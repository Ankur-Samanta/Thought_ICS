#!/usr/bin/env python3
"""
Baseline CoT Evaluation Script
Evaluates standard Chain of Thought (no Tree) with iterative correction
on the same 100 level-5 MATH problems for comparison.

Baseline conditions:
1. 0-shot CoT: Single attempt with standard CoT prompt
2. Iterative CoT (no GT): Standard CoT with iterative correction, no ground truth
3. Iterative CoT (with GT): Standard CoT with iterative correction, with ground truth
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import re
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

from thought_ics.thought_mdp import initialize_model
from thought_ics.chain_cache import load_initial_chains, save_initial_chains
from thought_ics.datasets import load_dataset_by_name, get_dataset_info, normalize_answer
from thought_ics.utils.wandb_utils import (
    WandbConfig, init_wandb_run, log_metrics,
    log_problem_result, log_summary_metrics, finish_run,
    create_run_name
)
from thought_ics.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration (defaults)
GENERATION_TEMP = 1.0  # Initial CoT generation (affects cache)
RESAMPLE_TEMP = 0.7    # Correction/regeneration (no cache impact)
JUDGE_TEMP = 0.3       # Error detection/verification (no cache impact)
MAX_DEPTH = 100  # Deep enough for complex problems, avoids recursion limit
MAX_TOKENS_PER_THOUGHT = None  # No limit
SEED = 42


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    if not text:
        return "NO ANSWER"

    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return "NO ANSWER"

    start_pos = matches[-1].end()
    brace_count = 1
    i = start_pos
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1

    if brace_count == 0:
        return text[start_pos:i-1].strip()

    return "NO ANSWER"


def identify_error_location_shared_prefix(
    manager,
    problem: str,
    solution: str,
    ground_truth: str,
    autonomy_level: int,
    temperature: float = 0.3
) -> Tuple[Optional[str], str]:
    """Identify error location by asking model to quote the erroneous text.

    This is used for shared-prefix correction mode.

    Returns:
        Tuple of (quoted_error_text, reasoning)
        quoted_error_text is None if no error found
    """

    if autonomy_level == 1:
        # L1: Oracle - model sees correct answer
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

The correct answer should be {ground_truth}.

Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows it's wrong
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

Your answer is incorrect. Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must self-verify
        prompt = f"""Problem: {problem}

Current solution:
{solution}

Carefully verify your solution step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), quote the EXACT text (word-for-word) where the first critical error occurs. This should be a continuous excerpt from your solution above.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}} if you found an error
- \\boxed{{CORRECT}} if the solution is correct
"""

    logger.info("Asking model to identify error location (shared prefix mode)...")

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=1024
    )

    response = outputs[0].strip()
    logger.info(f"Model response: {response[:200]}...")

    # Extract boxed answer
    boxed = extract_boxed_answer(response)

    if boxed == "CORRECT" or boxed == "NO_ERROR":
        logger.info("Model found no errors")
        return None, response

    # Try to extract quoted text
    # Format: ERROR_QUOTE: "quoted text"
    if "ERROR_QUOTE:" in boxed:
        quote_match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed)
        if quote_match:
            quoted_text = quote_match.group(1)
            logger.info(f"Extracted error quote: {quoted_text[:100]}...")
            return quoted_text, response

    # Fallback: try to find any quoted text in the response
    quote_matches = re.findall(r'"([^"]{20,})"', response)
    if quote_matches:
        quoted_text = quote_matches[0]
        logger.info(f"Found quoted text (fallback): {quoted_text[:100]}...")
        return quoted_text, response

    logger.warning("Could not extract error quote from response")
    return None, response


def find_truncation_point(solution: str, error_quote: str) -> Optional[int]:
    """Find where to truncate the solution based on the error quote.

    Uses exact matching only. Returns None if not found.
    """

    # Try exact match first
    idx = solution.find(error_quote)
    if idx != -1:
        logger.info(f"Found exact match at position {idx}")
        return idx

    # Try case-insensitive match
    idx = solution.lower().find(error_quote.lower())
    if idx != -1:
        logger.info(f"Found case-insensitive match at position {idx}")
        return idx

    # Could not find quote - will regenerate from scratch instead
    logger.warning(f"Could not find error quote in solution (will regenerate from scratch). Quote: {error_quote[:100]}")
    return None


def continue_from_prefix(
    manager,
    problem: str,
    prefix: str,
    max_tokens: int = 2048,
    temperature: float = 0.7
) -> str:
    """Continue generation from a prefix (shared prefix mode).

    Returns:
        Full solution (prefix + continuation)
    """

    prompt = f"""Solve the following math problem step by step. Show your reasoning clearly, then provide your final answer in the format \\boxed{{answer}}.

Problem: {problem}

Solution:
{prefix}"""

    logger.info(f"Continuing from prefix of length {len(prefix)} chars")

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=max_tokens
    )

    continuation = outputs[0].strip()

    # Return prefix + continuation
    return prefix + continuation


def convert_tot_chain_to_cot(chain: List[str]) -> str:
    """Convert a Tree of Thought chain to a standard CoT format by joining with newlines."""
    return "\n\n".join(chain)


def verify_solution_correctness_cot(manager, problem: str, solution: str, temperature: float = 0.3,
                                     mv_verify: bool = False, mv_k: int = 5,
                                     mv_criterion: str = "unanimous") -> Tuple[bool, str]:
    """Ask model directly if it thinks its final answer is correct.

    Args:
        manager: Model manager
        problem: Original problem statement
        solution: The solution text
        temperature: Sampling temperature for verification (default: 0.3)
        mv_verify: If True, use majority vote with k rollouts (default: False)
        mv_k: Number of rollouts for majority vote verification (default: 5)
        mv_criterion: Voting criterion - "unanimous" (all YES), "majority" (>50% YES), "any" (>=1 YES)

    Returns:
        Tuple of (believes_correct, reasoning)
        With mv_verify=True, believes_correct depends on mv_criterion
    """
    answer = extract_boxed_answer(solution)

    prompt = f"""You are reviewing a solution to a problem. Analyze it carefully to see if they arrived at the right answer.

Problem: {problem}

Solution to review:
{solution}

Final answer: {answer}

Verify the reasoning step by step and determine whether the final answer is correct or not.

Conclude with \\boxed{{YES}} if the solution is correct, or \\boxed{{NO}} if it contains errors."""

    if mv_verify:
        # Majority vote verification with k rollouts
        logger.info(f"MV Verification: generating {mv_k} rollouts...")

        outputs = manager.generate(
            prompts=[prompt] * mv_k,
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
        )

        # Parse each response
        votes = []
        for i, response in enumerate(outputs):
            response = response.strip()
            boxed = extract_boxed_answer(response).upper()

            if "YES" in boxed:
                votes.append("YES")
            elif "NO" in boxed:
                votes.append("NO")
            else:
                # Fallback: search for yes/no in response
                response_lower = response.lower()
                if "yes" in response_lower and "no" not in response_lower:
                    votes.append("YES")
                elif "no" in response_lower:
                    votes.append("NO")
                else:
                    votes.append("NO")  # Default to NO if unclear

        # Apply voting criterion
        yes_count = votes.count("YES")
        if mv_criterion == "unanimous":
            believes_correct = all(v == "YES" for v in votes)
        elif mv_criterion == "majority":
            believes_correct = yes_count > mv_k // 2  # >2 for k=5, i.e. ≥3
        elif mv_criterion == "any":
            believes_correct = yes_count >= 1
        else:
            raise ValueError(f"Unknown mv_criterion: {mv_criterion}")

        logger.info(f"MV Verification ({mv_criterion}, k={mv_k}): votes={votes}, yes={yes_count}/{mv_k}, believes_correct={believes_correct}")

        combined_reasoning = f"MV Verification ({mv_criterion}, k={mv_k}): votes={votes}, yes={yes_count}/{mv_k}, result={believes_correct}\n\n{outputs[0].strip()}"

        return believes_correct, combined_reasoning

    else:
        # Single rollout (existing behavior)
        logger.info("Asking model to verify if its final answer is correct...")

        outputs = manager.generate(
            prompts=[prompt],
            temperature=temperature,
            top_p=0.9,
            top_k=50,
            max_tokens=1024,
        )

        response = outputs[0].strip()
        logger.info(f"Verification response: {response[:200]}...")

        # Extract YES/NO from boxed answer (reuse existing function)
        boxed = extract_boxed_answer(response).upper()

        if "YES" in boxed:
            return True, response
        elif "NO" in boxed:
            return False, response

        # Fallback: search for yes/no in response
        response_lower = response.lower()
        if "yes" in response_lower and "no" not in response_lower:
            logger.warning("Could not parse boxed answer, but found 'yes' in response")
            return True, response
        elif "no" in response_lower:
            logger.warning("Could not parse boxed answer, but found 'no' in response")
            return False, response

        # Default: assume needs correction
        logger.warning("Could not determine YES/NO from response, assuming needs correction")
        return False, response


def generate_cot_solution(manager, problem: str, max_tokens: int = 2048, temperature: float = 1.0) -> str:
    """Generate a standard CoT solution without Tree of Thought."""

    prompt = f"""Solve the following math problem step by step. Show your reasoning clearly, then provide your final answer in the format \\boxed{{answer}}.

Problem: {problem}

Solution:"""

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=max_tokens
    )

    return outputs[0].strip()


def identify_error_step_cot(manager, problem: str, solution: str, ground_truth: str, autonomy_level: int = 1, temperature: float = 0.3) -> Tuple[bool, str]:
    """Ask model to identify if there's an error in the solution.

    Returns:
        Tuple of (has_error, reasoning)
    """

    if autonomy_level == 1:
        # L1: Oracle access - model sees correct answer
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

The correct answer should be {ground_truth}.

Analyze the solution step by step to identify where the error occurred (logical flaw, arithmetic error, or incorrect assumption).

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR}} if you found an error
- \\boxed{{CORRECT}} if the solution is actually correct
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows it's wrong but not the answer
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

Your answer is incorrect. Analyze the solution step by step to identify where the error occurred (logical flaw, arithmetic error, or incorrect assumption).

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR}} if you found an error
- \\boxed{{CORRECT}} if you cannot find the error
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must self-evaluate
        prompt = f"""Problem: {problem}

Current solution:
{solution}

Carefully verify your solution step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), analyze where the error occurred.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR}} if you found an error
- \\boxed{{CORRECT}} if the solution is correct
"""

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
    )

    response = outputs[0].strip()
    boxed = extract_boxed_answer(response)

    has_error = boxed == "ERROR"
    return has_error, response


def regenerate_solution(manager, problem: str, previous_solution: str, error_feedback: str, historical_context: Optional[str] = None, temperature: float = 0.7) -> str:
    """Regenerate solution given previous attempt and error feedback.

    Args:
        manager: Model manager
        problem: Original problem
        previous_solution: Previous attempt that had errors
        error_feedback: Analysis of what went wrong
        historical_context: With --context, context from earlier failed attempts
        temperature: Sampling temperature for regeneration
    """

    prompt = f"""Problem: {problem}

Previous attempt:
{previous_solution}

Error analysis:
{error_feedback}"""

    # Add historical context if provided (--context conditioning)
    if historical_context:
        prompt += f"""

Historical context (learn from these past mistakes):
{historical_context}"""

    prompt += """

Based on the error analysis, solve the problem again step by step. Show your corrected reasoning clearly, then provide your final answer in the format \\boxed{{answer}}.

Solution:"""

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=2048
    )

    return outputs[0].strip()


def baseline_cot_single(manager, problem: str, ground_truth: str) -> Dict:
    """Baseline: Single 0-shot CoT attempt."""

    logger.info("Generating 0-shot CoT solution...")
    solution = generate_cot_solution(manager, problem)
    answer = extract_boxed_answer(solution)
    correct = normalize_answer(answer) == normalize_answer(ground_truth)

    logger.info(f"Answer: {answer}, Correct: {correct}")

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'solution': solution,
        'answer': answer,
        'correct': correct,
        'iterations': 1
    }


def baseline_cot_iterative(manager, problem: str, ground_truth: str, max_iterations: int = 10, autonomy_level: int = 1,
                           initial_solution: Optional[str] = None, generation_temp: float = 1.0, resample_temp: float = 0.7,
                           judge_temp: float = 0.3, shared_prefix: bool = False, no_auto_stop: bool = False, use_context: bool = False,
                           use_3p_localize: bool = False, localize_api_key: Optional[str] = None, localize_model: str = "gpt-5",
                           verify: bool = False, mv_verify: bool = False, mv_k: int = 5, mv_criterion: str = "unanimous") -> Dict:
    """Baseline: Iterative CoT with correction.

    Args:
        initial_solution: If provided, use this as the initial solution instead of generating one
        generation_temp: Temperature for initial CoT generation
        resample_temp: Temperature for correction/regeneration
        judge_temp: Temperature for error detection/verification
        shared_prefix: If True, use token-level truncation and continuation (like ToT's step-level approach)
                      If False, regenerate entire solution each time (standard baseline)
    """

    autonomy_names = {1: "Oracle", 2: "Binary", 3: "Autonomous"}
    mode_str = "shared-prefix" if shared_prefix else "full-regen"
    logger.info(f"Running iterative CoT ({autonomy_names.get(autonomy_level, f'L{autonomy_level}')} - {mode_str})...")

    iterations_data = []

    # Initial attempt - use provided solution or generate new one
    if initial_solution is not None:
        logger.info("Using provided initial solution (from cached ToT chain)")
        solution = initial_solution
    else:
        logger.info("Generating new CoT solution")
        solution = generate_cot_solution(manager, problem, temperature=generation_temp)

    answer = extract_boxed_answer(solution)
    correct = normalize_answer(answer) == normalize_answer(ground_truth)

    iterations_data.append({
        'iteration': 0,
        'solution': solution,
        'answer': answer,
        'correct': correct,
        'error_reasoning': None,
        'verify_reasoning': None,
        'model_believes_correct': None
    })

    logger.info(f"Iteration 0: Answer = {answer}, Correct = {correct}")

    # Track historical attempts for context (if enabled)
    historical_attempts = []

    # Iterative correction
    for i in range(1, max_iterations + 1):
        if not no_auto_stop and correct:
            logger.info(f"SUCCESS! Correct answer found at iteration {i-1}")
            break

        # Optional verification: ask model if it thinks answer is correct
        # Initialize verification tracking variables
        iter_verify_reasoning = None
        iter_model_believes_correct = None

        if verify:
            believes_correct, verify_reasoning = verify_solution_correctness_cot(
                manager, problem, solution, temperature=judge_temp,
                mv_verify=mv_verify, mv_k=mv_k, mv_criterion=mv_criterion
            )
            is_actually_correct = normalize_answer(answer) == normalize_answer(ground_truth)
            logger.info(f"Verification result: model_believes_correct={believes_correct}, actually_correct={is_actually_correct}")

            # Store for inclusion in iteration data
            iter_verify_reasoning = verify_reasoning
            iter_model_believes_correct = believes_correct

            if believes_correct:
                correct = is_actually_correct  # Update correct so return value is accurate
                logger.info(f"Model believes answer is correct - stopping iteration.")
                iterations_data.append({
                    'iteration': i,
                    'solution': solution,
                    'answer': answer,
                    'correct': is_actually_correct,
                    'error_reasoning': None,
                    'verify_reasoning': verify_reasoning,
                    'model_believes_correct': True
                })
                break
            else:
                logger.info(f"Model believes answer is incorrect - continuing to error detection.")

        logger.info(f"Iteration {i}: Checking for errors...")

        # Initialize error localization variables
        error_quote = None
        truncation_idx = None
        prefix = None

        if shared_prefix:
            # SHARED PREFIX MODE: Use token-level error localization and truncation
            if use_3p_localize:
                # Use 3rd-party API for error localization
                from thought_ics.localization.third_party_api import call_3p_error_localization_cot_quote
                error_quote, error_reasoning = call_3p_error_localization_cot_quote(
                    problem, solution, ground_truth, localize_api_key, localize_model
                )
            else:
                # Use the evaluated model for error localization
                error_quote, error_reasoning = identify_error_location_shared_prefix(
                    manager, problem, solution, ground_truth, autonomy_level, judge_temp
                )

            if error_quote is None:
                if not no_auto_stop:
                    logger.warning("Model found no errors but answer is still wrong. Cannot proceed.")
                    break
                else:
                    # Model decided it's correct - stop iteration (true self-verification)
                    correct = normalize_answer(answer) == normalize_answer(ground_truth)
                    logger.info(f"Model found no errors - stopping iteration. Answer correct: {correct}")
                    iterations_data.append({
                        'iteration': i,
                        'solution': solution,
                        'answer': answer,
                        'correct': correct,
                        'error_reasoning': error_reasoning,
                        'verify_reasoning': iter_verify_reasoning,
                        'model_believes_correct': iter_model_believes_correct,
                        'error_quote': None,
                        'truncation_idx': None,
                        'prefix_kept': None
                    })
                    break
            else:
                # Find truncation point
                truncation_idx = find_truncation_point(solution, error_quote)

                if truncation_idx is None:
                    # Could not find exact quote - regenerate entire solution from scratch
                    logger.info("Quote not found - regenerating entire solution from scratch...")
                    solution = generate_cot_solution(manager, problem, temperature=resample_temp)
                else:
                    # Found quote - truncate at error point and continue from prefix
                    prefix = solution[:truncation_idx].rstrip()
                    logger.info(f"Truncating at position {truncation_idx}, prefix length: {len(prefix)}")
                    logger.info("Regenerating from prefix...")
                    solution = continue_from_prefix(manager, problem, prefix, temperature=resample_temp)

        else:
            # STANDARD MODE: Binary error detection and full regeneration
            has_error, error_reasoning = identify_error_step_cot(manager, problem, solution, ground_truth, autonomy_level, judge_temp)

            if not has_error:
                if not no_auto_stop:
                    logger.info("Model found no errors but answer is wrong - continuing with oracle verification")
                    # break  # Commented out: let oracle verification handle stopping for L2
                else:
                    # Model decided it's correct - stop iteration (true self-verification)
                    correct = normalize_answer(answer) == normalize_answer(ground_truth)
                    logger.info(f"Model found no errors - stopping iteration. Answer correct: {correct}")
                    iterations_data.append({
                        'iteration': i,
                        'solution': solution,
                        'answer': answer,
                        'correct': correct,
                        'error_reasoning': error_reasoning,
                        'verify_reasoning': iter_verify_reasoning,
                        'model_believes_correct': iter_model_believes_correct
                    })
                    break

            # Track historical attempts (if context enabled)
            if use_context:
                historical_attempts.append({
                    'solution': solution[:200] + "...",  # Truncate to save memory
                    'error': error_reasoning[:150] + "..."
                })

            # Regenerate solution
            logger.info("Regenerating solution based on error analysis...")

            # Build historical context (if enabled)
            historical_context = None
            if use_context and len(historical_attempts) > 0:
                historical_context = "Past failed attempts:\n"
                for idx, attempt in enumerate(historical_attempts[-2:], 1):  # Last 2 attempts
                    historical_context += f"{idx}. Error: {attempt['error']}\n"

            solution = regenerate_solution(manager, problem, solution, error_reasoning, historical_context, resample_temp)
        answer = extract_boxed_answer(solution)
        correct = normalize_answer(answer) == normalize_answer(ground_truth)

        # Build iteration data
        iteration_data = {
            'iteration': i,
            'solution': solution,
            'answer': answer,
            'correct': correct,
            'error_reasoning': error_reasoning,
            'verify_reasoning': iter_verify_reasoning,
            'model_believes_correct': iter_model_believes_correct
        }

        # Add error localization data for shared prefix mode
        if shared_prefix:
            iteration_data['error_quote'] = error_quote
            iteration_data['truncation_idx'] = truncation_idx
            iteration_data['prefix_kept'] = prefix

        iterations_data.append(iteration_data)

        logger.info(f"Iteration {i}: Answer = {answer}, Correct = {correct}")

        if not no_auto_stop and correct:
            logger.info(f"SUCCESS! Correct answer found at iteration {i}")
            break

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'iterations_data': iterations_data,
        'success': correct,
        'total_iterations': len(iterations_data)
    }


def baseline_cot_majority_vote(manager, problem: str, ground_truth: str, n_samples: int = 10, generation_temp: float = 1.0, cached_solution: str = None) -> Dict:
    """Baseline: Majority vote over N independent CoT samples.

    Args:
        manager: Model manager
        problem: Problem statement
        ground_truth: Correct answer
        n_samples: Number of independent samples to generate
        generation_temp: Sampling temperature for initial generation
        cached_solution: Optional cached CoT solution to use as first sample
    """

    logger.info(f"Generating {n_samples} CoT samples for majority vote...")

    samples = []
    answer_counts = {}
    start_idx = 0

    # Use cached solution as first sample if provided
    if cached_solution is not None:
        answer = extract_boxed_answer(cached_solution)
        samples.append({
            'solution': cached_solution,
            'answer': answer,
            'from_cache': True
        })
        if answer not in answer_counts:
            answer_counts[answer] = 0
        answer_counts[answer] += 1
        logger.info(f"Sample 1/{n_samples} (from cache): Answer: {answer}")
        start_idx = 1

    for i in range(start_idx, n_samples):
        logger.info(f"Sample {i+1}/{n_samples}")
        solution = generate_cot_solution(manager, problem, max_tokens=2048, temperature=generation_temp)
        answer = extract_boxed_answer(solution)

        samples.append({
            'solution': solution,
            'answer': answer,
            'from_cache': False
        })

        # Count answers
        if answer not in answer_counts:
            answer_counts[answer] = 0
        answer_counts[answer] += 1

        logger.info(f"  Answer: {answer}")

    # Find majority answer
    if answer_counts:
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
        majority_count = answer_counts[majority_answer]
    else:
        majority_answer = "NO ANSWER"
        majority_count = 0

    correct = normalize_answer(majority_answer) == normalize_answer(ground_truth)

    logger.info(f"Majority vote: {majority_answer} ({majority_count}/{n_samples} samples)")
    logger.info(f"Correct: {correct}")

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'samples': samples,
        'answer_counts': answer_counts,
        'majority_answer': majority_answer,
        'majority_count': majority_count,
        'n_samples': n_samples,
        'correct': correct,
        'used_cache': cached_solution is not None
    }


def run_baseline_evaluation(
    baseline_type: str,
    gpu_ids: str,
    tensor_parallel_size: int,
    n_problems: int = 100,
    max_iterations: int = 10,
    dataset: str = "math500",
    level: Optional[int] = None,
    model_name: str = "llama8b",
    autonomy_level: int = None,
    use_cached_chains: bool = True,
    regenerate_cot: bool = False,
    cache_type: str = "tot",
    generation_temp: float = 1.0,
    resample_temp: float = 0.7,
    judge_temp: float = 0.3,
    shared_prefix: bool = False,
    output_dir: str = "experiments",
    wandb_project: str = "thought-ics",
    wandb_entity: Optional[str] = None,
    enable_wandb: bool = False,
    no_auto_stop: bool = False,
    use_context: bool = False,
    use_3p_localize: bool = False,
    localize_api_key: Optional[str] = None,
    localize_model: str = "gpt-5",
    verify: bool = False,
    mv_verify: bool = False,
    mv_k: int = 5,
    mv_criterion: str = "unanimous"
):
    """Run baseline evaluation.

    Args:
        baseline_type: Type of baseline to run
        gpu_ids: GPU IDs to use
        tensor_parallel_size: Tensor parallel size
        n_problems: Number of problems to evaluate
        max_iterations: Max iterations/samples
        dataset: Dataset name ("math500", "gsm8k", or "amc23")
        level: For MATH-500, filter by difficulty level (1-5)
        model_name: Model nickname (default: "llama8b")
        autonomy_level: Autonomy level for iterative baselines
        use_cached_chains: If True, use cached chains
        regenerate_cot: If True, regenerate CoT instead of using cached chains (overrides use_cached_chains)
        cache_type: Type of cache to use - "tot" for ToT chains (converted to CoT), "cot" for CoT chains
        generation_temp: Temperature for initial CoT generation (affects cache)
        resample_temp: Temperature for correction/regeneration (no cache impact)
        judge_temp: Temperature for error detection/verification (no cache impact)
        shared_prefix: If True, use token-level truncation and continuation (like ToT)
        output_dir: Output directory for results
        wandb_project: Wandb project name
        wandb_entity: Wandb entity (username or team)
        enable_wandb: Whether to enable wandb logging
    """

    # Get dataset info
    dataset_info = get_dataset_info(dataset)

    baseline_names = {
        'single': 'Baseline_0shot_CoT',
        'iterative_l1': 'Baseline_Iterative_CoT_L1',
        'iterative_l2': 'Baseline_Iterative_CoT_L2',
        'iterative_l3': 'Baseline_Iterative_CoT_L3',
        'iterative_no_gt': 'Baseline_Iterative_CoT_L3',  # Alias for backward compat
        'iterative_with_gt': 'Baseline_Iterative_CoT_L1',  # Alias for backward compat
        'majority_vote': 'Baseline_MajorityVote_CoT'
    }

    baseline_name = baseline_names.get(baseline_type, baseline_type)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build experiment name with appropriate suffixes
    if use_cached_chains and not regenerate_cot:
        cache_suffix = f"_from_cached_{cache_type}"
    else:
        cache_suffix = "" if baseline_type == 'majority_vote' else "_regenerated"

    # Add shared prefix suffix if enabled
    prefix_suffix = "_shared_prefix" if shared_prefix else ""

    # Add context suffix if historical-context conditioning is enabled (any level)
    context_suffix = "_with_context" if use_context else ""

    # Add level suffix if level filter is specified
    level_suffix = f"_L{level}" if level else ""

    experiment_name = f"{baseline_name}{cache_suffix}{prefix_suffix}{context_suffix}_{model_name}_{dataset}{level_suffix}_{timestamp}"
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    log_file = exp_dir / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Save experiment config
    config = {
        'experiment_name': experiment_name,
        'baseline_type': baseline_type,
        'baseline_name': baseline_name,
        'model_name': model_name,
        'gpu_ids': gpu_ids,
        'tensor_parallel_size': tensor_parallel_size,
        'n_problems': n_problems,
        'max_iterations': max_iterations,
        'use_cached_chains': use_cached_chains and not regenerate_cot,
        'regenerate_cot': regenerate_cot,
        'cache_type': cache_type,
        'shared_prefix': shared_prefix,
        'generation_temp': generation_temp,
        'resample_temp': resample_temp,
        'judge_temp': judge_temp,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'level_filter': level,
        'seed': SEED,
        'no_auto_stop': no_auto_stop,
        'use_context': use_context,
        'use_3p_localize': use_3p_localize,
        'localize_model': localize_model if use_3p_localize else None,
        'timestamp': datetime.now().isoformat()
    }

    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize wandb
    wandb_run = None
    if enable_wandb:
        wandb_config = WandbConfig(
            enabled=True,
            project=wandb_project,
            entity=wandb_entity,
            tags=["baseline_cot", baseline_type, model_name, dataset],
            notes=f"Baseline CoT evaluation: {baseline_name} on {dataset}"
        )

        run_name = create_run_name(
            model_name=model_name,
            dataset_name=dataset,
            experiment_type=f"baseline_{baseline_type}",
            level=f"lvl{level}" if level else None
        )

        wandb_run = init_wandb_run(
            config=wandb_config,
            run_name=run_name,
            job_type="baseline_evaluation",
            run_config=config
        )
        logger.info(f"Wandb run initialized: {run_name}")

    logger.info("="*100)
    logger.info(f"BASELINE EVALUATION - {baseline_name}")
    logger.info("="*100)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dataset: {dataset_info['name']}")
    if level is not None:
        logger.info(f"Level filter: {level}")
    logger.info(f"Output dir: {exp_dir}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Tensor Parallel Size: {tensor_parallel_size}")
    logger.info(f"Number of problems: {n_problems}")
    logger.info(f"Generation temperature: {generation_temp}")
    logger.info(f"Resample temperature: {resample_temp}")
    logger.info(f"Judge temperature: {judge_temp}")
    if baseline_type != 'single':
        logger.info(f"Max iterations per problem: {max_iterations}")
    logger.info(f"Use cached ToT chains: {use_cached_chains and not regenerate_cot}")
    logger.info(f"Regenerate CoT: {regenerate_cot}")
    logger.info("="*100)

    # Load problems
    logger.info(f"Loading {dataset} dataset...")
    problems = load_dataset_by_name(
        dataset_name=dataset,
        n_problems=n_problems,
        level=level,
        seed=SEED
    )

    # Load cached chains if requested
    cached_chains = None
    if use_cached_chains and not regenerate_cot:
        logger.info(f"Attempting to load cached {cache_type.upper()} chains...")
        # For CoT cache, use max_depth=1 and max_tokens=4096 to match how caches are saved
        # For ToT cache, use the global defaults (MAX_DEPTH=100, MAX_TOKENS_PER_THOUGHT=None)
        if cache_type == "cot":
            cache_max_depth = 1
            cache_max_tokens = 4096
        else:
            cache_max_depth = MAX_DEPTH
            cache_max_tokens = MAX_TOKENS_PER_THOUGHT

        cached_chains = load_initial_chains(
            model_name=model_name,
            dataset_name=dataset,
            n_problems=n_problems,
            seed=SEED,
            temperature=generation_temp,
            max_depth=cache_max_depth,
            max_tokens_per_thought=cache_max_tokens,
            cache_type=cache_type
        )
        if cached_chains is not None:
            if cache_type == "tot":
                logger.info(f"Loaded {len(cached_chains)} cached ToT chains - will convert to CoT format")
            else:
                logger.info(f"Loaded {len(cached_chains)} cached CoT chains")
        else:
            logger.warning(f"No cached {cache_type.upper()} chains found! Will regenerate CoT solutions.")
            regenerate_cot = True

    # Initialize model
    logger.info(f"Initializing model '{model_name}' on GPUs {gpu_ids}...")
    manager = initialize_model(gpu_ids=gpu_ids, tensor_parallel_size=tensor_parallel_size, model_name=model_name)

    # Run evaluation
    # Check for existing progress (resume capability)
    checkpoint_file = exp_dir / "checkpoint.json"
    completed_problem_ids = set()
    results = []

    if checkpoint_file.exists():
        logger.info(f"Found existing checkpoint: {checkpoint_file}")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                results = checkpoint_data.get('results', [])
                completed_problem_ids = set(r['problem_id'] for r in results)
                logger.info(f"Resuming from checkpoint: {len(results)}/{len(problems)} problems already completed")
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}. Starting fresh.")
            results = []
            completed_problem_ids = set()

    stats = {
        'total_problems': len(problems),
        'successful': sum(1 for r in results if r.get('success', False)),
        'failed': sum(1 for r in results if not r.get('success', False)),
        'total_iterations': sum(r.get('total_iterations', 0) for r in results),
        'avg_iterations': 0.0,
        'baseline_type': baseline_type,
        'baseline_name': baseline_name,
        'gpu_ids': gpu_ids,
        'tensor_parallel_size': tensor_parallel_size,
        'timestamp': datetime.now().isoformat()
    }

    # Track freshly generated CoT chains for caching
    # Save cache whenever we regenerate fresh solutions (cache miss or forced regenerate)
    fresh_cot_chains = []
    fresh_cot_problem_ids = set()  # Track which problems have been added to prevent duplicates
    should_save_cot_cache = (cache_type == 'cot' and
                             (regenerate_cot or cached_chains is None))

    # Pre-populate fresh_cot_chains from checkpoint if resuming
    # This ensures cache can be saved even when resuming from a partial run
    # Use a dict keyed by problem_id to avoid duplicates
    if checkpoint_file.exists() and should_save_cot_cache and len(results) > 0:
        logger.info(f"Extracting solutions from {len(results)} checkpoint results for cache saving...")
        checkpoint_solutions = {}  # problem_id -> {solution, answer}
        for r in results:
            pid = r.get('problem_id')
            if pid is None or pid in checkpoint_solutions:
                continue  # Skip duplicates or entries without problem_id
            sol = None
            ans = None
            if 'iterations_data' in r and len(r['iterations_data']) > 0:
                sol = r['iterations_data'][0].get('solution', '')
                ans = r['iterations_data'][0].get('answer', '')
            elif 'solution' in r:
                sol = r['solution']
                ans = r.get('answer', '')
            if sol:
                checkpoint_solutions[pid] = {'solution': sol, 'answer': ans}
        # Add in order matching problems list
        for item in problems:
            pid = item['unique_id']
            if pid in checkpoint_solutions and pid not in fresh_cot_problem_ids:
                fresh_cot_chains.append(checkpoint_solutions[pid])
                fresh_cot_problem_ids.add(pid)
        logger.info(f"Pre-populated {len(fresh_cot_chains)} solutions from checkpoint")

    logger.info("\nStarting evaluation...")
    for idx, item in enumerate(tqdm(problems, desc=f"Evaluating {baseline_name}")):
        problem_num = idx + 1

        # Skip if already completed (resume capability)
        if item['unique_id'] in completed_problem_ids:
            logger.info(f"Skipping problem {problem_num}/{len(problems)} (already completed)")
            continue

        logger.info(f"\n{'='*80}")
        logger.info(f"Problem {problem_num}/{len(problems)}")
        logger.info(f"Subject: {item['subject']}, Level: {item['level']}")
        logger.info(f"{'='*80}")

        try:
            # Get initial solution
            initial_solution = None
            if cached_chains is not None and not regenerate_cot:
                cached_chain = cached_chains[idx]
                if cache_type == "tot":
                    # Convert cached ToT chain to CoT format
                    initial_solution = convert_tot_chain_to_cot(cached_chain['chain'])
                    logger.info(f"Using cached ToT chain ({len(cached_chain['chain'])} steps) converted to CoT")
                else:
                    # Use cached CoT chain directly
                    initial_solution = cached_chain.get('solution', cached_chain.get('chain', ''))
                    logger.info(f"Using cached CoT chain directly")

            if baseline_type == 'single':
                if initial_solution is not None:
                    # For single shot, just use the initial solution
                    answer = extract_boxed_answer(initial_solution)
                    result = {
                        'problem': item['problem'],
                        'ground_truth': item['answer'],
                        'solution': initial_solution,
                        'answer': answer,
                        'correct': normalize_answer(answer) == normalize_answer(item['answer']),
                        'iterations': 1,
                        'from_cache': cache_type
                    }
                else:
                    result = baseline_cot_single(manager, item['problem'], item['answer'])
                    result['from_cache'] = None
                    # Track fresh CoT for caching
                    if should_save_cot_cache and item['unique_id'] not in fresh_cot_problem_ids:
                        fresh_cot_chains.append({
                            'solution': result['solution'],
                            'answer': result['answer']
                        })
                        fresh_cot_problem_ids.add(item['unique_id'])

            elif baseline_type == 'majority_vote':
                # Majority vote baseline - generate N samples and take majority
                # Use cached CoT solution as first sample if available
                cached_solution = None
                if use_cached_chains and not regenerate_cot and cached_chains is not None:
                    cached_chain = cached_chains[idx]
                    if cache_type == 'cot':
                        cached_solution = cached_chain.get('solution', cached_chain.get('chain', ''))
                    elif cache_type == 'tot':
                        cached_solution = convert_tot_chain_to_cot(cached_chain.get('chain', ''))
                    logger.info(f"Using cached {cache_type} solution as first MV sample")

                result = baseline_cot_majority_vote(
                    manager,
                    item['problem'],
                    item['answer'],
                    n_samples=max_iterations if max_iterations > 1 else 10,
                    generation_temp=generation_temp,
                    cached_solution=cached_solution
                )
                result['from_cache'] = cache_type if cached_solution else None

            else:
                # Iterative baselines (L1, L2, L3)
                # Determine autonomy level
                if baseline_type in ['iterative_l1', 'iterative_with_gt']:
                    auto_level = 1
                elif baseline_type == 'iterative_l2':
                    auto_level = 2
                elif baseline_type in ['iterative_l3', 'iterative_no_gt']:
                    auto_level = 3
                else:
                    # Default to autonomy_level param if provided, else L3
                    auto_level = autonomy_level if autonomy_level is not None else 3

                result = baseline_cot_iterative(
                    manager,
                    item['problem'],
                    item['answer'],
                    max_iterations=max_iterations,
                    autonomy_level=auto_level,
                    initial_solution=initial_solution,
                    generation_temp=generation_temp,
                    resample_temp=resample_temp,
                    judge_temp=judge_temp,
                    shared_prefix=shared_prefix,
                    no_auto_stop=no_auto_stop,
                    use_context=use_context,
                    use_3p_localize=use_3p_localize,
                    localize_api_key=localize_api_key,
                    localize_model=localize_model,
                    verify=verify,
                    mv_verify=mv_verify,
                    mv_k=mv_k,
                    mv_criterion=mv_criterion
                )
                result['from_cache'] = cache_type if initial_solution is not None else None

                # Track fresh CoT for caching (when we had a cache miss)
                if should_save_cot_cache and initial_solution is None and item['unique_id'] not in fresh_cot_problem_ids:
                    # Extract initial solution from iterations_data
                    if result.get('iterations_data') and len(result['iterations_data']) > 0:
                        init_sol = result['iterations_data'][0].get('solution', '')
                        init_ans = result['iterations_data'][0].get('answer', '')
                        fresh_cot_chains.append({
                            'solution': init_sol,
                            'answer': init_ans
                        })
                        fresh_cot_problem_ids.add(item['unique_id'])

            result['problem_id'] = item['unique_id']
            result['subject'] = item['subject']
            result['level'] = item['level']
            result['problem_number'] = problem_num

            results.append(result)

            # Determine success based on baseline type
            if baseline_type in ['single', 'majority_vote']:
                is_success = result.get('correct', False)
            else:
                is_success = result.get('success', False)

            if is_success:
                stats['successful'] += 1
            else:
                stats['failed'] += 1

            total_iters = result.get('total_iterations', result.get('iterations', 1))
            stats['total_iterations'] += total_iters

            logger.info(f"Result: {'SUCCESS' if is_success else 'FAILED'} "
                       f"(iterations: {total_iters})")

            # Log to wandb
            if enable_wandb and wandb_run is not None:
                predicted_ans = result.get('final_answer', result.get('answer', 'NO ANSWER'))
                log_problem_result(
                    problem_id=item['unique_id'],
                    problem_number=problem_num,
                    predicted_answer=predicted_ans,
                    ground_truth=item['answer'],
                    correct=is_success,
                    iterations=total_iters,
                    additional_metrics={
                        'subject': item['subject'],
                        'level': item['level'],
                        'baseline_type': baseline_type
                    }
                )

            # Save checkpoint after each problem
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f, indent=2)

        except Exception as e:
            logger.error(f"Error on problem {problem_num}: {e}")
            results.append({
                'problem_id': item['unique_id'],
                'problem': item['problem'],
                'ground_truth': item['answer'],
                'subject': item['subject'],
                'level': item['level'],
                'problem_number': problem_num,
                'error': str(e),
                'correct': False,
                'success': False
            })
            stats['failed'] += 1

            # Save checkpoint even after errors
            with open(checkpoint_file, 'w') as f:
                json.dump({'results': results}, f, indent=2)

    # Compute final stats
    stats['avg_iterations'] = stats['total_iterations'] / stats['total_problems'] if stats['total_problems'] > 0 else 0
    stats['success_rate'] = (stats['successful'] / stats['total_problems']) * 100 if stats['total_problems'] > 0 else 0

    # Save freshly generated CoT chains to cache
    if should_save_cot_cache and len(fresh_cot_chains) == len(problems):
        logger.info(f"Saving {len(fresh_cot_chains)} freshly generated CoT chains to cache...")
        save_initial_chains(
            chains=fresh_cot_chains,
            model_name=model_name,
            dataset_name=dataset,
            n_problems=n_problems,
            seed=SEED,
            temperature=generation_temp,
            max_depth=1,  # CoT is single-step
            max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT if MAX_TOKENS_PER_THOUGHT else 4096,
            cache_type="cot"
        )
        logger.info(f"CoT cache saved successfully!")
    elif should_save_cot_cache and len(fresh_cot_chains) > 0:
        if len(fresh_cot_chains) > len(problems):
            logger.warning(f"BUG: More solutions ({len(fresh_cot_chains)}) than problems ({len(problems)}) - tracked {len(fresh_cot_problem_ids)} unique IDs")
        else:
            logger.warning(f"Only {len(fresh_cot_chains)}/{len(problems)} problems completed - not saving to cache")

    # Save results
    results_file = exp_dir / "results.json"

    full_results = {
        'stats': stats,
        'results': results
    }

    with open(results_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\n{'='*100}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*100}")
    logger.info(f"Total problems: {stats['total_problems']}")
    logger.info(f"Successful: {stats['successful']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Success rate: {stats['success_rate']:.2f}%")
    logger.info(f"Average iterations: {stats['avg_iterations']:.2f}")
    logger.info(f"Total iterations: {stats['total_iterations']}")
    logger.info(f"\nResults saved to: {exp_dir}")
    logger.info(f"  - Config: {config_file}")
    logger.info(f"  - Results: {results_file}")
    logger.info(f"{'='*100}")

    # Log summary to wandb
    if enable_wandb and wandb_run is not None:
        log_summary_metrics(
            total_problems=stats['total_problems'],
            correct_problems=stats['successful'],
            mean_iterations=stats['avg_iterations'],
            additional_summary={
                'success_rate': stats['success_rate'],
                'total_iterations': stats['total_iterations'],
                'baseline_type': baseline_type,
                'dataset': dataset,
                'model': model_name
            }
        )
        finish_run()

    # Compute comprehensive metrics
    logger.info(f"\n{'='*100}")
    logger.info("Computing comprehensive metrics...")
    logger.info(f"{'='*100}")
    try:
        metrics = compute_metrics(exp_dir)
        metrics_file = exp_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to: {metrics_file}")

        # Log key metrics
        overall = metrics.get("overall_performance", {})
        logger.info(f"  First Attempt Accuracy: {overall.get('first_attempt_accuracy', 0):.1%}")
        logger.info(f"  Final Accuracy: {overall.get('final_accuracy', 0):.1%}")
        logger.info(f"  Improvement: {overall.get('absolute_improvement', 0):.1%}")

        # Log C1 error detection if available
        if "error_detection_ability" in metrics:
            ed = metrics["error_detection_ability"]
            pm = ed.get("performance_metrics", {})
            logger.info(f"  Error Detection - Precision: {pm.get('precision', 0):.1%}, Recall: {pm.get('recall', 0):.1%}")
    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")
        logger.error("Continuing anyway...")

    # Cleanup
    manager.unload_base_model()

    return full_results


def main():
    parser = argparse.ArgumentParser(description='Baseline CoT Evaluation')
    parser.add_argument('--baseline-type', type=str,
                        choices=['single', 'iterative_l1', 'iterative_l2', 'iterative_l3',
                                'iterative_no_gt', 'iterative_with_gt', 'majority_vote'],
                        required=True,
                        help='Baseline type: single=0-shot, iterative_l1=Oracle, iterative_l2=Binary, '
                             'iterative_l3=Autonomous, majority_vote=N samples + vote. '
                             'Add --context for historical-context conditioning at any iterative level.')
    parser.add_argument('--gpus', type=str, required=True,
                        help='Comma-separated GPU IDs (e.g., "4,5")')
    parser.add_argument('--tensor-parallel-size', type=int, required=True,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--dataset', type=str, default='math500',
                        choices=['math500', 'gsm8k', 'amc23', 'aime', 'csqa', 'gpqa', 'svamp', 'mathqa', 'imo', 'imobench'],
                        help='Dataset to evaluate on (default: math500)')
    parser.add_argument('--level', type=int, default=None,
                        help='For MATH-500, filter by difficulty level 1-5 (default: all levels)')
    parser.add_argument('--model', type=str, default='llama8b',
                        choices=['llama8b', 'llama70b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen2b', 'llama3b', 'phi4b',
                                 'mistral3b', 'mistral8b', 'mistral14b', 'gptoss20b', 'gptoss120b'],
                        help='Model to use (default: llama8b)')
    parser.add_argument('--n-problems', type=int, default=100,
                        help='Number of problems to evaluate (default: 100)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='For iterative: max correction iterations. For majority_vote: number of samples (default: 10)')
    parser.add_argument('--use-cached-chains', action='store_true', default=True,
                        help='Use cached chains (default: True)')
    parser.add_argument('--regenerate-cot', action='store_true',
                        help='Regenerate CoT instead of using cached chains (overrides --use-cached-chains)')
    parser.add_argument('--cache-type', type=str, default='tot', choices=['tot', 'cot'],
                        help='Type of cache to use: "tot" for ToT chains (converted to CoT), "cot" for CoT chains (default: tot)')
    parser.add_argument('--generation-temp', type=float, default=1.0,
                        help='Temperature for initial CoT generation (affects cache) (default: 1.0)')
    parser.add_argument('--resample-temp', type=float, default=0.7,
                        help='Temperature for correction/regeneration (no cache impact) (default: 0.7)')
    parser.add_argument('--judge-temp', type=float, default=0.3,
                        help='Temperature for error detection/verification (no cache impact) (default: 0.3)')
    parser.add_argument('--shared-prefix', action='store_true',
                        help='Use shared-prefix mode: token-level truncation and continuation (like ToT step-level)')
    parser.add_argument('--3p-localize', action='store_true',
                        help='Use 3rd-party API (e.g., GPT-5) for error localization instead of evaluated model (requires --shared-prefix)')
    parser.add_argument('--3p-api-key', type=str, default=None,
                        help='API key for 3rd-party localization service (default: None, uses OPENAI_API_KEY env var)')
    parser.add_argument('--3p-model', type=str, default='gpt-5',
                        help='Model to use for 3rd-party localization (default: gpt-5)')
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='Disable auto-stopping on correct answer - always run max_iterations (default: False)')
    parser.add_argument('--context', action='store_true',
                        help='Condition the resample on prior failed attempts and their error analysis '
                             '(instead of a fresh stochastic sample). Combinable with any iterative baseline.')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory for results (default: experiments)')
    parser.add_argument('--wandb-project', type=str, default='thought-ics',
                        help='Wandb project name (default: SIERA)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity (username or team) (default: your logged-in wandb entity)')
    parser.add_argument('--enable-wandb', action='store_true',
                        help='Enable wandb logging (default: False)')
    parser.add_argument('--verify', action='store_true',
                        help='Enable solution verification before error detection (ask model if it thinks answer is correct)')
    parser.add_argument('--mv', action='store_true',
                        help='Enable majority vote verification (requires --verify)')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of rollouts for majority vote verification (default: 5)')
    parser.add_argument('--mv-criterion', type=str, default='unanimous',
                        choices=['unanimous', 'majority', 'any'],
                        help='MV voting criterion: unanimous (all YES), majority (>50%% YES), any (>=1 YES). Default: unanimous')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validation: --mv requires --verify
    if args.mv and not args.verify:
        raise ValueError("--mv requires --verify to be enabled")

    # Determine autonomy level from baseline type
    autonomy_level = None
    if 'l1' in args.baseline_type or 'with_gt' in args.baseline_type:
        autonomy_level = 1
    elif 'l2' in args.baseline_type:
        autonomy_level = 2
    elif 'l3' in args.baseline_type or 'no_gt' in args.baseline_type:
        autonomy_level = 3

    # Historical-context conditioning is orthogonal to the baseline level (--context).
    use_context = args.context

    # Get 3p API key from args or environment
    use_3p_localize = getattr(args, '3p_localize', False)
    localize_api_key = getattr(args, '3p_api_key', None)
    localize_model = getattr(args, '3p_model', 'gpt-5')

    # Validate 3p-localize configuration
    if use_3p_localize:
        if not args.shared_prefix:
            raise ValueError("--3p-localize requires --shared-prefix mode to be enabled")

        # Get API key from environment if not provided
        if localize_api_key is None:
            import os
            localize_api_key = os.environ.get('OPENAI_API_KEY')
            if localize_api_key is None:
                raise ValueError("--3p-localize requires --3p-api-key or OPENAI_API_KEY environment variable")

        logger.info(f"3P localization enabled: using {localize_model} for error localization")

    run_baseline_evaluation(
        baseline_type=args.baseline_type,
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        n_problems=args.n_problems,
        max_iterations=args.max_iterations,
        dataset=args.dataset,
        level=args.level,
        model_name=args.model,
        use_cached_chains=args.use_cached_chains,
        regenerate_cot=args.regenerate_cot,
        cache_type=args.cache_type,
        generation_temp=args.generation_temp,
        resample_temp=args.resample_temp,
        judge_temp=args.judge_temp,
        shared_prefix=args.shared_prefix,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        enable_wandb=args.enable_wandb,
        no_auto_stop=args.no_auto_stop,
        use_context=use_context,
        use_3p_localize=use_3p_localize,
        localize_api_key=localize_api_key,
        localize_model=localize_model,
        verify=args.verify,
        mv_verify=args.mv,
        mv_k=args.k,
        mv_criterion=args.mv_criterion
    )


if __name__ == "__main__":
    main()
