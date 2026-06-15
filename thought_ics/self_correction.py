#!/usr/bin/env python3
"""
Iterative Self-Correction Pipeline:
1. Generate full chain to completion
2. Identify error step (backtrack)
3. Regenerate from error point
4. Repeat L times or until correct
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import re
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from thought_ics.thought_mdp import (
    ToTAgent, ToTEnvironment, TreeSearch,
    initialize_model, get_completed_paths
)
from thought_ics.datasets import normalize_answer
from thought_ics import recommended_prompts  # active default localization prompts (L1/L2)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def generate_full_chain(manager, problem: str, temperature: float = 1.0, max_depth: int = 20, max_tokens_per_thought: int = 150) -> List[str]:
    """Generate a complete reasoning chain thought by thought."""
    logger.info("Generating initial chain...")

    agent = ToTAgent(manager, temperature=temperature, max_tokens=max_tokens_per_thought)
    env = ToTEnvironment(max_depth=max_depth)
    search = TreeSearch(agent, env, strategy="dfs", n_rollouts=1)

    root = search.search(problem, verbose=False)
    completed_paths = get_completed_paths(root)

    if not completed_paths:
        logger.warning("No completed paths found!")
        return []

    # Return first path (skip the question itself)
    chain = completed_paths[0][1:]  # Skip question
    answer = extract_boxed_answer(chain[-1] if chain else "")

    logger.info(f"Generated chain with {len(chain)} steps, answer: {answer}")
    return chain


def verify_single_step(manager, problem: str, context_steps: List[str], current_step_idx: int, ground_truth: str, autonomy_level: int, temperature: float = 0.3) -> Tuple[bool, str]:
    """Verify if a single step is correct given the context.

    Args:
        manager: Model manager
        problem: Original problem statement
        context_steps: List of steps from 1 to current_step_idx (inclusive)
        current_step_idx: The step number being verified (1-indexed)
        ground_truth: Correct answer (used for L1 prompting)
        autonomy_level: 1 (oracle), 2 (binary feedback), or 3 (full autonomy)
        temperature: Sampling temperature for verification (default: 0.3)

    Returns:
        Tuple of (is_correct, reasoning)
    """

    # Build context representation
    if current_step_idx == 1:
        context_text = ""
    else:
        context_text = "\n\nPrevious steps (already verified):"
        for i in range(current_step_idx - 1):
            context_text += f"\nStep {i+1}: {context_steps[i]}"

    current_step_text = context_steps[current_step_idx - 1]

    if autonomy_level == 1:
        # L1: Oracle access - model sees correct answer
        prompt = f"""Problem: {problem}
{context_text}

Current step to verify:
Step {current_step_idx}: {current_step_text}

The correct final answer should be {ground_truth}.

Question: Is Step {current_step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if current_step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {current_step_idx} is correct
- \\boxed{{NO}} if Step {current_step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows chain is wrong but not the answer
        prompt = f"""Problem: {problem}
{context_text}

Current step to verify:
Step {current_step_idx}: {current_step_text}

You are verifying a reasoning chain that led to an incorrect answer.

Question: Is Step {current_step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if current_step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {current_step_idx} is correct
- \\boxed{{NO}} if Step {current_step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must verify independently
        prompt = f"""Problem: {problem}
{context_text}

Current step to verify:
Step {current_step_idx}: {current_step_text}

Question: Is Step {current_step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if current_step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {current_step_idx} is correct
- \\boxed{{NO}} if Step {current_step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
    )

    response = outputs[0].strip()

    # Extract YES/NO from boxed answer
    answer = extract_boxed_answer(response).upper()

    if "YES" in answer:
        return True, response
    elif "NO" in answer:
        return False, response
    else:
        # Fallback: search for yes/no in response
        response_lower = response.lower()
        if "yes" in response_lower and "no" not in response_lower:
            logger.warning(f"Could not parse boxed answer, but found 'yes' in response")
            return True, response
        elif "no" in response_lower:
            logger.warning(f"Could not parse boxed answer, but found 'no' in response")
            return False, response
        else:
            logger.warning(f"Could not determine YES/NO from response, assuming correct")
            return True, response


def identify_error_step_incremental(manager, problem: str, chain: List[str], ground_truth: str, autonomy_level: int = 1, temperature: float = 0.3) -> Tuple[int, str]:
    """Incrementally verify each step to identify where the error occurred.

    Traverses the reasoning chain from top to bottom, verifying each step in context
    of previous steps until an error is found or the chain ends.

    Args:
        manager: Model manager
        problem: Original problem statement
        chain: List of reasoning steps
        ground_truth: Correct answer (used for verification and L1 prompting)
        autonomy_level: 1 (oracle), 2 (binary feedback), or 3 (full autonomy)
        temperature: Sampling temperature for error detection (default: 0.3)

    Returns:
        Tuple of (step_number, accumulated_reasoning)
    """

    logger.info("Using INCREMENTAL error detection: verifying each step sequentially...")

    accumulated_reasoning = []

    for current_step_idx in range(1, len(chain) + 1):
        logger.info(f"Verifying step {current_step_idx}/{len(chain)}...")

        # Verify this step in context of previous steps
        context_steps = chain[:current_step_idx]
        is_correct, step_reasoning = verify_single_step(
            manager, problem, context_steps, current_step_idx,
            ground_truth, autonomy_level, temperature
        )

        accumulated_reasoning.append(f"Step {current_step_idx} verification:\n{step_reasoning}")

        if not is_correct:
            # Found the error!
            logger.info(f"Error detected at step {current_step_idx}")
            full_reasoning = "\n\n".join(accumulated_reasoning)
            return current_step_idx, full_reasoning
        else:
            logger.info(f"Step {current_step_idx} verified correct, continuing...")

    # No error found in any step
    logger.info("All steps verified correct (no error detected)")
    full_reasoning = "\n\n".join(accumulated_reasoning)
    return 0, full_reasoning


def identify_error_step(manager, problem: str, chain: List[str], ground_truth: str, autonomy_level: int = 1, temperature: float = 0.3) -> Tuple[int, str]:
    """Ask model to identify which step contains the error with reasoning.

    Args:
        manager: Model manager
        problem: Original problem statement
        chain: List of reasoning steps
        ground_truth: Correct answer (used for verification and L1 prompting)
        autonomy_level: 1 (oracle), 2 (binary feedback), or 3 (full autonomy)
        temperature: Sampling temperature for error detection (default: 0.3)

    Returns:
        Tuple of (step_number, reasoning)
    """

    # Build chain representation
    chain_text = ""
    for i, step in enumerate(chain, 1):
        chain_text += f"\nStep {i}: {step}"

    if autonomy_level == 1:
        # L1: Oracle access - model sees correct answer.
        # Default: recommended localizer. Paper version: thought_ics.paper_prompts.
        prompt = recommended_prompts.localization_prompt(problem, chain, ground_truth=ground_truth)
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows it's wrong but not the answer.
        # Default: recommended (originating-cause) localizer. Paper version: thought_ics.paper_prompts.
        prompt = recommended_prompts.localization_prompt(problem, chain)
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must verify and identify errors
        prompt = f"""Problem: {problem}

Current reasoning chain:
{chain_text}

Carefully verify your reasoning chain step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), determine which step number (1 to {len(chain)}) contains the first critical error.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{step_number}} if you found an error
- \\boxed{{0}} if the reasoning is correct
"""

    logger.info("Asking model to identify error step with reasoning...")

    outputs = manager.generate(
        prompts=[prompt],
        temperature=temperature,
        top_p=0.9,
        top_k=50,
    )

    response = outputs[0].strip()
    logger.info(f"Model response: {response}")

    # Extract step number from boxed answer
    step_str = extract_boxed_answer(response)

    # Try to parse as integer
    try:
        step_num = int(step_str)
        if step_num == 0:
            logger.info("Model found no errors in the chain")
            return 0, response
        elif 1 <= step_num <= len(chain):
            logger.info(f"Identified error at step {step_num}")
            return step_num, response
    except (ValueError, TypeError):
        logger.warning(f"Could not parse step number from boxed answer '{step_str}'")

    # Fallback: try to find any number in response
    numbers = re.findall(r'\d+', response)
    if numbers:
        step_num = int(numbers[0])
        if 1 <= step_num <= len(chain):
            logger.warning(f"No valid boxed answer, using first number found: {step_num}")
            return step_num, response

    # Default to middle of chain if can't parse
    logger.warning(f"Could not parse step number from response, defaulting to middle")
    return max(1, len(chain) // 2), response


def identify_error_step_with_mv(
    manager,
    problem: str,
    chain: List[str],
    ground_truth: str,
    autonomy_level: int = 1,
    temperature: float = 0.5,
    mv_k: int = 10
) -> Tuple[int, str, List[Optional[int]]]:
    """Majority vote variant of identify_error_step.

    Generates mv_k samples and returns the majority vote step number along with
    all individual decisions for analysis.

    Args:
        manager: Model manager
        problem: Original problem statement
        chain: List of reasoning steps
        ground_truth: Correct answer (used for verification and L1 prompting)
        autonomy_level: 1 (oracle), 2 (binary feedback), or 3 (full autonomy)
        temperature: Sampling temperature for MV rollouts (default: 0.5)
        mv_k: Number of rollouts for majority vote (default: 10)

    Returns:
        Tuple of (mv_step_number, combined_reasoning, all_decisions)
        - mv_step_number: The majority vote step number
        - combined_reasoning: Combined text from all rollouts
        - all_decisions: List of all individual step decisions (for analysis)
    """
    from collections import Counter

    # Build chain representation
    chain_text = ""
    for i, step in enumerate(chain, 1):
        chain_text += f"\nStep {i}: {step}"

    # Build prompt based on autonomy level (same as identify_error_step)
    if autonomy_level == 1:
        prompt = f"""Problem: {problem}

Current reasoning chain (WRONG - got incorrect answer):
{chain_text}

The correct answer should be {ground_truth}.

Analyze the reasoning chain step by step to identify where the error occurred. Which step number (1 to {len(chain)}) contains the first critical error that led to the wrong answer?

Provide your reasoning, then conclude with the step number in the format: \\boxed{{step_number}}
"""
    elif autonomy_level == 2:
        prompt = f"""Problem: {problem}

Current reasoning chain (WRONG - got incorrect answer):
{chain_text}

Your answer is incorrect. Analyze the reasoning chain step by step to identify where the error occurred. Which step number (1 to {len(chain)}) contains the first critical error (logical flaw, arithmetic error, or incorrect assumption)?

Provide your reasoning, then conclude with the step number in the format: \\boxed{{step_number}}
"""
    else:  # autonomy_level == 3
        prompt = f"""Problem: {problem}

Current reasoning chain:
{chain_text}

Carefully verify your reasoning chain step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), determine which step number (1 to {len(chain)}) contains the first critical error.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{step_number}} if you found an error
- \\boxed{{0}} if the reasoning is correct
"""

    logger.info(f"MV Localization: generating {mv_k} rollouts at temperature {temperature}...")

    # Generate mv_k rollouts using vLLM's native n parameter
    outputs = manager.generate(
        prompts=[prompt],
        n=mv_k,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
    )

    # Parse all decisions
    all_decisions = []
    all_reasonings = []

    for i, response in enumerate(outputs):
        response = response.strip()
        all_reasonings.append(f"--- Rollout {i+1} ---\n{response}")

        # Extract step number from boxed answer
        step_str = extract_boxed_answer(response)

        # Try to parse as integer
        try:
            step_num = int(step_str)
            if step_num == 0:
                all_decisions.append(0)
            elif 1 <= step_num <= len(chain):
                all_decisions.append(step_num)
            else:
                # Out of range - fallback to finding any number
                numbers = re.findall(r'\d+', response)
                if numbers:
                    fallback = int(numbers[0])
                    if 1 <= fallback <= len(chain):
                        all_decisions.append(fallback)
                    else:
                        all_decisions.append(None)
                else:
                    all_decisions.append(None)
        except (ValueError, TypeError):
            # Fallback: try to find any number in response
            numbers = re.findall(r'\d+', response)
            if numbers:
                fallback = int(numbers[0])
                if 0 <= fallback <= len(chain):
                    all_decisions.append(fallback)
                else:
                    all_decisions.append(None)
            else:
                all_decisions.append(None)

    # Compute majority vote (filter out None values)
    valid_decisions = [d for d in all_decisions if d is not None]

    if not valid_decisions:
        logger.warning("MV Localization: No valid decisions parsed, defaulting to middle of chain")
        mv_step = max(1, len(chain) // 2)
    else:
        counter = Counter(valid_decisions)
        mv_step = counter.most_common(1)[0][0]

    # Log distribution
    if valid_decisions:
        counter = Counter(valid_decisions)
        logger.info(f"MV Localization: decisions={all_decisions}, distribution={dict(counter)}, mv_step={mv_step}")
    else:
        logger.info(f"MV Localization: all decisions failed to parse, using fallback mv_step={mv_step}")

    combined_reasoning = f"MV Localization (k={mv_k}, temp={temperature}): decisions={all_decisions}, mv_step={mv_step}\n\n" + "\n\n".join(all_reasonings)

    return mv_step, combined_reasoning, all_decisions


def verify_solution_correctness(manager, problem: str, chain: List[str], temperature: float = 0.3,
                                 mv_verify: bool = False, mv_k: int = 5,
                                 mv_criterion: str = "unanimous") -> Tuple[bool, str]:
    """Ask model directly if it thinks its final answer is correct.

    Args:
        manager: Model manager
        problem: Original problem statement
        chain: List of reasoning steps
        temperature: Sampling temperature for verification (default: 0.3)
        mv_verify: If True, use majority vote with k rollouts (default: False)
        mv_k: Number of rollouts for majority vote verification (default: 5)
        mv_criterion: Voting criterion - "unanimous" (all YES), "majority" (>50% YES), "any" (>=1 YES)

    Returns:
        Tuple of (believes_correct, reasoning)
        With mv_verify=True, believes_correct depends on mv_criterion
    """
    # Build chain representation
    chain_text = "\n".join(chain)
    answer = extract_boxed_answer(chain[-1] if chain else "")

    prompt = f"""You are reviewing a solution to a problem. Analyze it carefully to see if they arrived at the right answer.

Problem: {problem}

Solution to review:
{chain_text}

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


def generate_from_prefix(manager, problem: str, prefix: List[str], previous_chain: Optional[List[str]] = None, error_reasoning: Optional[str] = None, error_step: Optional[int] = None, temperature: float = 0.7) -> List[str]:
    """Generate new chain from a given prefix.

    Args:
        manager: Model manager
        problem: Problem statement
        prefix: Prefix of correct reasoning steps
        previous_chain: With --context, the previous chain that had an error (optional)
        error_reasoning: With --context, the error analysis from the previous attempt (optional)
        error_step: With --context, which step had the error (optional)
        temperature: Sampling temperature for regeneration (default: 0.7)
    """

    # Build prompt starting with problem
    prompt = problem

    # Historical-context conditioning: prepend the prior failed attempt + error analysis
    if previous_chain is not None and error_reasoning is not None and error_step is not None:
        logger.info(f"Regenerating from prefix of {len(prefix)} steps with historical context...")

        # Add full historical context before the prefix
        prompt += f"\n\n### Previous Failed Attempt\n"
        prompt += f"The following reasoning chain led to an incorrect answer:\n"
        for step in previous_chain:
            prompt += f"\n{step}"
        prompt += f"\n\n### Error Analysis\n"
        prompt += f"{error_reasoning}\n"
        prompt += f"\nNow let's try again with the correct approach:\n"

    # Add the prefix (correct steps to continue from)
    if prefix:
        for step in prefix:
            prompt += f"\n\n{step}"
        if previous_chain is None or error_reasoning is None:
            logger.info(f"Regenerating from prefix of {len(prefix)} steps...")
    else:
        if previous_chain is None or error_reasoning is None:
            logger.info("Regenerating from scratch...")

    agent = ToTAgent(manager, temperature=temperature, max_tokens=150)
    env = ToTEnvironment(max_depth=20)
    search = TreeSearch(agent, env, strategy="dfs", n_rollouts=1)

    root = search.search(prompt, verbose=False)
    completed_paths = get_completed_paths(root)

    if not completed_paths:
        logger.warning("No completed paths found during regeneration!")
        return prefix

    # Get new thoughts (skip the question)
    all_thoughts = completed_paths[0][1:]

    # The first len(prefix) thoughts are from the prefix
    # We want to return: prefix + new thoughts
    new_thoughts = all_thoughts[len(prefix):]
    full_chain = prefix + new_thoughts

    answer = extract_boxed_answer(full_chain[-1] if full_chain else "")
    logger.info(f"Generated new chain with {len(full_chain)} total steps ({len(new_thoughts)} new), answer: {answer}")

    return full_chain


def iterative_self_correction(manager, problem: str, ground_truth: str, L: int = 10, autonomy_level: int = 1, error_detection_method: str = 'batch', shared_prefix: bool = True, generation_temp: float = 1.0, resample_temp: float = 0.7, judge_temp: float = 0.3, no_auto_stop: bool = False, use_context: bool = False, verify: bool = False, mv_verify: bool = False, mv_k: int = 5, mv_criterion: str = "unanimous") -> Dict:
    """Run iterative self-correction for L iterations.

    Args:
        manager: Model manager
        problem: Problem statement
        ground_truth: Correct answer
        L: Maximum number of correction iterations
        autonomy_level: 1 (oracle), 2 (binary feedback), or 3 (full autonomy)
        error_detection_method: 'batch' (default, single-pass) or 'incremental' (step-by-step verification)
        shared_prefix: Whether to preserve correct prefix when regenerating (default: True)
        generation_temp: Temperature for initial chain generation (default: 1.0)
        resample_temp: Temperature for correction/regeneration (default: 0.7)
        judge_temp: Temperature for error detection/verification (default: 0.3)
    """

    autonomy_names = {1: "L1 (Oracle)", 2: "L2 (Binary Feedback)", 3: "L3 (Full Autonomy)"}

    logger.info("="*100)
    logger.info("ITERATIVE SELF-CORRECTION PIPELINE")
    logger.info("="*100)
    logger.info(f"Problem: {problem[:150]}...")
    logger.info(f"Ground truth answer: {ground_truth}")
    logger.info(f"Max iterations: {L}")
    logger.info(f"Autonomy level: {autonomy_names.get(autonomy_level, f'L{autonomy_level}')}")
    logger.info(f"Error detection method: {error_detection_method}")
    logger.info(f"Shared prefix: {shared_prefix}")
    logger.info("="*100)

    iterations = []

    # Generate initial chain
    chain = generate_full_chain(manager, problem, temperature=generation_temp)
    answer = extract_boxed_answer(chain[-1] if chain else "")

    iterations.append({
        'iteration': 0,
        'chain': chain,
        'answer': answer,
        'correct': normalize_answer(answer) == normalize_answer(ground_truth),
        'error_step': None,
        'error_reasoning': None,
        'verify_reasoning': None,
        'model_believes_correct': None,
        'prefix_length': None
    })

    logger.info(f"\nIteration 0: Answer = {answer}, Correct = {normalize_answer(answer) == normalize_answer(ground_truth)}")

    # Track previous chain for historical context (if enabled)
    previous_chain = None
    previous_error_reasoning = None

    # Iterative correction
    for i in range(1, L + 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"ITERATION {i}")
        logger.info(f"{'='*100}")

        # Check if we got it right
        if not no_auto_stop and normalize_answer(answer) == normalize_answer(ground_truth):
            logger.info(f"SUCCESS! Correct answer found at iteration {i-1}")
            break

        # Optional verification: ask model if it thinks answer is correct
        # Initialize verification tracking variables
        iter_verify_reasoning = None
        iter_model_believes_correct = None

        if verify:
            believes_correct, verify_reasoning = verify_solution_correctness(
                manager, problem, chain, temperature=judge_temp,
                mv_verify=mv_verify, mv_k=mv_k, mv_criterion=mv_criterion
            )
            is_actually_correct = normalize_answer(answer) == normalize_answer(ground_truth)
            logger.info(f"Verification result: model_believes_correct={believes_correct}, actually_correct={is_actually_correct}")

            # Store for inclusion in iteration data
            iter_verify_reasoning = verify_reasoning
            iter_model_believes_correct = believes_correct

            if believes_correct:
                logger.info(f"Model believes answer is correct - stopping iteration.")
                iterations.append({
                    'iteration': i,
                    'chain': chain,
                    'answer': answer,
                    'correct': is_actually_correct,
                    'error_step': None,
                    'error_reasoning': None,
                    'verify_reasoning': verify_reasoning,
                    'model_believes_correct': True,
                    'prefix_length': None
                })
                break
            else:
                logger.info(f"Model believes answer is incorrect - continuing to error detection.")

        # Identify error step using selected method
        if error_detection_method == 'incremental':
            error_step, error_reasoning = identify_error_step_incremental(manager, problem, chain, ground_truth, autonomy_level, judge_temp)
        else:  # default: 'batch'
            error_step, error_reasoning = identify_error_step(manager, problem, chain, ground_truth, autonomy_level, judge_temp)

        # Check if model found no errors
        if error_step == 0:
            is_correct = normalize_answer(answer) == normalize_answer(ground_truth)
            logger.info(f"Model found no errors - stopping iteration. Answer correct: {is_correct}")
            iterations.append({
                'iteration': i,
                'chain': chain,
                'answer': answer,
                'correct': is_correct,
                'error_step': 0,
                'error_reasoning': error_reasoning,
                'verify_reasoning': iter_verify_reasoning,
                'model_believes_correct': iter_model_believes_correct,
                'prefix_length': None
            })
            break

        # Generate new chain from before error
        if shared_prefix:
            prefix = chain[:error_step - 1]  # Steps before the error
            logger.info(f"Backtracking to step {error_step-1}, keeping {len(prefix)} steps as prefix")
        else:
            prefix = []  # Force full regeneration from scratch
            logger.info(f"Error at step {error_step}, regenerating entire solution from scratch (no shared prefix)")

        # Store the chain we're moving away from (if historical context enabled)
        if use_context:
            previous_chain = chain
            previous_error_reasoning = error_reasoning

        # Regenerate (with historical context if enabled)
        if use_context and previous_chain is not None:
            chain = generate_from_prefix(manager, problem, prefix,
                                        previous_chain=previous_chain,
                                        error_reasoning=previous_error_reasoning,
                                        error_step=error_step,
                                        temperature=resample_temp)
        else:
            chain = generate_from_prefix(manager, problem, prefix, temperature=resample_temp)

        answer = extract_boxed_answer(chain[-1] if chain else "")

        iterations.append({
            'iteration': i,
            'chain': chain,
            'answer': answer,
            'correct': normalize_answer(answer) == normalize_answer(ground_truth),
            'error_step': error_step,
            'error_reasoning': error_reasoning,
            'verify_reasoning': iter_verify_reasoning,
            'model_believes_correct': iter_model_believes_correct,
            'prefix_length': len(prefix)
        })

        logger.info(f"\nIteration {i}: Answer = {answer}, Correct = {normalize_answer(answer) == normalize_answer(ground_truth)}")

        if not no_auto_stop and normalize_answer(answer) == normalize_answer(ground_truth):
            logger.info(f"SUCCESS! Correct answer found at iteration {i}")
            break

    # Summary
    logger.info(f"\n{'='*100}")
    logger.info("SUMMARY")
    logger.info(f"{'='*100}")

    for it in iterations:
        status = "✓ CORRECT" if it['correct'] else "✗ WRONG"
        logger.info(f"Iteration {it['iteration']}: {it['answer']} {status}")

    final_correct = iterations[-1]['correct']
    logger.info(f"\nFinal result: {'SUCCESS' if final_correct else 'FAILED'}")
    logger.info(f"Iterations used: {len(iterations)}")

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'iterations': iterations,
        'success': final_correct,
        'total_iterations': len(iterations)
    }


def main():
    parser = argparse.ArgumentParser(description='Iterative Self-Correction Pipeline')
    parser.add_argument('--autonomy-level', type=int, choices=[1, 2, 3], default=1,
                        help='Autonomy level: 1=Oracle, 2=Binary Feedback, 3=Full Autonomy '
                             '(add --context for historical-context conditioning at any level)')
    parser.add_argument('--error-detection', type=str, choices=['batch', 'incremental'], default='batch',
                        help='Error detection method: batch=single-pass (default), incremental=step-by-step verification')
    parser.add_argument('--no-shared-prefix', action='store_true',
                        help='Disable shared prefix - regenerate entire solution from scratch instead of preserving correct steps')
    parser.add_argument('--gpus', type=str, default="1",
                        help='Comma-separated GPU IDs (e.g., "0,1" for multi-GPU)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism (default: 1)')
    parser.add_argument('--model', type=str, default='llama8b',
                        choices=['llama8b', 'llama70b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen2b', 'llama3b', 'phi4b'],
                        help='Model to use (default: llama8b)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Maximum number of correction iterations (default: 10)')
    parser.add_argument('--verify', action='store_true',
                        help='Enable solution verification before error detection (ask model if it thinks answer is correct)')
    parser.add_argument('--context', action='store_true',
                        help='Condition the resample on the prior iteration\'s failed chain and error '
                             'analysis (instead of a fresh stochastic sample). Combinable with any --autonomy-level.')
    args = parser.parse_args()

    # Test on Problem 7: Partial fractions problem
    logger.info("Testing iterative self-correction on Problem 7")

    problem = r"Find the product $CD$ of the integers $C$ and $D$ for which \[\frac{C}{x-3}+\frac{D}{x+8}=\frac{4x-23}{x^2+5x-24}\]for all real values of $x$ except $-8$ and $3$."
    ground_truth = "-5"

    # Initialize model
    logger.info(f"Initializing model '{args.model}' on GPUs {args.gpus}...")
    manager = initialize_model(gpu_ids=args.gpus, tensor_parallel_size=args.tensor_parallel_size, model_name=args.model)

    # Run pipeline
    result = iterative_self_correction(manager, problem, ground_truth, L=args.max_iterations,
                                      autonomy_level=args.autonomy_level,
                                      error_detection_method=args.error_detection,
                                      shared_prefix=not args.no_shared_prefix,
                                      use_context=args.context,
                                      verify=args.verify)

    # Save results
    level_suffix = f"_L{args.autonomy_level}"
    detection_suffix = f"_{args.error_detection}" if args.error_detection != 'batch' else ""
    prefix_suffix = "_no_shared_prefix" if args.no_shared_prefix else ""
    output_file = f"iterative_correction_result{level_suffix}{detection_suffix}{prefix_suffix}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Cleanup
    manager.unload_base_model()


if __name__ == "__main__":
    main()
