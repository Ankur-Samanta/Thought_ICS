#!/usr/bin/env python3
"""
Token-Level Self-Correction for CoT

This script implements self-correction at the token/character level,
similar to how ToT does step-level correction, but operating on the
continuous text of a CoT solution.

Key idea:
1. Generate initial CoT solution
2. Model identifies WHERE the error occurred (quotes the erroneous text)
3. Truncate solution at that point (find the quoted text)
4. Continue generation from the truncated prefix

This provides a fair comparison to ToT's step-level correction by giving
CoT the same capability to preserve correct work and regenerate from error points.
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

from thought_ics.thought_mdp import initialize_model
from thought_ics.datasets import load_dataset_by_name
from thought_ics.self_correction import extract_boxed_answer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_cot_solution(manager, problem: str, max_tokens: int = 2048, temperature: float = 1.0) -> str:
    """Generate a standard CoT solution."""

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


def identify_error_location(
    manager,
    problem: str,
    solution: str,
    ground_truth: str,
    autonomy_level: int = 3
) -> Tuple[Optional[str], str]:
    """Ask model to identify WHERE the error occurred by quoting the text.

    Args:
        manager: Model manager
        problem: Original problem
        solution: Current solution text
        ground_truth: Correct answer
        autonomy_level: 2 (binary feedback) or 3 (autonomous)

    Returns:
        Tuple of (quoted_error_text, reasoning)
        quoted_error_text is None if no error found
    """

    if autonomy_level == 2:
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

    logger.info("Asking model to identify error location...")

    outputs = manager.generate(
        prompts=[prompt],
        temperature=0.3,
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
        # Extract the quoted text
        quote_match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed)
        if quote_match:
            quoted_text = quote_match.group(1)
            logger.info(f"Extracted error quote: {quoted_text[:100]}...")
            return quoted_text, response

    # Fallback: try to find any quoted text in the response
    quote_matches = re.findall(r'"([^"]{20,})"', response)
    if quote_matches:
        # Use the first substantial quote
        quoted_text = quote_matches[0]
        logger.info(f"Found quoted text (fallback): {quoted_text[:100]}...")
        return quoted_text, response

    logger.warning("Could not extract error quote from response")
    return None, response


def find_truncation_point(solution: str, error_quote: str) -> Optional[int]:
    """Find where to truncate the solution based on the error quote.

    Uses only exact matching (exact or case-insensitive). Returns None if not found,
    which signals that we should regenerate from scratch instead of truncating.

    Args:
        solution: Full solution text
        error_quote: Quoted text where error occurs

    Returns:
        Character index to truncate at, or None if not found
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
    """Continue generation from a prefix.

    Args:
        manager: Model manager
        problem: Original problem
        prefix: The correct prefix to continue from
        max_tokens: Max tokens to generate
        temperature: Generation temperature

    Returns:
        Full solution (prefix + continuation)
    """

    # Build prompt: problem + prefix
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

    # The model might repeat some of the prefix, or it might just continue
    # We'll take the full output and return it
    return prefix + continuation


def token_level_self_correction(
    manager,
    problem: str,
    ground_truth: str,
    max_iterations: int = 10,
    autonomy_level: int = 3,
    initial_solution: Optional[str] = None,
    temperature: float = 1.0
) -> Dict:
    """Run token-level iterative self-correction.

    Args:
        manager: Model manager
        problem: Problem statement
        ground_truth: Correct answer
        max_iterations: Max correction iterations
        autonomy_level: 2 (binary) or 3 (autonomous)
        initial_solution: Optional initial solution to start from
        temperature: Generation temperature

    Returns:
        Dict with iteration history and results
    """

    logger.info(f"Starting token-level self-correction (L{autonomy_level})")

    iterations = []

    # Initial solution
    if initial_solution is not None:
        logger.info("Using provided initial solution")
        solution = initial_solution
    else:
        logger.info("Generating initial solution")
        solution = generate_cot_solution(manager, problem, temperature=temperature)

    answer = extract_boxed_answer(solution)
    correct = answer == ground_truth

    iterations.append({
        'iteration': 0,
        'solution': solution,
        'answer': answer,
        'correct': correct,
        'truncation_point': None,
        'error_quote': None
    })

    logger.info(f"Iteration 0: Answer = {answer}, Correct = {correct}")

    # Iterative correction
    for i in range(1, max_iterations + 1):
        if correct:
            logger.info(f"SUCCESS! Correct answer at iteration {i-1}")
            break

        logger.info(f"\nIteration {i}: Identifying error location...")

        # Identify where the error occurred
        error_quote, error_reasoning = identify_error_location(
            manager, problem, solution, ground_truth, autonomy_level
        )

        if error_quote is None:
            logger.warning("Model found no error but answer is wrong. Cannot proceed.")
            break

        # Find truncation point
        truncation_idx = find_truncation_point(solution, error_quote)

        if truncation_idx is None:
            # Could not find exact quote - regenerate entire solution from scratch
            logger.info("Quote not found - regenerating entire solution from scratch...")
            solution = generate_cot_solution(manager, problem, temperature=0.7)
        else:
            # Found quote - truncate at error point and continue from prefix
            prefix = solution[:truncation_idx].rstrip()
            logger.info(f"Truncating at position {truncation_idx}, prefix length: {len(prefix)}")
            logger.info("Regenerating from prefix...")
            solution = continue_from_prefix(manager, problem, prefix, temperature=0.7)

        answer = extract_boxed_answer(solution)
        correct = answer == ground_truth

        iterations.append({
            'iteration': i,
            'solution': solution,
            'answer': answer,
            'correct': correct,
            'truncation_point': truncation_idx,
            'error_quote': error_quote,
            'error_reasoning': error_reasoning
        })

        logger.info(f"Iteration {i}: Answer = {answer}, Correct = {correct}")

        if correct:
            logger.info(f"SUCCESS! Correct answer at iteration {i}")
            break

    return {
        'problem': problem,
        'ground_truth': ground_truth,
        'iterations': iterations,
        'success': correct,
        'total_iterations': len(iterations)
    }


def test_simple_examples(manager):
    """Test on simple examples with known errors."""

    logger.info("\n" + "="*80)
    logger.info("TESTING ON SIMPLE EXAMPLES")
    logger.info("="*80)

    # Example 1: Simple arithmetic error
    examples = [
        {
            'problem': 'What is 15 + 27?',
            'answer': '42',
            'initial_solution': """Let me solve this step by step.

First, I'll add the ones place: 5 + 7 = 12
So I write down 2 and carry 1.

Then, I'll add the tens place: 1 + 2 = 3, plus the carried 1 = 4.

Wait, let me recalculate: 15 + 27...
Actually, 5 + 7 = 13, not 12. Let me redo this.

Actually, I think the answer is 41.

Therefore, 15 + 27 = \\boxed{41}"""
        },
        {
            'problem': 'If a rectangle has length 8 and width 5, what is its area?',
            'answer': '40',
            'initial_solution': """To find the area of a rectangle, I use the formula: Area = length × width

The length is 8 and the width is 5.

So the area = 8 + 5 = 13

Therefore, the area is \\boxed{13}"""
        }
    ]

    results = []
    for idx, example in enumerate(examples, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Example {idx}/{len(examples)}")
        logger.info(f"{'='*60}")

        result = token_level_self_correction(
            manager=manager,
            problem=example['problem'],
            ground_truth=example['answer'],
            max_iterations=5,
            autonomy_level=2,  # Binary feedback for testing
            initial_solution=example['initial_solution']
        )

        results.append(result)

        logger.info(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        logger.info(f"Total iterations: {result['total_iterations']}")

    return results


def test_amc_problems(manager, n_problems: int = 5):
    """Test on AMC problems."""

    logger.info("\n" + "="*80)
    logger.info("TESTING ON AMC PROBLEMS")
    logger.info("="*80)

    # Load AMC problems
    problems = load_dataset_by_name(
        dataset_name='amc23',
        n_problems=n_problems,
        seed=42
    )

    results = []
    for idx, item in enumerate(problems, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"AMC Problem {idx}/{len(problems)}")
        logger.info(f"{'='*60}")
        logger.info(f"Problem: {item['problem'][:100]}...")

        result = token_level_self_correction(
            manager=manager,
            problem=item['problem'],
            ground_truth=item['answer'],
            max_iterations=10,
            autonomy_level=3,  # Autonomous
            initial_solution=None,
            temperature=1.0
        )

        results.append(result)

        logger.info(f"\nResult: {'SUCCESS' if result['success'] else 'FAILED'}")
        logger.info(f"Final answer: {result['iterations'][-1]['answer']}")
        logger.info(f"Ground truth: {item['answer']}")

    # Summary stats
    successful = sum(1 for r in results if r['success'])
    logger.info(f"\n{'='*80}")
    logger.info(f"AMC RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total problems: {len(results)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Success rate: {successful/len(results)*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Token-Level Self-Correction for CoT')
    parser.add_argument('--mode', type=str, choices=['simple', 'amc', 'both'], default='both',
                        help='Test mode: simple examples, AMC problems, or both')
    parser.add_argument('--n-problems', type=int, default=5,
                        help='Number of AMC problems to test (default: 5)')
    parser.add_argument('--model', type=str, default='llama8b',
                        help='Model to use (default: llama8b)')
    parser.add_argument('--gpu', type=str, default='7',
                        help='GPU ID to use (default: 7)')
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Output directory for results')

    args = parser.parse_args()

    # Initialize model
    logger.info(f"Initializing model '{args.model}' on GPU {args.gpu}...")
    manager = initialize_model(gpu_ids=args.gpu, tensor_parallel_size=1, model_name=args.model)

    # Run tests
    all_results = {}

    if args.mode in ['simple', 'both']:
        simple_results = test_simple_examples(manager)
        all_results['simple_examples'] = simple_results

    if args.mode in ['amc', 'both']:
        amc_results = test_amc_problems(manager, n_problems=args.n_problems)
        all_results['amc_problems'] = amc_results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"token_level_correction_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*80}")

    # Cleanup
    manager.unload_base_model()


if __name__ == "__main__":
    main()
