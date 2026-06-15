#!/usr/bin/env python3
"""
Self-Refine: Iterative Refinement with Self-Feedback (Madaan et al., 2023)

Implementation of the Self-Refine algorithm for mathematical reasoning tasks.
The model iteratively generates feedback on its own solution and refines it.

Algorithm:
1. Generate initial solution: y0 = M(problem)
2. For t = 0 to max_iter-1:
   a. Generate feedback: fb_t = M(problem, y_t)
   b. If feedback indicates correct, return y_t
   c. Refine: y_{t+1} = M(problem, y_t, fb_t)
3. Return final solution

Reference: https://arxiv.org/abs/2303.17651
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS FOR SELF-REFINE
# =============================================================================

# Initial generation prompt (standard CoT)
GENERATION_PROMPT = """Solve the following problem step by step. Show your reasoning clearly and put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

# Feedback prompt - ask model to evaluate its own solution
FEEDBACK_PROMPT = """You are reviewing a solution to a math problem. Analyze it carefully for errors.

Problem: {problem}

Solution to review:
{solution}

Provide feedback on this solution:
1. Is the solution correct? Answer YES or NO at the start.
2. If NO, identify the specific error(s) and explain what went wrong.
3. If YES, confirm the solution is correct.

Feedback:"""

# Refinement prompt - improve solution based on feedback
REFINE_PROMPT = """You previously attempted to solve a problem but received feedback indicating errors. Use the feedback to produce a corrected solution.

Problem: {problem}

Your previous solution:
{solution}

Feedback on your solution:
{feedback}

Now provide a corrected solution. Show your reasoning step by step and put your final answer in \\boxed{{}}.

Corrected Solution:"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SelfRefineResult:
    """Result of self-refine on a single problem."""
    problem: str
    initial_solution: str
    final_solution: str
    iterations: int
    history: List[Dict[str, str]] = field(default_factory=list)  # List of {solution, feedback}
    stopped_early: bool = False  # Whether stopped due to positive feedback


# =============================================================================
# CORE SELF-REFINE FUNCTIONS
# =============================================================================

def parse_feedback(feedback: str) -> Tuple[bool, str]:
    """Parse feedback to determine if solution is deemed correct.

    Returns:
        Tuple of (is_correct, reasoning)
    """
    feedback_lower = feedback.lower().strip()

    # Check for explicit YES/NO at the start
    if feedback_lower.startswith("yes"):
        return True, "Feedback indicates solution is correct"
    if feedback_lower.startswith("no"):
        return False, "Feedback indicates errors in solution"

    # Check for other positive indicators
    positive_patterns = [
        r"the solution is correct",
        r"this is correct",
        r"the answer is correct",
        r"correct solution",
        r"no errors",
        r"no issues",
        r"well done",
        r"correct!",
    ]

    for pattern in positive_patterns:
        if re.search(pattern, feedback_lower):
            return True, f"Matched positive pattern: {pattern}"

    # Check for negative indicators
    negative_patterns = [
        r"incorrect",
        r"error",
        r"mistake",
        r"wrong",
        r"should be",
        r"instead of",
        r"fix",
        r"correct this",
    ]

    for pattern in negative_patterns:
        if re.search(pattern, feedback_lower):
            return False, f"Matched negative pattern: {pattern}"

    # Default to assuming refinement is needed if unclear
    return False, "Unclear feedback, assuming refinement needed"


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


def self_refine_single(
    problem: str,
    manager,
    max_iterations: int = 4,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    cached_initial_solution: Optional[str] = None
) -> SelfRefineResult:
    """
    Run Self-Refine on a single problem.

    Args:
        problem: The problem text
        manager: Model manager with generate() method
        max_iterations: Maximum number of refinement iterations
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        cached_initial_solution: Optional pre-computed initial solution from CoT cache

    Returns:
        SelfRefineResult with solution history and final answer
    """
    history = []

    # Step 1: Generate initial solution (or use cached)
    if cached_initial_solution is not None:
        current_solution = cached_initial_solution.strip()
        logger.debug(f"Using cached initial solution: {current_solution[:200]}...")
    else:
        gen_prompt = GENERATION_PROMPT.format(problem=problem)
        outputs = manager.generate(
            prompts=[gen_prompt],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=50
        )
        current_solution = outputs[0].strip()
        logger.debug(f"Generated initial solution: {current_solution[:200]}...")

    # Iterative refinement loop
    for iteration in range(max_iterations):
        # Step 2a: Generate feedback
        fb_prompt = FEEDBACK_PROMPT.format(
            problem=problem,
            solution=current_solution
        )
        fb_outputs = manager.generate(
            prompts=[fb_prompt],
            temperature=temperature,
            max_tokens=1024,
            top_p=0.9,
            top_k=50
        )
        feedback = fb_outputs[0].strip()

        logger.debug(f"Iteration {iteration}: Feedback: {feedback[:200]}...")

        # Record history
        history.append({
            "iteration": iteration,
            "solution": current_solution,
            "feedback": feedback
        })

        # Step 2b: Check if feedback indicates correct solution
        is_correct, reason = parse_feedback(feedback)
        if is_correct:
            logger.info(f"Self-Refine stopped at iteration {iteration}: {reason}")
            return SelfRefineResult(
                problem=problem,
                initial_solution=history[0]["solution"] if history else current_solution,
                final_solution=current_solution,
                iterations=iteration + 1,
                history=history,
                stopped_early=True
            )

        # Step 2c: Refine based on feedback
        refine_prompt = REFINE_PROMPT.format(
            problem=problem,
            solution=current_solution,
            feedback=feedback
        )
        refine_outputs = manager.generate(
            prompts=[refine_prompt],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=50
        )
        current_solution = refine_outputs[0].strip()

        logger.debug(f"Iteration {iteration}: Refined solution: {current_solution[:200]}...")

    # Return final result after max iterations
    return SelfRefineResult(
        problem=problem,
        initial_solution=history[0]["solution"] if history else current_solution,
        final_solution=current_solution,
        iterations=max_iterations,
        history=history,
        stopped_early=False
    )


def self_refine_batch(
    problems: List[Dict[str, Any]],
    manager,
    max_iterations: int = 4,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    verbose: bool = True,
    cached_solutions: Optional[List[Dict[str, Any]]] = None
) -> List[SelfRefineResult]:
    """
    Run Self-Refine on a batch of problems.

    Note: This processes problems sequentially since each problem requires
    multiple rounds of generation. For efficiency, consider parallelizing
    at the SLURM job level.

    Args:
        problems: List of problem dicts with 'problem' key
        manager: Model manager with generate() method
        max_iterations: Maximum refinement iterations per problem
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        verbose: Whether to show progress
        cached_solutions: Optional list of cached CoT solutions (from chain_cache)
                         Each dict should have 'solution' key with initial solution text

    Returns:
        List of SelfRefineResult objects
    """
    from tqdm import tqdm

    results = []
    iterator = enumerate(tqdm(problems, desc="Self-Refine") if verbose else problems)

    for idx, prob_dict in iterator:
        problem = prob_dict.get("problem", prob_dict.get("question", ""))

        # Get cached initial solution if available
        cached_initial = None
        if cached_solutions is not None and idx < len(cached_solutions):
            cached_initial = cached_solutions[idx].get("solution")

        result = self_refine_single(
            problem=problem,
            manager=manager,
            max_iterations=max_iterations,
            temperature=temperature,
            max_tokens=max_tokens,
            cached_initial_solution=cached_initial
        )
        results.append(result)

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_final_answer(result: SelfRefineResult) -> str:
    """Extract the final boxed answer from a SelfRefineResult."""
    return extract_boxed_answer(result.final_solution)


def compute_self_refine_metrics(
    results: List[SelfRefineResult],
    ground_truths: List[str],
    normalize_fn=None
) -> Dict[str, Any]:
    """
    Compute metrics for Self-Refine results.

    Args:
        results: List of SelfRefineResult objects
        ground_truths: List of ground truth answers
        normalize_fn: Optional function to normalize answers for comparison

    Returns:
        Dictionary with accuracy, avg iterations, etc.
    """
    if normalize_fn is None:
        normalize_fn = lambda x: x.strip().lower()

    correct = 0
    total_iterations = 0
    early_stops = 0

    for result, gt in zip(results, ground_truths):
        pred = get_final_answer(result)
        pred_norm = normalize_fn(pred)
        gt_norm = normalize_fn(gt)

        if pred_norm == gt_norm:
            correct += 1

        total_iterations += result.iterations
        if result.stopped_early:
            early_stops += 1

    n = len(results)
    return {
        "accuracy": correct / n if n > 0 else 0,
        "correct": correct,
        "total": n,
        "avg_iterations": total_iterations / n if n > 0 else 0,
        "early_stop_rate": early_stops / n if n > 0 else 0,
        "early_stops": early_stops
    }


if __name__ == "__main__":
    # Simple test
    print("Self-Refine module loaded successfully.")
    print("Use self_refine_single() or self_refine_batch() with a model manager.")
