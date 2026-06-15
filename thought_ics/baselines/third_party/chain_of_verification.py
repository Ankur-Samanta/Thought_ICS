#!/usr/bin/env python3
"""
Chain-of-Verification (CoVe): Reduces Hallucination via Verification Questions
(Dhuliawala et al., 2023)

Implementation of the CoVe algorithm for mathematical reasoning tasks.
The model generates verification questions about its baseline response,
answers them independently, and uses the verified information to produce
a final refined response.

Algorithm (Factored Approach):
1. Generate baseline response: R_base = M(question)
2. Plan verification questions: Q_1...Q_n = M(question, R_base)
3. Execute verifications independently: A_i = M(Q_i) for each i
4. Generate final verified response: R_final = M(question, R_base, Q_1...Q_n, A_1...A_n)

Reference: https://arxiv.org/abs/2309.11495
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# PROMPTS FOR CHAIN-OF-VERIFICATION
# =============================================================================

# Baseline generation prompt (standard CoT)
BASELINE_PROMPT = """Solve the following problem step by step. Show your reasoning clearly and put your final answer in \\boxed{{}}.

Problem: {problem}

Solution:"""

# Plan verification questions prompt
PLAN_VERIFICATION_PROMPT = """You have provided a solution to a math problem. Now generate verification questions to check if your solution is correct.

Problem: {problem}

Your solution:
{baseline}

Generate 2-4 specific verification questions that would help verify the correctness of this solution. Each question should check a specific step, calculation, or reasoning in the solution.

Format your questions as a numbered list:
1. [First verification question]
2. [Second verification question]
...

Verification Questions:"""

# Execute verification prompt (factored - independent context)
EXECUTE_VERIFICATION_PROMPT = """Answer the following question carefully and precisely.

Question: {question}

Answer:"""

# Final verified response prompt
FINAL_RESPONSE_PROMPT = """You previously solved a problem and generated verification questions with answers. Use this verified information to produce a final, corrected solution.

Problem: {problem}

Your baseline solution:
{baseline}

Verification questions and answers:
{verifications}

Based on the verification results, provide your final solution. If the verifications revealed any errors, correct them. Show your reasoning step by step and put your final answer in \\boxed{{}}.

Final Solution:"""


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class VerificationQA:
    """A verification question and its answer."""
    question: str
    answer: str


@dataclass
class CoVeResult:
    """Result of Chain-of-Verification on a single problem."""
    problem: str
    baseline_response: str
    verification_questions: List[str]
    verification_answers: List[str]
    final_response: str
    num_verifications: int


# =============================================================================
# CORE CHAIN-OF-VERIFICATION FUNCTIONS
# =============================================================================

def parse_verification_questions(text: str) -> List[str]:
    """Parse numbered list of verification questions from model output.

    Args:
        text: Model output containing numbered questions

    Returns:
        List of question strings
    """
    questions = []

    # Try to match numbered patterns like "1.", "1)", "1:", etc.
    lines = text.strip().split('\n')

    for line in lines:
        line = line.strip()
        # Match patterns like "1. question" or "1) question" or "- question"
        match = re.match(r'^(?:\d+[.\):]|\-|\*)\s*(.+)$', line)
        if match:
            question = match.group(1).strip()
            if question and len(question) > 5:  # Filter out very short lines
                questions.append(question)

    # If no numbered list found, try to split by double newlines
    if not questions:
        paragraphs = text.strip().split('\n\n')
        for para in paragraphs:
            para = para.strip()
            if para and '?' in para:
                questions.append(para)

    # Limit to 4 questions max
    return questions[:4]


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


def cove_single(
    problem: str,
    manager,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    max_verifications: int = 4,
    cached_baseline: Optional[str] = None
) -> CoVeResult:
    """
    Run Chain-of-Verification on a single problem.

    Uses the "Factored" approach where verification questions are answered
    independently (without attending to the baseline response) to prevent
    repeating hallucinations.

    Args:
        problem: The problem text
        manager: Model manager with generate() method
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        max_verifications: Maximum number of verification questions
        cached_baseline: Optional pre-computed baseline response from CoT cache

    Returns:
        CoVeResult with baseline, verifications, and final response
    """
    # Step 1: Generate baseline response (or use cached)
    if cached_baseline is not None:
        baseline_response = cached_baseline.strip()
        logger.debug(f"Using cached baseline: {baseline_response[:200]}...")
    else:
        baseline_prompt = BASELINE_PROMPT.format(problem=problem)
        baseline_outputs = manager.generate(
            prompts=[baseline_prompt],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=50
        )
        baseline_response = baseline_outputs[0].strip()
        logger.debug(f"Generated baseline: {baseline_response[:200]}...")

    # Step 2: Plan verification questions
    plan_prompt = PLAN_VERIFICATION_PROMPT.format(
        problem=problem,
        baseline=baseline_response
    )
    plan_outputs = manager.generate(
        prompts=[plan_prompt],
        temperature=temperature,
        max_tokens=512,
        top_p=0.9,
        top_k=50
    )
    plan_text = plan_outputs[0].strip()

    verification_questions = parse_verification_questions(plan_text)
    verification_questions = verification_questions[:max_verifications]

    logger.debug(f"Planned {len(verification_questions)} verification questions")

    # Step 3: Execute verifications (factored - independent context)
    verification_answers = []
    for q in verification_questions:
        exec_prompt = EXECUTE_VERIFICATION_PROMPT.format(question=q)
        exec_outputs = manager.generate(
            prompts=[exec_prompt],
            temperature=temperature,
            max_tokens=512,
            top_p=0.9,
            top_k=50
        )
        answer = exec_outputs[0].strip()
        verification_answers.append(answer)

    logger.debug(f"Executed {len(verification_answers)} verifications")

    # Step 4: Generate final verified response
    # Format verifications for the prompt
    verifications_text = ""
    for i, (q, a) in enumerate(zip(verification_questions, verification_answers), 1):
        verifications_text += f"Q{i}: {q}\nA{i}: {a}\n\n"

    if not verifications_text.strip():
        # If no verifications were generated, just use baseline
        verifications_text = "(No verification questions generated)"

    final_prompt = FINAL_RESPONSE_PROMPT.format(
        problem=problem,
        baseline=baseline_response,
        verifications=verifications_text
    )
    final_outputs = manager.generate(
        prompts=[final_prompt],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        top_k=50
    )
    final_response = final_outputs[0].strip()

    logger.debug(f"Final response: {final_response[:200]}...")

    return CoVeResult(
        problem=problem,
        baseline_response=baseline_response,
        verification_questions=verification_questions,
        verification_answers=verification_answers,
        final_response=final_response,
        num_verifications=len(verification_questions)
    )


def cove_batch(
    problems: List[Dict[str, Any]],
    manager,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    max_verifications: int = 4,
    verbose: bool = True,
    cached_solutions: Optional[List[Dict[str, Any]]] = None
) -> List[CoVeResult]:
    """
    Run Chain-of-Verification on a batch of problems.

    Note: This processes problems sequentially since each problem requires
    multiple rounds of generation. For efficiency, consider parallelizing
    at the SLURM job level.

    Args:
        problems: List of problem dicts with 'problem' key
        manager: Model manager with generate() method
        temperature: Sampling temperature
        max_tokens: Maximum tokens per generation
        max_verifications: Maximum verification questions per problem
        verbose: Whether to show progress
        cached_solutions: Optional list of cached CoT solutions (from chain_cache)
                         Each dict should have 'solution' key with baseline response

    Returns:
        List of CoVeResult objects
    """
    from tqdm import tqdm

    results = []
    iterator = enumerate(tqdm(problems, desc="Chain-of-Verification") if verbose else problems)

    for idx, prob_dict in iterator:
        problem = prob_dict.get("problem", prob_dict.get("question", ""))

        # Get cached baseline if available
        cached_baseline = None
        if cached_solutions is not None and idx < len(cached_solutions):
            cached_baseline = cached_solutions[idx].get("solution")

        result = cove_single(
            problem=problem,
            manager=manager,
            temperature=temperature,
            max_tokens=max_tokens,
            max_verifications=max_verifications,
            cached_baseline=cached_baseline
        )
        results.append(result)

    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_final_answer(result: CoVeResult) -> str:
    """Extract the final boxed answer from a CoVeResult."""
    return extract_boxed_answer(result.final_response)


def get_baseline_answer(result: CoVeResult) -> str:
    """Extract the baseline boxed answer from a CoVeResult."""
    return extract_boxed_answer(result.baseline_response)


def compute_cove_metrics(
    results: List[CoVeResult],
    ground_truths: List[str],
    normalize_fn=None
) -> Dict[str, Any]:
    """
    Compute metrics for Chain-of-Verification results.

    Args:
        results: List of CoVeResult objects
        ground_truths: List of ground truth answers
        normalize_fn: Optional function to normalize answers for comparison

    Returns:
        Dictionary with accuracy, baseline accuracy, avg verifications, etc.
    """
    if normalize_fn is None:
        normalize_fn = lambda x: x.strip().lower()

    final_correct = 0
    baseline_correct = 0
    total_verifications = 0

    for result, gt in zip(results, ground_truths):
        # Final answer
        final_pred = get_final_answer(result)
        final_norm = normalize_fn(final_pred)
        gt_norm = normalize_fn(gt)

        if final_norm == gt_norm:
            final_correct += 1

        # Baseline answer (for comparison)
        baseline_pred = get_baseline_answer(result)
        baseline_norm = normalize_fn(baseline_pred)

        if baseline_norm == gt_norm:
            baseline_correct += 1

        total_verifications += result.num_verifications

    n = len(results)
    return {
        "final_accuracy": final_correct / n if n > 0 else 0,
        "baseline_accuracy": baseline_correct / n if n > 0 else 0,
        "improvement": (final_correct - baseline_correct) / n if n > 0 else 0,
        "final_correct": final_correct,
        "baseline_correct": baseline_correct,
        "total": n,
        "avg_verifications": total_verifications / n if n > 0 else 0,
    }


if __name__ == "__main__":
    # Simple test
    print("Chain-of-Verification module loaded successfully.")
    print("Use cove_single() or cove_batch() with a model manager.")
