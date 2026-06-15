"""StepCo verify-then-revise pipeline.

Faithful re-implementation of wzy6642/StepCo/StepCo/solving_pipeline.py,
adapted to use HTTP clients (BaseModelClient + MathShepherdClient) instead of
the original `answered_by_openai` and in-process HF transformers.

Algorithm (matches StepCo's solving_pipeline.py exactly):

  1. Initialization:
     - Generate initial reasoning with zero_shot_cot_prompt_template
     - OSV check: append one ки to the entire reasoning, score with verifier,
       take OSV[0] as a global "is this solution correct" probability
     - Extract numerical answer with get_numerical_answer_prompt_template
     - If OSV[0] >= threshold, return immediately

  2. Rectification loop (up to max_iterations):
     - Parse steps from previous iteration's reasoning via get_reasoning_steps
     - Build per-step verification input with one ки per step (filter steps < 5 chars)
     - PSV: get one score per step from verifier
     - find_first_smaller_index returns 1-indexed first step below threshold
       (or 0 if all steps pass)
     - If 0: return previous answer
     - Else: rectify with stepwise_rectify_prompt_template_v2
     - Re-verify globally (OSV) and re-extract answer
     - Stop if OSV[0] >= threshold OR if last two answers are identical
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .clients import BaseModelClient, MathShepherdClient, STEP_TAG
from .prompts import (
    zero_shot_cot_prompt_template,
    get_numerical_answer_prompt_template,
    stepwise_rectify_prompt_template_v2,
)
from .utils import get_reasoning_steps, find_first_smaller_index

logger = logging.getLogger(__name__)


# StepCo's defaults from config.py. We expose these as kwargs so the eval
# driver can override them to match the rest of the 3p baselines harness.
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_ITERATIONS = 5


@dataclass
class StepCoResult:
    """Result of StepCo on a single problem.

    The `record` field mirrors StepCo's solve_process_record dict so we can
    serialize the same per-iteration state (reasoning_path, OSV, PSV, answer).
    """
    problem: str
    initial_answer: str
    final_answer: str
    iterations: int  # number of rectification iterations actually run (0 if early exit)
    stopped_early: bool  # True if OSV passed at some point or answers converged
    record: Dict[str, Any] = field(default_factory=dict)


def _initialization(
    problem: str,
    record: Dict[str, Any],
    base_client: BaseModelClient,
    verifier: MathShepherdClient,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> str:
    """Mirror of StepCo's initialization()."""
    input_str = zero_shot_cot_prompt_template.format(instruction="\n", question=problem)
    initial_reasoning_path = base_client.generate(
        input_str,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    record["iter-0"] = {}
    record["iter-0"]["reasoning_path"] = initial_reasoning_path

    osv_input = f"Q: {problem} \n A: {initial_reasoning_path} {STEP_TAG}"
    output_supervised_verifier = verifier.step_verify_score(osv_input)
    record["iter-0"]["OSV"] = output_supervised_verifier

    answer_input = get_numerical_answer_prompt_template.format(
        **{' ': '', 'question': problem, 'reasoning_path': initial_reasoning_path}
    )
    initial_answer = base_client.generate(
        answer_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    record["iter-0"]["answer"] = initial_answer
    return initial_answer


def _rectification(
    problem: str,
    record: Dict[str, Any],
    base_client: BaseModelClient,
    verifier: MathShepherdClient,
    num_iter: int,
    threshold: float,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> str:
    """Mirror of StepCo's rectification()."""
    prev_reasoning = record[f"iter-{num_iter - 1}"]["reasoning_path"]
    reasoning_steps = get_reasoning_steps(prev_reasoning)
    reasoning_steps_with_tag = "\n".join(
        [step + f" {STEP_TAG}" for step in reasoning_steps if len(step) >= 5]
    )

    psv_input = f"Q: {record['problem']} \n A: {reasoning_steps_with_tag}"
    process_supervised_verifier = verifier.step_verify_score(psv_input)
    first_incorrect_step_idx = find_first_smaller_index(
        process_supervised_verifier, threshold
    )

    if first_incorrect_step_idx == 0:
        return record[f"iter-{num_iter - 1}"]["answer"]

    record[f"iter-{num_iter}"] = {}
    record[f"iter-{num_iter - 1}"]["PSV"] = process_supervised_verifier

    rectify_input = stepwise_rectify_prompt_template_v2.format(
        question=record["problem"],
        reasoning_path=prev_reasoning,
        step_index=first_incorrect_step_idx,
        probability=format(
            process_supervised_verifier[first_incorrect_step_idx - 1] * 100, ".2f"
        ),
    )
    rectified_reasoning_path = base_client.generate(
        rectify_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    record[f"iter-{num_iter}"]["reasoning_path"] = rectified_reasoning_path

    osv_input = f"Q: {problem} \n A: {rectified_reasoning_path} {STEP_TAG}"
    output_supervised_verifier = verifier.step_verify_score(osv_input)
    record[f"iter-{num_iter}"]["OSV"] = output_supervised_verifier

    answer_input = get_numerical_answer_prompt_template.format(
        **{' ': '', 'question': problem, 'reasoning_path': rectified_reasoning_path}
    )
    rectified_answer = base_client.generate(
        answer_input,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    record[f"iter-{num_iter}"]["answer"] = rectified_answer
    return rectified_answer


def stepco_single(
    problem: str,
    base_client: BaseModelClient,
    verifier: MathShepherdClient,
    threshold: float = DEFAULT_THRESHOLD,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    top_p: float = 0.9,
    top_k: int = 50,
) -> StepCoResult:
    """Run StepCo on a single problem. Mirror of StepCo's pipeline().

    The default temperature/max_tokens/top_p/top_k match the rest of the
    3p_baselines harness (self_refine, cove); StepCo's own config defaults
    were 0.7 / 2048 / 0.95 / 0.
    """
    record: Dict[str, Any] = {"problem": problem}

    initial_answer = _initialization(
        problem,
        record,
        base_client,
        verifier,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )

    answer_record_str: List[str] = [initial_answer]

    iter0_osv = record["iter-0"].get("OSV") or [0.0]
    if iter0_osv[0] >= threshold:
        return StepCoResult(
            problem=problem,
            initial_answer=initial_answer,
            final_answer=initial_answer,
            iterations=0,
            stopped_early=True,
            record=record,
        )

    logger.debug("[StepCo] Initial answer flagged as possibly incorrect; rectifying")
    final_answer = initial_answer
    iterations_run = 0
    stopped_early = False
    for num_iter in range(max_iterations):
        try:
            answer = _rectification(
                problem,
                record,
                base_client,
                verifier,
                num_iter + 1,
                threshold=threshold,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )
        except Exception as e:
            logger.error(f"[StepCo] Rectification iter {num_iter + 1} failed: {e}")
            break
        iterations_run += 1
        answer_record_str.append(answer)
        final_answer = answer

        next_iter_record = record.get(f"iter-{num_iter + 1}")
        if next_iter_record is not None:
            osv = next_iter_record.get("OSV") or [0.0]
            if osv[0] >= threshold:
                stopped_early = True
                break

        if len(answer_record_str) >= 2 and answer_record_str[-1] == answer_record_str[-2]:
            stopped_early = True
            break

    return StepCoResult(
        problem=problem,
        initial_answer=initial_answer,
        final_answer=final_answer,
        iterations=iterations_run,
        stopped_early=stopped_early,
        record=record,
    )
