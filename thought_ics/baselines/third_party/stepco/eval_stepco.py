#!/usr/bin/env python3
"""StepCo evaluation driver.

Mirrors the structure of 3p_baselines/eval_3p_baselines.py (same dataset
loaders, same output schema, same answer normalization), but uses two vLLM
HTTP servers instead of an in-process vLLM model:
  - one server hosts the base reasoner (1 GPU)
  - one server hosts the Math-Shepherd PRM verifier (2 GPUs, TP=2)

The user is responsible for starting the two servers before running this
script (see launch_servers.sh).

Usage:
    python 3p_baselines/stepco/eval_stepco.py \\
        --base-model-url http://localhost:8001 \\
        --base-model-name meta-llama/Llama-3.1-8B-Instruct \\
        --verifier-url http://localhost:8002 \\
        --model llama8b \\
        --dataset math500 \\
        --n-problems 100
"""

import os
os.environ.setdefault("VLLM_USE_V1", "1")

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import argparse
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from thought_ics.datasets import load_dataset_by_name, get_dataset_info, normalize_answer

from .clients import BaseModelClient, MathShepherdClient, MATH_SHEPHERD_MODEL
from .stepco import stepco_single, StepCoResult, DEFAULT_THRESHOLD, DEFAULT_MAX_ITERATIONS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42


# =============================================================================
# Answer extraction (matches the rest of the 3p baselines harness)
# =============================================================================

def extract_boxed_answer(text: str) -> str:
    """Extract the answer from \\boxed{}. StepCo's answer-extraction LLM call
    is prompted to wrap the final answer in \\boxed{}, so this should hit."""
    import re
    if not text:
        return "NO ANSWER"
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        # Fallback: try <ans>...</ans> tags from StepCo's CoT prompt
        ans_match = re.search(r'<ans>\s*(.*?)\s*</ans>', text)
        if ans_match:
            return ans_match.group(1).strip()
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
        return text[start_pos:i - 1].strip()
    return "NO ANSWER"


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_stepco(
    problems: List[Dict[str, Any]],
    base_client: BaseModelClient,
    verifier: MathShepherdClient,
    threshold: float,
    max_iterations: int,
    temperature: float,
    max_tokens: int,
    top_p: float,
    top_k: int,
) -> List[Dict[str, Any]]:
    results = []
    for idx, prob_dict in enumerate(tqdm(problems, desc="StepCo Evaluation")):
        problem = prob_dict.get("problem", prob_dict.get("question", ""))
        ground_truth = prob_dict.get("answer", prob_dict.get("ground_truth", ""))

        try:
            result: StepCoResult = stepco_single(
                problem=problem,
                base_client=base_client,
                verifier=verifier,
                threshold=threshold,
                max_iterations=max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
            )

            initial_answer_text = extract_boxed_answer(result.initial_answer)
            final_answer_text = extract_boxed_answer(result.final_answer)

            initial_norm = normalize_answer(initial_answer_text)
            final_norm = normalize_answer(final_answer_text)
            gt_norm = normalize_answer(str(ground_truth))

            initial_correct = (initial_norm == gt_norm)
            final_correct = (final_norm == gt_norm)

            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "initial_answer": initial_answer_text,
                "final_answer": final_answer_text,
                "initial_correct": initial_correct,
                "final_correct": final_correct,
                "iterations": result.iterations,
                "stopped_early": result.stopped_early,
                "record": result.record,
            })
        except Exception as e:
            logger.error(f"Problem {idx} failed: {e}", exc_info=True)
            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "initial_answer": "ERROR",
                "final_answer": "ERROR",
                "initial_correct": False,
                "final_correct": False,
                "iterations": 0,
                "stopped_early": False,
                "record": {},
                "error": str(e),
            })
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"final_accuracy": 0, "total": 0}
    final_correct = sum(1 for r in results if r["final_correct"])
    initial_correct = sum(1 for r in results if r["initial_correct"])
    avg_iterations = sum(r["iterations"] for r in results) / n
    early_stops = sum(1 for r in results if r["stopped_early"])
    return {
        "final_accuracy": final_correct / n,
        "initial_accuracy": initial_correct / n,
        "improvement": (final_correct - initial_correct) / n,
        "final_correct": final_correct,
        "initial_correct": initial_correct,
        "total": n,
        "avg_iterations": avg_iterations,
        "early_stop_rate": early_stops / n,
        "early_stops": early_stops,
    }


def save_results(results, metrics, exp_dir: Path):
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(exp_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    logger.info("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate StepCo (Wu et al., 2024)")

    # Server URLs and model identifiers (must match what the vLLM servers were started with)
    parser.add_argument("--base-model-url", type=str, required=True,
                        help="URL of the vLLM server hosting the base reasoner (e.g. http://localhost:8001)")
    parser.add_argument("--base-model-name", type=str, required=True,
                        help="HF model name the base vLLM server was launched with (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--verifier-url", type=str, required=True,
                        help="URL of the vLLM server hosting Math-Shepherd (e.g. http://localhost:8002)")
    parser.add_argument("--verifier-model-name", type=str, default=MATH_SHEPHERD_MODEL,
                        help="HF model name for the verifier (default: peiyi9979/math-shepherd-mistral-7b-prm)")

    # Bookkeeping nickname (matches eval_3p_baselines.py)
    parser.add_argument("--model", type=str, default="llama8b",
                        help="Short model nickname for output dir naming")

    # Dataset (matches eval_3p_baselines.py)
    parser.add_argument("--dataset", type=str, default="math500",
                        choices=["math500", "mathqa", "amc23", "aime", "csqa", "gpqa", "svamp"])
    parser.add_argument("--n-problems", type=int, default=100)
    parser.add_argument("--level", type=int, default=None)

    # Generation hyperparameters (defaults match eval_3p_baselines.py)
    parser.add_argument("--generation-temp", type=float, default=0.5)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)

    # StepCo-specific (defaults match StepCo's config.py)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)

    parser.add_argument("--output-dir", type=str, default="experiments")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Build experiment dir, matching eval_3p_baselines.py naming
    method_name = "Baseline_StepCo"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_suffix = f"_L{args.level}" if args.level else ""
    experiment_name = f"{method_name}_{args.model}_{args.dataset}{level_suffix}_{timestamp}"
    exp_dir = Path(args.output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(exp_dir / "run.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    config = {
        "experiment_name": experiment_name,
        "method": "stepco",
        "method_name": method_name,
        "base_model_url": args.base_model_url,
        "base_model_name": args.base_model_name,
        "verifier_url": args.verifier_url,
        "verifier_model_name": args.verifier_model_name,
        "model_nickname": args.model,
        "dataset": args.dataset,
        "n_problems": args.n_problems,
        "level": args.level,
        "threshold": args.threshold,
        "max_iterations": args.max_iterations,
        "generation_temp": args.generation_temp,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info("=" * 100)
    logger.info(f"STEPCO EVALUATION - {args.model} on {args.dataset}")
    logger.info("=" * 100)
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 100)

    # Load dataset
    problems = load_dataset_by_name(
        dataset_name=args.dataset,
        n_problems=args.n_problems,
        level=args.level,
        seed=SEED,
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Init clients (no model loading -- just HTTP wrappers + tokenizer)
    base_client = BaseModelClient(args.base_model_url, args.base_model_name)
    verifier = MathShepherdClient(args.verifier_url, args.verifier_model_name)

    # Smoke-test both servers before launching the eval
    logger.info("Smoke-testing base model server...")
    _ = base_client.generate("Q: What is 2+2?\nA:", max_tokens=8, temperature=0.0)
    logger.info("Smoke-testing verifier server...")
    _ = verifier.step_verify_score(f"Q: 1+1? \n A: 1+1=2 ки")
    logger.info("Both servers responsive.")

    # Run
    results = evaluate_stepco(
        problems=problems,
        base_client=base_client,
        verifier=verifier,
        threshold=args.threshold,
        max_iterations=args.max_iterations,
        temperature=args.generation_temp,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    metrics = compute_metrics(results)
    save_results(results, metrics, exp_dir)
    logger.info(f"Results saved to: {exp_dir}")

    logger.removeHandler(file_handler)
    file_handler.close()


if __name__ == "__main__":
    main()
