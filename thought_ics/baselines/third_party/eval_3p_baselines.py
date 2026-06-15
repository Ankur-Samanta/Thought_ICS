#!/usr/bin/env python3
"""
Unified Evaluation Script for Third-Party Baseline Methods

Evaluates Self-Refine and Chain-of-Verification methods on various datasets.
Uses the same argument structure as baseline_cot_eval.py for consistency.

Usage:
    python 3p_baselines/eval_3p_baselines.py \
        --method self_refine \
        --model llama8b \
        --dataset math500 \
        --n-problems 100 \
        --gpus 0,1 \
        --tensor-parallel-size 2

Supported methods:
    - self_refine: Self-Refine (Madaan et al., 2023)
    - cove: Chain-of-Verification (Dhuliawala et al., 2023)

Supported datasets:
    - math500, mathqa, amc23, aime, csqa, gpqa, svamp

Supported models:
    - llama3b, phi4b, qwen7b, llama8b, qwen14b, qwen32b, llama70b
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from tqdm import tqdm

from thought_ics.thought_mdp import initialize_model
from thought_ics.datasets import load_dataset_by_name, get_dataset_info, normalize_answer
from thought_ics.chain_cache import load_initial_chains

from .self_refine import (
    self_refine_single,
    SelfRefineResult,
    get_final_answer as get_sr_answer,
    extract_boxed_answer
)
from .chain_of_verification import (
    cove_single,
    CoVeResult,
    get_final_answer as get_cove_answer,
    get_baseline_answer as get_cove_baseline
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Experiment configuration (matching baseline_cot_eval.py)
SEED = 42


# =============================================================================
# UTILITY FUNCTIONS (matching baseline_cot_eval.py)
# =============================================================================

def convert_tot_chain_to_cot(chain: List[str]) -> str:
    """Convert a Tree of Thought chain to a standard CoT format by joining with newlines."""
    return "\n\n".join(chain)


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_self_refine(
    problems: List[Dict[str, Any]],
    manager,
    max_iterations: int = 4,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    cached_chains: Optional[List[Dict[str, Any]]] = None,
    cache_type: str = "cot"
) -> List[Dict[str, Any]]:
    """
    Evaluate Self-Refine on a list of problems.

    Args:
        problems: List of problem dicts with 'problem' and 'answer' keys
        manager: Model manager
        max_iterations: Maximum refinement iterations
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        cached_chains: Optional cached chains to use as initial solutions
        cache_type: Type of cache ("cot" or "tot")

    Returns:
        List of result dicts with predictions and metrics
    """
    results = []

    for idx, prob_dict in enumerate(tqdm(problems, desc="Self-Refine Evaluation")):
        problem = prob_dict.get("problem", prob_dict.get("question", ""))
        ground_truth = prob_dict.get("answer", prob_dict.get("ground_truth", ""))

        try:
            # Get cached initial solution if available
            # Match baseline_cot_eval.py exactly
            cached_initial = None
            if cached_chains is not None and idx < len(cached_chains):
                cached_chain = cached_chains[idx]
                if cache_type == "tot":
                    # Convert cached ToT chain to CoT format
                    cached_initial = convert_tot_chain_to_cot(cached_chain['chain'])
                    logger.info(f"Using cached ToT chain ({len(cached_chain['chain'])} steps) converted to CoT")
                else:
                    # Use cached CoT chain directly
                    cached_initial = cached_chain.get('solution', cached_chain.get('chain', ''))
                    logger.debug(f"Using cached CoT chain directly")

            # Run self-refine
            result = self_refine_single(
                problem=problem,
                manager=manager,
                max_iterations=max_iterations,
                temperature=temperature,
                max_tokens=max_tokens,
                cached_initial_solution=cached_initial
            )

            # Extract answers
            final_answer = get_sr_answer(result)
            initial_answer = extract_boxed_answer(result.initial_solution)

            # Normalize and compare
            final_norm = normalize_answer(final_answer)
            initial_norm = normalize_answer(initial_answer)
            gt_norm = normalize_answer(ground_truth)

            final_correct = (final_norm == gt_norm)
            initial_correct = (initial_norm == gt_norm)

            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "initial_answer": initial_answer,
                "final_answer": final_answer,
                "initial_correct": initial_correct,
                "final_correct": final_correct,
                "iterations": result.iterations,
                "stopped_early": result.stopped_early,
                "initial_solution": result.initial_solution,
                "final_solution": result.final_solution,
                "history": result.history
            })

        except Exception as e:
            logger.error(f"Problem {idx} failed: {e}")
            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "initial_answer": "ERROR",
                "final_answer": "ERROR",
                "initial_correct": False,
                "final_correct": False,
                "iterations": 0,
                "stopped_early": False,
                "initial_solution": "",
                "final_solution": "",
                "history": [],
                "error": str(e)
            })

    return results


def evaluate_cove(
    problems: List[Dict[str, Any]],
    manager,
    temperature: float = 0.5,
    max_tokens: int = 2048,
    max_verifications: int = 4,
    cached_chains: Optional[List[Dict[str, Any]]] = None,
    cache_type: str = "cot"
) -> List[Dict[str, Any]]:
    """
    Evaluate Chain-of-Verification on a list of problems.

    Args:
        problems: List of problem dicts with 'problem' and 'answer' keys
        manager: Model manager
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        max_verifications: Max verification questions
        cached_chains: Optional cached chains to use as baseline
        cache_type: Type of cache ("cot" or "tot")

    Returns:
        List of result dicts with predictions and metrics
    """
    results = []

    for idx, prob_dict in enumerate(tqdm(problems, desc="Chain-of-Verification Evaluation")):
        problem = prob_dict.get("problem", prob_dict.get("question", ""))
        ground_truth = prob_dict.get("answer", prob_dict.get("ground_truth", ""))

        try:
            # Get cached baseline if available
            # Match baseline_cot_eval.py exactly
            cached_baseline = None
            if cached_chains is not None and idx < len(cached_chains):
                cached_chain = cached_chains[idx]
                if cache_type == "tot":
                    # Convert cached ToT chain to CoT format
                    cached_baseline = convert_tot_chain_to_cot(cached_chain['chain'])
                    logger.info(f"Using cached ToT chain ({len(cached_chain['chain'])} steps) converted to CoT")
                else:
                    # Use cached CoT chain directly
                    cached_baseline = cached_chain.get('solution', cached_chain.get('chain', ''))
                    logger.debug(f"Using cached CoT chain directly")

            # Run CoVe
            result = cove_single(
                problem=problem,
                manager=manager,
                temperature=temperature,
                max_tokens=max_tokens,
                max_verifications=max_verifications,
                cached_baseline=cached_baseline
            )

            # Extract answers
            final_answer = get_cove_answer(result)
            baseline_answer = get_cove_baseline(result)

            # Normalize and compare
            final_norm = normalize_answer(final_answer)
            baseline_norm = normalize_answer(baseline_answer)
            gt_norm = normalize_answer(ground_truth)

            final_correct = (final_norm == gt_norm)
            baseline_correct = (baseline_norm == gt_norm)

            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "baseline_answer": baseline_answer,
                "final_answer": final_answer,
                "baseline_correct": baseline_correct,
                "final_correct": final_correct,
                "num_verifications": result.num_verifications,
                "verification_questions": result.verification_questions,
                "verification_answers": result.verification_answers,
                "baseline_response": result.baseline_response,
                "final_response": result.final_response
            })

        except Exception as e:
            logger.error(f"Problem {idx} failed: {e}")
            results.append({
                "problem": problem,
                "ground_truth": ground_truth,
                "baseline_answer": "ERROR",
                "final_answer": "ERROR",
                "baseline_correct": False,
                "final_correct": False,
                "num_verifications": 0,
                "verification_questions": [],
                "verification_answers": [],
                "baseline_response": "",
                "final_response": "",
                "error": str(e)
            })

    return results


def compute_metrics(results: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    """Compute aggregate metrics from evaluation results."""
    n = len(results)
    if n == 0:
        return {"accuracy": 0, "total": 0}

    if method == "self_refine":
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
            "early_stops": early_stops
        }

    elif method == "cove":
        final_correct = sum(1 for r in results if r["final_correct"])
        baseline_correct = sum(1 for r in results if r["baseline_correct"])
        avg_verifications = sum(r["num_verifications"] for r in results) / n

        return {
            "final_accuracy": final_correct / n,
            "baseline_accuracy": baseline_correct / n,
            "improvement": (final_correct - baseline_correct) / n,
            "final_correct": final_correct,
            "baseline_correct": baseline_correct,
            "total": n,
            "avg_verifications": avg_verifications
        }

    return {"accuracy": 0, "total": n}


def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    exp_dir: Path
):
    """Save evaluation results and metrics to disk (matching baseline_cot_eval.py format)."""
    # Save full results
    results_file = exp_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved results to {results_file}")

    # Save metrics summary
    metrics_file = exp_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_file}")

    # Print summary
    logger.info("="*60)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{key}: {value:.4f}")
        else:
            logger.info(f"{key}: {value}")
    logger.info("="*60)


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation(
    method: str,
    model_name: str = "llama8b",
    dataset: str = "math500",
    n_problems: int = 100,
    level: Optional[int] = None,
    gpu_ids: str = "0",
    tensor_parallel_size: int = 1,
    generation_temp: float = 0.5,
    max_tokens: int = 2048,
    max_iterations: int = 4,
    max_verifications: int = 4,
    use_cached_chains: bool = False,
    cache_type: str = "cot",
    output_dir: str = "experiments"
):
    """Run 3P baseline evaluation (matching baseline_cot_eval.py structure).

    Args:
        method: Method to evaluate ("self_refine" or "cove")
        model_name: Model nickname
        dataset: Dataset name
        n_problems: Number of problems to evaluate
        level: For MATH-500, filter by difficulty level (1-5)
        gpu_ids: Comma-separated GPU IDs
        tensor_parallel_size: Number of GPUs for tensor parallelism
        generation_temp: Temperature for generation
        max_tokens: Max tokens per generation
        max_iterations: Max iterations for Self-Refine
        max_verifications: Max verification questions for CoVe
        use_cached_chains: Whether to use cached CoT solutions
        cache_type: Type of cache ("cot" or "tot")
        output_dir: Base output directory for experiments
    """
    # Get dataset info
    dataset_info = get_dataset_info(dataset)

    # Method names matching baseline_cot_eval.py naming convention
    method_names = {
        'self_refine': 'Baseline_SelfRefine',
        'cove': 'Baseline_ChainOfVerification'
    }
    method_name = method_names.get(method, method)

    # Create experiment directory (matching baseline_cot_eval.py structure)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build experiment name with appropriate suffixes
    cache_suffix = f"_from_cached_{cache_type}" if use_cached_chains else ""
    level_suffix = f"_L{level}" if level else ""

    experiment_name = f"{method_name}{cache_suffix}_{model_name}_{dataset}{level_suffix}_{timestamp}"
    exp_dir = Path(output_dir) / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging (matching baseline_cot_eval.py)
    log_file = exp_dir / "run.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.info(f"Logging to: {log_file}")

    # Save experiment config (matching baseline_cot_eval.py)
    config = {
        'experiment_name': experiment_name,
        'method': method,
        'method_name': method_name,
        'model_name': model_name,
        'gpu_ids': gpu_ids,
        'tensor_parallel_size': tensor_parallel_size,
        'n_problems': n_problems,
        'max_iterations': max_iterations,
        'max_verifications': max_verifications,
        'use_cached_chains': use_cached_chains,
        'cache_type': cache_type,
        'generation_temp': generation_temp,
        'max_tokens': max_tokens,
        'dataset': dataset,
        'dataset_info': dataset_info,
        'level_filter': level,
        'seed': SEED,
        'timestamp': datetime.now().isoformat()
    }

    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    logger.info("="*100)
    logger.info(f"3P BASELINE EVALUATION - {method_name}")
    logger.info("="*100)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Dataset: {dataset_info['name']}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Method: {method}")
    logger.info(f"Problems: {n_problems}")
    if level:
        logger.info(f"Level filter: {level}")
    logger.info(f"Use cached chains: {use_cached_chains} ({cache_type})")
    logger.info(f"Generation temp: {generation_temp}")
    if method == "self_refine":
        logger.info(f"Max iterations: {max_iterations}")
    else:
        logger.info(f"Max verifications: {max_verifications}")
    logger.info("="*100)

    # Load dataset
    logger.info(f"Loading {dataset} dataset...")
    problems = load_dataset_by_name(
        dataset_name=dataset,
        n_problems=n_problems,
        level=level,
        seed=SEED
    )
    logger.info(f"Loaded {len(problems)} problems")

    # Load cached chains if requested
    cached_chains = None
    if use_cached_chains:
        logger.info(f"Attempting to load cached {cache_type.upper()} chains...")
        # For CoT cache, use max_depth=1 and max_tokens=4096 to match how caches are saved
        # For ToT cache, use the global defaults (MAX_DEPTH=100, MAX_TOKENS_PER_THOUGHT=None)
        if cache_type == "cot":
            cache_max_depth = 1
            cache_max_tokens = 4096
        else:
            cache_max_depth = 100
            cache_max_tokens = None

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
            logger.warning(f"No cached {cache_type.upper()} chains found! Will generate from scratch.")

    # Initialize model
    logger.info(f"Initializing model '{model_name}' on GPUs {gpu_ids}...")
    manager = initialize_model(
        gpu_ids=gpu_ids,
        tensor_parallel_size=tensor_parallel_size,
        model_name=model_name
    )
    logger.info("Model initialized successfully")

    # Run evaluation
    if method == "self_refine":
        logger.info("Running Self-Refine evaluation...")
        results = evaluate_self_refine(
            problems=problems,
            manager=manager,
            max_iterations=max_iterations,
            temperature=generation_temp,
            max_tokens=max_tokens,
            cached_chains=cached_chains,
            cache_type=cache_type
        )
    elif method == "cove":
        logger.info("Running Chain-of-Verification evaluation...")
        results = evaluate_cove(
            problems=problems,
            manager=manager,
            temperature=generation_temp,
            max_tokens=max_tokens,
            max_verifications=max_verifications,
            cached_chains=cached_chains,
            cache_type=cache_type
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Compute metrics
    metrics = compute_metrics(results, method)

    # Save results
    save_results(results, metrics, exp_dir)

    logger.info("="*100)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"Results saved to: {exp_dir}")
    logger.info("="*100)

    # Remove file handler to avoid duplicate logging in subsequent runs
    logger.removeHandler(file_handler)
    file_handler.close()

    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Self-Refine and Chain-of-Verification baselines"
    )

    # Method selection
    parser.add_argument('--method', type=str, required=True,
                        choices=['self_refine', 'cove'],
                        help='Method to evaluate: self_refine or cove')

    # Model configuration
    parser.add_argument('--model', type=str, default='llama8b',
                        choices=['llama3b', 'phi4b', 'qwen7b', 'llama8b',
                                 'qwen14b', 'qwen32b', 'llama70b', 'qwen2b',
                                 'mistral3b', 'mistral8b', 'mistral14b',
                                 'gptoss20b', 'gptoss120b'],
                        help='Model to use (default: llama8b)')

    # Dataset configuration
    parser.add_argument('--dataset', type=str, default='math500',
                        choices=['math500', 'mathqa', 'amc23', 'aime',
                                 'csqa', 'gpqa', 'svamp'],
                        help='Dataset to evaluate on (default: math500)')
    parser.add_argument('--n-problems', type=int, default=100,
                        help='Number of problems to evaluate (default: 100)')
    parser.add_argument('--level', type=int, default=None,
                        help='For MATH-500, filter by difficulty level 1-5')

    # GPU configuration
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (default: 0)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism (default: 1)')

    # Generation parameters
    parser.add_argument('--generation-temp', type=float, default=0.5,
                        help='Temperature for generation (default: 0.5)')
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Maximum tokens per generation (default: 2048)')

    # Method-specific parameters
    parser.add_argument('--max-iterations', type=int, default=4,
                        help='Max iterations for Self-Refine (default: 4)')
    parser.add_argument('--max-verifications', type=int, default=4,
                        help='Max verification questions for CoVe (default: 4)')

    # Cache options
    parser.add_argument('--use-cached-chains', action='store_true',
                        help='Use cached CoT solutions as initial/baseline responses')
    parser.add_argument('--cache-type', type=str, default='cot',
                        choices=['cot', 'tot'],
                        help='Type of cache to use (default: cot)')

    # Output
    parser.add_argument('--output-dir', type=str, default='experiments',
                        help='Base output directory (default: experiments)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Run evaluation
    run_evaluation(
        method=args.method,
        model_name=args.model,
        dataset=args.dataset,
        n_problems=args.n_problems,
        level=args.level,
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        generation_temp=args.generation_temp,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        max_verifications=args.max_verifications,
        use_cached_chains=args.use_cached_chains,
        cache_type=args.cache_type,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
