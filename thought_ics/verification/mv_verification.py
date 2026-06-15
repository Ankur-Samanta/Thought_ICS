#!/usr/bin/env python3
"""
Majority Vote Verification Evaluation Script

Evaluates if majority vote @ k can fix verification accuracy issues.
Loads cached CoT/ToT solutions and runs verification with multiple samples.
Computes confusion matrices for MV@1, MV@3, MV@5, MV@7, MV@9.

Usage:
    python evaluate_mv_verification.py --model llama3b --dataset aime --n-problems 100 --cache-type cot
    python evaluate_mv_verification.py --model llama3b --dataset aime --n-problems 100 --cache-type tot
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

from thought_ics.chain_cache import load_initial_chains
from thought_ics.thought_mdp import initialize_model
from thought_ics.baselines.cot_eval import extract_boxed_answer
from thought_ics.datasets import load_dataset_by_name, normalize_answer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default output directory (separate from main experiments)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "experiments_mv_verification"

# Cache parameters (must match what was used to generate the cached chains)
SEED = 42
GENERATION_TEMP = 0.5  # Temperature used for initial chain generation


def build_verification_prompt(problem: str, solution: str) -> str:
    """Build the verification prompt (same as --verify flag uses)."""
    answer = extract_boxed_answer(solution)

    prompt = f"""You are reviewing a solution to a problem. Analyze it carefully to see if they arrived at the right answer.

Problem: {problem}

Solution to review:
{solution}

Final answer: {answer}

Verify the reasoning step by step and determine whether the final answer is correct or not.

Conclude with \\boxed{{YES}} if the solution is correct, or \\boxed{{NO}} if it contains errors."""

    return prompt


def parse_verification_response(response: str) -> str:
    """Parse YES/NO from verification response."""
    boxed = extract_boxed_answer(response).upper()

    if "YES" in boxed:
        return "YES"
    elif "NO" in boxed:
        return "NO"

    # Fallback: search for yes/no in response
    response_lower = response.lower()
    if "yes" in response_lower and "no" not in response_lower:
        return "YES"
    elif "no" in response_lower:
        return "NO"

    # Default: conservative (assume incorrect)
    return "NO"


def verify_with_multiple_samples(
    manager,
    problem: str,
    solution: str,
    k: int = 9,
    temperature: float = 0.5
) -> Dict:
    """Run verification k times and return all votes.

    Args:
        manager: Model manager
        problem: Problem statement
        solution: Solution to verify
        k: Number of verification samples
        temperature: Temperature for verification (higher = more diverse)

    Returns:
        Dict with votes and responses
    """
    prompt = build_verification_prompt(problem, solution)

    # Single call with n=k for efficiency
    outputs = manager.generate(
        prompts=[prompt],
        n=k,
        temperature=temperature,
        top_p=0.9,
        top_k=50,
        max_tokens=1024
    )

    # Parse each response
    votes = []
    for response in outputs:
        vote = parse_verification_response(response)
        votes.append(vote)

    return {
        'votes': votes,
        'responses': outputs
    }


def compute_majority_vote(votes: List[str], k: int) -> str:
    """Compute majority vote from first k votes."""
    subset = votes[:k]
    yes_count = subset.count("YES")
    no_count = subset.count("NO")
    return "YES" if yes_count > no_count else "NO"


def compute_confusion_matrix(results: List[Dict], k: int) -> Dict:
    """Compute confusion matrix for MV@k."""
    confusion = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}

    for r in results:
        actually_correct = r['actually_correct']
        mv_result = compute_majority_vote(r['votes'], k)
        predicted_correct = (mv_result == "YES")

        if predicted_correct and actually_correct:
            confusion['TP'] += 1
        elif not predicted_correct and not actually_correct:
            confusion['TN'] += 1
        elif predicted_correct and not actually_correct:
            confusion['FP'] += 1
        else:
            confusion['FN'] += 1

    return confusion


def compute_metrics(confusion: Dict) -> Dict:
    """Compute accuracy, precision, and FP rate from confusion matrix."""
    total = sum(confusion.values())
    if total == 0:
        return {'accuracy': 0, 'precision': 0, 'fp_rate': 0, 'recall': 0}

    accuracy = (confusion['TP'] + confusion['TN']) / total

    # Precision: when we say "correct", how often is it actually correct?
    says_correct = confusion['TP'] + confusion['FP']
    precision = confusion['TP'] / says_correct if says_correct > 0 else 0

    # FP rate: how often do we wrongly say "correct"?
    fp_rate = confusion['FP'] / total

    # Recall: of all actually correct, how many do we identify?
    actually_correct = confusion['TP'] + confusion['FN']
    recall = confusion['TP'] / actually_correct if actually_correct > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'fp_rate': fp_rate,
        'recall': recall
    }


def evaluate_mv_verification(
    model_name: str,
    dataset_name: str,
    n_problems: int,
    k: int = 9,
    cache_type: str = "cot",
    temperature: float = 0.5,
    level: Optional[int] = None,
    gpus: str = "0",
    tensor_parallel_size: int = 1,
    output_dir: Optional[Path] = None
) -> Dict:
    """Main evaluation function.

    Args:
        model_name: Model to use (llama3b, qwen7b, etc.)
        dataset_name: Dataset name (aime, amc23, csqa, etc.)
        n_problems: Number of problems
        k: Number of verification samples (default: 9)
        cache_type: "cot" or "tot"
        temperature: Temperature for verification
        level: Level filter for MATH-500
        gpus: GPU IDs
        tensor_parallel_size: Tensor parallel size
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    # Setup output directory
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    level_str = f"_L{level}" if level else ""
    exp_name = f"mv_verify_{model_name}_{dataset_name}{level_str}_{cache_type}_{timestamp}"
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Setup file logging
    file_handler = logging.FileHandler(exp_dir / "run.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MAJORITY VOTE VERIFICATION EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"N problems: {n_problems}")
    logger.info(f"K samples: {k}")
    logger.info(f"Cache type: {cache_type}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Level: {level}")
    logger.info("=" * 80)

    # Save config
    config = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'n_problems': n_problems,
        'k': k,
        'cache_type': cache_type,
        'temperature': temperature,
        'level': level,
        'gpus': gpus,
        'tensor_parallel_size': tensor_parallel_size,
        'timestamp': timestamp
    }
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Initialize model
    logger.info(f"Initializing model {model_name} on GPUs {gpus}...")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    manager = initialize_model(
        gpu_ids=gpus,
        tensor_parallel_size=tensor_parallel_size,
        model_name=model_name
    )

    # Load dataset
    logger.info(f"Loading dataset {dataset_name}...")
    problems = load_dataset_by_name(dataset_name, n_problems=n_problems, level=level, seed=SEED)
    logger.info(f"Loaded {len(problems)} problems")

    # Load cached chains
    logger.info(f"Loading cached {cache_type.upper()} chains...")
    if cache_type == "cot":
        cache_max_depth = 1
        cache_max_tokens = 4096
    else:
        cache_max_depth = 100
        cache_max_tokens = None

    cached_chains = load_initial_chains(
        model_name=model_name,
        dataset_name=dataset_name,
        n_problems=n_problems,
        seed=SEED,
        temperature=GENERATION_TEMP,
        max_depth=cache_max_depth,
        max_tokens_per_thought=cache_max_tokens,
        cache_type=cache_type
    )

    if cached_chains is None:
        logger.error("Failed to load cached chains!")
        manager.unload_base_model()
        return None

    logger.info(f"Loaded {len(cached_chains)} cached chains")

    # Run verification
    results = []
    for idx, problem_data in enumerate(problems):
        if idx >= len(cached_chains):
            logger.warning(f"No cached chain for problem {idx}, skipping")
            continue

        # Get solution from cache
        if cache_type == "cot":
            solution = cached_chains[idx].get('solution', cached_chains[idx].get('chain', ''))
            if isinstance(solution, list):
                solution = "\n\n".join(solution)
        else:  # tot
            chain = cached_chains[idx]['chain']
            solution = "\n".join(chain)  # Match iterative_self_correction.py format

        # Get ground truth correctness
        answer = extract_boxed_answer(solution)
        ground_truth = problem_data.get('answer', problem_data.get('ground_truth', ''))
        actually_correct = normalize_answer(answer) == normalize_answer(ground_truth)

        logger.info(f"Problem {idx+1}/{len(problems)}: answer={answer}, ground_truth={ground_truth}, correct={actually_correct}")

        # Run verification with k samples
        mv_result = verify_with_multiple_samples(
            manager, problem_data['problem'], solution, k=k, temperature=temperature
        )

        # Compute MV results for different k values
        mv_at_k = {}
        for kv in [1, 3, 5, 7, 9]:
            if kv <= k:
                mv_at_k[f'mv@{kv}'] = compute_majority_vote(mv_result['votes'], kv)

        logger.info(f"  Votes: {mv_result['votes']}")
        logger.info(f"  MV results: {mv_at_k}")

        problem_text = problem_data['problem']
        results.append({
            'problem_id': idx,
            'problem': problem_text[:200] + "..." if len(problem_text) > 200 else problem_text,
            'answer': answer,
            'ground_truth': ground_truth,
            'actually_correct': actually_correct,
            'votes': mv_result['votes'],
            **mv_at_k
        })

    # Compute confusion matrices for each k value
    mv_results = {}
    for kv in [1, 3, 5, 7, 9]:
        if kv <= k:
            confusion = compute_confusion_matrix(results, kv)
            metrics = compute_metrics(confusion)
            mv_results[f'mv@{kv}'] = {
                'confusion': confusion,
                **metrics
            }
            logger.info(f"MV@{kv}: {confusion} | Accuracy: {metrics['accuracy']:.3f} | Precision: {metrics['precision']:.3f} | FP Rate: {metrics['fp_rate']:.3f}")

    # Final output
    output = {
        'config': config,
        'mv_results': mv_results,
        'per_problem_results': results
    }

    # Save results (without full responses to save space)
    with open(exp_dir / "results.json", 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to: {exp_dir}")

    # Cleanup
    manager.unload_base_model()

    return output


def main():
    parser = argparse.ArgumentParser(description="Majority Vote Verification Evaluation")

    parser.add_argument('--model', type=str, required=True,
                        help='Model name (llama3b, qwen7b, llama8b, qwen14b, gptoss20b, qwen32b, llama70b, gptoss120b)')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name (aime, amc23, csqa, gpqa, math500, mathqa)')
    parser.add_argument('--n-problems', type=int, required=True,
                        help='Number of problems')
    parser.add_argument('--k', type=int, default=9,
                        help='Number of verification samples (default: 9)')
    parser.add_argument('--cache-type', type=str, default='cot', choices=['cot', 'tot'],
                        help='Cache type: cot or tot (default: cot)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for verification (default: 0.5)')
    parser.add_argument('--level', type=int, default=None,
                        help='Level filter for MATH-500')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU IDs (default: 0)')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Tensor parallel size (default: 1)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: experiments_mv_verification/)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR

    evaluate_mv_verification(
        model_name=args.model,
        dataset_name=args.dataset,
        n_problems=args.n_problems,
        k=args.k,
        cache_type=args.cache_type,
        temperature=args.temperature,
        level=args.level,
        gpus=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
