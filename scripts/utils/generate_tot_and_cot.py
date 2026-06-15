#!/usr/bin/env python3
"""
Generate both ToT and CoT solutions for a dataset and cache them separately.
This script generates initial solutions (not iterative correction).
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import hashlib
import argparse
import logging
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm

from thought_ics.thought_mdp import (
    ToTAgent, ToTEnvironment, TreeSearch,
    initialize_model, get_completed_paths
)
from thought_ics.datasets import load_dataset_by_name, get_dataset_info

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
MAX_DEPTH = 100
MAX_TOKENS_PER_THOUGHT = None
SEED = 42


def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    import re
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


def get_cache_key(
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    solution_type: str  # 'tot' or 'cot'
) -> str:
    """Generate cache key from config parameters.

    Only includes solution_type for CoT to keep ToT caches compatible with existing format.
    """
    if solution_type == 'cot':
        # CoT: include solution_type in key for separate caching
        config_str = f"{model_name}_{dataset_name}_{n_problems}_{seed}_{temperature}_{max_depth}_{max_tokens_per_thought}_cot"
    else:
        # ToT: use old format (no solution_type) for backward compatibility
        config_str = f"{model_name}_{dataset_name}_{n_problems}_{seed}_{temperature}_{max_depth}_{max_tokens_per_thought}"
    return hashlib.md5(config_str.encode()).hexdigest()


def get_cache_path(cache_key: str, solution_type: str) -> Path:
    """Get cache file path for a given cache key and solution type."""
    if solution_type == 'cot':
        return CACHE_DIR / f"initial_cot_{cache_key}.json"
    else:
        # ToT: use old naming format for backward compatibility
        return CACHE_DIR / f"initial_chains_{cache_key}.json"


def save_initial_solutions(
    solutions: List[Dict],
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    solution_type: str
) -> None:
    """Save initial solutions to cache with metadata."""
    cache_key = get_cache_key(model_name, dataset_name, n_problems, seed, temperature, max_depth, max_tokens_per_thought, solution_type)
    cache_path = get_cache_path(cache_key, solution_type)

    cache_data = {
        'metadata': {
            'model_name': model_name,
            'dataset_name': dataset_name,
            'n_problems': n_problems,
            'seed': seed,
            'temperature': temperature,
            'max_depth': max_depth,
            'max_tokens_per_thought': max_tokens_per_thought,
            'solution_type': solution_type,
            'cache_key': cache_key
        },
        'solutions': solutions
    }

    with open(cache_path, 'w') as f:
        json.dump(cache_data, f, indent=2)

    logger.info(f"Saved {len(solutions)} {solution_type.upper()} solutions to cache: {cache_path.name}")


def load_initial_solutions(
    model_name: str,
    dataset_name: str,
    n_problems: int,
    seed: int,
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int,
    solution_type: str
) -> Optional[List[Dict]]:
    """Load initial solutions from cache if they exist.

    ToT uses the same cache format as existing scripts (initial_chains_*.json).
    CoT uses a separate cache format (initial_cot_*.json).
    """
    cache_key = get_cache_key(model_name, dataset_name, n_problems, seed, temperature, max_depth, max_tokens_per_thought, solution_type)
    cache_path = get_cache_path(cache_key, solution_type)

    if not cache_path.exists():
        logger.info(f"Cache not found for {solution_type.upper()} with key {cache_key}")
        return None

    with open(cache_path, 'r') as f:
        cache_data = json.load(f)

    # Verify metadata matches
    metadata = cache_data['metadata']

    # For ToT, metadata won't have solution_type (backward compatibility)
    # For CoT, metadata should have solution_type
    expected_solution_type = metadata.get('solution_type', 'tot')  # Default to 'tot' for old caches

    if (metadata.get('model_name') == model_name and
        metadata['dataset_name'] == dataset_name and
        metadata['n_problems'] == n_problems and
        metadata['seed'] == seed and
        metadata['temperature'] == temperature and
        metadata['max_depth'] == max_depth and
        metadata['max_tokens_per_thought'] == max_tokens_per_thought and
        expected_solution_type == solution_type):

        # Handle both old format (chains) and new format (solutions)
        solutions = cache_data.get('solutions', cache_data.get('chains'))
        logger.info(f"Loaded {len(solutions)} {solution_type.upper()} solutions from cache: {cache_path.name}")
        return solutions
    else:
        logger.warning(f"Cache metadata mismatch for {solution_type.upper()} with key {cache_key}")
        return None


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


def generate_tot_solutions(
    manager,
    problems: List[Dict],
    temperature: float,
    max_depth: int,
    max_tokens_per_thought: int
) -> List[Dict]:
    """Generate ToT solutions for a list of problems."""
    logger.info("Generating ToT solutions...")
    solutions = []

    for idx, item in enumerate(tqdm(problems, desc="Generating ToT solutions")):
        logger.info(f"Generating ToT solution for problem {idx+1}/{len(problems)}")

        agent = ToTAgent(manager, temperature=temperature, max_tokens=max_tokens_per_thought)
        env = ToTEnvironment(max_depth=max_depth)
        search = TreeSearch(agent, env, strategy="dfs", n_rollouts=1)

        root = search.search(item['problem'], verbose=False)
        completed_paths = get_completed_paths(root)

        if not completed_paths:
            logger.warning(f"No completed paths found for problem {idx+1}")
            chain = []
        else:
            chain = completed_paths[0][1:]  # Skip question

        answer = extract_boxed_answer(chain[-1] if chain else "")

        solutions.append({
            'problem_id': item['unique_id'],
            'problem_number': idx + 1,
            'chain': chain,
            'answer': answer,
            'correct': answer == item['answer']
        })

        logger.info(f"Problem {idx+1}: Generated ToT with {len(chain)} steps, answer: {answer}, correct: {answer == item['answer']}")

    return solutions


def generate_cot_solutions(
    manager,
    problems: List[Dict],
    temperature: float,
    max_tokens: int = 2048
) -> List[Dict]:
    """Generate CoT solutions for a list of problems."""
    logger.info("Generating CoT solutions...")
    solutions = []

    for idx, item in enumerate(tqdm(problems, desc="Generating CoT solutions")):
        logger.info(f"Generating CoT solution for problem {idx+1}/{len(problems)}")

        solution_text = generate_cot_solution(manager, item['problem'], max_tokens=max_tokens, temperature=temperature)
        answer = extract_boxed_answer(solution_text)

        solutions.append({
            'problem_id': item['unique_id'],
            'problem_number': idx + 1,
            'solution': solution_text,
            'answer': answer,
            'correct': answer == item['answer']
        })

        logger.info(f"Problem {idx+1}: Generated CoT, answer: {answer}, correct: {answer == item['answer']}")

    return solutions


def main():
    parser = argparse.ArgumentParser(description='Generate and cache ToT and CoT solutions')
    parser.add_argument('--gpus', type=str, required=True,
                        help='Comma-separated GPU IDs (e.g., "0,1")')
    parser.add_argument('--tensor-parallel-size', type=int, required=True,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--dataset', type=str, default='math500',
                        choices=['math500', 'gsm8k', 'amc23', 'aime', 'csqa', 'gpqa', 'svamp', 'mathqa', 'imo', 'imobench'],
                        help='Dataset to use (default: math500)')
    parser.add_argument('--level', type=int, default=None,
                        help='For MATH-500, filter by difficulty level 1-5 (default: all levels)')
    parser.add_argument('--model', type=str, default='llama8b',
                        choices=['llama8b', 'llama70b', 'qwen7b', 'qwen14b', 'qwen32b', 'qwen2b', 'llama3b', 'phi4b'],
                        help='Model to use (default: llama8b)')
    parser.add_argument('--n-problems', type=int, default=100,
                        help='Number of problems to process (default: 100)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature for generation (default: 1.0)')
    parser.add_argument('--solution-types', type=str, nargs='+', default=['tot', 'cot'],
                        choices=['tot', 'cot'],
                        help='Solution types to generate (default: both tot and cot)')
    parser.add_argument('--force-regenerate', action='store_true',
                        help='Force regeneration even if cache exists')

    args = parser.parse_args()

    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)

    logger.info("="*100)
    logger.info("ToT and CoT Solution Generation with Caching")
    logger.info("="*100)
    logger.info(f"Dataset: {dataset_info['name']}")
    if args.level is not None:
        logger.info(f"Level filter: {args.level}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"Number of problems: {args.n_problems}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Solution types: {args.solution_types}")
    logger.info(f"Force regenerate: {args.force_regenerate}")
    logger.info("="*100)

    # Load problems
    logger.info(f"Loading {args.dataset} dataset...")
    problems = load_dataset_by_name(
        dataset_name=args.dataset,
        n_problems=args.n_problems,
        level=args.level,
        seed=SEED
    )

    # Initialize model
    logger.info(f"Initializing model '{args.model}' on GPUs {args.gpus}...")
    manager = initialize_model(
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        model_name=args.model
    )

    # Generate or load ToT solutions
    if 'tot' in args.solution_types:
        logger.info("\n" + "="*100)
        logger.info("ToT Solution Generation")
        logger.info("="*100)

        if not args.force_regenerate:
            tot_solutions = load_initial_solutions(
                model_name=args.model,
                dataset_name=args.dataset,
                n_problems=args.n_problems,
                seed=SEED,
                temperature=args.temperature,
                max_depth=MAX_DEPTH,
                max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT,
                solution_type='tot'
            )
        else:
            tot_solutions = None

        if tot_solutions is None:
            logger.info("Generating new ToT solutions...")
            tot_solutions = generate_tot_solutions(
                manager=manager,
                problems=problems,
                temperature=args.temperature,
                max_depth=MAX_DEPTH,
                max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT
            )

            # Save to cache
            save_initial_solutions(
                solutions=tot_solutions,
                model_name=args.model,
                dataset_name=args.dataset,
                n_problems=args.n_problems,
                seed=SEED,
                temperature=args.temperature,
                max_depth=MAX_DEPTH,
                max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT,
                solution_type='tot'
            )

        # Print summary
        tot_correct = sum(1 for s in tot_solutions if s['correct'])
        logger.info(f"\nToT Summary:")
        logger.info(f"  Total problems: {len(tot_solutions)}")
        logger.info(f"  Correct: {tot_correct}/{len(tot_solutions)} ({100*tot_correct/len(tot_solutions):.1f}%)")

    # Generate or load CoT solutions
    if 'cot' in args.solution_types:
        logger.info("\n" + "="*100)
        logger.info("CoT Solution Generation")
        logger.info("="*100)

        if not args.force_regenerate:
            cot_solutions = load_initial_solutions(
                model_name=args.model,
                dataset_name=args.dataset,
                n_problems=args.n_problems,
                seed=SEED,
                temperature=args.temperature,
                max_depth=MAX_DEPTH,
                max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT,
                solution_type='cot'
            )
        else:
            cot_solutions = None

        if cot_solutions is None:
            logger.info("Generating new CoT solutions...")
            cot_solutions = generate_cot_solutions(
                manager=manager,
                problems=problems,
                temperature=args.temperature,
                max_tokens=2048
            )

            # Save to cache
            save_initial_solutions(
                solutions=cot_solutions,
                model_name=args.model,
                dataset_name=args.dataset,
                n_problems=args.n_problems,
                seed=SEED,
                temperature=args.temperature,
                max_depth=MAX_DEPTH,
                max_tokens_per_thought=MAX_TOKENS_PER_THOUGHT,
                solution_type='cot'
            )

        # Print summary
        cot_correct = sum(1 for s in cot_solutions if s['correct'])
        logger.info(f"\nCoT Summary:")
        logger.info(f"  Total problems: {len(cot_solutions)}")
        logger.info(f"  Correct: {cot_correct}/{len(cot_solutions)} ({100*cot_correct/len(cot_solutions):.1f}%)")

    # Cleanup
    logger.info("\n" + "="*100)
    logger.info("Generation complete!")
    logger.info("="*100)
    manager.unload_base_model()


if __name__ == "__main__":
    main()
