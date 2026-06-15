#!/usr/bin/env python3
"""
Explore alternative reasoning chains by generating rollouts from each step of a failed solution
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import logging
from typing import List, Dict
from thought_ics.thought_mdp import TreeOfThought, initialize_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_failed_problem(json_file: str, problem_idx: int):
    """Load a specific problem from the results file."""
    with open(json_file, 'r') as f:
        data = json.load(f)

    comparison = data['comparisons'][problem_idx - 1]
    return comparison

def generate_rollout_from_prefix(manager, problem: str, prefix_steps: List[str], n_rollouts: int = 2):
    """Generate complete rollouts starting from a given prefix of thoughts."""

    # Build the prompt with the prefix
    prompt = problem
    for step in prefix_steps:
        prompt += f"\n\n{step}"

    logger.info(f"\n{'='*80}")
    logger.info(f"Generating {n_rollouts} rollouts from prefix length {len(prefix_steps)}")
    logger.info(f"Prefix prompt:\n{prompt[:300]}...")
    logger.info(f"{'='*80}\n")

    # Generate complete reasoning chains
    tot = TreeOfThought(
        model_manager=manager,
        n_rollouts=n_rollouts,
        max_depth=20,
        temperature=0.7,
        max_tokens_per_thought=150,
        max_batch_size=32,
        traversal_strategy="dfs"
    )

    # Generate tree from this prefix
    root = tot.generate_tree(prompt, verbose=True)

    # Get completed paths
    completed_paths = tot.get_completed_paths(root)
    boxed_answers = tot.extract_boxed_answers(root)

    results = []
    for path in completed_paths:
        # The path will include the prefix prompt as first element
        # Extract just the new thoughts generated
        new_thoughts = path[1:]  # Skip the prompt

        results.append({
            'full_chain': [problem] + prefix_steps + new_thoughts,
            'new_thoughts': new_thoughts,
            'answer': None
        })

    # Match answers to paths
    for ans, path in boxed_answers:
        for result in results:
            if result['full_chain'] == path:
                result['answer'] = ans
                break

    return results

def explore_problem(manager, comparison: Dict, n_rollouts: int = 2):
    """Explore alternative chains at each step of the original failed solution."""

    problem_data = comparison['problem']
    original_path = comparison['tree']['completed_paths'][0]

    problem = original_path[0]
    original_thoughts = original_path[1:]

    logger.info(f"\n{'='*100}")
    logger.info(f"EXPLORING PROBLEM: {problem_data['subject']} (Level {problem_data['level']})")
    logger.info(f"{'='*100}")
    logger.info(f"Problem: {problem[:200]}...")
    logger.info(f"Original chain length: {len(original_thoughts)} steps")
    logger.info(f"Original answer: {comparison['tree']['primary_answer']}")
    logger.info(f"Correct answer: Should extract from solution")

    # Show original chain
    logger.info(f"\n{'='*100}")
    logger.info("ORIGINAL CHAIN")
    logger.info(f"{'='*100}")
    for i, thought in enumerate(original_thoughts, 1):
        logger.info(f"Step {i}: {thought}")

    # Explore alternatives at each step
    all_explorations = {}

    for step_idx in range(len(original_thoughts) + 1):
        logger.info(f"\n{'='*100}")
        logger.info(f"BRANCHING FROM STEP {step_idx} (using first {step_idx} original steps)")
        logger.info(f"{'='*100}")

        prefix = original_thoughts[:step_idx]

        # Generate rollouts from this prefix
        rollouts = generate_rollout_from_prefix(manager, problem, prefix, n_rollouts)

        all_explorations[f"step_{step_idx}"] = {
            'prefix_length': step_idx,
            'prefix': prefix,
            'rollouts': rollouts
        }

        logger.info(f"\nGenerated {len(rollouts)} complete chains from step {step_idx}:")
        for j, rollout in enumerate(rollouts, 1):
            logger.info(f"\n  Rollout {j}:")
            logger.info(f"    New thoughts: {len(rollout['new_thoughts'])} steps")
            logger.info(f"    Answer: {rollout['answer']}")
            logger.info(f"    Chain:")
            for k, thought in enumerate(rollout['new_thoughts'], step_idx + 1):
                logger.info(f"      Step {k}: {thought}")

    return all_explorations

def main():
    # Problem 5 is shortest wrong one: GCD problem, 6 steps, got wrong
    logger.info("Loading Problem 5 (Number Theory, Level 4)")
    logger.info("Question: Greatest possible value of gcd(n + 7, 2n + 1)")
    logger.info("Model got: 1, Correct answer: 13\n")

    json_file = "math_comparison_20251026_030347.json"
    comparison = load_failed_problem(json_file, problem_idx=5)

    # Initialize model
    logger.info("Initializing model on GPU 1...")
    manager = initialize_model(gpu_id=1)

    # Explore
    explorations = explore_problem(manager, comparison, n_rollouts=2)

    # Save results
    output_file = "branching_exploration.json"
    output_data = {
        'problem': comparison['problem'],
        'original_chain': comparison['tree']['completed_paths'][0],
        'original_answer': comparison['tree']['primary_answer'],
        'explorations': explorations
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n{'='*100}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*100}\n")

    # Cleanup
    manager.unload_base_model()

if __name__ == "__main__":
    main()
