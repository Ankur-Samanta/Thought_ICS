#!/usr/bin/env python3
"""
Explore alternative reasoning chains by generating trees from each prefix of a failed solution
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

def generate_tree_from_prefix(manager, problem: str, prefix_steps: List[str]):
    """Generate ONE tree starting from a given prefix of thoughts."""

    # Build the prompt with the prefix
    if prefix_steps:
        prompt = problem + "\n\n" + "\n\n".join(prefix_steps)
    else:
        prompt = problem

    logger.info(f"\n{'='*100}")
    logger.info(f"GENERATING TREE FROM PREFIX LENGTH {len(prefix_steps)}")
    logger.info(f"{'='*100}")
    if prefix_steps:
        logger.info(f"Prefix steps:")
        for i, step in enumerate(prefix_steps, 1):
            logger.info(f"  {i}. {step[:100]}...")
    else:
        logger.info(f"Starting from scratch (no prefix)")

    # Generate complete reasoning tree
    tot = TreeOfThought(
        model_manager=manager,
        n_rollouts=1,  # Just 1 rollout per node
        max_depth=15,  # Reasonable depth limit
        temperature=0.7,
        max_tokens_per_thought=150,
        max_batch_size=32,
        traversal_strategy="dfs"
    )

    # Generate tree from this prefix
    root = tot.generate_tree(prompt, verbose=False)  # Less verbose

    # Get completed paths
    completed_paths = tot.get_completed_paths(root)
    boxed_answers = tot.extract_boxed_answers(root)

    # Count nodes
    total_nodes = 0
    def count_nodes(node):
        nonlocal total_nodes
        total_nodes += 1
        for child in node.children:
            count_nodes(child)
    count_nodes(root)

    logger.info(f"\nTree generated:")
    logger.info(f"  Total nodes: {total_nodes}")
    logger.info(f"  Completed paths: {len(completed_paths)}")
    logger.info(f"  Boxed answers: {len(boxed_answers)}")

    results = []
    for path in completed_paths:
        # The path includes the full prompt as first element
        # Extract just the thoughts (skip the question)
        all_thoughts = path[1:]

        # The first len(prefix_steps) thoughts are from the prefix
        new_thoughts = all_thoughts[len(prefix_steps):]

        # Find answer for this path
        answer = None
        for ans, ans_path in boxed_answers:
            if ans_path == path:
                answer = ans
                break

        results.append({
            'prefix_steps': prefix_steps,
            'new_thoughts': new_thoughts,
            'full_chain': all_thoughts,
            'answer': answer
        })

    logger.info(f"\nAnswers found:")
    for i, result in enumerate(results, 1):
        logger.info(f"  Path {i}: {result['answer']} ({len(result['new_thoughts'])} new steps)")

    return {
        'prefix_length': len(prefix_steps),
        'total_nodes': total_nodes,
        'completed_paths': len(completed_paths),
        'paths': results
    }

def explore_problem(manager, comparison: Dict):
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

    # Show original chain
    logger.info(f"\nOriginal chain:")
    for i, thought in enumerate(original_thoughts, 1):
        logger.info(f"  Step {i}: {thought[:80]}...")

    # Explore alternatives at each prefix length
    all_explorations = []

    for step_idx in range(len(original_thoughts) + 1):
        prefix = original_thoughts[:step_idx]

        # Generate tree from this prefix
        exploration = generate_tree_from_prefix(manager, problem, prefix)
        all_explorations.append(exploration)

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
    explorations = explore_problem(manager, comparison)

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
    logger.info(f"SUMMARY")
    logger.info(f"{'='*100}")
    logger.info(f"Explored {len(explorations)} different prefix lengths")
    for i, exp in enumerate(explorations):
        logger.info(f"\nPrefix length {exp['prefix_length']}:")
        logger.info(f"  Nodes: {exp['total_nodes']}, Paths: {exp['completed_paths']}")
        answers = set(p['answer'] for p in exp['paths'] if p['answer'])
        logger.info(f"  Unique answers: {answers}")

    logger.info(f"\n{'='*100}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*100}\n")

    # Cleanup
    manager.unload_base_model()

if __name__ == "__main__":
    main()
