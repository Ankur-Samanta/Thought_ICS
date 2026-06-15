#!/usr/bin/env python3
"""
Generate Tree-of-Thought (ToT) structured solutions from ground-truth CoT solutions.
This converts linear chain-of-thought reasoning into tree-structured reasoning.
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import sys
from pathlib import Path
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

import json
import logging
from typing import List, Dict, Optional
from datasets import load_dataset
from thought_ics.thought_mdp import initialize_model, ToTAgent, ToTEnvironment, TreeSearch
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from text using \\boxed{} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None


def extract_first_solution(full_solution: str) -> str:
    """
    Extract the first solution approach from AIME dataset solutions.

    AIME solutions contain multiple approaches by different users, separated by ~username.
    This extracts just the first coherent solution.
    """
    # Find the first author marker (e.g., ~MRENTHUSIASM, ~Littlemouse, etc.)
    import re
    author_pattern = r'\n~[a-zA-Z0-9_]+'
    match = re.search(author_pattern, full_solution)

    if match:
        # Extract everything before the first author marker
        first_solution = full_solution[:match.start()].strip()
        return first_solution
    else:
        # If no author markers found, use the whole solution
        return full_solution.strip()


def create_verbatim_reconstruction_prompt(problem: str, solution: str) -> str:
    """
    Create a custom prompt template that instructs the model to convert the CoT into step-by-step ToT format.

    Uses in-context examples to show how to break down CoT into thought-by-thought steps.

    Note: We escape curly braces in the problem and solution to avoid conflicts with
    Python's .format() method that will be called later in tree_of_thought.py
    """
    # Extract just the first solution approach (not all of them)
    first_solution = extract_first_solution(solution)

    # Escape curly braces to avoid conflict with .format() calls later
    problem_escaped = problem.replace('{', '{{').replace('}', '}}')
    solution_escaped = first_solution.replace('{', '{{').replace('}', '}}')

    return f"""You are solving a problem step-by-step by breaking down a solution.

Instructions:
1. State your next reasoning step (one observation, calculation, or deduction)
2. End each step with </thought>
3. Continue until you reach the final answer, then write it in \\boxed{{{{answer}}}} format

Example of breaking down a solution:

Original Solution: "Let S be the set of even numbers from 2 to 100. The sum is 2 + 4 + ... + 100 = 2(1 + 2 + ... + 50) = 2 * (50*51/2) = 2550."

Step-by-step breakdown:
Let S be the set of even numbers from 2 to 100</thought>
The sum can be written as 2 + 4 + ... + 100</thought>
I can factor out 2: this equals 2(1 + 2 + ... + 50)</thought>
Using the formula for sum of first n integers: 1 + 2 + ... + 50 = 50*51/2</thought>
Therefore the sum is 2 * (50*51/2) = 2 * 1275 = \\boxed{{{{2550}}}}</thought>

Now break down this solution step-by-step:

Problem: {problem_escaped}

Solution to break down:
{solution_escaped}

Generate the next step of reasoning:
"""


def generate_tot_from_cot(
    problem: str,
    solution: str,
    correct_answer: str,
    agent: ToTAgent,
    env: ToTEnvironment,
    max_depth: int = 50
) -> Dict:
    """
    Generate a Tree-of-Thought structure from a linear CoT solution.

    Uses the actual ToT generator (agent + environment + search) with a custom prompt
    that instructs the model to reconstruct the ground truth CoT verbatim.

    Args:
        problem: The problem statement
        solution: Ground-truth CoT solution (linear)
        correct_answer: The correct final answer
        agent: ToT agent (the LLM policy)
        env: ToT environment (manages states and transitions)
        max_depth: Maximum depth for the tree

    Returns:
        Dictionary with ToT structure and metadata
    """
    logging.info(f"\nConverting CoT to ToT using actual generator...")
    logging.info(f"Problem: {problem[:100]}...")

    # Create custom prompt template for verbatim reconstruction
    verbatim_prompt_template = create_verbatim_reconstruction_prompt(problem, solution)

    # Create environment with custom prompt
    env.prompt_template = verbatim_prompt_template
    env.max_depth = max_depth

    # Run single-rollout DFS to generate the ToT
    search = TreeSearch(
        agent=agent,
        env=env,
        strategy="dfs",
        n_rollouts=1,  # Single path
    )

    # Generate the tree
    root = search.search(problem, verbose=False)

    # Extract the generated path
    from thought_ics.thought_mdp import get_completed_paths, extract_boxed_answers

    completed_paths = get_completed_paths(root)

    if not completed_paths:
        logging.warning("No completed paths found!")
        return {
            'problem': problem,
            'ground_truth_solution': solution,
            'ground_truth_answer': correct_answer,
            'tot_steps': [],
            'tot_final_answer': None,
            'num_steps': 0,
            'answer_matches': False
        }

    # Take the first completed path
    path = completed_paths[0]

    # Convert path to ToT steps (skip root which is the question)
    tot_steps = []
    for i, thought in enumerate(path[1:], 1):
        tot_steps.append({
            'step_number': i,
            'content': thought
        })

    # Extract final answer
    boxed_answers = extract_boxed_answers(root)
    final_answer = boxed_answers[0][0] if boxed_answers else None

    # Build result structure
    tot_structure = {
        'problem': problem,
        'ground_truth_solution': solution,
        'ground_truth_answer': correct_answer,
        'tot_steps': tot_steps,
        'tot_final_answer': final_answer,
        'num_steps': len(tot_steps),
        'answer_matches': final_answer == correct_answer if final_answer else False
    }

    logging.info(f"Generated ToT with {len(tot_steps)} steps")
    logging.info(f"Answer match: {tot_structure['answer_matches']}")

    return tot_structure


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate ToT from CoT solutions')
    parser.add_argument('--n-problems', type=int, default=3,
                      help='Number of problems to process (for testing)')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU to use')
    parser.add_argument('--output-dir', type=str, default='tot_dataset',
                      help='Output directory for generated ToT dataset')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logging.info("="*80)
    logging.info("ToT Dataset Generation from CoT Solutions")
    logging.info("="*80)
    logging.info(f"Processing {args.n_problems} problems")
    logging.info(f"GPU: {args.gpu}")
    logging.info(f"Output: {output_dir}")
    logging.info("="*80)

    # Load AIME dataset
    logging.info("Loading AIME dataset...")
    dataset = load_dataset("AI-MO/aimo-validation-aime")
    problems = dataset['train']

    # Load model
    logging.info(f"Loading model on GPU {args.gpu}...")
    model = initialize_model(
        gpu_ids=args.gpu,
        tensor_parallel_size=1,
        model_name="llama8b"
    )
    logging.info("Model loaded!")

    # Create agent and environment
    logging.info("Creating ToT agent and environment...")
    agent = ToTAgent(
        model_manager=model,
        temperature=0.3,  # Low temperature for faithful reconstruction
        max_tokens=512
    )
    env = ToTEnvironment(max_depth=50)
    logging.info("Agent and environment created!")

    # Process problems
    results = []
    for i in range(min(args.n_problems, len(problems))):
        problem = problems[i]
        logging.info("")
        logging.info("="*80)
        logging.info(f"Problem {i+1}/{args.n_problems}")
        logging.info("="*80)
        logging.info(f"Problem: {problem['problem'][:100]}...")
        logging.info(f"Answer: {problem['answer']}")

        # Generate ToT structure using actual ToT generator
        tot_result = generate_tot_from_cot(
            problem=problem['problem'],
            solution=problem['solution'],
            correct_answer=problem['answer'],
            agent=agent,
            env=env
        )

        # Add metadata
        tot_result['id'] = problem['id']
        tot_result['url'] = problem['url']

        results.append(tot_result)

        # Log results
        logging.info(f"Generated {tot_result['num_steps']} steps")
        logging.info(f"ToT answer: {tot_result['tot_final_answer']}")
        logging.info(f"Matches ground truth: {tot_result['answer_matches']}")

        # Print the tree structure
        logging.info("\nGenerated ToT structure:")
        for step in tot_result['tot_steps']:
            logging.info(f"  Step {step['step_number']}: {step['content'][:80]}...")

    # Save results
    output_file = output_dir / f'tot_from_cot_{args.n_problems}_problems.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("")
    logging.info("="*80)
    logging.info("Summary")
    logging.info("="*80)
    logging.info(f"Processed: {len(results)} problems")

    correct_count = sum(1 for r in results if r['answer_matches'])
    logging.info(f"Correct answers: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")

    avg_steps = sum(r['num_steps'] for r in results) / len(results)
    logging.info(f"Average steps: {avg_steps:.1f}")

    logging.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
