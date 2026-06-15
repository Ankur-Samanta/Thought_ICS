#!/usr/bin/env python3
"""
Simple approach: Pre-segment CoT solutions into logical steps and format as ToT.
No actual ToT generation - just parsing and reformatting.
"""

import json
import logging
import re
from pathlib import Path
from datasets import load_dataset
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def extract_first_solution(full_solution: str) -> str:
    """Extract the first solution approach from AIME dataset solutions."""
    import re
    author_pattern = r'\n~[a-zA-Z0-9_]+'
    match = re.search(author_pattern, full_solution)

    if match:
        first_solution = full_solution[:match.start()].strip()
        return first_solution
    else:
        return full_solution.strip()


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer from \\boxed{} format."""
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return None


def segment_solution_into_steps(solution: str) -> List[str]:
    """
    Segment a CoT solution into logical steps.

    Uses heuristics:
    - Split on sentences
    - Group into logical steps based on line breaks and content
    """
    # Split on line breaks first
    lines = [line.strip() for line in solution.split('\n') if line.strip()]

    steps = []
    current_step = []

    for line in lines:
        # LaTeX environments often mark logical boundaries
        if line.startswith('\\begin{') or line.startswith('\\end{'):
            if current_step:
                steps.append(' '.join(current_step))
                current_step = []
            current_step.append(line)
        # Standalone equations
        elif line.startswith('\\[') or line.endswith('\\]'):
            if current_step:
                current_step.append(line)
                steps.append(' '.join(current_step))
                current_step = []
            else:
                steps.append(line)
        # Regular text - accumulate
        else:
            current_step.append(line)
            # If line ends with period, might be end of thought
            if line.endswith('.') and len(' '.join(current_step)) > 50:
                steps.append(' '.join(current_step))
                current_step = []

    # Add remaining
    if current_step:
        steps.append(' '.join(current_step))

    return [s for s in steps if s]  # Filter empty


def convert_cot_to_tot_simple(
    problem: str,
    solution: str,
    correct_answer: str
) -> Dict:
    """
    Convert CoT to ToT by pre-segmenting and reformatting.
    Much simpler than using the actual ToT generator.
    """
    logging.info(f"\nConverting CoT to ToT (simple approach)...")
    logging.info(f"Problem: {problem[:100]}...")

    # Extract first solution
    first_solution = extract_first_solution(solution)

    # Segment into steps
    steps = segment_solution_into_steps(first_solution)

    # Format as ToT steps
    tot_steps = []
    for i, step_content in enumerate(steps, 1):
        tot_steps.append({
            'step_number': i,
            'content': step_content
        })

    # Extract final answer
    final_answer = extract_answer(first_solution)

    # Build result
    tot_structure = {
        'problem': problem,
        'ground_truth_solution': solution,
        'ground_truth_answer': correct_answer,
        'tot_steps': tot_steps,
        'tot_final_answer': final_answer,
        'num_steps': len(tot_steps),
        'answer_matches': final_answer == correct_answer if final_answer else False
    }

    logging.info(f"Generated {len(tot_steps)} steps")
    logging.info(f"Answer match: {tot_structure['answer_matches']}")

    return tot_structure


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert CoT to ToT (simple approach)')
    parser.add_argument('--n-problems', type=int, default=3,
                      help='Number of problems to process')
    parser.add_argument('--output-dir', type=str, default='tot_dataset',
                      help='Output directory for generated ToT dataset')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    logging.info("="*80)
    logging.info("ToT Dataset Generation from CoT Solutions (Simple Approach)")
    logging.info("="*80)
    logging.info(f"Processing {args.n_problems} problems")
    logging.info(f"Output: {output_dir}")
    logging.info("="*80)

    # Load AIME dataset
    logging.info("Loading AIME dataset...")
    dataset = load_dataset("AI-MO/aimo-validation-aime")
    problems = dataset['train']

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

        # Convert using simple approach
        tot_result = convert_cot_to_tot_simple(
            problem=problem['problem'],
            solution=problem['solution'],
            correct_answer=problem['answer']
        )

        # Add metadata
        tot_result['id'] = problem['id']
        tot_result['url'] = problem['url']

        results.append(tot_result)

        # Log results
        logging.info(f"Generated {tot_result['num_steps']} steps")
        logging.info(f"ToT answer: {tot_result['tot_final_answer']}")
        logging.info(f"Matches ground truth: {tot_result['answer_matches']}")

        # Print the steps
        logging.info("\nGenerated ToT structure:")
        for step in tot_result['tot_steps'][:5]:  # Show first 5
            logging.info(f"  Step {step['step_number']}: {step['content'][:80]}...")

    # Save results
    output_file = output_dir / f'tot_simple_{args.n_problems}_problems.json'
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
