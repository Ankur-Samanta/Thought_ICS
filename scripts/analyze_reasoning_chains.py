#!/usr/bin/env python3
"""
Analyze reasoning chains for Tree of Thought errors
"""

import json
import re
from pathlib import Path

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format."""
    if not text:
        return "NO ANSWER FOUND"

    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return "NO ANSWER FOUND"

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

    return "NO ANSWER FOUND"

def analyze_chain(problem_idx: int, comparison: dict):
    """Analyze the reasoning chain for a specific problem."""
    problem = comparison['problem']
    tree_result = comparison['tree']
    ground_truth = extract_boxed_answer(problem['solution'])
    tree_answer = tree_result['primary_answer']

    print(f"\n{'='*100}")
    print(f"PROBLEM {problem_idx}: {problem['subject']} (Level {problem['level']})")
    print(f"{'='*100}")
    print(f"\nQuestion: {problem['problem'][:200]}...")
    print(f"\nGround Truth Answer: {ground_truth}")
    print(f"Tree Answer: {tree_answer}")
    print(f"Tree Stats: {tree_result['stats']['total_nodes']} nodes, depth {tree_result['stats']['max_depth']}")

    if not tree_result['completed_paths']:
        print("\nNo completed paths found!")
        return

    # Analyze the first (primary) completed path
    path = tree_result['completed_paths'][0]

    print(f"\n{'='*100}")
    print("REASONING CHAIN ANALYSIS")
    print(f"{'='*100}")
    print(f"\nTotal steps in chain: {len(path) - 1}")  # -1 for the question

    # Show each step with analysis
    for i, thought in enumerate(path):
        if i == 0:
            print(f"\n[QUESTION]")
            print(f"{thought[:200]}...")
            continue

        print(f"\n[STEP {i}]")
        print(thought)

        # Check if this step contains the error
        # Look for numerical values, equations, assertions
        if i > 1:  # After first step
            # Check for arithmetic errors
            if '=' in thought:
                print(f"  → Contains equation/calculation")

            # Check for logical jumps
            if any(word in thought.lower() for word in ['therefore', 'thus', 'so', 'hence']):
                print(f"  → Makes conclusion")

            # Check for assumptions
            if any(word in thought.lower() for word in ['assume', 'suppose', 'let']):
                print(f"  → Makes assumption")

    print(f"\n{'='*100}")
    print("POTENTIAL ERROR ANALYSIS")
    print(f"{'='*100}")

    # Try to identify where it went wrong
    print("\nLooking for potential errors...")

    for i in range(1, len(path)):
        thought = path[i]

        # Check for early wrong assumptions
        if i <= 3:
            if 'assume' in thought.lower() or 'suppose' in thought.lower():
                print(f"\n⚠️  Early assumption at Step {i}:")
                print(f"   {thought[:150]}...")

        # Check for computational errors (looking for specific numbers)
        numbers = re.findall(r'-?\d+\.?\d*', thought)
        if numbers and len(numbers) > 2:
            print(f"\n⚠️  Multiple calculations at Step {i}:")
            print(f"   Numbers: {numbers}")
            print(f"   {thought[:150]}...")

    # Check final step
    if len(path) > 1:
        final_step = path[-1]
        print(f"\n🎯 Final step (Step {len(path)-1}):")
        print(f"   {final_step}")
        if '\\boxed{' in final_step:
            print(f"   ✓ Contains boxed answer")

def main():
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        results_files = sorted(Path(".").glob("math_comparison_*.json"))
        if not results_files:
            print("No results files found!")
            sys.exit(1)
        json_file = results_files[-1]

    print(f"Analyzing reasoning chains from: {json_file}\n")

    with open(json_file, 'r') as f:
        data = json.load(f)

    comparisons = data['comparisons']

    # Identify which problems Tree got wrong
    wrong_problems = []

    for i, comp in enumerate(comparisons, 1):
        ground_truth = extract_boxed_answer(comp['problem']['solution'])
        tree_answer = comp['tree']['primary_answer']

        # Normalize for comparison
        gt_norm = ground_truth.replace('\\', '').replace(' ', '').lower()
        tree_norm = tree_answer.replace('\\', '').replace(' ', '').lower()

        if gt_norm != tree_norm:
            wrong_problems.append((i, comp))

    print(f"Found {len(wrong_problems)} problems where Tree got wrong answer\n")

    # Analyze each wrong problem
    for problem_idx, comparison in wrong_problems:
        analyze_chain(problem_idx, comparison)
        print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
