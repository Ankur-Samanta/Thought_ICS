#!/usr/bin/env python3
"""
Analyze accuracy of baseline CoT vs Tree of Thought against ground truth answers
"""

import json
import re
from pathlib import Path
from collections import defaultdict

def extract_boxed_answer(text: str) -> str:
    """Extract answer from \\boxed{} format, handling nested braces."""
    if not text:
        return "NO ANSWER FOUND"

    # Find all \boxed{...} patterns
    matches = list(re.finditer(r'\\boxed\{', text))
    if not matches:
        return "NO ANSWER FOUND"

    # Start from the last \boxed{ occurrence (the final answer)
    start_pos = matches[-1].end()

    # Count braces to find the matching closing brace
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

def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    if answer == "NO ANSWER FOUND":
        return answer

    # Remove whitespace
    answer = answer.strip()

    # Remove LaTeX formatting
    answer = answer.replace('\\', '')
    answer = answer.replace(' ', '')
    answer = answer.replace(',', '')

    # Convert to lowercase for text answers
    answer_lower = answer.lower()

    return answer_lower

def answers_match(ans1: str, ans2: str) -> bool:
    """Check if two answers match after normalization."""
    norm1 = normalize_answer(ans1)
    norm2 = normalize_answer(ans2)

    # Direct match
    if norm1 == norm2:
        return True

    # Try evaluating as numbers
    try:
        # Handle fractions
        if 'frac' in ans1 or 'frac' in ans2:
            # Extract numerator and denominator
            def eval_frac(s):
                s = s.replace('\\', '').replace(' ', '')
                if 'frac' in s:
                    match = re.search(r'frac\{([^}]+)\}\{([^}]+)\}', s)
                    if match:
                        num, denom = match.groups()
                        return float(num) / float(denom)
                return float(s)

            val1 = eval_frac(ans1)
            val2 = eval_frac(ans2)
            return abs(val1 - val2) < 1e-6
    except:
        pass

    return False

def analyze_results(json_file: str):
    """Analyze accuracy of baseline vs tree against ground truth."""

    with open(json_file, 'r') as f:
        data = json.load(f)

    comparisons = data['comparisons']

    print("="*100)
    print("ACCURACY ANALYSIS: Baseline CoT vs Tree of Thought")
    print("="*100)

    baseline_correct = 0
    tree_correct = 0
    both_correct = 0
    both_wrong = 0
    baseline_only = 0
    tree_only = 0

    stats_by_level = defaultdict(lambda: {'baseline': 0, 'tree': 0, 'total': 0})
    stats_by_subject = defaultdict(lambda: {'baseline': 0, 'tree': 0, 'total': 0})

    for i, comp in enumerate(comparisons, 1):
        problem = comp['problem']
        baseline_ans = comp['baseline']['answer']
        tree_ans = comp['tree']['primary_answer']
        ground_truth = extract_boxed_answer(problem['solution'])

        level = problem['level']
        subject = problem['subject']

        baseline_match = answers_match(baseline_ans, ground_truth)
        tree_match = answers_match(tree_ans, ground_truth)

        print(f"\n{'='*100}")
        print(f"Problem {i}: {subject} (Level {level})")
        print(f"{'='*100}")
        print(f"Question: {problem['problem'][:150]}...")
        print(f"\nGround Truth: {ground_truth}")
        print(f"Baseline:     {baseline_ans}  {'✓' if baseline_match else '✗'}")
        print(f"Tree:         {tree_ans}  {'✓' if tree_match else '✗'}")

        if baseline_match:
            baseline_correct += 1
            stats_by_level[level]['baseline'] += 1
            stats_by_subject[subject]['baseline'] += 1
        if tree_match:
            tree_correct += 1
            stats_by_level[level]['tree'] += 1
            stats_by_subject[subject]['tree'] += 1

        stats_by_level[level]['total'] += 1
        stats_by_subject[subject]['total'] += 1

        if baseline_match and tree_match:
            both_correct += 1
            print("Status: Both correct ✓✓")
        elif baseline_match and not tree_match:
            baseline_only += 1
            print("Status: Only baseline correct ✓✗")
        elif tree_match and not baseline_match:
            tree_only += 1
            print("Status: Only tree correct ✗✓")
        else:
            both_wrong += 1
            print("Status: Both wrong ✗✗")

        # Show tree stats
        tree_stats = comp['tree']['stats']
        print(f"\nTree Stats: {tree_stats['total_nodes']} nodes, depth {tree_stats['max_depth']}, {tree_stats['completed_paths']} paths")

        # Show answer frequency if multiple answers
        if comp['tree']['unique_answers'] > 1:
            print(f"Tree found {comp['tree']['unique_answers']} different answers:")
            for ans_freq in comp['tree']['answer_frequency']:
                print(f"  - {ans_freq['answer']}: {ans_freq['count']} time(s)")

    # Overall summary
    total = len(comparisons)
    print(f"\n{'='*100}")
    print("OVERALL ACCURACY")
    print(f"{'='*100}")
    print(f"Baseline CoT:     {baseline_correct}/{total} ({baseline_correct/total*100:.1f}%)")
    print(f"Tree of Thought:  {tree_correct}/{total} ({tree_correct/total*100:.1f}%)")
    print(f"\nBreakdown:")
    print(f"  Both correct:        {both_correct}/{total}")
    print(f"  Only baseline:       {baseline_only}/{total}")
    print(f"  Only tree:           {tree_only}/{total}")
    print(f"  Both wrong:          {both_wrong}/{total}")

    # By difficulty level
    print(f"\n{'='*100}")
    print("ACCURACY BY DIFFICULTY LEVEL")
    print(f"{'='*100}")
    for level in sorted(stats_by_level.keys()):
        stats = stats_by_level[level]
        baseline_pct = stats['baseline']/stats['total']*100 if stats['total'] > 0 else 0
        tree_pct = stats['tree']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"Level {level}: Baseline {stats['baseline']}/{stats['total']} ({baseline_pct:.1f}%), "
              f"Tree {stats['tree']}/{stats['total']} ({tree_pct:.1f}%)")

    # By subject
    print(f"\n{'='*100}")
    print("ACCURACY BY SUBJECT")
    print(f"{'='*100}")
    for subject in sorted(stats_by_subject.keys()):
        stats = stats_by_subject[subject]
        baseline_pct = stats['baseline']/stats['total']*100 if stats['total'] > 0 else 0
        tree_pct = stats['tree']/stats['total']*100 if stats['total'] > 0 else 0
        print(f"{subject}: Baseline {stats['baseline']}/{stats['total']} ({baseline_pct:.1f}%), "
              f"Tree {stats['tree']}/{stats['total']} ({tree_pct:.1f}%)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Find the most recent results file
        results_files = sorted(Path(".").glob("math_comparison_*.json"))
        if not results_files:
            print("No results files found!")
            sys.exit(1)
        json_file = results_files[-1]

    print(f"Analyzing: {json_file}\n")
    analyze_results(json_file)
