#!/usr/bin/env python3
"""
Match Error Localization Results from Third Party Processing

This script matches judge responses from offline batch processing back to
the original prompts and computes agreement metrics.

Usage:
    python match_error_localization_results.py \
        --prompts prompts.jsonl \
        --responses responses.jsonl \
        --output agreement_results.json

Input response format (JSONL):
    {
        "row_id": "matching row_id from prompts.jsonl",
        "judge_response": "full response text from the judge model",
        "judge_decision": optional pre-parsed decision (step number, YES/NO, or error quote)
    }

Output:
    JSON file with agreement metrics and detailed results per experiment.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from difflib import SequenceMatcher


# =============================================================================
# PARSING UTILITIES
# =============================================================================

def extract_step_number(text: str) -> Optional[int]:
    """
    Extract step number from model response.

    Looks for patterns like:
    - \\boxed{5}
    - \\boxed{0}
    - \boxed{5}

    Returns None if not found.
    """
    if not text:
        return None

    # Look for \boxed{number} or \\boxed{number}
    pattern = r'\\*boxed\{(\d+)\}'
    matches = re.findall(pattern, text)

    if matches:
        # Take the last occurrence (model's final decision)
        return int(matches[-1])

    return None


def extract_yes_no(text: str) -> Optional[str]:
    """
    Extract YES/NO from incremental mode response.

    Returns 'YES', 'NO', or None
    """
    if not text:
        return None

    # Look for \boxed{YES} or \boxed{NO}
    pattern = r'\\*boxed\{(YES|NO)\}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return matches[-1].upper()

    return None


def extract_boxed_answer(text: str) -> str:
    """
    Extract answer from \\boxed{} format.
    Handles nested braces correctly.

    Returns:
        The content inside the last \\boxed{} in the text,
        or "NO ANSWER" if not found.
    """
    if not text:
        return "NO ANSWER"

    # Match \boxed{ with 0 or more backslashes for flexibility
    matches = list(re.finditer(r'\\*boxed\{', text))
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


def extract_error_quote_from_response(response: str) -> Optional[str]:
    """Extract quoted error text from judge response."""
    if not response:
        return None

    boxed = extract_boxed_answer(response)

    if boxed in ["CORRECT", "NO_ERROR", "NO ANSWER"]:
        return None

    if "ERROR_QUOTE:" in boxed:
        quote_match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed)
        if quote_match:
            return quote_match.group(1)

    # Fallback: find any quoted text (20+ chars) in the boxed content
    quote_matches = re.findall(r'"([^"]{20,})"', boxed)
    if quote_matches:
        return quote_matches[0]

    # Last resort: find quotes anywhere in response
    quote_matches = re.findall(r'"([^"]{20,})"', response)
    if quote_matches:
        return quote_matches[-1]

    return None


def find_truncation_point(solution: str, error_quote: Optional[str]) -> Optional[int]:
    """Find character position where error quote begins in solution."""
    if error_quote is None or not solution:
        return None

    idx = solution.find(error_quote)
    if idx != -1:
        return idx

    return None


def compute_quote_overlap(quote1: Optional[str], quote2: Optional[str]) -> float:
    """Compute overlap between two quoted strings as a percentage."""
    if quote1 is None or quote2 is None:
        return 0.0

    # Find longest common substring
    matcher = SequenceMatcher(None, quote1, quote2)
    match = matcher.find_longest_match(0, len(quote1), 0, len(quote2))

    if match.size == 0:
        return 0.0

    overlap_length = match.size
    min_length = min(len(quote1), len(quote2))

    return overlap_length / min_length


# =============================================================================
# AGREEMENT COMPUTATION
# =============================================================================

def parse_judge_decision(response_entry: Dict, prompt_entry: Dict) -> Dict:
    """
    Parse the judge's decision based on expected format.

    Returns a dict with:
        - judge_decision: parsed decision
        - parse_success: whether parsing succeeded
        - raw_response: the raw response text
    """
    expected_format = prompt_entry.get('expected_format', 'step_number')
    raw_response = response_entry.get('judge_response', '')

    # Check if pre-parsed decision provided
    if 'judge_decision' in response_entry and response_entry['judge_decision'] is not None:
        return {
            'judge_decision': response_entry['judge_decision'],
            'parse_success': True,
            'raw_response': raw_response
        }

    # Parse based on expected format
    if expected_format == 'step_number':
        decision = extract_step_number(raw_response)
        return {
            'judge_decision': decision,
            'parse_success': decision is not None,
            'raw_response': raw_response
        }

    elif expected_format == 'yes_no':
        decision = extract_yes_no(raw_response)
        return {
            'judge_decision': decision,
            'parse_success': decision is not None,
            'raw_response': raw_response
        }

    elif expected_format == 'error_quote':
        decision = extract_error_quote_from_response(raw_response)
        # For error_quote, None can mean CORRECT/NO_ERROR
        boxed = extract_boxed_answer(raw_response)
        is_correct = boxed in ["CORRECT", "NO_ERROR"]
        return {
            'judge_decision': decision,
            'judge_says_correct': is_correct,
            'parse_success': decision is not None or is_correct,
            'raw_response': raw_response
        }

    return {
        'judge_decision': None,
        'parse_success': False,
        'raw_response': raw_response
    }


def compute_agreement(prompt_entry: Dict, parsed_decision: Dict) -> Dict:
    """
    Compute agreement between original and judge decisions.

    Returns dict with agreement metrics.
    """
    expected_format = prompt_entry.get('expected_format', 'step_number')
    original_decision = prompt_entry.get('original_decision')
    judge_decision = parsed_decision.get('judge_decision')

    result = {
        'parse_success': parsed_decision['parse_success'],
        'original_decision': original_decision,
        'judge_decision': judge_decision,
        'expected_format': expected_format
    }

    if not parsed_decision['parse_success']:
        result['agreement'] = None
        return result

    if expected_format == 'step_number':
        # Exact match on step number
        result['agreement'] = (original_decision == judge_decision)
        result['both_no_error'] = (original_decision == 0 and judge_decision == 0)
        result['both_found_error'] = (original_decision > 0 and judge_decision > 0)
        result['same_step'] = result['agreement']

    elif expected_format == 'yes_no':
        # Exact match on YES/NO
        result['agreement'] = (original_decision == judge_decision)

    elif expected_format == 'error_quote':
        # More complex: need to compare error quotes
        judge_says_correct = parsed_decision.get('judge_says_correct', False)

        if original_decision is None:
            # Original said correct
            result['agreement'] = judge_says_correct
            result['both_correct'] = judge_says_correct
        elif judge_says_correct:
            # Judge says correct but original found error
            result['agreement'] = False
            result['both_correct'] = False
        else:
            # Both found errors - compute quote overlap
            quote_overlap = compute_quote_overlap(original_decision, judge_decision)
            result['quote_overlap'] = quote_overlap
            result['high_overlap'] = quote_overlap > 0.5
            result['agreement'] = quote_overlap > 0.5
            result['both_correct'] = False

    return result


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file into a list of dicts."""
    entries = []
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
    return entries


def main():
    parser = argparse.ArgumentParser(
        description='Match error localization results from third party processing'
    )
    parser.add_argument(
        '--prompts',
        required=True,
        help='Input JSONL file with prompts (from export_error_localization_prompts.py)'
    )
    parser.add_argument(
        '--responses',
        required=True,
        help='Input JSONL file with judge responses'
    )
    parser.add_argument(
        '--output',
        default='agreement_results.json',
        help='Output JSON file for results (default: agreement_results.json)'
    )
    parser.add_argument(
        '--detailed-output',
        help='Optional: Output JSONL file with per-instance details'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Load prompts
    print(f"Loading prompts from: {args.prompts}")
    prompts = load_jsonl(Path(args.prompts))
    print(f"  Loaded {len(prompts)} prompts")

    # Create lookup by row_id (skip prompts missing row_id)
    prompts_by_id = {p['row_id']: p for p in prompts if 'row_id' in p}

    # Load responses
    print(f"Loading responses from: {args.responses}")
    responses = load_jsonl(Path(args.responses))
    print(f"  Loaded {len(responses)} responses")

    # Match and compute agreements
    results = []
    matched_count = 0
    unmatched_responses = []

    for resp in responses:
        row_id = resp.get('row_id')
        if row_id not in prompts_by_id:
            unmatched_responses.append(row_id)
            continue

        prompt = prompts_by_id[row_id]
        parsed = parse_judge_decision(resp, prompt)
        agreement = compute_agreement(prompt, parsed)

        result = {
            'row_id': row_id,
            'experiment_id': prompt.get('experiment_id'),
            'experiment_type': prompt.get('experiment_type'),
            'model': prompt.get('model'),
            'dataset': prompt.get('dataset'),
            'autonomy_level': prompt.get('autonomy_level'),
            'problem_id': prompt.get('problem_id'),
            'iteration': prompt.get('iteration'),
            **agreement
        }
        results.append(result)
        matched_count += 1

    print(f"\nMatched {matched_count} responses")
    if unmatched_responses:
        print(f"Warning: {len(unmatched_responses)} responses could not be matched")
        if args.verbose:
            print(f"  Unmatched IDs: {unmatched_responses[:10]}...")

    # Find prompts without responses
    responded_ids = {r.get('row_id') for r in responses}
    missing_responses = [p['row_id'] for p in prompts if p['row_id'] not in responded_ids]
    if missing_responses:
        print(f"Warning: {len(missing_responses)} prompts have no responses")

    # Compute aggregate statistics
    print("\nComputing statistics...")

    # Overall stats
    total = len(results)
    parse_successes = sum(1 for r in results if r.get('parse_success'))
    agreements = sum(1 for r in results if r.get('agreement') == True)
    disagreements = sum(1 for r in results if r.get('agreement') == False)

    overall_stats = {
        'total_matched': total,
        'parse_successes': parse_successes,
        'parse_failures': total - parse_successes,
        'agreements': agreements,
        'disagreements': disagreements,
        'agreement_rate': agreements / parse_successes if parse_successes > 0 else 0,
        'parse_success_rate': parse_successes / total if total > 0 else 0
    }

    # Stats by experiment type
    by_experiment_type = defaultdict(lambda: {'total': 0, 'agreements': 0, 'parse_successes': 0})
    for r in results:
        exp_type = r.get('experiment_type', 'unknown')
        by_experiment_type[exp_type]['total'] += 1
        if r.get('parse_success'):
            by_experiment_type[exp_type]['parse_successes'] += 1
        if r.get('agreement') == True:
            by_experiment_type[exp_type]['agreements'] += 1

    for exp_type, stats in by_experiment_type.items():
        if stats['parse_successes'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['parse_successes']
        else:
            stats['agreement_rate'] = 0

    # Stats by model
    by_model = defaultdict(lambda: {'total': 0, 'agreements': 0, 'parse_successes': 0})
    for r in results:
        model = r.get('model', 'unknown')
        by_model[model]['total'] += 1
        if r.get('parse_success'):
            by_model[model]['parse_successes'] += 1
        if r.get('agreement') == True:
            by_model[model]['agreements'] += 1

    for model, stats in by_model.items():
        if stats['parse_successes'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['parse_successes']
        else:
            stats['agreement_rate'] = 0

    # Stats by dataset
    by_dataset = defaultdict(lambda: {'total': 0, 'agreements': 0, 'parse_successes': 0})
    for r in results:
        dataset = r.get('dataset', 'unknown')
        by_dataset[dataset]['total'] += 1
        if r.get('parse_success'):
            by_dataset[dataset]['parse_successes'] += 1
        if r.get('agreement') == True:
            by_dataset[dataset]['agreements'] += 1

    for dataset, stats in by_dataset.items():
        if stats['parse_successes'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['parse_successes']
        else:
            stats['agreement_rate'] = 0

    # Stats by autonomy level
    by_autonomy = defaultdict(lambda: {'total': 0, 'agreements': 0, 'parse_successes': 0})
    for r in results:
        autonomy = r.get('autonomy_level', 'unknown')
        by_autonomy[autonomy]['total'] += 1
        if r.get('parse_success'):
            by_autonomy[autonomy]['parse_successes'] += 1
        if r.get('agreement') == True:
            by_autonomy[autonomy]['agreements'] += 1

    for autonomy, stats in by_autonomy.items():
        if stats['parse_successes'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['parse_successes']
        else:
            stats['agreement_rate'] = 0

    # Stats by experiment (for per-experiment breakdown)
    by_experiment = defaultdict(lambda: {'total': 0, 'agreements': 0, 'parse_successes': 0})
    for r in results:
        exp_id = r.get('experiment_id', 'unknown')
        by_experiment[exp_id]['total'] += 1
        if r.get('parse_success'):
            by_experiment[exp_id]['parse_successes'] += 1
        if r.get('agreement') == True:
            by_experiment[exp_id]['agreements'] += 1

    for exp_id, stats in by_experiment.items():
        if stats['parse_successes'] > 0:
            stats['agreement_rate'] = stats['agreements'] / stats['parse_successes']
        else:
            stats['agreement_rate'] = 0

    # Compile final output
    output = {
        'analysis_timestamp': datetime.now().isoformat(),
        'prompts_file': args.prompts,
        'responses_file': args.responses,
        'overall': overall_stats,
        'by_experiment_type': dict(by_experiment_type),
        'by_model': dict(by_model),
        'by_dataset': dict(by_dataset),
        'by_autonomy_level': dict(by_autonomy),
        'by_experiment': dict(by_experiment),
        'warnings': {
            'unmatched_responses': len(unmatched_responses),
            'missing_responses': len(missing_responses)
        }
    }

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Write detailed output if requested
    if args.detailed_output:
        detailed_path = Path(args.detailed_output)
        with open(detailed_path, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        print(f"Detailed results saved to: {detailed_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nOverall:")
    print(f"  Total matched: {overall_stats['total_matched']}")
    print(f"  Parse success rate: {overall_stats['parse_success_rate']:.1%}")
    print(f"  Agreement rate: {overall_stats['agreement_rate']:.1%}")
    print(f"    Agreements: {overall_stats['agreements']}")
    print(f"    Disagreements: {overall_stats['disagreements']}")

    print(f"\nBy experiment type:")
    for exp_type, stats in sorted(by_experiment_type.items()):
        print(f"  {exp_type}: {stats['agreement_rate']:.1%} ({stats['agreements']}/{stats['parse_successes']})")

    print(f"\nBy model:")
    for model, stats in sorted(by_model.items()):
        print(f"  {model}: {stats['agreement_rate']:.1%} ({stats['agreements']}/{stats['parse_successes']})")

    print(f"\nBy dataset:")
    for dataset, stats in sorted(by_dataset.items()):
        print(f"  {dataset}: {stats['agreement_rate']:.1%} ({stats['agreements']}/{stats['parse_successes']})")

    print(f"\nBy autonomy level:")
    for autonomy, stats in sorted(by_autonomy.items()):
        print(f"  {autonomy}: {stats['agreement_rate']:.1%} ({stats['agreements']}/{stats['parse_successes']})")

    return 0


if __name__ == '__main__':
    exit(main())
