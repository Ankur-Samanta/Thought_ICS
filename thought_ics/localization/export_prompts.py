#!/usr/bin/env python3
"""
Export Error Localization Prompts for Offline Processing

This script exports all error localization prompts from completed experiments
to a JSONL file for offline batch processing by third parties.

Usage:
    python export_error_localization_prompts.py \
        --experiments "experiments_main/experiments/eval_*" \
        --output prompts.jsonl

Output format (JSONL - one JSON object per line):
    {
        "row_id": "unique identifier for matching",
        "experiment_id": "experiment folder name",
        "experiment_type": "tot_batch|tot_incremental|cot_shared_prefix",
        "model": "model name",
        "dataset": "dataset name",
        "autonomy_level": "L1|L2|L3",
        "problem_id": "problem identifier",
        "iteration": 2,
        "prompt": "full reconstructed prompt",
        "expected_format": "step_number|yes_no|error_quote",
        "original_decision": "model's original decision",
        "original_reasoning": "full model reasoning"
    }
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# =============================================================================
# EXPERIMENT TYPE DETECTION
# =============================================================================

def detect_experiment_type(config: Dict, stats: Dict) -> Tuple[str, int]:
    """
    Detect experiment type and autonomy level from config/stats.

    Returns:
        (experiment_type, autonomy_level)
        experiment_type: 'tot_batch', 'tot_incremental', 'cot_shared_prefix'
        autonomy_level: 1, 2, or 3
    """
    # Check for baseline CoT experiments
    baseline_type = stats.get('baseline_type', '')
    if baseline_type:
        # CoT experiment
        if 'l1' in baseline_type.lower():
            autonomy_level = 1
        elif 'l2' in baseline_type.lower():
            autonomy_level = 2
        else:
            autonomy_level = 3
        return 'cot_shared_prefix', autonomy_level

    # ToT experiment - check config
    autonomy_level = config.get('autonomy_level', 3)
    error_detection_method = config.get('error_detection_method', 'batch')

    if error_detection_method == 'incremental':
        return 'tot_incremental', autonomy_level
    else:
        return 'tot_batch', autonomy_level


# =============================================================================
# TOT PROMPT RECONSTRUCTION
# =============================================================================

def reconstruct_tot_batch_prompt(problem: str, chain: List[str], autonomy_level: int,
                                   ground_truth: Optional[str] = None) -> str:
    """
    Reconstruct the batch mode error localization prompt for ToT.

    Args:
        problem: The problem text
        chain: List of reasoning steps
        autonomy_level: 1, 2, or 3
        ground_truth: Required for L1, optional otherwise
    """
    # Build chain text
    chain_text = ""
    for i, step in enumerate(chain, 1):
        chain_text += f"\nStep {i}: {step}"

    if autonomy_level == 1:
        # L1: Oracle access - model sees correct answer
        prompt = f"""Problem: {problem}

Current reasoning chain (WRONG - got incorrect answer):
{chain_text}

The correct answer should be {ground_truth}.

Analyze the reasoning chain step by step to identify where the error occurred. Which step number (1 to {len(chain)}) contains the first critical error that led to the wrong answer?

Provide your reasoning, then conclude with the step number in the format: \\boxed{{step_number}}
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows it's wrong but not the answer
        prompt = f"""Problem: {problem}

Current reasoning chain (WRONG - got incorrect answer):
{chain_text}

Your answer is incorrect. Analyze the reasoning chain step by step to identify where the error occurred. Which step number (1 to {len(chain)}) contains the first critical error (logical flaw, arithmetic error, or incorrect assumption)?

Provide your reasoning, then conclude with the step number in the format: \\boxed{{step_number}}
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must verify and identify errors
        prompt = f"""Problem: {problem}

Current reasoning chain:
{chain_text}

Carefully verify your reasoning chain step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), determine which step number (1 to {len(chain)}) contains the first critical error.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{step_number}} if you found an error
- \\boxed{{0}} if the reasoning is correct
"""
    return prompt


def reconstruct_tot_incremental_prompt(problem: str, chain: List[str], step_idx: int,
                                         autonomy_level: int, ground_truth: Optional[str] = None) -> str:
    """
    Reconstruct the incremental mode error localization prompt for a specific step.

    NOTE: The L1 and L2 variants are extrapolated from the batch mode patterns since
    the original incremental mode implementation only supported L3. These variants
    follow the same pattern as batch mode (L1 includes ground truth, L2 mentions
    answer is wrong) but have not been verified against an original implementation.

    Args:
        problem: The problem text
        chain: List of reasoning steps
        step_idx: 1-indexed step number to verify
        autonomy_level: 1, 2, or 3
        ground_truth: Required for L1, optional otherwise
    """
    # Build context (previous steps)
    context_text = ""
    if step_idx > 1:
        context_text = "\n\nPrevious steps:"
        for i in range(step_idx - 1):
            context_text += f"\nStep {i + 1}: {chain[i]}"

    current_step_text = chain[step_idx - 1]

    if autonomy_level == 1:
        # L1: Oracle - includes ground truth
        prompt = f"""Problem: {problem}

The correct answer should be {ground_truth}.
{context_text}

Current step to verify:
Step {step_idx}: {current_step_text}

Question: Is Step {step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {step_idx} is correct
- \\boxed{{NO}} if Step {step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - knows overall answer is wrong
        prompt = f"""Problem: {problem}

Note: The overall solution led to an incorrect final answer.
{context_text}

Current step to verify:
Step {step_idx}: {current_step_text}

Question: Is Step {step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {step_idx} is correct
- \\boxed{{NO}} if Step {step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy
        prompt = f"""Problem: {problem}
{context_text}

Current step to verify:
Step {step_idx}: {current_step_text}

Question: Is Step {step_idx} logically correct and mathematically accurate given the problem{' and previous steps' if step_idx > 1 else ''}?

Analyze this specific step carefully. Then respond:
- \\boxed{{YES}} if Step {step_idx} is correct
- \\boxed{{NO}} if Step {step_idx} contains an error (logical flaw, arithmetic error, or incorrect assumption)

Provide your reasoning first, then your conclusion.
"""
    return prompt


# =============================================================================
# COT PROMPT RECONSTRUCTION
# =============================================================================

def reconstruct_cot_error_localization_prompt(problem: str, solution: str,
                                               autonomy_level: int,
                                               ground_truth: Optional[str] = None) -> str:
    """
    Reconstruct the error localization prompt for CoT with shared prefix.

    Args:
        problem: The problem text
        solution: The full solution text
        autonomy_level: 1, 2, or 3
        ground_truth: Required for L1, optional otherwise
    """
    if autonomy_level == 1:
        # L1: Oracle - model sees correct answer
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

The correct answer should be {ground_truth}.

Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""
    elif autonomy_level == 2:
        # L2: Binary feedback - model knows it's wrong
        prompt = f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

Your answer is incorrect. Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""
    else:  # autonomy_level == 3
        # L3: Full autonomy - model must self-verify
        prompt = f"""Problem: {problem}

Current solution:
{solution}

Carefully verify your solution step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), quote the EXACT text (word-for-word) where the first critical error occurs. This should be a continuous excerpt from your solution above.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}} if you found an error
- \\boxed{{CORRECT}} if the solution is correct
"""
    return prompt


# =============================================================================
# EXPERIMENT PROCESSING
# =============================================================================

def parse_model_name(experiment_id: str, config: Dict) -> str:
    """Extract model name from experiment ID or config."""
    model_name = config.get('model_name', '')
    if model_name:
        return model_name

    # Try to extract from experiment ID
    model_patterns = [
        'llama3b', 'llama8b', 'llama70b',
        'qwen7b', 'qwen14b', 'qwen32b',
        'gptoss20b', 'gptoss120b'
    ]
    experiment_lower = experiment_id.lower()
    for pattern in model_patterns:
        if pattern in experiment_lower:
            return pattern

    return 'unknown'


def parse_dataset_name(experiment_id: str, config: Dict) -> str:
    """Extract dataset name from experiment ID or config."""
    dataset = config.get('dataset', '')
    if dataset:
        return dataset

    # Try to extract from experiment ID
    dataset_patterns = ['aime', 'amc23', 'math500', 'csqa', 'gpqa', 'mathqa']
    experiment_lower = experiment_id.lower()
    for pattern in dataset_patterns:
        if pattern in experiment_lower:
            return pattern

    return 'unknown'


def process_tot_experiment(experiment_path: Path, results: Dict, config: Dict,
                           experiment_type: str, autonomy_level: int) -> List[Dict]:
    """
    Process a ToT experiment and extract error localization prompts.

    Returns list of prompt entries.
    """
    entries = []
    experiment_id = experiment_path.name
    model = parse_model_name(experiment_id, config)
    dataset = parse_dataset_name(experiment_id, config)

    results_list = results.get('results', [])

    for problem_idx, result in enumerate(results_list):
        problem = result.get('problem', '')
        ground_truth = result.get('ground_truth', '')
        problem_id = result.get('problem_id', f'prob_{problem_idx}')
        iterations = result.get('iterations', [])

        # Skip iteration 0 (initial generation, no error localization)
        # For iteration N, the error_step refers to where error was found in iteration N-1's chain
        for i, iteration_data in enumerate(iterations[1:], start=1):
            iteration_num = iteration_data.get('iteration', 0)
            # Use PREVIOUS iteration's chain - that's what the localizer analyzed
            prev_iteration = iterations[i - 1]
            chain = prev_iteration.get('chain', [])
            error_step = iteration_data.get('error_step')
            error_reasoning = iteration_data.get('error_reasoning', '')

            # Skip if no error localization occurred
            if error_step is None:
                continue

            # Skip if chain is empty (edge case - can't reconstruct prompt)
            if not chain:
                continue

            # Create unique row ID
            row_id = f"{experiment_id}_prob{problem_idx}_iter{iteration_num}"

            if experiment_type == 'tot_batch':
                # Batch mode: reconstruct full chain verification prompt
                prompt = reconstruct_tot_batch_prompt(
                    problem, chain, autonomy_level, ground_truth
                )
                expected_format = 'step_number'
                original_decision = error_step
            else:
                # Incremental mode: reconstruct step-specific prompt
                step_idx = error_step if error_step > 0 else len(chain)
                prompt = reconstruct_tot_incremental_prompt(
                    problem, chain, step_idx, autonomy_level, ground_truth
                )
                expected_format = 'yes_no'
                original_decision = 'NO' if error_step > 0 else 'YES'

            entry = {
                'row_id': row_id,
                'experiment_id': experiment_id,
                'experiment_type': experiment_type,
                'model': model,
                'dataset': dataset,
                'autonomy_level': f'L{autonomy_level}',
                'problem_id': problem_id,
                'iteration': iteration_num,
                'prompt': prompt,
                'expected_format': expected_format,
                'original_decision': original_decision,
                'original_reasoning': error_reasoning,
                'ground_truth': ground_truth,
                'chain_length': len(chain)
            }
            entries.append(entry)

    return entries


def process_cot_experiment(experiment_path: Path, results: Dict, config: Dict,
                           autonomy_level: int) -> List[Dict]:
    """
    Process a CoT shared prefix experiment and extract error localization prompts.

    Returns list of prompt entries.
    """
    entries = []
    experiment_id = experiment_path.name
    model = parse_model_name(experiment_id, config)
    dataset = parse_dataset_name(experiment_id, config)

    results_list = results.get('results', [])

    for problem_idx, result in enumerate(results_list):
        problem = result.get('problem', '')
        ground_truth = result.get('ground_truth', '')
        problem_id = result.get('problem_id', f'prob_{problem_idx}')
        iterations_data = result.get('iterations_data', [])

        # Skip iteration 0 (initial generation, no error localization)
        # For iteration N, the error_quote refers to where error was found in iteration N-1's solution
        for i, iter_data in enumerate(iterations_data[1:], start=1):
            iteration_num = iter_data.get('iteration', 0)
            # Use PREVIOUS iteration's solution - that's what the localizer analyzed
            prev_iter_data = iterations_data[i - 1]
            solution = prev_iter_data.get('solution', '')
            error_quote = iter_data.get('error_quote')
            error_reasoning = iter_data.get('error_reasoning', '')
            truncation_idx = iter_data.get('truncation_idx')

            # Skip if no error localization occurred
            if error_quote is None and error_reasoning == '':
                continue

            # Create unique row ID
            row_id = f"{experiment_id}_prob{problem_idx}_iter{iteration_num}"

            prompt = reconstruct_cot_error_localization_prompt(
                problem, solution, autonomy_level, ground_truth
            )

            entry = {
                'row_id': row_id,
                'experiment_id': experiment_id,
                'experiment_type': 'cot_shared_prefix',
                'model': model,
                'dataset': dataset,
                'autonomy_level': f'L{autonomy_level}',
                'problem_id': problem_id,
                'iteration': iteration_num,
                'prompt': prompt,
                'expected_format': 'error_quote',
                'original_decision': error_quote,
                'original_reasoning': error_reasoning,
                'ground_truth': ground_truth,
                'truncation_idx': truncation_idx,
                'solution_length': len(solution)
            }
            entries.append(entry)

    return entries


def process_experiment(experiment_path: Path) -> List[Dict]:
    """
    Process a single experiment and extract all error localization prompts.

    Returns list of prompt entries.
    """
    results_file = experiment_path / 'results.json'
    if not results_file.exists():
        return []

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not read {results_file}: {e}")
        return []

    # Handle different JSON structures
    if isinstance(data, list):
        # Some experiments store results as a list directly
        return []

    if not isinstance(data, dict):
        return []

    stats = data.get('stats', {})
    config = stats.get('config', {})

    # Also try to load config.json (some experiments store config separately)
    config_file = experiment_path / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                # Merge with existing config (file config takes precedence)
                for key, value in file_config.items():
                    if key not in config or not config[key]:
                        config[key] = value
        except (json.JSONDecodeError, IOError):
            pass

    # Detect experiment type
    experiment_type, autonomy_level = detect_experiment_type(config, stats)

    if experiment_type == 'cot_shared_prefix':
        return process_cot_experiment(experiment_path, data, config, autonomy_level)
    else:
        return process_tot_experiment(experiment_path, data, config,
                                       experiment_type, autonomy_level)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Export error localization prompts for offline processing'
    )
    parser.add_argument(
        '--experiments',
        required=True,
        help='Glob pattern for experiment directories (e.g., "experiments_main/experiments/eval_*")'
    )
    parser.add_argument(
        '--output',
        default='prompts.jsonl',
        help='Output JSONL file path (default: prompts.jsonl)'
    )
    parser.add_argument(
        '--filter-model',
        help='Only include experiments for this model (e.g., llama8b)'
    )
    parser.add_argument(
        '--filter-dataset',
        help='Only include experiments for this dataset (e.g., aime)'
    )
    parser.add_argument(
        '--filter-autonomy',
        help='Only include experiments with this autonomy level (e.g., L3)'
    )
    parser.add_argument(
        '--filter-type',
        choices=['tot_batch', 'tot_incremental', 'cot_shared_prefix'],
        help='Only include experiments of this type'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit total number of prompts exported (for testing)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose output'
    )

    args = parser.parse_args()

    # Find all matching experiment directories
    experiment_paths = sorted(glob.glob(args.experiments))

    if not experiment_paths:
        print(f"No experiments found matching: {args.experiments}")
        return 1

    print(f"Found {len(experiment_paths)} experiment directories")

    # Process all experiments
    all_entries = []
    experiments_processed = 0
    experiments_skipped = 0

    for exp_path in experiment_paths:
        exp_path = Path(exp_path)
        if not exp_path.is_dir():
            continue

        if args.verbose:
            print(f"Processing: {exp_path.name}")

        entries = process_experiment(exp_path)

        # Apply filters
        filtered_entries = []
        for entry in entries:
            if args.filter_model and entry['model'] != args.filter_model:
                continue
            if args.filter_dataset and entry['dataset'] != args.filter_dataset:
                continue
            if args.filter_autonomy and entry['autonomy_level'] != args.filter_autonomy:
                continue
            if args.filter_type and entry['experiment_type'] != args.filter_type:
                continue
            filtered_entries.append(entry)

        if filtered_entries:
            all_entries.extend(filtered_entries)
            experiments_processed += 1
            if args.verbose:
                print(f"  -> {len(filtered_entries)} prompts extracted")
        else:
            experiments_skipped += 1

        # Check limit
        if args.limit and len(all_entries) >= args.limit:
            all_entries = all_entries[:args.limit]
            print(f"Reached limit of {args.limit} prompts")
            break

    print(f"\nProcessed {experiments_processed} experiments")
    print(f"Skipped {experiments_skipped} experiments (no matching prompts)")
    print(f"Total prompts extracted: {len(all_entries)}")

    if not all_entries:
        print("No prompts to export!")
        return 1

    # Write output
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nExported to: {output_path}")

    # Also write metadata
    metadata = {
        'export_timestamp': datetime.now().isoformat(),
        'total_prompts': len(all_entries),
        'experiments_processed': experiments_processed,
        'experiment_pattern': args.experiments,
        'filters': {
            'model': args.filter_model,
            'dataset': args.filter_dataset,
            'autonomy': args.filter_autonomy,
            'type': args.filter_type
        },
        'by_experiment_type': {},
        'by_model': {},
        'by_dataset': {},
        'by_autonomy_level': {}
    }

    # Compute statistics
    for entry in all_entries:
        exp_type = entry['experiment_type']
        model = entry['model']
        dataset = entry['dataset']
        autonomy = entry['autonomy_level']

        metadata['by_experiment_type'][exp_type] = metadata['by_experiment_type'].get(exp_type, 0) + 1
        metadata['by_model'][model] = metadata['by_model'].get(model, 0) + 1
        metadata['by_dataset'][dataset] = metadata['by_dataset'].get(dataset, 0) + 1
        metadata['by_autonomy_level'][autonomy] = metadata['by_autonomy_level'].get(autonomy, 0) + 1

    metadata_path = output_path.with_suffix('.metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")

    # Print summary
    print("\nSummary by experiment type:")
    for exp_type, count in sorted(metadata['by_experiment_type'].items()):
        print(f"  {exp_type}: {count}")

    print("\nSummary by model:")
    for model, count in sorted(metadata['by_model'].items()):
        print(f"  {model}: {count}")

    print("\nSummary by dataset:")
    for dataset, count in sorted(metadata['by_dataset'].items()):
        print(f"  {dataset}: {count}")

    print("\nSummary by autonomy level:")
    for autonomy, count in sorted(metadata['by_autonomy_level'].items()):
        print(f"  {autonomy}: {count}")

    return 0


if __name__ == '__main__':
    exit(main())
