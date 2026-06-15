#!/usr/bin/env python3
"""
Majority Vote Localization Evaluation

Generates 10 localization decisions per prompt using majority voting:
- Rollout #1: Use existing original_decision from JSONL
- Rollouts #2-10: Generate 9 new via manager.generate(n=9, temperature=0.5)

Computes:
- MV@10: Majority vote among all 10 decisions
- Meta-localizer: Second-stage reasoning to pick best decision
- Random baseline: Random step/position for comparison

Usage:
    python evaluate_mv_localization.py --model llama3b --input-file l2_oracle_localization_prompts.jsonl
    python evaluate_mv_localization.py --model qwen7b --gpus 0,1 --tensor-parallel-size 2
"""

import os
os.environ['VLLM_USE_V1'] = '1'

import json
import re
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from tqdm import tqdm

import sys
sys.path.insert(0, str(next(_p for _p in Path(__file__).resolve().parents if (_p / 'thought_ics').is_dir())))

from thought_ics.thought_mdp import initialize_model
from thought_ics.baselines.cot_eval import extract_boxed_answer

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_GPU_CONFIG = {
    'llama3b': (1, 1, "0"),
    'qwen7b': (2, 2, "0,1"),
    'llama8b': (2, 2, "0,1"),
    'gptoss20b': (2, 2, "0,1"),
    'qwen32b': (4, 4, "0,1,2,3"),
    'gptoss120b': (4, 4, "0,1,2,3"),
    'llama70b': (4, 4, "0,1,2,3"),
}

TEMPERATURE = 0.5
MAX_TOKENS = 1024
N_ROLLOUTS = 9  # Generate 9 new, total 10 with original

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PARSING FUNCTIONS
# ============================================================================

def parse_step_number(text: str) -> Optional[int]:
    """Parse step number from \\boxed{N} format for tot_batch."""
    if not text:
        return None

    boxed = extract_boxed_answer(str(text))
    if boxed == "NO ANSWER":
        # Fallback: try to find any number in the text
        numbers = re.findall(r'\b(\d+)\b', str(text))
        if numbers:
            return int(numbers[-1])  # Take last number
        return None

    # Try to extract integer from boxed content
    try:
        # Handle various formats: "10", "Step 10", "step_number: 10", etc.
        match = re.search(r'(\d+)', boxed)
        if match:
            return int(match.group(1))
    except (ValueError, AttributeError):
        pass

    return None


def parse_error_quote(text: str) -> Optional[str]:
    """Parse error quote from \\boxed{ERROR_QUOTE: "..."} format for cot_shared_prefix."""
    if not text:
        return None

    boxed = extract_boxed_answer(str(text))
    if boxed == "NO ANSWER":
        return None

    # Normalize LaTeX escapes (e.g., NO\_ERROR -> NO_ERROR)
    boxed_normalized = boxed.replace('\\_', '_').replace('\\-', '-')

    # Check for special cases
    if boxed_normalized.upper() in ["CORRECT", "NO_ERROR", "NO ERROR"]:
        return "NO_ERROR"

    # Try to extract quoted text from ERROR_QUOTE format
    if "ERROR_QUOTE:" in boxed_normalized or "ERROR_QUOTE" in boxed_normalized:
        match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed_normalized)
        if match:
            return match.group(1)
        # Try without quotes
        match = re.search(r'ERROR_QUOTE:\s*(.+)', boxed_normalized)
        if match:
            return match.group(1).strip().strip('"')

    # Fallback: if boxed contains quoted text directly
    match = re.search(r'"([^"]{10,})"', boxed_normalized)
    if match:
        return match.group(1)

    # Return the boxed content as-is if it looks like a quote
    if len(boxed_normalized) > 10:
        return boxed_normalized

    return None


def parse_decision(text: str, expected_format: str) -> Any:
    """Parse decision based on expected format."""
    if expected_format == "step_number":
        return parse_step_number(text)
    elif expected_format == "error_quote":
        return parse_error_quote(text)
    else:
        raise ValueError(f"Unknown expected_format: {expected_format}")


def parse_original_decision(entry: Dict, expected_format: str) -> Any:
    """Parse the original decision from the JSONL entry.

    For tot_batch (step_number): original_decision contains the step number directly
    For cot_shared_prefix (error_quote): original_decision is often a placeholder,
        so we need to parse the actual error quote from original_reasoning
    """
    original_decision = entry.get('original_decision')
    original_reasoning = entry.get('original_reasoning', '')

    if expected_format == "step_number":
        # For tot_batch, original_decision IS the step number
        if original_decision is None:
            return None
        if isinstance(original_decision, int):
            return original_decision
        # Try to parse as int
        try:
            return int(original_decision)
        except (ValueError, TypeError):
            return parse_step_number(str(original_decision))
    else:
        # For cot_shared_prefix (error_quote):
        # original_decision often contains placeholder text like "Current solution (WRONG...)"
        # The actual error quote is in the original_reasoning field

        # First try to extract from original_reasoning (the model's full response)
        if original_reasoning:
            parsed = parse_error_quote(original_reasoning)
            if parsed and parsed != "NO_ERROR":
                return parsed

        # Fallback: try original_decision if it looks like a valid quote
        if original_decision and isinstance(original_decision, str):
            # Skip known placeholder patterns
            if "Current solution" in original_decision or "WRONG" in original_decision:
                return None
            if "\\boxed" in original_decision:
                return parse_error_quote(original_decision)
            # Return if it's long enough to be a real quote
            if len(original_decision) > 20:
                return original_decision

        return None


# ============================================================================
# ROLLOUT GENERATION
# ============================================================================

def generate_rollouts(
    manager,
    prompts: List[str],
    n: int = 9,
    temperature: float = 0.5,
    max_tokens: int = 1024
) -> List[List[str]]:
    """Generate n rollouts for each prompt using batched generation.

    Uses vLLM's native n parameter for efficiency (shared KV cache across completions).

    Args:
        manager: Model manager
        prompts: List of prompts
        n: Number of rollouts per prompt
        temperature: Sampling temperature
        max_tokens: Max tokens per generation

    Returns:
        List of lists, where each inner list contains n outputs for that prompt
    """
    if not prompts:
        return []

    # Use vLLM's native n parameter for efficient multi-sample generation
    # This generates n completions per prompt with shared KV cache
    outputs = manager.generate(
        prompts=prompts,
        n=n,  # Generate n completions per prompt
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        top_k=50,
    )

    # vLLM returns flattened list: [p1_c1, p1_c2, ..., p1_cn, p2_c1, ...]
    # Group by original prompt
    results = []
    for i in range(len(prompts)):
        start_idx = i * n
        end_idx = start_idx + n
        results.append(outputs[start_idx:end_idx])

    return results


# ============================================================================
# MAJORITY VOTE COMPUTATION
# ============================================================================

def compute_majority_vote(decisions: List[Any]) -> Any:
    """Compute majority vote from list of decisions.

    Returns the most common decision. Ties are broken by first occurrence.
    None values are filtered out.
    """
    valid_decisions = [d for d in decisions if d is not None]
    if not valid_decisions:
        return None

    counter = Counter(valid_decisions)
    return counter.most_common(1)[0][0]


# ============================================================================
# META-LOCALIZER
# ============================================================================

META_LOCALIZER_PROMPT_STEP = """You are analyzing error localization decisions for a reasoning problem.

Problem and Solution:
{prompt_content}

The following {n} localization attempts identified these error positions (step numbers):
{decisions_list}

Review the solution and the proposed error locations. Determine which error location is most accurate.

Think step by step about which step truly contains the first critical error. Then provide your final decision.

Conclude with your answer in the format: \\boxed{{N}} where N is the step number."""


META_LOCALIZER_PROMPT_QUOTE = """You are analyzing error localization decisions for a reasoning problem.

Problem and Solution:
{prompt_content}

The following {n} localization attempts identified these error locations (quoted text):
{decisions_list}

Review the solution and the proposed error locations. Determine which error quote most accurately identifies the first critical error.

Think step by step about which quote truly captures the first critical error. Then provide your final decision.

Conclude with your answer in the format: \\boxed{{ERROR_QUOTE: "exact quote"}}"""


def build_meta_localizer_prompt(
    original_prompt: str,
    decisions: List[Any],
    expected_format: str
) -> str:
    """Build meta-localizer prompt showing all decisions.

    Args:
        original_prompt: The original localization prompt (contains problem and solution)
        decisions: List of parsed decisions from all rollouts
        expected_format: "step_number" or "error_quote"

    Returns:
        Meta-localizer prompt string
    """
    # Format decisions list
    valid_decisions = [(i+1, d) for i, d in enumerate(decisions) if d is not None]

    if not valid_decisions:
        return None

    # Truncate original prompt if too long (keep first 6000 chars)
    prompt_content = original_prompt[:6000]
    if len(original_prompt) > 6000:
        prompt_content += "\n... [truncated]"

    if expected_format == "step_number":
        decisions_list = "\n".join([f"  Attempt {i}: Step {d}" for i, d in valid_decisions])
        template = META_LOCALIZER_PROMPT_STEP
    else:
        decisions_list = "\n".join([f'  Attempt {i}: "{d}"' for i, d in valid_decisions])
        template = META_LOCALIZER_PROMPT_QUOTE

    return template.format(
        prompt_content=prompt_content,
        n=len(valid_decisions),
        decisions_list=decisions_list
    )


def run_meta_localizer(
    manager,
    prompts: List[str],
    expected_formats: List[str],
    temperature: float = 0.5,
    max_tokens: int = 1024
) -> List[Tuple[Any, str]]:
    """Run meta-localizer for each prompt.

    Returns list of (decision, full_reasoning) tuples.
    """
    # Filter out None prompts
    valid_indices = [i for i, p in enumerate(prompts) if p is not None]
    valid_prompts = [prompts[i] for i in valid_indices]
    valid_formats = [expected_formats[i] for i in valid_indices]

    if not valid_prompts:
        return [(None, "") for _ in prompts]

    outputs = manager.generate(
        prompts=valid_prompts,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.9,
        top_k=50,
    )

    # Build full results list
    results = [(None, "") for _ in prompts]
    for idx, (output, expected_format) in enumerate(zip(outputs, valid_formats)):
        original_idx = valid_indices[idx]
        decision = parse_decision(output, expected_format)
        results[original_idx] = (decision, output.strip())

    return results


# ============================================================================
# RANDOM BASELINE
# ============================================================================

def generate_random_baseline(
    expected_format: str,
    solution_length: int = None,
    chain_length: int = None,
) -> Any:
    """Generate random baseline decision.

    For tot_batch (step_number): Random integer from 1 to chain_length
    For cot_shared_prefix (error_quote): Random position in solution (as integer index)
    """
    if expected_format == "step_number":
        # Random step number
        max_step = chain_length if chain_length and chain_length > 0 else 20
        return random.randint(1, max_step)
    else:
        # For error_quote, we return a random truncation index
        max_pos = solution_length if solution_length and solution_length > 0 else 1000
        return random.randint(0, max_pos)


# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def evaluate_mv_localization(
    model_name: str,
    input_file: str,
    output_dir: str,
    batch_size: int = 32,
    gpu_ids: str = None,
    tensor_parallel_size: int = None,
    start_idx: int = None,
    end_idx: int = None,
    skip_meta: bool = False,
):
    """Main evaluation function.

    Args:
        model_name: Model to use (llama3b, qwen7b, etc.)
        input_file: Path to input JSONL file
        output_dir: Directory for output files
        batch_size: Batch size for processing
        gpu_ids: GPU IDs to use (auto-detected if None)
        tensor_parallel_size: Tensor parallel size (auto-detected if None)
        start_idx: Start index for dataset slice
        end_idx: End index for dataset slice
        skip_meta: Skip meta-localizer step
    """
    # Auto-detect GPU config
    if model_name in MODEL_GPU_CONFIG:
        _, tp, default_gpus = MODEL_GPU_CONFIG[model_name]
        gpu_ids = gpu_ids or default_gpus
        tensor_parallel_size = tensor_parallel_size or tp
    else:
        gpu_ids = gpu_ids or "0"
        tensor_parallel_size = tensor_parallel_size or 1

    # Load input data
    logger.info(f"Loading data from {input_file}...")
    all_data = []
    with open(input_file) as f:
        for line in f:
            all_data.append(json.loads(line))

    logger.info(f"Loaded {len(all_data)} total entries")

    # Filter for this model
    model_data = [d for d in all_data if d['model'] == model_name]
    logger.info(f"Found {len(model_data)} entries for model {model_name}")

    if not model_data:
        logger.warning(f"No data found for model {model_name}")
        return []

    # Apply slicing if specified
    if start_idx is not None or end_idx is not None:
        start_idx = start_idx or 0
        end_idx = end_idx or len(model_data)
        model_data = model_data[start_idx:end_idx]
        logger.info(f"Processing slice [{start_idx}:{end_idx}] = {len(model_data)} entries")

    # Setup output
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slice_suffix = f"_part{start_idx}-{end_idx}" if start_idx is not None else ""
    output_file = output_dir / f"mv_localization_{model_name}{slice_suffix}_{timestamp}.jsonl"
    checkpoint_file = output_dir / f"checkpoint_mv_{model_name}{slice_suffix}.json"

    # Setup file logging
    log_file = output_dir / f"mv_localization_{model_name}{slice_suffix}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MAJORITY VOTE LOCALIZATION EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Model: {model_name}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Entries to process: {len(model_data)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Tensor parallel size: {tensor_parallel_size}")
    logger.info(f"Skip meta-localizer: {skip_meta}")
    logger.info("=" * 80)

    # Initialize model
    logger.info(f"Initializing model {model_name} on GPUs {gpu_ids}...")
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    manager = initialize_model(
        gpu_ids=gpu_ids,
        tensor_parallel_size=tensor_parallel_size,
        model_name=model_name
    )

    # Resume from checkpoint if exists
    completed_ids = set()
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
            completed_ids = set(checkpoint.get('completed_ids', []))
        logger.info(f"Resuming from checkpoint: {len(completed_ids)} already completed")

    # Process in batches
    all_results = []
    pending_data = [d for d in model_data if d['row_id'] not in completed_ids]
    logger.info(f"Pending entries: {len(pending_data)}")

    for batch_start in tqdm(range(0, len(pending_data), batch_size), desc=f"Processing {model_name}"):
        batch = pending_data[batch_start:batch_start + batch_size]

        # Extract prompts and metadata
        prompts = [d['prompt'] for d in batch]
        expected_formats = [d['expected_format'] for d in batch]

        # Generate 9 new rollouts per prompt
        logger.info(f"Generating {N_ROLLOUTS} rollouts for {len(batch)} prompts...")
        rollout_outputs = generate_rollouts(
            manager, prompts, n=N_ROLLOUTS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS
        )

        # Parse all decisions (original + 9 new = 10 total)
        batch_results = []
        for i, entry in enumerate(batch):
            # Rollout 1: original decision (parsed from entry)
            parsed_original = parse_original_decision(
                entry,
                entry['expected_format']
            )

            # Parse 9 new rollouts
            new_decisions = []
            new_responses = rollout_outputs[i] if i < len(rollout_outputs) else []
            for output in new_responses:
                parsed = parse_decision(output, entry['expected_format'])
                new_decisions.append(parsed)

            # All 10 decisions
            all_decisions = [parsed_original] + new_decisions

            # Compute MV@10
            mv_decision = compute_majority_vote(all_decisions)

            # Get chain_length for random baseline (for tot_batch)
            chain_length = entry.get('chain_length', 20)
            solution_length = entry.get('solution_length', 1000)

            # Random baseline
            random_decision = generate_random_baseline(
                entry['expected_format'],
                solution_length=solution_length,
                chain_length=chain_length,
            )

            batch_results.append({
                'entry': entry,
                'all_decisions': all_decisions,
                'mv_decision': mv_decision,
                'random_decision': random_decision,
                'rollout_outputs': new_responses,
            })

        # Run meta-localizer for batch (unless skipped)
        if not skip_meta:
            meta_prompts = []
            for br in batch_results:
                entry = br['entry']
                meta_prompt = build_meta_localizer_prompt(
                    original_prompt=entry['prompt'],
                    decisions=br['all_decisions'],
                    expected_format=entry['expected_format']
                )
                meta_prompts.append(meta_prompt)

            logger.info(f"Running meta-localizer for {len(batch)} prompts...")
            meta_results = run_meta_localizer(
                manager,
                meta_prompts,
                expected_formats,
                temperature=0.5,
                max_tokens=MAX_TOKENS
            )
        else:
            meta_results = [(None, "") for _ in batch_results]

        # Combine results
        for i, br in enumerate(batch_results):
            entry = br['entry']
            meta_decision, meta_reasoning = meta_results[i]

            # Convert decisions to strings for JSON serialization
            def to_str(d):
                if d is None:
                    return None
                return str(d)

            result = {
                'row_id': entry['row_id'],
                'model': entry['model'],
                'dataset': entry['dataset'],
                'experiment_type': entry['experiment_type'],
                'expected_format': entry['expected_format'],
                'problem_id': entry.get('problem_id'),
                'iteration': entry.get('iteration'),
                'ground_truth': entry.get('ground_truth'),
                'original_decision': to_str(entry.get('original_decision')),
                'rollout_decisions': [to_str(d) for d in br['all_decisions']],
                'mv10_decision': to_str(br['mv_decision']),
                'meta_localizer_decision': to_str(meta_decision),
                'meta_localizer_reasoning': meta_reasoning,
                'random_decision': to_str(br['random_decision']),
            }

            all_results.append(result)
            completed_ids.add(entry['row_id'])

        # Save checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'completed_ids': list(completed_ids)}, f)

        # Append to output file
        with open(output_file, 'a') as f:
            for result in all_results[-len(batch):]:
                f.write(json.dumps(result) + '\n')

        logger.info(f"Batch complete. Total processed: {len(all_results)}/{len(pending_data)}")

    # Cleanup
    logger.info("Unloading model...")
    manager.unload_base_model()

    logger.info(f"Completed! Results saved to {output_file}")
    logger.info(f"Total processed: {len(all_results)}")

    # Summary statistics
    if all_results:
        mv_not_none = sum(1 for r in all_results if r['mv10_decision'] is not None)
        meta_not_none = sum(1 for r in all_results if r['meta_localizer_decision'] is not None)
        logger.info(f"MV@10 decisions: {mv_not_none}/{len(all_results)}")
        logger.info(f"Meta-localizer decisions: {meta_not_none}/{len(all_results)}")

    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Majority Vote Localization Evaluation')

    parser.add_argument('--model', type=str, required=True,
                        help='Model to evaluate (llama3b, qwen7b, gptoss20b, gptoss120b, etc.)')
    parser.add_argument('--input-file', type=str,
                        default='l2_oracle_localization_prompts.jsonl',
                        help='Input JSONL file with localization prompts')
    parser.add_argument('--output-dir', type=str, default='experiments_mv_localization',
                        help='Output directory for results')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--gpus', type=str, default=None,
                        help='GPU IDs (auto-detected if not specified)')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                        help='Tensor parallel size (auto-detected if not specified)')
    parser.add_argument('--start-idx', type=int, default=None,
                        help='Start index for dataset slice')
    parser.add_argument('--end-idx', type=int, default=None,
                        help='End index for dataset slice')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip-meta', action='store_true',
                        help='Skip meta-localizer step')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    evaluate_mv_localization(
        model_name=args.model,
        input_file=args.input_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu_ids=args.gpus,
        tensor_parallel_size=args.tensor_parallel_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        skip_meta=args.skip_meta,
    )


if __name__ == '__main__':
    main()
