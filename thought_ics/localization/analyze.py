#!/usr/bin/env python3
"""
Error Localization Analysis Script

Analyzes error localization quality by:
1. Extracting error localization instances from experiment results
2. Reconstructing the exact prompts sent to the model
3. Re-evaluating with OpenAI's o1-mini model
4. Comparing decisions and reasoning

Supports both ToT (Tree of Thought) and CoT (Chain of Thought) with shared prefix.
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
from difflib import SequenceMatcher

try:
    from openai import OpenAI
except ImportError:
    print("Error: OpenAI library not installed. Install with: pip install openai")
    exit(1)


# =============================================================================
# MODE DETECTION
# =============================================================================

def detect_experiment_type(config: Dict) -> str:
    """
    Detect if experiment is ToT or CoT based on config.

    Returns:
        'tot': Tree of Thought with discrete steps
        'cot_shared_prefix': Chain of Thought with shared prefix continuation
        'cot_standard': Chain of Thought with full regeneration
    """
    if 'baseline_type' in config:
        # This is a baseline CoT experiment
        if config.get('shared_prefix', False):
            return 'cot_shared_prefix'
        else:
            return 'cot_standard'
    else:
        # This is a ToT experiment
        return 'tot'


def reconstruct_batch_prompt(problem: str, chain: List[str]) -> str:
    """
    Reconstruct the batch mode error localization prompt for L3.

    Based on iterative_self_correction.py lines 280-305
    """
    # Build chain text
    chain_text = ""
    for i, step in enumerate(chain, 1):
        chain_text += f"\nStep {i}: {step}"

    prompt = f"""Problem: {problem}

Current reasoning chain:
{chain_text}

Carefully verify your reasoning chain step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), determine which step number (1 to {len(chain)}) contains the first critical error.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{step_number}} if you found an error
- \\boxed{{0}} if the reasoning is correct
"""
    return prompt


def reconstruct_incremental_prompt(problem: str, chain: List[str], step_idx: int) -> str:
    """
    Reconstruct the incremental mode error localization prompt for a specific step.

    Based on iterative_self_correction.py lines 141-156
    Note: Incremental mode checks each step sequentially, so we need the step index.
    """
    # Build context (previous steps)
    context_text = ""
    if step_idx > 1:
        context_text = "\n\nPrevious steps:"
        for i in range(step_idx - 1):
            context_text += f"\nStep {i + 1}: {chain[i]}"

    current_step_text = chain[step_idx - 1]

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


def extract_step_number(text: str) -> Optional[int]:
    """
    Extract step number from model response.

    Looks for patterns like:
    - \\boxed{5}
    - \\boxed{0}
    - \boxed{5}

    Returns None if not found.
    """
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
    # Look for \boxed{YES} or \boxed{NO}
    pattern = r'\\*boxed\{(YES|NO)\}'
    matches = re.findall(pattern, text, re.IGNORECASE)

    if matches:
        return matches[-1].upper()

    return None


# =============================================================================
# COT (CHAIN OF THOUGHT) HELPERS
# =============================================================================

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


def extract_error_quote_from_response(response: str) -> Optional[str]:
    """Extract quoted error text from judge response."""
    boxed = extract_boxed_answer(response)

    if boxed in ["CORRECT", "NO_ERROR"]:
        return None

    if "ERROR_QUOTE:" in boxed:
        quote_match = re.search(r'ERROR_QUOTE:\s*"([^"]+)"', boxed)
        if quote_match:
            return quote_match.group(1)

    # Fallback: find any quoted text (20+ chars)
    quote_matches = re.findall(r'"([^"]{20,})"', response)
    if quote_matches:
        return quote_matches[0]

    return None


def find_truncation_point(solution: str, error_quote: Optional[str]) -> Optional[int]:
    """Find character position where error quote begins in solution."""
    if error_quote is None:
        return None

    idx = solution.find(error_quote)
    if idx != -1:
        return idx

    # Could add fuzzy matching here for minor variations
    return None


def compute_position_agreement(pos1: Optional[int], pos2: Optional[int], tolerance: int = 50) -> bool:
    """Check if two truncation positions agree within tolerance."""
    if pos1 is None or pos2 is None:
        return False
    return abs(pos1 - pos2) <= tolerance


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


def reconstruct_cot_error_localization_prompt(problem: str, solution: str, autonomy_level: int) -> str:
    """Reconstruct the exact prompt used for error localization in CoT."""
    if autonomy_level == 2:  # L2: Binary feedback
        return f"""Problem: {problem}

Current solution (WRONG - got incorrect answer):
{solution}

Your answer is incorrect. Analyze the solution step by step to identify where the error occurred. Quote the EXACT text (word-for-word) where the first critical error (logical flaw, arithmetic error, or incorrect assumption) begins. This should be a continuous excerpt from your solution above.

Provide your reasoning, then conclude with the exact quote in the format:
\\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}}

If you cannot find the error, respond with: \\boxed{{NO_ERROR}}
"""
    elif autonomy_level == 3:  # L3: Autonomous
        return f"""Problem: {problem}

Current solution:
{solution}

Carefully verify your solution step by step. If you identify any errors (logical flaw, arithmetic error, or incorrect assumption), quote the EXACT text (word-for-word) where the first critical error occurs. This should be a continuous excerpt from your solution above.

Provide your reasoning and analysis. Then conclude with:
- \\boxed{{ERROR_QUOTE: "exact text from solution where error occurs"}} if you found an error
- \\boxed{{CORRECT}} if the solution is correct
"""
    else:
        raise ValueError(f"Unsupported autonomy level for CoT: {autonomy_level}")


def call_openai_o1mini(prompt: str, client: OpenAI, max_retries: int = 3, model: str = "gpt-4o") -> Tuple[str, int]:
    """
    Call OpenAI model with the prompt.

    Returns:
        (response_text, tokens_used)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens

            return response_text, tokens_used

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                raise

    return "", 0


def analyze_experiment(experiment_path: str, openai_api_key: str, limit: Optional[int] = None, model: str = "gpt-4o") -> Dict:
    """
    Analyze error localization for a single experiment.

    Args:
        experiment_path: Path to experiment folder
        openai_api_key: OpenAI API key
        limit: Optional maximum number of instances to analyze (for testing)
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Dictionary with analysis results
    """
    experiment_path = Path(experiment_path)

    # Load results.json
    results_file = experiment_path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found in {experiment_path}")

    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    config = data.get('stats', {}).get('config', {})
    results = data.get('results', [])

    error_detection_method = config.get('error_detection_method', 'batch')
    autonomy_level = config.get('autonomy_level', 3)

    print(f"Experiment config:")
    print(f"  Model: {config.get('model_name', 'unknown')}")
    print(f"  Dataset: {config.get('dataset', 'unknown')}")
    print(f"  Error detection: {error_detection_method}")
    print(f"  Autonomy level: L{autonomy_level}")
    print(f"  Judge temp: {config.get('judge_temp', 'unknown')}")
    print(f"  Total problems: {len(results)}")

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Collect all error localization instances
    instances = []
    total_tokens = 0

    for problem_idx, result in enumerate(results):
        # Check if we've hit the limit
        if limit is not None and len(instances) >= limit:
            print(f"\nReached limit of {limit} instances, stopping...")
            break

        problem = result['problem']
        problem_id = result.get('problem_id', f"problem_{problem_idx}")
        iterations = result.get('iterations', [])

        # Skip iteration 0 (initial generation, no error localization)
        for iteration_data in iterations[1:]:
            # Check limit again (inner loop)
            if limit is not None and len(instances) >= limit:
                break

            iteration_num = iteration_data['iteration']
            chain = iteration_data['chain']
            error_step = iteration_data.get('error_step')
            error_reasoning = iteration_data.get('error_reasoning', '')

            # Skip if no error localization occurred
            if error_step is None:
                continue

            print(f"\nProcessing {problem_id} iteration {iteration_num}...")

            # Reconstruct prompt based on detection method
            if error_detection_method == 'batch':
                prompt = reconstruct_batch_prompt(problem, chain)

                # Get OpenAI model response
                try:
                    o1_response, tokens = call_openai_o1mini(prompt, client, model=model)
                    total_tokens += tokens

                    # Extract step number from response
                    o1_decision = extract_step_number(o1_response)

                    instance = {
                        'problem_id': problem_id,
                        'iteration': iteration_num,
                        'detection_method': 'batch',
                        'prompt': prompt,
                        'chain_length': len(chain),
                        'original_model_decision': error_step,
                        'original_model_reasoning': error_reasoning,
                        'judge_model_decision': o1_decision,
                        'judge_model_reasoning': o1_response,
                        'agreement': (error_step == o1_decision) if o1_decision is not None else None,
                        'tokens_used': tokens
                    }

                    instances.append(instance)
                    print(f"  Original: step {error_step}, Judge ({model}): step {o1_decision}, Agreement: {instance['agreement']}")

                except Exception as e:
                    print(f"  Error processing: {e}")
                    continue

            elif error_detection_method == 'incremental':
                # For incremental mode, we need to know which step was identified
                # The error_step tells us which step failed
                # We reconstruct the prompt for that specific step

                if error_step == 0:
                    # All steps passed, check the last step
                    step_idx = len(chain)
                else:
                    step_idx = error_step

                prompt = reconstruct_incremental_prompt(problem, chain, step_idx)

                try:
                    o1_response, tokens = call_openai_o1mini(prompt, client, model=model)
                    total_tokens += tokens

                    # Extract YES/NO from response
                    o1_yes_no = extract_yes_no(o1_response)

                    # Convert to comparable format
                    # Original: error_step > 0 means NO, error_step == 0 means YES for all
                    original_decision = 'NO' if error_step > 0 else 'YES'

                    instance = {
                        'problem_id': problem_id,
                        'iteration': iteration_num,
                        'detection_method': 'incremental',
                        'step_verified': step_idx,
                        'prompt': prompt,
                        'chain_length': len(chain),
                        'original_model_decision': original_decision,
                        'original_error_step': error_step,
                        'original_model_reasoning': error_reasoning,
                        'judge_model_decision': o1_yes_no,
                        'judge_model_reasoning': o1_response,
                        'agreement': (original_decision == o1_yes_no) if o1_yes_no is not None else None,
                        'tokens_used': tokens
                    }

                    instances.append(instance)
                    print(f"  Step {step_idx}: Original: {original_decision}, Judge ({model}): {o1_yes_no}, Agreement: {instance['agreement']}")

                except Exception as e:
                    print(f"  Error processing: {e}")
                    continue

    # Compute summary statistics
    total_instances = len(instances)
    agreements = sum(1 for inst in instances if inst.get('agreement') == True)
    disagreements = sum(1 for inst in instances if inst.get('agreement') == False)
    unparseable = sum(1 for inst in instances if inst.get('agreement') is None)

    agreement_rate = agreements / total_instances if total_instances > 0 else 0

    # Create analysis result
    analysis = {
        'experiment_path': str(experiment_path),
        'config': config,
        'judge_model': model,
        'analyzed_at': datetime.now().isoformat(),
        'total_instances': total_instances,
        'summary': {
            'agreements': agreements,
            'disagreements': disagreements,
            'unparseable_responses': unparseable,
            'agreement_rate': agreement_rate,
            'total_tokens_used': total_tokens,
            'estimated_cost_usd': total_tokens * 0.000003  # Rough estimate
        },
        'instances': instances
    }

    return analysis


# =============================================================================
# COT (CHAIN OF THOUGHT) ANALYSIS
# =============================================================================

def analyze_cot_error_localization(experiment_path: str, openai_api_key: str, limit: Optional[int] = None, model: str = "gpt-4o") -> Dict:
    """
    Analyze error localization for CoT with shared prefix.

    Args:
        experiment_path: Path to experiment folder
        openai_api_key: OpenAI API key
        limit: Optional maximum number of instances to analyze (for testing)
        model: OpenAI model to use (default: gpt-4o)

    Returns:
        Dictionary with analysis results
    """
    experiment_path = Path(experiment_path)

    # Load results.json
    results_file = experiment_path / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found in {experiment_path}")

    print(f"Loading results from {results_file}")
    with open(results_file, 'r') as f:
        data = json.load(f)

    config = data.get('stats', {})
    results = data.get('results', [])

    autonomy_level = 3  # Default to L3 for CoT
    if 'iterative_l2' in config.get('baseline_type', ''):
        autonomy_level = 2

    print(f"Experiment config:")
    print(f"  Model: {config.get('model_name', 'unknown')}")
    print(f"  Dataset: {config.get('dataset', 'unknown')}")
    print(f"  Autonomy level: L{autonomy_level}")
    print(f"  Shared prefix: {config.get('shared_prefix', False)}")
    print(f"  Total problems: {len(results)}")

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Collect all error localization instances
    instances = []
    total_tokens = 0

    for problem_idx, result in enumerate(results):
        # Check if we've hit the limit
        if limit is not None and len(instances) >= limit:
            print(f"\nReached limit of {limit} instances, stopping...")
            break

        problem = result['problem']
        problem_id = result.get('problem_id', f"problem_{problem_idx}")
        iterations_data = result.get('iterations_data', [])

        # Skip iteration 0 (initial generation, no error localization)
        for iter_data in iterations_data[1:]:
            # Check limit again (inner loop)
            if limit is not None and len(instances) >= limit:
                break

            iteration_num = iter_data['iteration']
            solution = iter_data['solution']

            # Check if error localization data exists
            error_quote = iter_data.get('error_quote')
            truncation_idx = iter_data.get('truncation_idx')
            error_reasoning = iter_data.get('error_reasoning', '')

            # Skip if no error localization occurred
            if error_quote is None:
                continue

            print(f"\nProcessing {problem_id} iteration {iteration_num}...")

            # Reconstruct prompt
            prompt = reconstruct_cot_error_localization_prompt(problem, solution, autonomy_level)

            # Get OpenAI model response
            try:
                judge_response, tokens = call_openai_o1mini(prompt, client, model=model)
                total_tokens += tokens

                # Extract error quote from judge response
                judge_quote = extract_error_quote_from_response(judge_response)
                judge_idx = find_truncation_point(solution, judge_quote) if judge_quote else None

                # Compute agreement metrics
                position_agreement = compute_position_agreement(truncation_idx, judge_idx, tolerance=50)
                quote_overlap = compute_quote_overlap(error_quote, judge_quote)

                # Overall agreement: either position agrees OR high overlap
                overall_agreement = position_agreement or (quote_overlap > 0.5)

                instance = {
                    'problem_id': problem_id,
                    'iteration': iteration_num,
                    'problem_snippet': problem[:100] + "..." if len(problem) > 100 else problem,
                    'solution_length': len(solution),
                    'model_error_quote': error_quote[:100] + "..." if len(error_quote) > 100 else error_quote,
                    'model_truncation_idx': truncation_idx,
                    'model_reasoning_snippet': error_reasoning[:200] + "..." if len(error_reasoning) > 200 else error_reasoning,
                    'judge_error_quote': judge_quote[:100] + "..." if judge_quote and len(judge_quote) > 100 else judge_quote,
                    'judge_truncation_idx': judge_idx,
                    'judge_reasoning_snippet': judge_response[:200] + "..." if len(judge_response) > 200 else judge_response,
                    'position_agreement': position_agreement,
                    'quote_overlap': quote_overlap,
                    'overall_agreement': overall_agreement,
                    'tokens_used': tokens
                }

                instances.append(instance)
                print(f"  Model pos: {truncation_idx}, Judge pos: {judge_idx}")
                print(f"  Position agreement: {position_agreement}, Quote overlap: {quote_overlap:.2f}")
                print(f"  Overall agreement: {overall_agreement}")

            except Exception as e:
                print(f"  Error processing: {e}")
                continue

    # Compute aggregate metrics
    total = len(instances)
    if total == 0:
        print("\nNo error localization instances found!")
        return {
            'experiment_path': str(experiment_path),
            'config': config,
            'total_instances': 0,
            'summary': {
                'position_agreements': 0,
                'high_overlaps': 0,
                'overall_agreements': 0,
                'position_agreement_rate': 0.0,
                'high_overlap_rate': 0.0,
                'overall_agreement_rate': 0.0,
                'avg_quote_overlap': 0.0,
                'total_tokens_used': 0,
                'estimated_cost_usd': 0.0
            },
            'instances': []
        }

    position_agreements = sum(1 for inst in instances if inst['position_agreement'])
    high_overlaps = sum(1 for inst in instances if inst['quote_overlap'] > 0.5)
    overall_agreements = sum(1 for inst in instances if inst['overall_agreement'])
    avg_overlap = sum(inst['quote_overlap'] for inst in instances) / total

    analysis = {
        'experiment_path': str(experiment_path),
        'config': config,
        'total_instances': total,
        'summary': {
            'position_agreements': position_agreements,
            'high_overlaps': high_overlaps,
            'overall_agreements': overall_agreements,
            'position_agreement_rate': position_agreements / total,
            'high_overlap_rate': high_overlaps / total,
            'overall_agreement_rate': overall_agreements / total,
            'avg_quote_overlap': avg_overlap,
            'total_tokens_used': total_tokens,
            'estimated_cost_usd': total_tokens * 0.000003  # Rough estimate
        },
        'instances': instances
    }

    return analysis


def main():
    parser = argparse.ArgumentParser(
        description='Analyze error localization quality by comparing with o1-mini'
    )
    parser.add_argument(
        'experiment_path',
        help='Path to experiment folder containing results.json'
    )
    parser.add_argument(
        '--api-key',
        help='OpenAI API key (or set OPENAI_API_KEY env variable)',
        default=None
    )
    parser.add_argument(
        '--output',
        help='Output filename (default: error_localization_analysis.json)',
        default='error_localization_analysis.json'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-analysis even if output file exists'
    )
    parser.add_argument(
        '--model',
        help='OpenAI model to use (default: gpt-4o)',
        default='gpt-4o'
    )
    parser.add_argument(
        '--mode',
        choices=['tot', 'cot', 'auto'],
        default='auto',
        help='Experiment type: tot (Tree of Thought), cot (Chain of Thought), or auto (auto-detect)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Provide via --api-key or OPENAI_API_KEY env variable")
        return 1

    # Check output file
    experiment_path = Path(args.experiment_path)
    output_file = experiment_path / args.output

    if output_file.exists() and not args.force:
        print(f"Analysis file already exists: {output_file}")
        print("Use --force to re-run analysis")
        return 0

    # Detect experiment type if auto
    mode = args.mode
    if mode == 'auto':
        # Load config to detect type
        config_file = experiment_path / "config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            exp_type = detect_experiment_type(config)
            mode = 'cot' if exp_type.startswith('cot') else 'tot'
            print(f"Auto-detected experiment type: {mode}")
        else:
            # Fallback: check results.json structure
            results_file = experiment_path / "results.json"
            with open(results_file, 'r') as f:
                data = json.load(f)
            config = data.get('stats', {})
            exp_type = detect_experiment_type(config)
            mode = 'cot' if exp_type.startswith('cot') else 'tot'
            print(f"Auto-detected experiment type: {mode}")

    # Run analysis
    try:
        print(f"Starting error localization analysis...")
        print(f"Experiment: {experiment_path}")
        print(f"Output: {output_file}")
        print(f"Model: {args.model}")
        print(f"Mode: {mode}")
        print("-" * 80)

        # Dispatch to appropriate analysis function
        if mode == 'cot':
            analysis = analyze_cot_error_localization(args.experiment_path, api_key, model=args.model)
        else:  # tot
            analysis = analyze_experiment(args.experiment_path, api_key, model=args.model)

        # Save results
        print(f"\n{'=' * 80}")
        print(f"Analysis complete!")
        print(f"  Total instances analyzed: {analysis['total_instances']}")

        # Display mode-specific metrics
        if mode == 'cot':
            print(f"  Position agreements: {analysis['summary']['position_agreements']}")
            print(f"  High overlaps (>50%): {analysis['summary']['high_overlaps']}")
            print(f"  Overall agreements: {analysis['summary']['overall_agreements']}")
            print(f"  Position agreement rate: {analysis['summary']['position_agreement_rate']:.1%}")
            print(f"  High overlap rate: {analysis['summary']['high_overlap_rate']:.1%}")
            print(f"  Overall agreement rate: {analysis['summary']['overall_agreement_rate']:.1%}")
            print(f"  Avg quote overlap: {analysis['summary']['avg_quote_overlap']:.2%}")
        else:  # tot
            print(f"  Agreements: {analysis['summary']['agreements']}")
            print(f"  Disagreements: {analysis['summary']['disagreements']}")
            print(f"  Agreement rate: {analysis['summary']['agreement_rate']:.1%}")

        print(f"  Total tokens used: {analysis['summary']['total_tokens_used']:,}")
        print(f"  Estimated cost: ${analysis['summary']['estimated_cost_usd']:.4f}")

        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        return 0

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
