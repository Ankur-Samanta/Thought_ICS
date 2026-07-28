#!/usr/bin/env python3
"""
Compute comprehensive metrics for self-correction experiments.

This version provides:
- Clear, unambiguous metric names
- Self-documenting with _description and _def fields
- No redundant metrics
- Proper handling of undefined metrics (null instead of 0.0)
- Logical organization
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys


class ExperimentType:
    """Enum for experiment types."""
    TOT = "tot"
    BASELINE_ITERATIVE = "baseline_iterative"
    MAJORITY_VOTE = "majority_vote"
    UNKNOWN = "unknown"


def detect_experiment_type(results: Dict[str, Any]) -> ExperimentType:
    """Detect experiment type from results structure."""
    if not results.get("results") or len(results["results"]) == 0:
        return ExperimentType.UNKNOWN

    first_result = results["results"][0]

    if "samples" in first_result and "answer_counts" in first_result:
        return ExperimentType.MAJORITY_VOTE

    if "iterations" in first_result:
        if len(first_result["iterations"]) > 0:
            first_iter = first_result["iterations"][0]
            if "chain" in first_iter and isinstance(first_iter.get("chain"), list):
                return ExperimentType.TOT

    if "iterations_data" in first_result:
        return ExperimentType.BASELINE_ITERATIVE

    return ExperimentType.UNKNOWN

# 计算每一轮结束后的整体准确率变化
def compute_accuracy_trajectory(results: List[Dict[str, Any]],
                                iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Compute accuracy evolution across iterations.

    For each iteration N:
    - accuracy: Fraction of problems correct at iteration N (using most recent state)
    - edited_count: Number of problems that performed iteration N
    - edited_now_correct: Of problems edited at N, how many are now correct
    - edited_accuracy: edited_now_correct / edited_count
    """
    all_iter_nums = set()
    for problem in results:
        iterations = problem.get(iter_key, [])
        for iteration in iterations:
            all_iter_nums.add(iteration["iteration"])

    by_iteration = {}
    for iter_num in sorted(all_iter_nums):
        # Overall accuracy at this iteration
        correct_count = 0
        total_count = 0

        for problem in results:
            iterations = problem.get(iter_key, [])
            most_recent = None
            for iteration in iterations:
                if iteration["iteration"] <= iter_num:
                    if most_recent is None or iteration["iteration"] > most_recent["iteration"]:
                        most_recent = iteration

            if most_recent is not None:
                total_count += 1
                if most_recent["correct"]:
                    correct_count += 1

        accuracy = round(correct_count / total_count, 4) if total_count > 0 else 0.0

        # Edited metrics (only for problems that performed this iteration)
        if iter_num == 0:
            edited_count = total_count
            edited_now_correct = correct_count
        else:
            edited_count = 0
            edited_now_correct = 0

            for problem in results:
                iterations = problem.get(iter_key, [])
                has_prev = any(it["iteration"] == iter_num - 1 for it in iterations)
                has_curr = any(it["iteration"] == iter_num for it in iterations)

                if has_prev and has_curr:
                    edited_count += 1
                    curr_iter = next(it for it in iterations if it["iteration"] == iter_num)
                    if curr_iter["correct"]:
                        edited_now_correct += 1

        edited_accuracy = round(edited_now_correct / edited_count, 4) if edited_count > 0 else 0.0

        by_iteration[f"iter_{iter_num}"] = {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count,
            "edited_count": edited_count,
            "edited_now_correct": edited_now_correct,
            "edited_accuracy": edited_accuracy
        }

    return {
        "_description": "Accuracy evolution across iterations",
        "by_iteration": by_iteration,
        "_definitions": {
            "accuracy": "Fraction of problems correct at this iteration (using most recent state up to this iter)",
            "edited_count": "Number of problems that performed an iteration at this step",
            "edited_now_correct": "Of problems edited at this iteration, how many are now correct",
            "edited_accuracy": "edited_now_correct / edited_count"
        }
    }


def compute_iteration_transitions(results: List[Dict[str, Any]],
                                  iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Analyze transitions between consecutive iterations.

    Tracks wrong->correct, wrong->wrong, correct->wrong, correct->correct.
    """
    transitions = {
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
        "correct_to_wrong": 0,
        "correct_to_correct": 0
    }

    total_transitions = 0

    for problem in results:
        iterations = problem.get(iter_key, [])
        if len(iterations) < 2:
            continue

        for i in range(1, len(iterations)):
            prev_correct = iterations[i-1]["correct"]
            curr_correct = iterations[i]["correct"]
            total_transitions += 1

            if not prev_correct and curr_correct:
                transitions["wrong_to_correct"] += 1
            elif prev_correct and not curr_correct:
                transitions["correct_to_wrong"] += 1
            elif not prev_correct and not curr_correct:
                transitions["wrong_to_wrong"] += 1
            else:
                transitions["correct_to_correct"] += 1

    rates = {}
    if total_transitions > 0:
        rates["correction_success_rate"] = round(transitions["wrong_to_correct"] / total_transitions, 4)
        rates["correction_failure_rate"] = round(transitions["wrong_to_wrong"] / total_transitions, 4)
        rates["error_introduction_rate"] = round(transitions["correct_to_wrong"] / total_transitions, 4)
        if transitions["correct_to_correct"] > 0:
            rates["maintained_correctness_rate"] = round(transitions["correct_to_correct"] / total_transitions, 4)
    else:
        rates["correction_success_rate"] = 0.0
        rates["correction_failure_rate"] = 0.0
        rates["error_introduction_rate"] = 0.0

    return {
        "_description": "Analysis of transitions between consecutive iterations",
        "total_transitions": total_transitions,
        "_note": "One transition per problem per iteration (except iter 0)",
        "transition_outcomes": transitions,
        "rates": rates,
        "_definitions": {
            "correction_success_rate": "wrong_to_correct / total_transitions",
            "correction_failure_rate": "wrong_to_wrong / total_transitions",
            "error_introduction_rate": "correct_to_wrong / total_transitions"
        }
    }


def compute_error_detection_ability(results: List[Dict[str, Any]],
                                    iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Compute C1 error detection metrics with confusion matrix.

    Measures model's ability to detect errors when attempting corrections.
    Only applies when model continues to next iteration.
    """
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0

    total_detection_attempts = 0

    for problem in results:
        iterations = problem.get(iter_key, [])
        if len(iterations) < 2:
            continue

        for i in range(1, len(iterations)):
            prev_iter = iterations[i-1]
            curr_iter = iterations[i]

            prev_answer_correct = prev_iter["correct"]
            error_step = curr_iter.get("error_step")

            if error_step is None:
                continue

            total_detection_attempts += 1
            model_detected_error = (error_step > 0)

            if not prev_answer_correct and model_detected_error:
                true_positive += 1
            elif prev_answer_correct and model_detected_error:
                false_positive += 1
            elif prev_answer_correct and not model_detected_error:
                true_negative += 1
            elif not prev_answer_correct and not model_detected_error:
                false_negative += 1

    # Compute metrics
    confusion_matrix = {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative
    }

    # Precision
    precision = round(true_positive / (true_positive + false_positive), 4) if (true_positive + false_positive) > 0 else None

    # Recall
    recall = round(true_positive / (true_positive + false_negative), 4) if (true_positive + false_negative) > 0 else None

    # F1 Score
    if precision is not None and recall is not None and precision + recall > 0:
        f1_score = round(2 * (precision * recall) / (precision + recall), 4)
    else:
        f1_score = None

    # Specificity
    specificity = round(true_negative / (true_negative + false_positive), 4) if (true_negative + false_positive) > 0 else None

    performance_metrics = {
        "precision": precision,
        "_def_precision": "TP / (TP + FP) - When model flags error, % it's correct",

        "recall": recall,
        "_def_recall": "TP / (TP + FN) - Of actual errors, % model detects",

        "f1_score": f1_score,
        "_def_f1": "Harmonic mean of precision and recall",

        "specificity": specificity,
        "_def_specificity": "TN / (TN + FP) - Of correct answers, % model correctly says no error"
    }

    # Add interpretation notes
    interpretation = {}
    if false_positive == 0 and true_negative == 0:
        interpretation["FP=0, TN=0"] = "Model ALWAYS stops immediately after getting correct answer"
        performance_metrics["_note_specificity"] = "null because model never continues after correct answer"

    if precision == 1.0:
        interpretation["Precision=100%"] = "When model detects error, it's always actually wrong"

    if recall is not None:
        interpretation[f"Recall={recall:.1%}"] = f"Model detects {recall:.1%} of actual errors, misses {(1-recall):.1%}"

    return {
        "_description": "Ability to detect errors when attempting corrections (C1 metric)",
        "_scope": "Only applies when model continues to next iteration (same events as iteration_transitions)",
        "confusion_matrix": {
            "_rows": "Ground truth: was previous answer wrong (pos) or correct (neg)?",
            "_cols": "Prediction: did model detect error (pos) or say no error (neg)?",
            "true_positive": true_positive,
            "_def_tp": "Model detected error (error_step > 0) AND previous answer was wrong",
            "false_negative": false_negative,
            "_def_fn": "Model said no error (error_step = 0) AND previous answer was wrong",
            "false_positive": false_positive,
            "_def_fp": "Model detected error AND previous answer was correct",
            "true_negative": true_negative,
            "_def_tn": "Model said no error AND previous answer was correct"
        },
        "performance_metrics": performance_metrics,
        "_interpretation": interpretation if interpretation else None
    }


def compute_stopping_behavior(results: List[Dict[str, Any]],
                              max_iterations: int,
                              iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Analyze when and why model stopped iterating.
    """
    stopped_before_max = 0
    hit_max_iterations = 0

    stopped_with_correct = 0
    stopped_with_wrong = 0

    continued_after_correct_count = 0
    problems_with_correct = 0

    for problem in results:
        iterations = problem.get(iter_key, [])
        if len(iterations) == 0:
            continue

        total_iters = len(iterations)
        final_correct = iterations[-1]["correct"]

        # Check stop reason
        if total_iters >= max_iterations:
            hit_max_iterations += 1
        else:
            stopped_before_max += 1
            if final_correct:
                stopped_with_correct += 1
            else:
                stopped_with_wrong += 1

        # Check if continued after correct
        first_correct_idx = None
        for i, iteration in enumerate(iterations):
            if iteration["correct"]:
                first_correct_idx = i
                problems_with_correct += 1
                break

        if first_correct_idx is not None and first_correct_idx < len(iterations) - 1:
            continued_after_correct_count += 1

    premature_stop_rate = round(stopped_with_wrong / stopped_before_max, 4) if stopped_before_max > 0 else 0.0
    continued_rate = round(continued_after_correct_count / problems_with_correct, 4) if problems_with_correct > 0 else 0.0

    return {
        "_description": "When and why did model stop iterating?",
        "stop_reasons": {
            "stopped_before_max": stopped_before_max,
            "_def": "Problems that stopped before hitting max_iterations limit",
            "hit_max_iterations": hit_max_iterations,
            "_def_hit_max": "Problems that reached max_iterations cap (correction failure)"
        },
        "stop_correctness": {
            "stopped_with_correct_answer": stopped_with_correct,
            "stopped_with_wrong_answer": stopped_with_wrong,
            "premature_stop_rate": premature_stop_rate,
            "_def_rate": "stopped_with_wrong / stopped_before_max"
        },
        "continued_after_correct": {
            "count": continued_after_correct_count,
            "rate": continued_rate,
            "_def": "Problems that found correct answer but continued iterating",
            "_denominator": f"problems_with_correct_answer = {problems_with_correct}"
        }
    }


def compute_convergence_patterns(results: List[Dict[str, Any]],
                                 iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Categorize problems by when they reached correct answer.
    """
    categories = {
        "immediate_success": 0,
        "early_convergence": 0,
        "late_convergence": 0,
        "never_correct": 0
    }

    oscillating_count = 0

    for problem in results:
        iterations = problem.get(iter_key, [])
        if len(iterations) == 0:
            continue

        correctness_seq = [it["correct"] for it in iterations]

        # Find first correct iteration
        first_correct_idx = None
        for i, correct in enumerate(correctness_seq):
            if correct:
                first_correct_idx = i
                break

        if first_correct_idx is None:
            categories["never_correct"] += 1
        elif first_correct_idx == 0:
            categories["immediate_success"] += 1
        elif first_correct_idx <= 3:
            categories["early_convergence"] += 1
        else:
            categories["late_convergence"] += 1

        # Check oscillation
        transitions = 0
        for i in range(1, len(correctness_seq)):
            if correctness_seq[i] != correctness_seq[i-1]:
                transitions += 1
        if transitions > 2:
            oscillating_count += 1

    return {
        "_description": "Categorize problems by when they reached correct answer",
        "categories": {
            "immediate_success": categories["immediate_success"],
            "_def_immediate": "Correct at iteration 0",
            "early_convergence": categories["early_convergence"],
            "_def_early": "First correct at iterations 1-3",
            "late_convergence": categories["late_convergence"],
            "_def_late": "First correct after iteration 3",
            "never_correct": categories["never_correct"],
            "_def_never": "Never reached correct answer"
        },
        "oscillation_analysis": {
            "oscillating_problems": oscillating_count,
            "_def": "Problems with >2 transitions between correct and wrong states",
            "_note": "Can overlap with early/late convergence categories"
        }
    }


def compute_iteration_statistics(results: List[Dict[str, Any]],
                                 iter_key: str = "iterations") -> Dict[str, Any]:
    """
    Efficiency and resource usage metrics.
    """
    total_iterations = 0
    successful_count = 0
    unsuccessful_count = 0
    iterations_for_successful = []
    iterations_for_unsuccessful = []

    for problem in results:
        iterations = problem.get(iter_key, [])
        num_iters = len(iterations)
        total_iterations += num_iters

        if problem.get("success", False):
            successful_count += 1
            iterations_for_successful.append(num_iters)
        else:
            unsuccessful_count += 1
            iterations_for_unsuccessful.append(num_iters)

    total_problems = len(results)
    avg_iterations = round(total_iterations / total_problems, 4) if total_problems > 0 else 0.0
    avg_successful = round(sum(iterations_for_successful) / len(iterations_for_successful), 4) if iterations_for_successful else 0.0
    avg_unsuccessful = round(sum(iterations_for_unsuccessful) / len(iterations_for_unsuccessful), 4) if iterations_for_unsuccessful else 0.0

    return {
        "_description": "Efficiency and resource usage metrics",
        "total_iterations": total_iterations,
        "avg_iterations_per_problem": avg_iterations,
        "successful_problems": {
            "count": successful_count,
            "avg_iterations": avg_successful
        },
        "unsuccessful_problems": {
            "count": unsuccessful_count,
            "avg_iterations": avg_unsuccessful
        }
    }

# 总入口
def compute_metrics(experiment_dir: Path) -> Dict[str, Any]:
    """
    Main function to compute all metrics with clean schema.
    """
    results_file = experiment_dir / "results.json"
    config_file = experiment_dir / "config.json"

    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found in {experiment_dir}")

    with open(results_file) as f:
        results_data = json.load(f)

    config = {}
    if config_file.exists():
        with open(config_file) as f:
            config = json.load(f)

    exp_type = detect_experiment_type(results_data)
    if exp_type == ExperimentType.UNKNOWN:
        raise ValueError(f"Could not detect experiment type for {experiment_dir}")

    # Build metrics
    metrics = {
        "metadata": {
            "experiment_name": experiment_dir.name,
            "experiment_type": exp_type,
            "computed_at": datetime.now().isoformat(),
            "config": {
                "model": config.get("model_name", results_data.get("stats", {}).get("baseline_name", "unknown")),
                "dataset": config.get("dataset", "unknown"),
                "max_iterations": config.get("max_iterations", 10),
                "total_problems": len(results_data.get("results", []))
            }
        }
    }

    results_list = results_data.get("results", [])
    max_iterations = config.get("max_iterations", 10)

    if exp_type == ExperimentType.MAJORITY_VOTE:
        # TODO: Implement majority vote metrics
        metrics["majority_vote"] = {"_todo": "Implement majority vote metrics"}

    elif exp_type in [ExperimentType.TOT, ExperimentType.BASELINE_ITERATIVE]:
        iter_key = "iterations" if exp_type == ExperimentType.TOT else "iterations_data"

        # Compute all sections
        trajectory = compute_accuracy_trajectory(results_list, iter_key)

        # Overall performance (derived from trajectory)
        first_iter = trajectory["by_iteration"].get("iter_0", {})
        last_iter_key = max(k for k in trajectory["by_iteration"].keys())
        last_iter = trajectory["by_iteration"][last_iter_key]

        first_acc = first_iter.get("accuracy", 0.0)
        final_acc = last_iter.get("accuracy", 0.0)

        metrics["overall_performance"] = {
            "_description": "High-level accuracy metrics",
            "first_attempt_accuracy": first_acc,
            "final_accuracy": final_acc,
            "absolute_improvement": round(final_acc - first_acc, 4),
            "relative_improvement": round((final_acc - first_acc) / first_acc, 4) if first_acc > 0 else 0.0,
            "_note": "relative_improvement = absolute_improvement / first_attempt_accuracy"
        }

        metrics["accuracy_trajectory"] = trajectory
        metrics["iteration_transitions"] = compute_iteration_transitions(results_list, iter_key)
        metrics["error_detection_ability"] = compute_error_detection_ability(results_list, iter_key)
        metrics["stopping_behavior"] = compute_stopping_behavior(results_list, max_iterations, iter_key)
        metrics["convergence_patterns"] = compute_convergence_patterns(results_list, iter_key)
        metrics["iteration_statistics"] = compute_iteration_statistics(results_list, iter_key)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute comprehensive metrics for self-correction experiments")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment directory")
    parser.add_argument("--output", type=str, help="Output file path (default: experiment_dir/metrics.json)")
    parser.add_argument("--pretty", action="store_true", help="Pretty print JSON output")

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.exists():
        print(f"Error: Directory {experiment_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Computing metrics for {experiment_dir.name}...")
        metrics = compute_metrics(experiment_dir)

        if args.output:
            output_file = Path(args.output)
        else:
            output_file = experiment_dir / "metrics.json"

        with open(output_file, 'w') as f:
            if args.pretty:
                json.dump(metrics, f, indent=2)
            else:
                json.dump(metrics, f)

        print(f"✓ Metrics saved to {output_file}")

        # Print summary
        print("\n=== Metrics Summary ===")
        overall = metrics.get("overall_performance", {})
        print(f"First Attempt: {overall.get('first_attempt_accuracy', 0):.1%}")
        print(f"Final Accuracy: {overall.get('final_accuracy', 0):.1%}")
        print(f"Improvement: {overall.get('absolute_improvement', 0):.1%}")

    except Exception as e:
        print(f"Error computing metrics: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
