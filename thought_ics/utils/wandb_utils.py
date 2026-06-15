#!/usr/bin/env python3
"""
Weights & Biases Integration Utilities

Provides helper functions for experiment tracking with wandb.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from contextlib import contextmanager

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logging.warning("wandb not installed. Install with: pip install wandb")

logger = logging.getLogger(__name__)


class WandbConfig:
    """Configuration for wandb integration."""

    def __init__(
        self,
        enabled: bool = True,
        project: str = "thought-ics",
        entity: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        mode: str = "online"  # online, offline, or disabled
    ):
        """
        Initialize wandb configuration.

        Args:
            enabled: Whether to enable wandb logging
            project: Wandb project name
            entity: Wandb entity (username or team)
            tags: List of tags for the run
            notes: Description of the run
            mode: wandb mode (online, offline, disabled)
        """
        self.enabled = enabled and WANDB_AVAILABLE
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.notes = notes
        self.mode = mode if enabled else "disabled"

        if enabled and not WANDB_AVAILABLE:
            logger.warning("wandb is not available. Logging will be disabled.")
            self.enabled = False


def init_wandb_run(
    config: WandbConfig,
    run_name: Optional[str] = None,
    job_type: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None,
    resume: Optional[str] = None
) -> Optional[Any]:
    """
    Initialize a wandb run.

    Args:
        config: WandbConfig object
        run_name: Name for this run
        job_type: Type of job (e.g., "train", "eval", "baseline")
        run_config: Dictionary of hyperparameters and configuration
        resume: Resume mode ("allow", "must", "never", or run_id)

    Returns:
        wandb.Run object if enabled, None otherwise
    """
    if not config.enabled:
        logger.info("wandb is disabled. Skipping initialization.")
        return None

    try:
        run = wandb.init(
            project=config.project,
            entity=config.entity,
            name=run_name,
            job_type=job_type,
            config=run_config or {},
            tags=config.tags,
            notes=config.notes,
            mode=config.mode,
            resume=resume
        )
        logger.info(f"Initialized wandb run: {run.name} (id: {run.id})")
        return run
    except Exception as e:
        logger.error(f"Failed to initialize wandb: {e}")
        return None


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metric names and values
        step: Optional step number
        commit: Whether to commit the metrics immediately
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        wandb.log(metrics, step=step, commit=commit)
    except Exception as e:
        logger.error(f"Failed to log metrics: {e}")


def log_artifact(
    artifact_path: str,
    artifact_name: str,
    artifact_type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a file or directory as a wandb artifact.

    Args:
        artifact_path: Path to file or directory
        artifact_name: Name for the artifact
        artifact_type: Type of artifact (e.g., "dataset", "model", "result")
        description: Description of the artifact
        metadata: Additional metadata dictionary
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            description=description,
            metadata=metadata
        )

        path = Path(artifact_path)
        if path.is_file():
            artifact.add_file(str(path))
        elif path.is_dir():
            artifact.add_dir(str(path))
        else:
            logger.warning(f"Artifact path does not exist: {artifact_path}")
            return

        wandb.run.log_artifact(artifact)
        logger.info(f"Logged artifact: {artifact_name} ({artifact_type})")
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}")


def log_tree_artifact(
    tree_json_path: str,
    problem_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a tree structure as an artifact.

    Args:
        tree_json_path: Path to tree JSON file
        problem_id: ID of the problem
        metadata: Additional metadata
    """
    artifact_name = f"tree_{problem_id}"
    log_artifact(
        artifact_path=tree_json_path,
        artifact_name=artifact_name,
        artifact_type="tree",
        description=f"Tree of Thought for problem {problem_id}",
        metadata=metadata
    )


def log_chain_artifact(
    chain_json_path: str,
    problem_id: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a reasoning chain as an artifact.

    Args:
        chain_json_path: Path to chain JSON file
        problem_id: ID of the problem
        metadata: Additional metadata
    """
    artifact_name = f"chain_{problem_id}"
    log_artifact(
        artifact_path=chain_json_path,
        artifact_name=artifact_name,
        artifact_type="chain",
        description=f"Reasoning chain for problem {problem_id}",
        metadata=metadata
    )


def log_table(
    table_name: str,
    columns: List[str],
    data: List[List[Any]]
) -> None:
    """
    Log a table to wandb.

    Args:
        table_name: Name of the table
        columns: Column names
        data: List of rows
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        table = wandb.Table(columns=columns, data=data)
        wandb.log({table_name: table})
    except Exception as e:
        logger.error(f"Failed to log table: {e}")


def finish_run() -> None:
    """Finish the current wandb run."""
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    try:
        wandb.finish()
        logger.info("Finished wandb run")
    except Exception as e:
        logger.error(f"Failed to finish wandb run: {e}")


@contextmanager
def wandb_experiment(
    config: WandbConfig,
    run_name: Optional[str] = None,
    job_type: Optional[str] = None,
    run_config: Optional[Dict[str, Any]] = None
):
    """
    Context manager for wandb experiments.

    Usage:
        with wandb_experiment(config, run_name="test", job_type="eval") as run:
            # Your experiment code here
            log_metrics({"accuracy": 0.95})

    Args:
        config: WandbConfig object
        run_name: Name for this run
        job_type: Type of job
        run_config: Dictionary of hyperparameters
    """
    run = init_wandb_run(config, run_name, job_type, run_config)
    try:
        yield run
    finally:
        finish_run()


def setup_wandb_env(api_key: Optional[str] = None) -> None:
    """
    Setup wandb environment variables.

    Args:
        api_key: Optional API key (will be set as environment variable)
    """
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        logger.info("Set WANDB_API_KEY from provided value")
    elif "WANDB_API_KEY" not in os.environ:
        logger.warning(
            "WANDB_API_KEY not set. You may need to run 'wandb login' "
            "or set the WANDB_API_KEY environment variable."
        )


def create_run_name(
    model_name: str,
    dataset_name: str,
    experiment_type: str,
    **kwargs
) -> str:
    """
    Create a standardized run name.

    Args:
        model_name: Model name (e.g., "llama8b")
        dataset_name: Dataset name (e.g., "math_level5")
        experiment_type: Type of experiment (e.g., "tot", "cot", "baseline")
        **kwargs: Additional parameters to include in name

    Returns:
        Formatted run name
    """
    parts = [experiment_type, model_name, dataset_name]

    # Add additional parameters
    for key, value in kwargs.items():
        if value is not None:
            parts.append(f"{key}={value}")

    return "_".join(parts)


def log_problem_result(
    problem_id: str,
    problem_number: int,
    predicted_answer: str,
    ground_truth: str,
    correct: bool,
    iterations: int = 1,
    additional_metrics: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log results for a single problem.

    Args:
        problem_id: Problem identifier
        problem_number: Problem number in dataset
        predicted_answer: Model's predicted answer
        ground_truth: Correct answer
        correct: Whether prediction is correct
        iterations: Number of iterations taken
        additional_metrics: Additional metrics to log
    """
    metrics = {
        "problem_id": problem_id,
        "problem_number": problem_number,
        "correct": int(correct),
        "iterations": iterations,
    }

    if additional_metrics:
        metrics.update(additional_metrics)

    log_metrics(metrics)


def log_summary_metrics(
    total_problems: int,
    correct_problems: int,
    mean_iterations: float,
    additional_summary: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log summary metrics for an experiment.

    Args:
        total_problems: Total number of problems
        correct_problems: Number of correct solutions
        mean_iterations: Mean number of iterations
        additional_summary: Additional summary metrics
    """
    if not WANDB_AVAILABLE or wandb.run is None:
        return

    summary = {
        "total_problems": total_problems,
        "correct_problems": correct_problems,
        "accuracy": correct_problems / total_problems if total_problems > 0 else 0.0,
        "mean_iterations": mean_iterations
    }

    if additional_summary:
        summary.update(additional_summary)

    # Update run summary
    for key, value in summary.items():
        wandb.run.summary[key] = value

    logger.info(f"Logged summary metrics: {summary}")
