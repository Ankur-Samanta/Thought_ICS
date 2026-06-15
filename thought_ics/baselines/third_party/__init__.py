"""
Third-Party Baseline Methods for Self-Correction Evaluation

This package implements benchmark methods for comparison:
- Self-Refine (Madaan et al., 2023): Iterative refinement with self-feedback
- Chain-of-Verification (Dhuliawala et al., 2023): Verification-based refinement
"""

from .self_refine import (
    self_refine_single,
    self_refine_batch,
    SelfRefineResult,
    get_final_answer,
    compute_self_refine_metrics
)

from .chain_of_verification import (
    cove_single,
    cove_batch,
    CoVeResult,
    get_final_answer as cove_get_final_answer,
    get_baseline_answer as cove_get_baseline_answer,
    compute_cove_metrics
)

__all__ = [
    # Self-Refine
    'self_refine_single',
    'self_refine_batch',
    'SelfRefineResult',
    'compute_self_refine_metrics',
    # Chain-of-Verification
    'cove_single',
    'cove_batch',
    'CoVeResult',
    'cove_get_final_answer',
    'cove_get_baseline_answer',
    'compute_cove_metrics',
]
