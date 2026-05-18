"""
lpi_b2 — LPI-B² (Latent Performance Indicator, Bootstrap-Bayesian)

Public API:
    LPIB2Evaluator          — main sklearn-compatible estimator
    get_label_noise_audit   — per-sample noise characterisation
    optimize_noise_threshold
    get_entropy_audit
"""

from .estimator import (
    LPIB2Evaluator,
    get_label_noise_audit,
    optimize_noise_threshold,
    get_entropy_audit,
)

__all__ = [
    "LPIB2Evaluator",
    "get_label_noise_audit",
    "optimize_noise_threshold",
    "get_entropy_audit",
]
