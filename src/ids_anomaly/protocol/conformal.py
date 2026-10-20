"""Split-conformal anomaly detection and p-value merging.

Given anomaly scores ``s`` (higher = more anomalous) on a *calibration* set of normal flows that
is exchangeable with future normal flows, the conformal p-value of a new flow with score ``s``,

    p(s) = (1 + #{calibration scores >= s}) / (n_cal + 1),

is super-uniform under the null "this flow is normal". Flagging ``p <= alpha`` therefore bounds
the false-positive rate on normal traffic by ``alpha`` **in finite samples, for any detector,
with no distributional assumptions**. The guarantee is exactly as strong as exchangeability, and
that is precisely what breaks under drift (E3) and dataset shift (E4), which is what the paper
measures.

Merging several detectors: twice the arithmetic mean of valid p-values is itself a valid
p-value (Vovk & Wang, 2020, "Combining p-values via averaging"), so a calibrated ensemble keeps
the finite-sample FPR guarantee while pooling evidence from heterogeneous detectors.
"""

from __future__ import annotations

import numpy as np


def conformal_pvalues(cal_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Conformal p-values of ``test_scores`` against normal calibration scores."""
    cal_sorted = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = len(cal_sorted)
    # number of calibration scores >= s  ==  n - (number strictly below s)
    n_ge = n - np.searchsorted(cal_sorted, np.asarray(test_scores, dtype=np.float64), side="left")
    return (1.0 + n_ge) / (n + 1.0)


def merge_pvalues_average(pvalues: np.ndarray) -> np.ndarray:
    """Valid merge of ``(n_detectors, n_flows)`` p-values: ``min(1, 2 * mean)``."""
    return np.minimum(1.0, 2.0 * np.asarray(pvalues).mean(axis=0))


def merge_pvalues_min(pvalues: np.ndarray) -> np.ndarray:
    """Bonferroni merge: ``min(1, K * min_k p_k)``. Valid but conservative."""
    p = np.asarray(pvalues)
    return np.minimum(1.0, p.shape[0] * p.min(axis=0))


def empirical_fpr(pvalues_normal: np.ndarray, alpha: float) -> float:
    """Fraction of *normal* flows flagged at level ``alpha`` (should be <= alpha if valid)."""
    return float((np.asarray(pvalues_normal) <= alpha).mean())


def coverage_curve(
    pvalues_normal: np.ndarray, alphas: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Realized FPR for a grid of nominal levels. A valid detector lies on or below y = x."""
    grid = np.linspace(0.005, 0.2, 40) if alphas is None else np.asarray(alphas)
    return grid, np.array([empirical_fpr(pvalues_normal, a) for a in grid])


def tpr_at_alpha(pvalues_attack: np.ndarray, alpha: float) -> float:
    return float((np.asarray(pvalues_attack) <= alpha).mean())
