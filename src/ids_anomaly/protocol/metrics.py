"""Threshold-free and operating-point metrics for anomaly scores."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def tpr_at_fpr(y: np.ndarray, s: np.ndarray, fpr_target: float = 0.01) -> float:
    """Best true-positive rate achievable at false-positive rate <= ``fpr_target``."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[np.searchsorted(fpr, fpr_target, side="right") - 1])


def operating_point(
    y: np.ndarray, s: np.ndarray, cal_scores: np.ndarray, alpha: float = 0.01
) -> dict[str, float]:
    """Metrics at a threshold fixed *before* seeing test data: the ``1 - alpha`` quantile of
    normal calibration scores. Realized FPR vs ``alpha`` is the calibration-honesty check."""
    thr = float(np.quantile(cal_scores, 1.0 - alpha))
    flagged = s >= thr
    normal, attack = ~y.astype(bool), y.astype(bool)
    tp = float((flagged & attack).sum())
    fp = float((flagged & normal).sum())
    fn = float((~flagged & attack).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "alpha": alpha,
        "threshold": thr,
        "fpr": fp / max(1.0, float(normal.sum())),
        "tpr": recall,
        "precision": precision,
        "f1": f1,
    }


def bootstrap_ci(
    y: np.ndarray,
    s: np.ndarray,
    metric=roc_auc,
    n_boot: int = 200,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Stratified bootstrap confidence interval over test flows."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        vals[i] = metric(y[idx], s[idx])
    lo, hi = np.quantile(vals, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI of ``current`` against ``reference`` using reference-quantile bins: a label-free drift
    signal (>0.1 is conventionally 'moderate', >0.25 'major' shift)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_frac = np.histogram(reference, edges)[0] / len(reference)
    cur_frac = np.histogram(current, edges)[0] / len(current)
    ref_frac, cur_frac = np.clip(ref_frac, 1e-6, None), np.clip(cur_frac, 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))
