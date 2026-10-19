"""Feature attribution for anomaly scores, with a faithfulness test.

For reconstruction detectors (autoencoder, masked transformer) the per-feature squared error is
an *exact* additive decomposition of the score, ``score = mean_j err_j``: no approximation, no
sampling, no extra model. That is the attribution the web app shows.

A decomposition being exact says nothing about whether it is *useful*, so two checks accompany it:

* **Deletion faithfulness**: replace the top-k attributed features of each flagged flow with the
  training median (0 after standardization) and re-score. A faithful attribution makes the score
  collapse much faster than deleting k random features.
* **Attribution clustering**: cluster the attribution vectors of flagged flows; if explanations
  carry attack-type information, clusters agree with the (held-out) attack categories (NMI),
  which is what an analyst triaging an alert queue actually needs.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


class _Attributable(Protocol):
    def score(self, X: np.ndarray) -> np.ndarray: ...
    def feature_errors(self, X: np.ndarray) -> np.ndarray: ...


def top_k_features(errors: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices and values of the ``k`` largest per-feature errors of each row, descending."""
    k = min(k, errors.shape[1])
    idx = np.argpartition(-errors, k - 1, axis=1)[:, :k]
    vals = np.take_along_axis(errors, idx, axis=1)
    order = np.argsort(-vals, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(vals, order, axis=1)


def deletion_curve(
    det: _Attributable, X: np.ndarray, ks: tuple[int, ...] = (1, 2, 3, 5, 8), seed: int = 0
) -> dict[str, list[float]]:
    """Mean fraction of the anomaly score *removed* after deleting k features.

    Returns ``{"k": [...], "attributed": [...], "random": [...]}``; faithful attributions give
    ``attributed >> random``.
    """
    rng = np.random.default_rng(seed)
    base = det.score(X)
    errors = det.feature_errors(X)
    order = np.argsort(-errors, axis=1)
    out: dict[str, list[float]] = {"k": [float(k) for k in ks], "attributed": [], "random": []}
    for k in ks:
        top = X.copy()
        np.put_along_axis(top, order[:, :k], 0.0, axis=1)
        rand = X.copy()
        rand_idx = np.argsort(rng.random(X.shape), axis=1)[:, :k]
        np.put_along_axis(rand, rand_idx, 0.0, axis=1)
        denom = np.maximum(base, 1e-12)
        out["attributed"].append(float(np.mean(np.clip((base - det.score(top)) / denom, 0, 1))))
        out["random"].append(float(np.mean(np.clip((base - det.score(rand)) / denom, 0, 1))))
    return out


def attribution_cluster_nmi(
    errors: np.ndarray, categories: np.ndarray, n_clusters: int | None = None, seed: int = 0
) -> float:
    """NMI between KMeans clusters of normalized attribution vectors and attack categories."""
    if len(errors) < 10:
        return float("nan")
    norm = errors / np.maximum(errors.sum(axis=1, keepdims=True), 1e-12)
    k = n_clusters or max(2, len(np.unique(categories)))
    labels = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(norm)
    return float(normalized_mutual_info_score(categories, labels))
