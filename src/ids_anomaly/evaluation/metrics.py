"""Common metrics used to put every method -- reduction, clustering, and anomaly detection --
on the same scoreboard, even though they optimize very different objectives.

Labels (``is_attack`` / ``attack_category``) are used here and only here: purely for scoring
already-fitted models, never for fitting them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    f1_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)


def anomaly_scoring_metrics(y_true_is_attack: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """ROC-AUC and PR-AUC (average precision), threshold-free -- the primary comparison axis
    across Isolation Forest / One-Class SVM / Deep SVDD / GMM-likelihood / AE-reconstruction.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true_is_attack, scores)),
        "pr_auc": float(average_precision_score(y_true_is_attack, scores)),
    }


def f1_at_quantile_threshold(
    y_true_is_attack: np.ndarray, scores: np.ndarray, quantile: float
) -> dict[str, float]:
    """F1 at a score threshold set from a target flagged-fraction (quantile), the operating
    point a SOC analyst would actually pick: "flag the top X% most anomalous flows."
    """
    threshold = np.quantile(scores, 1 - quantile)
    preds = (scores >= threshold).astype(int)
    return {
        "threshold_quantile": quantile,
        "threshold_value": float(threshold),
        "f1": float(f1_score(y_true_is_attack, preds, zero_division=0)),
        "flagged_fraction": float(preds.mean()),
    }


def clustering_metrics(
    X: np.ndarray, cluster_labels: np.ndarray, attack_category: np.ndarray
) -> dict[str, float]:
    """Internal (silhouette, needs no ground truth) + external (ARI/NMI against attack category,
    evaluation-only) clustering quality. Noise points (-1, from HDBSCAN) are excluded from
    silhouette since it is undefined for a "non-cluster" label.
    """
    metrics: dict[str, float] = {
        "ari": float(adjusted_rand_score(attack_category, cluster_labels)),
        "nmi": float(normalized_mutual_info_score(attack_category, cluster_labels)),
        "n_clusters": int(len(set(cluster_labels.tolist()) - {-1})),
    }
    mask = cluster_labels != -1
    unique_labels = set(cluster_labels[mask].tolist())
    if mask.sum() > 1 and 1 < len(unique_labels) < mask.sum():
        sample_size = min(10_000, int(mask.sum()))
        metrics["silhouette"] = float(
            silhouette_score(X[mask], cluster_labels[mask], sample_size=sample_size, random_state=42)
        )
    else:
        metrics["silhouette"] = float("nan")
    return metrics


def per_category_detection_rate(
    attack_category: np.ndarray, scores: np.ndarray, quantile: float
) -> pd.DataFrame:
    """Recall broken down by attack category at a fixed flagged-fraction threshold -- the core
    error-analysis table: which attack families does each method miss?
    """
    threshold = np.quantile(scores, 1 - quantile)
    flagged = scores >= threshold
    df = pd.DataFrame({"attack_category": attack_category, "flagged": flagged})
    summary = df.groupby("attack_category", observed=True).agg(
        n=("flagged", "size"), n_flagged=("flagged", "sum")
    )
    summary["detection_rate"] = summary["n_flagged"] / summary["n"]
    return summary.reset_index()
