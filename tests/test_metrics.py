from __future__ import annotations

import numpy as np

from ids_anomaly.evaluation.error_analysis import (
    consensus_misses,
    method_agreement,
    rank_normalized_scores,
)
from ids_anomaly.evaluation.metrics import (
    anomaly_scoring_metrics,
    clustering_metrics,
    f1_at_quantile_threshold,
    per_category_detection_rate,
)


def test_anomaly_scoring_metrics_perfect_separation_gives_auc_one():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    metrics = anomaly_scoring_metrics(y, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0


def test_anomaly_scoring_metrics_random_scores_near_half_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000)
    scores = rng.random(2000)
    metrics = anomaly_scoring_metrics(y, scores)
    assert 0.4 < metrics["roc_auc"] < 0.6


def test_f1_at_quantile_threshold_flags_expected_fraction():
    rng = np.random.default_rng(0)
    scores = rng.random(1000)
    y = (scores > 0.8).astype(int)
    result = f1_at_quantile_threshold(y, scores, quantile=0.2)
    assert abs(result["flagged_fraction"] - 0.2) < 0.02
    assert result["f1"] > 0.9  # threshold-selected top-20% should near-perfectly match y


def test_clustering_metrics_perfect_clustering_gives_ari_one():
    labels_true = np.array(["a"] * 50 + ["b"] * 50)
    X = np.vstack([np.random.default_rng(0).normal(0, 0.1, (50, 2)),
                    np.random.default_rng(1).normal(5, 0.1, (50, 2))])
    cluster_labels = np.array([0] * 50 + [1] * 50)
    metrics = clustering_metrics(X, cluster_labels, labels_true)
    assert metrics["ari"] == 1.0
    assert metrics["n_clusters"] == 2


def test_clustering_metrics_excludes_noise_from_silhouette():
    X = np.random.default_rng(0).normal(size=(30, 2))
    cluster_labels = np.array([-1] * 10 + [0] * 10 + [1] * 10)
    labels_true = np.array(["x"] * 30)
    metrics = clustering_metrics(X, cluster_labels, labels_true)
    assert not np.isnan(metrics["silhouette"])


def test_per_category_detection_rate_sums_to_group_sizes():
    cat = np.array(["dos"] * 10 + ["normal"] * 10)
    scores = np.concatenate([np.full(10, 0.9), np.full(10, 0.1)])
    df = per_category_detection_rate(cat, scores, quantile=0.5)
    dos_row = df[df["attack_category"] == "dos"].iloc[0]
    assert dos_row["detection_rate"] == 1.0


def test_rank_normalized_scores_are_bounded_in_unit_interval():
    scores = {"a": np.random.default_rng(0).random(100), "b": np.random.default_rng(1).random(100)}
    ranked = rank_normalized_scores(scores)
    assert ranked.min().min() > 0
    assert ranked.max().max() <= 1.0


def test_consensus_misses_only_returns_attacks():
    ranked = rank_normalized_scores({"a": np.array([0.1, 0.9, 0.5]), "b": np.array([0.2, 0.8, 0.4])})
    is_attack = np.array([True, True, False])
    misses = consensus_misses(ranked, is_attack, quantile=0.3, min_methods_missing=1)
    assert misses["is_attack"].all()


def test_method_agreement_diagonal_is_one():
    ranked = rank_normalized_scores({"a": np.random.default_rng(0).random(200)})
    agreement = method_agreement(ranked, quantile=0.2)
    assert agreement.loc["a", "a"] == 1.0
