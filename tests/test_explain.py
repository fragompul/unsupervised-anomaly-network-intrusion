from __future__ import annotations

import numpy as np

from ids_anomaly.detectors import build
from ids_anomaly.explain import attribution


def _data():
    rng = np.random.default_rng(0)
    d = 10
    train = rng.normal(size=(2500, d)).astype("float32")
    train[:, 5:] += train[:, :5]
    test = rng.normal(size=(200, d)).astype("float32")
    test[:, 5:] += test[:, :5]
    test[:, 2] += 6.0  # the anomaly is concentrated in feature 2 (and its dependent, 7)
    return train, test


def test_top_k_features_sorted_descending():
    err = np.array([[0.1, 0.9, 0.3, 0.5], [2.0, 0.0, 1.0, 3.0]])
    idx, vals = attribution.top_k_features(err, k=2)
    assert idx.tolist() == [[1, 3], [3, 0]]
    assert (np.diff(vals, axis=1) <= 0).all()


def test_ae_attribution_points_at_the_perturbed_feature():
    train, test = _data()
    det = build("ae", seed=0).fit(train)
    idx, _ = attribution.top_k_features(det.feature_errors(test), k=2)
    assert np.mean([2 in row for row in idx]) > 0.9


def test_deletion_is_far_more_effective_for_attributed_than_random_features():
    train, test = _data()
    det = build("ae", seed=0).fit(train)
    curve = attribution.deletion_curve(det, test, ks=(1, 2))
    assert curve["attributed"][0] > curve["random"][0] + 0.2
    assert all(0 <= v <= 1 for v in curve["attributed"] + curve["random"])


def test_attribution_clusters_align_with_categories():
    rng = np.random.default_rng(0)
    a = np.abs(rng.normal(size=(150, 8))) * np.r_[10, 10, np.ones(6)]
    b = np.abs(rng.normal(size=(150, 8))) * np.r_[np.ones(6), 10, 10]
    cats = np.array(["x"] * 150 + ["y"] * 150)
    assert attribution.attribution_cluster_nmi(np.vstack([a, b]), cats) > 0.8
    assert np.isnan(attribution.attribution_cluster_nmi(np.ones((3, 4)), np.array(["a", "b", "a"])))
