from __future__ import annotations

import numpy as np

from ids_anomaly.clustering.density import fit_hdbscan, noise_fraction
from ids_anomaly.clustering.kmeans_gmm import fit_gmm, fit_kmeans, gmm_anomaly_score


def _two_blobs(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=0, scale=0.2, size=(100, 3))
    b = rng.normal(loc=8, scale=0.2, size=(100, 3))
    return np.vstack([a, b]).astype(np.float32)


def test_kmeans_recovers_two_well_separated_blobs():
    X = _two_blobs()
    _, labels = fit_kmeans(X, n_clusters=2)
    # each true blob should map to a single cluster label (allowing for label permutation)
    first_blob_labels = set(labels[:100].tolist())
    second_blob_labels = set(labels[100:].tolist())
    assert len(first_blob_labels) == 1
    assert len(second_blob_labels) == 1
    assert first_blob_labels != second_blob_labels


def test_gmm_recovers_two_well_separated_blobs_and_scores_outliers_higher():
    X = _two_blobs()
    model, labels = fit_gmm(X, n_components=2)
    assert len(set(labels[:100].tolist())) == 1

    outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
    inlier_score = gmm_anomaly_score(model, X[:1]).mean()
    outlier_score = gmm_anomaly_score(model, outlier)[0]
    assert outlier_score > inlier_score


def test_hdbscan_flags_sparse_points_as_noise():
    # A *single* Gaussian blob has no density valley for HDBSCAN's stability-based selection to
    # split on, so it legitimately comes back all-noise (confirmed against sklearn's HDBSCAN too).
    # Two well-separated blobs give the density contrast HDBSCAN actually needs.
    rng = np.random.default_rng(0)
    blob_a = rng.normal(loc=0, scale=0.3, size=(150, 3))
    blob_b = rng.normal(loc=8, scale=0.3, size=(150, 3))
    sparse_outliers = rng.uniform(-10, 15, size=(20, 3))
    X = np.vstack([blob_a, blob_b, sparse_outliers]).astype(np.float32)
    _, labels = fit_hdbscan(X, min_cluster_size=20, min_samples=5)
    assert noise_fraction(labels) > 0
    # each dense blob's core should mostly land in a single non-noise cluster
    assert (labels[:150] != -1).mean() > 0.9
    assert (labels[150:300] != -1).mean() > 0.9
