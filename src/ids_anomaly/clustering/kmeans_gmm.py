"""Hard-partition (KMeans) and probabilistic (Gaussian Mixture) clustering baselines."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int = 42) -> tuple[KMeans, np.ndarray]:
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return model, labels


def fit_gmm(
    X: np.ndarray, n_components: int, covariance_type: str = "full", random_state: int = 42
) -> tuple[GaussianMixture, np.ndarray]:
    model = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type, random_state=random_state
    )
    labels = model.fit_predict(X)
    return model, labels


def gmm_anomaly_score(model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """Negative log-likelihood under the fitted mixture -- higher means more anomalous."""
    return -model.score_samples(X)
