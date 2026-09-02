"""Density-based clustering: HDBSCAN finds an arbitrary number of variable-density clusters
and natively labels sparse points as noise (-1), which doubles as an anomaly signal.
"""

from __future__ import annotations

import numpy as np
from hdbscan import HDBSCAN


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int | None = None,
) -> tuple[HDBSCAN, np.ndarray]:
    model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
    )
    labels = model.fit_predict(X)
    return model, labels


def noise_fraction(labels: np.ndarray) -> float:
    return float(np.mean(labels == -1))
