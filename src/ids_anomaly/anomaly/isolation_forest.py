"""Isolation Forest: tree-ensemble anomaly detector, the strong non-parametric baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


def fit_isolation_forest(
    X: np.ndarray,
    n_estimators: int = 200,
    contamination: float | str = "auto",
    max_samples: int | float | str = "auto",
    random_state: int = 42,
    n_jobs: int = -1,
) -> IsolationForest:
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X)
    return model


def anomaly_score(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous (sklearn's score_samples is the opposite sign; flip it)."""
    return -model.score_samples(X)
