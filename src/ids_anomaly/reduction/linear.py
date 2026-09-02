"""PCA baseline: the linear, convex reference point every other embedding is judged against."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(X: np.ndarray, n_components: int, random_state: int = 42) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components, random_state=random_state)
    embedding = pca.fit_transform(X)
    return pca, embedding


def explained_variance_curve(X: np.ndarray, max_components: int, random_state: int = 42) -> np.ndarray:
    """Cumulative explained variance ratio, used to justify the chosen n_components in docs/."""
    pca = PCA(n_components=max_components, random_state=random_state)
    pca.fit(X)
    return np.cumsum(pca.explained_variance_ratio_)
