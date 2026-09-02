"""Non-linear manifold embeddings: UMAP (scalable, has a transform for held-out data)
and t-SNE (visualization-only reference, refit per split since it has no out-of-sample transform).
"""

from __future__ import annotations

import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP


def fit_umap(
    X: np.ndarray,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    init: str = "spectral",
) -> tuple[UMAP, np.ndarray]:
    """``init="spectral"`` (UMAP's own default) gives the best embedding but depends on scipy's
    ARPACK eigensolver, which routinely fails to converge on a k-NN graph built with a small
    ``n_neighbors`` and silently falls back to LOBPCG, an order of magnitude slower. HPO sweeps
    a wide ``n_neighbors`` range specifically to find that failure edge, so every low-n_neighbors
    trial pays the slow-fallback cost. Pass ``init="random"`` from the HPO objective (never from
    the final best-params fit, where embedding quality matters and it is only fit once) to skip
    spectral initialization entirely during the sweep.
    """
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
        init=init,
        n_jobs=1,  # deterministic given random_state; UMAP disables parallelism otherwise
    )
    embedding = reducer.fit_transform(X)
    return reducer, embedding


def fit_tsne(
    X: np.ndarray,
    n_components: int = 3,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> np.ndarray:
    """t-SNE has no ``.transform`` for new data -- it is fit once per array it is shown.

    Used only for the exploratory/visualization comparison in docs/ and the dashboard's
    static reference view, on a bounded subsample (t-SNE is O(n^2)-ish and impractical
    on the full ~125k-row training split on CPU).
    """
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)
