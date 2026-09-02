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
) -> tuple[UMAP, np.ndarray]:
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
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
