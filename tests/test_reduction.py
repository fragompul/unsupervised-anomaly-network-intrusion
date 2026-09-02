from __future__ import annotations

import numpy as np

from ids_anomaly.reduction.autoencoder import (
    Autoencoder,
    TrainConfig,
    encode,
    get_device,
    reconstruction_error,
    train_autoencoder,
)
from ids_anomaly.reduction.linear import explained_variance_curve, fit_pca
from ids_anomaly.reduction.manifold import fit_tsne, fit_umap


def _blobs(n=300, d=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_pca_embedding_shape_and_orthogonality():
    X = _blobs()
    model, embedding = fit_pca(X, n_components=3)
    assert embedding.shape == (300, 3)
    # PCA components must be orthonormal
    gram = model.components_ @ model.components_.T
    np.testing.assert_allclose(gram, np.eye(3), atol=1e-6)


def test_explained_variance_curve_is_monotonic_and_bounded():
    X = _blobs()
    curve = explained_variance_curve(X, max_components=8)
    assert (np.diff(curve) >= -1e-9).all()
    assert curve[-1] <= 1.0 + 1e-9


def test_umap_embedding_shape_and_transform_on_new_data():
    X = _blobs(n=200)
    reducer, embedding = fit_umap(X, n_components=3, n_neighbors=10)
    assert embedding.shape == (200, 3)
    new_embedding = reducer.transform(_blobs(n=20, seed=99))
    assert new_embedding.shape == (20, 3)


def test_tsne_embedding_shape():
    X = _blobs(n=100)
    embedding = fit_tsne(X, n_components=2, perplexity=15)
    assert embedding.shape == (100, 2)


def test_autoencoder_reduces_reconstruction_error_with_training():
    X = _blobs(n=200, d=12)
    untrained = Autoencoder(n_features=12, latent_dim=3, hidden_dims=(16, 8))
    err_before = reconstruction_error(untrained.to(get_device()), X).mean()

    result = train_autoencoder(X, TrainConfig(max_epochs=30, patience=10, latent_dim=3))
    err_after = reconstruction_error(result.model, X).mean()

    assert err_after < err_before
    assert result.best_epoch >= 0


def test_autoencoder_encode_output_dim_matches_latent_dim():
    X = _blobs(n=100, d=8)
    result = train_autoencoder(X, TrainConfig(max_epochs=5, latent_dim=4, hidden_dims=(16,)))
    z = encode(result.model, X)
    assert z.shape == (100, 4)
