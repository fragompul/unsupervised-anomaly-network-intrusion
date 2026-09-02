from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from data import Artifacts
from projection import normalize_embedding, rotate_project


def test_normalize_embedding_gives_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    arr = rng.normal(loc=5, scale=3, size=(200, 3))
    normed = normalize_embedding(arr)
    np.testing.assert_allclose(normed.mean(axis=0), 0, atol=1e-6)
    np.testing.assert_allclose(normed.std(axis=0), 1, atol=1e-6)


def test_normalize_embedding_ignores_nan_rows_when_scaling():
    arr = np.array([[1.0, 1.0, 1.0], [np.nan, np.nan, np.nan], [3.0, 3.0, 3.0]])
    normed = normalize_embedding(arr)
    assert np.isnan(normed[1]).all()
    assert not np.isnan(normed[0]).any()


def test_rotate_project_at_zero_rotation_is_identity_on_xy():
    x3 = np.array([1.0, 2.0])
    y3 = np.array([3.0, 4.0])
    z3 = np.array([0.0, 0.0])
    x, y, size, alpha = rotate_project(x3, y3, z3, azimuth=0.0, elevation=0.0)
    np.testing.assert_allclose(x, x3)
    np.testing.assert_allclose(y, y3)
    assert (size > 0).all()
    assert (alpha >= 0).all()


def test_rotate_project_marks_nan_points_fully_transparent():
    x3 = np.array([np.nan, 1.0])
    y3 = np.array([np.nan, 1.0])
    z3 = np.array([np.nan, 1.0])
    _, _, _, alpha = rotate_project(x3, y3, z3, azimuth=0.5, elevation=0.5)
    assert alpha[0] == 0.0
    assert alpha[1] > 0.0


@pytest.fixture
def synthetic_results_dir(tmp_path: Path, monkeypatch) -> Path:
    results_dir = tmp_path / "results"
    (results_dir / "metrics").mkdir(parents=True)
    n = 50
    rng = np.random.default_rng(0)
    np.savez_compressed(
        results_dir / "embeddings_test.npz",
        pca=rng.normal(size=(n, 3)),
        umap=rng.normal(size=(n, 3)),
        autoencoder=rng.normal(size=(n, 3)),
        tsne=rng.normal(size=(10, 3)),
        tsne_idx=np.arange(10),
        is_attack=rng.integers(0, 2, size=n).astype(bool),
        attack_category=rng.choice(["normal", "dos"], size=n),
        label=rng.choice(["normal", "neptune"], size=n),
        if_score=rng.random(n),
        ocsvm_score=rng.random(n),
        svdd_score=rng.random(n),
        ae_recon_error=rng.random(n),
    )
    with open(results_dir / "metrics" / "results.json", "w") as f:
        json.dump({"dataset": {"n_test": n}}, f)

    import data as data_module

    monkeypatch.setattr(data_module, "RESULTS_DIR", results_dir)
    return results_dir


def test_artifacts_loads_and_scatters_tsne_into_full_length(synthetic_results_dir):
    artifacts = Artifacts()
    assert artifacts.n_test == 50
    assert artifacts.embeddings["tsne"].shape == (50, 3)
    # only the first 10 rows (tsne_idx) should be non-NaN
    assert not np.isnan(artifacts.embeddings["tsne"][:10]).any()
    assert np.isnan(artifacts.embeddings["tsne"][10:]).all()


def test_artifacts_base_columns_includes_all_score_fields(synthetic_results_dir):
    artifacts = Artifacts()
    cols = artifacts.base_columns()
    for key in ("if_score", "ocsvm_score", "svdd_score", "ae_recon_error"):
        assert key in cols
        assert len(cols[key]) == 50


def test_artifacts_raises_clear_error_when_missing(tmp_path, monkeypatch):
    import data as data_module

    monkeypatch.setattr(data_module, "RESULTS_DIR", tmp_path / "nonexistent")
    with pytest.raises(FileNotFoundError, match="ids_anomaly.cli"):
        Artifacts()
