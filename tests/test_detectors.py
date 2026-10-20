from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ids_anomaly.detectors import REGISTRY, TITLES, build
from ids_anomaly.detectors.base import Detector, batched_apply, fit_torch


@pytest.fixture(scope="module")
def toy():
    """Correlated 'normal' cloud plus a clearly shifted anomaly cluster."""
    rng = np.random.default_rng(0)
    d = 12
    base = rng.normal(size=(1200, d)).astype("float32")
    base[:, 6:] += base[:, :6]
    normal_test = rng.normal(size=(300, d)).astype("float32")
    normal_test[:, 6:] += normal_test[:, :6]
    anomalies = (rng.normal(size=(60, d)) * 1.5 + 2.0).astype("float32")
    X_test = np.vstack([normal_test, anomalies])
    y = np.r_[np.zeros(300), np.ones(60)]
    return base, X_test, y


def test_registry_and_titles_are_consistent():
    assert len(REGISTRY) == 15
    assert set(REGISTRY) == set(TITLES)
    for name, cls in REGISTRY.items():
        assert cls.name == name and cls.family


def test_build_rejects_unknown_name():
    with pytest.raises(KeyError):
        build("does-not-exist")


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_detector_scores_are_finite_and_rank_anomalies_higher(name, toy):
    X_train, X_test, y = toy
    det = build(name, seed=0)
    assert isinstance(det, Detector)
    scores = det.fit(X_train).score(X_test)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()
    # contrastive transformation learning targets semantic, not mean-shift, anomalies
    floor = 0.7 if name == "neutral" else 0.8
    assert roc_auc_score(y, scores) > floor, f"{name} fails to separate an obvious shift"


@pytest.mark.parametrize("name", ["iforest", "ae", "vae", "flow", "neutral", "mtt"])
def test_detector_is_deterministic_given_seed(name, toy):
    X_train, X_test, _ = toy
    a = build(name, seed=3).fit(X_train).score(X_test)
    b = build(name, seed=3).fit(X_train).score(X_test)
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)


def test_different_seeds_change_stochastic_models(toy):
    X_train, X_test, _ = toy
    a = build("ae", seed=1).fit(X_train).score(X_test)
    b = build("ae", seed=2).fit(X_train).score(X_test)
    assert not np.allclose(a, b)


def test_autoencoder_feature_errors_sum_to_score(toy):
    X_train, X_test, _ = toy
    det = build("ae", seed=0).fit(X_train)
    errs = det.feature_errors(X_test)
    assert errs.shape == X_test.shape
    np.testing.assert_allclose(errs.mean(axis=1), det.score(X_test), rtol=1e-4)


def test_masked_transformer_feature_errors_sum_to_score(toy):
    X_train, X_test, _ = toy
    det = build("mtt", seed=0).fit(X_train)
    errs = det.feature_errors(X_test)
    assert errs.shape == X_test.shape
    np.testing.assert_allclose(errs.mean(axis=1), det.score(X_test), rtol=1e-4)


def test_flow_density_is_higher_inside_the_training_distribution(toy):
    X_train, X_test, y = toy
    det = build("flow", seed=0).fit(X_train)
    scores = det.score(X_test)  # negative log-likelihood
    assert scores[y == 1].mean() > scores[y == 0].mean()


def test_fit_score_seconds_reports_timings(toy):
    X_train, X_test, _ = toy
    scores, fit_s, score_s = build("pca").fit_score_seconds(X_train, X_test)
    assert scores.shape == (len(X_test),) and fit_s >= 0 and score_s >= 0


def test_fit_torch_early_stops_when_validation_stalls_and_restores_weights():
    import torch
    from torch import nn

    X = np.random.default_rng(0).normal(size=(500, 4)).astype("float32")
    model = nn.Linear(4, 4)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # constant validation loss -> never improves on epoch 0 -> stops after `patience` stale epochs
    hist = fit_torch(model, lambda b: (model.weight * 0).sum() + 1.0, X, seed=0, epochs=200, patience=3)
    assert len(hist) == 4
    assert not model.training
    for k, v in model.state_dict().items():
        assert torch.allclose(v, before[k], atol=1e-6)  # AdamW decay is ~1e-8 here


def test_batched_apply_handles_empty_input():
    assert batched_apply(lambda b: b.sum(dim=1), np.empty((0, 3), dtype=np.float32)).size == 0
