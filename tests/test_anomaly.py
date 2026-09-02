from __future__ import annotations

import numpy as np

from ids_anomaly.anomaly.deep_svdd import DeepSVDDConfig, train_deep_svdd
from ids_anomaly.anomaly.deep_svdd import anomaly_score as svdd_score
from ids_anomaly.anomaly.isolation_forest import anomaly_score as if_score
from ids_anomaly.anomaly.isolation_forest import fit_isolation_forest
from ids_anomaly.anomaly.one_class_svm import anomaly_score as ocsvm_score
from ids_anomaly.anomaly.one_class_svm import fit_one_class_svm


def _normal_with_outliers(seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(loc=0, scale=1.0, size=(300, 6)).astype(np.float32)
    outliers = rng.normal(loc=12, scale=1.0, size=(30, 6)).astype(np.float32)
    return normal, outliers


def test_isolation_forest_scores_outliers_higher_than_normal():
    normal, outliers = _normal_with_outliers()
    model = fit_isolation_forest(normal, n_estimators=100)
    normal_scores = if_score(model, normal)
    outlier_scores = if_score(model, outliers)
    assert outlier_scores.mean() > normal_scores.mean()


def test_one_class_svm_scores_outliers_higher_than_normal():
    normal, outliers = _normal_with_outliers()
    model = fit_one_class_svm(normal, nu=0.05, max_train_samples=300)
    normal_scores = ocsvm_score(model, normal)
    outlier_scores = ocsvm_score(model, outliers)
    assert outlier_scores.mean() > normal_scores.mean()


def test_one_class_svm_subsamples_when_over_max_train_samples():
    normal, _ = _normal_with_outliers()
    # Should not raise even though normal has 300 rows > max_train_samples
    model = fit_one_class_svm(normal, max_train_samples=50)
    assert model.support_vectors_.shape[0] <= 50


def test_deep_svdd_scores_outliers_higher_than_normal():
    normal, outliers = _normal_with_outliers()
    result = train_deep_svdd(
        normal, DeepSVDDConfig(pretrain_epochs=5, train_epochs=15, latent_dim=4, hidden_dims=(16, 8))
    )
    normal_scores = svdd_score(result, normal)
    outlier_scores = svdd_score(result, outliers)
    assert outlier_scores.mean() > normal_scores.mean()


def test_deep_svdd_center_is_kept_away_from_origin():
    normal, _ = _normal_with_outliers()
    result = train_deep_svdd(normal, DeepSVDDConfig(pretrain_epochs=3, train_epochs=3))
    # a center collapsed at the origin would make the bias-free encoder's trivial
    # zero-function solution indistinguishable from a genuinely learned one
    assert result.center.abs().sum().item() > 0
