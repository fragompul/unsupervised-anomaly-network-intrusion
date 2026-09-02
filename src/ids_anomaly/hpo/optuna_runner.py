"""Optuna search spaces for every tunable method, plus a thin `run_study` wrapper.

Each objective is scored on a held-out validation slice carved out of the *training* split
(never the test split, which stays untouched until final evaluation) to avoid tuning
hyperparameters against the same data used for the headline benchmark numbers. Clustering
objectives use NMI against ``attack_category`` as the tuning target: it is only used to pick
hyperparameters, never to fit the clustering itself, mirroring how a security team would use a
small labeled validation set to select an operating configuration for an otherwise unsupervised
pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from ids_anomaly.anomaly.isolation_forest import anomaly_score as if_score
from ids_anomaly.anomaly.isolation_forest import fit_isolation_forest
from ids_anomaly.anomaly.one_class_svm import anomaly_score as ocsvm_score
from ids_anomaly.anomaly.one_class_svm import fit_one_class_svm
from ids_anomaly.clustering.density import fit_hdbscan
from ids_anomaly.clustering.kmeans_gmm import fit_gmm, fit_kmeans
from ids_anomaly.reduction.autoencoder import TrainConfig, reconstruction_error, train_autoencoder
from ids_anomaly.reduction.manifold import fit_umap

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class StudyResult:
    best_params: dict[str, Any]
    best_value: float
    study: optuna.Study


def run_study(
    objective: Callable[[optuna.Trial], float],
    n_trials: int,
    direction: str = "maximize",
    seed: int = 42,
) -> StudyResult:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return StudyResult(best_params=study.best_params, best_value=study.best_value, study=study)


# --- Anomaly detectors: maximize ROC-AUC on a labeled validation slice ------------------------


def objective_isolation_forest(
    trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray, y_val_is_attack: np.ndarray
) -> float:
    n_estimators = trial.suggest_int("n_estimators", 50, 400, step=50)
    max_samples = trial.suggest_float("max_samples", 0.1, 1.0)
    contamination = trial.suggest_float("contamination", 0.01, 0.3)
    model = fit_isolation_forest(
        X_train, n_estimators=n_estimators, contamination=contamination, max_samples=max_samples
    )
    return float(roc_auc_score(y_val_is_attack, if_score(model, X_val)))


def objective_one_class_svm(
    trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray, y_val_is_attack: np.ndarray
) -> float:
    nu = trial.suggest_float("nu", 0.01, 0.3, log=True)
    gamma = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
    model = fit_one_class_svm(X_train, nu=nu, gamma=gamma)
    return float(roc_auc_score(y_val_is_attack, ocsvm_score(model, X_val)))


def objective_autoencoder(
    trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray, y_val_is_attack: np.ndarray
) -> float:
    latent_dim = trial.suggest_categorical("latent_dim", [2, 3, 5, 8])
    hidden0 = trial.suggest_categorical("hidden0", [32, 64, 96])
    hidden1 = trial.suggest_categorical("hidden1", [8, 16, 24])
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    config = TrainConfig(
        latent_dim=latent_dim,
        hidden_dims=(hidden0, hidden1),
        lr=lr,
        max_epochs=40,
        patience=5,
    )
    result = train_autoencoder(X_train, config)
    scores = reconstruction_error(result.model, X_val)
    return float(roc_auc_score(y_val_is_attack, scores))


# --- Clustering: maximize NMI against attack_category (validation-only signal) ----------------


def objective_kmeans(
    trial: optuna.Trial, X: np.ndarray, attack_category_val: np.ndarray
) -> float:
    k = trial.suggest_int("n_clusters", 2, 20)
    _, labels = fit_kmeans(X, n_clusters=k)
    return float(normalized_mutual_info_score(attack_category_val, labels))


def objective_gmm(trial: optuna.Trial, X: np.ndarray, attack_category_val: np.ndarray) -> float:
    k = trial.suggest_int("n_components", 2, 20)
    cov_type = trial.suggest_categorical("covariance_type", ["full", "diag", "tied"])
    _, labels = fit_gmm(X, n_components=k, covariance_type=cov_type)
    return float(normalized_mutual_info_score(attack_category_val, labels))


def objective_hdbscan(
    trial: optuna.Trial, X: np.ndarray, attack_category_val: np.ndarray
) -> float:
    min_cluster_size = trial.suggest_int("min_cluster_size", 20, 500, log=True)
    _, labels = fit_hdbscan(X, min_cluster_size=min_cluster_size)
    return float(normalized_mutual_info_score(attack_category_val, labels))


def objective_umap(
    trial: optuna.Trial, X: np.ndarray, attack_category_val: np.ndarray
) -> float:
    """Score a UMAP embedding by how well a fixed downstream GMM recovers attack categories --
    a proxy for "does this embedding preserve the structure that matters," not just local
    neighborhood fidelity.
    """
    n_neighbors = trial.suggest_int("n_neighbors", 5, 50, log=True)
    min_dist = trial.suggest_float("min_dist", 0.0, 0.5)
    _, embedding = fit_umap(X, n_components=3, n_neighbors=n_neighbors, min_dist=min_dist)
    gmm = GaussianMixture(n_components=5, random_state=42)
    labels = gmm.fit_predict(embedding)
    return float(normalized_mutual_info_score(attack_category_val, labels))
