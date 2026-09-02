"""End-to-end experiment orchestration: load data -> tune -> fit every method -> evaluate ->
serialize everything the dashboard and docs/ figures need, to ``results/``.

Two training regimes are compared throughout, since this is a real methodological question for
network IDS in practice, not a formality:

* **fully unsupervised** -- fit on the raw (unlabeled, attack-contaminated) training split, the
  regime every clustering method and Isolation Forest actually runs in production, where you
  cannot assume incoming traffic is attack-free.
* **semi-supervised (normal-only)** -- fit One-Class SVM, Deep SVDD and the Autoencoder on a
  ``label == normal`` subset, the classical one-class setting those methods were designed for
  and which assumes a curated attack-free training window exists.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ids_anomaly.anomaly import isolation_forest as if_module
from ids_anomaly.anomaly import one_class_svm as ocsvm_module
from ids_anomaly.anomaly.deep_svdd import DeepSVDDConfig, train_deep_svdd
from ids_anomaly.anomaly.deep_svdd import anomaly_score as svdd_score
from ids_anomaly.clustering import density, kmeans_gmm
from ids_anomaly.data.download import download_raw
from ids_anomaly.data.preprocess import Dataset, load_datasets, normal_only
from ids_anomaly.evaluation.metrics import (
    anomaly_scoring_metrics,
    clustering_metrics,
    f1_at_quantile_threshold,
    per_category_detection_rate,
)
from ids_anomaly.reduction import linear, manifold
from ids_anomaly.reduction.autoencoder import (
    TrainConfig,
    encode,
    reconstruction_error,
    train_autoencoder,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
FLAG_QUANTILE = 0.20  # evaluate at "flag the top 20% most anomalous flows"


def _split_val(ds: Dataset, val_fraction: float = 0.15, seed: int = RANDOM_STATE) -> tuple[Dataset, Dataset]:
    """Carve a labeled validation slice out of the training split, for HPO and threshold
    selection -- kept strictly separate from the untouched test split used for final scoring.
    """
    rng = np.random.default_rng(seed)
    n = ds.X.shape[0]
    idx = rng.permutation(n)
    n_val = int(n * val_fraction)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train = Dataset(
        ds.X[train_idx], ds.is_attack[train_idx], ds.attack_category[train_idx],
        ds.label[train_idx], ds.feature_names,
    )
    val = Dataset(
        ds.X[val_idx], ds.is_attack[val_idx], ds.attack_category[val_idx],
        ds.label[val_idx], ds.feature_names,
    )
    return train, val


def run_full_pipeline(
    root_dir: Path,
    n_hpo_trials: int = 25,
    umap_components: int = 3,
    tsne_sample_size: int = 6000,
    hpo_n_jobs: int | None = None,
) -> dict[str, Any]:
    from ids_anomaly.hpo.optuna_runner import (
        objective_autoencoder,
        objective_gmm,
        objective_hdbscan,
        objective_isolation_forest,
        objective_kmeans,
        objective_one_class_svm,
        objective_umap,
        run_study,
    )

    # Every objective's inner work is native code that releases the GIL (BLAS, numba, PyTorch),
    # so threaded Optuna trials give real wall-clock speedup on a multi-core box. Leave a couple
    # of cores free for the OS rather than claiming every logical processor.
    if hpo_n_jobs is None:
        hpo_n_jobs = max(1, (os.cpu_count() or 4) - 2)
    logger.info("HPO parallelism: %d concurrent trials", hpo_n_jobs)
    timing: dict[str, float] = {}

    results_dir = root_dir / "results"
    models_dir = results_dir / "models"
    metrics_dir = results_dir / "metrics"
    figures_dir = results_dir / "figures"
    for d in (models_dir, metrics_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading raw data if needed...")
    download_raw(root_dir / "data" / "raw")

    logger.info("Loading + preprocessing NSL-KDD...")
    train_ds, test_ds, preprocessor = load_datasets(root_dir / "data" / "raw")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")

    fit_ds, val_ds = _split_val(train_ds)
    train_normal = normal_only(fit_ds)

    all_metrics: dict[str, Any] = {"dataset": {
        "n_train": int(train_ds.X.shape[0]),
        "n_test": int(test_ds.X.shape[0]),
        "n_features": int(train_ds.X.shape[1]),
        "train_attack_fraction": float(train_ds.is_attack.mean()),
        "test_attack_fraction": float(test_ds.is_attack.mean()),
    }}

    # ---------------------------------------------------------------- dimensionality reduction
    logger.info("PCA...")
    pca_model, _ = linear.fit_pca(fit_ds.X, n_components=umap_components)
    pca_test_embedding = pca_model.transform(test_ds.X)
    explained = linear.explained_variance_curve(fit_ds.X, max_components=20)
    all_metrics["pca"] = {"cumulative_explained_variance": explained.tolist()}

    logger.info("UMAP HPO (%d trials)...", n_hpo_trials)
    _t0 = time.perf_counter()
    # HPO only needs to rank hyperparameter configs against each other, not deliver a final
    # embedding, so score it on a bounded subsample rather than the full ~107k-row fit_ds. UMAP's
    # per-fit cost is well above linear in n once neighbor search and epoch-count scaling are
    # accounted for, and thread-based Optuna parallelism (n_jobs > 1) gives poor real speedup for
    # its numba-jitted internals (measured ~1.3x, not the ~14x the core count would suggest), so
    # the row count is what actually has to shrink for this stage to finish in a reasonable time.
    umap_hpo_size = min(20_000, fit_ds.X.shape[0])
    _hpo_idx = np.random.default_rng(RANDOM_STATE).choice(
        fit_ds.X.shape[0], size=umap_hpo_size, replace=False
    )
    umap_study = run_study(
        lambda trial: objective_umap(trial, fit_ds.X[_hpo_idx], fit_ds.attack_category[_hpo_idx]),
        n_trials=n_hpo_trials,
        n_jobs=hpo_n_jobs,
    )
    timing["umap_hpo"] = time.perf_counter() - _t0
    umap_model, _ = manifold.fit_umap(fit_ds.X, n_components=umap_components, **umap_study.best_params)
    umap_test_embedding = umap_model.transform(test_ds.X)
    all_metrics["umap"] = {"best_params": umap_study.best_params, "val_nmi": umap_study.best_value}
    joblib.dump(umap_model, models_dir / "umap.joblib")

    logger.info("t-SNE (visualization-only, subsampled)...")
    rng = np.random.default_rng(RANDOM_STATE)
    tsne_idx = rng.choice(test_ds.X.shape[0], size=min(tsne_sample_size, test_ds.X.shape[0]), replace=False)
    tsne_embedding = manifold.fit_tsne(test_ds.X[tsne_idx], n_components=umap_components)

    logger.info("Autoencoder HPO (%d trials)...", n_hpo_trials)
    _t0 = time.perf_counter()
    ae_study = run_study(
        lambda trial: objective_autoencoder(trial, train_normal.X, val_ds.X, val_ds.is_attack),
        n_trials=n_hpo_trials,
        n_jobs=hpo_n_jobs,
    )
    timing["autoencoder_hpo"] = time.perf_counter() - _t0
    ae_hidden = (ae_study.best_params["hidden0"], ae_study.best_params["hidden1"])
    ae_config = TrainConfig(
        latent_dim=ae_study.best_params["latent_dim"],
        hidden_dims=ae_hidden,
        lr=ae_study.best_params["lr"],
        max_epochs=150,
        patience=10,
    )
    ae_result = train_autoencoder(
        train_normal.X, ae_config, checkpoint_path=models_dir / "autoencoder_checkpoint.pt"
    )
    ae_test_embedding = encode(ae_result.model, test_ds.X)
    ae_test_recon_error = reconstruction_error(ae_result.model, test_ds.X)
    all_metrics["autoencoder"] = {
        "best_params": ae_study.best_params,
        "val_roc_auc": ae_study.best_value,
        "best_epoch": ae_result.best_epoch,
        "best_val_loss": ae_result.best_val_loss,
        "train_seconds": ae_result.train_seconds,
        "history": ae_result.history,
        **anomaly_scoring_metrics(test_ds.is_attack, ae_test_recon_error),
    }

    embeddings_test = {
        "pca": pca_test_embedding,
        "umap": umap_test_embedding,
        "autoencoder": ae_test_embedding,
    }

    # -------------------------------------------------------------------------------- clustering
    _t0 = time.perf_counter()
    clustering_results: dict[str, Any] = {}
    for embed_name, embed_test in embeddings_test.items():
        embed_train = {"pca": pca_model.transform(fit_ds.X), "umap": umap_model.transform(fit_ds.X),
                        "autoencoder": encode(ae_result.model, fit_ds.X)}[embed_name]

        logger.info("KMeans on %s embedding...", embed_name)
        km_study = run_study(
            lambda trial, e=embed_train: objective_kmeans(trial, e, fit_ds.attack_category),
            n_trials=n_hpo_trials,
            n_jobs=hpo_n_jobs,
        )
        _, km_test_labels = kmeans_gmm.fit_kmeans(embed_test, n_clusters=km_study.best_params["n_clusters"])

        logger.info("GMM on %s embedding...", embed_name)
        gmm_study = run_study(
            lambda trial, e=embed_train: objective_gmm(trial, e, fit_ds.attack_category),
            n_trials=n_hpo_trials,
            n_jobs=hpo_n_jobs,
        )
        gmm_model, gmm_test_labels = kmeans_gmm.fit_gmm(
            embed_test, n_components=gmm_study.best_params["n_components"],
            covariance_type=gmm_study.best_params["covariance_type"],
        )

        logger.info("HDBSCAN on %s embedding...", embed_name)
        hdb_study = run_study(
            lambda trial, e=embed_train: objective_hdbscan(trial, e, fit_ds.attack_category),
            n_trials=max(10, n_hpo_trials // 2),
            n_jobs=hpo_n_jobs,
        )
        _, hdb_test_labels = density.fit_hdbscan(embed_test, min_cluster_size=hdb_study.best_params["min_cluster_size"])

        clustering_results[embed_name] = {
            "kmeans": {"best_params": km_study.best_params,
                       **clustering_metrics(embed_test, km_test_labels, test_ds.attack_category)},
            "gmm": {"best_params": gmm_study.best_params,
                    **clustering_metrics(embed_test, gmm_test_labels, test_ds.attack_category),
                    **anomaly_scoring_metrics(test_ds.is_attack, kmeans_gmm.gmm_anomaly_score(gmm_model, embed_test))},
            "hdbscan": {"best_params": hdb_study.best_params,
                        "noise_fraction": density.noise_fraction(hdb_test_labels),
                        **clustering_metrics(embed_test, hdb_test_labels, test_ds.attack_category)},
        }
    all_metrics["clustering"] = clustering_results
    timing["clustering_hpo_all_embeddings"] = time.perf_counter() - _t0

    # --------------------------------------------------------------------------- anomaly detection
    logger.info("Isolation Forest HPO (fully unsupervised)...")
    _t0 = time.perf_counter()
    if_study = run_study(
        lambda trial: objective_isolation_forest(trial, fit_ds.X, val_ds.X, val_ds.is_attack),
        n_trials=n_hpo_trials,
        n_jobs=hpo_n_jobs,
    )
    timing["isolation_forest_hpo"] = time.perf_counter() - _t0
    if_model = if_module.fit_isolation_forest(fit_ds.X, **if_study.best_params)
    if_test_scores = if_module.anomaly_score(if_model, test_ds.X)
    joblib.dump(if_model, models_dir / "isolation_forest.joblib")

    logger.info("One-Class SVM HPO (semi-supervised, normal-only)...")
    _t0 = time.perf_counter()
    ocsvm_study = run_study(
        lambda trial: objective_one_class_svm(trial, train_normal.X, val_ds.X, val_ds.is_attack),
        n_trials=n_hpo_trials,
        n_jobs=hpo_n_jobs,
    )
    timing["one_class_svm_hpo"] = time.perf_counter() - _t0
    ocsvm_model = ocsvm_module.fit_one_class_svm(train_normal.X, **ocsvm_study.best_params)
    ocsvm_test_scores = ocsvm_module.anomaly_score(ocsvm_model, test_ds.X)
    joblib.dump(ocsvm_model, models_dir / "one_class_svm.joblib")

    logger.info("Deep SVDD (semi-supervised, normal-only)...")
    _t0 = time.perf_counter()
    svdd_result = train_deep_svdd(train_normal.X, DeepSVDDConfig())
    timing["deep_svdd_train"] = time.perf_counter() - _t0
    svdd_test_scores = svdd_score(svdd_result, test_ds.X)

    anomaly_results = {
        "isolation_forest": {"regime": "unsupervised", "best_params": if_study.best_params,
                              **anomaly_scoring_metrics(test_ds.is_attack, if_test_scores),
                              **f1_at_quantile_threshold(test_ds.is_attack, if_test_scores, FLAG_QUANTILE)},
        "one_class_svm": {"regime": "semi-supervised", "best_params": ocsvm_study.best_params,
                           **anomaly_scoring_metrics(test_ds.is_attack, ocsvm_test_scores),
                           **f1_at_quantile_threshold(test_ds.is_attack, ocsvm_test_scores, FLAG_QUANTILE)},
        "deep_svdd": {"regime": "semi-supervised",
                      "train_seconds": svdd_result.train_seconds,
                      **anomaly_scoring_metrics(test_ds.is_attack, svdd_test_scores),
                      **f1_at_quantile_threshold(test_ds.is_attack, svdd_test_scores, FLAG_QUANTILE)},
        "autoencoder_reconstruction": {"regime": "semi-supervised",
                                        **anomaly_scoring_metrics(test_ds.is_attack, ae_test_recon_error),
                                        **f1_at_quantile_threshold(test_ds.is_attack, ae_test_recon_error, FLAG_QUANTILE)},
    }
    all_metrics["anomaly_detection"] = anomaly_results

    # ------------------------------------------------------------------------------- error analysis
    score_map = {
        "isolation_forest": if_test_scores,
        "one_class_svm": ocsvm_test_scores,
        "deep_svdd": svdd_test_scores,
        "autoencoder_reconstruction": ae_test_recon_error,
    }
    per_category = {
        name: per_category_detection_rate(test_ds.attack_category, scores, FLAG_QUANTILE).to_dict(orient="records")
        for name, scores in score_map.items()
    }
    all_metrics["per_category_detection_rate"] = per_category
    all_metrics["timing_seconds"] = timing
    all_metrics["hpo_n_jobs"] = hpo_n_jobs

    # ------------------------------------------------------------------------------------ persist
    np.savez_compressed(
        results_dir / "embeddings_test.npz",
        pca=pca_test_embedding, umap=umap_test_embedding, autoencoder=ae_test_embedding,
        tsne=tsne_embedding, tsne_idx=tsne_idx,
        is_attack=test_ds.is_attack, attack_category=test_ds.attack_category, label=test_ds.label,
        if_score=if_test_scores, ocsvm_score=ocsvm_test_scores, svdd_score=svdd_test_scores,
        ae_recon_error=ae_test_recon_error,
    )
    with open(metrics_dir / "results.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    logger.info("Pipeline complete. Results written to %s", results_dir)
    return all_metrics
