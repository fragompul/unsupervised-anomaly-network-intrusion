"""Fast end-to-end smoke test on a small subsample -- catches API mismatches before committing
to the full multi-hour pipeline run. Not part of pytest (too slow/flaky for CI); run manually.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from ids_anomaly.anomaly.deep_svdd import DeepSVDDConfig, anomaly_score as svdd_score, train_deep_svdd
from ids_anomaly.anomaly.isolation_forest import anomaly_score as if_score, fit_isolation_forest
from ids_anomaly.anomaly.one_class_svm import anomaly_score as ocsvm_score, fit_one_class_svm
from ids_anomaly.clustering.density import fit_hdbscan
from ids_anomaly.clustering.kmeans_gmm import fit_gmm, fit_kmeans, gmm_anomaly_score
from ids_anomaly.data.download import download_raw
from ids_anomaly.data.preprocess import load_datasets, normal_only
from ids_anomaly.evaluation.metrics import anomaly_scoring_metrics, clustering_metrics
from ids_anomaly.hpo.optuna_runner import objective_isolation_forest, run_study
from ids_anomaly.reduction.autoencoder import TrainConfig, reconstruction_error, train_autoencoder
from ids_anomaly.reduction.linear import fit_pca
from ids_anomaly.reduction.manifold import fit_tsne, fit_umap

ROOT = Path(__file__).resolve().parents[1]

t0 = time.perf_counter()
download_raw(ROOT / "data" / "raw")
train, test, _ = load_datasets(ROOT / "data" / "raw")

rng = np.random.default_rng(0)
sub_idx = rng.choice(train.X.shape[0], size=3000, replace=False)
X = train.X[sub_idx]
y = train.is_attack[sub_idx]
cat = train.attack_category[sub_idx]
normal = normal_only(train)
X_normal = normal.X[: min(2000, normal.X.shape[0])]

print("[1/9] PCA")
_, pca_emb = fit_pca(X, n_components=3)
assert pca_emb.shape == (3000, 3)

print("[2/9] UMAP")
_, umap_emb = fit_umap(X, n_components=3, n_neighbors=10)
assert umap_emb.shape == (3000, 3)

print("[3/9] t-SNE")
tsne_emb = fit_tsne(X[:500], n_components=3, perplexity=20)
assert tsne_emb.shape == (500, 3)

print("[4/9] Autoencoder")
ae_result = train_autoencoder(X_normal, TrainConfig(max_epochs=3, patience=2))
err = reconstruction_error(ae_result.model, X)
print("   metrics:", anomaly_scoring_metrics(y, err))

print("[5/9] KMeans + GMM")
_, km_labels = fit_kmeans(pca_emb, n_clusters=5)
gmm_model, gmm_labels = fit_gmm(pca_emb, n_components=5)
print("   kmeans:", clustering_metrics(pca_emb, km_labels, cat))
print("   gmm:", clustering_metrics(pca_emb, gmm_labels, cat))
gscore = gmm_anomaly_score(gmm_model, pca_emb)
assert gscore.shape == (3000,)

print("[6/9] HDBSCAN")
_, hdb_labels = fit_hdbscan(pca_emb, min_cluster_size=30)
print("   hdbscan:", clustering_metrics(pca_emb, hdb_labels, cat))

print("[7/9] Isolation Forest")
if_model = fit_isolation_forest(X, n_estimators=50)
print("   metrics:", anomaly_scoring_metrics(y, if_score(if_model, X)))

print("[8/9] One-Class SVM")
ocsvm_model = fit_one_class_svm(X_normal, max_train_samples=1000)
print("   metrics:", anomaly_scoring_metrics(y, ocsvm_score(ocsvm_model, X)))

print("[9/9] Deep SVDD")
svdd_result = train_deep_svdd(X_normal, DeepSVDDConfig(pretrain_epochs=3, train_epochs=3))
print("   metrics:", anomaly_scoring_metrics(y, svdd_score(svdd_result, X)))

print("Optuna smoke (3 trials, isolation forest)")
val_idx = rng.choice(test.X.shape[0], size=500, replace=False)
study = run_study(
    lambda trial: objective_isolation_forest(trial, X, test.X[val_idx], test.is_attack[val_idx]),
    n_trials=3,
)
print("   best:", study.best_params, study.best_value)

print(f"\nAll smoke tests passed in {time.perf_counter() - t0:.1f}s")
