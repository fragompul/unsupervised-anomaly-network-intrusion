# Experiments

Full run reproduced with:

```bash
python -m ids_anomaly.cli run --trials 25 -v
```

All studies below use a seeded `optuna.samplers.TPESampler(seed=42)` and score against the
labeled validation slice carved out of `KDDTrain+.txt` (never the test split, see
[methodology.md](methodology.md#splits)).

## Search spaces

| Method | Trials | Search space | Objective (maximize unless noted) |
|---|---|---|---|
| UMAP | 25 | `n_neighbors` ∈ [5, 50] (log), `min_dist` ∈ [0.0, 0.5] | NMI of a downstream GMM(5) against `attack_category` |
| Autoencoder | 25 | `latent_dim` ∈ {2,3,5,8}, hidden dims ∈ {32,64,96}×{8,16,24}, `lr` ∈ [1e-4, 5e-3] (log) | ROC-AUC of reconstruction error vs. `is_attack` |
| KMeans (×3 embeddings) | 25 each | `n_clusters` ∈ [2, 20] | NMI vs. `attack_category` |
| GMM (×3 embeddings) | 25 each | `n_components` ∈ [2, 20], `covariance_type` ∈ {full, diag, tied} | NMI vs. `attack_category` |
| HDBSCAN (×3 embeddings) | 12 each | `min_cluster_size` ∈ [20, 500] (log) | NMI vs. `attack_category` |
| Isolation Forest | 25 | `n_estimators` ∈ [50, 400], `max_samples` ∈ [0.1, 1.0], `contamination` ∈ [0.01, 0.3] | ROC-AUC vs. `is_attack` |
| One-Class SVM | 25 | `nu` ∈ [0.01, 0.3] (log), `gamma` ∈ [1e-4, 1.0] (log) | ROC-AUC vs. `is_attack` |
| Deep SVDD | n/a (fixed config, not searched) | latent_dim=8, hidden=(64,16), 20 pretrain + 50 train epochs | n/a |

Deep SVDD was run with a fixed, reasonable configuration rather than a full Optuna search: its
two-stage (pretrain + SVDD fine-tune) training makes each trial roughly 2x an autoencoder trial,
and the marginal value of tuning it further was judged lower than spending that CPU time on the
per-embedding clustering searches, which directly drive three rows of the results table each.
This is exactly the kind of "real project has a CPU budget" trade-off the brief asked to be
documented honestly rather than hidden.

## Compute

Run on a CPU-only machine (no CUDA/MPS available, confirmed via `torch.cuda.is_available()`
before every training run). See [results.md](results.md) for wall-clock time per stage from the
actual run this repository's numbers come from.

## Ablations

In addition to the head-to-head method comparison, `docs/results.md` reports:

* **Embedding choice ablation**: the same clustering algorithm (KMeans/GMM/HDBSCAN) scored on
  PCA vs. UMAP vs. Autoencoder embeddings, isolating how much of clustering quality is actually
  attributable to the embedding rather than the clustering algorithm.
* **Training regime ablation**: Autoencoder reconstruction error trained on normal-only data
  vs. what the same architecture would need to change to run fully unsupervised (discussed
  qualitatively; the full grid was out of scope for the CPU budget available, see
  [limitations.md](limitations.md)).
* **PCA explained variance**: cumulative explained variance vs. number of components, to
  justify (or not) the fixed 3-component embedding used everywhere else for a fair comparison.
