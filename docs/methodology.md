# Methodology

## Preprocessing

* **Categorical features** (`protocol_type`, `service`, `flag`) are one-hot encoded with a
  vocabulary fit on the training split only (`handle_unknown="ignore"` so any unseen test
  category maps to an all-zero row rather than raising). 41 raw features expand to 122 columns.
* **Heavy-tailed count/byte columns** (`duration`, `src_bytes`, `dst_bytes`, `count`,
  `dst_host_count`, ...) get a `log1p` transform before standardization. These columns span
  several orders of magnitude in raw form and would otherwise dominate every Euclidean-distance-
  based method (KMeans, UMAP, RBF kernels) regardless of whether they carry more *signal* than
  the bounded rate features.
* **Bounded rate features** (already in [0, 1], e.g. `serror_rate`) and **binary flags**
  (`logged_in`, `root_shell`, ...) are standardized without the log transform.
* The preprocessor (a single `sklearn.compose.ColumnTransformer`) is fit once on training data
  and reused, unmodified, to transform the test split -- see
  `test_preprocessor_fit_only_on_train_but_transforms_test_consistently` in
  `tests/test_preprocess.py` for the regression test that pins this down.
* **Labels never enter the feature matrix or influence any fitted transform.** `is_attack` and
  `attack_category` are parsed alongside `X` but held in separate arrays, used only inside
  `evaluation/` and `hpo/` (see below).

## Where labels are, and are not, used

This is the methodological crux of the project and worth stating plainly:

| Used for | Labels used? |
|---|---|
| Fitting reduction / clustering / anomaly models | **No** |
| Preprocessing (scaling, encoding) | **No** |
| Optuna hyperparameter search (validation slice, carved from train) | **Yes** -- by design |
| Final metrics on the test split | **Yes** -- that is what evaluation means |

The HPO validation slice uses labels to *select* a hyperparameter configuration (e.g. "which
`min_cluster_size` gives clusters that best align with attack categories"), mirroring how a
security team would use a small labeled sample to tune an otherwise unsupervised pipeline before
deploying it -- a common and realistic middle ground between "fully blind" and "fully
supervised." It is a deliberate deviation from pure unsupervised learning, documented here rather
than left implicit.

## Two training regimes

See [architecture.md](architecture.md#two-training-regimes-compared-explicitly) for the full
comparison table. In short: KMeans / GMM / HDBSCAN / Isolation Forest are fit on the full
(unlabeled, attack-contaminated) training split; One-Class SVM / Deep SVDD / the Autoencoder are
fit on a normal-only subset, in the classical one-class regime those methods were designed for.

## Splits

* **Train** (`KDDTrain+.txt`, ~126k rows) is further split 85/15 into `fit_ds` (model fitting)
  and a labeled `val_ds` (HPO objective + threshold selection), seeded for reproducibility.
* **Test** (`KDDTest+.txt`, ~22.5k rows) is untouched until final scoring -- no model, scaler, or
  hyperparameter search ever sees it before the metrics in `results/metrics/results.json` are
  computed.
* NSL-KDD's test split intentionally contains attack types absent from train, which is a feature
  for this evaluation, not noise: it directly tests whether "model normal, flag deviation"
  generalizes to attacks a method has structurally never been shown.

## Reduction methods

* **PCA** -- linear baseline; `docs/results.md` reports the cumulative explained-variance curve
  used to sanity-check the chosen 3-component embedding against how much variance a linear
  projection can realistically capture.
* **UMAP** -- non-linear, has a `.transform()` for held-out data (unlike t-SNE), so it is the
  non-linear embedding actually used downstream for test-set clustering. `n_neighbors` and
  `min_dist` are tuned via Optuna against a downstream-clustering NMI proxy (see
  `hpo/optuna_runner.objective_umap`).
* **t-SNE** -- included for the visualization comparison only, fit on a bounded subsample (its
  cost scales badly and it has no out-of-sample transform, so it cannot participate in the
  train/test-consistent pipeline the other methods follow).
* **Autoencoder** -- a compact MLP (two hidden layers, latent dim tuned by Optuna) trained on
  normal-only data with early stopping on a held-out validation loss and checkpointing (best
  validation weights persisted to disk on every improvement, so a killed run never loses more
  than the epochs since the last improvement). Reconstruction error doubles as an anomaly score.

## Clustering methods

KMeans (hard partitions), Gaussian Mixture Models (soft/probabilistic partitions, with
per-sample negative log-likelihood as a bonus anomaly score), and HDBSCAN (variable-density
clusters with native noise labeling). All three are tuned and evaluated **per embedding** (PCA /
UMAP / Autoencoder), so the results table in `docs/results.md` answers "which reduction +
clustering combination works best" rather than conflating the two choices.

One HDBSCAN property worth flagging up front (documented in code and tests, not just here): a
single, unimodal cluster with no density valley separating it from anything else can legitimately
come back **all-noise** -- this was confirmed against scikit-learn's own HDBSCAN implementation
while writing `tests/test_clustering.py`, and is expected algorithmic behavior (HDBSCAN's
stability-based cluster selection needs density *contrast* to split on), not a bug.

## Anomaly detection methods

* **Isolation Forest** -- ensemble of random partitioning trees; anomalies isolate in fewer
  splits. Fit fully unsupervised on the mixed training split.
* **One-Class SVM** -- RBF-kernel boundary around the dense region of normal-only training data.
  Subsampled to a bounded training size (its kernel computation is O(n^2)-O(n^3), impractical on
  CPU at the full ~113k-row normal-only training set).
* **Deep SVDD** (Ruff et al., 2018) -- learns a feature map pulling normal data into a
  minimum-volume hypersphere; distance from the learned center is the anomaly score. Implemented
  with the paper's two safeguards against the trivial "collapse to a constant" solution:
  bias-free linear layers (so the network cannot simply learn `f(x) = c` via a bias term) and L2
  weight decay (which otherwise permits the *nearly*-trivial solution of shrinking weights toward
  zero). The encoder is pretrained as a bias-free autoencoder and its weights reused to
  initialize the SVDD network, as the original paper recommends.
* **Autoencoder reconstruction error** -- the same autoencoder trained for dimensionality
  reduction doubles as a fourth anomaly detector, letting the comparison include "is the
  embedding's reconstruction error itself a good anomaly signal" alongside the three
  purpose-built detectors.

## Hyperparameter optimization

Every tunable method has an Optuna search space (`hpo/optuna_runner.py`), each scored on the
labeled validation slice: ROC-AUC for anomaly detectors, NMI-against-`attack_category` for
clustering and UMAP, validation loss for the autoencoder's own training (with a separate ROC-AUC
objective for its architecture search). All studies use a seeded `TPESampler` for
reproducibility. See `docs/experiments.md` for the exact search spaces and trial counts used in
the reported run.

## Evaluation metrics

* **ROC-AUC / PR-AUC** -- threshold-free, the primary cross-method comparison axis.
* **F1 at a quantile threshold** -- "flag the top X% most anomalous flows," the operating point a
  SOC analyst would actually pick, evaluated at X = 20% by default.
* **ARI / NMI / silhouette** -- external (label-informed, evaluation-only) and internal
  (label-free) clustering quality, reported side by side since they can disagree.
* **Per-attack-category detection rate** -- the error-analysis table showing exactly which
  attack families (`dos`, `probe`, `r2l`, `u2r`) each method catches and misses, at the same
  quantile threshold used for F1.
* **Cross-method consensus** (`evaluation/error_analysis.py`) -- percentile-rank-normalizes
  every method's raw scores onto a comparable [0, 1] axis, then reports pairwise flagged-set
  overlap (Jaccard) and attacks every method jointly misses -- these tend to be the genuinely
  hard, statistically normal-looking attacks (typically R2L), not an artifact of any one method.
