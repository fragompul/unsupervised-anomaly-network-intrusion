# 🛡️ Unsupervised Network Intrusion Detection

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![UMAP](https://img.shields.io/badge/UMAP-manifold_learning-9b59b6)
![HDBSCAN](https://img.shields.io/badge/HDBSCAN-density_clustering-2c3e50)
![Optuna](https://img.shields.io/badge/Optuna-HPO-00A3E0)
![Bokeh](https://img.shields.io/badge/Bokeh-server_dashboard-e37e00)
![CI](https://github.com/fragompul/unsupervised-anomaly-network-intrusion/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

> **An unsupervised, label-blind benchmark of network intrusion detection: nine methods across
> dimensionality reduction, clustering, and anomaly detection, compared head-to-head on the same
> data and the same metrics -- with a live 3D Bokeh explorer for the flows each method flags.**

---

## 🎯 Executive Summary

Supervised intrusion detection has a labeling problem: attack labels are scarce, attacks evolve
faster than labeling pipelines, and a classifier trained on known signatures is, by construction,
blind to the ones it has never seen. This project asks a harder question instead: **how well can
you flag malicious network flows without ever showing a model a single attack label?**

It benchmarks the full unsupervised toolbox on [NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html)
network flow data -- **PCA, UMAP, t-SNE, and a custom autoencoder** for dimensionality reduction;
**KMeans, Gaussian Mixtures, and HDBSCAN** for clustering; **Isolation Forest, One-Class SVM, and
Deep SVDD** for anomaly detection -- all tuned with **Optuna**, evaluated on identical train/test
splits with identical metrics, and explored live through a **Bokeh server** dashboard with a
genuinely rotatable 3D embedding view linked to an anomaly drill-down table.

Labels are used for exactly two things, both stated explicitly rather than left implicit:
selecting hyperparameters on a held-out validation slice, and scoring the final test-split
metrics. Every model is fit without ever seeing a label. See
[docs/methodology.md](docs/methodology.md) for the full reasoning.

## 🧠 Why This Is Hard

- **No single "best" method.** Reduction, clustering, and anomaly detection each optimize a
  different objective -- this project reports ROC-AUC, PR-AUC, F1 at a realistic operating point,
  ARI/NMI, and silhouette side by side rather than collapsing everything into one leaderboard
  number.
- **Severe class imbalance.** DoS floods are common and statistically loud; R2L/U2R attacks are
  rare and can look, feature-wise, close to normal sessions. The per-attack-category breakdown in
  [docs/results.md](docs/results.md) is where most methods' real weaknesses show up.
- **Wildly heterogeneous feature scales.** NSL-KDD mixes near-binary flags, bounded [0, 1] rate
  features, and byte/duration counts spanning orders of magnitude -- handled deliberately with a
  `log1p` transform on skewed columns before scaling (see *Key Engineering Decisions* below).
- **Fully-unsupervised vs. semi-supervised (one-class) is a real methodological choice**, not a
  formality -- both regimes are benchmarked head-to-head on identical test data.

## 🏗️ Architecture

```mermaid
flowchart LR
    A[NSL-KDD raw flows] --> B[preprocess.py<br/>log1p + scale + one-hot]
    B --> C[train/val/test splits<br/>labels held out of X]
    C --> D[PCA / UMAP / Autoencoder]
    D --> E[KMeans / GMM / HDBSCAN<br/>per embedding]
    C --> F[Isolation Forest<br/>One-Class SVM / Deep SVDD]
    E --> G[results/metrics/results.json]
    F --> G
    D --> H[results/embeddings_test.npz]
    G --> I[Bokeh server dashboard<br/>3D explorer + anomaly table]
    H --> I
```

Every Optuna study is scored on a labeled validation slice carved from *training* data only --
the test split stays untouched until final scoring. See [docs/architecture.md](docs/architecture.md)
for the full data-flow diagram, the fully-unsupervised vs. semi-supervised training regime split,
and the dashboard's client/server callback architecture.

## 📊 Dashboard

The dashboard is a genuine **Bokeh server** app (`bokeh serve`, not a static export): a rotatable
3D embedding scatter (drag two sliders -- rotation runs entirely client-side via `CustomJS` for a
responsive drag) linked to an anomaly drill-down table (box/lasso-select points to filter the
table, or vice versa), with live switching between PCA / UMAP / Autoencoder / t-SNE embeddings,
four independent anomaly scores, and an adjustable "flag top X%" threshold.

*(Screenshot / GIF added once the dashboard is running against real pipeline results -- see
Installation below to run it yourself.)*

## ⚙️ Installation & Usage

**1. Clone and set up the environment:**

```bash
git clone https://github.com/fragompul/unsupervised-anomaly-network-intrusion.git
cd unsupervised-anomaly-network-intrusion
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on Linux/Mac
pip install -e ".[dev]"
```

**2. Run the full benchmark** (downloads NSL-KDD automatically, no account/API key needed --
CPU-only, auto-detects CUDA if available):

```bash
python -m ids_anomaly.cli run --trials 25 -v
```

This writes `results/metrics/results.json` (every metric in `docs/results.md`) and
`results/embeddings_test.npz` (what the dashboard reads). HPO trials run in parallel across your
CPU cores automatically (`--jobs` to override).

**3. Launch the dashboard:**

```bash
bokeh serve --show dashboard/app.py
```

**4. Run tests / lint:**

```bash
pytest --cov=ids_anomaly
ruff check .
```

## 📈 Key Results

See [docs/results.md](docs/results.md) for the full comparison tables (every reduction ×
clustering combination, all four anomaly detectors, per-attack-category detection rates, and
cross-method consensus analysis). Headline numbers below are filled in from the actual run this
repository ships with.

<!-- RESULTS_SUMMARY_START -->
*Populated automatically from `results/metrics/results.json` -- see docs/results.md for the full
tables.*
<!-- RESULTS_SUMMARY_END -->

## 🔑 Key Engineering Decisions

- **NSL-KDD over CICIDS2017.** CICIDS2017 (the more modern, higher-fidelity choice) was
  evaluated first, but its official host gates the download behind a form requiring name, email,
  organization, and country -- submitting personal data to a third-party site for a portfolio
  project wasn't worth it. NSL-KDD is a well-established, non-trivial alternative distributed
  without any access gate, and its test split's held-out novel attack types happen to be a good
  fit for testing unsupervised generalization anyway.
- **`log1p` before scaling, not scaling alone.** NSL-KDD's byte/count columns span orders of
  magnitude; standardizing them raw still leaves outlier-heavy tails that dominate every
  Euclidean-distance method (KMeans, UMAP, RBF-kernel SVM). `log1p` first, then scale --
  applied only to the skewed count/byte columns, not to already-bounded rate features or flags
  (see `src/ids_anomaly/data/preprocess.py`).
- **Two training regimes, compared explicitly, not blended.** KMeans/GMM/HDBSCAN/Isolation
  Forest fit on the full unlabeled (attack-contaminated) training data -- the only regime that
  matches production. One-Class SVM/Deep SVDD/the autoencoder fit on a normal-only subset -- the
  classical one-class regime those methods were designed for. Both are scored on the same test
  split so the comparison is honest about which assumption each method depends on.
- **Bias-free Deep SVDD, not a naive implementation.** Following Ruff et al. (2018) exactly:
  bias-free linear layers plus L2 weight decay to block the trivial "collapse to a constant"
  solution, with the encoder pretrained as a bias-free autoencoder first. A naive Deep SVDD
  implementation without these safeguards routinely collapses during training.
- **Parallelized Optuna HPO after profiling, not upfront.** The first full run showed UMAP HPO
  taking minutes per trial, single-threaded, on a 16-core machine sitting mostly idle. Every
  objective's inner work (BLAS, numba-jitted UMAP, PyTorch) releases the GIL, so switching to
  threaded Optuna trials (with per-trial estimators pinned to `n_jobs=1` to avoid nested
  oversubscription) gave a large real speedup -- documented in `src/ids_anomaly/pipeline.py`
  rather than silently reverted to serial.
- **A hand-rolled 3D rotation instead of a Three.js dependency.** Bokeh has no native 3D scatter.
  Rather than pull in a Three.js/WebGL extension (a build step, a Node.js dependency, more
  failure surface for a single-file `bokeh serve` app), rotation is a plain orthographic
  projection with a depth-based size/alpha cue, computed in ~30 lines of `CustomJS` shared with
  an equivalent NumPy implementation for the initial server-side render (see
  `dashboard/projection.py`). No build step, no extra runtime dependency, still genuinely
  interactive.
- **HDBSCAN's "all noise" result on a single blob is not a bug.** Confirmed against scikit-
  learn's own `HDBSCAN` implementation while writing tests: a single, unimodal Gaussian blob has
  no density valley for stability-based cluster selection to split on. Documented in
  `tests/test_clustering.py` rather than "fixed" by picking different test data and moving on.

## 📂 Project Structure

```
unsupervised-anomaly-network-intrusion/
├── src/ids_anomaly/
│   ├── data/            # download, schema, label-blind preprocessing
│   ├── reduction/         # PCA, UMAP, t-SNE, autoencoder
│   ├── clustering/         # KMeans, GMM, HDBSCAN
│   ├── anomaly/             # Isolation Forest, One-Class SVM, Deep SVDD
│   ├── evaluation/           # metrics, cross-method error analysis
│   ├── hpo/                   # Optuna search spaces
│   ├── pipeline.py              # orchestrates the full run
│   └── cli.py                    # `python -m ids_anomaly.cli run`
├── dashboard/            # Bokeh server app (3D explorer + anomaly table)
├── notebooks/              # exploration + docs/ figure generation only
├── tests/                    # pytest, network-free (synthetic fixture)
├── docs/                        # methodology, architecture, results, limitations
└── .github/workflows/ci.yml       # ruff + pytest on every push
```

See [docs/architecture.md](docs/architecture.md) for the full data-flow diagram.

## 📄 Documentation

- [docs/problem_statement.md](docs/problem_statement.md) -- why unsupervised, why this is hard
- [docs/methodology.md](docs/methodology.md) -- preprocessing, label usage, splits, every method
- [docs/architecture.md](docs/architecture.md) -- data flow, training regimes, dashboard design
- [docs/experiments.md](docs/experiments.md) -- exact Optuna search spaces and trial counts
- [docs/results.md](docs/results.md) -- full comparison tables and figures
- [docs/limitations.md](docs/limitations.md) -- honest limitations and future work

## 📜 License

MIT -- see [LICENSE](LICENSE).
