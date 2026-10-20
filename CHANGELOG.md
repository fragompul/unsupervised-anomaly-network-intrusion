# Changelog

## 0.2.0 — Noctua (2026-10-20)

**Research protocol**
* Three datasets (NSL-KDD, UNSW-NB15, CIC-IDS2017) with SHA-256-pinned, registration-free downloads;
  temporal CIC-IDS2017 protocol with timestamp repair; harmonized 18-feature UNSW↔CIC view.
* Fifteen detectors behind one interface (β-VAE, RealNVP flow, NeuTraL AD, masked tab-transformer,
  Deep SVDD, AE, PCA, GMM, kNN, LOF, ECOD, COPOD, HBOS, Isolation Forest, One-Class SVM).
* Split-conformal calibration with valid p-value merging; Consensus-Trimmed Refit (CTR) against
  training contamination; PSI drift alarm; exact AE attributions with deletion-faithfulness test.
* Experiments E1–E4 (main, contamination, drift, shift), resumable and fault-isolated; analysis
  recomputes every reported number from stored raw scores; Friedman/Nemenyi/Holm statistics.

**Noctua web app** (React 19, TypeScript, WebGL): 3D explorer, benchmark, compare, robustness lab,
live replay, in-browser playground, paper page. Static, GitHub-Pages-ready; optional FastAPI service.

**Engineering:** mypy, pre-commit, Python 3.10/3.12 CI matrix, Playwright e2e, MkDocs Material site,
Pages workflow, multi-stage Dockerfile, CITATION.cff, LaTeX paper generated from results.

**Legacy:** the original NSL-KDD pipeline (Optuna, clustering, Bokeh dashboard) is preserved.

## 0.1.0

Initial NSL-KDD benchmark: PCA/UMAP/t-SNE/autoencoder, KMeans/GMM/HDBSCAN, IF/OCSVM/Deep SVDD, Bokeh dashboard.
