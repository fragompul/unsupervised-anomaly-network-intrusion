# Noctua — Roadmap & Progress

Living tracker. Updated at the end of every phase (each phase = one branch + PR into `main`).

| # | Phase | Branch | Status |
|---|-------|--------|--------|
| 0 | Foundation: tooling, typing, pre-commit, CI matrix | `chore/foundation` | in progress |
| 1 | Multi-dataset layer: NSL-KDD, UNSW-NB15, CIC-IDS2017 + harmonized feature space | `feat/multi-dataset` | todo |
| 2 | Detector zoo: VAE, normalizing flow, contrastive (NeuTraL-style), masked tabular transformer, PyOD baselines | `feat/detectors` | todo |
| 3 | Research protocol: multi-seed, contamination sweep, drift, cross-dataset, conformal, stats | `feat/protocol` | todo |
| 4 | Explainability + cost profiling | `feat/explain` | todo |
| 5 | Export layer (static artifacts) + FastAPI service | `feat/export-api` | todo |
| 6 | Noctua web app (React + TS + WebGL) | `feat/app` | todo |
| 7 | Paper (LaTeX), MkDocs site, Docker, release, README overhaul | `feat/paper-docs` | todo |

## Data availability check (2026-10-19)

| Dataset | Source | Gate? | Verdict |
|---|---|---|---|
| NSL-KDD | GitHub mirror `defcom17/NSL_KDD` | none | used |
| UNSW-NB15 | HF `wwydmanski/UNSW-NB15` (official train/test CSV as parquet) | none | used. Note: the mirror's `train`/`test` names are **swapped** vs. the official split (its `train.parquet` has 82,332 rows = official *testing* set) |
| CIC-IDS2017 | HF `rdpahalavan/CIC-IDS2017` `Network-Flows/CICIDS_Flow.parquet` (2.83M flows, timestamps) | none (the official UNB host asks for a personal-data form) | used |
| UNSW-NB15 raw / CIC-IDS2018 | not needed | - | skipped |
