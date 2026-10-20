# Noctua: roadmap and progress

Living tracker. Updated at the end of every phase (each phase = one branch + PR into `main`).

| # | Phase | Branch | Status |
|---|-------|--------|--------|
| 0 | Foundation: tooling, typing, pre-commit, CI matrix | `chore/foundation` | done (PR #1) |
| 1 | Multi-dataset layer: NSL-KDD, UNSW-NB15, CIC-IDS2017 + harmonized space | `feat/multi-dataset` | done (PR #2) |
| 2 | Detector zoo: 15 detectors, CTR wrapper | `feat/detectors` | done (PR #3) |
| 3 | Research protocol: regimes, E1-E4, conformal, statistics | `feat/protocol` | done (PR #4) |
| 4 | Explainability + faithfulness checks | `feat/explain` | done (PR #5) |
| 5 | Export layer + FastAPI service | `feat/export-api` | done (PR #6) |
| 6-7 | Web app, docs site, CI, Docker, paper, README | `feat/app` | done (PR #7) |

All phases are stacked PRs (each targets the previous branch); merge them in order #1 → #7.

## Decisions log

* **Name:** Noctua (nocturnal owl). Repo and Python package names kept to preserve URLs and imports.
* **Hosting:** GitHub Pages is static-only, so the app runs entirely client-side from exported
  artifacts; the playground re-implements the AE forward pass in TypeScript (tested against PyTorch).
* **TabPFN:** weights are ungated but using it as an unsupervised detector needs one conditional
  prediction per feature per flow; impractical on CPU, license non-standard. Excluded and documented.
* **Findings that changed the narrative:** drift is *not* monotone (Tuesday is already shifted);
  the CTR screener committee is robust but not uniformly better than the best single screener;
  merged "all-15" ensembles are valid but very conservative (weak members kill power).

## Data availability check (2026-10-19)

| Dataset | Source | Gate? | Verdict |
|---|---|---|---|
| NSL-KDD | GitHub mirror `defcom17/NSL_KDD` | none | used |
| UNSW-NB15 | HF `wwydmanski/UNSW-NB15` (official train/test CSV as parquet) | none | used. Note: the mirror's `train`/`test` names are **swapped** vs. the official split (its `train.parquet` has 82,332 rows = official *testing* set) |
| CIC-IDS2017 | HF `rdpahalavan/CIC-IDS2017` `Network-Flows/CICIDS_Flow.parquet` (2.83M flows, timestamps) | none (the official UNB host asks for a personal-data form) | used |
| UNSW-NB15 raw / CIC-IDS2018 | not needed | - | skipped |
