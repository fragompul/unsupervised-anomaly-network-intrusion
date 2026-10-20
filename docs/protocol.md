# Research protocol

Noctua asks one question: **which unsupervised detector assumptions survive contact with
realistic network data?** Four experiments remove one textbook assumption at a time. Everything is
label-free at fit time; labels only score results.

## Shared conventions

* **Detector contract.** `fit(X)` sees unlabeled rows; `score(X)` returns one float per row, higher
  = more anomalous. 15 detectors across tree, kernel, density, distance, reconstruction,
  deep one-class and self-supervised families (`ids_anomaly.detectors`).
* **No per-dataset tuning.** Every detector uses one fixed, documented configuration on all
  datasets. Tuning on labels would turn "unsupervised" into a claim about the model instead of the
  pipeline. (The legacy NSL-KDD pipeline in `ids_anomaly.pipeline` keeps its Optuna study.)
* **Held-out calibration slice.** Before anything else, 15% of the training split's *clean normal*
  flows (capped at 20k) are held out. They are never fit on; they set thresholds and conformal
  p-values. Assuming a small curated calibration sample is far weaker than assuming a curated
  training set, and is stated as such.
* **Seeds.** Detector initialization, calibration split and contamination draws vary per seed;
  dataset splits are fixed. Default: 5 seeds (3 for the heaviest sweeps).
* **Compute vs. analysis are decoupled.** A *cell* fits all detectors on one training set and stores
  raw score vectors in `results/scores/<experiment>/*.npz`. `ids_anomaly.protocol.analysis`
  recomputes every table, figure input and statistic from those files.

## Metrics

| Metric | Meaning |
|---|---|
| ROC-AUC, PR-AUC | threshold-free ranking quality |
| TPR @ 1% FPR | recall at the strictest alarm budget |
| **Conformal operating point** | threshold fixed from calibration scores at level α ∈ {1%, 5%}; reports realized FPR, TPR, F1 |
| Realized FPR vs nominal α | the **calibration honesty check** |
| PSI of benign scores | label-free drift signal |

## Conformal layer

Given anomaly scores on the calibration set `C` (clean normals), the p-value of a new score `s` is

    p(s) = (1 + |{c in C : c >= s}|) / (|C| + 1)

If calibration and test normals are exchangeable, `P(p <= α) <= α` — a finite-sample, distribution-free
bound on the false-positive rate for **any** detector. Several detectors are merged with
`min(1, 2 · mean_k p_k)`, which is itself a valid p-value (Vovk & Wang, 2020). The unit tests verify
the guarantee by Monte-Carlo simulation and verify that it is *measurably violated* under shift.
For ranking (AUC), ensembles use the mean of `-log10 p_k`, which is not a p-value but is a strictly
better ranking signal than the merged p-value (which saturates at the floor).

## E1 — Main benchmark

3 datasets × 15 detectors × seeds × two training regimes:

* **clean**: training = normal-only (the one-class setting);
* **natural**: the training split exactly as released (NSL-KDD 47% attacks, UNSW-NB15 68%, CIC-IDS2017 Mon+Tue ≈ 1.4%).

Statistics: bootstrap CIs over seeds, Friedman test over (dataset, regime, seed) blocks with
Nemenyi critical differences (Demšar, 2006) and Holm-corrected Wilcoxon against the best-ranked
method. Seeds-as-replicates is optimistic; the analysis also reports the coarser
(dataset × regime) blocks.

## E2 — Training contamination

Attacks are injected into a normal base at an exact rate ρ ∈ {0, 1, 2, 5, 10, 20}% (30k training
rows). CIC-IDS2017 uses its iid split here so every attack family exists in the pool.

**Consensus-Trimmed Refit (CTR)** — our label-free robustification
(`ids_anomaly.detectors.purify`): score the training rows with three structurally different cheap
*screeners* (Isolation Forest, ECOD, PCA reconstruction), average the percentile ranks, drop the top
`trim` fraction, refit the base detector. Reported against: no purification, trim ∈ {5, 10, 20}%,
each single screener, and an *oracle* trim equal to the true ρ (headroom bound). The ablation is
reported as found: a single well-matched screener (ECOD) can match or beat the committee on some
datasets, while a poorly matched one (PCA alone) can hurt; the committee's role is robustness to that
choice, not uniform dominance.

## E3 — Temporal drift (CIC-IDS2017)

Fit on Monday's attack-free traffic; calibrate on held-out Monday flows; evaluate Tue–Fri.
Measures per-day detection AUC, realized FPR at nominal 1%/5%, and PSI of benign scores vs.
calibration. Question: does a **label-free** PSI alarm (≥ 0.25) predict the loss of the
conformal guarantee (realized FPR ≥ 3× nominal)?

## E4 — Dataset shift (UNSW-NB15 ↔ CIC-IDS2017)

Both datasets are mapped to 18 aligned flow features (see [datasets.md](datasets.md)). A detector
is fit on one dataset's clean normals and scored on the other, with the source scaler or with
per-domain standardization from the target's *unlabeled* statistics. Reports transfer AUC,
retention relative to in-dataset AUC, and calibration collapse.

## Reproduce

```bash
python -m ids_anomaly.cli data                     # download + verify (SHA-256) all datasets
python -m ids_anomaly.cli benchmark --experiment all --seeds 5 --workers 3
python -m ids_anomaly.cli analyze                  # results/analysis/summary.json
python -m ids_anomaly.export                       # static artifacts for the web app
```
