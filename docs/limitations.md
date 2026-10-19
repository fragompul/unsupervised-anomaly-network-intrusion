# Limitations and future work

Stated plainly, as part of the deliverable rather than an afterthought.

## Datasets

* **All three corpora are lab-generated.** NSL-KDD (1999 traffic re-cut in 2009), UNSW-NB15 (2015,
  synthetic IXIA traffic) and CIC-IDS2017 (a five-day testbed) do not reflect today's TLS 1.3 / QUIC
  / encrypted-DNS networks, and none is production traffic. Absolute numbers are *not*
  production-readiness numbers; the study's claim is about **relative** behaviour under
  contamination, drift and shift.
* **Known dataset issues.** CIC-IDS2017 has documented labeling and flow-construction problems
  (Engelen et al., 2021); we repair its mixed timestamp formats and drop identifiers (IPs, ports,
  flow id) but we do not re-label. The UNSW-NB15 mirror we use swaps the `train`/`test` file names
  relative to the official split (handled by row count). UNSW's released training split is 68%
  attacks, so the *natural* regime there is more adversarial than any realistic deployment, which
  is why E2 injects attacks at controlled rates instead.
* **CIC-IDS2017 is subsampled** (class-aware, rare classes kept whole), which tilts rare-attack
  prevalence upward relative to the raw capture.
* **The harmonized cross-dataset view is lossy** (18 features, two datasets). A failure to transfer
  there is evidence about *that* feature space, not about every possible representation.
* **Timestamps:** CIC-IDS2017's 12-hour clock without AM/PM is repaired by an 08:00–17:00
  assumption; within-day ordering is therefore approximate.

## Methodology

* **Fixed hyperparameters.** Every detector uses one configuration across datasets. This avoids
  label leakage into "unsupervised" tuning but under-serves detectors that would profit from
  per-dataset tuning (the legacy NSL-KDD pipeline keeps its Optuna study, and its numbers are
  not comparable to the new benchmark's).
* **A clean calibration slice is assumed** (15% of clean normals, capped at 20k). It is a weaker
  assumption than a clean training set but not a free one. The conformal guarantee is exactly as
  strong as exchangeability between calibration and deployment traffic, which E3/E4 show can fail.
* **Seeds are replicates over models, not over datasets.** Dataset splits are fixed, so
  Friedman tests over (dataset, regime, seed) blocks are optimistic; the analysis also reports
  coarser (dataset × regime) blocks.
* **CTR's trim is a hyperparameter.** It is an upper bound on assumed contamination; E2 reports
  the full trade-off (including an oracle), but a deployment must still choose it.
* **Anomalous ≠ malicious.** Every method detects deviation from learned normality; a benign but
  unusual flow scores like an attack. Triage and a supervised layer remain necessary.
* **TabPFN-style tabular foundation models were evaluated for feasibility and excluded.** The
  weights are downloadable, but using them as unsupervised detectors requires one conditional
  prediction per feature per flow (~70 per flow here), which is impractical on CPU for hundreds of
  thousands of flows, and their license is non-standard.
* **Single machine, CPU only.** Latency figures are relative (per 1,000 flows on a laptop CPU), not
  deployment benchmarks.

## What a real deployment would still need

* A drift response (the PSI alarm is a signal, not a policy): re-calibration schedule, re-fit
  triggers and analyst feedback.
* Streaming inference engineering; the replay is a simulation over stored flows.
* Adversarial evaluation: the benchmark does not test evasion or poisoning by an adaptive attacker.

## Natural next steps

* Add modern encrypted-traffic corpora and re-run the protocol unchanged.
* Online/streaming variants with conformal re-calibration under drift (e.g. weighted or adaptive
  conformal methods).
* Cost-sensitive operating points using analyst-triage budgets instead of a fixed α.
