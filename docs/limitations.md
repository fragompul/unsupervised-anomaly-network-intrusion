# Limitations and Future Work

Stated plainly, as part of the deliverable rather than an afterthought.

## Dataset

* **NSL-KDD is dated** (features engineered ~1999-2009). Its 41 hand-crafted flow statistics do
  not reflect modern traffic (TLS 1.3, QUIC, encrypted DNS) or contemporary attack techniques.
  CICIDS2017 would be the natural next step. Its official host requires a personal-data
  registration form to download, which this project deliberately avoided (see the root README's
  *Key Engineering Decisions*); a version obtainable without that gate (e.g. via an
  institutionally-agreed mirror) is the natural follow-up.
* **Attack-category imbalance** is severe (U2R in particular has a handful of training
  examples), which the per-category detection-rate breakdown in `docs/results.md` surfaces
  directly rather than hiding behind an aggregate accuracy number.
* NSL-KDD's simulated traffic is not real production traffic; base rates, attack mixes, and
  "normal" variability all differ from a live network, so absolute metric values should not be
  read as production-readiness numbers.

## Methodology

* **Anomalous ≠ malicious.** Every method here detects statistical deviation from learned
  "normal" structure. A legitimate but unusual traffic pattern (a new but benign application, an
  unusual but authorized bulk transfer) will score as anomalous exactly like an attack would.
  None of these methods, on their own, distinguish the two. Doing so requires either analyst
  triage on flagged flows (which the dashboard is built to support) or a supervised layer on top
  trained with confirmed labels.
* **HPO uses a labeled validation slice.** This project is explicit that hyperparameter
  selection is not "pure" unsupervised learning (see `methodology.md`), but it is worth
  restating: a deployment with zero labeled data at all would need a different tuning strategy
  (e.g. internal metrics only, or a much smaller/cheaper labeled sample than used here).
* **HDBSCAN and t-SNE do not have a native out-of-sample transform.** HDBSCAN's test-split
  labels are obtained by re-fitting on the test embedding with hyperparameters transferred from
  the training-embedding search, rather than via `approximate_predict` against the exact
  training-fitted model, a defensible, commonly used evaluation pattern for density clustering,
  but a design choice worth flagging rather than presenting as a true streaming "predict."
* **One-Class SVM is trained on a bounded subsample** (default 8000 rows) for CPU tractability;
  a full-data SVM (or a linear/Nyström-approximated kernel) might perform differently and was out
  of scope for this project's compute budget.
* **Deep SVDD's hyperparameters were not searched by Optuna** (see `docs/experiments.md`), a
  deliberate compute trade-off, not an oversight, but it means the Deep SVDD row in the results
  table is not on fully equal tuning footing with the other detectors.

## What a real deployment would still need

* A feedback loop to periodically re-fit against drifting "normal" traffic (unsupervised models
  degrade silently as the definition of normal shifts).
* Analyst-in-the-loop triage on flagged flows, since none of these methods alone separates
  "anomalous" from "malicious" (see above).
* Latency/throughput engineering for streaming inference. This project evaluates on static
  batches, not a live flow stream.

## Natural next steps

* Swap in CICIDS2017 (or another registration-free modern IDS dataset) once available without a
  personal-data form.
* Extend the training-regime ablation (unsupervised vs. semi-supervised) to every anomaly
  detector, not just the autoencoder, with a matched compute budget for a fair comparison.
* Add a streaming/online variant of at least one detector (e.g. incremental Isolation Forest) to
  address the "static batch" limitation above.
