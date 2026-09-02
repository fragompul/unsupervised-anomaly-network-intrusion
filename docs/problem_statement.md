# Problem Statement

## Why unsupervised

Supervised network intrusion detection has a labeling problem: labeled attack traffic is scarce,
attacks evolve faster than labeling pipelines, and a supervised classifier trained on known
attack signatures is, by construction, blind to the ones it has never seen. NSL-KDD itself
embeds this reality in its test split, which contains attack types absent from training
(`mailbomb`, `processtable`, `worm`, and others), a deliberate probe of generalization to
novel attacks.

Unsupervised methods sidestep the labeling bottleneck by modeling what *normal* traffic looks
like (density, clusters, a low-dimensional manifold) and flagging deviation from it, regardless
of whether the deviation matches a label anyone has seen before. That is also precisely their
weakness: without labels, there is no direct way to tell a genuinely novel attack from a
legitimate but unusual traffic pattern, and "anomalous" is not the same thing as "malicious."
This project takes that trade-off seriously rather than glossing over it, see
[limitations.md](limitations.md).

## Why this is a hard, non-toy problem

* **No single right answer.** Dimensionality reduction, clustering, and anomaly detection each
  optimize a different objective (reconstruction, density, isolation, likelihood, distance to a
  learned boundary), so "best method" is not well-posed without pinning down what "best" means
  for a specific deployment. This project reports several axes (ROC-AUC, PR-AUC, F1 at a
  realistic operating point, ARI/NMI, silhouette) instead of collapsing everything into one
  number.
* **Class imbalance and heterogeneity.** Attack categories are wildly imbalanced (DoS floods are
  common and statistically loud; U2R/R2L attacks are rare and can look, feature-wise, close to
  normal sessions), see [results.md](results.md) for the per-category breakdown, which is
  where most methods' real weaknesses show up.
* **Feature scale heterogeneity.** NSL-KDD mixes near-binary flags, bounded [0, 1] rate
  features, and heavy-tailed byte/duration counts spanning orders of magnitude, naive
  Euclidean-distance methods (KMeans, UMAP, RBF-kernel SVM) are dominated by whichever feature
  happens to have the largest raw scale unless this is handled deliberately (see
  [methodology.md](methodology.md)).
* **One-class vs. fully-unsupervised is a real methodological choice**, not a formality, see
  [architecture.md](architecture.md) for how both regimes are compared head-to-head on identical
  test data.

## Dataset

**NSL-KDD** (Tavallaee et al., 2009), a de-duplicated, difficulty-rebalanced revision of the
1999 KDD Cup intrusion detection set. 41 engineered flow features (basic TCP/IP properties,
content features, and time/host-based traffic statistics) across ~126k training and ~22.5k test
flows, labeled with one of 39 fine-grained attack types (or `normal`), grouped here into the
standard 5-class taxonomy: `normal`, `dos`, `probe`, `r2l`, `u2r`.

CICIDS2017, the more modern, higher-fidelity alternative, was evaluated first but its
official host requires submitting personal contact details through a registration form before
releasing a download link; NSL-KDD was used instead specifically to avoid pushing personal data
through a third-party form for a portfolio project. See the root README's *Key Engineering
Decisions* for the full reasoning.
