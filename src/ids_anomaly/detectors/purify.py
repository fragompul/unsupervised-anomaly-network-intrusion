"""Consensus-Trimmed Refit (CTR): label-free robustification against training contamination.

Problem. Reconstruction, density and one-class detectors assume (or at least profit from) a clean
training window; attacks hiding in it are *learned as normal*. Real deployments cannot guarantee
a clean window.

Method. Before fitting the base detector, score the training rows with a small committee of
cheap, structurally different *screeners* (Isolation Forest, ECOD, PCA reconstruction), convert
each score to a percentile rank, average the ranks, drop the ``trim`` fraction with the highest
consensus rank, and fit the base detector on what remains. Agreement across independent
inductive biases is a much more specific contamination signal than any single detector's tail,
and the screeners cost a fraction of the base model.

``trim`` is a single interpretable knob (the assumed upper bound on contamination). E2 reports
the full trade-off: too small leaves attacks in; too large shaves legitimate rare-but-benign
flows off the normal profile. The label-free oracle ``trim = true rho`` is the headroom bound.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from ids_anomaly.detectors import build
from ids_anomaly.detectors.base import Detector

DEFAULT_SCREENERS: tuple[str, ...] = ("iforest", "ecod", "pca")


def consensus_rank(X: np.ndarray, screeners: tuple[str, ...], seed: int) -> np.ndarray:
    """Mean percentile rank of ``X`` under each screener fitted on ``X`` itself, in (0, 1]."""
    ranks = []
    for name in screeners:
        det = build(name, seed=seed)
        scores = det.fit(X).score(X)
        ranks.append(rankdata(scores) / len(scores))
    return np.mean(ranks, axis=0)


class PurifiedDetector(Detector):
    family = "robust-wrapper"

    def __init__(
        self,
        base: str,
        trim: float = 0.10,
        screeners: tuple[str, ...] = DEFAULT_SCREENERS,
        seed: int = 42,
        **base_kwargs,
    ):
        super().__init__(seed)
        if not 0.0 <= trim < 0.9:
            raise ValueError(f"trim must be in [0, 0.9), got {trim}")
        self.base_name, self.trim, self.screeners = base, trim, screeners
        self.base_kwargs = base_kwargs
        self.name = f"{base}+ctr{trim:g}"  # type: ignore[misc]
        self.base = build(base, seed=seed, **base_kwargs)

    def fit(self, X: np.ndarray) -> PurifiedDetector:
        if self.trim > 0:
            consensus = consensus_rank(X, self.screeners, self.seed)
            keep = consensus <= np.quantile(consensus, 1.0 - self.trim)
            self.n_trimmed = int((~keep).sum())
            X = X[keep]
        else:
            self.n_trimmed = 0
        self.base.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.base.score(X)


def build_spec(spec: str, seed: int = 42, rho: float = 0.0) -> Detector:
    """Build a detector from a spec string.

    ``"ae"`` -> plain detector; ``"ae+ctr0.1"`` -> CTR with fixed trim 0.1;
    ``"ae+ctr@rho"`` -> oracle CTR whose trim equals the true contamination ``rho``;
    ``"ae+ctr0.1[gmm,pca]"`` -> CTR with an explicit screener committee (ablations).
    """
    if "+ctr" not in spec:
        return build(spec, seed=seed)
    base, _, arg = spec.partition("+ctr")
    screeners = DEFAULT_SCREENERS
    if arg.endswith("]"):
        arg, _, committee = arg[:-1].partition("[")
        screeners = tuple(committee.split(","))
    trim = rho if arg == "@rho" else float(arg)
    return PurifiedDetector(base, trim=trim, screeners=screeners, seed=seed)
