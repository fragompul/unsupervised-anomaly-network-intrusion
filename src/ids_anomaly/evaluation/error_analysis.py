"""Cross-method error analysis: where do all detectors agree, and which flows does every
method miss? Both are useful signals -- universal misses point at genuinely hard/ambiguous
traffic (e.g. R2L attacks that look statistically close to normal sessions), while universal
catches validate that the feature set carries a real signal independent of model choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_normalized_scores(scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """Percentile-rank each method's raw scores so they are comparable on a common [0, 1] axis
    (raw anomaly scores live on incomparable scales: SVDD distances vs. log-likelihoods vs.
    isolation depths).
    """
    df = pd.DataFrame(scores)
    return df.rank(pct=True)


def consensus_misses(
    ranked_scores: pd.DataFrame, is_attack: np.ndarray, quantile: float, min_methods_missing: int
) -> pd.DataFrame:
    """Attacks that at least ``min_methods_missing`` of the compared methods fail to flag."""
    flagged = ranked_scores >= (1 - quantile)
    n_missed = (~flagged).sum(axis=1)
    out = pd.DataFrame({"n_methods_missed": n_missed, "is_attack": is_attack})
    return out[(out["is_attack"]) & (out["n_methods_missed"] >= min_methods_missing)]


def method_agreement(ranked_scores: pd.DataFrame, quantile: float) -> pd.DataFrame:
    """Pairwise Jaccard overlap of each method's top-``quantile`` flagged set."""
    flagged = ranked_scores >= (1 - quantile)
    methods = flagged.columns.tolist()
    out = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for a in methods:
        for b in methods:
            inter = (flagged[a] & flagged[b]).sum()
            union = (flagged[a] | flagged[b]).sum()
            out.loc[a, b] = inter / union if union else float("nan")
    return out
