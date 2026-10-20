"""Shared preprocessing for the modern flow datasets (UNSW-NB15, CIC-IDS2017).

Those datasets are messier than NSL-KDD: byte/packet/duration columns are heavy-tailed *and*
can be negative (CICFlowMeter emits ``-1`` sentinels and negative IAT minima) or infinite
(``Flow Bytes/s`` on zero-duration flows). The recipe, fit on the training split only:

1. replace non-finite values with 0 (a zero-rate flow is the honest reading of ``inf/NaN`` here),
2. signed ``log1p`` (``sign(x) * log1p(|x|)``) to tame tails without destroying the sign,
3. standardize, then clip to ``+/- clip`` standard deviations so a single absurd test-time value
   cannot dominate every Euclidean-distance method downstream,
4. drop columns that are constant in training (they carry no information and break nothing, but
   inflate dimensionality for UMAP/KNN-style methods).

Categoricals are one-hot encoded with rare levels folded into an "infrequent" bucket, and unseen
test levels map to that same bucket.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def signed_log1p(x: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` with non-finite entries mapped to 0."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.sign(x) * np.log1p(np.abs(x))


def _clip(x: np.ndarray, bound: float) -> np.ndarray:
    return np.clip(x, -bound, bound)


def build_tabular_preprocessor(
    numeric: Sequence[str],
    categorical: Sequence[str] = (),
    *,
    clip: float = 10.0,
    min_frequency: float | None = 0.005,
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("signed_log1p", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            (
                "clip",
                FunctionTransformer(_clip, kw_args={"bound": clip}, feature_names_out="one-to-one"),
            ),
            ("drop_constant", VarianceThreshold(0.0)),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [("num", numeric_pipe, list(numeric))]
    if categorical:
        transformers.append(
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                    dtype=np.float32,
                ),
                list(categorical),
            )
        )
    return ColumnTransformer(transformers=transformers)


def stratified_cap(
    categories: np.ndarray,
    cap: int,
    rng: np.random.Generator,
    keep_whole_below: int = 2000,
) -> np.ndarray:
    """Indices of a subsample of roughly ``cap`` rows that keeps every rare class intact.

    Each category is down-sampled by the same global ratio, except categories with at most
    ``keep_whole_below`` rows, which are kept whole (a uniform ratio would erase Heartbleed,
    Infiltration and similar classes entirely). Class prevalence is therefore *slightly* tilted
    toward rare attacks; this is documented in docs/datasets.md.
    """
    n = len(categories)
    if n <= cap:
        return np.arange(n)
    ratio = cap / n
    keep: list[np.ndarray] = []
    for cat in pd.unique(categories):
        idx = np.flatnonzero(categories == cat)
        if len(idx) <= keep_whole_below:
            keep.append(idx)
        else:
            m = max(keep_whole_below, int(round(len(idx) * ratio)))
            keep.append(rng.choice(idx, size=min(m, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))
