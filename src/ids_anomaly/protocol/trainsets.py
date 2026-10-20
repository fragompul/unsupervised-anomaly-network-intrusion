"""Controlled training sets: the clean, natural and attack-injected regimes.

Three regimes, all unlabeled from the *model's* point of view (labels only decide which rows go
where, never what a model sees):

* ``clean``    -- normal-only (the classical one-class setting; needs a curated window);
* ``natural``  -- the training split exactly as released (NSL-KDD 47% attacks, UNSW-NB15 68%,
  CIC-IDS2017 Mon+Tue ~1.4%): "fully unsupervised in the wild";
* ``injected`` -- a normal base with attacks injected at an exact contamination rate ``rho``:
  the controlled axis of experiment E2.

In every regime a separate *calibration* slice of clean normals is held out first. It is never
fit on, only used to set thresholds and conformal p-values (a small curated sample is a far
weaker assumption than a curated training set, and it is stated as such in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ids_anomaly.data.preprocess import Dataset

Regime = Literal["clean", "natural", "injected"]


@dataclass
class FitSets:
    X_fit: np.ndarray
    X_cal: np.ndarray
    n_injected: int
    contamination: float  # realized fraction of attacks in X_fit


def make_fit_sets(
    train: Dataset,
    regime: Regime,
    *,
    rho: float = 0.0,
    n_fit: int | None = None,
    cal_fraction: float = 0.15,
    max_cal: int = 20_000,
    seed: int = 0,
) -> FitSets:
    rng = np.random.default_rng(seed)
    normal_idx = rng.permutation(np.flatnonzero(~train.is_attack))
    attack_idx = rng.permutation(np.flatnonzero(train.is_attack))

    n_cal = min(max_cal, max(1, int(len(normal_idx) * cal_fraction)))
    cal_idx, fit_normal = normal_idx[:n_cal], normal_idx[n_cal:]

    if regime == "clean":
        fit_idx = fit_normal if n_fit is None else fit_normal[:n_fit]
        n_inj = 0
    elif regime == "natural":
        fit_idx = rng.permutation(np.concatenate([fit_normal, attack_idx]))
        if n_fit is not None:
            fit_idx = fit_idx[:n_fit]
        n_inj = int(train.is_attack[fit_idx].sum())
    elif regime == "injected":
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        total = n_fit if n_fit is not None else len(fit_normal)
        n_inj = int(round(rho * total))
        n_norm = total - n_inj
        if n_inj > len(attack_idx) or n_norm > len(fit_normal):
            raise ValueError(
                f"cannot build n_fit={total} at rho={rho}: need {n_inj} attacks (have "
                f"{len(attack_idx)}) and {n_norm} normals (have {len(fit_normal)})"
            )
        fit_idx = rng.permutation(np.concatenate([fit_normal[:n_norm], attack_idx[:n_inj]]))
    else:
        raise ValueError(f"unknown regime {regime!r}")

    return FitSets(
        X_fit=train.X[fit_idx],
        X_cal=train.X[cal_idx],
        n_injected=n_inj,
        contamination=float(train.is_attack[fit_idx].mean()) if len(fit_idx) else 0.0,
    )
