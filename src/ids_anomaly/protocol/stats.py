"""Statistical comparison of detectors across datasets and regimes (Demšar, 2006)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass
class FriedmanResult:
    statistic: float
    pvalue: float
    mean_ranks: dict[str, float]
    critical_difference: float
    n_blocks: int


def rank_matrix(scores: np.ndarray) -> np.ndarray:
    """Per-block ranks, 1 = best (highest metric); ties get average ranks."""
    return np.vstack([stats.rankdata(-row) for row in scores])


def nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for ``k`` methods over ``n`` blocks."""
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    return float(q * np.sqrt(k * (k + 1) / (6.0 * n)))


def friedman(scores: np.ndarray, names: list[str], alpha: float = 0.05) -> FriedmanResult:
    """``scores`` has shape ``(n_blocks, n_methods)``; higher is better."""
    n, k = scores.shape
    res = stats.friedmanchisquare(*[scores[:, j] for j in range(k)])
    ranks = rank_matrix(scores).mean(axis=0)
    return FriedmanResult(
        statistic=float(res.statistic),
        pvalue=float(res.pvalue),
        mean_ranks=dict(zip(names, ranks.tolist(), strict=True)),
        critical_difference=nemenyi_cd(k, n, alpha),
        n_blocks=n,
    )


def cd_groups(mean_ranks: dict[str, float], cd: float) -> list[list[str]]:
    """Maximal cliques of methods whose mean ranks differ by less than ``cd`` (the bars in a
    critical-difference diagram)."""
    ordered = sorted(mean_ranks, key=mean_ranks.__getitem__)
    groups: list[list[str]] = []
    for i, a in enumerate(ordered):
        group = [a] + [b for b in ordered[i + 1 :] if mean_ranks[b] - mean_ranks[a] < cd]
        if len(group) > 1 and not any(set(group) <= set(g) for g in groups):
            groups.append(group)
    return groups


def holm_wilcoxon(
    scores: np.ndarray, names: list[str], baseline: str, alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Paired Wilcoxon signed-rank of every method vs ``baseline`` with Holm correction."""
    j0 = names.index(baseline)
    rows: list[dict[str, Any]] = []
    for j, name in enumerate(names):
        if j == j0:
            continue
        diff = scores[:, j] - scores[:, j0]
        p = 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff, zero_method="wilcox").pvalue)
        rows.append({"method": name, "mean_diff": float(diff.mean()), "p": p})
    order = np.argsort([r["p"] for r in rows])
    m = len(rows)
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * float(rows[idx]["p"]))
        running = max(running, adj)  # enforce monotone adjusted p-values
        rows[idx]["p_holm"] = running
        rows[idx]["significant"] = running < alpha
    return rows


def pairwise_win_matrix(scores: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    """Fraction of blocks where row-method beats column-method."""
    out: dict[str, dict[str, float]] = {a: {} for a in names}
    for (i, a), (j, b) in itertools.product(enumerate(names), repeat=2):
        out[a][b] = float((scores[:, i] > scores[:, j]).mean()) if i != j else 0.0
    return out


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
