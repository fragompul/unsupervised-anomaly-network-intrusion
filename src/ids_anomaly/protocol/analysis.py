"""Turn stored score files into the paper's tables and the web app's data.

Pipeline: every ``results/scores/<experiment>/*.npz`` cell is reduced to *long-format* rows
(one per detector x metric group), concatenated into pandas frames, and summarized with plain
groupbys. Outputs land in ``results/analysis/`` as CSV (full detail) and one ``summary.json``
(what the docs, the paper and the web app read).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ids_anomaly.detectors import TITLES
from ids_anomaly.protocol import conformal, metrics, stats

logger = logging.getLogger(__name__)

ALPHAS = (0.01, 0.05)
COVERAGE_GRID = np.linspace(0.005, 0.2, 40)
DIVERSE = ("iforest", "knn", "ae", "flow", "neutral")


def _title(spec: str) -> str:
    base, plus, rest = spec.partition("+")
    return TITLES.get(base, base) + (f" +{rest}" if plus else "")


def load_cell(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return {
        "specs": [str(s) for s in z["specs"]],
        "scores": {k.removeprefix("scores__"): z[k] for k in z.files if k.startswith("scores__")},
        "fit_seconds": z["fit_seconds"],
        "score_seconds": z["score_seconds"],
        "meta": meta,
    }


def load_labels(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def _valid(row: np.ndarray) -> bool:
    return bool(np.isfinite(row).all())


def _pvalues(cal: np.ndarray, eval_scores: np.ndarray) -> np.ndarray:
    return np.vstack([conformal.conformal_pvalues(c, s) for c, s in zip(cal, eval_scores, strict=True)])


def _ensemble_rows(
    specs: list[str], cal: np.ndarray, ev: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pseudo-detectors: (ranking score, merged conformal p-value) for 'all' and 'diverse' sets."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ok = [i for i, s in enumerate(specs) if _valid(ev[i]) and _valid(cal[i])]
    for name, members in (("ens_all", ok), ("ens_diverse", [i for i in ok if specs[i] in DIVERSE])):
        if len(members) < 2:
            continue
        p = _pvalues(cal[members], ev[members])
        merged = conformal.merge_pvalues_average(p)
        rank_score = -np.log10(np.clip(p, 1e-12, 1.0)).mean(axis=0)
        out[name] = (rank_score, merged)
    return out


def cell_rows(
    cell: dict[str, Any],
    labels: dict[str, np.ndarray],
    eval_name: str,
    y: np.ndarray,
    categories: np.ndarray | None,
    extra: dict[str, Any],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Metric rows, coverage rows and per-category rows for one evaluation set of one cell."""
    specs = cell["specs"]
    cal, ev = cell["scores"]["cal"], cell["scores"][eval_name]
    n_eval = ev.shape[1]
    normal = y == 0
    rows, cov_rows, cat_rows = [], [], []

    def add(name: str, score: np.ndarray, pvals: np.ndarray, i: int | None) -> None:
        row: dict[str, Any] = {
            **extra, "detector": name,
            "roc": metrics.roc_auc(y, score), "pr": metrics.pr_auc(y, score),
            "tpr_at_1pct_fpr": metrics.tpr_at_fpr(y, score, 0.01),
        }  # fmt: skip
        for alpha in ALPHAS:
            flagged = pvals <= alpha
            tp, fp = float((flagged & ~normal).sum()), float((flagged & normal).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / max(1.0, float((~normal).sum()))
            tag = f"a{int(alpha * 100)}"
            row[f"{tag}_fpr"] = fp / max(1.0, float(normal.sum()))
            row[f"{tag}_tpr"] = rec
            row[f"{tag}_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if i is not None:
            row["fit_s"] = float(cell["fit_seconds"][i])
            row["score_ms_per_1k"] = float(cell["score_seconds"][i] / n_eval * 1e6)
        rows.append(row)
        for a, f in zip(COVERAGE_GRID, [conformal.empirical_fpr(pvals[normal], a) for a in COVERAGE_GRID], strict=True):
            cov_rows.append({**extra, "detector": name, "alpha": float(a), "fpr": f})
        if categories is not None:
            flagged = pvals <= 0.05
            for cat in np.unique(categories):
                m = categories == cat
                cat_rows.append({**extra, "detector": name, "category": str(cat),
                                 "n": int(m.sum()), "recall_a5": float(flagged[m].mean())})  # fmt: skip

    for i, spec in enumerate(specs):
        if not (_valid(ev[i]) and _valid(cal[i])):
            logger.warning("skipping failed detector %s in %s", spec, extra)
            continue
        add(spec, ev[i], conformal.conformal_pvalues(cal[i], ev[i]), i)
    for name, (score, merged) in _ensemble_rows(specs, cal, ev).items():
        add(name, score, merged, None)
    return rows, cov_rows, cat_rows


# ------------------------------------------------------------------------------------- E1


def analyze_main(root: Path) -> dict[str, Any] | None:
    src = root / "results" / "scores" / "main"
    files = sorted(p for p in src.glob("*__seed*.npz"))
    if not files:
        return None
    rows, cov, cats = [], [], []
    for path in files:
        cell = load_cell(path)
        m = cell["meta"]
        labels = load_labels(src / f"labels__{m['dataset']}.npz")
        extra = {"dataset": m["dataset"], "regime": m["regime"], "seed": m["seed"],
                 "contamination": m["contamination"]}  # fmt: skip
        r, c, k = cell_rows(cell, labels, "test", labels["is_attack"].astype(int), labels["category"], extra)
        rows += r
        cov += c
        cats += k
    df, cov_df, cat_df = pd.DataFrame(rows), pd.DataFrame(cov), pd.DataFrame(cats)
    out_dir = root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "main_long.csv", index=False)

    summary: dict[str, Any] = {"table": [], "stats": {}, "coverage": [], "categories": []}
    keys = ["dataset", "regime", "detector"]
    for key, g in df.groupby(keys):
        entry = dict(zip(keys, key, strict=True))
        for metric in ("roc", "pr", "tpr_at_1pct_fpr", "a1_fpr", "a1_tpr", "a1_f1", "a5_fpr", "a5_tpr", "a5_f1", "fit_s", "score_ms_per_1k"):
            if metric in g and g[metric].notna().any():
                vals = g[metric].dropna().to_numpy()
                entry[metric] = float(vals.mean())
                entry[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                if metric == "roc" and len(vals) > 1:
                    entry["roc_ci"] = list(stats.bootstrap_mean_ci(vals))
        entry["n_seeds"] = int(len(g))
        summary["table"].append(entry)

    singles = df[~df.detector.str.startswith("ens_")]
    blocks = singles.pivot_table(index=["dataset", "regime", "seed"], columns="detector", values="roc").dropna(axis=1)
    if blocks.shape[0] >= 3 and blocks.shape[1] >= 3:
        names = list(blocks.columns)
        fr = stats.friedman(blocks.to_numpy(), names)
        coarse = singles.pivot_table(index=["dataset", "regime"], columns="detector", values="roc").dropna(axis=1)
        fr_c = stats.friedman(coarse.to_numpy(), list(coarse.columns)) if coarse.shape[0] >= 3 else None
        best = min(fr.mean_ranks, key=fr.mean_ranks.__getitem__)
        summary["stats"] = {
            "all_seeds": {"n_blocks": fr.n_blocks, "statistic": fr.statistic, "p": fr.pvalue,
                          "cd": fr.critical_difference, "mean_ranks": fr.mean_ranks,
                          "cd_groups": stats.cd_groups(fr.mean_ranks, fr.critical_difference)},
            "dataset_regime": None if fr_c is None else {
                "n_blocks": fr_c.n_blocks, "statistic": fr_c.statistic, "p": fr_c.pvalue,
                "cd": fr_c.critical_difference, "mean_ranks": fr_c.mean_ranks,
                "cd_groups": stats.cd_groups(fr_c.mean_ranks, fr_c.critical_difference)},
            "best_by_rank": best,
            "holm_vs_best": stats.holm_wilcoxon(blocks.to_numpy(), names, best),
            "win_matrix": stats.pairwise_win_matrix(blocks.to_numpy(), names),
        }  # fmt: skip
    cov_mean = cov_df.groupby(["dataset", "regime", "detector", "alpha"], as_index=False)["fpr"].mean()
    summary["coverage"] = json.loads(cov_mean.round(5).to_json(orient="records"))
    cat_mean = cat_df.groupby(["dataset", "regime", "detector", "category"], as_index=False).agg(
        n=("n", "first"), recall_a5=("recall_a5", "mean")
    )
    summary["categories"] = json.loads(cat_mean.round(5).to_json(orient="records"))
    cov_df.to_csv(out_dir / "main_coverage.csv", index=False)
    cat_df.to_csv(out_dir / "main_categories.csv", index=False)
    return summary


# ------------------------------------------------------------------------------------- E2


def analyze_contamination(root: Path) -> dict[str, Any] | None:
    src = root / "results" / "scores" / "contamination"
    files = sorted(src.glob("*__seed*.npz"))
    if not files:
        return None
    rows = []
    for path in files:
        cell = load_cell(path)
        m = cell["meta"]
        labels = load_labels(src / f"labels__{m['dataset']}.npz")
        extra = {"dataset": m["dataset"], "rho": m["rho"], "seed": m["seed"]}
        r, _, _ = cell_rows(cell, labels, "test", labels["is_attack"].astype(int), None, extra)
        rows += r
    df = pd.DataFrame(rows)
    out_dir = root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "contamination_long.csv", index=False)
    agg = (
        df.groupby(["dataset", "rho", "detector"])
        .agg(roc=("roc", "mean"), roc_std=("roc", "std"), pr=("pr", "mean"), a1_fpr=("a1_fpr", "mean"),
             a1_tpr=("a1_tpr", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )  # fmt: skip
    agg["roc_std"] = agg["roc_std"].fillna(0.0)

    # CTR gain over its own vanilla base, per dataset and rho (paired by seed)
    gains = []
    for (ds, rho), g in df.groupby(["dataset", "rho"]):
        piv = g.pivot_table(index="seed", columns="detector", values="roc")
        for col in piv.columns:
            if "+ctr" not in col:
                continue
            base = col.split("+ctr")[0]
            if base in piv.columns:
                d = (piv[col] - piv[base]).dropna()
                gains.append({"dataset": ds, "rho": rho, "detector": col, "base": base,
                              "gain": float(d.mean()), "gain_std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
                              "n": int(len(d))})  # fmt: skip
    # degradation: AUC at rho=0 minus AUC at rho=5%
    deg = []
    for (ds, det), g in agg.groupby(["dataset", "detector"]):
        a0 = g.loc[np.isclose(g.rho, 0.0), "roc"]
        a5 = g.loc[np.isclose(g.rho, 0.05), "roc"]
        if len(a0) and len(a5):
            deg.append({"dataset": ds, "detector": det, "roc_clean": float(a0.iloc[0]),
                        "roc_5pct": float(a5.iloc[0]), "drop": float(a0.iloc[0] - a5.iloc[0])})  # fmt: skip
    return {
        "table": json.loads(agg.round(5).to_json(orient="records")),
        "ctr_gain": gains,
        "degradation": deg,
    }


# ------------------------------------------------------------------------------------- E3


def analyze_drift(root: Path) -> dict[str, Any] | None:
    src = root / "results" / "scores" / "drift"
    files = sorted(src.glob("*__seed*.npz"))
    if not files:
        return None
    labels = load_labels(src / "labels__cicids2017.npz")
    rows, cov_rows = [], []
    for path in files:
        cell = load_cell(path)
        cal = cell["scores"]["cal"]
        for day in ("Tue", "Wed", "Thu", "Fri"):
            y = labels[f"{day}__is_attack"].astype(int)
            normal = y == 0
            ev = cell["scores"][day]
            for i, spec in enumerate(cell["specs"]):
                if not (_valid(ev[i]) and _valid(cal[i])):
                    continue
                p = conformal.conformal_pvalues(cal[i], ev[i])
                row: dict[str, Any] = {
                    "seed": cell["meta"]["seed"], "day": day, "detector": spec,
                    "n_attack": int(y.sum()), "n_normal": int(normal.sum()),
                    "psi_benign": metrics.population_stability_index(cal[i], ev[i][normal]),
                    "fpr_a1": conformal.empirical_fpr(p[normal], 0.01),
                    "fpr_a5": conformal.empirical_fpr(p[normal], 0.05),
                }  # fmt: skip
                if y.sum() > 20:
                    row["roc"] = metrics.roc_auc(y, ev[i])
                    row["tpr_a5"] = conformal.tpr_at_alpha(p[~normal], 0.05)
                rows.append(row)
                if day in ("Tue", "Fri"):
                    for a in COVERAGE_GRID:
                        cov_rows.append({"seed": cell["meta"]["seed"], "day": day, "detector": spec,
                                         "alpha": float(a), "fpr": conformal.empirical_fpr(p[normal], a)})  # fmt: skip
    df = pd.DataFrame(rows)
    out_dir = root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "drift_long.csv", index=False)
    agg = df.groupby(["detector", "day"], as_index=False).mean(numeric_only=True).drop(columns=["seed"])
    # label-free drift alarm: does benign-score PSI predict calibration breakdown (FPR >= 3x nominal)?
    valid = df.dropna(subset=["psi_benign", "fpr_a1"])
    rho_s = spearmanr(valid["psi_benign"], valid["fpr_a1"]) if len(valid) > 5 else None
    broken = valid["fpr_a1"] >= 0.03
    alarms = []
    for thr in (0.05, 0.10, 0.25):
        alarm = valid["psi_benign"] >= thr
        tp, fp = int((alarm & broken).sum()), int((alarm & ~broken).sum())
        fn = int((~alarm & broken).sum())
        alarms.append({"threshold_psi": thr, "tp": tp, "fp": fp, "fn": fn,
                       "precision": tp / (tp + fp) if tp + fp else None,
                       "recall": tp / (tp + fn) if tp + fn else None})  # fmt: skip
    return {
        "table": json.loads(agg.round(5).to_json(orient="records")),
        "psi_fpr_spearman": None if rho_s is None else {"rho": float(rho_s.statistic), "p": float(rho_s.pvalue)},
        "drift_alarm": alarms[2],
        "drift_alarms": alarms,
        "coverage": json.loads(
            pd.DataFrame(cov_rows).groupby(["detector", "day", "alpha"], as_index=False)["fpr"].mean().round(5).to_json(orient="records")
        ),
    }


# ------------------------------------------------------------------------------------- E4


def analyze_transfer(root: Path) -> dict[str, Any] | None:
    src = root / "results" / "scores" / "transfer"
    files = sorted(src.glob("*__seed*.npz"))
    if not files:
        return None
    rows = []
    for path in files:
        cell = load_cell(path)
        m = cell["meta"]
        labels = load_labels(src / f"labels__{m['target']}.npz")
        extra = {"source": m["source"], "target": m["target"], "scaler": m["scaler"], "seed": m["seed"]}
        r, _, _ = cell_rows(cell, labels, "target", labels["is_attack"].astype(int), None, extra)
        rows += r
    df = pd.DataFrame(rows)
    out_dir = root / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "transfer_long.csv", index=False)
    keys = ["source", "target", "scaler", "detector"]
    agg = (
        df.groupby(keys)
        .agg(roc=("roc", "mean"), roc_std=("roc", "std"), pr=("pr", "mean"), a1_fpr=("a1_fpr", "mean"),
             a5_fpr=("a5_fpr", "mean"), a5_tpr=("a5_tpr", "mean"), n_seeds=("seed", "nunique"))
        .reset_index()
    )  # fmt: skip
    agg["roc_std"] = agg["roc_std"].fillna(0.0)
    inds = agg[agg.source == agg.target].set_index(["target", "detector"])["roc"]
    ret = []
    for r in agg[agg.source != agg.target].itertuples():
        base = inds.get((r.target, r.detector))
        ret.append({"source": r.source, "target": r.target, "scaler": r.scaler, "detector": r.detector,
                    "roc": r.roc, "in_dataset_roc": None if base is None else float(base),
                    "retention": None if base is None else float(r.roc / base)})  # fmt: skip
    return {"table": json.loads(agg.round(5).to_json(orient="records")), "retention": ret}


def analyze_all(root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"titles": TITLES}
    for name, fn in (("main", analyze_main), ("contamination", analyze_contamination),
                     ("drift", analyze_drift), ("transfer", analyze_transfer)):  # fmt: skip
        logger.info("analyzing %s ...", name)
        res = fn(root)
        if res is not None:
            summary[name] = res
    out = root / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
