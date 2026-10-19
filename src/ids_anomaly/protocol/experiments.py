"""Experiment cells (E1-E4): fit detectors, score evaluation sets, persist raw scores.

Design: *compute* and *analysis* are decoupled. A cell fits a list of detector specs on one
training set and stores the raw score vectors for every named evaluation set (``cal``, ``test``,
per-day sets, ...) in a compressed ``.npz``. Everything reported in the paper (AUC, operating
points, conformal coverage, drift curves, statistics) is recomputed from those files by
:mod:`ids_anomaly.protocol.analysis`, so a metric can be added or fixed without re-training.

Cells are independent, resumable (finished files are skipped) and fault-isolated (a detector that
raises yields NaN scores and an error string instead of killing the run).
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from joblib import Parallel, delayed

from ids_anomaly.data import cicids, harmonized
from ids_anomaly.data.registry import Bundle, load_dataset
from ids_anomaly.data.tabular import build_tabular_preprocessor, stratified_cap
from ids_anomaly.detectors import REGISTRY
from ids_anomaly.detectors.purify import build_spec
from ids_anomaly.protocol.trainsets import make_fit_sets

logger = logging.getLogger(__name__)

ALL_DETECTORS: tuple[str, ...] = tuple(REGISTRY)
DATASETS: tuple[str, ...] = ("nsl_kdd", "unsw_nb15", "cicids2017")
CIC_TEST_CAP = 120_000

# E2 -----------------------------------------------------------------------------------------
E2_RHOS: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20)
E2_N_FIT = 30_000
E2_BASE: tuple[str, ...] = (
    "iforest", "pca", "gmm", "knn", "ecod", "ae", "vae", "deep_svdd", "flow", "neutral",
)  # fmt: skip
E2_CTR: tuple[str, ...] = tuple(f"{b}+ctr0.1" for b in ("ae", "vae", "flow", "knn", "gmm", "deep_svdd", "neutral"))
E2_ORACLE: tuple[str, ...] = tuple(f"{b}+ctr@rho" for b in ("ae", "knn", "flow"))
E2_ABLATION: tuple[str, ...] = (
    "ae+ctr0.1[iforest]", "ae+ctr0.1[ecod]", "ae+ctr0.1[pca]", "ae+ctr0.05", "ae+ctr0.2",
)  # fmt: skip


def e2_specs(rho: float) -> list[str]:
    specs = [*E2_BASE, *E2_CTR, *E2_ORACLE]
    if rho in (0.05, 0.20):
        specs += list(E2_ABLATION)
    return specs


@dataclass(frozen=True)
class Cell:
    experiment: str
    dataset: str
    seed: int
    variant: str = ""  # regime / rho / transfer mode: anything that distinguishes cells
    params: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    @property
    def key(self) -> str:
        v = f"__{self.variant}" if self.variant else ""
        return f"{self.dataset}{v}__seed{self.seed}"

    def p(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)


# --------------------------------------------------------------------------------- scoring core


def score_cell(
    specs: list[str], X_fit: np.ndarray, eval_sets: dict[str, np.ndarray], seed: int, rho: float = 0.0
) -> dict[str, Any]:
    """Fit every spec on ``X_fit`` and score each evaluation set; failures become NaN rows."""
    scores = {k: np.full((len(specs), len(v)), np.nan, dtype=np.float32) for k, v in eval_sets.items()}
    fit_s, score_s = np.zeros(len(specs)), np.zeros(len(specs))
    errors: dict[str, str] = {}
    for i, spec in enumerate(specs):
        try:
            det = build_spec(spec, seed=seed, rho=rho)
            t0 = time.perf_counter()
            det.fit(X_fit)
            fit_s[i] = time.perf_counter() - t0
            t1 = time.perf_counter()
            for name, X in eval_sets.items():
                scores[name][i] = det.score(X)
            score_s[i] = time.perf_counter() - t1
            logger.info("  %-22s fit %6.1fs score %6.1fs", spec, fit_s[i], score_s[i])
        except Exception:  # noqa: BLE001 - isolate one detector's failure from the whole cell
            errors[spec] = traceback.format_exc(limit=3)
            logger.error("  %s FAILED\n%s", spec, errors[spec])
    return {"scores": scores, "fit_seconds": fit_s, "score_seconds": score_s, "errors": errors}


def _tmp_path(path: Path) -> Path:
    """Per-process temp name: parallel workers may race to write the same shared labels file."""
    return path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")


def save_cell(path: Path, specs: list[str], result: dict[str, Any], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"scores__{k}": v for k, v in result["scores"].items()}
    tmp = _tmp_path(path)
    np.savez_compressed(
        tmp,
        specs=np.array(specs),
        fit_seconds=result["fit_seconds"],
        score_seconds=result["score_seconds"],
        meta=np.array(json.dumps({**meta, "errors": result["errors"]})),
        **arrays,
    )
    tmp.replace(path)


def save_labels(path: Path, **arrays: Any) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    # pandas>=3 string columns come out as object arrays, which npz would pickle (and analysis,
    # which loads with allow_pickle=False, could not read back): store them as fixed-width text.
    arrays = {k: (np.asarray(v).astype(str) if np.asarray(v).dtype == object else v) for k, v in arrays.items()}
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)  # atomic; concurrent workers writing identical content are harmless


# ----------------------------------------------------------------------------------- data cache


@lru_cache(maxsize=4)
def _bundle(dataset: str, raw_dir: str, variant: str = "") -> Bundle:
    kwargs: dict[str, Any] = {}
    if dataset == "cicids2017":
        kwargs = {"test_cap": CIC_TEST_CAP}
        if variant == "iid":
            kwargs["split"] = "iid"
    return load_dataset(dataset, Path(raw_dir), **kwargs)


# -------------------------------------------------------------------------------------- cells


def run_main_cell(cell: Cell, raw_dir: Path, out_dir: Path) -> None:
    """E1: one dataset x regime (clean | natural) x seed, all detectors."""
    regime = cell.p("regime")
    bundle = _bundle(cell.dataset, str(raw_dir))
    fs = make_fit_sets(
        bundle.train, regime, n_fit=100_000 if regime == "clean" else None, seed=cell.seed
    )
    test = bundle.test
    save_labels(
        out_dir / f"labels__{cell.dataset}.npz",
        is_attack=test.is_attack, category=test.attack_category.astype(str),
        group=np.array([]) if test.group is None else test.group.astype(str),
    )  # fmt: skip
    specs = list(cell.p("detectors", ALL_DETECTORS))
    res = score_cell(specs, fs.X_fit, {"cal": fs.X_cal, "test": test.X}, cell.seed)
    meta = {"experiment": "main", "dataset": cell.dataset, "regime": regime, "seed": cell.seed,
            "n_fit": len(fs.X_fit), "contamination": fs.contamination, "n_test": len(test)}  # fmt: skip
    save_cell(out_dir / f"{cell.key}.npz", specs, res, meta)


def run_contamination_cell(cell: Cell, raw_dir: Path, out_dir: Path) -> None:
    """E2: attacks injected into a normal base at exact rate rho; vanilla vs CTR detectors."""
    rho = float(cell.p("rho"))
    variant = "iid" if cell.dataset == "cicids2017" else ""
    bundle = _bundle(cell.dataset, str(raw_dir), variant)
    fs = make_fit_sets(bundle.train, "injected", rho=rho, n_fit=E2_N_FIT, seed=cell.seed)
    test = bundle.test
    save_labels(
        out_dir / f"labels__{cell.dataset}.npz",
        is_attack=test.is_attack, category=test.attack_category.astype(str),
    )  # fmt: skip
    specs = e2_specs(rho)
    res = score_cell(specs, fs.X_fit, {"cal": fs.X_cal, "test": test.X}, cell.seed, rho=rho)
    meta = {"experiment": "contamination", "dataset": cell.dataset, "rho": rho, "seed": cell.seed,
            "n_fit": len(fs.X_fit), "contamination": fs.contamination, "n_test": len(test)}  # fmt: skip
    save_cell(out_dir / f"{cell.key}.npz", specs, res, meta)


def run_drift_cell(cell: Cell, raw_dir: Path, out_dir: Path) -> None:
    """E3: fit on Monday's attack-free traffic; evaluate Tue-Fri with a Monday-calibrated threshold."""
    from ids_anomaly.data import fetch

    fetch.fetch_cicids(raw_dir)
    monday, days, numeric = _drift_frames(str(raw_dir))
    rng = np.random.default_rng(cell.seed)
    perm = rng.permutation(len(monday))
    n_cal = int(0.15 * len(monday))
    pre = build_tabular_preprocessor(numeric, cicids.CATEGORICAL)
    features = numeric + cicids.CATEGORICAL
    X_all = pre.fit_transform(monday[features].iloc[perm[n_cal:]]).astype(np.float32)
    eval_sets = {"cal": pre.transform(monday[features].iloc[perm[:n_cal]]).astype(np.float32)}
    label_arrays: dict[str, np.ndarray] = {}
    for day, frame in days.items():
        eval_sets[day] = pre.transform(frame[features]).astype(np.float32)
        label_arrays[f"{day}__is_attack"] = (frame["category"] != "normal").to_numpy()
        label_arrays[f"{day}__category"] = frame["category"].to_numpy(dtype=str)
        label_arrays[f"{day}__timestamp"] = frame["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    save_labels(out_dir / "labels__cicids2017.npz", **label_arrays)
    specs = list(cell.p("detectors", ALL_DETECTORS))
    res = score_cell(specs, X_all, eval_sets, cell.seed)
    meta = {"experiment": "drift", "dataset": "cicids2017", "seed": cell.seed, "n_fit": len(X_all)}
    save_cell(out_dir / f"{cell.key}.npz", specs, res, meta)


@lru_cache(maxsize=1)
def _drift_frames(raw_dir: str):
    return cicids.load_cicids_by_day(Path(raw_dir))


@lru_cache(maxsize=4)
def _harmonized(dataset: str, raw_dir: str):
    return harmonized.load_harmonized(dataset, Path(raw_dir))


def run_transfer_cell(cell: Cell, raw_dir: Path, out_dir: Path) -> None:
    """E4: train on one dataset's clean harmonized normals, detect on another (or itself)."""
    from ids_anomaly.data import fetch

    fetch.fetch_unsw(raw_dir)
    fetch.fetch_cicids(raw_dir)
    source, target, mode = cell.p("source"), cell.p("target"), cell.p("scaler")
    src_train, src_test = _harmonized(source, str(raw_dir))
    tgt_train, tgt_test = (src_train, src_test) if target == source else _harmonized(target, str(raw_dir))
    rng = np.random.default_rng(cell.seed)

    normal = np.flatnonzero(~src_train.is_attack)
    normal = rng.permutation(normal)
    n_cal = min(15_000, int(0.15 * len(normal)))
    cal_idx, fit_idx = normal[:n_cal], normal[n_cal:][:60_000]

    src_scaler = harmonized.fit_scaler(src_train)
    X_all_src = harmonized.transform(src_train, src_scaler)
    X_fit, X_cal = X_all_src[fit_idx], X_all_src[cal_idx]

    tgt_eval = tgt_test
    keep = stratified_cap(tgt_eval.attack_category, CIC_TEST_CAP, rng)
    sub = harmonized.RawSplit(
        tgt_eval.features.iloc[keep].reset_index(drop=True),
        tgt_eval.is_attack[keep], tgt_eval.attack_category[keep],
        None if tgt_eval.group is None else tgt_eval.group[keep],
    )  # fmt: skip
    # "target" mode standardizes the target with its own unlabeled statistics (domain-wise).
    tgt_scaler = src_scaler if mode == "source" else harmonized.fit_scaler(sub)
    X_tgt = harmonized.transform(sub, tgt_scaler)

    save_labels(
        out_dir / f"labels__{target}.npz",
        is_attack=sub.is_attack, category=sub.attack_category.astype(str),
    )  # fmt: skip
    specs = list(cell.p("detectors", ALL_DETECTORS))
    res = score_cell(specs, X_fit, {"cal": X_cal, "target": X_tgt}, cell.seed)
    meta = {"experiment": "transfer", "source": source, "target": target, "scaler": mode,
            "seed": cell.seed, "n_fit": len(X_fit), "n_test": len(X_tgt)}  # fmt: skip
    save_cell(out_dir / f"{cell.key}.npz", specs, res, meta)


# ---------------------------------------------------------------------------------- cell grids


def main_cells(seeds: int, datasets: tuple[str, ...] = DATASETS) -> list[Cell]:
    return [
        Cell("main", d, s, regime, (("regime", regime),))
        for s in range(seeds) for d in datasets for regime in ("clean", "natural")
    ]  # fmt: skip  (seed-major: a partial run is still balanced across datasets)


def contamination_cells(seeds: int, datasets: tuple[str, ...] = DATASETS) -> list[Cell]:
    return [
        Cell("contamination", d, s, f"rho{rho:g}", (("rho", rho),))
        for s in range(seeds) for d in datasets for rho in E2_RHOS
    ]  # fmt: skip


def drift_cells(seeds: int) -> list[Cell]:
    return [Cell("drift", "cicids2017", s) for s in range(seeds)]


def transfer_cells(seeds: int) -> list[Cell]:
    pairs = [("unsw_nb15", "unsw_nb15", "source"), ("cicids2017", "cicids2017", "source")]
    for a, b in (("unsw_nb15", "cicids2017"), ("cicids2017", "unsw_nb15")):
        pairs += [(a, b, "source"), (a, b, "target")]
    return [
        Cell("transfer", f"{a}_to_{b}", s, mode, (("source", a), ("target", b), ("scaler", mode)))
        for s in range(seeds) for a, b, mode in pairs
    ]  # fmt: skip


RUNNERS: dict[str, Callable[[Cell, Path, Path], None]] = {
    "main": run_main_cell,
    "contamination": run_contamination_cell,
    "drift": run_drift_cell,
    "transfer": run_transfer_cell,
}
GRIDS: dict[str, Callable[..., list[Cell]]] = {
    "main": main_cells,
    "contamination": contamination_cells,
    "drift": drift_cells,
    "transfer": transfer_cells,
}


def _run_one(cell: Cell, raw_dir: str, out_dir: str) -> str:
    logging.getLogger("ids_anomaly").setLevel(logging.INFO)
    out = Path(out_dir) / f"{cell.key}.npz"
    if out.exists():
        return f"skip {cell.key}"
    t0 = time.perf_counter()
    logger.info("[%s] %s start", cell.experiment, cell.key)
    RUNNERS[cell.experiment](cell, Path(raw_dir), Path(out_dir))
    return f"done {cell.key} in {time.perf_counter() - t0:.0f}s"


def run_experiment(
    experiment: str,
    root: Path,
    seeds: int = 5,
    n_jobs: int = 3,
    datasets: tuple[str, ...] | None = None,
) -> list[str]:
    """Run (or resume) every cell of one experiment, writing to ``results/scores/<experiment>/``."""
    cells = GRIDS[experiment](seeds, datasets) if datasets and experiment in ("main", "contamination") else GRIDS[experiment](seeds)
    out_dir = root / "results" / "scores" / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = root / "data" / "raw"
    logger.info("%s: %d cells, %d workers", experiment, len(cells), n_jobs)
    messages = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_run_one)(c, str(raw_dir), str(out_dir)) for c in cells
    )
    return list(messages)
