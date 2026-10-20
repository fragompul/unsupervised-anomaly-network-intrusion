"""Build the compact artifacts the Noctua web app loads.

Everything is a static file: JSON for structure, little-endian typed binaries for the point cloud
(quantized to int16/int8/uint8 so a 12k-point, 16-detector, 120-feature dataset is ~3 MB). The
app therefore runs from GitHub Pages with no server, including the in-browser *playground*, which
re-implements the autoencoder forward pass in JavaScript from the exported weights.

Per dataset (``<out>/<key>/``):

* ``embedding_pca.i16`` / ``embedding_umap.i16`` -- (n, 3) int16, coordinates scaled to [-1, 1];
* ``scores.u8``      -- (n_detectors, n) uint8: ``round(40 * -log10(conformal p-value))``;
* ``features.i8``    -- (n, d) int8: standardized features clipped to +/-8, scaled by 127/8;
* ``attribution.u8`` -- (n, 10) uint8: top-5 AE feature indices, then their share of the error;
* ``points.json``    -- category / attack / day index per point;
* ``ae.json``        -- autoencoder weights + calibration quantiles for the playground;
* ``curves.json``    -- ROC / PR curves and calibration data per detector.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_curve, roc_curve

from ids_anomaly.data.preprocess import Dataset
from ids_anomaly.data.tabular import stratified_cap
from ids_anomaly.detectors import REGISTRY, TITLES
from ids_anomaly.detectors.deep import AutoencoderDetector
from ids_anomaly.explain import attribution
from ids_anomaly.protocol import conformal

logger = logging.getLogger(__name__)

SCORE_SCALE = 40.0  # uint8 = round(40 * -log10 p): resolves p down to ~1e-6
FEATURE_CLIP = 8.0
CAL_QUANTILES = 2000


def _scale_coords(Z: np.ndarray) -> np.ndarray:
    Z = Z - np.median(Z, axis=0)
    bound = np.percentile(np.abs(Z), 99.5) or 1.0
    return np.clip(Z / bound, -1.2, 1.2) / 1.2


def _to_i16(Z: np.ndarray) -> np.ndarray:
    return np.round(np.clip(Z, -1, 1) * 32767).astype("<i2")


def nlogp_u8(pvalues: np.ndarray) -> np.ndarray:
    return np.clip(np.round(-np.log10(np.clip(pvalues, 1e-12, 1.0)) * SCORE_SCALE), 0, 255).astype(np.uint8)


def _umap_3d(X: np.ndarray, seed: int) -> np.ndarray:
    import umap

    return umap.UMAP(
        n_components=3, n_neighbors=30, min_dist=0.1, init="random", random_state=seed, n_jobs=1
    ).fit_transform(X)


def _mlp_weights(model: Any) -> list[dict[str, Any]]:
    """Dense layers of the autoencoder as plain nested lists; ``relu`` marks layers followed by ReLU."""
    import torch

    modules = list(model.enc) + list(model.dec)
    layers = []
    for i, m in enumerate(modules):
        if isinstance(m, torch.nn.Linear):
            followed_by_relu = i + 1 < len(modules) and isinstance(modules[i + 1], torch.nn.ReLU)
            layers.append(
                {
                    "w": np.round(m.weight.detach().numpy().astype(np.float64), 5).tolist(),
                    "b": np.round(m.bias.detach().numpy().astype(np.float64), 5).tolist(),
                    "relu": followed_by_relu,
                }
            )
    return layers


def _curve_points(y: np.ndarray, s: np.ndarray) -> dict[str, list[float]]:
    fpr, tpr, _ = roc_curve(y, s)
    grid = np.unique(np.concatenate([np.geomspace(1e-4, 1.0, 60), np.linspace(0, 1, 40)]))
    prec, rec, _ = precision_recall_curve(y, s)
    rgrid = np.linspace(0, 1, 60)
    return {
        "fpr": np.round(grid, 5).tolist(),
        "tpr": np.round(np.interp(grid, fpr, tpr), 4).tolist(),
        "recall": np.round(rgrid, 4).tolist(),
        "precision": np.round(np.interp(rgrid, rec[::-1], prec[::-1]), 4).tolist(),
    }


def export_dataset(
    key: str,
    title: str,
    test: Dataset,
    train_normal: np.ndarray,
    cell: dict[str, Any],
    out_dir: Path,
    n_points: int = 12_000,
    seed: int = 0,
    with_umap: bool = True,
) -> dict[str, Any]:
    """Write all artifacts of one dataset and return its manifest entry."""
    t0 = time.perf_counter()
    ddir = out_dir / key
    ddir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    specs: list[str] = cell["specs"]
    cal, ev = cell["scores"]["cal"], cell["scores"]["test"]
    if ev.shape[1] != len(test):
        raise ValueError(f"{key}: stored scores ({ev.shape[1]}) do not align with test set ({len(test)})")
    y = test.is_attack.astype(int)

    idx = stratified_cap(test.attack_category, n_points, rng, keep_whole_below=400)
    X = test.X[idx]
    categories = sorted(set(test.attack_category.tolist()), key=lambda c: (c != "normal", c))
    cat_index = {c: i for i, c in enumerate(categories)}

    # ---- scores: conformal -log10 p per detector, plus the two ensembles
    ok = [i for i in range(len(specs)) if np.isfinite(ev[i]).all() and np.isfinite(cal[i]).all()]
    pvals = {specs[i]: conformal.conformal_pvalues(cal[i], ev[i]) for i in ok}
    names = [specs[i] for i in ok]
    ens: dict[str, np.ndarray] = {}
    for ens_name, members in (("ens_all", names), ("ens_diverse", [n for n in names if n in ("iforest", "knn", "ae", "flow", "neutral")])):
        if len(members) >= 2:
            ens[ens_name] = conformal.merge_pvalues_average(np.vstack([pvals[m] for m in members]))
    all_p = {**pvals, **ens}
    det_names = list(all_p)
    score_mat = np.vstack([nlogp_u8(all_p[n][idx]) for n in det_names])
    score_mat.astype(np.uint8).tofile(ddir / "scores.u8")

    # ---- embeddings (transductive, for visualization only)
    pca = PCA(n_components=3, random_state=seed).fit_transform(X)
    _to_i16(_scale_coords(pca)).tofile(ddir / "embedding_pca.i16")
    embeddings = ["pca"]
    if with_umap:
        _to_i16(_scale_coords(_umap_3d(X, seed))).tofile(ddir / "embedding_umap.i16")
        embeddings.append("umap")

    # ---- features (standardized, int8)
    np.round(np.clip(X, -FEATURE_CLIP, FEATURE_CLIP) * (127.0 / FEATURE_CLIP)).astype(np.int8).tofile(
        ddir / "features.i8"
    )

    # ---- points metadata
    points: dict[str, Any] = {
        "category": [cat_index[c] for c in test.attack_category[idx]],
        "attack": test.is_attack[idx].astype(int).tolist(),
        "label": test.label[idx].astype(str).tolist() if (test.label != test.attack_category).any() else None,
    }
    if test.group is not None:
        days = sorted(set(test.group.tolist()))
        points["group_names"] = days
        points["group"] = [days.index(g) for g in test.group[idx]]
    (ddir / "points.json").write_text(json.dumps(points, separators=(",", ":")))

    # ---- showcase autoencoder: attributions, faithfulness, playground weights
    ae = AutoencoderDetector(seed=seed).fit(train_normal)
    errors = ae.feature_errors(X)
    top_idx, top_val = attribution.top_k_features(errors, k=5)
    ae_scores_sample = ae.score(X)
    # 10 bytes/point: top-5 feature indices (uint8) then each one's share of the total error (uint8/255)
    total = np.maximum(errors.sum(axis=1, keepdims=True), 1e-12)
    share = np.round(np.clip(top_val / total, 0, 1) * 255).astype(np.uint8)
    if errors.shape[1] > 256:
        raise ValueError("attribution.u8 stores feature indices as uint8")
    np.concatenate([top_idx.astype(np.uint8), share], axis=1).tofile(ddir / "attribution.u8")
    flagged = np.flatnonzero(all_p.get("ae", all_p[det_names[0]])[idx] <= 0.05)
    eval_rows = X[flagged[:600]] if len(flagged) >= 10 else X[:600]
    faithfulness = attribution.deletion_curve(ae, eval_rows)
    nmi = attribution.attribution_cluster_nmi(
        errors[flagged], test.attack_category[idx][flagged]
    ) if len(flagged) >= 30 else float("nan")

    cal_sorted = np.sort(cal[specs.index("ae")]) if "ae" in specs else np.sort(ae.score(train_normal[:5000]))
    q = np.quantile(cal_sorted, (np.arange(CAL_QUANTILES) + 0.5) / CAL_QUANTILES)
    (ddir / "ae.json").write_text(
        json.dumps(
            {
                "layers": _mlp_weights(ae.model),
                "cal_scores": np.round(q, 6).tolist(),
                "n_cal": int(len(cal_sorted)),
                "median_error_per_feature": np.round(np.median(ae.feature_errors(train_normal[:5000]), axis=0), 6).tolist(),
            },
            separators=(",", ":"),
        )
    )

    # ---- curves from the full test set
    curves = {n: _curve_points(y, nl) for n, nl in ((n, -np.log10(np.clip(all_p[n], 1e-12, 1))) for n in det_names)}
    coverage = {}
    for n in det_names:
        grid, fpr = conformal.coverage_curve(all_p[n][y == 0])
        coverage[n] = {"alpha": np.round(grid, 4).tolist(), "fpr": np.round(fpr, 5).tolist()}
    (ddir / "curves.json").write_text(json.dumps({"curves": curves, "coverage": coverage}, separators=(",", ":")))

    manifest = {
        "key": key,
        "title": title,
        "n_points": int(len(idx)),
        "n_test": int(len(test)),
        "n_features": int(X.shape[1]),
        "feature_names": [n.split("__", 1)[-1] for n in test.feature_names],
        "categories": categories,
        "embeddings": embeddings,
        "detectors": [
            {"id": n, "title": _display(n), "family": _family(n)} for n in det_names
        ],
        "score_scale": SCORE_SCALE,
        "feature_clip": FEATURE_CLIP,
        "attack_fraction": float(test.is_attack.mean()),
        "explain": {
            "deletion": faithfulness,
            "attribution_cluster_nmi": None if np.isnan(nmi) else nmi,
            "n_flagged_eval": int(len(flagged)),
        },
        "ae_sample_mean_error": float(ae_scores_sample.mean()),
    }
    logger.info("exported %s in %.0fs", key, time.perf_counter() - t0)
    return manifest


def _display(spec: str) -> str:
    if spec == "ens_all":
        return "Ensemble · all"
    if spec == "ens_diverse":
        return "Ensemble · diverse 5"
    base, plus, rest = spec.partition("+")
    return TITLES.get(base, base) + (f" +{rest}" if plus else "")


def _family(spec: str) -> str:
    if spec.startswith("ens_"):
        return "ensemble"
    base = spec.partition("+")[0]
    return REGISTRY[base].family if base in REGISTRY else "other"


REPLAY_DETECTORS = ("ae", "iforest", "flow", "knn", "ens_all")
REPLAY_PER_DAY = 1500


def export_replay(drift_cell: dict[str, Any], labels: dict[str, np.ndarray], out_dir: Path, seed: int = 0) -> dict[str, Any]:
    """A time-ordered Tue-Fri stream with conformal scores calibrated on Monday's benign traffic."""
    rng = np.random.default_rng(seed)
    specs = drift_cell["specs"]
    cal = drift_cell["scores"]["cal"]
    use = [d for d in REPLAY_DETECTORS if d == "ens_all" or d in specs]
    days = ["Tue", "Wed", "Thu", "Fri"]
    t_all: list[np.ndarray] = []
    attack_all: list[np.ndarray] = []
    day_all: list[np.ndarray] = []
    rows: dict[str, list[np.ndarray]] = {d: [] for d in use}
    ens_members = [i for i, s in enumerate(specs) if np.isfinite(drift_cell["scores"]["Tue"][i]).all()]
    for di, day in enumerate(days):
        ts = labels[f"{day}__timestamp"]
        order = np.argsort(ts, kind="stable")
        keep = np.sort(rng.choice(len(order), size=min(REPLAY_PER_DAY, len(order)), replace=False))
        sel = order[keep]
        t_all.append(ts[sel])
        attack_all.append(labels[f"{day}__is_attack"][sel].astype(int))
        day_all.append(np.full(len(sel), di))
        ev = drift_cell["scores"][day]
        pm = np.vstack([conformal.conformal_pvalues(cal[i], ev[i][sel]) for i in ens_members])
        for d in use:
            if d == "ens_all":
                p = conformal.merge_pvalues_average(pm)
            else:
                p = pm[ens_members.index(specs.index(d))]
            rows[d].append(nlogp_u8(p))
    t = np.concatenate(t_all)
    payload = {
        "days": days,
        "detectors": [{"id": d, "title": _display(d)} for d in use],
        "t": (t - t.min()).astype(int).tolist(),
        "attack": np.concatenate(attack_all).tolist(),
        "day": np.concatenate(day_all).tolist(),
        "nlogp": {d: np.concatenate(v).tolist() for d, v in rows.items()},
        "score_scale": SCORE_SCALE,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "replay.json").write_text(json.dumps(payload, separators=(",", ":")))
    return {"n": int(len(t)), "detectors": use}
