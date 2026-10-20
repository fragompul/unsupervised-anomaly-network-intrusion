"""End-to-end test of the analysis stage on synthetic stored cells (no training involved)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ids_anomaly.protocol import analysis, experiments
from ids_anomaly.protocol.trainsets import make_fit_sets

DETS = ["iforest", "knn", "ae", "flow", "neutral", "pca"]


def _scores(rng, y, shift, n_det=6):
    base = rng.normal(size=(n_det, len(y)))
    base[:, y == 1] += shift
    return base.astype(np.float32)


def _write_cell(path, rng, y, shift, meta, eval_name="test", extra_sets=None, specs=DETS):
    cal = rng.normal(size=(len(specs), 800)).astype(np.float32)
    sets = {"cal": cal, eval_name: _scores(rng, y, shift, len(specs))}
    if extra_sets:
        sets.update(extra_sets)
    res = {"scores": sets, "fit_seconds": np.ones(len(specs)), "score_seconds": np.ones(len(specs)) / 10, "errors": {}}
    experiments.save_cell(path, list(specs), res, meta)


@pytest.fixture
def fake_root(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1500), np.ones(500)].astype(int)
    cats = np.where(y == 1, rng.choice(["dos", "probe"], len(y)), "normal")
    # E1
    d = tmp_path / "results" / "scores" / "main"
    for ds in ("a", "b"):
        experiments.save_labels(d / f"labels__{ds}.npz", is_attack=y.astype(bool), category=cats)
        for regime in ("clean", "natural"):
            for seed in range(3):
                shift = 2.5 if regime == "clean" else 1.5
                meta = {"dataset": ds, "regime": regime, "seed": seed, "contamination": 0.0, "n_fit": 10}
                _write_cell(d / f"{ds}__{regime}__seed{seed}.npz", rng, y, shift + 0.1 * seed, meta)
    # E2
    d = tmp_path / "results" / "scores" / "contamination"
    experiments.save_labels(d / "labels__a.npz", is_attack=y.astype(bool), category=cats)
    specs = ["ae", "ae+ctr0.1", "knn", "knn+ctr0.1"]
    for rho in (0.0, 0.05, 0.2):
        for seed in range(2):
            shift = 3.0 - 6.0 * rho
            meta = {"dataset": "a", "rho": rho, "seed": seed, "n_fit": 10, "contamination": rho}
            _write_cell(d / f"a__rho{rho:g}__seed{seed}.npz", rng, y, shift, meta, specs=specs)
    # E3
    d = tmp_path / "results" / "scores" / "drift"
    lab = {}
    sets = {}
    for day in ("Tue", "Wed", "Thu", "Fri"):
        lab[f"{day}__is_attack"] = y.astype(bool)
        lab[f"{day}__category"] = cats
        sets[day] = _scores(rng, y, 2.0) + (0.5 if day == "Fri" else 0.0)
    experiments.save_labels(d / "labels__cicids2017.npz", **lab)
    for seed in range(2):
        _write_cell(d / f"cicids2017__seed{seed}.npz", rng, y, 2.0, {"dataset": "cicids2017", "seed": seed},
                    eval_name="Tue", extra_sets={k: v for k, v in sets.items() if k != "Tue"})  # fmt: skip
    # E4
    d = tmp_path / "results" / "scores" / "transfer"
    experiments.save_labels(d / "labels__a.npz", is_attack=y.astype(bool), category=cats)
    experiments.save_labels(d / "labels__b.npz", is_attack=y.astype(bool), category=cats)
    for src, tgt, shift in (("a", "a", 3.0), ("b", "b", 3.0), ("a", "b", 1.0)):
        for scaler in ("source", "target") if src != tgt else ("source",):
            for seed in range(2):
                meta = {"source": src, "target": tgt, "scaler": scaler, "seed": seed}
                _write_cell(d / f"{src}_to_{tgt}__{scaler}__seed{seed}.npz", rng, y, shift, meta, eval_name="target")
    return tmp_path


def test_full_analysis_produces_all_sections(fake_root: Path):
    summary = analysis.analyze_all(fake_root)
    assert {"main", "contamination", "drift", "transfer"} <= set(summary)
    json.dumps(summary, default=float)  # must be serializable
    assert (fake_root / "results" / "analysis" / "summary.json").exists()


def test_main_table_has_ensembles_and_sane_auc(fake_root: Path):
    main = analysis.analyze_main(fake_root)
    assert main is not None
    dets = {r["detector"] for r in main["table"]}
    assert {"ens_all", "ens_diverse", "ae"} <= dets
    clean = [r for r in main["table"] if r["regime"] == "clean" and r["detector"] == "ae"]
    natural = [r for r in main["table"] if r["regime"] == "natural" and r["detector"] == "ae"]
    assert clean[0]["roc"] > natural[0]["roc"] > 0.7
    assert clean[0]["n_seeds"] == 3 and "roc_ci" in clean[0]
    st = main["stats"]
    assert st["all_seeds"]["n_blocks"] == 12 and st["best_by_rank"] in DETS
    assert all(0 <= r["recall_a5"] <= 1 for r in main["categories"])


def test_conformal_coverage_is_nominal_for_iid_scores(fake_root: Path):
    main = analysis.analyze_main(fake_root)
    pts = [r for r in main["coverage"] if r["detector"] == "ae" and abs(r["alpha"] - 0.05) < 0.003]
    assert pts and all(abs(r["fpr"] - 0.05) < 0.03 for r in pts)


def test_contamination_summary_computes_paired_gain_and_degradation(fake_root: Path):
    res = analysis.analyze_contamination(fake_root)
    assert res is not None
    assert {g["base"] for g in res["ctr_gain"]} == {"ae", "knn"}
    deg = {(d["dataset"], d["detector"]): d for d in res["degradation"]}
    assert deg[("a", "ae")]["drop"] > 0  # signal shrinks with rho in the fake data


def test_drift_summary_detects_shifted_day(fake_root: Path):
    res = analysis.analyze_drift(fake_root)
    assert res is not None
    fri = [r for r in res["table"] if r["day"] == "Fri" and r["detector"] == "ae"][0]
    tue = [r for r in res["table"] if r["day"] == "Tue" and r["detector"] == "ae"][0]
    assert fri["psi_benign"] >= tue["psi_benign"] or fri["fpr_a1"] >= tue["fpr_a1"]
    assert res["drift_alarm"]["threshold_psi"] == 0.25


def test_transfer_summary_reports_retention(fake_root: Path):
    res = analysis.analyze_transfer(fake_root)
    assert res is not None
    r = [x for x in res["retention"] if x["detector"] == "ae" and x["scaler"] == "source"][0]
    assert r["source"] == "a" and r["target"] == "b"
    assert 0 < r["retention"] < 1  # transfer is worse than in-dataset in the fake data


def test_missing_experiments_are_skipped(tmp_path: Path):
    assert analysis.analyze_main(tmp_path) is None
    assert analysis.analyze_all(tmp_path).keys() == {"titles"}


def test_make_fit_sets_regimes_and_contamination():
    from ids_anomaly.data.preprocess import Dataset

    rng = np.random.default_rng(0)
    n = 4000
    is_attack = rng.random(n) < 0.4
    ds = Dataset(rng.normal(size=(n, 3)).astype("float32"), is_attack, np.where(is_attack, "x", "normal"),
                 np.where(is_attack, "x", "normal"), ["a", "b", "c"])  # fmt: skip
    clean = make_fit_sets(ds, "clean", seed=0)
    assert clean.contamination == 0.0 and len(clean.X_cal) > 0
    inj = make_fit_sets(ds, "injected", rho=0.1, n_fit=1000, seed=0)
    assert len(inj.X_fit) == 1000 and inj.n_injected == 100 and inj.contamination == pytest.approx(0.1)
    nat = make_fit_sets(ds, "natural", seed=0)
    assert nat.contamination == pytest.approx(is_attack.mean(), abs=0.05)
    with pytest.raises(ValueError):
        make_fit_sets(ds, "injected", rho=0.9, n_fit=3000, seed=0)  # not enough attacks
    # calibration rows never leak into the fit set
    assert not (set(map(tuple, clean.X_cal.tolist())) & set(map(tuple, clean.X_fit.tolist())))


def test_save_labels_never_pickles_object_arrays(tmp_path: Path):
    obj = np.array(["dos", "normal", "probe"], dtype=object)
    path = tmp_path / "labels.npz"
    experiments.save_labels(path, category=obj, is_attack=np.array([1, 0, 1], dtype=bool))
    loaded = analysis.load_labels(path)  # allow_pickle=False would raise on object dtype
    assert loaded["category"].tolist() == ["dos", "normal", "probe"]
    assert loaded["category"].dtype.kind == "U"
