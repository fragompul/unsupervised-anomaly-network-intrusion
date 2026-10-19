from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ids_anomaly.data.preprocess import Dataset
from ids_anomaly.detectors.deep import AutoencoderDetector
from ids_anomaly.export import build


def _toy(n=1200, d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.3
    X = rng.normal(size=(n, d)).astype("float32")
    X[y] += 2.5
    cats = np.where(y, rng.choice(["dos", "probe"], n), "normal")
    ds = Dataset(X, y, cats, cats, [f"num__f{i}" for i in range(d)], group=rng.choice(["Wed", "Thu"], n))
    train_normal = rng.normal(size=(2000, d)).astype("float32")
    return ds, train_normal


def _fake_cell(ds, specs):
    rng = np.random.default_rng(1)
    ev = rng.normal(size=(len(specs), len(ds))).astype("float32")
    ev[:, ds.is_attack] += 2.0
    cal = rng.normal(size=(len(specs), 500)).astype("float32")
    return {"specs": specs, "scores": {"cal": cal, "test": ev}, "fit_seconds": np.ones(len(specs)),
            "score_seconds": np.ones(len(specs)), "meta": {}}  # fmt: skip


@pytest.fixture(scope="module")
def exported(tmp_path_factory) -> tuple[Path, dict]:
    ds, train_normal = _toy()
    cell = _fake_cell(ds, ["iforest", "knn", "ae", "flow", "neutral"])
    out = tmp_path_factory.mktemp("export")
    manifest = build.export_dataset("toy", "Toy", ds, train_normal, cell, out, n_points=600, with_umap=False)
    return out / "toy", manifest


def test_binary_artifacts_have_expected_shapes(exported):
    d, m = exported
    n, nf, nd = m["n_points"], m["n_features"], len(m["detectors"])
    assert (d / "scores.u8").stat().st_size == nd * n
    assert (d / "features.i8").stat().st_size == n * nf
    assert (d / "embedding_pca.i16").stat().st_size == n * 3 * 2
    assert m["embeddings"] == ["pca"]
    ids = [x["id"] for x in m["detectors"]]
    assert {"ae", "ens_all", "ens_diverse"} <= set(ids)


def test_points_and_attribution_are_aligned(exported):
    d, m = exported
    pts = json.loads((d / "points.json").read_text())
    attr = json.loads((d / "attribution.json").read_text())
    assert len(pts["category"]) == len(pts["attack"]) == len(attr["idx"]) == m["n_points"]
    assert max(pts["category"]) < len(m["categories"]) and pts["group_names"] == ["Thu", "Wed"]
    assert all(len(r) == 5 and all(0 <= j < m["n_features"] for j in r) for r in attr["idx"])


def test_rare_categories_survive_sampling():
    ds, train_normal = _toy(n=3000)
    ds.attack_category[:5] = "worms"
    ds.is_attack[:5] = True
    cell = _fake_cell(ds, ["iforest", "ae"])
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        m = build.export_dataset("t", "T", ds, train_normal, cell, Path(tmp), n_points=500, with_umap=False)
        pts = json.loads((Path(tmp) / "t" / "points.json").read_text())
    assert "worms" in m["categories"]
    assert pts["category"].count(m["categories"].index("worms")) >= 3


def test_ae_weights_reproduce_the_python_score(exported):
    """The JSON export must be sufficient to recompute the AE score in another runtime."""
    d, m = exported
    ae = json.loads((d / "ae.json").read_text())
    ds, train_normal = _toy()
    rng = np.random.default_rng(3)
    x = rng.normal(size=(m["n_features"],)).astype("float64")

    h = x
    for layer in ae["layers"]:
        h = np.asarray(layer["w"]) @ h + np.asarray(layer["b"])
        if layer["relu"]:
            h = np.maximum(h, 0)
    manual = float(((h - x) ** 2).mean())
    reference = float(AutoencoderDetector(seed=0).fit(train_normal).score(x[None].astype("float32"))[0])
    assert manual == pytest.approx(reference, rel=1e-3)  # JS forward pass == PyTorch score
    assert len(ae["layers"]) == 6 and not ae["layers"][2]["relu"] and not ae["layers"][5]["relu"]
    assert len(ae["cal_scores"]) == build.CAL_QUANTILES and ae["cal_scores"] == sorted(ae["cal_scores"])


def test_curves_and_faithfulness_present(exported):
    d, m = exported
    curves = json.loads((d / "curves.json").read_text())
    c = curves["curves"]["ae"]
    assert c["tpr"][-1] == pytest.approx(1.0) and c["fpr"][0] < 1e-3
    assert 0.0 < m["explain"]["deletion"]["attributed"][0] <= 1.0
    cov = curves["coverage"]["ae"]
    assert len(cov["alpha"]) == len(cov["fpr"]) == 40


def test_quantization_helpers():
    p = np.array([1.0, 0.1, 0.01, 1e-9])
    q = build.nlogp_u8(p)
    assert q.dtype == np.uint8 and q.tolist() == [0, 40, 80, 255]


def test_alignment_guard(tmp_path: Path):
    ds, train_normal = _toy()
    cell = _fake_cell(ds, ["ae"])
    cell["scores"]["test"] = cell["scores"]["test"][:, :10]
    with pytest.raises(ValueError, match="do not align"):
        build.export_dataset("x", "X", ds, train_normal, cell, tmp_path, with_umap=False)


def test_replay_export(tmp_path: Path):
    rng = np.random.default_rng(0)
    specs = ["ae", "iforest", "flow", "knn"]
    labels, sets = {}, {}
    for day in ("Tue", "Wed", "Thu", "Fri"):
        n = 2500
        labels[f"{day}__timestamp"] = np.sort(rng.integers(0, 10_000, n))
        labels[f"{day}__is_attack"] = rng.random(n) < 0.2
        sets[day] = rng.normal(size=(4, n)).astype("float32")
    cell = {"specs": specs, "scores": {"cal": rng.normal(size=(4, 600)).astype("float32"), **sets}}
    info = build.export_replay(cell, labels, tmp_path)
    payload = json.loads((tmp_path / "replay.json").read_text())
    assert info["n"] == 4 * build.REPLAY_PER_DAY == len(payload["t"])
    assert set(payload["nlogp"]) == {"ae", "iforest", "flow", "knn", "ens_all"}
    assert payload["day"][0] == 0 and payload["day"][-1] == 3
