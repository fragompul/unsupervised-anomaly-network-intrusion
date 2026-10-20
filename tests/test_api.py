from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ids_anomaly.api import app as api_module
from ids_anomaly.export import build
from tests.test_export import _fake_cell, _toy


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    out: Path = tmp_path_factory.mktemp("api")
    ds, train_normal = _toy()
    manifest = build.export_dataset(
        "toy", "Toy", ds, train_normal, _fake_cell(ds, ["iforest", "ae"]), out, n_points=400, with_umap=False
    )
    (out / "manifest.json").write_text(json.dumps({"datasets": [manifest]}))
    (out / "summary.json").write_text(json.dumps({"main": {}}))
    import os

    os.environ["NOCTUA_DATA"] = str(out)
    api_module._load_json.cache_clear()
    yield TestClient(api_module.create_app())
    os.environ.pop("NOCTUA_DATA")
    api_module._load_json.cache_clear()


def test_health_and_manifest(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/manifest").json()["datasets"][0]["key"] == "toy"
    assert client.get("/api/summary").status_code == 200


def test_curves_endpoint_and_unknown_dataset(client):
    assert "curves" in client.get("/api/datasets/toy/curves").json()
    assert client.get("/api/datasets/nope/curves").status_code == 404


def test_score_separates_normal_from_extreme_flow(client):
    n = 8
    normal = client.post("/api/score", json={"dataset": "toy", "features": [0.0] * n}).json()
    extreme = client.post("/api/score", json={"dataset": "toy", "features": [7.0] * n}).json()
    assert extreme["score"] > normal["score"]
    assert extreme["p_value"] < normal["p_value"]
    assert extreme["verdict"] in {"anomalous", "suspicious"}
    assert len(extreme["top_features"]) == 5 and 0 < extreme["p_value"] <= 1


def test_score_validates_length(client):
    r = client.post("/api/score", json={"dataset": "toy", "features": [0.0, 1.0]})
    assert r.status_code == 422


def test_forward_matches_manual_layer_math():
    layers = [{"w": [[1.0, 0.0], [0.0, 2.0]], "b": [0.0, -1.0], "relu": True}, {"w": [[1.0, 1.0]], "b": [0.5], "relu": False}]
    out = api_module.forward(layers, np.array([2.0, 3.0]))
    assert out.tolist() == [2.0 + 5.0 + 0.5]
