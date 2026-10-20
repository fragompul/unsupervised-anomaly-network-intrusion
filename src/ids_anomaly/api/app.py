"""Noctua API.

The web app is fully static (it re-implements the autoencoder forward pass in JavaScript), so
this service is optional: it serves the exported artifacts and exposes the *reference*
implementation of playground scoring, which the browser version is tested against. Run with
``uvicorn ids_anomaly.api.app:app`` (``NOCTUA_DATA`` points at the artifacts directory).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ids_anomaly import __version__


def data_dir() -> Path:
    return Path(os.environ.get("NOCTUA_DATA", Path(__file__).resolve().parents[3] / "app" / "public" / "data"))


@lru_cache(maxsize=8)
def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text())


class ScoreRequest(BaseModel):
    dataset: str
    features: list[float] = Field(description="Standardized feature vector (z-scores), one per feature.")


class ScoreResponse(BaseModel):
    score: float
    p_value: float
    verdict: str
    top_features: list[dict[str, Any]]


def forward(layers: list[dict[str, Any]], x: np.ndarray) -> np.ndarray:
    """Autoencoder forward pass from exported weights (mirrors the JavaScript implementation)."""
    h = x
    for layer in layers:
        h = np.asarray(layer["w"]) @ h + np.asarray(layer["b"])
        if layer["relu"]:
            h = np.maximum(h, 0.0)
    return h


def create_app() -> FastAPI:
    app = FastAPI(title="Noctua API", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    def manifest() -> dict[str, Any]:
        path = data_dir() / "manifest.json"
        if not path.exists():
            raise HTTPException(503, "artifacts not exported yet: run `python -m ids_anomaly.export`")
        return _load_json(str(path))

    def dataset_entry(key: str) -> dict[str, Any]:
        for d in manifest()["datasets"]:
            if d["key"] == key:
                return d
        raise HTTPException(404, f"unknown dataset {key!r}")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/manifest")
    def get_manifest() -> dict[str, Any]:
        return manifest()

    @app.get("/api/summary")
    def get_summary() -> Any:
        path = data_dir() / "summary.json"
        if not path.exists():
            raise HTTPException(503, "summary not exported yet")
        return _load_json(str(path))

    @app.get("/api/datasets/{key}/curves")
    def get_curves(key: str) -> Any:
        dataset_entry(key)
        return _load_json(str(data_dir() / key / "curves.json"))

    @app.post("/api/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        entry = dataset_entry(req.dataset)
        if len(req.features) != entry["n_features"]:
            raise HTTPException(422, f"expected {entry['n_features']} features, got {len(req.features)}")
        ae = _load_json(str(data_dir() / req.dataset / "ae.json"))
        x = np.clip(np.asarray(req.features, dtype=np.float64), -entry["feature_clip"], entry["feature_clip"])
        errors = (forward(ae["layers"], x) - x) ** 2
        s = float(errors.mean())
        cal = np.asarray(ae["cal_scores"])
        p = float((1 + (len(cal) - np.searchsorted(cal, s, side="left")) * ae["n_cal"] / len(cal)) / (ae["n_cal"] + 1))
        order = np.argsort(-errors)[:5]
        names = entry["feature_names"]
        return ScoreResponse(
            score=s,
            p_value=min(1.0, p),
            verdict="anomalous" if p <= 0.01 else "suspicious" if p <= 0.05 else "normal",
            top_features=[{"feature": names[i], "error": float(errors[i])} for i in order],
        )

    # Optional: serve the built web app (and its data) from the same origin, as in the Docker image.
    web_dir = os.environ.get("NOCTUA_APP")
    if web_dir and Path(web_dir).is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="app")
    return app


app = create_app()
