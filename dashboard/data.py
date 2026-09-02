"""Load pipeline artifacts (``results/embeddings_test.npz`` + ``results/metrics/results.json``)
into the flat, JSON-serializable structures the Bokeh app renders and ships to the browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

EMBEDDING_KEYS = ["pca", "umap", "autoencoder", "tsne"]
SCORE_KEYS = {
    "if_score": "Isolation Forest",
    "ocsvm_score": "One-Class SVM",
    "svdd_score": "Deep SVDD",
    "ae_recon_error": "Autoencoder reconstruction error",
}


class Artifacts:
    """Thin wrapper that raises a clear, actionable error if the pipeline hasn't run yet."""

    def __init__(self) -> None:
        npz_path = RESULTS_DIR / "embeddings_test.npz"
        metrics_path = RESULTS_DIR / "metrics" / "results.json"
        if not npz_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                "No pipeline results found. Run `python -m ids_anomaly.cli run` first "
                "to generate results/embeddings_test.npz and results/metrics/results.json."
            )
        data = np.load(npz_path, allow_pickle=True)

        self.n_test = int(data["is_attack"].shape[0])
        self.tsne_idx = data["tsne_idx"].astype(int)

        # tsne is fit on a subsample only; scatter it into full-length arrays with NaNs
        # elsewhere so every embedding shares one row index space with the score/label arrays.
        tsne_full = np.full((self.n_test, data["tsne"].shape[1]), np.nan, dtype=np.float32)
        tsne_full[self.tsne_idx] = data["tsne"]

        self.embeddings: dict[str, np.ndarray] = {
            "pca": data["pca"],
            "umap": data["umap"],
            "autoencoder": data["autoencoder"],
            "tsne": tsne_full,
        }
        self.is_attack = data["is_attack"].astype(bool)
        self.attack_category = data["attack_category"].astype(str)
        self.label = data["label"].astype(str)
        self.scores: dict[str, np.ndarray] = {k: data[k].astype(np.float32) for k in SCORE_KEYS}

        with open(metrics_path) as f:
            self.metrics: dict[str, Any] = json.load(f)

    def base_columns(self) -> dict[str, np.ndarray]:
        cols = {
            "index": np.arange(self.n_test),
            "is_attack": self.is_attack,
            "attack_category": self.attack_category,
            "label": self.label,
        }
        for key in SCORE_KEYS:
            cols[key] = self.scores[key]
        return cols
