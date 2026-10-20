"""``python -m ids_anomaly.export``: build every artifact the web app needs from stored results."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path

from ids_anomaly import __version__
from ids_anomaly.data.registry import DATASETS
from ids_anomaly.export import build
from ids_anomaly.protocol import analysis, experiments
from ids_anomaly.protocol.trainsets import make_fit_sets

logger = logging.getLogger(__name__)


def export_all(root: Path, out_dir: Path, n_points: int = 12_000, with_umap: bool = True) -> dict:
    raw_dir = root / "data" / "raw"
    scores = root / "results" / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analysis.analyze_all(root)

    manifests = []
    for key, info in DATASETS.items():
        cell_path = scores / "main" / f"{key}__clean__seed0.npz"
        if not cell_path.exists():
            logger.warning("no stored scores for %s; skipping", key)
            continue
        bundle = experiments._bundle(key, str(raw_dir))
        fs = make_fit_sets(bundle.train, "clean", n_fit=100_000, seed=0)
        manifest = build.export_dataset(
            key, info.title, bundle.test, fs.X_fit, analysis.load_cell(cell_path), out_dir,
            n_points=n_points, with_umap=with_umap,
        )  # fmt: skip
        manifest.update(year=info.year, citation=info.citation)
        manifests.append(manifest)

    replay = None
    drift_cell = scores / "drift" / "cicids2017__seed0.npz"
    if drift_cell.exists():
        replay = build.export_replay(
            analysis.load_cell(drift_cell), analysis.load_labels(scores / "drift" / "labels__cicids2017.npz"), out_dir
        )

    (out_dir / "summary.json").write_text(json.dumps(summary, separators=(",", ":"), default=float))
    top = {
        "name": "Noctua",
        "version": __version__,
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "datasets": manifests,
        "replay": replay,
        "detector_titles": summary.get("titles", {}),
    }
    (out_dir / "manifest.json").write_text(json.dumps(top, separators=(",", ":"), default=float))
    return top


def main() -> None:
    parser = argparse.ArgumentParser(description="Export static artifacts for the Noctua web app.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--points", type=int, default=12_000)
    parser.add_argument("--no-umap", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export_all(args.root, args.out or args.root / "app" / "public" / "data", args.points, not args.no_umap)


if __name__ == "__main__":
    main()
