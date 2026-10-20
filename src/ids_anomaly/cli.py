"""CLI entrypoint: ``python -m ids_anomaly.cli run [--trials N]``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ids_anomaly.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unsupervised IDS benchmark pipeline.")
    parser.add_argument(
        "command", choices=["run", "data", "benchmark", "analyze"],
        help="`run`: legacy NSL-KDD pipeline. `data`: fetch datasets. `benchmark`: run experiment "
        "cells. `analyze`: build paper tables/artifacts from stored scores.",
    )
    parser.add_argument(
        "--dataset", default="all", help="Dataset key for `data` (nsl_kdd, unsw_nb15, cicids2017, all)."
    )
    parser.add_argument(
        "--experiment", default="all", help="main | contamination | drift | transfer | all (benchmark)."
    )
    parser.add_argument("--seeds", type=int, default=5, help="Seeds per cell (benchmark).")
    parser.add_argument("--workers", type=int, default=3, help="Parallel cells (benchmark).")
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials per HPO study.")
    parser.add_argument(
        "--jobs", type=int, default=None, help="Concurrent Optuna trials (default: cpu_count - 2)."
    )
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2], help="Project root."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Pipeline progress is genuinely useful even without -v: always show INFO for our own package.
    logging.getLogger("ids_anomaly").setLevel(logging.INFO)

    if args.command == "data":
        from ids_anomaly.data.registry import DATASETS

        keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
        for key in keys:
            DATASETS[key].ensure(args.root / "data" / "raw")
            print(f"{DATASETS[key].title}: ready")
    elif args.command == "benchmark":
        from ids_anomaly.protocol.experiments import GRIDS, run_experiment

        names = list(GRIDS) if args.experiment == "all" else [args.experiment]
        for name in names:
            for line in run_experiment(name, args.root, seeds=args.seeds, n_jobs=args.workers):
                print(line)
    elif args.command == "analyze":
        from ids_anomaly.protocol.analysis import analyze_all

        analyze_all(args.root)
    elif args.command == "run":
        run_full_pipeline(args.root, n_hpo_trials=args.trials, hpo_n_jobs=args.jobs)


if __name__ == "__main__":
    main()
