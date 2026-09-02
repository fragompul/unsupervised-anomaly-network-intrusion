"""CLI entrypoint: ``python -m ids_anomaly.cli run [--trials N]``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ids_anomaly.pipeline import run_full_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unsupervised IDS benchmark pipeline.")
    parser.add_argument("command", choices=["run"], help="Pipeline command to execute.")
    parser.add_argument("--trials", type=int, default=25, help="Optuna trials per HPO study.")
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

    if args.command == "run":
        run_full_pipeline(args.root, n_hpo_trials=args.trials)


if __name__ == "__main__":
    main()
