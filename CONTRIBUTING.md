# Contributing

Noctua is a research codebase; the bar is *reproducible and tested*.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api,paper,notebooks]"
pre-commit install
make check           # ruff + mypy + pytest (network-free)
cd app && npm ci && npm test && npm run lint
```

* **Adding a detector:** subclass `ids_anomaly.detectors.base.Detector` (`fit` unlabeled, `score` higher =
  more anomalous), register it in `detectors/__init__.py`, add a family and a display title. The
  parametrized test in `tests/test_detectors.py` picks it up automatically.
* **Adding a dataset:** add a loader returning `(train, test, preprocessor)` as
  `Dataset` objects, pin downloads by size and SHA-256 in `data/fetch.py`, and register it in
  `data/registry.py`. Document the split and any mirror quirks in `docs/datasets.md`.
* **Adding an experiment:** add a runner to `protocol/experiments.py` that stores raw scores,
  and an `analyze_*` function in `protocol/analysis.py`. Never compute reported numbers inside a
  runner: analysis must be re-derivable from the stored scores.
* Keep commits small and focused; use conventional-commit prefixes (`feat`, `fix`, `docs`, `test`, ...).
