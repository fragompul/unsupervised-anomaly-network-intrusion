.PHONY: install lint typecheck test check data benchmark app-dev app-build clean

PY ?= python

install:
	$(PY) -m pip install -e ".[dev,api,paper,notebooks]"

lint:
	ruff check .

typecheck:
	mypy

test:
	pytest -q -m "not slow"

check: lint typecheck test

data:
	$(PY) -m ids_anomaly.cli data --all

benchmark:
	$(PY) -m ids_anomaly.cli benchmark

app-dev:
	cd app && npm run dev

app-build:
	cd app && npm run build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist
