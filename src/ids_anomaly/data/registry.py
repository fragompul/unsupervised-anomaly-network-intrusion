"""One entry point for every benchmark dataset: ``load_dataset("cicids2017", raw_dir)``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ids_anomaly.data import cicids, fetch, unsw
from ids_anomaly.data.download import download_raw
from ids_anomaly.data.preprocess import Dataset, load_datasets


@dataclass
class Bundle:
    """A dataset ready for experiments: fitted preprocessing + train/test splits."""

    key: str
    train: Dataset
    test: Dataset
    preprocessor: Any


@dataclass(frozen=True)
class DatasetInfo:
    key: str
    title: str
    year: int
    citation: str
    ensure: Callable[[Path], None]
    load: Callable[[Path], tuple[Dataset, Dataset, Any]]


def _nsl_ensure(raw_dir: Path) -> None:
    download_raw(raw_dir)


def _unsw_ensure(raw_dir: Path) -> None:
    fetch.fetch_unsw(raw_dir)


def _cicids_ensure(raw_dir: Path) -> None:
    fetch.fetch_cicids(raw_dir)


DATASETS: dict[str, DatasetInfo] = {
    "nsl_kdd": DatasetInfo(
        "nsl_kdd", "NSL-KDD", 2009, "Tavallaee et al., 2009", _nsl_ensure, load_datasets
    ),
    "unsw_nb15": DatasetInfo(
        "unsw_nb15",
        "UNSW-NB15",
        2015,
        "Moustafa & Slay, 2015",
        _unsw_ensure,
        unsw.load_unsw,
    ),
    "cicids2017": DatasetInfo(
        "cicids2017",
        "CIC-IDS2017",
        2017,
        "Sharafaldin et al., 2018",
        _cicids_ensure,
        cicids.load_cicids,
    ),
}


def load_dataset(key: str, raw_dir: Path, download: bool = True) -> Bundle:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}")
    info = DATASETS[key]
    if download:
        info.ensure(raw_dir)
    train, test, pre = info.load(raw_dir)
    return Bundle(key=key, train=train, test=test, preprocessor=pre)
