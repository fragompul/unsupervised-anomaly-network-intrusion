"""UNSW-NB15 loader (Moustafa & Slay, 2015).

Uses the official partition: 175,341 training flows and 82,332 testing flows. UNSW-NB15 was
generated in the ACCS Cyber Range Lab with IXIA PerfectStorm, so it reflects (synthetic) 2015-era
traffic with nine modern attack families, a large step up from NSL-KDD's 1999-era hand-crafted
features. The mirror's file names are swapped relative to the official split (see
:mod:`ids_anomaly.data.fetch`); files are therefore identified by row count.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ids_anomaly.data.preprocess import Dataset
from ids_anomaly.data.tabular import build_tabular_preprocessor

CATEGORICAL = ["proto", "service", "state"]
# stcpb/dtcpb are TCP initial sequence numbers: effectively random identifiers, not behaviour.
DROPPED = ["stcpb", "dtcpb"]
TARGETS = ["attack_cat", "label"]
OFFICIAL_TRAIN_ROWS = 175_341
OFFICIAL_TEST_ROWS = 82_332


def _read(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_parquet(raw_dir / "unsw" / name) for name in ("train.parquet", "test.parquet")]
    frames.sort(key=len, reverse=True)  # official training set is the larger file
    train, test = frames
    if (len(train), len(test)) != (OFFICIAL_TRAIN_ROWS, OFFICIAL_TEST_ROWS):
        raise ValueError(
            f"Unexpected UNSW-NB15 row counts {len(train)}/{len(test)}; "
            f"expected {OFFICIAL_TRAIN_ROWS}/{OFFICIAL_TEST_ROWS}."
        )
    return train, test


def _to_dataset(df: pd.DataFrame, X: np.ndarray, names: list[str]) -> Dataset:
    category = df["attack_cat"].astype(str).str.lower().to_numpy()
    return Dataset(
        X=X.astype(np.float32),
        is_attack=(df["label"].to_numpy() == 1),
        attack_category=category,
        label=category,
        feature_names=names,
    )


def load_unsw(raw_dir: Path) -> tuple[Dataset, Dataset, object]:
    train_df, test_df = _read(raw_dir)
    for col in CATEGORICAL:
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
    numeric = [c for c in train_df.columns if c not in CATEGORICAL + TARGETS + DROPPED]
    features = numeric + CATEGORICAL

    pre = build_tabular_preprocessor(numeric, CATEGORICAL)
    X_train = pre.fit_transform(train_df[features])
    X_test = pre.transform(test_df[features])
    names = [str(n) for n in pre.get_feature_names_out()]
    return _to_dataset(train_df, X_train, names), _to_dataset(test_df, X_test, names), pre
