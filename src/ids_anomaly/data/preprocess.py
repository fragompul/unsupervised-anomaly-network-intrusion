"""Load NSL-KDD raw files into model-ready feature matrices.

Design choices (see docs/methodology.md for the full rationale):

* Categorical columns (``protocol_type``, ``service``, ``flag``) are
  one-hot encoded with a vocabulary fit on train only; unseen test
  categories map to an all-zero row via ``handle_unknown="ignore"``.
* Heavy-tailed count/byte columns get a ``log1p`` transform before
  standardization -- NSL-KDD's ``src_bytes``/``duration``/``count`` columns
  span several orders of magnitude and dominate Euclidean-distance-based
  methods (KMeans, UMAP, SVM kernels) if left on a raw scale.
* Bounded rate columns (already in [0, 1]) and binary flags are
  standardized without the log transform.
* Labels are parsed into a binary ``is_attack`` target and a 5-class
  ``attack_category`` for evaluation only -- they never enter the
  feature matrix or influence any fitted transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ids_anomaly.data.schema import (
    ALL_COLUMNS,
    BINARY_FLAG_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    attack_category,
)

RATE_COLUMNS: list[str] = [
    c for c in NUMERIC_COLUMNS if c.endswith("_rate") and c not in BINARY_FLAG_COLUMNS
]
SKEWED_COUNT_COLUMNS: list[str] = [
    c for c in NUMERIC_COLUMNS if c not in RATE_COLUMNS and c not in BINARY_FLAG_COLUMNS
]


@dataclass
class Dataset:
    """A preprocessed NSL-KDD split with features held separate from labels."""

    X: np.ndarray
    is_attack: np.ndarray
    attack_category: np.ndarray
    label: np.ndarray
    feature_names: list[str]


def _read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=ALL_COLUMNS)
    return df.drop(columns=["difficulty"])


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: log1p + scale skewed counts, scale rates/flags, one-hot categoricals."""
    log_scale = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("skewed", log_scale, SKEWED_COUNT_COLUMNS),
            ("rate", StandardScaler(), RATE_COLUMNS),
            ("flag", StandardScaler(), BINARY_FLAG_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def load_datasets(
    raw_dir: Path,
) -> tuple[Dataset, Dataset, ColumnTransformer]:
    """Load and jointly preprocess the NSL-KDD train/test splits.

    The preprocessor is fit on the training split only (standard practice to
    avoid test-set leakage) and reused to transform test. Returns both
    datasets plus the fitted preprocessor so the dashboard/inference code can
    transform new flows identically.
    """
    train_df = _read_raw(raw_dir / "KDDTrain+.txt")
    test_df = _read_raw(raw_dir / "KDDTest+.txt")

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])
    feature_names = list(preprocessor.get_feature_names_out())

    train_ds = Dataset(
        X=X_train.astype(np.float32),
        is_attack=(train_df["label"] != "normal").to_numpy(),
        attack_category=train_df["label"].map(attack_category).to_numpy(),
        label=train_df["label"].to_numpy(),
        feature_names=feature_names,
    )
    test_ds = Dataset(
        X=X_test.astype(np.float32),
        is_attack=(test_df["label"] != "normal").to_numpy(),
        attack_category=test_df["label"].map(attack_category).to_numpy(),
        label=test_df["label"].to_numpy(),
        feature_names=feature_names,
    )
    return train_ds, test_ds, preprocessor


def normal_only(ds: Dataset) -> Dataset:
    """Subset a Dataset to normal-labeled rows, for semi-supervised (one-class) training."""
    mask = ~ds.is_attack
    return Dataset(
        X=ds.X[mask],
        is_attack=ds.is_attack[mask],
        attack_category=ds.attack_category[mask],
        label=ds.label[mask],
        feature_names=ds.feature_names,
    )
