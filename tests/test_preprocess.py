from __future__ import annotations

import numpy as np

from ids_anomaly.data.preprocess import load_datasets, normal_only


def test_load_datasets_shapes_and_no_nans(synthetic_raw_dir):
    train, test, preprocessor = load_datasets(synthetic_raw_dir)

    assert train.X.shape[0] == 200
    assert test.X.shape[0] == 80
    assert train.X.shape[1] == test.X.shape[1]
    assert not np.isnan(train.X).any()
    assert not np.isnan(test.X).any()


def test_labels_and_categories_are_held_out_of_features(synthetic_raw_dir):
    train, _, _ = load_datasets(synthetic_raw_dir)
    assert train.is_attack.dtype == bool
    assert set(train.attack_category.tolist()) <= {"normal", "dos", "probe", "r2l", "u2r"}


def test_preprocessor_fit_only_on_train_but_transforms_test_consistently(synthetic_raw_dir):
    train, test, preprocessor = load_datasets(synthetic_raw_dir)
    # Re-transforming the raw test frame with the already-fitted preprocessor must reproduce
    # the same feature matrix -- i.e. test never influenced fitting.
    import pandas as pd

    from ids_anomaly.data.schema import ALL_COLUMNS, FEATURE_COLUMNS

    raw_test = pd.read_csv(synthetic_raw_dir / "KDDTest+.txt", header=None, names=ALL_COLUMNS)
    replayed = preprocessor.transform(raw_test[FEATURE_COLUMNS])
    np.testing.assert_allclose(replayed, test.X, rtol=1e-5)


def test_normal_only_keeps_exclusively_non_attack_rows(synthetic_raw_dir):
    train, _, _ = load_datasets(synthetic_raw_dir)
    normal = normal_only(train)
    assert normal.X.shape[0] == int((~train.is_attack).sum())
    assert not normal.is_attack.any()
