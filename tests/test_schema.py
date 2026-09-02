from ids_anomaly.data.schema import (
    ALL_COLUMNS,
    ATTACK_CATEGORY_MAP,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    attack_category,
)


def test_feature_columns_count_matches_nsl_kdd_spec():
    assert len(FEATURE_COLUMNS) == 41


def test_all_columns_appends_label_and_difficulty():
    assert [*FEATURE_COLUMNS, "label", "difficulty"] == ALL_COLUMNS


def test_categorical_columns_are_subset_of_features():
    assert set(CATEGORICAL_COLUMNS).issubset(set(FEATURE_COLUMNS))


def test_normal_maps_to_normal():
    assert attack_category("normal") == "normal"


def test_known_attacks_map_to_expected_categories():
    assert attack_category("neptune") == "dos"
    assert attack_category("satan") == "probe"
    assert attack_category("guess_passwd") == "r2l"
    assert attack_category("buffer_overflow") == "u2r"


def test_unseen_attack_label_falls_back_to_unknown_attack():
    assert attack_category("some_future_attack_2099") == "unknown_attack"


def test_every_mapped_value_is_a_valid_category():
    valid = {"normal", "dos", "probe", "r2l", "u2r"}
    assert set(ATTACK_CATEGORY_MAP.values()).issubset(valid)
