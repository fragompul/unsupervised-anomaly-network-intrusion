from __future__ import annotations

import numpy as np
import pytest

from ids_anomaly.protocol import conformal, metrics, stats


def test_conformal_pvalue_formula_and_extremes():
    cal = np.array([1.0, 2.0, 3.0, 4.0])
    p = conformal.conformal_pvalues(cal, np.array([0.0, 2.5, 10.0, 4.0]))
    # (1 + #cal >= s) / (n + 1)
    np.testing.assert_allclose(p, [5 / 5, 3 / 5, 1 / 5, 2 / 5])
    assert p.min() >= 1 / (len(cal) + 1)


def test_conformal_guarantees_fpr_on_exchangeable_normals():
    rng = np.random.default_rng(0)
    alphas = [0.01, 0.05, 0.1, 0.2]
    realized = {a: [] for a in alphas}
    for _ in range(40):
        cal = rng.normal(size=500)
        test = rng.normal(size=2000)  # same distribution => exchangeable
        p = conformal.conformal_pvalues(cal, test)
        for a in alphas:
            realized[a].append(conformal.empirical_fpr(p, a))
    for a in alphas:
        assert np.mean(realized[a]) <= a * 1.1  # <= alpha up to Monte-Carlo noise
        assert np.mean(realized[a]) >= a * 0.7  # and not absurdly conservative


def test_conformal_detects_violation_under_shift():
    rng = np.random.default_rng(1)
    cal = rng.normal(size=1000)
    shifted = rng.normal(loc=1.0, size=5000)
    p = conformal.conformal_pvalues(cal, shifted)
    assert conformal.empirical_fpr(p, 0.05) > 0.2  # guarantee broken, and measurably so


def test_merged_pvalues_remain_valid_for_independent_detectors():
    rng = np.random.default_rng(2)
    fprs_avg, fprs_min = [], []
    for _ in range(30):
        ps = np.vstack(
            [conformal.conformal_pvalues(rng.normal(size=400), rng.normal(size=2000)) for _ in range(5)]
        )
        fprs_avg.append(conformal.empirical_fpr(conformal.merge_pvalues_average(ps), 0.05))
        fprs_min.append(conformal.empirical_fpr(conformal.merge_pvalues_min(ps), 0.05))
    assert np.mean(fprs_avg) <= 0.05 * 1.15
    assert np.mean(fprs_min) <= 0.05 * 1.15


def test_coverage_curve_shapes():
    rng = np.random.default_rng(0)
    p = conformal.conformal_pvalues(rng.normal(size=500), rng.normal(size=500))
    grid, fpr = conformal.coverage_curve(p)
    assert grid.shape == fpr.shape and np.all(np.diff(fpr) >= 0)


def test_metrics_perfect_and_random():
    y = np.r_[np.zeros(500), np.ones(500)]
    perfect = np.r_[np.zeros(500), np.ones(500)] + np.random.default_rng(0).normal(0, 1e-3, 1000)
    assert metrics.roc_auc(y, perfect) == pytest.approx(1.0)
    assert metrics.pr_auc(y, perfect) == pytest.approx(1.0)
    assert metrics.tpr_at_fpr(y, perfect, 0.01) == pytest.approx(1.0)
    rnd = np.random.default_rng(1).normal(size=1000)
    assert 0.4 < metrics.roc_auc(y, rnd) < 0.6


def test_operating_point_uses_calibration_threshold_only():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=10_000)
    y = np.r_[np.zeros(5000), np.ones(5000)]
    s = np.r_[rng.normal(size=5000), rng.normal(loc=4.0, size=5000)]
    op = metrics.operating_point(y, s, cal, alpha=0.05)
    assert op["fpr"] == pytest.approx(0.05, abs=0.015)
    assert op["tpr"] > 0.95 and 0 < op["f1"] <= 1


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400), np.ones(400)]
    s = np.r_[rng.normal(size=400), rng.normal(loc=1.0, size=400)]
    lo, hi = metrics.bootstrap_ci(y, s, n_boot=100)
    assert lo < metrics.roc_auc(y, s) < hi and hi - lo < 0.1


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    assert metrics.population_stability_index(ref, rng.normal(size=20_000)) < 0.02
    assert metrics.population_stability_index(ref, rng.normal(loc=1.5, size=20_000)) > 0.25


def test_nemenyi_critical_difference_matches_demsar_table():
    assert stats.nemenyi_cd(10, 14) == pytest.approx(3.164 * np.sqrt(10 * 11 / (6 * 14)), rel=1e-3)


def test_friedman_ranks_and_cd_groups():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(20, 1))
    scores = np.hstack([base + 2.0, base + 1.0, base + 0.0, base + 0.01]) + rng.normal(0, 0.01, (20, 4))
    names = ["a", "b", "c", "d"]
    res = stats.friedman(scores, names)
    assert res.pvalue < 1e-6
    assert res.mean_ranks["a"] == pytest.approx(1.0) and res.mean_ranks["b"] == pytest.approx(2.0)
    groups = stats.cd_groups(res.mean_ranks, res.critical_difference)
    assert any({"c", "d"} <= set(g) for g in groups)
    assert not any({"a", "c"} <= set(g) for g in groups)


def test_holm_wilcoxon_flags_clear_winner_and_not_noise():
    rng = np.random.default_rng(0)
    n = 25
    baseline = rng.normal(size=n)
    scores = np.column_stack(
        [baseline, baseline + 1.0 + rng.normal(0, 0.1, n), baseline + rng.normal(0, 0.1, n)]
    )
    rows = {r["method"]: r for r in stats.holm_wilcoxon(scores, ["base", "better", "same"], "base")}
    assert rows["better"]["significant"] and not rows["same"]["significant"]
    assert rows["better"]["p_holm"] >= rows["better"]["p"]


def test_pairwise_win_matrix_is_antisymmetric_without_ties():
    scores = np.array([[3.0, 2.0, 1.0], [3.0, 1.0, 2.0]])
    w = stats.pairwise_win_matrix(scores, ["x", "y", "z"])
    assert w["x"]["y"] == 1.0 and w["y"]["x"] == 0.0 and w["y"]["z"] == 0.5


def test_bootstrap_mean_ci():
    lo, hi = stats.bootstrap_mean_ci(np.random.default_rng(0).normal(5, 1, 50))
    assert lo < 5.0 < hi
