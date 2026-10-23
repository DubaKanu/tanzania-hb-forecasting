"""
Tests for src/metrics.py: the error measures and the exact-coverage rule every selector shares.
Small synthetic inputs with hand-worked answers.
"""
import numpy as np

from metrics import (
    keep_mask, mae_at_coverage, mean_absolute_error, mean_signed_error, n_keep,
    paired_bootstrap_difference, rmse, root_mean_square,
)


def test_metrics_basic_values():
    y_true = np.array([10.0, 12.0, 8.0])
    y_pred = np.array([11.0, 12.0, 8.0])
    assert abs(mean_absolute_error(y_true, y_pred) - (1 / 3)) < 1e-9
    assert rmse(y_true, y_pred) > 0
    assert mean_signed_error(y_true, y_pred) > 0  # predictions run slightly high


def test_rmse_equals_root_mean_square_of_the_errors():
    y_true, y_pred = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.5, 1.0, 3.0, 6.0])
    assert abs(rmse(y_true, y_pred) - root_mean_square(y_true - y_pred)) < 1e-12
    assert abs(root_mean_square([3.0, 4.0]) - np.sqrt(12.5)) < 1e-12


def test_mae_at_coverage_picks_lowest_scores():
    errors = np.array([1.0, 5.0, 2.0, 4.0])
    scores = np.array([0.1, 0.9, 0.2, 0.8])  # low score == "more trustworthy"
    # 50% coverage should retain the two lowest-scored (indices 0, 2) -> errors 1.0, 2.0
    assert abs(mae_at_coverage(errors, scores, 0.5) - 1.5) < 1e-9


def test_n_keep_follows_the_protocol_numbers():
    assert n_keep(399, 0.8) == 319   # the locked-test number in the protocol
    assert n_keep(402, 0.8) == 322
    assert n_keep(399, 1.0) == 399
    assert n_keep(5, 0.7) == 4       # 3.5 rounds up
    assert n_keep(10, 0.9) == 9


def test_n_keep_is_not_thrown_off_by_floating_point_noise():
    # 70% of 45 women is exactly 31.5 on paper, so the rule rounds up to 32. In floating point
    # 0.7 * 45 is 31.499999999999996, and a plain floor(x + 0.5) would wrongly give 31.
    assert 0.7 * 45 < 31.5
    assert int(np.floor(0.7 * 45 + 0.5)) == 31
    assert n_keep(45, 0.7) == 32
    assert n_keep(85, 0.7) == 60     # 59.5 on paper
    assert n_keep(0, 0.8) == 0


def test_ties_are_broken_by_record_id_not_by_row_position():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    ids = np.array([40, 10, 30, 20])
    kept = keep_mask(scores, 0.5, ids)
    assert set(ids[kept]) == {10, 20}
    order = np.array([2, 0, 3, 1])  # shuffle the rows: the same women are kept
    assert set(ids[order][keep_mask(scores[order], 0.5, ids[order])]) == {10, 20}
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae_at_coverage(errors, np.zeros(4), 0.5, np.array([4, 3, 2, 1])) == 3.5


def test_keep_mask_keeps_exactly_the_requested_number():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=137)
    for coverage in (1.0, 0.9, 0.8, 0.7, 0.5):
        assert keep_mask(scores, coverage).sum() == n_keep(137, coverage)


def test_keep_mask_keeps_the_lowest_scores():
    scores = np.array([0.5, 0.1, 0.9, 0.3])
    assert list(keep_mask(scores, 0.5)) == [False, True, False, True]


def test_paired_bootstrap_difference_on_a_constant_gap():
    a, b = np.full(50, 1.0), np.full(50, 2.0)
    d = paired_bootstrap_difference(a, b, n_boot=200, seed=0)
    assert d["difference"] == -1.0 and d["ci95_low"] == -1.0 and d["ci95_high"] == -1.0


def test_paired_bootstrap_difference_is_repeatable_and_brackets_the_estimate():
    rng = np.random.default_rng(1)
    a, b = rng.uniform(0, 2, 80), rng.uniform(0, 2, 80)
    first = paired_bootstrap_difference(a, b, n_boot=300, seed=5)
    assert first == paired_bootstrap_difference(a, b, n_boot=300, seed=5)
    assert first["ci95_low"] <= first["difference"] <= first["ci95_high"]
