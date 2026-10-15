"""
Smoke tests for src/pipeline.py. These check the code behaves correctly on
small synthetic inputs; they are not, and must never be reported as, results
on the real Tanzania cohort.
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import (  # noqa: E402
    FEATURE_COLUMNS, TARGET_COLUMN, GESTATION_VISIT2_COLUMN,
    apply_eligibility, fixed_split, mean_absolute_error, rmse,
    identical_vector_groups, split_group_violations, train_stage1_candidates,
    mean_signed_error, mae_at_coverage, assign_cv_folds, n_keep, keep_mask,
    paired_bootstrap_difference,
)


def _toy_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 5, n) for c in FEATURE_COLUMNS})
    df["hb_visit1_g_dl"] = rng.uniform(6, 14, n)
    df["gestation_visit1_weeks"] = rng.uniform(5, 20, n)
    df[GESTATION_VISIT2_COLUMN] = df["gestation_visit1_weeks"] + rng.uniform(1, 5, n)
    df[TARGET_COLUMN] = df["hb_visit1_g_dl"] + rng.normal(0, 1, n)
    return df


def test_eligibility_drops_out_of_range_hb():
    df = _toy_df()
    df.loc[0, "hb_visit1_g_dl"] = 25  # implausible, should be dropped
    out = apply_eligibility(df)
    assert len(out) == len(df) - 1


def test_eligibility_drops_non_increasing_gestation():
    df = _toy_df()
    df.loc[0, GESTATION_VISIT2_COLUMN] = df.loc[0, "gestation_visit1_weeks"] - 1
    out = apply_eligibility(df)
    assert len(out) == len(df) - 1


def test_fixed_split_sizes_and_disjoint():
    df = _toy_df(n=3000)
    elig = apply_eligibility(df).reset_index(drop=True)
    elig["_row_id"] = elig.index  # unique id to check disjointness after fixed_split resets the index
    train, dev, test = fixed_split(elig, seed=1)
    assert (len(train), len(dev), len(test)) == (1884, 402, 399)
    # partitions must not overlap (compare original row identity, not the reset positional index)
    assert set(train["_row_id"]).isdisjoint(set(dev["_row_id"]))
    assert set(train["_row_id"]).isdisjoint(set(test["_row_id"]))
    assert set(dev["_row_id"]).isdisjoint(set(test["_row_id"]))


def test_metrics_basic_values():
    y_true = np.array([10.0, 12.0, 8.0])
    y_pred = np.array([11.0, 12.0, 8.0])
    assert abs(mean_absolute_error(y_true, y_pred) - (1 / 3)) < 1e-9
    assert rmse(y_true, y_pred) > 0
    assert mean_signed_error(y_true, y_pred) > 0  # predictions run slightly high


def test_mae_at_coverage_picks_lowest_scores():
    errors = np.array([1.0, 5.0, 2.0, 4.0])
    scores = np.array([0.1, 0.9, 0.2, 0.8])  # low score == "more trustworthy"
    # 50% coverage should retain the two lowest-scored (indices 0, 2) -> errors 1.0, 2.0
    assert abs(mae_at_coverage(errors, scores, 0.5) - 1.5) < 1e-9


def test_identical_vectors_stay_in_one_partition():
    df = _toy_df(n=2700, seed=3)
    rng = np.random.default_rng(5)
    src = rng.choice(len(df), size=60, replace=False)
    dst = rng.choice(np.setdiff1d(np.arange(len(df)), src), size=60, replace=False)
    for a, b in zip(src, dst):  # force 60 identical-vector pairs
        df.loc[b, FEATURE_COLUMNS] = df.loc[a, FEATURE_COLUMNS].values
    train, dev, test = fixed_split(df, seed=1)
    assert (len(train), len(dev), len(test)) == (1884, 402, 399)
    assert split_group_violations(train, dev, test) == 0


def test_identical_vector_groups_treat_missing_as_equal():
    df = _toy_df(n=10)
    df.loc[1, FEATURE_COLUMNS] = df.loc[0, FEATURE_COLUMNS].values
    df.loc[[0, 1], "pulse_visit1_bpm"] = np.nan
    g = identical_vector_groups(df)
    assert g[0] == g[1]
    assert len(set(g)) == 9


def test_stage1_candidates_cover_all_five_models():
    df = _toy_df(n=300, seed=2)
    res = train_stage1_candidates(df, n_folds=3, seed=0)
    assert list(res) == ["median", "carry_forward", "two_input_linear", "ridge", "random_forest"]
    assert all(np.isfinite(r.cv_mae) for r in res.values())


def test_stage1_candidates_keep_out_of_fold_forecasts():
    df = _toy_df(n=300, seed=2)
    res = train_stage1_candidates(df, n_folds=3, seed=0)
    oof = res["random_forest"].oof_pred
    assert len(oof) == len(df) and np.isfinite(oof).all()


def test_cv_fold_rule_matches_a_hand_worked_example():
    # Group 0 has two rows, so it goes first (to fold 0). The single-row groups then go,
    # in group order, to the lightest fold, the lowest fold number winning ties.
    groups = np.array([0, 1, 2, 0, 3, 4, 5, 6])
    assert list(assign_cv_folds(groups, 3)) == [0, 1, 2, 0, 1, 2, 0, 1]


def test_cv_folds_are_repeatable_balanced_and_never_split_a_group():
    df = _toy_df(n=300, seed=4)
    src = np.arange(0, 20)
    dst = np.arange(100, 120)
    for a, b in zip(src, dst):  # 20 identical-vector pairs
        df.loc[b, FEATURE_COLUMNS] = df.loc[a, FEATURE_COLUMNS].values
    groups = identical_vector_groups(df)
    folds = assign_cv_folds(groups, 3)
    assert (folds == assign_cv_folds(groups, 3)).all()
    assert set(folds) == {0, 1, 2}
    for g in np.unique(groups):
        assert len(set(folds[groups == g])) == 1
    assert np.bincount(folds).max() - np.bincount(folds).min() <= 2


def test_n_keep_follows_the_protocol_numbers():
    assert n_keep(399, 0.8) == 319   # the locked-test number in the protocol
    assert n_keep(402, 0.8) == 322
    assert n_keep(399, 1.0) == 399
    assert n_keep(5, 0.7) == 4       # 3.5 rounds up
    assert n_keep(10, 0.9) == 9


def test_ties_are_broken_by_record_id_not_by_row_position():
    scores = np.array([1.0, 1.0, 1.0, 1.0])
    ids = np.array([40, 10, 30, 20])
    kept = keep_mask(scores, 0.5, ids)
    assert set(ids[kept]) == {10, 20}
    order = np.array([2, 0, 3, 1])  # shuffle the rows: the same women are kept
    assert set(ids[order][keep_mask(scores[order], 0.5, ids[order])]) == {10, 20}
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    assert mae_at_coverage(errors, np.zeros(4), 0.5, np.array([4, 3, 2, 1])) == 3.5


def test_paired_bootstrap_difference_on_a_constant_gap():
    a, b = np.full(50, 1.0), np.full(50, 2.0)
    d = paired_bootstrap_difference(a, b, n_boot=200, seed=0)
    assert d["difference"] == -1.0 and d["ci95_low"] == -1.0 and d["ci95_high"] == -1.0
