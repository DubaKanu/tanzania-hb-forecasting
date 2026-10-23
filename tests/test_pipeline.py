"""
Tests for src/pipeline.py: eligibility, the fixed split, identical-record groups and the fixed
cross-validation fold rule. They use small synthetic inputs; they are not, and must never be
reported as, results on the real Tanzania cohort.
"""
import numpy as np

from helpers import toy_df as _toy_df
from pipeline import (
    FEATURE_COLUMNS, GESTATION_VISIT2_COLUMN, apply_eligibility, assign_cv_folds, fixed_split,
    identical_vector_groups, split_group_violations,
)


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
