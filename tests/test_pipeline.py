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
    mean_signed_error, mae_at_coverage,
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
