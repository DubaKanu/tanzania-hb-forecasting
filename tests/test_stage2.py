"""
Tests for src/stage2.py: the out-of-fold table, the Stage 2 inputs, the Stage2ErrorEstimator class
and the run ID. Synthetic cohorts only; none of this is a result on the real cohort.
"""
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from helpers import toy_cohort
from pipeline import FEATURE_COLUMNS, STAGE1_CONFIG, TARGET_COLUMN, assign_cv_folds, identical_vector_groups
from stage1 import RandomForestModel, prep_matrix, tree_disagreement
from stage2 import (
    STAGE2_CONFIG, STAGE2_INPUT_COLUMNS, Stage2ErrorEstimator, build_oof_table, make_run_id, stage2_features,
)


def _folds(df):
    return assign_cv_folds(identical_vector_groups(df), 3)


# ---------------------------------------------------------------------------
# Out-of-fold table
# ---------------------------------------------------------------------------
def test_oof_table_has_the_columns_the_brief_asks_for():
    df = toy_cohort(300)
    oof = build_oof_table(df, _folds(df))
    assert list(oof.columns) == ["record_id", "fold", "observed_hb", "oof_forecast",
                                 "tree_disagreement", "abs_error"]
    assert np.array_equal(oof["record_id"], df["_source_row"])
    assert np.allclose(oof["abs_error"], np.abs(oof["observed_hb"] - oof["oof_forecast"]))
    assert oof.notna().all().all()


def test_each_oof_forecast_comes_from_a_model_that_never_saw_that_woman():
    df = toy_cohort(300)
    folds = _folds(df)
    oof = build_oof_table(df, folds)
    k = 1
    held_in, held_out = df[folds != k], df[folds == k]
    X_in, imputer = prep_matrix(held_in, fit=True)
    X_out, _ = prep_matrix(held_out, imputer=imputer, fit=False)
    forest = RandomForestRegressor(**STAGE1_CONFIG["random_forest"], random_state=STAGE1_CONFIG["seed"])
    forest.fit(X_in, held_in[TARGET_COLUMN])
    assert np.allclose(oof.loc[folds == k, "oof_forecast"], forest.predict(X_out))
    assert np.allclose(oof.loc[folds == k, "tree_disagreement"], tree_disagreement(forest, X_out))


def test_in_sample_residuals_are_too_optimistic_which_is_why_they_are_not_used():
    df = toy_cohort(300)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    in_sample = np.abs(df[TARGET_COLUMN] - stage1.predict(df)).mean()
    assert in_sample < oof["abs_error"].mean()


# ---------------------------------------------------------------------------
# Stage 2 inputs and estimator
# ---------------------------------------------------------------------------
def test_stage2_inputs_are_the_twelve_permitted_columns():
    assert STAGE2_INPUT_COLUMNS == FEATURE_COLUMNS + ["stage1_forecast", "tree_disagreement"]
    assert len(STAGE2_INPUT_COLUMNS) == 12
    for c in STAGE2_INPUT_COLUMNS:
        assert not any(bad in c.lower() for bad in ("visit2", "target", "risk", "delivery"))


def test_stage2_features_are_filled_with_training_medians_and_in_the_fixed_order():
    df = toy_cohort(120)
    stage1 = RandomForestModel().fit(df)
    forecast, disagreement = stage1.predict_with_disagreement(df)
    features = stage2_features(df, stage1.imputer_, forecast, disagreement)
    assert list(features.columns) == STAGE2_INPUT_COLUMNS and not features.isna().any().any()
    missing = df["respiratory_rate_visit1"].isna()
    assert missing.any()
    assert np.allclose(features.loc[missing, "respiratory_rate_visit1"], df["respiratory_rate_visit1"].median())
    assert np.allclose(features["stage1_forecast"], forecast)


def test_stage2_learns_from_the_out_of_fold_table_it_is_given():
    df = toy_cohort(300)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    oof["abs_error"] = 2.0 * np.abs(df["hb_visit1_g_dl"].to_numpy() - 11)  # a clean function of an input
    estimator = Stage2ErrorEstimator().fit(oof, df, stage1.imputer_)
    pred = estimator.predict(stage2_features(df, stage1.imputer_, oof["oof_forecast"], oof["tree_disagreement"]))
    assert np.corrcoef(pred, oof["abs_error"])[0, 1] > 0.9


def test_the_estimator_refuses_a_table_in_a_different_row_order():
    df = toy_cohort(200)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    with pytest.raises(AssertionError):
        Stage2ErrorEstimator().fit(oof.sample(frac=1.0, random_state=0).reset_index(drop=True), df, stage1.imputer_)


def test_the_estimator_owns_its_own_forest_with_the_declared_settings():
    df = toy_cohort(200)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    estimator = Stage2ErrorEstimator().fit(oof, df, stage1.imputer_)
    assert estimator.forest_ is not stage1.estimator_          # composition: a separate instance
    for key, value in STAGE2_CONFIG["random_forest"].items():
        assert getattr(estimator.forest_, key) == value
    assert estimator.forest_.random_state == STAGE2_CONFIG["seed"]
    other = Stage2ErrorEstimator()
    assert other.forest_ is not estimator.forest_               # every estimator builds its own


def test_from_fitted_wraps_a_saved_forest_without_refitting():
    df = toy_cohort(200)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    fitted = Stage2ErrorEstimator().fit(oof, df, stage1.imputer_)
    wrapped = Stage2ErrorEstimator.from_fitted(fitted.forest_)
    features = stage2_features(df, stage1.imputer_, oof["oof_forecast"], oof["tree_disagreement"])
    assert wrapped.forest_ is fitted.forest_
    assert np.allclose(wrapped.predict(features), fitted.predict(features))


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
def test_run_id_is_stable_and_changes_when_a_setting_changes():
    a = make_run_id({"x": 1, "cfg": {"depth": 6}})
    assert a == make_run_id({"cfg": {"depth": 6}, "x": 1}) and a.startswith("run-")
    assert a != make_run_id({"x": 1, "cfg": {"depth": 7}})


# ---------------------------------------------------------------------------
# The estimator's fast route gives its forest's numbers; inputs are read by name
# ---------------------------------------------------------------------------
def _fitted_estimator():
    df = toy_cohort(300)
    oof = build_oof_table(df, _folds(df))
    stage1 = RandomForestModel().fit(df)
    estimator = Stage2ErrorEstimator().fit(oof, df, stage1.imputer_)
    forecast, spread = stage1.predict_with_disagreement(df)
    return estimator, stage2_features(df, stage1.imputer_, forecast, spread)


def test_the_estimator_gives_exactly_the_numbers_of_its_own_forest():
    estimator, features = _fitted_estimator()
    assert np.array_equal(estimator.predict(features), estimator.forest_.predict(features))


def test_the_estimator_reads_its_inputs_by_name_so_column_order_does_not_matter():
    estimator, features = _fitted_estimator()
    shuffled = features[list(reversed(STAGE2_INPUT_COLUMNS))]
    assert np.array_equal(estimator.predict(shuffled), estimator.predict(features))


def test_the_estimator_refuses_a_table_that_lacks_one_of_its_inputs():
    estimator, features = _fitted_estimator()
    with pytest.raises(KeyError):
        estimator.predict(features.drop(columns=["tree_disagreement"]))
