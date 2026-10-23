"""
Tests for src/forecaster.py: the two stages used together, the demo cutoff, and saving and loading.

The last tests compare the forecaster, built from the saved models in models/, with the
development results that were saved before the code was reorganised into classes.
"""
import os

import joblib
import numpy as np
import pandas as pd
import pytest

from forecaster import DEMO_CUTOFF_RULE, ForecastBatch, TwoStageForecaster
from helpers import toy_cohort
from pipeline import FEATURE_COLUMNS
from stage1 import MedianBaseline, RandomForestModel, prep_matrix
from stage2 import STAGE2_INPUT_COLUMNS, stage2_features

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = [os.path.join(ROOT, "models", "stage1_model.joblib"), os.path.join(ROOT, "models", "stage2_model.joblib")]
DATA = os.path.join(ROOT, "data", "Josephine_African_Data_Review", "Tanzania_Followup_Haemoglobin")
DEV_DATA = os.path.join(DATA, "validation.csv")
DEV_TABLE = os.path.join(ROOT, "reports", "stage2_dev_table.csv")


# ---------------------------------------------------------------------------
# Using the two stages together
# ---------------------------------------------------------------------------
def test_forecast_is_stage1_then_stage2():
    df = toy_cohort(250)
    forecaster = TwoStageForecaster.fit(df)
    rows = df.head(15)
    batch = forecaster.forecast(rows)
    forecast, disagreement = forecaster.stage1.predict_with_disagreement(rows)
    assert isinstance(batch, ForecastBatch)
    assert np.allclose(batch.forecast, forecast) and np.allclose(batch.tree_disagreement, disagreement)
    features = stage2_features(rows, forecaster.stage1.imputer_, forecast, disagreement)
    assert np.allclose(batch.estimated_abs_error, forecaster.stage2.predict(features))
    assert (batch.estimated_abs_error >= 0).all()   # an estimated absolute error cannot be negative


def test_the_batch_has_one_entry_per_woman_in_the_same_order():
    df = toy_cohort(200)
    forecaster = TwoStageForecaster.fit(df)
    whole = forecaster.forecast(df.head(20))
    one_by_one = [forecaster.forecast(df.iloc[[i]]).forecast[0] for i in range(20)]
    assert len(whole.forecast) == 20
    assert np.allclose(whole.forecast, one_by_one)


def test_fit_keeps_the_out_of_fold_table_and_folds_it_used():
    df = toy_cohort(200)
    forecaster = TwoStageForecaster.fit(df)
    assert len(forecaster.oof_table_) == 200 and len(forecaster.folds_) == 200
    assert forecaster.train_hb_median == pytest.approx(np.median(df["hb_visit1_g_dl"]))


def test_fit_can_reuse_a_stage1_forest_that_was_already_fitted():
    df = toy_cohort(200)
    stage1 = RandomForestModel().fit(df)
    forecaster = TwoStageForecaster.fit(df, stage1=stage1)
    assert forecaster.stage1 is stage1


# ---------------------------------------------------------------------------
# The demo cutoff
# ---------------------------------------------------------------------------
def test_demo_cutoff_is_a_percentile_of_the_development_scores_and_decides_kept_or_not():
    df = toy_cohort(200)
    forecaster = TwoStageForecaster.fit(df)
    scores = np.linspace(0.1, 1.0, 101)
    cutoff = forecaster.calibrate_demo_cutoff(scores)
    assert cutoff == pytest.approx(np.percentile(scores, 80))
    assert forecaster.demo_cutoff == cutoff
    assert forecaster.is_kept(cutoff) is True            # at the cutoff: kept
    assert forecaster.is_kept(cutoff + 1e-9) is False    # just above: set aside


def test_is_kept_refuses_to_answer_before_a_cutoff_is_set():
    forecaster = TwoStageForecaster.fit(toy_cohort(150))
    with pytest.raises(ValueError, match="no demo cutoff"):
        forecaster.is_kept(0.5)


# ---------------------------------------------------------------------------
# Saving and loading
# ---------------------------------------------------------------------------
def test_bundles_round_trip_to_an_identical_forecaster(tmp_path):
    df = toy_cohort(250)
    original = TwoStageForecaster.fit(df)
    original.run_id = "run-test"
    original.calibrate_demo_cutoff(original.forecast(df).estimated_abs_error)
    bundle1, bundle2 = original.to_bundles({"python": "x"})
    joblib.dump(bundle1, tmp_path / "s1.joblib")
    joblib.dump(bundle2, tmp_path / "s2.joblib")
    loaded = TwoStageForecaster.from_files(str(tmp_path / "s1.joblib"), str(tmp_path / "s2.joblib"))
    rows = df.head(40)
    a, b = original.forecast(rows), loaded.forecast(rows)
    assert np.array_equal(a.forecast, b.forecast) and np.array_equal(a.estimated_abs_error, b.estimated_abs_error)
    assert (loaded.run_id, loaded.demo_cutoff, loaded.train_hb_median) == \
        ("run-test", original.demo_cutoff, original.train_hb_median)


def test_the_stage2_bundle_has_the_layout_the_old_files_had():
    forecaster = TwoStageForecaster.fit(toy_cohort(150))
    forecaster.calibrate_demo_cutoff([0.1, 0.2, 0.3])
    _, bundle2 = forecaster.to_bundles({"python": "x"})
    assert set(bundle2) == {"model", "run_id", "config", "input_columns", "train_hb_median", "demo_cutoff",
                            "demo_cutoff_rule", "versions"}
    assert bundle2["input_columns"] == STAGE2_INPUT_COLUMNS and bundle2["demo_cutoff_rule"] == DEMO_CUTOFF_RULE


def test_loading_refuses_a_stage1_model_that_is_not_the_forest():
    df = toy_cohort(120)
    forest = TwoStageForecaster.fit(df)
    _, bundle2 = forest.to_bundles({})
    not_forest = {"model": MedianBaseline().fit(df), "imputer": forest.stage1.imputer_, "name": "median"}
    with pytest.raises(ValueError, match="needs the forest"):
        TwoStageForecaster.from_bundles(not_forest, bundle2)


# ---------------------------------------------------------------------------
# The saved models give the same numbers as the results saved before the reorganisation
# ---------------------------------------------------------------------------
needs_models = pytest.mark.skipif(not all(os.path.exists(p) for p in MODELS + [DEV_DATA, DEV_TABLE]),
                                  reason="saved models or development files not found")


@needs_models
def test_the_saved_models_reproduce_the_saved_development_table_for_all_402_women():
    dev = pd.read_csv(DEV_DATA)
    saved = pd.read_csv(DEV_TABLE)
    forecaster = TwoStageForecaster.from_files(*MODELS)
    batch = forecaster.forecast(dev)
    assert len(dev) == len(saved) == 402
    assert np.array_equal(dev["_source_row"].to_numpy(), saved["record_id"].to_numpy())
    assert np.abs(batch.forecast - saved["forecast"].to_numpy()).max() < 6e-7       # the table is rounded to 6 d.p.
    assert np.abs(batch.estimated_abs_error - saved["stage2_score"].to_numpy()).max() < 6e-7
    assert np.abs(batch.tree_disagreement - saved["tree_disagreement"].to_numpy()).max() < 6e-7


@needs_models
def test_the_saved_forecaster_has_the_run_id_and_cutoff_written_in_the_results():
    import json
    with open(os.path.join(ROOT, "reports", "stage2_dev_results.json"), encoding="utf8") as f:
        results = json.load(f)
    forecaster = TwoStageForecaster.from_files(*MODELS)
    assert forecaster.run_id == results["run_id"]
    assert forecaster.demo_cutoff == pytest.approx(results["demo_cutoff"])
    assert list(forecaster.stage2.forest_.feature_names_in_) == STAGE2_INPUT_COLUMNS
    assert set(FEATURE_COLUMNS) < set(STAGE2_INPUT_COLUMNS)


@needs_models
def test_the_fast_route_equals_scikit_learn_for_the_saved_models_on_all_402_women():
    dev = pd.read_csv(DEV_DATA)
    forecaster = TwoStageForecaster.from_files(*MODELS)
    batch = forecaster.forecast(dev)
    X, _ = prep_matrix(dev, imputer=forecaster.stage1.imputer_)
    slow_forecast = forecaster.stage1.estimator_.predict(X)
    slow_spread = np.stack([t.predict(X.values) for t in forecaster.stage1.estimator_.estimators_], axis=1).std(axis=1)
    features = stage2_features(dev, forecaster.stage1.imputer_, slow_forecast, slow_spread)
    assert np.array_equal(batch.forecast, slow_forecast)
    assert np.array_equal(batch.tree_disagreement, slow_spread)
    assert np.array_equal(batch.estimated_abs_error, forecaster.stage2.forest_.predict(features))
