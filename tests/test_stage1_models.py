"""
Tests for src/stage1.py: the Stage1Model class hierarchy, polymorphic use, the preprocessing and
out-of-fold forecasts. Synthetic cohorts only; none of this is a result on the real cohort.
"""
import pickle

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from helpers import toy_cohort, toy_df
from metrics import mean_absolute_error
from pipeline import FEATURE_COLUMNS, STAGE1_CONFIG, TARGET_COLUMN, assign_cv_folds, identical_vector_groups
from stage1 import (
    CarryForwardBaseline, MatrixModel, MedianBaseline, RandomForestModel, RidgeModel, Stage1Model,
    TwoInputLinear, cross_validated_predictions, forest_mean, prep_matrix, stage1_candidate_factories,
    train_stage1_candidates, tree_disagreement, tree_predictions,
)

CONCRETE = [MedianBaseline, CarryForwardBaseline, TwoInputLinear, RidgeModel, RandomForestModel]


# ---------------------------------------------------------------------------
# The class hierarchy (inheritance and abstraction)
# ---------------------------------------------------------------------------
def test_the_five_models_share_one_abstract_parent():
    for cls in CONCRETE:
        assert issubclass(cls, Stage1Model)
    assert issubclass(RidgeModel, MatrixModel) and issubclass(RandomForestModel, MatrixModel)
    with pytest.raises(TypeError):
        Stage1Model()      # abstract: fit and predict are not implemented
    with pytest.raises(TypeError):
        MatrixModel()      # abstract: it does not say which estimator to build


def test_every_model_is_used_through_the_same_calls():
    df = toy_df(120, seed=1)
    for name, make in stage1_candidate_factories(seed=3).items():
        model = make()
        assert model.fit(df) is model                   # fit returns the model itself
        forecast = model.predict(df)
        assert model.name == name
        assert forecast.shape == (len(df),) and np.isfinite(forecast).all()


def test_only_the_forest_reports_tree_disagreement():
    df = toy_df(120, seed=2)
    for make in stage1_candidate_factories().values():
        model = make().fit(df)
        forecast, disagreement = model.predict_with_disagreement(df)
        assert np.allclose(forecast, model.predict(df))
        if isinstance(model, RandomForestModel):
            assert disagreement.shape == (len(df),) and (disagreement >= 0).all()
        else:
            assert disagreement is None


def test_a_sixth_model_plugs_into_cross_validation_without_changing_it():
    class TrainingMean(Stage1Model):
        name = "training_mean"

        def fit(self, train_df):
            self.mean_ = float(train_df[TARGET_COLUMN].mean())
            return self

        def predict(self, df):
            return np.full(len(df), self.mean_)

    df = toy_df(90, seed=3)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof, disagreement, fold_maes = cross_validated_predictions(df, TrainingMean, folds)
    assert disagreement is None and len(fold_maes) == 3
    for k in range(3):
        expected = df.loc[folds != k, TARGET_COLUMN].mean()
        assert np.allclose(oof[folds == k], expected)


# ---------------------------------------------------------------------------
# What each model does
# ---------------------------------------------------------------------------
def test_median_and_carry_forward_baselines():
    df = toy_df(60, seed=4)
    assert np.allclose(MedianBaseline().fit(df).predict(df), np.median(df[TARGET_COLUMN]))
    assert np.allclose(CarryForwardBaseline().fit(df).predict(df), df["hb_visit1_g_dl"])


def test_two_input_linear_reads_only_first_hb_and_first_gestation():
    df = toy_df(80, seed=5)
    model = TwoInputLinear().fit(df)
    changed = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in ("hb_visit1_g_dl", "gestation_visit1_weeks"):
            changed[column] = changed[column] + 100.0
    assert np.allclose(model.predict(df), model.predict(changed))


def test_forest_and_ridge_use_the_declared_settings():
    forest = RandomForestModel()._make_estimator()
    for key, value in STAGE1_CONFIG["random_forest"].items():
        assert getattr(forest, key) == value
    assert forest.random_state == STAGE1_CONFIG["seed"]
    ridge = RidgeModel()._make_estimator()
    assert ridge[-1].alpha == STAGE1_CONFIG["ridge"]["alpha"]


def test_tree_disagreement_is_the_spread_of_the_individual_trees():
    df = toy_df(100, seed=6)
    forest = RandomForestModel().fit(df)
    X, _ = prep_matrix(df, imputer=forest.imputer_)
    per_tree = np.array([tree.predict(X.values) for tree in forest.estimator_.estimators_])
    assert np.allclose(tree_disagreement(forest.estimator_, X), per_tree.std(axis=0))
    # and the forest forecast is the mean of its trees
    assert np.allclose(forest.predict(df), per_tree.mean(axis=0))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def test_prep_matrix_gives_twenty_columns_with_a_missing_flag_per_input():
    df = toy_cohort(60, seed=1)
    X, imputer = prep_matrix(df, fit=True)
    assert X.shape == (60, 20) and not X.isna().any().any()
    flags = X[[f"{c}_missing" for c in FEATURE_COLUMNS]]
    assert np.array_equal(flags.to_numpy(), df[FEATURE_COLUMNS].isna().to_numpy().astype(int))
    assert imputer is not None


def test_prep_matrix_fills_with_the_training_median_never_the_new_rows():
    train = toy_df(50, seed=7)
    new = toy_df(10, seed=8)
    new.loc[0, "pulse_visit1_bpm"] = np.nan
    _, imputer = prep_matrix(train, fit=True)
    filled, same_imputer = prep_matrix(new, imputer=imputer, fit=False)
    assert same_imputer is imputer                       # applied, not refitted
    assert filled.loc[0, "pulse_visit1_bpm"] == pytest.approx(train["pulse_visit1_bpm"].median())
    assert filled.loc[0, "pulse_visit1_bpm_missing"] == 1


# ---------------------------------------------------------------------------
# Out-of-fold forecasts and the candidate comparison
# ---------------------------------------------------------------------------
def test_out_of_fold_forecasts_come_from_models_that_never_saw_the_row():
    df = toy_df(150, seed=9)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof, _, _ = cross_validated_predictions(df, MedianBaseline, folds)
    for k in range(3):
        assert np.allclose(oof[folds == k], np.median(df.loc[folds != k, TARGET_COLUMN]))


def test_fold_scores_are_the_mean_absolute_error_of_each_held_out_fold():
    df = toy_df(150, seed=10)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof, _, fold_maes = cross_validated_predictions(df, CarryForwardBaseline, folds)
    for k, fold_mae in enumerate(fold_maes):
        held = folds == k
        assert fold_mae == pytest.approx(mean_absolute_error(df.loc[held, TARGET_COLUMN], oof[held]))


def test_asking_for_disagreement_from_a_model_without_trees_is_an_error():
    df = toy_df(60, seed=11)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    with pytest.raises(ValueError, match="no tree disagreement"):
        cross_validated_predictions(df, MedianBaseline, folds, want_disagreement=True)
    _, disagreement, _ = cross_validated_predictions(df, RandomForestModel, folds, want_disagreement=True)
    assert disagreement.shape == (60,) and np.isfinite(disagreement).all()


def test_stage1_candidates_cover_all_five_models():
    df = toy_df(300, seed=2)
    res = train_stage1_candidates(df, n_folds=3, seed=0)
    assert list(res) == ["median", "carry_forward", "two_input_linear", "ridge", "random_forest"]
    assert all(np.isfinite(r.cv_mae) for r in res.values())
    assert all(isinstance(r.model, Stage1Model) for r in res.values())


def test_stage1_candidates_keep_out_of_fold_forecasts():
    df = toy_df(300, seed=2)
    res = train_stage1_candidates(df, n_folds=3, seed=0)
    oof = res["random_forest"].oof_pred
    assert len(oof) == len(df) and np.isfinite(oof).all()


def test_matrix_models_expose_their_training_fitted_imputer_and_the_others_do_not():
    df = toy_df(120, seed=12)
    res = train_stage1_candidates(df, n_folds=3, seed=0)
    assert res["random_forest"].imputer is not None and res["ridge"].imputer is not None
    assert res["median"].imputer is None and res["carry_forward"].imputer is None


# ---------------------------------------------------------------------------
# Saving: plain scikit-learn objects, not this code's classes
# ---------------------------------------------------------------------------
def test_only_matrix_models_can_be_saved_as_a_bundle():
    df = toy_df(80, seed=13)
    for cls in (MedianBaseline, CarryForwardBaseline, TwoInputLinear):
        with pytest.raises(TypeError):
            cls().fit(df).to_bundle()
    bundle = RandomForestModel().fit(df).to_bundle()
    assert set(bundle) == {"model", "imputer", "name"} and bundle["name"] == "random_forest"
    assert isinstance(bundle["model"], RandomForestRegressor)


def test_a_saved_bundle_pickles_no_class_from_this_code(tmp_path):
    df = toy_df(80, seed=14)
    model = RandomForestModel().fit(df)
    blob = pickle.dumps(model.to_bundle())
    for own_module in (b"stage1", b"stage2", b"forecaster", b"pipeline", b"__main__"):
        assert own_module not in blob
    path = tmp_path / "stage1.joblib"
    joblib.dump(model.to_bundle(), path)
    loaded = joblib.load(path)
    rebuilt = RandomForestModel.from_fitted(loaded["model"], loaded["imputer"])
    assert np.allclose(rebuilt.predict(df), model.predict(df))


def test_from_fitted_wraps_without_refitting():
    df = toy_df(80, seed=15)
    fitted = RandomForestModel().fit(df)
    wrapped = RandomForestModel.from_fitted(fitted.estimator_, fitted.imputer_)
    assert wrapped.estimator_ is fitted.estimator_ and wrapped.imputer_ is fitted.imputer_
    assert np.allclose(wrapped.predict_with_disagreement(df)[1], fitted.predict_with_disagreement(df)[1])


# ---------------------------------------------------------------------------
# The fast route through the trees gives the numbers scikit-learn gives, to the last digit
# ---------------------------------------------------------------------------
def _fitted_forest(n=120, seed=3):
    df = toy_df(n, seed=seed)
    forest = RandomForestModel().fit(df)
    X, _ = prep_matrix(df, imputer=forest.imputer_)
    return df, forest, X


def test_each_trees_forecast_equals_the_one_scikit_learn_gives():
    _, forest, X = _fitted_forest()
    fast = tree_predictions(forest.estimator_, X)
    slow = np.stack([tree.predict(X.values) for tree in forest.estimator_.estimators_], axis=1)
    assert fast.shape == (len(X), STAGE1_CONFIG["random_forest"]["n_estimators"])
    assert np.array_equal(fast, slow)


def test_the_forest_forecast_is_identical_to_the_last_digit_not_just_close():
    _, forest, X = _fitted_forest()
    assert np.array_equal(forest_mean(tree_predictions(forest.estimator_, X)), forest.estimator_.predict(X))


def test_forest_mean_is_the_sum_of_the_trees_divided_by_how_many_there_are():
    assert forest_mean(np.array([[1.0, 2.0, 6.0], [0.0, 0.0, 3.0]])).tolist() == [3.0, 1.0]


def test_predict_with_disagreement_equals_the_slow_route_exactly():
    df, forest, X = _fitted_forest()
    forecast, spread = forest.predict_with_disagreement(df)
    slow_spread = np.stack([t.predict(X.values) for t in forest.estimator_.estimators_], axis=1).std(axis=1)
    assert np.array_equal(forecast, forest.estimator_.predict(X))
    assert np.array_equal(spread, slow_spread)


def test_a_woman_gets_the_same_numbers_alone_as_inside_a_batch():
    df, forest, _ = _fitted_forest()
    batch_forecast, batch_spread = forest.predict_with_disagreement(df.head(10))
    for i in range(10):
        forecast, spread = forest.predict_with_disagreement(df.iloc[[i]])
        assert forecast[0] == batch_forecast[i] and spread[0] == batch_spread[i]
