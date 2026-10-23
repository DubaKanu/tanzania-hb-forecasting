"""
Stage 2: estimate how large each Stage 1 forecast's error is likely to be.

The idea in plain words
-----------------------
Stage 1 forecasts a woman's haemoglobin at her next recorded visit. Its average error
hides which individual forecasts are bad. Stage 2 learns to predict the size of each
forecast's error, so forecasts can be ranked from most to least reliable. A LOWER
Stage 2 score means a MORE reliable forecast.

How Stage 2 is trained (no leakage)
-----------------------------------
1. Run the same 3-fold cross-validation as Stage 1 on the training women, using the
   frozen Stage 1 recipe (the forest). Every woman gets a forecast from a model that
   never saw her: an out-of-fold forecast. Her error is |observed Hb - that forecast|.
2. Train a separate random forest on those out-of-fold errors. Its inputs are the ten
   first-visit values, the out-of-fold forecast and the disagreement among the forest's
   individual trees. It never sees in-sample residuals, which are too optimistic.
3. For development and test women the frozen Stage 1 forest (fitted on all training
   women) supplies the forecast and the tree disagreement, and Stage 2 turns those
   plus the ten inputs into an estimated absolute error.

The same frozen Stage 1 forecasts feed every selector, so the comparison is paired.

Stage2ErrorEstimator owns its forest (composition, Figure 4 of the proposal): that forest
has no purpose outside the estimator. The estimator and the Stage 1 model are used
together through forecaster.TwoStageForecaster.

This module contains no execution code. It is used by run_stage2.py (development
data), run_locked_test.py (the locked test, once) and, through forecaster.py, by the demo API.
"""

import hashlib
import json
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from stage1 import RandomForestModel, cross_validated_predictions, forest_mean, tree_predictions

# ---------------------------------------------------------------------------
# Fixed configuration. Nothing here is tuned on any rows whose errors become
# Stage 2 targets, and none of it changes after the protocol is frozen.
# ---------------------------------------------------------------------------
STAGE2_INPUT_COLUMNS = FEATURE_COLUMNS + ["stage1_forecast", "tree_disagreement"]

STAGE2_CONFIG = {
    "seed": 42,
    "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
    "target": "absolute error of the out-of-fold Stage 1 forecast",
    "inputs": ("ten first-visit values (training-fitted median imputation), "
               "out-of-fold Stage 1 forecast, tree disagreement"),
}


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
def make_run_id(parts: dict) -> str:
    """Short, repeatable ID built from the things that define a run (data, split, folds, settings)."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return "run-" + hashlib.sha256(payload.encode("utf8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Out-of-fold table
# ---------------------------------------------------------------------------
def build_oof_table(train_df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Out-of-fold artefact: one row per training woman.

    Columns: record_id, fold, observed_hb, oof_forecast, tree_disagreement, abs_error.
    The forecast for each woman comes from a forest fitted on the other folds only.
    """
    oof, td, _ = cross_validated_predictions(train_df, RandomForestModel, folds, want_disagreement=True)
    y = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    return pd.DataFrame({
        "record_id": train_df["_source_row"].to_numpy(),
        "fold": np.asarray(folds, dtype=int),
        "observed_hb": y,
        "oof_forecast": oof,
        "tree_disagreement": td,
        "abs_error": np.abs(y - oof),
    })


def stage2_features(df: pd.DataFrame, imputer, forecast, disagreement) -> pd.DataFrame:
    """The 12 Stage 2 inputs: ten first-visit values (median-imputed with the training medians),
    the Stage 1 forecast and the tree disagreement."""
    filled = pd.DataFrame(imputer.transform(df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS,
                          index=df.index)
    filled["stage1_forecast"] = np.asarray(forecast, dtype=float)
    filled["tree_disagreement"] = np.asarray(disagreement, dtype=float)
    return filled[STAGE2_INPUT_COLUMNS]


# ---------------------------------------------------------------------------
# The Stage 2 estimator
# ---------------------------------------------------------------------------
class Stage2ErrorEstimator:
    """Predicts the absolute error of a Stage 1 forecast. Owns its own random forest (composition).

    The forest is a separate instance from the Stage 1 forest. It is trained on the
    whole out-of-fold table, never on residuals of a model fitted on the same rows.
    """

    def __init__(self, config: Optional[dict] = None):
        """Creates an unfitted forest with the fixed Stage 2 settings."""
        self.config = STAGE2_CONFIG if config is None else config
        self.forest_ = RandomForestRegressor(**self.config["random_forest"], random_state=self.config["seed"])

    @classmethod
    def from_fitted(cls, forest: RandomForestRegressor, config: Optional[dict] = None) -> "Stage2ErrorEstimator":
        """Wrap an already fitted forest (used when loading the saved Stage 2 model)."""
        estimator = cls(config)
        estimator.forest_ = forest
        return estimator

    def fit(self, oof_table: pd.DataFrame, train_df: pd.DataFrame, imputer) -> "Stage2ErrorEstimator":
        """Fit on the whole out-of-fold table; the table must be in the row order of train_df."""
        assert np.array_equal(oof_table["record_id"].to_numpy(), train_df["_source_row"].to_numpy()), \
            "out-of-fold table and training table are not in the same row order"
        X = stage2_features(train_df, imputer, oof_table["oof_forecast"], oof_table["tree_disagreement"])
        self.forest_.fit(X, oof_table["abs_error"].to_numpy(dtype=float))
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Estimated absolute error (g/dL) for rows built by stage2_features.

        The columns are put in the training order first, so a table with a column missing is
        refused and one with its columns shuffled still gets the right answer.
        """
        return forest_mean(tree_predictions(self.forest_, features[STAGE2_INPUT_COLUMNS]))
