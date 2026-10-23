"""
The two stages used together.

TwoStageForecaster holds the frozen Stage 1 forest and the Stage 2 error estimator and answers
one question: for these women, what is the forecast, and how large is its error likely to be?
The evaluation scripts, the locked test and the demo API all ask it through the same method
(forecast), so the API output equals the offline output for the same woman by construction.

Saved models are plain scikit-learn objects in two dictionaries ("bundles"), so the files in
models/ load without any class from this code. from_bundles / to_bundles are the only places
that know the bundle layout.

This module contains no execution code.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from pipeline import STAGE1_CONFIG, assign_cv_folds, identical_vector_groups
from stage1 import RandomForestModel
from stage2 import STAGE2_INPUT_COLUMNS, Stage2ErrorEstimator, build_oof_table, stage2_features

DEMO_CUTOFF_RULE = ("80th percentile of the Stage 2 scores of the development women; "
                    "fitted on development data, so different from exact-coverage evaluation")


@dataclass(frozen=True)
class ForecastBatch:
    """What the forecaster returns for a table of women (one entry per woman, in the same order)."""

    forecast: np.ndarray             # Stage 1: haemoglobin at the next recorded visit (g/dL)
    tree_disagreement: np.ndarray    # spread of the forest's individual trees (g/dL)
    estimated_abs_error: np.ndarray  # Stage 2: how far the forecast is likely to be from the real value (g/dL)


class TwoStageForecaster:
    """Stage 1 forest + Stage 2 error estimator, used as one object (composition)."""

    def __init__(self, stage1: RandomForestModel, stage2: Stage2ErrorEstimator, train_hb_median: float,
                 demo_cutoff: Optional[float] = None, run_id: Optional[str] = None):
        """Takes the fitted Stage 1 forest, the fitted Stage 2 estimator, the training median first Hb
        (for the simple rule) and the demonstration cutoff."""
        self.stage1 = stage1
        self.stage2 = stage2
        self.train_hb_median = float(train_hb_median)  # the simple rule needs it
        self.demo_cutoff = demo_cutoff
        self.run_id = run_id

    # -- building ------------------------------------------------------------------------
    @classmethod
    def fit(cls, train_df: pd.DataFrame, stage1: Optional[RandomForestModel] = None) -> "TwoStageForecaster":
        """Out-of-fold table, then Stage 2, for one training set.

        If stage1 is None the Stage 1 forest is fitted here (used by the complete-case
        sensitivity analysis, which repeats everything on a subset). The out-of-fold table and
        the folds are kept as oof_table_ and folds_.
        """
        stage1 = RandomForestModel().fit(train_df) if stage1 is None else stage1
        folds = assign_cv_folds(identical_vector_groups(train_df), STAGE1_CONFIG["n_folds"])
        oof = build_oof_table(train_df, folds)
        stage2 = Stage2ErrorEstimator().fit(oof, train_df, stage1.imputer_)
        built = cls(stage1, stage2, float(np.median(train_df["hb_visit1_g_dl"])))
        built.oof_table_, built.folds_ = oof, folds
        return built

    # -- using ---------------------------------------------------------------------------
    def forecast(self, df: pd.DataFrame) -> ForecastBatch:
        """Forecast, tree disagreement and estimated absolute error for every row of df."""
        forecast, disagreement = self.stage1.predict_with_disagreement(df)
        features = stage2_features(df, self.stage1.imputer_, forecast, disagreement)
        return ForecastBatch(forecast, disagreement, self.stage2.predict(features))

    def calibrate_demo_cutoff(self, development_scores, percentile: float = 80.0) -> float:
        """Set the API cutoff: a percentile of the Stage 2 scores of the development women.

        One woman cannot be ranked against others, so the API needs a cutoff. It is fitted on
        development data and is NOT the exact-coverage rule used in the evaluation.
        """
        self.demo_cutoff = float(np.percentile(development_scores, percentile))
        return self.demo_cutoff

    def is_kept(self, estimated_abs_error: float) -> bool:
        """True if the estimated error is at or below the demo cutoff (the forecast would be kept)."""
        if self.demo_cutoff is None:
            raise ValueError("no demo cutoff has been set; call calibrate_demo_cutoff first")
        return estimated_abs_error <= self.demo_cutoff

    # -- saving and loading --------------------------------------------------------------
    def to_bundles(self, versions: dict) -> Tuple[dict, dict]:
        """(Stage 1 bundle, Stage 2 bundle): the plain-object layout of the two files in models/."""
        bundle2 = {
            "model": self.stage2.forest_, "run_id": self.run_id, "config": self.stage2.config,
            "input_columns": STAGE2_INPUT_COLUMNS, "train_hb_median": self.train_hb_median,
            "demo_cutoff": self.demo_cutoff, "demo_cutoff_rule": DEMO_CUTOFF_RULE,
            "versions": versions,
        }
        return self.stage1.to_bundle(), bundle2

    @classmethod
    def from_bundles(cls, bundle1: dict, bundle2: dict) -> "TwoStageForecaster":
        """Rebuild the forecaster from the two saved dictionaries."""
        if bundle1["name"] != RandomForestModel.name:
            raise ValueError("Stage 2 needs the forest: tree disagreement exists only for a forest")
        return cls(
            RandomForestModel.from_fitted(bundle1["model"], bundle1["imputer"]),
            Stage2ErrorEstimator.from_fitted(bundle2["model"], bundle2.get("config")),
            bundle2["train_hb_median"], bundle2.get("demo_cutoff"), bundle2.get("run_id"),
        )

    @classmethod
    def from_files(cls, stage1_path: str, stage2_path: str) -> "TwoStageForecaster":
        """Load the two saved model files."""
        return cls.from_bundles(joblib.load(stage1_path), joblib.load(stage2_path))
