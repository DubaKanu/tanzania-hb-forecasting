"""
Stage 1: forecast a woman's haemoglobin at her next recorded visit from her first visit.

The design (Figure 4 of the proposal)
-------------------------------------
Stage1Model is an abstract class with five concrete implementations (inheritance):

    Stage1Model (abstract: fit, predict)
    |-- MedianBaseline          the training median, the same for everyone
    |-- CarryForwardBaseline    her own first Hb
    |-- TwoInputLinear          a line on first Hb and first gestational week only
    `-- MatrixModel (abstract)  models that read the median-imputed matrix with flags
        |-- RidgeModel
        `-- RandomForestModel   the only one with tree disagreement

Every model is fitted on, and predicts from, a DataFrame of women. Cross-validation,
evaluation and saving therefore treat all five the same way (polymorphism), and a sixth
model is added by writing one subclass. Nothing is tuned: every setting comes from
STAGE1_CONFIG in pipeline.py.

This module contains no execution code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from metrics import mean_absolute_error
from pipeline import (
    FEATURE_COLUMNS, STAGE1_CONFIG, TARGET_COLUMN, assign_cv_folds, identical_vector_groups,
)

TWO_INPUT_COLUMNS = ["hb_visit1_g_dl", "gestation_visit1_weeks"]


# ---------------------------------------------------------------------------
# Preprocessing shared by the models that need a matrix
# ---------------------------------------------------------------------------
def prep_matrix(X: pd.DataFrame, imputer: SimpleImputer = None, fit: bool = False):
    """Median-impute the ten inputs and add a missing-value flag for each, fit only on training data.

    With fit=True a new imputer is fitted on X (the rows of one training fold or one
    training partition); with fit=False the given imputer is applied unchanged.
    Returns (matrix with 20 columns, imputer).
    """
    ind = X[FEATURE_COLUMNS].isna().astype(int).add_suffix("_missing")
    if fit:
        imputer = SimpleImputer(strategy="median")
        vals = imputer.fit_transform(X[FEATURE_COLUMNS])
    else:
        vals = imputer.transform(X[FEATURE_COLUMNS])
    filled = pd.DataFrame(vals, columns=FEATURE_COLUMNS, index=X.index)
    return pd.concat([filled, ind], axis=1), imputer


def tree_predictions(forest: RandomForestRegressor, X) -> np.ndarray:
    """What every tree of the forest forecasts for every row: an array of shape (rows, trees).

    scikit-learn's own predict() re-checks its input once per tree. For a single woman that
    checking costs far more than the forecast, so the matrix is converted once to the float32 the
    trees work in and each fitted tree is asked directly. The numbers are the ones scikit-learn
    gives; tests/test_stage1_models.py checks that they match to the last digit.

    Cost: one pass of every tree over the rows, O(trees x rows x depth).
    """
    matrix = np.ascontiguousarray(np.asarray(X), dtype=np.float32)
    return np.stack([tree.tree_.predict(matrix)[:, 0] for tree in forest.estimators_], axis=1)


def forest_mean(per_tree: np.ndarray) -> np.ndarray:
    """The forest's forecast from its trees' forecasts, as RandomForestRegressor.predict adds them up.

    The trees are added one at a time, in order, and the sum is divided by the number of trees.
    numpy's own mean adds in a different order and can differ in the last digit, which is why it is not used.
    """
    total = np.zeros(per_tree.shape[0])
    for one_tree in per_tree.T:
        total += one_tree
    return total / per_tree.shape[1]


def tree_disagreement(rf: RandomForestRegressor, X) -> np.ndarray:
    """Std. dev. across the individual trees' forecasts: the 'simple uncertainty' comparator."""
    return tree_predictions(rf, X).std(axis=1)


# ---------------------------------------------------------------------------
# The abstract class and its implementations
# ---------------------------------------------------------------------------
class Stage1Model(ABC):
    """A Stage 1 forecaster of next-visit haemoglobin (abstract)."""

    name: str = ""

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "Stage1Model":
        """Fit on training women only (the target column must be present) and return self."""

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Forecast the next-visit haemoglobin (g/dL) of every row of df."""

    def predict_with_disagreement(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """(forecast, disagreement among trees). Only a forest has a disagreement; others return None."""
        return self.predict(df), None

    def to_bundle(self) -> dict:
        """The plain-object form saved in models/stage1_model.joblib (see MatrixModel)."""
        raise TypeError(f"{self.name} reads the raw columns and is not saved as a model bundle")


class MedianBaseline(Stage1Model):
    """The training median, the same forecast for every woman."""

    name = "median"

    def fit(self, train_df):
        """Remember the training median of the target."""
        self.median_ = float(np.median(train_df[TARGET_COLUMN]))
        return self

    def predict(self, df):
        """The same training median for every woman."""
        return np.full(len(df), self.median_)


class CarryForwardBaseline(Stage1Model):
    """Forecasts visit-2 Hb as equal to the woman's own visit-1 Hb."""

    name = "carry_forward"

    def fit(self, train_df):
        """Nothing to learn."""
        return self

    def predict(self, df):
        """The first Hb is the forecast."""
        return np.asarray(df["hb_visit1_g_dl"])


class TwoInputLinear(Stage1Model):
    """Linear regression on first Hb and first gestational week only.

    Tests whether the other eight first-contact measurements add anything
    beyond these two simple inputs.
    """

    name = "two_input_linear"

    def fit(self, train_df):
        """Fit a line on first Hb and gestational week."""
        self.model_ = LinearRegression().fit(train_df[TWO_INPUT_COLUMNS], train_df[TARGET_COLUMN])
        return self

    def predict(self, df):
        """Forecast from first Hb and gestational week."""
        return self.model_.predict(df[TWO_INPUT_COLUMNS])


class MatrixModel(Stage1Model):
    """A model that reads the median-imputed matrix with a missing-value flag per input.

    The imputation medians are fitted on the rows given to fit() only, so inside
    cross-validation every fold imputes with its own training rows and nothing leaks.
    Subclasses only say which scikit-learn estimator to use.
    """

    @abstractmethod
    def _make_estimator(self):
        """A new, unfitted scikit-learn estimator with the fixed settings."""

    def fit(self, train_df):
        """Fit the imputer on these rows, then the estimator on the imputed matrix."""
        X, self.imputer_ = prep_matrix(train_df, fit=True)
        self.estimator_ = self._make_estimator().fit(X, train_df[TARGET_COLUMN])
        return self

    def predict(self, df):
        """Forecast from the imputed matrix, which carries a missing-value flag per input."""
        X, _ = prep_matrix(df, imputer=self.imputer_, fit=False)
        return self.estimator_.predict(X)

    def to_bundle(self) -> dict:
        """Plain scikit-learn objects in a dict, so the saved file loads without this code's classes."""
        return {"model": self.estimator_, "imputer": self.imputer_, "name": self.name}


class RidgeModel(MatrixModel):
    """Ridge regression on the standardised matrix."""

    name = "ridge"

    def __init__(self, seed: int = STAGE1_CONFIG["seed"]):
        """`seed` fixes the model's random state."""
        self.seed = seed

    def _make_estimator(self):
        """Standard scaling, then ridge regression with the fixed alpha."""
        return make_pipeline(StandardScaler(), Ridge(alpha=STAGE1_CONFIG["ridge"]["alpha"], random_state=self.seed))


class RandomForestModel(MatrixModel):
    """The compact random forest: the frozen Stage 1 model, and the source of tree disagreement."""

    name = "random_forest"

    def __init__(self, seed: int = STAGE1_CONFIG["seed"]):
        """`seed` fixes the forest's random state."""
        self.seed = seed

    def _make_estimator(self):
        """The forest with the fixed settings (200 trees, depth 6, minimum leaf 5)."""
        return RandomForestRegressor(**STAGE1_CONFIG["random_forest"], random_state=self.seed)

    def predict_with_disagreement(self, df):
        """The forecast and the spread of the trees, both from one pass over the trees."""
        X, _ = prep_matrix(df, imputer=self.imputer_, fit=False)
        per_tree = tree_predictions(self.estimator_, X)
        return forest_mean(per_tree), per_tree.std(axis=1)

    @classmethod
    def from_fitted(cls, estimator: RandomForestRegressor, imputer: SimpleImputer,
                    seed: int = STAGE1_CONFIG["seed"]) -> "RandomForestModel":
        """Wrap an already fitted forest and its training-fitted imputer (used when loading a saved model)."""
        model = cls(seed)
        model.estimator_, model.imputer_ = estimator, imputer
        return model


def stage1_candidate_factories(seed: int = 42) -> dict:
    """The five Stage 1 candidates, in the order they are reported (nothing is tuned)."""
    return {
        MedianBaseline.name: MedianBaseline,
        CarryForwardBaseline.name: CarryForwardBaseline,
        TwoInputLinear.name: TwoInputLinear,
        RidgeModel.name: lambda: RidgeModel(seed),
        RandomForestModel.name: lambda: RandomForestModel(seed),
    }


# ---------------------------------------------------------------------------
# Out-of-fold forecasts and the candidate comparison
# ---------------------------------------------------------------------------
@dataclass
class Stage1Result:
    """One candidate after training: refitted on the whole training partition, with its CV scores."""

    name: str
    model: Stage1Model
    cv_mae: float
    oof_pred: Optional[np.ndarray] = None  # out-of-fold forecast for every training row

    @property
    def imputer(self) -> Optional[SimpleImputer]:
        """The training-fitted imputer, for models that use one."""
        return getattr(self.model, "imputer_", None)


def cross_validated_predictions(train_df: pd.DataFrame, make_model: Callable[[], Stage1Model],
                                folds: np.ndarray, want_disagreement: bool = False):
    """Out-of-fold forecast for every training row.

    Each row is forecast by a model fitted on the other folds only, so no row is ever
    predicted by a model that saw it. Imputation and the missing-value flags are fitted
    inside each fold (the model does it in fit). If want_disagreement is True the model
    must be a forest: the disagreement among its individual trees is recorded for every
    row as well.

    Returns (forecast, disagreement or None, list of the mean absolute error of each fold).
    """
    n = len(train_df)
    oof = np.full(n, np.nan)
    td = np.full(n, np.nan) if want_disagreement else None
    fold_maes = []
    for k in np.unique(folds):
        train_i = np.where(folds != k)[0]
        val_i = np.where(folds == k)[0]
        fold_train, fold_val = train_df.iloc[train_i], train_df.iloc[val_i]
        model = make_model().fit(fold_train)
        if want_disagreement:
            pred, disagreement = model.predict_with_disagreement(fold_val)
            if disagreement is None:
                raise ValueError(f"{model.name} has no tree disagreement; use a random forest")
            td[val_i] = disagreement
        else:
            pred = model.predict(fold_val)
        oof[val_i] = pred
        fold_maes.append(mean_absolute_error(fold_val[TARGET_COLUMN], pred))
    return oof, td, fold_maes


def train_stage1_candidates(train_df: pd.DataFrame, n_folds: int = 3, seed: int = 42):
    """Train and compare the five Stage 1 candidates inside the training partition.

    Cross-validation is 3-fold. Folds come from assign_cv_folds, which keeps rows that
    share an identical predictor vector together (a proxy for patient grouping, not
    proof of independence). Imputation, missingness indicators and scaling are fitted
    inside each fold. Settings are fixed (STAGE1_CONFIG), not tuned.

    Returns a dict of Stage1Result so the caller can freeze the winner by
    cv_mae (lower is better) without looking at development data.
    """
    folds = assign_cv_folds(identical_vector_groups(train_df), n_folds)

    results = {}
    for name, make_model in stage1_candidate_factories(seed).items():
        oof, _, fold_maes = cross_validated_predictions(train_df, make_model, folds)
        final_model = make_model().fit(train_df)  # refit on the full training partition
        results[name] = Stage1Result(
            name=name, model=final_model, cv_mae=float(np.mean(fold_maes)), oof_pred=oof,
        )
    return results


def stage1_predict(result: Stage1Result, df: pd.DataFrame) -> np.ndarray:
    """Forecast every row of df with a trained candidate."""
    return result.model.predict(df)
