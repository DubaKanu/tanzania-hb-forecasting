"""
Tanzania antenatal haemoglobin forecasting pipeline.

Implements the methodology specified in the capstone proposal:
- Cohort construction with explicit eligibility rules
- Fixed train/dev/test partitioning
- Stage 1: forecast next-contact Hb (median, carry-forward, ridge, compact RF)
- Stage 2: estimate forecast reliability from out-of-fold Stage 1 errors
- Coverage-based evaluation

This module contains no execution code; it is imported by the notebook
(for exploration and reporting) and by app/main.py (for the demo API).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Column contract (must match the data dictionary supplied by the supervisor)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "hb_visit1_g_dl",
    "gestation_visit1_weeks",
    "maternal_age_years",
    "pregnancy_count",
    "height_cm",
    "weight_visit1_kg",
    "pulse_visit1_bpm",
    "respiratory_rate_visit1",
    "systolic_bp_visit1",
    "diastolic_bp_visit1",
]
TARGET_COLUMN = "target_hb_visit2_g_dl"
GESTATION_VISIT2_COLUMN = "gestation_visit2_weeks"

# Fixed partition sizes, per the frozen protocol (do not change post hoc)
N_TRAIN, N_DEV, N_TEST = 1884, 402, 399


# ---------------------------------------------------------------------------
# Cohort construction
# ---------------------------------------------------------------------------
def apply_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the prespecified, audit-only eligibility screen.

    These are plausibility / completeness rules, not diagnostic thresholds.
    Never impute the outcome or silently repair implausible values here;
    excluded rows should be logged separately (see exclusion_log below).
    """
    out = df.copy()
    hb1_ok = out["hb_visit1_g_dl"].between(3, 20)
    hb2_ok = out[TARGET_COLUMN].between(3, 20)
    g1_ok = out["gestation_visit1_weeks"].between(1, 42)
    g2_ok = out[GESTATION_VISIT2_COLUMN].between(1, 42)
    order_ok = out[GESTATION_VISIT2_COLUMN] > out["gestation_visit1_weeks"]
    both_recorded = out["hb_visit1_g_dl"].notna() & out[TARGET_COLUMN].notna()

    eligible = hb1_ok & hb2_ok & g1_ok & g2_ok & order_ok & both_recorded
    return out.loc[eligible].reset_index(drop=True)


def exclusion_log(df: pd.DataFrame) -> pd.DataFrame:
    """Return a row-level log of which eligibility rule (if any) excluded each record."""
    reasons = []
    for _, row in df.iterrows():
        r = []
        if pd.isna(row.get("hb_visit1_g_dl")) or pd.isna(row.get(TARGET_COLUMN)):
            r.append("missing_hb_reading")
        else:
            if not (3 <= row["hb_visit1_g_dl"] <= 20):
                r.append("hb_visit1_out_of_range")
            if not (3 <= row[TARGET_COLUMN] <= 20):
                r.append("hb_visit2_out_of_range")
        if not (1 <= row.get("gestation_visit1_weeks", -1) <= 42):
            r.append("gestation1_out_of_range")
        if not (1 <= row.get(GESTATION_VISIT2_COLUMN, -1) <= 42):
            r.append("gestation2_out_of_range")
        if row.get(GESTATION_VISIT2_COLUMN, 0) <= row.get("gestation_visit1_weeks", 0):
            r.append("non_increasing_gestation")
        reasons.append(";".join(r) if r else "eligible")
    return pd.DataFrame({"reason": reasons})


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-feature missingness count among eligible records (Table 4 in the proposal)."""
    counts = df[FEATURE_COLUMNS].isna().sum()
    return counts.rename("n_missing").to_frame().assign(
        pct_missing=lambda d: (d["n_missing"] / len(df) * 100).round(1)
    )


# ---------------------------------------------------------------------------
# Fixed partitioning
# ---------------------------------------------------------------------------
def fixed_split(df: pd.DataFrame, seed: int = 42):
    """Deterministic train/dev/test split at the frozen sizes.

    NOTE: this is an internal split (no facility identifier is available to
    prove separation of repeated patients/facilities). Do not describe this
    as temporal or external validation.
    """
    assert len(df) >= N_TRAIN + N_DEV + N_TEST, (
        f"Expected at least {N_TRAIN + N_DEV + N_TEST} eligible rows, got {len(df)}"
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    train_idx = idx[:N_TRAIN]
    dev_idx = idx[N_TRAIN:N_TRAIN + N_DEV]
    test_idx = idx[N_TRAIN + N_DEV:N_TRAIN + N_DEV + N_TEST]
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[dev_idx].reset_index(drop=True),
        df.iloc[test_idx].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Stage 1 models
# ---------------------------------------------------------------------------
class MedianBaseline:
    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class CarryForwardBaseline:
    """Forecasts visit-2 Hb as equal to the woman's own visit-1 Hb."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.asarray(X["hb_visit1_g_dl"])


def _prep_matrix(X: pd.DataFrame, imputer: SimpleImputer = None, fit: bool = False):
    """Median-impute + missingness indicators, fit only on training data."""
    ind = X[FEATURE_COLUMNS].isna().astype(int).add_suffix("_missing")
    if fit:
        imputer = SimpleImputer(strategy="median")
        vals = imputer.fit_transform(X[FEATURE_COLUMNS])
    else:
        vals = imputer.transform(X[FEATURE_COLUMNS])
    filled = pd.DataFrame(vals, columns=FEATURE_COLUMNS, index=X.index)
    return pd.concat([filled, ind], axis=1), imputer


@dataclass
class Stage1Result:
    name: str
    model: object
    imputer: SimpleImputer
    cv_mae: float


def train_stage1_candidates(train_df: pd.DataFrame, n_folds: int = 3, seed: int = 42):
    """Train & compare the four Stage 1 candidates with grouped CV inside training.

    Returns a dict of Stage1Result, so the caller can pick the frozen winner
    by cv_mae (lower is better) rather than by peeking at development data.
    """
    y = train_df[TARGET_COLUMN].values
    groups = np.arange(len(train_df))  # no repeated-patient identifier available
    gkf = GroupKFold(n_splits=n_folds)

    candidates = {
        "median": lambda: MedianBaseline(),
        "carry_forward": lambda: CarryForwardBaseline(),
        "ridge": lambda: Ridge(alpha=1.0, random_state=seed),
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=seed
        ),
    }

    results = {}
    for name, make_model in candidates.items():
        fold_errors = []
        for train_i, val_i in gkf.split(train_df, groups=groups):
            fold_train, fold_val = train_df.iloc[train_i], train_df.iloc[val_i]
            Xtr, imp = _prep_matrix(fold_train, fit=True)
            Xval, _ = _prep_matrix(fold_val, imputer=imp, fit=False)
            model = make_model()
            if name in ("median", "carry_forward"):
                model.fit(fold_train, fold_train[TARGET_COLUMN])
                pred = model.predict(fold_val)
            else:
                model.fit(Xtr, fold_train[TARGET_COLUMN])
                pred = model.predict(Xval)
            fold_errors.append(mean_absolute_error(fold_val[TARGET_COLUMN], pred))

        # refit on the full training partition for the frozen candidate
        Xtr_full, imp_full = _prep_matrix(train_df, fit=True)
        final_model = make_model()
        if name in ("median", "carry_forward"):
            final_model.fit(train_df, y)
        else:
            final_model.fit(Xtr_full, y)

        results[name] = Stage1Result(
            name=name, model=final_model, imputer=imp_full, cv_mae=float(np.mean(fold_errors))
        )
    return results


def stage1_predict(result: "Stage1Result", df: pd.DataFrame) -> np.ndarray:
    if result.name in ("median", "carry_forward"):
        return result.model.predict(df)
    X, _ = _prep_matrix(df, imputer=result.imputer, fit=False)
    return result.model.predict(X)


def tree_disagreement(rf: RandomForestRegressor, X: pd.DataFrame) -> np.ndarray:
    """Std. dev. across individual trees' predictions -- the 'simple uncertainty' comparator."""
    per_tree = np.stack([t.predict(X.values) for t in rf.estimators_], axis=1)
    return per_tree.std(axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mean_signed_error(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mae_at_coverage(errors: np.ndarray, scores: np.ndarray, coverage: float) -> float:
    """Retain the `coverage` fraction with the LOWEST scores (most 'trustworthy'), report MAE."""
    n_keep = int(round(len(errors) * coverage))
    order = np.argsort(scores)  # ascending: most trustworthy first
    keep = order[:n_keep]
    return float(np.mean(errors[keep]))
