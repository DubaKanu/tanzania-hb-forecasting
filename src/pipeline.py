"""
Tanzania antenatal haemoglobin forecasting pipeline.

Implements the methodology specified in the capstone proposal:
- Cohort construction with explicit eligibility rules
- Fixed train/dev/test partitioning (identical predictor vectors kept together)
- Stage 1: forecast next-contact Hb (median, carry-forward, two-input linear, ridge, compact RF)
- Cross-validation folds assigned by a fixed rule written in this file
- Coverage-based evaluation helpers shared by every selector

Stage 2 (reliability estimation and the selector comparison) lives in
stage2.py and builds on the functions here.

This module contains no execution code; it is imported by the notebook
(for exploration and reporting), by the run scripts and by app/main.py
(for the demo API).
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

# Fixed Stage 1 configuration. Nothing here is tuned on the data: the settings
# are set once, in advance, so held-out rows never influence model selection
# for the predictions that later become Stage 2 error targets.
STAGE1_CONFIG = {
    "n_folds": 3,
    "seed": 42,
    "ridge": {"alpha": 1.0, "scaler": "StandardScaler fitted inside each fold"},
    "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
}
TWO_INPUT_COLUMNS = ["hb_visit1_g_dl", "gestation_visit1_weeks"]
# Models that read the raw columns directly instead of the prepared matrix
DF_MODELS = ("median", "carry_forward", "two_input_linear")


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
def identical_vector_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group id per row; rows with identical values on the ten model
    inputs share an id (missing values count as equal to missing).

    This is a PROXY for "same woman", not proof. The release has no verified
    patient identifier, so independence between partitions cannot be guaranteed.
    """
    return df.groupby(FEATURE_COLUMNS, dropna=False, sort=False).ngroup().to_numpy()


def _keep_groups_together(parts: dict, groups: np.ndarray) -> dict:
    """Move members of any split group into one partition, swapping in an
    ungrouped row the other way so every partition keeps its exact size."""
    sizes = np.bincount(groups)
    part_of = {i: name for name, ids in parts.items() for i in ids}
    for g in np.where(sizes > 1)[0]:
        members = [int(i) for i in np.where(groups == g)[0] if int(i) in part_of]
        if not members:
            continue
        home = part_of[members[0]]
        for m in members[1:]:
            away = part_of[m]
            if away == home:
                continue
            swap = next(r for r in parts[home] if sizes[groups[r]] == 1)
            parts[home].remove(swap)
            parts[away].remove(m)
            parts[home].append(m)
            parts[away].append(swap)
            part_of[m], part_of[swap] = home, away
    return parts


def fixed_split(df: pd.DataFrame, seed: int = 42):
    """Deterministic train/dev/test split at the frozen sizes.

    Rows with identical predictor vectors are kept in the same partition. That
    reduces exact-duplicate leakage but does not prove separation of repeated
    pregnancies, patients or facilities (no verified patient or facility
    identifier exists in the release). Describe this as internal validation,
    never as temporal or external validation.
    """
    assert len(df) >= N_TRAIN + N_DEV + N_TEST, (
        f"Expected at least {N_TRAIN + N_DEV + N_TEST} eligible rows, got {len(df)}"
    )
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    parts = {
        "train": idx[:N_TRAIN].tolist(),
        "dev": idx[N_TRAIN:N_TRAIN + N_DEV].tolist(),
        "test": idx[N_TRAIN + N_DEV:N_TRAIN + N_DEV + N_TEST].tolist(),
    }
    parts = _keep_groups_together(parts, identical_vector_groups(df))
    return tuple(df.iloc[parts[k]].reset_index(drop=True) for k in ("train", "dev", "test"))


def partition_manifest(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Which source row sits in which partition (the saved split manifest)."""
    return pd.concat([
        pd.DataFrame({"_source_row": train["_source_row"], "partition": "train"}),
        pd.DataFrame({"_source_row": dev["_source_row"], "partition": "dev"}),
        pd.DataFrame({"_source_row": test["_source_row"], "partition": "test"}),
    ], ignore_index=True)


CV_FOLD_RULE = (
    "Groups of rows with an identical ten-input vector are never split. Groups are taken "
    "largest first (ties by group number, which follows the row order of train.csv) and each "
    "goes to the fold with the fewest rows so far (ties to the lowest fold number). "
    "Folds are numbered 0, 1, 2."
)


def assign_cv_folds(groups: np.ndarray, n_folds: int = 3) -> np.ndarray:
    """Fold number (0 .. n_folds-1) for every row, by a fixed rule.

    The rule is written out here instead of left to a library default. A library
    default (scikit-learn's GroupKFold) breaks ties between equal-sized groups in a
    way that changed between versions and moved the cross-validation MAE by about
    0.003 g/dL, which would change the out-of-fold table that Stage 2 learns from.
    This rule gives the same folds on every machine and every library version.
    """
    groups = np.asarray(groups)
    sizes = np.bincount(groups)
    # Largest group first; equal sizes in group-number order (lexsort is stable).
    order = np.lexsort((np.arange(len(sizes)), -sizes))
    load = np.zeros(n_folds, dtype=int)
    fold_of_group = np.zeros(len(sizes), dtype=int)
    for g in order:
        f = int(np.argmin(load))  # argmin returns the lowest fold number on a tie
        fold_of_group[g] = f
        load[f] += sizes[g]
    return fold_of_group[groups]


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


class TwoInputLinear:
    """Linear regression on first Hb and first gestational week only.

    Tests whether the other eight first-contact measurements add anything
    beyond these two simple inputs.
    """

    def fit(self, X, y):
        self.model_ = LinearRegression().fit(X[TWO_INPUT_COLUMNS], y)
        return self

    def predict(self, X):
        return self.model_.predict(X[TWO_INPUT_COLUMNS])


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
    oof_pred: Optional[np.ndarray] = None  # out-of-fold forecast for every training row


def stage1_candidate_factories(seed: int = 42) -> dict:
    """The five Stage 1 candidates, each built from the fixed STAGE1_CONFIG (nothing is tuned)."""
    cfg = STAGE1_CONFIG
    return {
        "median": lambda: MedianBaseline(),
        "carry_forward": lambda: CarryForwardBaseline(),
        "two_input_linear": lambda: TwoInputLinear(),
        "ridge": lambda: make_pipeline(
            StandardScaler(), Ridge(alpha=cfg["ridge"]["alpha"], random_state=seed)
        ),
        "random_forest": lambda: RandomForestRegressor(**cfg["random_forest"], random_state=seed),
    }


def cross_validated_predictions(train_df: pd.DataFrame, make_model, name: str, folds: np.ndarray,
                                want_disagreement: bool = False):
    """Out-of-fold forecast for every training row.

    Each row is forecast by a model fitted on the other folds only, so no row is ever
    predicted by a model that saw it. Imputation and the missing-value flags are fitted
    inside each fold. If want_disagreement is True and the model is a forest, the
    disagreement among its individual trees is recorded for every row as well.

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
        Xtr, imp = _prep_matrix(fold_train, fit=True)
        Xval, _ = _prep_matrix(fold_val, imputer=imp, fit=False)
        model = make_model()
        if name in DF_MODELS:
            model.fit(fold_train, fold_train[TARGET_COLUMN])
            pred = model.predict(fold_val)
        else:
            model.fit(Xtr, fold_train[TARGET_COLUMN])
            pred = model.predict(Xval)
            if want_disagreement:
                td[val_i] = tree_disagreement(model, Xval)
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
    y = train_df[TARGET_COLUMN].values
    folds = assign_cv_folds(identical_vector_groups(train_df), n_folds)

    results = {}
    for name, make_model in stage1_candidate_factories(seed).items():
        oof, _, fold_maes = cross_validated_predictions(train_df, make_model, name, folds)

        # refit on the full training partition for the frozen candidate
        Xtr_full, imp_full = _prep_matrix(train_df, fit=True)
        final_model = make_model()
        if name in DF_MODELS:
            final_model.fit(train_df, y)
        else:
            final_model.fit(Xtr_full, y)

        results[name] = Stage1Result(
            name=name, model=final_model, imputer=imp_full,
            cv_mae=float(np.mean(fold_maes)), oof_pred=oof,
        )
    return results


def stage1_predict(result: "Stage1Result", df: pd.DataFrame) -> np.ndarray:
    if result.name in DF_MODELS:
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


def n_keep(n_total: int, coverage: float) -> int:
    """How many forecasts an exact-coverage rule keeps: round(coverage x N), halves rounded up.

    For 80% of the 399 locked-test women this is 319; for 80% of the 402 development
    women it is 322. (The extra round() guards against floating-point noise such as
    3.4999999999999996 standing in for 3.5.)
    """
    return int(np.floor(round(n_total * coverage, 9) + 0.5))


def keep_mask(scores, coverage: float, record_ids=None) -> np.ndarray:
    """True for the forecasts kept at this coverage: the n lowest scores.

    A LOWER score means a MORE reliable forecast. Ties are broken by record ID,
    ascending, so the result never depends on a sorting library's tie behaviour.
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.arange(len(scores)) if record_ids is None else np.asarray(record_ids)
    order = np.lexsort((ids, scores))  # last key is the primary one: score, then record ID
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:n_keep(len(scores), coverage)]] = True
    return mask


def mae_at_coverage(errors, scores, coverage: float, record_ids=None) -> float:
    """MAE of the retained forecasts when the `coverage` fraction with the lowest scores is kept."""
    errors = np.asarray(errors, dtype=float)
    kept = keep_mask(scores, coverage, record_ids)
    return float(np.mean(errors[kept])) if kept.any() else float("nan")


def paired_bootstrap_difference(values_a, values_b, groups=None, n_boot: int = 5000, seed: int = 42):
    """Mean of (a - b) over paired cases with a 95% percentile interval.

    Resamples whole identical-vector groups with replacement (effectively single
    records, because only one pair of identical records exists). Used to ask whether
    one model's per-woman absolute errors are lower than another's on the same women.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    d = a - b
    g = np.arange(len(d)) if groups is None else np.asarray(groups)
    members = [np.where(g == k)[0] for k in np.unique(g)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        means[i] = d[np.concatenate([members[j] for j in pick])].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"difference": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}


def split_group_violations(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> int:
    """Number of identical-vector groups that appear in more than one partition (should be 0)."""
    parts = pd.concat([train.assign(_p="train"), dev.assign(_p="dev"), test.assign(_p="test")], ignore_index=True)
    g = identical_vector_groups(parts)
    n_parts = pd.Series(parts["_p"].values).groupby(g).nunique()
    return int((n_parts > 1).sum())
