"""
Tanzania antenatal haemoglobin forecasting: data preparation.

Implements the data side of the methodology specified in the capstone proposal:
- Cohort construction with explicit eligibility rules
- Fixed train/dev/test partitioning (identical predictor vectors kept together)
- Cross-validation folds assigned by a fixed rule written in this file

The models live elsewhere, one module per concern:
    stage1.py      Stage 1 forecasters (an abstract Stage1Model and five implementations)
    stage2.py      Stage 2 error estimator and the out-of-fold table
    forecaster.py  the two stages used together (also what the demo API loads)
    metrics.py     error metrics and the exact-coverage helpers
    selection.py   the ways of choosing which forecasts to keep
    evaluation.py  the protocol's comparison of those ways
    reporting.py   tables, figures and the written report

This module contains no execution code; it is imported by the notebook
(for exploration and reporting), by the run scripts and by the tests.
"""

import numpy as np
import pandas as pd

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


def _move_to_partition(parts: dict, part_of: dict, sizes: np.ndarray, groups: np.ndarray, m: int, home: str):
    """Put row m into partition `home` and send an ungrouped row of `home` the other way, so sizes stay exact."""
    away = part_of[m]
    swap = next(r for r in parts[home] if sizes[groups[r]] == 1)
    parts[home].remove(swap)
    parts[away].remove(m)
    parts[home].append(m)
    parts[away].append(swap)
    part_of[m], part_of[swap] = home, away


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
            if part_of[m] != home:
                _move_to_partition(parts, part_of, sizes, groups, m, home)
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


def split_group_violations(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> int:
    """Number of identical-vector groups that appear in more than one partition (should be 0)."""
    parts = pd.concat([train.assign(_p="train"), dev.assign(_p="dev"), test.assign(_p="test")], ignore_index=True)
    g = identical_vector_groups(parts)
    n_parts = pd.Series(parts["_p"].values).groupby(g).nunique()
    return int((n_parts > 1).sum())


# ---------------------------------------------------------------------------
# Cross-validation folds
# ---------------------------------------------------------------------------
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

    It is the greedy "longest processing time first" rule for balancing loads:
    sort the groups by size, then give each to the lightest fold. Cost: O(G log G)
    for G groups.
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
