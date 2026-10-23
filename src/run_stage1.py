"""
Versioned rerun of preparation and Stage 1 baselines.

Raw release -> eligible cohort -> frozen partitions -> Stage 1 comparison ->
frozen Stage 1 model. Run from the repository root:

    python src/run_stage1.py

The reserved test partition is written to disk but is never evaluated here.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_adapter import load_and_map  # noqa: E402
from metrics import mean_absolute_error, mean_signed_error, paired_bootstrap_difference, rmse  # noqa: E402
from pipeline import (  # noqa: E402
    CV_FOLD_RULE, STAGE1_CONFIG, TARGET_COLUMN, apply_eligibility, assign_cv_folds, fixed_split,
    identical_vector_groups, partition_manifest, split_group_violations,
)
from provenance import library_versions, md5_of_file, md5_of_frame  # noqa: E402
from stage1 import stage1_predict, train_stage1_candidates  # noqa: E402

RAW = "data/raw/maternal_dataset_csv.csv"
RAW_MD5 = "0253d2d385dd80d6c9bb82df6b85fcc8"  # pinned Zenodo release, Version 1
OUT = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"
SIMPLER_MODELS = ("median", "carry_forward", "two_input_linear", "ridge")
BOOTSTRAP_REPLICATES = 5000


def prepare_partitions(seed: int):
    """Raw file -> eligible cohort -> fixed train / development / test partitions."""
    assert md5_of_file(RAW) == RAW_MD5, "Raw file does not match the pinned release checksum"
    eligible = apply_eligibility(load_and_map(RAW))
    train, dev, test = fixed_split(eligible, seed=seed)
    violations = split_group_violations(train, dev, test)
    assert violations == 0, f"{violations} identical-vector groups straddle partitions"
    return eligible, train, dev, test, violations


def save_partitions(eligible, train, dev, test) -> pd.DataFrame:
    """Write the cohort, the three partitions and the manifest of who is in which."""
    os.makedirs(OUT, exist_ok=True)
    eligible.to_csv(f"{OUT}/model_ready.csv", index=False)
    train.to_csv(f"{OUT}/train.csv", index=False)
    dev.to_csv(f"{OUT}/validation.csv", index=False)
    test.to_csv(f"{OUT}/test.csv", index=False)  # never evaluated until the frozen final step
    manifest = partition_manifest(train, dev, test)
    manifest.to_csv(f"{OUT}/split_manifest.csv", index=False)
    return manifest


def save_folds(train) -> pd.DataFrame:
    """Cross-validation folds: fixed rule, saved so every later step reads the same folds."""
    folds = assign_cv_folds(identical_vector_groups(train), STAGE1_CONFIG["n_folds"])
    folds_table = pd.DataFrame({"_source_row": train["_source_row"], "cv_fold": folds})
    folds_table.to_csv(f"{OUT}/cv_folds.csv", index=False)
    return folds_table


def score_candidates(results, train, dev):
    """Each candidate's cross-validation and development errors, and its per-woman absolute errors."""
    rows, dev_abs_err, oof_abs_err = {}, {}, {}
    y_dev = dev[TARGET_COLUMN].to_numpy()
    y_train = train[TARGET_COLUMN].to_numpy()
    for name, result in results.items():
        pred = stage1_predict(result, dev)
        rows[name] = {
            "train_cv_mae": round(result.cv_mae, 3),
            "dev_mae": round(mean_absolute_error(dev[TARGET_COLUMN], pred), 3),
            "dev_rmse": round(rmse(dev[TARGET_COLUMN], pred), 3),
            "dev_mean_signed_error": round(mean_signed_error(dev[TARGET_COLUMN], pred), 3),
        }
        dev_abs_err[name] = np.abs(y_dev - pred)
        oof_abs_err[name] = np.abs(y_train - result.oof_pred)
    return rows, dev_abs_err, oof_abs_err


def compare_forest_with_simpler_models(train, dev, oof_abs_err, dev_abs_err, seed: int) -> dict:
    """Does the forest beat each simpler model on the same women?

    Paired bootstrap of the difference in mean absolute error (forest minus the other model;
    negative favours the forest), on the training out-of-fold errors and on the development errors.
    """
    paired = {}
    for label, errs, groups in (("training_out_of_fold", oof_abs_err, identical_vector_groups(train)),
                                ("development", dev_abs_err, identical_vector_groups(dev))):
        paired[label] = {}
        for other in SIMPLER_MODELS:
            d = paired_bootstrap_difference(errs["random_forest"], errs[other], groups=groups,
                                            n_boot=BOOTSTRAP_REPLICATES, seed=seed)
            paired[label][f"random_forest_minus_{other}"] = {k: round(v, 3) for k, v in d.items()}
    return paired


def freeze_stage1_model(results) -> str:
    """Frozen Stage 1 model = lowest training cross-validation MAE (not development MAE). Saves it."""
    frozen = min(results, key=lambda name: results[name].cv_mae)
    os.makedirs("models", exist_ok=True)
    joblib.dump(results[frozen].model.to_bundle(), "models/stage1_model.joblib")
    return frozen


def build_report(eligible, train, dev, test, violations, manifest, folds_table, frozen, rows, paired) -> dict:
    """Everything run_stage2.py and the locked test later check against."""
    return {
        "raw_file": "maternal_dataset_csv.csv (Zenodo 10.5281/zenodo.15309733, Version 1)",
        "raw_md5": RAW_MD5,
        "eligible_rows": len(eligible),
        "partitions": {"train": len(train), "dev": len(dev), "test": len(test)},
        "identical_vector_groups_split_across_partitions": violations,
        "manifest_md5": md5_of_frame(manifest),
        "cv_folds_md5": md5_of_frame(folds_table),
        "cv_fold_rule": CV_FOLD_RULE,
        "config": STAGE1_CONFIG,
        "frozen_stage1_model": frozen,
        "metrics_g_per_dl": rows,
        "paired_comparisons_mae_difference_g_per_dl": {
            "meaning": "random_forest minus the other model, mean absolute error on the same women; "
                       "negative favours the forest. Paired bootstrap, 5000 replicates, seed 42, 95% interval.",
            **paired,
        },
        "versions": library_versions(),
    }


def print_summary(eligible, train, dev, test, violations, rows, frozen, paired):
    """Print the partition counts and the Stage 1 comparison."""
    print(f"eligible: {len(eligible)}  partitions: {len(train)}/{len(dev)}/{len(test)}  group violations: {violations}")
    print(f"{'model':18s} {'train CV MAE':>12s} {'dev MAE':>9s} {'dev RMSE':>9s}")
    for name, m in rows.items():
        print(f"{name:18s} {m['train_cv_mae']:12.3f} {m['dev_mae']:9.3f} {m['dev_rmse']:9.3f}")
    print(f"frozen Stage 1 model (lowest training CV MAE): {frozen}")
    print("forest minus two-input line (g/dL, negative favours the forest):")
    for label in ("training_out_of_fold", "development"):
        d = paired[label]["random_forest_minus_two_input_linear"]
        print(f"  {label:22s} {d['difference']:+.3f}  (95% interval {d['ci95_low']:+.3f} to {d['ci95_high']:+.3f})")


def main():
    """Prepare the partitions and folds, compare the five Stage 1 models, freeze the best and save the results."""
    seed = STAGE1_CONFIG["seed"]
    eligible, train, dev, test, violations = prepare_partitions(seed)
    manifest = save_partitions(eligible, train, dev, test)
    folds_table = save_folds(train)

    results = train_stage1_candidates(train, n_folds=STAGE1_CONFIG["n_folds"], seed=seed)
    rows, dev_abs_err, oof_abs_err = score_candidates(results, train, dev)
    paired = compare_forest_with_simpler_models(train, dev, oof_abs_err, dev_abs_err, seed)
    frozen = freeze_stage1_model(results)

    os.makedirs("reports", exist_ok=True)
    report = build_report(eligible, train, dev, test, violations, manifest, folds_table, frozen, rows, paired)
    with open("reports/stage1_results.json", "w") as f:
        json.dump(report, f, indent=2)
    print_summary(eligible, train, dev, test, violations, rows, frozen, paired)


if __name__ == "__main__":
    main()
