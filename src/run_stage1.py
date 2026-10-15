"""
Versioned rerun of preparation and Stage 1 baselines.

Raw release -> eligible cohort -> frozen partitions -> Stage 1 comparison ->
frozen Stage 1 model. Run from the repository root:

    python src/run_stage1.py

The reserved test partition is written to disk but is never evaluated here.
"""
import hashlib
import json
import os
import platform
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, os.path.dirname(__file__))
from data_adapter import load_and_map  # noqa: E402
from pipeline import (  # noqa: E402
    STAGE1_CONFIG, TARGET_COLUMN, CV_FOLD_RULE, apply_eligibility, fixed_split, partition_manifest,
    split_group_violations, train_stage1_candidates, stage1_predict, assign_cv_folds,
    identical_vector_groups, paired_bootstrap_difference,
    mean_absolute_error, rmse, mean_signed_error,
)

RAW = "data/raw/maternal_dataset_csv.csv"
RAW_MD5 = "0253d2d385dd80d6c9bb82df6b85fcc8"  # pinned Zenodo release, Version 1
OUT = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert md5_of(RAW) == RAW_MD5, "Raw file does not match the pinned release checksum"
    seed = STAGE1_CONFIG["seed"]

    eligible = apply_eligibility(load_and_map(RAW))
    train, dev, test = fixed_split(eligible, seed=seed)
    violations = split_group_violations(train, dev, test)
    assert violations == 0, f"{violations} identical-vector groups straddle partitions"

    os.makedirs(OUT, exist_ok=True)
    eligible.to_csv(f"{OUT}/model_ready.csv", index=False)
    train.to_csv(f"{OUT}/train.csv", index=False)
    dev.to_csv(f"{OUT}/validation.csv", index=False)
    test.to_csv(f"{OUT}/test.csv", index=False)  # never evaluated until the frozen final step
    manifest = partition_manifest(train, dev, test)
    manifest.to_csv(f"{OUT}/split_manifest.csv", index=False)

    # Cross-validation folds: fixed rule, saved so every later step reads the same folds.
    folds = assign_cv_folds(identical_vector_groups(train), STAGE1_CONFIG["n_folds"])
    folds_table = pd.DataFrame({"_source_row": train["_source_row"], "cv_fold": folds})
    folds_table.to_csv(f"{OUT}/cv_folds.csv", index=False)

    results = train_stage1_candidates(train, n_folds=STAGE1_CONFIG["n_folds"], seed=seed)
    rows = {}
    dev_abs_err, oof_abs_err = {}, {}
    y_dev = dev[TARGET_COLUMN].to_numpy()
    y_train = train[TARGET_COLUMN].to_numpy()
    for name, r in results.items():
        pred = stage1_predict(r, dev)
        rows[name] = {
            "train_cv_mae": round(r.cv_mae, 3),
            "dev_mae": round(mean_absolute_error(dev[TARGET_COLUMN], pred), 3),
            "dev_rmse": round(rmse(dev[TARGET_COLUMN], pred), 3),
            "dev_mean_signed_error": round(mean_signed_error(dev[TARGET_COLUMN], pred), 3),
        }
        dev_abs_err[name] = np.abs(y_dev - pred)
        oof_abs_err[name] = np.abs(y_train - r.oof_pred)

    # Does the forest beat each simpler model on the same women? Paired bootstrap of the
    # difference in mean absolute error (forest minus the other model; negative favours the forest).
    paired = {}
    for label, errs, groups in (("training_out_of_fold", oof_abs_err, identical_vector_groups(train)),
                                ("development", dev_abs_err, identical_vector_groups(dev))):
        paired[label] = {}
        for other in ("median", "carry_forward", "two_input_linear", "ridge"):
            d = paired_bootstrap_difference(errs["random_forest"], errs[other], groups=groups,
                                            n_boot=5000, seed=seed)
            paired[label][f"random_forest_minus_{other}"] = {k: round(v, 3) for k, v in d.items()}

    # Frozen Stage 1 model = lowest training cross-validation MAE (not development MAE).
    frozen = min(results, key=lambda n: results[n].cv_mae)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": results[frozen].model, "imputer": results[frozen].imputer, "name": frozen},
                "models/stage1_model.joblib")

    os.makedirs("reports", exist_ok=True)
    report = {
        "raw_file": "maternal_dataset_csv.csv (Zenodo 10.5281/zenodo.15309733, Version 1)",
        "raw_md5": RAW_MD5,
        "eligible_rows": len(eligible),
        "partitions": {"train": len(train), "dev": len(dev), "test": len(test)},
        "identical_vector_groups_split_across_partitions": violations,
        "manifest_md5": hashlib.md5(manifest.to_csv(index=False).encode()).hexdigest(),
        "cv_folds_md5": hashlib.md5(folds_table.to_csv(index=False).encode()).hexdigest(),
        "cv_fold_rule": CV_FOLD_RULE,
        "config": STAGE1_CONFIG,
        "frozen_stage1_model": frozen,
        "metrics_g_per_dl": rows,
        "paired_comparisons_mae_difference_g_per_dl": {
            "meaning": "random_forest minus the other model, mean absolute error on the same women; "
                       "negative favours the forest. Paired bootstrap, 5000 replicates, seed 42, 95% interval.",
            **paired,
        },
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    with open("reports/stage1_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"eligible: {len(eligible)}  partitions: {len(train)}/{len(dev)}/{len(test)}  group violations: {violations}")
    print(f"{'model':18s} {'train CV MAE':>12s} {'dev MAE':>9s} {'dev RMSE':>9s}")
    for name, m in rows.items():
        print(f"{name:18s} {m['train_cv_mae']:12.3f} {m['dev_mae']:9.3f} {m['dev_rmse']:9.3f}")
    print(f"frozen Stage 1 model (lowest training CV MAE): {frozen}")
    print("forest minus two-input line (g/dL, negative favours the forest):")
    for label in ("training_out_of_fold", "development"):
        d = paired[label]["random_forest_minus_two_input_linear"]
        print(f"  {label:22s} {d['difference']:+.3f}  (95% interval {d['ci95_low']:+.3f} to {d['ci95_high']:+.3f})")


if __name__ == "__main__":
    main()
