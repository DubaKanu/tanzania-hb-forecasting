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
    STAGE1_CONFIG, TARGET_COLUMN, apply_eligibility, fixed_split, partition_manifest,
    split_group_violations, train_stage1_candidates, stage1_predict,
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

    results = train_stage1_candidates(train, n_folds=STAGE1_CONFIG["n_folds"], seed=seed)
    rows = {}
    for name, r in results.items():
        pred = stage1_predict(r, dev)
        rows[name] = {
            "train_cv_mae": round(r.cv_mae, 3),
            "dev_mae": round(mean_absolute_error(dev[TARGET_COLUMN], pred), 3),
            "dev_rmse": round(rmse(dev[TARGET_COLUMN], pred), 3),
            "dev_mean_signed_error": round(mean_signed_error(dev[TARGET_COLUMN], pred), 3),
        }

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
        "config": STAGE1_CONFIG,
        "frozen_stage1_model": frozen,
        "metrics_g_per_dl": rows,
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


if __name__ == "__main__":
    main()
