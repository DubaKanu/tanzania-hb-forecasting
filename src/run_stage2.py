"""
Stage 2 on the training and development women. Run from the repository root,
after run_stage1.py:

    python src/run_stage2.py

What it does
  1. Rebuilds the out-of-fold Stage 1 forecasts for the 1,884 training women with the
     fixed fold rule and saves them with a run ID (reports/oof_stage1.csv).
  2. Trains the Stage 2 forest on those out-of-fold errors only.
  3. Compares the learned score, tree disagreement, a simple rule and random selection on
     the 402 development women: every coverage level, the paired bootstrap, the audit of
     who is kept, and a complete-case sensitivity check.
  4. Saves the Stage 2 model (models/stage2_model.joblib), tables and figures.

This script reads train.csv and validation.csv only. It never opens the locked test
file. The locked test is run once, by run_locked_test.py, after the protocol is frozen.
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
from pipeline import (  # noqa: E402
    FEATURE_COLUMNS, STAGE1_CONFIG, assign_cv_folds, identical_vector_groups,
)
from stage2 import (  # noqa: E402
    PROTOCOL, STAGE2_CONFIG, STAGE2_INPUT_COLUMNS, build_oof_table, evaluate_partition,
    fit_stage1_forest, fit_stage2, fit_two_stage, interval_sentence, make_run_id,
    plot_error_vs_coverage, plot_forecast_vs_observed, plot_stage2_check, primary_row,
    render_report, results_to_json, stage1_outputs,
)

DATA = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"
REPORTS = "reports"
MODELS = "models"


def md5_of_frame(df):
    return hashlib.md5(df.to_csv(index=False).encode()).hexdigest()


def versions():
    return {"python": platform.python_version(), "pandas": pd.__version__,
            "numpy": np.__version__, "scikit-learn": sklearn.__version__}


def main():
    with open(f"{REPORTS}/stage1_results.json") as f:
        s1 = json.load(f)
    assert s1["frozen_stage1_model"] == "random_forest", \
        "Stage 2 needs the forest: tree disagreement exists only for a forest"

    # --- Only the training and development files are opened here. -------------------
    manifest = pd.read_csv(f"{DATA}/split_manifest.csv")
    assert md5_of_frame(manifest) == s1["manifest_md5"], "split manifest changed since run_stage1.py"
    train = pd.read_csv(f"{DATA}/train.csv")
    dev = pd.read_csv(f"{DATA}/validation.csv")
    for name, frame in (("train", train), ("dev", dev)):
        assert frame["_source_row"].is_unique, f"record IDs repeat in {name}"
        in_manifest = set(manifest.loc[manifest["partition"] == name, "_source_row"])
        assert set(frame["_source_row"]) == in_manifest, f"{name} does not match the split manifest"

    # --- Folds: the same fixed rule, checked against the saved file. -----------------
    folds = assign_cv_folds(identical_vector_groups(train), STAGE1_CONFIG["n_folds"])
    saved = pd.read_csv(f"{DATA}/cv_folds.csv")
    assert np.array_equal(saved["_source_row"], train["_source_row"]) and np.array_equal(saved["cv_fold"], folds), \
        "cv_folds.csv does not match the fold rule"
    assert md5_of_frame(saved) == s1["cv_folds_md5"], "cv_folds.csv changed since run_stage1.py"

    # --- The frozen Stage 1 forest, checked against a fresh fit. ---------------------
    bundle1 = joblib.load(f"{MODELS}/stage1_model.joblib")
    assert bundle1["name"] == "random_forest"
    fresh = fit_stage1_forest(train)
    frozen_out, fresh_out = stage1_outputs(bundle1, dev), stage1_outputs(fresh, dev)
    assert np.allclose(frozen_out[0], fresh_out[0]) and np.allclose(frozen_out[1], fresh_out[1]), \
        "the saved Stage 1 model differs from a fresh fit; rerun run_stage1.py"

    # --- Out-of-fold artefact. --------------------------------------------------------
    oof = build_oof_table(train, folds)
    oof_mae = float(oof["abs_error"].mean())
    assert round(oof_mae, 3) == s1["metrics_g_per_dl"]["random_forest"]["train_cv_mae"], \
        "out-of-fold MAE does not match the Stage 1 cross-validation MAE"
    run_id = make_run_id({
        "raw_md5": s1["raw_md5"], "manifest_md5": s1["manifest_md5"], "cv_folds_md5": s1["cv_folds_md5"],
        "stage1": s1["config"], "stage2": STAGE2_CONFIG,
    })
    os.makedirs(REPORTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    oof.insert(0, "run_id", run_id)
    oof.to_csv(f"{REPORTS}/oof_stage1.csv", index=False, float_format="%.6f")
    with open(f"{REPORTS}/oof_run.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "rows": int(len(oof)),
            "columns": list(oof.columns),
            "folds": {"rule": s1["cv_fold_rule"], "rows_per_fold": {int(k): int(v) for k, v in
                                                                  oof["fold"].value_counts().sort_index().items()}},
            "pooled_oof_mae_g_per_dl": round(oof_mae, 3),
            "stage1_config": s1["config"],
            "raw_md5": s1["raw_md5"], "manifest_md5": s1["manifest_md5"], "cv_folds_md5": s1["cv_folds_md5"],
            "versions": versions(),
        }, f, indent=2)

    # --- Stage 2 model. ----------------------------------------------------------------
    stage2_model = fit_stage2(oof, train, bundle1["imputer"])
    train_hb_median = float(np.median(train["hb_visit1_g_dl"]))

    # --- Development comparison. -------------------------------------------------------
    res = evaluate_partition(dev, bundle1, stage2_model, train_hb_median)

    # Demo cutoff for the API (one woman cannot be ranked). It is fitted on the development
    # women and is NOT the exact-coverage rule used in the evaluation above.
    cutoff = float(np.percentile(res["table"]["stage2_score"], 80))
    joblib.dump({
        "model": stage2_model, "run_id": run_id, "config": STAGE2_CONFIG,
        "input_columns": STAGE2_INPUT_COLUMNS, "train_hb_median": train_hb_median,
        "demo_cutoff": cutoff,
        "demo_cutoff_rule": ("80th percentile of the Stage 2 scores of the development women; "
                             "fitted on development data, so different from exact-coverage evaluation"),
        "versions": versions(),
    }, f"{MODELS}/stage2_model.joblib")

    # --- Complete-case sensitivity: repeat everything on women with all ten inputs. ----
    train_cc = train[train[FEATURE_COLUMNS].notna().all(axis=1)].reset_index(drop=True)
    dev_cc = dev[dev[FEATURE_COLUMNS].notna().all(axis=1)].reset_index(drop=True)
    b1_cc, m2_cc, oof_cc, _ = fit_two_stage(train_cc)
    res_cc = evaluate_partition(dev_cc, b1_cc, m2_cc, float(np.median(train_cc["hb_visit1_g_dl"])))
    cc_note = (f"The whole pipeline is repeated on the complete-case women only: {len(train_cc)} training and "
               f"{len(dev_cc)} development women (of {len(train)} and {len(dev)}). Folds, the out-of-fold table, "
               "Stage 1, Stage 2 and the comparison are all refitted on these women, with the same fixed settings.")

    # --- Save results. -----------------------------------------------------------------
    meta = {"run_id": run_id, "versions": versions(),
            "extra_lines": [f"Stage 1 pooled out-of-fold MAE on the training women: {oof_mae:.3f}",
                            f"Training median first Hb used by the simple rule: {train_hb_median:.2f} g/dL",
                            f"Demo cutoff for the API (development-fitted, not the evaluation rule): {cutoff:.3f} g/dL"]}
    out = {
        "label": "development",
        "run_id": run_id,
        "protocol": PROTOCOL,
        "stage2_config": STAGE2_CONFIG,
        "versions": versions(),
        "stage1_pooled_oof_mae": round(oof_mae, 3),
        "demo_cutoff": cutoff,
        "main": results_to_json(res),
        "complete_case": {"n_train": int(len(train_cc)), "n_dev": int(len(dev_cc)),
                          **results_to_json(res_cc)},
    }
    with open(f"{REPORTS}/stage2_dev_results.json", "w") as f:
        json.dump(out, f, indent=2)
    res["table"].round(6).to_csv(f"{REPORTS}/stage2_dev_table.csv", index=False)
    with open(f"{REPORTS}/stage2_dev_results.md", "w") as f:
        f.write(render_report("Stage 2 results on the development women", "development", meta, res, res_cc, cc_note))

    plot_error_vs_coverage(res, f"{REPORTS}/fig_error_vs_coverage_dev.png",
                           "Error of the kept forecasts against coverage (development women)")
    plot_forecast_vs_observed(res, f"{REPORTS}/fig_forecast_vs_observed_dev.png", "Stage 1 forecast (development women)")
    plot_stage2_check(res, f"{REPORTS}/fig_stage2_check_dev.png", "Stage 2 estimate against real error (development women)")

    # --- Console summary. --------------------------------------------------------------
    cov = res["coverage"]
    print(f"run ID {run_id}   out-of-fold MAE (training) {oof_mae:.3f}   development MAE {res['stage1']['mae']:.3f}")
    print(f"{'selector':22s}" + "".join(f"{int(c * 100):>9d}%" for c in PROTOCOL["coverage_levels"]))
    for s in ("learned", "tree_disagreement", "simple_rule", "random", "oracle"):
        row = cov[cov["selector"] == s].set_index("coverage")["mae"]
        print(f"{s:22s}" + "".join(f"{row[c]:10.3f}" for c in PROTOCOL["coverage_levels"]))
    print(interval_sentence(primary_row(res["bootstrap"]), "development"))
    print("complete-case:", interval_sentence(primary_row(res_cc["bootstrap"]), "development complete-case"))
    print(f"wrote {REPORTS}/oof_stage1.csv, stage2_dev_results.md/.json, figures, {MODELS}/stage2_model.joblib")


if __name__ == "__main__":
    main()
