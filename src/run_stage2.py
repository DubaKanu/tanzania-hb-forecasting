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
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import PROTOCOL, evaluate_partition, interval_sentence, primary_row  # noqa: E402
from forecaster import TwoStageForecaster  # noqa: E402
from pipeline import FEATURE_COLUMNS, STAGE1_CONFIG, assign_cv_folds, identical_vector_groups  # noqa: E402
from provenance import library_versions, md5_of_frame  # noqa: E402
from reporting import (  # noqa: E402
    plot_error_vs_coverage, plot_forecast_vs_observed, plot_stage2_check, render_report, results_to_json,
)
from stage1 import RandomForestModel  # noqa: E402
from stage2 import STAGE2_CONFIG, Stage2ErrorEstimator, build_oof_table, make_run_id  # noqa: E402

DATA = "data/Josephine_African_Data_Review/Tanzania_Followup_Haemoglobin"
REPORTS = "reports"
MODELS = "models"


# ---------------------------------------------------------------------------
# Inputs, each checked against what run_stage1.py recorded
# ---------------------------------------------------------------------------
def load_stage1_results() -> dict:
    """Read the saved Stage 1 results. Stop unless the frozen model is the forest, which supplies the disagreement."""
    with open(f"{REPORTS}/stage1_results.json") as f:
        s1 = json.load(f)
    assert s1["frozen_stage1_model"] == "random_forest", \
        "Stage 2 needs the forest: tree disagreement exists only for a forest"
    return s1


def load_training_and_development(s1: dict):
    """Only the training and development files are opened here."""
    manifest = pd.read_csv(f"{DATA}/split_manifest.csv")
    assert md5_of_frame(manifest) == s1["manifest_md5"], "split manifest changed since run_stage1.py"
    train = pd.read_csv(f"{DATA}/train.csv")
    dev = pd.read_csv(f"{DATA}/validation.csv")
    for name, frame in (("train", train), ("dev", dev)):
        assert frame["_source_row"].is_unique, f"record IDs repeat in {name}"
        in_manifest = set(manifest.loc[manifest["partition"] == name, "_source_row"])
        assert set(frame["_source_row"]) == in_manifest, f"{name} does not match the split manifest"
    return train, dev


def load_folds(train, s1: dict) -> np.ndarray:
    """The fixed fold rule, checked against the saved file."""
    folds = assign_cv_folds(identical_vector_groups(train), STAGE1_CONFIG["n_folds"])
    saved = pd.read_csv(f"{DATA}/cv_folds.csv")
    assert np.array_equal(saved["_source_row"], train["_source_row"]) and np.array_equal(saved["cv_fold"], folds), \
        "cv_folds.csv does not match the fold rule"
    assert md5_of_frame(saved) == s1["cv_folds_md5"], "cv_folds.csv changed since run_stage1.py"
    return folds


def load_frozen_stage1(train, dev) -> RandomForestModel:
    """The frozen Stage 1 forest, checked against a fresh fit."""
    bundle = joblib.load(f"{MODELS}/stage1_model.joblib")
    assert bundle["name"] == "random_forest"
    stage1 = RandomForestModel.from_fitted(bundle["model"], bundle["imputer"])
    fresh = RandomForestModel().fit(train)
    frozen_out, fresh_out = stage1.predict_with_disagreement(dev), fresh.predict_with_disagreement(dev)
    assert np.allclose(frozen_out[0], fresh_out[0]) and np.allclose(frozen_out[1], fresh_out[1]), \
        "the saved Stage 1 model differs from a fresh fit; rerun run_stage1.py"
    return stage1


# ---------------------------------------------------------------------------
# Out-of-fold artefact
# ---------------------------------------------------------------------------
def build_and_save_oof(train, folds, s1: dict):
    """The out-of-fold table with its run ID, saved with a record of how it was made."""
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
            "versions": library_versions(),
        }, f, indent=2)
    return oof, oof_mae, run_id


# ---------------------------------------------------------------------------
# Development comparison
# ---------------------------------------------------------------------------
def complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the women with all ten inputs recorded."""
    return frame[frame[FEATURE_COLUMNS].notna().all(axis=1)].reset_index(drop=True)


def run_complete_case_check(train, dev):
    """Complete-case sensitivity: repeat everything on women with all ten inputs."""
    train_cc, dev_cc = complete_cases(train), complete_cases(dev)
    res_cc = evaluate_partition(dev_cc, TwoStageForecaster.fit(train_cc))
    note = (f"The whole pipeline is repeated on the complete-case women only: {len(train_cc)} training and "
            f"{len(dev_cc)} development women (of {len(train)} and {len(dev)}). Folds, the out-of-fold table, "
            "Stage 1, Stage 2 and the comparison are all refitted on these women, with the same fixed settings.")
    return res_cc, note, len(train_cc), len(dev_cc)


def results_payload(run_id, oof_mae, cutoff, res, res_cc, n_train_cc, n_dev_cc) -> dict:
    """Everything the development run reports, as a dictionary for the JSON file."""
    return {
        "label": "development",
        "run_id": run_id,
        "protocol": PROTOCOL,
        "stage2_config": STAGE2_CONFIG,
        "versions": library_versions(),
        "stage1_pooled_oof_mae": round(oof_mae, 3),
        "demo_cutoff": cutoff,
        "main": results_to_json(res),
        "complete_case": {"n_train": n_train_cc, "n_dev": n_dev_cc, **results_to_json(res_cc)},
    }


def save_development_outputs(run_id, oof_mae, train_hb_median, cutoff, res, res_cc, cc_note, payload):
    """Save the development results: JSON, table, markdown report and figures."""
    with open(f"{REPORTS}/stage2_dev_results.json", "w") as f:
        json.dump(payload, f, indent=2)
    res["table"].round(6).to_csv(f"{REPORTS}/stage2_dev_table.csv", index=False)
    meta = {"run_id": run_id, "versions": library_versions(),
            "extra_lines": [f"Stage 1 pooled out-of-fold MAE on the training women: {oof_mae:.3f}",
                            f"Training median first Hb used by the simple rule: {train_hb_median:.2f} g/dL",
                            "Demo cutoff for the API (development-fitted, not the evaluation rule): "
                            f"{cutoff:.3f} g/dL"]}
    with open(f"{REPORTS}/stage2_dev_results.md", "w") as f:
        f.write(render_report("Stage 2 results on the development women", "development", meta, res, res_cc,
                              cc_note))
    plot_error_vs_coverage(res, f"{REPORTS}/fig_error_vs_coverage_dev.png",
                           "Error of the kept forecasts against coverage (development women)")
    plot_forecast_vs_observed(res, f"{REPORTS}/fig_forecast_vs_observed_dev.png",
                              "Stage 1 forecast (development women)")
    plot_stage2_check(res, f"{REPORTS}/fig_stage2_check_dev.png",
                      "Stage 2 estimate against real error (development women)")


def print_summary(run_id, oof_mae, res, res_cc):
    """Print the development comparison."""
    cov = res["coverage"]
    print(f"run ID {run_id}   out-of-fold MAE (training) {oof_mae:.3f}   development MAE {res['stage1']['mae']:.3f}")
    print(f"{'selector':22s}" + "".join(f"{int(c * 100):>9d}%" for c in PROTOCOL["coverage_levels"]))
    for s in ("learned", "tree_disagreement", "simple_rule", "random", "oracle"):
        row = cov[cov["selector"] == s].set_index("coverage")["mae"]
        print(f"{s:22s}" + "".join(f"{row[c]:10.3f}" for c in PROTOCOL["coverage_levels"]))
    print(interval_sentence(primary_row(res["bootstrap"]), "development"))
    print("complete-case:", interval_sentence(primary_row(res_cc["bootstrap"]), "development complete-case"))
    print(f"wrote {REPORTS}/oof_stage1.csv, stage2_dev_results.md/.json, figures, {MODELS}/stage2_model.joblib")


def main():
    """Build the out-of-fold table, fit Stage 2, evaluate the development women and save everything."""
    s1 = load_stage1_results()
    train, dev = load_training_and_development(s1)
    folds = load_folds(train, s1)
    stage1 = load_frozen_stage1(train, dev)

    oof, oof_mae, run_id = build_and_save_oof(train, folds, s1)
    stage2 = Stage2ErrorEstimator().fit(oof, train, stage1.imputer_)
    train_hb_median = float(np.median(train["hb_visit1_g_dl"]))
    forecaster = TwoStageForecaster(stage1, stage2, train_hb_median, run_id=run_id)

    res = evaluate_partition(dev, forecaster)

    # Demo cutoff for the API (one woman cannot be ranked). It is fitted on the development
    # women and is NOT the exact-coverage rule used in the evaluation above.
    cutoff = forecaster.calibrate_demo_cutoff(res["table"]["stage2_score"])
    joblib.dump(forecaster.to_bundles(library_versions())[1], f"{MODELS}/stage2_model.joblib")

    res_cc, cc_note, n_train_cc, n_dev_cc = run_complete_case_check(train, dev)
    payload = results_payload(run_id, oof_mae, cutoff, res, res_cc, n_train_cc, n_dev_cc)
    save_development_outputs(run_id, oof_mae, train_hb_median, cutoff, res, res_cc, cc_note, payload)
    print_summary(run_id, oof_mae, res, res_cc)


if __name__ == "__main__":
    main()
