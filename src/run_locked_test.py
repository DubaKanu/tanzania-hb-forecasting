"""
The locked test, run once. From the repository root, after freeze_protocol.py:

    python src/run_locked_test.py

This is the only script that evaluates the 399 locked-test women. It refuses to run
unless the protocol has been frozen, refuses to run if any frozen file has changed,
and refuses to run a second time. Every result is saved, including any that does not
favour the learned score.

It applies exactly the same analysis code as the development run (evaluation.evaluate_partition).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from evaluation import PROTOCOL, evaluate_partition, interval_sentence, primary_row  # noqa: E402
from forecaster import TwoStageForecaster  # noqa: E402
from freeze_protocol import (  # noqa: E402
    DATA, FREEZE_PATH, LOCKED_RESULTS_PATH, sha256_of, verify_freeze,
)
from metrics import mean_absolute_error, mean_signed_error, paired_bootstrap_difference, rmse  # noqa: E402
from pipeline import FEATURE_COLUMNS, STAGE1_CONFIG, TARGET_COLUMN, identical_vector_groups  # noqa: E402
from provenance import library_versions  # noqa: E402
from reporting import (  # noqa: E402
    md_table, plot_error_vs_coverage, plot_forecast_vs_observed, plot_stage2_check, render_report,
    results_to_json,
)
from stage1 import stage1_predict, train_stage1_candidates  # noqa: E402

REPORTS = "reports"
LIBRARIES = ("numpy", "pandas", "scikit-learn")
SIMPLER_MODELS = ("median", "carry_forward", "two_input_linear", "ridge")


# ---------------------------------------------------------------------------
# The guards: nothing below the call to ensure_ready runs unless every one passes
# ---------------------------------------------------------------------------
def library_mismatches(record: dict) -> list:
    """Libraries whose version differs from the one recorded at the freeze.

    Python itself is not compared: the same results were obtained on Python 3.9 to 3.12.
    """
    frozen = record.get("versions", {})
    now = library_versions()
    return [f"{lib}: frozen with {frozen.get(lib, 'an unknown version')}, running with {now[lib]}"
            for lib in LIBRARIES if frozen.get(lib) != now[lib]]


def ensure_ready(root: str = ".") -> dict:
    """Stop unless the test may be run now. Returns the freeze record."""
    if os.path.exists(os.path.join(root, LOCKED_RESULTS_PATH)):
        sys.exit("The locked test has already been run and its results are saved. It is not run twice.")
    freeze_file = os.path.join(root, FREEZE_PATH)
    if not os.path.exists(freeze_file):
        sys.exit("The protocol is not frozen yet. Run python src/freeze_protocol.py first.")
    with open(freeze_file) as f:
        record = json.load(f)
    problems = verify_freeze(record, root)
    if problems:
        sys.exit("The locked test will not run because the frozen files no longer match:\n  - " +
                 "\n  - ".join(problems) +
                 "\nRestore them, or (if a change was truly needed) delete the freeze file, freeze again "
                 "and write down why.")
    wrong = library_mismatches(record)
    if wrong:
        sys.exit("The locked test will not run because the libraries differ from the ones used at the freeze:\n  - " +
                 "\n  - ".join(wrong) +
                 "\nInstall the versions in requirements.txt and run it again.")
    return record


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_partitions():
    """Training, development and (for the first and only time) the locked-test women."""
    manifest = pd.read_csv(f"{DATA}/split_manifest.csv")
    train = pd.read_csv(f"{DATA}/train.csv")
    dev = pd.read_csv(f"{DATA}/validation.csv")
    test = pd.read_csv(f"{DATA}/test.csv")  # the first and only evaluation use of these women
    assert len(test) == 399, "the locked test partition must have 399 women"
    assert set(test["_source_row"]) == set(manifest.loc[manifest["partition"] == "test", "_source_row"]), \
        "test.csv does not match the split manifest"
    assert test["_source_row"].is_unique
    return train, dev, test


def complete_cases(frame: pd.DataFrame) -> pd.DataFrame:
    """Only the women with all ten inputs recorded."""
    return frame[frame[FEATURE_COLUMNS].notna().all(axis=1)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------
def run_complete_case_analysis(train, dev, test, dev_run: dict):
    """Complete-case sensitivity: refit on complete-case training women, score complete-case test women."""
    train_cc, dev_cc, test_cc = complete_cases(train), complete_cases(dev), complete_cases(test)
    forecaster_cc = TwoStageForecaster.fit(train_cc)
    # The refit must reproduce the development result that was saved before the freeze.
    check = evaluate_partition(dev_cc, forecaster_cc, n_boot=20, repeats=5)
    stored = dev_run["complete_case"]["primary"]["difference"]
    assert abs(primary_row(check["bootstrap"])["difference"] - stored) < 1e-9, \
        "complete-case refit does not reproduce the development result"
    res_cc = evaluate_partition(test_cc, forecaster_cc)
    note = (f"The whole pipeline is repeated on complete-case women only: {len(train_cc)} training and "
            f"{len(test_cc)} locked-test women (of {len(train)} and {len(test)}). It was refitted from the "
            "training women with the same fixed settings.")
    return res_cc, note, len(train_cc), len(test_cc)


def compare_stage1_candidates(forecaster, train, dev, test):
    """Every Stage 1 candidate on the test women, and the forest against each simpler model (paired bootstrap)."""
    candidates = train_stage1_candidates(train, n_folds=STAGE1_CONFIG["n_folds"], seed=STAGE1_CONFIG["seed"])
    same = np.allclose(stage1_predict(candidates["random_forest"], dev), forecaster.stage1.predict(dev))
    assert same, "a fresh Stage 1 fit does not match the frozen forest"
    y = test[TARGET_COLUMN].to_numpy()
    groups = identical_vector_groups(test)
    abs_err, rows = {}, []
    for name, result in candidates.items():
        pred = stage1_predict(result, test)
        abs_err[name] = np.abs(y - pred)
        rows.append({"model": name, "MAE": mean_absolute_error(y, pred), "RMSE": rmse(y, pred),
                     "mean signed error": mean_signed_error(y, pred)})
    table = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    paired = {f"random_forest_minus_{other}": paired_bootstrap_difference(
        abs_err["random_forest"], abs_err[other], groups=groups,
        n_boot=PROTOCOL["bootstrap_replicates"], seed=PROTOCOL["bootstrap_seed"]) for other in SIMPLER_MODELS}
    return table, paired


# ---------------------------------------------------------------------------
# Saving. The results file is written last: it is also the 'already run' marker.
# ---------------------------------------------------------------------------
def save_report(freeze, res, res_cc, cc_note, stage1_table, paired):
    """Write the locked-test report (markdown) with the freeze details at the top."""
    meta = {"run_id": freeze["run_id"], "versions": library_versions(),
            "extra_lines": [f"Protocol frozen at {freeze['frozen_at_utc']} UTC",
                            f"Freeze record SHA-256: {sha256_of(FREEZE_PATH)}"]}
    report = render_report("Locked test results (evaluated once)", "locked test", meta, res, res_cc, cc_note)
    report += ("\n## Stage 1 on the locked-test women\n\n" + md_table(stage1_table) + "\n\n"
               "Forest minus each simpler model, mean absolute error on the same women (negative favours the "
               "forest), paired bootstrap:\n\n" +
               md_table(pd.DataFrame([{"comparison": k.replace("_", " "), "difference (g/dL)": v["difference"],
                                       "95% low": v["ci95_low"], "95% high": v["ci95_high"]}
                                      for k, v in paired.items()])) + "\n")
    with open(f"{REPORTS}/locked_test_results.md", "w") as f:
        f.write(report)


def save_tables_and_figures(res):
    """Save the locked-test table and figures."""
    res["table"].round(6).to_csv(f"{REPORTS}/locked_test_table.csv", index=False)
    plot_error_vs_coverage(res, f"{REPORTS}/fig_error_vs_coverage_test.png",
                           "Error of the kept forecasts against coverage (locked test women)")
    plot_forecast_vs_observed(res, f"{REPORTS}/fig_forecast_vs_observed_test.png",
                              "Stage 1 forecast (locked test women)")
    plot_stage2_check(res, f"{REPORTS}/fig_stage2_check_test.png",
                      "Stage 2 estimate against real error (locked test women)")


def save_results_json(freeze, res, res_cc, n_train_cc, n_test_cc, stage1_table, paired):
    """Save every number of the locked test as JSON."""
    out = {
        "label": "locked test",
        "run_id": freeze["run_id"],
        "frozen_at_utc": freeze["frozen_at_utc"],
        "freeze_record_sha256": sha256_of(FREEZE_PATH),
        "protocol": PROTOCOL,
        "versions": library_versions(),
        "main": results_to_json(res),
        "complete_case": {"n_train": n_train_cc, "n_test": n_test_cc, **results_to_json(res_cc)},
        "stage1_on_test": json.loads(stage1_table.to_json(orient="records")),
        "stage1_paired_mae_difference": paired,
    }
    with open(LOCKED_RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)


def print_summary(freeze, test, res):
    """Print the locked-test result, with the reminder that it must not be run again."""
    cov = res["coverage"]
    print("LOCKED TEST RESULT (run once; do not run again)")
    print(f"run ID {freeze['run_id']}   test women {len(test)}   Stage 1 MAE {res['stage1']['mae']:.3f}")
    print(f"{'selector':22s}" + "".join(f"{int(c * 100):>9d}%" for c in PROTOCOL["coverage_levels"]))
    for s in ("learned", "tree_disagreement", "simple_rule", "random", "oracle"):
        row = cov[cov["selector"] == s].set_index("coverage")["mae"]
        print(f"{s:22s}" + "".join(f"{row[c]:10.3f}" for c in PROTOCOL["coverage_levels"]))
    print(interval_sentence(primary_row(res["bootstrap"]), "locked test"))
    print(f"Saved {LOCKED_RESULTS_PATH}, locked_test_results.md, locked_test_table.csv and three figures.")


def main():
    """Check the freeze, evaluate the locked-test women once and save the results."""
    freeze = ensure_ready()

    with open(f"{REPORTS}/stage2_dev_results.json") as f:
        dev_run = json.load(f)
    train, dev, test = load_partitions()
    forecaster = TwoStageForecaster.from_files("models/stage1_model.joblib", "models/stage2_model.joblib")
    assert forecaster.run_id == freeze["run_id"]

    # Main analysis: exactly the development code, on the test women.
    res = evaluate_partition(test, forecaster)
    res_cc, cc_note, n_train_cc, n_test_cc = run_complete_case_analysis(train, dev, test, dev_run)
    stage1_table, paired = compare_stage1_candidates(forecaster, train, dev, test)

    save_report(freeze, res, res_cc, cc_note, stage1_table, paired)
    save_tables_and_figures(res)
    save_results_json(freeze, res, res_cc, n_train_cc, n_test_cc, stage1_table, paired)
    print_summary(freeze, test, res)


if __name__ == "__main__":
    main()
