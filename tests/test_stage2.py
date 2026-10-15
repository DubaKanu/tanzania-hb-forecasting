"""
Tests for Stage 2 (src/stage2.py) and the freeze and locked-test guards.

These use small synthetic cohorts. They check that the code does what the protocol
says; they are not results on the real Tanzania cohort and must never be reported as such.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import (  # noqa: E402
    FEATURE_COLUMNS, GESTATION_VISIT2_COLUMN, STAGE1_CONFIG, TARGET_COLUMN, _prep_matrix,
    assign_cv_folds, identical_vector_groups, n_keep, tree_disagreement,
)
from stage2 import (  # noqa: E402
    PROTOCOL, SELECTORS, STAGE2_CONFIG, STAGE2_INPUT_COLUMNS, build_oof_table, coverage_table,
    evaluate_partition, fit_stage1_forest, fit_stage2, fit_two_stage, forecast_with_reliability,
    interval_sentence, make_run_id, paired_bootstrap_selectors, simple_rule_score, stage1_outputs,
    stage2_features,
)
import freeze_protocol  # noqa: E402
import run_locked_test  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _toy_cohort(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 5, n) for c in FEATURE_COLUMNS})
    df["hb_visit1_g_dl"] = rng.uniform(6, 14, n)
    df["gestation_visit1_weeks"] = rng.integers(5, 20, n).astype(float)
    df[GESTATION_VISIT2_COLUMN] = df["gestation_visit1_weeks"] + rng.integers(1, 9, n)
    # The error is larger when the first Hb is far from 11, so Stage 2 has something to learn.
    noise = rng.normal(0, 1, n) * (0.3 + 0.2 * np.abs(df["hb_visit1_g_dl"] - 11))
    df[TARGET_COLUMN] = 0.7 * df["hb_visit1_g_dl"] + 3.3 + noise
    df["_source_row"] = np.arange(1000, 1000 + n)
    df.loc[rng.choice(n, n // 10, replace=False), "respiratory_rate_visit1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Out-of-fold table
# ---------------------------------------------------------------------------
def test_oof_table_has_the_columns_the_brief_asks_for():
    df = _toy_cohort(300)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof = build_oof_table(df, folds)
    assert list(oof.columns) == ["record_id", "fold", "observed_hb", "oof_forecast",
                                 "tree_disagreement", "abs_error"]
    assert np.array_equal(oof["record_id"], df["_source_row"])
    assert np.allclose(oof["abs_error"], np.abs(oof["observed_hb"] - oof["oof_forecast"]))
    assert oof.notna().all().all()


def test_each_oof_forecast_comes_from_a_model_that_never_saw_that_woman():
    df = _toy_cohort(300)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof = build_oof_table(df, folds)
    k = 1
    held_in, held_out = df[folds != k], df[folds == k]
    X_in, imputer = _prep_matrix(held_in, fit=True)
    X_out, _ = _prep_matrix(held_out, imputer=imputer, fit=False)
    forest = RandomForestRegressor(**STAGE1_CONFIG["random_forest"], random_state=STAGE1_CONFIG["seed"])
    forest.fit(X_in, held_in[TARGET_COLUMN])
    assert np.allclose(oof.loc[folds == k, "oof_forecast"], forest.predict(X_out))
    assert np.allclose(oof.loc[folds == k, "tree_disagreement"], tree_disagreement(forest, X_out))


def test_in_sample_residuals_are_too_optimistic_which_is_why_they_are_not_used():
    df = _toy_cohort(300)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof = build_oof_table(df, folds)
    b1 = fit_stage1_forest(df)
    in_sample = np.abs(df[TARGET_COLUMN] - stage1_outputs(b1, df)[0]).mean()
    assert in_sample < oof["abs_error"].mean()


# ---------------------------------------------------------------------------
# Stage 2 model
# ---------------------------------------------------------------------------
def test_stage2_inputs_are_the_twelve_permitted_columns():
    assert STAGE2_INPUT_COLUMNS == FEATURE_COLUMNS + ["stage1_forecast", "tree_disagreement"]
    assert len(STAGE2_INPUT_COLUMNS) == 12
    for c in STAGE2_INPUT_COLUMNS:
        assert not any(bad in c.lower() for bad in ("visit2", "target", "risk", "delivery"))


def test_stage2_learns_from_the_out_of_fold_table_it_is_given():
    df = _toy_cohort(300)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof = build_oof_table(df, folds)
    b1 = fit_stage1_forest(df)
    oof["abs_error"] = 2.0 * np.abs(df["hb_visit1_g_dl"].to_numpy() - 11)  # a clean function of an input
    model = fit_stage2(oof, df, b1["imputer"])
    pred = model.predict(stage2_features(df, b1["imputer"], oof["oof_forecast"], oof["tree_disagreement"]))
    assert np.corrcoef(pred, oof["abs_error"])[0, 1] > 0.9


def test_fit_stage2_refuses_a_table_in_a_different_row_order():
    df = _toy_cohort(200)
    folds = assign_cv_folds(identical_vector_groups(df), 3)
    oof = build_oof_table(df, folds)
    b1 = fit_stage1_forest(df)
    with pytest.raises(AssertionError):
        fit_stage2(oof.sample(frac=1.0, random_state=0).reset_index(drop=True), df, b1["imputer"])


def test_stage2_is_a_separate_fitted_instance_with_the_declared_settings():
    df = _toy_cohort(200)
    b1, model, _, _ = fit_two_stage(df)
    assert model is not b1["model"]
    for key, value in STAGE2_CONFIG["random_forest"].items():
        assert getattr(model, key) == value


def test_forecast_with_reliability_is_stage1_then_stage2():
    df = _toy_cohort(250)
    b1, model, _, _ = fit_two_stage(df)
    rows = df.head(15)
    forecast, disagreement, estimate = forecast_with_reliability(b1, model, rows)
    f2, d2 = stage1_outputs(b1, rows)
    assert np.allclose(forecast, f2) and np.allclose(disagreement, d2)
    assert np.allclose(estimate, model.predict(stage2_features(rows, b1["imputer"], f2, d2)))
    assert (estimate >= 0).all()  # an estimated absolute error cannot be negative


def test_run_id_is_stable_and_changes_when_a_setting_changes():
    a = make_run_id({"x": 1, "cfg": {"depth": 6}})
    assert a == make_run_id({"cfg": {"depth": 6}, "x": 1}) and a.startswith("run-")
    assert a != make_run_id({"x": 1, "cfg": {"depth": 7}})


# ---------------------------------------------------------------------------
# Selectors and comparison
# ---------------------------------------------------------------------------
def test_simple_rule_orders_by_missing_inputs_then_distance_from_the_median():
    df = _toy_cohort(6, seed=1)
    df[FEATURE_COLUMNS] = 1.0
    df["hb_visit1_g_dl"] = [11.0, 14.0, 11.0, 9.0, 11.0, 12.0]
    df.loc[2, "pulse_visit1_bpm"] = np.nan                                  # one missing
    df.loc[4, ["pulse_visit1_bpm", "respiratory_rate_visit1"]] = np.nan     # two missing
    score = simple_rule_score(df, train_hb_median=11.0)
    # most reliable first: 0 missing & on the median, then 0 missing & 1 away, 2 away, 3 away, then 1 missing, then 2 missing
    assert list(np.argsort(score, kind="stable")) == [0, 5, 3, 1, 2, 4]


def _toy_table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    err = rng.uniform(0.05, 2.0, n)
    return pd.DataFrame({"record_id": np.arange(n)[::-1] + 500, "abs_error": err,
                         "signed_error": err * rng.choice([-1, 1], n)})


def test_coverage_table_oracle_is_best_and_random_matches_the_overall_average():
    tab = _toy_table()
    rng = np.random.default_rng(5)
    scores = {s: rng.normal(size=len(tab)) for s in SELECTORS}
    cov = coverage_table(tab, scores, [1.0, 0.8], repeats=100, seed=3)
    at = lambda sel, c: float(cov[(cov.selector == sel) & (cov.coverage == c)]["mae"].iloc[0])  # noqa: E731
    assert at("oracle", 0.8) <= min(at(s, 0.8) for s in SELECTORS) + 1e-12
    assert abs(at("random", 0.8) - tab["abs_error"].mean()) < 0.1
    for sel in SELECTORS + ["oracle", "random"]:
        assert abs(at(sel, 1.0) - tab["abs_error"].mean()) < 1e-12
    kept = cov[(cov.coverage == 0.8) & (cov.selector == "learned")]["n_kept"].iloc[0]
    assert kept == n_keep(len(tab), 0.8)


def test_paired_bootstrap_matches_a_hand_written_version():
    rng0 = np.random.default_rng(3)
    n, n_boot = 30, 40
    tab = _toy_table(n, seed=3)
    scores = {"learned": rng0.normal(size=n), "tree_disagreement": rng0.normal(size=n),
              "simple_rule": np.round(rng0.uniform(0, 3, n))}  # many ties on purpose
    out = paired_bootstrap_selectors(tab, scores, [0.8], np.arange(n), n_boot, seed=7)

    rng = np.random.default_rng(7)
    k = int(np.floor(round(n * 0.8, 9) + 0.5))
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        err, ids = tab["abs_error"].to_numpy()[pick], tab["record_id"].to_numpy()[pick]

        def mae(sc):
            order = sorted(range(n), key=lambda i: (sc[pick][i], ids[i]))
            return float(np.mean([err[i] for i in order[:k]]))

        diffs.append(mae(scores["learned"]) - mae(scores["tree_disagreement"]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    row = out[(out["comparison"] == "learned - tree_disagreement") & (out["coverage"] == 0.8)].iloc[0]
    assert np.isclose(row["ci95_low"], lo) and np.isclose(row["ci95_high"], hi)


def test_partition_evaluation_audit_counts_add_up():
    train, test = _toy_cohort(400, seed=1), _toy_cohort(120, seed=2)
    b1, model, _, _ = fit_two_stage(train)
    res = evaluate_partition(test, b1, model, float(np.median(train["hb_visit1_g_dl"])), n_boot=30, repeats=10)
    n = len(test)
    audit = res["audit"]
    for sel in SELECTORS:
        for grouping, part in audit[audit.selector == sel].groupby("grouping"):
            assert part["n_women"].sum() == n
            assert part["n_kept"].sum() == n_keep(n, PROTOCOL["primary_coverage"])
    assert res["n_kept_primary"] == n_keep(n, 0.8)
    assert set(res["bootstrap"]["coverage"]) == {0.9, 0.8, 0.7}


def test_interval_sentence_never_claims_more_than_the_interval_shows():
    crossing = interval_sentence({"difference": -0.02, "ci95_low": -0.06, "ci95_high": 0.03}, "development")
    assert "neither superiority nor equivalence" in crossing and "does not establish equivalence" in crossing
    below = interval_sentence({"difference": -0.05, "ci95_low": -0.08, "ci95_high": -0.01}, "development")
    assert "lower average error" in below and "neither" not in below
    above = interval_sentence({"difference": 0.05, "ci95_low": 0.01, "ci95_high": 0.08}, "development")
    assert "higher average error" in above


# ---------------------------------------------------------------------------
# The locked test cannot be opened by accident
# ---------------------------------------------------------------------------
def test_development_script_never_opens_the_test_file():
    with open(os.path.join(ROOT, "src", "run_stage2.py"), encoding="utf8") as f:
        assert "test.csv" not in f.read()


def _write_freeze(root, files):
    record = {"frozen_at_utc": "2026-01-01 00:00:00", "run_id": "run-x",
              "versions": run_locked_test.versions(),
              "files": {rel: {"sha256": freeze_protocol.sha256_of(os.path.join(root, rel)),
                              "bytes": 1} for rel in files}}
    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    with open(os.path.join(root, freeze_protocol.FREEZE_PATH), "w") as f:
        json.dump(record, f)


def test_locked_test_needs_a_freeze_and_stops_if_a_frozen_file_changes(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    with pytest.raises(SystemExit, match="not frozen"):
        run_locked_test.ensure_ready(root)
    _write_freeze(root, ["protocol.md"])
    assert run_locked_test.ensure_ready(root)["run_id"] == "run-x"   # intact: allowed
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan, quietly edited")
    with pytest.raises(SystemExit, match="changed since the freeze"):
        run_locked_test.ensure_ready(root)


def test_locked_test_stops_if_the_libraries_differ_from_the_freeze(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    _write_freeze(root, ["protocol.md"])
    run_locked_test.ensure_ready(root)                      # same libraries: allowed
    path = tmp_path / freeze_protocol.FREEZE_PATH
    record = json.loads(path.read_text())
    record["versions"]["scikit-learn"] = "0.0.1"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="libraries differ") as stop:
        run_locked_test.ensure_ready(root)
    assert "scikit-learn" in str(stop.value)
    record["versions"].pop("scikit-learn")                  # a freeze record with no version is refused too
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="libraries differ"):
        run_locked_test.ensure_ready(root)


def test_locked_test_cannot_run_twice(tmp_path):
    root = str(tmp_path)
    with open(tmp_path / "protocol.md", "w") as f:
        f.write("plan")
    _write_freeze(root, ["protocol.md"])
    with open(tmp_path / freeze_protocol.LOCKED_RESULTS_PATH, "w") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="already been run"):
        run_locked_test.ensure_ready(root)


def test_nothing_can_be_frozen_after_the_test_has_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("reports")
    with open(freeze_protocol.LOCKED_RESULTS_PATH, "w") as f:
        f.write("{}")
    with pytest.raises(SystemExit, match="already been run"):
        freeze_protocol.main()
