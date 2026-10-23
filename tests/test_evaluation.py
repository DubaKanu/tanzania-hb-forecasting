"""
Tests for src/evaluation.py: the comparison of selectors, the paired bootstrap, the audit and the
plain-language interval sentence. Synthetic data only; none of this is a result on the real cohort.
"""
import numpy as np
import pandas as pd

from evaluation import (
    PROTOCOL, audit_bands, coverage_table, evaluate_partition, interval_sentence,
    paired_bootstrap_selectors, primary_row, retention_audit, stage2_calibration,
)
from forecaster import TwoStageForecaster
from helpers import FixedScoreSelector, toy_cohort, toy_table
from metrics import n_keep
from selection import SELECTOR_NAMES


def _selectors(n, seed=5):
    rng = np.random.default_rng(seed)
    return [FixedScoreSelector(name, rng.normal(size=n)) for name in SELECTOR_NAMES]


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------
def test_coverage_table_oracle_is_best_and_random_matches_the_overall_average():
    tab = toy_table()
    selectors = _selectors(len(tab))
    cov = coverage_table(tab, selectors, [1.0, 0.8], repeats=100, seed=3)
    at = lambda sel, c: float(cov[(cov.selector == sel) & (cov.coverage == c)]["mae"].iloc[0])  # noqa: E731
    assert at("oracle", 0.8) <= min(at(s, 0.8) for s in SELECTOR_NAMES) + 1e-12
    assert abs(at("random", 0.8) - tab["abs_error"].mean()) < 0.1
    for sel in SELECTOR_NAMES + ["oracle", "random"]:
        assert abs(at(sel, 1.0) - tab["abs_error"].mean()) < 1e-12
    kept = cov[(cov.coverage == 0.8) & (cov.selector == "learned")]["n_kept"].iloc[0]
    assert kept == n_keep(len(tab), 0.8)


def test_a_new_selector_is_compared_without_changing_the_evaluation_code():
    tab = toy_table(50)
    extra = FixedScoreSelector("my_new_idea", np.random.default_rng(9).normal(size=50))
    cov = coverage_table(tab, [*_selectors(50), extra], [0.8], repeats=10, seed=1)
    assert "my_new_idea" in set(cov["selector"])
    row = cov[(cov.selector == "my_new_idea")].iloc[0]
    assert row["n_kept"] == n_keep(50, 0.8)


def test_achieved_coverage_is_the_exact_fraction_asked_for():
    tab = toy_table(399)
    cov = coverage_table(tab, _selectors(399), [0.8], repeats=5, seed=1)
    kept = cov[(cov.selector == "learned") & (cov.coverage == 0.8)].iloc[0]
    assert kept["n_kept"] == 319            # the number written in the frozen protocol
    assert abs(kept["achieved_coverage"] - 319 / 399) < 1e-12


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------
def test_paired_bootstrap_matches_a_hand_written_version():
    rng0 = np.random.default_rng(3)
    n, n_boot = 30, 40
    tab = toy_table(n, seed=3)
    scores = {"learned": rng0.normal(size=n), "tree_disagreement": rng0.normal(size=n),
              "simple_rule": np.round(rng0.uniform(0, 3, n))}  # many ties on purpose
    selectors = [FixedScoreSelector(name, values) for name, values in scores.items()]
    out = paired_bootstrap_selectors(tab, selectors, [0.8], np.arange(n), n_boot, seed=7)

    k = int(np.floor(round(n * 0.8, 9) + 0.5))

    def retained_mae(score, pick):
        """Plain Python: sort by (score, record ID), keep the first k, average their errors."""
        err, ids = tab["abs_error"].to_numpy()[pick], tab["record_id"].to_numpy()[pick]
        order = sorted(range(n), key=lambda i: (score[pick][i], ids[i]))
        return float(np.mean([err[i] for i in order[:k]]))

    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        diffs.append(retained_mae(scores["learned"], pick) - retained_mae(scores["tree_disagreement"], pick))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    row = out[(out["comparison"] == "learned - tree_disagreement") & (out["coverage"] == 0.8)].iloc[0]
    assert np.isclose(row["ci95_low"], lo) and np.isclose(row["ci95_high"], hi)


def test_paired_bootstrap_is_repeatable_and_has_one_row_per_pair_and_coverage():
    n = 40
    tab = toy_table(n, seed=2)
    selectors = _selectors(n, seed=4)
    first = paired_bootstrap_selectors(tab, selectors, [0.9, 0.8], np.arange(n), 20, seed=1)
    again = paired_bootstrap_selectors(tab, selectors, [0.9, 0.8], np.arange(n), 20, seed=1)
    pd.testing.assert_frame_equal(first, again)
    # three selector pairs plus three 'against keeping everything', at two coverages
    assert len(first) == (3 + 3) * 2
    assert (first["ci95_low"] <= first["ci95_high"]).all()


def test_identical_records_are_resampled_as_whole_groups():
    # If every woman is in one group, every replicate is the whole cohort again, so there is no
    # spread at all and each interval collapses onto the point estimate.
    n = 20
    tab = toy_table(n, seed=6)
    out = paired_bootstrap_selectors(tab, _selectors(n, seed=8), [0.8], np.zeros(n, dtype=int), 25, seed=2)
    assert (out["bootstrap_se"] == 0).all()
    assert np.allclose(out["ci95_low"], out["difference"]) and np.allclose(out["ci95_high"], out["difference"])
    # With every woman in her own group there is real spread.
    spread = paired_bootstrap_selectors(tab, _selectors(n, seed=8), [0.8], np.arange(n), 25, seed=2)
    assert (spread["bootstrap_se"] > 0).all()


# ---------------------------------------------------------------------------
# Audit and calibration
# ---------------------------------------------------------------------------
def test_audit_bands_use_the_prespecified_cut_points():
    tab = pd.DataFrame({"first_hb": [9.9, 10.0, 10.9, 11.0], "n_missing_inputs": [0, 1, 0, 2],
                        "gap_weeks": [2, 3, 8, 9], "observed_hb": [9.0, 10.0, 11.0, 12.0]})
    bands = audit_bands(tab)
    assert list(bands["first_hb"][0]) == ["<10", "10 to <11", "10 to <11", ">=11"]
    assert list(bands["inputs"][0]) == ["complete", "incomplete", "complete", "incomplete"]
    assert list(bands["followup_gap"][0]) == ["<=2 wk", "3-4 wk", "5-8 wk", ">8 wk"]


def test_retention_audit_counts_add_up_for_every_selector():
    train, test = toy_cohort(400, seed=1), toy_cohort(120, seed=2)
    result = evaluate_partition(test, TwoStageForecaster.fit(train), n_boot=10, repeats=5)
    n = len(test)
    audit = result["audit"]
    for sel in SELECTOR_NAMES:
        for _, part in audit[audit.selector == sel].groupby("grouping"):
            assert part["n_women"].sum() == n
            assert part["n_kept"].sum() == n_keep(n, PROTOCOL["primary_coverage"])
    direct = retention_audit(result["table"], result["selectors"], 0.8)
    pd.testing.assert_frame_equal(audit, direct)


def test_calibration_has_five_fifths_that_cover_every_woman():
    tab = pd.DataFrame({"stage2_score": np.linspace(0.1, 1.0, 50), "abs_error": np.linspace(0.2, 1.1, 50)})
    cal = stage2_calibration(tab)
    assert list(cal["fifth_of_stage2_score"]) == [1, 2, 3, 4, 5] and cal["n_women"].sum() == 50
    assert cal["mean_real_abs_error"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# One whole partition
# ---------------------------------------------------------------------------
def test_partition_evaluation_returns_everything_the_protocol_asks_for():
    train, test = toy_cohort(400, seed=1), toy_cohort(120, seed=2)
    result = evaluate_partition(test, TwoStageForecaster.fit(train), n_boot=30, repeats=10)
    assert set(result) >= {"table", "selectors", "stage1", "coverage", "bootstrap", "audit", "calibration",
                           "spearman_with_real_error", "n_kept_primary"}
    assert result["n_kept_primary"] == n_keep(len(test), 0.8)
    assert set(result["bootstrap"]["coverage"]) == {0.9, 0.8, 0.7}
    assert len(result["table"]) == len(test)
    assert primary_row(result["bootstrap"])["comparison"] == "learned - tree_disagreement"


def test_partition_evaluation_is_repeatable():
    train, test = toy_cohort(300, seed=1), toy_cohort(100, seed=2)
    forecaster = TwoStageForecaster.fit(train)
    first = evaluate_partition(test, forecaster, n_boot=15, repeats=5)
    second = evaluate_partition(test, forecaster, n_boot=15, repeats=5)
    pd.testing.assert_frame_equal(first["table"], second["table"])
    pd.testing.assert_frame_equal(first["bootstrap"], second["bootstrap"])


def test_partition_evaluation_accepts_another_set_of_selectors():
    train, test = toy_cohort(300, seed=1), toy_cohort(100, seed=2)
    forecaster = TwoStageForecaster.fit(train)
    custom = [FixedScoreSelector("alpha", np.arange(100.0)), FixedScoreSelector("beta", np.arange(100.0)[::-1])]
    result = evaluate_partition(test, forecaster, n_boot=10, repeats=3, selectors=custom)
    assert set(result["coverage"]["selector"]) == {"alpha", "beta", "oracle", "random"}


def test_stage1_summary_matches_the_table():
    train, test = toy_cohort(300, seed=1), toy_cohort(100, seed=2)
    result = evaluate_partition(test, TwoStageForecaster.fit(train), n_boot=10, repeats=3)
    tab = result["table"]
    assert np.isclose(result["stage1"]["mae"], tab["abs_error"].mean())
    assert np.isclose(result["stage1"]["mean_signed_error"], tab["signed_error"].mean())
    assert (tab["abs_error"] == (tab["observed_hb"] - tab["forecast"]).abs()).all()


# ---------------------------------------------------------------------------
# What the interval is allowed to claim
# ---------------------------------------------------------------------------
def test_interval_sentence_never_claims_more_than_the_interval_shows():
    crossing = interval_sentence({"difference": -0.02, "ci95_low": -0.06, "ci95_high": 0.03}, "development")
    assert "neither superiority nor equivalence" in crossing and "does not establish equivalence" in crossing
    below = interval_sentence({"difference": -0.05, "ci95_low": -0.08, "ci95_high": -0.01}, "development")
    assert "lower average error" in below and "neither" not in below
    above = interval_sentence({"difference": 0.05, "ci95_low": 0.01, "ci95_high": 0.08}, "development")
    assert "higher average error" in above
