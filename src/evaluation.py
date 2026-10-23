"""
The protocol's comparison of the ways of choosing which forecasts to keep.

For one partition of women this module computes everything the frozen protocol asks for:
every selector at every coverage level, the paired bootstrap between selectors, the
descriptive audit of who is kept, and how well the Stage 2 score tracks the real error.

evaluate_partition is used unchanged for the development women now and for the locked-test
women once, so both are analysed by exactly the same code. The selectors come from
selection.py and the forecasts from forecaster.TwoStageForecaster; nothing here fits a model.

This module contains no execution code.
"""

from itertools import combinations
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from metrics import n_keep, root_mean_square
from pipeline import FEATURE_COLUMNS, GESTATION_VISIT2_COLUMN, TARGET_COLUMN, identical_vector_groups
from selection import OracleSelector, RandomSelector, ScoreSelector, SimpleRuleSelector, default_selectors

# Fixed by the protocol. Nothing here changes after the protocol is frozen.
PROTOCOL = {
    "coverage_levels": [1.0, 0.9, 0.8, 0.7],
    "primary_coverage": 0.8,
    "primary_comparison": ("learned minus tree disagreement: mean absolute error of the "
                           "retained forecasts at 80% coverage; negative favours the learned score"),
    "exact_coverage_rule": "keep the n lowest-scoring forecasts, n = round(coverage x N), halves up",
    "ties": "record ID, ascending",
    "random_repeats": 200,
    "random_seed": 42,
    "bootstrap_replicates": 5000,
    "bootstrap_seed": 42,
    "bootstrap_interval": "95% percentile interval, paired, resampling identical-record groups, "
                          "exact-coverage rule re-applied inside every replicate",
    "simple_rule": ("number of missing inputs (more missing = less reliable); ties broken by the "
                    "distance of first Hb from the training median (farther = less reliable); then record ID"),
    "first_hb_bands": ["<10", "10 to <11", ">=11"],
    "gap_bands_weeks": ["<=2", "3-4", "5-8", ">8"],
}


def pct(c: float) -> int:
    """A coverage fraction as a whole percentage (0.8 gives 80)."""
    return int(round(c * 100))


# ---------------------------------------------------------------------------
# Every selector at every coverage level
# ---------------------------------------------------------------------------
def coverage_table(tab: pd.DataFrame, selectors: Sequence[ScoreSelector], coverages, repeats: int,
                   seed: int) -> pd.DataFrame:
    """Every selector at every coverage level, on the same Stage 1 forecasts.

    The oracle (the forecasts that really were closest) is added as a reference, and random
    selection is repeated with fixed seeds. Every selector is asked the same question
    (summarise), so none of them is treated as a special case.
    """
    n = len(tab)
    rows = []
    for selector in [*selectors, OracleSelector(), RandomSelector(repeats, seed)]:
        for c in coverages:
            summary = selector.summarise(tab, c)
            rows.append(dict(selector=selector.name, coverage=c, n_total=n,
                             achieved_coverage=summary["n_kept"] / n, **summary))
    return pd.DataFrame(rows)


def _selector_pairs(names):
    """Every pair of selectors, then each selector against keeping everything."""
    return list(combinations(names, 2)) + [(name, "all") for name in names]


def _retained_mae(scores, rows, err, ids, coverages):
    """MAE of the kept forecasts for every selector and coverage, on one resample (`rows`), and the MAE of all."""
    e, r = err[rows], ids[rows]
    mae = {}
    for name in scores:
        order = np.lexsort((r, scores[name][rows]))
        for c in coverages:
            mae[(name, c)] = e[order[:n_keep(len(rows), c)]].mean()
    return mae, e.mean()


def _bootstrap_draws(tab, scores, pairs, coverages, groups, n_boot, seed):
    """For each replicate: resample identical-record groups, re-apply the exact-coverage rule, and
    record the difference in retained MAE for every pair and coverage."""
    ids = tab["record_id"].to_numpy()
    err = tab["abs_error"].to_numpy()
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    draws = {(a, b, c): np.empty(n_boot) for a, b in pairs for c in coverages}

    for i in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        rows = np.concatenate([members[j] for j in pick])
        mae, all_mae = _retained_mae(scores, rows, err, ids, coverages)
        for a, b in pairs:
            for c in coverages:
                draws[(a, b, c)][i] = mae[(a, c)] - (all_mae if b == "all" else mae[(b, c)])
    return draws


def paired_bootstrap_selectors(tab: pd.DataFrame, selectors: Sequence[ScoreSelector], coverages, groups,
                               n_boot: int, seed: int) -> pd.DataFrame:
    """Paired bootstrap of the difference in retained MAE between selectors.

    Every replicate resamples identical-record groups (effectively single women) with
    replacement, then re-applies the exact-coverage rule to the resampled cases. All
    selectors are scored on the same resampled women, so the comparison is paired.
    A negative 'A - B' means A kept forecasts with the lower error.

    Cost: n_boot replicates, each one sort per selector, O(n_boot x selectors x n log n).
    """
    err = tab["abs_error"].to_numpy()
    scores = {s.name: s.score(tab) for s in selectors}
    pairs = _selector_pairs(list(scores))
    draws = _bootstrap_draws(tab, scores, pairs, coverages, groups, n_boot, seed)

    # Point estimates come from the real (not resampled) women.
    point = {(s.name, c): err[s.keep_mask(tab, c)].mean() for s in selectors for c in coverages}
    all_point = err.mean()

    rows_out = []
    for a, b in pairs:
        for c in coverages:
            d = draws[(a, b, c)]
            other = all_point if b == "all" else point[(b, c)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows_out.append(dict(
                comparison=f"{a} - {b}", coverage=c,
                difference=float(point[(a, c)] - other),
                ci95_low=float(lo), ci95_high=float(hi), bootstrap_se=float(d.std(ddof=1)),
            ))
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Who is kept? (descriptive audit, never used to choose forecasts)
# ---------------------------------------------------------------------------
def audit_bands(tab: pd.DataFrame) -> dict:
    """The prespecified descriptive groupings: (label per woman, bands in natural order)."""
    hb_bins = [-np.inf, 10, 11, np.inf]
    hb_labels = PROTOCOL["first_hb_bands"]
    gap_labels = [f"{b} wk" for b in PROTOCOL["gap_bands_weeks"]]
    return {
        "first_hb": (pd.cut(tab["first_hb"], hb_bins, right=False, labels=hb_labels).astype(str),
                     hb_labels),
        "inputs": (pd.Series(np.where(tab["n_missing_inputs"] == 0, "complete", "incomplete"),
                             index=tab.index), ["complete", "incomplete"]),
        "followup_gap": (pd.cut(tab["gap_weeks"], [-np.inf, 2, 4, 8, np.inf],
                                labels=gap_labels).astype(str), gap_labels),
        "observed_followup_hb": (pd.cut(tab["observed_hb"], hb_bins, right=False,
                                        labels=hb_labels).astype(str), hb_labels),
    }


def retention_audit(tab: pd.DataFrame, selectors: Sequence[ScoreSelector], coverage: float) -> pd.DataFrame:
    """Retention rate and error by group, for each selector at one coverage level.

    Descriptive only. These groups are never used to choose which forecasts to keep.
    """
    err = tab["abs_error"].to_numpy()
    bands = audit_bands(tab)
    rows = []
    for selector in selectors:
        kept = selector.keep_mask(tab, coverage)
        for g_key, (labels, order) in bands.items():
            for b in order:
                inb = (labels == b).to_numpy()
                if not inb.any():
                    continue
                rows.append(dict(
                    selector=selector.name, grouping=g_key, group=b, n_women=int(inb.sum()),
                    n_kept=int((inb & kept).sum()),
                    retention_pct=float(100.0 * (inb & kept).sum() / inb.sum()),
                    mae_all=float(err[inb].mean()),
                    mae_kept=float(err[inb & kept].mean()) if (inb & kept).any() else float("nan"),
                ))
    return pd.DataFrame(rows)


def stage2_calibration(tab: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Average Stage 2 estimate versus average real error, by fifths of the Stage 2 score.

    A descriptive check that higher scores really go with larger errors. It does NOT make
    the score a calibrated confidence interval.
    """
    ranks = tab["stage2_score"].rank(method="first")
    fifth = pd.qcut(ranks, n_bins, labels=False) + 1
    g = tab.groupby(fifth)
    return pd.DataFrame({
        "fifth_of_stage2_score": g.size().index.astype(int),
        "n_women": g.size().to_numpy(),
        "mean_stage2_estimate": g["stage2_score"].mean().to_numpy(),
        "mean_real_abs_error": g["abs_error"].mean().to_numpy(),
    })


# ---------------------------------------------------------------------------
# One partition, everything the protocol asks for
# ---------------------------------------------------------------------------
def build_woman_table(df: pd.DataFrame, batch, train_hb_median: float) -> pd.DataFrame:
    """One row per woman: what was forecast, the real value, the errors and every selector's inputs."""
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    tab = pd.DataFrame({
        "record_id": df["_source_row"].to_numpy(),
        "observed_hb": y,
        "forecast": batch.forecast,
        "abs_error": np.abs(y - batch.forecast),
        "signed_error": batch.forecast - y,
        "tree_disagreement": batch.tree_disagreement,
        "stage2_score": batch.estimated_abs_error,
        "n_missing_inputs": df[FEATURE_COLUMNS].isna().sum(axis=1).to_numpy(),
        "first_hb": df["hb_visit1_g_dl"].to_numpy(dtype=float),
        "gap_weeks": (df[GESTATION_VISIT2_COLUMN] - df["gestation_visit1_weeks"]).to_numpy(dtype=float),
    })
    tab["simple_rule_score"] = SimpleRuleSelector(train_hb_median).score(tab)
    return tab


def stage1_summary(tab: pd.DataFrame) -> dict:
    """Stage 1 error over all the women, before any forecast is set aside."""
    return {
        "n_women": int(len(tab)),
        "mae": float(tab["abs_error"].mean()),
        "rmse": root_mean_square(tab["abs_error"]),
        "mean_signed_error": float(tab["signed_error"].mean()),
    }


def evaluate_partition(df: pd.DataFrame, forecaster, n_boot: Optional[int] = None,
                       repeats: Optional[int] = None,
                       selectors: Optional[Sequence[ScoreSelector]] = None) -> dict:
    """All comparisons the protocol specifies, for one partition of women.

    forecaster is a forecaster.TwoStageForecaster. selectors defaults to the protocol's
    three (learned, tree disagreement, simple rule); passing others compares them instead
    without changing this function.
    """
    n_boot = PROTOCOL["bootstrap_replicates"] if n_boot is None else n_boot
    repeats = PROTOCOL["random_repeats"] if repeats is None else repeats
    coverages = PROTOCOL["coverage_levels"]

    tab = build_woman_table(df, forecaster.forecast(df), forecaster.train_hb_median)
    selectors = default_selectors(forecaster.train_hb_median) if selectors is None else list(selectors)

    groups = identical_vector_groups(df)
    spearman = {s.name: float(pd.Series(s.score(tab)).corr(tab["abs_error"], method="spearman"))
                for s in selectors}
    return {
        "table": tab,
        "selectors": selectors,
        "stage1": stage1_summary(tab),
        "coverage": coverage_table(tab, selectors, coverages, repeats, PROTOCOL["random_seed"]),
        "bootstrap": paired_bootstrap_selectors(tab, selectors, [c for c in coverages if c < 1.0], groups,
                                                n_boot, PROTOCOL["bootstrap_seed"]),
        "audit": retention_audit(tab, selectors, PROTOCOL["primary_coverage"]),
        "calibration": stage2_calibration(tab),
        "spearman_with_real_error": spearman,
        "n_kept_primary": n_keep(len(tab), PROTOCOL["primary_coverage"]),
    }


def primary_row(boot: pd.DataFrame) -> dict:
    """The one pre-declared primary result: learned minus tree disagreement at 80% coverage."""
    r = boot[(boot["comparison"] == "learned - tree_disagreement") &
             (boot["coverage"] == PROTOCOL["primary_coverage"])].iloc[0]
    return {k: (float(r[k]) if k != "comparison" else r[k]) for k in r.index}


def interval_sentence(row: dict, partition_label: str) -> str:
    """Plain statement of what the primary interval does and does not show. No spin."""
    d, lo, hi = row["difference"], row["ci95_low"], row["ci95_high"]
    base = (f"On the {partition_label} women, the learned score minus tree disagreement at 80% coverage "
            f"is {d:+.3f} g/dL (95% interval {lo:+.3f} to {hi:+.3f}).")
    if hi < 0:
        return (base + " The whole interval lies below zero, so in this partition the learned score kept "
                "forecasts with a lower average error than tree disagreement did.")
    if lo > 0:
        return (base + " The whole interval lies above zero, so in this partition the learned score kept "
                "forecasts with a higher average error than tree disagreement did.")
    return (base + " The interval crosses zero. This shows neither superiority nor equivalence: the study "
            "reports the estimated difference and its uncertainty and does not establish equivalence.")
