"""
Stage 2: estimate how large each Stage 1 forecast's error is likely to be, and
compare ways of choosing which forecasts to keep.

The idea in plain words
-----------------------
Stage 1 forecasts a woman's haemoglobin at her next recorded visit. Its average error
hides which individual forecasts are bad. Stage 2 learns to predict the size of each
forecast's error, so forecasts can be ranked from most to least reliable. A LOWER
Stage 2 score means a MORE reliable forecast.

How Stage 2 is trained (no leakage)
-----------------------------------
1. Run the same 3-fold cross-validation as Stage 1 on the training women, using the
   frozen Stage 1 recipe (the forest). Every woman gets a forecast from a model that
   never saw her: an out-of-fold forecast. Her error is |observed Hb - that forecast|.
2. Train a separate random forest on those out-of-fold errors. Its inputs are the ten
   first-visit values, the out-of-fold forecast and the disagreement among the forest's
   individual trees. It never sees in-sample residuals, which are too optimistic.
3. For development and test women the frozen Stage 1 forest (fitted on all training
   women) supplies the forecast and the tree disagreement, and Stage 2 turns those
   plus the ten inputs into an estimated absolute error.

The same frozen Stage 1 forecasts feed every selector, so the comparison is paired.

This module contains no execution code. It is used by run_stage2.py (development
data), run_locked_test.py (the locked test, once) and app/main.py (the demo API).
"""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from pipeline import (
    FEATURE_COLUMNS, TARGET_COLUMN, GESTATION_VISIT2_COLUMN, STAGE1_CONFIG,
    _prep_matrix, tree_disagreement, assign_cv_folds, identical_vector_groups,
    cross_validated_predictions, keep_mask, n_keep,
)

# ---------------------------------------------------------------------------
# Fixed configuration. Nothing here is tuned on any rows whose errors become
# Stage 2 targets, and none of it changes after the protocol is frozen.
# ---------------------------------------------------------------------------
STAGE2_INPUT_COLUMNS = FEATURE_COLUMNS + ["stage1_forecast", "tree_disagreement"]

STAGE2_CONFIG = {
    "seed": 42,
    "random_forest": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 5},
    "target": "absolute error of the out-of-fold Stage 1 forecast",
    "inputs": ("ten first-visit values (training-fitted median imputation), "
               "out-of-fold Stage 1 forecast, tree disagreement"),
}

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

SELECTORS = ["learned", "tree_disagreement", "simple_rule"]
SELECTOR_LABELS = {
    "learned": "Learned Stage 2 score",
    "tree_disagreement": "Tree disagreement",
    "simple_rule": "Simple rule",
    "random": "Random selection",
    "oracle": "Oracle (reference only)",
}


def pct(c: float) -> int:
    return int(round(c * 100))


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------
def make_run_id(parts: dict) -> str:
    """Short, repeatable ID built from the things that define a run (data, split, folds, settings)."""
    payload = json.dumps(parts, sort_keys=True, default=str)
    return "run-" + hashlib.sha256(payload.encode("utf8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Out-of-fold table and Stage 2 model
# ---------------------------------------------------------------------------
def build_oof_table(train_df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """Out-of-fold artefact: one row per training woman.

    Columns: record_id, fold, observed_hb, oof_forecast, tree_disagreement, abs_error.
    The forecast for each woman comes from a forest fitted on the other folds only.
    """
    cfg = STAGE1_CONFIG
    seed = cfg["seed"]

    def make_forest():
        return RandomForestRegressor(**cfg["random_forest"], random_state=seed)

    oof, td, _ = cross_validated_predictions(train_df, make_forest, "random_forest", folds,
                                             want_disagreement=True)
    y = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    return pd.DataFrame({
        "record_id": train_df["_source_row"].to_numpy(),
        "fold": np.asarray(folds, dtype=int),
        "observed_hb": y,
        "oof_forecast": oof,
        "tree_disagreement": td,
        "abs_error": np.abs(y - oof),
    })


def stage2_features(df: pd.DataFrame, imputer, forecast, disagreement) -> pd.DataFrame:
    """The 12 Stage 2 inputs: ten first-visit values (median-imputed with the training medians),
    the Stage 1 forecast and the tree disagreement."""
    filled = pd.DataFrame(imputer.transform(df[FEATURE_COLUMNS]), columns=FEATURE_COLUMNS,
                          index=df.index)
    filled["stage1_forecast"] = np.asarray(forecast, dtype=float)
    filled["tree_disagreement"] = np.asarray(disagreement, dtype=float)
    return filled[STAGE2_INPUT_COLUMNS]


def fit_stage2(oof_table: pd.DataFrame, train_df: pd.DataFrame, imputer) -> RandomForestRegressor:
    """Fit the Stage 2 forest on the whole out-of-fold table (a separate instance from Stage 1)."""
    assert np.array_equal(oof_table["record_id"].to_numpy(), train_df["_source_row"].to_numpy()), \
        "out-of-fold table and training table are not in the same row order"
    X = stage2_features(train_df, imputer, oof_table["oof_forecast"], oof_table["tree_disagreement"])
    cfg = STAGE2_CONFIG
    model = RandomForestRegressor(**cfg["random_forest"], random_state=cfg["seed"])
    return model.fit(X, oof_table["abs_error"].to_numpy(dtype=float))


def fit_stage1_forest(train_df: pd.DataFrame) -> dict:
    """The frozen Stage 1 recipe fitted on a whole training set: forest + training-fitted imputer."""
    cfg = STAGE1_CONFIG
    X, imputer = _prep_matrix(train_df, fit=True)
    forest = RandomForestRegressor(**cfg["random_forest"], random_state=cfg["seed"])
    forest.fit(X, train_df[TARGET_COLUMN].to_numpy(dtype=float))
    return {"model": forest, "imputer": imputer, "name": "random_forest"}


def fit_two_stage(train_df: pd.DataFrame, bundle1: dict = None):
    """Out-of-fold table, then Stage 2, for one training set.

    bundle1 is the frozen Stage 1 forest. If it is None, it is fitted here (used by the
    complete-case sensitivity analysis, which repeats everything on a subset).
    Returns (bundle1, stage2_model, oof_table, folds).
    """
    if bundle1 is None:
        bundle1 = fit_stage1_forest(train_df)
    folds = assign_cv_folds(identical_vector_groups(train_df), STAGE1_CONFIG["n_folds"])
    oof = build_oof_table(train_df, folds)
    return bundle1, fit_stage2(oof, train_df, bundle1["imputer"]), oof, folds


def stage1_outputs(bundle1: dict, df: pd.DataFrame):
    """Stage 1 forecast and tree disagreement from the frozen forest."""
    X, _ = _prep_matrix(df, imputer=bundle1["imputer"], fit=False)
    model = bundle1["model"]
    return model.predict(X), tree_disagreement(model, X)


def forecast_with_reliability(bundle1: dict, stage2_model, df: pd.DataFrame):
    """(forecast, tree disagreement, Stage 2 estimated absolute error) for every row of df.

    The one function used by the evaluation scripts AND the demo API, so the API output
    equals the offline output for the same woman by construction.
    """
    forecast, disagreement = stage1_outputs(bundle1, df)
    X2 = stage2_features(df, bundle1["imputer"], forecast, disagreement)
    return forecast, disagreement, stage2_model.predict(X2)


# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------
def simple_rule_score(df: pd.DataFrame, train_hb_median: float) -> np.ndarray:
    """The simple rule a clinic could apply by hand, fixed before any results were seen.

    More missing inputs means a less reliable forecast. Ties are broken by how far the
    first haemoglobin is from the training median (farther = less reliable). Written as
    one number so it can be ranked like the other scores: the distance is divided by 100
    so it can only separate women who have the same number of missing inputs.
    """
    n_missing = df[FEATURE_COLUMNS].isna().sum(axis=1).to_numpy(dtype=float)
    distance = np.abs(df["hb_visit1_g_dl"].to_numpy(dtype=float) - float(train_hb_median))
    assert distance.max() < 100, "distance would no longer act only as a tie-break"
    return n_missing + distance / 100.0


def _rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def _retained_metrics(abs_err, signed_err, mask):
    return {
        "n_kept": int(mask.sum()),
        "mae": float(abs_err[mask].mean()),
        "rmse": _rmse(abs_err[mask]),
        "mean_signed_error": float(signed_err[mask].mean()),
    }


def coverage_table(tab: pd.DataFrame, scores: dict, coverages, repeats: int, seed: int) -> pd.DataFrame:
    """Every selector at every coverage level, on the same Stage 1 forecasts."""
    ids = tab["record_id"].to_numpy()
    abs_err = tab["abs_error"].to_numpy()
    signed = tab["signed_error"].to_numpy()
    n = len(tab)
    rows = []
    for name in SELECTORS:
        for c in coverages:
            m = keep_mask(scores[name], c, ids)
            rows.append(dict(selector=name, coverage=c, n_total=n, achieved_coverage=m.mean(),
                             **_retained_metrics(abs_err, signed, m)))
    # Oracle: keeps the forecasts that really were closest. Not a usable selector;
    # shown only to mark how much room any selector has.
    for c in coverages:
        m = keep_mask(abs_err, c, ids)
        rows.append(dict(selector="oracle", coverage=c, n_total=n, achieved_coverage=m.mean(),
                         **_retained_metrics(abs_err, signed, m)))
    # Random selection, repeated with fixed seeds (one stream per coverage level).
    for c in coverages:
        k = n_keep(n, c)
        rng = np.random.default_rng([seed, pct(c)])
        maes, rmses, sgn = [], [], []
        for _ in range(repeats):
            idx = rng.choice(n, size=k, replace=False)
            maes.append(abs_err[idx].mean())
            rmses.append(_rmse(abs_err[idx]))
            sgn.append(signed[idx].mean())
        rows.append(dict(selector="random", coverage=c, n_total=n, achieved_coverage=k / n,
                         n_kept=k, mae=float(np.mean(maes)), rmse=float(np.mean(rmses)),
                         mean_signed_error=float(np.mean(sgn)),
                         mae_p2_5=float(np.percentile(maes, 2.5)),
                         mae_p97_5=float(np.percentile(maes, 97.5))))
    out = pd.DataFrame(rows)
    return out


def paired_bootstrap_selectors(tab: pd.DataFrame, scores: dict, coverages, groups, n_boot: int, seed: int):
    """Paired bootstrap of the difference in retained MAE between selectors.

    Every replicate resamples identical-record groups (effectively single women) with
    replacement, then re-applies the exact-coverage rule to the resampled cases. All
    selectors are scored on the same resampled women, so the comparison is paired.
    A negative 'A - B' means A kept forecasts with the lower error.
    """
    ids = tab["record_id"].to_numpy()
    err = tab["abs_error"].to_numpy()
    members = [np.where(groups == g)[0] for g in np.unique(groups)]
    n_groups = len(members)
    rng = np.random.default_rng(seed)

    pairs = [("learned", "tree_disagreement"), ("learned", "simple_rule"),
             ("tree_disagreement", "simple_rule"),
             ("learned", "all"), ("tree_disagreement", "all"), ("simple_rule", "all")]
    draws = {(a, b, c): np.empty(n_boot) for a, b in pairs for c in coverages}

    for i in range(n_boot):
        pick = rng.integers(0, n_groups, size=n_groups)
        rows = np.concatenate([members[j] for j in pick])
        e, r = err[rows], ids[rows]
        all_mae = e.mean()
        mae = {}
        for s in SELECTORS:
            order = np.lexsort((r, scores[s][rows]))
            for c in coverages:
                mae[(s, c)] = e[order[:n_keep(len(rows), c)]].mean()
        for a, b in pairs:
            for c in coverages:
                other = all_mae if b == "all" else mae[(b, c)]
                draws[(a, b, c)][i] = mae[(a, c)] - other

    # Point estimates come from the real (not resampled) women.
    point = {}
    for s in SELECTORS:
        for c in coverages:
            point[(s, c)] = err[keep_mask(scores[s], c, ids)].mean()
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


GROUPING_LABELS = {
    "first_hb": "First haemoglobin (g/dL)",
    "inputs": "First-visit inputs",
    "followup_gap": "Follow-up gap",
    "observed_followup_hb": "Observed follow-up haemoglobin (g/dL)",
}


def retention_audit(tab: pd.DataFrame, scores: dict, coverage: float) -> pd.DataFrame:
    """Retention rate and error by group, for each selector at one coverage level.

    Descriptive only. These groups are never used to choose which forecasts to keep.
    """
    ids = tab["record_id"].to_numpy()
    err = tab["abs_error"].to_numpy()
    bands = audit_bands(tab)
    rows = []
    for s in SELECTORS:
        kept = keep_mask(scores[s], coverage, ids)
        for g_key, (labels, order) in bands.items():
            for b in order:
                inb = (labels == b).to_numpy()
                if not inb.any():
                    continue
                rows.append(dict(
                    selector=s, grouping=g_key, group=b, n_women=int(inb.sum()),
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
def evaluate_partition(df: pd.DataFrame, bundle1: dict, stage2_model, train_hb_median: float,
                       n_boot: int = None, repeats: int = None) -> dict:
    """All comparisons the protocol specifies, for one partition of women.

    Used unchanged for the development partition now and for the locked test partition
    once, so both are analysed by exactly the same code.
    """
    n_boot = PROTOCOL["bootstrap_replicates"] if n_boot is None else n_boot
    repeats = PROTOCOL["random_repeats"] if repeats is None else repeats
    coverages = PROTOCOL["coverage_levels"]

    forecast, disagreement, learned = forecast_with_reliability(bundle1, stage2_model, df)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    tab = pd.DataFrame({
        "record_id": df["_source_row"].to_numpy(),
        "observed_hb": y,
        "forecast": forecast,
        "abs_error": np.abs(y - forecast),
        "signed_error": forecast - y,
        "tree_disagreement": disagreement,
        "stage2_score": learned,
        "n_missing_inputs": df[FEATURE_COLUMNS].isna().sum(axis=1).to_numpy(),
        "first_hb": df["hb_visit1_g_dl"].to_numpy(dtype=float),
        "gap_weeks": (df[GESTATION_VISIT2_COLUMN] - df["gestation_visit1_weeks"]).to_numpy(dtype=float),
    })
    tab["simple_rule_score"] = simple_rule_score(df, train_hb_median)
    scores = {"learned": learned, "tree_disagreement": disagreement,
              "simple_rule": tab["simple_rule_score"].to_numpy()}

    groups = identical_vector_groups(df)
    cov = coverage_table(tab, scores, coverages, repeats, PROTOCOL["random_seed"])
    boot = paired_bootstrap_selectors(tab, scores, [c for c in coverages if c < 1.0], groups, n_boot,
                                      PROTOCOL["bootstrap_seed"])
    audit = retention_audit(tab, scores, PROTOCOL["primary_coverage"])

    spearman = {s: float(pd.Series(scores[s]).corr(tab["abs_error"], method="spearman"))
                for s in SELECTORS}
    return {
        "table": tab,
        "stage1": {
            "n_women": int(len(tab)),
            "mae": float(tab["abs_error"].mean()),
            "rmse": _rmse(tab["abs_error"]),
            "mean_signed_error": float(tab["signed_error"].mean()),
        },
        "coverage": cov,
        "bootstrap": boot,
        "audit": audit,
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


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def md_table(df: pd.DataFrame) -> str:
    """Pipe table from a DataFrame; integers stay integers, floats get three decimals."""
    cols = list(df.columns)
    cells = {}
    for c in cols:
        col = df[c]
        if pd.api.types.is_integer_dtype(col):
            cells[c] = [str(int(v)) for v in col]
        elif pd.api.types.is_float_dtype(col):
            cells[c] = ["" if np.isnan(v) else f"{v:.3f}" for v in col]
        else:
            cells[c] = [str(v) for v in col]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for i in range(len(df)):
        lines.append("| " + " | ".join(cells[c][i] for c in cols) + " |")
    return "\n".join(lines)


def coverage_for_display(cov: pd.DataFrame) -> pd.DataFrame:
    out = cov.copy()
    out["selector"] = out["selector"].map(SELECTOR_LABELS)
    out["coverage"] = out["coverage"].map(lambda c: f"{pct(c)}%")
    out["achieved_coverage"] = out["achieved_coverage"].map(lambda c: f"{100 * c:.1f}%")
    out["random 2.5th to 97.5th pct of MAE"] = [
        (f"{lo:.3f} to {hi:.3f}" if not np.isnan(lo) else "")
        for lo, hi in zip(out.get("mae_p2_5", np.nan), out.get("mae_p97_5", np.nan))
    ]
    keep = ["selector", "coverage", "n_kept", "achieved_coverage", "mae", "rmse",
            "mean_signed_error", "random 2.5th to 97.5th pct of MAE"]
    out = out[keep].rename(columns={"mae": "MAE", "rmse": "RMSE", "mean_signed_error": "mean signed error",
                                    "n_kept": "kept", "achieved_coverage": "achieved coverage"})
    return out


def bootstrap_for_display(boot: pd.DataFrame) -> pd.DataFrame:
    out = boot.copy()
    out["coverage"] = out["coverage"].map(lambda c: f"{pct(c)}%")
    out["comparison"] = out["comparison"].str.replace("_", " ").str.replace(" - all", " - keep all")
    return out.rename(columns={"difference": "difference (g/dL)", "ci95_low": "95% low",
                               "ci95_high": "95% high", "bootstrap_se": "bootstrap SE"})


def audit_for_display(audit: pd.DataFrame) -> pd.DataFrame:
    out = audit.copy()
    out["selector"] = out["selector"].map(SELECTOR_LABELS)
    out["grouping"] = out["grouping"].map(GROUPING_LABELS)
    out["retention_pct"] = out["retention_pct"].map(lambda v: f"{v:.1f}%")
    return out.rename(columns={"n_women": "women", "n_kept": "kept", "retention_pct": "share kept",
                               "mae_all": "MAE all women in group", "mae_kept": "MAE kept women in group"})


def results_to_json(res: dict) -> dict:
    """Plain-data version of evaluate_partition's output (the per-woman table is saved separately)."""
    def records(df):
        return json.loads(df.to_json(orient="records"))
    return {
        "stage1": res["stage1"],
        "n_kept_primary": res["n_kept_primary"],
        "coverage": records(res["coverage"]),
        "bootstrap": records(res["bootstrap"]),
        "primary": primary_row(res["bootstrap"]),
        "audit": records(res["audit"]),
        "calibration": records(res["calibration"]),
        "spearman_with_real_error": res["spearman_with_real_error"],
    }


# ---------------------------------------------------------------------------
# Figures (matplotlib is imported only when a figure is drawn, so the API does not need it)
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_error_vs_coverage(res: dict, path: str, title: str) -> None:
    plt = _plt()
    tab = res["table"]
    ids = tab["record_id"].to_numpy()
    err = tab["abs_error"].to_numpy()
    scores = {"learned": tab["stage2_score"].to_numpy(),
              "tree_disagreement": tab["tree_disagreement"].to_numpy(),
              "simple_rule": tab["simple_rule_score"].to_numpy()}
    grid = np.round(np.arange(1.0, 0.4999, -0.01), 2)
    colours = {"learned": "#1b6ca8", "tree_disagreement": "#d9822b", "simple_rule": "#6a994e"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in SELECTORS:
        ax.plot(grid * 100, [err[keep_mask(scores[s], c, ids)].mean() for c in grid],
                label=SELECTOR_LABELS[s], color=colours[s], lw=2)
    ax.plot(grid * 100, [err[keep_mask(err, c, ids)].mean() for c in grid], color="#999999", ls="--",
            lw=1.2, label=SELECTOR_LABELS["oracle"])
    cov = res["coverage"]
    r = cov[cov["selector"] == "random"]
    lower = np.maximum(r["mae"] - r["mae_p2_5"], 0.0)  # floating-point noise can make this -1e-16
    upper = np.maximum(r["mae_p97_5"] - r["mae"], 0.0)
    ax.errorbar(r["coverage"] * 100, r["mae"], yerr=[lower, upper],
                fmt="o", color="black", capsize=3, ms=4, label="Random selection (mean, 2.5th to 97.5th pct)")
    ax.axvline(80, color="#bbbbbb", lw=1)
    ax.text(80.4, 0.03, "80% (primary)", transform=ax.get_xaxis_transform(), fontsize=8, va="bottom",
            color="#666666")
    ax.set_xlabel("Coverage: share of forecasts kept (%)")
    ax.set_ylabel("MAE of the kept forecasts (g/dL)")
    ax.set_title(title, fontsize=11)
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_forecast_vs_observed(res: dict, path: str, title: str) -> None:
    plt = _plt()
    tab = res["table"]
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(tab["observed_hb"], tab["forecast"], s=14, alpha=0.55, color="#1b6ca8")
    lo = float(min(tab["observed_hb"].min(), tab["forecast"].min())) - 0.3
    hi = float(max(tab["observed_hb"].max(), tab["forecast"].max())) + 0.3
    ax.plot([lo, hi], [lo, hi], color="#888888", ls="--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Observed haemoglobin at the next recorded visit (g/dL)")
    ax.set_ylabel("Stage 1 forecast (g/dL)")
    ax.set_title(f"{title}\nMAE {res['stage1']['mae']:.3f} g/dL, n = {res['stage1']['n_women']}", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_stage2_check(res: dict, path: str, title: str) -> None:
    plt = _plt()
    cal = res["calibration"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot(cal["fifth_of_stage2_score"], cal["mean_stage2_estimate"], "o-", color="#1b6ca8",
            label="Stage 2 estimate (average)")
    ax.plot(cal["fifth_of_stage2_score"], cal["mean_real_abs_error"], "s-", color="#d9822b",
            label="Real absolute error (average)")
    ax.set_xticks(cal["fifth_of_stage2_score"])
    ax.set_xlabel("Fifth of women, from lowest to highest Stage 2 score")
    ax.set_ylabel("Absolute error (g/dL)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Written report (the same text generator is used for development and for the locked test)
# ---------------------------------------------------------------------------
LIMITATIONS = [
    "This is internal validation on one public dataset (five Tanzanian districts, not nationally "
    "representative). It is not temporal or external validation.",
    "There is no verified patient identifier. Keeping rows with an identical ten-input vector together "
    "is only a proxy for the same woman and cannot guarantee independence between partitions.",
    "The target is haemoglobin at the next recorded visit. The gap between visits runs from 1 to 28 "
    "weeks (median 4), so this is not a fixed-interval forecast.",
    "The out-of-fold forests that make the Stage 2 training table are fitted on two thirds of the training "
    "women, so the forecasts Stage 2 later receives come from a slightly stronger forest than the one that "
    "produced its training examples.",
    "The Stage 2 score is used to rank forecasts. It is not a calibrated confidence interval.",
    "Eligibility requires a repeat test, which selects women who were tested twice.",
    "'Set aside' means a research output is not counted in this evaluation. It does not mean withholding "
    "care or testing, and clinical usefulness has not been verified.",
]


def _primary_block(res: dict, label: str) -> list:
    prow = primary_row(res["bootstrap"])
    cov = res["coverage"]
    c80 = cov[(cov["coverage"] == PROTOCOL["primary_coverage"]) & (cov["selector"].isin(
        ["learned", "tree_disagreement", "simple_rule", "random", "oracle"]))]
    lines = [interval_sentence(prow, label), "",
             f"At 80% coverage the exact-coverage rule keeps {res['n_kept_primary']} of "
             f"{res['stage1']['n_women']} women. Mean absolute error of the kept forecasts:", ""]
    for _, r in c80.iterrows():
        extra = ""
        if r["selector"] == "random":
            extra = f" (2.5th to 97.5th percentile of {PROTOCOL['random_repeats']} repeats: " \
                    f"{r['mae_p2_5']:.3f} to {r['mae_p97_5']:.3f})"
        lines.append(f"- {SELECTOR_LABELS[r['selector']]}: {r['mae']:.3f} g/dL{extra}")
    all_row = cov[(cov["coverage"] == 1.0) & (cov["selector"] == "learned")].iloc[0]
    lines.append(f"- Keeping every forecast (100% coverage): {all_row['mae']:.3f} g/dL")
    return lines


def render_report(title: str, label: str, meta: dict, res: dict, res_cc: dict = None,
                  cc_note: str = "") -> str:
    """Markdown report for one partition. label is 'development' or 'locked test'."""
    L = [f"# {title}", ""]
    L.append(f"- Run ID: `{meta['run_id']}`")
    L.append(f"- Partition: {label} ({res['stage1']['n_women']} women)")
    L.append(f"- Python {meta['versions']['python']}, scikit-learn {meta['versions']['scikit-learn']}, "
             f"pandas {meta['versions']['pandas']}, numpy {meta['versions']['numpy']}")
    for extra in meta.get("extra_lines", []):
        L.append(f"- {extra}")
    L += ["", "Lower scores mean more reliable forecasts. Errors are in g/dL. MAE is mean absolute error.", ""]

    s1 = res["stage1"]
    L += ["## Stage 1 forecasts on these women", "",
          f"MAE {s1['mae']:.3f}, RMSE {s1['rmse']:.3f}, mean signed error {s1['mean_signed_error']:+.3f} "
          "(positive means the forecast ran high).", ""]

    L += [f"## Primary comparison (declared in advance): learned minus tree disagreement at 80% coverage", ""]
    L += _primary_block(res, label)
    L += ["", "## Every selector at every coverage level", "",
          md_table(coverage_for_display(res["coverage"])), "",
          f"Random selection is repeated {PROTOCOL['random_repeats']} times with fixed seeds. The oracle keeps "
          "the forecasts that really were closest; it cannot be used in practice and only marks how much "
          "room any selector has.", ""]
    L += ["## All paired comparisons (paired bootstrap, "
          f"{PROTOCOL['bootstrap_replicates']} replicates)", "",
          "A negative 'A - B' means A kept forecasts with the lower average error. 'keep all' is the "
          "average error of all forecasts at 100% coverage. Only the 80% learned-minus-tree-disagreement "
          "row was declared as primary; every other row is secondary.", "",
          md_table(bootstrap_for_display(res["bootstrap"])), ""]
    L += ["## Who is kept at 80% coverage? (descriptive audit only)", "",
          "These groups are never used to choose which forecasts to keep.", "",
          md_table(audit_for_display(res["audit"])), ""]
    sp = res["spearman_with_real_error"]
    L += ["## Does a higher score go with a larger real error?", "",
          "Spearman correlation between each score and the real absolute error: " +
          ", ".join(f"{SELECTOR_LABELS[k].lower()} {v:.3f}" for k, v in sp.items()) + ".", "",
          md_table(res["calibration"].rename(columns={
              "fifth_of_stage2_score": "fifth of Stage 2 score", "n_women": "women",
              "mean_stage2_estimate": "mean Stage 2 estimate", "mean_real_abs_error": "mean real error"})), "",
          "This table checks the ordering. It does not make the score a calibrated confidence interval.", ""]
    if res_cc is not None:
        prow = primary_row(res_cc["bootstrap"])
        L += ["## Complete-case sensitivity (women with all ten inputs)", "",
              cc_note, "", interval_sentence(prow, label + " complete-case"), "",
              md_table(coverage_for_display(res_cc["coverage"])), ""]
    L += ["## Limits that apply to every number above", ""] + [f"- {t}" for t in LIMITATIONS] + [""]
    return "\n".join(L)
