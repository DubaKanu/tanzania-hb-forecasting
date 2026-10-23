"""
Tables, figures and the written report.

Everything here turns the output of evaluation.evaluate_partition into something a person
reads: pipe tables, plain-data JSON, three figures and a Markdown report. The same text
generator is used for the development women and for the locked test, so the two reports are
worded the same way. Nothing here computes a result.

matplotlib is imported only when a figure is drawn, so the API does not need it.

This module contains no execution code.
"""

import json

import numpy as np
import pandas as pd

from evaluation import PROTOCOL, interval_sentence, pct, primary_row
from selection import SELECTOR_LABELS, OracleSelector

GROUPING_LABELS = {
    "first_hb": "First haemoglobin (g/dL)",
    "inputs": "First-visit inputs",
    "followup_gap": "Follow-up gap",
    "observed_followup_hb": "Observed follow-up haemoglobin (g/dL)",
}

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


# ---------------------------------------------------------------------------
# Tables
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
    """The coverage table with readable labels, for the markdown report."""
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
    """The paired-bootstrap table with readable labels."""
    out = boot.copy()
    out["coverage"] = out["coverage"].map(lambda c: f"{pct(c)}%")
    out["comparison"] = out["comparison"].str.replace("_", " ").str.replace(" - all", " - keep all")
    return out.rename(columns={"difference": "difference (g/dL)", "ci95_low": "95% low",
                               "ci95_high": "95% high", "bootstrap_se": "bootstrap SE"})


def audit_for_display(audit: pd.DataFrame) -> pd.DataFrame:
    """The who-is-kept table with readable labels."""
    out = audit.copy()
    out["selector"] = out["selector"].map(SELECTOR_LABELS)
    out["grouping"] = out["grouping"].map(GROUPING_LABELS)
    out["retention_pct"] = out["retention_pct"].map(lambda v: f"{v:.1f}%")
    return out.rename(columns={"n_women": "women", "n_kept": "kept", "retention_pct": "share kept",
                               "mae_all": "MAE all women in group", "mae_kept": "MAE kept women in group"})


def results_to_json(res: dict) -> dict:
    """Plain-data version of evaluate_partition's output (the per-woman table is saved separately)."""
    def records(df):
        """A data frame as a list of plain dictionaries, which json can save."""
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
# Figures (matplotlib is imported only when a figure is drawn)
# ---------------------------------------------------------------------------
def _plt():
    """matplotlib's pyplot with the non-interactive backend, imported only when a figure is drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_error_vs_coverage(res: dict, path: str, title: str) -> None:
    """Draw the error of the kept forecasts against coverage for each selector and save it to `path`."""
    plt = _plt()
    tab = res["table"]
    grid = np.round(np.arange(1.0, 0.4999, -0.01), 2)
    colours = {"learned": "#1b6ca8", "tree_disagreement": "#d9822b", "simple_rule": "#6a994e"}
    err = tab["abs_error"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for selector in res["selectors"]:
        ax.plot(grid * 100, [err[selector.keep_mask(tab, c)].mean() for c in grid],
                label=SELECTOR_LABELS[selector.name], color=colours.get(selector.name), lw=2)
    oracle = OracleSelector()
    ax.plot(grid * 100, [err[oracle.keep_mask(tab, c)].mean() for c in grid], color="#999999", ls="--",
            lw=1.2, label=SELECTOR_LABELS[oracle.name])
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
    """Draw each woman's forecast against her observed follow-up Hb and save it to `path`."""
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
    """Draw the mean Stage 2 estimate against the mean real error in each fifth of the score and save it to `path`."""
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
def _primary_block(res: dict, label: str) -> list:
    """The sentences and numbers of the primary comparison (learned minus tree disagreement at 80%)."""
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


def _header_section(title: str, label: str, meta: dict, res: dict) -> list:
    """The title and the run details at the top of the report."""
    lines = [f"# {title}", "",
             f"- Run ID: `{meta['run_id']}`",
             f"- Partition: {label} ({res['stage1']['n_women']} women)",
             f"- Python {meta['versions']['python']}, scikit-learn {meta['versions']['scikit-learn']}, "
             f"pandas {meta['versions']['pandas']}, numpy {meta['versions']['numpy']}"]
    lines += [f"- {extra}" for extra in meta.get("extra_lines", [])]
    lines += ["", "Lower scores mean more reliable forecasts. Errors are in g/dL. MAE is mean absolute error.", ""]
    return lines


def _stage1_section(res: dict) -> list:
    """The Stage 1 error on these women."""
    s1 = res["stage1"]
    return ["## Stage 1 forecasts on these women", "",
            f"MAE {s1['mae']:.3f}, RMSE {s1['rmse']:.3f}, mean signed error {s1['mean_signed_error']:+.3f} "
            "(positive means the forecast ran high).", ""]


def _primary_section(res: dict, label: str) -> list:
    """The primary comparison, declared in advance."""
    return (["## Primary comparison (declared in advance): learned minus tree disagreement at 80% coverage", ""]
            + _primary_block(res, label))


def _coverage_section(res: dict) -> list:
    """Every selector at every coverage level."""
    return ["", "## Every selector at every coverage level", "",
            md_table(coverage_for_display(res["coverage"])), "",
            f"Random selection is repeated {PROTOCOL['random_repeats']} times with fixed seeds. The oracle keeps "
            "the forecasts that really were closest; it cannot be used in practice and only marks how much "
            "room any selector has.", ""]


def _bootstrap_section(res: dict) -> list:
    """All paired comparisons with their bootstrap intervals."""
    return ["## All paired comparisons (paired bootstrap, "
            f"{PROTOCOL['bootstrap_replicates']} replicates)", "",
            "A negative 'A - B' means A kept forecasts with the lower average error. 'keep all' is the "
            "average error of all forecasts at 100% coverage. Only the 80% learned-minus-tree-disagreement "
            "row was declared as primary; every other row is secondary.", "",
            md_table(bootstrap_for_display(res["bootstrap"])), ""]


def _audit_section(res: dict) -> list:
    """Who is kept at 80% coverage (descriptive only)."""
    return ["## Who is kept at 80% coverage? (descriptive audit only)", "",
            "These groups are never used to choose which forecasts to keep.", "",
            md_table(audit_for_display(res["audit"])), ""]


def _ordering_section(res: dict) -> list:
    """Whether a higher score goes with a larger real error."""
    sp = res["spearman_with_real_error"]
    return ["## Does a higher score go with a larger real error?", "",
            "Spearman correlation between each score and the real absolute error: " +
            ", ".join(f"{SELECTOR_LABELS[k].lower()} {v:.3f}" for k, v in sp.items()) + ".", "",
            md_table(res["calibration"].rename(columns={
                "fifth_of_stage2_score": "fifth of Stage 2 score", "n_women": "women",
                "mean_stage2_estimate": "mean Stage 2 estimate", "mean_real_abs_error": "mean real error"})), "",
            "This table checks the ordering. It does not make the score a calibrated confidence interval.", ""]


def _complete_case_section(res_cc: dict, label: str, cc_note: str) -> list:
    """The repeat of the whole comparison on the women with all ten inputs."""
    prow = primary_row(res_cc["bootstrap"])
    return ["## Complete-case sensitivity (women with all ten inputs)", "",
            cc_note, "", interval_sentence(prow, label + " complete-case"), "",
            md_table(coverage_for_display(res_cc["coverage"])), ""]


def render_report(title: str, label: str, meta: dict, res: dict, res_cc: dict = None,
                  cc_note: str = "") -> str:
    """Markdown report for one partition. label is 'development' or 'locked test'."""
    lines = (_header_section(title, label, meta, res) + _stage1_section(res) + _primary_section(res, label)
             + _coverage_section(res) + _bootstrap_section(res) + _audit_section(res) + _ordering_section(res))
    if res_cc is not None:
        lines += _complete_case_section(res_cc, label, cc_note)
    lines += ["## Limits that apply to every number above", ""] + [f"- {t}" for t in LIMITATIONS] + [""]
    return "\n".join(lines)
