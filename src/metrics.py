"""
Error metrics and the exact-coverage selection helpers shared by every selector.

Pure functions with no state, used by the Stage 1 code, the selector comparison and
the run scripts. Nothing here reads a file or fits a model.

A LOWER score always means a MORE reliable forecast.
"""

import numpy as np


def mean_absolute_error(y_true, y_pred) -> float:
    """Mean of |true - predicted|."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Square root of the mean squared difference."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mean_signed_error(y_true, y_pred) -> float:
    """Mean of (predicted - true). Positive means the forecasts ran high."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def root_mean_square(values) -> float:
    """Square root of the mean of the squares (the RMSE of a vector of errors)."""
    return float(np.sqrt(np.mean(np.square(values))))


def n_keep(n_total: int, coverage: float) -> int:
    """How many forecasts an exact-coverage rule keeps: round(coverage x N), halves rounded up.

    For 80% of the 399 locked-test women this is 319; for 80% of the 402 development
    women it is 322. (The extra round() guards against floating-point noise such as
    3.4999999999999996 standing in for 3.5.)
    """
    return int(np.floor(round(n_total * coverage, 9) + 0.5))


def keep_mask(scores, coverage: float, record_ids=None) -> np.ndarray:
    """True for the forecasts kept at this coverage: the n lowest scores.

    A LOWER score means a MORE reliable forecast. Ties are broken by record ID,
    ascending, so the result never depends on a sorting library's tie behaviour.
    Cost: one sort, O(n log n).
    """
    scores = np.asarray(scores, dtype=float)
    ids = np.arange(len(scores)) if record_ids is None else np.asarray(record_ids)
    order = np.lexsort((ids, scores))  # last key is the primary one: score, then record ID
    mask = np.zeros(len(scores), dtype=bool)
    mask[order[:n_keep(len(scores), coverage)]] = True
    return mask


def mae_at_coverage(errors, scores, coverage: float, record_ids=None) -> float:
    """MAE of the retained forecasts when the `coverage` fraction with the lowest scores is kept."""
    errors = np.asarray(errors, dtype=float)
    kept = keep_mask(scores, coverage, record_ids)
    return float(np.mean(errors[kept])) if kept.any() else float("nan")


def paired_bootstrap_difference(values_a, values_b, groups=None, n_boot: int = 5000, seed: int = 42):
    """Mean of (a - b) over paired cases with a 95% percentile interval.

    Resamples whole identical-vector groups with replacement (effectively single
    records, because only one pair of identical records exists). Used to ask whether
    one model's per-woman absolute errors are lower than another's on the same women.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    d = a - b
    g = np.arange(len(d)) if groups is None else np.asarray(groups)
    members = [np.where(g == k)[0] for k in np.unique(g)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(members), size=len(members))
        means[i] = d[np.concatenate([members[j] for j in pick])].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"difference": float(d.mean()), "ci95_low": float(lo), "ci95_high": float(hi)}
