"""
The ways of choosing which forecasts to keep ("selectors").

A selector gives every woman a score. A LOWER score means a MORE reliable forecast. At a
given coverage the forecasts with the lowest scores are kept (metrics.keep_mask: exact
coverage, ties by record ID). Every selector is scored on the same Stage 1 forecasts, so
the comparison is paired.

    Selector (abstract: name, label)
    |-- ScoreSelector (abstract: score)         ranks women by one number
    |   |-- LearnedStage2Selector               the Stage 2 estimated error
    |   |-- TreeDisagreementSelector            spread of the forest's trees
    |   |-- SimpleRuleSelector                  missing inputs, then distance from the median
    |   `-- OracleSelector                      the real error (reference only, unusable)
    `-- RandomSelector                          repeated random draws, summarised

The evaluation treats every selector the same way (polymorphism): it asks each one to
summarise itself at a coverage level. A new way of choosing is added by writing one subclass,
without changing the evaluation (open for extension, closed for modification).

The score methods read the per-woman table built by evaluation.evaluate_partition
(columns stage2_score, tree_disagreement, n_missing_inputs, first_hb, abs_error, record_id).

This module contains no execution code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from metrics import keep_mask, n_keep, root_mean_square


class Selector(ABC):
    """Anything that can be compared with the others at a given coverage."""

    name: str = ""
    label: str = ""

    @abstractmethod
    def summarise(self, tab: pd.DataFrame, coverage: float) -> dict:
        """Error of the forecasts this selector keeps at `coverage`: n_kept, mae, rmse, mean_signed_error."""


class ScoreSelector(Selector):
    """A selector that gives each woman a score; the lowest scores are kept."""

    @abstractmethod
    def score(self, tab: pd.DataFrame) -> np.ndarray:
        """One score per row of the per-woman table. Lower means more reliable."""

    def keep_mask(self, tab: pd.DataFrame, coverage: float) -> np.ndarray:
        """True for the forecasts kept at this coverage (exact-coverage rule, ties by record ID)."""
        return keep_mask(self.score(tab), coverage, tab["record_id"].to_numpy())

    def summarise(self, tab, coverage):
        """Keep the lowest scores at this coverage and report n_kept, mae, rmse and mean_signed_error."""
        kept = self.keep_mask(tab, coverage)
        abs_err = tab["abs_error"].to_numpy()[kept]
        return dict(n_kept=int(kept.sum()), mae=float(abs_err.mean()), rmse=root_mean_square(abs_err),
                    mean_signed_error=float(tab["signed_error"].to_numpy()[kept].mean()))


class LearnedStage2Selector(ScoreSelector):
    """The learned Stage 2 score: the estimated absolute error of the forecast."""

    name = "learned"
    label = "Learned Stage 2 score"

    def score(self, tab):
        """The Stage 2 estimate of the absolute error."""
        return tab["stage2_score"].to_numpy(dtype=float)


class TreeDisagreementSelector(ScoreSelector):
    """Standard deviation of the individual trees' forecasts: free, because the forest is already there."""

    name = "tree_disagreement"
    label = "Tree disagreement"

    def score(self, tab):
        """The standard deviation of the forest's trees."""
        return tab["tree_disagreement"].to_numpy(dtype=float)


class SimpleRuleSelector(ScoreSelector):
    """The simple rule a clinic could apply by hand, fixed before any results were seen.

    More missing inputs means a less reliable forecast. Ties are broken by how far the
    first haemoglobin is from the training median (farther = less reliable). Written as
    one number so it can be ranked like the other scores: the distance is divided by 100
    so it can only separate women who have the same number of missing inputs.
    """

    name = "simple_rule"
    label = "Simple rule"

    def __init__(self, train_hb_median: float):
        """`train_hb_median` is the median first Hb of the training women."""
        self.train_hb_median = float(train_hb_median)

    def score(self, tab):
        """The number of missing inputs, plus the distance of first Hb from the training median divided by 100."""
        n_missing = tab["n_missing_inputs"].to_numpy(dtype=float)
        distance = np.abs(tab["first_hb"].to_numpy(dtype=float) - self.train_hb_median)
        assert distance.max() < 100, "distance would no longer act only as a tie-break"
        return n_missing + distance / 100.0


class OracleSelector(ScoreSelector):
    """Keeps the forecasts that really were closest. Not a usable selector: it needs the answer.

    Shown only to mark how much room any selector has.
    """

    name = "oracle"
    label = "Oracle (reference only)"

    def score(self, tab):
        """The real absolute error."""
        return tab["abs_error"].to_numpy(dtype=float)


class RandomSelector(Selector):
    """Random selection, repeated with fixed seeds (one stream per coverage level).

    It has no score, so it is not a ScoreSelector: it reports the mean and the spread of the
    error of the kept forecasts over many random draws.
    """

    name = "random"
    label = "Random selection"

    def __init__(self, repeats: int, seed: int):
        """`repeats` random draws per coverage level, from fixed streams made from `seed`."""
        self.repeats = repeats
        self.seed = seed

    def summarise(self, tab: pd.DataFrame, coverage: float) -> dict:
        """Mean MAE, RMSE and signed error over the repeats, and the 2.5th to 97.5th percentile of MAE."""
        abs_err = tab["abs_error"].to_numpy()
        signed = tab["signed_error"].to_numpy()
        n = len(tab)
        k = n_keep(n, coverage)
        rng = np.random.default_rng([self.seed, int(round(coverage * 100))])
        maes, rmses, sgn = [], [], []
        for _ in range(self.repeats):
            idx = rng.choice(n, size=k, replace=False)
            maes.append(abs_err[idx].mean())
            rmses.append(root_mean_square(abs_err[idx]))
            sgn.append(signed[idx].mean())
        return dict(n_kept=k, mae=float(np.mean(maes)), rmse=float(np.mean(rmses)),
                    mean_signed_error=float(np.mean(sgn)),
                    mae_p2_5=float(np.percentile(maes, 2.5)),
                    mae_p97_5=float(np.percentile(maes, 97.5)))


# ---------------------------------------------------------------------------
# The standard line-up used by the protocol
# ---------------------------------------------------------------------------
SELECTOR_NAMES: List[str] = [LearnedStage2Selector.name, TreeDisagreementSelector.name, SimpleRuleSelector.name]
SELECTOR_LABELS: Dict[str, str] = {c.name: c.label for c in (
    LearnedStage2Selector, TreeDisagreementSelector, SimpleRuleSelector, RandomSelector, OracleSelector)}


def default_selectors(train_hb_median: float) -> List[ScoreSelector]:
    """The three usable selectors of the protocol, in reporting order."""
    return [LearnedStage2Selector(), TreeDisagreementSelector(), SimpleRuleSelector(train_hb_median)]
