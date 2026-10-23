"""Small synthetic cohorts and fake parts used by several test files.

Everything here is made up. These cohorts check that the code behaves as the protocol says;
they are not, and must never be reported as, results on the real Tanzania cohort.
"""
import numpy as np
import pandas as pd

from pipeline import FEATURE_COLUMNS, GESTATION_VISIT2_COLUMN, TARGET_COLUMN
from selection import ScoreSelector


def toy_df(n=50, seed=0):
    """A plain cohort: next Hb follows first Hb plus noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 5, n) for c in FEATURE_COLUMNS})
    df["hb_visit1_g_dl"] = rng.uniform(6, 14, n)
    df["gestation_visit1_weeks"] = rng.uniform(5, 20, n)
    df[GESTATION_VISIT2_COLUMN] = df["gestation_visit1_weeks"] + rng.uniform(1, 5, n)
    df[TARGET_COLUMN] = df["hb_visit1_g_dl"] + rng.normal(0, 1, n)
    return df


def toy_cohort(n=400, seed=0):
    """A cohort with record IDs, some missing inputs, and errors that grow away from Hb 11,
    so that Stage 2 has something to learn."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(50, 5, n) for c in FEATURE_COLUMNS})
    df["hb_visit1_g_dl"] = rng.uniform(6, 14, n)
    df["gestation_visit1_weeks"] = rng.integers(5, 20, n).astype(float)
    df[GESTATION_VISIT2_COLUMN] = df["gestation_visit1_weeks"] + rng.integers(1, 9, n)
    noise = rng.normal(0, 1, n) * (0.3 + 0.2 * np.abs(df["hb_visit1_g_dl"] - 11))
    df[TARGET_COLUMN] = 0.7 * df["hb_visit1_g_dl"] + 3.3 + noise
    df["_source_row"] = np.arange(1000, 1000 + n)
    df.loc[rng.choice(n, n // 10, replace=False), "respiratory_rate_visit1"] = np.nan
    return df


def toy_table(n=60, seed=0):
    """The per-woman table the selectors read (record ID and errors only)."""
    rng = np.random.default_rng(seed)
    err = rng.uniform(0.05, 2.0, n)
    return pd.DataFrame({"record_id": np.arange(n)[::-1] + 500, "abs_error": err,
                         "signed_error": err * rng.choice([-1, 1], n)})


class FixedScoreSelector(ScoreSelector):
    """A selector written for a test: it simply carries scores it was given.

    It plugs into the evaluation exactly like the protocol's selectors, which shows the
    evaluation code does not need to change when a new way of choosing is added.
    """

    def __init__(self, name, scores):
        self.name = name
        self.label = name
        self._scores = np.asarray(scores, dtype=float)

    def score(self, tab):
        assert len(tab) == len(self._scores)
        return self._scores
