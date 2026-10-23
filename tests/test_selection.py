"""
Tests for src/selection.py: the selector classes and how they are compared.
"""
import numpy as np
import pandas as pd
import pytest

from helpers import FixedScoreSelector, toy_table
from metrics import n_keep
from selection import (
    SELECTOR_LABELS, SELECTOR_NAMES, LearnedStage2Selector, OracleSelector, RandomSelector, ScoreSelector,
    Selector, SimpleRuleSelector, TreeDisagreementSelector, default_selectors,
)


def _table(**columns):
    n = len(next(iter(columns.values())))
    base = {"record_id": np.arange(n), "abs_error": np.linspace(0.1, 1.0, n)}
    base.update(columns)
    return pd.DataFrame(base)


def test_selectors_form_one_hierarchy():
    for cls in (LearnedStage2Selector, TreeDisagreementSelector, SimpleRuleSelector, OracleSelector):
        assert issubclass(cls, ScoreSelector) and issubclass(cls, Selector)
    assert issubclass(RandomSelector, Selector) and not issubclass(RandomSelector, ScoreSelector)
    with pytest.raises(TypeError):
        ScoreSelector()    # abstract: score is not implemented


def test_the_protocol_line_up_is_learned_tree_disagreement_simple_rule():
    selectors = default_selectors(train_hb_median=11.0)
    assert [s.name for s in selectors] == SELECTOR_NAMES == ["learned", "tree_disagreement", "simple_rule"]
    assert all(isinstance(s, ScoreSelector) for s in selectors)
    assert set(SELECTOR_LABELS) >= {"learned", "tree_disagreement", "simple_rule", "random", "oracle"}


def test_each_selector_reads_its_own_column_of_the_table():
    tab = _table(stage2_score=[0.3, 0.1, 0.2], tree_disagreement=[0.9, 0.8, 0.7])
    assert list(LearnedStage2Selector().score(tab)) == [0.3, 0.1, 0.2]
    assert list(TreeDisagreementSelector().score(tab)) == [0.9, 0.8, 0.7]
    assert np.allclose(OracleSelector().score(tab), tab["abs_error"])


def test_simple_rule_orders_by_missing_inputs_then_distance_from_the_median():
    tab = _table(n_missing_inputs=[0, 0, 1, 0, 2, 0], first_hb=[11.0, 12.0, 11.0, 9.0, 11.0, 14.0])
    score = SimpleRuleSelector(train_hb_median=11.0).score(tab)
    # most reliable first: 0 missing and on the median, then 0 missing and 1, 2, 3 away, then 1 missing, then 2 missing
    assert list(np.argsort(score, kind="stable")) == [0, 1, 3, 5, 2, 4]


def test_simple_rule_distance_only_breaks_ties_between_equal_missing_counts():
    tab = _table(n_missing_inputs=[0, 1], first_hb=[19.9, 11.0])
    score = SimpleRuleSelector(11.0).score(tab)
    assert score[0] < score[1]     # one missing input is always worse than a large distance


def test_keep_mask_keeps_the_exact_number_with_the_lowest_scores():
    tab = toy_table(60)
    scores = np.random.default_rng(1).normal(size=60)
    selector = FixedScoreSelector("fixed", scores)
    for coverage in (1.0, 0.9, 0.8, 0.7):
        mask = selector.keep_mask(tab, coverage)
        assert mask.sum() == n_keep(60, coverage)
        assert scores[mask].max() <= scores[~mask].min() if (~mask).any() else True


def test_ties_between_equal_scores_go_to_the_lower_record_id():
    tab = _table(stage2_score=[1.0, 1.0, 1.0, 1.0])
    tab["record_id"] = [40, 10, 30, 20]
    kept = LearnedStage2Selector().keep_mask(tab, 0.5)
    assert set(tab.loc[kept, "record_id"]) == {10, 20}


def test_the_oracle_is_never_beaten_by_another_selector():
    tab = toy_table(80)
    oracle_mae = tab.loc[OracleSelector().keep_mask(tab, 0.8), "abs_error"].mean()
    rng = np.random.default_rng(2)
    for i in range(5):
        other = FixedScoreSelector(f"r{i}", rng.normal(size=80))
        assert oracle_mae <= tab.loc[other.keep_mask(tab, 0.8), "abs_error"].mean() + 1e-12


def test_random_selector_is_repeatable_and_centred_on_the_overall_error():
    tab = toy_table(200)
    summary = RandomSelector(repeats=200, seed=42).summarise(tab, 0.8)
    assert summary == RandomSelector(repeats=200, seed=42).summarise(tab, 0.8)
    assert summary["n_kept"] == n_keep(200, 0.8)
    assert summary["mae_p2_5"] <= summary["mae"] <= summary["mae_p97_5"]
    assert abs(summary["mae"] - tab["abs_error"].mean()) < 0.05
    assert summary != RandomSelector(repeats=200, seed=43).summarise(tab, 0.8)
