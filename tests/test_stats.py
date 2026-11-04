"""The paired statistics, on small tables worked by hand."""

import numpy as np
import pytest

from marketreplay.evaluate import Row
from marketreplay.stats import bootstrap_mean_ci, paired_day_means, summarize


def row(day, parent, strategy, bps, excluded="", forced=0, maker=0, taker=10):
    return Row(day, parent, 1, strategy, excluded, bps, forced, maker, taker, 0, 0, 1.0)


def test_paired_means_average_within_a_day_first():
    rows = [
        row(1, 1, "a", 3.0), row(1, 1, "b", 1.0),
        row(1, 2, "a", 5.0), row(1, 2, "b", 1.0),
        row(2, 3, "a", 0.0), row(2, 3, "b", 1.0),
    ]  # fmt: skip
    days, means = paired_day_means(rows, "a", "b")
    assert days == [1, 2]
    assert means.tolist() == [3.0, -1.0]


def test_an_excluded_parent_is_dropped_for_both_strategies():
    rows = [row(1, 1, "a", 9.0, excluded="halt"), row(1, 1, "b", 1.0), row(1, 2, "a", 2.0), row(1, 2, "b", 1.0)]
    _, means = paired_day_means(rows, "a", "b")
    assert means.tolist() == [1.0]


def test_a_constant_day_value_has_a_zero_width_interval():
    assert bootstrap_mean_ci(np.full(10, 2.5)) == (2.5, 2.5, 2.5)


def test_the_interval_contains_the_mean_and_shrinks_with_more_days():
    rng = np.random.default_rng(1)
    few, many = rng.normal(size=10), rng.normal(size=1000)
    mean, low, high = bootstrap_mean_ci(few)
    assert low < mean < high
    _, low_many, high_many = bootstrap_mean_ci(many)
    assert high_many - low_many < high - low


def test_summaries_keep_completion_apart_from_cost():
    rows = [row(1, 1, "a", 1.0, forced=5, taker=5), row(1, 2, "a", 3.0, forced=0, taker=10)]
    summary = summarize(rows, "a")
    assert summary["mean_bps"] == pytest.approx(2.0)
    assert summary["voluntary_completion_pct"] == pytest.approx(50.0)
    assert summary["forced_lot_pct"] == pytest.approx(25.0)
