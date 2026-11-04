"""Label logic on small feeds worked by hand."""

import pytest

from marketreplay.events import Event, Kind, Side
from marketreplay.labels import kaplan_meier, observed_outcomes
from marketreplay.synthetic import NS_PER_S

S = NS_PER_S


def test_kaplan_meier_handles_censoring_by_hand():
    # Times 1(event) 2(censored) 3(event) 4(censored).
    # S after t=1: 3/4. After t=3, 1 of 2 at risk: 3/4 x 1/2 = 3/8. Fill = 5/8.
    assert kaplan_meier([(1, True), (2, False), (3, True), (4, False)], 10) == pytest.approx(5 / 8)


def test_kaplan_meier_with_no_censoring_is_the_plain_rate():
    assert kaplan_meier([(1, True), (2, False), (9, False), (3, True)], 5) == pytest.approx(0.5)


def test_observed_outcomes_counts_each_fate():
    events = [
        Event(1, 0, Kind.ADD, Side.BID, 1, 100, 5),  # Not joining: no prior best on this side.
        Event(2, 0, Kind.ADD, Side.ASK, 2, 102, 5),
        Event(3, 1 * S, Kind.ADD, Side.BID, 3, 100, 2),  # Joins. Executes at 3 s.
        Event(4, 2 * S, Kind.ADD, Side.BID, 4, 100, 2),  # Joins. Cancels at 4 s.
        Event(5, 3 * S, Kind.ADD, Side.BID, 5, 100, 2),  # Joins. Still there at the end.
        Event(6, 3 * S, Kind.EXECUTE, Side.BID, 3, 100, 2),
        Event(7, 4 * S, Kind.CANCEL, Side.BID, 4, 100, 2),
        Event(8, 60 * S, Kind.ADD, Side.BID, 6, 99, 1),
    ]
    outcome = observed_outcomes(events, horizon_s=15)
    assert (outcome.orders, outcome.fully_executed, outcome.cancelled_first, outcome.still_resting) == (3, 1, 1, 1)
