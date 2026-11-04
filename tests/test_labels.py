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
    # Nothing is censored before the horizon, so this is simply 2 of 4.
    assert kaplan_meier([(1, True), (3, True), (9, False), (9, False)], 5) == pytest.approx(0.5)


def opening():
    return [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.ADD, Side.BID, 1, 100, 5),
        Event(3, 0, Kind.ADD, Side.ASK, 2, 102, 5),
        Event(4, 0, Kind.SNAPSHOT_END),
    ]


def test_observed_outcomes_counts_each_fate():
    events = opening() + [
        Event(5, 1 * S, Kind.ADD, Side.BID, 3, 100, 2),  # Joins. Executes at 3 s.
        Event(6, 2 * S, Kind.ADD, Side.BID, 4, 100, 2),  # Joins. Cancels at 4 s.
        Event(7, 3 * S, Kind.ADD, Side.BID, 5, 100, 2),  # Joins. Still there at the end.
        Event(8, 3 * S, Kind.EXECUTE, Side.BID, 3, 100, 2),
        Event(9, 4 * S, Kind.CANCEL, Side.BID, 4, 100, 2),
        Event(10, 60 * S, Kind.ADD, Side.BID, 6, 99, 1),
    ]
    outcome = observed_outcomes(events, horizon_s=15)
    assert (outcome.orders, outcome.fully_executed, outcome.cancelled_first, outcome.still_resting) == (3, 1, 1, 1)


def test_a_gap_censors_tracked_orders_instead_of_crashing():
    events = opening() + [
        Event(5, 1 * S, Kind.ADD, Side.BID, 3, 100, 2),
        Event(9, 2 * S, Kind.EXECUTE, Side.BID, 77, 100, 1),  # Gap, and an id the book never saw.
        Event(10, 3 * S, Kind.ADD, Side.BID, 8, 100, 1),
    ]
    outcome = observed_outcomes(events, horizon_s=15)
    assert (outcome.orders, outcome.fully_executed, outcome.cancelled_first) == (1, 0, 1)
