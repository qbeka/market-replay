"""Assert every reconstructed state of the hand-worked feed."""

import csv
import json
from pathlib import Path

import pytest

from marketreplay.book import Book, BookError, PriorityRules, ReplayedBook
from marketreplay.events import Event, Kind, Side, from_array, to_array

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures"


def load_trace(name="hand_trace.csv") -> list[Event]:
    rows = (line for line in open(FIXTURES / name) if not line.startswith("#"))
    return [
        Event(int(r[0]), int(r[1]), Kind[r[2]], Side[r[3]], int(r[4]), int(r[5]), int(r[6]))
        for r in csv.reader(rows)
    ]


def test_every_state_of_the_hand_trace():
    expected = json.loads((FIXTURES / "hand_trace_expected.json").read_text())
    events = load_trace()
    assert len(events) == len(expected["states"])
    replay = ReplayedBook()
    replay.valid = True  # The fixture starts from an empty, known book.
    replay.started = True
    replay.invalid_since_ns = None
    for event, state in zip(events, expected["states"]):
        replay.apply(event)
        assert replay.valid == state["valid"], f"line {state['line']}"
        if not state["valid"]:
            continue
        book = replay.book
        assert [list(row) for row in book.depth(Side.BID, 5)] == state["bids"], f"line {state['line']}"
        assert [list(row) for row in book.depth(Side.ASK, 5)] == state["asks"], f"line {state['line']}"
        best = book.best(Side.BID)
        queue = book.queue(Side.BID, best) if best is not None else []
        assert queue == state["best_bid_queue"], f"line {state['line']}"
    assert [list(i) for i in replay.invalid_intervals_ns] == expected["invalid_intervals_ns"]


def test_a_gap_stops_book_updates_until_resynchronized():
    replay = ReplayedBook()
    for e in [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.ADD, Side.BID, 1, 100, 5),
        Event(3, 0, Kind.SNAPSHOT_END),
        Event(5, 10, Kind.ADD, Side.BID, 2, 100, 5),
    ]:
        replay.apply(e)
    assert not replay.valid
    assert replay.problems == ["gap: seq 3 then 5"]
    assert 2 not in replay.book.order_index


def test_an_impossible_event_invalidates_instead_of_guessing():
    replay = ReplayedBook()
    for e in [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.SNAPSHOT_END),
        Event(3, 5, Kind.CANCEL, Side.BID, 99, 100, 1),
    ]:
        replay.apply(e)
    assert not replay.valid
    assert "unknown" in replay.problems[0]


def test_venue_rule_can_let_a_size_increase_keep_priority():
    rules = PriorityRules(size_increase_loses_priority=False)
    book = Book(rules)
    book.apply(Event(1, 0, Kind.ADD, Side.BID, 1, 100, 5))
    book.apply(Event(2, 0, Kind.ADD, Side.BID, 2, 100, 5))
    book.apply(Event(3, 0, Kind.MODIFY, Side.BID, 1, 100, 9))
    assert book.queue(Side.BID, 100) == [1, 2]


def test_default_rule_sends_a_size_increase_to_the_back():
    book = Book()
    book.apply(Event(1, 0, Kind.ADD, Side.BID, 1, 100, 5))
    book.apply(Event(2, 0, Kind.ADD, Side.BID, 2, 100, 5))
    book.apply(Event(3, 0, Kind.MODIFY, Side.BID, 1, 100, 9))
    assert book.queue(Side.BID, 100) == [2, 1]


def test_a_crossing_add_is_rejected():
    book = Book()
    book.apply(Event(1, 0, Kind.ADD, Side.ASK, 1, 100, 5))
    with pytest.raises(BookError):
        book.apply(Event(2, 0, Kind.ADD, Side.BID, 2, 100, 5))


def test_binary_round_trip_preserves_every_event():
    events = load_trace()
    assert from_array(to_array(events)) == events
