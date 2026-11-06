"""Counterexamples for three engine biases. Each one failed before its fix."""

from dataclasses import replace

import pytest

from marketreplay.adapters.lobster import Message, convert
from marketreplay.book import Book, BookError, PriorityRules
from marketreplay.compiled import by_run, reference_lines, run_exec, write_tape
from marketreplay.events import Event, Kind, Side, to_array
from marketreplay.execution import ExecConfig, Intent, Parent, ScriptedStrategy, run_session
from marketreplay.stats import bootstrap_mean_ci

import numpy as np

CONFIG = ExecConfig(
    obs_latency_ns=30, order_latency_ns=5, cancel_latency_ns=5, taker_fee_mticks=300, maker_fee_mticks=-100,
    decision_interval_ns=10, snapshot_interval_ns=10,
)  # fmt: skip


def opening() -> list[Event]:
    return [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.ADD, Side.BID, 1, 99, 10),
        Event(3, 0, Kind.ADD, Side.ASK, 2, 101, 10),
        Event(4, 0, Kind.SNAPSHOT_END),
    ]


class Recorder:
    name = "recorder"

    def __init__(self, script):
        self.script = script
        self.views = []

    def decide(self, view):
        self.views.append(view)
        return self.script.get(view.now_ns)


def test_a_hidden_print_does_not_burn_visible_queue_ahead():
    feed = opening() + [
        Event(5, 40, Kind.TRADE_HIDDEN, Side.BID, 0, 99, 10),
        Event(6, 50, Kind.EXECUTE, Side.BID, 1, 99, 4),
        Event(7, 500, Kind.ADD, Side.BID, 9, 98, 1),
    ]
    [run] = run_session(feed, [(Parent(1, Side.BID, 5, 30, 100), ScriptedStrategy({30: Intent.rest(5)}))], CONFIG)
    assert [f.role.name for f in run.ledger.fills] == ["FORCED"]


def test_a_hidden_print_fills_a_child_with_nothing_ahead():
    feed = opening() + [
        Event(5, 35, Kind.EXECUTE, Side.BID, 1, 99, 10),
        Event(6, 36, Kind.ADD, Side.BID, 3, 98, 5),
        Event(7, 40, Kind.TRADE_HIDDEN, Side.BID, 0, 99, 2),
        Event(8, 500, Kind.ADD, Side.BID, 9, 97, 1),
    ]
    [run] = run_session(feed, [(Parent(1, Side.BID, 2, 30, 100), ScriptedStrategy({30: Intent.rest(2)}))], CONFIG)
    assert [(f.role.name, f.qty) for f in run.ledger.fills][0] == ("MAKER", 2)


def test_own_order_state_is_as_late_as_the_observation_clock():
    feed = opening() + [
        Event(5, 40, Kind.EXECUTE, Side.BID, 1, 99, 10),
        Event(6, 45, Kind.ADD, Side.BID, 3, 98, 5),
        Event(7, 46, Kind.EXECUTE, Side.BID, 3, 98, 5),  # Trades through and fills the child at 46.
        Event(8, 47, Kind.ADD, Side.BID, 4, 98, 5),
        Event(9, 500, Kind.ADD, Side.BID, 9, 97, 1),
    ]
    spy = Recorder({30: Intent.rest(5)})
    run_session(feed, [(Parent(1, Side.BID, 5, 30, 200), spy)], CONFIG)
    views = {v.now_ns: v for v in spy.views}
    assert views[70].filled_seen == 0 and views[70].rest_price == 99 and views[70].rest_open == 5
    assert views[80].filled_seen == 5 and views[80].rest_price is None


def test_the_arrival_mid_a_strategy_sees_is_delayed():
    feed = opening() + [
        Event(5, 90, Kind.ADD, Side.BID, 3, 100, 1),  # True mid moves to 100.5 just before arrival.
        Event(6, 500, Kind.ADD, Side.BID, 9, 97, 1),
    ]
    spy = Recorder({})
    [run] = run_session(feed, [(Parent(1, Side.BID, 5, 100, 200), spy)], CONFIG)
    assert run.ledger.arrival_mid_x2 == 201  # The benchmark uses the true market.
    assert spy.views[0].arrival_mid_seen == 100.0  # A strategy only sees the delayed one.


def test_the_deadline_prices_from_the_last_far_side_price_when_it_empties():
    feed = opening() + [
        Event(5, 10, Kind.CANCEL, Side.ASK, 2, 101, 10),
        Event(6, 20, Kind.ADD, Side.ASK, 3, 200, 10),
        Event(7, 150, Kind.CANCEL, Side.ASK, 3, 200, 10),
        Event(8, 500, Kind.ADD, Side.BID, 9, 97, 1),
    ]
    config = replace(CONFIG, unfilled_extra_ticks=10)
    [run] = run_session(feed, [(Parent(1, Side.BID, 5, 100, 200), ScriptedStrategy({}))], config)
    assert [(f.price, f.qty) for f in run.ledger.fills] == [(210, 5)]


def test_the_compiled_replay_follows_non_default_priority_rules(tmp_path):
    rules = PriorityRules(size_increase_loses_priority=False, size_decrease_loses_priority=True)
    config = replace(CONFIG, priority=rules, record_trace=True)
    feed = opening() + [
        Event(5, 40, Kind.ADD, Side.BID, 3, 99, 5),
        Event(6, 45, Kind.MODIFY, Side.BID, 1, 99, 12),
        Event(7, 50, Kind.EXECUTE, Side.BID, 1, 99, 12),
        Event(8, 60, Kind.MODIFY, Side.BID, 3, 99, 4),
        Event(9, 70, Kind.EXECUTE, Side.BID, 3, 99, 4),
        Event(10, 500, Kind.ADD, Side.BID, 9, 97, 1),
    ]
    runs = run_session(feed, [(Parent(1, Side.BID, 5, 30, 100), ScriptedStrategy({30: Intent.rest(5)}))], config)
    path, tape = tmp_path / "events.bin", tmp_path / "tape.txt"
    path.write_bytes(to_array(feed).tobytes())
    write_tape(runs, tape)
    assert run_exec(path, tape, config).lines == by_run(reference_lines(runs))


def test_the_snapshot_grid_stays_on_absolute_time_when_the_feed_starts_late():
    feed = [Event(e.seq, e.ts_ns + 123, e.kind, e.side, e.order_id, e.price, e.qty) for e in opening()]
    feed.append(Event(5, 2000, Kind.ADD, Side.BID, 9, 97, 1))
    config = replace(CONFIG, obs_latency_ns=3)
    spy = Recorder({})
    run_session(feed, [(Parent(1, Side.BID, 5, 500, 600), spy)], config)
    assert spy.views and all(v.market.ts_ns == v.now_ns - 3 for v in spy.views)


def test_a_modify_on_the_wrong_side_is_refused():
    book = Book()
    book.apply(Event(1, 0, Kind.ADD, Side.BID, 1, 100, 5))
    with pytest.raises(BookError):
        book.apply(Event(2, 0, Kind.MODIFY, Side.ASK, 1, 100, 4))


def test_a_depth_of_zero_levels_is_empty():
    book = Book()
    book.apply(Event(1, 0, Kind.ADD, Side.BID, 1, 100, 5))
    assert book.depth(Side.BID, 0) == []


def test_a_single_day_has_no_bootstrap_interval():
    mean, low, high = bootstrap_mean_ci(np.array([1.5]))
    assert mean == 1.5 and np.isnan(low) and np.isnan(high)


def test_lobster_events_carry_a_receive_time_equal_to_the_event_time():
    events = convert([Message(5, 1, 1, 1, 10000, 1), Message(6, 1, 2, 1, 10000, 1)], [10100, 1, 10000, 1]).events
    assert all(e.recv_ns == e.ts_ns for e in events)
