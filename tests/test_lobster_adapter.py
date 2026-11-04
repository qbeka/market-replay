"""The LOBSTER adapter on a hand-written window, then a mid-session window with an independent book."""

import csv
import io
from pathlib import Path

import pytest

from marketreplay.adapters.lobster import (
    EMPTY_ASK,
    EMPTY_BID,
    Message,
    compare_to_snapshots,
    convert,
    levels_of,
    parse_time_ns,
    read_messages,
    read_orderbook,
)
from marketreplay.book import ReplayedBook
from marketreplay.events import Event, Kind, Side, from_array
from marketreplay.synthetic import StudyPlan, generate_day

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures"


def fixture():
    return read_messages(FIXTURES / "lobster_message.csv"), read_orderbook(FIXTURES / "lobster_orderbook.csv")


def test_time_parsing_keeps_every_nanosecond():
    assert parse_time_ns("34200.000000001") == 34_200_000_000_001
    assert parse_time_ns("34200.5") == 34_200_500_000_000
    assert parse_time_ns("34200") == 34_200_000_000_000


def test_empty_levels_are_left_out():
    bids, asks = levels_of([100, 5, 99, 4, EMPTY_ASK, 0, EMPTY_BID, 0])
    assert bids == [(99, 4)] and asks == [(100, 5)]


def test_every_row_of_the_hand_window_reconstructs():
    messages, orderbook = fixture()
    report = compare_to_snapshots(convert(messages, orderbook[0]), orderbook)
    assert report.match_pct_by_level == (100.0, 100.0)
    assert report.first_mismatch_row is None
    assert report.book_problems == 0
    # Row 10 deletes an order at a price nobody can see, so it cannot be applied.
    assert report.dropped_unknown == 1


def test_a_hidden_trade_between_ticks_rounds_away_from_a_simulated_order():
    messages = [Message(1, 1, 1, 1, 100000, 1), Message(2, 5, 0, 3, 100050, 1), Message(3, 5, 0, 3, 100050, -1)]
    events = convert(messages, [100100, 1, 100000, 1]).events
    assert [(e.side, e.price) for e in events[-2:]] == [(Side.BID, 1001), (Side.ASK, 1000)]


def test_a_trading_halt_maps_to_halt_events():
    messages, orderbook = fixture()
    kinds = [e.kind for e in convert(messages, orderbook[0]).events]
    assert kinds.count(Kind.HALT_START) == 1 and kinds.count(Kind.HALT_END) == 1


def test_an_auction_cross_is_bracketed_as_a_halt():
    messages = [Message(1, 1, 1, 1, 100, 1), Message(2, 6, -1, 5, 100, 1)]
    kinds = [e.kind for e in convert(messages, [200, 1, 100, 1]).events]
    assert kinds[-2:] == [Kind.HALT_START, Kind.HALT_END]


def test_an_unexpected_direction_is_refused():
    with pytest.raises(ValueError):
        convert([Message(1, 1, 1, 1, 100, 1), Message(2, 1, 2, 1, 100, -2)], [200, 1, 100, 1])


def test_a_corrupted_snapshot_is_reported():
    messages, orderbook = fixture()
    damaged = [list(r) for r in orderbook]
    damaged[4][1] = 999
    report = compare_to_snapshots(convert(messages, orderbook[0]), damaged)
    assert report.first_mismatch_row == 4
    assert report.match_pct_by_level[0] < 100


class NaiveBook:
    """Independent reference: a flat list of orders, re-sorted for every snapshot.

    It shares no code with Book, so agreement between them is real evidence.
    """

    def __init__(self):
        self.orders = {}

    def apply(self, e: Event):
        if e.kind == Kind.ADD:
            self.orders[e.order_id] = [int(e.side), e.price, e.qty]
        elif e.kind in (Kind.CANCEL, Kind.EXECUTE):
            self.orders[e.order_id][2] -= e.qty
            if self.orders[e.order_id][2] == 0:
                del self.orders[e.order_id]
        elif e.kind == Kind.MODIFY:
            self.orders[e.order_id][1:] = [e.price, e.qty]
        elif e.kind == Kind.CLEAR:
            self.orders = {}

    def row(self, levels: int) -> list[int]:
        totals = {1: {}, -1: {}}
        for side, price, qty in self.orders.values():
            totals[side][price] = totals[side].get(price, 0) + qty
        bids = sorted(totals[1].items(), reverse=True)
        asks = sorted(totals[-1].items())
        out = []
        for k in range(levels):
            ask = asks[k] if k < len(asks) else (EMPTY_ASK, 0)
            bid = bids[k] if k < len(bids) else (EMPTY_BID, 0)
            out += [ask[0] * 100 if ask[1] else EMPTY_ASK, ask[1], bid[0] * 100 if bid[1] else EMPTY_BID, bid[1]]
        return out


def export_window(events: list[Event], start: int, stop: int, levels: int) -> tuple[list[Message], list[list[int]]]:
    """Write events as LOBSTER rows, starting mid-session so earlier orders are unknown."""
    naive = NaiveBook()
    messages, rows = [], []
    for index, e in enumerate(events[:stop]):
        naive.apply(e)
        if index < start:
            continue
        code = {Kind.ADD: 1, Kind.EXECUTE: 4, Kind.TRADE_HIDDEN: 5}.get(e.kind)
        if e.kind == Kind.CANCEL:
            code = 3 if e.order_id not in naive.orders else 2
        if code is None:
            continue
        messages.append(Message(e.ts_ns, code, e.order_id, e.qty, e.price * 100, int(e.side)))
        rows.append(naive.row(levels))
    return messages, rows


def plain_events(day: int) -> list[Event]:
    """A session with MODIFY rewritten as delete then add, because LOBSTER has no modify."""
    plan = StudyPlan(days=4, session_s=300)
    records, _ = generate_day(plan, day)
    book = ReplayedBook()
    out = []
    for e in from_array(records):
        if e.kind == Kind.MODIFY:
            side, price = book.book.order_index[e.order_id]
            qty = book.book.order_qty(e.order_id)
            out.append(Event(0, e.ts_ns, Kind.CANCEL, Side(side), e.order_id, price, qty))
            out.append(Event(0, e.ts_ns, Kind.ADD, Side(side), e.order_id, e.price, e.qty))
        else:
            out.append(e)
        book.apply(e)
    return out


def test_a_window_from_the_session_start_matches_an_independent_book_at_every_row():
    events = plain_events(1)
    messages, rows = export_window(events, 0, 6000, levels=10)
    report = compare_to_snapshots(convert(messages, rows[0]), rows)
    assert report.rows == len(rows) > 3000
    assert report.book_problems == 0


def test_a_mid_session_window_stays_exact_at_the_top_of_book_for_a_while():
    events = plain_events(1)
    messages, rows = export_window(events, 3000, 4000, levels=10)
    report = compare_to_snapshots(convert(messages, rows[0]), rows)
    assert report.rows > 500
    assert report.match_pct_by_level[0] >= 80.0
    assert report.match_pct_by_level[0] >= report.match_pct_by_level[9]
