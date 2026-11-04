"""LOBSTER message and orderbook files to normalized events. Adapter version 1.

Format, from LOBSTER's documentation as quoted by two independent secondary sources
(the site's own page did not render here, see docs/data_rights.md):

    message file    time, type, order id, size, price, direction
    orderbook file  ask price 1, ask size 1, bid price 1, bid size 1, then level 2, ...

    time            seconds after midnight, up to nanosecond precision
    price           USD x 10,000
    direction       1 buy limit order, -1 sell limit order. For an execution it names
                    the resting order, so -1 is a buyer-initiated trade.
    type            1 new order, 2 partial cancel, 3 full delete, 4 visible execution,
                    5 hidden execution, 6 auction cross, 7 trading halt
    type 7 price    -1 halt, 0 quoting resumes, 1 trading resumes
    empty level     price -9999999999 (bid) or 9999999999 (ask), size 0
    row k           of the orderbook file is the book after row k of the message file

The hard part is that a window starts mid-session. Orders placed before the first
message are invisible, and the file only shows the top levels. So the first orderbook
row seeds one anonymous order per level, and a later event on an unknown id draws from
that price's anonymous order. An event on an unknown id beyond the seeded depth cannot
be applied, so it is counted and dropped, and reconstruction reports its effect.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

from ..events import Event, Kind, Side

ADAPTER_VERSION = 1
EMPTY_BID = -9999999999
EMPTY_ASK = 9999999999
NS_PER_S = 1_000_000_000


@dataclass(frozen=True)
class Units:
    """USD x 10,000 per tick. The default is one US equity tick of $0.01."""

    price_per_tick: int = 100
    shares_per_lot: int = 1


@dataclass(frozen=True)
class Message:
    time_ns: int
    event_type: int
    order_id: int
    size: int
    price: int
    direction: int


def parse_time_ns(text: str) -> int:
    """Exact integer nanoseconds. A float would lose the last digits of a 15-digit time."""
    whole, _, fraction = text.partition(".")
    return int(whole) * NS_PER_S + int(fraction.ljust(9, "0")[:9] or 0)


def read_messages(path: Path) -> list[Message]:
    with open(path, newline="") as handle:
        return [Message(parse_time_ns(r[0]), int(r[1]), int(r[2]), int(r[3]), int(r[4]), int(r[5])) for r in csv.reader(handle)]


def read_orderbook(path: Path) -> list[list[int]]:
    with open(path, newline="") as handle:
        return [[int(float(v)) for v in row] for row in csv.reader(handle)]


def levels_of(row: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """(bids, asks), best first, in raw units, leaving out empty levels."""
    bids, asks = [], []
    for i in range(0, len(row), 4):
        ask_price, ask_size, bid_price, bid_size = row[i : i + 4]
        if ask_price != EMPTY_ASK and ask_size > 0:
            asks.append((ask_price, ask_size))
        if bid_price != EMPTY_BID and bid_size > 0:
            bids.append((bid_price, bid_size))
    return bids, asks


@dataclass
class Conversion:
    events: list[Event]
    dropped_unknown: int  # Events on orders that predate the window and sit beyond the seed.
    rows_of_events: list[int]  # For each event, the orderbook row it should agree with.


def _side(direction: int) -> Side:
    if direction == 1:
        return Side.BID
    if direction == -1:
        return Side.ASK
    raise ValueError(f"direction {direction} is not 1 or -1")


def _ticks(price: int, units: Units) -> int:
    if price % units.price_per_tick:
        raise ValueError(f"price {price} is not a whole number of ticks")
    return price // units.price_per_tick


def _hidden_ticks(price: int, side: Side, units: Units) -> int:
    """Hidden trades can print between ticks, at the midpoint, so they need rounding.

    Round away from any simulated order on that side. A resting bid rounds up and a
    resting ask rounds down, so a half-tick print can never look like a trade through.
    """
    whole, part = divmod(price, units.price_per_tick)
    return whole + 1 if part and side == Side.BID else whole


def convert(messages: list[Message], first_row: list[int], units: Units = Units()) -> Conversion:
    """Convert a window. The first message is already inside first_row, so it seeds, not replays."""
    seed_ts = messages[0].time_ns
    events = [Event(0, seed_ts, Kind.CLEAR)]
    rows = [0]
    anonymous: dict[tuple[Side, int], int] = {}  # (side, price) -> anonymous order id
    remaining: dict[int, int] = {}  # Anonymous id -> lots left.
    bids, asks = levels_of(first_row)
    next_anonymous = -1
    for side, levels in ((Side.BID, bids), (Side.ASK, asks)):
        for raw_price, raw_size in levels:
            price = _ticks(raw_price, units)
            lots = raw_size // units.shares_per_lot
            anonymous[(side, price)] = next_anonymous
            remaining[next_anonymous] = lots
            events.append(Event(0, seed_ts, Kind.ADD, side, next_anonymous, price, lots))
            rows.append(0)
            next_anonymous -= 1
    events.append(Event(0, seed_ts, Kind.SNAPSHOT_END))
    rows.append(0)

    live: set[int] = set()
    dropped = 0

    def push(row: int, *fields) -> None:
        events.append(Event(0, *fields))
        rows.append(row)

    for row, m in enumerate(messages[1:], start=1):
        if m.event_type == 1:
            live.add(m.order_id)
            push(row, m.time_ns, Kind.ADD, _side(m.direction), m.order_id, _ticks(m.price, units), m.size // units.shares_per_lot)
        elif m.event_type in (2, 3, 4):
            side, price, lots = _side(m.direction), _ticks(m.price, units), m.size // units.shares_per_lot
            order_id = m.order_id
            if order_id not in live:
                order_id = anonymous.get((side, price))
                if order_id is None or remaining.get(order_id, 0) < lots:
                    dropped += 1
                    continue
                remaining[order_id] -= lots
            kind = Kind.EXECUTE if m.event_type == 4 else Kind.CANCEL
            push(row, m.time_ns, kind, side, order_id, price, lots)
        elif m.event_type == 5:
            side = _side(m.direction)
            push(row, m.time_ns, Kind.TRADE_HIDDEN, side, 0, _hidden_ticks(m.price, side, units), m.size // units.shares_per_lot)
        elif m.event_type == 6:
            # An auction cross. Continuous matching does not apply, so bracket it as a halt.
            push(row, m.time_ns, Kind.HALT_START)
            push(row, m.time_ns, Kind.HALT_END)
        elif m.event_type == 7:
            if m.price == -1:
                push(row, m.time_ns, Kind.HALT_START)
            elif m.price == 1:
                push(row, m.time_ns, Kind.HALT_END)
        else:
            raise ValueError(f"unknown LOBSTER event type {m.event_type}")
    numbered = [
        Event(i + 1, e.ts_ns, e.kind, e.side, e.order_id, e.price, e.qty, e.recv_ns) for i, e in enumerate(events)
    ]
    return Conversion(numbered, dropped, rows)


@dataclass(frozen=True)
class Reconstruction:
    rows: int
    levels: int
    match_pct_by_level: tuple[float, ...]  # Share of rows where the top k levels all agree.
    first_mismatch_row: int | None  # First row where level 1 disagrees.
    dropped_unknown: int
    book_problems: int


def compare_to_snapshots(conversion: Conversion, orderbook: list[list[int]], units: Units = Units()) -> Reconstruction:
    """Replay converted events and compare each row against the supplied snapshots."""
    from ..book import ReplayedBook

    replay = ReplayedBook()
    levels = len(orderbook[0]) // 4
    matches = [0] * levels
    first_mismatch = None
    by_row: dict[int, int] = {}
    for index, row in enumerate(conversion.rows_of_events):
        by_row[row] = index
    cursor = 0
    events = conversion.events
    for row, snapshot in enumerate(orderbook):
        last = by_row.get(row)
        if last is not None:
            while cursor <= last:
                replay.apply(events[cursor])
                cursor += 1
        bids, asks = levels_of(snapshot)
        got_bids = replay.book.depth(Side.BID, levels)
        got_asks = replay.book.depth(Side.ASK, levels)
        for k in range(1, levels + 1):
            want_bids = [(_ticks(p, units), s // units.shares_per_lot) for p, s in bids[:k]]
            want_asks = [(_ticks(p, units), s // units.shares_per_lot) for p, s in asks[:k]]
            if got_bids[:k] == want_bids and got_asks[:k] == want_asks:
                matches[k - 1] += 1
            elif k == 1 and first_mismatch is None:
                first_mismatch = row
    total = len(orderbook)
    return Reconstruction(
        total,
        levels,
        tuple(round(100 * m / total, 4) for m in matches),
        first_mismatch,
        conversion.dropped_unknown,
        len(replay.problems),
    )
