"""The normalized feed record every adapter produces.

Units are explicit and integer so replay is exact and portable:

    ts_ns, recv_ns   nanoseconds since the start of the trading date
    price            integer ticks
    qty              integer lots
    seq              feed sequence number, +1 per event within a session
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Kind(IntEnum):
    ADD = 1  # New visible resting order.
    CANCEL = 2  # Remove qty lots from a resting order, partial or full.
    MODIFY = 3  # Change an order to (price, qty). Priority depends on venue rules.
    EXECUTE = 4  # Resting visible order traded qty lots. side is the resting side.
    TRADE_HIDDEN = 5  # Print against hidden liquidity. No visible book change.
    CLEAR = 6  # Empty the book. Starts a session or a resynchronization.
    SNAPSHOT_END = 7  # Rebuild after CLEAR is complete. The book is valid again.
    HALT_START = 8  # Auction or halt. Continuous matching assumptions stop holding.
    HALT_END = 9  # Continuous trading resumes.


class Side(IntEnum):
    NONE = 0
    BID = 1
    ASK = -1


@dataclass(frozen=True, slots=True)
class Event:
    seq: int
    ts_ns: int
    kind: Kind
    side: Side = Side.NONE
    order_id: int = 0
    price: int = 0
    qty: int = 0
    recv_ns: int = 0


# On-disk layout. compiled/replay.c reads the same 48-byte little-endian record.
EVENT_DTYPE = np.dtype(
    [
        ("seq", "<i8"),
        ("ts_ns", "<i8"),
        ("recv_ns", "<i8"),
        ("order_id", "<i8"),
        ("price", "<i8"),
        ("qty", "<i4"),
        ("kind", "<i2"),
        ("side", "<i2"),
    ]
)
assert EVENT_DTYPE.itemsize == 48


def to_array(events: list[Event]) -> np.ndarray:
    """Pack events into the on-disk layout."""
    records = np.zeros(len(events), dtype=EVENT_DTYPE)
    for i, e in enumerate(events):
        records[i] = (e.seq, e.ts_ns, e.recv_ns, e.order_id, e.price, e.qty, int(e.kind), int(e.side))
    return records


def from_array(records: np.ndarray) -> list[Event]:
    """Unpack on-disk records. Replay keeps int and enum fields, never floats."""
    return [
        Event(seq, ts, Kind(kind), Side(side), oid, price, qty, recv)
        for seq, ts, recv, oid, price, qty, kind, side in records.tolist()
    ]
