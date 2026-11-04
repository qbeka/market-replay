"""Reference limit order book. Correctness first, speed second.

Book replays the recorded feed and nothing else. Simulated orders live in
execution.py and never touch it, so a hypothetical fill cannot rewrite history.

Determinism: priority within a price level is feed order. A Python dict keeps
insertion order, so each level's dict is its FIFO queue.
"""

from bisect import bisect_left, insort
from dataclasses import dataclass

from .events import Event, Kind, Side


class BookError(Exception):
    """The feed asked for something impossible given the current book."""


@dataclass(frozen=True)
class PriorityRules:
    """Which MODIFY events lose time priority. Varies by venue, so it is configuration."""

    size_increase_loses_priority: bool = True
    size_decrease_loses_priority: bool = False


class Book:
    def __init__(self, rules: PriorityRules = PriorityRules()):
        self.rules = rules
        self.clear()

    def clear(self) -> None:
        self.levels: dict[int, dict[int, dict[int, int]]] = {Side.BID: {}, Side.ASK: {}}
        # Sorted ascending by key so the best price is always the last element:
        # bids sort by price, asks by negated price.
        self._sorted_keys: dict[int, list[int]] = {Side.BID: [], Side.ASK: []}
        self.order_index: dict[int, tuple[int, int]] = {}  # order_id -> (side, price)

    def best(self, side: Side) -> int | None:
        keys = self._sorted_keys[side]
        return keys[-1] * side if keys else None

    def depth(self, side: Side, levels: int) -> list[tuple[int, int]]:
        """Top levels as (price_ticks, total_lots), best first."""
        keys = self._sorted_keys[side]
        rows = []
        for key in reversed(keys[-levels:] if levels > 0 else ()):
            price = key * side
            rows.append((price, sum(self.levels[side][price].values())))
        return rows

    def queue(self, side: Side, price: int) -> list[int]:
        """Order ids at a price, front of the queue first."""
        return list(self.levels[side].get(price, ()))

    def order_qty(self, order_id: int) -> int:
        side, price = self.order_index[order_id]
        return self.levels[side][price][order_id]

    def apply(self, e: Event) -> None:
        if e.kind == Kind.ADD:
            self._add(e.side, e.order_id, e.price, e.qty)
        elif e.kind == Kind.CANCEL or e.kind == Kind.EXECUTE:
            self._reduce(e)
        elif e.kind == Kind.MODIFY:
            self._modify(e)
        elif e.kind == Kind.CLEAR:
            self.clear()
        # TRADE_HIDDEN, SNAPSHOT_END and HALT events do not change visible depth.

    def _add(self, side: Side, order_id: int, price: int, qty: int) -> None:
        if order_id in self.order_index:
            raise BookError(f"order {order_id} added twice")
        if qty <= 0:
            raise BookError(f"order {order_id} added with qty {qty}")
        opposite = self.best(Side(-side))
        if opposite is not None and (price - opposite) * side >= 0:
            raise BookError(f"order {order_id} at {price} locks or crosses {opposite}")
        level = self.levels[side].get(price)
        if level is None:
            level = self.levels[side][price] = {}
            insort(self._sorted_keys[side], price * side)
        level[order_id] = qty
        self.order_index[order_id] = (side, price)

    def _reduce(self, e: Event) -> None:
        located = self.order_index.get(e.order_id)
        if located is None:
            raise BookError(f"order {e.order_id} is unknown")
        side, price = located
        if e.price != price or e.side != side:
            raise BookError(f"order {e.order_id} is at {side}/{price}, not {e.side}/{e.price}")
        remaining = self.levels[side][price][e.order_id] - e.qty
        if remaining < 0 or e.qty <= 0:
            raise BookError(f"order {e.order_id} reduced by {e.qty}, more than it holds")
        if remaining == 0:
            self._remove(side, price, e.order_id)
        else:
            self.levels[side][price][e.order_id] = remaining

    def _modify(self, e: Event) -> None:
        located = self.order_index.get(e.order_id)
        if located is None:
            raise BookError(f"order {e.order_id} is unknown")
        side, price = located
        if e.side != side:
            raise BookError(f"order {e.order_id} is on side {side}, not {e.side}")
        old_qty = self.levels[side][price][e.order_id]
        if self.loses_priority(e, price, old_qty):
            self._remove(side, price, e.order_id)
            self._add(side, e.order_id, e.price, e.qty)
        elif e.qty <= 0:
            raise BookError(f"order {e.order_id} modified to qty {e.qty}")
        else:
            self.levels[side][price][e.order_id] = e.qty

    def loses_priority(self, e: Event, old_price: int, old_qty: int) -> bool:
        if e.price != old_price:
            return True  # A new price is a new queue on every venue.
        if e.qty > old_qty:
            return self.rules.size_increase_loses_priority
        if e.qty < old_qty:
            return self.rules.size_decrease_loses_priority
        return False

    def _remove(self, side: int, price: int, order_id: int) -> None:
        level = self.levels[side][price]
        del level[order_id]
        del self.order_index[order_id]
        if not level:
            del self.levels[side][price]
            keys = self._sorted_keys[side]
            del keys[bisect_left(keys, price * side)]


class ReplayedBook:
    """A Book guarded by feed integrity checks.

    Any sequence gap or impossible event invalidates the book. It stays invalid
    until a CLEAR and SNAPSHOT_END rebuild it. Parent orders that overlap an
    invalid interval are excluded, because the book they would have seen is unknown.
    """

    def __init__(self, rules: PriorityRules = PriorityRules()):
        self.book = Book(rules)
        self.valid = False
        self.rebuilding = False
        self.last_seq: int | None = None
        self.last_ts_ns = 0
        self.invalid_since_ns: int | None = 0
        self.invalid_intervals_ns: list[tuple[int, int]] = []
        self.started = False
        self.problems: list[str] = []

    def apply(self, e: Event) -> None:
        if e.kind == Kind.CLEAR:
            # A CLEAR is the one event allowed to restart the sequence.
            self._invalidate(self.last_ts_ns)
            self.book.clear()
            self.rebuilding = True
            self._mark_seen(e)
            return

        if self.last_seq is not None and e.seq != self.last_seq + 1:
            self.rebuilding = False
            self._invalidate(self.last_ts_ns)
            self.problems.append(f"gap: seq {self.last_seq} then {e.seq}")

        if self.valid or self.rebuilding:
            try:
                self.book.apply(e)
            except BookError as error:
                self.rebuilding = False
                self._invalidate(self.last_ts_ns)
                self.problems.append(f"seq {e.seq}: {error}")

        if e.kind == Kind.SNAPSHOT_END and self.rebuilding:
            self._validate(e.ts_ns)
        self._mark_seen(e)

    def _mark_seen(self, e: Event) -> None:
        self.last_seq = e.seq
        self.last_ts_ns = e.ts_ns

    def _invalidate(self, ts_ns: int) -> None:
        if self.valid:
            self.valid = False
            self.invalid_since_ns = ts_ns

    def _validate(self, ts_ns: int) -> None:
        self.rebuilding = False
        self.valid = True
        if self.started:
            self.invalid_intervals_ns.append((self.invalid_since_ns, ts_ns))
        self.started = True
        self.invalid_since_ns = None
