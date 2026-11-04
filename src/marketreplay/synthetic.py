"""A seeded synthetic order-by-order market.

This replaces licensed multi-day data. It is a stochastic model, not a market,
so results on it validate the pipeline and show how decisions respond to known
structure. They say nothing about any real venue.

How it works, in one paragraph. A hidden fair value drifts with a hidden signal.
Market orders lean toward fair value, so they run over stale quotes, which
creates adverse selection for resting orders. Limit orders join near the touch,
lean with the signal, and cancel at random. Day parameters follow slow random
walks, so consecutive days resemble each other. That is why the study splits by
contiguous days with a gap.
"""

import math
import random
from dataclasses import asdict, dataclass

import numpy as np

from .book import Book
from .events import Event, Kind, Side, to_array

NS_PER_S = 1_000_000_000
BASE_PRICE = 10_000  # 1 tick is about 1 bp at this level.


@dataclass(frozen=True)
class DayParams:
    add_rate_per_s: float = 18.0
    market_rate_per_s: float = 2.6
    modify_rate_per_s: float = 0.8
    target_orders: int = 60  # Live orders at balance. Each order cancels at rate add_rate / target.
    fair_vol_ticks_per_sqrt_s: float = 0.22
    signal_half_life_s: float = 25.0
    signal_drift_ticks_per_s: float = 0.035
    flow_lean: float = 1.2  # How strongly market orders lean toward fair value.
    add_lean: float = 0.25  # How strongly limit orders lean with the signal.
    mean_add_lots: float = 5.0
    mean_market_lots: float = 3.0
    placement_decay: float = 0.60  # Geometric chance of each extra tick behind the touch.
    hidden_trade_share: float = 0.04
    recv_median_ms: float = 2.0


@dataclass(frozen=True)
class StudyPlan:
    days: int = 120
    session_s: int = 1800
    seed: int = 20261104
    halt_day_every: int = 11  # A session with a halt, excluded by parents that overlap it.
    gap_day_every: int = 13  # A session with a sequence gap and resynchronization.


def day_params(plan: StudyPlan) -> list[DayParams]:
    """Slowly varying day parameters, with occasional volatile and sparse days."""
    rng = random.Random(plan.seed)
    log_vol, log_rate, log_size = 0.0, 0.0, 0.0
    days = []
    for _ in range(plan.days):
        log_vol = 0.92 * log_vol + rng.gauss(0, 0.10)
        log_rate = 0.92 * log_rate + rng.gauss(0, 0.08)
        log_size = 0.92 * log_size + rng.gauss(0, 0.08)
        volatile = rng.random() < 0.20
        sparse = rng.random() < 0.20
        base = DayParams()
        days.append(
            DayParams(
                add_rate_per_s=base.add_rate_per_s * math.exp(log_rate) * (0.65 if sparse else 1.0),
                market_rate_per_s=base.market_rate_per_s * math.exp(log_rate) * (0.65 if sparse else 1.0),
                target_orders=30 if sparse else base.target_orders,
                fair_vol_ticks_per_sqrt_s=base.fair_vol_ticks_per_sqrt_s
                * math.exp(log_vol)
                * (2.0 if volatile else 1.0),
                signal_drift_ticks_per_s=base.signal_drift_ticks_per_s * (1.6 if volatile else 1.0),
                mean_add_lots=base.mean_add_lots * math.exp(log_size) * (0.45 if sparse else 1.0),
                placement_decay=0.72 if sparse else base.placement_decay,
                recv_median_ms=base.recv_median_ms * rng.uniform(0.7, 1.5),
            )
        )
    return days


def geometric(rng: random.Random, mean: float) -> int:
    """A positive integer with the given mean."""
    if mean <= 1:
        return 1
    return 1 + int(math.log(max(rng.random(), 1e-12)) / math.log(1 - 1 / mean))


class Generator:
    def __init__(self, params: DayParams, session_s: int, seed: int):
        self.p = params
        self.session_ns = session_s * NS_PER_S
        self.rng = random.Random(seed)
        self.book = Book()
        self.events: list[Event] = []
        self.next_id = 1
        self.ids: list[int] = []  # Live ids, for uniform random picks.
        self.id_slot: dict[int, int] = {}
        self.fair = float(BASE_PRICE)
        self.signal = 0.0
        self.ts = 0

    def emit(self, kind: Kind, side: Side = Side.NONE, order_id: int = 0, price: int = 0, qty: int = 0) -> None:
        event = Event(len(self.events) + 1, self.ts, kind, side, order_id, price, qty)
        self.book.apply(event)
        self.events.append(event)

    def _track(self, order_id: int) -> None:
        self.id_slot[order_id] = len(self.ids)
        self.ids.append(order_id)

    def _untrack(self, order_id: int) -> None:
        slot = self.id_slot.pop(order_id)
        last = self.ids.pop()
        if last != order_id:
            self.ids[slot] = last
            self.id_slot[last] = slot

    def add(self, side: Side, price: int, qty: int) -> None:
        order_id = self.next_id
        self.next_id += 1
        self.emit(Kind.ADD, side, order_id, price, qty)
        self._track(order_id)

    def open_session(self) -> None:
        self.emit(Kind.CLEAR)
        for level in range(1, 11):
            for _ in range(3):
                self.add(Side.BID, BASE_PRICE - level, geometric(self.rng, self.p.mean_add_lots))
                self.add(Side.ASK, BASE_PRICE + level, geometric(self.rng, self.p.mean_add_lots))
        self.emit(Kind.SNAPSHOT_END)

    def evolve(self, dt_s: float) -> None:
        rate = math.log(2) / self.p.signal_half_life_s
        keep = math.exp(-rate * dt_s)
        self.signal = self.signal * keep + math.sqrt(max(1 - keep * keep, 0)) * self.rng.gauss(0, 1)
        self.fair += self.p.signal_drift_ticks_per_s * self.signal * dt_s
        self.fair += self.p.fair_vol_ticks_per_sqrt_s * math.sqrt(dt_s) * self.rng.gauss(0, 1)

    def market_order(self) -> None:
        bid, ask = self.book.best(Side.BID), self.book.best(Side.ASK)
        mid = (bid + ask) / 2
        gap = (self.fair - mid) * self.p.flow_lean
        p_buy = 0.5 + 0.45 * math.tanh(gap + 0.35 * self.signal)
        aggressor = Side.BID if self.rng.random() < p_buy else Side.ASK
        resting = Side(-aggressor)
        lots = geometric(self.rng, self.p.mean_market_lots)
        if self.rng.random() < self.p.hidden_trade_share:
            self.emit(Kind.TRADE_HIDDEN, resting, 0, self.book.best(resting), lots)
            return
        while lots > 0:
            price = self.book.best(resting)
            if price is None or len(self.book.levels[resting]) == 1 and len(self.book.queue(resting, price)) == 1:
                break  # Leave one order so the book stays two-sided.
            front = self.book.queue(resting, price)[0]
            take = min(lots, self.book.order_qty(front))
            self.emit(Kind.EXECUTE, resting, front, price, take)
            if front not in self.book.order_index:
                self._untrack(front)
            lots -= take

    def limit_order(self) -> None:
        bid, ask = self.book.best(Side.BID), self.book.best(Side.ASK)
        p_bid = 0.5 + 0.45 * math.tanh(self.p.add_lean * self.signal)
        side = Side.BID if self.rng.random() < p_bid else Side.ASK
        behind = 0
        while self.rng.random() < self.p.placement_decay and behind < 12:
            behind += 1
        reference = math.floor(self.fair - 0.5) if side == Side.BID else math.ceil(self.fair + 0.5)
        touch = bid if side == Side.BID else ask
        anchor = touch + side if (ask - bid > 1 and self.rng.random() < 0.6) else touch
        if (reference - anchor) * side < 0:
            anchor = reference  # Quotes do not chase past fair value.
        price = anchor - side * behind
        opposite = ask if side == Side.BID else bid
        if (price - opposite) * side >= 0:
            price = opposite - side
        self.add(side, price, geometric(self.rng, self.p.mean_add_lots))

    def cancel(self) -> None:
        order_id = self._pick_order()
        if order_id is None:
            return
        side, price = self.book.order_index[order_id]
        qty = self.book.order_qty(order_id)
        take = qty if self.rng.random() < 0.8 else self.rng.randint(1, qty)
        self.emit(Kind.CANCEL, side, order_id, price, take)
        if order_id not in self.book.order_index:
            self._untrack(order_id)

    def modify(self) -> None:
        order_id = self._pick_order()
        if order_id is None:
            return
        side, price = self.book.order_index[order_id]
        qty = self.book.order_qty(order_id)
        roll = self.rng.random()
        if roll < 0.5 and qty > 1:
            self.emit(Kind.MODIFY, side, order_id, price, qty - 1)  # Keeps priority.
        elif roll < 0.8:
            self.emit(Kind.MODIFY, side, order_id, price, qty + self.rng.randint(1, 3))  # Loses it.
        else:
            target = price - side  # Steps back one tick, a new queue.
            self.emit(Kind.MODIFY, side, order_id, target, qty)

    def _pick_order(self) -> int | None:
        """A random order, except the last one on each side."""
        for _ in range(4):
            order_id = self.rng.choice(self.ids)
            side, price = self.book.order_index[order_id]
            if len(self.book.levels[side]) > 1 or len(self.book.levels[side][price]) > 1:
                return order_id
        return None

    def run(self) -> list[Event]:
        """Thinned event loop. Cancels scale with live orders, which keeps the book in balance."""
        self.open_session()
        p = self.p
        cancel_rate_per_order = p.add_rate_per_s / p.target_orders
        fixed = p.add_rate_per_s + p.market_rate_per_s + p.modify_rate_per_s
        while True:
            cancel_rate = cancel_rate_per_order * len(self.ids)
            total = fixed + cancel_rate
            dt_s = self.rng.expovariate(total)
            self.ts += max(1, int(dt_s * NS_PER_S))
            if self.ts >= self.session_ns:
                return self.events
            self.evolve(dt_s)
            roll = self.rng.random() * total
            if roll < p.market_rate_per_s:
                self.market_order()
            elif roll < p.market_rate_per_s + p.add_rate_per_s:
                self.limit_order()
            elif roll < fixed:
                self.modify()
            else:
                self.cancel()


def with_receive_times(events: list[Event], median_ms: float, rng: random.Random) -> list[Event]:
    """Add a feed arrival time. Lognormal delay, so a latency can be estimated from data."""
    out = []
    for e in events:
        delay_ns = int(median_ms * 1e6 * math.exp(0.4 * rng.gauss(0, 1)))
        out.append(Event(e.seq, e.ts_ns, e.kind, e.side, e.order_id, e.price, e.qty, e.ts_ns + delay_ns))
    return out


def renumber(events: list[Event]) -> list[Event]:
    return [Event(i + 1, e.ts_ns, e.kind, e.side, e.order_id, e.price, e.qty, e.recv_ns) for i, e in enumerate(events)]


def snapshot_events(events: list[Event], ts_ns: int) -> list[Event]:
    """CLEAR, the full book left by `events`, then SNAPSHOT_END, all stamped ts_ns."""
    book = Book()
    for e in events:
        book.apply(e)
    rebuilt = [Event(0, ts_ns, Kind.CLEAR)]
    for side in (Side.BID, Side.ASK):
        for price in sorted(book.levels[side]):
            for order_id, qty in book.levels[side][price].items():
                rebuilt.append(Event(0, ts_ns, Kind.ADD, side, order_id, price, qty))
    rebuilt.append(Event(0, ts_ns, Kind.SNAPSHOT_END))
    return rebuilt


def insert_halt(events: list[Event], start_ns: int, length_ns: int) -> list[Event]:
    """Silence the feed for a window, then rebuild the book at the end of the halt."""
    end_ns = start_ns + length_ns
    before = [e for e in events if e.ts_ns < start_ns]
    during = [e for e in events if start_ns <= e.ts_ns < end_ns]
    after = [e for e in events if e.ts_ns >= end_ns]
    book_events = [e for e in before + during if e.kind != Kind.SNAPSHOT_END]
    halt = [Event(0, start_ns, Kind.HALT_START), Event(0, end_ns, Kind.HALT_END)]
    return renumber(before + halt + snapshot_events(book_events, end_ns) + after)


def insert_gap(events: list[Event], start: int, lost: int, blackout: int) -> list[Event]:
    """Lose `lost` events at index `start`, then resynchronize `blackout` events later.

    The surviving events keep their numbers, so a monitor sees exactly one jump.
    """
    numbered = renumber(events)
    resync_at = start + lost + blackout
    head = numbered[:start] + numbered[start + lost : resync_at]
    rebuilt = snapshot_events(numbered[:resync_at], numbered[resync_at - 1].ts_ns)
    tail = renumber(rebuilt + numbered[resync_at:])
    offset = head[-1].seq
    tail = [Event(e.seq + offset, e.ts_ns, e.kind, e.side, e.order_id, e.price, e.qty, e.recv_ns) for e in tail]
    return head + tail


def generate_day(plan: StudyPlan, day: int) -> tuple[np.ndarray, dict]:
    """Generate one session. The result depends only on the plan and the day number."""
    params = day_params(plan)[day]
    rng = random.Random(plan.seed * 1009 + day)
    events = Generator(params, plan.session_s, rng.getrandbits(32)).run()
    labels = {"day": day, "params": asdict(params), "halt": None, "gap": None}
    if plan.halt_day_every and day % plan.halt_day_every == plan.halt_day_every - 1:
        start_s = rng.uniform(0.3, 0.7) * plan.session_s
        events = insert_halt(events, int(start_s * NS_PER_S), 60 * NS_PER_S)
        labels["halt"] = {"start_s": start_s, "length_s": 60}
    if plan.gap_day_every and day % plan.gap_day_every == plan.gap_day_every - 1:
        start = int(rng.uniform(0.3, 0.7) * len(events))
        events = insert_gap(events, start, lost=40, blackout=300)
        labels["gap"] = {"start_index": start, "lost": 40, "blackout": 300}
    events = with_receive_times(events, params.recv_median_ms, rng)
    return to_array(events), labels
