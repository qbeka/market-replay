"""Simulated orders against a replayed feed.

Three clocks drive everything:

    exchange      ts_ns of each recorded feed event
    observation   a strategy at time t sees market snapshots no newer than
                  t - obs_latency_ns, and its own fills obs_latency_ns late
    order         an order decided at t arrives at t + order_latency_ns, and a
                  cancel decided at t lands at t + cancel_latency_ns

The feed is never changed. Each simulated parent order has its own Ledger and
its own record of consumed liquidity, so one pass over the feed can run many
strategies side by side without any of them affecting another. The cost of that
design is the central limit of this project: the replayed market cannot react to
a simulated order. See docs/assumptions.md.

Ties: a feed event at time t is processed before a simulated action at time t.
Simulated actions at the same time run in the order they were scheduled.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from itertools import count
from typing import Protocol

from .book import PriorityRules, ReplayedBook
from .events import Event, Kind, Side
from .ledger import Fill, Ledger, MTICKS_PER_TICK, Role


@dataclass(frozen=True)
class ExecConfig:
    obs_latency_ns: int = 20_000_000
    order_latency_ns: int = 20_000_000
    cancel_latency_ns: int = 20_000_000
    taker_fee_mticks: int = 300  # Per lot. 1000 mticks = 1 tick.
    maker_fee_mticks: int = -100  # Per lot. Negative is a rebate.
    decision_interval_ns: int = 2_000_000_000
    snapshot_interval_ns: int = 1_000_000_000
    trade_window_ns: int = 10_000_000_000  # Trailing window for trade flow in snapshots.
    cross_slippage_ticks: int = 1  # A cross may reach this far past the observed touch.
    cancel_ahead_credit_pct: int = 100  # 100 is exact FIFO. 0 never advances on cancels.
    queue_ahead_pct: int = 100  # Above 100 assumes extra unseen lots ahead at insertion.
    trade_through_only: bool = False  # True ignores fills at the touch, keeping only trade throughs.
    adverse_fill_mticks: int = 0  # Stress: extra cost per lot on every passive fill.
    participation_cap_pct: int | None = None
    participation_window_ns: int = 60_000_000_000
    participation_free_lots: int = 0  # Allowance so an order can start on a quiet tape.
    completion_penalty_ticks: int = 2  # Deadline rule: extra cost per forced lot.
    unfilled_extra_ticks: int = 10  # Deadline rule: price past the book when depth runs out.
    book_levels: int = 5
    priority: PriorityRules = PriorityRules()
    record_trace: bool = False


@dataclass(frozen=True)
class Parent:
    parent_id: int
    side: Side
    target_lots: int
    start_ns: int
    deadline_ns: int


class Action(IntEnum):
    WAIT = 0  # Hold no resting order.
    REST = 1  # Keep a passive order at the near touch.
    CROSS = 2  # Send an immediate-or-cancel order across the spread.


@dataclass(frozen=True)
class Intent:
    action: Action
    lots: int = 0

    @classmethod
    def wait(cls) -> "Intent":
        return cls(Action.WAIT)

    @classmethod
    def rest(cls, lots: int) -> "Intent":
        return cls(Action.REST, lots)

    @classmethod
    def cross(cls, lots: int) -> "Intent":
        return cls(Action.CROSS, lots)


@dataclass(frozen=True)
class Snapshot:
    """Market state at ts_ns. Strategies only ever receive these, delayed."""

    ts_ns: int
    bids: tuple[tuple[int, int], ...]
    asks: tuple[tuple[int, int], ...]
    buy_lots: int  # Trailing buyer-initiated volume.
    sell_lots: int  # Trailing seller-initiated volume.
    cum_lots: int  # Volume since the session started.


@dataclass
class View:
    """Everything a strategy may read. Built only from delayed information."""

    now_ns: int
    parent: Parent
    arrival_mid_x2: int
    market: Snapshot
    history: list[Snapshot]  # Observable snapshots, oldest first, ending at market.
    filled_seen: int
    open_seen: int  # Lots in orders not yet confirmed finished.
    rest_price: int | None  # Own resting order, if one is live and not being cancelled.
    rest_open: int
    queue_ahead_seen: int | None

    @property
    def remaining_seen(self) -> int:
        return self.parent.target_lots - self.filled_seen

    @property
    def sendable(self) -> int:
        """Lots that can be sent without risking an overfill."""
        return self.remaining_seen - self.open_seen

    @property
    def time_left_ns(self) -> int:
        return self.parent.deadline_ns - self.now_ns


class Strategy(Protocol):
    name: str

    def decide(self, view: View) -> Intent | None:
        """Return the desired state. None leaves everything as it is."""


class ScriptedStrategy:
    """Replays a fixed {time: Intent} script. Used by tests and the compiled check."""

    name = "scripted"

    def __init__(self, script: dict[int, Intent]):
        self.script = script

    def decide(self, view: View) -> Intent | None:
        return self.script.get(view.now_ns)


@dataclass(eq=False)
class Child:
    child_id: int
    run: "Run"
    side: Side
    price: int
    lots: int
    immediate: bool
    filled: int = 0
    arrived: bool = False
    live: bool = False
    finished: bool = False
    cancel_requested: bool = False
    queue_ahead: int = 0
    queue_trail: deque = field(default_factory=deque)  # (ts_ns, queue_ahead)

    @property
    def open_lots(self) -> int:
        return self.lots - self.filled


@dataclass(eq=False)
class Run:
    """One strategy working one parent order."""

    parent: Parent
    strategy: Strategy
    ledger: Ledger | None = None
    excluded: str | None = None  # Why this parent was dropped, for every strategy alike.
    finished: bool = False
    children: list[Child] = field(default_factory=list)
    resting: Child | None = None
    notices: deque = field(default_factory=deque)  # (visible_ns, child, filled_qty, done)
    filled_seen: int = 0
    open_seen: dict = field(default_factory=dict)  # child -> open lots as last known
    consumed: dict = field(default_factory=dict)  # order_id -> lots this ledger took
    own_window: deque = field(default_factory=deque)  # (ts_ns, qty)
    own_window_lots: int = 0
    trace: list = field(default_factory=list)

    def log(self, ts_ns: int, what: str, **detail) -> None:
        self.trace.append({"ts_ns": ts_ns, "what": what, **detail})


class Engine:
    def __init__(self, config: ExecConfig):
        self.cfg = config
        self.replay = ReplayedBook(config.priority)
        self.book = self.replay.book
        self.heap: list = []
        self.ticket = count()
        self.child_ids = count(1)
        self.active: dict[int, list[Child]] = {Side.BID: [], Side.ASK: []}
        self.watchers: dict[int, list[Child]] = {}  # Book order id -> children queued behind it.
        self.live_runs: list[Run] = []
        self.snapshots: list[Snapshot] = []
        self.visible_snapshots = 0  # Moves forward only, so lookups stay cheap.
        self.halted = False
        self.cum_lots = 0
        self.tape: deque = deque()  # (ts_ns, aggressor_side, lots) for trade flow.
        self.tape_buy = 0
        self.tape_sell = 0
        self.cap_tape: deque = deque()  # (ts_ns, lots) for the participation cap.
        self.cap_tape_lots = 0

    # Scheduling -----------------------------------------------------------

    def at(self, ts_ns: int, action, *args) -> None:
        heapq.heappush(self.heap, (ts_ns, next(self.ticket), action, args))

    def run_until(self, ts_ns: int) -> None:
        """Run simulated actions strictly before ts_ns."""
        while self.heap and self.heap[0][0] < ts_ns:
            when, _, action, args = heapq.heappop(self.heap)
            action(when, *args)

    # Feed hooks. These run before the book applies the event ---------------

    def on_feed(self, e: Event) -> None:
        kind = e.kind
        if kind == Kind.EXECUTE or kind == Kind.TRADE_HIDDEN:
            self._note_trade(e)
            if self.active[e.side]:
                self._historical_trade(e)
        elif not self.watchers and not self.active[Side.BID] and not self.active[Side.ASK]:
            pass  # Nothing simulated is working, so only the book needs this event.
        elif kind == Kind.CANCEL:
            self._cancel_ahead(e.order_id, e.qty, e.qty == self._book_qty(e.order_id))
        elif kind == Kind.MODIFY:
            self._modify_ahead(e)
        elif kind == Kind.ADD and self.active[-e.side]:
            self._crossing_add(e)
        if kind == Kind.EXECUTE and self.watchers and e.qty == self._book_qty(e.order_id):
            self.watchers.pop(e.order_id, None)

    def on_feed_applied(self, e: Event) -> None:
        if e.kind == Kind.HALT_START:
            self.halted = True
        elif e.kind == Kind.HALT_END:
            self.halted = False
        if (self.halted or not self.replay.valid) and self.live_runs:
            reason = "halt" if self.halted else "invalid book"
            for run in list(self.live_runs):
                self._exclude(run, e.ts_ns, reason)

    def _book_qty(self, order_id: int) -> int | None:
        located = self.book.order_index.get(order_id)
        return None if located is None else self.book.levels[located[0]][located[1]][order_id]

    def _note_trade(self, e: Event) -> None:
        self.cum_lots += e.qty
        self.tape.append((e.ts_ns, -e.side, e.qty))
        if e.side == Side.ASK:
            self.tape_buy += e.qty  # A resting ask was lifted, so the aggressor bought.
        else:
            self.tape_sell += e.qty
        if self.cfg.participation_cap_pct is not None:
            self.cap_tape.append((e.ts_ns, e.qty))
            self.cap_tape_lots += e.qty

    def _historical_trade(self, e: Event) -> None:
        """An aggressor traded e.qty at e.price against resting side e.side.

        A simulated order at the same price sits in the FIFO queue: the trade first
        burns the lots ahead of it. A simulated order at a better price would have
        met the aggressor first, so nothing is ahead of it and it fills. One
        historical trade can pay out at most e.qty lots to a ledger, however many
        of that ledger's orders it touches.
        """
        budgets: dict[Run, int] = {}
        for child in list(self.active[e.side]):
            improvement = (child.price - e.price) * e.side
            if improvement < 0:
                continue
            run = child.run
            passing = e.qty
            if improvement > 0:
                child.queue_ahead = 0
            elif self.cfg.trade_through_only:
                continue
            else:
                burned = min(child.queue_ahead, passing)
                child.queue_ahead -= burned
                passing -= burned
            budget = budgets.get(run, e.qty)
            qty = min(passing, child.open_lots, budget, run.ledger.remaining_lots)
            if qty > 0:
                qty = self._capped(run, e.ts_ns, qty)
            if qty > 0:
                budgets[run] = budget - qty
                self._fill_passive(child, e.ts_ns, qty, e.seq)

    def _cancel_ahead(self, order_id: int, qty: int, order_gone: bool) -> None:
        children = self.watchers.get(order_id)
        if children is None:
            return
        credit = qty * self.cfg.cancel_ahead_credit_pct // 100
        for child in children:
            child.queue_ahead = max(0, child.queue_ahead - credit)
        if order_gone:
            del self.watchers[order_id]

    def _modify_ahead(self, e: Event) -> None:
        located = self.book.order_index.get(e.order_id)
        if located is None or e.order_id not in self.watchers:
            return
        _, old_price = located
        old_qty = self._book_qty(e.order_id)
        if self.book.loses_priority(e, old_price, old_qty):
            self._cancel_ahead(e.order_id, old_qty, True)  # It leaves the queue ahead.
        elif e.qty < old_qty:
            self._cancel_ahead(e.order_id, old_qty - e.qty, False)
        else:
            for child in self.watchers[e.order_id]:  # It grew and kept its place.
                child.queue_ahead += e.qty - old_qty

    def _crossing_add(self, e: Event) -> None:
        """A recorded add can land on a simulated order that improved on the book.

        In history the add rested. In the replayed market it would have met the
        simulated order, so that order fills at its own limit price.
        """
        for child in list(self.active[-e.side]):
            run = child.run
            if (child.price - e.price) * child.side < 0:
                continue
            child.queue_ahead = 0
            available = e.qty - run.consumed.get(e.order_id, 0)
            qty = min(available, child.open_lots, run.ledger.remaining_lots)
            if qty > 0:
                qty = self._capped(run, e.ts_ns, qty)
            if qty > 0:
                run.consumed[e.order_id] = run.consumed.get(e.order_id, 0) + qty
                self._fill_passive(child, e.ts_ns, qty, e.seq)

    # Simulated actions ------------------------------------------------------

    def activate(self, now: int, run: Run) -> None:
        bid, ask = self.book.best(Side.BID), self.book.best(Side.ASK)
        if not self.replay.valid or self.halted:
            run.excluded = "halt" if self.halted else "invalid book"
            return
        if bid is None or ask is None:
            run.excluded = "one-sided book at arrival"
            return
        run.ledger = Ledger(run.parent.side, run.parent.target_lots, bid + ask)
        self.live_runs.append(run)
        run.log(now, "arrival", mid_x2=bid + ask)
        self.at(now, self.decide, run)
        self.at(run.parent.deadline_ns, self.deadline, run)

    def decide(self, now: int, run: Run) -> None:
        if run.finished:
            return
        view = self._view(now, run)
        if view is not None:
            intent = run.strategy.decide(view)
            if self.cfg.record_trace:
                info = getattr(run.strategy, "last_info", None)
                run.log(now, "decision", intent=None if intent is None else (intent.action.name, intent.lots), info=info)
            if intent is not None:
                self._enact(now, run, view, intent)
        next_tick = now + self.cfg.decision_interval_ns
        if next_tick < run.parent.deadline_ns:
            self.at(next_tick, self.decide, run)

    def _view(self, now: int, run: Run) -> View | None:
        while run.notices and run.notices[0][0] <= now:
            _, child, qty, done = run.notices.popleft()
            run.filled_seen += qty
            run.open_seen[child] -= qty
            if done:
                del run.open_seen[child]
        snapshots = self.snapshots
        limit = now - self.cfg.obs_latency_ns
        while self.visible_snapshots < len(snapshots) and snapshots[self.visible_snapshots].ts_ns <= limit:
            self.visible_snapshots += 1
        visible = self.visible_snapshots  # Actions run in time order, so this only moves forward.
        if visible == 0:
            return None
        resting = run.resting
        rest_live = resting is not None and not resting.cancel_requested and not resting.finished
        queue_seen = None
        if rest_live:
            trail = resting.queue_trail
            while len(trail) > 1 and trail[1][0] <= limit:
                trail.popleft()
            if trail and trail[0][0] <= limit:
                queue_seen = trail[0][1]
        return View(
            now_ns=now,
            parent=run.parent,
            arrival_mid_x2=run.ledger.arrival_mid_x2,
            market=snapshots[visible - 1],
            history=snapshots[max(0, visible - 64) : visible],
            filled_seen=run.filled_seen,
            open_seen=sum(run.open_seen.values()),
            rest_price=resting.price if rest_live else None,
            rest_open=run.open_seen.get(resting, 0) if rest_live else 0,
            queue_ahead_seen=queue_seen,
        )

    def _enact(self, now: int, run: Run, view: View, intent: Intent) -> None:
        side = run.parent.side
        near = view.market.bids if side == Side.BID else view.market.asks
        far = view.market.asks if side == Side.BID else view.market.bids
        if intent.action == Action.WAIT:
            self._cancel_resting(now, run)
        elif intent.action == Action.CROSS:
            self._cancel_resting(now, run)
            lots = min(intent.lots, view.sendable)
            if lots > 0 and far:
                limit = far[0][0] + side * self.cfg.cross_slippage_ticks
                self._send(now, run, limit, lots, immediate=True)
        elif intent.action == Action.REST:
            if not near:
                return
            target = near[0][0]
            if view.rest_price == target:
                return  # Already there. Staying keeps the queue position.
            self._cancel_resting(now, run)
            lots = min(intent.lots, view.sendable)
            if lots > 0:
                run.resting = self._send(now, run, target, lots, immediate=False)

    def _send(self, now: int, run: Run, price: int, lots: int, immediate: bool) -> Child:
        child = Child(next(self.child_ids), run, run.parent.side, price, lots, immediate)
        run.children.append(child)
        run.open_seen[child] = lots
        run.log(now, "send", child=child.child_id, price=price, lots=lots, immediate=immediate)
        self.at(now + self.cfg.order_latency_ns, self.arrive, child)
        return child

    def _cancel_resting(self, now: int, run: Run) -> None:
        child = run.resting
        if child is not None and not child.cancel_requested and not child.finished:
            child.cancel_requested = True
            run.log(now, "cancel_request", child=child.child_id)
            self.at(now + self.cfg.cancel_latency_ns, self.cancel_lands, child)

    def arrive(self, now: int, child: Child) -> None:
        run = child.run
        if run.finished or child.finished:
            return
        child.arrived = True
        if child.immediate:
            self._take(now, child)
            self._finish(now, child)
            return
        best_opposite = self.book.best(Side(-child.side))
        if best_opposite is not None and (child.price - best_opposite) * child.side >= 0:
            run.log(now, "reject_post_only", child=child.child_id)
            self._finish(now, child)
            return
        self._join_queue(now, child)

    def _join_queue(self, now: int, child: Child) -> None:
        level = self.book.levels[child.side].get(child.price, {})
        ahead = 0
        for order_id, qty in level.items():
            ahead += qty
            self.watchers.setdefault(order_id, []).append(child)
        for other in child.run.children:
            if other is not child and other.live and other.price == child.price:
                ahead += other.open_lots
        child.queue_ahead = ahead * self.cfg.queue_ahead_pct // 100
        child.queue_trail.append((now, child.queue_ahead))
        child.live = True
        self.active[child.side].append(child)
        child.run.log(now, "rest", child=child.child_id, queue_ahead=child.queue_ahead)

    def cancel_lands(self, now: int, child: Child) -> None:
        if child.finished:
            return
        child.run.log(now, "cancelled", child=child.child_id, lots=child.open_lots)
        self._finish(now, child)

    def _finish(self, now: int, child: Child) -> None:
        child.finished = True
        if child.live:
            child.live = False
            self.active[child.side].remove(child)
            for order_id in [o for o, kids in self.watchers.items() if child in kids]:
                self.watchers[order_id].remove(child)
                if not self.watchers[order_id]:
                    del self.watchers[order_id]
        child.run.notices.append((now + self.cfg.obs_latency_ns, child, 0, True))

    def _capped(self, run: Run, now: int, qty: int) -> int:
        cap = self.cfg.participation_cap_pct
        if cap is None:
            return qty
        horizon = now - self.cfg.participation_window_ns
        while self.cap_tape and self.cap_tape[0][0] <= horizon:
            self.cap_tape_lots -= self.cap_tape.popleft()[1]
        while run.own_window and run.own_window[0][0] <= horizon:
            run.own_window_lots -= run.own_window.popleft()[1]
        allowed = self.cap_tape_lots * cap // 100 + self.cfg.participation_free_lots - run.own_window_lots
        return max(0, min(qty, allowed))

    def _note_own(self, run: Run, now: int, qty: int) -> None:
        if self.cfg.participation_cap_pct is not None:
            run.own_window.append((now, qty))
            run.own_window_lots += qty

    def _fill_passive(self, child: Child, now: int, qty: int, seq: int) -> None:
        run = child.run
        fee = self.cfg.maker_fee_mticks * qty
        fill = Fill(now, child.child_id, child.price, qty, Role.MAKER, fee, self.cfg.adverse_fill_mticks * qty, seq)
        run.ledger.record(fill)
        child.filled += qty
        self._note_own(run, now, qty)
        run.log(now, "fill", child=child.child_id, price=child.price, qty=qty, role="maker", queue_ahead=child.queue_ahead)
        run.notices.append((now + self.cfg.obs_latency_ns, child, qty, False))
        if child.open_lots == 0:
            self._finish(now, child)

    def _take(self, now: int, child: Child) -> None:
        """Walk the opposite side up to the limit, skipping lots this ledger already took."""
        run = child.run
        side = Side(-child.side)
        wanted = min(child.lots, run.ledger.remaining_lots)
        for price, _ in self.book.depth(side, self.cfg.book_levels * 4):
            if wanted == 0 or (price - child.price) * child.side > 0:
                break
            level_qty = self._available_at(run, side, price, wanted)
            level_qty = self._capped(run, now, level_qty) if level_qty else 0
            if level_qty == 0:
                continue
            self._commit_consumption(run, side, price, level_qty)
            fill = Fill(now, child.child_id, price, level_qty, Role.TAKER, self.cfg.taker_fee_mticks * level_qty, 0, 0)
            run.ledger.record(fill)
            child.filled += level_qty
            wanted -= level_qty
            self._note_own(run, now, level_qty)
            run.log(now, "fill", child=child.child_id, price=price, qty=level_qty, role="taker")
            run.notices.append((now + self.cfg.obs_latency_ns, child, level_qty, False))

    def _available_at(self, run: Run, side: Side, price: int, wanted: int) -> int:
        available = 0
        for order_id, qty in self.book.levels[side][price].items():
            available += max(0, qty - run.consumed.get(order_id, 0))
            if available >= wanted:
                return wanted
        return available

    def _commit_consumption(self, run: Run, side: Side, price: int, qty: int) -> None:
        for order_id, order_qty in self.book.levels[side][price].items():
            used = run.consumed.get(order_id, 0)
            take = min(qty, max(0, order_qty - used))
            if take:
                run.consumed[order_id] = used + take
                qty -= take
            if qty == 0:
                return

    def deadline(self, now: int, run: Run) -> None:
        """Complete the order whatever it costs. Forced lots pay the taker fee and a penalty.

        This rule is what stops a strategy from scoring well by not finishing:
        residual inventory is always bought at a walked price plus a penalty.
        """
        if run.finished:
            return
        run.finished = True
        for child in run.children:
            if not child.finished:
                self._finish(now, child)
        ledger = run.ledger
        side = Side(-run.parent.side)
        worst = None
        for price, _ in self.book.depth(side, 10**6):
            if ledger.remaining_lots == 0:
                break
            qty = self._available_at(run, side, price, ledger.remaining_lots)
            worst = price
            if qty:
                self._commit_consumption(run, side, price, qty)
                self._force(now, run, price, qty)
        if ledger.remaining_lots:
            base = worst if worst is not None else (ledger.arrival_mid_x2 + 1) // 2
            self._force(now, run, base + run.parent.side * self.cfg.unfilled_extra_ticks, ledger.remaining_lots)
        self.live_runs.remove(run)
        run.log(now, "complete", filled=ledger.filled_lots)

    def _force(self, now: int, run: Run, price: int, qty: int) -> None:
        cfg = self.cfg
        penalty = cfg.completion_penalty_ticks * MTICKS_PER_TICK * qty
        run.ledger.record(Fill(now, 0, price, qty, Role.FORCED, cfg.taker_fee_mticks * qty, penalty, 0))
        run.log(now, "fill", child=0, price=price, qty=qty, role="forced")

    def _exclude(self, run: Run, now: int, reason: str) -> None:
        run.excluded = reason
        run.finished = True
        for child in run.children:
            if not child.finished:
                self._finish(now, child)
        self.live_runs.remove(run)

    def capture(self, now: int, last_ts_ns: int) -> None:
        cfg = self.cfg
        horizon = now - cfg.trade_window_ns
        while self.tape and self.tape[0][0] <= horizon:
            _, aggressor, lots = self.tape.popleft()
            if aggressor == Side.BID:
                self.tape_buy -= lots
            else:
                self.tape_sell -= lots
        self.snapshots.append(
            Snapshot(
                now,
                tuple(self.book.depth(Side.BID, cfg.book_levels)),
                tuple(self.book.depth(Side.ASK, cfg.book_levels)),
                self.tape_buy,
                self.tape_sell,
                self.cum_lots,
            )
        )
        for side in (Side.BID, Side.ASK):
            for child in self.active[side]:
                child.queue_trail.append((now, child.queue_ahead))
        if now + cfg.snapshot_interval_ns <= last_ts_ns:
            self.at(now + cfg.snapshot_interval_ns, self.capture, last_ts_ns)


def run_session(events: list[Event], work: list[tuple[Parent, Strategy]], config: ExecConfig) -> list[Run]:
    """Replay one session and return one Run per (parent, strategy) pair, in input order."""
    engine = Engine(config)
    runs = [Run(parent, strategy) for parent, strategy in work]
    for run in runs:
        engine.at(run.parent.start_ns, engine.activate, run)
    if events:
        engine.at(events[0].ts_ns, engine.capture, events[-1].ts_ns)
    replay_apply = engine.replay.apply
    heap = engine.heap
    for event in events:
        if heap and heap[0][0] < event.ts_ns:
            engine.run_until(event.ts_ns)
        engine.on_feed(event)
        replay_apply(event)
        engine.on_feed_applied(event)
    engine.run_until(2**63)
    return runs
