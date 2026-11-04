"""A strategy must never receive information newer than its observation clock."""

import random

from marketreplay.events import Event, Kind, Side
from marketreplay.execution import ExecConfig, Intent, Parent, run_session
from marketreplay.metrics import shortfall_bps

ONE_SIDED = "one-sided book at arrival"
CONFIG = ExecConfig(
    obs_latency_ns=30,
    order_latency_ns=11,
    cancel_latency_ns=17,
    taker_fee_mticks=300,
    maker_fee_mticks=-100,
    decision_interval_ns=25,
    snapshot_interval_ns=10,
)


def random_feed(seed: int, steps: int = 400) -> list[Event]:
    """A small valid random feed. Bids below 100 and asks above 100 so it never crosses."""
    rng = random.Random(seed)
    events = [Event(1, 0, Kind.CLEAR)]
    orders: dict[int, list] = {}
    seq, order_id, ts = 1, 0, 0

    def push(kind, side=Side.NONE, oid=0, price=0, qty=0):
        nonlocal seq
        seq += 1
        events.append(Event(seq, ts, kind, side, oid, price, qty))

    for side, price in ((Side.BID, 99), (Side.ASK, 101)):
        order_id += 1
        orders[order_id] = [side, price, 50]
        push(Kind.ADD, side, order_id, price, 50)
    push(Kind.SNAPSHOT_END)
    for _ in range(steps):
        ts += rng.randint(1, 9)
        roll = rng.random()
        if roll < 0.5 or len(orders) < 4:
            order_id += 1
            side = rng.choice((Side.BID, Side.ASK))
            price = (99 - rng.randint(0, 3)) if side == Side.BID else (101 + rng.randint(0, 3))
            qty = rng.randint(1, 9)
            orders[order_id] = [side, price, qty]
            push(Kind.ADD, side, order_id, price, qty)
        else:
            oid = rng.choice(list(orders))
            side, price, qty = orders[oid]
            take = rng.randint(1, qty)
            push(Kind.EXECUTE if roll < 0.75 else Kind.CANCEL, side, oid, price, take)
            orders[oid][2] -= take
            if orders[oid][2] == 0:
                del orders[oid]
    return events


class Spy:
    """Records everything it is shown and plays a random seeded policy."""

    name = "spy"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.seen = []

    def decide(self, view):
        self.seen.append(view)
        roll = self.rng.random()
        lots = self.rng.randint(1, 8)
        if roll < 0.4:
            return Intent.rest(lots)
        if roll < 0.7:
            return Intent.cross(lots)
        return Intent.wait()


def test_no_view_contains_a_snapshot_newer_than_the_observation_clock():
    for seed in range(20):
        spy = Spy(seed)
        [run] = run_session(random_feed(seed), [(Parent(1, Side.BID, 30, 100, 1500), spy)], CONFIG)
        assert spy.seen or run.excluded == ONE_SIDED
        for view in spy.seen:
            limit = view.now_ns - CONFIG.obs_latency_ns
            assert view.market.ts_ns <= limit
            assert all(s.ts_ns <= limit for s in view.history)


def test_reported_fills_are_delayed_by_the_observation_clock():
    for seed in range(20):
        spy = Spy(seed)
        [run] = run_session(random_feed(seed), [(Parent(1, Side.BID, 30, 100, 1500), spy)], CONFIG)
        if run.excluded:
            continue
        for view in spy.seen:
            known = sum(f.qty for f in run.ledger.fills if f.ts_ns + CONFIG.obs_latency_ns <= view.now_ns)
            assert view.filled_seen == known


def test_changing_the_future_cannot_change_past_decisions():
    """Rewrite the feed after time T. Every decision the observation clock places before T must match."""
    for seed in range(20):
        events = random_feed(seed, steps=300)
        cut = events[len(events) // 2].ts_ns
        future = [e for e in events if e.ts_ns >= cut]
        keep = [e for e in events if e.ts_ns < cut]
        changed = keep + [
            Event(e.seq, e.ts_ns, e.kind, e.side, e.order_id, e.price, e.qty) for e in future[:1]
        ]
        spy_a, spy_b = Spy(seed), Spy(seed)
        parent = Parent(1, Side.BID, 30, 100, 2000)
        run_session(events, [(parent, spy_a)], CONFIG)
        run_session(changed, [(parent, spy_b)], CONFIG)
        before = [v for v in spy_a.seen if v.market.ts_ns < cut]
        twins = [v for v in spy_b.seen if v.market.ts_ns < cut]
        for a, b in zip(before, twins):
            assert (a.now_ns, a.market, a.filled_seen, a.open_seen) == (b.now_ns, b.market, b.filled_seen, b.open_seen)


def test_random_runs_conserve_everything_and_always_complete():
    for seed in range(50):
        spy = Spy(seed)
        [run] = run_session(random_feed(seed), [(Parent(1, Side.ASK, 25, 100, 1500), spy)], CONFIG)
        if run.excluded:
            assert run.excluded == ONE_SIDED  # No midpoint exists, so there is no benchmark.
            continue
        run.ledger.check_conservation()
        assert run.ledger.remaining_lots == 0
        shortfall_bps(run.ledger)


def test_replay_is_deterministic():
    results = []
    for _ in range(2):
        [run] = run_session(random_feed(7), [(Parent(1, Side.BID, 30, 100, 1500), Spy(7))], CONFIG)
        results.append(run.ledger.fills)
    assert results[0] == results[1]




def test_a_decision_sees_a_snapshot_exactly_one_observation_latency_old():
    config = ExecConfig(obs_latency_ns=3, decision_interval_ns=20, snapshot_interval_ns=10)
    spy = Spy(1)
    run_session(random_feed(1), [(Parent(1, Side.BID, 30, 100, 1500), spy)], config)
    assert spy.seen and all(v.market.ts_ns == v.now_ns - 3 for v in spy.seen)
