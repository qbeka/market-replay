"""Two different sources of fill labels. They answer different questions.

Hypothetical probes: place a simulated passive order, run it through the queue
model and record what happened. These are model outputs, not observations. They
match the policy's own orders in size and placement, and they are never cancelled
early, so they are not censored. The policy trains on these.

Observed outcomes: follow real resting orders in the recorded feed. These are
facts, but real traders chose to cancel many of those orders, so cancellation
censors them. They act as an outside check on the probes.
"""

import random
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from .book import ReplayedBook
from .events import Event, Kind, Side
from .execution import ExecConfig, Intent, Parent, Snapshot, View, replay_session
from .features import FEATURE_NAMES, features, mid
from .study import Study, load_day
from .synthetic import NS_PER_S

PROBE_LOTS = 16
LABEL_HORIZON_S = 15
PROBE_SPACING_S = 4


class Probe:
    """Rest once at the near touch, record the features, then leave the order alone."""

    name = "probe"

    def __init__(self, time_left_s: float, remaining_lots: int):
        self.time_left_s = time_left_s
        self.remaining_lots = remaining_lots
        self.row = None
        self.observed_mid = None
        self.spread = None

    def decide(self, view: View) -> Intent | None:
        if self.row is not None:
            return None
        row = features(view)
        if row is None:
            return None
        row[FEATURE_NAMES.index("time_left_s")] = self.time_left_s
        row[FEATURE_NAMES.index("remaining_lots")] = self.remaining_lots
        self.row = row
        self.observed_mid = mid(view.market)
        self.spread = row[0]
        return Intent.rest(PROBE_LOTS)


def mid_at(snapshots: list[Snapshot], times: list[int], ts_ns: int) -> float | None:
    index = bisect_right(times, ts_ns) - 1
    return None if index < 0 else mid(snapshots[index])


def probe_day(study: Study, config: ExecConfig, day: int) -> dict:
    """Probes on both sides of a grid. Each one lives until its label horizon or its deadline."""
    events = load_day(study.plan, day)
    rng = random.Random(study.plan.seed * 17 + day)
    session_ns = study.plan.session_s * NS_PER_S
    probe_config = ExecConfig(
        **{**config.__dict__, "decision_interval_ns": 2 * NS_PER_S, "participation_cap_pct": None}
    )
    work = []
    start = 30 * NS_PER_S
    next_id = 0
    while start + (LABEL_HORIZON_S + 1) * NS_PER_S < session_ns:
        for side in (Side.BID, Side.ASK):
            time_left_s = rng.uniform(2, study.parents.horizon_s)
            life_s = min(LABEL_HORIZON_S, time_left_s)
            remaining = rng.randint(PROBE_LOTS, study.parents.target_lots)
            parent = Parent(next_id, side, PROBE_LOTS, start, start + int(life_s * NS_PER_S))
            work.append((parent, Probe(time_left_s, remaining)))
            next_id += 1
        start += PROBE_SPACING_S * NS_PER_S
    runs, snapshots = replay_session(events, work, probe_config)
    times = [s.ts_ns for s in snapshots]
    rows, filled, fraction, moves = [], [], [], []
    for run in runs:
        probe = run.strategy
        if run.excluded or probe.row is None:
            continue
        maker = sum(f.qty for f in run.ledger.fills if f.role.name == "MAKER")
        end_mid = mid_at(snapshots, times, run.parent.deadline_ns - config.obs_latency_ns)
        if end_mid is None:
            continue
        rows.append(probe.row)
        filled.append(maker == PROBE_LOTS)
        fraction.append(maker / PROBE_LOTS)
        moves.append(run.parent.side * (end_mid - probe.observed_mid))
    return {
        "x": np.array(rows),
        "filled": np.array(filled, dtype=int),
        "fraction": np.array(fraction),
        "move": np.array(moves),
        "day": np.full(len(rows), day),
    }


@dataclass(frozen=True)
class ObservedOutcomes:
    """Fate of recorded orders that joined a touch, within the label horizon."""

    orders: int
    fully_executed: int
    cancelled_first: int  # Censored: we never learn if these would have filled.
    still_resting: int
    kaplan_meier_fill: float  # Fill chance by the horizon, treating cancels as censoring.


def observed_outcomes(events: list[Event], horizon_s: int = LABEL_HORIZON_S) -> ObservedOutcomes:
    """Follow every recorded add that joined the best price on its side."""
    replay = ReplayedBook()
    book = replay.book
    horizon_ns = horizon_s * NS_PER_S
    joined: dict[int, tuple[int, int]] = {}  # order_id -> (join ts, qty)
    durations: list[tuple[int, bool]] = []
    last_ts = events[-1].ts_ns if events else 0

    def censor_all(ts_ns: int) -> None:
        for join_ts, _ in joined.values():
            durations.append((min(ts_ns - join_ts, horizon_ns), False))
        joined.clear()

    for e in events:
        if not replay.valid or e.kind == Kind.CLEAR:
            censor_all(e.ts_ns)  # Across a gap or a reset, an order's fate is unknown.
        elif e.kind == Kind.ADD and book.best(e.side) is not None and e.price == book.best(e.side):
            joined[e.order_id] = (e.ts_ns, e.qty)
        elif e.kind in (Kind.EXECUTE, Kind.CANCEL) and e.order_id in joined:
            join_ts, _ = joined[e.order_id]
            gone = book.order_index.get(e.order_id) is not None and book.order_qty(e.order_id) == e.qty
            age = e.ts_ns - join_ts
            if age >= horizon_ns:
                del joined[e.order_id]
                durations.append((horizon_ns, False))
            elif gone:
                del joined[e.order_id]
                durations.append((age, e.kind == Kind.EXECUTE))
        elif e.kind == Kind.MODIFY and e.order_id in joined:
            join_ts, _ = joined.pop(e.order_id)
            durations.append((min(e.ts_ns - join_ts, horizon_ns), False))  # Terms changed: censored.
        replay.apply(e)
    for order_id, (join_ts, _) in joined.items():
        durations.append((min(last_ts - join_ts, horizon_ns), False))
    executed = sum(1 for _, hit in durations if hit)
    cancelled = sum(1 for age, hit in durations if not hit and age < horizon_ns)
    still = sum(1 for age, hit in durations if not hit and age >= horizon_ns)
    return ObservedOutcomes(len(durations), executed, cancelled, still, kaplan_meier(durations, horizon_ns))


def kaplan_meier(durations: list[tuple[int, bool]], horizon_ns: int) -> float:
    """Chance the event happens by horizon_ns, with non-events as right censoring."""
    ordered = sorted(durations)
    survival = 1.0
    at_risk = len(ordered)
    for age, hit in ordered:
        if age > horizon_ns:
            break
        if hit:
            survival *= 1 - 1 / at_risk
        at_risk -= 1
    return 1 - survival
