"""Study definition: inputs on disk, the day split, parent orders and session labels."""

import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path

import numpy as np

from .book import ReplayedBook
from .events import EVENT_DTYPE, Event, Kind, Side, from_array
from .execution import ExecConfig, Parent
from .synthetic import NS_PER_S, StudyPlan, generate_day

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "synthetic"
WORKERS = 8


@dataclass(frozen=True)
class ParentSpec:
    """One grid of same-sized parents per session, on whole even seconds."""

    target_lots: int = 40
    horizon_s: int = 300
    first_start_s: int = 120
    spacing_s: int = 120


@dataclass(frozen=True)
class Split:
    """Contiguous days, inclusive. Days between ranges are an embargo.

    Labels look at most 20 s ahead and stay inside one session, so no label
    crosses a boundary. The embargo day is there because day parameters follow a
    slow random walk, so a neighboring day shares state with the one before it.
    """

    train: tuple[int, int] = (0, 59)
    validation: tuple[int, int] = (61, 84)
    test: tuple[int, int] = (86, 119)

    def days(self, name: str) -> list[int]:
        first, last = getattr(self, name)
        return list(range(first, last + 1))


@dataclass(frozen=True)
class Study:
    plan: StudyPlan = StudyPlan()
    parents: ParentSpec = ParentSpec()
    split: Split = Split()
    exec_config: ExecConfig = field(default_factory=ExecConfig)


def plan_folder(plan: StudyPlan) -> Path:
    key = hashlib.sha256(json.dumps(asdict(plan), sort_keys=True).encode()).hexdigest()[:12]
    return DATA / key


def _write_day(plan: StudyPlan, folder: Path, day: int) -> tuple[int, int, str]:
    records, labels = generate_day(plan, day)
    path = folder / f"day_{day:03d}.bin"
    path.write_bytes(records.tobytes())
    (folder / f"day_{day:03d}.json").write_text(json.dumps(labels, indent=1))
    return day, len(records), hashlib.sha256(records.tobytes()).hexdigest()


def ensure_data(plan: StudyPlan) -> dict:
    """Generate the plan's days once. Returns an index with an event count and hash per day."""
    folder = plan_folder(plan)
    index_path = folder / "index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(partial(_write_day, plan, folder), range(plan.days)))
    index = {
        "plan": asdict(plan),
        "events": sum(r[1] for r in rows),
        "days": {str(day): {"events": n, "sha256": digest} for day, n, digest in rows},
        "sha256": hashlib.sha256("".join(r[2] for r in rows).encode()).hexdigest(),
    }
    index_path.write_text(json.dumps(index, indent=1))
    return index


def load_records(plan: StudyPlan, day: int) -> np.ndarray:
    return np.fromfile(plan_folder(plan) / f"day_{day:03d}.bin", dtype=EVENT_DTYPE)


def load_day(plan: StudyPlan, day: int) -> list[Event]:
    return from_array(load_records(plan, day))


def parents_for_day(study: Study, day: int) -> list[Parent]:
    """The same grid every day. Sides come from a seeded coin, never from the market."""
    spec = study.parents
    rng = random.Random(study.plan.seed * 31 + day)
    parents = []
    start_s = spec.first_start_s
    index = 0
    while start_s + spec.horizon_s < study.plan.session_s:
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        parent_id = day * 1000 + index
        parents.append(
            Parent(parent_id, side, spec.target_lots, start_s * NS_PER_S, (start_s + spec.horizon_s) * NS_PER_S)
        )
        start_s += spec.spacing_s
        index += 1
    return parents


@dataclass(frozen=True)
class SessionStats:
    day: int
    events: int
    traded_lots: int
    realized_vol_ticks: float  # Std of 60 s midpoint changes.
    median_spread_ticks: float
    median_touch_lots: float
    volume_curve: tuple[float, ...]  # Lots per 60 s bucket.
    recv_delay_p50_ns: float
    recv_delay_p95_ns: float


def session_stats(study: Study, day: int) -> SessionStats:
    records = load_records(study.plan, day)
    replay = ReplayedBook()
    next_ts = 0
    mids, spreads, touch = [], [], []
    buckets = [0] * (study.plan.session_s // 60 + 1)
    traded = 0
    for e in from_array(records):
        replay.apply(e)
        if e.kind == Kind.EXECUTE or e.kind == Kind.TRADE_HIDDEN:
            traded += e.qty
            buckets[e.ts_ns // (60 * NS_PER_S)] += e.qty
        if e.ts_ns >= next_ts and replay.valid:
            next_ts = e.ts_ns + NS_PER_S
            bid, ask = replay.book.best(Side.BID), replay.book.best(Side.ASK)
            if bid is not None and ask is not None:
                mids.append((bid + ask) / 2)
                spreads.append(ask - bid)
                touch.append(min(replay.book.depth(Side.BID, 1)[0][1], replay.book.depth(Side.ASK, 1)[0][1]))
    changes = np.diff(np.array(mids)[::60])
    delays = (records["recv_ns"] - records["ts_ns"]).astype(float)
    return SessionStats(
        day,
        len(records),
        traded,
        float(changes.std()),
        float(np.median(spreads)),
        float(np.median(touch)),
        tuple(buckets),
        float(np.percentile(delays, 50)),
        float(np.percentile(delays, 95)),
    )


def all_session_stats(study: Study, days: list[int]) -> list[SessionStats]:
    with ProcessPoolExecutor(WORKERS) as pool:
        return list(pool.map(partial(session_stats, study), days))


def fit_volume_curve(train: list[SessionStats]) -> list[float]:
    """Cumulative fraction of session volume by minute, averaged over earlier days."""
    totals = np.sum([s.volume_curve for s in train], axis=0).astype(float)
    return (np.cumsum(totals) / totals.sum()).tolist()


def fit_latencies(train: list[SessionStats], base: ExecConfig) -> ExecConfig:
    """Observation latency is the 95th percentile feed delay on earlier days.

    The feed cannot reveal order entry latency, so that part is an assumption: the
    observation latency plus a fixed gateway delay of 5 ms in each direction.
    """
    obs_ns = int(np.median([s.recv_delay_p95_ns for s in train]))
    gateway_ns = 5_000_000
    return ExecConfig(
        **{
            **base.__dict__,
            "obs_latency_ns": obs_ns,
            "order_latency_ns": obs_ns + gateway_ns,
            "cancel_latency_ns": obs_ns + gateway_ns,
        }
    )


def fit_regimes(train: list[SessionStats]) -> dict:
    """Thresholds for session labels, fixed on training days before any evaluation."""
    return {
        "volatile_vol_ticks": float(np.percentile([s.realized_vol_ticks for s in train], 80)),
        "sparse_touch_lots": float(np.percentile([s.median_touch_lots for s in train], 20)),
    }


def regime(stats: SessionStats, thresholds: dict) -> str:
    """Sparse wins over volatile because a thin book changes fills more than volatility does."""
    if stats.median_touch_lots <= thresholds["sparse_touch_lots"]:
        return "sparse"
    if stats.realized_vol_ticks >= thresholds["volatile_vol_ticks"]:
        return "volatile"
    return "calm"
