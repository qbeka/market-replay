"""Run strategies over days and turn runs into one flat row per (parent, strategy)."""

import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from threadpoolctl import threadpool_limits

from .execution import ExecConfig, Run, run_session
from .ledger import Role
from .metrics import forced_lots, shortfall_bps
from .roster import Fitted, build
from .study import Study, WORKERS, load_day, parents_for_day

ROW_FIELDS = [
    "day", "parent_id", "side", "strategy", "excluded", "shortfall_bps", "forced_lots",
    "maker_lots", "taker_lots", "fees_mticks", "penalties_mticks",
    "last_voluntary_fill_s",
]  # fmt: skip


@dataclass(frozen=True)
class Row:
    day: int
    parent_id: int
    side: int
    strategy: str
    excluded: str
    shortfall_bps: float
    forced_lots: int
    maker_lots: int
    taker_lots: int
    fees_mticks: int
    penalties_mticks: int
    last_voluntary_fill_s: float


def to_row(day: int, name: str, run: Run) -> Row:
    parent = run.parent
    if run.excluded:
        return Row(day, parent.parent_id, parent.side, name, run.excluded, float("nan"), 0, 0, 0, 0, 0, float("nan"))
    ledger = run.ledger
    ledger.check_conservation()
    by_role = {role: sum(f.qty for f in ledger.fills if f.role == role) for role in Role}
    voluntary = [f.ts_ns for f in ledger.fills if f.role != Role.FORCED]
    last = (max(voluntary) - parent.start_ns) / 1e9 if voluntary else float("nan")
    return Row(
        day, parent.parent_id, parent.side, name, "", shortfall_bps(ledger), forced_lots(ledger),
        by_role[Role.MAKER], by_role[Role.TAKER], ledger.fees_mticks, ledger.penalties_mticks, last,
    )  # fmt: skip


def evaluate_day(study: Study, fitted: Fitted, names: list[str], config: ExecConfig, day: int) -> list[Row]:
    """Every strategy trades every parent of the day in one pass over the feed."""
    events = load_day(study.plan, day)
    parents = parents_for_day(study, day)
    work = [(parent, build(fitted, name, parent)) for parent in parents for name in names]
    labels = [name for _ in parents for name in names]
    runs = run_session(events, work, config)
    return [to_row(day, name, run) for name, run in zip(labels, runs)]


def _one_thread_per_worker() -> None:
    threadpool_limits(1)  # Workers run days in parallel, so models must not also spawn threads.


def evaluate_days(study: Study, fitted: Fitted, names: list[str], config: ExecConfig, days: list[int]) -> list[Row]:
    with ProcessPoolExecutor(WORKERS, initializer=_one_thread_per_worker) as pool:
        per_day = pool.map(partial(evaluate_day, study, fitted, names, config), days)
        return [row for rows in per_day for row in rows]


def write_rows(rows: list[Row], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ROW_FIELDS)
        for row in rows:
            writer.writerow([getattr(row, name) for name in ROW_FIELDS])
