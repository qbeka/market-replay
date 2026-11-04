"""Drive compiled/replay.c and compare its results with the Python reference.

Strategies remain Python. A recorded run gives the exact commands each one sent, so
the compiled replay is checked on book and execution logic, with no strategy in the way.
"""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .events import Event, Side
from .execution import ExecConfig, Run

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "compiled" / "replay.c"
BINARY = REPO / "compiled" / "build" / "replay"
FLAGS = ["-O2", "-std=c11", "-Wall", "-Wextra", "-Werror"]


def build() -> Path:
    """Compile if the binary is older than its source."""
    if not BINARY.exists() or BINARY.stat().st_mtime < SOURCE.stat().st_mtime:
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["clang", *FLAGS, "-o", str(BINARY), str(SOURCE)], check=True)
    return BINARY


def write_tape(runs: list[Run], path: Path) -> None:
    """Every parent, then each run's commands in the order Python scheduled them."""
    lines = [f"R {len(runs)}"]
    for index, run in enumerate(runs):
        p = run.parent
        lines.append(f"P {index} {int(p.side)} {p.target_lots} {p.start_ns} {p.deadline_ns}")
    for index, run in enumerate(runs):
        for entry in run.trace:
            if entry["what"] == "send":
                immediate = int(entry["immediate"])
                lines.append(f"S {index} {entry['ts_ns']} {entry['child']} {entry['price']} {entry['lots']} {immediate}")
            elif entry["what"] == "cancel_request":
                lines.append(f"C {index} {entry['ts_ns']} {entry['child']}")
    path.write_text("\n".join(lines) + "\n")


def options(config: ExecConfig) -> list[str]:
    values = {
        "taker_fee": config.taker_fee_mticks,
        "maker_fee": config.maker_fee_mticks,
        "credit_pct": config.cancel_ahead_credit_pct,
        "ahead_pct": config.queue_ahead_pct,
        "adverse": config.adverse_fill_mticks,
        "cap_pct": -1 if config.participation_cap_pct is None else config.participation_cap_pct,
        "cap_window": config.participation_window_ns,
        "free_lots": config.participation_free_lots,
        "penalty_ticks": config.completion_penalty_ticks,
        "extra_ticks": config.unfilled_extra_ticks,
        "take_levels": config.book_levels * 4,
        "trade_through_only": int(config.trade_through_only),
        "size_up_loses": int(config.priority.size_increase_loses_priority),
        "size_down_loses": int(config.priority.size_decrease_loses_priority),
        "order_latency": config.order_latency_ns,
        "cancel_latency": config.cancel_latency_ns,
    }
    return [f"{key}={value}" for key, value in values.items()]


def reference_lines(runs: list[Run]) -> list[str]:
    """The Python ledgers, in the compiled program's line format."""
    lines = []
    for index, run in enumerate(runs):
        # An excluded parent can hold partial fills from before a halt or a gap. They
        # stay in the comparison, because the two engines must agree on them too.
        for f in run.ledger.fills if run.ledger else ():
            lines.append(
                f"F {index} {f.ts_ns} {f.child_id} {f.price} {f.qty} {int(f.role)} {f.fee_mticks} {f.penalty_mticks} {f.source_seq}"
            )
    for index, run in enumerate(runs):
        if run.excluded:
            lines.append(f"E {index} {run.excluded}")
        else:
            ledger = run.ledger
            lines.append(
                f"L {index} {ledger.filled_lots} {ledger.notional_ticks} {ledger.fees_mticks} {ledger.penalties_mticks} {ledger.cash_mticks}"
            )
    return lines


def by_run(lines: list[str]) -> list[str]:
    """Fills from all runs interleave in time. Sort stably by run so the two forms line up."""
    fills = [line for line in lines if line.startswith("F")]
    fills.sort(key=lambda line: int(line.split()[1]))
    return fills + [line for line in lines if not line.startswith("F")]


@dataclass(frozen=True)
class ExecResult:
    lines: list[str]
    seconds: float

    @property
    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.lines).encode()).hexdigest()


def run_exec(events_path: Path, tape_path: Path, config: ExecConfig) -> ExecResult:
    done = subprocess.run(
        [str(build()), "exec", str(events_path), str(tape_path), *options(config)],
        capture_output=True, text=True, check=True,
    )  # fmt: skip
    seconds = float(done.stderr.split()[-1])
    return ExecResult(by_run(done.stdout.splitlines()), seconds)


def run_book(events_path: Path, repeats: int = 1) -> dict:
    done = subprocess.run(
        [str(build()), "book", str(events_path), str(repeats)], capture_output=True, text=True, check=True
    )
    fields = done.stdout.split()
    return {"events": int(fields[1]), "seconds": float(fields[5]), "checksum": fields[7]}


FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
MASK = 2**64 - 1


def _mix(value: int, state: int) -> int:
    return ((state ^ (value & MASK)) * FNV_PRIME) & MASK


def python_book_checksum(events: list[Event]) -> str:
    """The same checksum, computed from the reference book after every event."""
    from .book import ReplayedBook

    replay = ReplayedBook()
    state = FNV_OFFSET
    book = replay.book
    for e in events:
        replay.apply(e)
        bid, ask = book.best(Side.BID), book.best(Side.ASK)
        state = _mix(-1 if bid is None else bid, state)
        state = _mix(-1 if ask is None else ask, state)
        state = _mix(0 if bid is None else sum(book.levels[Side.BID][bid].values()), state)
        state = _mix(0 if ask is None else sum(book.levels[Side.ASK][ask].values()), state)
        state = _mix(int(replay.valid), state)
    return f"{state:016x}"


