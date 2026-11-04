"""One-minute check that the whole path works on a tiny study. Nothing here is a result."""

import csv
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from .book import ReplayedBook
from .compiled import by_run, reference_lines, run_exec, write_tape
from .events import Event, Kind, Side
from .execution import ExecConfig, Intent, Parent, ScriptedStrategy, replay_session
from .metrics import shortfall_bps
from .roster import Fitted, build
from .stats import summarize
from .evaluate import evaluate_day
from .study import ParentSpec, REPO, Split, Study, all_session_stats, ensure_data, fit_latencies, fit_volume_curve, load_day, plan_folder, parents_for_day
from .synthetic import StudyPlan

FIXTURES = REPO / "data" / "fixtures"
SMOKE = Study(
    plan=StudyPlan(days=6, session_s=300, halt_day_every=0, gap_day_every=0),
    parents=ParentSpec(target_lots=120, horizon_s=60, first_start_s=30, spacing_s=30),
    split=Split(train=(0, 2), validation=(3, 3), test=(4, 5)),
)


def check_hand_trace() -> int:
    """Replay the hand-worked feed and assert every state in the fixture."""
    expected = json.loads((FIXTURES / "hand_trace_expected.json").read_text())
    rows = (line for line in open(FIXTURES / "hand_trace.csv") if not line.startswith("#"))
    replay = ReplayedBook()
    replay.valid, replay.started, replay.invalid_since_ns = True, True, None
    states = 0
    for row, state in zip(csv.reader(rows), expected["states"]):
        replay.apply(Event(int(row[0]), int(row[1]), Kind[row[2]], Side[row[3]], int(row[4]), int(row[5]), int(row[6])))
        assert replay.valid == state["valid"], f"line {state['line']}"
        if state["valid"]:
            assert [list(r) for r in replay.book.depth(Side.BID, 5)] == state["bids"], f"line {state['line']}"
            assert [list(r) for r in replay.book.depth(Side.ASK, 5)] == state["asks"], f"line {state['line']}"
        states += 1
    return states


def check_hand_execution() -> float:
    """The worked example in tests/test_execution_hand_worked.py, through the normal engine."""
    book = [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.ADD, Side.BID, 1, 100, 4),
        Event(3, 0, Kind.ADD, Side.BID, 2, 100, 3),
        Event(4, 0, Kind.ADD, Side.BID, 6, 99, 10),
        Event(5, 0, Kind.ADD, Side.ASK, 3, 102, 5),
        Event(6, 0, Kind.ADD, Side.ASK, 4, 103, 5),
        Event(7, 0, Kind.SNAPSHOT_END),
        Event(8, 110, Kind.ADD, Side.BID, 5, 100, 2),
        Event(9, 120, Kind.CANCEL, Side.BID, 2, 100, 3),
        Event(10, 130, Kind.EXECUTE, Side.BID, 1, 100, 3),
        Event(11, 140, Kind.EXECUTE, Side.BID, 1, 100, 1),
        Event(12, 150, Kind.EXECUTE, Side.BID, 5, 100, 2),
        Event(13, 165, Kind.EXECUTE, Side.BID, 6, 99, 3),
        Event(14, 1100, Kind.ADD, Side.BID, 7, 98, 1),
    ]
    config = ExecConfig(
        obs_latency_ns=10, order_latency_ns=5, cancel_latency_ns=7, taker_fee_mticks=2000, maker_fee_mticks=-500,
        decision_interval_ns=20, snapshot_interval_ns=10,
    )  # fmt: skip
    script = {100: Intent.rest(6), 160: Intent.wait(), 180: Intent.cross(3), 200: Intent.cross(3)}
    [run], _ = replay_session(book, [(Parent(1, Side.BID, 11, 100, 1000), ScriptedStrategy(script))], config)
    run.ledger.check_conservation()
    assert run.ledger.cash_mticks == -1_122_500
    return shortfall_bps(run.ledger)


def check_compiled(fitted: Fitted) -> int:
    day = 4
    config = replace(fitted.config, record_trace=True)
    work = [(p, build(fitted, n, p)) for p in parents_for_day(SMOKE, day) for n in ("urgency", "immediate", "passive_only")]
    runs, _ = replay_session(load_day(SMOKE.plan, day), work, config)
    with tempfile.TemporaryDirectory() as folder:
        tape = Path(folder) / "tape.txt"
        write_tape(runs, tape)
        result = run_exec(plan_folder(SMOKE.plan) / f"day_{day:03d}.bin", tape, config)
    expected = by_run(reference_lines(runs))
    assert result.lines == expected, "compiled ledgers differ from the reference"
    return len(expected)


def run_smoke() -> None:
    started = time.perf_counter()
    print(f"hand-worked book: {check_hand_trace()} states match")
    print(f"hand-worked execution: {check_hand_execution():.4f} bps, cash conserved")
    ensure_data(SMOKE.plan)
    stats = all_session_stats(SMOKE, SMOKE.split.days("train"))
    fitted = Fitted(fit_latencies(stats, SMOKE.exec_config), fit_volume_curve(stats))
    names = ["immediate", "equal_time", "passive_only", "urgency"]
    rows = [r for d in SMOKE.split.days("test") for r in evaluate_day(SMOKE, fitted, names, fitted.config, d)]
    print("tiny synthetic run, 2 days. This is a pipeline check, not a result:")
    for name in names:
        s = summarize(rows, name)
        print(f"  {name:<14} parents {s['parents']:>3}  mean {s['mean_bps']:7.3f} bps  forced {s['forced_lot_pct']:5.1f}%")
    print(f"compiled replay: {check_compiled(fitted)} ledger lines identical to the reference")
    print(f"smoke passed in {time.perf_counter() - started:.1f} s")
