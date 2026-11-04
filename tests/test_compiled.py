"""The compiled replay must equal the Python reference exactly, or it does not count."""

import random
import shutil
from dataclasses import replace

import pytest

from marketreplay.compiled import by_run, python_book_checksum, reference_lines, run_book, run_exec, write_tape
from marketreplay.events import Side, from_array
from marketreplay.execution import ExecConfig, Intent, Parent, run_session
from marketreplay.synthetic import NS_PER_S, StudyPlan, generate_day

pytestmark = pytest.mark.skipif(shutil.which("clang") is None, reason="clang is not installed")

PLAN = StudyPlan(days=30, session_s=240, halt_day_every=5, gap_day_every=7)
PLAIN_DAY, HALT_DAY, GAP_DAY = 1, 4, 6


class Noisy:
    """Random seeded commands, to reach far more engine states than a sensible strategy would."""

    name = "noisy"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def decide(self, view):
        roll = self.rng.random()
        lots = self.rng.randint(1, 30)
        if roll < 0.45:
            return Intent.rest(lots)
        if roll < 0.7:
            return Intent.cross(lots)
        if roll < 0.85:
            return Intent.wait()
        return None


BASE = ExecConfig(
    obs_latency_ns=3_000_000,
    order_latency_ns=8_000_000,
    cancel_latency_ns=6_000_000,
    decision_interval_ns=NS_PER_S,
    record_trace=True,
)

CONFIGS = {
    "base": BASE,
    "capped": replace(BASE, participation_cap_pct=25, participation_free_lots=10),
    "no_cancel_credit": replace(BASE, cancel_ahead_credit_pct=0),
    "half_credit_long_queue": replace(BASE, cancel_ahead_credit_pct=50, queue_ahead_pct=175),
    "trade_through_only": replace(BASE, trade_through_only=True),
    "adverse_and_fees": replace(BASE, adverse_fill_mticks=700, taker_fee_mticks=900, maker_fee_mticks=250),
    "cancel_slower_than_orders": replace(BASE, cancel_latency_ns=250_000_000),
    "order_slower_than_cancel": replace(BASE, order_latency_ns=250_000_000),
}


def setup(tmp_path, day: int, config: ExecConfig):
    records, _ = generate_day(PLAN, day)
    events_path = tmp_path / "events.bin"
    events_path.write_bytes(records.tobytes())
    events = from_array(records)
    rng = random.Random(day)
    work = []
    for i in range(24):
        start = (10 + 8 * i) * NS_PER_S
        side = rng.choice((Side.BID, Side.ASK))
        work.append((Parent(i, side, rng.randint(20, 150), start, start + rng.randint(5, 40) * NS_PER_S), Noisy(i)))
    runs = run_session(events, work, config)
    tape = tmp_path / "tape.txt"
    write_tape(runs, tape)
    return events, events_path, tape, runs


@pytest.mark.parametrize("name", CONFIGS)
@pytest.mark.parametrize("day", [PLAIN_DAY, HALT_DAY, GAP_DAY])
def test_ledgers_are_identical_to_the_reference(tmp_path, name, day):
    config = CONFIGS[name]
    _, events_path, tape, runs = setup(tmp_path, day, config)
    result = run_exec(events_path, tape, config)
    assert result.lines == by_run(reference_lines(runs))


def test_the_gap_and_halt_days_really_exclude_parents(tmp_path):
    excluded = set()
    for day in (HALT_DAY, GAP_DAY):
        _, _, _, runs = setup(tmp_path, day, BASE)
        excluded |= {run.excluded for run in runs if run.excluded}
    assert excluded >= {"halt", "invalid book"}


def test_book_states_match_after_every_event(tmp_path):
    for day in (PLAIN_DAY, HALT_DAY, GAP_DAY):
        events, events_path, _, _ = setup(tmp_path, day, BASE)
        assert run_book(events_path)["checksum"] == python_book_checksum(events)
