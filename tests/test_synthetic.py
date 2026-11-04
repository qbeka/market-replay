"""The generator must be deterministic, and its feed must replay cleanly."""

import numpy as np

from marketreplay.book import ReplayedBook
from marketreplay.events import Kind, from_array
from marketreplay.synthetic import StudyPlan, generate_day

PLAN = StudyPlan(days=30, session_s=120, halt_day_every=5, gap_day_every=7)


def test_the_same_day_is_byte_identical():
    first, _ = generate_day(PLAN, 3)
    second, _ = generate_day(PLAN, 3)
    assert np.array_equal(first, second)


def test_different_days_differ():
    assert not np.array_equal(generate_day(PLAN, 1)[0], generate_day(PLAN, 2)[0])


def test_timestamps_never_run_backward():
    for day in (3, 4, 6):
        records, _ = generate_day(PLAN, day)
        assert np.all(np.diff(records["ts_ns"]) >= 0)


def test_a_plain_day_replays_with_no_problems():
    records, labels = generate_day(PLAN, 1)
    assert labels["gap"] is None and labels["halt"] is None
    replay = ReplayedBook()
    for event in from_array(records):
        replay.apply(event)
    assert replay.valid and replay.problems == []


def test_a_gap_day_is_detected_once_and_resynchronized():
    records, labels = generate_day(PLAN, 6)
    assert labels["gap"] is not None
    replay = ReplayedBook()
    for event in from_array(records):
        replay.apply(event)
    assert len(replay.problems) == 1 and replay.problems[0].startswith("gap")
    assert len(replay.invalid_intervals_ns) == 1
    assert replay.valid


def test_a_halt_day_marks_the_halt_and_resumes_valid():
    records, labels = generate_day(PLAN, 4)
    assert labels["halt"] is not None
    kinds = records["kind"].tolist()
    assert kinds.count(int(Kind.HALT_START)) == 1 and kinds.count(int(Kind.HALT_END)) == 1
    replay = ReplayedBook()
    for event in from_array(records):
        replay.apply(event)
    assert replay.valid and replay.problems == []
