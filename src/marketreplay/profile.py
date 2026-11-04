"""Speed and memory of the reference and the compiled replay, on one recorded session.

Every measurement runs in its own single-threaded process, five times, and the median is
reported with the range. Peak memory comes from /usr/bin/time, which reports the whole
process. The Python number includes the interpreter and numpy.
"""

import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .book import ReplayedBook
from .compiled import build, run_book, run_exec, write_tape
from .events import from_array
from .execution import replay_session
from .pipeline import RESULTS, load_fitted
from .roster import build as build_strategy
from .study import Study, load_day, load_records, parents_for_day, plan_folder

REPEATS = 5
PROFILE_DAY = 100


class Timed:
    """Wraps a strategy and records how long each decision took."""

    def __init__(self, inner):
        self.inner = inner
        self.name = inner.name
        self.seconds: list[float] = []

    def decide(self, view):
        start = time.perf_counter()
        intent = self.inner.decide(view)
        self.seconds.append(time.perf_counter() - start)
        return intent


def _python_book_seconds(day_path: str) -> None:
    """Child process entry. Prints seconds for one guarded replay of the day."""
    from .events import EVENT_DTYPE
    import numpy as np

    events = from_array(np.fromfile(day_path, dtype=EVENT_DTYPE))
    start = time.perf_counter()
    replay = ReplayedBook()
    apply = replay.apply
    for event in events:
        apply(event)
    print(time.perf_counter() - start)


def _peak_rss_bytes(command: list[str]) -> tuple[int, str]:
    done = subprocess.run(["/usr/bin/time", "-l", *command], capture_output=True, text=True, check=True)
    return int(re.search(r"(\d+)\s+maximum resident set size", done.stderr).group(1)), done.stdout


def _spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": len(values)}


def profile(study: Study) -> dict:
    fitted = load_fitted()
    day = PROFILE_DAY
    day_path = plan_folder(study.plan) / f"day_{day:03d}.bin"
    records = load_records(study.plan, day)
    events = load_day(study.plan, day)
    build()

    python_book = []
    python_rss = 0
    for _ in range(REPEATS):
        rss, out = _peak_rss_bytes(
            [sys.executable, "-c", f"from marketreplay.profile import _python_book_seconds as f; f({str(day_path)!r})"]
        )
        python_rss = max(python_rss, rss)
        python_book.append(float(out.strip()))
    compiled_book = [run_book(day_path)["seconds"] for _ in range(REPEATS)]
    compiled_rss, _ = _peak_rss_bytes([str(build()), "book", str(day_path), "1"])

    # The same strategies through both engines, from one recorded command tape.
    config = fitted.config.__class__(**{**fitted.config.__dict__, "record_trace": True})
    names = ["urgency", "immediate", "passive_only", "equal_time"]
    parents = parents_for_day(study, day)
    python_exec = []
    for _ in range(REPEATS):
        fresh = [(p, build_strategy(fitted, n, p)) for p in parents for n in names]
        start = time.perf_counter()
        runs, _ = replay_session(events, fresh, config)
        python_exec.append(time.perf_counter() - start)
    with tempfile.TemporaryDirectory() as folder:
        tape = Path(folder) / "tape.txt"
        write_tape(runs, tape)
        compiled_exec = [run_exec(day_path, tape, config).seconds for _ in range(REPEATS)]

    timed = []
    work = []
    for parent in parents:
        strategy = Timed(build_strategy(fitted, "learned", parent))
        timed.append(strategy)
        work.append((parent, strategy))
    replay_session(events, work, fitted.config)
    decisions = sorted(s for strategy in timed for s in strategy.seconds)

    count = len(records)
    python_book_s = _spread(python_book)
    compiled_book_s = _spread(compiled_book)
    python_exec_s = _spread(python_exec)
    compiled_exec_s = _spread(compiled_exec)
    payload = {
        "day": day,
        "events": count,
        "strategies_in_exec_run": len(names),
        "parents_in_exec_run": len(parents),
        "python_book": {**python_book_s, "events_per_s": count / python_book_s["median"]},
        "compiled_book": {**compiled_book_s, "events_per_s": count / compiled_book_s["median"]},
        "python_exec": {**python_exec_s, "events_per_s": count / python_exec_s["median"]},
        "compiled_exec": {**compiled_exec_s, "events_per_s": count / compiled_exec_s["median"]},
        "speedup_book": python_book_s["median"] / compiled_book_s["median"],
        "speedup_exec": python_exec_s["median"] / compiled_exec_s["median"],
        "peak_rss_mb": {"python_book": python_rss / 2**20, "compiled_book": compiled_rss / 2**20},
        "policy_decision_us": {
            "count": len(decisions),
            "p50": decisions[len(decisions) // 2] * 1e6,
            "p95": decisions[int(len(decisions) * 0.95)] * 1e6,
        },
    }
    (RESULTS / "profile.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload
