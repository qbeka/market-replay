"""Record the test count and coverage from a real run, so no document can overstate them."""

import json
import re
import subprocess
import sys

from .pipeline import RESULTS
from .study import REPO


# The modules that decide fills, cost and strategy behavior. Scripts and reports are not in this number.
ENGINE = {"book", "events", "execution", "ledger", "metrics", "policy", "stats", "strategies", "synthetic"}


def record_evidence() -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--cov=marketreplay", "--cov-report=term"],
        cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    summary = re.search(r"(\d+) passed", done.stdout)
    failed = re.search(r"(\d+) failed", done.stdout)
    coverage = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", done.stdout, re.M)
    if done.returncode != 0 or failed or not summary or not coverage:
        raise SystemExit("The test suite did not pass, so no evidence was recorded.\n" + done.stdout[-1500:])
    engine = [(int(s), int(m)) for name, s, m in re.findall(r"^\S+/(\w+)\.py\s+(\d+)\s+(\d+)\s+\d+%", done.stdout, re.M) if name in ENGINE]
    result = {
        "tests": int(summary.group(1)),
        "coverage_pct": int(coverage.group(1)),
        "engine_coverage_pct": round(100 * (1 - sum(m for _, m in engine) / sum(s for s, _ in engine))),
    }
    (RESULTS / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
