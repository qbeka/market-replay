"""Record the test count and coverage from a real run, so no document can overstate them."""

import json
import re
import subprocess
import sys

from .pipeline import RESULTS
from .study import REPO


def record_evidence() -> dict:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--cov=marketreplay", "--cov-report=term", "-q"],
        cwd=REPO, capture_output=True, text=True,
    )  # fmt: skip
    summary = re.search(r"(\d+) passed", done.stdout)
    failed = re.search(r"(\d+) failed", done.stdout)
    coverage = re.search(r"^TOTAL\s+\d+\s+\d+\s+(\d+)%", done.stdout, re.M)
    if done.returncode != 0 or failed or not summary or not coverage:
        raise SystemExit("The test suite did not pass, so no evidence was recorded.\n" + done.stdout[-1500:])
    result = {"tests": int(summary.group(1)), "coverage_pct": int(coverage.group(1))}
    (RESULTS / "evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
