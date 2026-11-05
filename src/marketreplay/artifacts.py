"""The artifact manifest: every saved result, what made it and its hash."""

import hashlib
import json

from .study import REPO

MANIFEST = REPO / "results" / "manifest.json"
ARTIFACTS = {
    "results/fit.json": "mreplay fit",
    "results/fitted/fitted.pkl": "mreplay fit",
    "results/tune.json": "mreplay tune",
    "results/raw/validation.csv": "mreplay tune",
    "results/freeze.json": "mreplay freeze",
    "results/final.json": "mreplay final",
    "results/raw/test.csv": "mreplay final",
    "results/stress.json": "mreplay stress",
    "results/raw/stress": "mreplay stress",
    "results/profile.json": "mreplay profile",
    "results/fuzz.json": "mreplay fuzz",
    "results/evidence.json": "mreplay evidence",
    "docs/report.md": "mreplay report",
    "docs/resume.md": "mreplay report",
    "site/index.html": "mreplay demo",
    "results/demo_run.json": "mreplay demo",
}


def entries() -> list[dict]:
    rows = []
    for name, command in ARTIFACTS.items():
        path = REPO / name
        for file in sorted(path.iterdir()) if path.is_dir() else [path]:
            data = file.read_bytes()
            rows.append(
                {
                    "path": str(file.relative_to(REPO)),
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "made_by": command,
                }
            )
    return rows


def write_manifest() -> str:
    rows = entries()
    MANIFEST.write_text(json.dumps({"simulated": True, "artifacts": rows}, indent=1) + "\n")
    return f"Wrote {MANIFEST.relative_to(REPO)} with {len(rows)} files"


def check_manifest() -> str:
    """Name every file that no longer matches, so a rerun can be compared in one command."""
    recorded = {row["path"]: row["sha256"] for row in json.loads(MANIFEST.read_text())["artifacts"]}
    current = {row["path"]: row["sha256"] for row in entries()}
    changed = sorted(path for path in recorded if current.get(path) != recorded[path])
    if changed:
        raise SystemExit("These files differ from the manifest:\n" + "\n".join(changed))
    return f"All {len(recorded)} files match the manifest"
