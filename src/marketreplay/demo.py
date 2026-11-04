"""Build the offline website demo from a recorded run. One HTML file, no network.

Scenes are chosen by rule from the raw test results, so they cannot be cherry-picked:

    typical     the parent whose paired difference is closest to the median
    worst loss  the parent where the learned policy lost the most to the baseline

Every number on the page comes from the replay of that parent.
"""

import json
from dataclasses import replace

import numpy as np

from .evaluate import read_rows
from .execution import replay_session
from .features import mid
from .metrics import shortfall_bps
from .pipeline import FREEZE_PATH, RESULTS, load_fitted
from .roster import build
from .study import REPO, Study, load_day, parents_for_day

TEMPLATE = REPO / "site" / "template.html"
OUTPUT = REPO / "site" / "index.html"


def choose_scenes(rows, baseline: str) -> list[dict]:
    pairs = {}
    for row in rows:
        if row.strategy in ("learned", baseline) and not row.excluded:
            pairs.setdefault((row.day, row.parent_id), {})[row.strategy] = row.shortfall_bps
    differences = [
        (pair["learned"] - pair[baseline], key) for key, pair in pairs.items() if len(pair) == 2
    ]
    values = np.array([d for d, _ in differences])
    median = float(np.median(values))
    typical = min(differences, key=lambda item: (abs(item[0] - median), item[1]))
    worst = max(differences, key=lambda item: (item[0], item[1]))
    return [
        {"title": "A typical order", "rule": "The parent closest to the median paired difference", "key": typical[1]},
        {"title": "The worst loss", "rule": "The parent where the learned policy lost the most", "key": worst[1]},
    ]


def fill_rows(run, start_ns: int) -> list[dict]:
    ledger = run.ledger
    arrival = ledger.arrival_mid_x2 / 2
    running = 0.0
    rows = []
    for f in ledger.fills:
        running += ledger.side * (f.price - arrival) * f.qty + (f.fee_mticks + f.penalty_mticks) / 1000
        rows.append(
            {
                "t": round((f.ts_ns - start_ns) / 1e9, 3),
                "price": f.price,
                "qty": f.qty,
                "role": f.role.name.lower(),
                "cost_bps": round(10_000 * running / (ledger.target_lots * arrival), 3),
            }
        )
    return rows


def describe(entry: dict, start_ns: int) -> dict:
    keep = {k: v for k, v in entry.items() if k not in ("ts_ns", "what", "info", "child")}
    if "info" in entry and entry["info"]:
        keep.update(entry["info"])
    return {"t": round((entry["ts_ns"] - start_ns) / 1e9, 3), "what": entry["what"], **keep}


def scene_data(study: Study, fitted, scene: dict, baseline: str) -> dict:
    day, parent_id = scene["key"]
    parent = next(p for p in parents_for_day(study, day) if p.parent_id == parent_id)
    config = replace(fitted.config, record_trace=True)
    names = ["learned", baseline]
    work = [(parent, build(fitted, name, parent)) for name in names]
    runs, snapshots = replay_session(load_day(study.plan, day), work, config)
    start, end = parent.start_ns, parent.deadline_ns
    book = [
        {
            "t": round((s.ts_ns - start) / 1e9, 3),
            "bids": [list(level) for level in s.bids],
            "asks": [list(level) for level in s.asks],
            "mid": mid(s),
        }
        for s in snapshots
        if start <= s.ts_ns <= end
    ]
    policies = []
    for name, run in zip(names, runs):
        run.ledger.check_conservation()
        events = [describe(e, start) for e in run.trace if e["what"] != "queue"]
        queue = [
            {"t": round((e["ts_ns"] - start) / 1e9, 3), "ahead": e["queue_ahead"], "price": e["price"]}
            for e in run.trace
            if e["what"] == "queue"
        ]
        policies.append(
            {
                "name": name,
                "shortfall_bps": round(shortfall_bps(run.ledger), 3),
                "fills": fill_rows(run, start),
                "events": events,
                "queue": queue,
            }
        )
    return {
        "title": scene["title"],
        "rule": scene["rule"],
        "day": day,
        "parent_id": parent_id,
        "side": "buy" if parent.side > 0 else "sell",
        "target_lots": parent.target_lots,
        "horizon_s": (end - start) / 1e9,
        "arrival_mid": runs[0].ledger.arrival_mid_x2 / 2,
        "book": book,
        "policies": policies,
    }


def write_demo(study: Study) -> str:
    frozen = json.loads(FREEZE_PATH.read_text())
    baseline = frozen["best_baseline"]
    fitted = load_fitted()
    rows = read_rows(RESULTS / "raw" / "test.csv")
    scenes = [scene_data(study, fitted, scene, baseline) for scene in choose_scenes(rows, baseline)]
    final = json.loads((RESULTS / "final.json").read_text())
    payload = {
        "scenes": scenes,
        "baseline": baseline,
        "commit": final["manifest"]["commit"],
        "input_sha256": final["manifest"]["input_sha256"],
        "summary": {
            "days": final["verdict"]["paired"]["days"],
            "parents": final["summaries"]["learned"]["parents"],
            "mean_diff_bps": final["verdict"]["paired"]["mean_diff_bps"],
            "ci_low": final["verdict"]["paired"]["ci_low"],
            "ci_high": final["verdict"]["paired"]["ci_high"],
            "improved": final["verdict"]["improved"],
        },
    }
    (RESULTS / "demo_run.json").write_text(json.dumps(payload, indent=1) + "\n")
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(payload, separators=(",", ":")))
    OUTPUT.write_text(html)
    return f"Wrote {OUTPUT.relative_to(REPO)}, {len(html) // 1024} KB, {len(scenes)} scenes"
