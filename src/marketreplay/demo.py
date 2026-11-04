"""Build the offline website demo from a recorded run. One HTML file, no network.

Scenes are chosen by rule from the raw test results, so they cannot be cherry-picked:

    typical     the median parent among those where the two strategies differed
    biggest win the parent where the learned policy gained the most on the baseline
    worst loss  the parent where it lost the most

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
from .study import REPO, Study, ensure_data, load_day, parents_for_day

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
    disagreed = [item for item in differences if abs(item[0]) >= 0.01]
    median = float(np.median([d for d, _ in disagreed]))
    typical = min(disagreed, key=lambda item: (abs(item[0] - median), item[1]))
    best = min(disagreed, key=lambda item: (item[0], item[1]))
    worst = max(disagreed, key=lambda item: (item[0], item[1]))
    share = 100 * len(disagreed) / len(differences)
    return [
        {
            "title": "A typical order",
            "rule": f"The traders acted differently on {share:.0f}% of orders. This is the median of those orders.",
            "key": typical[1],
        },
        {"title": "The biggest win", "rule": "The order where the learned policy finished furthest ahead of the other trader.", "key": best[1]},
        {"title": "The worst loss", "rule": "The order where the learned policy finished furthest behind the other trader.", "key": worst[1]},
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
            "bid": s.bids[0][0] if s.bids else None,
            "ask": s.asks[0][0] if s.asks else None,
        }
        for s in snapshots
        if start <= s.ts_ns <= end and s.bids and s.asks
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


def session_ridge(snapshots: list, points: int = 120) -> list[float]:
    """One session's midpoint, smoothed and thinned into a calm line behind the page."""
    mids = np.array([m for m in (mid(s) for s in snapshots) if m is not None])
    width = max(1, len(mids) // 40)
    smooth = np.convolve(mids, np.ones(width) / width, mode="valid")
    return [round(float(v), 2) for v in smooth[:: max(1, len(smooth) // points)]]


def compact(value: float) -> str:
    """11648571 -> 11.6M. Tile numbers stay short and keep one decimal."""
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:.1f}{suffix}"
    return f"{value:.0f}"


def verdict_sentence(verdict: dict, baseline: str) -> str:
    paired = verdict["paired"]
    name = baseline.replace("_", " ")
    span = f"{paired['mean_diff_bps']:+.3f} bps (95% interval {paired['ci_low']:.3f} to {paired['ci_high']:.3f}), resampling {paired['days']} days"
    if verdict["improved"]:
        return f"The learned policy passed the rule frozen before the test. Against the {name} rule it averaged {span}, with no worse tail and no more forced completion."
    failed = ", ".join(key.replace("_", " ") for key, ok in verdict["checks"].items() if not ok)
    return f"The learned policy did not pass the rule frozen before the test. Against the {name} rule it averaged {span}. Failed: {failed}. A negative result is a valid finding."


def page_content(final: dict, frozen: dict, stress: dict, profile: dict, fuzz: dict, evidence: dict, events: int) -> dict:
    """Every sentence and number on the page, read from saved results."""
    summaries = final["summaries"]
    order = sorted((n for n in summaries if n != "oracle_volume"), key=lambda n: summaries[n]["mean_bps"])
    learned = summaries["learned"]
    paired = final["verdict"]["paired"]
    changed = sum(1 for e in stress["scenarios"].values() if e["conclusion_changes"])
    return {
        "tiles": [
            {"value": compact(events), "label": "order events replayed"},
            {"value": f"{learned['parents']:,}", "label": f"paired orders, {learned['days']} test days"},
            {"value": f"{compact(profile['compiled_book']['events_per_s'])}/s", "label": "events on one core, in C"},
            {"value": str(fuzz["ledger_divergences"] + fuzz["book_divergences"]), "label": f"divergences in {compact(fuzz['cases'])} fuzz cases"},
        ],
        "verdict": {"sentence": verdict_sentence(final["verdict"], frozen["best_baseline"])},
        "table": {
            "parents": learned["parents"],
            "days": learned["days"],
            "rows": [
                {"name": n, "mean": summaries[n]["mean_bps"], "p95": summaries[n]["p95_bps"], "forced": summaries[n]["forced_lot_pct"]}
                for n in order
            ],
            "note": f"Positive numbers are costs. Forced means lots bought by the deadline rule. Under {len(stress['scenarios']) - 1} moved fill assumptions, the paired result changes sign in {changed}.",
        },
        "checks": [
            f"{evidence['tests']} tests at {evidence['coverage_pct']}% line coverage, starting from a hand-worked trace with an assertion for every book state.",
            f"The compiled C replay matches the Python reference byte for byte, with {fuzz['ledger_divergences'] + fuzz['book_divergences']} divergences in {fuzz['cases']:,} hostile fuzz cases under AddressSanitizer and UBSan.",
            "Strategies receive only delayed data. A test rewrites the future and requires every earlier view to stay identical.",
            f"The hypothesis, split and decision rule were committed before any test day was read. The interval resamples whole days, {paired['days']} of them.",
            "An independent review found three biases before the freeze. Each one became a failing test, then a fix.",
        ],
        "limits": [
            "Market impact. The feed is fixed, so a simulated order never moves a price and nobody reacts to it.",
            "A real venue. The feed is a seeded synthetic market, and its signal strength was set by this project.",
            "Thin markets. Sparse synthetic days can show spreads of 50 ticks, far wider than normal, so their largest gains and losses are not realistic.",
            "Queue reality. Fills come from a queue model, so the stress tests show how much the answer depends on it.",
        ],
    }


def write_demo(study: Study) -> str:
    frozen = json.loads(FREEZE_PATH.read_text())
    baseline = frozen["best_baseline"]
    fitted = load_fitted()
    rows = read_rows(RESULTS / "raw" / "test.csv")
    scenes = [scene_data(study, fitted, scene, baseline) for scene in choose_scenes(rows, baseline)]
    final = json.loads((RESULTS / "final.json").read_text())
    stress = json.loads((RESULTS / "stress.json").read_text())
    profile = json.loads((RESULTS / "profile.json").read_text())
    fuzz = json.loads((RESULTS / "fuzz.json").read_text())
    evidence = json.loads((RESULTS / "evidence.json").read_text())
    day = scenes[0]["day"]
    snapshots = replay_session(load_day(study.plan, day), [], fitted.config)[1]
    payload = {
        "scenes": scenes,
        "baseline": baseline,
        "ridge": session_ridge(snapshots),
        "commit": final["manifest"]["commit"],
        "input_sha256": final["manifest"]["input_sha256"],
        **page_content(final, frozen, stress, profile, fuzz, evidence, ensure_data(study.plan)["events"]),
    }
    (RESULTS / "demo_run.json").write_text(json.dumps(payload, indent=1) + "\n")
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(payload, separators=(",", ":")))
    OUTPUT.write_text(html)
    return f"Wrote {OUTPUT.relative_to(REPO)}, {len(html) // 1024} KB, {len(scenes)} scenes"
