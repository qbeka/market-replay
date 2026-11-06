"""Build the offline website demo from a recorded run. One HTML file, no network.

Scenes are chosen by rule from the raw test results, so they cannot be cherry-picked:

    typical     the median parent among those where the two strategies differed
    biggest win the parent where the learned policy gained the most on the baseline
    worst loss  the parent where it lost the most

Every number on the page comes from the replay of that parent. The page draws only the
frames made here, one for each moment a recorded event changed something, so a trade
lands at its recorded time however smoothly the clock moves between frames.
"""

import base64
import json
from dataclasses import replace

import numpy as np

from .evaluate import read_rows
from .execution import replay_session
from .metrics import shortfall_bps
from .pipeline import FREEZE_PATH, RESULTS, load_fitted
from .roster import build
from .study import REPO, Study, load_day, parents_for_day

TEMPLATE = REPO / "site" / "template.html"
OUTPUT = REPO / "site" / "index.html"
FONTS = {
    "/*GEIST_SERIF*/": REPO / "site" / "fonts" / "GeistSerif-Regular.woff2",
    "/*GEIST_REGULAR*/": REPO / "site" / "fonts" / "Geist-Regular.woff2",
    "/*GEIST_MEDIUM*/": REPO / "site" / "fonts" / "Geist-Medium.woff2",
}
LABELS = {
    "learned": "Learned policy",
    "immediate": "Immediate cross",
    "equal_time": "Equal time",
    "volume_schedule": "Volume schedule",
    "passive_only": "Passive only",
    "urgency": "Urgency rule",
    "fitted_q": "Fitted Q",
    "hv_random": "Random",
    "oracle_volume": "Oracle schedule",
}
SHORT = {"learned": "Learned", "urgency": "Urgency"}
MOVED = {
    "participation_10pct": "a 10% participation cap",
    "latency_plus_50ms": "50 ms of added latency",
    "latency_plus_250ms": "250 ms of added latency",
}
SETTLE_S = 0.05  # A decision's own orders land inside this window, at the recorded latency.
THIN_SPREAD_TICKS = 10
MINUS = "\u2212"


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
        {"title": "Typical order", "rule": f"This is the median of the {share:.0f}% of orders where the two acted differently", "key": typical[1]},
        {"title": "Biggest win", "rule": "This is the order where the learned policy finished furthest ahead", "key": best[1]},
        {"title": "Worst loss", "rule": "This is the order where the learned policy finished furthest behind", "key": worst[1]},
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


def signed(value: float, places: int = 2) -> str:
    sign = "+" if value > 0 else MINUS if value < 0 else ""
    return f"{sign}{abs(value):.{places}f}"


def lots_word(lots: int) -> str:
    return f"{lots} lot" if lots == 1 else f"{lots} lots"


def price_words(prices: list[int]) -> str:
    low, high = min(prices), max(prices)
    if low == high:
        return str(low)
    return f"{low} and {high}" if len(set(prices)) == 2 else f"{low} to {high}"


def why(name: str, entry: dict, limits: dict) -> str:
    """The recorded reason for a decision, as one sentence."""
    if not entry.get("intent"):
        return "It has nothing left to send."
    action, lots = entry["intent"]
    if name != "learned":
        limit = limits["urgency_behind"]
        return f"It is {lots} lots behind schedule, past its limit of {limit}." if action == "CROSS" else f"It stays within {limit} lots of schedule, so it rests."
    if entry.get("rule") == "final":
        return f"In the last {limits['final_s']:g} seconds it crosses what is left."
    if entry.get("rule") == "behind":
        return f"It is {entry['behind_lots']} lots behind schedule, past its limit of {limits['behind_lots']}."
    if entry.get("rule") == "model":
        rest, cross = (signed(entry["costs"][k]).removeprefix("+") for k in ("rest", "cross"))
        return f"At a fill chance of {entry['p_fill']:.0%}, it prices resting at {rest} ticks and crossing at {cross}."
    return "The book is one-sided, so it rests."


def fill_sentence(side: str, fills: list[dict], asked: int | None) -> str:
    """What the fills since the last decision came to."""
    done, plain = ("Sold", "sell") if side == "sell" else ("Bought", "buy")
    forced = [f for f in fills if f["role"] == "forced"]
    if forced:
        return f"Forced to {plain} {lots_word(sum(f['qty'] for f in forced))} at {price_words([f['price'] for f in forced])}."
    lots = sum(f["qty"] for f in fills)
    where = price_words([f["price"] for f in fills])
    if asked and lots < asked:
        return f"{done} {lots} of the {asked} lots it asked for, at {where}."
    return f"{done} {lots_word(lots)} at {where}."


def build_frames(name: str, events: list[dict], queue: list[dict], fills: list[dict], side: str, target: int, limits: dict) -> list[dict]:
    """Every state the page can show for one trader, each at the recorded time it began."""
    samples = [{**q, "what": "queue"} for q in queue]
    # Sorting is stable, so events keep their recorded order and a sample follows them.
    stream = sorted(events + samples, key=lambda e: e["t"])
    rows = iter(fills)
    frames: list[dict] = []
    action, lots, reason, cost, left = None, None, "", 0.0, target
    resting, resting_lots, ahead, sent = None, 0, 0, None
    window: list[dict] = []
    for entry in stream:
        what = entry["what"]
        if what == "decision":
            action, lots = entry["intent"] or ("HOLD", None)
            reason, window = why(name, entry, limits), []
        elif what == "send":
            sent = entry
        elif what == "rest":
            resting, resting_lots, ahead = sent["price"], sent["lots"], entry["queue_ahead"]
        elif what == "queue":
            if resting != entry["price"]:
                resting, resting_lots = entry["price"], sent["lots"]
            ahead = entry["ahead"]
        elif what == "cancelled":
            resting, resting_lots = None, 0
        elif what == "fill":
            row = next(rows)
            cost, left = row["cost_bps"], left - row["qty"]
            window = [*window, row]
            if row["role"] == "maker":
                resting_lots, ahead = resting_lots - row["qty"], 0
                if resting_lots <= 0:
                    resting = None
        elif what == "complete":
            forced = sum(f["qty"] for f in fills if f["role"] == "forced")
            action, lots, resting = "DONE", None, None
            reason = f"The deadline forced {lots_word(forced)}." if forced else "The deadline forced nothing."
        if window:
            said = fill_sentence(side, window, lots if action == "CROSS" else None)
        elif resting is not None:
            said = f"Resting at {resting}, " + ("first in the queue." if ahead == 0 else f"with {lots_word(ahead)} ahead.")
        elif action == "CROSS":
            said = "Nothing has filled yet."
        elif sent and not sent["immediate"]:
            said = f"On its way to {sent['price']}."
        else:
            said = ""
        frame = {
            "t": entry["t"], "action": action, "lots": lots, "why": reason, "said": said, "left": left,
            "cost": cost, "rest": resting, "hit": window[-1]["price"] if window else None,
        }  # fmt: skip
        if frames and frames[-1]["t"] == frame["t"]:
            frames[-1] = frame
        elif not frames or any(frames[-1][k] != v for k, v in frame.items() if k != "t"):
            frames.append(frame)
    return frames


def decision_stops(policies: list[dict]) -> list[dict]:
    """Decisions where one trader rested and the other crossed, once both orders landed."""
    first, second = ([e for e in p["events"] if e["what"] == "decision"] for p in policies)
    stops = []
    for a, b in zip(first, second):
        if {(e["intent"] or ("HOLD",))[0] for e in (a, b)} == {"REST", "CROSS"}:
            landed = [f["t"] for p in policies for f in p["frames"] if a["t"] <= f["t"] < a["t"] + SETTLE_S]
            stops.append({"t": max(landed), "decided": a["t"]})
    return stops


def result_sentence(policies: list[dict]) -> str:
    learned, other = policies
    gap = other["shortfall_bps"] - learned["shortfall_bps"]
    return f"Learned finished {abs(gap):.2f} bps {'better' if gap > 0 else 'worse'} than {other['short']}"


def scene_data(study: Study, fitted, scene: dict, baseline: str, limits: dict) -> dict:
    day, parent_id = scene["key"]
    parent = next(p for p in parents_for_day(study, day) if p.parent_id == parent_id)
    config = replace(fitted.config, record_trace=True)
    names = ["learned", baseline]
    work = [(parent, build(fitted, name, parent)) for name in names]
    runs, snapshots = replay_session(load_day(study.plan, day), work, config)
    start, end = parent.start_ns, parent.deadline_ns
    # The last snapshot before arrival is what the first decision saw.
    first = max(s.ts_ns for s in snapshots if s.ts_ns <= start and s.bids and s.asks)
    book = [
        {
            "t": round((s.ts_ns - start) / 1e9, 3),
            "bids": [list(level) for level in s.bids],
            "asks": [list(level) for level in s.asks],
            "bid": s.bids[0][0],
            "ask": s.asks[0][0],
        }
        for s in snapshots
        if first <= s.ts_ns <= end and s.bids and s.asks
    ]
    side = "buy" if parent.side > 0 else "sell"
    policies = []
    for name, run in zip(names, runs):
        run.ledger.check_conservation()
        events = [describe(e, start) for e in run.trace if e["what"] != "queue"]
        queue = [
            {"t": round((e["ts_ns"] - start) / 1e9, 3), "ahead": e["queue_ahead"], "price": e["price"]}
            for e in run.trace
            if e["what"] == "queue"
        ]
        fills = fill_rows(run, start)
        policies.append(
            {
                "name": name,
                "label": LABELS[name],
                "short": SHORT[name],
                "shortfall_bps": round(shortfall_bps(run.ledger), 3),
                "fills": fills,
                "events": events,
                "frames": build_frames(name, events, queue, fills, side, parent.target_lots, limits),
            }
        )
    spreads = sorted(row["ask"] - row["bid"] for row in book)
    spread = spreads[len(spreads) // 2]
    thin = f", in a thin market with a median spread of {spread} ticks" if spread >= THIN_SPREAD_TICKS else ""
    return {
        "title": scene["title"],
        "rule": scene["rule"] + thin,
        "day": day,
        "parent_id": parent_id,
        "side": side,
        "target_lots": parent.target_lots,
        "horizon_s": (end - start) / 1e9,
        "arrival_mid": runs[0].ledger.arrival_mid_x2 / 2,
        "book": book,
        "policies": policies,
        "start": max(next(f["t"] for f in p["frames"] if f["rest"] is not None) for p in policies),
        "stops": decision_stops(policies),
        "result": result_sentence(policies),
    }


def compact(value: float) -> str:
    """11648571 -> 11.6M. Tile numbers stay short, with one decimal at most."""
    for size, suffix in ((1e6, "M"), (1e3, "K")):
        if abs(value) >= size:
            return f"{value / size:.1f}".removesuffix(".0") + suffix
    return f"{value:.0f}"


def verdict_sentence(verdict: dict, baseline: str) -> str:
    paired = verdict["paired"]
    name = baseline.replace("_", " ")
    span = f"{paired['mean_diff_bps']:+.3f} bps (95% interval {paired['ci_low']:.3f} to {paired['ci_high']:.3f}), resampling {paired['days']} days"
    if verdict["improved"]:
        saved = f"{-paired['mean_diff_bps']:.3f} bps per order (95% interval {-paired['ci_high']:.3f} to {-paired['ci_low']:.3f}), resampling {paired['days']} days"
        return f"The learned policy passed the rule frozen before the test. It saved {saved} against the {name} rule, with a smaller tail and far less forced completion."
    failed = ", ".join(key.replace("_", " ") for key, ok in verdict["checks"].items() if not ok)
    return f"The learned policy did not pass the rule frozen before the test. Against the {name} rule it averaged {span}. Failed: {failed}. A negative result is a valid finding."


def moved_sentence(name: str, entry: dict) -> str:
    """One stress scenario that changed the conclusion, stated plainly."""
    moved, c = MOVED.get(name, name.replace("_", " ")), entry["learned_vs_baseline"]
    span = f"{signed(c['mean_diff_bps'], 3)} bps with a 95% interval of {signed(c['ci_low'], 3)} to {signed(c['ci_high'], 3)}"
    if c["ci_low"] > 0:
        return f"With {moved} the result reverses, and the learned policy averages {span} against the urgency rule."
    return f"With {moved} the interval reaches zero, at {span} against the urgency rule."


def page_content(final: dict, frozen: dict, stress: dict, profile: dict, fuzz: dict, evidence: dict) -> dict:
    """Every sentence and number in Details, read from saved results."""
    baseline = frozen["best_baseline"]
    summaries = final["summaries"]
    order = sorted((n for n in summaries if n != "oracle_volume"), key=lambda n: summaries[n]["mean_bps"])
    learned, other = summaries["learned"], summaries[baseline]
    verdict, paired = final["verdict"], final["verdict"]["paired"]
    immediate = next(c for c in final["comparisons"] if c["b"] == "immediate")
    orders = f"{learned['parents']} paired simulated orders over {learned['days']} test days"
    changed = {n: e for n, e in stress["scenarios"].items() if n != "base" and e["conclusion_changes"]}
    moved = len(stress["scenarios"]) - 1
    sparse = final["regimes"]["sparse"]["learned_vs_best_baseline"]
    fitted_q = summaries["fitted_q"]
    return {
        "about": [
            "Two simulated traders work the same order on the same simulated market. To rest is to wait in the queue for a better price, and to cross is to pay the spread and fill now.",
            f"The learned policy uses a fill model to price both choices at every decision, and the urgency rule rests until it falls more than {frozen['urgency_behind_lots']} lots behind an even schedule.",
            "Playback runs at four times simulated speed and stops wherever one trader rests while the other crosses. Trades land at their recorded times, the book updates once a second as it was recorded, and a negative cost means the fills beat the price at arrival.",
        ],
        "verdict": verdict_sentence(verdict, baseline),
        "results": [
            {"name": "Frozen rule", "value": "Passed" if verdict["improved"] else "Did not pass", "scope": f"Learned policy against the urgency rule on {orders}."},
            {"name": "Mean cost", "value": f"{-paired['mean_diff_bps']:.3f} bps lower", "scope": f"95% interval {-paired['ci_high']:.3f} to {-paired['ci_low']:.3f} bps, resampling whole days, against the urgency rule."},
            {"name": "Forced lots", "value": f"{learned['forced_lot_pct']:.1f}% against {other['forced_lot_pct']:.1f}%", "scope": "Share of lots finished by the deadline rule, against the urgency rule on the same orders."},
            {"name": "95th percentile cost", "value": f"{signed(learned['p95_bps'])} against {signed(other['p95_bps'])} bps", "scope": "Against the urgency rule on the same orders."},
            {"name": "Crossing at once", "value": f"{-immediate['mean_diff_bps']:.2f} bps lower", "scope": f"Mean cost of the learned policy against immediate crossing, 95% interval {-immediate['ci_high']:.2f} to {-immediate['ci_low']:.2f} bps."},
            {"name": "Replay speed", "value": f"{compact(profile['compiled_book']['events_per_s'])} events per second", "scope": f"The C book on one core, best of {profile['compiled_book']['runs']} runs on a shared machine, {profile['speedup_book']:.1f}x the Python reference."},
            {"name": "Fuzzing", "value": f"{fuzz['ledger_divergences'] + fuzz['book_divergences']} divergences", "scope": f"C against the Python reference in {fuzz['cases']:,} cases under AddressSanitizer and UBSan."},
            {"name": "Tests", "value": f"{evidence['tests']} passing", "scope": f"{evidence['engine_coverage_pct']}% line coverage of the engine and {evidence['coverage_pct']}% of the package when the study was recorded."},
        ],
        "strategies": {
            "scope": f"Mean and 95th percentile cost in bps on {orders}, with the share of lots forced at the deadline. Positive numbers are costs.",
            "rows": [
                {"name": LABELS[n], "mean": signed(summaries[n]["mean_bps"]), "p95": signed(summaries[n]["p95_bps"]), "forced": f"{summaries[n]['forced_lot_pct']:.1f}%"}
                for n in order
            ],
        },
        "failed": [
            *[moved_sentence(n, changed[n]) for n in sorted(changed, key=lambda n: (stress["scenarios"][n]["learned_vs_baseline"]["ci_low"] <= 0, n))],
            f"The other {moved - len(changed)} of {moved} moved assumptions left the conclusion in place.",
            f"On the {sparse['days']} sparse test days the difference is unclear, at {signed(sparse['mean_diff_bps'], 3)} bps with a 95% interval of {signed(sparse['ci_low'], 3)} to {signed(sparse['ci_high'], 3)}.",
            f"The fitted Q comparator lost clearly, at {signed(fitted_q['mean_bps'])} bps with {fitted_q['forced_lot_pct']:.1f}% of lots forced, and it had no tuning of its own.",
        ],
        "limits": [
            "The feed is fixed, so a simulated order never moves a price and nobody reacts to it.",
            "The feed is a seeded synthetic market whose signal strength was set by this project, so none of this shows what a real venue would pay.",
            "Sparse synthetic days can show spreads of 50 ticks, far wider than normal, so the largest gains and losses here are not realistic.",
            "Fills come from a queue model, and the stress tests show how much the answer depends on it.",
        ],
    }


def render(payload: dict) -> str:
    """Fill the template. Fonts are embedded so the page makes no network requests."""
    html = TEMPLATE.read_text().replace("/*DATA*/null", json.dumps(payload, separators=(",", ":")))
    for mark, path in FONTS.items():
        html = html.replace(mark, base64.b64encode(path.read_bytes()).decode())
    return html


def write_demo(study: Study) -> str:
    frozen = json.loads(FREEZE_PATH.read_text())
    baseline = frozen["best_baseline"]
    limits = {**frozen["policy"], "urgency_behind": frozen["urgency_behind_lots"]}
    fitted = load_fitted()
    rows = read_rows(RESULTS / "raw" / "test.csv")
    scenes = [scene_data(study, fitted, scene, baseline, limits) for scene in choose_scenes(rows, baseline)]
    final = json.loads((RESULTS / "final.json").read_text())
    stress = json.loads((RESULTS / "stress.json").read_text())
    profile = json.loads((RESULTS / "profile.json").read_text())
    fuzz = json.loads((RESULTS / "fuzz.json").read_text())
    evidence = json.loads((RESULTS / "evidence.json").read_text())
    payload = {"scenes": scenes, "baseline": baseline, **page_content(final, frozen, stress, profile, fuzz, evidence)}
    (RESULTS / "demo_run.json").write_text(json.dumps(payload, indent=1) + "\n")
    html = render(payload)
    OUTPUT.write_text(html)
    return f"Wrote {OUTPUT.relative_to(REPO)}, {len(html) // 1024} KB, {len(scenes)} scenes"
