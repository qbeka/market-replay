"""The study, stage by stage. Each stage reads the saved output of the one before it.

    fit       models and schedules from training days
    tune      strategy settings from validation days
    freeze    write the hypothesis, split and decision rule, before the final test
    final     frozen strategies on test days, once
    stress    frozen strategies under other fill assumptions

Test days are only read by final and stress, and both refuse to run unless the frozen
file matches the saved artifacts and is committed.
"""

import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import numpy as np

from .comparator import fit_fitted_q
from .evaluate import Row, evaluate_days, write_rows
from .execution import ExecConfig
from .features import FEATURE_NAMES
from .fillmodel import (
    FAMILIES,
    TreeMoveModel,
    MoveModel,
    brier,
    calibration_table,
    expected_calibration_error,
    tune_family,
)
from .labels import observed_outcomes, probe_day
from .policy import PolicyParams
from .roster import CLIP_LOTS, Fitted
from .stats import bootstrap_mean_ci, paired_day_means, summarize
from .study import (
    REPO,
    Study,
    all_session_stats,
    ensure_data,
    fit_latencies,
    fit_regimes,
    fit_volume_curve,
    load_day,
    regime,
)

RESULTS = REPO / "results"
FITTED_PATH = RESULTS / "fitted" / "fitted.pkl"
FREEZE_PATH = RESULTS / "freeze.json"
BASELINES = ["immediate", "equal_time", "volume_schedule", "passive_only", "urgency", "hv_random"]
SEED = 11


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(study: Study, command: str, seconds: float, extra: dict) -> dict:
    """Everything needed to rerun a result and to tell if it is stale."""
    return {
        "command": command,
        "commit": _git("rev-parse", "HEAD"),
        "code_clean": _git("status", "--porcelain", "--", ".", ":!results", ":!docs/report.md", ":!site/index.html") == "",
        "seeds": {"plan": study.plan.seed, "pipeline": SEED},
        "input_sha256": ensure_data(study.plan)["sha256"],
        "study": json.loads(json.dumps(asdict(study), default=str)),
        "hardware": {
            "machine": platform.machine(),
            "cpu": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
            "cores": os.cpu_count(),
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "elapsed_s": round(seconds, 2),
        **extra,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def collect_probes(study: Study, config: ExecConfig, days: list[int]) -> dict:
    with ProcessPoolExecutor(8) as pool:
        parts = list(pool.map(partial(probe_day, study, config), days))
    return {
        "x": np.vstack([p["x"] for p in parts]),
        "filled": np.concatenate([p["filled"] for p in parts]),
        "move": np.concatenate([p["move"] for p in parts]),
        "day": np.concatenate([p["day"] for p in parts]),
    }


def fit(study: Study) -> dict:
    """Fit everything on training days. Fit, calibrate and score use separate day ranges."""
    started = time.perf_counter()
    days = study.split.days("train")
    stats = all_session_stats(study, days)
    config = fit_latencies(stats, study.exec_config)
    probes = collect_probes(study, config, days)
    x, filled, move, day = probes["x"], probes["filled"], probes["move"], probes["day"]
    fit_days, cal_days, score_days = days[:40], days[40:50], days[50:]
    fit_mask = day <= fit_days[-1]
    cal_mask = (day >= cal_days[0]) & (day <= cal_days[-1])
    score_mask = day >= score_days[0]

    report = {"fill_models": {}}
    models = {}
    for family in FAMILIES:
        model, trial = tune_family(family, x[fit_mask], filled[fit_mask], x[cal_mask], filled[cal_mask], SEED)
        models[family] = model
        predicted = model.predict(x[score_mask])
        report["fill_models"][family] = {
            "trials": len(trial.trials),
            "brier": brier(filled[score_mask], predicted),
            "ece": expected_calibration_error(filled[score_mask], predicted),
            "calibration": calibration_table(filled[score_mask], predicted),
        }
    report["base_rate_brier"] = brier(filled[score_mask], np.full(score_mask.sum(), filled[fit_mask].mean()))
    report["probe_rows"] = {"fit": int(fit_mask.sum()), "calibrate": int(cal_mask.sum()), "score": int(score_mask.sum())}
    report["adverse_selection_ticks"] = {
        "move_after_fill": float(move[score_mask & (filled == 1)].mean()),
        "move_without_fill": float(move[score_mask & (filled == 0)].mean()),
    }
    winner = min(FAMILIES, key=lambda f: report["fill_models"][f]["brier"])
    report["fill_winner"] = winner

    move_scores = {}
    for name, cls in (("ridge", MoveModel), ("trees", TreeMoveModel)):
        candidate = cls().fit(x[fit_mask], move[fit_mask], filled[fit_mask])
        _, overall = candidate.predict(x[score_mask])
        residual = ((move[score_mask] - overall) ** 2).sum()
        total = ((move[score_mask] - move[score_mask].mean()) ** 2).sum()
        move_scores[name] = (float(1 - residual / total), candidate)
    report["move_r2"] = {name: score for name, (score, _) in move_scores.items()}
    move_model = max(move_scores.values(), key=lambda pair: pair[0])[1]

    # Observed outcomes check the probes against real resting orders.
    sample = score_days[:4]
    outcomes = [observed_outcomes(load_day(study.plan, d)) for d in sample]
    report["observed_resting_orders"] = {
        "days": sample,
        "orders": sum(o.orders for o in outcomes),
        "fully_executed": sum(o.fully_executed for o in outcomes),
        "cancelled_first": sum(o.cancelled_first for o in outcomes),
        "still_resting": sum(o.still_resting for o in outcomes),
        "kaplan_meier_fill": float(np.mean([o.kaplan_meier_fill for o in outcomes])),
        "probe_fill_rate": float(filled[score_mask].mean()),
    }

    comparator, transitions = fit_fitted_q(study, config, fit_days, CLIP_LOTS)
    report["comparator_transitions"] = transitions

    fitted = Fitted(
        config=config,
        volume_curve=fit_volume_curve(stats),
        fill_model=models[winner],
        move_model=move_model,
        comparator=comparator,
    )
    regimes = fit_regimes(stats)
    FITTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    FITTED_PATH.write_bytes(pickle.dumps(fitted))
    payload = {**report, "regime_thresholds": regimes, "fitted_sha256": sha256_file(FITTED_PATH)}
    payload["manifest"] = manifest(study, "mreplay fit", time.perf_counter() - started, {})
    write_json(RESULTS / "fit.json", payload)
    return payload


def load_fitted() -> Fitted:
    return pickle.loads(FITTED_PATH.read_bytes())


URGENCY_GRID = [8, 12, 20, 30, 45, 60]
POLICY_GRID = [
    PolicyParams(CLIP_LOTS, behind, catch_up, final, margin)
    for behind in (20, 30)
    for catch_up in (0.0, 0.5)
    for final in (4.0, 8.0)
    for margin in (0.5, 1.5, 99.0)  # 99 means never wait, so the data can reject waiting.
]


def tune(study: Study) -> dict:
    """Pick settings on validation days by mean cost, among those whose p95 stays in reach."""
    started = time.perf_counter()
    fitted = load_fitted()
    fitted.urgency_behind = {f"urgency_{b}": b for b in URGENCY_GRID}
    fitted.policies = {
        f"learned_b{p.behind_lots}_c{p.catch_up_ticks}_f{p.final_s}_w{p.wait_margin_ticks}": p for p in POLICY_GRID
    }
    names = [n for n in BASELINES if n != "urgency"] + list(fitted.urgency_behind) + list(fitted.policies) + ["fitted_q"]
    rows = evaluate_days(study, fitted, names, fitted.config, study.split.days("validation"))
    write_rows(rows, RESULTS / "raw" / "validation.csv")
    summaries = {name: summarize(rows, name) for name in names}

    def pick(prefix: str) -> str:
        group = [s for n, s in summaries.items() if n.startswith(prefix)]
        tail_limit = min(s["p95_bps"] for s in group) + 0.5
        return min((s for s in group if s["p95_bps"] <= tail_limit), key=lambda s: s["mean_bps"])["strategy"]

    best_urgency = pick("urgency_")
    best_policy = pick("learned_")
    tuned = Fitted(
        config=fitted.config,
        volume_curve=fitted.volume_curve,
        fill_model=fitted.fill_model,
        move_model=fitted.move_model,
        comparator=fitted.comparator,
        policies={"learned": fitted.policies[best_policy]},
        urgency_behind={"urgency": fitted.urgency_behind[best_urgency]},
    )
    FITTED_PATH.write_bytes(pickle.dumps(tuned))
    eligible = [n for n in ("immediate", "equal_time", "volume_schedule", "passive_only", best_urgency)]
    best_baseline = min(eligible, key=lambda n: summaries[n]["mean_bps"])
    payload = {
        "summaries": summaries,
        "chosen_urgency": best_urgency,
        "chosen_policy": best_policy,
        "best_baseline": "urgency" if best_baseline == best_urgency else best_baseline,
        "fitted_sha256": sha256_file(FITTED_PATH),
        "manifest": manifest(study, "mreplay tune", time.perf_counter() - started, {"strategies": len(names)}),
    }
    write_json(RESULTS / "tune.json", payload)
    return payload


# Freeze -----------------------------------------------------------------------------

DECISION_RULE = {
    "primary_comparison": "learned against the best causal baseline chosen on validation days",
    "unit_of_replication": "trading day",
    "improved_only_if_all_hold": [
        "95% day-bootstrap interval of mean(learned - baseline) lies entirely below 0 bps",
        "p95 shortfall of learned is at most baseline p95 + 0.5 bps",
        "forced lot percentage of learned is at most baseline's + 1 percentage point",
    ],
    "otherwise": "report no demonstrated improvement. A negative result is a valid finding.",
}

STRESS_STRATEGIES = ["immediate", "equal_time", "passive_only", "urgency", "learned", "fitted_q"]


def stress_scenarios(base: ExecConfig) -> dict[str, ExecConfig]:
    """Every assumption gets moved on its own. The last two move several together."""

    def slower(extra_ms: int) -> ExecConfig:
        extra = extra_ms * 1_000_000
        return replace(
            base,
            obs_latency_ns=base.obs_latency_ns + extra,
            order_latency_ns=base.order_latency_ns + extra,
            cancel_latency_ns=base.cancel_latency_ns + extra,
        )

    return {
        "base": base,
        "latency_plus_50ms": slower(50),
        "latency_plus_250ms": slower(250),
        "latency_plus_900ms": slower(900),
        "taker_fee_zero": replace(base, taker_fee_mticks=0),
        "taker_fee_double": replace(base, taker_fee_mticks=base.taker_fee_mticks * 2),
        "maker_rebate_none": replace(base, maker_fee_mticks=0),
        "maker_fee_charged": replace(base, maker_fee_mticks=100),
        "participation_10pct": replace(base, participation_cap_pct=10),
        "participation_60pct": replace(base, participation_cap_pct=60),
        "cancel_credit_50pct": replace(base, cancel_ahead_credit_pct=50),
        "cancel_credit_0pct": replace(base, cancel_ahead_credit_pct=0),
        "queue_ahead_150pct": replace(base, queue_ahead_pct=150),
        "queue_ahead_250pct": replace(base, queue_ahead_pct=250),
        "trade_through_only": replace(base, trade_through_only=True),
        "adverse_fill_300": replace(base, adverse_fill_mticks=300),
        "adverse_fill_1000": replace(base, adverse_fill_mticks=1000),
        "pessimistic_fills": replace(
            base, cancel_ahead_credit_pct=50, queue_ahead_pct=150, adverse_fill_mticks=300
        ),
        "worst_case_fills": replace(
            base, cancel_ahead_credit_pct=0, queue_ahead_pct=250, adverse_fill_mticks=1000, trade_through_only=True
        ),
    }


def freeze(study: Study) -> dict:
    """Write down what the final test will do, before it runs. Commit this file first."""
    tuned = json.loads((RESULTS / "tune.json").read_text())
    fit_report = json.loads((RESULTS / "fit.json").read_text())
    fitted = load_fitted()
    payload = {
        "hypothesis": (
            "A calibrated fill model inside a constrained controller lowers mean implementation "
            "shortfall against the best causal baseline on paired parents, without a worse tail "
            "and without more forced completion."
        ),
        "split": asdict(study.split),
        "workload": asdict(study.parents),
        "input_sha256": ensure_data(study.plan)["sha256"],
        "fitted_sha256": sha256_file(FITTED_PATH),
        "baselines": BASELINES + ["fitted_q", "oracle_volume"],
        "oracle_note": "oracle_volume sees realized volume. It is labeled and never eligible as the best baseline.",
        "metrics": ["mean_bps", "median_bps", "p95_bps", "voluntary_completion_pct", "forced_lot_pct"],
        "decision_rule": DECISION_RULE,
        "best_baseline": tuned["best_baseline"],
        "policy": asdict(next(iter(fitted.policies.values()))),
        "urgency_behind_lots": next(iter(fitted.urgency_behind.values())),
        "config": json.loads(json.dumps(asdict(fitted.config), default=str)),
        "regime_thresholds": fit_report["regime_thresholds"],
        "stress_scenarios": list(stress_scenarios(fitted.config)),
        "stress_strategies": STRESS_STRATEGIES,
        "bootstrap": {"draws": 10_000, "seed": 5, "level": 0.95},
        "code_commit_at_freeze": _git("rev-parse", "HEAD"),
    }
    write_json(FREEZE_PATH, payload)
    return payload


def require_freeze(study: Study) -> dict:
    """Refuse to touch test days unless the frozen file is committed and still matches."""
    if not FREEZE_PATH.exists():
        raise SystemExit("No freeze file. Run `mreplay freeze` and commit results/freeze.json.")
    frozen = json.loads(FREEZE_PATH.read_text())
    if _git("status", "--porcelain", "--", str(FREEZE_PATH)):
        raise SystemExit("results/freeze.json is not committed. Commit it before the final test.")
    if frozen["fitted_sha256"] != sha256_file(FITTED_PATH):
        raise SystemExit("Fitted artifacts changed after the freeze.")
    if frozen["input_sha256"] != ensure_data(study.plan)["sha256"]:
        raise SystemExit("Inputs changed after the freeze.")
    return frozen


# Final test and stress ---------------------------------------------------------------


def compare(rows, a: str, b: str) -> dict:
    days, differences = paired_day_means(rows, a, b)
    mean, low, high = bootstrap_mean_ci(differences)
    return {"a": a, "b": b, "days": len(days), "mean_diff_bps": mean, "ci_low": low, "ci_high": high}


def verdict(rows, baseline: str) -> dict:
    """Apply the frozen decision rule. Each test is reported on its own."""
    paired = compare(rows, "learned", baseline)
    learned, other = summarize(rows, "learned"), summarize(rows, baseline)
    checks = {
        "mean_cost_lower": paired["ci_high"] < 0,
        "tail_not_worse": learned["p95_bps"] <= other["p95_bps"] + 0.5,
        "forced_not_worse": learned["forced_lot_pct"] <= other["forced_lot_pct"] + 1.0,
    }
    return {"paired": paired, "checks": checks, "improved": all(checks.values())}


def session_regimes(study: Study, days: list[int], thresholds: dict) -> dict[int, str]:
    return {s.day: regime(s, thresholds) for s in all_session_stats(study, days)}


def final(study: Study) -> dict:
    """The one evaluation on test days."""
    started = time.perf_counter()
    frozen = require_freeze(study)
    fitted = load_fitted()
    days = study.split.days("test")
    names = BASELINES + ["learned", "fitted_q", "oracle_volume"]
    rows = evaluate_days(study, fitted, names, fitted.config, days)
    write_rows(rows, RESULTS / "raw" / "test.csv")
    regimes = session_regimes(study, days, frozen["regime_thresholds"])
    by_regime = {}
    for label in ("calm", "volatile", "sparse"):
        subset = [r for r in rows if regimes[r.day] == label]
        if not subset:
            continue
        by_regime[label] = {
            "days": sorted({r.day for r in subset}),
            "summaries": {n: summarize(subset, n) for n in names},
            "learned_vs_best_baseline": compare(subset, "learned", frozen["best_baseline"]),
        }
    probes = collect_probes(study, fitted.config, days)
    predicted = fitted.fill_model.predict(probes["x"])
    payload = {
        "verdict": verdict(rows, frozen["best_baseline"]),
        "summaries": {n: summarize(rows, n) for n in names},
        "comparisons": [compare(rows, "learned", n) for n in names if n != "learned"],
        "regimes": by_regime,
        "exclusions": {
            reason: sum(1 for r in rows if r.excluded == reason and r.strategy == "learned")
            for reason in sorted({r.excluded for r in rows if r.excluded})
        },
        "fill_model_on_test": {
            "rows": len(predicted),
            "brier": brier(probes["filled"], predicted),
            "base_rate_brier": brier(probes["filled"], np.full(len(predicted), probes["filled"].mean())),
            "ece": expected_calibration_error(probes["filled"], predicted),
            "calibration": calibration_table(probes["filled"], predicted),
        },
        "manifest": manifest(study, "mreplay final", time.perf_counter() - started, {"raw_sha256": sha256_file(RESULTS / "raw" / "test.csv")}),
    }
    write_json(RESULTS / "final.json", payload)
    return payload


def stress(study: Study) -> dict:
    """Frozen strategies under moved assumptions. Nothing is refit, and the policy keeps its fee beliefs."""
    started = time.perf_counter()
    frozen = require_freeze(study)
    fitted = load_fitted()
    days = study.split.days("test")
    baseline = frozen["best_baseline"]
    names = list(dict.fromkeys(STRESS_STRATEGIES + [baseline]))
    table = {}
    for name, config in stress_scenarios(fitted.config).items():
        rows = evaluate_days(study, fitted, names, config, days)
        write_rows(rows, RESULTS / "raw" / "stress" / f"{name}.csv")
        summaries = {n: summarize(rows, n) for n in names}
        table[name] = {
            "summaries": summaries,
            "cheapest": min(names, key=lambda n: summaries[n]["mean_bps"]),
            "learned_vs_baseline": compare(rows, "learned", baseline),
            "learned_vs_immediate": compare(rows, "learned", "immediate"),
            "verdict": verdict(rows, baseline),
        }
    base = table["base"]["learned_vs_baseline"]
    for name, entry in table.items():
        paired = entry["learned_vs_baseline"]
        base_sign = (base["ci_high"] < 0) - (base["ci_low"] > 0)
        sign = (paired["ci_high"] < 0) - (paired["ci_low"] > 0)
        entry["conclusion_changes"] = sign != base_sign
    payload = {
        "baseline": baseline,
        "scenarios": table,
        "manifest": manifest(study, "mreplay stress", time.perf_counter() - started, {"days": len(days)}),
    }
    write_json(RESULTS / "stress.json", payload)
    return payload
