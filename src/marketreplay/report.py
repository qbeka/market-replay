"""Write docs/report.md and its plots from saved results. No number here is typed by hand."""

import json

from .pipeline import FREEZE_PATH, RESULTS
from .study import REPO, Study

DOCS = REPO / "docs"
FIGURES = DOCS / "figures"
NAMES = {
    "immediate": "Immediate cross",
    "equal_time": "Equal time",
    "volume_schedule": "Volume schedule",
    "passive_only": "Passive only",
    "urgency": "Urgency rule",
    "hv_random": "Random (after Hafsi and Vittori)",
    "learned": "Learned policy",
    "fitted_q": "Fitted Q comparator",
    "oracle_volume": "Oracle volume schedule (not achievable)",
}
INK, INK_2, GRID, SERIES = "#0b0b0b", "#52514e", "#dedcd5", "#2a78d6"


def strategy_table(summaries: dict) -> str:
    lines = [
        "| Strategy | Parents | Mean bps | Median bps | P95 bps | Finished before deadline | Forced lots | Maker lots |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in sorted(summaries.items(), key=lambda item: item[1]["mean_bps"]):
        lines.append(
            f"| {NAMES.get(name, name)} | {s['parents']} | {s['mean_bps']:.3f} | {s['median_bps']:.3f} | {s['p95_bps']:.3f} "
            f"| {s['voluntary_completion_pct']:.1f}% | {s['forced_lot_pct']:.2f}% | {s['maker_lot_pct']:.1f}% |"
        )
    return "\n".join(lines)


def comparison_table(comparisons: list[dict]) -> str:
    lines = ["| Learned minus | Mean difference bps | 95% interval | Days |", "|---|---:|---|---:|"]
    for c in comparisons:
        lines.append(
            f"| {NAMES.get(c['b'], c['b'])} | {c['mean_diff_bps']:.3f} | {c['ci_low']:.3f} to {c['ci_high']:.3f} | {c['days']} |"
        )
    return "\n".join(lines)


def regime_table(regimes: dict, baseline: str) -> str:
    lines = [
        f"| Session type | Days | Learned mean bps | {NAMES[baseline]} mean bps | Difference bps | 95% interval |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, entry in regimes.items():
        c = entry["learned_vs_best_baseline"]
        learned, base = entry["summaries"]["learned"], entry["summaries"][baseline]
        lines.append(
            f"| {label} | {c['days']} | {learned['mean_bps']:.3f} | {base['mean_bps']:.3f} "
            f"| {c['mean_diff_bps']:.3f} | {c['ci_low']:.3f} to {c['ci_high']:.3f} |"
        )
    return "\n".join(lines)


def stress_table(stress: dict) -> str:
    lines = [
        "| Scenario | Cheapest strategy | Learned minus baseline bps | 95% interval | Rule passes | Sign changes |",
        "|---|---|---:|---|---|---|",
    ]
    for name, entry in stress["scenarios"].items():
        c = entry["learned_vs_baseline"]
        lines.append(
            f"| {name} | {NAMES.get(entry['cheapest'], entry['cheapest'])} | {c['mean_diff_bps']:.3f} "
            f"| {c['ci_low']:.3f} to {c['ci_high']:.3f} | {'Yes' if entry['verdict']['improved'] else 'No'} "
            f"| {'Yes' if entry['conclusion_changes'] else 'No'} |"
        )
    return "\n".join(lines)


def reversals(stress: dict) -> str:
    """Plain statements of what changes across scenarios, read from the table."""
    base = stress["scenarios"]["base"]
    items = []
    changed = [n for n, e in stress["scenarios"].items() if e["conclusion_changes"]]
    items.append(
        f"- The sign of the paired result changes in {len(changed)} of {len(stress['scenarios']) - 1} moved scenarios"
        + (f": {', '.join(changed)}." if changed else ".")
    )
    other_cheapest = [n for n, e in stress["scenarios"].items() if e["cheapest"] != base["cheapest"]]
    items.append(
        f"- The cheapest strategy under base assumptions is {NAMES.get(base['cheapest'])}. It loses that place in "
        f"{len(other_cheapest)} scenarios"
        + (f": {', '.join(other_cheapest)}." if other_cheapest else ".")
    )
    worst = stress["scenarios"].get("worst_case_fills")
    if worst:
        items.append(
            f"- Under worst_case_fills the cheapest strategy is {NAMES.get(worst['cheapest'])}, at "
            f"{worst['summaries'][worst['cheapest']]['mean_bps']:.3f} bps."
        )
    return "\n".join(items)


def plot_stress(stress: dict) -> None:
    """Forest plot on one axis: paired difference with its day-bootstrap interval."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(stress["scenarios"])[::-1]
    entries = [stress["scenarios"][n]["learned_vs_baseline"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 0.34 * len(names) + 1.2))
    for y, c in enumerate(entries):
        ax.plot([c["ci_low"], c["ci_high"]], [y, y], color=SERIES, linewidth=2, solid_capstyle="round")
        ax.plot(c["mean_diff_bps"], y, "o", color=SERIES, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.axvline(0, color=INK_2, linewidth=1)
    ax.set_yticks(range(len(names)), names, fontsize=9, color=INK)
    ax.set_xlabel(f"Learned minus {NAMES[stress['baseline']].lower()}, bps. Below zero favors learned.", fontsize=9, color=INK_2)
    ax.tick_params(axis="x", labelsize=9, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "stress_forest.svg")
    plt.close(fig)


def plot_calibration(table: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1)
    ax.plot(
        [r["predicted"] for r in table], [r["observed"] for r in table], "-o", color=SERIES, linewidth=2,
        markersize=7, markeredgecolor="white", markeredgewidth=1.5,
    )  # fmt: skip
    ax.set_xlabel("Predicted fill chance", fontsize=9, color=INK_2)
    ax.set_ylabel("Filled in the queue model", fontsize=9, color=INK_2)
    ax.tick_params(labelsize=9, colors=INK_2, length=0)
    ax.grid(color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.svg")
    plt.close(fig)


def write_report(study: Study) -> str:
    frozen = json.loads(FREEZE_PATH.read_text())
    fit = json.loads((RESULTS / "fit.json").read_text())
    tune = json.loads((RESULTS / "tune.json").read_text())
    final = json.loads((RESULTS / "final.json").read_text())
    stress = json.loads((RESULTS / "stress.json").read_text())
    profile = json.loads((RESULTS / "profile.json").read_text())
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_stress(stress)
    plot_calibration(final["fill_model_on_test"]["calibration"])

    baseline = frozen["best_baseline"]
    verdict = final["verdict"]
    paired = verdict["paired"]
    checks = verdict["checks"]
    learned = final["summaries"]["learned"]
    base = final["summaries"][baseline]
    fill = final["fill_model_on_test"]
    observed = fit["observed_resting_orders"]
    manifest = final["manifest"]
    if verdict["improved"]:
        headline = (
            f"**The learned policy passed the frozen rule against the {NAMES[baseline].lower()} on this synthetic feed.** "
            f"It cost {-paired['mean_diff_bps']:.3f} bps less per order (95% interval "
            f"{-paired['ci_high']:.3f} to {-paired['ci_low']:.3f} bps), across {paired['days']} trading days."
        )
    else:
        failed = [k.replace("_", " ") for k, ok in checks.items() if not ok]
        headline = (
            f"**The learned policy did not pass the frozen rule against the {NAMES[baseline].lower()} on this synthetic feed.** "
            f"The mean paired difference was {paired['mean_diff_bps']:.3f} bps (95% interval {paired['ci_low']:.3f} to "
            f"{paired['ci_high']:.3f} bps) across {paired['days']} trading days. Failed checks: {', '.join(failed)}. "
            "This is a valid negative result."
        )
    text = f"""# Benchmark report

Generated by `uv run mreplay report` from saved results. Do not edit by hand.

Every order here is simulated, and the market is a seeded synthetic feed. Nothing here describes a real venue. Economic results are exploratory. See [assumptions.md](assumptions.md) and [data_rights.md](data_rights.md).

## Result

{headline}

| Frozen check | Learned | {NAMES[baseline]} | Passed |
|---|---:|---:|---|
| 95% interval of the mean paired difference below 0 bps | {paired['ci_low']:.3f} to {paired['ci_high']:.3f} | | {'Yes' if checks['mean_cost_lower'] else 'No'} |
| P95 no more than 0.5 bps above the baseline | {learned['p95_bps']:.3f} | {base['p95_bps']:.3f} | {'Yes' if checks['tail_not_worse'] else 'No'} |
| Forced lots no more than 1 point above the baseline | {learned['forced_lot_pct']:.2f}% | {base['forced_lot_pct']:.2f}% | {'Yes' if checks['forced_not_worse'] else 'No'} |

Unit of replication: one trading day. Denominator: {learned['parents']} parents over {learned['days']} days, with {sum(final['exclusions'].values())} parents excluded for every strategy ({', '.join(f'{k}: {v}' for k, v in final['exclusions'].items())}).

## Cost by strategy

Test days only. Positive numbers are costs. Each row trades the same parents.

{strategy_table(final['summaries'])}

The oracle schedule sees each session's final volume, so no real trader could run it. It is shown for scale and was never eligible as the baseline.

{comparison_table(final['comparisons'])}

## Session types

Thresholds were fixed on training days: volatile means 60 s midpoint volatility of at least {frozen['regime_thresholds']['volatile_vol_ticks']:.2f} ticks, and sparse means a median touch of at most {frozen['regime_thresholds']['sparse_touch_lots']:.1f} lots. Sparse takes priority. Small day counts make these intervals wide.

{regime_table(final['regimes'], baseline)}

## What changes under other fill assumptions

Strategies stay frozen in every scenario. Nothing is refit, and the policy keeps its original fee beliefs.

{reversals(stress)}

![Paired difference by scenario](figures/stress_forest.svg)

{stress_table(stress)}

## Fill model

Boosted trees beat logistic regression at an equal budget of {fit['fill_models']['boosted']['trials']} random trials each, on later training days.

| Measured on | Rows | Boosted Brier | Logistic Brier | Base rate Brier | Boosted calibration error |
|---|---:|---:|---:|---:|---:|
| Training days 50 to 59 | {fit['probe_rows']['score']} | {fit['fill_models']['boosted']['brier']:.4f} | {fit['fill_models']['logistic']['brier']:.4f} | {fit['base_rate_brier']:.4f} | {fit['fill_models']['boosted']['ece']:.4f} |
| Test days | {fill['rows']} | {fill['brier']:.4f} | not run | {fill['base_rate_brier']:.4f} | {fill['ece']:.4f} |

![Calibration on test days](figures/calibration.svg)

The label is the queue model's own output: did a 16 lot probe fill within 15 s or its deadline. Calibration here shows the model learned the simulator, not that the simulator is right.

## Modeled against observed

| Quantity | Kind | Value |
|---|---|---:|
| Probe fill rate, 16 lots | Modeled | {100 * observed['probe_fill_rate']:.1f}% |
| Fill chance of recorded touch orders by 15 s, Kaplan-Meier | Observed in the feed | {100 * observed['kaplan_meier_fill']:.1f}% |
| Recorded touch orders followed | Observed | {observed['orders']:,} |
| Share cancelled before the horizon | Observed, censored | {100 * observed['cancelled_first'] / observed['orders']:.1f}% |
| Price move after a probe filled | Modeled fill, observed price | {fit['adverse_selection_ticks']['move_after_fill']:.3f} ticks |
| Price move after a probe did not fill | Modeled fill, observed price | {fit['adverse_selection_ticks']['move_without_fill']:.3f} ticks |

Recorded orders are smaller than probes, so their rate should run higher. With {100 * observed['cancelled_first'] / observed['orders']:.0f}% censored, the Kaplan-Meier figure assumes cancels carry no information about fills. Real traders cancel when they expect trouble, so treat it as an upper estimate.

## Speed and memory

Measured on {manifest['hardware']['cpu']}, one core, day {profile['day']} with {profile['events']:,} events, median of {profile['python_book']['runs']} runs. The 1 minute load average was {profile['load_average_1m_before']:.1f} before the run and {profile['load_average_1m_after']:.1f} after it. The compiled replay was profiled only after its ledgers matched the reference exactly.

| Path | Events per second | Seconds, median | Range |
|---|---:|---:|---|
| Book replay, Python | {profile['python_book']['events_per_s']:,.0f} | {profile['python_book']['median']:.4f} | {profile['python_book']['min']:.4f} to {profile['python_book']['max']:.4f} |
| Book replay, compiled C | {profile['compiled_book']['events_per_s']:,.0f} | {profile['compiled_book']['median']:.4f} | {profile['compiled_book']['min']:.4f} to {profile['compiled_book']['max']:.4f} |
| Replay and execution, Python | {profile['python_exec']['events_per_s']:,.0f} | {profile['python_exec']['median']:.4f} | {profile['python_exec']['min']:.4f} to {profile['python_exec']['max']:.4f} |
| Replay and execution, compiled C | {profile['compiled_exec']['events_per_s']:,.0f} | {profile['compiled_exec']['median']:.4f} | {profile['compiled_exec']['min']:.4f} to {profile['compiled_exec']['max']:.4f} |

Execution rows cover {profile['strategies_in_exec_run']} strategies on {profile['parents_in_exec_run']} parents in one pass. The Python execution time includes those strategies' decisions, and the compiled time replays their recorded commands, so that speedup of {profile['speedup_exec']:.1f}x does not compare like with like. The book speedup of {profile['speedup_book']:.1f}x does. The compiled book checksum also updates after each event, which slows it slightly.

| Memory, peak | MB |
|---|---:|
| Python book replay, whole process | {profile['peak_rss_mb']['python_book']:.1f} |
| Compiled book replay, whole process | {profile['peak_rss_mb']['compiled_book']:.1f} |

Each policy decision took {profile['policy_decision_us']['p50']:.0f} us at the median and {profile['policy_decision_us']['p95']:.0f} us at p95, over {profile['policy_decision_us']['count']:,} decisions. Decisions in this study run every 2 s, so model speed does not limit it.

## Validation record

The tuning stage chose `{tune['chosen_policy']}` and `{tune['chosen_urgency']}` on validation days, then the freeze named `{baseline}` as the baseline. Failed experiments are in [experiments.md](experiments.md).

## How to reproduce

| Item | Value |
|---|---|
| Final run commit | `{manifest['commit']}` |
| Input hash | `{manifest['input_sha256']}` |
| Fitted artifact hash | `{frozen['fitted_sha256']}` |
| Raw results hash | `{manifest['raw_sha256']}` |
| Seeds | plan {manifest['seeds']['plan']}, pipeline {manifest['seeds']['pipeline']}, bootstrap {frozen['bootstrap']['seed']} |
| Machine | {manifest['hardware']['cpu']}, {manifest['hardware']['cores']} cores, Python {manifest['hardware']['python']} |

Raw per-parent measurements are in `results/raw/`. Plots are built from them, never the other way around.
"""
    (DOCS / "report.md").write_text(text)
    return "Wrote docs/report.md and docs/figures/"
