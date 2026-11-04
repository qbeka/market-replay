"""Paired comparison across trading days.

Parents on one day share a market, so a day is the unit of replication. The
bootstrap resamples whole days. Resampling ticks or parents would overstate how much
independent evidence exists.
"""

import numpy as np

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 5


def paired_day_means(rows, strategy_a: str, strategy_b: str) -> tuple[list[int], np.ndarray]:
    """Mean per-parent difference a - b for each day, using parents both completed."""
    by_key = {}
    for row in rows:
        if row.strategy in (strategy_a, strategy_b) and not row.excluded:
            by_key.setdefault((row.day, row.parent_id), {})[row.strategy] = row.shortfall_bps
    by_day: dict[int, list[float]] = {}
    for (day, _), pair in by_key.items():
        if strategy_a in pair and strategy_b in pair:
            by_day.setdefault(day, []).append(pair[strategy_a] - pair[strategy_b])
    days = sorted(by_day)
    return days, np.array([np.mean(by_day[d]) for d in days])


def bootstrap_mean_ci(day_values: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and percentile interval from resampling days with replacement."""
    if len(day_values) < 2:
        return float(day_values.mean()), float("nan"), float("nan")  # One day gives no spread.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(day_values), size=(BOOTSTRAP_DRAWS, len(day_values)))
    means = day_values[draws].mean(axis=1)
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(day_values.mean()), float(low), float(high)


def summarize(rows, name: str) -> dict:
    """Median cost, tail cost and completion, kept separate on purpose."""
    done = [r for r in rows if r.strategy == name and not r.excluded]
    values = np.array([r.shortfall_bps for r in done])
    target = sum(r.maker_lots + r.taker_lots + r.forced_lots for r in done)
    forced = sum(r.forced_lots for r in done)
    return {
        "strategy": name,
        "parents": len(done),
        "days": len({r.day for r in done}),
        "mean_bps": float(values.mean()),
        "median_bps": float(np.median(values)),
        "p95_bps": float(np.percentile(values, 95)),
        "worst_bps": float(values.max()),
        "voluntary_completion_pct": float(100 * np.mean([r.forced_lots == 0 for r in done])),
        "forced_lot_pct": float(100 * forced / target),
        "maker_lot_pct": float(100 * sum(r.maker_lots for r in done) / target),
    }
