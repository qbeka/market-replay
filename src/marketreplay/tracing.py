"""One parent's complete story, printed. This is where a failure gets inspected."""

from dataclasses import replace

from .execution import replay_session
from .metrics import forced_lots, shortfall_bps
from .pipeline import load_fitted
from .roster import build
from .study import Study, load_day, parents_for_day


def format_entry(start_ns: int, entry: dict) -> str:
    detail = {k: v for k, v in entry.items() if k not in ("ts_ns", "what")}
    return f"{(entry['ts_ns'] - start_ns) / 1e9:9.3f}s  {entry['what']:<16} {detail}"


def print_trace(study: Study, day: int, parent_index: int, strategy: str) -> None:
    fitted = load_fitted()
    config = replace(fitted.config, record_trace=True)
    parent = parents_for_day(study, day)[parent_index]
    [run], _ = replay_session(load_day(study.plan, day), [(parent, build(fitted, strategy, parent))], config)
    print(f"SIMULATED. Day {day}, parent {parent.parent_id}, side {parent.side.name}, {parent.target_lots} lots, {strategy}")
    print(f"Latency ms: observation {config.obs_latency_ns / 1e6:.2f}, order {config.order_latency_ns / 1e6:.2f}, cancel {config.cancel_latency_ns / 1e6:.2f}")
    for entry in run.trace:
        print(format_entry(parent.start_ns, entry))
    if run.excluded:
        print(f"Excluded: {run.excluded}")
        return
    ledger = run.ledger
    ledger.check_conservation()
    print(
        f"Shortfall {shortfall_bps(ledger):.3f} bps. Filled {ledger.filled_lots} lots, {forced_lots(ledger)} forced. "
        f"Fees {ledger.fees_mticks} mticks, penalties {ledger.penalties_mticks} mticks."
    )
