"""Signed implementation shortfall in basis points of the arrival midpoint.

    IS = 10,000 x [s x sum(q_i x (p_i - p_arrival)) + costs] / (Q x p_arrival)

s is +1 for a buy and -1 for a sell. A positive number is a cost. Costs are
fees plus any deadline penalty. Only complete orders are scored, and the
deadline rule guarantees that every order completes. That is why a strategy
cannot improve its score by leaving an expensive order unfinished.
"""

from .ledger import Ledger, MTICKS_PER_TICK, Role


def shortfall_bps(ledger: Ledger) -> float:
    if ledger.remaining_lots != 0:
        raise ValueError("shortfall is defined for complete orders only")
    # Everything is doubled so the half-tick arrival midpoint stays exact.
    slippage_x2_ticks = ledger.side * (2 * ledger.notional_ticks - ledger.target_lots * ledger.arrival_mid_x2)
    costs_x2_ticks = 2 * (ledger.fees_mticks + ledger.penalties_mticks) / MTICKS_PER_TICK
    return 10_000 * (slippage_x2_ticks + costs_x2_ticks) / (ledger.target_lots * ledger.arrival_mid_x2)


def forced_lots(ledger: Ledger) -> int:
    return sum(f.qty for f in ledger.fills if f.role == Role.FORCED)
