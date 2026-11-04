"""Isolated accounting for one simulated parent order.

Cash is in millitick-lots: 1 tick x 1 lot = 1000 mticks. Using integers means
conservation holds exactly, with no rounding to explain away.
"""

from dataclasses import dataclass, field
from enum import IntEnum

from .events import Side

MTICKS_PER_TICK = 1000


class Role(IntEnum):
    MAKER = 1  # A simulated resting order was hit.
    TAKER = 2  # A simulated order crossed the spread.
    FORCED = 3  # Deadline completion rule. See Ledger.record and docs/assumptions.md.


@dataclass(frozen=True)
class Fill:
    ts_ns: int
    child_id: int
    price: int
    qty: int
    role: Role
    fee_mticks: int
    penalty_mticks: int
    source_seq: int  # The feed event behind this fill. 0 for deadline completion.


@dataclass
class Ledger:
    side: Side
    target_lots: int
    arrival_mid_x2: int  # Twice the midpoint, so a half-tick mid stays an integer.
    fills: list[Fill] = field(default_factory=list)
    filled_lots: int = 0
    notional_ticks: int = 0  # Sum of price x qty over fills.
    fees_mticks: int = 0
    penalties_mticks: int = 0
    cash_mticks: int = 0

    @property
    def remaining_lots(self) -> int:
        return self.target_lots - self.filled_lots

    def record(self, fill: Fill) -> None:
        if fill.qty <= 0 or fill.qty > self.remaining_lots:
            raise ValueError(f"fill of {fill.qty} with {self.remaining_lots} lots remaining")
        self.fills.append(fill)
        self.filled_lots += fill.qty
        self.notional_ticks += fill.price * fill.qty
        self.fees_mticks += fill.fee_mticks
        self.penalties_mticks += fill.penalty_mticks
        self.cash_mticks -= (
            self.side * fill.price * fill.qty * MTICKS_PER_TICK + fill.fee_mticks + fill.penalty_mticks
        )

    def check_conservation(self) -> None:
        """Recompute every total from the fills. Raises if any running total drifted."""
        qty = sum(f.qty for f in self.fills)
        notional = sum(f.price * f.qty for f in self.fills)
        fees = sum(f.fee_mticks for f in self.fills)
        penalties = sum(f.penalty_mticks for f in self.fills)
        assert qty == self.filled_lots
        assert qty + self.remaining_lots == self.target_lots
        assert notional == self.notional_ticks
        assert fees == self.fees_mticks
        assert penalties == self.penalties_mticks
        assert self.cash_mticks == -(self.side * notional * MTICKS_PER_TICK + fees + penalties)
