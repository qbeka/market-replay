"""The learned execution policy: a fill model inside a constrained controller.

At each decision the policy prices three choices in ticks per lot, relative to the
observed midpoint, and takes the cheapest one that the constraints allow:

    cross = half_spread + taker
    rest  = p_fill x (-half_spread + maker)
            + (1 - p_fill) x (move_if_unfilled + half_spread + taker + catch_up)
    wait  = expected_move + min(cross, rest)

p_fill and the moves come from models fitted on earlier days. Keeping the rule this
plain means any change in cost can be traced to one decision, which an end to end
reinforcement learner would not allow.

Constraints, checked first:

    behind_lots   cross the shortfall once behind an equal-time schedule by more
    final_s       inside this many seconds, cross whatever remains
"""

from dataclasses import dataclass

from .execution import Intent, View
from .features import FEATURE_NAMES, features
from .ledger import MTICKS_PER_TICK
from .strategies import lots_due, time_fraction

QUEUE_COLUMN = FEATURE_NAMES.index("queue_ahead_lots")


@dataclass(frozen=True)
class PolicyParams:
    clip_lots: int = 16
    behind_lots: int = 20
    catch_up_ticks: float = 0.5
    final_s: float = 4.0


class LearnedPolicy:
    def __init__(self, fill_model, move_model, params: PolicyParams, taker_mticks: int, maker_mticks: int, name: str):
        self.fill_model = fill_model
        self.move_model = move_model
        self.params = params
        self.taker_ticks = taker_mticks / MTICKS_PER_TICK
        self.maker_ticks = maker_mticks / MTICKS_PER_TICK
        self.name = name
        self.last_info = None

    def decide(self, view: View) -> Intent | None:
        params = self.params
        time_left_s = view.time_left_ns / 1e9
        if time_left_s <= params.final_s:
            self.last_info = {"rule": "final"}
            return Intent.cross(view.sendable) if view.sendable > 0 else None
        behind = lots_due(view, time_fraction(view)) - view.filled_seen
        if behind > params.behind_lots:
            self.last_info = {"rule": "behind", "behind_lots": behind}
            return Intent.cross(behind)
        row = features(view)
        if row is None:
            self.last_info = {"rule": "one-sided book"}
            return Intent.rest(params.clip_lots)
        batch = row.reshape(1, -1)
        p_fill = float(self.fill_model.predict(batch)[0])
        move_unfilled, move_overall = (float(a[0]) for a in self.move_model.predict(batch))
        costs = self.price_choices(row[0] / 2, p_fill, move_unfilled, move_overall)
        choice = min(costs, key=costs.get)
        self.last_info = {
            "rule": "model", "p_fill": round(p_fill, 4), "move_unfilled": round(move_unfilled, 4),
            "move_overall": round(move_overall, 4), "costs": {k: round(v, 4) for k, v in costs.items()},
        }  # fmt: skip
        if choice == "cross":
            return Intent.cross(params.clip_lots)
        if choice == "rest":
            return Intent.rest(params.clip_lots)
        return Intent.wait()

    def price_choices(self, half_spread: float, p_fill: float, move_unfilled: float, move_overall: float) -> dict:
        cross = half_spread + self.taker_ticks
        catch_up = move_unfilled + half_spread + self.taker_ticks + self.params.catch_up_ticks
        rest = p_fill * (-half_spread + self.maker_ticks) + (1 - p_fill) * catch_up
        wait = move_overall + min(cross, rest)
        return {"cross": cross, "rest": rest, "wait": wait}
