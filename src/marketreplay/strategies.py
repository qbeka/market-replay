"""Baseline strategies. All of them see only a delayed View.

Every strategy trades the same parents, so results compare paired orders.
A Schedule says how many lots should be done by a given fraction of the horizon.
"""

import math
import random

from .execution import Intent, View


def lots_due(view: View, fraction_done: float) -> int:
    return math.ceil(view.parent.target_lots * min(fraction_done, 1.0))


def time_fraction(view: View) -> float:
    parent = view.parent
    return (view.now_ns - parent.start_ns) / (parent.deadline_ns - parent.start_ns)


class ImmediateCross:
    """Cross everything now, and retry each tick if depth runs out."""

    name = "immediate"

    def decide(self, view: View) -> Intent | None:
        return Intent.cross(view.sendable) if view.sendable > 0 else None


class EqualTime:
    """Cross equal slices through the horizon."""

    name = "equal_time"

    def __init__(self, slices: int = 10):
        self.slices = slices

    def decide(self, view: View) -> Intent | None:
        # Slice k is due at the start of its interval, so the last slice has time to finish.
        k = math.floor(time_fraction(view) * self.slices) + 1
        return _cross_up_to(view, lots_due(view, k / self.slices))


class VolumeSchedule:
    """Cross in proportion to a volume curve fitted on earlier days.

    curve[i] is the cumulative fraction of session volume by the end of bucket i.
    """

    name = "volume_schedule"

    def __init__(self, curve: list[float], bucket_ns: int, slices: int = 10, name: str | None = None):
        self.curve = curve
        self.bucket_ns = bucket_ns
        self.slices = slices
        if name:
            self.name = name

    def cumulative(self, ts_ns: int) -> float:
        position = ts_ns / self.bucket_ns
        index = int(position)
        if index >= len(self.curve) - 1:
            return self.curve[-1]
        before = self.curve[index - 1] if index > 0 else 0.0
        return before + (self.curve[index] - before) * (position - index)

    def decide(self, view: View) -> Intent | None:
        parent = view.parent
        start = self.cumulative(parent.start_ns)
        span = self.cumulative(parent.deadline_ns) - start
        k = math.floor(time_fraction(view) * self.slices) + 1
        next_ts = parent.start_ns + (parent.deadline_ns - parent.start_ns) * k // self.slices
        due = (self.cumulative(next_ts) - start) / span if span > 0 else time_fraction(view)
        return _cross_up_to(view, lots_due(view, due))


def _cross_up_to(view: View, due: int) -> Intent | None:
    done_or_sent = view.parent.target_lots - view.sendable
    short = due - done_or_sent
    return Intent.cross(min(short, view.sendable)) if short > 0 else None


class PassiveOnly:
    """Rest at the near touch and accept the deadline rule for whatever is left."""

    name = "passive_only"

    def __init__(self, clip_lots: int = 8):
        self.clip_lots = clip_lots

    def decide(self, view: View) -> Intent | None:
        return Intent.rest(self.clip_lots)


class UrgencyRule:
    """Rest passively, and cross when too far behind an equal-time schedule."""

    name = "urgency"

    def __init__(self, behind_lots: int = 6, clip_lots: int = 8):
        self.behind_lots = behind_lots
        self.clip_lots = clip_lots

    def decide(self, view: View) -> Intent | None:
        behind = lots_due(view, time_fraction(view)) - view.filled_seen
        if behind > self.behind_lots:
            return Intent.cross(behind)
        return Intent.rest(self.clip_lots)


class RandomPolicy:
    """Hafsi and Vittori's random baseline, restated on this action set."""

    name = "hv_random"

    def __init__(self, seed: int, clip_lots: int = 8):
        self.rng = random.Random(seed)
        self.clip_lots = clip_lots

    def decide(self, view: View) -> Intent | None:
        if self.rng.random() < 0.5:
            return Intent.wait()
        return Intent.cross(self.rng.randint(1, self.clip_lots))
