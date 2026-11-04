"""The decision rule, checked with fixed model outputs."""

from dataclasses import dataclass

import numpy as np
import pytest

from marketreplay.events import Side
from marketreplay.execution import Action, Parent, Snapshot, View
from marketreplay.policy import LearnedPolicy, PolicyParams


class FixedFill:
    def __init__(self, p):
        self.p = p

    def predict(self, x):
        return np.array([self.p])


@dataclass
class FixedMove:
    unfilled: float = 0.0
    overall: float = 0.0

    def predict(self, x):
        return np.array([self.unfilled]), np.array([self.overall])


def view(now_s=10.0, filled=0, sent=0):
    market = Snapshot(0, ((100, 10),), ((102, 10),), 5, 5, 0)
    parent = Parent(1, Side.BID, 120, 0, 60_000_000_000)
    return View(int(now_s * 1e9), parent, 202, market, [market], filled, sent, None, 0, None)


def policy(p, unfilled=0.0, overall=0.0, **params):
    return LearnedPolicy(FixedFill(p), FixedMove(unfilled, overall), PolicyParams(**params), 300, -100, "learned")


def test_choices_match_the_hand_calculation():
    costs = policy(0.5, unfilled=2.0, overall=-1.0, catch_up_ticks=0.5).price_choices(1.0, 0.5, 2.0, -1.0)
    assert costs["cross"] == pytest.approx(1.3)
    # 0.5 x (-1 - 0.1) + 0.5 x (2 + 1 + 0.3 + 0.5) = -0.55 + 1.9
    assert costs["rest"] == pytest.approx(1.35)
    assert costs["wait"] == pytest.approx(-1.0 + 1.3)


def test_a_likely_fill_rests():
    assert policy(0.95).decide(view()).action == Action.REST


def test_an_unlikely_fill_with_a_price_running_away_crosses():
    assert policy(0.05, unfilled=3.0, overall=1.0).decide(view()).action == Action.CROSS


def test_a_price_expected_to_fall_waits():
    assert policy(0.2, unfilled=1.0, overall=-4.0).decide(view()).action == Action.WAIT


def test_the_behind_constraint_overrides_the_model():
    decision = policy(0.95, behind_lots=20).decide(view(now_s=30))
    assert decision.action == Action.CROSS
    assert decision.lots == 60


def test_the_final_constraint_crosses_everything_left():
    decision = policy(0.95, behind_lots=500, final_s=4).decide(view(now_s=57, filled=100))
    assert (decision.action, decision.lots) == (Action.CROSS, 20)
