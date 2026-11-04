"""Finite-horizon reinforcement learning comparator.

Hafsi and Vittori (arXiv:2411.06389v2) train deep Q-learning for a finite-horizon
inventory task in the ABIDES simulator, with market-order actions only. This file
keeps the task shape and changes the rest: fitted Q-iteration on replay data, with
the study's own actions (wait, rest, cross). It is a comparator, not a reproduction.

Unlike LearnedPolicy it has no hard constraints. The deadline rule is its only limit,
so it shows what an unconstrained learner does with the same information.
"""

import random
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .execution import Action, ExecConfig, Intent, Parent, View, run_session
from .features import features, mid
from .ledger import MTICKS_PER_TICK, Role
from .study import Study, WORKERS, load_day, parents_for_day

ACTIONS = (Action.WAIT, Action.REST, Action.CROSS)
# With no discounting, each iteration carries cost back one decision. The horizon is 30
# decisions, so fewer iterations would leave early states blind to the deadline penalty.
Q_ITERATIONS = 30


def state_row(view: View) -> np.ndarray | None:
    row = features(view)
    if row is None:
        return None
    start = view.arrival_mid_seen if view.arrival_mid_seen is not None else mid(view.market)
    drift = view.parent.side * (mid(view.market) - start)
    return np.append(row, drift)


def intent_for(action: Action, clip_lots: int) -> Intent:
    if action == Action.CROSS:
        return Intent.cross(clip_lots)
    if action == Action.REST:
        return Intent.rest(clip_lots)
    return Intent.wait()


class Explorer:
    """Behavior policy. Each parent draws its own cross rate, so urgency varies across the data."""

    name = "explorer"

    def __init__(self, seed: int, clip_lots: int):
        self.rng = random.Random(seed)
        self.clip_lots = clip_lots
        self.cross_rate = self.rng.uniform(0.0, 0.35)  # Low rates reach the deadline, which it must learn.
        self.steps = []  # (now_ns, state, action index)

    def decide(self, view: View) -> Intent | None:
        state = state_row(view)
        if state is None or view.remaining_seen == 0:
            return None
        if self.rng.random() < self.cross_rate:
            index = 2
        else:
            index = self.rng.choice((0, 1, 1))
        self.steps.append((view.now_ns, state, index))
        return intent_for(ACTIONS[index], self.clip_lots)


def step_costs(run, target_lots: int) -> list[float]:
    """Cost in ticks per parent lot, charged to the step in which each fill landed."""
    steps = run.strategy.steps
    costs = [0.0] * len(steps)
    arrival = run.ledger.arrival_mid_x2 / 2
    times = [s[0] for s in steps]
    for fill in run.ledger.fills:
        index = max(0, np.searchsorted(times, fill.ts_ns, side="right") - 1)
        slippage = run.parent.side * (fill.price - arrival) * fill.qty
        extra = (fill.fee_mticks + fill.penalty_mticks) / MTICKS_PER_TICK
        costs[index] += (slippage + extra) / target_lots
    return costs


def explore_day(study: Study, config: ExecConfig, clip_lots: int, day: int) -> list[tuple]:
    work = [(p, Explorer(p.parent_id, clip_lots)) for p in parents_for_day(study, day)]
    transitions = []
    for run in run_session(load_day(study.plan, day), work, config):
        steps = run.strategy.steps
        if run.excluded or not steps:
            continue
        costs = step_costs(run, run.parent.target_lots)
        for i, (_, state, action) in enumerate(steps):
            next_state = steps[i + 1][1] if i + 1 < len(steps) else None
            transitions.append((state, action, costs[i], next_state))
    return transitions


class FittedQ:
    def __init__(self, models: list, clip_lots: int):
        self.models = models
        self.clip_lots = clip_lots

    def strategy(self, name: str) -> "FittedQStrategy":
        return FittedQStrategy(self, name)

    def action_values(self, states: np.ndarray) -> np.ndarray:
        return np.column_stack([model.predict(states) for model in self.models])


class FittedQStrategy:
    def __init__(self, q: FittedQ, name: str):
        self.q = q
        self.name = name

    def decide(self, view: View) -> Intent | None:
        state = state_row(view)
        if state is None:
            return Intent.rest(self.q.clip_lots)
        values = self.q.action_values(state.reshape(1, -1))[0]
        return intent_for(ACTIONS[int(np.argmin(values))], self.q.clip_lots)


def fit_fitted_q(study: Study, config: ExecConfig, days: list[int], clip_lots: int) -> tuple[FittedQ, int]:
    with ProcessPoolExecutor(WORKERS) as pool:
        per_day = pool.map(partial(explore_day, study, config, clip_lots), days)
        transitions = [t for rows in per_day for t in rows]
    states = np.array([t[0] for t in transitions])
    actions = np.array([t[1] for t in transitions])
    costs = np.array([t[2] for t in transitions])
    has_next = np.array([t[3] is not None for t in transitions])
    next_states = np.array([t[3] for t in transitions if t[3] is not None])
    targets = costs.copy()
    models = []
    for _ in range(Q_ITERATIONS):
        models = []
        for index in range(len(ACTIONS)):
            chosen = actions == index
            model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, min_samples_leaf=100, random_state=0)
            models.append(model.fit(states[chosen], targets[chosen]))
        best_next = np.column_stack([m.predict(next_states) for m in models]).min(axis=1)
        targets = costs.copy()
        targets[has_next] += best_next
    return FittedQ(models, clip_lots), len(transitions)
