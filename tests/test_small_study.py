"""One tiny study, seeded data to rows, with models fit on it. Slowest test file, about 10 s."""

from dataclasses import replace

import numpy as np
import pytest

from marketreplay import comparator, evaluate
from marketreplay.comparator import explore_day, fit_fitted_q, state_row
from marketreplay.evaluate import evaluate_days
from marketreplay.execution import ExecConfig
from marketreplay.fillmodel import MoveModel
from marketreplay.labels import observed_outcomes, probe_day
from marketreplay.policy import PolicyParams
from marketreplay.roster import Fitted, build
from marketreplay.study import (
    ParentSpec, Split, Study, ensure_data, fit_latencies, fit_regimes, fit_volume_curve, load_day,
    parents_for_day, regime, session_stats,
)  # fmt: skip
from marketreplay.synthetic import StudyPlan

STUDY = Study(
    plan=StudyPlan(days=3, session_s=300, seed=7),
    parents=ParentSpec(target_lots=40, horizon_s=30, first_start_s=120, spacing_s=60),
    split=Split(train=(0, 1), validation=(2, 2), test=(2, 2)),
)


class InlinePool:
    """Runs a pool's work in this process, so a failure shows its own traceback."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def map(self, function, items):
        return map(function, items)


class ConstantFill:
    def predict(self, x):
        return np.full(len(x), 0.7)


@pytest.fixture(scope="module")
def trained():
    index = ensure_data(STUDY.plan)
    stats = [session_stats(STUDY, day) for day in STUDY.split.days("train")]
    config = fit_latencies(stats, STUDY.exec_config)
    probes = [probe_day(STUDY, config, day) for day in STUDY.split.days("train")]
    x = np.vstack([p["x"] for p in probes])
    move = np.concatenate([p["move"] for p in probes])
    filled = np.concatenate([p["filled"] for p in probes])
    fitted = Fitted(
        config, fit_volume_curve(stats), ConstantFill(), MoveModel().fit(x, move, filled),
        {"learned": PolicyParams()}, {"urgency": 20},
    )  # fmt: skip
    return index, stats, fitted


def test_data_rebuilds_to_the_same_hash(trained):
    assert ensure_data(STUDY.plan)["sha256"] == trained[0]["sha256"]


def test_fitted_latency_keeps_the_gateway_gap(trained):
    config = trained[2].config
    assert config.order_latency_ns == config.obs_latency_ns + 5_000_000
    assert config.cancel_latency_ns == config.order_latency_ns


def test_regime_thresholds_label_every_session(trained):
    thresholds = fit_regimes(trained[1])
    assert {regime(s, thresholds) for s in trained[1]} <= {"calm", "volatile", "sparse"}


def test_recorded_orders_give_a_usable_outside_check():
    outcome = observed_outcomes(load_day(STUDY.plan, 0))
    assert outcome.orders > 0


def test_every_strategy_trades_every_parent_and_conserves(trained, monkeypatch):
    monkeypatch.setattr(evaluate, "ProcessPoolExecutor", InlinePool)
    names = ["immediate", "equal_time", "volume_schedule", "passive_only", "urgency", "hv_random", "learned", "oracle_volume"]
    rows = evaluate_days(STUDY, trained[2], names, trained[2].config, [2])
    parents = parents_for_day(STUDY, 2)
    assert len(rows) == len(parents) * len(names)
    for r in rows:
        if not r.excluded:
            assert r.maker_lots + r.taker_lots + r.forced_lots == STUDY.parents.target_lots


def test_a_strategy_name_nobody_defined_is_refused(trained):
    with pytest.raises(KeyError):
        build(trained[2], "imaginary", parents_for_day(STUDY, 2)[0])


def test_the_comparator_learns_values_for_all_three_actions(trained, monkeypatch):
    monkeypatch.setattr(comparator, "ProcessPoolExecutor", InlinePool)
    monkeypatch.setattr(comparator, "Q_ITERATIONS", 2)
    config = trained[2].config
    transitions = explore_day(STUDY, config, 16, 0)
    assert transitions and {t[1] for t in transitions} <= {0, 1, 2}
    q, count = fit_fitted_q(STUDY, config, [0, 1], 16)
    assert count >= len(transitions)
    states = np.array([t[0] for t in transitions[:5]])
    assert q.action_values(states).shape == (5, 3)


def test_comparator_state_adds_drift_to_the_shared_features(trained, monkeypatch):
    fitted = replace(trained[2])
    seen = []

    class Recorder:
        name = "recorder"

        def decide(self, view):
            state = state_row(view)
            if state is not None:
                seen.append(state)
            return None

    from marketreplay.execution import run_session

    parent = parents_for_day(STUDY, 2)[0]
    run_session(load_day(STUDY.plan, 2), [(parent, Recorder())], fitted.config)
    assert seen and len(seen[0]) == 14
    assert isinstance(fitted.config, ExecConfig)
