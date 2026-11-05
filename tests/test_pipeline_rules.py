"""The frozen decision rule, the freeze guard and the stress plan."""

import json
from dataclasses import fields

import pytest

from marketreplay import pipeline
from marketreplay.evaluate import Row
from marketreplay.execution import ExecConfig
from marketreplay.pipeline import require_freeze, sha256_file, stress_scenarios, verdict
from marketreplay.study import Study


def row(day, parent, strategy, cost, forced=0, maker=100):
    return Row(day, parent, 1, strategy, "", cost, forced, maker, 100 - maker - forced, 0, 0, 1.0)


def paired(learned_costs, baseline_costs, learned_forced=0, baseline_forced=0):
    rows = []
    for day, (a, b) in enumerate(zip(learned_costs, baseline_costs)):
        rows.append(row(day, 1, "learned", a, learned_forced, 100 - learned_forced))
        rows.append(row(day, 1, "urgency", b, baseline_forced, 100 - baseline_forced))
    return rows


def test_a_clear_win_passes_all_three_checks():
    result = verdict(paired([0.0, 0.1, 0.2, 0.1, 0.0, 0.1], [1.0, 1.1, 1.2, 1.1, 1.0, 1.1]), "urgency")
    assert result["improved"] and all(result["checks"].values())
    assert result["paired"]["days"] == 6


def test_an_interval_that_reaches_zero_fails_the_mean_check():
    result = verdict(paired([1, -1, 1, -1, 1, -1], [0, 0, 0, 0, 0, 0]), "urgency")
    assert not result["checks"]["mean_cost_lower"]
    assert not result["improved"]


def test_a_cheap_mean_with_a_bad_tail_fails():
    learned = [-9.0] * 18 + [30.0, 30.0]
    result = verdict(paired(learned, [0.0] * 20), "urgency")
    assert not result["checks"]["tail_not_worse"]
    assert not result["improved"]


def test_saving_money_by_forcing_more_lots_fails():
    result = verdict(paired([0.0] * 6, [1.0] * 6, learned_forced=5, baseline_forced=1), "urgency")
    assert result["checks"]["mean_cost_lower"]
    assert not result["checks"]["forced_not_worse"]
    assert not result["improved"]


def test_every_stress_scenario_moves_what_its_name_says():
    base = ExecConfig(obs_latency_ns=4, order_latency_ns=9, cancel_latency_ns=9, participation_cap_pct=30)
    scenarios = stress_scenarios(base)
    assert scenarios["base"] == base
    assert len(scenarios) == 19
    names = [f.name for f in fields(ExecConfig)]
    for name, config in scenarios.items():
        moved = {n for n in names if getattr(config, n) != getattr(base, n)}
        assert (name == "base") == (not moved), name
    assert scenarios["latency_plus_50ms"].order_latency_ns == 9 + 50_000_000
    assert scenarios["worst_case_fills"].trade_through_only


@pytest.fixture
def frozen_files(tmp_path, monkeypatch):
    fitted, freeze = tmp_path / "fitted.pkl", tmp_path / "freeze.json"
    fitted.write_bytes(b"model")
    freeze.write_text(json.dumps({"fitted_sha256": sha256_file(fitted), "input_sha256": "input"}))
    monkeypatch.setattr(pipeline, "FITTED_PATH", fitted)
    monkeypatch.setattr(pipeline, "FREEZE_PATH", freeze)
    monkeypatch.setattr(pipeline, "ensure_data", lambda plan: {"sha256": "input"})
    monkeypatch.setattr(pipeline, "_git", lambda *args: "")
    return fitted, freeze


def test_the_guard_accepts_a_matching_freeze(frozen_files):
    assert require_freeze(Study())["input_sha256"] == "input"


def test_the_guard_refuses_a_missing_freeze(frozen_files):
    frozen_files[1].unlink()
    with pytest.raises(SystemExit, match="No freeze file"):
        require_freeze(Study())


def test_the_guard_refuses_an_uncommitted_freeze(frozen_files, monkeypatch):
    monkeypatch.setattr(pipeline, "_git", lambda *args: " M results/freeze.json")
    with pytest.raises(SystemExit, match="not committed"):
        require_freeze(Study())


def test_the_guard_refuses_models_changed_after_the_freeze(frozen_files):
    frozen_files[0].write_bytes(b"retrained")
    with pytest.raises(SystemExit, match="Fitted artifacts changed"):
        require_freeze(Study())


def test_the_guard_refuses_changed_inputs(frozen_files, monkeypatch):
    monkeypatch.setattr(pipeline, "ensure_data", lambda plan: {"sha256": "other"})
    with pytest.raises(SystemExit, match="Inputs changed"):
        require_freeze(Study())
