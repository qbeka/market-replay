"""Scoring and fitting on small data whose right answers are known."""

import numpy as np
import pytest

from marketreplay import fillmodel
from marketreplay.fillmodel import MoveModel, brier, calibration_table, expected_calibration_error, tune_family


def separable(rows=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    y = (x[:, 0] + rng.normal(0, 0.5, rows) > 0).astype(int)
    return x, y


def test_brier_by_hand():
    assert brier(np.array([1, 0]), np.array([0.8, 0.4])) == pytest.approx((0.04 + 0.16) / 2)


def test_calibration_bins_hold_equal_counts_and_every_row():
    y, p = np.array([0, 1] * 50), np.linspace(0, 1, 100)
    table = calibration_table(y, p, bins=5)
    assert [row["count"] for row in table] == [20] * 5
    assert sum(row["count"] * row["observed"] for row in table) == pytest.approx(y.sum())


def test_expected_calibration_error_is_zero_when_predictions_match_the_rate():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.5), bins=1) == pytest.approx(0)


def test_expected_calibration_error_sees_a_constant_overconfident_model():
    y = np.array([0, 1] * 50)
    assert expected_calibration_error(y, np.full(100, 0.9), bins=1) == pytest.approx(0.4)


@pytest.mark.parametrize("family", ["logistic", "boosted"])
def test_both_families_get_the_same_budget_and_beat_the_base_rate(family, monkeypatch):
    monkeypatch.setattr(fillmodel, "TUNING_TRIALS", 3)
    x, y = separable()
    model, trial = tune_family(family, x[:1200], y[:1200], x[1200:1600], y[1200:1600], seed=1)
    predicted = model.predict(x[1600:])
    assert len(trial.trials) == 3
    assert trial.best_brier == min(trial.trials)
    assert brier(y[1600:], predicted) < brier(y[1600:], np.full(400, y[1600:].mean()))
    assert predicted.min() >= 0 and predicted.max() <= 1


def test_tuning_repeats_exactly_from_its_seed(monkeypatch):
    monkeypatch.setattr(fillmodel, "TUNING_TRIALS", 2)
    x, y = separable(800)
    first = tune_family("logistic", x[:400], y[:400], x[400:], y[400:], seed=9)[1]
    second = tune_family("logistic", x[:400], y[:400], x[400:], y[400:], seed=9)[1]
    assert first.trials == second.trials


def test_the_unfilled_move_model_ignores_filled_rows():
    x = np.linspace(-1, 1, 200).reshape(-1, 1)
    filled = np.array([0, 1] * 100)
    move = np.where(filled == 1, 100.0, x[:, 0])  # Filled rows carry a value that must not leak in.
    unfilled, overall = MoveModel().fit(x, move, filled).predict(np.array([[0.0]]))
    assert abs(unfilled[0]) < 1
    assert overall[0] > 40
