"""Calibrated fill probability and expected price move.

Fit order, all on earlier data:

    fit days        models learn here
    calibration     later training days pick hyperparameters and fit calibration

Both model families get the same number of random trials, so neither wins by
getting more tuning. Scores are reported on days neither stage used.
"""

import random
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TUNING_TRIALS = 12


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-count bins of predicted probability against the observed rate."""
    order = np.argsort(p)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append(
            {
                "predicted": float(p[chunk].mean()),
                "observed": float(y[chunk].mean()),
                "count": int(len(chunk)),
            }
        )
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(y, p, bins)
    return float(sum(r["count"] * abs(r["predicted"] - r["observed"]) for r in table) / len(y))


def logistic_trial(rng: random.Random):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=10 ** rng.uniform(-3, 2), max_iter=500),
    )


def boosted_trial(rng: random.Random):
    return HistGradientBoostingClassifier(
        learning_rate=10 ** rng.uniform(-2, -0.7),
        max_leaf_nodes=rng.choice([7, 15, 31]),
        min_samples_leaf=rng.choice([50, 100, 200, 400]),
        l2_regularization=10 ** rng.uniform(-3, 1),
        max_iter=rng.choice([100, 200, 300]),
        random_state=0,
    )


FAMILIES = {"logistic": logistic_trial, "boosted": boosted_trial}


@dataclass
class TrialResult:
    family: str
    best_brier: float
    trials: list[float]
    params: dict


class CalibratedModel:
    def __init__(self, model, calibrator: IsotonicRegression):
        self.model = model
        self.calibrator = calibrator

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.calibrator.predict(self.model.predict_proba(x)[:, 1])


def tune_family(family: str, x_fit, y_fit, x_cal, y_cal, seed: int) -> tuple[CalibratedModel, TrialResult]:
    """Random search with a fixed budget. Calibration is fit on half the calibration days' rows."""
    rng = random.Random(seed)
    halves = np.arange(len(y_cal)) % 2 == 0
    best, best_score, scores = None, float("inf"), []
    for _ in range(TUNING_TRIALS):
        model = FAMILIES[family](rng)
        model.fit(x_fit, y_fit)
        score = brier(y_cal[~halves], model.predict_proba(x_cal[~halves])[:, 1])
        scores.append(score)
        if score < best_score:
            best, best_score = model, score
    raw = best.predict_proba(x_cal[halves])[:, 1]
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, y_cal[halves])
    return CalibratedModel(best, calibrator), TrialResult(family, best_score, scores, {})


class MoveModel:
    """Expected signed price move over the label horizon, split by fill outcome.

    filled and unfilled are separate regressions because their gap is the adverse
    selection a resting order faces.
    """

    def __init__(self):
        self.unfilled = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        self.overall = make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    def fit(self, x: np.ndarray, move: np.ndarray, filled: np.ndarray) -> "MoveModel":
        self.overall.fit(x, move)
        self.unfilled.fit(x[filled == 0], move[filled == 0])
        return self

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.unfilled.predict(x), self.overall.predict(x)


class TreeMoveModel(MoveModel):
    def __init__(self):
        self.unfilled = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, min_samples_leaf=200, random_state=0)
        self.overall = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=15, min_samples_leaf=200, random_state=0)
