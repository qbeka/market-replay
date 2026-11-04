"""Causal features, built only from a delayed View.

Every signed feature is expressed from the parent's point of view, so a buy and a
sell in the same situation get the same numbers. near is the parent's own side of
the book, far is the side it would cross into.
"""

import math

import numpy as np

from .events import Side
from .execution import Snapshot, View

FEATURE_NAMES = [
    "spread_ticks",
    "near_touch_lots",
    "far_touch_lots",
    "touch_imbalance",  # Positive means more near than far lots.
    "depth_imbalance",  # Same, over the visible levels.
    "queue_ahead_lots",  # Own queue if resting, otherwise the near touch.
    "flow_toward_fill",  # Positive means aggressors are hitting our side.
    "trade_rate_lots_per_s",
    "mid_move_5s_ticks",  # Positive means the price moved against this parent.
    "mid_move_20s_ticks",
    "mid_vol_20s_ticks",
    "time_left_s",
    "remaining_lots",
]


def mid(snapshot: Snapshot) -> float | None:
    if not snapshot.bids or not snapshot.asks:
        return None
    return (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2


def features(view: View, trade_window_s: float = 10.0) -> np.ndarray | None:
    """One row of FEATURE_NAMES, or None when the observed book is one-sided."""
    market = view.market
    now_mid = mid(market)
    if now_mid is None:
        return None
    side = view.parent.side
    near_levels, far_levels = (market.bids, market.asks) if side == Side.BID else (market.asks, market.bids)
    near_lots, far_lots = near_levels[0][1], far_levels[0][1]
    near_depth = sum(q for _, q in near_levels)
    far_depth = sum(q for _, q in far_levels)
    hitting_us = market.sell_lots if side == Side.BID else market.buy_lots
    other_way = market.buy_lots if side == Side.BID else market.sell_lots
    history = view.history

    def move(seconds: int) -> float:
        if len(history) <= seconds:
            return 0.0
        then = mid(history[-1 - seconds])
        return 0.0 if then is None else side * (now_mid - then)

    recent = [m for m in (mid(s) for s in history[-21:]) if m is not None]
    changes = np.diff(recent) if len(recent) > 2 else np.zeros(1)
    queue = view.queue_ahead_seen if view.queue_ahead_seen is not None else near_lots
    return np.array(
        [
            market.asks[0][0] - market.bids[0][0],
            near_lots,
            far_lots,
            (near_lots - far_lots) / (near_lots + far_lots),
            (near_depth - far_depth) / (near_depth + far_depth),
            queue,
            (hitting_us - other_way) / (hitting_us + other_way + 1),
            (hitting_us + other_way) / trade_window_s,
            move(5),
            move(20),
            float(np.std(changes)) * math.sqrt(20),
            view.time_left_ns / 1e9,
            view.remaining_seen,
        ],
        dtype=float,
    )
