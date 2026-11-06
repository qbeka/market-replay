"""The page may draw only what the recorded run says, at the time the run says it."""

import json

import pytest

from marketreplay.demo import build_frames, decision_stops, fill_sentence, price_words, why
from marketreplay.pipeline import RESULTS

LIMITS = {"behind_lots": 30, "final_s": 4.0, "urgency_behind": 30}
RUN = json.loads((RESULTS / "demo_run.json").read_text())
PARENTS = [(s["title"], s) for s in RUN["scenes"]]


def decision(t, action, lots, **info):
    return {"t": t, "what": "decision", "intent": [action, lots], **info}


def fill(t, qty, price, role, cost):
    return {"t": t, "price": price, "qty": qty, "role": role, "cost_bps": cost}


def event(row):
    return {"t": row["t"], "what": "fill", "price": row["price"], "qty": row["qty"], "role": row["role"]}


def test_a_trade_changes_the_frame_only_at_its_recorded_time():
    fills = [fill(2.009, 4, 100, "taker", 0.5)]
    events = [
        decision(0.0, "REST", 16),
        {"t": 0.0, "what": "send", "price": 101, "lots": 16, "immediate": False},
        {"t": 0.009, "what": "rest", "queue_ahead": 3},
        decision(2.0, "CROSS", 4),
        {"t": 2.0, "what": "cancel_request"},
        {"t": 2.0, "what": "send", "price": 100, "lots": 4, "immediate": True},
        {"t": 2.009, "what": "cancelled", "lots": 16},
        event(fills[0]),
    ]
    frames = build_frames("urgency", events, [], fills, "sell", 4, LIMITS)
    assert [f["t"] for f in frames] == [0.0, 0.009, 2.0, 2.009]
    assert [f["left"] for f in frames] == [4, 4, 4, 0]
    assert frames[1]["said"] == "Resting at 101, with 3 lots ahead."
    assert frames[3]["rest"] is None and frames[3]["hit"] == 100
    assert frames[3]["said"] == "Sold 4 lots at 100."


def test_a_cross_that_falls_short_says_so():
    assert fill_sentence("sell", [fill(1, 7, 90, "taker", 0)], 31) == "Sold 7 of the 31 lots it asked for, at 90."
    assert fill_sentence("buy", [fill(1, 2, 90, "maker", 0), fill(1, 5, 95, "forced", 0)], None) == "Forced to buy 5 lots at 95."


def test_prices_read_as_one_a_pair_or_a_range():
    assert [price_words([5, 5]), price_words([5, 6]), price_words([5, 6, 9])] == ["5", "5 and 6", "5 to 9"]


def test_each_reason_comes_from_the_recorded_rule():
    model = decision(0, "CROSS", 16, rule="model", p_fill=0.4066, costs={"rest": -1.5, "cross": 1.3})
    assert why("learned", model, LIMITS) == "At a fill chance of 41%, it prices resting at −1.50 ticks and crossing at 1.30."
    assert "last 4 seconds" in why("learned", decision(0, "CROSS", 5, rule="final"), LIMITS)
    assert why("urgency", decision(0, "CROSS", 31), LIMITS).startswith("It is 31 lots behind")


def policy(actions):
    events = [decision(float(t), action, 16) for t, action in actions]
    return {"events": events, "frames": [{"t": e["t"] + 0.009} for e in events]}


def test_playback_stops_only_where_one_rests_and_the_other_crosses():
    stops = decision_stops([policy([(0, "REST"), (2, "CROSS"), (4, "CROSS")]), policy([(0, "REST"), (2, "REST"), (4, "CROSS")])])
    assert stops == [{"t": 2.009, "decided": 2.0}]


@pytest.mark.parametrize("title,scene", PARENTS, ids=[t for t, _ in PARENTS])
def test_recorded_frames_add_up(title, scene):
    assert scene["stops"], title
    for p in scene["policies"]:
        frames = p["frames"]
        times = [f["t"] for f in frames]
        assert times == sorted(set(times))
        assert frames[-1]["left"] == 0 and frames[-1]["cost"] == p["shortfall_bps"]
        assert frames[-1]["action"] == "DONE"
        assert all(f["left"] >= 0 for f in frames)


@pytest.mark.parametrize("title,scene", PARENTS, ids=[t for t, _ in PARENTS])
def test_a_resting_fill_lands_where_the_page_shows_the_order(title, scene):
    for p in scene["policies"]:
        for row in (f for f in p["fills"] if f["role"] == "maker"):
            before = [f for f in p["frames"] if f["t"] < row["t"]]
            assert before[-1]["rest"] == row["price"], (title, p["name"], row)


def test_the_opening_book_is_the_one_the_first_decision_saw():
    for scene in RUN["scenes"]:
        first = scene["book"][0]
        assert first["t"] <= 0 < scene["book"][1]["t"]
        assert (first["bid"] + first["ask"]) / 2 == scene["arrival_mid"]


def test_the_worst_loss_stays_on_the_page():
    worst = next(s for s in RUN["scenes"] if s["title"] == "Worst loss")
    learned, other = worst["policies"]
    assert learned["shortfall_bps"] > other["shortfall_bps"]
    assert "worse" in worst["result"]
