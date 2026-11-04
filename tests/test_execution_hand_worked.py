"""A deterministic execution worked by hand before the engine existed.

Feed, in ticks and lots. Buy parent of 11 lots, start 100 ns, deadline 1000 ns.
Latencies: observation 10, order 5, cancel 7. Fees: taker +2000, maker -500 mticks/lot.

    t=0    bids 100: id1 x4, id2 x3 | 99: id6 x10    asks 102: id3 x5 | 103: id4 x5
    t=100  arrival mid is 101. Decide REST 6 at 100.
    t=105  child 1 arrives behind 7 lots. Queue ahead = 7.
    t=110  id5 x2 adds at 100, behind child 1.
    t=120  id2 cancels 3. Queue ahead 7 -> 4.
    t=130  id1 executes 3. Queue ahead 4 -> 1. No fill.
    t=140  id1 executes 1. Queue ahead 1 -> 0. No fill.
    t=150  id5 executes 2. In the replayed market id5 sits behind child 1, so
           child 1 is hit instead: fill 2 at 100, maker.
    t=160  Decide WAIT. Cancel requested, effective t=167.
    t=165  id6 executes 3 at 99. Sellers went through 100 while the cancel was
           in flight: fill 3 at 100, maker. Child 1 has 1 lot left.
    t=167  Cancel lands. 1 lot cancelled. Child 1 ends with 5 filled.
    t=180  Decide CROSS 3, limit 103. Arrives t=185: 3 at 102, taker.
    t=200  Decide CROSS 3. Arrives t=205. 102 shows 5 but 3 were consumed by
           this ledger, so only 2 remain: 2 at 102, then 1 at 103, taker.

    Fills: 2@100 M, 3@100 M, 3@102 T, 2@102 T, 1@103 T = 11 lots
    Notional: 200 + 300 + 306 + 204 + 103 = 1113 ticks
    Slippage against 101: -5 + 5 + 2 = +2 ticks = 2000 mticks
    Fees: 5 x -500 + 6 x 2000 = 9500 mticks
    Cash: -(1113 x 1000) - 9500 = -1,122,500 mticks
    IS: 10,000 x (2000 + 9500) / 1000 / (11 x 101) = 103.5103... bps
"""

import pytest

from marketreplay.events import Event, Kind, Side
from marketreplay.execution import ExecConfig, Intent, Parent, ScriptedStrategy, run_session
from marketreplay.ledger import Role
from marketreplay.metrics import forced_lots, shortfall_bps

CONFIG = ExecConfig(
    obs_latency_ns=10,
    order_latency_ns=5,
    cancel_latency_ns=7,
    taker_fee_mticks=2000,
    maker_fee_mticks=-500,
    decision_interval_ns=20,
    snapshot_interval_ns=10,
    completion_penalty_ticks=1,
)


def opening() -> list[Event]:
    return [
        Event(1, 0, Kind.CLEAR),
        Event(2, 0, Kind.ADD, Side.BID, 1, 100, 4),
        Event(3, 0, Kind.ADD, Side.BID, 2, 100, 3),
        Event(4, 0, Kind.ADD, Side.BID, 6, 99, 10),
        Event(5, 0, Kind.ADD, Side.ASK, 3, 102, 5),
        Event(6, 0, Kind.ADD, Side.ASK, 4, 103, 5),
        Event(7, 0, Kind.SNAPSHOT_END),
    ]


def feed() -> list[Event]:
    return opening() + [
        Event(8, 110, Kind.ADD, Side.BID, 5, 100, 2),
        Event(9, 120, Kind.CANCEL, Side.BID, 2, 100, 3),
        Event(10, 130, Kind.EXECUTE, Side.BID, 1, 100, 3),
        Event(11, 140, Kind.EXECUTE, Side.BID, 1, 100, 1),
        Event(12, 150, Kind.EXECUTE, Side.BID, 5, 100, 2),
        Event(13, 165, Kind.EXECUTE, Side.BID, 6, 99, 3),
        Event(14, 1100, Kind.ADD, Side.BID, 7, 98, 1),
    ]


def run(script, events=None, config=CONFIG, parent=None):
    parent = parent or Parent(1, Side.BID, 11, 100, 1000)
    strategy = ScriptedStrategy(script)
    runs = run_session(events or feed(), [(parent, strategy)], config)
    return runs[0]


SCRIPT = {
    100: Intent.rest(6),
    160: Intent.wait(),
    180: Intent.cross(3),
    200: Intent.cross(3),
}


def test_the_hand_worked_fills():
    ledger = run(SCRIPT).ledger
    assert [(f.ts_ns, f.price, f.qty, f.role, f.fee_mticks) for f in ledger.fills] == [
        (150, 100, 2, Role.MAKER, -1000),
        (165, 100, 3, Role.MAKER, -1500),
        (185, 102, 3, Role.TAKER, 6000),
        (205, 102, 2, Role.TAKER, 4000),
        (205, 103, 1, Role.TAKER, 2000),
    ]


def test_cash_inventory_remaining_and_fees_are_conserved():
    ledger = run(SCRIPT).ledger
    ledger.check_conservation()
    assert ledger.cash_mticks == -1_122_500
    assert ledger.filled_lots == 11
    assert ledger.remaining_lots == 0
    assert ledger.fees_mticks == 9500


def test_the_shortfall_matches_the_hand_calculation():
    assert shortfall_bps(run(SCRIPT).ledger) == pytest.approx(10_000 * 11.5 / (11 * 101))


def test_the_recorded_feed_is_never_rewritten():
    events = feed()
    before = list(events)
    run(SCRIPT, events)
    assert events == before


def test_a_sell_parent_mirrors_the_sign_of_cost():
    parent = Parent(1, Side.ASK, 2, 100, 1000)
    ledger = run({100: Intent.cross(2)}, parent=parent).ledger
    assert [(f.price, f.qty) for f in ledger.fills] == [(100, 2)]
    # Sold 2 at 100 against an arrival mid of 101, so slippage is +1 tick per lot.
    assert shortfall_bps(ledger) == pytest.approx(10_000 * (2 + 4) / (2 * 101))


def test_an_unfinished_order_is_completed_at_the_deadline_with_a_penalty():
    ledger = run({}, parent=Parent(1, Side.BID, 7, 100, 1000)).ledger
    assert [(f.ts_ns, f.price, f.qty, f.role) for f in ledger.fills] == [
        (1000, 102, 5, Role.FORCED),
        (1000, 103, 2, Role.FORCED),
    ]
    assert ledger.penalties_mticks == 7000
    assert forced_lots(ledger) == 7
    ledger.check_conservation()


def test_doing_nothing_costs_more_than_crossing_at_once():
    idle = shortfall_bps(run({}, parent=Parent(1, Side.BID, 7, 100, 1000)).ledger)
    crossed = shortfall_bps(run({100: Intent.cross(7)}, parent=Parent(1, Side.BID, 7, 100, 1000)).ledger)
    assert idle > crossed


def test_a_cross_never_reuses_liquidity_it_already_consumed():
    script = {100: Intent.cross(4), 120: Intent.cross(4)}
    ledger = run(script, parent=Parent(1, Side.BID, 8, 100, 1000)).ledger
    assert [(f.price, f.qty) for f in ledger.fills] == [(102, 4), (102, 1), (103, 3)]


def test_two_ledgers_do_not_share_consumed_liquidity():
    parents = [
        (Parent(1, Side.BID, 5, 100, 1000), ScriptedStrategy({100: Intent.cross(5)})),
        (Parent(2, Side.BID, 5, 100, 1000), ScriptedStrategy({100: Intent.cross(5)})),
    ]
    first, second = run_session(feed(), parents, CONFIG)
    assert [(f.price, f.qty) for f in first.ledger.fills] == [(102, 5)]
    assert [(f.price, f.qty) for f in second.ledger.fills] == [(102, 5)]


def test_no_cancel_credit_keeps_the_order_behind_the_queue():
    config = ExecConfig(**{**CONFIG.__dict__, "cancel_ahead_credit_pct": 0})
    # 7 ahead, cancel of 3 gives no credit, executes of 3 + 1 + 2 leave 1 ahead.
    # The 3 lots at 99 trade through, which clears the queue: fill 3 at 100.
    ledger = run({100: Intent.rest(6), 980: Intent.cross(8)}, config=config).ledger
    assert ledger.fills[0].ts_ns == 165
    assert (ledger.fills[0].price, ledger.fills[0].qty) == (100, 3)


def test_requiring_a_trade_through_removes_at_touch_fills():
    config = ExecConfig(**{**CONFIG.__dict__, "trade_through_only": True})
    ledger = run({100: Intent.rest(6), 160: Intent.wait(), 180: Intent.cross(8)}, config=config).ledger
    assert (ledger.fills[0].ts_ns, ledger.fills[0].qty) == (165, 3)


def test_the_participation_cap_limits_fills_inside_the_window():
    config = ExecConfig(
        **{**CONFIG.__dict__, "participation_cap_pct": 50, "participation_window_ns": 1000, "participation_free_lots": 0}
    )
    # At t=150 the tape shows 3 + 1 + 2 = 6 lots, so 50% allows 3 and the 2 lot fill passes.
    # At t=165 the tape shows 9 lots, so 50% allows 4 in all. 2 are used, so 2 of 3 pass.
    ledger = run(SCRIPT, config=config).ledger
    assert [(f.ts_ns, f.qty) for f in ledger.fills][:2] == [(150, 2), (165, 2)]


def test_a_new_order_cannot_overfill_while_a_cancel_is_in_flight():
    # Rest 6, then ask to cross everything while the cancel is unconfirmed. Only the
    # lots not already open may be sent, so total fills can never exceed the parent.
    script = {100: Intent.rest(6), 160: Intent.cross(11)}
    ledger = run(script).ledger
    ledger.check_conservation()
    assert ledger.filled_lots == 11
