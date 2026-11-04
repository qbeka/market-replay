"""Each baseline, run on the hand-worked book, follows its stated rule."""

from marketreplay.events import Event, Kind, Side
from marketreplay.execution import ExecConfig, Parent, run_session
from marketreplay.strategies import EqualTime, ImmediateCross, PassiveOnly, UrgencyRule, VolumeSchedule

CONFIG = ExecConfig(
    obs_latency_ns=1, order_latency_ns=1, cancel_latency_ns=1, decision_interval_ns=10, snapshot_interval_ns=1
)


def deep_book():
    events = [Event(1, 0, Kind.CLEAR)]
    for i, (side, price) in enumerate([(Side.BID, 100), (Side.ASK, 101)]):
        events.append(Event(2 + i, 0, Kind.ADD, side, 1 + i, price, 1000))
    events += [Event(4, 0, Kind.SNAPSHOT_END), Event(5, 2000, Kind.ADD, Side.BID, 9, 99, 1)]
    return events


def fills(strategy, parent=Parent(1, Side.BID, 20, 100, 1100)):
    [run] = run_session(deep_book(), [(parent, strategy)], CONFIG)
    run.ledger.check_conservation()
    return run.ledger.fills


def test_immediate_crosses_everything_on_the_first_tick():
    assert [(f.ts_ns, f.qty) for f in fills(ImmediateCross())] == [(101, 20)]


def test_equal_time_sends_equal_slices():
    result = fills(EqualTime(slices=4))
    assert [f.qty for f in result] == [5, 5, 5, 5]
    assert [f.ts_ns for f in result] == [101, 351, 601, 851]


def test_a_flat_volume_curve_reproduces_equal_time():
    curve = [(i + 1) / 2000 for i in range(2000)]
    assert [f.ts_ns for f in fills(VolumeSchedule(curve, 1, slices=4))] == [101, 351, 601, 851]


def test_volume_schedule_shifts_lots_toward_heavy_volume():
    # All volume comes after t=600, so the first slice of the horizon is empty.
    curve = [0.0 if i < 600 else (i - 599) / 1400 for i in range(2000)]
    assert [(f.ts_ns, f.qty) for f in fills(VolumeSchedule(curve, 1, slices=2))] == [(601, 20)]


def test_passive_only_rests_and_then_pays_the_deadline():
    result = fills(PassiveOnly())
    assert {f.role.name for f in result} == {"FORCED"}


def test_urgency_crosses_once_behind_by_more_than_its_threshold():
    result = fills(UrgencyRule(behind_lots=4))
    assert result[0].role.name == "TAKER"
    assert result[0].ts_ns > 101
