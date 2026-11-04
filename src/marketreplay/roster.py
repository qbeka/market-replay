"""Every named strategy in the study, built from artifacts fitted on earlier data."""

from dataclasses import dataclass, field

from .execution import ExecConfig, Parent, Strategy
from .policy import LearnedPolicy, PolicyParams
from .strategies import EqualTime, ImmediateCross, PassiveOnly, RandomPolicy, UrgencyRule, VolumeSchedule
from .synthetic import NS_PER_S

CLIP_LOTS = 16
ORACLE_BUCKET_NS = 2 * NS_PER_S


@dataclass
class Fitted:
    """What a strategy may know. Everything here comes from days before the ones it trades."""

    config: ExecConfig
    volume_curve: list[float]
    fill_model: object = None
    move_model: object = None
    policies: dict = field(default_factory=dict)  # name -> PolicyParams
    urgency_behind: dict = field(default_factory=lambda: {"urgency": 20})  # name -> lots
    comparator: object = None


def build(fitted: Fitted, name: str, parent: Parent, realized_curve: list[float] | None = None) -> Strategy:
    """A fresh strategy for one parent. Seeds come from the parent, so paired runs repeat exactly.

    realized_curve is the session's own final volume. Only oracle_volume may use it. That
    strategy cannot exist in practice, so it is reported apart and is never a baseline to beat.
    """
    if name == "oracle_volume":
        return VolumeSchedule(realized_curve, ORACLE_BUCKET_NS, name=name)
    if name == "immediate":
        return ImmediateCross()
    if name == "equal_time":
        return EqualTime()
    if name == "volume_schedule":
        return VolumeSchedule(fitted.volume_curve, 60 * NS_PER_S)
    if name == "passive_only":
        return PassiveOnly(CLIP_LOTS)
    if name == "hv_random":
        return RandomPolicy(parent.parent_id, CLIP_LOTS)
    if name in fitted.urgency_behind:
        return UrgencyRule(fitted.urgency_behind[name], CLIP_LOTS)
    if name in fitted.policies:
        config = fitted.config
        return LearnedPolicy(
            fitted.fill_model, fitted.move_model, fitted.policies[name],
            config.taker_fee_mticks, config.maker_fee_mticks, name,
        )  # fmt: skip
    if name == "fitted_q" and fitted.comparator is not None:
        return fitted.comparator.strategy(name)
    raise KeyError(name)
