"""
Temporal Fusion for Subjective Logic Opinions.

Combines temporal decay (:mod:`confidence_decay`) with fusion
(:mod:`confidence_algebra`) and optional Byzantine filtering
(:mod:`confidence_byzantine`) in a single composable pipeline.

Core idea: **newer evidence should count more**.  Before fusing
a set of timestamped opinions, each is decayed according to its
age relative to a reference time.  The decayed opinions are then
fused (cumulative or averaging), and optionally filtered for
adversarial agents.

Three entry points, from simple to full-featured:

    temporal_fuse()
        Decay → Fuse.  Uniform half-life for all sources.

    temporal_fuse_weighted()
        Decay → Fuse.  Per-source half-lives (e.g., academic sources
        decay slower than social-media posts).

    temporal_byzantine_fuse()
        Decay → Byzantine filter → Fuse.  The complete pipeline.

All three are **additive** — they compose existing modules without
modifying them.

Usage::

    from jsonld_ex.confidence_temporal_fusion import (
        TimestampedOpinion, TemporalFusionConfig, temporal_fuse,
    )

    opinions = [
        TimestampedOpinion(opinion_a, datetime(2025, 6, 1, tzinfo=UTC)),
        TimestampedOpinion(opinion_b, datetime(2025, 7, 1, tzinfo=UTC)),
    ]
    cfg = TemporalFusionConfig(half_life=86400.0 * 7)  # 1-week half-life
    report = temporal_fuse(opinions, cfg)
    print(report.fused)

References:
    Jøsang, A. (2016). Subjective Logic, §10.4 (Opinion Aging).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal, Optional, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
)
from jsonld_ex.confidence_decay import (
    DecayFunction,
    decay_opinion,
    exponential_decay,
)
from jsonld_ex.confidence_byzantine import (
    ByzantineConfig,
    ByzantineFusionReport,
    byzantine_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TimestampedOpinion:
    """An opinion tagged with the time it was formed.

    Attributes:
        opinion:   The subjective opinion.
        timestamp: UTC datetime when the opinion was formed / evidence
                   was gathered.
        source_id: Optional identifier for the source (used by
                   :func:`temporal_fuse_weighted` to look up
                   per-source half-lives).
    """

    opinion: Opinion
    timestamp: datetime
    source_id: Optional[str] = None


@dataclass(frozen=True)
class TemporalFusionConfig:
    """Configuration for :func:`temporal_fuse`.

    Attributes:
        half_life:      Time (in seconds) for belief/disbelief to halve.
        decay_fn:       Custom decay function.  ``None`` = exponential.
        fusion_method:  ``"cumulative"`` or ``"averaging"``.
        reference_time: The "now" against which age is measured.
                        ``None`` = ``datetime.now(UTC)`` at call time.
    """

    half_life: float
    decay_fn: Optional[DecayFunction] = None
    fusion_method: Literal["cumulative", "averaging"] = "cumulative"
    reference_time: Optional[datetime] = None


@dataclass(frozen=True)
class TemporalFusionReport:
    """Report from :func:`temporal_fuse`.

    Attributes:
        fused:            The final fused opinion.
        decayed_opinions: List of opinions after temporal decay
                          (same order as input).
        reference_time:   The reference time that was used.
        config:           The config that was used (for reproducibility).
    """

    fused: Opinion
    decayed_opinions: list[Opinion]
    reference_time: datetime
    config: TemporalFusionConfig


# ═══════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════


def _resolve_reference_time(cfg: TemporalFusionConfig) -> datetime:
    """Return the reference time, defaulting to UTC now."""
    return cfg.reference_time if cfg.reference_time is not None else datetime.now(timezone.utc)


def _decay_one(
    ts_opinion: TimestampedOpinion,
    ref: datetime,
    half_life: float,
    decay_fn: Optional[DecayFunction],
) -> Opinion:
    """Decay a single timestamped opinion relative to *ref*."""
    elapsed = (ref - ts_opinion.timestamp).total_seconds()
    if elapsed < 0:
        raise ValueError(
            f"Opinion timestamp {ts_opinion.timestamp.isoformat()} is in the "
            f"future relative to reference_time {ref.isoformat()}"
        )
    return decay_opinion(
        ts_opinion.opinion,
        elapsed=elapsed,
        half_life=half_life,
        decay_fn=decay_fn,
    )


def _fuse_many(
    opinions: list[Opinion],
    method: Literal["cumulative", "averaging"],
) -> Opinion:
    """Fuse a list of opinions using the specified method."""
    if method == "averaging":
        return averaging_fuse(*opinions)
    return cumulative_fuse(*opinions)


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════


def temporal_fuse(
    opinions: Sequence[TimestampedOpinion],
    config: TemporalFusionConfig,
) -> TemporalFusionReport:
    """Decay each opinion by age, then fuse.

    All opinions share the same ``config.half_life``.  For per-source
    half-lives, use :func:`temporal_fuse_weighted`.

    Args:
        opinions: Timestamped opinions to fuse.
        config:   Temporal fusion configuration.

    Returns:
        :class:`TemporalFusionReport`.

    Raises:
        ValueError: If opinions is empty or any timestamp is in the
                    future relative to reference_time.
    """
    if len(opinions) == 0:
        raise ValueError("temporal_fuse requires at least one opinion")

    ref = _resolve_reference_time(config)

    decayed = [
        _decay_one(ts_op, ref, config.half_life, config.decay_fn)
        for ts_op in opinions
    ]

    fused = _fuse_many(decayed, config.fusion_method)

    return TemporalFusionReport(
        fused=fused,
        decayed_opinions=decayed,
        reference_time=ref,
        config=config,
    )


def temporal_fuse_weighted(
    opinions: Sequence[TimestampedOpinion],
    half_life_map: dict[str, float],
    default_half_life: float = 86400.0,
    decay_fn: Optional[DecayFunction] = None,
    fusion_method: Literal["cumulative", "averaging"] = "cumulative",
    reference_time: Optional[datetime] = None,
) -> TemporalFusionReport:
    """Decay with per-source half-lives, then fuse.

    Each opinion's ``source_id`` is looked up in *half_life_map*.
    Sources not found use *default_half_life*.

    Args:
        opinions:          Timestamped opinions.
        half_life_map:     ``{source_id: half_life_seconds}``.
        default_half_life: Fallback half-life for unknown sources.
        decay_fn:          Custom decay function (default: exponential).
        fusion_method:     ``"cumulative"`` or ``"averaging"``.
        reference_time:    ``None`` = UTC now.

    Returns:
        :class:`TemporalFusionReport`.

    Raises:
        ValueError: If opinions is empty or any timestamp is future.
    """
    if len(opinions) == 0:
        raise ValueError("temporal_fuse_weighted requires at least one opinion")

    ref = reference_time if reference_time is not None else datetime.now(timezone.utc)

    decayed: list[Opinion] = []
    for ts_op in opinions:
        hl = half_life_map.get(ts_op.source_id, default_half_life) if ts_op.source_id else default_half_life
        decayed.append(_decay_one(ts_op, ref, hl, decay_fn))

    fused = _fuse_many(decayed, fusion_method)

    # Build a config object for the report (uses default_half_life as representative)
    cfg = TemporalFusionConfig(
        half_life=default_half_life,
        decay_fn=decay_fn,
        fusion_method=fusion_method,
        reference_time=ref,
    )

    return TemporalFusionReport(
        fused=fused,
        decayed_opinions=decayed,
        reference_time=ref,
        config=cfg,
    )


def temporal_byzantine_fuse(
    opinions: Sequence[TimestampedOpinion],
    temporal_config: TemporalFusionConfig,
    byzantine_config: Optional[ByzantineConfig] = None,
) -> ByzantineFusionReport:
    """Full pipeline: decay by age → Byzantine filter → fuse.

    1. Each opinion is decayed according to its age.
    2. The decayed opinions are passed to :func:`byzantine_fuse`
       which removes adversarial agents and fuses the remainder.

    Args:
        opinions:          Timestamped opinions.
        temporal_config:   Temporal decay configuration.
        byzantine_config:  Byzantine fusion configuration.
                           ``None`` uses defaults.

    Returns:
        :class:`ByzantineFusionReport` (from confidence_byzantine).
        The ``removed`` indices refer to positions in the original
        *opinions* list.

    Raises:
        ValueError: If opinions is empty or timestamps are future.
    """
    if len(opinions) == 0:
        raise ValueError("temporal_byzantine_fuse requires at least one opinion")

    ref = _resolve_reference_time(temporal_config)

    decayed = [
        _decay_one(ts_op, ref, temporal_config.half_life, temporal_config.decay_fn)
        for ts_op in opinions
    ]

    return byzantine_fuse(decayed, config=byzantine_config)
