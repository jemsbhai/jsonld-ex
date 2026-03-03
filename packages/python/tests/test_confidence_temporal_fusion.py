"""Tests for temporal fusion (confidence_temporal_fusion.py).

TDD: Tests written FIRST — all will fail until implementation exists.

New types and functions:

    TimestampedOpinion — dataclass(opinion, timestamp, source_id?)
    TemporalFusionConfig — dataclass(half_life, decay_fn, fusion_method, reference_time?)
    TemporalFusionReport — dataclass(fused, decayed_opinions, reference_time, config)

    temporal_fuse(opinions, config) → TemporalFusionReport
    temporal_fuse_weighted(opinions, half_life_map, ...) → TemporalFusionReport
    temporal_byzantine_fuse(opinions, temporal_config, byzantine_config) → ByzantineFusionReport

Design:
    - Composes with confidence_decay (decay each opinion by age) then fuses
    - Composes with confidence_byzantine for the full pipeline
    - All additive — no modifications to existing modules
"""

import pytest
from datetime import datetime, timezone, timedelta

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
from jsonld_ex.confidence_decay import exponential_decay, linear_decay
from jsonld_ex.confidence_byzantine import ByzantineConfig, ByzantineFusionReport

from jsonld_ex.confidence_temporal_fusion import (
    TimestampedOpinion,
    TemporalFusionConfig,
    TemporalFusionReport,
    temporal_fuse,
    temporal_fuse_weighted,
    temporal_byzantine_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# TimestampedOpinion
# ═══════════════════════════════════════════════════════════════════


class TestTimestampedOpinion:

    def test_basic_construction(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        top = TimestampedOpinion(opinion=o, timestamp=ts)
        assert top.opinion is o
        assert top.timestamp == ts
        assert top.source_id is None

    def test_with_source_id(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        top = TimestampedOpinion(opinion=o, timestamp=ts, source_id="reuters-001")
        assert top.source_id == "reuters-001"


# ═══════════════════════════════════════════════════════════════════
# TemporalFusionConfig
# ═══════════════════════════════════════════════════════════════════


class TestTemporalFusionConfig:

    def test_default_values(self):
        cfg = TemporalFusionConfig(half_life=86400.0)
        assert cfg.half_life == 86400.0
        assert cfg.decay_fn is None  # uses exponential_decay
        assert cfg.fusion_method == "cumulative"
        assert cfg.reference_time is None  # uses utcnow

    def test_custom_values(self):
        ref = datetime(2025, 7, 1, tzinfo=timezone.utc)
        cfg = TemporalFusionConfig(
            half_life=3600.0,
            decay_fn=linear_decay,
            fusion_method="averaging",
            reference_time=ref,
        )
        assert cfg.half_life == 3600.0
        assert cfg.decay_fn is linear_decay
        assert cfg.fusion_method == "averaging"
        assert cfg.reference_time == ref


# ═══════════════════════════════════════════════════════════════════
# temporal_fuse — core
# ═══════════════════════════════════════════════════════════════════


class TestTemporalFuse:

    def _now(self):
        return datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_fresh_opinions_barely_decay(self):
        """Opinions from 1 second ago with 1-day half-life → negligible decay."""
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(seconds=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                now - timedelta(seconds=1),
            ),
        ]
        cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        # Almost no decay → belief should stay high
        assert report.fused.belief > 0.7

    def test_old_opinions_decay_heavily(self):
        """Opinions from 10 half-lives ago → nearly vacuous."""
        now = self._now()
        half_life = 3600.0  # 1 hour
        old_time = now - timedelta(seconds=half_life * 10)
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
                old_time,
            ),
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                old_time,
            ),
        ]
        cfg = TemporalFusionConfig(half_life=half_life, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        # After 10 half-lives, belief ≈ 0.9 * 2^-10 ≈ 0.0009
        assert report.fused.belief < 0.05

    def test_newer_opinion_dominates(self):
        """A fresh opinion should contribute more than a stale one."""
        now = self._now()
        half_life = 3600.0
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.2, disbelief=0.7, uncertainty=0.1),
                now - timedelta(seconds=1),  # fresh: DISBELIEVES
            ),
            TimestampedOpinion(
                Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
                now - timedelta(hours=10),  # stale: believes
            ),
        ]
        cfg = TemporalFusionConfig(half_life=half_life, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        # Fresh disbelief should dominate stale belief
        assert report.fused.disbelief > report.fused.belief

    def test_returns_report(self):
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
        ]
        cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        assert isinstance(report, TemporalFusionReport)
        assert isinstance(report.fused, Opinion)
        assert len(report.decayed_opinions) == 1
        assert report.reference_time == now

    def test_decayed_opinions_in_report(self):
        """Report should contain the post-decay opinions for inspection."""
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                now - timedelta(hours=5),
            ),
        ]
        cfg = TemporalFusionConfig(half_life=3600.0, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        assert len(report.decayed_opinions) == 2
        # First should have less decay than second
        assert report.decayed_opinions[0].belief > report.decayed_opinions[1].belief

    def test_custom_decay_fn(self):
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(hours=3),
            ),
        ]
        cfg = TemporalFusionConfig(
            half_life=3600.0,
            decay_fn=linear_decay,
            reference_time=now,
        )
        report = temporal_fuse(opinions, cfg)
        assert isinstance(report.fused, Opinion)

    def test_averaging_fusion_method(self):
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
                now - timedelta(hours=1),
            ),
        ]
        cfg = TemporalFusionConfig(
            half_life=86400.0,
            fusion_method="averaging",
            reference_time=now,
        )
        report = temporal_fuse(opinions, cfg)
        assert isinstance(report.fused, Opinion)

    def test_empty_raises(self):
        cfg = TemporalFusionConfig(half_life=3600.0)
        with pytest.raises(ValueError):
            temporal_fuse([], cfg)

    def test_single_opinion(self):
        now = self._now()
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        opinions = [TimestampedOpinion(o, now - timedelta(seconds=1))]
        cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        report = temporal_fuse(opinions, cfg)
        # Barely decayed
        assert report.fused.belief == pytest.approx(o.belief, abs=0.01)

    def test_reference_time_defaults_to_now(self):
        """When reference_time is None, it should use current UTC time."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        opinions = [TimestampedOpinion(o, datetime.now(timezone.utc) - timedelta(seconds=1))]
        cfg = TemporalFusionConfig(half_life=86400.0)
        report = temporal_fuse(opinions, cfg)
        assert report.reference_time is not None
        assert report.fused.belief > 0.7

    def test_future_timestamp_raises(self):
        """Opinions from the future (negative elapsed) should raise."""
        now = datetime(2025, 7, 1, tzinfo=timezone.utc)
        future = now + timedelta(hours=1)
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                future,
            ),
        ]
        cfg = TemporalFusionConfig(half_life=3600.0, reference_time=now)
        with pytest.raises(ValueError, match="future"):
            temporal_fuse(opinions, cfg)


# ═══════════════════════════════════════════════════════════════════
# temporal_fuse_weighted — per-source half-lives
# ═══════════════════════════════════════════════════════════════════


class TestTemporalFuseWeighted:

    def _now(self):
        return datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_academic_decays_slower(self):
        """Academic source (long half-life) retains more belief than news."""
        now = self._now()
        age = timedelta(hours=12)
        academic = TimestampedOpinion(
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            now - age,
            source_id="nature-001",
        )
        news = TimestampedOpinion(
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            now - age,
            source_id="reddit-001",
        )
        half_life_map = {
            "nature-001": 86400.0 * 30,  # 30 days
            "reddit-001": 3600.0,         # 1 hour
        }
        report = temporal_fuse_weighted(
            [academic, news],
            half_life_map=half_life_map,
            reference_time=now,
        )
        # Academic opinion should dominate
        assert report.decayed_opinions[0].belief > report.decayed_opinions[1].belief

    def test_default_half_life_for_unknown_source(self):
        """Sources not in half_life_map use default_half_life."""
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1),
                now - timedelta(hours=1),
                source_id="unknown-source",
            ),
        ]
        report = temporal_fuse_weighted(
            opinions,
            half_life_map={},
            default_half_life=86400.0,
            reference_time=now,
        )
        assert isinstance(report.fused, Opinion)
        assert report.fused.belief > 0.6

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            temporal_fuse_weighted(
                [],
                half_life_map={},
                default_half_life=3600.0,
            )

    def test_source_id_none_uses_default(self):
        """Opinions without source_id use default_half_life."""
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
        ]
        report = temporal_fuse_weighted(
            opinions,
            half_life_map={"academic": 86400.0},
            default_half_life=86400.0,
            reference_time=now,
        )
        assert isinstance(report.fused, Opinion)


# ═══════════════════════════════════════════════════════════════════
# temporal_byzantine_fuse — composed pipeline
# ═══════════════════════════════════════════════════════════════════


class TestTemporalByzantineFuse:
    """Full pipeline: decay by age → filter adversarial → fuse."""

    def _now(self):
        return datetime(2025, 7, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_stale_rogue_removed(self):
        """A stale rogue should be doubly disadvantaged:
        decayed (less influence) AND conflicting (removed)."""
        now = self._now()
        honest_fresh = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(minutes=5),
            ),
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                now - timedelta(minutes=10),
            ),
            TimestampedOpinion(
                Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
                now - timedelta(minutes=15),
            ),
        ]
        rogue_stale = TimestampedOpinion(
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            now - timedelta(hours=10),
        )

        t_cfg = TemporalFusionConfig(half_life=3600.0, reference_time=now)
        b_cfg = ByzantineConfig(threshold=0.15)

        report = temporal_byzantine_fuse(
            honest_fresh + [rogue_stale],
            temporal_config=t_cfg,
            byzantine_config=b_cfg,
        )
        assert isinstance(report, ByzantineFusionReport)
        assert report.fused.belief > 0.3

    def test_returns_byzantine_report(self):
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
                now - timedelta(hours=2),
            ),
        ]
        t_cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        b_cfg = ByzantineConfig()
        report = temporal_byzantine_fuse(opinions, t_cfg, b_cfg)
        assert isinstance(report, ByzantineFusionReport)
        assert isinstance(report.fused, Opinion)
        assert len(report.conflict_matrix) == 2

    def test_empty_raises(self):
        t_cfg = TemporalFusionConfig(half_life=3600.0)
        b_cfg = ByzantineConfig()
        with pytest.raises(ValueError):
            temporal_byzantine_fuse([], t_cfg, b_cfg)

    def test_fresh_rogue_still_removed(self):
        """Even a fresh rogue should be removed by Byzantine filtering."""
        now = self._now()
        honest = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(minutes=5),
            ),
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                now - timedelta(minutes=10),
            ),
            TimestampedOpinion(
                Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
                now - timedelta(minutes=15),
            ),
        ]
        fresh_rogue = TimestampedOpinion(
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            now - timedelta(minutes=1),  # very fresh
        )

        t_cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        b_cfg = ByzantineConfig(threshold=0.15)

        report = temporal_byzantine_fuse(honest + [fresh_rogue], t_cfg, b_cfg)
        removed_indices = [r.index for r in report.removed]
        assert 3 in removed_indices
        assert report.fused.belief > 0.5

    def test_with_trust_weights(self):
        """Combined pipeline with trust-weighted Byzantine strategy."""
        now = self._now()
        opinions = [
            TimestampedOpinion(
                Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
                now - timedelta(hours=1),
            ),
            TimestampedOpinion(
                Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
                now - timedelta(hours=1),
            ),
        ]
        t_cfg = TemporalFusionConfig(half_life=86400.0, reference_time=now)
        b_cfg = ByzantineConfig(
            strategy="combined",
            trust_weights=[0.9, 0.9, 0.1],
        )
        report = temporal_byzantine_fuse(opinions, t_cfg, b_cfg)
        assert isinstance(report, ByzantineFusionReport)
