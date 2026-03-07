"""Tests for infer_temporal_diff() -- opinion change over time (Tier 3, Step 7).

infer_temporal_diff() runs infer_at() at two time points and reports
how the inferred opinion changed. The result includes both opinions
and their component-wise deltas.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import SLNode, SLEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.temporal import TemporalDiffResult


# -- Fixtures --

T0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2025, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2026, 1, 1, tzinfo=timezone.utc)
ONE_DAY = 86400.0


def _build_chain(timestamp: datetime) -> SLNetwork:
    """Build rain -> wet_grass with all timestamps at the given time."""
    net = SLNetwork()
    net.add_node(SLNode(
        node_id="rain",
        opinion=Opinion(0.7, 0.1, 0.2),
        timestamp=timestamp,
    ))
    net.add_node(SLNode(
        node_id="wet_grass",
        opinion=Opinion(0.5, 0.3, 0.2),
        timestamp=timestamp,
    ))
    net.add_edge(SLEdge(
        source_id="rain",
        target_id="wet_grass",
        conditional=Opinion(0.9, 0.05, 0.05),
        counterfactual=Opinion(0.1, 0.7, 0.2),
        timestamp=timestamp,
    ))
    return net


class TestTemporalDiffResultStructure:
    """TemporalDiffResult has the expected fields."""

    def test_fields_present(self):
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        assert isinstance(result, TemporalDiffResult)
        assert result.node_id == "wet_grass"
        assert result.t1 == T0
        assert result.t2 == T1
        assert isinstance(result.opinion_at_t1, Opinion)
        assert isinstance(result.opinion_at_t2, Opinion)
        assert isinstance(result.delta_belief, float)
        assert isinstance(result.delta_disbelief, float)
        assert isinstance(result.delta_uncertainty, float)


class TestTemporalDiffDeltas:
    """Deltas are computed as t2 - t1 component-wise."""

    def test_uncertainty_increases_over_time(self):
        """As time passes, uncertainty grows due to decay."""
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        # At T0, no elapsed time, so opinion is fresh.
        # At T1, several months of decay have occurred.
        assert result.delta_uncertainty > 0.0
        assert result.delta_belief < 0.0

    def test_deltas_are_t2_minus_t1(self):
        """delta_x = opinion_at_t2.x - opinion_at_t1.x."""
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        assert result.delta_belief == pytest.approx(
            result.opinion_at_t2.belief - result.opinion_at_t1.belief,
        )
        assert result.delta_disbelief == pytest.approx(
            result.opinion_at_t2.disbelief - result.opinion_at_t1.disbelief,
        )
        assert result.delta_uncertainty == pytest.approx(
            result.opinion_at_t2.uncertainty - result.opinion_at_t1.uncertainty,
        )

    def test_zero_interval_zero_deltas(self):
        """If t1 == t2, all deltas are zero."""
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T1, t2=T1, default_half_life=ONE_DAY,
        )
        assert result.delta_belief == pytest.approx(0.0)
        assert result.delta_disbelief == pytest.approx(0.0)
        assert result.delta_uncertainty == pytest.approx(0.0)


class TestTemporalDiffMatchesInferAt:
    """The opinions in the diff result match calling infer_at directly."""

    def test_opinions_match_infer_at(self):
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        direct_t1 = net.infer_at(
            "wet_grass", reference_time=T0, default_half_life=ONE_DAY,
        )
        direct_t2 = net.infer_at(
            "wet_grass", reference_time=T1, default_half_life=ONE_DAY,
        )
        assert result.opinion_at_t1.belief == pytest.approx(direct_t1.opinion.belief)
        assert result.opinion_at_t1.uncertainty == pytest.approx(direct_t1.opinion.uncertainty)
        assert result.opinion_at_t2.belief == pytest.approx(direct_t2.opinion.belief)
        assert result.opinion_at_t2.uncertainty == pytest.approx(direct_t2.opinion.uncertainty)


class TestTemporalDiffBDUInvariant:
    """Both opinions in the result satisfy b+d+u=1."""

    def test_invariant(self):
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        for op in [result.opinion_at_t1, result.opinion_at_t2]:
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)


class TestTemporalDiffKwargsForwarded:
    """Additional kwargs are forwarded to infer_at."""

    def test_decay_model_forwarded(self):
        net = _build_chain(T0)
        result_nodes = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1,
            default_half_life=ONE_DAY, decay_model="nodes",
        )
        result_both = net.infer_temporal_diff(
            "wet_grass", t1=T0, t2=T1,
            default_half_life=ONE_DAY, decay_model="both",
        )
        # Different decay models produce different opinions at t2
        # (at t1 == T0, elapsed is 0 so both are identical)
        assert result_nodes.opinion_at_t2.belief != pytest.approx(
            result_both.opinion_at_t2.belief, abs=1e-6,
        )


class TestTemporalDiffRootNode:
    """infer_temporal_diff works on root nodes too."""

    def test_root_node(self):
        net = _build_chain(T0)
        result = net.infer_temporal_diff(
            "rain", t1=T0, t2=T1, default_half_life=ONE_DAY,
        )
        assert result.node_id == "rain"
        assert result.delta_uncertainty > 0.0
