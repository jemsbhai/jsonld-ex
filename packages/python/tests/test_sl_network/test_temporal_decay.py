"""Tests for temporal decay on SLNetwork (Tier 3, Steps 2-3).

Step 2: Model A -- node-level decay (decay_network_nodes).
Step 3: Model B -- edge-level decay (decay_network_edges).

Both functions return a NEW network with decayed opinions,
leaving the original unchanged (immutability).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import math

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay, linear_decay
from jsonld_ex.sl_network.types import SLNode, SLEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.temporal import (
    decay_network_nodes,
    decay_network_edges,
)


# -- Fixtures --

VACUOUS = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_HOUR = 3600.0
ONE_DAY = 86400.0


def _approx_opinion(op: Opinion, rel: float = 1e-9) -> None:
    """Assert b+d+u=1 for a decayed opinion."""
    assert op.belief + op.disbelief + op.uncertainty == pytest.approx(
        1.0, rel=rel
    )


def _build_simple_network(
    node_timestamps: dict[str, datetime | None] | None = None,
    node_half_lives: dict[str, float | None] | None = None,
) -> SLNetwork:
    """Build a two-node chain: rain -> wet_grass."""
    node_timestamps = node_timestamps or {}
    node_half_lives = node_half_lives or {}

    net = SLNetwork()
    net.add_node(SLNode(
        node_id="rain",
        opinion=Opinion(0.7, 0.1, 0.2),
        timestamp=node_timestamps.get("rain"),
        half_life=node_half_lives.get("rain"),
    ))
    net.add_node(SLNode(
        node_id="wet_grass",
        opinion=Opinion(0.5, 0.3, 0.2),
        timestamp=node_timestamps.get("wet_grass"),
        half_life=node_half_lives.get("wet_grass"),
    ))
    net.add_edge(SLEdge(
        source_id="rain",
        target_id="wet_grass",
        conditional=Opinion(0.9, 0.05, 0.05),
        counterfactual=Opinion(0.1, 0.7, 0.2),
    ))
    return net


# ===============================================================
# Model A: decay_network_nodes
# ===============================================================


class TestDecayNetworkNodesBasic:
    """Basic behavior of decay_network_nodes."""

    def test_zero_elapsed_is_identity(self):
        """If reference_time == node timestamp, no decay occurs."""
        net = _build_simple_network(
            node_timestamps={"rain": NOW, "wet_grass": NOW},
        )
        decayed = decay_network_nodes(net, reference_time=NOW, default_half_life=ONE_DAY)
        rain = decayed.get_node("rain")
        assert rain.opinion.belief == pytest.approx(0.7)
        assert rain.opinion.disbelief == pytest.approx(0.1)
        assert rain.opinion.uncertainty == pytest.approx(0.2)

    def test_positive_elapsed_increases_uncertainty(self):
        """After some time, belief and disbelief decrease, uncertainty increases."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": one_day_ago},
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        rain = decayed.get_node("rain")
        # After one half-life, decay factor ~0.5
        assert rain.opinion.uncertainty > 0.2
        assert rain.opinion.belief < 0.7
        _approx_opinion(rain.opinion)

    def test_original_network_unchanged(self):
        """decay_network_nodes returns a new network; original is unmodified."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": one_day_ago},
        )
        original_rain = net.get_node("rain").opinion
        _ = decay_network_nodes(net, reference_time=NOW, default_half_life=ONE_DAY)
        assert net.get_node("rain").opinion.belief == original_rain.belief

    def test_matches_manual_decay_opinion(self):
        """Decayed node opinion must exactly match calling decay_opinion directly."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": NOW},
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        # rain was observed 1 day ago with default half-life of 1 day
        expected = decay_opinion(
            Opinion(0.7, 0.1, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
        )
        rain = decayed.get_node("rain")
        assert rain.opinion.belief == pytest.approx(expected.belief)
        assert rain.opinion.disbelief == pytest.approx(expected.disbelief)
        assert rain.opinion.uncertainty == pytest.approx(expected.uncertainty)

    def test_node_without_timestamp_uses_no_decay(self):
        """Nodes with no timestamp are left unchanged (no age to compute)."""
        net = _build_simple_network()  # no timestamps on either node
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        rain = decayed.get_node("rain")
        assert rain.opinion.belief == pytest.approx(0.7)
        assert rain.opinion.uncertainty == pytest.approx(0.2)

    def test_edges_preserved(self):
        """Edges are carried over to the new network unchanged."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": one_day_ago},
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("rain", "wet_grass")
        # Edge conditional should be unchanged (Model A only decays nodes)
        assert edge.conditional.belief == pytest.approx(0.9)


class TestDecayNetworkNodesHalfLife:
    """Per-node half_life overrides the default."""

    def test_per_node_half_life_used(self):
        """Node's own half_life takes precedence over default_half_life."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": one_day_ago},
            node_half_lives={"rain": ONE_DAY * 7},  # 7-day half-life
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        rain = decayed.get_node("rain")
        # rain has a 7-day half-life, so after 1 day the decay is much less
        expected = decay_opinion(
            Opinion(0.7, 0.1, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY * 7,
        )
        assert rain.opinion.belief == pytest.approx(expected.belief)

        # wet_grass uses the default 1-day half-life
        wg = decayed.get_node("wet_grass")
        expected_wg = decay_opinion(
            Opinion(0.5, 0.3, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
        )
        assert wg.opinion.belief == pytest.approx(expected_wg.belief)


class TestDecayNetworkNodesDecayFn:
    """Custom decay functions are forwarded to decay_opinion."""

    def test_linear_decay_fn(self):
        """Using linear_decay instead of exponential_decay."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": one_day_ago, "wet_grass": one_day_ago},
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
            decay_fn=linear_decay,
        )
        rain = decayed.get_node("rain")
        expected = decay_opinion(
            Opinion(0.7, 0.1, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
            decay_fn=linear_decay,
        )
        assert rain.opinion.belief == pytest.approx(expected.belief)
        assert rain.opinion.uncertainty == pytest.approx(expected.uncertainty)


class TestDecayNetworkNodesBDUInvariant:
    """b+d+u=1 invariant holds at every node after decay."""

    def test_invariant_all_nodes(self):
        """Every decayed node satisfies b+d+u=1."""
        timestamps = {
            "rain": NOW - timedelta(hours=6),
            "wet_grass": NOW - timedelta(days=3),
        }
        net = _build_simple_network(node_timestamps=timestamps)
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        for node_id in ["rain", "wet_grass"]:
            op = decayed.get_node(node_id).opinion
            _approx_opinion(op)


class TestDecayNetworkNodesFutureTimestamp:
    """Nodes with timestamps in the future relative to reference_time."""

    def test_future_timestamp_no_decay(self):
        """If a node's timestamp is after reference_time, elapsed is 0 (no decay)."""
        future = NOW + timedelta(days=1)
        net = _build_simple_network(
            node_timestamps={"rain": future, "wet_grass": NOW},
        )
        decayed = decay_network_nodes(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        rain = decayed.get_node("rain")
        assert rain.opinion.belief == pytest.approx(0.7)
        assert rain.opinion.uncertainty == pytest.approx(0.2)


# ===============================================================
# Model B: decay_network_edges
# ===============================================================


class TestDecayNetworkEdgesBasic:
    """Basic behavior of decay_network_edges."""

    def test_zero_elapsed_is_identity(self):
        """If reference_time == edge timestamp, no decay on conditionals."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=Opinion(0.1, 0.7, 0.2),
            timestamp=NOW,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.conditional.belief == pytest.approx(0.9)
        assert edge.counterfactual.belief == pytest.approx(0.1)

    def test_positive_elapsed_decays_conditional(self):
        """After time, edge conditional opinions decay toward vacuous."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=Opinion(0.1, 0.7, 0.2),
            timestamp=one_day_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.conditional.uncertainty > 0.05
        assert edge.conditional.belief < 0.9
        _approx_opinion(edge.conditional)

    def test_counterfactual_also_decayed(self):
        """Both conditional and counterfactual opinions decay."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=Opinion(0.1, 0.7, 0.2),
            timestamp=one_day_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        # Counterfactual also decayed
        expected_cf = decay_opinion(
            Opinion(0.1, 0.7, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
        )
        assert edge.counterfactual.belief == pytest.approx(expected_cf.belief)
        assert edge.counterfactual.uncertainty == pytest.approx(expected_cf.uncertainty)

    def test_none_counterfactual_stays_none(self):
        """If counterfactual is None, it remains None after decay."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=None,
            timestamp=one_day_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.counterfactual is None
        assert edge.conditional.belief < 0.9  # conditional still decayed

    def test_original_network_unchanged(self):
        """Edge decay returns a new network; original is unmodified."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            timestamp=one_day_ago,
        ))
        _ = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = net.get_edge("a", "b")
        assert edge.conditional.belief == pytest.approx(0.9)

    def test_matches_manual_decay_opinion(self):
        """Decayed edge conditional matches calling decay_opinion directly."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            timestamp=one_day_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        expected = decay_opinion(
            Opinion(0.9, 0.05, 0.05),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.conditional.belief == pytest.approx(expected.belief)
        assert edge.conditional.uncertainty == pytest.approx(expected.uncertainty)

    def test_edge_without_timestamp_no_decay(self):
        """Edges with no timestamp are left unchanged."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            # no timestamp
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.conditional.belief == pytest.approx(0.9)

    def test_nodes_preserved(self):
        """Node opinions are not changed by edge decay (Model B only)."""
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            timestamp=one_day_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        assert decayed.get_node("a").opinion.belief == pytest.approx(0.7)
        assert decayed.get_node("b").opinion.belief == pytest.approx(0.5)


class TestDecayNetworkEdgesHalfLife:
    """Per-edge half_life overrides the default."""

    def test_per_edge_half_life_used(self):
        one_day_ago = NOW - timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            timestamp=one_day_ago,
            half_life=ONE_DAY * 30,  # 30-day half-life (slow decay)
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        expected = decay_opinion(
            Opinion(0.9, 0.05, 0.05),
            elapsed=ONE_DAY,
            half_life=ONE_DAY * 30,
        )
        assert edge.conditional.belief == pytest.approx(expected.belief)


class TestDecayNetworkEdgesFutureTimestamp:
    """Edges with future timestamps get no decay."""

    def test_future_edge_no_decay(self):
        future = NOW + timedelta(days=1)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            timestamp=future,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        edge = decayed.get_edge("a", "b")
        assert edge.conditional.belief == pytest.approx(0.9)


class TestDecayNetworkEdgesBDUInvariant:
    """b+d+u=1 invariant on all decayed edge opinions."""

    def test_invariant_all_edges(self):
        two_days_ago = NOW - timedelta(days=2)
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode(node_id="c", opinion=Opinion(0.3, 0.3, 0.4)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=Opinion(0.1, 0.7, 0.2),
            timestamp=two_days_ago,
        ))
        net.add_edge(SLEdge(
            source_id="b", target_id="c",
            conditional=Opinion(0.6, 0.2, 0.2),
            counterfactual=Opinion(0.2, 0.5, 0.3),
            timestamp=two_days_ago,
        ))
        decayed = decay_network_edges(
            net, reference_time=NOW, default_half_life=ONE_DAY,
        )
        for src, tgt in [("a", "b"), ("b", "c")]:
            edge = decayed.get_edge(src, tgt)
            _approx_opinion(edge.conditional)
            if edge.counterfactual is not None:
                _approx_opinion(edge.counterfactual)
