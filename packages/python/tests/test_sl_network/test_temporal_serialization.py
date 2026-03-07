"""Tests for serialization of temporal fields (Tier 3, Step 9).

Verifies that temporal fields on SLNode, SLEdge, TrustEdge, and
AttestationEdge survive round-trip through to_dict/from_dict and
to_jsonld/from_jsonld.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    SLNode, SLEdge, TrustEdge, AttestationEdge,
)
from jsonld_ex.sl_network.network import SLNetwork


# -- Fixtures --

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
YESTERDAY = NOW - timedelta(days=1)
TOMORROW = NOW + timedelta(days=1)
ONE_DAY = 86400.0
ONE_WEEK = ONE_DAY * 7

OP_A = Opinion(0.7, 0.1, 0.2)
OP_B = Opinion(0.5, 0.3, 0.2)
COND = Opinion(0.9, 0.05, 0.05)
CF = Opinion(0.1, 0.7, 0.2)
TRUST_OP = Opinion(0.85, 0.05, 0.1)
ATT_OP = Opinion(0.75, 0.1, 0.15)
VACUOUS = Opinion(0.0, 0.0, 1.0, base_rate=0.5)


def _build_temporal_network() -> SLNetwork:
    """Build a network with temporal fields on all edge/node types."""
    net = SLNetwork(name="temporal_test")

    # Content nodes with temporal fields
    net.add_node(SLNode(
        node_id="rain", opinion=OP_A,
        timestamp=YESTERDAY, half_life=ONE_DAY,
    ))
    net.add_node(SLNode(
        node_id="wet_grass", opinion=OP_B,
        timestamp=NOW, half_life=ONE_WEEK,
    ))

    # Deduction edge with full temporal fields
    net.add_edge(SLEdge(
        source_id="rain", target_id="wet_grass",
        conditional=COND, counterfactual=CF,
        timestamp=YESTERDAY, half_life=ONE_DAY * 30,
        valid_from=YESTERDAY, valid_until=TOMORROW,
    ))

    # Agent nodes
    net.add_node(SLNode(
        node_id="alice", opinion=VACUOUS, node_type="agent",
    ))
    net.add_node(SLNode(
        node_id="bob", opinion=VACUOUS, node_type="agent",
        timestamp=NOW, half_life=ONE_WEEK,
    ))

    # Trust edge with temporal fields
    net.add_trust_edge(TrustEdge(
        source_id="alice", target_id="bob",
        trust_opinion=TRUST_OP,
        timestamp=YESTERDAY, half_life=ONE_DAY * 365,
    ))

    # Attestation edge with temporal fields
    net.add_attestation(AttestationEdge(
        agent_id="bob", content_id="rain",
        opinion=ATT_OP,
        timestamp=NOW, half_life=ONE_DAY * 7,
    ))

    return net


# ===============================================================
# to_dict / from_dict round-trip
# ===============================================================


class TestDictRoundTripTemporalNodes:
    """Node temporal fields survive dict round-trip."""

    def test_node_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.get_node("rain").timestamp == YESTERDAY

    def test_node_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.get_node("rain").half_life == ONE_DAY

    def test_node_none_temporal_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        # alice has no temporal fields
        assert restored.get_node("alice").timestamp is None
        assert restored.get_node("alice").half_life is None


class TestDictRoundTripTemporalEdges:
    """Edge temporal fields survive dict round-trip."""

    def test_edge_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.timestamp == YESTERDAY

    def test_edge_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.half_life == ONE_DAY * 30

    def test_edge_validity_window_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until == TOMORROW


class TestDictRoundTripTemporalTrust:
    """Trust edge temporal fields survive dict round-trip."""

    def test_trust_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        edges = restored.get_trust_edges_from("alice")
        assert len(edges) == 1
        assert edges[0].timestamp == YESTERDAY

    def test_trust_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        edges = restored.get_trust_edges_from("alice")
        assert edges[0].half_life == ONE_DAY * 365


class TestDictRoundTripTemporalAttestation:
    """Attestation edge temporal fields survive dict round-trip."""

    def test_attestation_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        atts = restored.get_attestations_for("rain")
        assert len(atts) == 1
        assert atts[0].timestamp == NOW

    def test_attestation_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_dict(net.to_dict())
        atts = restored.get_attestations_for("rain")
        assert atts[0].half_life == ONE_DAY * 7


# ===============================================================
# to_jsonld / from_jsonld round-trip
# ===============================================================


class TestJsonLdRoundTripTemporalNodes:
    """Node temporal fields survive JSON-LD round-trip."""

    def test_node_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.get_node("rain").timestamp == YESTERDAY

    def test_node_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.get_node("rain").half_life == ONE_DAY

    def test_node_none_temporal_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.get_node("alice").timestamp is None
        assert restored.get_node("alice").half_life is None


class TestJsonLdRoundTripTemporalEdges:
    """Edge temporal fields survive JSON-LD round-trip."""

    def test_edge_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.timestamp == YESTERDAY

    def test_edge_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.half_life == ONE_DAY * 30

    def test_edge_validity_window_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edge = restored.get_edge("rain", "wet_grass")
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until == TOMORROW


class TestJsonLdRoundTripTemporalTrust:
    """Trust edge temporal fields survive JSON-LD round-trip."""

    def test_trust_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edges = restored.get_trust_edges_from("alice")
        assert len(edges) == 1
        assert edges[0].timestamp == YESTERDAY

    def test_trust_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edges = restored.get_trust_edges_from("alice")
        assert edges[0].half_life == ONE_DAY * 365


class TestJsonLdRoundTripTemporalAttestation:
    """Attestation edge temporal fields survive JSON-LD round-trip."""

    def test_attestation_timestamp_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        atts = restored.get_attestations_for("rain")
        assert len(atts) == 1
        assert atts[0].timestamp == NOW

    def test_attestation_half_life_preserved(self):
        net = _build_temporal_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        atts = restored.get_attestations_for("rain")
        assert atts[0].half_life == ONE_DAY * 7
