"""
Tests for trust edge serialization (Tier 2, Step 7).

Covers:
    - to_dict / from_dict roundtrip with trust edges and attestations
    - to_jsonld / from_jsonld roundtrip with trust edges and attestations
    - Fidelity: trust opinions, attestation opinions, agent node types,
      and metadata survive the roundtrip exactly.
    - Mixed network: agents + content + trust + attestation + deduction
    - Edge cases: network with only agents/trust, no attestations

TDD RED PHASE: These tests should FAIL until serialization is updated.
"""

from __future__ import annotations

import json

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    AttestationEdge,
    SLEdge,
    SLNode,
    TrustEdge,
)


_TOL = 1e-12

VAC = Opinion(0.0, 0.0, 1.0)


def _assert_opinions_equal(a: Opinion, b: Opinion, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    assert abs(a.belief - b.belief) < _TOL, f"{prefix}belief"
    assert abs(a.disbelief - b.disbelief) < _TOL, f"{prefix}disbelief"
    assert abs(a.uncertainty - b.uncertainty) < _TOL, f"{prefix}uncertainty"
    assert abs(a.base_rate - b.base_rate) < _TOL, f"{prefix}base_rate"


def _build_mixed_network() -> SLNetwork:
    """Build a network with agents, content, trust, attestation, deduction."""
    net = SLNetwork(name="mixed_trust_net")

    # Agents
    net.add_agent(SLNode(node_id="alice", opinion=VAC, node_type="agent"))
    net.add_agent(SLNode(
        node_id="bob", opinion=VAC, node_type="agent",
        label="Bob the reviewer",
        metadata={"role": "reviewer"},
    ))

    # Content
    net.add_node(SLNode(
        node_id="claim_x", opinion=Opinion(0.7, 0.15, 0.15),
    ))
    net.add_node(SLNode(node_id="claim_y", opinion=VAC))

    # Trust edges
    net.add_trust_edge(TrustEdge(
        source_id="alice", target_id="bob",
        trust_opinion=Opinion(0.85, 0.05, 0.1),
        metadata={"context": "peer-review"},
    ))

    # Attestations
    net.add_attestation(AttestationEdge(
        agent_id="alice", content_id="claim_x",
        opinion=Opinion(0.8, 0.1, 0.1),
        metadata={"method": "NER"},
    ))
    net.add_attestation(AttestationEdge(
        agent_id="bob", content_id="claim_y",
        opinion=Opinion(0.6, 0.2, 0.2),
    ))

    # Deduction edge
    net.add_edge(SLEdge(
        source_id="claim_x", target_id="claim_y",
        conditional=Opinion(0.9, 0.03, 0.07),
    ))

    return net


# ═══════════════════════════════════════════════════════════════════
# to_dict / from_dict ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════


class TestDictRoundtrip:
    """Trust edges and attestations survive dict serialization."""

    def test_roundtrip_preserves_trust_edges(self) -> None:
        """Trust edges are present after from_dict(to_dict())."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        edges = restored.get_trust_edges_from("alice")
        assert len(edges) == 1
        assert edges[0].target_id == "bob"
        _assert_opinions_equal(
            edges[0].trust_opinion,
            Opinion(0.85, 0.05, 0.1),
            "trust_opinion",
        )

    def test_roundtrip_preserves_trust_metadata(self) -> None:
        """Trust edge metadata survives roundtrip."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        edges = restored.get_trust_edges_from("alice")
        assert edges[0].metadata == {"context": "peer-review"}

    def test_roundtrip_preserves_attestations(self) -> None:
        """Attestation edges are present after roundtrip."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        atts = restored.get_attestations_for("claim_x")
        assert len(atts) == 1
        assert atts[0].agent_id == "alice"
        _assert_opinions_equal(
            atts[0].opinion,
            Opinion(0.8, 0.1, 0.1),
            "attestation_opinion",
        )

    def test_roundtrip_preserves_attestation_metadata(self) -> None:
        """Attestation metadata survives roundtrip."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        atts = restored.get_attestations_for("claim_x")
        assert atts[0].metadata == {"method": "NER"}

    def test_roundtrip_preserves_agent_node_type(self) -> None:
        """Agent nodes retain node_type='agent' after roundtrip."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        assert restored.get_node("alice").node_type == "agent"
        assert restored.get_node("bob").node_type == "agent"
        assert restored.get_node("claim_x").node_type == "content"

    def test_roundtrip_preserves_deduction_edges(self) -> None:
        """Deduction edges coexist with trust edges after roundtrip."""
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())

        assert restored.has_edge("claim_x", "claim_y")

    def test_roundtrip_preserves_network_name(self) -> None:
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.name == "mixed_trust_net"

    def test_roundtrip_node_counts(self) -> None:
        net = _build_mixed_network()
        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.node_count() == net.node_count()
        assert len(restored.get_agents()) == 2
        assert len(restored.get_content_nodes()) == 2

    def test_dict_is_json_serializable(self) -> None:
        """to_dict output can be JSON-serialized (no custom types)."""
        net = _build_mixed_network()
        d = net.to_dict()
        serialized = json.dumps(d)  # should not raise
        assert isinstance(serialized, str)


# ═══════════════════════════════════════════════════════════════════
# to_jsonld / from_jsonld ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════


class TestJsonldRoundtrip:
    """Trust edges and attestations survive JSON-LD serialization."""

    def test_roundtrip_preserves_trust_edges(self) -> None:
        net = _build_mixed_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())

        edges = restored.get_trust_edges_from("alice")
        assert len(edges) == 1
        assert edges[0].target_id == "bob"
        _assert_opinions_equal(
            edges[0].trust_opinion,
            Opinion(0.85, 0.05, 0.1),
            "trust_opinion",
        )

    def test_roundtrip_preserves_attestations(self) -> None:
        net = _build_mixed_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())

        atts = restored.get_attestations_for("claim_x")
        assert len(atts) == 1
        assert atts[0].agent_id == "alice"
        _assert_opinions_equal(
            atts[0].opinion,
            Opinion(0.8, 0.1, 0.1),
            "attestation_opinion",
        )

    def test_roundtrip_preserves_agent_node_type(self) -> None:
        net = _build_mixed_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())

        assert restored.get_node("alice").node_type == "agent"
        assert restored.get_node("bob").node_type == "agent"

    def test_roundtrip_preserves_all_edge_types(self) -> None:
        """Deduction, trust, and attestation all survive."""
        net = _build_mixed_network()
        restored = SLNetwork.from_jsonld(net.to_jsonld())

        assert restored.has_edge("claim_x", "claim_y")
        assert len(restored.get_trust_edges_from("alice")) == 1
        assert len(restored.get_attestations_for("claim_x")) == 1
        assert len(restored.get_attestations_for("claim_y")) == 1

    def test_jsonld_is_json_serializable(self) -> None:
        net = _build_mixed_network()
        jld = net.to_jsonld()
        serialized = json.dumps(jld)
        assert isinstance(serialized, str)

    def test_jsonld_has_context(self) -> None:
        net = _build_mixed_network()
        jld = net.to_jsonld()
        assert "@context" in jld

    def test_jsonld_trust_edges_have_type(self) -> None:
        """Trust edges in JSON-LD output have @type annotation."""
        net = _build_mixed_network()
        jld = net.to_jsonld()
        trust_edges = jld.get("trustEdges", [])
        assert len(trust_edges) >= 1
        assert trust_edges[0].get("@type") == "TrustEdge"

    def test_jsonld_attestations_have_type(self) -> None:
        """Attestation edges in JSON-LD output have @type annotation."""
        net = _build_mixed_network()
        jld = net.to_jsonld()
        attestations = jld.get("attestations", [])
        assert len(attestations) >= 1
        assert attestations[0].get("@type") == "AttestationEdge"


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestSerializationEdgeCases:
    """Edge cases for trust serialization."""

    def test_agents_only_dict_roundtrip(self) -> None:
        """Network with agents and trust but no content/attestation."""
        net = SLNetwork(name="trust_only")
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="b", opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(
            source_id="a", target_id="b",
            trust_opinion=Opinion(0.7, 0.1, 0.2),
        ))

        restored = SLNetwork.from_dict(net.to_dict())
        assert len(restored.get_agents()) == 2
        assert len(restored.get_trust_edges_from("a")) == 1

    def test_agents_only_jsonld_roundtrip(self) -> None:
        """Network with agents and trust but no content/attestation (JSON-LD)."""
        net = SLNetwork(name="trust_only")
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="b", opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(
            source_id="a", target_id="b",
            trust_opinion=Opinion(0.7, 0.1, 0.2),
        ))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert len(restored.get_agents()) == 2
        assert len(restored.get_trust_edges_from("a")) == 1

    def test_empty_network_roundtrip_still_works(self) -> None:
        """Existing empty-network roundtrip not broken by trust additions."""
        net = SLNetwork(name="empty")
        restored_dict = SLNetwork.from_dict(net.to_dict())
        restored_jld = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored_dict.node_count() == 0
        assert restored_jld.node_count() == 0
