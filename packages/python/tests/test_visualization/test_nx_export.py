"""
Tests for NetworkX export (optional dependency).

Covers:
    - to_networkx returns a networkx.DiGraph
    - Content nodes have correct attributes (opinion, node_type)
    - Agent nodes have correct attributes
    - Deduction edges have correct attributes (conditional, edge_type)
    - Trust edges present with correct attributes
    - Attestation edges present with correct attributes
    - Node/edge counts match the original network
    - Empty network
    - Mixed network with all types
    - Import guard: clear error when networkx not installed

TDD RED PHASE: These tests should FAIL until visualization/nx_export.py
is implemented.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    AttestationEdge,
    SLEdge,
    SLNode,
    TrustEdge,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

VAC = Opinion(0.0, 0.0, 1.0)

nx = pytest.importorskip("networkx", reason="networkx not installed")


def _build_content_only() -> SLNetwork:
    """Simple content-only: fever → infection."""
    net = SLNetwork(name="content_net")
    net.add_node(SLNode(
        node_id="fever", opinion=Opinion(0.8, 0.1, 0.1), label="Fever",
    ))
    net.add_node(SLNode(
        node_id="infection", opinion=Opinion(0.5, 0.2, 0.3),
    ))
    net.add_edge(SLEdge(
        source_id="fever", target_id="infection",
        conditional=Opinion(0.9, 0.05, 0.05),
    ))
    return net


def _build_mixed() -> SLNetwork:
    """Mixed network: agents + content + all edge types."""
    net = SLNetwork(name="mixed_nx")

    net.add_agent(SLNode(
        node_id="alice", opinion=VAC, node_type="agent", label="Alice",
    ))
    net.add_agent(SLNode(
        node_id="bob", opinion=VAC, node_type="agent",
    ))
    net.add_node(SLNode(
        node_id="claim_x", opinion=Opinion(0.7, 0.15, 0.15),
    ))
    net.add_node(SLNode(node_id="claim_y", opinion=VAC))

    net.add_trust_edge(TrustEdge(
        source_id="alice", target_id="bob",
        trust_opinion=Opinion(0.85, 0.05, 0.1),
    ))
    net.add_attestation(AttestationEdge(
        agent_id="alice", content_id="claim_x",
        opinion=Opinion(0.8, 0.1, 0.1),
    ))
    net.add_attestation(AttestationEdge(
        agent_id="bob", content_id="claim_y",
        opinion=Opinion(0.6, 0.2, 0.2),
    ))
    net.add_edge(SLEdge(
        source_id="claim_x", target_id="claim_y",
        conditional=Opinion(0.9, 0.03, 0.07),
    ))

    return net


# ═══════════════════════════════════════════════════════════════════
# BASIC STRUCTURE
# ═══════════════════════════════════════════════════════════════════


class TestNxBasicStructure:
    """to_networkx returns a valid DiGraph."""

    def test_returns_digraph(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert isinstance(G, nx.DiGraph)

    def test_graph_name(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert G.graph.get("name") == "content_net"

    def test_empty_network(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = SLNetwork(name="empty")
        G = to_networkx(net)
        assert isinstance(G, nx.DiGraph)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


# ═══════════════════════════════════════════════════════════════════
# CONTENT NODES
# ═══════════════════════════════════════════════════════════════════


class TestNxContentNodes:
    """Content nodes carry opinion and type attributes."""

    def test_node_count(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert G.number_of_nodes() == 2

    def test_node_has_opinion_components(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        data = G.nodes["fever"]
        assert abs(data["belief"] - 0.8) < 1e-9
        assert abs(data["disbelief"] - 0.1) < 1e-9
        assert abs(data["uncertainty"] - 0.1) < 1e-9

    def test_node_has_type(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert G.nodes["fever"]["node_type"] == "content"

    def test_node_has_label(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert G.nodes["fever"]["label"] == "Fever"

    def test_node_projected_probability(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        # P(ω) = b + a*u = 0.8 + 0.5*0.1 = 0.85
        assert abs(G.nodes["fever"]["projected_probability"] - 0.85) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# AGENT NODES
# ═══════════════════════════════════════════════════════════════════


class TestNxAgentNodes:
    """Agent nodes carry agent-specific attributes."""

    def test_agent_node_type(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        assert G.nodes["alice"]["node_type"] == "agent"
        assert G.nodes["bob"]["node_type"] == "agent"

    def test_agent_label(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        assert G.nodes["alice"]["label"] == "Alice"


# ═══════════════════════════════════════════════════════════════════
# DEDUCTION EDGES
# ═══════════════════════════════════════════════════════════════════


class TestNxDeductionEdges:
    """Deduction edges carry conditional opinion attributes."""

    def test_edge_exists(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        assert G.has_edge("fever", "infection")

    def test_edge_has_type(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        data = G.edges["fever", "infection"]
        assert data["edge_type"] == "deduction"

    def test_edge_has_conditional_opinion(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_content_only()
        G = to_networkx(net)
        data = G.edges["fever", "infection"]
        assert abs(data["cond_belief"] - 0.9) < 1e-9
        assert abs(data["cond_disbelief"] - 0.05) < 1e-9
        assert abs(data["cond_uncertainty"] - 0.05) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# TRUST EDGES
# ═══════════════════════════════════════════════════════════════════


class TestNxTrustEdges:
    """Trust edges carry trust opinion attributes."""

    def test_trust_edge_exists(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        assert G.has_edge("alice", "bob")

    def test_trust_edge_type(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        data = G.edges["alice", "bob"]
        assert data["edge_type"] == "trust"

    def test_trust_edge_opinion(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        data = G.edges["alice", "bob"]
        assert abs(data["trust_belief"] - 0.85) < 1e-9
        assert abs(data["trust_disbelief"] - 0.05) < 1e-9
        assert abs(data["trust_uncertainty"] - 0.1) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# ATTESTATION EDGES
# ═══════════════════════════════════════════════════════════════════


class TestNxAttestationEdges:
    """Attestation edges carry attestation opinion attributes."""

    def test_attestation_edge_exists(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        assert G.has_edge("alice", "claim_x")

    def test_attestation_edge_type(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        data = G.edges["alice", "claim_x"]
        assert data["edge_type"] == "attestation"

    def test_attestation_edge_opinion(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        data = G.edges["alice", "claim_x"]
        assert abs(data["attest_belief"] - 0.8) < 1e-9
        assert abs(data["attest_disbelief"] - 0.1) < 1e-9
        assert abs(data["attest_uncertainty"] - 0.1) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# MIXED NETWORK COUNTS
# ═══════════════════════════════════════════════════════════════════


class TestNxMixedCounts:
    """Mixed network has correct node and edge counts."""

    def test_total_node_count(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        # 2 agents + 2 content = 4
        assert G.number_of_nodes() == 4

    def test_total_edge_count(self) -> None:
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)
        # 1 deduction + 1 trust + 2 attestation = 4
        assert G.number_of_edges() == 4

    def test_can_filter_by_edge_type(self) -> None:
        """Edge type attribute enables filtering."""
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)

        trust_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "trust"
        ]
        deduction_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "deduction"
        ]
        attestation_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "attestation"
        ]

        assert len(trust_edges) == 1
        assert len(deduction_edges) == 1
        assert len(attestation_edges) == 2

    def test_can_filter_by_node_type(self) -> None:
        """Node type attribute enables filtering."""
        from jsonld_ex.visualization.nx_export import to_networkx

        net = _build_mixed()
        G = to_networkx(net)

        agents = [n for n, d in G.nodes(data=True) if d["node_type"] == "agent"]
        content = [n for n, d in G.nodes(data=True) if d["node_type"] == "content"]

        assert len(agents) == 2
        assert len(content) == 2
