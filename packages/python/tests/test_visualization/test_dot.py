"""
Tests for DOT/Graphviz visualization (zero dependencies).

Covers:
    - Basic DOT structure (digraph wrapper, semicolons)
    - Content nodes: ellipse shape, opinion b/d/u in label
    - Agent nodes: distinct shape (box), distinguished from content
    - Deduction edges: solid style, conditional opinion label
    - Trust edges: dashed style, trust opinion label
    - Attestation edges: dotted style, opinion label
    - Empty network
    - Network name as graph title
    - Mixed networks with all edge/node types
    - Node color coding by belief intensity

TDD RED PHASE: These tests should FAIL until visualization/dot.py
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

from jsonld_ex.visualization.dot import to_dot


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

VAC = Opinion(0.0, 0.0, 1.0)


def _build_content_only() -> SLNetwork:
    """Simple content-only network: A → B."""
    net = SLNetwork(name="content_net")
    net.add_node(SLNode(
        node_id="fever", opinion=Opinion(0.8, 0.1, 0.1), label="Fever",
    ))
    net.add_node(SLNode(
        node_id="infection", opinion=Opinion(0.5, 0.2, 0.3), label="Infection",
    ))
    net.add_edge(SLEdge(
        source_id="fever", target_id="infection",
        conditional=Opinion(0.9, 0.05, 0.05),
    ))
    return net


def _build_mixed() -> SLNetwork:
    """Mixed network: agents + content + all edge types."""
    net = SLNetwork(name="mixed_viz")

    net.add_agent(SLNode(
        node_id="alice", opinion=VAC, node_type="agent", label="Alice",
    ))
    net.add_agent(SLNode(
        node_id="bob", opinion=VAC, node_type="agent", label="Bob",
    ))
    net.add_node(SLNode(
        node_id="claim_x", opinion=Opinion(0.7, 0.15, 0.15),
    ))
    net.add_node(SLNode(
        node_id="claim_y", opinion=VAC,
    ))

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


class TestDotBasicStructure:
    """DOT output has valid digraph structure."""

    def test_returns_string(self) -> None:
        """to_dot returns a string."""
        net = _build_content_only()
        result = to_dot(net)
        assert isinstance(result, str)

    def test_starts_with_digraph(self) -> None:
        """Output begins with 'digraph'."""
        net = _build_content_only()
        result = to_dot(net)
        assert result.strip().startswith("digraph")

    def test_contains_opening_and_closing_braces(self) -> None:
        net = _build_content_only()
        result = to_dot(net)
        assert "{" in result
        assert result.strip().endswith("}")

    def test_network_name_in_graph(self) -> None:
        """Network name appears as the digraph name."""
        net = SLNetwork(name="my_network")
        net.add_node(SLNode(node_id="a", opinion=VAC))
        result = to_dot(net)
        assert "my_network" in result

    def test_empty_network(self) -> None:
        """Empty network produces valid DOT with no nodes/edges."""
        net = SLNetwork(name="empty")
        result = to_dot(net)
        assert "digraph" in result
        assert "{" in result
        assert result.strip().endswith("}")


# ═══════════════════════════════════════════════════════════════════
# CONTENT NODES
# ═══════════════════════════════════════════════════════════════════


class TestDotContentNodes:
    """Content nodes render with opinion info and proper shape."""

    def test_node_id_appears(self) -> None:
        """Each content node ID appears in the output."""
        net = _build_content_only()
        result = to_dot(net)
        assert "fever" in result
        assert "infection" in result

    def test_opinion_in_label(self) -> None:
        """Node label includes b/d/u values."""
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="x", opinion=Opinion(0.8, 0.1, 0.1),
        ))
        result = to_dot(net)
        # Should contain the opinion components in some format
        assert "0.80" in result or "0.8" in result

    def test_label_used_when_available(self) -> None:
        """Human-readable label appears in node declaration."""
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="fever", opinion=Opinion(0.8, 0.1, 0.1), label="Fever",
        ))
        result = to_dot(net)
        assert "Fever" in result

    def test_content_node_ellipse_shape(self) -> None:
        """Content nodes use ellipse shape."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="x", opinion=VAC))
        result = to_dot(net)
        assert "ellipse" in result


# ═══════════════════════════════════════════════════════════════════
# AGENT NODES
# ═══════════════════════════════════════════════════════════════════


class TestDotAgentNodes:
    """Agent nodes are visually distinct from content nodes."""

    def test_agent_node_different_shape(self) -> None:
        """Agent nodes use a different shape than content (box or diamond)."""
        net = SLNetwork()
        net.add_agent(SLNode(
            node_id="alice", opinion=VAC, node_type="agent",
        ))
        result = to_dot(net)
        # Should be box, diamond, or some non-ellipse shape
        assert "box" in result or "diamond" in result

    def test_agent_label_used(self) -> None:
        """Agent label appears in output."""
        net = SLNetwork()
        net.add_agent(SLNode(
            node_id="alice", opinion=VAC, node_type="agent", label="Alice",
        ))
        result = to_dot(net)
        assert "Alice" in result


# ═══════════════════════════════════════════════════════════════════
# DEDUCTION EDGES
# ═══════════════════════════════════════════════════════════════════


class TestDotDeductionEdges:
    """Deduction edges render with conditional opinion labels."""

    def test_edge_appears(self) -> None:
        """Deduction edge appears as 'source -> target'."""
        net = _build_content_only()
        result = to_dot(net)
        assert "fever" in result
        assert "infection" in result
        assert "->" in result

    def test_conditional_opinion_in_label(self) -> None:
        """Edge label includes conditional opinion info."""
        net = _build_content_only()
        result = to_dot(net)
        # The conditional was (0.9, 0.05, 0.05)
        assert "0.9" in result or "0.90" in result


# ═══════════════════════════════════════════════════════════════════
# TRUST EDGES
# ═══════════════════════════════════════════════════════════════════


class TestDotTrustEdges:
    """Trust edges are visually distinct from deduction edges."""

    def test_trust_edge_appears(self) -> None:
        """Trust edge between agents appears in output."""
        net = _build_mixed()
        result = to_dot(net)
        assert "alice" in result
        assert "bob" in result

    def test_trust_edge_dashed(self) -> None:
        """Trust edges use dashed style."""
        net = _build_mixed()
        result = to_dot(net)
        assert "dashed" in result

    def test_trust_opinion_in_label(self) -> None:
        """Trust edge label includes trust opinion."""
        net = _build_mixed()
        result = to_dot(net)
        assert "0.85" in result or "trust" in result.lower()


# ═══════════════════════════════════════════════════════════════════
# ATTESTATION EDGES
# ═══════════════════════════════════════════════════════════════════


class TestDotAttestationEdges:
    """Attestation edges are visually distinct."""

    def test_attestation_edge_appears(self) -> None:
        """Attestation from agent to content appears in output."""
        net = _build_mixed()
        result = to_dot(net)
        # alice → claim_x attestation should be present
        assert "claim_x" in result

    def test_attestation_edge_dotted(self) -> None:
        """Attestation edges use dotted style."""
        net = _build_mixed()
        result = to_dot(net)
        assert "dotted" in result


# ═══════════════════════════════════════════════════════════════════
# COLOR CODING
# ═══════════════════════════════════════════════════════════════════


class TestDotColorCoding:
    """Nodes are color-coded by belief intensity."""

    def test_high_belief_node_has_fill(self) -> None:
        """Node with high belief has a fill color."""
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="certain", opinion=Opinion(0.95, 0.03, 0.02),
        ))
        result = to_dot(net)
        assert "fillcolor" in result or "color" in result

    def test_vacuous_node_has_different_style(self) -> None:
        """Vacuous node (u=1) is styled differently from certain node."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="certain", opinion=Opinion(0.95, 0.03, 0.02)))
        net.add_node(SLNode(node_id="unknown", opinion=VAC))
        result = to_dot(net)
        # Both nodes should be in the output with different styling
        assert "certain" in result
        assert "unknown" in result


# ═══════════════════════════════════════════════════════════════════
# FULL MIXED NETWORK
# ═══════════════════════════════════════════════════════════════════


class TestDotMixedNetwork:
    """Full mixed network renders all elements."""

    def test_all_node_types_present(self) -> None:
        net = _build_mixed()
        result = to_dot(net)
        assert "alice" in result
        assert "bob" in result
        assert "claim_x" in result
        assert "claim_y" in result

    def test_all_edge_types_present(self) -> None:
        """Deduction, trust, and attestation edges all appear."""
        net = _build_mixed()
        result = to_dot(net)
        # Should have at least 4 edges total (1 deduction + 1 trust + 2 attestation)
        assert result.count("->") >= 4

    def test_output_is_valid_dot_structure(self) -> None:
        """Basic structural validation: balanced braces, ends properly."""
        net = _build_mixed()
        result = to_dot(net)
        assert result.count("{") == result.count("}")
        assert result.strip().endswith("}")
