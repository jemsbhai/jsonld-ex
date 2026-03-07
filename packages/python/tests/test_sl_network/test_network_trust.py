"""
Tests for Tier 2 network API extensions (Tier 2, Step 4).

Covers the trust-aware query and convenience methods added to
SLNetwork:
    - add_agent() — convenience for adding agent-typed nodes
    - get_agents() / get_content_nodes() — filtered node lists
    - get_trust_subgraph() / get_content_subgraph()
    - propagate_trust() as a network method
    - add_trust_edge / add_attestation validation edge cases

TDD RED PHASE: These tests should FAIL until network.py is extended.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, trust_discount
from jsonld_ex.sl_network.network import (
    CycleError,
    NodeNotFoundError,
    SLNetwork,
)
from jsonld_ex.sl_network.types import (
    AttestationEdge,
    SLEdge,
    SLNode,
    TrustEdge,
    TrustPropagationResult,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

VAC = Opinion(0.0, 0.0, 1.0)
HIGH_TRUST = Opinion(0.9, 0.02, 0.08)
CONTENT_OP = Opinion(0.75, 0.1, 0.15)


def _build_mixed_network() -> SLNetwork:
    """Build a network with both agent and content nodes.

    Agents: alice, bob
    Content: claim_x, claim_y
    Trust: alice → bob
    Attestation: alice → claim_x, bob → claim_y
    Deduction: claim_x → claim_y
    """
    net = SLNetwork(name="mixed")

    # Agent nodes
    net.add_agent(SLNode(node_id="alice", opinion=VAC, node_type="agent"))
    net.add_agent(SLNode(node_id="bob", opinion=VAC, node_type="agent"))

    # Content nodes
    net.add_node(SLNode(node_id="claim_x", opinion=CONTENT_OP))
    net.add_node(SLNode(node_id="claim_y", opinion=VAC))

    # Trust edge
    net.add_trust_edge(TrustEdge(
        source_id="alice", target_id="bob", trust_opinion=HIGH_TRUST,
    ))

    # Attestations
    net.add_attestation(AttestationEdge(
        agent_id="alice", content_id="claim_x", opinion=CONTENT_OP,
    ))
    net.add_attestation(AttestationEdge(
        agent_id="bob", content_id="claim_y",
        opinion=Opinion(0.6, 0.2, 0.2),
    ))

    # Deduction edge
    net.add_edge(SLEdge(
        source_id="claim_x", target_id="claim_y",
        conditional=Opinion(0.85, 0.05, 0.1),
    ))

    return net


# ═══════════════════════════════════════════════════════════════════
# add_agent()
# ═══════════════════════════════════════════════════════════════════


class TestAddAgent:
    """add_agent() is a convenience for adding agent-typed nodes."""

    def test_add_agent_basic(self) -> None:
        """add_agent stores the node and it's retrievable."""
        net = SLNetwork()
        node = SLNode(node_id="alice", opinion=VAC, node_type="agent")
        net.add_agent(node)
        assert net.has_node("alice")
        assert net.get_node("alice").node_type == "agent"

    def test_add_agent_rejects_content_node(self) -> None:
        """add_agent rejects a node with node_type='content'."""
        net = SLNetwork()
        node = SLNode(node_id="prop_x", opinion=VAC, node_type="content")
        with pytest.raises((ValueError, TypeError)):
            net.add_agent(node)

    def test_add_agent_duplicate_raises(self) -> None:
        """Adding an agent with existing ID raises ValueError."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="alice", opinion=VAC, node_type="agent"))
        with pytest.raises(ValueError, match="already exists"):
            net.add_agent(SLNode(node_id="alice", opinion=VAC, node_type="agent"))

    def test_add_agent_non_slnode_raises(self) -> None:
        """add_agent rejects non-SLNode input."""
        net = SLNetwork()
        with pytest.raises(TypeError):
            net.add_agent("alice")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# get_agents() / get_content_nodes()
# ═══════════════════════════════════════════════════════════════════


class TestNodeFiltering:
    """get_agents() and get_content_nodes() return filtered, sorted lists."""

    def test_get_agents_returns_agent_ids(self) -> None:
        """get_agents returns only agent-typed node IDs."""
        net = _build_mixed_network()
        agents = net.get_agents()
        assert agents == ["alice", "bob"]

    def test_get_content_nodes_returns_content_ids(self) -> None:
        """get_content_nodes returns only content-typed node IDs."""
        net = _build_mixed_network()
        content = net.get_content_nodes()
        assert content == ["claim_x", "claim_y"]

    def test_get_agents_empty_network(self) -> None:
        """Empty network returns empty agents list."""
        net = SLNetwork()
        assert net.get_agents() == []

    def test_get_content_nodes_agent_only_network(self) -> None:
        """Agent-only network returns empty content list."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        assert net.get_content_nodes() == []

    def test_sorted_output(self) -> None:
        """Results are sorted alphabetically."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="zara", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="alice", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="z_prop", opinion=VAC))
        net.add_node(SLNode(node_id="a_prop", opinion=VAC))
        assert net.get_agents() == ["alice", "zara"]
        assert net.get_content_nodes() == ["a_prop", "z_prop"]


# ═══════════════════════════════════════════════════════════════════
# get_trust_subgraph() / get_content_subgraph()
# ═══════════════════════════════════════════════════════════════════


class TestSubgraphExtraction:
    """Subgraph extraction for trust and content sub-networks."""

    def test_trust_subgraph_contains_only_agents(self) -> None:
        """get_trust_subgraph has only agent nodes."""
        net = _build_mixed_network()
        trust_sub = net.get_trust_subgraph()
        assert set(trust_sub.get_agents()) == {"alice", "bob"}
        assert trust_sub.get_content_nodes() == []

    def test_trust_subgraph_contains_trust_edges(self) -> None:
        """get_trust_subgraph preserves trust edges."""
        net = _build_mixed_network()
        trust_sub = net.get_trust_subgraph()
        edges = trust_sub.get_trust_edges_from("alice")
        assert len(edges) == 1
        assert edges[0].target_id == "bob"

    def test_trust_subgraph_no_deduction_edges(self) -> None:
        """get_trust_subgraph has no deduction edges."""
        net = _build_mixed_network()
        trust_sub = net.get_trust_subgraph()
        assert trust_sub.edge_count() == 0  # deduction edge count

    def test_content_subgraph_contains_only_content(self) -> None:
        """get_content_subgraph has only content nodes."""
        net = _build_mixed_network()
        content_sub = net.get_content_subgraph()
        assert content_sub.get_agents() == []
        assert set(content_sub.get_content_nodes()) == {"claim_x", "claim_y"}

    def test_content_subgraph_preserves_deduction_edges(self) -> None:
        """get_content_subgraph preserves deduction edges."""
        net = _build_mixed_network()
        content_sub = net.get_content_subgraph()
        assert content_sub.has_edge("claim_x", "claim_y")

    def test_content_subgraph_no_trust_edges(self) -> None:
        """get_content_subgraph has no trust edges."""
        net = _build_mixed_network()
        content_sub = net.get_content_subgraph()
        # No agents, so no trust edges possible
        assert len(content_sub._trust_edges) == 0

    def test_empty_network_subgraphs(self) -> None:
        """Subgraphs of empty network are empty."""
        net = SLNetwork()
        assert net.get_trust_subgraph().node_count() == 0
        assert net.get_content_subgraph().node_count() == 0


# ═══════════════════════════════════════════════════════════════════
# propagate_trust() as network method
# ═══════════════════════════════════════════════════════════════════


class TestNetworkPropagateTrust:
    """SLNetwork.propagate_trust() delegates to trust.propagate_trust()."""

    def test_delegates_correctly(self) -> None:
        """Network method returns same result as module function."""
        from jsonld_ex.sl_network.trust import (
            propagate_trust as module_propagate,
        )

        net = SLNetwork(name="delegation_test")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=HIGH_TRUST,
        ))

        net_result = net.propagate_trust("Q")
        mod_result = module_propagate(net, "Q")

        assert net_result.querying_agent == mod_result.querying_agent
        assert set(net_result.derived_trusts.keys()) == set(mod_result.derived_trusts.keys())
        for key in net_result.derived_trusts:
            nr = net_result.derived_trusts[key]
            mr = mod_result.derived_trusts[key]
            assert abs(nr.belief - mr.belief) < 1e-9
            assert abs(nr.disbelief - mr.disbelief) < 1e-9
            assert abs(nr.uncertainty - mr.uncertainty) < 1e-9

    def test_returns_trust_propagation_result(self) -> None:
        """Network method returns TrustPropagationResult."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        result = net.propagate_trust("Q")
        assert isinstance(result, TrustPropagationResult)

    def test_nonexistent_agent_raises(self) -> None:
        """Network method raises for nonexistent agent."""
        net = SLNetwork()
        with pytest.raises(Exception):
            net.propagate_trust("ghost")


# ═══════════════════════════════════════════════════════════════════
# add_trust_edge / add_attestation validation
# ═══════════════════════════════════════════════════════════════════


class TestTrustEdgeValidation:
    """Edge cases for add_trust_edge and add_attestation."""

    def test_trust_edge_missing_source_raises(self) -> None:
        """Trust edge referencing nonexistent source raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="b", opinion=VAC, node_type="agent"))
        with pytest.raises(NodeNotFoundError):
            net.add_trust_edge(TrustEdge(
                source_id="ghost", target_id="b", trust_opinion=HIGH_TRUST,
            ))

    def test_trust_edge_missing_target_raises(self) -> None:
        """Trust edge referencing nonexistent target raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        with pytest.raises(NodeNotFoundError):
            net.add_trust_edge(TrustEdge(
                source_id="a", target_id="ghost", trust_opinion=HIGH_TRUST,
            ))

    def test_duplicate_trust_edge_raises(self) -> None:
        """Adding same trust edge twice raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="b", opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(
            source_id="a", target_id="b", trust_opinion=HIGH_TRUST,
        ))
        with pytest.raises(ValueError, match="already exists"):
            net.add_trust_edge(TrustEdge(
                source_id="a", target_id="b", trust_opinion=HIGH_TRUST,
            ))

    def test_attestation_missing_agent_raises(self) -> None:
        """Attestation referencing nonexistent agent raises."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="c", opinion=VAC))
        with pytest.raises(NodeNotFoundError):
            net.add_attestation(AttestationEdge(
                agent_id="ghost", content_id="c", opinion=CONTENT_OP,
            ))

    def test_attestation_missing_content_raises(self) -> None:
        """Attestation referencing nonexistent content raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        with pytest.raises(NodeNotFoundError):
            net.add_attestation(AttestationEdge(
                agent_id="a", content_id="ghost", opinion=CONTENT_OP,
            ))

    def test_duplicate_attestation_raises(self) -> None:
        """Adding same attestation twice raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="a", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="c", opinion=VAC))
        net.add_attestation(AttestationEdge(
            agent_id="a", content_id="c", opinion=CONTENT_OP,
        ))
        with pytest.raises(ValueError, match="already exists"):
            net.add_attestation(AttestationEdge(
                agent_id="a", content_id="c", opinion=CONTENT_OP,
            ))
