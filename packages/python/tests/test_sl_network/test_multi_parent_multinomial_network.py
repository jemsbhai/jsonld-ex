"""Tests for MultiParentMultinomialEdge integration with SLNetwork.

TDD RED phase for Phase C, Step C.1.

Verifies that SLNetwork can store, retrieve, count, and clean up
MultiParentMultinomialEdge instances, following the same patterns
established for MultiParentEdge and MultinomialEdge.

References:
    SLNetworks_plan.md §1.6.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.network import SLNetwork, CycleError, NodeNotFoundError
from jsonld_ex.sl_network.types import (
    MultiParentMultinomialEdge,
    SLEdge,
    SLNode,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _make_opinion() -> Opinion:
    return Opinion(0.0, 0.0, 1.0)  # vacuous


def _make_multinomial_opinion(
    b_h: float = 0.0, b_l: float = 0.0
) -> MultinomialOpinion:
    u = 1.0 - b_h - b_l
    return MultinomialOpinion(
        beliefs={"H": b_h, "L": b_l},
        uncertainty=u,
        base_rates={"H": 0.5, "L": 0.5},
    )


def _make_network_with_3_nodes() -> SLNetwork:
    """Create network with nodes A, B, C (all vacuous)."""
    net = SLNetwork(name="test")
    for nid in ["A", "B", "C"]:
        net.add_node(SLNode(node_id=nid, opinion=_make_opinion()))
    return net


def _make_two_parent_multinomial_edge(
    parents: tuple[str, ...] = ("A", "B"),
    target: str = "C",
) -> MultiParentMultinomialEdge:
    """A,B → C with binary parent states and binary child domain {H,L}."""
    conditionals: dict[tuple[str, ...], MultinomialOpinion] = {}
    for s_a in ["s0", "s1"]:
        for s_b in ["s0", "s1"]:
            conditionals[(s_a, s_b)] = _make_multinomial_opinion(0.3, 0.3)
    return MultiParentMultinomialEdge(
        parent_ids=parents,
        target_id=target,
        conditionals=conditionals,
    )


# ═══════════════════════════════════════════════════════════════════
# ADD EDGE DISPATCH
# ═══════════════════════════════════════════════════════════════════


class TestAddMultiParentMultinomialEdge:
    """Test add_edge() dispatch for MultiParentMultinomialEdge."""

    def test_add_edge_accepts_multi_parent_multinomial(self) -> None:
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)  # Should not raise

    def test_add_edge_updates_adjacency_parents(self) -> None:
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        parents = net.get_parents("C")
        assert set(parents) == {"A", "B"}

    def test_add_edge_updates_adjacency_children(self) -> None:
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        assert "C" in net.get_children("A")
        assert "C" in net.get_children("B")

    def test_add_edge_node_not_found_target(self) -> None:
        net = SLNetwork(name="test")
        net.add_node(SLNode(node_id="A", opinion=_make_opinion()))
        net.add_node(SLNode(node_id="B", opinion=_make_opinion()))
        edge = _make_two_parent_multinomial_edge()  # target="C" missing
        with pytest.raises(NodeNotFoundError):
            net.add_edge(edge)

    def test_add_edge_node_not_found_parent(self) -> None:
        net = SLNetwork(name="test")
        net.add_node(SLNode(node_id="A", opinion=_make_opinion()))
        # B missing
        net.add_node(SLNode(node_id="C", opinion=_make_opinion()))
        edge = _make_two_parent_multinomial_edge()
        with pytest.raises(NodeNotFoundError):
            net.add_edge(edge)

    def test_add_edge_duplicate_target_raises(self) -> None:
        """Cannot add two MultiParentMultinomialEdges for the same target."""
        net = _make_network_with_3_nodes()
        edge1 = _make_two_parent_multinomial_edge()
        net.add_edge(edge1)
        edge2 = _make_two_parent_multinomial_edge()
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge(edge2)

    def test_add_edge_cycle_detection(self) -> None:
        """Adding edge that would create cycle raises CycleError."""
        net = SLNetwork(name="test")
        for nid in ["A", "B", "C"]:
            net.add_node(SLNode(node_id=nid, opinion=_make_opinion()))
        # A → B (simple edge)
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=_make_opinion(),
        ))
        # Now try B,C → A which creates cycle B→...→A→B
        cond = {("s0", "s1"): _make_multinomial_opinion(0.3, 0.3)}
        edge = MultiParentMultinomialEdge(
            parent_ids=("B", "C"), target_id="A", conditionals=cond,
        )
        with pytest.raises(CycleError):
            net.add_edge(edge)

    def test_cross_type_conflict_with_sl_edge(self) -> None:
        """Cannot add MultiParentMultinomialEdge if SLEdge exists for same parent→target pair."""
        net = _make_network_with_3_nodes()
        net.add_edge(SLEdge(
            source_id="A", target_id="C",
            conditional=_make_opinion(),
        ))
        edge = _make_two_parent_multinomial_edge()  # A,B→C
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge(edge)


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════


class TestGetMultiParentMultinomialEdge:
    """Test retrieval methods."""

    def test_get_multi_parent_multinomial_edge(self) -> None:
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        retrieved = net.get_multi_parent_multinomial_edge("C")
        assert retrieved is edge

    def test_get_multi_parent_multinomial_edge_not_found(self) -> None:
        net = _make_network_with_3_nodes()
        with pytest.raises(ValueError, match="not found"):
            net.get_multi_parent_multinomial_edge("C")

    def test_has_multi_parent_multinomial_edge(self) -> None:
        net = _make_network_with_3_nodes()
        assert not net.has_multi_parent_multinomial_edge("C")
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        assert net.has_multi_parent_multinomial_edge("C")


# ═══════════════════════════════════════════════════════════════════
# EDGE COUNT
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCountWithMultiParentMultinomial:
    """Test that edge_count() includes multi-parent multinomial edges."""

    def test_edge_count_includes_multi_parent_multinomial(self) -> None:
        """A,B→C contributes 2 to edge_count (one per parent)."""
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        assert net.edge_count() == 2  # 2 parents

    def test_edge_count_mixed(self) -> None:
        """Mix of SLEdge and MultiParentMultinomialEdge."""
        net = SLNetwork(name="test")
        for nid in ["A", "B", "C", "D"]:
            net.add_node(SLNode(node_id=nid, opinion=_make_opinion()))
        # A → B (SLEdge)
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=_make_opinion(),
        ))
        # B,C → D (MultiParentMultinomialEdge)
        cond = {
            ("s0", "s1"): _make_multinomial_opinion(0.3, 0.3),
        }
        net.add_edge(MultiParentMultinomialEdge(
            parent_ids=("B", "C"), target_id="D", conditionals=cond,
        ))
        assert net.edge_count() == 3  # 1 SLEdge + 2 multi-parent


# ═══════════════════════════════════════════════════════════════════
# REMOVE NODE CLEANUP
# ═══════════════════════════════════════════════════════════════════


class TestRemoveNodeCleansUpMultiParentMultinomial:
    """Test that remove_node() properly cleans up MultiParentMultinomialEdge."""

    def test_remove_target_node_removes_edge(self) -> None:
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        net.remove_node("C")
        assert not net.has_multi_parent_multinomial_edge("C")
        assert net.edge_count() == 0

    def test_remove_parent_node_removes_edge(self) -> None:
        """Removing a parent invalidates the entire multi-parent edge."""
        net = _make_network_with_3_nodes()
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        net.remove_node("A")
        assert not net.has_multi_parent_multinomial_edge("C")
        # B should no longer have C as child
        assert "C" not in net.get_children("B")

    def test_remove_unrelated_node_keeps_edge(self) -> None:
        """Removing an unrelated node doesn't affect the edge."""
        net = SLNetwork(name="test")
        for nid in ["A", "B", "C", "D"]:
            net.add_node(SLNode(node_id=nid, opinion=_make_opinion()))
        edge = _make_two_parent_multinomial_edge()
        net.add_edge(edge)
        net.remove_node("D")
        assert net.has_multi_parent_multinomial_edge("C")


# ═══════════════════════════════════════════════════════════════════
# NO REGRESSIONS
# ═══════════════════════════════════════════════════════════════════


class TestNoRegressions:
    """Verify existing edge types still work correctly."""

    def test_binary_multi_parent_still_works(self) -> None:
        """Adding binary MultiParentEdge is unaffected."""
        from jsonld_ex.sl_network.types import MultiParentEdge

        net = _make_network_with_3_nodes()
        mpe = MultiParentEdge(
            target_id="C",
            parent_ids=("A", "B"),
            conditionals={
                (True, True): _make_opinion(),
                (True, False): _make_opinion(),
                (False, True): _make_opinion(),
                (False, False): _make_opinion(),
            },
        )
        net.add_edge(mpe)
        assert net.edge_count() == 2

    def test_sl_edge_still_works(self) -> None:
        net = SLNetwork(name="test")
        for nid in ["A", "B"]:
            net.add_node(SLNode(node_id=nid, opinion=_make_opinion()))
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=_make_opinion(),
        ))
        assert net.edge_count() == 1
