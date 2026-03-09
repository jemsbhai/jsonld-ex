"""
Tests for SLNetwork multinomial edge support (Phase B, Step 3a).

Extends SLNetwork to store, retrieve, and manage MultinomialEdge
instances alongside existing SLEdge and MultiParentEdge types.

Design contract (consistent with existing edge patterns):
    - ``add_edge()`` accepts ``MultinomialEdge`` in addition to
      ``SLEdge`` and ``MultiParentEdge``.
    - MultinomialEdges contribute to adjacency structure (parents/children).
    - Cycle detection applies to MultinomialEdges.
    - Node existence is validated on add.
    - Duplicate edges are rejected.
    - ``remove_node()`` cleans up MultinomialEdges.
    - ``get_multinomial_edge()`` and ``has_multinomial_edge()`` for retrieval.
    - ``get_parents()`` and ``get_children()`` reflect MultinomialEdges.

TDD: RED phase — all tests should FAIL until network.py is updated.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.network import CycleError, NodeNotFoundError, SLNetwork
from jsonld_ex.sl_network.types import MultinomialEdge, SLEdge, SLNode


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


def _vacuous_binomial() -> Opinion:
    """Vacuous binomial opinion for placeholder nodes."""
    return Opinion(0.0, 0.0, 1.0)


def _make_ternary_conditionals() -> dict[str, MultinomialOpinion]:
    """Conditional table for ternary parent {A, B, C} → binary child {H, L}."""
    br = {"H": 0.5, "L": 0.5}
    return {
        "A": MultinomialOpinion(
            beliefs={"H": 0.7, "L": 0.1}, uncertainty=0.2, base_rates=br,
        ),
        "B": MultinomialOpinion(
            beliefs={"H": 0.3, "L": 0.4}, uncertainty=0.3, base_rates=br,
        ),
        "C": MultinomialOpinion(
            beliefs={"H": 0.1, "L": 0.6}, uncertainty=0.3, base_rates=br,
        ),
    }


def _make_binary_conditionals() -> dict[str, MultinomialOpinion]:
    """Conditional table for binary parent {T, F} → binary child {H, L}."""
    br = {"H": 0.5, "L": 0.5}
    return {
        "T": MultinomialOpinion(
            beliefs={"H": 0.8, "L": 0.1}, uncertainty=0.1, base_rates=br,
        ),
        "F": MultinomialOpinion(
            beliefs={"H": 0.1, "L": 0.7}, uncertainty=0.2, base_rates=br,
        ),
    }


@pytest.fixture
def net() -> SLNetwork:
    """Network with three nodes: X, Y, Z (no edges yet)."""
    network = SLNetwork(name="multinomial_test")
    network.add_node(SLNode(node_id="X", opinion=_vacuous_binomial()))
    network.add_node(SLNode(node_id="Y", opinion=_vacuous_binomial()))
    network.add_node(SLNode(node_id="Z", opinion=_vacuous_binomial()))
    return network


@pytest.fixture
def ternary_edge() -> MultinomialEdge:
    """A ternary parent → binary child multinomial edge."""
    return MultinomialEdge(
        source_id="X",
        target_id="Y",
        conditionals=_make_ternary_conditionals(),
    )


@pytest.fixture
def binary_edge() -> MultinomialEdge:
    """A binary parent → binary child multinomial edge."""
    return MultinomialEdge(
        source_id="Y",
        target_id="Z",
        conditionals=_make_binary_conditionals(),
    )


# ═══════════════════════════════════════════════════════════════════
# ADD EDGE
# ═══════════════════════════════════════════════════════════════════


class TestAddMultinomialEdge:
    """MultinomialEdge can be added via add_edge()."""

    def test_add_multinomial_edge(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """add_edge() accepts MultinomialEdge."""
        net.add_edge(ternary_edge)
        assert net.has_multinomial_edge("X", "Y")

    def test_add_updates_children(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """MultinomialEdge contributes to children adjacency."""
        net.add_edge(ternary_edge)
        assert "Y" in net.get_children("X")

    def test_add_updates_parents(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """MultinomialEdge contributes to parents adjacency."""
        net.add_edge(ternary_edge)
        assert "X" in net.get_parents("Y")

    def test_add_chain_of_multinomial_edges(
        self,
        net: SLNetwork,
        ternary_edge: MultinomialEdge,
        binary_edge: MultinomialEdge,
    ) -> None:
        """Multiple multinomial edges forming a chain X→Y→Z."""
        net.add_edge(ternary_edge)
        net.add_edge(binary_edge)
        assert net.has_multinomial_edge("X", "Y")
        assert net.has_multinomial_edge("Y", "Z")
        assert "Y" in net.get_children("X")
        assert "Z" in net.get_children("Y")

    def test_add_mixed_edge_types(self, net: SLNetwork) -> None:
        """SLEdge and MultinomialEdge can coexist on different node pairs."""
        # SLEdge X→Y
        binomial_edge = SLEdge(
            source_id="X",
            target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
        )
        net.add_edge(binomial_edge)

        # MultinomialEdge Y→Z
        multi_edge = MultinomialEdge(
            source_id="Y",
            target_id="Z",
            conditionals=_make_binary_conditionals(),
        )
        net.add_edge(multi_edge)

        assert net.has_edge("X", "Y")  # SLEdge
        assert net.has_multinomial_edge("Y", "Z")  # MultinomialEdge

    def test_topological_sort_includes_multinomial(
        self,
        net: SLNetwork,
        ternary_edge: MultinomialEdge,
        binary_edge: MultinomialEdge,
    ) -> None:
        """Topological sort respects multinomial edge ordering."""
        net.add_edge(ternary_edge)  # X→Y
        net.add_edge(binary_edge)  # Y→Z
        topo = net.topological_sort()
        assert topo.index("X") < topo.index("Y")
        assert topo.index("Y") < topo.index("Z")

    def test_roots_and_leaves_with_multinomial(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """Roots and leaves reflect multinomial edges."""
        net.add_edge(ternary_edge)  # X→Y
        roots = net.get_roots()
        leaves = net.get_leaves()
        assert "X" in roots
        assert "Y" not in roots
        # Z is isolated, so it's both a root and leaf
        assert "Z" in roots
        assert "Z" in leaves


# ═══════════════════════════════════════════════════════════════════
# VALIDATION ON ADD
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeValidation:
    """Validation when adding MultinomialEdge to the network."""

    def test_source_node_must_exist(self, net: SLNetwork) -> None:
        """Source node must be in the network."""
        edge = MultinomialEdge(
            source_id="NONEXISTENT",
            target_id="Y",
            conditionals=_make_binary_conditionals(),
        )
        with pytest.raises(NodeNotFoundError):
            net.add_edge(edge)

    def test_target_node_must_exist(self, net: SLNetwork) -> None:
        """Target node must be in the network."""
        edge = MultinomialEdge(
            source_id="X",
            target_id="NONEXISTENT",
            conditionals=_make_binary_conditionals(),
        )
        with pytest.raises(NodeNotFoundError):
            net.add_edge(edge)

    def test_duplicate_multinomial_edge_raises(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """Cannot add the same multinomial edge twice."""
        net.add_edge(ternary_edge)
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge(ternary_edge)

    def test_cycle_detection_direct(self, net: SLNetwork) -> None:
        """Adding an edge that creates a direct cycle is rejected."""
        edge_xy = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=_make_binary_conditionals(),
        )
        edge_yx = MultinomialEdge(
            source_id="Y", target_id="X",
            conditionals=_make_binary_conditionals(),
        )
        net.add_edge(edge_xy)
        with pytest.raises(CycleError):
            net.add_edge(edge_yx)

    def test_cycle_detection_transitive(self, net: SLNetwork) -> None:
        """Adding an edge that creates a transitive cycle is rejected."""
        edge_xy = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=_make_binary_conditionals(),
        )
        edge_yz = MultinomialEdge(
            source_id="Y", target_id="Z",
            conditionals=_make_binary_conditionals(),
        )
        edge_zx = MultinomialEdge(
            source_id="Z", target_id="X",
            conditionals=_make_binary_conditionals(),
        )
        net.add_edge(edge_xy)
        net.add_edge(edge_yz)
        with pytest.raises(CycleError):
            net.add_edge(edge_zx)

    def test_cycle_detection_mixed_edge_types(self, net: SLNetwork) -> None:
        """Cycle detection works across SLEdge and MultinomialEdge."""
        # SLEdge X→Y
        net.add_edge(SLEdge(
            source_id="X", target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
        ))
        # MultinomialEdge Y→Z
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="Z",
            conditionals=_make_binary_conditionals(),
        ))
        # MultinomialEdge Z→X would create cycle
        with pytest.raises(CycleError):
            net.add_edge(MultinomialEdge(
                source_id="Z", target_id="X",
                conditionals=_make_binary_conditionals(),
            ))

    def test_conflict_with_existing_sl_edge_raises(
        self, net: SLNetwork
    ) -> None:
        """Cannot add MultinomialEdge on same (src, tgt) as existing SLEdge."""
        net.add_edge(SLEdge(
            source_id="X", target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
        ))
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge(MultinomialEdge(
                source_id="X", target_id="Y",
                conditionals=_make_binary_conditionals(),
            ))

    def test_conflict_sl_edge_over_multinomial_raises(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """Cannot add SLEdge on same (src, tgt) as existing MultinomialEdge."""
        net.add_edge(ternary_edge)
        with pytest.raises(ValueError, match="already exists"):
            net.add_edge(SLEdge(
                source_id="X", target_id="Y",
                conditional=Opinion(0.8, 0.1, 0.1),
            ))


# ═══════════════════════════════════════════════════════════════════
# RETRIEVAL
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeRetrieval:
    """Retrieve multinomial edges from the network."""

    def test_get_multinomial_edge(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """get_multinomial_edge returns the stored edge."""
        net.add_edge(ternary_edge)
        retrieved = net.get_multinomial_edge("X", "Y")
        assert retrieved is ternary_edge

    def test_get_nonexistent_raises(self, net: SLNetwork) -> None:
        """get_multinomial_edge raises for nonexistent edge."""
        with pytest.raises(ValueError, match="[Mm]ultinomial.*not found|does not exist"):
            net.get_multinomial_edge("X", "Y")

    def test_has_multinomial_edge_true(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """has_multinomial_edge returns True when edge exists."""
        net.add_edge(ternary_edge)
        assert net.has_multinomial_edge("X", "Y") is True

    def test_has_multinomial_edge_false(self, net: SLNetwork) -> None:
        """has_multinomial_edge returns False when edge doesn't exist."""
        assert net.has_multinomial_edge("X", "Y") is False

    def test_has_multinomial_edge_not_sl_edge(
        self, net: SLNetwork
    ) -> None:
        """has_multinomial_edge returns False for SLEdge on same pair."""
        net.add_edge(SLEdge(
            source_id="X", target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
        ))
        assert net.has_multinomial_edge("X", "Y") is False


# ═══════════════════════════════════════════════════════════════════
# EDGE COUNT
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeCount:
    """MultinomialEdges are counted in edge_count()."""

    def test_edge_count_includes_multinomial(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """edge_count() includes multinomial edges."""
        assert net.edge_count() == 0
        net.add_edge(ternary_edge)
        assert net.edge_count() == 1

    def test_edge_count_mixed(self, net: SLNetwork) -> None:
        """edge_count() sums SLEdge + MultinomialEdge."""
        net.add_edge(SLEdge(
            source_id="X", target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
        ))
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="Z",
            conditionals=_make_binary_conditionals(),
        ))
        assert net.edge_count() == 2


# ═══════════════════════════════════════════════════════════════════
# REMOVE NODE CLEANUP
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeRemoveNode:
    """remove_node() cleans up MultinomialEdges."""

    def test_remove_source_cleans_multinomial_edge(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """Removing the source node removes the multinomial edge."""
        net.add_edge(ternary_edge)  # X→Y
        net.remove_node("X")
        assert not net.has_multinomial_edge("X", "Y")
        # Y should have no parents now
        assert net.get_parents("Y") == []

    def test_remove_target_cleans_multinomial_edge(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """Removing the target node removes the multinomial edge."""
        net.add_edge(ternary_edge)  # X→Y
        net.remove_node("Y")
        assert not net.has_node("Y")
        # X should have no children now
        assert net.get_children("X") == []

    def test_remove_middle_node_cleans_chain(
        self,
        net: SLNetwork,
        ternary_edge: MultinomialEdge,
        binary_edge: MultinomialEdge,
    ) -> None:
        """Removing a middle node cleans both incident edges."""
        net.add_edge(ternary_edge)  # X→Y
        net.add_edge(binary_edge)   # Y→Z
        net.remove_node("Y")
        assert not net.has_multinomial_edge("X", "Y")
        assert not net.has_multinomial_edge("Y", "Z")
        assert net.get_children("X") == []
        assert net.get_parents("Z") == []


# ═══════════════════════════════════════════════════════════════════
# IS_TREE INTERACTION
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeIsTree:
    """is_tree() accounts for MultinomialEdges."""

    def test_single_multinomial_edge_is_tree(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """A single multinomial edge X→Y (plus isolated Z) is a tree."""
        net.add_edge(ternary_edge)
        # Each node has at most one parent, so tree structure holds
        assert net.is_tree() is True

    def test_diamond_via_multinomial_not_tree(self, net: SLNetwork) -> None:
        """Two multinomial edges converging on same target → not a tree."""
        net.add_node(SLNode(node_id="W", opinion=_vacuous_binomial()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="W",
            conditionals=_make_binary_conditionals(),
        ))
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="W",
            conditionals=_make_binary_conditionals(),
        ))
        # W has two parents → not a tree
        assert net.is_tree() is False


# ═══════════════════════════════════════════════════════════════════
# REPR / LEN
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeNetworkRepr:
    """Network repr and len reflect multinomial edges."""

    def test_len_includes_multinomial_edges(
        self, net: SLNetwork, ternary_edge: MultinomialEdge
    ) -> None:
        """len(network) counts nodes (unchanged behavior)."""
        assert len(net) == 3
        net.add_edge(ternary_edge)
        assert len(net) == 3  # len is node count, not edge count
