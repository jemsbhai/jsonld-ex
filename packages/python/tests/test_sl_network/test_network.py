"""
Tests for SLNetwork graph construction (Tier 1, Step 2).

Covers: add/remove nodes and edges, cycle detection, topological sort,
graph queries (roots, leaves, parents, children), structural analysis
(is_tree, is_dag), serialization, and all error paths.

TDD: These tests define the contract that network.py must satisfy.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    InferenceResult,
    InferenceStep,
    MultiParentEdge,
    SLEdge,
    SLNode,
)
from jsonld_ex.sl_network.network import (
    CycleError,
    NodeNotFoundError,
    SLNetwork,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def op_high() -> Opinion:
    return Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)


@pytest.fixture
def op_mod() -> Opinion:
    return Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)


@pytest.fixture
def op_low() -> Opinion:
    return Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2)


@pytest.fixture
def op_vacuous() -> Opinion:
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def empty_net() -> SLNetwork:
    return SLNetwork(name="test_network")


@pytest.fixture
def linear_net(op_high: Opinion, op_mod: Opinion, op_low: Opinion) -> SLNetwork:
    """A → B → C linear chain."""
    net = SLNetwork(name="linear")
    net.add_node(SLNode("A", op_high))
    net.add_node(SLNode("B", op_mod))
    net.add_node(SLNode("C", op_low))
    net.add_edge(SLEdge("A", "B", conditional=op_high))
    net.add_edge(SLEdge("B", "C", conditional=op_mod))
    return net


@pytest.fixture
def diamond_net(op_high: Opinion, op_mod: Opinion, op_low: Opinion) -> SLNetwork:
    """Diamond DAG: A → B, A → C, B → D, C → D."""
    net = SLNetwork(name="diamond")
    net.add_node(SLNode("A", op_high))
    net.add_node(SLNode("B", op_mod))
    net.add_node(SLNode("C", op_mod))
    net.add_node(SLNode("D", op_low))
    net.add_edge(SLEdge("A", "B", conditional=op_high))
    net.add_edge(SLEdge("A", "C", conditional=op_high))
    net.add_edge(SLEdge("B", "D", conditional=op_mod))
    net.add_edge(SLEdge("C", "D", conditional=op_mod))
    return net


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCTOR
# ═══════════════════════════════════════════════════════════════════


class TestSLNetworkConstructor:
    """Test SLNetwork creation."""

    def test_empty_network(self) -> None:
        net = SLNetwork()
        assert net.name is None
        assert net.node_count() == 0
        assert net.edge_count() == 0

    def test_named_network(self) -> None:
        net = SLNetwork(name="medical")
        assert net.name == "medical"

    def test_repr_empty(self) -> None:
        net = SLNetwork()
        r = repr(net)
        assert "nodes=0" in r
        assert "edges=0" in r

    def test_repr_named(self) -> None:
        net = SLNetwork(name="test")
        assert "'test'" in repr(net)

    def test_len(self, linear_net: SLNetwork) -> None:
        assert len(linear_net) == 3


# ═══════════════════════════════════════════════════════════════════
# ADD NODE
# ═══════════════════════════════════════════════════════════════════


class TestAddNode:
    """Test adding nodes to the network."""

    def test_add_single_node(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        node = SLNode("X", op_high)
        empty_net.add_node(node)
        assert empty_net.node_count() == 1
        assert empty_net.has_node("X")

    def test_add_multiple_nodes(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_high))
        empty_net.add_node(SLNode("C", op_high))
        assert empty_net.node_count() == 3

    def test_get_node_returns_same_object(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        node = SLNode("X", op_high)
        empty_net.add_node(node)
        retrieved = empty_net.get_node("X")
        assert retrieved is node

    def test_duplicate_node_raises(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("X", op_high))
        with pytest.raises(ValueError, match="already exists"):
            empty_net.add_node(SLNode("X", op_high))

    def test_non_slnode_raises(self, empty_net: SLNetwork) -> None:
        with pytest.raises(TypeError, match="SLNode"):
            empty_net.add_node({"node_id": "X"})  # type: ignore[arg-type]

    def test_isolated_node_is_root_and_leaf(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("X", op_high))
        assert "X" in empty_net.get_roots()
        assert "X" in empty_net.get_leaves()


# ═══════════════════════════════════════════════════════════════════
# ADD EDGE
# ═══════════════════════════════════════════════════════════════════


class TestAddEdge:
    """Test adding edges to the network."""

    def test_add_simple_edge(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        edge = SLEdge("A", "B", conditional=op_high)
        empty_net.add_edge(edge)
        assert empty_net.edge_count() == 1
        assert empty_net.has_edge("A", "B")

    def test_get_edge_returns_same_object(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        edge = SLEdge("A", "B", conditional=op_high)
        empty_net.add_edge(edge)
        retrieved = empty_net.get_edge("A", "B")
        assert retrieved is edge

    def test_missing_source_raises(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("B", op_mod))
        with pytest.raises(NodeNotFoundError):
            empty_net.add_edge(SLEdge("A", "B", conditional=op_high))

    def test_missing_target_raises(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        with pytest.raises(NodeNotFoundError):
            empty_net.add_edge(SLEdge("A", "B", conditional=op_high))

    def test_duplicate_edge_raises(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        with pytest.raises(ValueError, match="already exists"):
            empty_net.add_edge(SLEdge("A", "B", conditional=op_mod))

    def test_non_edge_type_raises(self, empty_net: SLNetwork) -> None:
        with pytest.raises(TypeError, match="SLEdge or MultiParentEdge"):
            empty_net.add_edge(("A", "B"))  # type: ignore[arg-type]

    def test_edge_updates_parents_children(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        assert empty_net.get_children("A") == ["B"]
        assert empty_net.get_parents("B") == ["A"]
        assert empty_net.get_parents("A") == []
        assert empty_net.get_children("B") == []


# ═══════════════════════════════════════════════════════════════════
# MULTI-PARENT EDGE
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentEdgeInNetwork:
    """Test adding MultiParentEdge to the network."""

    def test_add_multi_parent_edge(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion, op_low: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_node(SLNode("Y", op_low))
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_low,
        }
        mpe = MultiParentEdge(target_id="Y", parent_ids=("A", "B"), conditionals=conds)
        empty_net.add_edge(mpe)
        # 2 parent→target edges
        assert empty_net.edge_count() == 2
        assert empty_net.get_parents("Y") == ["A", "B"]

    def test_get_multi_parent_edge(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion, op_low: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_node(SLNode("Y", op_low))
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_low,
        }
        mpe = MultiParentEdge(target_id="Y", parent_ids=("A", "B"), conditionals=conds)
        empty_net.add_edge(mpe)
        retrieved = empty_net.get_multi_parent_edge("Y")
        assert retrieved is mpe

    def test_multi_parent_missing_node_raises(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        # "B" is missing
        empty_net.add_node(SLNode("Y", op_mod))
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_mod,
        }
        mpe = MultiParentEdge(target_id="Y", parent_ids=("A", "B"), conditionals=conds)
        with pytest.raises(NodeNotFoundError):
            empty_net.add_edge(mpe)

    def test_multi_parent_duplicate_raises(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion, op_low: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_node(SLNode("Y", op_low))
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_low,
        }
        mpe = MultiParentEdge(target_id="Y", parent_ids=("A", "B"), conditionals=conds)
        empty_net.add_edge(mpe)
        with pytest.raises(ValueError, match="already exists"):
            empty_net.add_edge(mpe)

    def test_get_multi_parent_edge_missing_raises(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("X", op_high))
        with pytest.raises(ValueError, match="No MultiParentEdge"):
            empty_net.get_multi_parent_edge("X")

    def test_get_multi_parent_edge_no_node_raises(
        self, empty_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_net.get_multi_parent_edge("nonexistent")


# ═══════════════════════════════════════════════════════════════════
# CYCLE DETECTION
# ═══════════════════════════════════════════════════════════════════


class TestCycleDetection:
    """Test that cycles are rejected on edge insertion."""

    def test_direct_cycle_rejected(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        """A → B, then B → A creates a cycle."""
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        with pytest.raises(CycleError) as exc_info:
            empty_net.add_edge(SLEdge("B", "A", conditional=op_mod))
        assert "cycle" in str(exc_info.value).lower()
        assert isinstance(exc_info.value.path, list)

    def test_indirect_cycle_rejected(
        self, linear_net: SLNetwork, op_high: Opinion
    ) -> None:
        """A → B → C, then C → A creates a cycle."""
        with pytest.raises(CycleError):
            linear_net.add_edge(SLEdge("C", "A", conditional=op_high))

    def test_long_cycle_rejected(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        """A → B → C → D → E, then E → A creates a 5-node cycle."""
        for nid in "ABCDE":
            empty_net.add_node(SLNode(nid, op_high))
        for src, tgt in [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")]:
            empty_net.add_edge(SLEdge(src, tgt, conditional=op_high))
        with pytest.raises(CycleError):
            empty_net.add_edge(SLEdge("E", "A", conditional=op_high))

    def test_non_cycle_edge_accepted(
        self, linear_net: SLNetwork, op_high: Opinion
    ) -> None:
        """A → B → C.  Adding A → C is NOT a cycle (it's a shortcut)."""
        linear_net.add_edge(SLEdge("A", "C", conditional=op_high))
        assert linear_net.has_edge("A", "C")

    def test_cycle_error_has_path(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        """CycleError includes the cycle path."""
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_high))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        with pytest.raises(CycleError) as exc_info:
            empty_net.add_edge(SLEdge("B", "A", conditional=op_high))
        path = exc_info.value.path
        assert len(path) >= 2
        # Path should start and end with the same node (cycle)
        assert path[0] == path[-1]

    def test_multi_parent_cycle_rejected(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion
    ) -> None:
        """MultiParentEdge that would create a cycle is rejected."""
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_node(SLNode("C", op_mod))
        # C → A exists
        empty_net.add_edge(SLEdge("C", "A", conditional=op_high))
        # Now try to add MultiParentEdge with parents (A, B) → C
        # This creates A→C, but C→A already exists → cycle
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_mod,
        }
        mpe = MultiParentEdge(target_id="C", parent_ids=("A", "B"), conditionals=conds)
        with pytest.raises(CycleError):
            empty_net.add_edge(mpe)

    def test_existing_simple_edge_blocks_multi_parent(
        self, empty_net: SLNetwork, op_high: Opinion, op_mod: Opinion, op_low: Opinion
    ) -> None:
        """If A→Y already exists as SLEdge, can't add MultiParentEdge (A,B)→Y."""
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_mod))
        empty_net.add_node(SLNode("Y", op_low))
        empty_net.add_edge(SLEdge("A", "Y", conditional=op_high))
        conds = {
            (True, True): op_high,
            (True, False): op_mod,
            (False, True): op_mod,
            (False, False): op_low,
        }
        mpe = MultiParentEdge(target_id="Y", parent_ids=("A", "B"), conditionals=conds)
        with pytest.raises(ValueError, match="already exists as SLEdge"):
            empty_net.add_edge(mpe)


# ═══════════════════════════════════════════════════════════════════
# REMOVE NODE
# ═══════════════════════════════════════════════════════════════════


class TestRemoveNode:
    """Test removing nodes from the network."""

    def test_remove_isolated_node(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("X", op_high))
        empty_net.remove_node("X")
        assert empty_net.node_count() == 0
        assert not empty_net.has_node("X")

    def test_remove_node_removes_outgoing_edges(
        self, linear_net: SLNetwork
    ) -> None:
        """Removing A from A→B→C removes the A→B edge."""
        linear_net.remove_node("A")
        assert not linear_net.has_edge("A", "B")
        assert linear_net.get_parents("B") == []
        assert linear_net.node_count() == 2

    def test_remove_node_removes_incoming_edges(
        self, linear_net: SLNetwork
    ) -> None:
        """Removing C from A→B→C removes the B→C edge."""
        linear_net.remove_node("C")
        assert not linear_net.has_edge("B", "C")
        assert linear_net.get_children("B") == []
        assert linear_net.node_count() == 2

    def test_remove_middle_node(self, linear_net: SLNetwork) -> None:
        """Removing B from A→B→C removes both edges."""
        linear_net.remove_node("B")
        assert linear_net.node_count() == 2
        assert linear_net.edge_count() == 0
        assert linear_net.get_children("A") == []
        assert linear_net.get_parents("C") == []

    def test_remove_nonexistent_raises(self, empty_net: SLNetwork) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_net.remove_node("X")

    def test_node_not_found_error_has_node_id(
        self, empty_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError) as exc_info:
            empty_net.remove_node("ghost")
        assert exc_info.value.node_id == "ghost"


# ═══════════════════════════════════════════════════════════════════
# REMOVE EDGE
# ═══════════════════════════════════════════════════════════════════


class TestRemoveEdge:
    """Test removing edges from the network."""

    def test_remove_edge(self, linear_net: SLNetwork) -> None:
        linear_net.remove_edge("A", "B")
        assert not linear_net.has_edge("A", "B")
        assert linear_net.edge_count() == 1
        assert linear_net.get_children("A") == []
        assert linear_net.get_parents("B") == []

    def test_remove_edge_preserves_nodes(self, linear_net: SLNetwork) -> None:
        linear_net.remove_edge("A", "B")
        assert linear_net.has_node("A")
        assert linear_net.has_node("B")

    def test_remove_nonexistent_edge_raises(
        self, linear_net: SLNetwork
    ) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            linear_net.remove_edge("A", "C")

    def test_remove_edge_missing_source_raises(
        self, linear_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            linear_net.remove_edge("Z", "A")

    def test_remove_edge_missing_target_raises(
        self, linear_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            linear_net.remove_edge("A", "Z")


# ═══════════════════════════════════════════════════════════════════
# GRAPH QUERIES
# ═══════════════════════════════════════════════════════════════════


class TestGraphQueries:
    """Test graph query methods."""

    def test_get_node_missing_raises(self, empty_net: SLNetwork) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_net.get_node("X")

    def test_get_edge_missing_raises(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_high))
        with pytest.raises(ValueError, match="does not exist"):
            empty_net.get_edge("A", "B")

    def test_get_parents_missing_node_raises(
        self, empty_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_net.get_parents("X")

    def test_get_children_missing_node_raises(
        self, empty_net: SLNetwork
    ) -> None:
        with pytest.raises(NodeNotFoundError):
            empty_net.get_children("X")

    def test_roots_of_linear(self, linear_net: SLNetwork) -> None:
        assert linear_net.get_roots() == ["A"]

    def test_leaves_of_linear(self, linear_net: SLNetwork) -> None:
        assert linear_net.get_leaves() == ["C"]

    def test_roots_of_diamond(self, diamond_net: SLNetwork) -> None:
        assert diamond_net.get_roots() == ["A"]

    def test_leaves_of_diamond(self, diamond_net: SLNetwork) -> None:
        assert diamond_net.get_leaves() == ["D"]

    def test_roots_sorted(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        """Multiple roots are returned in sorted order."""
        for nid in ["C", "A", "B"]:
            empty_net.add_node(SLNode(nid, op_high))
        assert empty_net.get_roots() == ["A", "B", "C"]

    def test_leaves_sorted(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        """Multiple leaves are returned in sorted order."""
        for nid in ["C", "A", "B"]:
            empty_net.add_node(SLNode(nid, op_high))
        assert empty_net.get_leaves() == ["A", "B", "C"]

    def test_has_node_true(self, linear_net: SLNetwork) -> None:
        assert linear_net.has_node("A") is True

    def test_has_node_false(self, linear_net: SLNetwork) -> None:
        assert linear_net.has_node("Z") is False

    def test_has_edge_true(self, linear_net: SLNetwork) -> None:
        assert linear_net.has_edge("A", "B") is True

    def test_has_edge_false(self, linear_net: SLNetwork) -> None:
        assert linear_net.has_edge("A", "C") is False

    def test_node_count(self, diamond_net: SLNetwork) -> None:
        assert diamond_net.node_count() == 4

    def test_edge_count(self, diamond_net: SLNetwork) -> None:
        assert diamond_net.edge_count() == 4

    def test_parents_returns_copy(self, linear_net: SLNetwork) -> None:
        """get_parents returns a copy, not a reference to internal state."""
        parents = linear_net.get_parents("B")
        parents.append("HACKED")
        assert linear_net.get_parents("B") == ["A"]

    def test_children_returns_copy(self, linear_net: SLNetwork) -> None:
        """get_children returns a copy, not a reference to internal state."""
        children = linear_net.get_children("A")
        children.append("HACKED")
        assert linear_net.get_children("A") == ["B"]


# ═══════════════════════════════════════════════════════════════════
# TOPOLOGICAL SORT
# ═══════════════════════════════════════════════════════════════════


class TestTopologicalSort:
    """Test topological sort correctness and determinism."""

    def test_empty_network(self, empty_net: SLNetwork) -> None:
        assert empty_net.topological_sort() == []

    def test_single_node(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        empty_net.add_node(SLNode("A", op_high))
        assert empty_net.topological_sort() == ["A"]

    def test_linear_chain(self, linear_net: SLNetwork) -> None:
        """A → B → C must produce [A, B, C]."""
        assert linear_net.topological_sort() == ["A", "B", "C"]

    def test_diamond(self, diamond_net: SLNetwork) -> None:
        """Diamond: A first, D last, B and C in between (sorted)."""
        topo = diamond_net.topological_sort()
        assert topo[0] == "A"
        assert topo[-1] == "D"
        # B and C can be in either order, but lexicographic gives B before C
        assert topo == ["A", "B", "C", "D"]

    def test_deterministic(self, diamond_net: SLNetwork) -> None:
        """Same graph always produces the same sort order."""
        result1 = diamond_net.topological_sort()
        result2 = diamond_net.topological_sort()
        assert result1 == result2

    def test_disconnected_components(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        """Disconnected nodes are sorted lexicographically."""
        for nid in ["C", "A", "B"]:
            empty_net.add_node(SLNode(nid, op_high))
        assert empty_net.topological_sort() == ["A", "B", "C"]

    def test_forest(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        """Two independent chains: A→B, C→D."""
        for nid in "ABCD":
            empty_net.add_node(SLNode(nid, op_high))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        empty_net.add_edge(SLEdge("C", "D", conditional=op_high))
        topo = empty_net.topological_sort()
        # A before B, C before D (topological constraint)
        assert topo.index("A") < topo.index("B")
        assert topo.index("C") < topo.index("D")
        # Kahn's with lexicographic queue: after processing A,
        # B becomes available (in-degree 0) and is smaller than C
        assert topo == ["A", "B", "C", "D"]

    def test_wide_graph(self, empty_net: SLNetwork, op_high: Opinion) -> None:
        """Root → many leaves: A → B, A → C, A → D, A → E."""
        empty_net.add_node(SLNode("A", op_high))
        for child in "EDCB":  # Add in reverse to test sorting
            empty_net.add_node(SLNode(child, op_high))
            empty_net.add_edge(SLEdge("A", child, conditional=op_high))
        topo = empty_net.topological_sort()
        assert topo[0] == "A"
        assert topo[1:] == ["B", "C", "D", "E"]

    def test_parent_always_before_child(
        self, diamond_net: SLNetwork
    ) -> None:
        """Every parent appears before its children in topo order."""
        topo = diamond_net.topological_sort()
        idx = {nid: i for i, nid in enumerate(topo)}
        assert idx["A"] < idx["B"]
        assert idx["A"] < idx["C"]
        assert idx["B"] < idx["D"]
        assert idx["C"] < idx["D"]


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════


class TestStructuralAnalysis:
    """Test is_tree and is_dag."""

    def test_empty_is_tree(self, empty_net: SLNetwork) -> None:
        assert empty_net.is_tree() is True

    def test_single_node_is_tree(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        assert empty_net.is_tree() is True

    def test_linear_chain_is_tree(self, linear_net: SLNetwork) -> None:
        assert linear_net.is_tree() is True

    def test_diamond_is_not_tree(self, diamond_net: SLNetwork) -> None:
        """D has two parents → not a tree."""
        assert diamond_net.is_tree() is False

    def test_forest_is_tree(
        self, empty_net: SLNetwork, op_high: Opinion
    ) -> None:
        """Two disconnected chains: each node has ≤1 parent."""
        for nid in "ABCD":
            empty_net.add_node(SLNode(nid, op_high))
        empty_net.add_edge(SLEdge("A", "B", conditional=op_high))
        empty_net.add_edge(SLEdge("C", "D", conditional=op_high))
        assert empty_net.is_tree() is True

    def test_empty_is_dag(self, empty_net: SLNetwork) -> None:
        assert empty_net.is_dag() is True

    def test_linear_is_dag(self, linear_net: SLNetwork) -> None:
        assert linear_net.is_dag() is True

    def test_diamond_is_dag(self, diamond_net: SLNetwork) -> None:
        assert diamond_net.is_dag() is True


# ═══════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


class TestSerialization:
    """Test to_dict serialization."""

    def test_empty_to_dict(self, empty_net: SLNetwork) -> None:
        d = empty_net.to_dict()
        assert d["name"] == "test_network"
        assert d["nodes"] == {}
        assert d["edges"] == {}
        assert d["multi_parent_edges"] == {}

    def test_linear_to_dict(self, linear_net: SLNetwork) -> None:
        d = linear_net.to_dict()
        assert len(d["nodes"]) == 3
        assert "A" in d["nodes"]
        assert "B" in d["nodes"]
        assert "C" in d["nodes"]
        assert len(d["edges"]) == 2
        assert "A->B" in d["edges"]
        assert "B->C" in d["edges"]

    def test_node_dict_structure(self, linear_net: SLNetwork) -> None:
        d = linear_net.to_dict()
        node_a = d["nodes"]["A"]
        assert node_a["node_id"] == "A"
        assert "belief" in node_a["opinion"]
        assert node_a["node_type"] == "content"

    def test_edge_dict_structure(self, linear_net: SLNetwork) -> None:
        d = linear_net.to_dict()
        edge = d["edges"]["A->B"]
        assert edge["source_id"] == "A"
        assert edge["target_id"] == "B"
        assert "belief" in edge["conditional"]

    def test_counterfactual_none_serializes(
        self, linear_net: SLNetwork
    ) -> None:
        d = linear_net.to_dict()
        edge = d["edges"]["A->B"]
        assert edge["counterfactual"] is None

    def test_counterfactual_present_serializes(
        self, empty_net: SLNetwork, op_high: Opinion, op_vacuous: Opinion
    ) -> None:
        empty_net.add_node(SLNode("A", op_high))
        empty_net.add_node(SLNode("B", op_high))
        empty_net.add_edge(SLEdge(
            "A", "B",
            conditional=op_high,
            counterfactual=op_vacuous,
        ))
        d = empty_net.to_dict()
        edge = d["edges"]["A->B"]
        assert edge["counterfactual"] is not None
        assert "belief" in edge["counterfactual"]


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Verify SLNetwork does not affect existing modules."""

    def test_existing_operators_still_work(self) -> None:
        """Importing SLNetwork doesn't break confidence_algebra."""
        from jsonld_ex.confidence_algebra import (
            Opinion,
            cumulative_fuse,
            deduce,
            trust_discount,
        )

        op = Opinion(0.7, 0.2, 0.1)
        fused = cumulative_fuse(op, op)
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_import_from_package(self) -> None:
        """SLNetwork is importable from sl_network package."""
        from jsonld_ex.sl_network import SLNetwork, CycleError, NodeNotFoundError
        assert SLNetwork is not None
        assert issubclass(CycleError, ValueError)
        assert issubclass(NodeNotFoundError, KeyError)
