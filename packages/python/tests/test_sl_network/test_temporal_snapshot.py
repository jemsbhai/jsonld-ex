"""Tests for network_at_time() -- point-in-time snapshots (Tier 3, Step 5).

Model C: filter a network to only include edges whose validity windows
(@validFrom / @validUntil) contain the given timestamp. Nodes with no
inbound valid edges and no children become isolated but are still
included (they may be roots with intrinsic opinions).

Validity rules:
    - valid_from is None:  edge is valid from the beginning of time.
    - valid_until is None: edge is valid until the end of time.
    - Both None:           edge is always valid.
    - Edge is valid at t iff (valid_from is None or valid_from <= t)
                         and (valid_until is None or t <= valid_until).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import SLNode, SLEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.temporal import network_at_time


# -- Fixtures --

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
T1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
T2 = datetime(2025, 1, 1, tzinfo=timezone.utc)
T3 = datetime(2025, 6, 1, tzinfo=timezone.utc)

OP_A = Opinion(0.7, 0.1, 0.2)
OP_B = Opinion(0.5, 0.3, 0.2)
OP_C = Opinion(0.3, 0.3, 0.4)
COND = Opinion(0.9, 0.05, 0.05)
CF = Opinion(0.1, 0.7, 0.2)


class TestNetworkAtTimeEdgeFiltering:
    """Edges outside their validity window are excluded."""

    def test_edge_within_window_included(self):
        """An edge valid from T0 to T2 is included at T1."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND, counterfactual=CF,
            valid_from=T0, valid_until=T2,
        ))
        snapshot = network_at_time(net, T1)
        assert snapshot.has_edge("a", "b")

    def test_edge_before_valid_from_excluded(self):
        """An edge valid from T1 onward is excluded at T0."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND, counterfactual=CF,
            valid_from=T1, valid_until=T3,
        ))
        snapshot = network_at_time(net, T0)
        assert not snapshot.has_edge("a", "b")

    def test_edge_after_valid_until_excluded(self):
        """An edge valid until T1 is excluded at T2."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND, counterfactual=CF,
            valid_from=T0, valid_until=T1,
        ))
        snapshot = network_at_time(net, T2)
        assert not snapshot.has_edge("a", "b")

    def test_edge_at_exact_valid_from_included(self):
        """Boundary: edge is valid at exactly valid_from."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T1, valid_until=T3,
        ))
        snapshot = network_at_time(net, T1)
        assert snapshot.has_edge("a", "b")

    def test_edge_at_exact_valid_until_included(self):
        """Boundary: edge is valid at exactly valid_until."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T0, valid_until=T1,
        ))
        snapshot = network_at_time(net, T1)
        assert snapshot.has_edge("a", "b")


class TestNetworkAtTimeNullBounds:
    """None bounds mean open-ended validity."""

    def test_no_valid_from_means_always_valid_before(self):
        """valid_from=None, valid_until=T2: valid at any time <= T2."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=None, valid_until=T2,
        ))
        snapshot = network_at_time(net, T0)
        assert snapshot.has_edge("a", "b")

    def test_no_valid_until_means_always_valid_after(self):
        """valid_from=T0, valid_until=None: valid at any time >= T0."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T0, valid_until=None,
        ))
        snapshot = network_at_time(net, T3)
        assert snapshot.has_edge("a", "b")

    def test_both_none_always_valid(self):
        """valid_from=None, valid_until=None: always valid."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=None, valid_until=None,
        ))
        snapshot = network_at_time(net, T3)
        assert snapshot.has_edge("a", "b")


class TestNetworkAtTimeNodePreservation:
    """All nodes are preserved regardless of edge filtering."""

    def test_all_nodes_present_even_if_edges_removed(self):
        """Nodes whose edges are filtered out still appear in the snapshot."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T2, valid_until=T3,  # not valid at T0
        ))
        snapshot = network_at_time(net, T0)
        assert snapshot.has_node("a")
        assert snapshot.has_node("b")
        assert not snapshot.has_edge("a", "b")

    def test_node_opinions_preserved(self):
        """Node opinions are copied exactly, not decayed."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T0, valid_until=T3,
        ))
        snapshot = network_at_time(net, T1)
        assert snapshot.get_node("a").opinion.belief == pytest.approx(OP_A.belief)
        assert snapshot.get_node("b").opinion.belief == pytest.approx(OP_B.belief)


class TestNetworkAtTimeMultipleEdges:
    """Networks with multiple edges where some are filtered."""

    def test_partial_edge_filtering(self):
        """Only edges valid at the query time survive."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_node(SLNode(node_id="c", opinion=OP_C))
        # a->b valid in [T0, T1]
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND, counterfactual=CF,
            valid_from=T0, valid_until=T1,
        ))
        # b->c valid in [T1, T3]
        net.add_edge(SLEdge(
            source_id="b", target_id="c",
            conditional=COND, counterfactual=CF,
            valid_from=T1, valid_until=T3,
        ))
        # Query at T2: a->b expired, b->c valid
        snapshot = network_at_time(net, T2)
        assert not snapshot.has_edge("a", "b")
        assert snapshot.has_edge("b", "c")

    def test_all_edges_valid(self):
        """If all edges are valid at query time, the snapshot matches."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_node(SLNode(node_id="c", opinion=OP_C))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T0, valid_until=T3,
        ))
        net.add_edge(SLEdge(
            source_id="b", target_id="c",
            conditional=COND,
            valid_from=T0, valid_until=T3,
        ))
        snapshot = network_at_time(net, T1)
        assert snapshot.has_edge("a", "b")
        assert snapshot.has_edge("b", "c")
        assert snapshot.node_count() == 3


class TestNetworkAtTimeOriginalUnchanged:
    """network_at_time does not modify the original."""

    def test_original_preserved(self):
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T2, valid_until=T3,
        ))
        _ = network_at_time(net, T0)
        # Original still has the edge
        assert net.has_edge("a", "b")


class TestNetworkAtTimeDAGInvariant:
    """The snapshot is always a valid DAG."""

    def test_snapshot_is_dag(self):
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=OP_A))
        net.add_node(SLNode(node_id="b", opinion=OP_B))
        net.add_node(SLNode(node_id="c", opinion=OP_C))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=COND,
            valid_from=T0, valid_until=T3,
        ))
        net.add_edge(SLEdge(
            source_id="b", target_id="c",
            conditional=COND,
            valid_from=T0, valid_until=T1,
        ))
        snapshot = network_at_time(net, T2)
        assert snapshot.is_dag()


# ═══════════════════════════════════════════════════════════════════
# Gap A: network_at_time must preserve multinomial edge types
# ═══════════════════════════════════════════════════════════════════


class TestNetworkAtTimeMultinomialEdges:
    """network_at_time must copy MultinomialEdge and MultiParentMultinomialEdge."""

    def test_multinomial_edge_preserved(self) -> None:
        """MultinomialEdge survives point-in-time snapshot."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        br = {"H": 0.5, "L": 0.5}
        net = SLNetwork()
        net.add_node(SLNode("A", COND))
        net.add_node(SLNode("B", COND))
        net.add_edge(MultinomialEdge(
            source_id="A", target_id="B",
            conditionals={
                "a": MultinomialOpinion(beliefs={"H": 0.7, "L": 0.1},
                                        uncertainty=0.2, base_rates=br),
                "b": MultinomialOpinion(beliefs={"H": 0.2, "L": 0.5},
                                        uncertainty=0.3, base_rates=br),
            },
        ))

        snapshot = network_at_time(net, T1)
        assert snapshot.has_multinomial_edge("A", "B")

    def test_multi_parent_multinomial_edge_preserved(self) -> None:
        """MultiParentMultinomialEdge survives point-in-time snapshot."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultiParentMultinomialEdge

        br = {"H": 0.5, "L": 0.5}
        cond = MultinomialOpinion(
            beliefs={"H": 0.5, "L": 0.3}, uncertainty=0.2, base_rates=br,
        )

        net = SLNetwork()
        net.add_node(SLNode("P1", COND))
        net.add_node(SLNode("P2", COND))
        net.add_node(SLNode("C", COND))
        net.add_edge(MultiParentMultinomialEdge(
            parent_ids=("P1", "P2"), target_id="C",
            conditionals={("a", "x"): cond, ("b", "x"): cond},
        ))

        snapshot = network_at_time(net, T1)
        assert snapshot.has_multi_parent_multinomial_edge("C")
