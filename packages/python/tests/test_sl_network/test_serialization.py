"""
Tests for SLNetwork serialization (Tier 1, Step 8).

Covers:
    - to_dict / from_dict roundtrip
    - to_jsonld / from_jsonld roundtrip
    - Fidelity: all node opinions, edge conditionals, counterfactuals,
      multi-parent edges, metadata, labels, and network name survive
      the roundtrip exactly.
    - Edge cases: empty network, single node, counterfactual=None,
      multi-parent edges with 2 and 3 parents.
    - Inference equivalence: inference on the deserialized network
      produces the same result as inference on the original.
    - _parse_bool_tuple helper correctness.

TDD: These tests define the contract for serialization roundtrips.
"""

from __future__ import annotations

import json

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.inference import infer_node
from jsonld_ex.sl_network.network import SLNetwork, _parse_bool_tuple
from jsonld_ex.sl_network.types import MultiParentEdge, SLEdge, SLNode


_TOL = 1e-12


def _assert_opinions_equal(a: Opinion, b: Opinion, label: str = "") -> None:
    prefix = f"[{label}] " if label else ""
    assert abs(a.belief - b.belief) < _TOL, f"{prefix}belief"
    assert abs(a.disbelief - b.disbelief) < _TOL, f"{prefix}disbelief"
    assert abs(a.uncertainty - b.uncertainty) < _TOL, f"{prefix}uncertainty"
    assert abs(a.base_rate - b.base_rate) < _TOL, f"{prefix}base_rate"


# ═══════════════════════════════════════════════════════════════════
# _parse_bool_tuple HELPER
# ═══════════════════════════════════════════════════════════════════


class TestParseBoolTuple:
    """Test the stringified bool-tuple parser."""

    def test_two_element_with_parens(self) -> None:
        result = _parse_bool_tuple("(True, False)", 2)
        assert result == (True, False)

    def test_two_element_without_parens(self) -> None:
        result = _parse_bool_tuple("True, False", 2)
        assert result == (True, False)

    def test_three_element(self) -> None:
        result = _parse_bool_tuple("(True, False, True)", 3)
        assert result == (True, False, True)

    def test_all_true(self) -> None:
        result = _parse_bool_tuple("(True, True)", 2)
        assert result == (True, True)

    def test_all_false(self) -> None:
        result = _parse_bool_tuple("(False, False)", 2)
        assert result == (False, False)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 2 elements"):
            _parse_bool_tuple("(True, False, True)", 2)

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 'True' or 'False'"):
            _parse_bool_tuple("(True, Maybe)", 2)

    def test_extra_whitespace(self) -> None:
        result = _parse_bool_tuple("  ( True ,  False )  ", 2)
        assert result == (True, False)


# ═══════════════════════════════════════════════════════════════════
# to_dict / from_dict ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════


class TestDictRoundtrip:
    """Test to_dict → from_dict roundtrip fidelity."""

    def test_empty_network(self) -> None:
        net = SLNetwork(name="empty")
        d = net.to_dict()
        restored = SLNetwork.from_dict(d)
        assert restored.name == "empty"
        assert restored.node_count() == 0
        assert restored.edge_count() == 0

    def test_single_node(self) -> None:
        op = Opinion(0.7, 0.2, 0.1, base_rate=0.6)
        net = SLNetwork(name="single")
        net.add_node(SLNode("X", op, label="test node", metadata={"k": "v"}))

        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.node_count() == 1
        node = restored.get_node("X")
        _assert_opinions_equal(node.opinion, op, "single node opinion")
        assert node.label == "test node"
        assert node.metadata == {"k": "v"}

    def test_linear_chain(self) -> None:
        """A → B → C: nodes, edges, and opinions preserved."""
        op_a = Opinion(0.8, 0.1, 0.1)
        op_b = Opinion(0.5, 0.2, 0.3)
        op_c = Opinion(0.3, 0.4, 0.3)
        cond_ab = Opinion(0.9, 0.05, 0.05)
        cond_bc = Opinion(0.7, 0.15, 0.15)

        net = SLNetwork(name="chain")
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", op_b))
        net.add_node(SLNode("C", op_c))
        net.add_edge(SLEdge("A", "B", conditional=cond_ab))
        net.add_edge(SLEdge("B", "C", conditional=cond_bc))

        restored = SLNetwork.from_dict(net.to_dict())

        assert restored.node_count() == 3
        assert restored.edge_count() == 2
        assert restored.has_edge("A", "B")
        assert restored.has_edge("B", "C")

        _assert_opinions_equal(
            restored.get_node("A").opinion, op_a, "A opinion"
        )
        _assert_opinions_equal(
            restored.get_edge("A", "B").conditional, cond_ab, "A→B cond"
        )

    def test_explicit_counterfactual_preserved(self) -> None:
        op = Opinion(0.7, 0.2, 0.1)
        cond = Opinion(0.9, 0.05, 0.05)
        cf = Opinion(0.2, 0.5, 0.3, base_rate=0.4)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_edge(SLEdge("A", "B", conditional=cond, counterfactual=cf))

        restored = SLNetwork.from_dict(net.to_dict())
        edge = restored.get_edge("A", "B")
        assert edge.counterfactual is not None
        _assert_opinions_equal(edge.counterfactual, cf, "counterfactual")

    def test_none_counterfactual_preserved(self) -> None:
        op = Opinion(0.7, 0.2, 0.1)
        cond = Opinion(0.9, 0.05, 0.05)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        restored = SLNetwork.from_dict(net.to_dict())
        edge = restored.get_edge("A", "B")
        assert edge.counterfactual is None

    def test_multi_parent_edge(self) -> None:
        """MultiParentEdge survives roundtrip with all 2^k conditionals."""
        op = Opinion(0.5, 0.3, 0.2)
        cond_tt = Opinion(0.9, 0.05, 0.05)
        cond_tf = Opinion(0.6, 0.2, 0.2)
        cond_ft = Opinion(0.5, 0.3, 0.2)
        cond_ff = Opinion(0.1, 0.7, 0.2)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_node(SLNode("Y", op))

        conditionals = {
            (True, True): cond_tt,
            (True, False): cond_tf,
            (False, True): cond_ft,
            (False, False): cond_ff,
        }
        mpe = MultiParentEdge(
            target_id="Y", parent_ids=("A", "B"), conditionals=conditionals
        )
        net.add_edge(mpe)

        restored = SLNetwork.from_dict(net.to_dict())
        restored_mpe = restored.get_multi_parent_edge("Y")

        assert restored_mpe.parent_ids == ("A", "B")
        assert len(restored_mpe.conditionals) == 4
        _assert_opinions_equal(
            restored_mpe.conditionals[(True, True)], cond_tt, "MPE (T,T)"
        )
        _assert_opinions_equal(
            restored_mpe.conditionals[(False, False)], cond_ff, "MPE (F,F)"
        )

    def test_diamond_dag(self) -> None:
        """Diamond DAG survives roundtrip."""
        op = Opinion(0.6, 0.2, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork(name="diamond")
        for nid in "ABCD":
            net.add_node(SLNode(nid, op))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        net.add_edge(SLEdge("A", "C", conditional=cond))
        net.add_edge(SLEdge("B", "D", conditional=cond))
        net.add_edge(SLEdge("C", "D", conditional=cond))

        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.node_count() == 4
        assert restored.edge_count() == 4
        assert restored.name == "diamond"
        assert not restored.is_tree()
        assert restored.is_dag()

    def test_metadata_preserved(self) -> None:
        op = Opinion(0.7, 0.2, 0.1)
        meta = {"source": "http://example.org", "version": 3, "nested": {"a": 1}}

        net = SLNetwork()
        net.add_node(SLNode("X", op, metadata=meta))
        net.add_node(SLNode("Y", op))
        net.add_edge(SLEdge(
            "X", "Y", conditional=op, metadata={"weight": 0.95}
        ))

        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.get_node("X").metadata == meta
        assert restored.get_edge("X", "Y").metadata == {"weight": 0.95}

    def test_json_serializable(self) -> None:
        """to_dict output is JSON-serializable."""
        op = Opinion(0.7, 0.2, 0.1)
        net = SLNetwork(name="json_test")
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_edge(SLEdge("A", "B", conditional=op))

        d = net.to_dict()
        json_str = json.dumps(d)  # Should not raise
        parsed = json.loads(json_str)
        restored = SLNetwork.from_dict(parsed)
        assert restored.node_count() == 2

    def test_name_none_preserved(self) -> None:
        net = SLNetwork()  # name=None
        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.name is None


# ═══════════════════════════════════════════════════════════════════
# to_jsonld / from_jsonld ROUNDTRIP
# ═══════════════════════════════════════════════════════════════════


class TestJsonLdRoundtrip:
    """Test to_jsonld → from_jsonld roundtrip fidelity."""

    def test_empty_network(self) -> None:
        net = SLNetwork(name="empty")
        jld = net.to_jsonld()
        assert jld["@type"] == "SLNetwork"
        assert "@context" in jld

        restored = SLNetwork.from_jsonld(jld)
        assert restored.name == "empty"
        assert restored.node_count() == 0

    def test_linear_chain(self) -> None:
        op_a = Opinion(0.8, 0.1, 0.1)
        op_b = Opinion(0.5, 0.2, 0.3)
        cond = Opinion(0.9, 0.05, 0.05)

        net = SLNetwork(name="chain")
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", op_b))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        restored = SLNetwork.from_jsonld(net.to_jsonld())

        assert restored.node_count() == 2
        assert restored.edge_count() == 1
        _assert_opinions_equal(
            restored.get_node("A").opinion, op_a, "jsonld A"
        )
        _assert_opinions_equal(
            restored.get_edge("A", "B").conditional, cond, "jsonld cond"
        )

    def test_jsonld_has_context(self) -> None:
        net = SLNetwork()
        net.add_node(SLNode("X", Opinion(0.5, 0.3, 0.2)))
        jld = net.to_jsonld()
        assert "@context" in jld
        assert "slnet" in jld["@context"]

    def test_jsonld_node_has_type(self) -> None:
        net = SLNetwork()
        net.add_node(SLNode("X", Opinion(0.5, 0.3, 0.2)))
        jld = net.to_jsonld()
        assert jld["nodes"][0]["@type"] == "SLNode"

    def test_explicit_counterfactual(self) -> None:
        op = Opinion(0.7, 0.2, 0.1)
        cond = Opinion(0.9, 0.05, 0.05)
        cf = Opinion(0.1, 0.6, 0.3, base_rate=0.3)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_edge(SLEdge("A", "B", conditional=cond, counterfactual=cf))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        edge = restored.get_edge("A", "B")
        assert edge.counterfactual is not None
        _assert_opinions_equal(edge.counterfactual, cf, "jsonld cf")

    def test_multi_parent_edge(self) -> None:
        op = Opinion(0.5, 0.3, 0.2)
        conds = {
            (True, True): Opinion(0.9, 0.05, 0.05),
            (True, False): Opinion(0.6, 0.2, 0.2),
            (False, True): Opinion(0.5, 0.3, 0.2),
            (False, False): Opinion(0.1, 0.7, 0.2),
        }

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_node(SLNode("Y", op))
        net.add_edge(MultiParentEdge(
            target_id="Y", parent_ids=("A", "B"), conditionals=conds
        ))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        mpe = restored.get_multi_parent_edge("Y")
        assert mpe.parent_ids == ("A", "B")
        _assert_opinions_equal(
            mpe.conditionals[(True, True)],
            conds[(True, True)],
            "jsonld MPE (T,T)",
        )

    def test_jsonld_is_json_serializable(self) -> None:
        """to_jsonld output is JSON-serializable."""
        op = Opinion(0.7, 0.2, 0.1)
        net = SLNetwork(name="json_test")
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", op))
        net.add_edge(SLEdge("A", "B", conditional=op))

        jld = net.to_jsonld()
        json_str = json.dumps(jld, indent=2)  # Should not raise
        parsed = json.loads(json_str)
        restored = SLNetwork.from_jsonld(parsed)
        assert restored.node_count() == 2

    def test_no_multi_parent_edges_serializes_none(self) -> None:
        """When no MPEs exist, multiParentEdges is None."""
        net = SLNetwork()
        net.add_node(SLNode("X", Opinion(0.5, 0.3, 0.2)))
        jld = net.to_jsonld()
        assert jld["multiParentEdges"] is None


# ═══════════════════════════════════════════════════════════════════
# INFERENCE EQUIVALENCE
# ═══════════════════════════════════════════════════════════════════


class TestInferenceEquivalence:
    """Inference on a deserialized network produces the same result
    as inference on the original."""

    def test_dict_roundtrip_inference_tree(self) -> None:
        op_a = Opinion(0.7, 0.1, 0.2)
        cond_ab = Opinion(0.9, 0.05, 0.05)
        cond_bc = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond_ab))
        net.add_edge(SLEdge("B", "C", conditional=cond_bc))

        original = infer_node(net, "C")
        restored = SLNetwork.from_dict(net.to_dict())
        roundtripped = infer_node(restored, "C")

        _assert_opinions_equal(
            roundtripped.opinion, original.opinion, "dict tree inference"
        )

    def test_jsonld_roundtrip_inference_tree(self) -> None:
        op_a = Opinion(0.7, 0.1, 0.2)
        cond = Opinion(0.9, 0.05, 0.05)

        net = SLNetwork()
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        original = infer_node(net, "B")
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        roundtripped = infer_node(restored, "B")

        _assert_opinions_equal(
            roundtripped.opinion, original.opinion, "jsonld tree inference"
        )

    def test_dict_roundtrip_inference_dag(self) -> None:
        op = Opinion(0.6, 0.2, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        for nid in "ABCD":
            net.add_node(SLNode(nid, op))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        net.add_edge(SLEdge("A", "C", conditional=cond))
        net.add_edge(SLEdge("B", "D", conditional=cond))
        net.add_edge(SLEdge("C", "D", conditional=cond))

        original = infer_node(net, "D", method="approximate")
        restored = SLNetwork.from_dict(net.to_dict())
        roundtripped = infer_node(restored, "D", method="approximate")

        _assert_opinions_equal(
            roundtripped.opinion, original.opinion, "dict DAG inference"
        )

    def test_jsonld_roundtrip_inference_dag(self) -> None:
        op = Opinion(0.6, 0.2, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        for nid in "ABCD":
            net.add_node(SLNode(nid, op))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        net.add_edge(SLEdge("A", "C", conditional=cond))
        net.add_edge(SLEdge("B", "D", conditional=cond))
        net.add_edge(SLEdge("C", "D", conditional=cond))

        original = infer_node(net, "D", method="approximate")
        restored = SLNetwork.from_jsonld(net.to_jsonld())
        roundtripped = infer_node(restored, "D", method="approximate")

        _assert_opinions_equal(
            roundtripped.opinion, original.opinion, "jsonld DAG inference"
        )
