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


# ═══════════════════════════════════════════════════════════════════
# Gap 3: Multinomial Component Serialization
# ═══════════════════════════════════════════════════════════════════


def _assert_multinomial_opinions_equal(
    a: "MultinomialOpinion",
    b: "MultinomialOpinion",
    label: str = "",
) -> None:
    """Assert two MultinomialOpinions are equal within tolerance."""
    from jsonld_ex.multinomial_algebra import MultinomialOpinion

    prefix = f"[{label}] " if label else ""
    assert a.domain == b.domain, f"{prefix}domain mismatch: {a.domain} vs {b.domain}"
    assert abs(a.uncertainty - b.uncertainty) < _TOL, f"{prefix}uncertainty"
    for x in a.domain:
        assert abs(a.beliefs[x] - b.beliefs[x]) < _TOL, f"{prefix}belief[{x}]"
        assert abs(a.base_rates[x] - b.base_rates[x]) < _TOL, f"{prefix}base_rate[{x}]"


class TestDictRoundtripMultinomialNode:
    """Gap 3a: SLNode.multinomial_opinion must survive to_dict/from_dict."""

    def test_node_with_multinomial_opinion_dict_roundtrip(self) -> None:
        """A node carrying a multinomial_opinion survives dict roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        mop = MultinomialOpinion(
            beliefs={"H": 0.4, "M": 0.2, "L": 0.1},
            uncertainty=0.3,
            base_rates={"H": 1 / 3, "M": 1 / 3, "L": 1 / 3},
        )
        node = SLNode(
            node_id="weather",
            opinion=Opinion(0.4, 0.3, 0.3),
            multinomial_opinion=mop,
        )
        net = SLNetwork()
        net.add_node(node)

        restored = SLNetwork.from_dict(net.to_dict())
        rn = restored.get_node("weather")

        assert rn.multinomial_opinion is not None
        _assert_multinomial_opinions_equal(
            rn.multinomial_opinion, mop, "dict node multinomial"
        )

    def test_node_without_multinomial_opinion_dict_roundtrip(self) -> None:
        """A node without multinomial_opinion stays None after roundtrip."""
        node = SLNode(node_id="X", opinion=Opinion(0.5, 0.3, 0.2))
        net = SLNetwork()
        net.add_node(node)

        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.get_node("X").multinomial_opinion is None

    def test_mixed_nodes_dict_roundtrip(self) -> None:
        """Network with both binomial-only and multinomial nodes."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        mop = MultinomialOpinion(
            beliefs={"a": 0.3, "b": 0.3},
            uncertainty=0.4,
            base_rates={"a": 0.5, "b": 0.5},
        )
        net = SLNetwork()
        net.add_node(SLNode("binary_only", Opinion(0.6, 0.2, 0.2)))
        net.add_node(SLNode("with_multi", Opinion(0.3, 0.3, 0.4), multinomial_opinion=mop))

        restored = SLNetwork.from_dict(net.to_dict())
        assert restored.get_node("binary_only").multinomial_opinion is None
        assert restored.get_node("with_multi").multinomial_opinion is not None
        _assert_multinomial_opinions_equal(
            restored.get_node("with_multi").multinomial_opinion, mop,
        )


class TestDictRoundtripMultinomialEdge:
    """Gap 3b: MultinomialEdge must survive to_dict/from_dict."""

    def test_multinomial_edge_dict_roundtrip(self) -> None:
        """A simple multinomial edge survives dict roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        mop_parent = MultinomialOpinion(
            beliefs={"sunny": 0.5, "cloudy": 0.2, "rainy": 0.0},
            uncertainty=0.3,
            base_rates={"sunny": 1 / 3, "cloudy": 1 / 3, "rainy": 1 / 3},
        )
        cond_sunny = MultinomialOpinion(
            beliefs={"H": 0.7, "L": 0.1}, uncertainty=0.2,
            base_rates={"H": 0.5, "L": 0.5},
        )
        cond_cloudy = MultinomialOpinion(
            beliefs={"H": 0.3, "L": 0.3}, uncertainty=0.4,
            base_rates={"H": 0.5, "L": 0.5},
        )
        cond_rainy = MultinomialOpinion(
            beliefs={"H": 0.1, "L": 0.6}, uncertainty=0.3,
            base_rates={"H": 0.5, "L": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("W", Opinion(0.5, 0.2, 0.3), multinomial_opinion=mop_parent))
        net.add_node(SLNode("Y", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultinomialEdge(
            source_id="W",
            target_id="Y",
            conditionals={"sunny": cond_sunny, "cloudy": cond_cloudy, "rainy": cond_rainy},
        ))

        d = net.to_dict()
        assert "multinomial_edges" in d

        restored = SLNetwork.from_dict(d)
        assert restored.has_multinomial_edge("W", "Y")
        re = restored.get_multinomial_edge("W", "Y")

        _assert_multinomial_opinions_equal(
            re.conditionals["sunny"], cond_sunny, "sunny conditional"
        )
        _assert_multinomial_opinions_equal(
            re.conditionals["cloudy"], cond_cloudy, "cloudy conditional"
        )
        _assert_multinomial_opinions_equal(
            re.conditionals["rainy"], cond_rainy, "rainy conditional"
        )

    def test_multinomial_edge_temporal_fields_dict_roundtrip(self) -> None:
        """Temporal fields on MultinomialEdge survive dict roundtrip."""
        from datetime import datetime, timezone
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        ts = datetime(2025, 6, 1, tzinfo=timezone.utc)
        vf = datetime(2025, 1, 1, tzinfo=timezone.utc)
        vu = datetime(2025, 12, 31, tzinfo=timezone.utc)

        cond_a = MultinomialOpinion(
            beliefs={"y1": 0.5, "y2": 0.3}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"y1": 0.2, "y2": 0.4}, uncertainty=0.4,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("B", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultinomialEdge(
            source_id="A", target_id="B",
            conditionals={"a": cond_a, "b": cond_b},
            timestamp=ts, half_life=86400.0,
            valid_from=vf, valid_until=vu,
        ))

        restored = SLNetwork.from_dict(net.to_dict())
        re = restored.get_multinomial_edge("A", "B")

        assert re.timestamp == ts
        assert re.half_life == 86400.0
        assert re.valid_from == vf
        assert re.valid_until == vu


class TestDictRoundtripMultiParentMultinomialEdge:
    """Gap 3c: MultiParentMultinomialEdge must survive to_dict/from_dict."""

    def test_multi_parent_multinomial_edge_dict_roundtrip(self) -> None:
        """A MultiParentMultinomialEdge survives dict roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultiParentMultinomialEdge

        cond_aa = MultinomialOpinion(
            beliefs={"y1": 0.7, "y2": 0.1}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_ab = MultinomialOpinion(
            beliefs={"y1": 0.4, "y2": 0.3}, uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_ba = MultinomialOpinion(
            beliefs={"y1": 0.2, "y2": 0.5}, uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_bb = MultinomialOpinion(
            beliefs={"y1": 0.1, "y2": 0.4}, uncertainty=0.5,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("P1", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("P2", Opinion(0.4, 0.3, 0.3)))
        net.add_node(SLNode("C", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultiParentMultinomialEdge(
            parent_ids=("P1", "P2"),
            target_id="C",
            conditionals={
                ("a", "a"): cond_aa,
                ("a", "b"): cond_ab,
                ("b", "a"): cond_ba,
                ("b", "b"): cond_bb,
            },
        ))

        d = net.to_dict()
        assert "multi_parent_multinomial_edges" in d

        restored = SLNetwork.from_dict(d)
        assert restored.has_multi_parent_multinomial_edge("C")
        rmpe = restored.get_multi_parent_multinomial_edge("C")

        assert rmpe.parent_ids == ("P1", "P2")
        _assert_multinomial_opinions_equal(
            rmpe.conditionals[("a", "a")], cond_aa, "cond (a,a)"
        )
        _assert_multinomial_opinions_equal(
            rmpe.conditionals[("b", "b")], cond_bb, "cond (b,b)"
        )

    def test_multi_parent_multinomial_edge_temporal_fields(self) -> None:
        """Temporal fields on MultiParentMultinomialEdge survive roundtrip."""
        from datetime import datetime, timezone
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultiParentMultinomialEdge

        ts = datetime(2025, 7, 1, tzinfo=timezone.utc)

        cond = MultinomialOpinion(
            beliefs={"y1": 0.5, "y2": 0.3}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("P", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("C", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultiParentMultinomialEdge(
            parent_ids=("P",),
            target_id="C",
            conditionals={("a",): cond, ("b",): cond},
            timestamp=ts, half_life=3600.0,
        ))

        restored = SLNetwork.from_dict(net.to_dict())
        rmpe = restored.get_multi_parent_multinomial_edge("C")
        assert rmpe.timestamp == ts
        assert rmpe.half_life == 3600.0


class TestJsonLdRoundtripMultinomialNode:
    """Gap 3a: SLNode.multinomial_opinion must survive to_jsonld/from_jsonld."""

    def test_node_with_multinomial_opinion_jsonld_roundtrip(self) -> None:
        """A node carrying a multinomial_opinion survives JSON-LD roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        mop = MultinomialOpinion(
            beliefs={"H": 0.4, "M": 0.2, "L": 0.1},
            uncertainty=0.3,
            base_rates={"H": 1 / 3, "M": 1 / 3, "L": 1 / 3},
        )
        net = SLNetwork()
        net.add_node(SLNode("weather", Opinion(0.4, 0.3, 0.3), multinomial_opinion=mop))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        rn = restored.get_node("weather")

        assert rn.multinomial_opinion is not None
        _assert_multinomial_opinions_equal(
            rn.multinomial_opinion, mop, "jsonld node multinomial"
        )

    def test_node_without_multinomial_opinion_jsonld_roundtrip(self) -> None:
        """A node without multinomial_opinion stays None after JSON-LD roundtrip."""
        net = SLNetwork()
        net.add_node(SLNode("X", Opinion(0.5, 0.3, 0.2)))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.get_node("X").multinomial_opinion is None


class TestJsonLdRoundtripMultinomialEdge:
    """Gap 3b: MultinomialEdge must survive to_jsonld/from_jsonld."""

    def test_multinomial_edge_jsonld_roundtrip(self) -> None:
        """A MultinomialEdge survives JSON-LD roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        cond_a = MultinomialOpinion(
            beliefs={"y1": 0.6, "y2": 0.2}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"y1": 0.1, "y2": 0.5}, uncertainty=0.4,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("B", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultinomialEdge(
            source_id="A", target_id="B",
            conditionals={"a": cond_a, "b": cond_b},
        ))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.has_multinomial_edge("A", "B")
        re = restored.get_multinomial_edge("A", "B")

        _assert_multinomial_opinions_equal(
            re.conditionals["a"], cond_a, "jsonld cond a"
        )
        _assert_multinomial_opinions_equal(
            re.conditionals["b"], cond_b, "jsonld cond b"
        )


class TestJsonLdRoundtripMultiParentMultinomialEdge:
    """Gap 3c: MultiParentMultinomialEdge must survive to_jsonld/from_jsonld."""

    def test_multi_parent_multinomial_edge_jsonld_roundtrip(self) -> None:
        """A MultiParentMultinomialEdge survives JSON-LD roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultiParentMultinomialEdge

        cond_aa = MultinomialOpinion(
            beliefs={"y1": 0.7, "y2": 0.1}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_ab = MultinomialOpinion(
            beliefs={"y1": 0.3, "y2": 0.4}, uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork()
        net.add_node(SLNode("P1", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("P2", Opinion(0.4, 0.3, 0.3)))
        net.add_node(SLNode("C", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(MultiParentMultinomialEdge(
            parent_ids=("P1", "P2"),
            target_id="C",
            conditionals={
                ("a", "a"): cond_aa,
                ("a", "b"): cond_ab,
            },
        ))

        restored = SLNetwork.from_jsonld(net.to_jsonld())
        assert restored.has_multi_parent_multinomial_edge("C")
        rmpe = restored.get_multi_parent_multinomial_edge("C")

        assert rmpe.parent_ids == ("P1", "P2")
        _assert_multinomial_opinions_equal(
            rmpe.conditionals[("a", "a")], cond_aa, "jsonld cond (a,a)"
        )
        _assert_multinomial_opinions_equal(
            rmpe.conditionals[("a", "b")], cond_ab, "jsonld cond (a,b)"
        )


class TestMixedBinaryMultinomialNetworkRoundtrip:
    """Round-trip a network with both binary and multinomial components."""

    def test_mixed_network_dict_roundtrip(self) -> None:
        """Network with SLEdge + MultinomialEdge + binary nodes + multinomial nodes."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        mop = MultinomialOpinion(
            beliefs={"H": 0.5, "M": 0.2, "L": 0.0},
            uncertainty=0.3,
            base_rates={"H": 1 / 3, "M": 1 / 3, "L": 1 / 3},
        )

        cond_h = MultinomialOpinion(
            beliefs={"y1": 0.8, "y2": 0.1}, uncertainty=0.1,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_m = MultinomialOpinion(
            beliefs={"y1": 0.4, "y2": 0.3}, uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_l = MultinomialOpinion(
            beliefs={"y1": 0.1, "y2": 0.6}, uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork(name="mixed")
        # Binary-only node
        net.add_node(SLNode("A", Opinion(0.6, 0.2, 0.2)))
        # Multinomial node
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3), multinomial_opinion=mop))
        # Target node
        net.add_node(SLNode("C", Opinion(0.3, 0.3, 0.4)))

        # Binary edge A→C
        net.add_edge(SLEdge("A", "C", conditional=Opinion(0.8, 0.1, 0.1)))
        # Multinomial edge B→C
        net.add_edge(MultinomialEdge(
            source_id="B", target_id="C",
            conditionals={"H": cond_h, "M": cond_m, "L": cond_l},
        ))

        restored = SLNetwork.from_dict(net.to_dict())

        assert restored.name == "mixed"
        assert len(restored) == 3
        assert restored.get_node("A").multinomial_opinion is None
        assert restored.get_node("B").multinomial_opinion is not None
        assert restored.has_edge("A", "C")
        assert restored.has_multinomial_edge("B", "C")

    def test_mixed_network_jsonld_roundtrip(self) -> None:
        """Same mixed network survives JSON-LD roundtrip."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.sl_network.types import MultinomialEdge

        mop = MultinomialOpinion(
            beliefs={"a": 0.3, "b": 0.3},
            uncertainty=0.4,
            base_rates={"a": 0.5, "b": 0.5},
        )
        cond_a = MultinomialOpinion(
            beliefs={"y1": 0.6, "y2": 0.2}, uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"y1": 0.1, "y2": 0.5}, uncertainty=0.4,
            base_rates={"y1": 0.5, "y2": 0.5},
        )

        net = SLNetwork(name="mixed_jsonld")
        net.add_node(SLNode("X", Opinion(0.5, 0.3, 0.2)))
        net.add_node(SLNode("Y", Opinion(0.4, 0.2, 0.4), multinomial_opinion=mop))
        net.add_node(SLNode("Z", Opinion(0.3, 0.3, 0.4)))
        net.add_edge(SLEdge("X", "Z", conditional=Opinion(0.7, 0.2, 0.1)))
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="Z",
            conditionals={"a": cond_a, "b": cond_b},
        ))

        restored = SLNetwork.from_jsonld(net.to_jsonld())

        assert restored.name == "mixed_jsonld"
        assert restored.get_node("Y").multinomial_opinion is not None
        assert restored.has_multinomial_edge("Y", "Z")
