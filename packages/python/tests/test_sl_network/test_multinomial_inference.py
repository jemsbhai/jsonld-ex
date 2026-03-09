"""
Tests for multinomial inference dispatch (Phase B, Step 3b).

Extends the inference engine to handle MultinomialEdge edges by
dispatching to ``multinomial_deduce()`` when:
    1. The parent node has a ``multinomial_opinion`` (is_multinomial).
    2. The edge connecting parent → child is a ``MultinomialEdge``.

Design contract:
    - Multinomial inference results are stored in
      ``InferenceResult.multinomial_intermediate_opinions``.
    - Binomial ``intermediate_opinions`` still contains each node's
      binomial ``opinion`` (backward compatible).
    - Existing binary inference is completely unaffected.
    - Mathematical correctness: results match direct calls to
      ``multinomial_deduce()``.

TDD: RED phase — all tests should FAIL until inference.py and
types.py are updated.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import (
    MultinomialOpinion,
    coarsen,
    multinomial_deduce,
    multinomial_cumulative_fuse,
)
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    InferenceResult,
    MultinomialEdge,
    SLEdge,
    SLNode,
)
from jsonld_ex.sl_network.inference import infer_node, infer_all


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _vacuous_binomial() -> Opinion:
    return Opinion(0.0, 0.0, 1.0)


def _make_ternary_opinion() -> MultinomialOpinion:
    """Parent opinion over {A, B, C}."""
    return MultinomialOpinion(
        beliefs={"A": 0.5, "B": 0.2, "C": 0.1},
        uncertainty=0.2,
        base_rates={"A": 0.5, "B": 0.3, "C": 0.2},
    )


def _make_binary_multinomial() -> MultinomialOpinion:
    """Binary multinomial opinion over {H, L}."""
    return MultinomialOpinion(
        beliefs={"H": 0.6, "L": 0.2},
        uncertainty=0.2,
        base_rates={"H": 0.5, "L": 0.5},
    )


def _make_conditionals_ternary_to_binary() -> dict[str, MultinomialOpinion]:
    """Conditional table: ternary parent {A,B,C} → binary child {H,L}."""
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


def _make_conditionals_binary_to_binary() -> dict[str, MultinomialOpinion]:
    """Conditional table: binary parent {H,L} → binary child {P,Q}."""
    br = {"P": 0.5, "Q": 0.5}
    return {
        "H": MultinomialOpinion(
            beliefs={"P": 0.8, "Q": 0.1}, uncertainty=0.1, base_rates=br,
        ),
        "L": MultinomialOpinion(
            beliefs={"P": 0.1, "Q": 0.7}, uncertainty=0.2, base_rates=br,
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# INFERENCE RESULT EXTENSION
# ═══════════════════════════════════════════════════════════════════


class TestInferenceResultMultinomialField:
    """InferenceResult has an optional multinomial_intermediate_opinions field."""

    def test_default_is_empty_dict(self) -> None:
        """Without multinomial nodes, the field defaults to empty dict."""
        result = InferenceResult(
            query_node="X",
            opinion=Opinion(0.5, 0.3, 0.2),
            steps=[],
            intermediate_opinions={},
            topological_order=[],
        )
        assert result.multinomial_intermediate_opinions == {}

    def test_backward_compatible_construction(self) -> None:
        """Existing construction without the new field still works."""
        result = InferenceResult(
            query_node="X",
            opinion=Opinion(0.5, 0.3, 0.2),
            steps=[],
            intermediate_opinions={"X": Opinion(0.5, 0.3, 0.2)},
            topological_order=["X"],
        )
        assert result.query_node == "X"
        assert result.multinomial_intermediate_opinions == {}

    def test_explicit_multinomial_opinions(self) -> None:
        """Can construct with explicit multinomial_intermediate_opinions."""
        multi_op = _make_ternary_opinion()
        result = InferenceResult(
            query_node="X",
            opinion=Opinion(0.5, 0.3, 0.2),
            steps=[],
            intermediate_opinions={},
            topological_order=[],
            multinomial_intermediate_opinions={"X": multi_op},
        )
        assert "X" in result.multinomial_intermediate_opinions
        assert result.multinomial_intermediate_opinions["X"] is multi_op


# ═══════════════════════════════════════════════════════════════════
# MULTINOMIAL ROOT PASSTHROUGH
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialRootPassthrough:
    """Root nodes with multinomial_opinion pass it through to results."""

    def test_single_multinomial_root(self) -> None:
        """A single multinomial root node's opinion is passed through."""
        multi_op = _make_ternary_opinion()
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X",
            opinion=_vacuous_binomial(),
            multinomial_opinion=multi_op,
        ))

        result = infer_node(net, "X")
        assert "X" in result.multinomial_intermediate_opinions
        assert result.multinomial_intermediate_opinions["X"] is multi_op

    def test_binary_root_has_no_multinomial(self) -> None:
        """A binary root node has no entry in multinomial_intermediate_opinions."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="X", opinion=Opinion(0.7, 0.2, 0.1)))

        result = infer_node(net, "X")
        assert "X" not in result.multinomial_intermediate_opinions

    def test_mixed_roots_only_multinomial_stored(self) -> None:
        """Only multinomial roots appear in multinomial_intermediate_opinions."""
        multi_op = _make_ternary_opinion()
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X",
            opinion=_vacuous_binomial(),
            multinomial_opinion=multi_op,
        ))
        net.add_node(SLNode(node_id="Y", opinion=Opinion(0.7, 0.2, 0.1)))

        result = infer_node(net, "X")
        assert "X" in result.multinomial_intermediate_opinions
        # Y may or may not be in the result depending on connectivity,
        # but it should NOT be in multinomial_intermediate_opinions
        assert "Y" not in result.multinomial_intermediate_opinions


# ═══════════════════════════════════════════════════════════════════
# SINGLE MULTINOMIAL EDGE INFERENCE
# ═══════════════════════════════════════════════════════════════════


class TestSingleMultinomialEdgeInference:
    """Inference through a single MultinomialEdge."""

    def _build_simple_network(self) -> tuple[SLNetwork, MultinomialOpinion]:
        """Build X→Y where X is ternary, edge is MultinomialEdge."""
        parent_op = _make_ternary_opinion()
        conditionals = _make_conditionals_ternary_to_binary()

        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X",
            opinion=_vacuous_binomial(),
            multinomial_opinion=parent_op,
        ))
        net.add_node(SLNode(
            node_id="Y",
            opinion=_vacuous_binomial(),
        ))
        net.add_edge(MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=conditionals,
        ))
        return net, parent_op

    def test_child_gets_multinomial_result(self) -> None:
        """Inference through MultinomialEdge produces MultinomialOpinion for child."""
        net, _ = self._build_simple_network()
        result = infer_node(net, "Y")
        assert "Y" in result.multinomial_intermediate_opinions
        child_op = result.multinomial_intermediate_opinions["Y"]
        assert isinstance(child_op, MultinomialOpinion)

    def test_child_domain_matches_conditionals(self) -> None:
        """Inferred child opinion has the child domain from the edge."""
        net, _ = self._build_simple_network()
        result = infer_node(net, "Y")
        child_op = result.multinomial_intermediate_opinions["Y"]
        assert child_op.domain == ("H", "L")

    def test_result_matches_direct_multinomial_deduce(self) -> None:
        """Inferred result matches a direct call to multinomial_deduce."""
        parent_op = _make_ternary_opinion()
        conditionals = _make_conditionals_ternary_to_binary()

        # Direct computation
        expected = multinomial_deduce(parent_op, conditionals)

        # Network inference
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X",
            opinion=_vacuous_binomial(),
            multinomial_opinion=parent_op,
        ))
        net.add_node(SLNode(
            node_id="Y",
            opinion=_vacuous_binomial(),
        ))
        net.add_edge(MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=conditionals,
        ))

        result = infer_node(net, "Y")
        actual = result.multinomial_intermediate_opinions["Y"]

        # Verify component-wise equality
        assert actual.domain == expected.domain
        assert abs(actual.uncertainty - expected.uncertainty) < 1e-9
        for state in expected.domain:
            assert abs(actual.beliefs[state] - expected.beliefs[state]) < 1e-9
            assert abs(actual.base_rates[state] - expected.base_rates[state]) < 1e-9

    def test_additivity_preserved(self) -> None:
        """Inferred opinion satisfies Σb + u = 1."""
        net, _ = self._build_simple_network()
        result = infer_node(net, "Y")
        child_op = result.multinomial_intermediate_opinions["Y"]
        total = sum(child_op.beliefs.values()) + child_op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_binomial_intermediate_still_populated(self) -> None:
        """The binomial intermediate_opinions dict is still populated for all nodes."""
        net, _ = self._build_simple_network()
        result = infer_node(net, "Y")
        # Both nodes should have binomial entries
        assert "X" in result.intermediate_opinions
        assert "Y" in result.intermediate_opinions

    def test_parent_multinomial_also_in_results(self) -> None:
        """The parent's multinomial_opinion appears in multinomial results."""
        net, parent_op = self._build_simple_network()
        result = infer_node(net, "Y")
        assert "X" in result.multinomial_intermediate_opinions
        assert result.multinomial_intermediate_opinions["X"] is parent_op


# ═══════════════════════════════════════════════════════════════════
# MULTINOMIAL CHAIN INFERENCE
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialChainInference:
    """Inference through a chain of MultinomialEdges: X→Y→Z."""

    def _build_chain_network(self) -> SLNetwork:
        """Build X→Y→Z with multinomial edges.

        X: ternary {A,B,C}
        X→Y: ternary→binary {H,L}
        Y→Z: binary {H,L}→binary {P,Q}
        """
        parent_op = _make_ternary_opinion()
        conds_xy = _make_conditionals_ternary_to_binary()
        conds_yz = _make_conditionals_binary_to_binary()

        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X",
            opinion=_vacuous_binomial(),
            multinomial_opinion=parent_op,
        ))
        net.add_node(SLNode(
            node_id="Y",
            opinion=_vacuous_binomial(),
        ))
        net.add_node(SLNode(
            node_id="Z",
            opinion=_vacuous_binomial(),
        ))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=conds_xy,
        ))
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="Z",
            conditionals=conds_yz,
        ))
        return net

    def test_chain_produces_multinomial_at_leaf(self) -> None:
        """Inference propagates through multinomial chain to leaf."""
        net = self._build_chain_network()
        result = infer_node(net, "Z")
        assert "Z" in result.multinomial_intermediate_opinions
        z_op = result.multinomial_intermediate_opinions["Z"]
        assert isinstance(z_op, MultinomialOpinion)
        assert z_op.domain == ("P", "Q")

    def test_chain_intermediate_node_also_multinomial(self) -> None:
        """Intermediate node Y also gets a multinomial result."""
        net = self._build_chain_network()
        result = infer_node(net, "Z")
        assert "Y" in result.multinomial_intermediate_opinions
        y_op = result.multinomial_intermediate_opinions["Y"]
        assert y_op.domain == ("H", "L")

    def test_chain_matches_manual_two_step_deduce(self) -> None:
        """Chain result matches two sequential multinomial_deduce calls."""
        parent_op = _make_ternary_opinion()
        conds_xy = _make_conditionals_ternary_to_binary()
        conds_yz = _make_conditionals_binary_to_binary()

        # Manual two-step computation
        y_expected = multinomial_deduce(parent_op, conds_xy)
        z_expected = multinomial_deduce(y_expected, conds_yz)

        # Network inference
        net = self._build_chain_network()
        result = infer_node(net, "Z")
        z_actual = result.multinomial_intermediate_opinions["Z"]

        # Verify
        assert z_actual.domain == z_expected.domain
        assert abs(z_actual.uncertainty - z_expected.uncertainty) < 1e-9
        for state in z_expected.domain:
            assert abs(z_actual.beliefs[state] - z_expected.beliefs[state]) < 1e-9

    def test_chain_additivity_at_every_node(self) -> None:
        """Σb + u = 1 at every multinomial node in the chain."""
        net = self._build_chain_network()
        result = infer_node(net, "Z")
        for nid, op in result.multinomial_intermediate_opinions.items():
            total = sum(op.beliefs.values()) + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"Additivity violated at node {nid!r}: Σb + u = {total}"
            )


# ═══════════════════════════════════════════════════════════════════
# BINARY INFERENCE UNAFFECTED
# ═══════════════════════════════════════════════════════════════════


class TestBinaryInferenceUnaffected:
    """Existing binary inference is completely unchanged."""

    def test_binary_network_no_multinomial_opinions(self) -> None:
        """A purely binary network has empty multinomial_intermediate_opinions."""
        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.8, 0.1, 0.1)))
        net.add_node(SLNode("B", Opinion(0.0, 0.0, 1.0)))
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=Opinion(0.9, 0.05, 0.05),
        ))

        result = infer_node(net, "B")
        assert result.multinomial_intermediate_opinions == {}

    def test_binary_result_values_unchanged(self) -> None:
        """Binary inference produces the same opinion values as before."""
        from jsonld_ex.confidence_algebra import deduce

        parent_op = Opinion(0.8, 0.1, 0.1)
        conditional = Opinion(0.9, 0.05, 0.05)
        counterfactual = Opinion(0.0, 0.0, 1.0)  # vacuous

        expected = deduce(parent_op, conditional, counterfactual)

        net = SLNetwork()
        net.add_node(SLNode("A", parent_op))
        net.add_node(SLNode("B", Opinion(0.0, 0.0, 1.0)))
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=conditional,
        ))

        result = infer_node(net, "B", counterfactual_fn="vacuous")
        assert abs(result.opinion.belief - expected.belief) < 1e-9
        assert abs(result.opinion.disbelief - expected.disbelief) < 1e-9
        assert abs(result.opinion.uncertainty - expected.uncertainty) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# MIXED BINARY + MULTINOMIAL NETWORKS
# ═══════════════════════════════════════════════════════════════════


class TestMixedBinaryMultinomialNetwork:
    """Networks with both binary and multinomial edges."""

    def test_binary_branch_unaffected_by_multinomial_branch(self) -> None:
        """Binary and multinomial branches coexist independently.

        Network:
            X (ternary) --MultinomialEdge--> Y (binary child {H,L})
            A (binary)  --SLEdge--> B (binary)

        Binary branch A→B should be completely unaffected.
        """
        # Multinomial branch
        net = SLNetwork()
        multi_op = _make_ternary_opinion()
        net.add_node(SLNode(
            node_id="X", opinion=_vacuous_binomial(),
            multinomial_opinion=multi_op,
        ))
        net.add_node(SLNode(node_id="Y", opinion=_vacuous_binomial()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=_make_conditionals_ternary_to_binary(),
        ))

        # Binary branch
        parent_op = Opinion(0.7, 0.2, 0.1)
        net.add_node(SLNode(node_id="A", opinion=parent_op))
        net.add_node(SLNode(node_id="B", opinion=_vacuous_binomial()))
        conditional = Opinion(0.9, 0.05, 0.05)
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=conditional,
        ))

        # Infer binary branch
        result_b = infer_node(net, "B")
        assert "B" not in result_b.multinomial_intermediate_opinions
        assert result_b.opinion.belief > 0  # Should have real inference

        # Infer multinomial branch
        result_y = infer_node(net, "Y")
        assert "Y" in result_y.multinomial_intermediate_opinions


# ═══════════════════════════════════════════════════════════════════
# INFER_ALL WITH MULTINOMIAL
# ═══════════════════════════════════════════════════════════════════


class TestInferAllMultinomial:
    """infer_all() handles multinomial nodes correctly."""

    def test_infer_all_includes_multinomial(self) -> None:
        """infer_all returns multinomial results for multinomial nodes."""
        parent_op = _make_ternary_opinion()
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X", opinion=_vacuous_binomial(),
            multinomial_opinion=parent_op,
        ))
        net.add_node(SLNode(node_id="Y", opinion=_vacuous_binomial()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=_make_conditionals_ternary_to_binary(),
        ))

        results = infer_all(net)
        assert "X" in results
        assert "Y" in results
        # Y's result should contain multinomial opinions
        assert "Y" in results["Y"].multinomial_intermediate_opinions


# ═══════════════════════════════════════════════════════════════════
# DOGMATIC / VACUOUS EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialInferenceEdgeCases:
    """Edge cases for multinomial inference."""

    def test_vacuous_parent_yields_base_rate_weighted_result(self) -> None:
        """Vacuous parent (u=1) yields child beliefs weighted by base rates.

        When parent has u=1, multinomial_deduce produces:
            b_Y(y) = u_X · Σ_i a_X(x_i) · b_{Y|x_i}(y) = Σ_i a_X(x_i) · b_{Y|x_i}(y)
        """
        vacuous_parent = MultinomialOpinion(
            beliefs={"A": 0.0, "B": 0.0, "C": 0.0},
            uncertainty=1.0,
            base_rates={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        )
        conditionals = _make_conditionals_ternary_to_binary()

        expected = multinomial_deduce(vacuous_parent, conditionals)

        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X", opinion=_vacuous_binomial(),
            multinomial_opinion=vacuous_parent,
        ))
        net.add_node(SLNode(node_id="Y", opinion=_vacuous_binomial()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=conditionals,
        ))

        result = infer_node(net, "Y")
        actual = result.multinomial_intermediate_opinions["Y"]
        assert abs(actual.uncertainty - expected.uncertainty) < 1e-9
        for state in expected.domain:
            assert abs(actual.beliefs[state] - expected.beliefs[state]) < 1e-9

    def test_dogmatic_parent_yields_deterministic_conditional(self) -> None:
        """Dogmatic parent (b(A)=1, u=0) selects conditional A exactly.

        When parent is dogmatic for state A:
            b_Y(y) = b_X(A) · b_{Y|A}(y) = b_{Y|A}(y)
            u_Y = b_X(A) · u_{Y|A} = u_{Y|A}
        """
        dogmatic_parent = MultinomialOpinion(
            beliefs={"A": 1.0, "B": 0.0, "C": 0.0},
            uncertainty=0.0,
            base_rates={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
        )
        conditionals = _make_conditionals_ternary_to_binary()
        expected_cond = conditionals["A"]

        net = SLNetwork()
        net.add_node(SLNode(
            node_id="X", opinion=_vacuous_binomial(),
            multinomial_opinion=dogmatic_parent,
        ))
        net.add_node(SLNode(node_id="Y", opinion=_vacuous_binomial()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=conditionals,
        ))

        result = infer_node(net, "Y")
        actual = result.multinomial_intermediate_opinions["Y"]

        # With dogmatic parent at A, the child beliefs should match
        # the conditional for A exactly
        for state in expected_cond.domain:
            assert abs(actual.beliefs[state] - expected_cond.beliefs[state]) < 1e-9
        assert abs(actual.uncertainty - expected_cond.uncertainty) < 1e-9
