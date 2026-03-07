"""
Tests for exact tree inference (Tier 1, Step 4).

Correctness criterion: SLNetwork tree inference must produce
results IDENTICAL to manual deduce() chains.  This is verified
by building test networks and comparing network inference against
hand-computed deduce() calls for:
    - 2-hop linear chains (A → B)
    - 3-hop linear chains (A → B → C)
    - 4-hop linear chains (A → B → C → D)
    - Forest (multiple disconnected trees)
    - Single-node (root only)
    - All three counterfactual strategies
    - Explicit counterfactual on edges

Also covers: inference trace correctness, error paths,
method selection, and the b+d+u=1 invariant at every node.

TDD: These tests define the contract for tree inference.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, deduce
from jsonld_ex.sl_network.counterfactuals import (
    adversarial_counterfactual,
    prior_counterfactual,
    vacuous_counterfactual,
)
from jsonld_ex.sl_network.inference import infer_all, infer_node
from jsonld_ex.sl_network.network import NodeNotFoundError, SLNetwork
from jsonld_ex.sl_network.types import (
    InferenceResult,
    SLEdge,
    SLNode,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

_TOL = 1e-12


def _assert_opinions_equal(actual: Opinion, expected: Opinion, label: str = "") -> None:
    """Assert two opinions are numerically identical within tolerance."""
    prefix = f"[{label}] " if label else ""
    assert abs(actual.belief - expected.belief) < _TOL, (
        f"{prefix}belief: {actual.belief} != {expected.belief}"
    )
    assert abs(actual.disbelief - expected.disbelief) < _TOL, (
        f"{prefix}disbelief: {actual.disbelief} != {expected.disbelief}"
    )
    assert abs(actual.uncertainty - expected.uncertainty) < _TOL, (
        f"{prefix}uncertainty: {actual.uncertainty} != {expected.uncertainty}"
    )
    assert abs(actual.base_rate - expected.base_rate) < _TOL, (
        f"{prefix}base_rate: {actual.base_rate} != {expected.base_rate}"
    )


def _assert_additivity(opinion: Opinion, label: str = "") -> None:
    """Assert b+d+u=1."""
    total = opinion.belief + opinion.disbelief + opinion.uncertainty
    prefix = f"[{label}] " if label else ""
    assert abs(total - 1.0) < _TOL, f"{prefix}b+d+u={total}"


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def op_root() -> Opinion:
    """Root node opinion: strong belief with some uncertainty."""
    return Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)


@pytest.fixture
def cond_ab() -> Opinion:
    """Conditional: ω_{B|A=True}."""
    return Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)


@pytest.fixture
def cond_bc() -> Opinion:
    """Conditional: ω_{C|B=True}."""
    return Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)


@pytest.fixture
def cond_cd() -> Opinion:
    """Conditional: ω_{D|C=True}."""
    return Opinion(belief=0.7, disbelief=0.15, uncertainty=0.15)


@pytest.fixture
def net_2hop(op_root: Opinion, cond_ab: Opinion) -> SLNetwork:
    """A → B (2-node chain)."""
    net = SLNetwork(name="2hop")
    net.add_node(SLNode("A", op_root))
    net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))  # Prior, overridden by inference
    net.add_edge(SLEdge("A", "B", conditional=cond_ab))
    return net


@pytest.fixture
def net_3hop(op_root: Opinion, cond_ab: Opinion, cond_bc: Opinion) -> SLNetwork:
    """A → B → C (3-node chain)."""
    net = SLNetwork(name="3hop")
    net.add_node(SLNode("A", op_root))
    net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
    net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
    net.add_edge(SLEdge("A", "B", conditional=cond_ab))
    net.add_edge(SLEdge("B", "C", conditional=cond_bc))
    return net


@pytest.fixture
def net_4hop(
    op_root: Opinion, cond_ab: Opinion, cond_bc: Opinion, cond_cd: Opinion
) -> SLNetwork:
    """A → B → C → D (4-node chain)."""
    net = SLNetwork(name="4hop")
    net.add_node(SLNode("A", op_root))
    net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
    net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
    net.add_node(SLNode("D", Opinion(0.5, 0.2, 0.3)))
    net.add_edge(SLEdge("A", "B", conditional=cond_ab))
    net.add_edge(SLEdge("B", "C", conditional=cond_bc))
    net.add_edge(SLEdge("C", "D", conditional=cond_cd))
    return net


# ═══════════════════════════════════════════════════════════════════
# 2-HOP: A → B (matches manual deduce)
# ═══════════════════════════════════════════════════════════════════


class TestTwoHopExact:
    """2-node chain: network inference must match a single deduce() call."""

    def test_matches_manual_deduce_vacuous(
        self, net_2hop: SLNetwork, op_root: Opinion, cond_ab: Opinion
    ) -> None:
        """Network inference at B == deduce(ω_A, cond_AB, vacuous(cond_AB))."""
        cf = vacuous_counterfactual(cond_ab)
        expected = deduce(op_root, cond_ab, cf)

        result = infer_node(net_2hop, "B")
        _assert_opinions_equal(result.opinion, expected, "2hop vacuous")

    def test_matches_manual_deduce_adversarial(
        self, net_2hop: SLNetwork, op_root: Opinion, cond_ab: Opinion
    ) -> None:
        cf = adversarial_counterfactual(cond_ab)
        expected = deduce(op_root, cond_ab, cf)

        result = infer_node(net_2hop, "B", counterfactual_fn="adversarial")
        _assert_opinions_equal(result.opinion, expected, "2hop adversarial")

    def test_matches_manual_deduce_prior(
        self, net_2hop: SLNetwork, op_root: Opinion, cond_ab: Opinion
    ) -> None:
        cf = prior_counterfactual(cond_ab)
        expected = deduce(op_root, cond_ab, cf)

        result = infer_node(net_2hop, "B", counterfactual_fn="prior")
        _assert_opinions_equal(result.opinion, expected, "2hop prior")

    def test_root_unchanged(self, net_2hop: SLNetwork, op_root: Opinion) -> None:
        """Querying the root returns its marginal opinion unchanged."""
        result = infer_node(net_2hop, "A")
        _assert_opinions_equal(result.opinion, op_root, "root A")

    def test_additivity_at_all_nodes(self, net_2hop: SLNetwork) -> None:
        result = infer_node(net_2hop, "B")
        for nid, op in result.intermediate_opinions.items():
            _assert_additivity(op, nid)


# ═══════════════════════════════════════════════════════════════════
# 3-HOP: A → B → C (chained deduce)
# ═══════════════════════════════════════════════════════════════════


class TestThreeHopExact:
    """3-node chain: inference at C must match two chained deduce() calls."""

    def test_matches_manual_chain(
        self,
        net_3hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
        cond_bc: Opinion,
    ) -> None:
        """ω_C = deduce(deduce(ω_A, cond_AB, cf_AB), cond_BC, cf_BC)."""
        cf_ab = vacuous_counterfactual(cond_ab)
        omega_b = deduce(op_root, cond_ab, cf_ab)

        cf_bc = vacuous_counterfactual(cond_bc)
        expected_c = deduce(omega_b, cond_bc, cf_bc)

        result = infer_node(net_3hop, "C")
        _assert_opinions_equal(result.opinion, expected_c, "3hop C")

    def test_intermediate_b_matches(
        self,
        net_3hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
    ) -> None:
        """Intermediate opinion at B matches manual deduce."""
        cf_ab = vacuous_counterfactual(cond_ab)
        expected_b = deduce(op_root, cond_ab, cf_ab)

        result = infer_node(net_3hop, "C")
        actual_b = result.intermediate_opinions["B"]
        _assert_opinions_equal(actual_b, expected_b, "intermediate B")

    def test_additivity_all_nodes(self, net_3hop: SLNetwork) -> None:
        result = infer_node(net_3hop, "C")
        for nid, op in result.intermediate_opinions.items():
            _assert_additivity(op, nid)

    def test_adversarial_3hop(
        self,
        net_3hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
        cond_bc: Opinion,
    ) -> None:
        """3-hop with adversarial counterfactual matches manual chain."""
        cf_ab = adversarial_counterfactual(cond_ab)
        omega_b = deduce(op_root, cond_ab, cf_ab)

        cf_bc = adversarial_counterfactual(cond_bc)
        expected_c = deduce(omega_b, cond_bc, cf_bc)

        result = infer_node(net_3hop, "C", counterfactual_fn="adversarial")
        _assert_opinions_equal(result.opinion, expected_c, "3hop adversarial C")


# ═══════════════════════════════════════════════════════════════════
# 4-HOP: A → B → C → D (3 chained deductions)
# ═══════════════════════════════════════════════════════════════════


class TestFourHopExact:
    """4-node chain: inference at D must match three chained deduce() calls."""

    def test_matches_manual_chain(
        self,
        net_4hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
        cond_bc: Opinion,
        cond_cd: Opinion,
    ) -> None:
        cf_ab = vacuous_counterfactual(cond_ab)
        omega_b = deduce(op_root, cond_ab, cf_ab)

        cf_bc = vacuous_counterfactual(cond_bc)
        omega_c = deduce(omega_b, cond_bc, cf_bc)

        cf_cd = vacuous_counterfactual(cond_cd)
        expected_d = deduce(omega_c, cond_cd, cf_cd)

        result = infer_node(net_4hop, "D")
        _assert_opinions_equal(result.opinion, expected_d, "4hop D")

    def test_all_intermediates_match(
        self,
        net_4hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
        cond_bc: Opinion,
        cond_cd: Opinion,
    ) -> None:
        """Every intermediate opinion matches the manual chain."""
        cf_ab = vacuous_counterfactual(cond_ab)
        omega_b = deduce(op_root, cond_ab, cf_ab)

        cf_bc = vacuous_counterfactual(cond_bc)
        omega_c = deduce(omega_b, cond_bc, cf_bc)

        result = infer_node(net_4hop, "D")
        _assert_opinions_equal(
            result.intermediate_opinions["A"], op_root, "inter A"
        )
        _assert_opinions_equal(
            result.intermediate_opinions["B"], omega_b, "inter B"
        )
        _assert_opinions_equal(
            result.intermediate_opinions["C"], omega_c, "inter C"
        )

    def test_additivity_all_nodes(self, net_4hop: SLNetwork) -> None:
        result = infer_node(net_4hop, "D")
        for nid, op in result.intermediate_opinions.items():
            _assert_additivity(op, nid)

    def test_uncertainty_trends(self, net_4hop: SLNetwork) -> None:
        """With vacuous counterfactuals, uncertainty should generally
        increase along the chain (information loss per hop).

        This is a soft check — not a mathematical guarantee for all
        parameter choices, but true for our test configuration.
        """
        result = infer_node(net_4hop, "D")
        u_a = result.intermediate_opinions["A"].uncertainty
        u_d = result.intermediate_opinions["D"].uncertainty
        # D should have more uncertainty than A due to deduction chain
        assert u_d > u_a, (
            f"Expected uncertainty increase: u_A={u_a}, u_D={u_d}"
        )


# ═══════════════════════════════════════════════════════════════════
# EXPLICIT COUNTERFACTUAL ON EDGE
# ═══════════════════════════════════════════════════════════════════


class TestExplicitCounterfactual:
    """Test edges with counterfactual explicitly set."""

    def test_explicit_overrides_strategy(self, op_root: Opinion) -> None:
        """When SLEdge.counterfactual is set, the strategy function
        is NOT called — the explicit value is used directly."""
        conditional = Opinion(0.9, 0.05, 0.05)
        explicit_cf = Opinion(0.3, 0.5, 0.2)  # Custom, not vacuous

        net = SLNetwork()
        net.add_node(SLNode("A", op_root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge(
            "A", "B",
            conditional=conditional,
            counterfactual=explicit_cf,
        ))

        # Manual computation with explicit counterfactual
        expected = deduce(op_root, conditional, explicit_cf)

        # Inference should use explicit_cf, regardless of strategy
        result_vac = infer_node(net, "B", counterfactual_fn="vacuous")
        result_adv = infer_node(net, "B", counterfactual_fn="adversarial")

        _assert_opinions_equal(result_vac.opinion, expected, "explicit+vacuous")
        _assert_opinions_equal(result_adv.opinion, expected, "explicit+adversarial")

        # Both should be identical (explicit overrides strategy)
        _assert_opinions_equal(
            result_vac.opinion, result_adv.opinion, "vac==adv with explicit"
        )


# ═══════════════════════════════════════════════════════════════════
# SINGLE NODE (ROOT ONLY)
# ═══════════════════════════════════════════════════════════════════


class TestSingleNode:
    """Inference on a single-node network (no edges)."""

    def test_root_returns_marginal(self) -> None:
        op = Opinion(0.6, 0.3, 0.1)
        net = SLNetwork()
        net.add_node(SLNode("X", op))
        result = infer_node(net, "X")
        _assert_opinions_equal(result.opinion, op, "single node")

    def test_passthrough_step(self) -> None:
        op = Opinion(0.6, 0.3, 0.1)
        net = SLNetwork()
        net.add_node(SLNode("X", op))
        result = infer_node(net, "X")
        assert len(result.steps) == 1
        assert result.steps[0].operation == "passthrough"
        assert result.steps[0].node_id == "X"


# ═══════════════════════════════════════════════════════════════════
# FOREST (multiple disconnected trees)
# ═══════════════════════════════════════════════════════════════════


class TestForest:
    """Inference on disconnected trees."""

    def test_infer_in_one_tree(self) -> None:
        """Inference on a node processes only its connected component."""
        op_a = Opinion(0.8, 0.1, 0.1)
        op_c = Opinion(0.3, 0.4, 0.3)
        cond = Opinion(0.7, 0.2, 0.1)

        net = SLNetwork()
        # Tree 1: A → B
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        # Tree 2: C (isolated)
        net.add_node(SLNode("C", op_c))

        # Infer B — should only involve A → B
        result = infer_node(net, "B")
        cf = vacuous_counterfactual(cond)
        expected = deduce(op_a, cond, cf)
        _assert_opinions_equal(result.opinion, expected, "forest B")

    def test_infer_isolated_node(self) -> None:
        """Isolated node returns its marginal."""
        op = Opinion(0.3, 0.4, 0.3)
        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.8, 0.1, 0.1)))
        net.add_node(SLNode("C", op))
        result = infer_node(net, "C")
        _assert_opinions_equal(result.opinion, op, "isolated C")


# ═══════════════════════════════════════════════════════════════════
# INFERENCE TRACE
# ═══════════════════════════════════════════════════════════════════


class TestInferenceTrace:
    """Test the inference trace (steps, topological_order, intermediate_opinions)."""

    def test_trace_step_count_3hop(self, net_3hop: SLNetwork) -> None:
        """3-node chain: 1 passthrough + 2 deductions = 3 steps."""
        result = infer_node(net_3hop, "C")
        assert len(result.steps) == 3

    def test_trace_operations(self, net_3hop: SLNetwork) -> None:
        result = infer_node(net_3hop, "C")
        ops = [s.operation for s in result.steps]
        assert ops == ["passthrough", "deduce", "deduce"]

    def test_trace_node_order(self, net_3hop: SLNetwork) -> None:
        result = infer_node(net_3hop, "C")
        nodes = [s.node_id for s in result.steps]
        assert nodes == ["A", "B", "C"]

    def test_topological_order(self, net_3hop: SLNetwork) -> None:
        result = infer_node(net_3hop, "C")
        assert result.topological_order == ["A", "B", "C"]

    def test_intermediate_opinions_keys(self, net_3hop: SLNetwork) -> None:
        result = infer_node(net_3hop, "C")
        assert set(result.intermediate_opinions.keys()) == {"A", "B", "C"}

    def test_deduce_step_has_correct_inputs(
        self, net_3hop: SLNetwork, op_root: Opinion, cond_ab: Opinion
    ) -> None:
        """The deduce step for B should have parent, conditional, counterfactual."""
        result = infer_node(net_3hop, "C")
        step_b = result.steps[1]  # B's deduce step
        assert step_b.node_id == "B"
        assert "parent" in step_b.inputs
        assert "conditional" in step_b.inputs
        assert "counterfactual" in step_b.inputs
        _assert_opinions_equal(step_b.inputs["parent"], op_root, "step B parent")
        _assert_opinions_equal(
            step_b.inputs["conditional"], cond_ab, "step B conditional"
        )


# ═══════════════════════════════════════════════════════════════════
# ERROR PATHS
# ═══════════════════════════════════════════════════════════════════


class TestErrorPaths:
    """Test error conditions for inference."""

    def test_nonexistent_node_raises(self) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.5, 0.3, 0.2)))
        with pytest.raises(NodeNotFoundError):
            infer_node(net, "Z")

    def test_exact_on_dag_raises(self) -> None:
        """method='exact' on a diamond DAG raises ValueError."""
        op = Opinion(0.5, 0.3, 0.2)
        net = SLNetwork()
        for nid in "ABCD":
            net.add_node(SLNode(nid, op))
        net.add_edge(SLEdge("A", "B", conditional=op))
        net.add_edge(SLEdge("A", "C", conditional=op))
        net.add_edge(SLEdge("B", "D", conditional=op))
        net.add_edge(SLEdge("C", "D", conditional=op))

        with pytest.raises(ValueError, match="tree-structured"):
            infer_node(net, "D", method="exact")

    def test_auto_on_dag_uses_approximate(self) -> None:
        """method='auto' on a DAG uses approximate inference (Step 5)."""
        op = Opinion(0.5, 0.3, 0.2)
        net = SLNetwork()
        for nid in "ABCD":
            net.add_node(SLNode(nid, op))
        net.add_edge(SLEdge("A", "B", conditional=op))
        net.add_edge(SLEdge("A", "C", conditional=op))
        net.add_edge(SLEdge("B", "D", conditional=op))
        net.add_edge(SLEdge("C", "D", conditional=op))

        # Should succeed with approximate inference
        result = infer_node(net, "D", method="auto")
        total = result.opinion.belief + result.opinion.disbelief + result.opinion.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_approximate_on_tree_succeeds(self, net_3hop: SLNetwork) -> None:
        """Approximate method works on trees too (exact single-parent)."""
        result = infer_node(net_3hop, "C", method="approximate")
        total = result.opinion.belief + result.opinion.disbelief + result.opinion.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_enumerate_on_tree_succeeds(self, net_3hop: SLNetwork) -> None:
        """Enumerate method works on trees (falls back to exact)."""
        result = infer_node(net_3hop, "C", method="enumerate")
        total = result.opinion.belief + result.opinion.disbelief + result.opinion.uncertainty
        assert abs(total - 1.0) < 1e-12

    def test_unknown_method_raises(self, net_3hop: SLNetwork) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            infer_node(net_3hop, "C", method="magic")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════
# infer_all
# ═══════════════════════════════════════════════════════════════════


class TestInferAll:
    """Test infer_all convenience function."""

    def test_empty_network(self) -> None:
        net = SLNetwork()
        results = infer_all(net)
        assert results == {}

    def test_single_node(self) -> None:
        op = Opinion(0.6, 0.3, 0.1)
        net = SLNetwork()
        net.add_node(SLNode("X", op))
        results = infer_all(net)
        assert "X" in results
        _assert_opinions_equal(results["X"].opinion, op, "infer_all single")

    def test_3hop_all_nodes(
        self,
        net_3hop: SLNetwork,
        op_root: Opinion,
        cond_ab: Opinion,
        cond_bc: Opinion,
    ) -> None:
        """infer_all returns correct opinions for every node."""
        results = infer_all(net_3hop)
        assert set(results.keys()) == {"A", "B", "C"}

        # Root is unchanged
        _assert_opinions_equal(results["A"].opinion, op_root, "all A")

        # B matches manual
        cf_ab = vacuous_counterfactual(cond_ab)
        expected_b = deduce(op_root, cond_ab, cf_ab)
        _assert_opinions_equal(results["B"].opinion, expected_b, "all B")

        # C matches manual chain
        cf_bc = vacuous_counterfactual(cond_bc)
        expected_c = deduce(expected_b, cond_bc, cf_bc)
        _assert_opinions_equal(results["C"].opinion, expected_c, "all C")


# ═══════════════════════════════════════════════════════════════════
# MATHEMATICAL PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestMathProperties:
    """Verify mathematical invariants across inference."""

    def test_additivity_4hop(self, net_4hop: SLNetwork) -> None:
        """b+d+u=1 at every node in a 4-hop chain."""
        result = infer_node(net_4hop, "D")
        for nid, op in result.intermediate_opinions.items():
            _assert_additivity(op, f"4hop {nid}")

    def test_classical_limit_dogmatic_chain(self) -> None:
        """When ALL three opinions are dogmatic (u=0), SL deduction
        reduces exactly to the scalar law of total probability:

            P(B) = P(A)·P(B|A) + P(¬A)·P(B|¬A)

        Key requirement: ALL three inputs must be dogmatic — antecedent,
        conditional, AND counterfactual.  A vacuous counterfactual has
        u=1 and would make the result non-dogmatic.

        Derivation for the dogmatic case:
            b_x=P(A), d_x=1-P(A), u_x=0
            c_{y|x} dogmatic, c_{y|¬x} dogmatic

            b_y = b_x · b_{y|x} + d_x · b_{y|¬x} + 0·(...)
                = P(A)·P(B|A) + (1-P(A))·P(B|¬A)
            d_y = 1 - b_y   (since u_y = 0 when all inputs have u=0)
            u_y = 0

            P(ω_B) = b_y + a_y·0 = b_y = P(A)·P(B|A) + (1-P(A))·P(B|¬A)

        This is the standard law of total probability.
        """
        # All three opinions are dogmatic (u=0)
        op_a = Opinion(0.8, 0.2, 0.0)           # P(A) = 0.8
        cond = Opinion(0.9, 0.1, 0.0)           # P(B|A=T) = 0.9
        counterfactual = Opinion(0.3, 0.7, 0.0)  # P(B|A=F) = 0.3

        net = SLNetwork()
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge(
            "A", "B",
            conditional=cond,
            counterfactual=counterfactual,
        ))

        result = infer_node(net, "B")

        # Hand-computed via deduction formula:
        # b_B = 0.8*0.9 + 0.2*0.3 + 0*(...)  = 0.72 + 0.06 = 0.78
        # d_B = 0.8*0.1 + 0.2*0.7 + 0*(...)  = 0.08 + 0.14 = 0.22
        # u_B = 0.8*0.0 + 0.2*0.0 + 0*(...)  = 0.00
        assert abs(result.opinion.belief - 0.78) < _TOL
        assert abs(result.opinion.disbelief - 0.22) < _TOL
        assert abs(result.opinion.uncertainty - 0.0) < _TOL

        # Projected probability = law of total probability
        # P(B) = P(A)·P(B|A) + (1-P(A))·P(B|¬A)
        #      = 0.8 * 0.9 + 0.2 * 0.3 = 0.78
        expected_p = 0.8 * 0.9 + 0.2 * 0.3
        actual_p = result.opinion.projected_probability()
        assert abs(actual_p - expected_p) < _TOL, (
            f"P(ω_B)={actual_p} != {expected_p}"
        )
        # Since u=0, P(ω) = b, confirming the classical limit
        assert abs(actual_p - result.opinion.belief) < _TOL

    def test_vacuous_root_produces_prior_based_result(self) -> None:
        """When root opinion is vacuous (u=1), the deduced opinion
        should depend primarily on the base rate.

        From the formula with b_A=0, d_A=0, u_A=1:
            c_y = 0 + 0 + 1 · (a_A · c_{y|x} + ā_A · c_{y|¬x})
            c_y = a_A · c_{y|x} + (1-a_A) · c_{y|¬x}

        With vacuous counterfactual, this is a weighted average
        of the conditional and vacuous, weighted by a_A.
        """
        op_a = Opinion(0.0, 0.0, 1.0, base_rate=0.5)
        cond = Opinion(0.9, 0.05, 0.05)

        net = SLNetwork()
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        result = infer_node(net, "B")
        # b_B = 0 + 0 + 1 * (0.5 * 0.9 + 0.5 * 0) = 0.45
        # d_B = 0 + 0 + 1 * (0.5 * 0.05 + 0.5 * 0) = 0.025
        # u_B = 0 + 0 + 1 * (0.5 * 0.05 + 0.5 * 1.0) = 0.525
        assert abs(result.opinion.belief - 0.45) < _TOL
        assert abs(result.opinion.disbelief - 0.025) < _TOL
        assert abs(result.opinion.uncertainty - 0.525) < _TOL

    def test_custom_callable_counterfactual(self) -> None:
        """Custom callable counterfactual function works in inference."""
        def mild_cf(cond: Opinion) -> Opinion:
            return Opinion(
                belief=cond.belief * 0.5,
                disbelief=cond.disbelief * 0.5,
                uncertainty=1.0 - cond.belief * 0.5 - cond.disbelief * 0.5,
                base_rate=cond.base_rate,
            )

        op_a = Opinion(0.7, 0.1, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        net.add_node(SLNode("A", op_a))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        # Manual
        cf = mild_cf(cond)
        expected = deduce(op_a, cond, cf)

        result = infer_node(net, "B", counterfactual_fn=mild_cf)
        _assert_opinions_equal(result.opinion, expected, "custom cf")


# ═══════════════════════════════════════════════════════════════════
# MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Verify inference functions are importable from package."""

    def test_import_from_sl_network(self) -> None:
        from jsonld_ex.sl_network import infer_all, infer_node
        assert infer_node is not None
        assert infer_all is not None
