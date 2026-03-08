"""
Tests for Bayesian Network ↔ SLNetwork interoperability.

Build order (TDD):
    Sub-step 1: from_bayesian_network() for tree-structured BNs
    Sub-step 2: from_bayesian_network() for multi-parent BNs (CPT → MultiParentEdge)
    Sub-step 3: to_bayesian_network()
    Sub-step 4: Round-trip tests (BN→SL→BN)
    Sub-step 5: Expressiveness divergence tests
    Sub-step 6: Error handling

References:
    Jøsang, A. (2016). Subjective Logic, §3.2 (evidence-to-opinion mapping),
    §12.6 (deduction operator).
    SLNetworks_plan.md §1.6 (BN interop spec).
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import SLEdge, SLNode, MultiParentEdge

# pgmpy is an optional dependency and requires Python >= 3.10
# (uses PEP 604 `int | float` syntax internally).
# Skip the entire module if pgmpy cannot be imported for any reason.
try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as BN
    except (ImportError, TypeError):
        pytest.skip("pgmpy not available", allow_module_level=True)
except TypeError:
    # pgmpy uses PEP 604 union types internally, which fail on Python < 3.10
    pytest.skip("pgmpy requires Python >= 3.10", allow_module_level=True)

from pgmpy.factors.discrete import TabularCPD


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _make_two_node_bn() -> BN:
    """Create a simple A → B Bayesian network.

    CPDs:
        P(A=1) = 0.6
        P(B=1|A=1) = 0.9, P(B=1|A=0) = 0.2
    """
    model = BN([("A", "B")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.4], [0.6]])
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[
            [0.8, 0.1],  # P(B=0|A=0), P(B=0|A=1)
            [0.2, 0.9],  # P(B=1|A=0), P(B=1|A=1)
        ],
        evidence=["A"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_a, cpd_b)
    model.check_model()
    return model


def _make_three_node_chain_bn() -> BN:
    """Create a chain A → B → C Bayesian network.

    CPDs:
        P(A=1) = 0.7
        P(B=1|A=1) = 0.8, P(B=1|A=0) = 0.3
        P(C=1|B=1) = 0.95, P(C=1|B=0) = 0.1
    """
    model = BN([("A", "B"), ("B", "C")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.3], [0.7]])
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[
            [0.7, 0.2],  # P(B=0|A=0), P(B=0|A=1)
            [0.3, 0.8],  # P(B=1|A=0), P(B=1|A=1)
        ],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.9, 0.05],  # P(C=0|B=0), P(C=0|B=1)
            [0.1, 0.95],  # P(C=1|B=0), P(C=1|B=1)
        ],
        evidence=["B"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


def _make_diverging_bn() -> BN:
    """Create a diverging (common cause) A → B, A → C network.

    CPDs:
        P(A=1) = 0.5
        P(B=1|A=1) = 0.8, P(B=1|A=0) = 0.1
        P(C=1|A=1) = 0.7, P(C=1|A=0) = 0.3
    """
    model = BN([("A", "B"), ("A", "C")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[
            [0.9, 0.2],  # P(B=0|A=0), P(B=0|A=1)
            [0.1, 0.8],  # P(B=1|A=0), P(B=1|A=1)
        ],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.7, 0.3],  # P(C=0|A=0), P(C=0|A=1)
            [0.3, 0.7],  # P(C=1|A=0), P(C=1|A=1)
        ],
        evidence=["A"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


# ═══════════════════════════════════════════════════════════════════
# Sub-step 1: from_bayesian_network() — tree-structured BNs
# ═══════════════════════════════════════════════════════════════════


class TestFromBayesianNetworkTreeStructure:
    """Test BN → SLNetwork conversion for tree-structured BNs."""

    def test_two_node_bn_creates_correct_nodes(self) -> None:
        """A → B BN should produce 2 content nodes."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        assert net.node_count() == 2
        assert "A" in [n for n in net.get_roots()]
        assert "B" in [n for n in net.get_leaves()]

    def test_two_node_bn_root_opinion_from_evidence(self) -> None:
        """Root node A opinion should match Opinion.from_evidence(pos=P(A=1)*N, neg=P(A=0)*N)."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100
        net = from_bayesian_network(bn, default_sample_count=N)

        node_a = net.get_node("A")
        # P(A=1) = 0.6, so pos=60, neg=40
        expected = Opinion.from_evidence(positive=0.6 * N, negative=0.4 * N)
        assert abs(node_a.opinion.belief - expected.belief) < 1e-9
        assert abs(node_a.opinion.disbelief - expected.disbelief) < 1e-9
        assert abs(node_a.opinion.uncertainty - expected.uncertainty) < 1e-9

    def test_two_node_bn_root_with_custom_sample_count(self) -> None:
        """Per-node sample_counts override default_sample_count."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(
            bn, sample_counts={"A": 1000}, default_sample_count=100
        )

        node_a = net.get_node("A")
        expected = Opinion.from_evidence(positive=0.6 * 1000, negative=0.4 * 1000)
        assert abs(node_a.opinion.belief - expected.belief) < 1e-9
        assert abs(node_a.opinion.uncertainty - expected.uncertainty) < 1e-9

    def test_two_node_bn_edge_conditional(self) -> None:
        """Edge A→B conditional should reflect P(B=1|A=1) = 0.9."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100
        net = from_bayesian_network(bn, default_sample_count=N)

        # Get the edge from A to B
        edges = net._edges  # Access internal for testing
        edge_key = ("A", "B")
        assert edge_key in edges

        edge = edges[edge_key]
        assert isinstance(edge, SLEdge)

        # conditional: Opinion from evidence P(B=1|A=1)=0.9
        cond_expected = Opinion.from_evidence(positive=0.9 * N, negative=0.1 * N)
        assert abs(edge.conditional.belief - cond_expected.belief) < 1e-9
        assert abs(edge.conditional.disbelief - cond_expected.disbelief) < 1e-9
        assert abs(edge.conditional.uncertainty - cond_expected.uncertainty) < 1e-9

    def test_two_node_bn_edge_counterfactual(self) -> None:
        """Edge A→B counterfactual should reflect P(B=1|A=0) = 0.2."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100
        net = from_bayesian_network(bn, default_sample_count=N)

        edge = net._edges[("A", "B")]
        assert edge.counterfactual is not None

        # counterfactual: Opinion from evidence P(B=1|A=0)=0.2
        cf_expected = Opinion.from_evidence(positive=0.2 * N, negative=0.8 * N)
        assert abs(edge.counterfactual.belief - cf_expected.belief) < 1e-9
        assert abs(edge.counterfactual.disbelief - cf_expected.disbelief) < 1e-9
        assert abs(edge.counterfactual.uncertainty - cf_expected.uncertainty) < 1e-9

    def test_three_node_chain_structure(self) -> None:
        """A → B → C chain should produce 3 nodes and 2 edges."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_node_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        assert net.node_count() == 3
        assert net.edge_count() == 2
        assert set(net.get_roots()) == {"A"}
        assert set(net.get_leaves()) == {"C"}

    def test_three_node_chain_root_opinion(self) -> None:
        """Root A in chain should get correct evidence-based opinion."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_node_chain_bn()
        N = 200
        net = from_bayesian_network(bn, default_sample_count=N)

        node_a = net.get_node("A")
        expected = Opinion.from_evidence(positive=0.7 * N, negative=0.3 * N)
        assert abs(node_a.opinion.belief - expected.belief) < 1e-9

    def test_three_node_chain_intermediate_is_non_root(self) -> None:
        """Non-root nodes B and C should not be in get_roots()."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_node_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        roots = set(net.get_roots())
        assert "B" not in roots
        assert "C" not in roots

    def test_diverging_structure(self) -> None:
        """A → B, A → C should produce 3 nodes, 2 edges, 1 root."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_diverging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        assert net.node_count() == 3
        assert net.edge_count() == 2
        assert set(net.get_roots()) == {"A"}
        assert set(net.get_leaves()) == {"B", "C"}

    def test_non_root_node_gets_vacuous_opinion(self) -> None:
        """Non-root nodes should get a vacuous (placeholder) opinion.

        Their actual opinion will be computed by inference, not assigned
        from the BN marginal. Using the BN marginal for non-root nodes
        would bake in BN inference results instead of letting SL
        inference compute them from the graph structure.
        """
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        node_b = net.get_node("B")
        # Non-root should have a vacuous opinion (all uncertainty)
        assert abs(node_b.opinion.uncertainty - 1.0) < 1e-9
        assert abs(node_b.opinion.belief) < 1e-9
        assert abs(node_b.opinion.disbelief) < 1e-9

    def test_custom_base_rate(self) -> None:
        """Custom base_rate should propagate to all opinions."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(
            bn, default_sample_count=100, base_rate=0.3
        )

        node_a = net.get_node("A")
        assert abs(node_a.opinion.base_rate - 0.3) < 1e-9

        edge = net._edges[("A", "B")]
        assert abs(edge.conditional.base_rate - 0.3) < 1e-9
        assert abs(edge.counterfactual.base_rate - 0.3) < 1e-9

    def test_all_content_nodes(self) -> None:
        """All converted nodes should be content nodes."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_node_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        for node_id in ["A", "B", "C"]:
            node = net.get_node(node_id)
            assert node.node_type == "content"

    def test_all_deduction_edges(self) -> None:
        """All converted edges should be deduction edges."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        edge = net._edges[("A", "B")]
        assert edge.edge_type == "deduction"

    def test_bdu_invariant_root_opinion(self) -> None:
        """Root node opinions must satisfy b+d+u=1."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        node_a = net.get_node("A")
        op = node_a.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_bdu_invariant_edge_conditional(self) -> None:
        """Edge conditional opinions must satisfy b+d+u=1."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        edge = net._edges[("A", "B")]
        op = edge.conditional
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_bdu_invariant_edge_counterfactual(self) -> None:
        """Edge counterfactual opinions must satisfy b+d+u=1."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        edge = net._edges[("A", "B")]
        assert edge.counterfactual is not None
        op = edge.counterfactual
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromBayesianNetworkDogmaticLimit:
    """Test that projected probabilities match BN probabilities in the dogmatic limit.

    When sample_count is very large, uncertainty → 0 and the SL projected
    probability P(ω) = b + a*u ≈ b should match the original BN probability.
    This verifies that the conversion preserves the probabilistic semantics.
    """

    def test_root_projected_probability_large_n(self) -> None:
        """With large N, root node P(ω) ≈ P(X=1) from BN."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)

        node_a = net.get_node("A")
        # P(A=1) = 0.6
        assert abs(node_a.opinion.projected_probability() - 0.6) < 0.01

    def test_conditional_projected_probability_large_n(self) -> None:
        """With large N, conditional P(ω) ≈ P(B=1|A=1) from BN."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)

        edge = net._edges[("A", "B")]
        # P(B=1|A=1) = 0.9
        assert abs(edge.conditional.projected_probability() - 0.9) < 0.01

    def test_counterfactual_projected_probability_large_n(self) -> None:
        """With large N, counterfactual P(ω) ≈ P(B=1|A=0) from BN."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)

        edge = net._edges[("A", "B")]
        assert edge.counterfactual is not None
        # P(B=1|A=0) = 0.2
        assert abs(edge.counterfactual.projected_probability() - 0.2) < 0.01

    def test_uncertainty_decreases_with_larger_n(self) -> None:
        """Uncertainty should decrease as sample count increases."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        net_small = from_bayesian_network(bn, default_sample_count=10)
        net_large = from_bayesian_network(bn, default_sample_count=10_000)

        node_small = net_small.get_node("A")
        node_large = net_large.get_node("A")

        assert node_small.opinion.uncertainty > node_large.opinion.uncertainty

    def test_chain_all_conditionals_match_bn_large_n(self) -> None:
        """All conditionals in 3-node chain match BN CPT values at large N."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_node_chain_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)

        # Edge A→B: P(B=1|A=1)=0.8, P(B=1|A=0)=0.3
        edge_ab = net._edges[("A", "B")]
        assert abs(edge_ab.conditional.projected_probability() - 0.8) < 0.01
        assert abs(edge_ab.counterfactual.projected_probability() - 0.3) < 0.01

        # Edge B→C: P(C=1|B=1)=0.95, P(C=1|B=0)=0.1
        edge_bc = net._edges[("B", "C")]
        assert abs(edge_bc.conditional.projected_probability() - 0.95) < 0.01
        assert abs(edge_bc.counterfactual.projected_probability() - 0.1) < 0.01


# ═══════════════════════════════════════════════════════════════════
# MULTI-PARENT BN HELPERS
# ═══════════════════════════════════════════════════════════════════


def _make_converging_bn() -> BN:
    """Create a converging (v-structure) A → C ← B network.

    CPDs:
        P(A=1) = 0.6
        P(B=1) = 0.4
        P(C=1|A,B):
            A=0, B=0: 0.05
            A=1, B=0: 0.6
            A=0, B=1: 0.5
            A=1, B=1: 0.95
    """
    model = BN([("A", "C"), ("B", "C")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.4], [0.6]])
    cpd_b = TabularCPD(variable="B", variable_card=2, values=[[0.6], [0.4]])
    # pgmpy column order: A varies slowest, B varies fastest
    # Columns: (A=0,B=0), (A=0,B=1), (A=1,B=0), (A=1,B=1)
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.95, 0.50, 0.40, 0.05],  # P(C=0|...)
            [0.05, 0.50, 0.60, 0.95],  # P(C=1|...)
        ],
        evidence=["A", "B"],
        evidence_card=[2, 2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


def _make_three_parent_bn() -> BN:
    """Create a 3-parent network: A, B, C → D.

    CPDs:
        P(A=1) = 0.5, P(B=1) = 0.5, P(C=1) = 0.5
        P(D=1|A,B,C): see table below (8 entries for 2^3 configs)
    """
    model = BN([("A", "D"), ("B", "D"), ("C", "D")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(variable="B", variable_card=2, values=[[0.5], [0.5]])
    cpd_c = TabularCPD(variable="C", variable_card=2, values=[[0.5], [0.5]])
    # 8 columns: (A=0,B=0,C=0), (A=0,B=0,C=1), ..., (A=1,B=1,C=1)
    # A varies slowest, C varies fastest
    p_d1 = [0.01, 0.10, 0.15, 0.40, 0.20, 0.50, 0.60, 0.99]
    p_d0 = [1.0 - p for p in p_d1]
    cpd_d = TabularCPD(
        variable="D",
        variable_card=2,
        values=[p_d0, p_d1],
        evidence=["A", "B", "C"],
        evidence_card=[2, 2, 2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
    model.check_model()
    return model


def _make_mixed_bn() -> BN:
    """Create a mixed network: A → B, A → C ← B (single + multi-parent).

    CPDs:
        P(A=1) = 0.5
        P(B=1|A=1) = 0.8, P(B=1|A=0) = 0.2
        P(C=1|A,B): A=0,B=0: 0.1; A=1,B=0: 0.4; A=0,B=1: 0.5; A=1,B=1: 0.9
    """
    model = BN([("A", "B"), ("A", "C"), ("B", "C")])
    cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.5], [0.5]])
    cpd_b = TabularCPD(
        variable="B",
        variable_card=2,
        values=[
            [0.8, 0.2],  # P(B=0|A=0), P(B=0|A=1)
            [0.2, 0.8],  # P(B=1|A=0), P(B=1|A=1)
        ],
        evidence=["A"],
        evidence_card=[2],
    )
    # Columns: (A=0,B=0), (A=0,B=1), (A=1,B=0), (A=1,B=1)
    cpd_c = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.9, 0.5, 0.6, 0.1],  # P(C=0|...)
            [0.1, 0.5, 0.4, 0.9],  # P(C=1|...)
        ],
        evidence=["A", "B"],
        evidence_card=[2, 2],
    )
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.check_model()
    return model


# ═══════════════════════════════════════════════════════════════════
# Sub-step 2: from_bayesian_network() — multi-parent BNs
# ═══════════════════════════════════════════════════════════════════


class TestFromBayesianNetworkMultiParent:
    """Test BN → SLNetwork conversion for multi-parent (converging) nodes."""

    def test_converging_produces_multi_parent_edge(self) -> None:
        """A → C ← B should produce a MultiParentEdge for C."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        # C should have a multi-parent edge, not a regular SLEdge
        assert "C" in net._multi_parent_edges
        mpe = net._multi_parent_edges["C"]
        assert isinstance(mpe, MultiParentEdge)

    def test_converging_multi_parent_edge_parents(self) -> None:
        """MultiParentEdge should reference parents A and B."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        mpe = net._multi_parent_edges["C"]
        assert set(mpe.parent_ids) == {"A", "B"}
        assert mpe.target_id == "C"

    def test_converging_multi_parent_edge_has_4_entries(self) -> None:
        """2 parents → 2^2 = 4 entries in conditional table."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        mpe = net._multi_parent_edges["C"]
        assert len(mpe.conditionals) == 4

    def test_converging_conditional_table_correct_values(self) -> None:
        """Each entry in the conditional table should match the BN CPT.

        The bool tuple order matches parent_ids order.
        Expected CPT for P(C=1|A,B):
            (False, False): 0.05
            (True,  False): 0.60
            (False, True):  0.50
            (True,  True):  0.95
        """
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        N = 100_000  # Large N for dogmatic limit
        net = from_bayesian_network(bn, default_sample_count=N)

        mpe = net._multi_parent_edges["C"]
        parent_order = mpe.parent_ids
        a_idx = parent_order.index("A")
        b_idx = parent_order.index("B")

        # Build lookup keyed by (A_state, B_state)
        def _get_p(a_val: bool, b_val: bool) -> float:
            key = [False, False]
            key[a_idx] = a_val
            key[b_idx] = b_val
            return mpe.conditionals[tuple(key)].projected_probability()

        assert abs(_get_p(False, False) - 0.05) < 0.01
        assert abs(_get_p(True, False) - 0.60) < 0.01
        assert abs(_get_p(False, True) - 0.50) < 0.01
        assert abs(_get_p(True, True) - 0.95) < 0.01

    def test_converging_root_nodes_have_evidence_opinions(self) -> None:
        """Both roots A and B should have evidence-based opinions."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        N = 100
        net = from_bayesian_network(bn, default_sample_count=N)

        node_a = net.get_node("A")
        expected_a = Opinion.from_evidence(positive=0.6 * N, negative=0.4 * N)
        assert abs(node_a.opinion.belief - expected_a.belief) < 1e-9

        node_b = net.get_node("B")
        expected_b = Opinion.from_evidence(positive=0.4 * N, negative=0.6 * N)
        assert abs(node_b.opinion.belief - expected_b.belief) < 1e-9

    def test_converging_child_gets_vacuous_opinion(self) -> None:
        """Multi-parent child C should get vacuous opinion."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        node_c = net.get_node("C")
        assert abs(node_c.opinion.uncertainty - 1.0) < 1e-9

    def test_three_parent_produces_8_entry_table(self) -> None:
        """3 parents → 2^3 = 8 entries in conditional table."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_parent_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        assert "D" in net._multi_parent_edges
        mpe = net._multi_parent_edges["D"]
        assert len(mpe.conditionals) == 8
        assert set(mpe.parent_ids) == {"A", "B", "C"}

    def test_three_parent_conditional_values_match_cpt(self) -> None:
        """All 8 entries in 3-parent CPT should match at large N."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_three_parent_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)

        mpe = net._multi_parent_edges["D"]
        parent_order = mpe.parent_ids
        a_idx = parent_order.index("A")
        b_idx = parent_order.index("B")
        c_idx = parent_order.index("C")

        # Expected P(D=1|A,B,C) from the CPT
        expected = {
            (False, False, False): 0.01,
            (False, False, True): 0.10,
            (False, True, False): 0.15,
            (False, True, True): 0.40,
            (True, False, False): 0.20,
            (True, False, True): 0.50,
            (True, True, False): 0.60,
            (True, True, True): 0.99,
        }

        for (a_val, b_val, c_val), expected_p in expected.items():
            key = [False, False, False]
            key[a_idx] = a_val
            key[b_idx] = b_val
            key[c_idx] = c_val
            actual_p = mpe.conditionals[tuple(key)].projected_probability()
            assert abs(actual_p - expected_p) < 0.01, (
                f"Mismatch at A={a_val}, B={b_val}, C={c_val}: "
                f"expected {expected_p}, got {actual_p}"
            )

    def test_mixed_network_single_and_multi_parent_edges(self) -> None:
        """A→B (single parent) and A,B→C (multi-parent) coexist."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        # A→B should be a regular SLEdge
        assert ("A", "B") in net._edges
        assert isinstance(net._edges[("A", "B")], SLEdge)

        # C should have a MultiParentEdge
        assert "C" in net._multi_parent_edges
        mpe = net._multi_parent_edges["C"]
        assert isinstance(mpe, MultiParentEdge)
        assert set(mpe.parent_ids) == {"A", "B"}

    def test_mixed_network_edge_count(self) -> None:
        """Mixed network: 1 SLEdge (A→B) + 2 multi-parent edges (A→C, B→C) = 3."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        # edge_count = len(SLEdges) + sum(len(parent_ids) for multi-parent)
        assert net.edge_count() == 3  # 1 + 2

    def test_bdu_invariant_multi_parent_conditionals(self) -> None:
        """All opinions in multi-parent conditional table must satisfy b+d+u=1."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)

        mpe = net._multi_parent_edges["C"]
        for key, opinion in mpe.conditionals.items():
            total = opinion.belief + opinion.disbelief + opinion.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"b+d+u != 1 for key {key}: {total}"
            )

    def test_multi_parent_uncertainty_decreases_with_n(self) -> None:
        """MultiParentEdge conditional uncertainty should decrease with larger N."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_converging_bn()
        net_small = from_bayesian_network(bn, default_sample_count=10)
        net_large = from_bayesian_network(bn, default_sample_count=10_000)

        mpe_small = net_small._multi_parent_edges["C"]
        mpe_large = net_large._multi_parent_edges["C"]

        key = (True, True)  # Any valid key
        # Find matching key in both (parent order may vary)
        # Use first key from small
        first_key = next(iter(mpe_small.conditionals))
        assert (
            mpe_small.conditionals[first_key].uncertainty
            > mpe_large.conditionals[first_key].uncertainty
        )


# ═══════════════════════════════════════════════════════════════════
# Sub-step 3: to_bayesian_network()
# ═══════════════════════════════════════════════════════════════════


class TestToBayesianNetwork:
    """Test SLNetwork → pgmpy BN conversion (lossy: uncertainty collapsed)."""

    def test_two_node_returns_pgmpy_bn(self) -> None:
        """to_bayesian_network() should return a pgmpy BN instance."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        # Should be a valid pgmpy BN (DiscreteBayesianNetwork or BayesianNetwork)
        assert hasattr(bn_out, "nodes")
        assert hasattr(bn_out, "edges")
        assert hasattr(bn_out, "get_cpds")

    def test_two_node_preserves_structure(self) -> None:
        """Output BN should have the same nodes and edges as the input."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        assert set(bn_out.nodes()) == {"A", "B"}
        assert ("A", "B") in bn_out.edges()

    def test_two_node_root_cpd_matches_projected_probability(self) -> None:
        """Root node CPD P(A=1) should equal the opinion's projected probability."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        cpd_a = bn_out.get_cpds("A")
        values = cpd_a.get_values()
        p_a1 = float(values[1, 0])  # P(A=1)

        expected = net.get_node("A").opinion.projected_probability()
        assert abs(p_a1 - expected) < 1e-9

    def test_two_node_child_cpd_conditional(self) -> None:
        """Child CPD P(B=1|A=1) should equal conditional's projected probability."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        cpd_b = bn_out.get_cpds("B")
        values = cpd_b.get_values()
        # Column 1 = parent A=1 (True)
        p_b1_given_a1 = float(values[1, 1])

        edge = net._edges[("A", "B")]
        expected = edge.conditional.projected_probability()
        assert abs(p_b1_given_a1 - expected) < 1e-9

    def test_two_node_child_cpd_counterfactual(self) -> None:
        """Child CPD P(B=1|A=0) should equal counterfactual's projected probability."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        cpd_b = bn_out.get_cpds("B")
        values = cpd_b.get_values()
        # Column 0 = parent A=0 (False)
        p_b1_given_a0 = float(values[1, 0])

        edge = net._edges[("A", "B")]
        expected = edge.counterfactual.projected_probability()
        assert abs(p_b1_given_a0 - expected) < 1e-9

    def test_three_node_chain_structure(self) -> None:
        """A → B → C chain should produce correct BN structure."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_three_node_chain_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        assert set(bn_out.nodes()) == {"A", "B", "C"}
        assert ("A", "B") in bn_out.edges()
        assert ("B", "C") in bn_out.edges()

    def test_output_bn_is_valid(self) -> None:
        """Output BN should pass pgmpy's check_model() validation."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_two_node_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        # This validates CPD dimensions, probabilities sum to 1, etc.
        assert bn_out.check_model()

    def test_chain_output_bn_is_valid(self) -> None:
        """Chain BN output should pass pgmpy's check_model()."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_three_node_chain_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        assert bn_out.check_model()

    def test_converging_output_bn_is_valid(self) -> None:
        """Converging (multi-parent) BN output should pass check_model()."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_converging_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        assert bn_out.check_model()

    def test_converging_cpd_column_count(self) -> None:
        """Multi-parent node C should have 4-column CPD (2^2 parent configs)."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_converging_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        cpd_c = bn_out.get_cpds("C")
        values = cpd_c.get_values()
        assert values.shape == (2, 4)  # 2 states x 4 parent configs

    def test_manually_built_network(self) -> None:
        """to_bayesian_network() should work on hand-built SLNetworks, not just round-trips."""
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = SLNetwork(name="manual")
        net.add_node(SLNode(
            node_id="X",
            opinion=Opinion(0.7, 0.2, 0.1),
        ))
        net.add_node(SLNode(
            node_id="Y",
            opinion=Opinion(0.0, 0.0, 1.0),  # vacuous
        ))
        net.add_edge(SLEdge(
            source_id="X",
            target_id="Y",
            conditional=Opinion(0.8, 0.1, 0.1),
            counterfactual=Opinion(0.2, 0.6, 0.2),
        ))

        bn_out = to_bayesian_network(net)

        assert set(bn_out.nodes()) == {"X", "Y"}
        assert bn_out.check_model()

        # Root CPD should use projected probability
        cpd_x = bn_out.get_cpds("X")
        p_x1 = float(cpd_x.get_values()[1, 0])
        assert abs(p_x1 - Opinion(0.7, 0.2, 0.1).projected_probability()) < 1e-9

    def test_cpd_rows_sum_to_one(self) -> None:
        """Every column in every CPD should sum to 1.0."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn_orig = _make_three_node_chain_bn()
        net = from_bayesian_network(bn_orig, default_sample_count=100)
        bn_out = to_bayesian_network(net)

        for node in bn_out.nodes():
            cpd = bn_out.get_cpds(node)
            values = cpd.get_values()
            for col_idx in range(values.shape[1]):
                col_sum = sum(values[:, col_idx])
                assert abs(col_sum - 1.0) < 1e-9, (
                    f"CPD column {col_idx} for {node} sums to {col_sum}"
                )


# ═══════════════════════════════════════════════════════════════════
# Sub-step 4: Round-trip tests (BN → SL → BN)
# ═══════════════════════════════════════════════════════════════════


class TestRoundTrip:
    """Test BN → SLNetwork → BN round-trip fidelity.

    In the dogmatic limit (large N, uncertainty ≈ 0), the round-trip
    should preserve probabilities closely.  With smaller N, information
    loss from uncertainty collapse is expected and quantified.
    """

    def _round_trip(
        self, bn_orig: BN, sample_count: int = 100_000
    ) -> BN:
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        net = from_bayesian_network(bn_orig, default_sample_count=sample_count)
        return to_bayesian_network(net)

    def test_two_node_root_probability_preserved(self) -> None:
        """P(A=1) should survive round-trip at large N."""
        bn_orig = _make_two_node_bn()
        bn_rt = self._round_trip(bn_orig)

        orig_vals = bn_orig.get_cpds("A").get_values()
        rt_vals = bn_rt.get_cpds("A").get_values()

        assert abs(float(orig_vals[1, 0]) - float(rt_vals[1, 0])) < 0.01

    def test_two_node_conditional_preserved(self) -> None:
        """P(B=1|A=1) and P(B=1|A=0) should survive round-trip at large N."""
        bn_orig = _make_two_node_bn()
        bn_rt = self._round_trip(bn_orig)

        orig_vals = bn_orig.get_cpds("B").get_values()
        rt_vals = bn_rt.get_cpds("B").get_values()

        # P(B=1|A=0) -- column 0
        assert abs(float(orig_vals[1, 0]) - float(rt_vals[1, 0])) < 0.01
        # P(B=1|A=1) -- column 1
        assert abs(float(orig_vals[1, 1]) - float(rt_vals[1, 1])) < 0.01

    def test_chain_all_cpds_preserved(self) -> None:
        """All CPDs in A → B → C chain survive round-trip at large N."""
        bn_orig = _make_three_node_chain_bn()
        bn_rt = self._round_trip(bn_orig)

        for node in ["A", "B", "C"]:
            orig = bn_orig.get_cpds(node).get_values()
            rt = bn_rt.get_cpds(node).get_values()
            for row in range(orig.shape[0]):
                for col in range(orig.shape[1]):
                    assert abs(float(orig[row, col]) - float(rt[row, col])) < 0.01, (
                        f"Mismatch at node {node}, row {row}, col {col}: "
                        f"orig={orig[row, col]}, rt={rt[row, col]}"
                    )

    def test_converging_cpd_preserved(self) -> None:
        """Multi-parent CPD for converging BN survives round-trip at large N."""
        bn_orig = _make_converging_bn()
        bn_rt = self._round_trip(bn_orig)

        for node in ["A", "B", "C"]:
            orig = bn_orig.get_cpds(node).get_values()
            rt = bn_rt.get_cpds(node).get_values()
            for row in range(orig.shape[0]):
                for col in range(orig.shape[1]):
                    assert abs(float(orig[row, col]) - float(rt[row, col])) < 0.01, (
                        f"Mismatch at node {node}, row {row}, col {col}"
                    )

    def test_round_trip_output_is_valid(self) -> None:
        """Round-tripped BN should pass pgmpy check_model() for all topologies."""
        for make_fn in [
            _make_two_node_bn,
            _make_three_node_chain_bn,
            _make_diverging_bn,
            _make_converging_bn,
            _make_mixed_bn,
        ]:
            bn_orig = make_fn()
            bn_rt = self._round_trip(bn_orig)
            assert bn_rt.check_model(), f"check_model() failed for {make_fn.__name__}"

    def test_round_trip_structure_preserved(self) -> None:
        """Round-trip preserves nodes and edges for all topologies."""
        for make_fn in [
            _make_two_node_bn,
            _make_three_node_chain_bn,
            _make_diverging_bn,
            _make_converging_bn,
            _make_mixed_bn,
        ]:
            bn_orig = make_fn()
            bn_rt = self._round_trip(bn_orig)
            assert set(bn_orig.nodes()) == set(bn_rt.nodes()), (
                f"Nodes differ for {make_fn.__name__}"
            )
            assert set(bn_orig.edges()) == set(bn_rt.edges()), (
                f"Edges differ for {make_fn.__name__}"
            )

    def test_information_loss_at_small_n(self) -> None:
        """At small N, round-trip error should be larger than at large N.

        This documents the expected information loss: small N means
        high uncertainty, which gets collapsed during SL → BN, shifting
        probabilities toward the base rate.
        """
        bn_orig = _make_two_node_bn()

        bn_small = self._round_trip(bn_orig, sample_count=5)
        bn_large = self._round_trip(bn_orig, sample_count=100_000)

        orig_p = float(bn_orig.get_cpds("A").get_values()[1, 0])  # P(A=1)=0.6
        small_p = float(bn_small.get_cpds("A").get_values()[1, 0])
        large_p = float(bn_large.get_cpds("A").get_values()[1, 0])

        error_small = abs(orig_p - small_p)
        error_large = abs(orig_p - large_p)

        # Large N should have strictly less error than small N
        assert error_large < error_small
        # Large N error should be negligible
        assert error_large < 0.001

    def test_uncertainty_causes_base_rate_pull(self) -> None:
        """With uncertainty > 0, projected probability is pulled toward base_rate.

        P(ω) = b + a*u.  When a=0.5 (default base rate), non-zero u
        pulls the projected probability toward 0.5.  This means
        round-trip at small N should show probabilities closer to 0.5
        than the original.
        """
        bn_orig = _make_two_node_bn()
        bn_rt = self._round_trip(bn_orig, sample_count=5)

        orig_p = float(bn_orig.get_cpds("A").get_values()[1, 0])  # 0.6
        rt_p = float(bn_rt.get_cpds("A").get_values()[1, 0])

        # Original is 0.6, which is > 0.5
        # With base rate pull, round-trip should be between 0.5 and 0.6
        assert 0.5 <= rt_p <= orig_p + 0.001


# ═══════════════════════════════════════════════════════════════════
# Sub-step 5: Expressiveness divergence (SL inference ≠ BN inference)
# ═══════════════════════════════════════════════════════════════════


class TestExpressivenessDivergence:
    """Demonstrate that SL inference ≠ BN scalar inference.

    This is the key scientific result for experiment E7.4: when
    opinions carry genuine uncertainty (u > 0), SL inference produces
    different results than BN inference using projected probabilities.

    The divergence arises because BN inference collapses uncertainty
    before propagation, while SL inference propagates the full
    (b, d, u) triple through the deduction operator.

    References:
        NeurIPS experiment roadmap E7.4
        Jøsang (2016), §12.6 (deduction operator)
    """

    def test_sl_inference_differs_from_bn_at_moderate_n(self) -> None:
        """SL inferred probability at leaf ≠ BN inferred probability.

        With moderate N (non-trivial uncertainty), SL deduction
        propagates uncertainty through the graph, yielding a different
        projected probability at the leaf than BN variable elimination.
        """
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        from jsonld_ex.sl_network.inference import infer_node

        bn_orig = _make_two_node_bn()
        N = 10  # Low N = high uncertainty
        net = from_bayesian_network(bn_orig, default_sample_count=N)

        # SL inference: propagate opinions through the graph
        result = infer_node(net, "B")
        sl_prob = result.opinion.projected_probability()

        # BN inference: use pgmpy's variable elimination
        from pgmpy.inference import VariableElimination
        bn_for_inference = to_bayesian_network(net)
        ve = VariableElimination(bn_for_inference)
        bn_result = ve.query(["B"])
        bn_prob = float(bn_result.values[1])  # P(B=1)

        # They should differ because SL propagates uncertainty
        # while BN collapses it before inference
        assert abs(sl_prob - bn_prob) > 1e-6, (
            f"Expected divergence but SL={sl_prob} == BN={bn_prob}"
        )

    def test_divergence_vanishes_at_large_n(self) -> None:
        """As N → ∞, SL and BN inference converge.

        In the dogmatic limit (u ≈ 0), SL deduction reduces to
        standard probability propagation, so the results should match.
        """
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        from jsonld_ex.sl_network.inference import infer_node
        from pgmpy.inference import VariableElimination

        bn_orig = _make_two_node_bn()
        N = 100_000
        net = from_bayesian_network(bn_orig, default_sample_count=N)

        sl_prob = infer_node(net, "B").opinion.projected_probability()

        bn_for_inference = to_bayesian_network(net)
        ve = VariableElimination(bn_for_inference)
        bn_prob = float(ve.query(["B"]).values[1])

        # Should be very close at large N
        assert abs(sl_prob - bn_prob) < 0.01

    def test_divergence_increases_with_uncertainty(self) -> None:
        """Higher uncertainty (smaller N) → larger SL-vs-BN divergence."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        from jsonld_ex.sl_network.inference import infer_node
        from pgmpy.inference import VariableElimination

        bn_orig = _make_two_node_bn()

        divergences = []
        for N in [5, 50, 500, 50_000]:
            net = from_bayesian_network(bn_orig, default_sample_count=N)
            sl_prob = infer_node(net, "B").opinion.projected_probability()

            bn_rt = to_bayesian_network(net)
            ve = VariableElimination(bn_rt)
            bn_prob = float(ve.query(["B"]).values[1])

            divergences.append(abs(sl_prob - bn_prob))

        # Divergence should be monotonically decreasing as N grows
        for i in range(len(divergences) - 1):
            assert divergences[i] >= divergences[i + 1] - 1e-9, (
                f"Divergence not monotonically decreasing: {divergences}"
            )

    def test_sl_preserves_uncertainty_at_leaf(self) -> None:
        """SL inference propagates uncertainty to the leaf node.

        BN inference produces a scalar probability with no uncertainty
        information. SL inference produces a full opinion where u > 0
        when the inputs have uncertainty.
        """
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network
        from jsonld_ex.sl_network.inference import infer_node

        bn_orig = _make_two_node_bn()
        N = 10  # High uncertainty
        net = from_bayesian_network(bn_orig, default_sample_count=N)

        result = infer_node(net, "B")

        # Leaf should have non-trivial uncertainty
        assert result.opinion.uncertainty > 0.01, (
            f"Expected uncertainty > 0.01, got {result.opinion.uncertainty}"
        )
        # b+d+u=1 still holds
        total = (
            result.opinion.belief
            + result.opinion.disbelief
            + result.opinion.uncertainty
        )
        assert abs(total - 1.0) < 1e-9

    def test_chain_divergence_amplifies(self) -> None:
        """In a chain A → B → C, divergence at C > divergence at B.

        Uncertainty compounds through deduction chains: each deduction
        step takes an input that already diverges from the BN result,
        so the gap grows with path length.

        Note: This is demonstrated for the specific test network
        parameters, not claimed as a universal theorem.  Pathological
        parameter choices could theoretically produce cancellation.
        """
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        from jsonld_ex.sl_network.inference import infer_node
        from pgmpy.inference import VariableElimination

        bn_orig = _make_three_node_chain_bn()
        N = 10
        net = from_bayesian_network(bn_orig, default_sample_count=N)

        bn_rt = to_bayesian_network(net)
        ve = VariableElimination(bn_rt)

        divergence_b = abs(
            infer_node(net, "B").opinion.projected_probability()
            - float(ve.query(["B"]).values[1])
        )
        divergence_c = abs(
            infer_node(net, "C").opinion.projected_probability()
            - float(ve.query(["C"]).values[1])
        )

        # Divergence should amplify through the chain
        assert divergence_c > divergence_b, (
            f"Expected chain amplification: div_C={divergence_c} "
            f"should be > div_B={divergence_b}"
        )


# ═══════════════════════════════════════════════════════════════════
# Sub-step 6: Error handling
# ═══════════════════════════════════════════════════════════════════


class TestBnInteropErrorHandling:
    """Test error handling for BN interop functions."""

    def test_from_bn_pgmpy_not_installed(self) -> None:
        """from_bayesian_network should raise ImportError with clear message
        when pgmpy is not installed.

        Note: We can't truly test this since pgmpy IS installed in our
        test environment. Instead we test the _require_pgmpy helper
        indirectly by verifying the function works (pgmpy is present).
        This test documents the expected behavior.
        """
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        # Should not raise when pgmpy is installed
        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=10)
        assert net.node_count() > 0

    def test_to_bn_pgmpy_not_installed(self) -> None:
        """to_bayesian_network should not raise when pgmpy is installed."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        bn = _make_two_node_bn()
        net = from_bayesian_network(bn, default_sample_count=10)
        bn_out = to_bayesian_network(net)
        assert bn_out is not None

    def test_from_bn_zero_sample_count(self) -> None:
        """Zero sample count should still work (all uncertainty)."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        # from_evidence with pos=0, neg=0 should yield vacuous
        # but prior_weight=2 prevents division by zero
        net = from_bayesian_network(bn, default_sample_count=0)
        node_a = net.get_node("A")
        # With pos=0, neg=0: b=0, d=0, u=1
        assert abs(node_a.opinion.uncertainty - 1.0) < 1e-9

    def test_from_bn_negative_sample_count_raises(self) -> None:
        """Negative sample count should raise ValueError."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_two_node_bn()
        with pytest.raises(ValueError):
            from_bayesian_network(bn, default_sample_count=-10)

    def test_from_bn_empty_network(self) -> None:
        """Empty BN (no nodes) should produce empty SLNetwork."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = BN()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.node_count() == 0

    def test_from_bn_single_isolated_node(self) -> None:
        """BN with a single node (no edges) should convert correctly."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = BN()
        bn.add_node("X")
        cpd_x = TabularCPD(variable="X", variable_card=2, values=[[0.3], [0.7]])
        bn.add_cpds(cpd_x)

        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.node_count() == 1
        node = net.get_node("X")
        assert abs(node.opinion.projected_probability() - 0.7) < 0.1


class TestBnInteropExports:
    """Verify that BN interop functions are accessible from sl_network package."""

    def test_from_bayesian_network_importable_from_package(self) -> None:
        """from_bayesian_network should be importable from jsonld_ex.sl_network."""
        from jsonld_ex.sl_network import from_bayesian_network

        assert callable(from_bayesian_network)

    def test_to_bayesian_network_importable_from_package(self) -> None:
        """to_bayesian_network should be importable from jsonld_ex.sl_network."""
        from jsonld_ex.sl_network import to_bayesian_network

        assert callable(to_bayesian_network)

    def test_exports_in_all(self) -> None:
        """Both functions should be listed in __all__."""
        import jsonld_ex.sl_network as pkg

        assert "from_bayesian_network" in pkg.__all__
        assert "to_bayesian_network" in pkg.__all__
