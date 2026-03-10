"""Tests for k-ary Bayesian Network → SLNetwork conversion.

TDD RED phase for Phase C, Step C.2.

Extends from_bayesian_network() to handle any-cardinality discrete
variables using MultinomialOpinion, MultinomialEdge, and
MultiParentMultinomialEdge.

Binary variables continue to use the existing binary path (Opinion,
SLEdge, MultiParentEdge) — zero behavior change for existing tests.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 3.5, Ch. 9.
    SLNetworks_plan.md §1.6.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    MultinomialEdge,
    MultiParentMultinomialEdge,
    SLEdge,
    SLNode,
)

# pgmpy is optional; skip if unavailable.
try:
    from pgmpy.models import DiscreteBayesianNetwork as BN
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as BN
    except (ImportError, TypeError):
        pytest.skip("pgmpy not available", allow_module_level=True)
except TypeError:
    pytest.skip("pgmpy requires Python >= 3.10", allow_module_level=True)

from pgmpy.factors.discrete import TabularCPD


# ═══════════════════════════════════════════════════════════════════
# HELPERS: K-ary BN construction
# ═══════════════════════════════════════════════════════════════════


def _make_ternary_root_bn() -> BN:
    """Single ternary root node X with P(X=0)=0.2, P(X=1)=0.5, P(X=2)=0.3."""
    model = BN()
    model.add_node("X")
    cpd = TabularCPD(
        variable="X", variable_card=3,
        values=[[0.2], [0.5], [0.3]],
    )
    model.add_cpds(cpd)
    model.check_model()
    return model


def _make_ternary_chain_bn() -> BN:
    """Chain: X(3 states) → Y(3 states).

    P(X): [0.2, 0.5, 0.3]
    P(Y|X): 3x3 CPT — columns are parent states, rows are child states.
    """
    model = BN([("X", "Y")])
    cpd_x = TabularCPD(
        variable="X", variable_card=3,
        values=[[0.2], [0.5], [0.3]],
    )
    # P(Y|X): 3 rows (Y states) x 3 columns (X states)
    #          X=0    X=1    X=2
    cpd_y = TabularCPD(
        variable="Y", variable_card=3,
        values=[
            [0.7, 0.1, 0.2],   # P(Y=0|X)
            [0.2, 0.6, 0.3],   # P(Y=1|X)
            [0.1, 0.3, 0.5],   # P(Y=2|X)
        ],
        evidence=["X"],
        evidence_card=[3],
    )
    model.add_cpds(cpd_x, cpd_y)
    model.check_model()
    return model


def _make_mixed_cardinality_chain_bn() -> BN:
    """Chain: X(binary) → Y(ternary).

    Tests cross-cardinality edges.
    """
    model = BN([("X", "Y")])
    cpd_x = TabularCPD(
        variable="X", variable_card=2,
        values=[[0.4], [0.6]],
    )
    # P(Y|X): 3 rows (Y states) x 2 columns (X states)
    cpd_y = TabularCPD(
        variable="Y", variable_card=3,
        values=[
            [0.6, 0.1],   # P(Y=0|X)
            [0.3, 0.7],   # P(Y=1|X)
            [0.1, 0.2],   # P(Y=2|X)
        ],
        evidence=["X"],
        evidence_card=[2],
    )
    model.add_cpds(cpd_x, cpd_y)
    model.check_model()
    return model


def _make_ternary_converging_bn() -> BN:
    """Converging: X(3) → Z(3) ← Y(3).

    Multi-parent with k-ary variables.
    """
    model = BN([("X", "Z"), ("Y", "Z")])
    cpd_x = TabularCPD(
        variable="X", variable_card=3,
        values=[[0.2], [0.5], [0.3]],
    )
    cpd_y = TabularCPD(
        variable="Y", variable_card=3,
        values=[[0.3], [0.4], [0.3]],
    )
    # P(Z|X,Y): 3 rows x 9 columns (X varies slowest)
    # 9 columns: (X=0,Y=0), (X=0,Y=1), (X=0,Y=2), (X=1,Y=0), ...
    cpd_z = TabularCPD(
        variable="Z", variable_card=3,
        values=[
            # Z=0 row
            [0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.3, 0.1, 0.05],
            # Z=1 row
            [0.2, 0.3, 0.4, 0.4, 0.5, 0.4, 0.3, 0.4, 0.35],
            # Z=2 row
            [0.1, 0.2, 0.3, 0.2, 0.3, 0.5, 0.4, 0.5, 0.60],
        ],
        evidence=["X", "Y"],
        evidence_card=[3, 3],
    )
    model.add_cpds(cpd_x, cpd_y, cpd_z)
    model.check_model()
    return model


def _make_mixed_converging_bn() -> BN:
    """Converging: X(binary) → Z(ternary) ← Y(binary).

    Multi-parent with mixed cardinalities.
    """
    model = BN([("X", "Z"), ("Y", "Z")])
    cpd_x = TabularCPD(
        variable="X", variable_card=2,
        values=[[0.4], [0.6]],
    )
    cpd_y = TabularCPD(
        variable="Y", variable_card=2,
        values=[[0.5], [0.5]],
    )
    # P(Z|X,Y): 3 rows x 4 columns (X varies slowest)
    cpd_z = TabularCPD(
        variable="Z", variable_card=3,
        values=[
            [0.7, 0.3, 0.4, 0.1],   # Z=0
            [0.2, 0.5, 0.4, 0.3],   # Z=1
            [0.1, 0.2, 0.2, 0.6],   # Z=2
        ],
        evidence=["X", "Y"],
        evidence_card=[2, 2],
    )
    model.add_cpds(cpd_x, cpd_y, cpd_z)
    model.check_model()
    return model


# ═══════════════════════════════════════════════════════════════════
# TERNARY ROOT NODE
# ═══════════════════════════════════════════════════════════════════


class TestFromBNKaryRootNode:
    """Test k-ary root node conversion."""

    def test_ternary_root_creates_node(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.node_count() == 1
        assert net.has_node("X")

    def test_ternary_root_has_multinomial_opinion(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        node = net.get_node("X")
        assert node.is_multinomial
        assert node.multinomial_opinion is not None
        assert node.multinomial_opinion.cardinality == 3

    def test_ternary_root_state_names_match_cpd(self) -> None:
        """State names should be string representations of state indices."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        node = net.get_node("X")
        # pgmpy state indices are 0, 1, 2 → str names "0", "1", "2"
        assert set(node.multinomial_opinion.domain) == {"0", "1", "2"}

    def test_ternary_root_projected_probability_large_n(self) -> None:
        """At large N, projected probabilities should match CPD values."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)
        node = net.get_node("X")
        pp = node.multinomial_opinion.projected_probability()
        assert abs(pp["0"] - 0.2) < 0.01
        assert abs(pp["1"] - 0.5) < 0.01
        assert abs(pp["2"] - 0.3) < 0.01

    def test_ternary_root_uncertainty_decreases_with_n(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        net_small = from_bayesian_network(bn, default_sample_count=10)
        net_large = from_bayesian_network(bn, default_sample_count=10_000)
        u_small = net_small.get_node("X").multinomial_opinion.uncertainty
        u_large = net_large.get_node("X").multinomial_opinion.uncertainty
        assert u_small > u_large

    def test_ternary_root_base_rates_uniform_by_default(self) -> None:
        """Default base rates should be uniform 1/k."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_root_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        node = net.get_node("X")
        for state in node.multinomial_opinion.domain:
            assert abs(node.multinomial_opinion.base_rates[state] - 1 / 3) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# SINGLE-PARENT K-ARY EDGES
# ═══════════════════════════════════════════════════════════════════


class TestFromBNKarySingleParentEdge:
    """Test k-ary single-parent edge conversion."""

    def test_ternary_chain_creates_multinomial_edge(self) -> None:
        """X(3) → Y(3) should use MultinomialEdge."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.has_multinomial_edge("X", "Y")

    def test_ternary_chain_edge_has_correct_conditionals(self) -> None:
        """MultinomialEdge should have one conditional per parent state."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        edge = net.get_multinomial_edge("X", "Y")
        assert edge.parent_cardinality == 3  # 3 parent states
        assert edge.child_cardinality == 3   # 3 child states

    def test_ternary_chain_conditionals_match_cpt_large_n(self) -> None:
        """At large N, conditional projected probabilities match CPT."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)
        edge = net.get_multinomial_edge("X", "Y")

        # Check P(Y|X=0): [0.7, 0.2, 0.1]
        pp_x0 = edge.conditionals["0"].projected_probability()
        assert abs(pp_x0["0"] - 0.7) < 0.01
        assert abs(pp_x0["1"] - 0.2) < 0.01
        assert abs(pp_x0["2"] - 0.1) < 0.01

        # Check P(Y|X=1): [0.1, 0.6, 0.3]
        pp_x1 = edge.conditionals["1"].projected_probability()
        assert abs(pp_x1["0"] - 0.1) < 0.01
        assert abs(pp_x1["1"] - 0.6) < 0.01
        assert abs(pp_x1["2"] - 0.3) < 0.01

    def test_mixed_cardinality_uses_multinomial_edge(self) -> None:
        """X(binary) → Y(ternary) should use MultinomialEdge (child is k>2)."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_cardinality_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        # The child has k=3, so we need MultinomialEdge
        assert net.has_multinomial_edge("X", "Y")

    def test_mixed_cardinality_edge_parent_states(self) -> None:
        """Binary parent → 2 conditionals (states '0' and '1')."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_cardinality_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        edge = net.get_multinomial_edge("X", "Y")
        assert edge.parent_cardinality == 2
        assert edge.child_cardinality == 3

    def test_mixed_cardinality_conditionals_match_large_n(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_cardinality_chain_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)
        edge = net.get_multinomial_edge("X", "Y")

        # P(Y|X=0): [0.6, 0.3, 0.1]
        pp = edge.conditionals["0"].projected_probability()
        assert abs(pp["0"] - 0.6) < 0.01
        assert abs(pp["1"] - 0.3) < 0.01
        assert abs(pp["2"] - 0.1) < 0.01

    def test_child_node_has_multinomial_opinion_vacuous(self) -> None:
        """Non-root k-ary node should have vacuous MultinomialOpinion."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        node_y = net.get_node("Y")
        assert node_y.is_multinomial
        assert abs(node_y.multinomial_opinion.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# MULTI-PARENT K-ARY EDGES
# ═══════════════════════════════════════════════════════════════════


class TestFromBNKaryMultiParentEdge:
    """Test k-ary multi-parent edge conversion."""

    def test_ternary_converging_creates_multi_parent_multinomial(self) -> None:
        """X(3),Y(3) → Z(3) should use MultiParentMultinomialEdge."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.has_multi_parent_multinomial_edge("Z")

    def test_ternary_converging_has_9_entries(self) -> None:
        """3 x 3 parent configs = 9 entries."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        mpe = net.get_multi_parent_multinomial_edge("Z")
        assert mpe.num_parent_configs == 9

    def test_ternary_converging_child_domain(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        mpe = net.get_multi_parent_multinomial_edge("Z")
        assert mpe.child_cardinality == 3

    def test_ternary_converging_conditionals_match_large_n(self) -> None:
        """Spot-check a few entries against the CPT at large N."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        N = 100_000
        net = from_bayesian_network(bn, default_sample_count=N)
        mpe = net.get_multi_parent_multinomial_edge("Z")

        parent_order = mpe.parent_ids
        x_idx = parent_order.index("X")
        y_idx = parent_order.index("Y")

        def _get_pp(x_state: str, y_state: str) -> dict[str, float]:
            key = ["", ""]
            key[x_idx] = x_state
            key[y_idx] = y_state
            return mpe.conditionals[tuple(key)].projected_probability()

        # P(Z|X=0,Y=0): [0.7, 0.2, 0.1]
        pp = _get_pp("0", "0")
        assert abs(pp["0"] - 0.7) < 0.01
        assert abs(pp["1"] - 0.2) < 0.01
        assert abs(pp["2"] - 0.1) < 0.01

        # P(Z|X=2,Y=2): [0.05, 0.35, 0.60]
        pp = _get_pp("2", "2")
        assert abs(pp["0"] - 0.05) < 0.01
        assert abs(pp["1"] - 0.35) < 0.01
        assert abs(pp["2"] - 0.60) < 0.01

    def test_mixed_converging_creates_multi_parent_multinomial(self) -> None:
        """X(2),Y(2) → Z(3): child is k>2 so use MultiParentMultinomialEdge."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.has_multi_parent_multinomial_edge("Z")

    def test_mixed_converging_has_4_entries(self) -> None:
        """2 x 2 parent configs = 4 entries."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_mixed_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        mpe = net.get_multi_parent_multinomial_edge("Z")
        assert mpe.num_parent_configs == 4
        assert mpe.child_cardinality == 3


# ═══════════════════════════════════════════════════════════════════
# BINARY BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════


class TestFromBNBinaryBackwardCompatibility:
    """Verify that purely binary BNs still use binary types (no regressions)."""

    def test_binary_root_no_multinomial_opinion(self) -> None:
        """Binary root should NOT have multinomial_opinion set."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        model = BN([("A", "B")])
        cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.4], [0.6]])
        cpd_b = TabularCPD(
            variable="B", variable_card=2,
            values=[[0.8, 0.1], [0.2, 0.9]],
            evidence=["A"], evidence_card=[2],
        )
        model.add_cpds(cpd_a, cpd_b)
        model.check_model()

        net = from_bayesian_network(model, default_sample_count=100)
        node_a = net.get_node("A")
        assert not node_a.is_multinomial
        assert node_a.multinomial_opinion is None

    def test_binary_edge_still_uses_sl_edge(self) -> None:
        """Binary A→B should still produce SLEdge, not MultinomialEdge."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        model = BN([("A", "B")])
        cpd_a = TabularCPD(variable="A", variable_card=2, values=[[0.4], [0.6]])
        cpd_b = TabularCPD(
            variable="B", variable_card=2,
            values=[[0.8, 0.1], [0.2, 0.9]],
            evidence=["A"], evidence_card=[2],
        )
        model.add_cpds(cpd_a, cpd_b)
        model.check_model()

        net = from_bayesian_network(model, default_sample_count=100)
        assert net.has_edge("A", "B")
        assert not net.has_multinomial_edge("A", "B")


# ═══════════════════════════════════════════════════════════════════
# NETWORK STRUCTURE INVARIANTS
# ═══════════════════════════════════════════════════════════════════


class TestFromBNKaryStructure:
    """Verify structural correctness of k-ary converted networks."""

    def test_ternary_chain_node_count(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.node_count() == 2

    def test_ternary_chain_edge_count(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert net.edge_count() == 1  # single MultinomialEdge

    def test_ternary_converging_edge_count(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        # MultiParentMultinomialEdge with 2 parents → 2 edges
        assert net.edge_count() == 2

    def test_ternary_chain_roots_and_leaves(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_chain_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        assert set(net.get_roots()) == {"X"}
        assert set(net.get_leaves()) == {"Y"}

    def test_all_nodes_are_content_type(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        bn = _make_ternary_converging_bn()
        net = from_bayesian_network(bn, default_sample_count=100)
        for nid in ["X", "Y", "Z"]:
            assert net.get_node(nid).node_type == "content"
