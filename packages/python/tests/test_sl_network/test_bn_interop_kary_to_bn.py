"""Tests for k-ary SLNetwork → Bayesian Network conversion.

TDD RED phase for Phase C, Step C.3.

Extends to_bayesian_network() to handle any-cardinality discrete
variables.  K-ary nodes use MultinomialOpinion projected probabilities,
MultinomialEdge conditionals, and MultiParentMultinomialEdge tables.

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
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _vacuous_opinion() -> Opinion:
    return Opinion(0.0, 0.0, 1.0)


def _build_ternary_root_network() -> SLNetwork:
    """Network with a single ternary root node X."""
    net = SLNetwork(name="ternary_root")
    mn_op = MultinomialOpinion(
        beliefs={"0": 0.2, "1": 0.5, "2": 0.2},
        uncertainty=0.1,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    net.add_node(SLNode(
        node_id="X",
        opinion=_vacuous_opinion(),
        multinomial_opinion=mn_op,
    ))
    return net


def _build_ternary_chain_network() -> SLNetwork:
    """X(3) → Y(3) using MultinomialEdge."""
    net = SLNetwork(name="ternary_chain")

    mn_x = MultinomialOpinion(
        beliefs={"0": 0.18, "1": 0.48, "2": 0.28},
        uncertainty=0.06,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    net.add_node(SLNode(
        node_id="X", opinion=_vacuous_opinion(),
        multinomial_opinion=mn_x,
    ))

    mn_y_vac = MultinomialOpinion(
        beliefs={"0": 0.0, "1": 0.0, "2": 0.0},
        uncertainty=1.0,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    net.add_node(SLNode(
        node_id="Y", opinion=_vacuous_opinion(),
        multinomial_opinion=mn_y_vac,
    ))

    # Conditionals: P(Y|X=i) for i in {0, 1, 2}
    cond_0 = MultinomialOpinion(
        beliefs={"0": 0.65, "1": 0.18, "2": 0.08},
        uncertainty=0.09,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    cond_1 = MultinomialOpinion(
        beliefs={"0": 0.08, "1": 0.55, "2": 0.28},
        uncertainty=0.09,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    cond_2 = MultinomialOpinion(
        beliefs={"0": 0.18, "1": 0.28, "2": 0.45},
        uncertainty=0.09,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    edge = MultinomialEdge(
        source_id="X", target_id="Y",
        conditionals={"0": cond_0, "1": cond_1, "2": cond_2},
    )
    net.add_edge(edge)
    return net


def _build_ternary_converging_network() -> SLNetwork:
    """X(3), Y(3) → Z(3) using MultiParentMultinomialEdge."""
    net = SLNetwork(name="ternary_conv")

    for nid in ["X", "Y"]:
        mn = MultinomialOpinion(
            beliefs={"0": 0.3, "1": 0.4, "2": 0.2},
            uncertainty=0.1,
            base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
        )
        net.add_node(SLNode(
            node_id=nid, opinion=_vacuous_opinion(),
            multinomial_opinion=mn,
        ))

    mn_z_vac = MultinomialOpinion(
        beliefs={"0": 0.0, "1": 0.0, "2": 0.0},
        uncertainty=1.0,
        base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
    )
    net.add_node(SLNode(
        node_id="Z", opinion=_vacuous_opinion(),
        multinomial_opinion=mn_z_vac,
    ))

    # 3x3 = 9 entries, child domain = {0,1,2}
    conditionals: dict[tuple[str, ...], MultinomialOpinion] = {}
    idx = 0
    for sx in ["0", "1", "2"]:
        for sy in ["0", "1", "2"]:
            b0 = 0.1 * (idx % 5 + 1)
            b1 = 0.1 * ((idx + 2) % 5 + 1)
            b2 = max(0.0, 1.0 - b0 - b1 - 0.1)  # leave u=0.1
            conditionals[(sx, sy)] = MultinomialOpinion(
                beliefs={"0": b0, "1": b1, "2": b2},
                uncertainty=0.1,
                base_rates={"0": 1 / 3, "1": 1 / 3, "2": 1 / 3},
            )
            idx += 1

    mpe = MultiParentMultinomialEdge(
        parent_ids=("X", "Y"),
        target_id="Z",
        conditionals=conditionals,
    )
    net.add_edge(mpe)
    return net


# ═══════════════════════════════════════════════════════════════════
# TERNARY ROOT → BN
# ═══════════════════════════════════════════════════════════════════


class TestToBNKaryRoot:
    """Test k-ary root node conversion to BN."""

    def test_ternary_root_returns_valid_bn(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_root_network()
        bn = to_bayesian_network(net)
        assert bn.check_model()

    def test_ternary_root_cpd_has_3_rows(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_root_network()
        bn = to_bayesian_network(net)
        cpd = bn.get_cpds("X")
        values = cpd.get_values()
        assert values.shape == (3, 1)

    def test_ternary_root_cpd_matches_projected_probability(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_root_network()
        bn = to_bayesian_network(net)
        cpd = bn.get_cpds("X")
        values = cpd.get_values()

        node = net.get_node("X")
        pp = node.multinomial_opinion.projected_probability()
        for i, state in enumerate(sorted(pp.keys())):
            assert abs(float(values[i, 0]) - pp[state]) < 1e-9

    def test_ternary_root_cpd_columns_sum_to_one(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_root_network()
        bn = to_bayesian_network(net)
        cpd = bn.get_cpds("X")
        values = cpd.get_values()
        assert abs(sum(float(values[i, 0]) for i in range(3)) - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# TERNARY CHAIN → BN
# ═══════════════════════════════════════════════════════════════════


class TestToBNKaryChain:
    """Test k-ary single-parent edge conversion to BN."""

    def test_ternary_chain_returns_valid_bn(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_chain_network()
        bn = to_bayesian_network(net)
        assert bn.check_model()

    def test_ternary_chain_child_cpd_shape(self) -> None:
        """Y has 3 states and 1 parent with 3 states → (3, 3) CPD."""
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_chain_network()
        bn = to_bayesian_network(net)
        cpd_y = bn.get_cpds("Y")
        values = cpd_y.get_values()
        assert values.shape == (3, 3)

    def test_ternary_chain_child_cpd_matches_conditionals(self) -> None:
        """Each column should match the conditional's projected probability."""
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_chain_network()
        bn = to_bayesian_network(net)
        cpd_y = bn.get_cpds("Y")
        values = cpd_y.get_values()

        edge = net.get_multinomial_edge("X", "Y")
        for col_idx, parent_state in enumerate(sorted(edge.conditionals.keys())):
            pp = edge.conditionals[parent_state].projected_probability()
            for row_idx, child_state in enumerate(sorted(pp.keys())):
                assert abs(
                    float(values[row_idx, col_idx]) - pp[child_state]
                ) < 1e-9, (
                    f"Mismatch at Y={child_state}|X={parent_state}"
                )

    def test_ternary_chain_cpd_columns_sum_to_one(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_chain_network()
        bn = to_bayesian_network(net)
        cpd_y = bn.get_cpds("Y")
        values = cpd_y.get_values()
        for col in range(values.shape[1]):
            col_sum = sum(float(values[row, col]) for row in range(3))
            assert abs(col_sum - 1.0) < 1e-9

    def test_ternary_chain_structure(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_chain_network()
        bn = to_bayesian_network(net)
        assert set(bn.nodes()) == {"X", "Y"}
        assert ("X", "Y") in bn.edges()


# ═══════════════════════════════════════════════════════════════════
# TERNARY CONVERGING → BN
# ═══════════════════════════════════════════════════════════════════


class TestToBNKaryConverging:
    """Test k-ary multi-parent edge conversion to BN."""

    def test_ternary_converging_returns_valid_bn(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_converging_network()
        bn = to_bayesian_network(net)
        assert bn.check_model()

    def test_ternary_converging_child_cpd_shape(self) -> None:
        """Z has 3 states, 2 parents each with 3 states → (3, 9) CPD."""
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_converging_network()
        bn = to_bayesian_network(net)
        cpd_z = bn.get_cpds("Z")
        values = cpd_z.get_values()
        assert values.shape == (3, 9)

    def test_ternary_converging_cpd_columns_sum_to_one(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_converging_network()
        bn = to_bayesian_network(net)
        cpd_z = bn.get_cpds("Z")
        values = cpd_z.get_values()
        for col in range(values.shape[1]):
            col_sum = sum(float(values[row, col]) for row in range(3))
            assert abs(col_sum - 1.0) < 1e-9

    def test_ternary_converging_structure(self) -> None:
        from jsonld_ex.sl_network.bn_interop import to_bayesian_network

        net = _build_ternary_converging_network()
        bn = to_bayesian_network(net)
        assert set(bn.nodes()) == {"X", "Y", "Z"}
        edges = set(bn.edges())
        assert ("X", "Z") in edges
        assert ("Y", "Z") in edges


# ═══════════════════════════════════════════════════════════════════
# ROUND-TRIP FIDELITY (k-ary BN → SL → BN)
# ═══════════════════════════════════════════════════════════════════


class TestKaryRoundTrip:
    """Test BN → SL → BN round-trip for k-ary variables."""

    def _round_trip(self, bn_orig: BN, sample_count: int = 100_000) -> BN:
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )
        net = from_bayesian_network(bn_orig, default_sample_count=sample_count)
        return to_bayesian_network(net)

    def test_ternary_root_round_trip(self) -> None:
        """Single ternary root survives round-trip at large N."""
        model = BN()
        model.add_node("X")
        cpd = TabularCPD(
            variable="X", variable_card=3,
            values=[[0.2], [0.5], [0.3]],
        )
        model.add_cpds(cpd)
        model.check_model()

        bn_rt = self._round_trip(model)
        assert bn_rt.check_model()

        orig = model.get_cpds("X").get_values()
        rt = bn_rt.get_cpds("X").get_values()
        for i in range(3):
            assert abs(float(orig[i, 0]) - float(rt[i, 0])) < 0.01

    def test_ternary_chain_round_trip(self) -> None:
        """X(3) → Y(3) survives round-trip at large N."""
        model = BN([("X", "Y")])
        cpd_x = TabularCPD(
            variable="X", variable_card=3,
            values=[[0.2], [0.5], [0.3]],
        )
        cpd_y = TabularCPD(
            variable="Y", variable_card=3,
            values=[
                [0.7, 0.1, 0.2],
                [0.2, 0.6, 0.3],
                [0.1, 0.3, 0.5],
            ],
            evidence=["X"], evidence_card=[3],
        )
        model.add_cpds(cpd_x, cpd_y)
        model.check_model()

        bn_rt = self._round_trip(model)
        assert bn_rt.check_model()

        # Check Y CPD
        orig = model.get_cpds("Y").get_values()
        rt = bn_rt.get_cpds("Y").get_values()
        for row in range(3):
            for col in range(3):
                assert abs(float(orig[row, col]) - float(rt[row, col])) < 0.01, (
                    f"Mismatch at row={row}, col={col}"
                )

    def test_ternary_converging_round_trip(self) -> None:
        """X(3),Y(3) → Z(3) survives round-trip at large N."""
        model = BN([("X", "Z"), ("Y", "Z")])
        cpd_x = TabularCPD(
            variable="X", variable_card=3,
            values=[[0.2], [0.5], [0.3]],
        )
        cpd_y = TabularCPD(
            variable="Y", variable_card=3,
            values=[[0.3], [0.4], [0.3]],
        )
        cpd_z = TabularCPD(
            variable="Z", variable_card=3,
            values=[
                [0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.3, 0.1, 0.05],
                [0.2, 0.3, 0.4, 0.4, 0.5, 0.4, 0.3, 0.4, 0.35],
                [0.1, 0.2, 0.3, 0.2, 0.3, 0.5, 0.4, 0.5, 0.60],
            ],
            evidence=["X", "Y"], evidence_card=[3, 3],
        )
        model.add_cpds(cpd_x, cpd_y, cpd_z)
        model.check_model()

        bn_rt = self._round_trip(model)
        assert bn_rt.check_model()

        orig = model.get_cpds("Z").get_values()
        rt = bn_rt.get_cpds("Z").get_values()
        for row in range(3):
            for col in range(9):
                assert abs(float(orig[row, col]) - float(rt[row, col])) < 0.01, (
                    f"Z CPD mismatch at row={row}, col={col}"
                )

    def test_round_trip_structure_preserved(self) -> None:
        """Round-trip preserves nodes and edges."""
        model = BN([("X", "Y")])
        cpd_x = TabularCPD(
            variable="X", variable_card=3,
            values=[[0.2], [0.5], [0.3]],
        )
        cpd_y = TabularCPD(
            variable="Y", variable_card=3,
            values=[
                [0.7, 0.1, 0.2],
                [0.2, 0.6, 0.3],
                [0.1, 0.3, 0.5],
            ],
            evidence=["X"], evidence_card=[3],
        )
        model.add_cpds(cpd_x, cpd_y)
        model.check_model()

        bn_rt = self._round_trip(model)
        assert set(model.nodes()) == set(bn_rt.nodes())
        assert set(model.edges()) == set(bn_rt.edges())
