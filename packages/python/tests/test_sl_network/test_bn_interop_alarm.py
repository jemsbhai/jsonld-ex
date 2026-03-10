"""Integration tests with the ALARM benchmark Bayesian network.

TDD RED phase for Phase C, Step C.4.

The ALARM network (A Logical Alarm Reduction Mechanism) is a classic
benchmark BN used in probabilistic reasoning literature.  It has 37
nodes with mixed cardinalities (2, 3, and 4 states) and 46 edges,
including multi-parent nodes with k-ary variables.

These tests verify that from_bayesian_network() and to_bayesian_network()
handle a real-world, non-trivial multi-valued BN correctly, including:
    - Correct node count and edge count after conversion
    - Binary and k-ary nodes produce the right opinion types
    - Round-trip fidelity at large N
    - Output BN passes pgmpy's check_model()

References:
    Beinlich et al. (1989). "The ALARM Monitoring System."
    pgmpy.utils.get_example_model('alarm').
"""

from __future__ import annotations

import pytest

# pgmpy is optional; skip if unavailable.
try:
    from pgmpy.models import DiscreteBayesianNetwork as BNClass
except ImportError:
    try:
        from pgmpy.models import BayesianNetwork as BNClass
    except (ImportError, TypeError):
        pytest.skip("pgmpy not available", allow_module_level=True)
except TypeError:
    pytest.skip("pgmpy requires Python >= 3.10", allow_module_level=True)

try:
    from pgmpy.utils import get_example_model
except ImportError:
    pytest.skip(
        "pgmpy.utils.get_example_model not available",
        allow_module_level=True,
    )

# Try to load ALARM; skip if network data not downloadable
try:
    _ALARM_BN = get_example_model("alarm")
except Exception:
    pytest.skip(
        "Could not load ALARM network (network download may have failed)",
        allow_module_level=True,
    )


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _get_node_card(bn, node_name: str) -> int:
    """Return the cardinality of a node in the BN."""
    return int(bn.get_cpds(node_name).get_values().shape[0])


def _categorize_alarm_nodes() -> dict[str, list[str]]:
    """Categorize ALARM nodes by cardinality."""
    categories: dict[str, list[str]] = {
        "binary": [],
        "ternary": [],
        "quaternary": [],
    }
    for node in _ALARM_BN.nodes():
        card = _get_node_card(_ALARM_BN, node)
        if card == 2:
            categories["binary"].append(node)
        elif card == 3:
            categories["ternary"].append(node)
        elif card == 4:
            categories["quaternary"].append(node)
    return categories


# ═══════════════════════════════════════════════════════════════════
# STRUCTURAL TESTS: BN → SLNetwork
# ═══════════════════════════════════════════════════════════════════


class TestAlarmFromBN:
    """Test from_bayesian_network() with the ALARM network."""

    def test_alarm_converts_without_error(self) -> None:
        """from_bayesian_network() should complete without raising."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        assert net is not None

    def test_alarm_node_count_matches(self) -> None:
        """Converted network should have the same number of nodes."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        assert net.node_count() == len(_ALARM_BN.nodes())

    def test_alarm_edge_count_matches(self) -> None:
        """Converted network should have the same number of directed edges."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        assert net.edge_count() == len(_ALARM_BN.edges())

    def test_alarm_binary_nodes_use_binomial_opinion(self) -> None:
        """Binary (card=2) nodes should NOT have multinomial_opinion."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        categories = _categorize_alarm_nodes()

        for node_name in categories["binary"]:
            node = net.get_node(node_name)
            assert not node.is_multinomial, (
                f"Binary node {node_name} should not have multinomial opinion"
            )

    def test_alarm_kary_nodes_use_multinomial_opinion(self) -> None:
        """K-ary (card>2) nodes should have multinomial_opinion set."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        categories = _categorize_alarm_nodes()

        for node_name in categories["ternary"] + categories["quaternary"]:
            node = net.get_node(node_name)
            assert node.is_multinomial, (
                f"K-ary node {node_name} should have multinomial opinion"
            )
            expected_card = _get_node_card(_ALARM_BN, node_name)
            assert node.multinomial_opinion.cardinality == expected_card

    def test_alarm_root_nodes_have_evidence_opinions(self) -> None:
        """Root nodes should have non-vacuous opinions (u < 1)."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        roots = set(net.get_roots())
        assert len(roots) > 0, "ALARM should have root nodes"

        for root_name in roots:
            node = net.get_node(root_name)
            if node.is_multinomial:
                assert node.multinomial_opinion.uncertainty < 1.0, (
                    f"Root {root_name} multinomial should be non-vacuous"
                )
            else:
                assert node.opinion.uncertainty < 1.0, (
                    f"Root {root_name} should be non-vacuous"
                )

    def test_alarm_non_root_nodes_are_vacuous(self) -> None:
        """Non-root nodes should have vacuous opinions (u ≈ 1)."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        roots = set(net.get_roots())

        for node_name in _ALARM_BN.nodes():
            if node_name in roots:
                continue
            node = net.get_node(node_name)
            if node.is_multinomial:
                assert abs(node.multinomial_opinion.uncertainty - 1.0) < 1e-9, (
                    f"Non-root {node_name} multinomial should be vacuous"
                )
            else:
                assert abs(node.opinion.uncertainty - 1.0) < 1e-9, (
                    f"Non-root {node_name} should be vacuous"
                )

    def test_alarm_has_mixed_cardinalities(self) -> None:
        """Sanity check: ALARM should have nodes with card 2, 3, and 4."""
        categories = _categorize_alarm_nodes()
        assert len(categories["binary"]) > 0, "Expected binary nodes"
        assert len(categories["ternary"]) > 0, "Expected ternary nodes"
        assert len(categories["quaternary"]) > 0, "Expected quaternary nodes"


# ═══════════════════════════════════════════════════════════════════
# TO BN: SLNetwork → BN
# ═══════════════════════════════════════════════════════════════════


class TestAlarmToBN:
    """Test to_bayesian_network() after converting ALARM."""

    def test_alarm_round_trip_produces_valid_bn(self) -> None:
        """Round-tripped ALARM should pass pgmpy's check_model()."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100_000)
        bn_rt = to_bayesian_network(net)
        assert bn_rt.check_model()

    def test_alarm_round_trip_preserves_structure(self) -> None:
        """Round-trip should preserve nodes and edges."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        bn_rt = to_bayesian_network(net)

        assert set(bn_rt.nodes()) == set(_ALARM_BN.nodes())
        assert set(bn_rt.edges()) == set(_ALARM_BN.edges())

    def test_alarm_round_trip_preserves_cardinalities(self) -> None:
        """All node cardinalities should be preserved through round-trip."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        bn_rt = to_bayesian_network(net)

        for node_name in _ALARM_BN.nodes():
            orig_card = _get_node_card(_ALARM_BN, node_name)
            rt_card = _get_node_card(bn_rt, node_name)
            assert orig_card == rt_card, (
                f"Cardinality mismatch for {node_name}: "
                f"orig={orig_card}, rt={rt_card}"
            )

    def test_alarm_round_trip_cpd_fidelity_roots(self) -> None:
        """Root node CPDs should survive round-trip at large N."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        N = 100_000
        net = from_bayesian_network(_ALARM_BN, default_sample_count=N)
        bn_rt = to_bayesian_network(net)

        roots = [
            n for n in _ALARM_BN.nodes()
            if len(list(_ALARM_BN.get_parents(n))) == 0
        ]

        for root_name in roots:
            orig = _ALARM_BN.get_cpds(root_name).get_values()
            rt = bn_rt.get_cpds(root_name).get_values()
            card = orig.shape[0]
            for i in range(card):
                assert abs(float(orig[i, 0]) - float(rt[i, 0])) < 0.01, (
                    f"Root {root_name} state {i}: "
                    f"orig={orig[i, 0]:.4f}, rt={rt[i, 0]:.4f}"
                )

    def test_alarm_round_trip_cpd_fidelity_all_nodes(self) -> None:
        """All CPDs should survive round-trip at large N (< 0.02 error)."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        N = 100_000
        net = from_bayesian_network(_ALARM_BN, default_sample_count=N)
        bn_rt = to_bayesian_network(net)

        max_error = 0.0
        worst_node = ""

        for node_name in _ALARM_BN.nodes():
            orig = _ALARM_BN.get_cpds(node_name).get_values()
            rt = bn_rt.get_cpds(node_name).get_values()

            assert orig.shape == rt.shape, (
                f"Shape mismatch for {node_name}: "
                f"orig={orig.shape}, rt={rt.shape}"
            )

            for row in range(orig.shape[0]):
                for col in range(orig.shape[1]):
                    err = abs(float(orig[row, col]) - float(rt[row, col]))
                    if err > max_error:
                        max_error = err
                        worst_node = node_name

        # At N=100k, max error should be very small
        assert max_error < 0.02, (
            f"Max CPD round-trip error {max_error:.6f} at node {worst_node} "
            f"exceeds tolerance 0.02"
        )

    def test_alarm_cpd_columns_sum_to_one(self) -> None:
        """Every column of every CPD in the round-tripped BN should sum to 1."""
        from jsonld_ex.sl_network.bn_interop import (
            from_bayesian_network,
            to_bayesian_network,
        )

        net = from_bayesian_network(_ALARM_BN, default_sample_count=100)
        bn_rt = to_bayesian_network(net)

        for node_name in bn_rt.nodes():
            cpd = bn_rt.get_cpds(node_name)
            values = cpd.get_values()
            for col in range(values.shape[1]):
                col_sum = sum(float(values[row, col]) for row in range(values.shape[0]))
                assert abs(col_sum - 1.0) < 1e-9, (
                    f"CPD column {col} for {node_name} sums to {col_sum}"
                )


# ═══════════════════════════════════════════════════════════════════
# EXPRESSIVENESS: SL carries more information than BN
# ═══════════════════════════════════════════════════════════════════


class TestAlarmExpressiveness:
    """Demonstrate that SLNetwork carries richer information than BN.

    After conversion, the SLNetwork retains epistemic uncertainty
    that the original BN cannot express.  This is the key added value
    of the SL representation.
    """

    def test_alarm_sl_retains_uncertainty(self) -> None:
        """Converted SLNetwork should have non-zero uncertainty at all roots."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        N = 100  # Moderate N → meaningful uncertainty
        net = from_bayesian_network(_ALARM_BN, default_sample_count=N)

        roots = net.get_roots()
        for root_name in roots:
            node = net.get_node(root_name)
            if node.is_multinomial:
                u = node.multinomial_opinion.uncertainty
            else:
                u = node.opinion.uncertainty
            assert u > 0.001, (
                f"Root {root_name} should retain meaningful uncertainty "
                f"at N={N}, got u={u}"
            )

    def test_alarm_uncertainty_decreases_with_evidence(self) -> None:
        """More evidence (larger N) → less uncertainty, monotonically."""
        from jsonld_ex.sl_network.bn_interop import from_bayesian_network

        # Pick the first root node
        roots = [
            n for n in _ALARM_BN.nodes()
            if len(list(_ALARM_BN.get_parents(n))) == 0
        ]
        root_name = sorted(roots)[0]

        uncertainties = []
        for N in [10, 100, 1000, 10_000]:
            net = from_bayesian_network(_ALARM_BN, default_sample_count=N)
            node = net.get_node(root_name)
            if node.is_multinomial:
                uncertainties.append(node.multinomial_opinion.uncertainty)
            else:
                uncertainties.append(node.opinion.uncertainty)

        for i in range(len(uncertainties) - 1):
            assert uncertainties[i] > uncertainties[i + 1], (
                f"Uncertainty should decrease monotonically: {uncertainties}"
            )
