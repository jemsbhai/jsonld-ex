"""
Integration tests for mixed binary/multinomial SLNetwork inference
(Phase B, Step 4).

End-to-end scenarios that exercise multiple components together:
    - Mixed topologies (binary + multinomial subtrees)
    - Coarsening round-trip consistency
    - Multi-level multinomial chains
    - infer_all on mixed networks
    - Mathematical invariants across mixed inference

These tests verify that Phase B steps 1–3 integrate correctly
and that existing binary inference remains completely unaffected.

TDD: RED phase — these are integration tests that exercise
the full stack.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion, deduce
from jsonld_ex.multinomial_algebra import (
    MultinomialOpinion,
    coarsen,
    multinomial_cumulative_fuse,
    multinomial_deduce,
)
from jsonld_ex.sl_network.inference import infer_all, infer_node
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import (
    MultinomialEdge,
    SLEdge,
    SLNode,
)

TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _vacuous() -> Opinion:
    return Opinion(0.0, 0.0, 1.0)


def _assert_multinomial_close(
    actual: MultinomialOpinion,
    expected: MultinomialOpinion,
    tol: float = TOL,
) -> None:
    """Assert two MultinomialOpinions are component-wise close."""
    assert actual.domain == expected.domain, (
        f"Domains differ: {actual.domain} vs {expected.domain}"
    )
    assert abs(actual.uncertainty - expected.uncertainty) < tol, (
        f"Uncertainty: {actual.uncertainty} vs {expected.uncertainty}"
    )
    for state in expected.domain:
        assert abs(actual.beliefs[state] - expected.beliefs[state]) < tol, (
            f"beliefs[{state}]: {actual.beliefs[state]} vs "
            f"{expected.beliefs[state]}"
        )
        assert abs(actual.base_rates[state] - expected.base_rates[state]) < tol, (
            f"base_rates[{state}]: {actual.base_rates[state]} vs "
            f"{expected.base_rates[state]}"
        )


def _assert_additivity(op: MultinomialOpinion, tol: float = TOL) -> None:
    """Assert Σb + u = 1."""
    total = sum(op.beliefs.values()) + op.uncertainty
    assert abs(total - 1.0) < tol, f"Σb + u = {total}, expected 1.0"


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 1: Parallel binary and multinomial branches
# ═══════════════════════════════════════════════════════════════════


class TestParallelBranches:
    """Binary and multinomial branches operating independently.

    Network topology:
        A (binary) --SLEdge--> B (binary)
        X (ternary) --MultinomialEdge--> Y (binary multinomial {H,L})

    The two branches share no edges and should not interfere.
    """

    @pytest.fixture
    def parallel_net(self) -> SLNetwork:
        net = SLNetwork(name="parallel_branches")

        # Binary branch
        net.add_node(SLNode("A", Opinion(0.8, 0.1, 0.1)))
        net.add_node(SLNode("B", _vacuous()))
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=Opinion(0.9, 0.05, 0.05),
        ))

        # Multinomial branch
        multi_op = MultinomialOpinion(
            beliefs={"X1": 0.5, "X2": 0.3, "X3": 0.0},
            uncertainty=0.2,
            base_rates={"X1": 0.4, "X2": 0.3, "X3": 0.3},
        )
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=multi_op))
        net.add_node(SLNode("Y", _vacuous()))

        br = {"H": 0.5, "L": 0.5}
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals={
                "X1": MultinomialOpinion(
                    beliefs={"H": 0.8, "L": 0.1}, uncertainty=0.1,
                    base_rates=br,
                ),
                "X2": MultinomialOpinion(
                    beliefs={"H": 0.3, "L": 0.5}, uncertainty=0.2,
                    base_rates=br,
                ),
                "X3": MultinomialOpinion(
                    beliefs={"H": 0.1, "L": 0.6}, uncertainty=0.3,
                    base_rates=br,
                ),
            },
        ))
        return net

    def test_binary_branch_unaffected(self, parallel_net: SLNetwork) -> None:
        """Binary branch yields same result as a standalone network."""
        # Standalone binary network
        standalone = SLNetwork()
        standalone.add_node(SLNode("A", Opinion(0.8, 0.1, 0.1)))
        standalone.add_node(SLNode("B", _vacuous()))
        standalone.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=Opinion(0.9, 0.05, 0.05),
        ))
        expected = infer_node(standalone, "B")

        result = infer_node(parallel_net, "B")
        assert abs(result.opinion.belief - expected.opinion.belief) < TOL
        assert abs(result.opinion.disbelief - expected.opinion.disbelief) < TOL
        assert abs(result.opinion.uncertainty - expected.opinion.uncertainty) < TOL

    def test_multinomial_branch_produces_result(
        self, parallel_net: SLNetwork
    ) -> None:
        """Multinomial branch produces a valid MultinomialOpinion."""
        result = infer_node(parallel_net, "Y")
        assert "Y" in result.multinomial_intermediate_opinions
        y_op = result.multinomial_intermediate_opinions["Y"]
        assert y_op.domain == ("H", "L")
        _assert_additivity(y_op)

    def test_infer_all_covers_both_branches(
        self, parallel_net: SLNetwork
    ) -> None:
        """infer_all returns results for all 4 nodes."""
        results = infer_all(parallel_net)
        assert set(results.keys()) == {"A", "B", "X", "Y"}
        # Binary branch: no multinomial
        assert "A" not in results["B"].multinomial_intermediate_opinions
        assert "B" not in results["B"].multinomial_intermediate_opinions
        # Multinomial branch: has multinomial
        assert "Y" in results["Y"].multinomial_intermediate_opinions


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 2: Three-level multinomial chain
# ═══════════════════════════════════════════════════════════════════


class TestThreeLevelMultinomialChain:
    """Three-level chain: X (ternary) → Y (binary) → Z (binary).

    Tests that multinomial inference propagates correctly through
    multiple hops, and matches manual two-step computation.
    """

    @pytest.fixture
    def chain_net(self) -> tuple[SLNetwork, MultinomialOpinion, dict, dict]:
        parent_op = MultinomialOpinion(
            beliefs={"A": 0.4, "B": 0.3, "C": 0.1},
            uncertainty=0.2,
            base_rates={"A": 0.5, "B": 0.3, "C": 0.2},
        )
        br_hl = {"H": 0.5, "L": 0.5}
        conds_xy = {
            "A": MultinomialOpinion(
                beliefs={"H": 0.7, "L": 0.1}, uncertainty=0.2,
                base_rates=br_hl,
            ),
            "B": MultinomialOpinion(
                beliefs={"H": 0.2, "L": 0.5}, uncertainty=0.3,
                base_rates=br_hl,
            ),
            "C": MultinomialOpinion(
                beliefs={"H": 0.1, "L": 0.4}, uncertainty=0.5,
                base_rates=br_hl,
            ),
        }
        br_pq = {"P": 0.5, "Q": 0.5}
        conds_yz = {
            "H": MultinomialOpinion(
                beliefs={"P": 0.9, "Q": 0.0}, uncertainty=0.1,
                base_rates=br_pq,
            ),
            "L": MultinomialOpinion(
                beliefs={"P": 0.0, "Q": 0.8}, uncertainty=0.2,
                base_rates=br_pq,
            ),
        }

        net = SLNetwork(name="three_level_chain")
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=parent_op))
        net.add_node(SLNode("Y", _vacuous()))
        net.add_node(SLNode("Z", _vacuous()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y", conditionals=conds_xy,
        ))
        net.add_edge(MultinomialEdge(
            source_id="Y", target_id="Z", conditionals=conds_yz,
        ))
        return net, parent_op, conds_xy, conds_yz

    def test_leaf_matches_manual_computation(self, chain_net) -> None:
        """Z's result matches two sequential multinomial_deduce calls."""
        net, parent_op, conds_xy, conds_yz = chain_net
        y_manual = multinomial_deduce(parent_op, conds_xy)
        z_manual = multinomial_deduce(y_manual, conds_yz)

        result = infer_node(net, "Z")
        z_actual = result.multinomial_intermediate_opinions["Z"]
        _assert_multinomial_close(z_actual, z_manual)

    def test_intermediate_y_correct(self, chain_net) -> None:
        """Y's multinomial result matches direct deduction from X."""
        net, parent_op, conds_xy, _ = chain_net
        y_manual = multinomial_deduce(parent_op, conds_xy)

        result = infer_node(net, "Z")
        y_actual = result.multinomial_intermediate_opinions["Y"]
        _assert_multinomial_close(y_actual, y_manual)

    def test_all_nodes_satisfy_additivity(self, chain_net) -> None:
        """Σb + u = 1 at every multinomial node."""
        net, _, _, _ = chain_net
        result = infer_node(net, "Z")
        for nid, op in result.multinomial_intermediate_opinions.items():
            _assert_additivity(op)

    def test_topological_order_is_xyz(self, chain_net) -> None:
        """Topological sort produces X, Y, Z."""
        net, _, _, _ = chain_net
        result = infer_node(net, "Z")
        topo = result.topological_order
        assert topo.index("X") < topo.index("Y") < topo.index("Z")


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 3: Coarsening consistency
# ═══════════════════════════════════════════════════════════════════


class TestCoarseningConsistency:
    """Verify that coarsening an inferred MultinomialOpinion produces
    a binomial opinion with the correct projected probability.

    This is NOT enforced by inference — it's a mathematical property
    that should hold when the coarsen() operator is applied to
    inference results.
    """

    def test_coarsen_preserves_projected_probability(self) -> None:
        """P_binomial(focus) == P_multinomial(focus) after coarsening."""
        parent_op = MultinomialOpinion(
            beliefs={"A": 0.6, "B": 0.2, "C": 0.0},
            uncertainty=0.2,
            base_rates={"A": 0.5, "B": 0.3, "C": 0.2},
        )
        br = {"H": 0.5, "L": 0.5}
        conds = {
            "A": MultinomialOpinion(
                beliefs={"H": 0.8, "L": 0.1}, uncertainty=0.1,
                base_rates=br,
            ),
            "B": MultinomialOpinion(
                beliefs={"H": 0.3, "L": 0.4}, uncertainty=0.3,
                base_rates=br,
            ),
            "C": MultinomialOpinion(
                beliefs={"H": 0.1, "L": 0.5}, uncertainty=0.4,
                base_rates=br,
            ),
        }

        net = SLNetwork()
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=parent_op))
        net.add_node(SLNode("Y", _vacuous()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y", conditionals=conds,
        ))

        result = infer_node(net, "Y")
        y_multi = result.multinomial_intermediate_opinions["Y"]

        # Coarsen to focus on "H"
        y_binomial = coarsen(y_multi, "H")

        # Projected probabilities should match
        multi_proj = y_multi.projected_probability()
        binom_proj = y_binomial.projected_probability()

        assert abs(binom_proj - multi_proj["H"]) < TOL, (
            f"P_binomial={binom_proj} != P_multinomial(H)={multi_proj['H']}"
        )


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 4: Dogmatic and vacuous limits
# ═══════════════════════════════════════════════════════════════════


class TestExtremeOpinions:
    """Integration tests with extreme (dogmatic/vacuous) opinions."""

    def test_dogmatic_chain_selects_path(self) -> None:
        """Dogmatic parent at state A, child conditional A is high-H.

        Expected: child inherits conditional-A's opinion exactly.
        """
        parent = MultinomialOpinion(
            beliefs={"A": 1.0, "B": 0.0},
            uncertainty=0.0,
            base_rates={"A": 0.5, "B": 0.5},
        )
        br = {"H": 0.5, "L": 0.5}
        cond_a = MultinomialOpinion(
            beliefs={"H": 0.9, "L": 0.0}, uncertainty=0.1, base_rates=br,
        )
        cond_b = MultinomialOpinion(
            beliefs={"H": 0.0, "L": 0.8}, uncertainty=0.2, base_rates=br,
        )

        net = SLNetwork()
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=parent))
        net.add_node(SLNode("Y", _vacuous()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals={"A": cond_a, "B": cond_b},
        ))

        result = infer_node(net, "Y")
        y_op = result.multinomial_intermediate_opinions["Y"]

        # With dogmatic parent b(A)=1, child = conditional A
        assert abs(y_op.beliefs["H"] - cond_a.beliefs["H"]) < TOL
        assert abs(y_op.beliefs["L"] - cond_a.beliefs["L"]) < TOL
        assert abs(y_op.uncertainty - cond_a.uncertainty) < TOL

    def test_vacuous_parent_yields_prior_weighted(self) -> None:
        """Vacuous parent (u=1) yields base-rate-weighted child."""
        parent = MultinomialOpinion(
            beliefs={"A": 0.0, "B": 0.0},
            uncertainty=1.0,
            base_rates={"A": 0.7, "B": 0.3},
        )
        br = {"H": 0.5, "L": 0.5}
        cond_a = MultinomialOpinion(
            beliefs={"H": 0.8, "L": 0.0}, uncertainty=0.2, base_rates=br,
        )
        cond_b = MultinomialOpinion(
            beliefs={"H": 0.0, "L": 0.6}, uncertainty=0.4, base_rates=br,
        )

        net = SLNetwork()
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=parent))
        net.add_node(SLNode("Y", _vacuous()))
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals={"A": cond_a, "B": cond_b},
        ))

        expected = multinomial_deduce(parent, {"A": cond_a, "B": cond_b})
        result = infer_node(net, "Y")
        actual = result.multinomial_intermediate_opinions["Y"]
        _assert_multinomial_close(actual, expected)
        _assert_additivity(actual)


# ═══════════════════════════════════════════════════════════════════
# SCENARIO 5: infer_all comprehensive
# ═══════════════════════════════════════════════════════════════════


class TestInferAllComprehensive:
    """infer_all on a mixed network returns correct per-node results."""

    def test_infer_all_mixed_network(self) -> None:
        """Four-node mixed network: binary root A→B, multinomial root X→Y.

        Every node should get a result, and multinomial results
        should appear only for multinomial nodes.
        """
        net = SLNetwork()

        # Binary branch
        net.add_node(SLNode("A", Opinion(0.7, 0.2, 0.1)))
        net.add_node(SLNode("B", _vacuous()))
        net.add_edge(SLEdge(
            source_id="A", target_id="B",
            conditional=Opinion(0.85, 0.1, 0.05),
        ))

        # Multinomial branch
        multi_op = MultinomialOpinion(
            beliefs={"S1": 0.4, "S2": 0.3},
            uncertainty=0.3,
            base_rates={"S1": 0.5, "S2": 0.5},
        )
        net.add_node(SLNode("X", _vacuous(), multinomial_opinion=multi_op))
        net.add_node(SLNode("Y", _vacuous()))
        br = {"P": 0.5, "Q": 0.5}
        net.add_edge(MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals={
                "S1": MultinomialOpinion(
                    beliefs={"P": 0.7, "Q": 0.1}, uncertainty=0.2,
                    base_rates=br,
                ),
                "S2": MultinomialOpinion(
                    beliefs={"P": 0.2, "Q": 0.5}, uncertainty=0.3,
                    base_rates=br,
                ),
            },
        ))

        results = infer_all(net)

        # All four nodes present
        assert set(results.keys()) == {"A", "B", "X", "Y"}

        # Binary nodes: no multinomial results
        for nid in ("A", "B"):
            assert nid not in results[nid].multinomial_intermediate_opinions

        # Multinomial nodes: have multinomial results
        y_result = results["Y"]
        assert "X" in y_result.multinomial_intermediate_opinions
        assert "Y" in y_result.multinomial_intermediate_opinions
        y_op = y_result.multinomial_intermediate_opinions["Y"]
        _assert_additivity(y_op)

        # Binary B result is real inference (not vacuous)
        assert results["B"].opinion.belief > 0.0
        assert results["B"].opinion.uncertainty < 1.0
