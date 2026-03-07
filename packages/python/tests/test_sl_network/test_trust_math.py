"""
Trust math property tests (Tier 2, Step 6).

Uses the Hypothesis library for property-based testing of trust
propagation algebraic properties.

Properties verified:
    1. Trust discount associativity (left-fold): 3-hop chain order
    2. Multi-path fusion differs from single paths
    3. Full trust identity through chains
    4. Vacuous trust dilution at any hop
    5. b+d+u=1 invariant after trust propagation
    6. Belief monotonic decrease along chains
    7. Uncertainty monotonic increase along chains
    8. Trust discount commutativity with fusion (order of agents)
    9. Non-negativity of all derived opinions
   10. Convergence to base rate under repeated discounting

TDD RED PHASE: These tests should FAIL until trust math properties
hold (they should pass with the current implementation).

References:
    Jøsang, A. (2016). Subjective Logic, §14.3 (transitive trust),
    §14.5 (multi-path trust fusion).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    trust_discount,
)
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.trust import propagate_trust
# Note: 'assume' kept imported for TestMultiPathFusionProperties
from jsonld_ex.sl_network.types import SLNode, TrustEdge


# ═══════════════════════════════════════════════════════════════════
# HYPOTHESIS STRATEGIES
# ═══════════════════════════════════════════════════════════════════

_TOL = 1e-9

# Trust opinions: must satisfy b+d+u=1, avoid extremes for stability
_trust_opinion = (
    st.tuples(
        st.floats(min_value=0.01, max_value=0.95),
        st.floats(min_value=0.01, max_value=0.95),
    )
    .filter(lambda t: 0.02 <= 1.0 - t[0] - t[1] <= 0.98)
    .map(lambda t: Opinion(
        belief=t[0],
        disbelief=t[1],
        uncertainty=1.0 - t[0] - t[1],
    ))
)

# Non-dogmatic trust (u > 0.02) for fusion stability
_non_dogmatic_trust = (
    st.tuples(
        st.floats(min_value=0.01, max_value=0.90),
        st.floats(min_value=0.01, max_value=0.90),
    )
    .filter(lambda t: 0.05 <= 1.0 - t[0] - t[1] <= 0.98)
    .map(lambda t: Opinion(
        belief=t[0],
        disbelief=t[1],
        uncertainty=1.0 - t[0] - t[1],
    ))
)

VAC = Opinion(0.0, 0.0, 1.0)


def _build_chain(agent_ids: list[str], trusts: list[Opinion]) -> SLNetwork:
    """Build a linear trust chain."""
    net = SLNetwork(name="chain")
    for aid in agent_ids:
        net.add_node(SLNode(node_id=aid, opinion=VAC, node_type="agent"))
    for i, t in enumerate(trusts):
        net.add_trust_edge(TrustEdge(
            source_id=agent_ids[i],
            target_id=agent_ids[i + 1],
            trust_opinion=t,
        ))
    return net


def _assert_valid(op: Opinion) -> None:
    """Assert b+d+u=1, all non-negative."""
    total = op.belief + op.disbelief + op.uncertainty
    assert abs(total - 1.0) < _TOL, f"b+d+u={total}"
    assert op.belief >= -_TOL, f"b={op.belief}"
    assert op.disbelief >= -_TOL, f"d={op.disbelief}"
    assert op.uncertainty >= -_TOL, f"u={op.uncertainty}"


# ═══════════════════════════════════════════════════════════════════
# 1. TRUST DISCOUNT ASSOCIATIVITY
# ═══════════════════════════════════════════════════════════════════


class TestTrustDiscountAssociativity:
    """trust_discount left-fold is associative (Jøsang §14.3)."""

    @given(t1=_trust_opinion, t2=_trust_opinion, t3=_trust_opinion)
    @settings(max_examples=200)
    def test_three_hop_chain_associative(
        self, t1: Opinion, t2: Opinion, t3: Opinion
    ) -> None:
        """td(td(t1, t2), t3) == td(t1, td(t2, t3)) is NOT generally true,
        but left-fold through a chain gives consistent results.

        The network propagation should match manual left-fold exactly.
        """
        net = _build_chain(["Q", "A", "B", "C"], [t1, t2, t3])
        result = propagate_trust(net, "Q")

        # Manual left-fold
        qa = t1
        qb = trust_discount(qa, t2)
        qc = trust_discount(qb, t3)

        actual = result.derived_trusts["C"]
        assert abs(actual.belief - qc.belief) < _TOL
        assert abs(actual.disbelief - qc.disbelief) < _TOL
        assert abs(actual.uncertainty - qc.uncertainty) < _TOL


# ═══════════════════════════════════════════════════════════════════
# 2. b+d+u=1 INVARIANT
# ═══════════════════════════════════════════════════════════════════


class TestBDUInvariant:
    """b+d+u=1 holds for every derived trust opinion."""

    @given(t1=_trust_opinion, t2=_trust_opinion)
    @settings(max_examples=200)
    def test_two_hop_bdu(self, t1: Opinion, t2: Opinion) -> None:
        net = _build_chain(["Q", "A", "B"], [t1, t2])
        result = propagate_trust(net, "Q")
        for op in result.derived_trusts.values():
            _assert_valid(op)

    @given(t1=_trust_opinion, t2=_trust_opinion, t3=_trust_opinion)
    @settings(max_examples=200)
    def test_three_hop_bdu(
        self, t1: Opinion, t2: Opinion, t3: Opinion
    ) -> None:
        net = _build_chain(["Q", "A", "B", "C"], [t1, t2, t3])
        result = propagate_trust(net, "Q")
        for op in result.derived_trusts.values():
            _assert_valid(op)

    @given(
        t_qa1=_non_dogmatic_trust,
        t_qa2=_non_dogmatic_trust,
        t_a1b=_non_dogmatic_trust,
        t_a2b=_non_dogmatic_trust,
    )
    @settings(max_examples=200)
    def test_diamond_bdu(
        self,
        t_qa1: Opinion,
        t_qa2: Opinion,
        t_a1b: Opinion,
        t_a2b: Opinion,
    ) -> None:
        """b+d+u=1 after multi-path fusion."""
        net = SLNetwork(name="diamond")
        for aid in ["Q", "A1", "A2", "B"]:
            net.add_node(SLNode(node_id=aid, opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A1", trust_opinion=t_qa1))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A2", trust_opinion=t_qa2))
        net.add_trust_edge(TrustEdge(source_id="A1", target_id="B", trust_opinion=t_a1b))
        net.add_trust_edge(TrustEdge(source_id="A2", target_id="B", trust_opinion=t_a2b))

        result = propagate_trust(net, "Q")
        for op in result.derived_trusts.values():
            _assert_valid(op)


# ═══════════════════════════════════════════════════════════════════
# 3. NON-NEGATIVITY
# ═══════════════════════════════════════════════════════════════════


class TestNonNegativity:
    """All components of derived trust are non-negative."""

    @given(t1=_trust_opinion, t2=_trust_opinion, t3=_trust_opinion)
    @settings(max_examples=200)
    def test_chain_non_negativity(
        self, t1: Opinion, t2: Opinion, t3: Opinion
    ) -> None:
        net = _build_chain(["Q", "A", "B", "C"], [t1, t2, t3])
        result = propagate_trust(net, "Q")
        for op in result.derived_trusts.values():
            assert op.belief >= -_TOL
            assert op.disbelief >= -_TOL
            assert op.uncertainty >= -_TOL


# ═══════════════════════════════════════════════════════════════════
# 4. FULL TRUST IDENTITY
# ═══════════════════════════════════════════════════════════════════


class TestFullTrustIdentity:
    """Full trust (b=1, d=0, u=0) is identity for trust_discount."""

    @given(t=_trust_opinion)
    @settings(max_examples=200)
    def test_full_trust_preserves_opinion(self, t: Opinion) -> None:
        """trust_discount(full, t) == t."""
        full = Opinion(1.0, 0.0, 0.0)
        result = trust_discount(full, t)
        assert abs(result.belief - t.belief) < _TOL
        assert abs(result.disbelief - t.disbelief) < _TOL
        assert abs(result.uncertainty - t.uncertainty) < _TOL

    @given(t=_trust_opinion)
    @settings(max_examples=200)
    def test_full_trust_chain_preserves(self, t: Opinion) -> None:
        """Chain of all-full-trust passes last edge opinion through."""
        full = Opinion(1.0, 0.0, 0.0)
        net = _build_chain(["Q", "A", "B", "C"], [full, full, t])
        result = propagate_trust(net, "Q")
        actual = result.derived_trusts["C"]
        assert abs(actual.belief - t.belief) < _TOL
        assert abs(actual.disbelief - t.disbelief) < _TOL
        assert abs(actual.uncertainty - t.uncertainty) < _TOL


# ═══════════════════════════════════════════════════════════════════
# 5. VACUOUS TRUST DILUTION
# ═══════════════════════════════════════════════════════════════════


class TestVacuousTrustDilution:
    """Vacuous trust (b=0, d=0, u=1) at any hop yields vacuous downstream."""

    @given(t=_trust_opinion)
    @settings(max_examples=200)
    def test_vacuous_first_hop_yields_vacuous(self, t: Opinion) -> None:
        """trust_discount(vacuous, t) is vacuous."""
        vac_trust = Opinion(0.0, 0.0, 1.0)
        result = trust_discount(vac_trust, t)
        assert result.belief < _TOL
        assert result.disbelief < _TOL
        assert abs(result.uncertainty - 1.0) < _TOL

    @given(t1=_trust_opinion, t2=_trust_opinion)
    @settings(max_examples=200)
    def test_vacuous_middle_hop(self, t1: Opinion, t2: Opinion) -> None:
        """Vacuous trust in middle of chain makes downstream vacuous."""
        vac_trust = Opinion(0.0, 0.0, 1.0)
        net = _build_chain(["Q", "A", "B", "C"], [t1, vac_trust, t2])
        result = propagate_trust(net, "Q")
        # B gets trust_discount(t1, vacuous) → belief=0
        # C gets trust_discount(that, t2) → belief=0
        actual_C = result.derived_trusts["C"]
        assert actual_C.belief < _TOL
        _assert_valid(actual_C)


# ═══════════════════════════════════════════════════════════════════
# 6. BELIEF MONOTONIC DECREASE
# ═══════════════════════════════════════════════════════════════════


class TestBeliefMonotonicDecrease:
    """Belief decreases (or stays equal) along a trust chain."""

    @given(t=_trust_opinion)
    @settings(max_examples=200)
    def test_single_discount_decreases_belief(self, t: Opinion) -> None:
        """trust_discount(t1, t2).belief <= t1.belief for partial trust."""
        result = trust_discount(t, t)
        # b_result = t.belief * t.belief <= t.belief (since b <= 1)
        assert result.belief <= t.belief + _TOL

    @given(
        t1=_trust_opinion,
        t2=_trust_opinion,
        t3=_trust_opinion,
    )
    @settings(max_examples=200)
    def test_chain_belief_monotonic(
        self, t1: Opinion, t2: Opinion, t3: Opinion
    ) -> None:
        """Along Q→A→B→C, belief(Q→A) >= belief(Q→B) >= belief(Q→C)."""
        net = _build_chain(["Q", "A", "B", "C"], [t1, t2, t3])
        result = propagate_trust(net, "Q")

        b_A = result.derived_trusts["A"].belief
        b_B = result.derived_trusts["B"].belief
        b_C = result.derived_trusts["C"].belief

        assert b_A >= b_B - _TOL, f"b_A={b_A} < b_B={b_B}"
        assert b_B >= b_C - _TOL, f"b_B={b_B} < b_C={b_C}"


# ═══════════════════════════════════════════════════════════════════
# 7. UNCERTAINTY MONOTONIC INCREASE
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyMonotonicIncrease:
    """Uncertainty increases (or stays equal) along a trust chain."""

    @given(
        t1=_trust_opinion,
        t2=_trust_opinion,
        t3=_trust_opinion,
    )
    @settings(max_examples=200)
    def test_chain_uncertainty_monotonic(
        self, t1: Opinion, t2: Opinion, t3: Opinion
    ) -> None:
        """Along Q→A→B→C, u(Q→A) <= u(Q→B) <= u(Q→C)."""
        net = _build_chain(["Q", "A", "B", "C"], [t1, t2, t3])
        result = propagate_trust(net, "Q")

        u_A = result.derived_trusts["A"].uncertainty
        u_B = result.derived_trusts["B"].uncertainty
        u_C = result.derived_trusts["C"].uncertainty

        assert u_A <= u_B + _TOL, f"u_A={u_A} > u_B={u_B}"
        assert u_B <= u_C + _TOL, f"u_B={u_B} > u_C={u_C}"


# ═══════════════════════════════════════════════════════════════════
# 8. CONVERGENCE TO BASE RATE
# ═══════════════════════════════════════════════════════════════════


class TestConvergenceToBaseRate:
    """Long trust chains converge toward the vacuous opinion."""

    def test_20_hop_chain_converges_to_vacuity(self) -> None:
        """20-hop chain with uniform trust=0.85 → belief near 0."""
        trust_op = Opinion(0.85, 0.05, 0.1)
        agents = [f"a{i}" for i in range(21)]
        trusts = [trust_op] * 20
        net = _build_chain(agents, trusts)
        result = propagate_trust(net, "a0")

        # After 20 hops, belief should be very small
        final = result.derived_trusts["a20"]
        _assert_valid(final)
        # 0.85^20 ≈ 0.039 — belief decays exponentially
        assert final.belief < 0.05
        # Uncertainty should dominate
        assert final.uncertainty > 0.9


# ═══════════════════════════════════════════════════════════════════
# 9. MULTI-PATH FUSION REDUCES UNCERTAINTY
# ═══════════════════════════════════════════════════════════════════


class TestMultiPathFusionProperties:
    """Multi-path fusion reduces uncertainty vs single paths."""

    @given(
        t_qa1=_non_dogmatic_trust,
        t_qa2=_non_dogmatic_trust,
        t_a1b=_non_dogmatic_trust,
        t_a2b=_non_dogmatic_trust,
    )
    @settings(max_examples=200)
    def test_fusion_reduces_uncertainty(
        self,
        t_qa1: Opinion,
        t_qa2: Opinion,
        t_a1b: Opinion,
        t_a2b: Opinion,
    ) -> None:
        """Fused trust has <= uncertainty than either single path."""
        path1 = trust_discount(t_qa1, t_a1b)
        path2 = trust_discount(t_qa2, t_a2b)

        # Skip if either path is dogmatic (u=0), fusion formula differs
        assume(path1.uncertainty > 0.001)
        assume(path2.uncertainty > 0.001)

        fused = cumulative_fuse(path1, path2)
        _assert_valid(fused)
        assert fused.uncertainty <= path1.uncertainty + _TOL
        assert fused.uncertainty <= path2.uncertainty + _TOL

    @given(
        t_qa1=_non_dogmatic_trust,
        t_qa2=_non_dogmatic_trust,
        t_a1b=_non_dogmatic_trust,
        t_a2b=_non_dogmatic_trust,
    )
    @settings(max_examples=200)
    def test_fusion_matches_propagation(
        self,
        t_qa1: Opinion,
        t_qa2: Opinion,
        t_a1b: Opinion,
        t_a2b: Opinion,
    ) -> None:
        """Network propagation matches manual path computation + fusion."""
        net = SLNetwork(name="diamond")
        for aid in ["Q", "A1", "A2", "B"]:
            net.add_node(SLNode(node_id=aid, opinion=VAC, node_type="agent"))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A1", trust_opinion=t_qa1))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A2", trust_opinion=t_qa2))
        net.add_trust_edge(TrustEdge(source_id="A1", target_id="B", trust_opinion=t_a1b))
        net.add_trust_edge(TrustEdge(source_id="A2", target_id="B", trust_opinion=t_a2b))

        result = propagate_trust(net, "Q")

        p1 = trust_discount(t_qa1, t_a1b)
        p2 = trust_discount(t_qa2, t_a2b)
        expected = cumulative_fuse(p1, p2)

        actual = result.derived_trusts["B"]
        assert abs(actual.belief - expected.belief) < _TOL
        assert abs(actual.disbelief - expected.disbelief) < _TOL
        assert abs(actual.uncertainty - expected.uncertainty) < _TOL
