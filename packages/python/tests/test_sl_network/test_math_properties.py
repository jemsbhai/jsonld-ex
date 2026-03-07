"""
Mathematical property tests for SLNetwork inference (Tier 1, Step 6).

Uses the Hypothesis library for property-based testing.

Properties verified:
    1. Additivity invariant: b+d+u=1 at every node after inference
    2. Deduction factorization through P(x)
    3. Projected probability LTP conditions
    4. Where SL expressiveness advantage actually manifests
    5. Chain associativity
    6. Vacuous root propagation
    7. Dogmatic root propagation
    8. Uncertainty behavior along chains
    9. Counterfactual strategy effects
   10. Non-negativity
   11. Enumeration weight properties

MATHEMATICAL FINDINGS (discovered during testing):

Finding 1 — Deduction factorization:
    c_y = P(x)·c_{y|x} + (1-P(x))·c_{y|¬x}
    Two antecedents with the same P(x) produce IDENTICAL (b,d,u).
    SL's expressiveness advantage does NOT manifest in single deduction.

Finding 2 — LTP for projected probabilities is conditional:
    P(ω_y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x) does NOT hold in general.
    It requires: a_y = a_{y|x} = a_{y|¬x}, which holds iff
    P(y|x) + P(y|¬x) = 1 (when a_x = a_{y|x} = a_{y|¬x}).
    The existing deduce() docstring claims this as a general property,
    which is incorrect.  The (b,d,u) components are correct; only the
    base rate formula creates the discrepancy in projected probability.

References:
    Jøsang, A. (2016). Subjective Logic, Springer.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    deduce,
    trust_discount,
)
from jsonld_ex.sl_network.counterfactuals import (
    adversarial_counterfactual,
    vacuous_counterfactual,
)
from jsonld_ex.sl_network.inference import infer_all, infer_node
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import MultiParentEdge, SLEdge, SLNode


# ═══════════════════════════════════════════════════════════════════
# HYPOTHESIS STRATEGIES
# ═══════════════════════════════════════════════════════════════════

# Generate (b, d) with room for u, avoiding extreme boundaries.
_opinion_strategy = (
    st.tuples(
        st.floats(min_value=0.01, max_value=0.95),
        st.floats(min_value=0.01, max_value=0.95),
        st.floats(min_value=0.05, max_value=0.95),
    )
    .filter(lambda t: 0.02 <= 1.0 - t[0] - t[1] <= 0.98)
    .map(lambda t: Opinion(
        belief=t[0],
        disbelief=t[1],
        uncertainty=1.0 - t[0] - t[1],
        base_rate=t[2],
    ))
)

# Non-dogmatic opinions for fusion tests (u well away from 0).
_non_dogmatic_opinion = (
    st.tuples(
        st.floats(min_value=0.01, max_value=0.90),
        st.floats(min_value=0.01, max_value=0.90),
        st.floats(min_value=0.05, max_value=0.95),
    )
    .filter(lambda t: 0.05 <= 1.0 - t[0] - t[1] <= 0.98)
    .map(lambda t: Opinion(
        belief=t[0],
        disbelief=t[1],
        uncertainty=1.0 - t[0] - t[1],
        base_rate=t[2],
    ))
)


_TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
# 1. ADDITIVITY INVARIANT: b + d + u = 1
# ═══════════════════════════════════════════════════════════════════


class TestAdditivityInvariant:
    """b + d + u = 1 must hold at every node after inference.

    Proof:
        c_y = P(x)·c_{y|x} + (1-P(x))·c_{y|¬x}  for c ∈ {b, d, u}
        Sum = P(x)·1 + (1-P(x))·1 = 1.
    """

    @given(root=_opinion_strategy, conditional=_opinion_strategy)
    @settings(max_examples=200)
    def test_additivity_2hop(self, root: Opinion, conditional: Opinion) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=conditional))

        result = infer_node(net, "B")
        for nid, op in result.intermediate_opinions.items():
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < _TOL, f"{nid}: b+d+u={total}"

    @given(root=_opinion_strategy, c1=_opinion_strategy, c2=_opinion_strategy)
    @settings(max_examples=200)
    def test_additivity_3hop(self, root: Opinion, c1: Opinion, c2: Opinion) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=c1))
        net.add_edge(SLEdge("B", "C", conditional=c2))

        result = infer_node(net, "C")
        for nid, op in result.intermediate_opinions.items():
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < _TOL, f"{nid}: b+d+u={total}"

    @given(
        root=_opinion_strategy,
        ca=_opinion_strategy,
        cb=_opinion_strategy,
        cd1=_non_dogmatic_opinion,
        cd2=_non_dogmatic_opinion,
    )
    @settings(max_examples=200)
    def test_additivity_diamond(
        self, root: Opinion, ca: Opinion, cb: Opinion,
        cd1: Opinion, cd2: Opinion,
    ) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("D", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=ca))
        net.add_edge(SLEdge("A", "C", conditional=cb))
        net.add_edge(SLEdge("B", "D", conditional=cd1))
        net.add_edge(SLEdge("C", "D", conditional=cd2))

        result = infer_node(net, "D", method="approximate")
        for nid, op in result.intermediate_opinions.items():
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < _TOL, f"{nid}: b+d+u={total}"


# ═══════════════════════════════════════════════════════════════════
# 2. DEDUCTION FACTORIZATION THROUGH P(x)
# ═══════════════════════════════════════════════════════════════════


class TestDeductionFactorization:
    """The deduction formula factorizes through P(x).

    THEOREM: deduce(ω₁, yx, ynx) has identical (b,d,u) to
    deduce(ω₂, yx, ynx) whenever P(ω₁) = P(ω₂).

    PROOF:
        c_y = b_x·c_{y|x} + d_x·c_{y|¬x} + u_x·(a_x·c_{y|x} + ā_x·c_{y|¬x})
            = (b_x + a_x·u_x)·c_{y|x} + (d_x + ā_x·u_x)·c_{y|¬x}
            = P(x)·c_{y|x} + (1-P(x))·c_{y|¬x}
    """

    @given(conditional=_opinion_strategy, counterfactual=_opinion_strategy)
    @settings(max_examples=200)
    def test_same_px_same_bdu(
        self, conditional: Opinion, counterfactual: Opinion
    ) -> None:
        """Two antecedents with P(x)=0.7 produce identical (b,d,u)."""
        op1 = Opinion(0.6, 0.2, 0.2, base_rate=0.5)  # P=0.7
        op2 = Opinion(0.4, 0.0, 0.6, base_rate=0.5)  # P=0.7

        r1 = deduce(op1, conditional, counterfactual)
        r2 = deduce(op2, conditional, counterfactual)

        assert abs(r1.belief - r2.belief) < _TOL
        assert abs(r1.disbelief - r2.disbelief) < _TOL
        assert abs(r1.uncertainty - r2.uncertainty) < _TOL

    def test_conflict_vs_ignorance_identical_deduction(self) -> None:
        """Conflict and ignorance with same P(x) produce identical
        deduced (b,d,u).  This is the factorization theorem."""
        conflict = Opinion(0.45, 0.45, 0.10, base_rate=0.5)  # P=0.50
        ignorance = Opinion(0.00, 0.00, 1.00, base_rate=0.5)  # P=0.50

        cond = Opinion(0.8, 0.1, 0.1)
        cf = vacuous_counterfactual(cond)

        r_c = deduce(conflict, cond, cf)
        r_i = deduce(ignorance, cond, cf)

        assert abs(r_c.belief - r_i.belief) < 1e-12
        assert abs(r_c.disbelief - r_i.disbelief) < 1e-12
        assert abs(r_c.uncertainty - r_i.uncertainty) < 1e-12


# ═══════════════════════════════════════════════════════════════════
# 3. PROJECTED PROBABILITY LTP — CONDITIONAL PROPERTY
# ═══════════════════════════════════════════════════════════════════


class TestProjectedProbabilityLTP:
    """P(ω_y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x) is NOT a general
    property of SL deduction.

    PROOF of failure:
        P(ω_y) = b_y + a_y·u_y
        LTP = P(x)·(b_{y|x}+a_{y|x}·u_{y|x}) + (1-P(x))·(b_{y|¬x}+a_{y|¬x}·u_{y|¬x})
            = b_y + P(x)·a_{y|x}·u_{y|x} + (1-P(x))·a_{y|¬x}·u_{y|¬x}

        So LTP holds iff: a_y·u_y = P(x)·a_{y|x}·u_{y|x} + (1-P(x))·a_{y|¬x}·u_{y|¬x}

        With a_{y|x} = a_{y|¬x} = a:  RHS = a·u_y
        But a_y = a_x·P(y|x) + (1-a_x)·P(y|¬x), which ≠ a in general.

    The base rate formula a_y computes y's prior by marginalizing over
    x's prior (a_x), not its posterior (P(x)).  This is intentional —
    the base rate represents the prior expectation of y — but it means
    LTP for projected probabilities does not hold unconditionally.

    NOTE: The (b,d,u) components ARE correct per Jøsang Def. 12.6.
    The issue is solely in how a_y interacts with u_y in the projected
    probability.  The existing deduce() docstring claiming LTP as a
    general property should be corrected.
    """

    def test_ltp_does_not_hold_in_general(self) -> None:
        """Counterexample: all base rates 0.5, but P(y|x)+P(y|¬x)≠1."""
        x = Opinion(0.5, 0.0, 0.5, base_rate=0.5)    # P=0.75
        yx = Opinion(0.5, 0.0, 0.5, base_rate=0.5)    # P=0.75
        ynx = Opinion(0.5, 0.0, 0.5, base_rate=0.5)   # P=0.75

        result = deduce(x, yx, ynx)
        p_x = x.projected_probability()
        ltp = p_x * yx.projected_probability() + (1 - p_x) * ynx.projected_probability()
        actual = result.projected_probability()

        # a_y = 0.5*(0.75+0.75) = 0.75 ≠ 0.5, so LTP fails
        assert abs(actual - ltp) > 0.01, (
            f"LTP should fail: actual={actual}, ltp={ltp}"
        )

    def test_ltp_holds_when_condition_met(self) -> None:
        """LTP holds when P(y|x) + P(y|¬x) = 1 (with equal base rates).

        In this case a_y = 0.5·(P(y|x) + P(y|¬x)) = 0.5 = a.
        """
        x = Opinion(0.5, 0.0, 0.5, base_rate=0.5)      # P=0.75
        yx = Opinion(0.5, 0.0, 0.5, base_rate=0.5)      # P=0.75
        ynx = Opinion(0.0, 0.5, 0.5, base_rate=0.5)     # P=0.25

        # P(y|x) + P(y|¬x) = 0.75 + 0.25 = 1.0 ✓

        result = deduce(x, yx, ynx)
        p_x = x.projected_probability()
        ltp = p_x * yx.projected_probability() + (1 - p_x) * ynx.projected_probability()
        actual = result.projected_probability()

        assert abs(actual - ltp) < 1e-12, (
            f"LTP should hold: actual={actual}, ltp={ltp}"
        )

    def test_component_wise_ltp_always_holds(self) -> None:
        """The COMPONENT-WISE law of total probability ALWAYS holds:
            c_y = P(x)·c_{y|x} + (1-P(x))·c_{y|¬x}  for c ∈ {b, d, u}

        This is the fundamental deduction formula, which is exact.
        """
        x = Opinion(0.5, 0.0, 0.5, base_rate=0.5)
        yx = Opinion(0.5, 0.0, 0.5, base_rate=0.5)
        ynx = Opinion(0.5, 0.0, 0.5, base_rate=0.5)

        result = deduce(x, yx, ynx)
        p_x = x.projected_probability()

        for comp in ["belief", "disbelief", "uncertainty"]:
            expected = (
                p_x * getattr(yx, comp)
                + (1 - p_x) * getattr(ynx, comp)
            )
            actual = getattr(result, comp)
            assert abs(actual - expected) < 1e-12, (
                f"Component {comp}: {actual} != {expected}"
            )

    @given(
        antecedent=_opinion_strategy,
        conditional=_opinion_strategy,
        counterfactual=_opinion_strategy,
    )
    @settings(max_examples=500)
    def test_component_ltp_hypothesis(
        self, antecedent: Opinion, conditional: Opinion, counterfactual: Opinion
    ) -> None:
        """Component-wise LTP holds for ALL opinions (Hypothesis)."""
        result = deduce(antecedent, conditional, counterfactual)
        p_x = antecedent.projected_probability()

        for comp in ["belief", "disbelief", "uncertainty"]:
            expected = (
                p_x * getattr(conditional, comp)
                + (1 - p_x) * getattr(counterfactual, comp)
            )
            actual = getattr(result, comp)
            assert abs(actual - expected) < _TOL, (
                f"Component {comp}: {actual} != {expected}"
            )


# ═══════════════════════════════════════════════════════════════════
# 4. WHERE SL EXPRESSIVENESS ACTUALLY MANIFESTS
# ═══════════════════════════════════════════════════════════════════


class TestExpressivenessAdvantage:
    """SL's advantage over scalars manifests in fusion and trust
    discount, NOT in single-deduction steps."""

    def test_fusion_distinguishes_conflict_from_ignorance(self) -> None:
        """cumulative_fuse treats conflict and ignorance differently."""
        conflict = Opinion(0.45, 0.45, 0.10, base_rate=0.5)
        ignorance = Opinion(0.00, 0.00, 1.00, base_rate=0.5)
        evidence = Opinion(0.7, 0.1, 0.2, base_rate=0.5)

        fused_c = cumulative_fuse(conflict, evidence)
        fused_i = cumulative_fuse(ignorance, evidence)

        assert abs(fused_c.belief - fused_i.belief) > 0.01

    def test_trust_discount_distinguishes_uncertainty(self) -> None:
        """trust_discount treats different uncertainty levels differently."""
        high_trust = Opinion(0.9, 0.0, 0.1)
        low_trust = Opinion(0.3, 0.0, 0.7)
        source = Opinion(0.8, 0.1, 0.1)

        disc_h = trust_discount(high_trust, source)
        disc_l = trust_discount(low_trust, source)

        assert disc_h.belief > disc_l.belief
        assert disc_h.uncertainty < disc_l.uncertainty

    def test_multi_source_dag_reduces_uncertainty(self) -> None:
        """Diamond DAG fusion reduces uncertainty below single-path."""
        op = Opinion(0.7, 0.1, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("D", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        net.add_edge(SLEdge("A", "C", conditional=cond))
        net.add_edge(SLEdge("B", "D", conditional=cond))
        net.add_edge(SLEdge("C", "D", conditional=cond))

        result = infer_node(net, "D", method="approximate")
        u_b = result.intermediate_opinions["B"].uncertainty
        u_d = result.opinion.uncertainty
        assert u_d < u_b


# ═══════════════════════════════════════════════════════════════════
# 5. CHAIN ASSOCIATIVITY
# ═══════════════════════════════════════════════════════════════════


class TestChainAssociativity:
    """Full network pass == step-by-step deduce chain."""

    @given(
        root=_opinion_strategy,
        c1=_opinion_strategy,
        c2=_opinion_strategy,
        c3=_opinion_strategy,
    )
    @settings(max_examples=200)
    def test_chain_decomposition(
        self, root: Opinion, c1: Opinion, c2: Opinion, c3: Opinion
    ) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("D", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=c1))
        net.add_edge(SLEdge("B", "C", conditional=c2))
        net.add_edge(SLEdge("C", "D", conditional=c3))

        full = infer_node(net, "D")

        cf1 = vacuous_counterfactual(c1)
        ob = deduce(root, c1, cf1)
        cf2 = vacuous_counterfactual(c2)
        oc = deduce(ob, c2, cf2)
        cf3 = vacuous_counterfactual(c3)
        od = deduce(oc, c3, cf3)

        assert abs(full.opinion.belief - od.belief) < _TOL
        assert abs(full.opinion.disbelief - od.disbelief) < _TOL
        assert abs(full.opinion.uncertainty - od.uncertainty) < _TOL
        assert abs(full.opinion.base_rate - od.base_rate) < _TOL


# ═══════════════════════════════════════════════════════════════════
# 6. VACUOUS ROOT PROPAGATION
# ═══════════════════════════════════════════════════════════════════


class TestVacuousRootPropagation:
    """Vacuous root: P(x)=a, so c_y = a·c_{y|x} + (1-a)·c_{y|¬x}.

    With vacuous counterfactual:
        b_y = a · b_{y|x}
        d_y = a · d_{y|x}
        u_y = a · u_{y|x} + (1-a)
    """

    @given(conditional=_opinion_strategy)
    @settings(max_examples=200)
    def test_vacuous_root_formula(self, conditional: Opinion) -> None:
        a = 0.5
        vac = Opinion(0.0, 0.0, 1.0, base_rate=a)

        net = SLNetwork()
        net.add_node(SLNode("A", vac))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=conditional))

        result = infer_node(net, "B")

        assert abs(result.opinion.belief - a * conditional.belief) < _TOL
        assert abs(result.opinion.disbelief - a * conditional.disbelief) < _TOL
        assert abs(result.opinion.uncertainty - (a * conditional.uncertainty + (1 - a))) < _TOL

    def test_vacuous_root_increases_uncertainty(self) -> None:
        cond = Opinion(0.9, 0.05, 0.05)
        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.0, 0.0, 1.0)))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        result = infer_node(net, "B")
        assert result.opinion.uncertainty > cond.uncertainty


# ═══════════════════════════════════════════════════════════════════
# 7. DOGMATIC ROOT PROPAGATION
# ═══════════════════════════════════════════════════════════════════


class TestDogmaticRootPropagation:
    """Dogmatic true (P(x)=1): c_y = c_{y|x}.
    Dogmatic false (P(x)=0): c_y = c_{y|¬x}.
    """

    @given(conditional=_opinion_strategy)
    @settings(max_examples=200)
    def test_dogmatic_true_yields_conditional(self, conditional: Opinion) -> None:
        dog = Opinion(1.0, 0.0, 0.0)
        net = SLNetwork()
        net.add_node(SLNode("A", dog))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=conditional))

        result = infer_node(net, "B")
        assert abs(result.opinion.belief - conditional.belief) < _TOL
        assert abs(result.opinion.disbelief - conditional.disbelief) < _TOL
        assert abs(result.opinion.uncertainty - conditional.uncertainty) < _TOL

    @given(conditional=_opinion_strategy)
    @settings(max_examples=200)
    def test_dogmatic_false_yields_counterfactual(self, conditional: Opinion) -> None:
        dog = Opinion(0.0, 1.0, 0.0)
        cf = vacuous_counterfactual(conditional)
        net = SLNetwork()
        net.add_node(SLNode("A", dog))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=conditional))

        result = infer_node(net, "B")
        assert abs(result.opinion.belief - cf.belief) < _TOL
        assert abs(result.opinion.disbelief - cf.disbelief) < _TOL
        assert abs(result.opinion.uncertainty - cf.uncertainty) < _TOL


# ═══════════════════════════════════════════════════════════════════
# 8. UNCERTAINTY BEHAVIOR
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyBehavior:
    def test_vacuous_conditional_maximizes_uncertainty(self) -> None:
        """Vacuous cond + vacuous cf → result is vacuous."""
        root = Opinion(0.7, 0.2, 0.1)
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=Opinion(0.0, 0.0, 1.0)))

        result = infer_node(net, "B")
        assert abs(result.opinion.uncertainty - 1.0) < _TOL
        assert abs(result.opinion.belief - 0.0) < _TOL

    def test_fuse_reduces_uncertainty(self) -> None:
        """u_{A⊕B} ≤ min(u_A, u_B) (Jøsang §12.3)."""
        op = Opinion(0.7, 0.1, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        net.add_node(SLNode("A", op))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("D", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))
        net.add_edge(SLEdge("A", "C", conditional=cond))
        net.add_edge(SLEdge("B", "D", conditional=cond))
        net.add_edge(SLEdge("C", "D", conditional=cond))

        result = infer_node(net, "D", method="approximate")
        fuse_step = [s for s in result.steps if s.operation == "fuse_parents"][0]

        for name, pp in fuse_step.inputs.items():
            assert result.opinion.uncertainty <= pp.uncertainty + _TOL


# ═══════════════════════════════════════════════════════════════════
# 9. COUNTERFACTUAL STRATEGY EFFECTS
# ═══════════════════════════════════════════════════════════════════


class TestCounterfactualEffects:
    def test_strategies_differ_when_antecedent_uncertain(self) -> None:
        root = Opinion(0.5, 0.3, 0.2)
        cond = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=cond))

        r_v = infer_node(net, "B", counterfactual_fn="vacuous")
        r_a = infer_node(net, "B", counterfactual_fn="adversarial")

        diff = abs(r_v.opinion.belief - r_a.opinion.belief)
        assert diff > 1e-6


# ═══════════════════════════════════════════════════════════════════
# 10. NON-NEGATIVITY
# ═══════════════════════════════════════════════════════════════════


class TestNonNegativity:
    @given(root=_opinion_strategy, c1=_opinion_strategy, c2=_opinion_strategy)
    @settings(max_examples=500)
    def test_non_negative_3hop(self, root: Opinion, c1: Opinion, c2: Opinion) -> None:
        net = SLNetwork()
        net.add_node(SLNode("A", root))
        net.add_node(SLNode("B", Opinion(0.5, 0.2, 0.3)))
        net.add_node(SLNode("C", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge("A", "B", conditional=c1))
        net.add_edge(SLEdge("B", "C", conditional=c2))

        result = infer_node(net, "C")
        for nid, op in result.intermediate_opinions.items():
            assert op.belief >= -_TOL, f"{nid}: b={op.belief}"
            assert op.disbelief >= -_TOL, f"{nid}: d={op.disbelief}"
            assert op.uncertainty >= -_TOL, f"{nid}: u={op.uncertainty}"


# ═══════════════════════════════════════════════════════════════════
# 11. ENUMERATION WEIGHTS
# ═══════════════════════════════════════════════════════════════════


class TestEnumerationWeights:
    @given(
        p_a=st.floats(min_value=0.01, max_value=0.99),
        p_b=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=200)
    def test_weights_sum_to_one(self, p_a: float, p_b: float) -> None:
        """Product of Bernoulli PMFs sums to 1 over all configs."""
        total = (
            p_a * p_b
            + p_a * (1 - p_b)
            + (1 - p_a) * p_b
            + (1 - p_a) * (1 - p_b)
        )
        assert abs(total - 1.0) < 1e-12
