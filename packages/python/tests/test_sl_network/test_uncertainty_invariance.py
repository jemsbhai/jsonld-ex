"""Property tests encoding the Uncertainty Invariance Theorem for SL deduction.

These tests encode four mathematical findings discovered during Phase D
(DBN interop) development:

Finding 1 — Uncertainty Invariance Theorem (Binomial):
    When u_{Y|x} = u_{Y|~x} = u_c, the deduced uncertainty u_Y = u_c
    regardless of the parent opinion omega_X.

    Proof:
        u_Y = b_X * u_{Y|x} + d_X * u_{Y|~x}
            + u_X * (a_X * u_{Y|x} + (1 - a_X) * u_{Y|~x})
        When u_{Y|x} = u_{Y|~x} = u_c:
            u_Y = (b_X + d_X) * u_c + u_X * (a_X + (1 - a_X)) * u_c
                = (b_X + d_X + u_X) * u_c = 1 * u_c = u_c   [by additivity]

Finding 2 — Uncertainty Invariance Theorem (Multinomial):
    The invariance extends to multinomial deduction. When all conditional
    opinions share the same uncertainty u_c:
        u_Y = sum_i b_X(x_i) * u_c + u_X * sum_i a_X(x_i) * u_c
            = (sum_i b_X(x_i) + u_X) * u_c = 1 * u_c = u_c

Finding 3 — Projected-Probability-Weighted Uncertainty Average:
    In the general heterogeneous case:
        u_Y = sum_i P_X(x_i) * u_{Y|x_i}
    Deduced uncertainty is the projected-probability-weighted average
    of conditional uncertainties. (This is the component-wise LTP
    applied to the uncertainty component.)

Finding 4 — Chain Convergence:
    Repeated deduction with fixed heterogeneous conditionals converges
    to a fixed point (P_inf, u_inf) where u_inf = P_inf * u_{Y|x}
    + (1 - P_inf) * u_{Y|~x}.

These findings have implications for the NeurIPS paper:
    - The SL-vs-BN expressiveness divergence does NOT arise from
      "uncertainty compounding through chains" under homogeneous evidence.
    - SL's advantage arises from: (a) belief mass redistribution through
      deduction, (b) heterogeneous evidence quality across network edges,
      and (c) preservation of the (b, d, u) triple that BN collapses.

References:
    Jøsang, A. (2016). Subjective Logic, §12.6 (Deduction Operator).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import Opinion, deduce
from jsonld_ex.multinomial_algebra import MultinomialOpinion, multinomial_deduce


# ═══════════════════════════════════════════════════════════════════
# FINDING 1: BINOMIAL UNCERTAINTY INVARIANCE
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyInvarianceBinomial:
    """When all conditional uncertainties equal u_c, deduced u = u_c."""

    @given(
        b=st.floats(min_value=0.0, max_value=1.0),
        u_parent=st.floats(min_value=0.0, max_value=1.0),
        a=st.floats(min_value=0.01, max_value=0.99),
        u_c=st.floats(min_value=0.01, max_value=0.99),
        b_cond_ratio=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=200)
    def test_invariance_property_based(
        self, b: float, u_parent: float, a: float, u_c: float,
        b_cond_ratio: float,
    ) -> None:
        """For arbitrary parent opinions, u_Y = u_c when conditionals
        share the same uncertainty u_c.

        Property-based test using Hypothesis.
        """
        # Build valid parent opinion
        d = 1.0 - b - u_parent
        assume(d >= 0.0)
        assume(abs(b + d + u_parent - 1.0) < 1e-9)
        parent = Opinion(b, d, u_parent, base_rate=a)

        # Build conditional and counterfactual with equal u_c
        b_cond = b_cond_ratio * (1.0 - u_c)
        d_cond = (1.0 - b_cond_ratio) * (1.0 - u_c)
        assume(b_cond >= 0.0 and d_cond >= 0.0)

        cond = Opinion(b_cond, d_cond, u_c)
        cf = Opinion(d_cond, b_cond, u_c)  # Swapped b/d

        result = deduce(parent, cond, cf)
        assert abs(result.uncertainty - u_c) < 1e-9, (
            f"Expected u_Y = u_c = {u_c}, got {result.uncertainty}"
        )

    def test_invariance_dogmatic_parent(self) -> None:
        """Dogmatic parent (u=0) still yields u_Y = u_c."""
        parent = Opinion(0.8, 0.2, 0.0)
        cond = Opinion(0.63, 0.27, 0.1)
        cf = Opinion(0.27, 0.63, 0.1)
        result = deduce(parent, cond, cf)
        assert abs(result.uncertainty - 0.1) < 1e-9

    def test_invariance_vacuous_parent(self) -> None:
        """Vacuous parent (u=1) still yields u_Y = u_c."""
        parent = Opinion(0.0, 0.0, 1.0)
        cond = Opinion(0.42, 0.18, 0.4)
        cf = Opinion(0.18, 0.42, 0.4)
        result = deduce(parent, cond, cf)
        assert abs(result.uncertainty - 0.4) < 1e-9

    def test_invariance_biased_base_rate(self) -> None:
        """Non-default base rate does not break invariance."""
        parent = Opinion(0.3, 0.3, 0.4, base_rate=0.8)
        cond = Opinion(0.35, 0.40, 0.25)
        cf = Opinion(0.40, 0.35, 0.25)
        result = deduce(parent, cond, cf)
        assert abs(result.uncertainty - 0.25) < 1e-9

    def test_invariance_through_chain(self) -> None:
        """Uncertainty stays exactly u_c through a 5-step deduction chain."""
        u_c = 0.15
        b_c = 0.6 * (1 - u_c)
        d_c = 0.4 * (1 - u_c)
        cond = Opinion(b_c, d_c, u_c)
        cf = Opinion(d_c, b_c, u_c)

        current = Opinion(0.5, 0.3, 0.2)
        for step in range(5):
            current = deduce(current, cond, cf)
            assert abs(current.uncertainty - u_c) < 1e-9, (
                f"Chain step {step + 1}: u = {current.uncertainty}, expected {u_c}"
            )


# ═══════════════════════════════════════════════════════════════════
# FINDING 2: MULTINOMIAL UNCERTAINTY INVARIANCE
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyInvarianceMultinomial:
    """Extends to multinomial deduction with uniform conditional uncertainty."""

    def test_ternary_invariance(self) -> None:
        """Ternary parent, ternary child, all conditionals u = u_c."""
        u_c = 0.2
        parent = MultinomialOpinion(
            beliefs={"A": 0.3, "B": 0.4, "C": 0.02},
            uncertainty=0.28,
            base_rates={"A": 0.4, "B": 0.35, "C": 0.25},
        )

        remaining = 1.0 - u_c
        conditionals = {}
        for state in ["A", "B", "C"]:
            conditionals[state] = MultinomialOpinion(
                beliefs={"X": remaining * 0.5, "Y": remaining * 0.3, "Z": remaining * 0.2},
                uncertainty=u_c,
                base_rates={"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3},
            )

        result = multinomial_deduce(parent, conditionals)
        assert abs(result.uncertainty - u_c) < 1e-9

    def test_quaternary_invariance(self) -> None:
        """4-state parent, 3-state child."""
        u_c = 0.12
        parent = MultinomialOpinion(
            beliefs={"a": 0.2, "b": 0.3, "c": 0.15, "d": 0.1},
            uncertainty=0.25,
            base_rates={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        )

        remaining = 1.0 - u_c
        conditionals = {}
        for state in ["a", "b", "c", "d"]:
            conditionals[state] = MultinomialOpinion(
                beliefs={"X": remaining * 0.4, "Y": remaining * 0.35, "Z": remaining * 0.25},
                uncertainty=u_c,
                base_rates={"X": 1 / 3, "Y": 1 / 3, "Z": 1 / 3},
            )

        result = multinomial_deduce(parent, conditionals)
        assert abs(result.uncertainty - u_c) < 1e-9

    def test_multinomial_vacuous_parent(self) -> None:
        """Vacuous multinomial parent still yields u_Y = u_c."""
        u_c = 0.3
        parent = MultinomialOpinion(
            beliefs={"A": 0.0, "B": 0.0, "C": 0.0},
            uncertainty=1.0,
            base_rates={"A": 0.5, "B": 0.3, "C": 0.2},
        )

        remaining = 1.0 - u_c
        conditionals = {
            s: MultinomialOpinion(
                beliefs={"X": remaining * 0.6, "Y": remaining * 0.4},
                uncertainty=u_c,
                base_rates={"X": 0.5, "Y": 0.5},
            )
            for s in ["A", "B", "C"]
        }

        result = multinomial_deduce(parent, conditionals)
        assert abs(result.uncertainty - u_c) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# FINDING 3: PROJECTED-PROBABILITY-WEIGHTED UNCERTAINTY AVERAGE
# ═══════════════════════════════════════════════════════════════════


class TestWeightedUncertaintyAverage:
    """u_Y = sum_i P_X(x_i) * u_{Y|x_i} for heterogeneous conditionals."""

    @given(
        b=st.floats(min_value=0.0, max_value=0.9),
        u_parent=st.floats(min_value=0.0, max_value=0.9),
        a=st.floats(min_value=0.01, max_value=0.99),
        u_cond=st.floats(min_value=0.01, max_value=0.99),
        u_cf=st.floats(min_value=0.01, max_value=0.99),
    )
    @settings(max_examples=200)
    def test_weighted_average_property(
        self, b: float, u_parent: float, a: float,
        u_cond: float, u_cf: float,
    ) -> None:
        """Deduced uncertainty = P(x)*u_{Y|x} + P(~x)*u_{Y|~x}."""
        d = 1.0 - b - u_parent
        assume(d >= 0.0)
        assume(abs(b + d + u_parent - 1.0) < 1e-9)
        parent = Opinion(b, d, u_parent, base_rate=a)

        # Build conditionals with different uncertainties
        b_c = 0.6 * (1.0 - u_cond)
        d_c = 0.4 * (1.0 - u_cond)
        b_cf = 0.4 * (1.0 - u_cf)
        d_cf = 0.6 * (1.0 - u_cf)

        cond = Opinion(b_c, d_c, u_cond)
        cf = Opinion(b_cf, d_cf, u_cf)

        result = deduce(parent, cond, cf)

        px = parent.projected_probability()
        expected_u = px * u_cond + (1.0 - px) * u_cf
        assert abs(result.uncertainty - expected_u) < 1e-9, (
            f"Expected u_Y = {expected_u}, got {result.uncertainty}"
        )

    def test_concentrated_on_low_uncertainty(self) -> None:
        """Parent favoring the well-evidenced conditional → low u_Y."""
        parent = Opinion(0.85, 0.05, 0.1, base_rate=0.9)
        cond = Opinion(0.693, 0.297, 0.01)   # u = 0.01 (fresh)
        cf = Opinion(0.21, 0.49, 0.3)        # u = 0.30 (stale)

        result = deduce(parent, cond, cf)
        px = parent.projected_probability()  # ~0.94
        expected = px * 0.01 + (1 - px) * 0.3
        assert abs(result.uncertainty - expected) < 1e-9
        assert result.uncertainty < 0.05  # Much less than average of 0.155

    def test_concentrated_on_high_uncertainty(self) -> None:
        """Parent favoring the poorly-evidenced conditional → high u_Y."""
        parent = Opinion(0.05, 0.85, 0.1, base_rate=0.1)
        cond = Opinion(0.693, 0.297, 0.01)
        cf = Opinion(0.21, 0.49, 0.3)

        result = deduce(parent, cond, cf)
        px = parent.projected_probability()  # ~0.06
        expected = px * 0.01 + (1 - px) * 0.3
        assert abs(result.uncertainty - expected) < 1e-9
        assert result.uncertainty > 0.25  # Much more than average of 0.155

    def test_multinomial_weighted_average(self) -> None:
        """Multinomial version: u_Y = sum_i P(x_i) * u_{Y|x_i}."""
        parent = MultinomialOpinion(
            beliefs={"A": 0.5, "B": 0.3, "C": 0.0},
            uncertainty=0.2,
            base_rates={"A": 0.4, "B": 0.35, "C": 0.25},
        )

        # Different uncertainties per conditional
        cond_a = MultinomialOpinion(
            beliefs={"X": 0.45, "Y": 0.45}, uncertainty=0.1,
            base_rates={"X": 0.5, "Y": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"X": 0.35, "Y": 0.35}, uncertainty=0.3,
            base_rates={"X": 0.5, "Y": 0.5},
        )
        cond_c = MultinomialOpinion(
            beliefs={"X": 0.25, "Y": 0.25}, uncertainty=0.5,
            base_rates={"X": 0.5, "Y": 0.5},
        )

        result = multinomial_deduce(
            parent, {"A": cond_a, "B": cond_b, "C": cond_c}
        )

        pp = parent.projected_probability()
        expected_u = pp["A"] * 0.1 + pp["B"] * 0.3 + pp["C"] * 0.5
        assert abs(result.uncertainty - expected_u) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# FINDING 4: CHAIN CONVERGENCE
# ═══════════════════════════════════════════════════════════════════


class TestChainConvergence:
    """Repeated deduction with fixed conditionals converges to a fixed point."""

    def test_convergence_within_20_steps(self) -> None:
        """Chain should converge to within 1e-6 in at most 20 steps."""
        cond = Opinion(0.693, 0.297, 0.01)
        cf = Opinion(0.21, 0.49, 0.3)

        current = Opinion(0.6, 0.3, 0.1)
        for _ in range(20):
            prev = current
            current = deduce(current, cond, cf)

        # Should have converged: consecutive steps nearly identical
        assert abs(current.uncertainty - prev.uncertainty) < 1e-6
        assert abs(current.belief - prev.belief) < 1e-6
        assert abs(current.projected_probability() - prev.projected_probability()) < 1e-6

    def test_fixed_point_satisfies_self_consistency(self) -> None:
        """At the fixed point, deducing again yields the same opinion."""
        cond = Opinion(0.693, 0.297, 0.01)
        cf = Opinion(0.21, 0.49, 0.3)

        # Run to convergence
        current = Opinion(0.5, 0.3, 0.2)
        for _ in range(50):
            current = deduce(current, cond, cf)

        # Apply one more step
        next_step = deduce(current, cond, cf)
        assert abs(next_step.belief - current.belief) < 1e-12
        assert abs(next_step.disbelief - current.disbelief) < 1e-12
        assert abs(next_step.uncertainty - current.uncertainty) < 1e-12

    def test_fixed_point_uncertainty_formula(self) -> None:
        """u_inf = P_inf * u_cond + (1 - P_inf) * u_cf."""
        u_cond = 0.01
        u_cf = 0.3
        cond = Opinion(0.6 * (1 - u_cond), 0.4 * (1 - u_cond), u_cond)
        cf = Opinion(0.4 * (1 - u_cf), 0.6 * (1 - u_cf), u_cf)

        # Converge
        current = Opinion(0.5, 0.3, 0.2)
        for _ in range(50):
            current = deduce(current, cond, cf)

        p_inf = current.projected_probability()
        expected_u = p_inf * u_cond + (1 - p_inf) * u_cf
        assert abs(current.uncertainty - expected_u) < 1e-9

    def test_different_initial_conditions_same_fixed_point(self) -> None:
        """Fixed point is independent of initial parent opinion."""
        cond = Opinion(0.693, 0.297, 0.01)
        cf = Opinion(0.21, 0.49, 0.3)

        starts = [
            Opinion(0.9, 0.05, 0.05),
            Opinion(0.0, 0.0, 1.0),
            Opinion(0.3, 0.6, 0.1),
            Opinion(0.5, 0.5, 0.0),
        ]

        fixed_points = []
        for start in starts:
            current = start
            for _ in range(50):
                current = deduce(current, cond, cf)
            fixed_points.append(current)

        # All should converge to the same fixed point
        ref = fixed_points[0]
        for i, fp in enumerate(fixed_points[1:], 1):
            assert abs(fp.belief - ref.belief) < 1e-9, (
                f"Start {i}: belief {fp.belief} != {ref.belief}"
            )
            assert abs(fp.uncertainty - ref.uncertainty) < 1e-9, (
                f"Start {i}: u {fp.uncertainty} != {ref.uncertainty}"
            )
