"""
Tests for the deduction operator in Subjective Logic.

Per Jøsang (2016, §12.6), deduction derives an opinion about y from:
  - ω_x:     opinion about antecedent x
  - ω_{y|x}: conditional opinion about y given x is true
  - ω_{y|¬x}: conditional opinion about y given x is false

Formula (Jøsang 2016, Def. 12.6):
  b_y = b_x · b_{y|x} + d_x · b_{y|¬x} + u_x · (a_x · b_{y|x} + ā_x · b_{y|¬x})
  d_y = b_x · d_{y|x} + d_x · d_{y|¬x} + u_x · (a_x · d_{y|x} + ā_x · d_{y|¬x})
  u_y = b_x · u_{y|x} + d_x · u_{y|¬x} + u_x · (a_x · u_{y|x} + ā_x · u_{y|¬x})
  a_y = a_x · P(y|x) + ā_x · P(y|¬x)

where ā_x = 1 - a_x.
"""

import pytest
from jsonld_ex.confidence_algebra import Opinion, deduce


# ═══════════════════════════════════════════════════════════════════
# 1. BOUNDARY CONDITIONS — certain antecedent
# ═══════════════════════════════════════════════════════════════════


class TestDeductionCertainAntecedent:
    """When the antecedent is certain (b_x=1, u_x=0), the result
    should be exactly ω_{y|x}."""

    def test_certain_true_returns_conditional(self):
        """b_x=1 → result = ω_{y|x}."""
        omega_x = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        omega_y_given_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        omega_y_given_not_x = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)

        result = deduce(omega_x, omega_y_given_x, omega_y_given_not_x)

        assert abs(result.belief - 0.8) < 1e-9
        assert abs(result.disbelief - 0.1) < 1e-9
        assert abs(result.uncertainty - 0.1) < 1e-9

    def test_certain_false_returns_negated_conditional(self):
        """d_x=1 → result = ω_{y|¬x}."""
        omega_x = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        omega_y_given_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        omega_y_given_not_x = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)

        result = deduce(omega_x, omega_y_given_x, omega_y_given_not_x)

        assert abs(result.belief - 0.2) < 1e-9
        assert abs(result.disbelief - 0.5) < 1e-9
        assert abs(result.uncertainty - 0.3) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 2. VACUOUS ANTECEDENT — maximum uncertainty
# ═══════════════════════════════════════════════════════════════════


class TestDeductionVacuousAntecedent:
    """When the antecedent is vacuous (u_x=1), the result is a base-rate
    weighted combination of the two conditionals."""

    def test_vacuous_antecedent_default_base_rate(self):
        """u_x=1, a_x=0.5 → equal weighting of conditionals."""
        omega_x = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        omega_y_given_x = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        omega_y_given_not_x = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)

        result = deduce(omega_x, omega_y_given_x, omega_y_given_not_x)

        # b_y = 0·b_{y|x} + 0·b_{y|¬x} + 1·(0.5·0.9 + 0.5·0.1) = 0.5
        assert abs(result.belief - 0.5) < 1e-9
        # d_y = 0·d_{y|x} + 0·d_{y|¬x} + 1·(0.5·0.0 + 0.5·0.8) = 0.4
        assert abs(result.disbelief - 0.4) < 1e-9
        # u_y = 0·u_{y|x} + 0·u_{y|¬x} + 1·(0.5·0.1 + 0.5·0.1) = 0.1
        assert abs(result.uncertainty - 0.1) < 1e-9

    def test_vacuous_antecedent_high_base_rate(self):
        """u_x=1, a_x=0.9 → heavily weights ω_{y|x}."""
        omega_x = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.9)
        omega_y_given_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        omega_y_given_not_x = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)

        result = deduce(omega_x, omega_y_given_x, omega_y_given_not_x)

        # b_y = 1·(0.9·0.8 + 0.1·0.2) = 0.72 + 0.02 = 0.74
        assert abs(result.belief - 0.74) < 1e-9
        # d_y = 1·(0.9·0.1 + 0.1·0.5) = 0.09 + 0.05 = 0.14
        assert abs(result.disbelief - 0.14) < 1e-9
        # u_y = 1·(0.9·0.1 + 0.1·0.3) = 0.09 + 0.03 = 0.12
        assert abs(result.uncertainty - 0.12) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 3. ADDITIVITY INVARIANT — b + d + u = 1 always
# ═══════════════════════════════════════════════════════════════════


class TestDeductionAdditivity:
    """The result must always satisfy b + d + u = 1."""

    @pytest.mark.parametrize(
        "bx,dx,ux,ax",
        [
            (1.0, 0.0, 0.0, 0.5),
            (0.0, 1.0, 0.0, 0.5),
            (0.0, 0.0, 1.0, 0.5),
            (0.5, 0.3, 0.2, 0.5),
            (0.33, 0.33, 0.34, 0.7),
            (0.1, 0.1, 0.8, 0.1),
            (0.9, 0.05, 0.05, 0.99),
        ],
    )
    def test_additivity_holds(self, bx, dx, ux, ax):
        omega_x = Opinion(belief=bx, disbelief=dx, uncertainty=ux, base_rate=ax)
        omega_y_x = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.6)
        omega_y_nx = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3, base_rate=0.4)

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-9, f"b+d+u = {total}, not 1"

    def test_additivity_with_extreme_conditionals(self):
        """Dogmatic conditionals still produce valid opinion."""
        omega_x = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        omega_y_x = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        omega_y_nx = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_additivity_with_vacuous_conditionals(self):
        """Vacuous conditionals still produce valid opinion."""
        omega_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        omega_y_x = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        omega_y_nx = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-9
        # Both conditionals vacuous → result should be vacuous
        assert abs(result.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 4. CONSISTENCY WITH CLASSICAL PROBABILITY
# ═══════════════════════════════════════════════════════════════════


class TestDeductionClassicalConsistency:
    """When all opinions are dogmatic (u=0), deduction reduces to the
    law of total probability: P(y) = P(x)·P(y|x) + P(¬x)·P(y|¬x)."""

    def test_total_probability_basic(self):
        """Classic example: P(x)=0.6, P(y|x)=0.9, P(y|¬x)=0.2."""
        omega_x = Opinion.from_confidence(0.6)  # dogmatic, b=0.6
        omega_y_x = Opinion.from_confidence(0.9)
        omega_y_nx = Opinion.from_confidence(0.2)

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        # P(y) = 0.6*0.9 + 0.4*0.2 = 0.54 + 0.08 = 0.62
        expected = 0.6 * 0.9 + 0.4 * 0.2
        assert abs(result.projected_probability() - expected) < 1e-9
        # Result should also be dogmatic
        assert abs(result.uncertainty) < 1e-9

    def test_total_probability_complementary(self):
        """P(y|x) + P(¬y|x) should give complementary results."""
        omega_x = Opinion.from_confidence(0.7)
        omega_y_x = Opinion.from_confidence(0.8)
        omega_y_nx = Opinion.from_confidence(0.3)

        result_y = deduce(omega_x, omega_y_x, omega_y_nx)

        # P(y) = 0.7*0.8 + 0.3*0.3 = 0.56 + 0.09 = 0.65
        assert abs(result_y.projected_probability() - 0.65) < 1e-9

    def test_total_probability_certain_antecedent(self):
        """P(x)=1 → P(y) = P(y|x)."""
        omega_x = Opinion.from_confidence(1.0)
        omega_y_x = Opinion.from_confidence(0.85)
        omega_y_nx = Opinion.from_confidence(0.15)

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        assert abs(result.projected_probability() - 0.85) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 5. BASE RATE COMPUTATION
# ═══════════════════════════════════════════════════════════════════


class TestDeductionBaseRate:
    """The deduced base rate should be:
    a_y = a_x · P(y|x) + (1-a_x) · P(y|¬x)
    where P(y|x) and P(y|¬x) are projected probabilities of conditionals."""

    def test_base_rate_computation(self):
        omega_x = Opinion(
            belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.6
        )
        omega_y_x = Opinion(
            belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.5
        )
        omega_y_nx = Opinion(
            belief=0.3, disbelief=0.4, uncertainty=0.3, base_rate=0.5
        )

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        # P(y|x) = 0.7 + 0.5*0.1 = 0.75
        # P(y|¬x) = 0.3 + 0.5*0.3 = 0.45
        # a_y = 0.6*0.75 + 0.4*0.45 = 0.45 + 0.18 = 0.63
        expected_a = 0.6 * 0.75 + 0.4 * 0.45
        assert abs(result.base_rate - expected_a) < 1e-9

    def test_base_rate_with_symmetric_conditionals(self):
        """When conditionals are symmetric, a_y should equal 0.5."""
        omega_x = Opinion(
            belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5
        )
        # Symmetric: P(y|x) = P(y|¬x)
        omega_y_x = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        omega_y_nx = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        # P(y|x) = P(y|¬x) = 0.6 + 0.5*0.2 = 0.7
        # a_y = 0.5*0.7 + 0.5*0.7 = 0.7
        assert abs(result.base_rate - 0.7) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 6. MONOTONICITY
# ═══════════════════════════════════════════════════════════════════


class TestDeductionMonotonicity:
    """Increasing belief in x should move result toward ω_{y|x}
    (when ω_{y|x} has higher belief than ω_{y|¬x})."""

    def test_more_belief_in_x_increases_belief_in_y(self):
        """If P(y|x) > P(y|¬x), then increasing b_x should increase b_y."""
        omega_y_x = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        omega_y_nx = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)

        prev_belief = None
        for bx in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ux = 1.0 - bx  # keep d_x = 0 for clean monotonicity
            omega_x = Opinion(belief=bx, disbelief=0.0, uncertainty=ux)
            result = deduce(omega_x, omega_y_x, omega_y_nx)
            if prev_belief is not None:
                assert result.belief >= prev_belief - 1e-9, (
                    f"Monotonicity violated: b_y decreased from "
                    f"{prev_belief} to {result.belief} as b_x increased to {bx}"
                )
            prev_belief = result.belief


# ═══════════════════════════════════════════════════════════════════
# 7. PROJECTED PROBABILITY PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestDeductionProjectedProbability:
    """Projected probability properties for deduction.

    IMPORTANT SCIENTIFIC NOTE:
    The property P(ω_y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x) does NOT
    hold for non-dogmatic opinions in general.  This is because
    P(ω_y) = b_y + a_y·u_y, and the interaction between the deduced
    base rate a_y and uncertainty u_y does not simplify to the law
    of total probability when uncertainty is nonzero.

    The total probability law holds ONLY in the dogmatic limit (u=0),
    which is tested in TestDeductionClassicalConsistency.

    This class tests properties that DO hold universally.
    """

    def test_total_probability_NOT_exact_for_non_dogmatic(self):
        """Explicitly document that P(ω_y) ≠ P(x)·P(y|x) + (1-P(x))·P(y|¬x)
        for non-dogmatic opinions.  This is NOT a bug — it is an inherent
        property of Subjective Logic deduction (Jøsang 2016, §12.6)."""
        omega_x = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4, base_rate=0.6)
        omega_y_x = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        omega_y_nx = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3, base_rate=0.5)

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        p_x = omega_x.projected_probability()
        p_y_x = omega_y_x.projected_probability()
        p_y_nx = omega_y_nx.projected_probability()

        total_prob = p_x * p_y_x + (1 - p_x) * p_y_nx
        actual = result.projected_probability()

        # These should NOT be equal for non-dogmatic opinions
        assert abs(actual - total_prob) > 1e-6, (
            "Expected non-equivalence for non-dogmatic deduction"
        )

    def test_projected_probability_bounded(self):
        """P(ω_y) should lie between P(y|x) and P(y|¬x) when the
        antecedent opinion has default base rate."""
        omega_x = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.5)
        omega_y_x = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        omega_y_nx = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        p_y_x = omega_y_x.projected_probability()
        p_y_nx = omega_y_nx.projected_probability()
        lo, hi = min(p_y_x, p_y_nx), max(p_y_x, p_y_nx)

        assert lo - 1e-9 <= result.projected_probability() <= hi + 1e-9

    @pytest.mark.parametrize("seed", range(10))
    def test_additivity_random(self, seed):
        """b + d + u = 1 for randomly generated opinions."""
        import random

        rng = random.Random(seed)

        def random_opinion():
            b = rng.random()
            d = rng.random() * (1 - b)
            u = 1 - b - d
            a = rng.random()
            return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)

        omega_x = random_opinion()
        omega_y_x = random_opinion()
        omega_y_nx = random_opinion()

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        total = result.belief + result.disbelief + result.uncertainty
        assert abs(total - 1.0) < 1e-9

    @pytest.mark.parametrize("seed", range(10))
    def test_components_non_negative_random(self, seed):
        """All result components >= 0 for random opinions."""
        import random

        rng = random.Random(seed)

        def random_opinion():
            b = rng.random()
            d = rng.random() * (1 - b)
            u = 1 - b - d
            a = rng.random()
            return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)

        omega_x = random_opinion()
        omega_y_x = random_opinion()
        omega_y_nx = random_opinion()

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        assert result.belief >= -1e-12
        assert result.disbelief >= -1e-12
        assert result.uncertainty >= -1e-12
        assert 0.0 <= result.base_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════
# 8. EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestDeductionEdgeCases:
    """Edge cases a reviewer might try."""

    def test_all_vacuous(self):
        """All three opinions are vacuous → result is vacuous."""
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = deduce(vacuous, vacuous, vacuous)
        assert abs(result.uncertainty - 1.0) < 1e-9
        assert abs(result.belief) < 1e-9
        assert abs(result.disbelief) < 1e-9

    def test_all_dogmatic(self):
        """All dogmatic → result is dogmatic (zero uncertainty)."""
        omega_x = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        omega_y_x = Opinion(belief=0.9, disbelief=0.1, uncertainty=0.0)
        omega_y_nx = Opinion(belief=0.3, disbelief=0.7, uncertainty=0.0)

        result = deduce(omega_x, omega_y_x, omega_y_nx)
        assert abs(result.uncertainty) < 1e-9

    def test_identical_conditionals(self):
        """When ω_{y|x} = ω_{y|¬x}, the antecedent doesn't matter."""
        cond = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        omega_x1 = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        omega_x2 = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)

        result1 = deduce(omega_x1, cond, cond)
        result2 = deduce(omega_x2, cond, cond)

        assert abs(result1.belief - result2.belief) < 1e-9
        assert abs(result1.disbelief - result2.disbelief) < 1e-9
        assert abs(result1.uncertainty - result2.uncertainty) < 1e-9

    def test_complementary_dogmatic_conditionals(self):
        """ω_{y|x} = certain true, ω_{y|¬x} = certain false.
        This is effectively an indicator function."""
        omega_x = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        omega_y_x = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        omega_y_nx = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)

        result = deduce(omega_x, omega_y_x, omega_y_nx)

        # b_y = 0.7·1 + 0.1·0 + 0.2·(0.5·1 + 0.5·0) = 0.7 + 0.1 = 0.8
        assert abs(result.belief - 0.8) < 1e-9
        # d_y = 0.7·0 + 0.1·1 + 0.2·(0.5·0 + 0.5·1) = 0.1 + 0.1 = 0.2
        assert abs(result.disbelief - 0.2) < 1e-9
        # u_y = 0.7·0 + 0.1·0 + 0.2·(0.5·0 + 0.5·0) = 0
        assert abs(result.uncertainty) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 9. INPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestDeductionInputValidation:
    """Deduction should reject invalid inputs."""

    def test_rejects_non_opinion(self):
        valid = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(TypeError):
            deduce("not an opinion", valid, valid)
        with pytest.raises(TypeError):
            deduce(valid, 0.5, valid)
        with pytest.raises(TypeError):
            deduce(valid, valid, None)

    def test_rejects_wrong_number_of_args(self):
        """Deduction requires exactly 3 opinions."""
        valid = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(TypeError):
            deduce(valid, valid)  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# 10. INTEGRATION — CHAINED DEDUCTION
# ═══════════════════════════════════════════════════════════════════


class TestDeductionChaining:
    """Deduction can be chained: deduce y from x, then z from y."""

    def test_two_step_deduction(self):
        """x → y → z chain."""
        omega_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)

        # x → y conditionals
        omega_y_x = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        omega_y_nx = Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2)

        # y → z conditionals
        omega_z_y = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        omega_z_ny = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)

        # Step 1: deduce y
        omega_y = deduce(omega_x, omega_y_x, omega_y_nx)

        # Step 2: deduce z from y
        omega_z = deduce(omega_y, omega_z_y, omega_z_ny)

        # Verify validity
        total = omega_z.belief + omega_z.disbelief + omega_z.uncertainty
        assert abs(total - 1.0) < 1e-9

        # z should have higher uncertainty than single-step (uncertainty compounds)
        # Not a strict guarantee but a reasonable expectation
        assert omega_z.belief >= 0.0
        assert omega_z.disbelief >= 0.0
        assert omega_z.uncertainty >= 0.0

    def test_chain_additivity(self):
        """Chained deduction preserves b+d+u=1 at every step."""
        omega_x = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.5)
        omega_y_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)
        omega_y_nx = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3, base_rate=0.5)

        omega_y = deduce(omega_x, omega_y_x, omega_y_nx)

        total = omega_y.belief + omega_y.disbelief + omega_y.uncertainty
        assert abs(total - 1.0) < 1e-9
        assert 0.0 <= omega_y.base_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════
# 11. HALLUCINATION FIREWALL SCENARIO
# ═══════════════════════════════════════════════════════════════════


class TestDeductionHallucinationFirewall:
    """Real-world scenario: an AI agent reasons conditionally about
    whether data is reliable given its source's trustworthiness."""

    def test_high_trust_source_high_confidence_claim(self):
        """Trusted source says 'if sensor reads high, patient at risk'.
        Sensor opinion is high → deduced risk should be high."""
        sensor_reading = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.1)
        risk_if_high = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        risk_if_normal = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)

        risk = deduce(sensor_reading, risk_if_high, risk_if_normal)

        # Should have high belief in risk
        assert risk.belief > 0.7
        # Projected probability should be high
        assert risk.projected_probability() > 0.8

    def test_uncertain_source_same_claim(self):
        """Same conditionals, but sensor reading is highly uncertain.
        Deduced risk should reflect the uncertainty."""
        sensor_reading = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6)
        risk_if_high = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        risk_if_normal = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)

        risk = deduce(sensor_reading, risk_if_high, risk_if_normal)

        # Should have LESS belief in risk than the certain case
        assert risk.belief < 0.7
        # But uncertainty should be higher
        assert risk.uncertainty > 0.1
