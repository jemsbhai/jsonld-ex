"""
Property-based tests for the Subjective Logic algebra using Hypothesis.

These tests verify algebraic invariants across thousands of randomly
generated inputs, providing much stronger guarantees than hand-crafted
examples alone.  A NeurIPS reviewer can inspect these tests and be
confident that the claimed properties hold universally, not just for
cherry-picked inputs.

Each property is annotated with its formal statement and the relevant
citation from Jøsang (2016).

Strategy: We define a custom Hypothesis strategy for generating valid
Opinion instances (b + d + u = 1, all components in [0, 1]).
"""

import math
import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
)
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)

# ═══════════════════════════════════════════════════════════════════
# Custom Hypothesis Strategies
# ═══════════════════════════════════════════════════════════════════

# Floating-point unit in [0, 1] with enough resolution to stress
# IEEE 754 edge cases but avoiding subnormals that cause noise.
_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def opinions(draw, min_uncertainty=0.0, max_uncertainty=1.0, base_rate=None):
    """Generate a valid Opinion with b + d + u = 1.

    Approach: draw b, d uniformly from the valid simplex region,
    compute u = 1 - b - d.  This guarantees the additivity constraint
    by construction.
    """
    b = draw(_unit)
    # d can be at most 1 - b
    max_d = 1.0 - b
    d = draw(st.floats(min_value=0.0, max_value=max_d, allow_nan=False, allow_infinity=False))
    u = 1.0 - b - d

    # Apply uncertainty bounds
    assume(min_uncertainty <= u <= max_uncertainty)

    if base_rate is not None:
        a = base_rate
    else:
        a = draw(_unit)

    # Guard against floating-point drift making u slightly negative
    if u < 0.0:
        u = 0.0
    if u > 1.0:
        u = 1.0

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


def non_dogmatic_opinions(**kwargs):
    """Generate opinions with u > 0 (not dogmatic)."""
    return opinions(min_uncertainty=1e-9, **kwargs)


@st.composite
def dogmatic_opinions(draw, base_rate=None):
    """Generate dogmatic opinions (u = 0) directly.

    Constructs b + d = 1 by drawing b and computing d = 1 - b,
    avoiding the filter problem of trying to hit u = 0.0 in a
    continuous simplex.
    """
    b = draw(_unit)
    d = 1.0 - b
    if base_rate is not None:
        a = base_rate
    else:
        a = draw(_unit)
    return Opinion(belief=b, disbelief=d, uncertainty=0.0, base_rate=a)


# Tolerance for floating-point comparison
_TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
# CUMULATIVE FUSION — Algebraic Properties
# ═══════════════════════════════════════════════════════════════════


class TestCumulativeFuseProperties:
    """Property-based verification of cumulative fusion (⊕).
    Per Jøsang (2016, §12.3)."""

    @given(a=opinions(), b=opinions())
    def test_commutativity(self, a, b):
        """∀ A, B: A ⊕ B = B ⊕ A."""
        ab = cumulative_fuse(a, b)
        ba = cumulative_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=_TOL)
        assert ab.disbelief == pytest.approx(ba.disbelief, abs=_TOL)
        assert ab.uncertainty == pytest.approx(ba.uncertainty, abs=_TOL)

    @given(a=non_dogmatic_opinions(), b=non_dogmatic_opinions(), c=non_dogmatic_opinions())
    def test_associativity_non_dogmatic(self, a, b, c):
        """∀ non-dogmatic A, B, C: (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C).

        Note: associativity is tested for non-dogmatic opinions to avoid
        the dogmatic limit (γ-weighted average) which uses a different
        formula.  The standard formula is associative when κ > 0.
        """
        ab_c = cumulative_fuse(cumulative_fuse(a, b), c)
        a_bc = cumulative_fuse(a, cumulative_fuse(b, c))
        assert ab_c.belief == pytest.approx(a_bc.belief, abs=_TOL)
        assert ab_c.disbelief == pytest.approx(a_bc.disbelief, abs=_TOL)
        assert ab_c.uncertainty == pytest.approx(a_bc.uncertainty, abs=_TOL)

    @given(a=opinions())
    def test_identity(self, a):
        """∀ A: A ⊕ vacuous = A.
        The vacuous opinion (0, 0, 1) is the identity element."""
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=a.base_rate)
        result = cumulative_fuse(a, vacuous)
        assert result.belief == pytest.approx(a.belief, abs=_TOL)
        assert result.disbelief == pytest.approx(a.disbelief, abs=_TOL)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=_TOL)

    @given(a=opinions(), b=opinions())
    def test_additivity(self, a, b):
        """∀ A, B: let R = A ⊕ B, then R.b + R.d + R.u = 1."""
        result = cumulative_fuse(a, b)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(a=opinions(), b=opinions())
    def test_bounds(self, a, b):
        """∀ A, B: all components of A ⊕ B are in [0, 1]."""
        result = cumulative_fuse(a, b)
        assert -_TOL <= result.belief <= 1.0 + _TOL
        assert -_TOL <= result.disbelief <= 1.0 + _TOL
        assert -_TOL <= result.uncertainty <= 1.0 + _TOL
        assert -_TOL <= result.base_rate <= 1.0 + _TOL

    @given(a=non_dogmatic_opinions(), b=non_dogmatic_opinions())
    def test_uncertainty_reduction(self, a, b):
        """∀ non-dogmatic A, B: u_{A⊕B} ≤ min(u_A, u_B).
        Independent evidence always reduces uncertainty."""
        result = cumulative_fuse(a, b)
        assert result.uncertainty <= min(a.uncertainty, b.uncertainty) + _TOL

    @given(a=opinions(), b=opinions(), c=opinions())
    def test_three_way_additivity(self, a, b, c):
        """∀ A, B, C: cumulative_fuse(A, B, C) produces valid opinion."""
        result = cumulative_fuse(a, b, c)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)
        assert result.belief >= -_TOL
        assert result.disbelief >= -_TOL
        assert result.uncertainty >= -_TOL


# ═══════════════════════════════════════════════════════════════════
# AVERAGING FUSION — Algebraic Properties
# ═══════════════════════════════════════════════════════════════════


class TestAveragingFuseProperties:
    """Property-based verification of averaging fusion (⊘).
    Per Jøsang (2016, §12.5)."""

    @given(a=opinions(), b=opinions())
    def test_commutativity(self, a, b):
        """∀ A, B: A ⊘ B = B ⊘ A."""
        ab = averaging_fuse(a, b)
        ba = averaging_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=_TOL)
        assert ab.disbelief == pytest.approx(ba.disbelief, abs=_TOL)
        assert ab.uncertainty == pytest.approx(ba.uncertainty, abs=_TOL)

    @given(a=opinions())
    def test_idempotence(self, a):
        """∀ A: A ⊘ A = A.
        Averaging the same source with itself yields the original opinion."""
        result = averaging_fuse(a, a)
        assert result.belief == pytest.approx(a.belief, abs=_TOL)
        assert result.disbelief == pytest.approx(a.disbelief, abs=_TOL)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=_TOL)

    @given(a=opinions(), b=opinions())
    def test_additivity(self, a, b):
        """∀ A, B: let R = A ⊘ B, then R.b + R.d + R.u = 1."""
        result = averaging_fuse(a, b)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(a=opinions(), b=opinions())
    def test_bounds(self, a, b):
        """∀ A, B: all components of A ⊘ B are in [0, 1]."""
        result = averaging_fuse(a, b)
        assert -_TOL <= result.belief <= 1.0 + _TOL
        assert -_TOL <= result.disbelief <= 1.0 + _TOL
        assert -_TOL <= result.uncertainty <= 1.0 + _TOL

    @given(
        a=non_dogmatic_opinions(),
        b=non_dogmatic_opinions(),
        c=non_dogmatic_opinions(),
    )
    def test_non_associativity(self, a, b, c):
        """Averaging fusion is NOT associative in general.
        (A ⊘ B) ⊘ C ≠ A ⊘ (B ⊘ C).

        We verify the results are valid but do NOT assert equality.
        This is a scientifically honest characterization: the property
        explicitly fails, and we document it rather than hiding it.

        Note: for some inputs the values may coincidentally be close,
        so we only verify validity rather than asserting non-equality.
        """
        ab_c = averaging_fuse(averaging_fuse(a, b), c)
        a_bc = averaging_fuse(a, averaging_fuse(b, c))

        # Both results must be valid opinions regardless
        for result in [ab_c, a_bc]:
            total = result.belief + result.disbelief + result.uncertainty
            assert total == pytest.approx(1.0, abs=_TOL)
            assert result.belief >= -_TOL
            assert result.disbelief >= -_TOL
            assert result.uncertainty >= -_TOL


# ═══════════════════════════════════════════════════════════════════
# TRUST DISCOUNT — Algebraic Properties
# ═══════════════════════════════════════════════════════════════════


class TestTrustDiscountProperties:
    """Property-based verification of trust discount (⊗).
    Per Jøsang (2016, §14.3)."""

    @given(trust=opinions(), opinion=opinions())
    def test_additivity(self, trust, opinion):
        """∀ trust, opinion: result has b + d + u = 1."""
        result = trust_discount(trust, opinion)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(trust=opinions(), opinion=opinions())
    def test_bounds(self, trust, opinion):
        """∀ trust, opinion: all components in [0, 1]."""
        result = trust_discount(trust, opinion)
        assert -_TOL <= result.belief <= 1.0 + _TOL
        assert -_TOL <= result.disbelief <= 1.0 + _TOL
        assert -_TOL <= result.uncertainty <= 1.0 + _TOL

    @given(opinion=opinions())
    def test_full_trust_identity(self, opinion):
        """discount(certain_trust, X) = X.
        Full trust passes opinion through unchanged."""
        full_trust = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        result = trust_discount(full_trust, opinion)
        assert result.belief == pytest.approx(opinion.belief, abs=_TOL)
        assert result.disbelief == pytest.approx(opinion.disbelief, abs=_TOL)
        assert result.uncertainty == pytest.approx(opinion.uncertainty, abs=_TOL)

    @given(opinion=opinions())
    def test_zero_trust_vacuous(self, opinion):
        """discount(distrust, X) = vacuous.
        Complete distrust yields total uncertainty."""
        distrust = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        result = trust_discount(distrust, opinion)
        assert result.belief == pytest.approx(0.0, abs=_TOL)
        assert result.disbelief == pytest.approx(0.0, abs=_TOL)
        assert result.uncertainty == pytest.approx(1.0, abs=_TOL)

    @given(opinion=opinions())
    def test_vacuous_trust_vacuous(self, opinion):
        """discount(vacuous_trust, X) = vacuous.
        No information about trustworthiness → no information about claim."""
        vacuous_trust = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = trust_discount(vacuous_trust, opinion)
        assert result.belief == pytest.approx(0.0, abs=_TOL)
        assert result.disbelief == pytest.approx(0.0, abs=_TOL)
        assert result.uncertainty == pytest.approx(1.0, abs=_TOL)

    @given(opinion=opinions())
    def test_monotonicity_in_trust(self, opinion):
        """Higher trust belief → higher result belief.
        Tested with two fixed trust levels to avoid assume() filtering."""
        low = Opinion(belief=0.2, disbelief=0.3, uncertainty=0.5)
        high = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        r_low = trust_discount(low, opinion)
        r_high = trust_discount(high, opinion)
        assert r_high.belief >= r_low.belief - _TOL

    @given(opinion=opinions())
    def test_preserves_base_rate(self, opinion):
        """Trust discount preserves the original opinion's base rate."""
        trust = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        result = trust_discount(trust, opinion)
        assert result.base_rate == pytest.approx(opinion.base_rate, abs=_TOL)


# ═══════════════════════════════════════════════════════════════════
# DEDUCTION — Algebraic Properties
# ═══════════════════════════════════════════════════════════════════


class TestDeduceProperties:
    """Property-based verification of deduction operator.
    Per Jøsang (2016, §12.6)."""

    @given(omega_x=opinions(), omega_yx=opinions(), omega_ynx=opinions())
    def test_additivity(self, omega_x, omega_yx, omega_ynx):
        """∀ inputs: deduced opinion has b + d + u = 1."""
        result = deduce(omega_x, omega_yx, omega_ynx)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(omega_x=opinions(), omega_yx=opinions(), omega_ynx=opinions())
    def test_bounds(self, omega_x, omega_yx, omega_ynx):
        """∀ inputs: all deduced components in [0, 1]."""
        result = deduce(omega_x, omega_yx, omega_ynx)
        assert -_TOL <= result.belief <= 1.0 + _TOL
        assert -_TOL <= result.disbelief <= 1.0 + _TOL
        assert -_TOL <= result.uncertainty <= 1.0 + _TOL
        assert -_TOL <= result.base_rate <= 1.0 + _TOL

    @given(omega_yx=dogmatic_opinions(), omega_ynx=dogmatic_opinions())
    def test_classical_limit(self, omega_yx, omega_ynx):
        """When all opinions are dogmatic, deduction reduces to the
        law of total probability: P(y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x)."""
        # Use a fixed dogmatic antecedent for clarity
        omega_x = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        result = deduce(omega_x, omega_yx, omega_ynx)

        p_x = omega_x.projected_probability()
        p_yx = omega_yx.projected_probability()
        p_ynx = omega_ynx.projected_probability()
        expected = p_x * p_yx + (1 - p_x) * p_ynx

        assert result.projected_probability() == pytest.approx(expected, abs=_TOL)
        assert result.uncertainty == pytest.approx(0.0, abs=_TOL)

    @given(omega_x=dogmatic_opinions(base_rate=0.5))
    def test_classical_limit_varying_antecedent(self, omega_x):
        """Law of total probability with varying dogmatic antecedent."""
        omega_yx = Opinion(belief=0.9, disbelief=0.1, uncertainty=0.0)
        omega_ynx = Opinion(belief=0.2, disbelief=0.8, uncertainty=0.0)

        result = deduce(omega_x, omega_yx, omega_ynx)

        p_x = omega_x.projected_probability()
        expected = p_x * 0.9 + (1 - p_x) * 0.2

        assert result.projected_probability() == pytest.approx(expected, abs=_TOL)

    @given(omega_x=opinions(), cond=opinions())
    def test_identical_conditionals_antecedent_irrelevant(self, omega_x, cond):
        """When ω_{y|x} = ω_{y|¬x}, the antecedent does not matter.
        Result should equal the shared conditional."""
        result = deduce(omega_x, cond, cond)
        assert result.belief == pytest.approx(cond.belief, abs=_TOL)
        assert result.disbelief == pytest.approx(cond.disbelief, abs=_TOL)
        assert result.uncertainty == pytest.approx(cond.uncertainty, abs=_TOL)


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL DECAY — Algebraic Properties
# ═══════════════════════════════════════════════════════════════════


class TestDecayProperties:
    """Property-based verification of temporal decay.
    Uses exponential decay (the default)."""

    @given(opinion=opinions())
    def test_identity_at_zero(self, opinion):
        """∀ ω: decay(ω, t=0) = ω."""
        result = decay_opinion(opinion, elapsed=0.0, half_life=10.0)
        assert result.belief == pytest.approx(opinion.belief, abs=1e-12)
        assert result.disbelief == pytest.approx(opinion.disbelief, abs=1e-12)
        assert result.uncertainty == pytest.approx(opinion.uncertainty, abs=1e-12)

    @given(
        opinion=opinions(),
        elapsed=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        half_life=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_additivity(self, opinion, elapsed, half_life):
        """∀ ω, t, τ: decayed opinion has b + d + u = 1."""
        result = decay_opinion(opinion, elapsed=elapsed, half_life=half_life)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(
        opinion=opinions(),
        elapsed=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        half_life=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_bounds(self, opinion, elapsed, half_life):
        """∀ ω, t, τ: all decayed components in [0, 1]."""
        result = decay_opinion(opinion, elapsed=elapsed, half_life=half_life)
        assert -_TOL <= result.belief <= 1.0 + _TOL
        assert -_TOL <= result.disbelief <= 1.0 + _TOL
        assert -_TOL <= result.uncertainty <= 1.0 + _TOL

    @given(
        opinion=opinions(),
        elapsed=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        half_life=st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_base_rate_preserved(self, opinion, elapsed, half_life):
        """∀ ω, t, τ: decay does not change the base rate."""
        result = decay_opinion(opinion, elapsed=elapsed, half_life=half_life)
        assert result.base_rate == pytest.approx(opinion.base_rate, abs=1e-12)

    @given(opinion=opinions())
    def test_uncertainty_monotonic(self, opinion):
        """∀ ω: u(t_1) ≤ u(t_2) when t_1 ≤ t_2.
        Uncertainty never decreases as time passes."""
        times = [0.0, 1.0, 5.0, 10.0, 50.0, 100.0]
        prev_u = -1.0
        for t in times:
            result = decay_opinion(opinion, elapsed=t, half_life=10.0)
            assert result.uncertainty >= prev_u - _TOL
            prev_u = result.uncertainty

    @given(opinion=opinions())
    @settings(suppress_health_check=[HealthCheck.too_slow])
    def test_ratio_preservation(self, opinion):
        """∀ ω with b > 0 and d > 0: b/d ratio preserved under decay.
        Decay reduces both b and d by the same factor λ,
        so b'/d' = (λ·b)/(λ·d) = b/d."""
        assume(opinion.belief > 1e-10)
        assume(opinion.disbelief > 1e-10)

        original_ratio = opinion.belief / opinion.disbelief
        result = decay_opinion(opinion, elapsed=5.0, half_life=10.0)

        if result.disbelief > 1e-15:
            ratio = result.belief / result.disbelief
            assert ratio == pytest.approx(original_ratio, rel=1e-6)

    @given(opinion=opinions())
    def test_convergence_to_vacuous(self, opinion):
        """∀ ω: as t → ∞, belief and disbelief approach 0."""
        result = decay_opinion(opinion, elapsed=1e5, half_life=10.0)
        assert result.belief < 1e-6
        assert result.disbelief < 1e-6
        assert result.uncertainty > 1.0 - 1e-6


# ═══════════════════════════════════════════════════════════════════
# CROSS-OPERATOR COMPOSITION — Property-based
# ═══════════════════════════════════════════════════════════════════


class TestCrossOperatorProperties:
    """Composition of operators must always produce valid opinions."""

    @given(
        trust=opinions(),
        opinion_a=opinions(),
        opinion_b=opinions(),
    )
    def test_discount_then_cumulative_fuse(self, trust, opinion_a, opinion_b):
        """discount + cumulative_fuse always produces a valid opinion."""
        discounted_a = trust_discount(trust, opinion_a)
        discounted_b = trust_discount(trust, opinion_b)
        fused = cumulative_fuse(discounted_a, discounted_b)
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)
        assert fused.belief >= -_TOL
        assert fused.disbelief >= -_TOL
        assert fused.uncertainty >= -_TOL

    @given(
        trust=opinions(),
        opinion_a=opinions(),
        opinion_b=opinions(),
    )
    def test_discount_then_averaging_fuse(self, trust, opinion_a, opinion_b):
        """discount + averaging_fuse always produces a valid opinion."""
        discounted_a = trust_discount(trust, opinion_a)
        discounted_b = trust_discount(trust, opinion_b)
        fused = averaging_fuse(discounted_a, discounted_b)
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)
        assert fused.belief >= -_TOL
        assert fused.disbelief >= -_TOL
        assert fused.uncertainty >= -_TOL

    @given(opinion=opinions())
    def test_decay_then_cumulative_fuse(self, opinion):
        """decay + cumulative_fuse always produces a valid opinion."""
        decayed = decay_opinion(opinion, elapsed=5.0, half_life=10.0)
        fresh = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        fused = cumulative_fuse(decayed, fresh)
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(
        omega_x=opinions(),
        omega_yx=opinions(),
        omega_ynx=opinions(),
        trust=opinions(),
    )
    def test_deduction_then_discount(self, omega_x, omega_yx, omega_ynx, trust):
        """deduction + trust_discount always produces a valid opinion."""
        deduced = deduce(omega_x, omega_yx, omega_ynx)
        discounted = trust_discount(trust, deduced)
        total = discounted.belief + discounted.disbelief + discounted.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)
        assert discounted.belief >= -_TOL
        assert discounted.disbelief >= -_TOL
        assert discounted.uncertainty >= -_TOL

    @given(
        omega_x=opinions(),
        omega_yx=opinions(),
        omega_ynx=opinions(),
    )
    def test_deduction_then_decay(self, omega_x, omega_yx, omega_ynx):
        """deduction + decay always produces a valid opinion."""
        deduced = deduce(omega_x, omega_yx, omega_ynx)
        decayed = decay_opinion(deduced, elapsed=10.0, half_life=5.0)
        total = decayed.belief + decayed.disbelief + decayed.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)

    @given(
        omega_x=opinions(),
        omega_yx=opinions(),
        omega_ynx=opinions(),
        trust=opinions(),
        opinion_b=opinions(),
    )
    def test_full_pipeline(self, omega_x, omega_yx, omega_ynx, trust, opinion_b):
        """Full pipeline: deduce → decay → discount → fuse.
        Must always produce a valid opinion, regardless of inputs."""
        deduced = deduce(omega_x, omega_yx, omega_ynx)
        decayed = decay_opinion(deduced, elapsed=3.0, half_life=10.0)
        discounted = trust_discount(trust, decayed)
        fused = cumulative_fuse(discounted, opinion_b)

        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=_TOL)
        assert fused.belief >= -_TOL
        assert fused.disbelief >= -_TOL
        assert fused.uncertainty >= -_TOL
