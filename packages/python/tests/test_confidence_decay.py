"""Tests for temporal decay of opinions.

Temporal decay models the natural degradation of confidence over time:
as evidence ages, it becomes less reliable, and uncertainty should
increase.  This is critical for ML systems where model predictions
have a natural shelf life.

Mathematical invariants:
  - At t=0: opinion unchanged (identity)
  - At t=half_life: belief and disbelief halved
  - As t→∞: opinion approaches vacuous (total uncertainty)
  - b/d ratio preserved (direction of evidence unchanged)
  - b + d + u = 1 maintained (valid opinion throughout)
  - Monotonicity: more elapsed time → more uncertainty

Extensibility:
  - Users can provide any decay function matching the protocol:
    decay_fn(elapsed: float, half_life: float) -> float in [0, 1]
  - Built-in functions: exponential_decay, linear_decay, step_decay
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)


# ═══════════════════════════════════════════════════════════════════
# Built-in decay functions
# ═══════════════════════════════════════════════════════════════════


class TestExponentialDecay:
    """λ(t, τ) = 2^(−t/τ)."""

    def test_at_zero(self):
        assert exponential_decay(0.0, 10.0) == pytest.approx(1.0)

    def test_at_half_life(self):
        assert exponential_decay(10.0, 10.0) == pytest.approx(0.5)

    def test_at_two_half_lives(self):
        assert exponential_decay(20.0, 10.0) == pytest.approx(0.25)

    def test_approaches_zero(self):
        assert exponential_decay(1000.0, 10.0) < 1e-10

    def test_underflows_to_zero_at_extreme(self):
        """IEEE 754 underflow: 2^(-1e6) = 0.0 in floating point.
        Mathematically positive, but not representable."""
        assert exponential_decay(1e6, 1.0) >= 0.0

    def test_monotonically_decreasing(self):
        prev = 1.0
        for t in [1, 5, 10, 50, 100]:
            val = exponential_decay(float(t), 10.0)
            assert val < prev
            prev = val


class TestLinearDecay:
    """λ(t, τ) = max(0, 1 − t/(2τ)).

    Reaches zero at t = 2τ (twice the half_life).
    """

    def test_at_zero(self):
        assert linear_decay(0.0, 10.0) == pytest.approx(1.0)

    def test_at_half_life(self):
        assert linear_decay(10.0, 10.0) == pytest.approx(0.5)

    def test_at_full_decay(self):
        """Reaches zero at t = 2 * half_life."""
        assert linear_decay(20.0, 10.0) == pytest.approx(0.0)

    def test_clamps_at_zero(self):
        """Beyond full decay, remains at zero."""
        assert linear_decay(100.0, 10.0) == pytest.approx(0.0)

    def test_monotonically_decreasing(self):
        prev = 1.0
        for t in [1, 5, 10, 15, 20]:
            val = linear_decay(float(t), 10.0)
            assert val <= prev
            prev = val


class TestStepDecay:
    """λ(t, τ) = 1 if t < τ, else 0.

    Binary: evidence is either fresh or completely stale.
    """

    def test_before_threshold(self):
        assert step_decay(5.0, 10.0) == pytest.approx(1.0)

    def test_at_threshold(self):
        """At exactly the half_life, evidence expires."""
        assert step_decay(10.0, 10.0) == pytest.approx(0.0)

    def test_after_threshold(self):
        assert step_decay(15.0, 10.0) == pytest.approx(0.0)

    def test_at_zero(self):
        assert step_decay(0.0, 10.0) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════
# decay_opinion: core behavior
# ═══════════════════════════════════════════════════════════════════


class TestDecayOpinionIdentity:
    """At t=0, opinion is unchanged."""

    def test_zero_elapsed(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        result = decay_opinion(o, elapsed=0.0, half_life=10.0)
        assert result.belief == pytest.approx(o.belief, abs=1e-12)
        assert result.disbelief == pytest.approx(o.disbelief, abs=1e-12)
        assert result.uncertainty == pytest.approx(o.uncertainty, abs=1e-12)

    def test_zero_elapsed_dogmatic(self):
        o = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        result = decay_opinion(o, elapsed=0.0, half_life=10.0)
        assert result.belief == pytest.approx(o.belief, abs=1e-12)


class TestDecayOpinionHalfLife:
    """At t=half_life, belief and disbelief are halved."""

    def test_belief_halved(self):
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = decay_opinion(o, elapsed=10.0, half_life=10.0)
        assert result.belief == pytest.approx(0.4, abs=1e-9)
        assert result.disbelief == pytest.approx(0.05, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.55, abs=1e-9)

    def test_dogmatic_gains_uncertainty(self):
        o = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        result = decay_opinion(o, elapsed=10.0, half_life=10.0)
        assert result.belief == pytest.approx(0.3)
        assert result.disbelief == pytest.approx(0.2)
        assert result.uncertainty == pytest.approx(0.5)


class TestDecayOpinionAsymptotic:
    """As t→∞, opinion approaches vacuous."""

    def test_very_large_elapsed(self):
        o = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        result = decay_opinion(o, elapsed=10000.0, half_life=10.0)
        assert result.belief < 1e-10
        assert result.disbelief < 1e-10
        assert result.uncertainty == pytest.approx(1.0, abs=1e-9)

    def test_vacuous_stays_vacuous(self):
        o = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = decay_opinion(o, elapsed=100.0, half_life=10.0)
        assert result.uncertainty == pytest.approx(1.0, abs=1e-12)


class TestDecayOpinionInvariants:
    """Mathematical invariants that must hold for ANY valid decay."""

    def test_additivity_preserved(self):
        """b + d + u = 1 after decay."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.3)
        for t in [0, 1, 5, 10, 50, 100, 1000]:
            result = decay_opinion(o, elapsed=float(t), half_life=10.0)
            total = result.belief + result.disbelief + result.uncertainty
            assert total == pytest.approx(1.0, abs=1e-9), f"Failed at t={t}: {total}"

    def test_belief_disbelief_ratio_preserved(self):
        """b/d ratio is constant throughout decay (evidence direction unchanged)."""
        o = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        original_ratio = o.belief / o.disbelief

        for t in [1, 5, 10, 20]:
            result = decay_opinion(o, elapsed=float(t), half_life=10.0)
            if result.disbelief > 1e-15:  # avoid div by zero at extreme decay
                ratio = result.belief / result.disbelief
                assert ratio == pytest.approx(original_ratio, rel=1e-9), \
                    f"Ratio changed at t={t}: {ratio} vs {original_ratio}"

    def test_uncertainty_monotonically_increases(self):
        """More time → more uncertainty (never decreases)."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        prev_u = o.uncertainty
        for t in [1, 5, 10, 20, 50]:
            result = decay_opinion(o, elapsed=float(t), half_life=10.0)
            assert result.uncertainty >= prev_u - 1e-12
            prev_u = result.uncertainty

    def test_belief_monotonically_decreases(self):
        """More time → less belief."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        prev_b = o.belief
        for t in [1, 5, 10, 20, 50]:
            result = decay_opinion(o, elapsed=float(t), half_life=10.0)
            assert result.belief <= prev_b + 1e-12
            prev_b = result.belief

    def test_components_stay_in_bounds(self):
        """All components remain in [0, 1]."""
        o = Opinion(belief=0.99, disbelief=0.005, uncertainty=0.005)
        for t in [0.01, 0.1, 1, 10, 100, 10000]:
            result = decay_opinion(o, elapsed=t, half_life=5.0)
            assert 0.0 <= result.belief <= 1.0, f"belief out of bounds at t={t}"
            assert 0.0 <= result.disbelief <= 1.0, f"disbelief out of bounds at t={t}"
            assert 0.0 <= result.uncertainty <= 1.0, f"uncertainty out of bounds at t={t}"

    def test_base_rate_preserved(self):
        """Base rate is not affected by decay."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.3)
        result = decay_opinion(o, elapsed=15.0, half_life=10.0)
        assert result.base_rate == pytest.approx(0.3)

    def test_projected_probability_moves_toward_base_rate(self):
        """As evidence decays, P(ω) → base_rate (prior takes over)."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.3)
        p_original = o.projected_probability()

        result = decay_opinion(o, elapsed=100.0, half_life=10.0)
        p_decayed = result.projected_probability()

        # Should be much closer to base_rate (0.3) than original
        assert abs(p_decayed - 0.3) < abs(p_original - 0.3)


# ═══════════════════════════════════════════════════════════════════
# Custom decay functions
# ═══════════════════════════════════════════════════════════════════


class TestCustomDecayFunction:
    """Users can provide their own decay_fn(elapsed, half_life) -> factor."""

    def test_constant_decay(self):
        """Custom function that always returns 0.5 (ignores time)."""
        always_half = lambda elapsed, half_life: 0.5
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = decay_opinion(o, elapsed=999.0, half_life=1.0, decay_fn=always_half)
        assert result.belief == pytest.approx(0.4, abs=1e-9)
        assert result.disbelief == pytest.approx(0.05, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.55, abs=1e-9)

    def test_instant_expire(self):
        """Custom function: any elapsed > 0 → fully decayed."""
        instant = lambda elapsed, half_life: 0.0 if elapsed > 0 else 1.0
        o = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        result = decay_opinion(o, elapsed=0.001, half_life=10.0, decay_fn=instant)
        assert result.belief == pytest.approx(0.0)
        assert result.uncertainty == pytest.approx(1.0)

    def test_linear_decay_via_parameter(self):
        """Pass the built-in linear_decay explicitly."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = decay_opinion(o, elapsed=10.0, half_life=10.0, decay_fn=linear_decay)
        # linear_decay(10, 10) = 0.5
        assert result.belief == pytest.approx(0.4, abs=1e-9)

    def test_step_decay_via_parameter(self):
        """Pass the built-in step_decay explicitly."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)

        # Before threshold: unchanged
        before = decay_opinion(o, elapsed=5.0, half_life=10.0, decay_fn=step_decay)
        assert before.belief == pytest.approx(0.8, abs=1e-9)

        # After threshold: vacuous
        after = decay_opinion(o, elapsed=15.0, half_life=10.0, decay_fn=step_decay)
        assert after.belief == pytest.approx(0.0, abs=1e-9)
        assert after.uncertainty == pytest.approx(1.0, abs=1e-9)

    def test_custom_with_additivity_check(self):
        """Any custom function must still produce valid opinions."""
        # Sigmoid-like decay
        sigmoid_decay = lambda t, tau: 1.0 / (1.0 + math.exp((t - tau) / (tau * 0.3)))
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        for t in [0, 5, 10, 20, 50]:
            result = decay_opinion(o, elapsed=float(t), half_life=10.0, decay_fn=sigmoid_decay)
            total = result.belief + result.disbelief + result.uncertainty
            assert total == pytest.approx(1.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# Input validation
# ═══════════════════════════════════════════════════════════════════


class TestDecayValidation:
    """Bad inputs should be caught early with clear errors."""

    def test_negative_elapsed(self):
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(ValueError, match="elapsed.*non-negative"):
            decay_opinion(o, elapsed=-1.0, half_life=10.0)

    def test_zero_half_life(self):
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(ValueError, match="half_life.*positive"):
            decay_opinion(o, elapsed=5.0, half_life=0.0)

    def test_negative_half_life(self):
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(ValueError, match="half_life.*positive"):
            decay_opinion(o, elapsed=5.0, half_life=-10.0)

    def test_decay_fn_returns_above_one(self):
        """Custom function returning > 1 should be clamped or rejected."""
        bad_fn = lambda t, tau: 1.5
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(ValueError, match="decay factor.*\\[0.*1\\]"):
            decay_opinion(o, elapsed=5.0, half_life=10.0, decay_fn=bad_fn)

    def test_decay_fn_returns_negative(self):
        """Custom function returning < 0 should be rejected."""
        bad_fn = lambda t, tau: -0.1
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(ValueError, match="decay factor.*\\[0.*1\\]"):
            decay_opinion(o, elapsed=5.0, half_life=10.0, decay_fn=bad_fn)


# ═══════════════════════════════════════════════════════════════════
# Integration with fusion operators
# ═══════════════════════════════════════════════════════════════════


class TestDecayWithFusion:
    """Temporal decay composes correctly with fusion operators."""

    def test_fresh_opinion_dominates_stale(self):
        """When fusing a fresh and stale opinion, the fresh one
        should have more influence because the stale one has
        higher uncertainty."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        fresh = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        stale_original = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        stale = decay_opinion(stale_original, elapsed=30.0, half_life=10.0)

        # Stale opinion should now have high uncertainty
        # factor = 2^(-30/10) = 0.125, so u = 1 - 0.125*(0.9+0.05) = 0.88125
        assert stale.uncertainty > 0.8

        # Fusing should be dominated by the fresh opinion
        fused = cumulative_fuse(fresh, stale)
        assert abs(fused.belief - fresh.belief) < abs(fused.belief - stale_original.belief)

    def test_two_equally_stale_opinions(self):
        """Two opinions with the same decay should fuse symmetrically."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)

        a_decayed = decay_opinion(a, elapsed=5.0, half_life=10.0)
        b_decayed = decay_opinion(b, elapsed=5.0, half_life=10.0)

        fused = cumulative_fuse(a_decayed, b_decayed)
        assert fused.belief + fused.disbelief + fused.uncertainty == pytest.approx(1.0)
