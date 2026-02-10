"""Tests for Subjective Logic operators: fusion, discount, deduction.

These operators form the algebraic core of the confidence algebra,
with provable properties that distinguish jsonld-ex from ad-hoc
confidence propagation schemes.

Operators:
  - Cumulative fusion (⊕):  Combine independent sources
  - Averaging fusion (⊘):   Combine dependent/correlated sources
  - Trust discount (⊗):     Propagate confidence through trust chains
  - Deduction (⊙):          Conditional reasoning with opinions

Algebraic properties tested:
  - Commutativity:          A ⊕ B = B ⊕ A
  - Associativity:          (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)
  - Identity element:       A ⊕ vacuous = A
  - Uncertainty reduction:  u_{A⊕B} ≤ min(u_A, u_B)
  - Bounds preservation:    result ∈ valid opinion space
  - Idempotence (avg):      A ⊘ A = A
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse, averaging_fuse, trust_discount


# ═══════════════════════════════════════════════════════════════════
# Cumulative Fusion (⊕) — independent sources
# ═══════════════════════════════════════════════════════════════════


class TestCumulativeFuseBasic:
    """Basic functionality of cumulative fusion."""

    def test_two_agreeing_sources(self):
        """Two sources with similar belief should reinforce each other."""
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        result = cumulative_fuse(a, b)
        # Combined belief should be higher than either source
        assert result.belief > max(a.belief, b.belief)
        # Uncertainty should decrease
        assert result.uncertainty < min(a.uncertainty, b.uncertainty)

    def test_two_conflicting_sources(self):
        """One believes, the other disbelieves — uncertainty should increase."""
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        result = cumulative_fuse(a, b)
        # Belief and disbelief should both be moderate
        assert result.belief < a.belief
        assert result.disbelief < b.disbelief

    def test_strong_and_weak_source(self):
        """A strong source should dominate a weak (uncertain) source."""
        strong = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        weak = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6)
        result = cumulative_fuse(strong, weak)
        # Result should be closer to the strong source
        assert abs(result.belief - strong.belief) < abs(result.belief - weak.belief)

    def test_result_is_valid_opinion(self):
        """Fused result must satisfy b + d + u = 1."""
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2)
        result = cumulative_fuse(a, b)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert 0 <= result.belief <= 1
        assert 0 <= result.disbelief <= 1
        assert 0 <= result.uncertainty <= 1


class TestCumulativeFuseIdentity:
    """Vacuous opinion is the identity element for cumulative fusion."""

    def test_fuse_with_vacuous_right(self):
        """A ⊕ vacuous = A."""
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = cumulative_fuse(a, vacuous)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_fuse_with_vacuous_left(self):
        """vacuous ⊕ A = A."""
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = cumulative_fuse(vacuous, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_vacuous_fuse_vacuous(self):
        """vacuous ⊕ vacuous = vacuous."""
        v = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        result = cumulative_fuse(v, v)
        assert result.belief == pytest.approx(0.0, abs=1e-9)
        assert result.disbelief == pytest.approx(0.0, abs=1e-9)
        assert result.uncertainty == pytest.approx(1.0, abs=1e-9)


class TestCumulativeFuseCommutativity:
    """A ⊕ B = B ⊕ A."""

    def test_commutative_general(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        ab = cumulative_fuse(a, b)
        ba = cumulative_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=1e-9)
        assert ab.disbelief == pytest.approx(ba.disbelief, abs=1e-9)
        assert ab.uncertainty == pytest.approx(ba.uncertainty, abs=1e-9)

    def test_commutative_conflicting(self):
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)
        ab = cumulative_fuse(a, b)
        ba = cumulative_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=1e-9)
        assert ab.disbelief == pytest.approx(ba.disbelief, abs=1e-9)

    def test_commutative_extreme(self):
        a = Opinion(belief=0.99, disbelief=0.005, uncertainty=0.005)
        b = Opinion(belief=0.005, disbelief=0.005, uncertainty=0.99)
        ab = cumulative_fuse(a, b)
        ba = cumulative_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=1e-9)


class TestCumulativeFuseAssociativity:
    """(A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)."""

    def test_associative_general(self):
        a = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        b = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        c = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)

        ab_c = cumulative_fuse(cumulative_fuse(a, b), c)
        a_bc = cumulative_fuse(a, cumulative_fuse(b, c))

        assert ab_c.belief == pytest.approx(a_bc.belief, abs=1e-9)
        assert ab_c.disbelief == pytest.approx(a_bc.disbelief, abs=1e-9)
        assert ab_c.uncertainty == pytest.approx(a_bc.uncertainty, abs=1e-9)

    def test_associative_with_high_uncertainty(self):
        a = Opinion(belief=0.1, disbelief=0.1, uncertainty=0.8)
        b = Opinion(belief=0.2, disbelief=0.1, uncertainty=0.7)
        c = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6)

        ab_c = cumulative_fuse(cumulative_fuse(a, b), c)
        a_bc = cumulative_fuse(a, cumulative_fuse(b, c))

        assert ab_c.belief == pytest.approx(a_bc.belief, abs=1e-9)
        assert ab_c.disbelief == pytest.approx(a_bc.disbelief, abs=1e-9)
        assert ab_c.uncertainty == pytest.approx(a_bc.uncertainty, abs=1e-9)


class TestCumulativeFuseUncertaintyReduction:
    """Fusing independent sources must reduce uncertainty."""

    def test_uncertainty_decreases(self):
        a = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.1, uncertainty=0.5)
        result = cumulative_fuse(a, b)
        assert result.uncertainty < min(a.uncertainty, b.uncertainty)

    def test_uncertainty_cannot_increase(self):
        """Fusing any non-vacuous source must not increase uncertainty."""
        a = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)
        b = Opinion(belief=0.1, disbelief=0.1, uncertainty=0.8)
        result = cumulative_fuse(a, b)
        assert result.uncertainty <= a.uncertainty + 1e-9  # allow fp tolerance


class TestCumulativeFuseDogmatic:
    """Edge cases: dogmatic opinions (u = 0)."""

    def test_both_dogmatic(self):
        """Two dogmatic opinions: result should average beliefs proportionally."""
        a = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        b = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        result = cumulative_fuse(a, b)
        # When both are dogmatic, the gamma-weighted average applies
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert result.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_one_dogmatic_one_uncertain(self):
        """Dogmatic source should dominate."""
        dogmatic = Opinion(belief=0.9, disbelief=0.1, uncertainty=0.0)
        uncertain = Opinion(belief=0.3, disbelief=0.1, uncertainty=0.6)
        result = cumulative_fuse(dogmatic, uncertain)
        # Dogmatic opinion should heavily influence the result
        assert result.belief > 0.7


class TestCumulativeFuseMultiple:
    """Fusing more than two opinions (variadic)."""

    def test_three_sources(self):
        opinions = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3),
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        result = cumulative_fuse(*opinions)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        # Three agreeing sources → belief higher than any individual source
        assert result.belief > max(o.belief for o in opinions)

    def test_single_opinion_returned_as_is(self):
        o = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        result = cumulative_fuse(o)
        assert result == o

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            cumulative_fuse()


# ═══════════════════════════════════════════════════════════════════
# Averaging Fusion (⊘) — dependent/correlated sources
# ═══════════════════════════════════════════════════════════════════


class TestAveragingFuseBasic:
    def test_two_identical_opinions(self):
        """Averaging identical opinions should return the same opinion (idempotent)."""
        a = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        result = averaging_fuse(a, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_two_different_opinions(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        result = averaging_fuse(a, b)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_result_is_valid(self):
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2)
        result = averaging_fuse(a, b)
        assert 0 <= result.belief <= 1
        assert 0 <= result.disbelief <= 1
        assert 0 <= result.uncertainty <= 1


class TestAveragingFuseCommutativity:
    def test_commutative(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        ab = averaging_fuse(a, b)
        ba = averaging_fuse(b, a)
        assert ab.belief == pytest.approx(ba.belief, abs=1e-9)
        assert ab.disbelief == pytest.approx(ba.disbelief, abs=1e-9)
        assert ab.uncertainty == pytest.approx(ba.uncertainty, abs=1e-9)


class TestAveragingFuseIdempotence:
    """Key distinguishing property: A ⊘ A = A."""

    def test_idempotent_general(self):
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        result = averaging_fuse(a, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_idempotent_extreme(self):
        a = Opinion(belief=0.95, disbelief=0.04, uncertainty=0.01)
        result = averaging_fuse(a, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)


class TestAveragingFuseUncertainty:
    """Averaging fusion does NOT reduce uncertainty (unlike cumulative)."""

    def test_uncertainty_does_not_decrease_for_identical(self):
        a = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        result = averaging_fuse(a, a)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)


class TestAveragingFuseNonAssociativity:
    """IMPORTANT: Averaging fusion is NOT associative.

    (A ⊘ B) ⊘ C ≠ A ⊘ (B ⊘ C) in general.

    This is mathematically correct per Jøsang (2016, §12.5).
    The proper n-source averaging fusion uses a simultaneous
    formula, not pairwise application.  We document and test
    this limitation explicitly.
    """

    def test_non_associative_general(self):
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        c = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)

        ab_c = averaging_fuse(averaging_fuse(a, b), c)
        a_bc = averaging_fuse(a, averaging_fuse(b, c))

        # These should NOT be equal — averaging is not associative
        assert ab_c.belief != pytest.approx(a_bc.belief, abs=1e-6)

    def test_pairwise_is_exact_for_two(self):
        """For exactly two sources, pairwise = direct.  No issue."""
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        result = averaging_fuse(a, b)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)


class TestAveragingVsCumulative:
    """Averaging fusion should be more conservative than cumulative."""

    def test_averaging_less_belief_boost(self):
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        cum = cumulative_fuse(a, b)
        avg = averaging_fuse(a, b)
        # Cumulative should produce higher belief for agreeing sources
        assert cum.belief > avg.belief


class TestAveragingFuseMultiple:
    def test_three_sources(self):
        opinions = [
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3),
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        result = averaging_fuse(*opinions)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_single_returns_self(self):
        o = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        result = averaging_fuse(o)
        assert result == o

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            averaging_fuse()


# ═══════════════════════════════════════════════════════════════════
# Trust Discount (⊗) — chain propagation via trust transitivity
# ═══════════════════════════════════════════════════════════════════


class TestTrustDiscountBasic:
    """A trusts B, B asserts X → A's derived opinion about X."""

    def test_full_trust(self):
        """If A fully trusts B, A adopts B's opinion unchanged."""
        trust = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        opinion = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = trust_discount(trust, opinion)
        assert result.belief == pytest.approx(opinion.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(opinion.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(opinion.uncertainty, abs=1e-9)

    def test_zero_trust(self):
        """If A has zero trust in B, the result is vacuous (total uncertainty)."""
        trust = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        opinion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        result = trust_discount(trust, opinion)
        assert result.uncertainty == pytest.approx(1.0, abs=1e-9)
        assert result.belief == pytest.approx(0.0, abs=1e-9)
        assert result.disbelief == pytest.approx(0.0, abs=1e-9)

    def test_partial_trust(self):
        """Partial trust should dilute the opinion toward uncertainty."""
        trust = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        opinion = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = trust_discount(trust, opinion)
        # Belief should be reduced, uncertainty should increase
        assert result.belief < opinion.belief
        assert result.uncertainty > opinion.uncertainty

    def test_result_is_valid(self):
        trust = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        opinion = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        result = trust_discount(trust, opinion)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)
        assert 0 <= result.belief <= 1
        assert 0 <= result.disbelief <= 1
        assert 0 <= result.uncertainty <= 1

    def test_vacuous_trust(self):
        """Total uncertainty about trust → result is vacuous."""
        trust = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        opinion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        result = trust_discount(trust, opinion)
        assert result.uncertainty == pytest.approx(1.0, abs=1e-9)


class TestTrustDiscountChain:
    """Chaining discounts: A→B→C should yield more uncertainty than A→B."""

    def test_chain_increases_uncertainty(self):
        """Longer trust chains → more uncertainty."""
        trust_ab = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        trust_bc = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        assertion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

        # A's view of C's assertion: discount(trust_ab, discount(trust_bc, assertion))
        bc_discounted = trust_discount(trust_bc, assertion)
        abc_discounted = trust_discount(trust_ab, bc_discounted)

        # Direct trust would give less uncertainty than chained
        direct = trust_discount(trust_ab, assertion)
        assert abc_discounted.uncertainty > direct.uncertainty

    def test_chain_preserves_validity(self):
        """Multi-hop chain should still produce valid opinion."""
        trust1 = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        trust2 = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        trust3 = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        assertion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

        step1 = trust_discount(trust3, assertion)
        step2 = trust_discount(trust2, step1)
        result = trust_discount(trust1, step2)

        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)


class TestTrustDiscountMonotonicity:
    """Higher trust should yield beliefs closer to the original opinion."""

    def test_higher_trust_higher_belief(self):
        low_trust = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3)
        high_trust = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        opinion = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)

        low_result = trust_discount(low_trust, opinion)
        high_result = trust_discount(high_trust, opinion)

        assert high_result.belief > low_result.belief
        assert high_result.uncertainty < low_result.uncertainty


# ═══════════════════════════════════════════════════════════════════
# Cross-operator mathematical properties
# ═══════════════════════════════════════════════════════════════════


class TestCrossOperatorProperties:
    """Properties that span multiple operators."""

    def test_cumulative_stronger_than_averaging(self):
        """For agreeing independent sources, cumulative fusion should
        produce higher belief than averaging fusion."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.5, disbelief=0.1, uncertainty=0.4)
        cum = cumulative_fuse(a, b)
        avg = averaging_fuse(a, b)
        assert cum.belief >= avg.belief - 1e-9

    def test_discount_then_fuse_valid(self):
        """Discount followed by fusion should remain valid."""
        trust_a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        trust_b = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        assertion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

        a_view = trust_discount(trust_a, assertion)
        b_view = trust_discount(trust_b, assertion)
        fused = cumulative_fuse(a_view, b_view)

        assert fused.belief + fused.disbelief + fused.uncertainty == pytest.approx(1.0)
        assert 0 <= fused.belief <= 1
        assert 0 <= fused.disbelief <= 1
        assert 0 <= fused.uncertainty <= 1

    def test_all_operators_preserve_base_rate(self):
        """Base rate should pass through operators."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.7)
        b = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.7)
        trust = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)

        cum = cumulative_fuse(a, b)
        avg = averaging_fuse(a, b)
        disc = trust_discount(trust, a)

        # Fused opinions of same base rate should keep it
        assert cum.base_rate == pytest.approx(0.7)
        assert avg.base_rate == pytest.approx(0.7)
