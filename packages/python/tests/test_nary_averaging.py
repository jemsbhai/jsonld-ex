"""Tests for n-ary simultaneous averaging fusion.

Per Jøsang (2016, §12.5), averaging fusion of n > 2 sources requires
a simultaneous formula — pairwise left-fold is NOT equivalent because
averaging fusion is not associative.

This module verifies:
  1. Backward compatibility: 2-source n-ary = pairwise (exact).
  2. N-ary correctness via hand-computed examples.
  3. Algebraic properties: idempotence, commutativity, additivity.
  4. Divergence from pairwise left-fold for n > 2.
  5. Boundary conditions: dogmatic opinions, mixed dogmatic/non-dogmatic.

Mathematical reference:
  Given n opinions ω_i = (b_i, d_i, u_i) with equal weight:

    U_i = ∏_{j≠i} u_j   (product of all OTHER uncertainties)
    κ   = Σ_i U_i

    b = Σ_i (b_i · U_i) / κ
    d = Σ_i (d_i · U_i) / κ
    u = n · ∏_i u_i / κ

  When all u_i = 0 (all dogmatic): simple average.
"""

import math
import pytest

from jsonld_ex.confidence_algebra import (
    Opinion,
    averaging_fuse,
    _averaging_fuse_pair,
)


# ═══════════════════════════════════════════════════════════════════
# Backward Compatibility: 2-source case
# ═══════════════════════════════════════════════════════════════════


class TestNaryBackwardCompatibility:
    """The n-ary formula for n=2 MUST produce identical results to the
    existing pairwise formula.  Any deviation is a regression."""

    def test_two_source_matches_pairwise(self):
        """averaging_fuse(a, b) must be identical before and after."""
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        pairwise = _averaging_fuse_pair(a, b)
        nary = averaging_fuse(a, b)
        assert nary.belief == pytest.approx(pairwise.belief, abs=1e-12)
        assert nary.disbelief == pytest.approx(pairwise.disbelief, abs=1e-12)
        assert nary.uncertainty == pytest.approx(pairwise.uncertainty, abs=1e-12)

    def test_two_source_dogmatic_matches_pairwise(self):
        """Dogmatic 2-source case: both formulas must agree."""
        a = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        b = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        pairwise = _averaging_fuse_pair(a, b)
        nary = averaging_fuse(a, b)
        assert nary.belief == pytest.approx(pairwise.belief, abs=1e-12)
        assert nary.disbelief == pytest.approx(pairwise.disbelief, abs=1e-12)
        assert nary.uncertainty == pytest.approx(pairwise.uncertainty, abs=1e-12)

    def test_two_source_high_uncertainty_matches_pairwise(self):
        """High-uncertainty 2-source case."""
        a = Opinion(belief=0.1, disbelief=0.05, uncertainty=0.85)
        b = Opinion(belief=0.05, disbelief=0.1, uncertainty=0.85)
        pairwise = _averaging_fuse_pair(a, b)
        nary = averaging_fuse(a, b)
        assert nary.belief == pytest.approx(pairwise.belief, abs=1e-12)
        assert nary.disbelief == pytest.approx(pairwise.disbelief, abs=1e-12)
        assert nary.uncertainty == pytest.approx(pairwise.uncertainty, abs=1e-12)

    def test_single_opinion_unchanged(self):
        """averaging_fuse(a) = a."""
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        result = averaging_fuse(a)
        assert result == a

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            averaging_fuse()


# ═══════════════════════════════════════════════════════════════════
# N-ary Correctness: Hand-Computed Examples
# ═══════════════════════════════════════════════════════════════════


class TestNaryHandComputed:
    """Verify against manually calculated values.

    Example:
        ω_1 = (0.6, 0.1, 0.3)
        ω_2 = (0.4, 0.2, 0.4)
        ω_3 = (0.3, 0.3, 0.4)

        U_1 = 0.4 × 0.4 = 0.16
        U_2 = 0.3 × 0.4 = 0.12
        U_3 = 0.3 × 0.4 = 0.12
        κ = 0.40

        b = 0.180 / 0.40 = 0.45
        d = 0.076 / 0.40 = 0.19
        u = 3 × 0.048 / 0.40 = 0.36
    """

    def test_three_source_example(self):
        w1 = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        w2 = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        w3 = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        result = averaging_fuse(w1, w2, w3)

        assert result.belief == pytest.approx(0.45, abs=1e-9)
        assert result.disbelief == pytest.approx(0.19, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.36, abs=1e-9)

    def test_three_source_equal_uncertainty(self):
        """When all u_i are equal, n-ary reduces to simple average of b,d
        with uncertainty preserved.

        ω_i all have u=0.3 ⟹ U_i = 0.3² = 0.09 for all i,
        κ = 3 × 0.09 = 0.27.

        b = Σ(b_i × 0.09) / 0.27 = Σb_i / 3
        d = Σ(d_i × 0.09) / 0.27 = Σd_i / 3
        u = 3 × 0.3³ / 0.27 = 0.081/0.27 = 0.3
        """
        w1 = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        w2 = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        w3 = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)

        result = averaging_fuse(w1, w2, w3)

        assert result.belief == pytest.approx(0.5, abs=1e-9)
        assert result.disbelief == pytest.approx(0.2, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.3, abs=1e-9)

    def test_four_source(self):
        """Verify with n=4.

        ω_1 = (0.5, 0.2, 0.3)
        ω_2 = (0.4, 0.3, 0.3)
        ω_3 = (0.3, 0.2, 0.5)
        ω_4 = (0.6, 0.1, 0.3)

        U_1 = u_2·u_3·u_4 = 0.3 × 0.5 × 0.3 = 0.045
        U_2 = u_1·u_3·u_4 = 0.3 × 0.5 × 0.3 = 0.045
        U_3 = u_1·u_2·u_4 = 0.3 × 0.3 × 0.3 = 0.027
        U_4 = u_1·u_2·u_3 = 0.3 × 0.3 × 0.5 = 0.045

        κ = 0.045 + 0.045 + 0.027 + 0.045 = 0.162

        b = (0.5×0.045 + 0.4×0.045 + 0.3×0.027 + 0.6×0.045) / 0.162
          = (0.0225 + 0.018 + 0.0081 + 0.027) / 0.162
          = 0.0756 / 0.162
          ≈ 0.466667

        d = (0.2×0.045 + 0.3×0.045 + 0.2×0.027 + 0.1×0.045) / 0.162
          = (0.009 + 0.0135 + 0.0054 + 0.0045) / 0.162
          = 0.0324 / 0.162
          = 0.2

        u = 4 × (0.3×0.3×0.5×0.3) / 0.162
          = 4 × 0.0135 / 0.162
          = 0.054 / 0.162
          = 0.333333
        """
        w1 = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        w2 = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        w3 = Opinion(belief=0.3, disbelief=0.2, uncertainty=0.5)
        w4 = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)

        result = averaging_fuse(w1, w2, w3, w4)

        assert result.belief == pytest.approx(0.0756 / 0.162, abs=1e-9)
        assert result.disbelief == pytest.approx(0.2, abs=1e-9)
        assert result.uncertainty == pytest.approx(1.0 / 3.0, abs=1e-9)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# Divergence from Pairwise Left-Fold
# ═══════════════════════════════════════════════════════════════════


class TestNaryDivergenceFromPairwise:
    """Demonstrate that the n-ary formula differs from BOTH pairwise
    groupings when n > 2.  This is the core scientific motivation for
    implementing the proper simultaneous formula."""

    def test_nary_differs_from_left_fold(self):
        """averaging_fuse(a, b, c) ≠ averaging_fuse(averaging_fuse(a, b), c)."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        nary = averaging_fuse(a, b, c)
        left_fold = averaging_fuse(averaging_fuse(a, b), c)

        # They should NOT be equal (different formulas for n>2)
        assert nary.belief != pytest.approx(left_fold.belief, abs=1e-6)

    def test_nary_differs_from_right_fold(self):
        """averaging_fuse(a, b, c) ≠ averaging_fuse(a, averaging_fuse(b, c))."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        nary = averaging_fuse(a, b, c)
        right_fold = averaging_fuse(a, averaging_fuse(b, c))

        assert nary.belief != pytest.approx(right_fold.belief, abs=1e-6)

    def test_both_folds_differ_from_each_other(self):
        """Sanity check: left-fold ≠ right-fold (non-associativity)."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        left_fold = averaging_fuse(averaging_fuse(a, b), c)
        right_fold = averaging_fuse(a, averaging_fuse(b, c))

        assert left_fold.belief != pytest.approx(right_fold.belief, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════
# Algebraic Properties for n > 2
# ═══════════════════════════════════════════════════════════════════


class TestNaryAlgebraicProperties:
    """Properties that must hold for the simultaneous n-ary formula."""

    def test_idempotence_three(self):
        """A ⊘ A ⊘ A = A.
        Idempotence holds for any number of identical opinions.

        Proof sketch: U_i = u^(n-1) for all i, κ = n·u^(n-1).
        b_result = n·b·u^(n-1) / (n·u^(n-1)) = b.  QED.
        """
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        result = averaging_fuse(a, a, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_idempotence_five(self):
        """A ⊘ A ⊘ A ⊘ A ⊘ A = A."""
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        result = averaging_fuse(a, a, a, a, a)
        assert result.belief == pytest.approx(a.belief, abs=1e-9)
        assert result.disbelief == pytest.approx(a.disbelief, abs=1e-9)
        assert result.uncertainty == pytest.approx(a.uncertainty, abs=1e-9)

    def test_commutativity_three(self):
        """All permutations of 3 opinions yield the same result."""
        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        abc = averaging_fuse(a, b, c)
        acb = averaging_fuse(a, c, b)
        bac = averaging_fuse(b, a, c)
        bca = averaging_fuse(b, c, a)
        cab = averaging_fuse(c, a, b)
        cba = averaging_fuse(c, b, a)

        for perm in [acb, bac, bca, cab, cba]:
            assert perm.belief == pytest.approx(abc.belief, abs=1e-9)
            assert perm.disbelief == pytest.approx(abc.disbelief, abs=1e-9)
            assert perm.uncertainty == pytest.approx(abc.uncertainty, abs=1e-9)

    def test_additivity_three(self):
        """b + d + u = 1 for 3-source fusion."""
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        b = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3)
        c = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        result = averaging_fuse(a, b, c)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_additivity_five(self):
        """b + d + u = 1 for 5-source fusion."""
        opinions = [
            Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3),
            Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3),
            Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3),
            Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3),
            Opinion(belief=0.2, disbelief=0.1, uncertainty=0.7),
        ]
        result = averaging_fuse(*opinions)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_bounds(self):
        """All components in [0, 1] for 3-source fusion."""
        a = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        b = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        c = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)
        result = averaging_fuse(a, b, c)
        assert 0.0 <= result.belief <= 1.0
        assert 0.0 <= result.disbelief <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Boundary Conditions: Dogmatic Opinions
# ═══════════════════════════════════════════════════════════════════


class TestNaryDogmatic:
    """Boundary behavior when one or more opinions have u = 0."""

    def test_all_dogmatic_three(self):
        """All dogmatic: simple average.

        When all u_i = 0, every U_i = 0 and κ = 0.
        Fallback: b = Σb_i/n, d = Σd_i/n, u = 0.
        """
        a = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        b = Opinion(belief=0.6, disbelief=0.4, uncertainty=0.0)
        c = Opinion(belief=0.4, disbelief=0.6, uncertainty=0.0)

        result = averaging_fuse(a, b, c)

        assert result.belief == pytest.approx(0.6, abs=1e-9)
        assert result.disbelief == pytest.approx(0.4, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_one_dogmatic_dominates(self):
        """A single dogmatic opinion among non-dogmatic sources dominates.

        When ω_1 has u_1 = 0:
          U_1 = ∏_{j≠1} u_j > 0  (non-zero)
          U_j = u_1 · ∏_{k≠j,k≠1} u_k = 0 for all j ≠ 1
          κ = U_1

        Result: b = b_1, d = d_1, u = 0.  The dogmatic source
        completely overrides all non-dogmatic sources.

        This is mathematically correct per Jøsang: a dogmatic source
        has infinite relative weight (zero uncertainty = infinite
        evidence).  We test and document this rather than hiding it.
        """
        dogmatic = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        uncertain1 = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)
        uncertain2 = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)

        result = averaging_fuse(dogmatic, uncertain1, uncertain2)

        assert result.belief == pytest.approx(0.8, abs=1e-9)
        assert result.disbelief == pytest.approx(0.2, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_two_dogmatic_among_three(self):
        """Two dogmatic + one non-dogmatic: dogmatic sources dominate.

        ω_1 = (0.8, 0.2, 0.0), ω_2 = (0.4, 0.6, 0.0), ω_3 = (0.5, 0.2, 0.3)

        U_1 = u_2·u_3 = 0 × 0.3 = 0
        U_2 = u_1·u_3 = 0 × 0.3 = 0
        U_3 = u_1·u_2 = 0 × 0   = 0
        κ = 0  → dogmatic fallback

        Limit analysis (ε-perturbation, §12.5):
        Replace u_1 → ε, u_2 → ε, keep u_3 = 0.3, take ε → 0.
          U_1 = ε·0.3 = O(ε),  U_2 = ε·0.3 = O(ε),  U_3 = ε² = O(ε²)
        The non-dogmatic source's U_3 vanishes at higher order.
        Result: average of the dogmatic subset only.

        b = (0.8 + 0.4) / 2 = 0.6
        d = (0.2 + 0.6) / 2 = 0.4
        u = 0
        """
        a = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        b = Opinion(belief=0.4, disbelief=0.6, uncertainty=0.0)
        c = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)

        result = averaging_fuse(a, b, c)

        assert result.belief == pytest.approx(0.6, abs=1e-9)
        assert result.disbelief == pytest.approx(0.4, abs=1e-9)
        assert result.uncertainty == pytest.approx(0.0, abs=1e-9)

    def test_near_dogmatic_stability(self):
        """Opinions with very small but non-zero u should not cause
        numerical instability."""
        a = Opinion(belief=0.8, disbelief=0.2 - 1e-12, uncertainty=1e-12)
        b = Opinion(belief=0.6, disbelief=0.4 - 1e-12, uncertainty=1e-12)
        c = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        result = averaging_fuse(a, b, c)
        total = result.belief + result.disbelief + result.uncertainty
        assert total == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= result.belief <= 1.0
        assert 0.0 <= result.disbelief <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0


# ═══════════════════════════════════════════════════════════════════
# Base Rate Handling
# ═══════════════════════════════════════════════════════════════════


class TestNaryBaseRate:
    """Base rate of the n-ary result should be the average of inputs."""

    def test_same_base_rate_preserved(self):
        a = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.6)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.6)
        c = Opinion(belief=0.3, disbelief=0.2, uncertainty=0.5, base_rate=0.6)
        result = averaging_fuse(a, b, c)
        assert result.base_rate == pytest.approx(0.6, abs=1e-9)

    def test_different_base_rates_averaged(self):
        a = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3, base_rate=0.3)
        b = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3, base_rate=0.6)
        c = Opinion(belief=0.3, disbelief=0.2, uncertainty=0.5, base_rate=0.9)
        result = averaging_fuse(a, b, c)
        assert result.base_rate == pytest.approx(0.6, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# Integration with Existing Property-Based Tests
# ═══════════════════════════════════════════════════════════════════


class TestNaryWithExistingOperators:
    """N-ary averaging fusion composes correctly with other operators."""

    def test_nary_then_trust_discount(self):
        """Fuse three sources, then discount by trust: valid opinion."""
        from jsonld_ex.confidence_algebra import trust_discount

        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)
        trust = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)

        fused = averaging_fuse(a, b, c)
        discounted = trust_discount(trust, fused)

        total = discounted.belief + discounted.disbelief + discounted.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_nary_then_cumulative_fuse(self):
        """N-ary average + cumulative fuse with independent source."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)
        independent = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)

        avg = averaging_fuse(a, b, c)
        fused = cumulative_fuse(avg, independent)

        total = fused.belief + fused.disbelief + fused.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_nary_then_decay(self):
        """N-ary average then temporal decay: valid opinion."""
        from jsonld_ex.confidence_decay import decay_opinion

        a = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        b = Opinion(belief=0.4, disbelief=0.2, uncertainty=0.4)
        c = Opinion(belief=0.3, disbelief=0.3, uncertainty=0.4)

        avg = averaging_fuse(a, b, c)
        decayed = decay_opinion(avg, elapsed=5.0, half_life=10.0)

        total = decayed.belief + decayed.disbelief + decayed.uncertainty
        assert total == pytest.approx(1.0, abs=1e-9)
