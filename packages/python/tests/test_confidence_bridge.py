"""Tests for the bridge between scalar inference API and formal algebra.

The bridge layer demonstrates the relationship between existing ad-hoc
methods in inference.py and the Subjective Logic operators.

Exact equivalences (proven):
  - Multiply chain ≡ iterated trust discount with base_rate=0
  - Scalar average (2 sources) ≡ averaging fusion of dogmatic opinions

Generalization relationships (not exact equivalence):
  - Noisy-OR and cumulative fusion both combine independent sources
    but use different formulas.  Cumulative fusion preserves the
    uncertainty dimension; noisy-OR collapses it.
  - The algebra is strictly more expressive: it distinguishes
    epistemic states that scalar confidence conflates.

The bridge provides upgrade paths for users who want richer
uncertainty semantics without changing their existing code.
"""

import math
import pytest

from jsonld_ex.inference import (
    propagate_confidence,
    combine_sources,
    resolve_conflict,
    PropagationResult,
)
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
)
from jsonld_ex.confidence_bridge import (
    combine_opinions_from_scalars,
    propagate_opinions_from_scalars,
    resolve_conflict_with_opinions,
    scalar_propagate_via_algebra,
)


# ═══════════════════════════════════════════════════════════════════
# EXACT EQUIVALENCE: multiply chain ≡ trust discount (base_rate=0)
# ═══════════════════════════════════════════════════════════════════


class TestMultiplyChainEquivalence:
    """Prove: propagate(chain, 'multiply') ≡ chained trust_discount(base_rate=0).

    When each step's confidence is modeled as a dogmatic trust opinion
    (b=c, d=1-c, u=0) and the assertion base_rate=0, the projected
    probability equals the product of confidences exactly.

    This works because:
      - trust_discount produces b = ∏(b_trust_i), u = accumulated
      - With base_rate=0: P(ω) = b + 0·u = b = ∏c_i
    """

    def test_two_step_chain(self):
        chain = [0.9, 0.8]
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_three_step_chain(self):
        chain = [0.9, 0.8, 0.7]
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_long_chain(self):
        chain = [0.9] * 10
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_single_step(self):
        chain = [0.75]
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_zero_in_chain(self):
        chain = [0.9, 0.0, 0.8]
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_varied_chain(self):
        chain = [0.95, 0.6, 0.85, 0.7]
        scalar_result = propagate_confidence(chain, "multiply").score
        algebra_result = _chain_discount_a0(chain)
        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)


class TestMultiplyInductiveProof:
    """Constructive inductive proof: after n trust discounts,
    b_n = ∏ c_i and u_n = 1 − ∏ c_i.

    This verifies the closed-form at each step, not just the
    final projected probability.
    """

    def test_base_case(self):
        """n=1: b = c, u = 1-c."""
        c = 0.85
        assertion = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.0)
        trust = Opinion(belief=c, disbelief=1-c, uncertainty=0.0)
        result = trust_discount(trust, assertion)

        assert result.belief == pytest.approx(c, abs=1e-12)
        assert result.disbelief == pytest.approx(0.0, abs=1e-12)
        assert result.uncertainty == pytest.approx(1 - c, abs=1e-12)

    def test_inductive_step_verify_each_intermediate(self):
        """For chain [c1, c2, c3, c4], verify b_k = ∏_{i=1}^k c_i at each step."""
        chain = [0.9, 0.8, 0.7, 0.6]

        current = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.0)
        running_product = 1.0

        for c in chain:
            trust = Opinion(belief=c, disbelief=1-c, uncertainty=0.0)
            current = trust_discount(trust, current)
            running_product *= c

            # At each step: b == product so far, d == 0, u == 1 - product
            assert current.belief == pytest.approx(running_product, abs=1e-12)
            assert current.disbelief == pytest.approx(0.0, abs=1e-12)
            assert current.uncertainty == pytest.approx(1 - running_product, abs=1e-12)


def _chain_discount_a0(chain: list[float]) -> float:
    """Chain trust_discount with dogmatic opinions and base_rate=0."""
    current = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.0)
    for c in reversed(chain):
        trust = Opinion(belief=c, disbelief=1.0 - c, uncertainty=0.0)
        current = trust_discount(trust, current)
    return current.to_confidence()


# ═══════════════════════════════════════════════════════════════════
# EXACT EQUIVALENCE: average (2 sources) ≡ averaging fusion (dogmatic)
# ═══════════════════════════════════════════════════════════════════


class TestAverageEquivalence:
    """Prove: average(a, b) ≡ averaging_fuse(dogmatic_a, dogmatic_b).to_confidence()

    For two dogmatic opinions (u=0), averaging fusion reduces to
    arithmetic mean of beliefs.  This is exact for n=2.
    """

    def test_two_sources(self):
        scores = [0.8, 0.6]
        scalar_result = combine_sources(scores, "average").score

        opinions = [Opinion.from_confidence(p) for p in scores]
        algebra_result = averaging_fuse(*opinions).to_confidence()

        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_two_sources_varied(self):
        scores = [0.95, 0.3]
        scalar_result = combine_sources(scores, "average").score

        opinions = [Opinion.from_confidence(p) for p in scores]
        algebra_result = averaging_fuse(*opinions).to_confidence()

        assert scalar_result == pytest.approx(algebra_result, abs=1e-9)

    def test_three_sources_NOT_equivalent(self):
        """For n>2, pairwise averaging fusion ≠ scalar arithmetic mean.

        This is NOT a bug — averaging fusion is not associative,
        so sequential pairwise application diverges from the
        simultaneous n-ary mean.  We test this explicitly to
        prevent anyone from claiming n>2 equivalence.
        """
        scores = [0.9, 0.7, 0.5]
        scalar_mean = combine_sources(scores, "average").score  # 0.7

        opinions = [Opinion.from_confidence(p) for p in scores]
        algebra_result = averaging_fuse(*opinions).to_confidence()

        # These should NOT match for n>2
        assert scalar_mean != pytest.approx(algebra_result, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# RELATIONSHIP (not exact equivalence): noisy-OR and cumulative fusion
# ═══════════════════════════════════════════════════════════════════


class TestNoisyOrRelationship:
    """Noisy-OR and cumulative fusion are related but distinct operations.

    Both combine independent sources, but:
      - Noisy-OR: 1 - ∏(1-p_i), produces a scalar
      - Cumulative fusion: produces an Opinion (b, d, u)

    They agree on the direction (more agreeing sources → higher result)
    but differ in magnitude because cumulative fusion preserves
    the uncertainty dimension rather than collapsing it.
    """

    def test_both_increase_with_more_sources(self):
        """Both methods should give higher results with more agreeing sources."""
        scores_2 = [0.8, 0.7]
        scores_3 = [0.8, 0.7, 0.6]

        nor_2 = combine_sources(scores_2, "noisy_or").score
        nor_3 = combine_sources(scores_3, "noisy_or").score

        ops_2 = [Opinion(belief=p, disbelief=0.0, uncertainty=1-p) for p in scores_2]
        ops_3 = [Opinion(belief=p, disbelief=0.0, uncertainty=1-p) for p in scores_3]
        cf_2 = cumulative_fuse(*ops_2).to_confidence()
        cf_3 = cumulative_fuse(*ops_3).to_confidence()

        assert nor_3 > nor_2
        assert cf_3 > cf_2

    def test_cumulative_fusion_preserves_uncertainty(self):
        """Unlike noisy-OR, cumulative fusion retains uncertainty info."""
        scores = [0.8, 0.7]
        opinions = [Opinion(belief=p, disbelief=0.0, uncertainty=1-p) for p in scores]
        result = cumulative_fuse(*opinions)

        # Noisy-OR gives a single number: 0.94
        nor = combine_sources(scores, "noisy_or").score

        # Cumulative fusion gives a full opinion with non-zero uncertainty
        assert result.uncertainty > 0
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_single_source_matches(self):
        """For a single source with base_rate=0.5 and d=0, both agree
        since P = b + 0.5*u = p + 0.5*(1-p) = 0.5 + 0.5*p ... no.
        Actually for a single source, noisy-OR = p and cumulative fuse
        of a single opinion just returns it, so P = b + a*u.
        With base_rate=0 and d=0: P = b = p. Exact match.
        """
        p = 0.85
        nor = combine_sources([p], "noisy_or").score
        opinion = Opinion(belief=p, disbelief=0.0, uncertainty=1-p, base_rate=0.0)
        cf = cumulative_fuse(opinion).to_confidence()
        assert nor == pytest.approx(cf, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════
# Bridge convenience functions
# ═══════════════════════════════════════════════════════════════════


class TestCombineOpinionsFromScalars:
    """combine_opinions_from_scalars: scalar scores → Opinion-based fusion."""

    def test_returns_opinion(self):
        result = combine_opinions_from_scalars([0.9, 0.7])
        assert isinstance(result, Opinion)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_more_sources_increase_belief(self):
        two = combine_opinions_from_scalars([0.8, 0.7])
        three = combine_opinions_from_scalars([0.8, 0.7, 0.6])
        assert three.belief > two.belief

    def test_more_sources_decrease_uncertainty(self):
        two = combine_opinions_from_scalars([0.8, 0.7])
        three = combine_opinions_from_scalars([0.8, 0.7, 0.6])
        assert three.uncertainty < two.uncertainty

    def test_with_explicit_uncertainty(self):
        """When uncertainty > 0, results differ from default."""
        default = combine_opinions_from_scalars([0.9, 0.7])
        uncertain = combine_opinions_from_scalars([0.9, 0.7], uncertainty=0.3)
        assert default.to_confidence() != pytest.approx(uncertain.to_confidence(), abs=0.01)

    def test_averaging_mode(self):
        result = combine_opinions_from_scalars([0.8, 0.6], fusion="averaging")
        assert isinstance(result, Opinion)
        assert result.belief + result.disbelief + result.uncertainty == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_opinions_from_scalars([])


class TestPropagateOpinionsFromScalars:
    """propagate_opinions_from_scalars: scalar chain → trust discount → Opinion."""

    def test_dogmatic_matches_multiply(self):
        """With trust_uncertainty=0 and base_rate=0, matches multiply exactly."""
        result = propagate_opinions_from_scalars([0.9, 0.8])
        scalar = propagate_confidence([0.9, 0.8], "multiply").score
        assert result.to_confidence() == pytest.approx(scalar, abs=1e-9)

    def test_with_uncertainty_degrades_more(self):
        """With trust uncertainty, result is lower than pure multiply."""
        dogmatic = propagate_opinions_from_scalars([0.9, 0.8])
        uncertain = propagate_opinions_from_scalars([0.9, 0.8], trust_uncertainty=0.2)
        # Uncertain trust should have lower belief
        assert uncertain.belief < dogmatic.belief
        # And higher uncertainty
        assert uncertain.uncertainty > dogmatic.uncertainty

    def test_returns_full_opinion(self):
        result = propagate_opinions_from_scalars([0.9, 0.8, 0.7])
        assert isinstance(result, Opinion)

    def test_long_chain(self):
        result = propagate_opinions_from_scalars([0.9] * 10)
        scalar = propagate_confidence([0.9] * 10, "multiply").score
        assert result.to_confidence() == pytest.approx(scalar, abs=1e-9)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            propagate_opinions_from_scalars([])


class TestResolveConflictWithOpinions:
    """resolve_conflict_with_opinions: adds Opinion metadata to conflict results."""

    def test_basic_resolution(self):
        assertions = [
            {"@value": "Engineer", "@confidence": 0.9},
            {"@value": "Manager", "@confidence": 0.7},
        ]
        report = resolve_conflict_with_opinions(assertions)
        assert report.winner["@value"] == "Engineer"
        assert "@opinion" in report.winner

    def test_opinion_is_valid(self):
        assertions = [
            {"@value": "A", "@confidence": 0.8},
            {"@value": "B", "@confidence": 0.6},
        ]
        report = resolve_conflict_with_opinions(assertions)
        opinion = Opinion.from_jsonld(report.winner["@opinion"])
        assert opinion.belief + opinion.disbelief + opinion.uncertainty == pytest.approx(1.0)


class TestScalarPropagateViaAlgebra:
    """scalar_propagate_via_algebra: drop-in for propagate_confidence('multiply')."""

    def test_matches_multiply(self):
        for chain in [[0.9, 0.8], [0.9, 0.8, 0.7], [0.9] * 10]:
            original = propagate_confidence(chain, "multiply").score
            via_algebra = scalar_propagate_via_algebra(chain, "multiply").score
            assert original == pytest.approx(via_algebra, abs=1e-9)

    def test_returns_propagation_result(self):
        result = scalar_propagate_via_algebra([0.9, 0.8], "multiply")
        assert isinstance(result, PropagationResult)
        assert result.method == "multiply"

    def test_unsupported_method_raises(self):
        with pytest.raises(ValueError, match="only supports"):
            scalar_propagate_via_algebra([0.9], "bayesian")  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# The key insight: uncertainty-aware versions are STRICTLY better
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyAwareSuperior:
    """Demonstrate that the algebra provides strictly more information
    than scalar confidence alone.

    The same scalar confidence can arise from very different
    epistemic states. The algebra distinguishes them.
    """

    def test_same_confidence_different_opinions(self):
        """Two opinions with the same projected probability but
        different uncertainty — the algebra distinguishes them."""
        # "I'm fairly sure" — moderate evidence
        informed = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        # "I have no idea" — no evidence, base rate happens to be 0.75
        ignorant = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.75)

        # Both project to 0.75
        assert informed.projected_probability() == pytest.approx(0.75)
        assert ignorant.projected_probability() == pytest.approx(0.75)

        # But they are fundamentally different
        assert informed.uncertainty != ignorant.uncertainty
        assert informed != ignorant

    def test_fusion_distinguishes_informed_from_ignorant(self):
        """Fusing with an informed source vs an ignorant source
        should produce different results, even if both have the
        same projected probability."""
        base = Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)

        informed = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3, base_rate=0.5)
        ignorant = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.75)

        fused_informed = cumulative_fuse(base, informed)
        fused_ignorant = cumulative_fuse(base, ignorant)

        # Fusing with the informed source should change belief more
        assert fused_informed.belief != pytest.approx(fused_ignorant.belief, abs=0.01)
        # Fusing with ignorant source should barely change anything (identity-like)
        assert fused_ignorant.belief == pytest.approx(base.belief, abs=1e-9)

    def test_hallucination_firewall_scenario(self):
        """The 'Hallucination Firewall' use case:

        Agent A (vision model) says "it's a cat" with ~0.8 confidence.
        Agent B (logic agent) must decide whether to act on this.

        With scalar confidence alone, ~0.8 always looks the same.
        With opinions, B can distinguish:
          - 0.8 from a well-calibrated model (low uncertainty)
          - ~0.8 from an unreliable model (high uncertainty)
        """
        # Well-calibrated model: lots of evidence (100 observations)
        calibrated = Opinion.from_evidence(positive=80, negative=20)
        # Unreliable model: very little evidence (5 observations)
        unreliable = Opinion.from_evidence(positive=3, negative=0)

        # Both have similar projected probability (~0.79-0.80)
        assert abs(calibrated.to_confidence() - unreliable.to_confidence()) < 0.05

        # But vastly different uncertainty
        assert calibrated.uncertainty < 0.03    # ~0.02: lots of evidence
        assert unreliable.uncertainty > 0.3     # ~0.4: very little evidence

        # A downstream agent with a threshold can reject the unreliable one
        UNCERTAINTY_THRESHOLD = 0.1
        assert calibrated.uncertainty < UNCERTAINTY_THRESHOLD   # accept
        assert unreliable.uncertainty > UNCERTAINTY_THRESHOLD    # reject → human-in-the-loop
