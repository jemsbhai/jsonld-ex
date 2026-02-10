"""Tests for the formal Confidence Algebra (Subjective Logic foundation).

The confidence algebra formalizes uncertainty representation and propagation
in jsonld-ex, grounding it in Jøsang's Subjective Logic framework.

An Opinion ω = (b, d, u, a) where:
  - b ∈ [0,1]: belief (evidence FOR the proposition)
  - d ∈ [0,1]: disbelief (evidence AGAINST the proposition)
  - u ∈ [0,1]: uncertainty (lack of evidence)
  - a ∈ [0,1]: base rate (prior probability)
  - Constraint: b + d + u = 1

The projected probability is: P(ω) = b + a·u
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# Step 1: Opinion construction and validation
# ═══════════════════════════════════════════════════════════════════


class TestOpinionConstruction:
    """Test that Opinion objects are created correctly with validation."""

    def test_valid_opinion(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        assert o.belief == 0.7
        assert o.disbelief == 0.2
        assert o.uncertainty == 0.1
        assert o.base_rate == 0.5  # default

    def test_custom_base_rate(self):
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.7)
        assert o.base_rate == 0.7

    def test_dogmatic_opinion(self):
        """u = 0: full certainty, no remaining uncertainty."""
        o = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        assert o.uncertainty == 0.0

    def test_vacuous_opinion(self):
        """u = 1: total ignorance, no evidence at all."""
        o = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert o.uncertainty == 1.0

    def test_absolute_belief(self):
        """b = 1: absolute certainty FOR the proposition."""
        o = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert o.belief == 1.0

    def test_absolute_disbelief(self):
        """d = 1: absolute certainty AGAINST the proposition."""
        o = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        assert o.disbelief == 1.0


class TestOpinionValidation:
    """Test that invalid opinions are rejected."""

    def test_sum_exceeds_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            Opinion(belief=0.5, disbelief=0.4, uncertainty=0.2)

    def test_sum_below_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            Opinion(belief=0.3, disbelief=0.3, uncertainty=0.3)

    def test_negative_belief(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=-0.1, disbelief=0.6, uncertainty=0.5)

    def test_negative_disbelief(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=0.5, disbelief=-0.1, uncertainty=0.6)

    def test_negative_uncertainty(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=0.5, disbelief=0.5, uncertainty=-0.0001)

    def test_belief_above_one(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=1.1, disbelief=0.0, uncertainty=0.0)

    def test_negative_base_rate(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=-0.1)

    def test_base_rate_above_one(self):
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=1.1)

    def test_nan_belief(self):
        with pytest.raises(ValueError, match="must be finite"):
            Opinion(belief=float("nan"), disbelief=0.5, uncertainty=0.5)

    def test_inf_disbelief(self):
        with pytest.raises(ValueError, match="must be finite"):
            Opinion(belief=0.0, disbelief=float("inf"), uncertainty=0.0)

    def test_non_numeric_belief(self):
        with pytest.raises(TypeError, match="must be a number"):
            Opinion(belief="0.5", disbelief=0.3, uncertainty=0.2)  # type: ignore

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="must be a number"):
            Opinion(belief=True, disbelief=0.0, uncertainty=0.0)  # type: ignore

    def test_tolerance_for_floating_point(self):
        """b + d + u may not be exactly 1.0 due to IEEE 754.
        We allow a tolerance of 1e-9."""
        # 0.1 + 0.2 + 0.7 can have floating point issues
        o = Opinion(belief=0.1, disbelief=0.2, uncertainty=0.7)
        assert o.belief == pytest.approx(0.1)


# ═══════════════════════════════════════════════════════════════════
# Step 2: Projected probability and scalar conversions
# ═══════════════════════════════════════════════════════════════════


class TestProjectedProbability:
    """P(ω) = b + a·u — the expected probability."""

    def test_dogmatic_opinion(self):
        """With no uncertainty, P(ω) = b."""
        o = Opinion(belief=0.8, disbelief=0.2, uncertainty=0.0)
        assert o.projected_probability() == pytest.approx(0.8)

    def test_vacuous_opinion_default_base_rate(self):
        """With total uncertainty, P(ω) = a = 0.5."""
        o = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert o.projected_probability() == pytest.approx(0.5)

    def test_vacuous_opinion_custom_base_rate(self):
        """With total uncertainty and a = 0.3, P(ω) = 0.3."""
        o = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.3)
        assert o.projected_probability() == pytest.approx(0.3)

    def test_mixed_opinion(self):
        """b=0.6, d=0.1, u=0.3, a=0.5 → P = 0.6 + 0.5*0.3 = 0.75."""
        o = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        assert o.projected_probability() == pytest.approx(0.75)

    def test_absolute_belief(self):
        o = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert o.projected_probability() == pytest.approx(1.0)

    def test_absolute_disbelief(self):
        o = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        assert o.projected_probability() == pytest.approx(0.0)


class TestFromConfidence:
    """Map scalar confidence c ∈ [0,1] to an Opinion."""

    def test_default_mapping_is_dogmatic(self):
        """By default, confidence maps to a dogmatic opinion (u=0)."""
        o = Opinion.from_confidence(0.8)
        assert o.belief == pytest.approx(0.8)
        assert o.disbelief == pytest.approx(0.2)
        assert o.uncertainty == pytest.approx(0.0)

    def test_with_uncertainty(self):
        """Explicit uncertainty redistributes mass from belief/disbelief."""
        o = Opinion.from_confidence(0.8, uncertainty=0.3)
        assert o.uncertainty == pytest.approx(0.3)
        # Remaining 0.7 split proportionally: belief = 0.8*0.7 = 0.56
        assert o.belief == pytest.approx(0.56)
        assert o.disbelief == pytest.approx(0.14)
        # b + d + u = 0.56 + 0.14 + 0.3 = 1.0
        assert o.belief + o.disbelief + o.uncertainty == pytest.approx(1.0)

    def test_full_uncertainty(self):
        """u=1.0 → vacuous opinion regardless of confidence."""
        o = Opinion.from_confidence(0.9, uncertainty=1.0)
        assert o.belief == pytest.approx(0.0)
        assert o.disbelief == pytest.approx(0.0)
        assert o.uncertainty == pytest.approx(1.0)

    def test_zero_confidence(self):
        o = Opinion.from_confidence(0.0)
        assert o.belief == pytest.approx(0.0)
        assert o.disbelief == pytest.approx(1.0)
        assert o.uncertainty == pytest.approx(0.0)

    def test_perfect_confidence(self):
        o = Opinion.from_confidence(1.0)
        assert o.belief == pytest.approx(1.0)
        assert o.disbelief == pytest.approx(0.0)

    def test_custom_base_rate(self):
        o = Opinion.from_confidence(0.7, base_rate=0.3)
        assert o.base_rate == 0.3

    def test_roundtrip_dogmatic(self):
        """from_confidence → projected_probability should recover c for dogmatic."""
        for c in [0.0, 0.1, 0.5, 0.73, 0.99, 1.0]:
            o = Opinion.from_confidence(c)
            assert o.projected_probability() == pytest.approx(c, abs=1e-9)

    def test_invalid_confidence_above_one(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(1.5)

    def test_invalid_confidence_negative(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(-0.1)

    def test_invalid_uncertainty_above_one(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(0.5, uncertainty=1.5)

    def test_invalid_uncertainty_negative(self):
        with pytest.raises(ValueError):
            Opinion.from_confidence(0.5, uncertainty=-0.1)


class TestToConfidence:
    """to_confidence() is an alias for projected_probability()."""

    def test_alias(self):
        o = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        assert o.to_confidence() == o.projected_probability()


class TestFromEvidenceCounts:
    """Map positive/negative evidence counts to an Opinion.

    Using Jøsang's evidence-to-opinion mapping:
      b = r / (r + s + W)
      d = s / (r + s + W)
      u = W / (r + s + W)
    where r = positive evidence, s = negative evidence, W = non-informative
    prior weight (default W = 2).
    """

    def test_no_evidence(self):
        """Zero evidence → vacuous opinion."""
        o = Opinion.from_evidence(positive=0, negative=0)
        assert o.belief == pytest.approx(0.0)
        assert o.disbelief == pytest.approx(0.0)
        assert o.uncertainty == pytest.approx(1.0)

    def test_only_positive(self):
        """10 positive, 0 negative → high belief, some uncertainty."""
        o = Opinion.from_evidence(positive=10, negative=0)
        # b = 10/(10+0+2) = 10/12 ≈ 0.833
        assert o.belief == pytest.approx(10 / 12)
        assert o.disbelief == pytest.approx(0.0)
        assert o.uncertainty == pytest.approx(2 / 12)

    def test_only_negative(self):
        """0 positive, 10 negative → high disbelief."""
        o = Opinion.from_evidence(positive=0, negative=10)
        assert o.belief == pytest.approx(0.0)
        assert o.disbelief == pytest.approx(10 / 12)
        assert o.uncertainty == pytest.approx(2 / 12)

    def test_balanced_evidence(self):
        """Equal positive and negative → uncertainty dominant."""
        o = Opinion.from_evidence(positive=5, negative=5)
        # b = 5/12, d = 5/12, u = 2/12
        assert o.belief == pytest.approx(5 / 12)
        assert o.disbelief == pytest.approx(5 / 12)
        assert o.uncertainty == pytest.approx(2 / 12)

    def test_large_evidence_reduces_uncertainty(self):
        """More evidence → less uncertainty."""
        small = Opinion.from_evidence(positive=2, negative=1)
        large = Opinion.from_evidence(positive=200, negative=100)
        assert large.uncertainty < small.uncertainty

    def test_custom_prior_weight(self):
        """W=10 makes the prior more influential."""
        o = Opinion.from_evidence(positive=5, negative=5, prior_weight=10)
        # b = 5/20, d = 5/20, u = 10/20
        assert o.belief == pytest.approx(0.25)
        assert o.uncertainty == pytest.approx(0.5)

    def test_negative_evidence_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            Opinion.from_evidence(positive=-1, negative=0)

    def test_negative_prior_weight_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            Opinion.from_evidence(positive=1, negative=0, prior_weight=0)


# ═══════════════════════════════════════════════════════════════════
# Equality and representation
# ═══════════════════════════════════════════════════════════════════


class TestOpinionEquality:
    def test_equal_opinions(self):
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        b = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        assert a == b

    def test_different_base_rate(self):
        a = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=0.3)
        b = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=0.7)
        assert a != b

    def test_repr_contains_components(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        r = repr(o)
        assert "0.7" in r
        assert "0.2" in r
        assert "0.1" in r


class TestOpinionSerialization:
    """Opinions must serialize to/from JSON-LD annotation format."""

    def test_to_jsonld(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.5)
        j = o.to_jsonld()
        assert j["@type"] == "Opinion"
        assert j["belief"] == 0.7
        assert j["disbelief"] == 0.2
        assert j["uncertainty"] == 0.1
        assert j["baseRate"] == 0.5

    def test_from_jsonld(self):
        j = {
            "@type": "Opinion",
            "belief": 0.6,
            "disbelief": 0.3,
            "uncertainty": 0.1,
            "baseRate": 0.4,
        }
        o = Opinion.from_jsonld(j)
        assert o.belief == 0.6
        assert o.disbelief == 0.3
        assert o.uncertainty == 0.1
        assert o.base_rate == 0.4

    def test_roundtrip(self):
        original = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.6)
        restored = Opinion.from_jsonld(original.to_jsonld())
        assert original == restored
