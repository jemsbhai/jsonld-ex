"""Tests for the quantization-to-SL-uncertainty bridge.

TDD RED phase — all tests should FAIL until implementation is complete.

This module bridges vector quantization metadata to Subjective Logic
opinions, modeling quantization distortion as epistemic uncertainty.

Mathematical foundation:
  For b-bit quantization of unit vectors, inner product distortion is:
    D(b) = k_method · 4^{-b}
  where k_method depends on the algorithm.

  The SL uncertainty is:
    u = min(1, sqrt(D(b)))
  mapping RMS error to uncertainty mass.

This is a novel contribution: unlike heuristic ML confidence, quantization
distortion bounds are provable and well-characterized by information theory.
"""

import math
import pytest
from jsonld_ex.quantization_bridge import (
    DISTORTION_CONSTANTS,
    quantization_distortion,
    distortion_to_uncertainty,
    similarity_to_confidence,
    quantization_to_opinion,
)
from jsonld_ex.confidence_algebra import Opinion


# ── DISTORTION_CONSTANTS registry ──────────────────────────────────────


class TestDistortionConstants:
    """The distortion constant k_method for known quantization methods."""

    def test_scalar_defined(self):
        assert "scalar" in DISTORTION_CONSTANTS

    def test_turboquant_defined(self):
        assert "turboquant" in DISTORTION_CONSTANTS

    def test_polarquant_defined(self):
        assert "polarquant" in DISTORTION_CONSTANTS

    def test_qjl_defined(self):
        assert "qjl" in DISTORTION_CONSTANTS

    def test_product_quantization_defined(self):
        assert "product_quantization" in DISTORTION_CONSTANTS

    def test_all_positive(self):
        for method, k in DISTORTION_CONSTANTS.items():
            assert k > 0, f"{method} has non-positive k={k}"

    def test_turboquant_better_than_scalar(self):
        """TurboQuant achieves lower distortion than naive scalar."""
        assert DISTORTION_CONSTANTS["turboquant"] <= DISTORTION_CONSTANTS["scalar"]


# ── quantization_distortion() ──────────────────────────────────────────


class TestQuantizationDistortion:
    """D(b) = k_method · 4^{-b} — the normalized distortion rate."""

    def test_scalar_4bit(self):
        d = quantization_distortion(4, "scalar")
        k = DISTORTION_CONSTANTS["scalar"]
        assert d == pytest.approx(k / 256.0)

    def test_turboquant_4bit(self):
        d = quantization_distortion(4, "turboquant")
        k = DISTORTION_CONSTANTS["turboquant"]
        assert d == pytest.approx(k / 256.0)

    def test_distortion_decreases_with_bits(self):
        """More bits → less distortion, strictly monotonic."""
        prev = quantization_distortion(1, "scalar")
        for b in range(2, 17):
            curr = quantization_distortion(b, "scalar")
            assert curr < prev, f"D({b}) >= D({b-1})"
            prev = curr

    def test_high_bits_near_zero(self):
        """At 16 bits, distortion should be negligible."""
        d = quantization_distortion(16, "scalar")
        assert d < 1e-9

    def test_1bit_meaningful_distortion(self):
        """At 1 bit, distortion should be substantial."""
        d = quantization_distortion(1, "scalar")
        assert d > 0.01

    def test_unknown_method_uses_scalar(self):
        """Unknown methods fall back to scalar constant."""
        d_unknown = quantization_distortion(4, "my_custom_method")
        d_scalar = quantization_distortion(4, "scalar")
        assert d_unknown == d_scalar

    def test_bit_width_zero_raises(self):
        with pytest.raises(ValueError, match="bit_width"):
            quantization_distortion(0, "scalar")

    def test_bit_width_negative_raises(self):
        with pytest.raises(ValueError, match="bit_width"):
            quantization_distortion(-1, "scalar")

    def test_all_methods_same_formula(self):
        """All methods use D = k · 4^{-b}, just with different k."""
        for method, k in DISTORTION_CONSTANTS.items():
            d = quantization_distortion(8, method)
            assert d == pytest.approx(k * 4.0 ** (-8))


# ── distortion_to_uncertainty() ────────────────────────────────────────


class TestDistortionToUncertainty:
    """u = min(1.0, sqrt(D)) — maps MSE to SL uncertainty."""

    def test_zero_distortion(self):
        assert distortion_to_uncertainty(0.0) == 0.0

    def test_small_distortion(self):
        u = distortion_to_uncertainty(0.01)
        assert u == pytest.approx(0.1)

    def test_unit_distortion(self):
        """D=1.0 maps to u=1.0."""
        assert distortion_to_uncertainty(1.0) == pytest.approx(1.0)

    def test_large_distortion_capped(self):
        """D > 1.0 is capped at u = 1.0."""
        assert distortion_to_uncertainty(4.0) == 1.0

    def test_negative_distortion_clamped(self):
        """Negative distortion (should not happen) clamped to 0."""
        assert distortion_to_uncertainty(-0.01) == 0.0

    def test_monotonically_increasing(self):
        """More distortion → more uncertainty."""
        prev = distortion_to_uncertainty(0.0)
        for d_val in [0.001, 0.01, 0.1, 0.5, 1.0]:
            curr = distortion_to_uncertainty(d_val)
            assert curr >= prev
            prev = curr

    def test_sqrt_relationship(self):
        """u = sqrt(D) for D in [0, 1]."""
        for d_val in [0.04, 0.09, 0.16, 0.25, 0.36]:
            u = distortion_to_uncertainty(d_val)
            assert u == pytest.approx(math.sqrt(d_val))


# ── similarity_to_confidence() ─────────────────────────────────────────


class TestSimilarityToConfidence:
    """Map similarity scores to confidence in [0, 1]."""

    def test_cosine_range_default(self):
        """Default range is [-1, 1] (cosine similarity)."""
        assert similarity_to_confidence(1.0) == pytest.approx(1.0)
        assert similarity_to_confidence(-1.0) == pytest.approx(0.0)
        assert similarity_to_confidence(0.0) == pytest.approx(0.5)

    def test_midpoint(self):
        c = similarity_to_confidence(0.5)
        assert c == pytest.approx(0.75)

    def test_custom_range(self):
        """Euclidean distance: [0, 10] inverted — 0 is most similar."""
        c = similarity_to_confidence(0.0, range_min=0.0, range_max=10.0)
        assert c == pytest.approx(0.0)
        c = similarity_to_confidence(10.0, range_min=0.0, range_max=10.0)
        assert c == pytest.approx(1.0)

    def test_clamped_above(self):
        """Values above range_max are clamped to 1.0."""
        c = similarity_to_confidence(2.0, range_min=-1.0, range_max=1.0)
        assert c == pytest.approx(1.0)

    def test_clamped_below(self):
        """Values below range_min are clamped to 0.0."""
        c = similarity_to_confidence(-2.0, range_min=-1.0, range_max=1.0)
        assert c == pytest.approx(0.0)

    def test_degenerate_range_raises(self):
        """range_min == range_max is degenerate."""
        with pytest.raises(ValueError, match="range"):
            similarity_to_confidence(0.5, range_min=1.0, range_max=1.0)

    def test_inverted_range_raises(self):
        """range_min > range_max is invalid."""
        with pytest.raises(ValueError, match="range"):
            similarity_to_confidence(0.5, range_min=1.0, range_max=-1.0)


# ── quantization_to_opinion() ─────────────────────────────────────────


class TestQuantizationToOpinion:
    """Create an SL Opinion from a quantized similarity score."""

    def test_returns_opinion(self):
        op = quantization_to_opinion(0.9, bit_width=4, method="scalar")
        assert isinstance(op, Opinion)

    def test_additivity(self):
        """b + d + u must equal 1."""
        op = quantization_to_opinion(0.7, bit_width=4, method="turboquant")
        assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)

    def test_high_bits_low_uncertainty(self):
        """At 16 bits, opinion should be nearly dogmatic."""
        op = quantization_to_opinion(0.8, bit_width=16, method="scalar")
        assert op.uncertainty < 0.001

    def test_low_bits_high_uncertainty(self):
        """At 1 bit, opinion should have substantial uncertainty."""
        op = quantization_to_opinion(0.8, bit_width=1, method="scalar")
        assert op.uncertainty > 0.1

    def test_uncertainty_decreases_with_bits(self):
        """More bits → less uncertainty, monotonically."""
        prev_u = quantization_to_opinion(0.5, bit_width=1, method="scalar").uncertainty
        for b in range(2, 9):
            curr_u = quantization_to_opinion(0.5, bit_width=b, method="scalar").uncertainty
            assert curr_u <= prev_u, f"u({b}) > u({b-1})"
            prev_u = curr_u

    def test_perfect_similarity(self):
        """Cosine similarity 1.0 → high confidence (belief)."""
        op = quantization_to_opinion(1.0, bit_width=8, method="scalar")
        assert op.belief > 0.9

    def test_zero_similarity(self):
        """Cosine similarity 0.0 → midpoint confidence."""
        op = quantization_to_opinion(0.0, bit_width=8, method="scalar")
        # 0.0 cosine → 0.5 confidence → belief ≈ 0.5 * (1-u)
        assert op.belief == pytest.approx(0.5 * (1.0 - op.uncertainty), abs=1e-6)

    def test_negative_similarity(self):
        """Cosine similarity -1.0 → low confidence (high disbelief)."""
        op = quantization_to_opinion(-1.0, bit_width=8, method="scalar")
        assert op.disbelief > 0.9

    def test_custom_range(self):
        """Custom similarity range [0, 1] for dot product."""
        op = quantization_to_opinion(
            0.5, bit_width=4, method="scalar",
            similarity_range=(0.0, 1.0),
        )
        # 0.5 in [0,1] maps to confidence 0.5
        assert op.belief == pytest.approx(0.5 * (1.0 - op.uncertainty), abs=1e-6)

    def test_custom_base_rate(self):
        op = quantization_to_opinion(0.8, bit_width=4, method="scalar", base_rate=0.3)
        assert op.base_rate == pytest.approx(0.3)

    def test_projected_probability_preserves_confidence(self):
        """P(ω) should approximate the normalized confidence.

        When base_rate matches the confidence-implied prior, projected
        probability should equal the normalized similarity.
        """
        # For base_rate=0.5 and symmetric mapping, P(ω) ≈ confidence
        sim = 0.6
        op = quantization_to_opinion(sim, bit_width=8, method="scalar")
        expected_conf = (sim + 1.0) / 2.0  # 0.8
        # P(ω) = b + a·u ≈ expected_conf (exact when a=0.5)
        assert op.projected_probability() == pytest.approx(expected_conf, abs=0.01)

    def test_turboquant_less_uncertainty_than_scalar(self):
        """TurboQuant's better distortion → less uncertainty."""
        op_scalar = quantization_to_opinion(0.5, bit_width=4, method="scalar")
        op_turbo = quantization_to_opinion(0.5, bit_width=4, method="turboquant")
        assert op_turbo.uncertainty <= op_scalar.uncertainty

    def test_bit_width_zero_raises(self):
        with pytest.raises(ValueError):
            quantization_to_opinion(0.5, bit_width=0, method="scalar")


# ── Round-trip consistency ─────────────────────────────────────────────


class TestRoundTripConsistency:
    """Verify that the bridge is consistent with the rest of jsonld-ex."""

    def test_opinion_to_jsonld_roundtrip(self):
        """Opinion from quantization can serialize/deserialize."""
        op = quantization_to_opinion(0.75, bit_width=4, method="turboquant")
        jsonld = op.to_jsonld()
        restored = Opinion.from_jsonld(jsonld)
        assert restored.belief == pytest.approx(op.belief)
        assert restored.disbelief == pytest.approx(op.disbelief)
        assert restored.uncertainty == pytest.approx(op.uncertainty)
        assert restored.base_rate == pytest.approx(op.base_rate)

    def test_fuse_quantized_opinions(self):
        """Two quantized similarity opinions can be fused."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        op1 = quantization_to_opinion(0.8, bit_width=4, method="turboquant")
        op2 = quantization_to_opinion(0.7, bit_width=8, method="scalar")
        fused = cumulative_fuse(op1, op2)
        assert isinstance(fused, Opinion)
        # Fused should have less uncertainty than either input
        assert fused.uncertainty <= max(op1.uncertainty, op2.uncertainty)

    def test_trust_discount_quantized_opinion(self):
        """A quantized opinion can be trust-discounted."""
        from jsonld_ex.confidence_algebra import trust_discount

        trust = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        op = quantization_to_opinion(0.8, bit_width=4, method="turboquant")
        discounted = trust_discount(trust, op)
        assert isinstance(discounted, Opinion)
        # Trust discount increases uncertainty
        assert discounted.uncertainty >= op.uncertainty
