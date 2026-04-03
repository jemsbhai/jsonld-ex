"""Tests for Opinion fast-path construction optimization.

RED phase: tests that verify _create_unchecked produces identical results
to the validated constructor, and that all operators still produce correct
outputs after optimization.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest packages/python/tests/test_opinion_fastpath.py -v
"""
import math
import pytest
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
)
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay


class TestUncheckedConstructor:
    """_create_unchecked must produce identical Opinion objects."""

    def test_unchecked_exists(self):
        """The fast-path constructor must exist."""
        assert hasattr(Opinion, '_create_unchecked')

    def test_unchecked_matches_validated(self):
        """Unchecked produces the same object as validated constructor."""
        validated = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.6)
        unchecked = Opinion._create_unchecked(0.7, 0.2, 0.1, 0.6)
        assert unchecked.belief == validated.belief
        assert unchecked.disbelief == validated.disbelief
        assert unchecked.uncertainty == validated.uncertainty
        assert unchecked.base_rate == validated.base_rate

    def test_unchecked_projected_probability(self):
        """Unchecked opinion computes correct projected probability."""
        op = Opinion._create_unchecked(0.7, 0.2, 0.1, 0.5)
        assert abs(op.projected_probability() - (0.7 + 0.5 * 0.1)) < 1e-12

    def test_unchecked_is_frozen(self):
        """Unchecked opinion must still be immutable."""
        op = Opinion._create_unchecked(0.7, 0.2, 0.1, 0.5)
        with pytest.raises(AttributeError):
            op.belief = 0.9

    def test_unchecked_equality(self):
        """Unchecked and validated opinions with same values must be equal."""
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
        b = Opinion._create_unchecked(0.5, 0.3, 0.2, 0.5)
        assert a == b

    def test_unchecked_hash(self):
        """Unchecked and validated opinions must hash identically."""
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2, base_rate=0.5)
        b = Opinion._create_unchecked(0.5, 0.3, 0.2, 0.5)
        assert hash(a) == hash(b)


class TestFromConfidenceAfterOptimization:
    """from_confidence must produce identical results after optimization."""

    def test_dogmatic(self):
        op = Opinion.from_confidence(0.85)
        assert abs(op.belief - 0.85) < 1e-12
        assert abs(op.disbelief - 0.15) < 1e-12
        assert abs(op.uncertainty) < 1e-12

    def test_with_uncertainty(self):
        op = Opinion.from_confidence(0.85, uncertainty=0.1)
        assert abs(op.belief - 0.85 * 0.9) < 1e-12
        assert abs(op.disbelief - 0.15 * 0.9) < 1e-12
        assert abs(op.uncertainty - 0.1) < 1e-12

    def test_projected_probability_preserved(self):
        """P(w) must equal confidence when base_rate matches."""
        for c in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            op = Opinion.from_confidence(c, uncertainty=0.0)
            assert abs(op.projected_probability() - c) < 1e-12, f"Failed at c={c}"

    def test_boundary_values(self):
        op0 = Opinion.from_confidence(0.0)
        assert abs(op0.belief) < 1e-12
        assert abs(op0.disbelief - 1.0) < 1e-12

        op1 = Opinion.from_confidence(1.0)
        assert abs(op1.belief - 1.0) < 1e-12
        assert abs(op1.disbelief) < 1e-12

    def test_full_uncertainty(self):
        op = Opinion.from_confidence(0.5, uncertainty=1.0)
        assert abs(op.belief) < 1e-12
        assert abs(op.disbelief) < 1e-12
        assert abs(op.uncertainty - 1.0) < 1e-12

    def test_validation_still_rejects_bad_input(self):
        """Public API must still reject invalid inputs."""
        with pytest.raises(ValueError):
            Opinion.from_confidence(-0.1)
        with pytest.raises(ValueError):
            Opinion.from_confidence(1.5)
        with pytest.raises(TypeError):
            Opinion.from_confidence("hello")
        with pytest.raises(ValueError):
            Opinion.from_confidence(float('nan'))


class TestFromEvidenceAfterOptimization:
    """from_evidence must produce identical results."""

    def test_basic(self):
        op = Opinion.from_evidence(positive=10, negative=5)
        total = 10 + 5 + 2.0  # default prior_weight=2
        assert abs(op.belief - 10 / total) < 1e-12
        assert abs(op.disbelief - 5 / total) < 1e-12
        assert abs(op.uncertainty - 2.0 / total) < 1e-12

    def test_additivity(self):
        op = Opinion.from_evidence(positive=100, negative=50)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-12


class TestOperatorsAfterOptimization:
    """All operators must produce identical results."""

    def test_cumulative_fuse(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        b = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.5)
        result = cumulative_fuse(a, b)
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-12
        assert 0.0 <= result.belief <= 1.0
        assert 0.0 <= result.disbelief <= 1.0
        assert 0.0 <= result.uncertainty <= 1.0

    def test_averaging_fuse(self):
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        b = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2, base_rate=0.5)
        result = averaging_fuse(a, b)
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-12

    def test_trust_discount(self):
        trust = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)
        source = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05, base_rate=0.5)
        result = trust_discount(trust, source)
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-12
        # Trust discount reduces belief
        assert result.belief <= source.belief
        # Trust discount increases uncertainty
        assert result.uncertainty >= source.uncertainty

    def test_deduce(self):
        x = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        y_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)
        y_nx = Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2, base_rate=0.5)
        result = deduce(x, y_x, y_nx)
        assert abs(result.belief + result.disbelief + result.uncertainty - 1.0) < 1e-12

    def test_decay_opinion(self):
        op = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1, base_rate=0.5)
        decayed = decay_opinion(op, elapsed=5.0, half_life=2.0, decay_fn=exponential_decay)
        assert abs(decayed.belief + decayed.disbelief + decayed.uncertainty - 1.0) < 1e-12
        # Decay increases uncertainty
        assert decayed.uncertainty >= op.uncertainty

    def test_chain_of_operations(self):
        """A realistic pipeline: from_confidence -> trust_discount -> fuse -> decay."""
        raw1 = Opinion.from_confidence(0.85, uncertainty=0.1)
        raw2 = Opinion.from_confidence(0.72, uncertainty=0.15)
        trust = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

        disc1 = trust_discount(trust, raw1)
        disc2 = trust_discount(trust, raw2)
        fused = cumulative_fuse(disc1, disc2)
        final = decay_opinion(fused, elapsed=3.0, half_life=10.0, decay_fn=exponential_decay)

        assert abs(final.belief + final.disbelief + final.uncertainty - 1.0) < 1e-12
        assert 0.0 <= final.belief <= 1.0
        assert 0.0 <= final.disbelief <= 1.0
        assert 0.0 <= final.uncertainty <= 1.0
        # The projected probability should be reasonable
        pp = final.projected_probability()
        assert 0.0 <= pp <= 1.0
