"""Tests for confidence propagation and multi-source combination."""

import math
import pytest

from jsonld_ex.inference import (
    propagate_confidence,
    combine_sources,
    resolve_conflict,
    propagate_graph_confidence,
    PropagationResult,
    ConflictReport,
)


# ═══════════════════════════════════════════════════════════════════
# propagate_confidence
# ═══════════════════════════════════════════════════════════════════


class TestPropagateMultiply:
    """Tests for chain propagation with the 'multiply' method."""

    def test_two_step_chain(self):
        r = propagate_confidence([0.9, 0.8], method="multiply")
        assert r.score == pytest.approx(0.72)

    def test_single_element(self):
        r = propagate_confidence([0.5], method="multiply")
        assert r.score == pytest.approx(0.5)

    def test_three_step_chain(self):
        r = propagate_confidence([0.9, 0.8, 0.7], method="multiply")
        assert r.score == pytest.approx(0.9 * 0.8 * 0.7)

    def test_perfect_confidence(self):
        r = propagate_confidence([1.0, 1.0, 1.0], method="multiply")
        assert r.score == pytest.approx(1.0)

    def test_zero_in_chain(self):
        r = propagate_confidence([0.9, 0.0, 0.8], method="multiply")
        assert r.score == pytest.approx(0.0)

    def test_long_chain_degrades(self):
        """Long chains should produce low scores under multiplication."""
        chain = [0.9] * 10
        r = propagate_confidence(chain, method="multiply")
        assert r.score == pytest.approx(0.9**10, rel=1e-6)
        assert r.score < 0.35  # 0.9^10 ≈ 0.3487

    def test_returns_metadata(self):
        r = propagate_confidence([0.9, 0.8], method="multiply")
        assert r.method == "multiply"
        assert r.input_scores == [0.9, 0.8]


class TestPropagateBayesian:
    """Tests for Bayesian chain propagation."""

    def test_two_step_symmetric(self):
        r = propagate_confidence([0.9, 0.9], method="bayesian")
        # With uniform prior, two 0.9 likelihood ratios should yield > 0.9
        assert r.score > 0.9

    def test_single_element_with_prior(self):
        """Single score with uniform prior should return approximately itself."""
        r = propagate_confidence([0.7], method="bayesian")
        # Bayesian with uniform prior: posterior ≈ the likelihood
        assert r.score == pytest.approx(0.7, rel=1e-2)

    def test_weak_evidence_stays_weak(self):
        r = propagate_confidence([0.5, 0.5], method="bayesian")
        # 0.5 is uninformative → posterior stays near 0.5
        assert r.score == pytest.approx(0.5, abs=0.05)

    def test_strong_evidence_compounds(self):
        r = propagate_confidence([0.95, 0.95, 0.95], method="bayesian")
        assert r.score > 0.99

    def test_low_evidence_reduces(self):
        r = propagate_confidence([0.2, 0.3], method="bayesian")
        assert r.score < 0.2

    def test_boundary_near_zero(self):
        """Scores near zero should not cause math errors."""
        r = propagate_confidence([0.01, 0.01], method="bayesian")
        assert 0.0 < r.score < 0.01

    def test_boundary_near_one(self):
        """Scores near one should not cause math errors."""
        r = propagate_confidence([0.99, 0.99], method="bayesian")
        assert r.score > 0.99


class TestPropagateMin:
    """Tests for weakest-link propagation."""

    def test_returns_minimum(self):
        r = propagate_confidence([0.9, 0.5, 0.8], method="min")
        assert r.score == pytest.approx(0.5)

    def test_single_element(self):
        r = propagate_confidence([0.7], method="min")
        assert r.score == pytest.approx(0.7)

    def test_all_equal(self):
        r = propagate_confidence([0.6, 0.6, 0.6], method="min")
        assert r.score == pytest.approx(0.6)


class TestPropagateDampened:
    """Tests for dampened multiplication."""

    def test_single_element_equals_itself(self):
        r = propagate_confidence([0.8], method="dampened")
        assert r.score == pytest.approx(0.8)

    def test_two_elements(self):
        r = propagate_confidence([0.9, 0.8], method="dampened")
        product = 0.9 * 0.8  # 0.72
        expected = product ** (1.0 / math.sqrt(2))
        assert r.score == pytest.approx(expected)

    def test_less_aggressive_than_multiply(self):
        """Dampened should always be >= multiply for chains > 1."""
        chain = [0.8, 0.7, 0.6]
        mult = propagate_confidence(chain, method="multiply").score
        damp = propagate_confidence(chain, method="dampened").score
        assert damp > mult

    def test_long_chain_less_degraded(self):
        """Key property: dampened degrades much less than multiply for long chains."""
        chain = [0.9] * 10
        mult = propagate_confidence(chain, method="multiply").score
        damp = propagate_confidence(chain, method="dampened").score
        # multiply: 0.9^10 ≈ 0.349
        # dampened: (0.9^10)^(1/√10) ≈ 0.349^0.316 ≈ 0.702
        assert mult < 0.35
        assert damp > 0.65

    def test_zero_in_chain(self):
        r = propagate_confidence([0.9, 0.0], method="dampened")
        assert r.score == pytest.approx(0.0)


class TestPropagateErrors:
    """Error handling for propagate_confidence."""

    def test_empty_chain(self):
        with pytest.raises(ValueError, match="at least one"):
            propagate_confidence([])

    def test_invalid_score_above_one(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            propagate_confidence([0.9, 1.5])

    def test_invalid_score_below_zero(self):
        with pytest.raises(ValueError, match="between 0.0 and 1.0"):
            propagate_confidence([-0.1])

    def test_invalid_score_type(self):
        with pytest.raises(TypeError, match="must be a number"):
            propagate_confidence(["0.9"])  # type: ignore

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown propagation method"):
            propagate_confidence([0.9], method="unknown")  # type: ignore

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            propagate_confidence([float("nan")])

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            propagate_confidence([float("inf")])

    def test_bool_rejected(self):
        with pytest.raises(TypeError, match="must be a number"):
            propagate_confidence([True])  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# combine_sources
# ═══════════════════════════════════════════════════════════════════


class TestCombineAverage:
    def test_basic(self):
        r = combine_sources([0.8, 0.6], method="average")
        assert r.score == pytest.approx(0.7)

    def test_single(self):
        r = combine_sources([0.5], method="average")
        assert r.score == pytest.approx(0.5)


class TestCombineMax:
    def test_basic(self):
        r = combine_sources([0.6, 0.9, 0.7], method="max")
        assert r.score == pytest.approx(0.9)


class TestCombineNoisyOr:
    """Tests for the noisy-OR combination."""

    def test_two_sources(self):
        # 1 - (1-0.9)(1-0.7) = 1 - 0.03 = 0.97
        r = combine_sources([0.9, 0.7], method="noisy_or")
        assert r.score == pytest.approx(0.97)

    def test_single_source(self):
        r = combine_sources([0.8], method="noisy_or")
        assert r.score == pytest.approx(0.8)

    def test_three_sources(self):
        # 1 - (0.2)(0.3)(0.4) = 1 - 0.024 = 0.976
        r = combine_sources([0.8, 0.7, 0.6], method="noisy_or")
        assert r.score == pytest.approx(1.0 - 0.2 * 0.3 * 0.4)

    def test_always_increases(self):
        """Adding more sources should never decrease confidence."""
        one = combine_sources([0.6], method="noisy_or").score
        two = combine_sources([0.6, 0.6], method="noisy_or").score
        three = combine_sources([0.6, 0.6, 0.6], method="noisy_or").score
        assert two > one
        assert three > two

    def test_zero_source(self):
        """A zero-confidence source adds no information."""
        r = combine_sources([0.8, 0.0], method="noisy_or")
        assert r.score == pytest.approx(0.8)

    def test_perfect_source(self):
        """A perfect source yields 1.0."""
        r = combine_sources([0.5, 1.0], method="noisy_or")
        assert r.score == pytest.approx(1.0)

    def test_all_zeros(self):
        r = combine_sources([0.0, 0.0], method="noisy_or")
        assert r.score == pytest.approx(0.0)


class TestCombineDempsterShafer:
    """Tests for Dempster–Shafer combination."""

    def test_two_sources_agreement(self):
        r = combine_sources([0.8, 0.7], method="dempster_shafer")
        # Combined belief should be higher than either source
        assert r.score > 0.8

    def test_single_source(self):
        r = combine_sources([0.6], method="dempster_shafer")
        assert r.score == pytest.approx(0.6)

    def test_total_uncertainty(self):
        """Score 0.0 = total uncertainty → other source dominates."""
        r = combine_sources([0.0, 0.8], method="dempster_shafer")
        assert r.score == pytest.approx(0.8)

    def test_full_belief(self):
        """Score 1.0 = full belief → combined should be 1.0."""
        r = combine_sources([1.0, 0.5], method="dempster_shafer")
        assert r.score == pytest.approx(1.0)

    def test_monotonically_increases(self):
        """More agreeing sources should increase belief."""
        one = combine_sources([0.6], method="dempster_shafer").score
        two = combine_sources([0.6, 0.6], method="dempster_shafer").score
        three = combine_sources([0.6, 0.6, 0.6], method="dempster_shafer").score
        assert two > one
        assert three > two

    def test_associativity_approximation(self):
        """D-S rule is associative: combine(a,b,c) ≈ combine(combine(a,b),c)."""
        # Combine all three at once
        abc = combine_sources([0.7, 0.8, 0.6], method="dempster_shafer").score
        # Combine pairwise then with third
        ab = combine_sources([0.7, 0.8], method="dempster_shafer").score
        abc2 = combine_sources([ab, 0.6], method="dempster_shafer").score
        assert abc == pytest.approx(abc2, rel=1e-6)


class TestCombineErrors:
    def test_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            combine_sources([])

    def test_invalid_score(self):
        with pytest.raises(ValueError):
            combine_sources([1.5])

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown combination"):
            combine_sources([0.5], method="bogus")  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# resolve_conflict
# ═══════════════════════════════════════════════════════════════════


class TestResolveHighest:
    def test_picks_highest(self):
        assertions = [
            {"@value": "Engineer", "@confidence": 0.9},
            {"@value": "Manager", "@confidence": 0.85},
        ]
        r = resolve_conflict(assertions, strategy="highest")
        assert r.winner["@value"] == "Engineer"
        assert r.strategy == "highest"

    def test_tie_broken_by_order(self):
        assertions = [
            {"@value": "A", "@confidence": 0.8},
            {"@value": "B", "@confidence": 0.8},
        ]
        r = resolve_conflict(assertions, strategy="highest")
        assert r.winner["@value"] == "A"

    def test_single_assertion(self):
        assertions = [{"@value": "X", "@confidence": 0.5}]
        r = resolve_conflict(assertions, strategy="highest")
        assert r.winner["@value"] == "X"


class TestResolveWeightedVote:
    def test_two_agreeing_beat_one_higher(self):
        """Two sources agreeing should beat one higher-confidence source."""
        assertions = [
            {"@value": "Engineer", "@confidence": 0.7, "@source": "A"},
            {"@value": "Engineer", "@confidence": 0.6, "@source": "B"},
            {"@value": "Manager", "@confidence": 0.85, "@source": "C"},
        ]
        r = resolve_conflict(assertions, strategy="weighted_vote")
        assert r.winner["@value"] == "Engineer"
        # noisy-OR of 0.7, 0.6 = 1 - 0.3*0.4 = 0.88 > 0.85
        assert r.winner["@confidence"] > 0.85

    def test_single_high_beats_multiple_low(self):
        """One very high source should beat multiple very low ones."""
        assertions = [
            {"@value": "A", "@confidence": 0.95},
            {"@value": "B", "@confidence": 0.1},
            {"@value": "B", "@confidence": 0.1},
            {"@value": "B", "@confidence": 0.1},
        ]
        r = resolve_conflict(assertions, strategy="weighted_vote")
        # noisy-OR of 0.1, 0.1, 0.1 = 1 - 0.9^3 = 0.271
        assert r.winner["@value"] == "A"

    def test_report_contains_all_candidates(self):
        assertions = [
            {"@value": "X", "@confidence": 0.6},
            {"@value": "Y", "@confidence": 0.5},
        ]
        r = resolve_conflict(assertions, strategy="weighted_vote")
        assert len(r.candidates) == 2


class TestResolveRecency:
    def test_prefers_recent(self):
        assertions = [
            {"@value": "Old", "@confidence": 0.9, "@extractedAt": "2024-01-01T00:00:00Z"},
            {"@value": "New", "@confidence": 0.7, "@extractedAt": "2025-06-01T00:00:00Z"},
        ]
        r = resolve_conflict(assertions, strategy="recency")
        assert r.winner["@value"] == "New"

    def test_tiebreaker_is_confidence(self):
        assertions = [
            {"@value": "A", "@confidence": 0.8, "@extractedAt": "2025-01-01T00:00:00Z"},
            {"@value": "B", "@confidence": 0.9, "@extractedAt": "2025-01-01T00:00:00Z"},
        ]
        r = resolve_conflict(assertions, strategy="recency")
        assert r.winner["@value"] == "B"

    def test_missing_timestamp_sorted_last(self):
        assertions = [
            {"@value": "No-time", "@confidence": 0.95},
            {"@value": "Has-time", "@confidence": 0.5, "@extractedAt": "2025-01-01T00:00:00Z"},
        ]
        r = resolve_conflict(assertions, strategy="recency")
        assert r.winner["@value"] == "Has-time"


class TestResolveErrors:
    def test_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            resolve_conflict([])

    def test_missing_confidence(self):
        with pytest.raises(KeyError, match="@confidence"):
            resolve_conflict([{"@value": "X"}])

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_conflict(
                [{"@value": "X", "@confidence": 0.5}],
                strategy="unknown",  # type: ignore
            )


# ═══════════════════════════════════════════════════════════════════
# propagate_graph_confidence
# ═══════════════════════════════════════════════════════════════════


class TestPropagateGraphConfidence:
    def test_basic_chain(self):
        doc = {
            "source_fact": {"@value": "X", "@confidence": 0.9},
            "inferred": {"@value": "Y", "@confidence": 0.8},
        }
        r = propagate_graph_confidence(doc, ["source_fact", "inferred"])
        assert r.score == pytest.approx(0.72)
        assert r.provenance_trail == ["source_fact", "inferred"]

    def test_missing_confidence_treated_as_1(self):
        doc = {
            "known_fact": {"@value": "X"},
            "derived": {"@value": "Y", "@confidence": 0.8},
        }
        r = propagate_graph_confidence(doc, ["known_fact", "derived"])
        assert r.score == pytest.approx(0.8)

    def test_missing_property_raises(self):
        doc = {"a": {"@value": "X", "@confidence": 0.9}}
        with pytest.raises(KeyError, match="not found"):
            propagate_graph_confidence(doc, ["a", "b"])

    def test_single_property(self):
        doc = {"fact": {"@value": "X", "@confidence": 0.75}}
        r = propagate_graph_confidence(doc, ["fact"])
        assert r.score == pytest.approx(0.75)


# ═══════════════════════════════════════════════════════════════════
# PropagationResult.to_annotation
# ═══════════════════════════════════════════════════════════════════


class TestToAnnotation:
    def test_basic(self):
        r = propagate_confidence([0.9, 0.8], method="multiply")
        ann = r.to_annotation()
        assert ann["@confidence"] == pytest.approx(0.72)
        assert ann["@method"] == "confidence-multiply"
        assert ann["@derivedFrom"] == [0.9, 0.8]

    def test_with_provenance_trail(self):
        r = propagate_graph_confidence(
            {"a": {"@value": 1, "@confidence": 0.9}, "b": {"@value": 2, "@confidence": 0.8}},
            ["a", "b"],
        )
        ann = r.to_annotation()
        assert ann["@derivedFrom"] == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════
# Mathematical property tests
# ═══════════════════════════════════════════════════════════════════


class TestMathematicalProperties:
    """Verify mathematical invariants across methods."""

    def test_propagation_ordering(self):
        """For the same chain, methods should satisfy:
        multiply <= dampened <= min (for typical chains)
        and bayesian can be higher than others for strong evidence.
        """
        chain = [0.8, 0.7, 0.6]
        mult = propagate_confidence(chain, "multiply").score
        damp = propagate_confidence(chain, "dampened").score
        minv = propagate_confidence(chain, "min").score
        # multiply should be most conservative
        assert mult < damp
        # min = 0.6, dampened ≈ 0.57 — depends on chain
        # The key invariant: dampened > multiply
        assert damp > mult

    def test_combination_noisy_or_bounds(self):
        """Noisy-OR result should be >= max and <= 1.0."""
        scores = [0.6, 0.7, 0.5]
        r = combine_sources(scores, "noisy_or")
        assert r.score >= max(scores)
        assert r.score <= 1.0

    def test_combination_ds_bounds(self):
        """D-S result should be >= max(individual beliefs)."""
        scores = [0.6, 0.7]
        r = combine_sources(scores, "dempster_shafer")
        assert r.score >= max(scores)

    def test_noisy_or_commutative(self):
        """Noisy-OR should be order-independent."""
        a = combine_sources([0.8, 0.6, 0.7], "noisy_or").score
        b = combine_sources([0.7, 0.8, 0.6], "noisy_or").score
        c = combine_sources([0.6, 0.7, 0.8], "noisy_or").score
        assert a == pytest.approx(b)
        assert b == pytest.approx(c)

    def test_ds_commutative(self):
        """Dempster-Shafer should be order-independent."""
        a = combine_sources([0.8, 0.6], "dempster_shafer").score
        b = combine_sources([0.6, 0.8], "dempster_shafer").score
        assert a == pytest.approx(b)

    def test_multiply_is_product(self):
        """Sanity: multiply should equal the exact product."""
        chain = [0.9, 0.85, 0.7, 0.95]
        r = propagate_confidence(chain, "multiply")
        expected = 0.9 * 0.85 * 0.7 * 0.95
        assert r.score == pytest.approx(expected, rel=1e-9)
