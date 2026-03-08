"""
Tests for Multinomial Confidence Algebra.

Implements Jøsang's multinomial opinions (Ch. 3.5) — the generalization
of binomial opinions to k-ary domains via Dirichlet distributions.

Build order (TDD):
    Step 1: MultinomialOpinion dataclass + validation + projected_probability + from_evidence
    Step 2: Coarsening (multinomial → binomial) and promotion (binomial → multinomial)
    Step 3: multinomial_cumulative_fuse()
    Step 4: multinomial_deduce()

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 3.5 (Multinomial Opinions),
    §3.5.2 (Dirichlet Multinomial Model), §3.5.5 (Mapping).
    Jøsang, A. (2022). FUSION 2022 Tutorial, slides 20–30.
"""

from __future__ import annotations

import pytest


# ═══════════════════════════════════════════════════════════════════
# Step 1: MultinomialOpinion dataclass
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialOpinionConstruction:
    """Test MultinomialOpinion construction and validation."""

    def test_ternary_opinion_basic(self) -> None:
        """Construct a valid 3-state opinion."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        assert op.uncertainty == 0.4
        assert op.beliefs["x1"] == 0.3
        assert op.base_rates["x3"] == 0.2

    def test_binary_opinion_valid(self) -> None:
        """Binary (k=2) case should work — equivalent to binomial."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"T": 0.6, "F": 0.2},
            uncertainty=0.2,
            base_rates={"T": 0.5, "F": 0.5},
        )
        assert len(op.beliefs) == 2

    def test_four_state_opinion(self) -> None:
        """4-state domain should construct correctly."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.0},
            uncertainty=0.4,
            base_rates={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        )
        assert len(op.beliefs) == 4

    def test_domain_property(self) -> None:
        """domain should return the sorted set of state names."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        assert op.domain == ("x1", "x2", "x3")

    def test_cardinality_property(self) -> None:
        """cardinality should return the number of states."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        assert op.cardinality == 3


class TestMultinomialOpinionValidation:
    """Test input validation — catch errors early."""

    def test_beliefs_plus_uncertainty_must_sum_to_one(self) -> None:
        """sum(beliefs) + uncertainty must equal 1."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError, match="sum to 1"):
            MultinomialOpinion(
                beliefs={"x1": 0.5, "x2": 0.5},
                uncertainty=0.5,  # sum = 1.5
                base_rates={"x1": 0.5, "x2": 0.5},
            )

    def test_base_rates_must_sum_to_one(self) -> None:
        """sum(base_rates) must equal 1."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError, match="base_rates.*sum to 1"):
            MultinomialOpinion(
                beliefs={"x1": 0.3, "x2": 0.3},
                uncertainty=0.4,
                base_rates={"x1": 0.6, "x2": 0.6},  # sum = 1.2
            )

    def test_beliefs_and_base_rates_must_have_same_keys(self) -> None:
        """Domain mismatch between beliefs and base_rates should raise."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError, match="domain"):
            MultinomialOpinion(
                beliefs={"x1": 0.3, "x2": 0.3},
                uncertainty=0.4,
                base_rates={"x1": 0.5, "x3": 0.5},  # x3 ≠ x2
            )

    def test_negative_belief_raises(self) -> None:
        """Negative belief mass should raise ValueError."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError):
            MultinomialOpinion(
                beliefs={"x1": -0.1, "x2": 0.7},
                uncertainty=0.4,
                base_rates={"x1": 0.5, "x2": 0.5},
            )

    def test_negative_uncertainty_raises(self) -> None:
        """Negative uncertainty should raise ValueError."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError):
            MultinomialOpinion(
                beliefs={"x1": 0.5, "x2": 0.5},
                uncertainty=-0.1,
                base_rates={"x1": 0.5, "x2": 0.5},
            )

    def test_negative_base_rate_raises(self) -> None:
        """Negative base rate should raise ValueError."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError):
            MultinomialOpinion(
                beliefs={"x1": 0.3, "x2": 0.3},
                uncertainty=0.4,
                base_rates={"x1": -0.1, "x2": 1.1},
            )

    def test_single_state_raises(self) -> None:
        """Domain with fewer than 2 states should raise."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError, match="at least 2"):
            MultinomialOpinion(
                beliefs={"x1": 0.5},
                uncertainty=0.5,
                base_rates={"x1": 1.0},
            )

    def test_empty_domain_raises(self) -> None:
        """Empty domain should raise."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError):
            MultinomialOpinion(
                beliefs={},
                uncertainty=1.0,
                base_rates={},
            )

    def test_float_tolerance_accepted(self) -> None:
        """Small floating-point deviations from sum=1 should be tolerated."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        # These sum to 1.0 within floating-point tolerance
        op = MultinomialOpinion(
            beliefs={"x1": 0.1, "x2": 0.2, "x3": 0.3},
            uncertainty=0.4 + 1e-15,  # tiny overshoot
            base_rates={"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3},
        )
        assert op is not None


class TestMultinomialOpinionProjectedProbability:
    """Test projected probability: P(x) = b(x) + a(x) * u."""

    def test_projected_probability_basic(self) -> None:
        """P(x) = b(x) + a(x) * u for each state."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        pp = op.projected_probability()

        assert abs(pp["x1"] - (0.3 + 0.5 * 0.4)) < 1e-9  # 0.5
        assert abs(pp["x2"] - (0.2 + 0.3 * 0.4)) < 1e-9  # 0.32
        assert abs(pp["x3"] - (0.1 + 0.2 * 0.4)) < 1e-9  # 0.18

    def test_projected_probability_sums_to_one(self) -> None:
        """Projected probabilities must sum to 1."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        pp = op.projected_probability()
        assert abs(sum(pp.values()) - 1.0) < 1e-9

    def test_dogmatic_opinion_pp_equals_belief(self) -> None:
        """When u=0, projected probability equals belief distribution."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.5, "x2": 0.3, "x3": 0.2},
            uncertainty=0.0,
            base_rates={"x1": 1 / 3, "x2": 1 / 3, "x3": 1 / 3},
        )
        pp = op.projected_probability()
        assert abs(pp["x1"] - 0.5) < 1e-9
        assert abs(pp["x2"] - 0.3) < 1e-9
        assert abs(pp["x3"] - 0.2) < 1e-9

    def test_vacuous_opinion_pp_equals_base_rate(self) -> None:
        """When u=1, projected probability equals base rate."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.0, "x2": 0.0, "x3": 0.0},
            uncertainty=1.0,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        pp = op.projected_probability()
        assert abs(pp["x1"] - 0.5) < 1e-9
        assert abs(pp["x2"] - 0.3) < 1e-9
        assert abs(pp["x3"] - 0.2) < 1e-9


class TestMultinomialOpinionFromEvidence:
    """Test from_evidence() — Dirichlet-based evidence mapping.

    Formulas (Jøsang 2016, §3.5.2, §3.5.5):
        b(x) = r(x) / (Σr + W)
        u    = W / (Σr + W)

    Dynamic prior weight W (Jøsang FUSION 2022, slide 25):
        W = (k + 2*k*Σr) / (1 + k*Σr)
        where k = cardinality, Σr = sum of all evidence counts.
        Initially (Σr=0): W = k.
        As Σr → ∞: W → 2.
    """

    def test_vacuous_with_zero_evidence(self) -> None:
        """Zero evidence for all states should give vacuous opinion (u=1)."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 0, "x2": 0, "x3": 0},
        )
        assert abs(op.uncertainty - 1.0) < 1e-9
        for b in op.beliefs.values():
            assert abs(b) < 1e-9

    def test_uniform_evidence_equal_beliefs(self) -> None:
        """Equal evidence across all states gives equal beliefs."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 10, "x2": 10, "x3": 10},
        )
        assert abs(op.beliefs["x1"] - op.beliefs["x2"]) < 1e-9
        assert abs(op.beliefs["x2"] - op.beliefs["x3"]) < 1e-9

    def test_uncertainty_decreases_with_evidence(self) -> None:
        """More evidence → less uncertainty."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op_small = MultinomialOpinion.from_evidence(
            evidence={"x1": 1, "x2": 1, "x3": 1},
        )
        op_large = MultinomialOpinion.from_evidence(
            evidence={"x1": 100, "x2": 100, "x3": 100},
        )
        assert op_small.uncertainty > op_large.uncertainty

    def test_evidence_produces_correct_beliefs(self) -> None:
        """Beliefs should be proportional to evidence counts."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 60, "x2": 30, "x3": 10},
        )
        # x1 has 6x the evidence of x3, so belief should be ~6x
        assert op.beliefs["x1"] > op.beliefs["x2"] > op.beliefs["x3"]

    def test_bdu_constraint_holds(self) -> None:
        """sum(beliefs) + uncertainty must equal 1."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 7, "x2": 3, "x3": 5, "x4": 2},
        )
        total = sum(op.beliefs.values()) + op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_custom_base_rates(self) -> None:
        """Custom base rates should be preserved."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        br = {"x1": 0.6, "x2": 0.3, "x3": 0.1}
        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 10, "x2": 10, "x3": 10},
            base_rates=br,
        )
        assert abs(op.base_rates["x1"] - 0.6) < 1e-9
        assert abs(op.base_rates["x2"] - 0.3) < 1e-9

    def test_default_base_rates_are_uniform(self) -> None:
        """If no base_rates given, default to uniform."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 10, "x2": 10, "x3": 10},
        )
        expected = 1.0 / 3.0
        for a in op.base_rates.values():
            assert abs(a - expected) < 1e-9

    def test_projected_probability_matches_frequency_at_large_n(self) -> None:
        """At large N, projected probability converges to observed frequency.

        This is the dogmatic limit: as evidence grows, uncertainty → 0,
        and P(x) → r(x)/Σr (the observed frequency).
        """
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        N = 100_000
        op = MultinomialOpinion.from_evidence(
            evidence={"x1": int(0.5 * N), "x2": int(0.3 * N), "x3": int(0.2 * N)},
        )
        pp = op.projected_probability()
        assert abs(pp["x1"] - 0.5) < 0.01
        assert abs(pp["x2"] - 0.3) < 0.01
        assert abs(pp["x3"] - 0.2) < 0.01

    def test_dynamic_prior_weight_starts_at_cardinality(self) -> None:
        """With zero evidence, W should equal k (cardinality).

        W = (k + 2*k*Σr) / (1 + k*Σr)
        When Σr = 0: W = k/1 = k.
        """
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        # k=3, zero evidence → W = 3 → u = W/(Σr+W) = 3/3 = 1.0
        op = MultinomialOpinion.from_evidence(
            evidence={"x1": 0, "x2": 0, "x3": 0},
        )
        assert abs(op.uncertainty - 1.0) < 1e-9

    def test_dynamic_prior_weight_converges_to_two(self) -> None:
        """With large evidence, W converges to 2 (like binomial case).

        W = (k + 2*k*Σr) / (1 + k*Σr)
        As Σr → ∞: W → 2k*Σr / (k*Σr) = 2.
        """
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        N = 1_000_000
        op = MultinomialOpinion.from_evidence(
            evidence={"x1": N, "x2": N, "x3": N},
        )
        # W ≈ 2, Σr = 3N, u = W/(Σr+W) ≈ 2/(3N+2) ≈ very small
        # For binomial with same total evidence, u = 2/(2N+2)
        # Just verify uncertainty is tiny
        assert op.uncertainty < 0.001

    def test_negative_evidence_raises(self) -> None:
        """Negative evidence count should raise ValueError."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        with pytest.raises(ValueError):
            MultinomialOpinion.from_evidence(
                evidence={"x1": -5, "x2": 10},
            )


class TestMultinomialOpinionImmutability:
    """MultinomialOpinion should be immutable (frozen)."""

    def test_frozen_beliefs(self) -> None:
        """Cannot mutate beliefs dict after construction."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        with pytest.raises((TypeError, AttributeError)):
            op.beliefs["x1"] = 0.9  # type: ignore

    def test_frozen_uncertainty(self) -> None:
        """Cannot mutate uncertainty after construction."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        with pytest.raises((TypeError, AttributeError)):
            op.uncertainty = 0.9  # type: ignore
