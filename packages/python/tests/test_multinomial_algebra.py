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

    def test_k2_matches_binomial_from_evidence(self) -> None:
        """from_evidence with k=2 must match binomial Opinion.from_evidence.

        For k=2, the dynamic prior weight W = (2 + 4Σr) / (1 + 2Σr) = 2
        always, so multinomial from_evidence reduces exactly to the
        binomial case.  This is a critical consistency requirement.
        """
        from jsonld_ex.confidence_algebra import Opinion
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        pos, neg = 30, 70
        bin_op = Opinion.from_evidence(positive=pos, negative=neg)
        multi_op = MultinomialOpinion.from_evidence(
            evidence={"T": pos, "F": neg},
            base_rates={"T": 0.5, "F": 0.5},
        )

        assert abs(multi_op.beliefs["T"] - bin_op.belief) < 1e-9
        assert abs(multi_op.beliefs["F"] - bin_op.disbelief) < 1e-9
        assert abs(multi_op.uncertainty - bin_op.uncertainty) < 1e-9

    def test_dynamic_w_equals_two_for_k2(self) -> None:
        """For k=2, dynamic W = (2+4Σr)/(1+2Σr) = 2 for all Σr.

        This is a mathematical identity: 2(1+2Σr)/(1+2Σr) = 2.
        Verifying it ensures multinomial and binomial are consistent.
        """
        from jsonld_ex.multinomial_algebra import MultinomialOpinion
        from jsonld_ex.confidence_algebra import Opinion

        for total_evidence in [0, 1, 10, 100, 10000]:
            r = total_evidence // 2
            s = total_evidence - r

            bin_op = Opinion.from_evidence(positive=r, negative=s)
            multi_op = MultinomialOpinion.from_evidence(
                evidence={"T": r, "F": s},
                base_rates={"T": 0.5, "F": 0.5},
            )
            assert abs(multi_op.uncertainty - bin_op.uncertainty) < 1e-9, (
                f"Mismatch at total_evidence={total_evidence}: "
                f"multi_u={multi_op.uncertainty}, bin_u={bin_op.uncertainty}"
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


# ═══════════════════════════════════════════════════════════════════
# Step 2: Coarsening and Promotion
# ═══════════════════════════════════════════════════════════════════


class TestCoarsenToBinomial:
    """Test coarsening: MultinomialOpinion → binomial Opinion.

    Coarsening selects one state as the "focus" (positive) state.
    All other states collapse into the complementary (negative) class.

    Given MultinomialOpinion with focus state x_f:
        belief    = b(x_f)
        disbelief = Σ b(x_i) for i ≠ f   (= 1 - u - b(x_f))
        uncertainty = u  (unchanged)
        base_rate = a(x_f)

    References:
        Jøsang (2016), §3.5.4 (Coarsening Example: From Ternary to Binary)
    """

    def test_coarsen_basic(self) -> None:
        """Coarsening a ternary opinion to binary."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen
        from jsonld_ex.confidence_algebra import Opinion

        mop = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        op = coarsen(mop, focus_state="x1")

        assert isinstance(op, Opinion)
        assert abs(op.belief - 0.3) < 1e-9
        assert abs(op.disbelief - 0.3) < 1e-9  # 0.2 + 0.1
        assert abs(op.uncertainty - 0.4) < 1e-9
        assert abs(op.base_rate - 0.5) < 1e-9

    def test_coarsen_preserves_bdu_sum(self) -> None:
        """Coarsened opinion must satisfy b+d+u=1."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.0},
            uncertainty=0.4,
            base_rates={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        )
        op = coarsen(mop, focus_state="c")
        total = op.belief + op.disbelief + op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_coarsen_preserves_projected_probability(self) -> None:
        """P(x_f) in multinomial should equal P(ω) in coarsened binomial.

        This is the key correctness requirement: coarsening must not
        change the projected probability of the focus state.
        """
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        pp_multi = mop.projected_probability()["x1"]
        op = coarsen(mop, focus_state="x1")
        pp_bin = op.projected_probability()

        assert abs(pp_multi - pp_bin) < 1e-9

    def test_coarsen_each_state(self) -> None:
        """Coarsening to each state should produce valid opinions."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        for state in ["x1", "x2", "x3"]:
            op = coarsen(mop, focus_state=state)
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9
            assert op.belief == mop.beliefs[state]
            assert abs(op.base_rate - mop.base_rates[state]) < 1e-9

    def test_coarsen_vacuous(self) -> None:
        """Coarsening a vacuous multinomial gives a vacuous binomial."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"x1": 0.0, "x2": 0.0, "x3": 0.0},
            uncertainty=1.0,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        op = coarsen(mop, focus_state="x1")
        assert abs(op.uncertainty - 1.0) < 1e-9
        assert abs(op.belief) < 1e-9
        assert abs(op.disbelief) < 1e-9

    def test_coarsen_dogmatic(self) -> None:
        """Coarsening a dogmatic multinomial gives a dogmatic binomial."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"x1": 0.5, "x2": 0.3, "x3": 0.2},
            uncertainty=0.0,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        op = coarsen(mop, focus_state="x2")
        assert abs(op.uncertainty) < 1e-9
        assert abs(op.belief - 0.3) < 1e-9
        assert abs(op.disbelief - 0.7) < 1e-9

    def test_coarsen_invalid_state_raises(self) -> None:
        """Coarsening on a state not in the domain should raise."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen

        mop = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        with pytest.raises(ValueError, match="not in domain"):
            coarsen(mop, focus_state="x99")


class TestPromoteFromBinomial:
    """Test promotion: binomial Opinion → MultinomialOpinion.

    Converts an Opinion(b, d, u, a) to a MultinomialOpinion with k=2
    states named by the caller (default: "T", "F").

    Mapping:
        beliefs = {true_state: b, false_state: d}
        uncertainty = u
        base_rates = {true_state: a, false_state: 1-a}
    """

    def test_promote_basic(self) -> None:
        """Promote a binomial opinion to MultinomialOpinion."""
        from jsonld_ex.multinomial_algebra import promote
        from jsonld_ex.confidence_algebra import Opinion

        op = Opinion(0.6, 0.2, 0.2, 0.5)
        mop = promote(op)

        assert mop.cardinality == 2
        assert abs(mop.beliefs["T"] - 0.6) < 1e-9
        assert abs(mop.beliefs["F"] - 0.2) < 1e-9
        assert abs(mop.uncertainty - 0.2) < 1e-9
        assert abs(mop.base_rates["T"] - 0.5) < 1e-9
        assert abs(mop.base_rates["F"] - 0.5) < 1e-9

    def test_promote_custom_state_names(self) -> None:
        """Promote with custom state names."""
        from jsonld_ex.multinomial_algebra import promote
        from jsonld_ex.confidence_algebra import Opinion

        op = Opinion(0.7, 0.1, 0.2, 0.6)
        mop = promote(op, true_state="positive", false_state="negative")

        assert mop.domain == ("negative", "positive")
        assert abs(mop.beliefs["positive"] - 0.7) < 1e-9
        assert abs(mop.beliefs["negative"] - 0.1) < 1e-9
        assert abs(mop.base_rates["positive"] - 0.6) < 1e-9
        assert abs(mop.base_rates["negative"] - 0.4) < 1e-9

    def test_promote_preserves_projected_probability(self) -> None:
        """Projected probability must be preserved through promotion."""
        from jsonld_ex.multinomial_algebra import promote
        from jsonld_ex.confidence_algebra import Opinion

        op = Opinion(0.6, 0.2, 0.2, 0.7)
        pp_bin = op.projected_probability()

        mop = promote(op)
        pp_multi = mop.projected_probability()

        assert abs(pp_multi["T"] - pp_bin) < 1e-9
        assert abs(pp_multi["F"] - (1.0 - pp_bin)) < 1e-9

    def test_round_trip_promote_coarsen(self) -> None:
        """Promote then coarsen should return the original opinion."""
        from jsonld_ex.multinomial_algebra import promote, coarsen
        from jsonld_ex.confidence_algebra import Opinion

        op_orig = Opinion(0.5, 0.3, 0.2, 0.6)
        mop = promote(op_orig)
        op_rt = coarsen(mop, focus_state="T")

        assert abs(op_rt.belief - op_orig.belief) < 1e-9
        assert abs(op_rt.disbelief - op_orig.disbelief) < 1e-9
        assert abs(op_rt.uncertainty - op_orig.uncertainty) < 1e-9
        assert abs(op_rt.base_rate - op_orig.base_rate) < 1e-9

    def test_round_trip_coarsen_promote(self) -> None:
        """Coarsen then promote should give a k=2 MultinomialOpinion
        that matches the coarsened opinion's values."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion, promote, coarsen,
        )

        mop_orig = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        op_bin = coarsen(mop_orig, focus_state="x1")
        mop_rt = promote(op_bin, true_state="x1", false_state="not_x1")

        assert abs(mop_rt.beliefs["x1"] - op_bin.belief) < 1e-9
        assert abs(mop_rt.beliefs["not_x1"] - op_bin.disbelief) < 1e-9
        assert abs(mop_rt.uncertainty - op_bin.uncertainty) < 1e-9

    def test_promote_same_state_names_raises(self) -> None:
        """true_state and false_state must be different."""
        from jsonld_ex.multinomial_algebra import promote
        from jsonld_ex.confidence_algebra import Opinion

        op = Opinion(0.5, 0.3, 0.2, 0.5)
        with pytest.raises(ValueError, match="must be different"):
            promote(op, true_state="X", false_state="X")


# ═══════════════════════════════════════════════════════════════════
# Step 3: Multinomial cumulative fusion
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialCumulativeFuse:
    """Test multinomial cumulative belief fusion.

    Cumulative fusion combines independent evidence additively,
    reducing uncertainty.  The formula generalizes the binomial
    case to k-ary domains.

    For two opinions with at least one non-dogmatic (u > 0):
        κ = u_A + u_B − u_A · u_B
        b_fused(x) = (b_A(x) · u_B + b_B(x) · u_A) / κ
        u_fused = (u_A · u_B) / κ

    References:
        Jøsang (2016), §12.3 (Cumulative Fusion).
    """

    def test_two_ternary_opinions(self) -> None:
        """Fuse two ternary opinions."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op_a = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        op_b = MultinomialOpinion(
            beliefs={"x1": 0.1, "x2": 0.4, "x3": 0.2},
            uncertainty=0.3,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        result = multinomial_cumulative_fuse(op_a, op_b)

        # b+u=1 constraint must hold
        total = sum(result.beliefs.values()) + result.uncertainty
        assert abs(total - 1.0) < 1e-9

        # Uncertainty must decrease
        assert result.uncertainty < op_a.uncertainty
        assert result.uncertainty < op_b.uncertainty

    def test_uncertainty_is_product_over_kappa(self) -> None:
        """u_fused = (u_A · u_B) / κ where κ = u_A + u_B - u_A*u_B."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op_a = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        op_b = MultinomialOpinion(
            beliefs={"x1": 0.2, "x2": 0.5},
            uncertainty=0.3,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        result = multinomial_cumulative_fuse(op_a, op_b)

        kappa = 0.4 + 0.3 - 0.4 * 0.3
        expected_u = (0.4 * 0.3) / kappa
        assert abs(result.uncertainty - expected_u) < 1e-9

    def test_commutativity(self) -> None:
        """A ⊕ B = B ⊕ A."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op_a = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        op_b = MultinomialOpinion(
            beliefs={"x1": 0.1, "x2": 0.4, "x3": 0.2},
            uncertainty=0.3,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        ab = multinomial_cumulative_fuse(op_a, op_b)
        ba = multinomial_cumulative_fuse(op_b, op_a)

        for x in ["x1", "x2", "x3"]:
            assert abs(ab.beliefs[x] - ba.beliefs[x]) < 1e-9
        assert abs(ab.uncertainty - ba.uncertainty) < 1e-9

    def test_identity_with_vacuous(self) -> None:
        """A ⊕ vacuous = A (vacuous is the identity element)."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        vacuous = MultinomialOpinion(
            beliefs={"x1": 0.0, "x2": 0.0, "x3": 0.0},
            uncertainty=1.0,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        result = multinomial_cumulative_fuse(op, vacuous)

        for x in ["x1", "x2", "x3"]:
            assert abs(result.beliefs[x] - op.beliefs[x]) < 1e-9
        assert abs(result.uncertainty - op.uncertainty) < 1e-9

    def test_dogmatic_both(self) -> None:
        """When both are dogmatic (u=0), result is average of beliefs."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op_a = MultinomialOpinion(
            beliefs={"x1": 0.6, "x2": 0.3, "x3": 0.1},
            uncertainty=0.0,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        op_b = MultinomialOpinion(
            beliefs={"x1": 0.2, "x2": 0.5, "x3": 0.3},
            uncertainty=0.0,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        result = multinomial_cumulative_fuse(op_a, op_b)

        assert abs(result.uncertainty) < 1e-9
        assert abs(result.beliefs["x1"] - 0.4) < 1e-9  # (0.6+0.2)/2
        assert abs(result.beliefs["x2"] - 0.4) < 1e-9  # (0.3+0.5)/2
        assert abs(result.beliefs["x3"] - 0.2) < 1e-9  # (0.1+0.3)/2

    def test_three_opinions(self) -> None:
        """Fusing three opinions should work via left-fold."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        ops = [
            MultinomialOpinion(
                beliefs={"a": 0.2, "b": 0.3},
                uncertainty=0.5,
                base_rates={"a": 0.5, "b": 0.5},
            ),
            MultinomialOpinion(
                beliefs={"a": 0.4, "b": 0.1},
                uncertainty=0.5,
                base_rates={"a": 0.5, "b": 0.5},
            ),
            MultinomialOpinion(
                beliefs={"a": 0.1, "b": 0.4},
                uncertainty=0.5,
                base_rates={"a": 0.5, "b": 0.5},
            ),
        ]
        result = multinomial_cumulative_fuse(*ops)

        total = sum(result.beliefs.values()) + result.uncertainty
        assert abs(total - 1.0) < 1e-9
        # Uncertainty should be much less than any single input
        assert result.uncertainty < 0.5

    def test_single_opinion_returned_unchanged(self) -> None:
        """Fusing a single opinion returns it unchanged."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        result = multinomial_cumulative_fuse(op)
        assert result == op

    def test_no_opinions_raises(self) -> None:
        """Fusing zero opinions should raise."""
        from jsonld_ex.multinomial_algebra import multinomial_cumulative_fuse

        with pytest.raises(ValueError):
            multinomial_cumulative_fuse()

    def test_domain_mismatch_raises(self) -> None:
        """Fusing opinions with different domains should raise."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_cumulative_fuse,
        )

        op_a = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        op_b = MultinomialOpinion(
            beliefs={"a": 0.3, "b": 0.3},
            uncertainty=0.4,
            base_rates={"a": 0.5, "b": 0.5},
        )
        with pytest.raises(ValueError, match="domain"):
            multinomial_cumulative_fuse(op_a, op_b)

    def test_consistency_with_binomial_fuse(self) -> None:
        """Multinomial fusion of k=2 should match binomial cumulative_fuse.

        This is the critical consistency test: when applied to promoted
        binomial opinions, multinomial fusion must give the same result
        as the existing binomial cumulative_fuse().
        """
        from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
        from jsonld_ex.multinomial_algebra import (
            promote,
            coarsen,
            multinomial_cumulative_fuse,
        )

        op_a = Opinion(0.5, 0.2, 0.3, 0.6)
        op_b = Opinion(0.3, 0.4, 0.3, 0.6)

        # Binomial fusion
        bin_result = cumulative_fuse(op_a, op_b)

        # Multinomial fusion via promote → fuse → coarsen
        mop_a = promote(op_a)
        mop_b = promote(op_b)
        multi_result = multinomial_cumulative_fuse(mop_a, mop_b)
        coarsened = coarsen(multi_result, focus_state="T")

        assert abs(coarsened.belief - bin_result.belief) < 1e-9
        assert abs(coarsened.disbelief - bin_result.disbelief) < 1e-9
        assert abs(coarsened.uncertainty - bin_result.uncertainty) < 1e-9
        assert abs(coarsened.base_rate - bin_result.base_rate) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Step 4: Multinomial deduction
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialDeduce:
    """Test multinomial deduction operator.

    Generalizes binomial deduce() to k-ary parent and child domains.

    Given parent opinion ω_X over domain X = {x_1, ..., x_k_X}
    and conditional opinions ω_{Y|x_i} for each x_i, each over
    domain Y = {y_1, ..., y_k_Y}:

    For each y_j:
        b_Y(y_j) = Σ_i b_X(x_i) · b_{Y|x_i}(y_j)
                 + u_X · Σ_i a_X(x_i) · b_{Y|x_i}(y_j)

        u_Y = Σ_i b_X(x_i) · u_{Y|x_i}
            + u_X · Σ_i a_X(x_i) · u_{Y|x_i}

        a_Y(y_j) = Σ_i a_X(x_i) · P_{Y|x_i}(y_j)

    Reduces to binomial deduce() when k_X = k_Y = 2.

    References:
        Jøsang (2016), Ch. 9 (Multinomial Deduction), §12.6 (Binomial).
    """

    def test_basic_ternary_parent_binary_child(self) -> None:
        """Ternary parent, binary child: produces a valid result."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.4, "x2": 0.2, "x3": 0.1},
            uncertainty=0.3,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.7, "y2": 0.1},
                uncertainty=0.2,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.5},
                uncertainty=0.3,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x3": MultinomialOpinion(
                beliefs={"y1": 0.1, "y2": 0.3},
                uncertainty=0.6,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        # Must have the child domain
        assert result.domain == ("y1", "y2")

        # b+u=1 constraint
        total = sum(result.beliefs.values()) + result.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_additivity_always_holds(self) -> None:
        """Prove: Σ b_Y(y_j) + u_Y = 1 for arbitrary inputs."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"a": 0.15, "b": 0.25, "c": 0.35},
            uncertainty=0.25,
            base_rates={"a": 0.2, "b": 0.3, "c": 0.5},
        )
        conditionals = {
            "a": MultinomialOpinion(
                beliefs={"y1": 0.5, "y2": 0.3, "y3": 0.0},
                uncertainty=0.2,
                base_rates={"y1": 0.4, "y2": 0.4, "y3": 0.2},
            ),
            "b": MultinomialOpinion(
                beliefs={"y1": 0.1, "y2": 0.1, "y3": 0.4},
                uncertainty=0.4,
                base_rates={"y1": 0.4, "y2": 0.4, "y3": 0.2},
            ),
            "c": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.6, "y3": 0.1},
                uncertainty=0.1,
                base_rates={"y1": 0.4, "y2": 0.4, "y3": 0.2},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        total = sum(result.beliefs.values()) + result.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_base_rates_sum_to_one(self) -> None:
        """Deduced base rates must sum to 1."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.4, "x2": 0.2},
            uncertainty=0.4,
            base_rates={"x1": 0.6, "x2": 0.4},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.7, "y2": 0.1},
                uncertainty=0.2,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.5},
                uncertainty=0.3,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        assert abs(sum(result.base_rates.values()) - 1.0) < 1e-9

    def test_dogmatic_parent_dogmatic_conditionals(self) -> None:
        """When all opinions are dogmatic, reduces to law of total probability.

        P_Y(y_j) = Σ_i P_X(x_i) · P_{Y|x_i}(y_j)
        with u_Y = 0.
        """
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.6, "x2": 0.4},
            uncertainty=0.0,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.9, "y2": 0.1},
                uncertainty=0.0,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.8},
                uncertainty=0.0,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        # u_Y should be 0
        assert abs(result.uncertainty) < 1e-9

        # b_Y(y1) = 0.6*0.9 + 0.4*0.2 = 0.62
        assert abs(result.beliefs["y1"] - 0.62) < 1e-9
        # b_Y(y2) = 0.6*0.1 + 0.4*0.8 = 0.38
        assert abs(result.beliefs["y2"] - 0.38) < 1e-9

    def test_vacuous_parent_gives_base_rate_weighted_result(self) -> None:
        """With vacuous parent (u=1), result is base-rate-weighted conditionals."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.0, "x2": 0.0},
            uncertainty=1.0,
            base_rates={"x1": 0.7, "x2": 0.3},
        )
        cond_x1 = MultinomialOpinion(
            beliefs={"y1": 0.8, "y2": 0.0},
            uncertainty=0.2,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        cond_x2 = MultinomialOpinion(
            beliefs={"y1": 0.1, "y2": 0.6},
            uncertainty=0.3,
            base_rates={"y1": 0.5, "y2": 0.5},
        )
        result = multinomial_deduce(parent, {"x1": cond_x1, "x2": cond_x2})

        # b_Y(y1) = 0 + 1.0 * (0.7*0.8 + 0.3*0.1) = 0.59
        assert abs(result.beliefs["y1"] - 0.59) < 1e-9
        # u_Y = 0 + 1.0 * (0.7*0.2 + 0.3*0.3) = 0.23
        assert abs(result.uncertainty - 0.23) < 1e-9

    def test_uncertainty_propagation(self) -> None:
        """Uncertainty from both parent and conditionals propagates to child."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        # Moderate uncertainty parent
        parent = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        # Moderate uncertainty conditionals
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.4, "y2": 0.2},
                uncertainty=0.4,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.4},
                uncertainty=0.4,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        # u_Y = 0.3*0.4 + 0.3*0.4 + 0.4*(0.5*0.4 + 0.5*0.4) = 0.24 + 0.16 = 0.40
        assert abs(result.uncertainty - 0.40) < 1e-9

    def test_consistency_with_binomial_deduce(self) -> None:
        """Multinomial deduce on promoted binomial opinions must match
        the existing binomial deduce().

        This is the critical backward-compatibility test.
        """
        from jsonld_ex.confidence_algebra import Opinion, deduce
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
            promote,
            coarsen,
        )

        op_x = Opinion(0.5, 0.3, 0.2, 0.6)
        op_y_given_x = Opinion(0.8, 0.1, 0.1, 0.5)
        op_y_given_not_x = Opinion(0.2, 0.6, 0.2, 0.5)

        # Binomial deduce
        bin_result = deduce(op_x, op_y_given_x, op_y_given_not_x)

        # Multinomial deduce via promote
        m_parent = promote(op_x, true_state="x", false_state="not_x")
        m_conditionals = {
            "x": promote(op_y_given_x),
            "not_x": promote(op_y_given_not_x),
        }
        m_result = multinomial_deduce(m_parent, m_conditionals)
        coarsened = coarsen(m_result, focus_state="T")

        assert abs(coarsened.belief - bin_result.belief) < 1e-9
        assert abs(coarsened.disbelief - bin_result.disbelief) < 1e-9
        assert abs(coarsened.uncertainty - bin_result.uncertainty) < 1e-9
        assert abs(coarsened.base_rate - bin_result.base_rate) < 1e-9

    def test_missing_conditional_raises(self) -> None:
        """Conditionals must cover all parent states."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.4, "x2": 0.2, "x3": 0.1},
            uncertainty=0.3,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        # Missing x3 conditional
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.5, "y2": 0.3}, uncertainty=0.2,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.3, "y2": 0.4}, uncertainty=0.3,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        with pytest.raises(ValueError, match="conditional"):
            multinomial_deduce(parent, conditionals)

    def test_child_domain_mismatch_raises(self) -> None:
        """All conditionals must share the same child domain."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.5, "x2": 0.2},
            uncertainty=0.3,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.5, "y2": 0.3}, uncertainty=0.2,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"z1": 0.3, "z2": 0.4}, uncertainty=0.3,
                base_rates={"z1": 0.5, "z2": 0.5},
            ),
        }
        with pytest.raises(ValueError, match="domain"):
            multinomial_deduce(parent, conditionals)

    def test_component_wise_ltp_holds(self) -> None:
        """Component-wise LTP holds exactly for beliefs and uncertainty.

        For each child state y_j:
            b_Y(y_j) = Σ_i P_X(x_i) · b_{Y|x_i}(y_j)
            u_Y      = Σ_i P_X(x_i) · u_{Y|x_i}

        where P_X(x_i) = b_X(x_i) + a_X(x_i) · u_X.

        This is exact, not an approximation.
        """
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.7, "y2": 0.1},
                uncertainty=0.2,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.5},
                uncertainty=0.3,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
            "x3": MultinomialOpinion(
                beliefs={"y1": 0.1, "y2": 0.3},
                uncertainty=0.6,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        pp_parent = parent.projected_probability()

        # Belief component-wise LTP
        for y in ["y1", "y2"]:
            expected_b = sum(
                pp_parent[x] * conditionals[x].beliefs[y]
                for x in ["x1", "x2", "x3"]
            )
            assert abs(result.beliefs[y] - expected_b) < 1e-9, (
                f"b_Y({y}): expected={expected_b}, got={result.beliefs[y]}"
            )

        # Uncertainty component-wise LTP
        expected_u = sum(
            pp_parent[x] * conditionals[x].uncertainty
            for x in ["x1", "x2", "x3"]
        )
        assert abs(result.uncertainty - expected_u) < 1e-9

    def test_projected_probability_ltp_does_not_hold_in_general(self) -> None:
        """Document: projected probability LTP does NOT hold in general.

        P_Y(y_j) ≠ Σ_i P_X(x_i) · P_{Y|x_i}(y_j)  in general.

        The discrepancy arises because P_Y(y) = b_Y(y) + a_Y(y)·u_Y
        mixes two different weighting schemes: a_Y uses parent base
        rates a_X(x_i), while u_Y uses projected probabilities P_X(x_i).
        The product a_Y·u_Y is therefore not a simple weighted sum.

        This is consistent with binomial deduce() which documents
        the same limitation (see confidence_algebra.py).

        The LTP DOES hold in the dogmatic limit (u=0 for all opinions)
        because the base rate weighting becomes irrelevant.
        """
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.7, "y2": 0.1},
                uncertainty=0.2,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.2, "y2": 0.5},
                uncertainty=0.3,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
            "x3": MultinomialOpinion(
                beliefs={"y1": 0.1, "y2": 0.3},
                uncertainty=0.6,
                base_rates={"y1": 0.6, "y2": 0.4},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        pp_parent = parent.projected_probability()
        pp_result = result.projected_probability()

        # Compute what LTP would predict
        for y in ["y1", "y2"]:
            ltp = sum(
                pp_parent[x] * conditionals[x].projected_probability()[y]
                for x in ["x1", "x2", "x3"]
            )
            actual = pp_result[y]
            # They should NOT be equal (with non-trivial uncertainty)
            # but should be close — the gap is the documented discrepancy
            assert abs(actual - ltp) > 1e-6, (
                f"Expected discrepancy for P_Y({y}) but got exact match: "
                f"LTP={ltp}, actual={actual}"
            )

    def test_projected_probability_ltp_holds_in_dogmatic_limit(self) -> None:
        """In the dogmatic limit (all u=0), projected probability LTP holds exactly."""
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion,
            multinomial_deduce,
        )

        parent = MultinomialOpinion(
            beliefs={"x1": 0.6, "x2": 0.3, "x3": 0.1},
            uncertainty=0.0,
            base_rates={"x1": 1/3, "x2": 1/3, "x3": 1/3},
        )
        conditionals = {
            "x1": MultinomialOpinion(
                beliefs={"y1": 0.8, "y2": 0.2}, uncertainty=0.0,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x2": MultinomialOpinion(
                beliefs={"y1": 0.3, "y2": 0.7}, uncertainty=0.0,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
            "x3": MultinomialOpinion(
                beliefs={"y1": 0.1, "y2": 0.9}, uncertainty=0.0,
                base_rates={"y1": 0.5, "y2": 0.5},
            ),
        }
        result = multinomial_deduce(parent, conditionals)

        pp_parent = parent.projected_probability()
        pp_result = result.projected_probability()

        for y in ["y1", "y2"]:
            ltp = sum(
                pp_parent[x] * conditionals[x].projected_probability()[y]
                for x in ["x1", "x2", "x3"]
            )
            assert abs(pp_result[y] - ltp) < 1e-9, (
                f"Dogmatic limit: P_Y({y}) should match LTP"
            )


# ═══════════════════════════════════════════════════════════════════
# Step 5: MultinomialOpinion Serialization (Gap 2)
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialOpinionToDict:
    """Test to_dict() serialization."""

    def test_to_dict_basic_ternary(self) -> None:
        """to_dict() returns beliefs, uncertainty, base_rates as plain dict."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        d = op.to_dict()

        assert isinstance(d, dict)
        assert d["beliefs"] == {"x1": 0.3, "x2": 0.2, "x3": 0.1}
        assert d["uncertainty"] == 0.4
        assert d["base_rates"] == {"x1": 0.5, "x2": 0.3, "x3": 0.2}

    def test_to_dict_binary(self) -> None:
        """to_dict() works for k=2 (binary) opinions."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"T": 0.6, "F": 0.2},
            uncertainty=0.2,
            base_rates={"T": 0.5, "F": 0.5},
        )
        d = op.to_dict()

        assert d["beliefs"]["T"] == 0.6
        assert d["beliefs"]["F"] == 0.2
        assert d["uncertainty"] == 0.2

    def test_to_dict_returns_plain_dicts(self) -> None:
        """to_dict() must return plain dicts, not MappingProxyType."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        d = op.to_dict()

        assert type(d["beliefs"]) is dict
        assert type(d["base_rates"]) is dict

    def test_to_dict_vacuous(self) -> None:
        """Vacuous opinion serializes correctly."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"a": 0.0, "b": 0.0, "c": 0.0},
            uncertainty=1.0,
            base_rates={"a": 1 / 3, "b": 1 / 3, "c": 1 / 3},
        )
        d = op.to_dict()

        assert d["uncertainty"] == 1.0
        for v in d["beliefs"].values():
            assert v == 0.0


class TestMultinomialOpinionFromDict:
    """Test from_dict() deserialization."""

    def test_from_dict_basic(self) -> None:
        """from_dict() reconstructs a MultinomialOpinion."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        data = {
            "beliefs": {"x1": 0.3, "x2": 0.2, "x3": 0.1},
            "uncertainty": 0.4,
            "base_rates": {"x1": 0.5, "x2": 0.3, "x3": 0.2},
        }
        op = MultinomialOpinion.from_dict(data)

        assert op.beliefs["x1"] == 0.3
        assert op.uncertainty == 0.4
        assert op.base_rates["x3"] == 0.2

    def test_round_trip_to_dict_from_dict(self) -> None:
        """to_dict() → from_dict() round-trip preserves values."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        reconstructed = MultinomialOpinion.from_dict(original.to_dict())
        assert reconstructed == original

    def test_round_trip_10_state(self) -> None:
        """Round-trip works for a 10-state domain."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        k = 10
        beliefs = {f"s{i}": 0.05 for i in range(k)}  # 10 * 0.05 = 0.5
        base_rates = {f"s{i}": 1.0 / k for i in range(k)}
        original = MultinomialOpinion(
            beliefs=beliefs,
            uncertainty=0.5,
            base_rates=base_rates,
        )
        reconstructed = MultinomialOpinion.from_dict(original.to_dict())
        assert reconstructed == original

    def test_round_trip_near_zero_values(self) -> None:
        """Round-trip preserves near-zero belief values."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion(
            beliefs={"a": 1e-15, "b": 1.0 - 1e-15 - 0.3, "c": 0.3},
            uncertainty=0.0,
            base_rates={"a": 0.1, "b": 0.5, "c": 0.4},
        )
        reconstructed = MultinomialOpinion.from_dict(original.to_dict())
        assert reconstructed == original

    def test_round_trip_from_evidence_factory(self) -> None:
        """Round-trip works for opinions created via from_evidence()."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion.from_evidence(
            evidence={"x1": 30, "x2": 50, "x3": 20},
        )
        reconstructed = MultinomialOpinion.from_dict(original.to_dict())
        assert reconstructed == original


class TestMultinomialOpinionToJsonLD:
    """Test to_jsonld() serialization."""

    def test_to_jsonld_has_type(self) -> None:
        """to_jsonld() must include @type: MultinomialOpinion."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        d = op.to_jsonld()

        assert d["@type"] == "MultinomialOpinion"

    def test_to_jsonld_uses_camel_case(self) -> None:
        """to_jsonld() uses baseRates (camelCase), matching Opinion pattern."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.3},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.5},
        )
        d = op.to_jsonld()

        assert "baseRates" in d
        assert "base_rates" not in d
        assert d["baseRates"] == {"x1": 0.5, "x2": 0.5}

    def test_to_jsonld_structure(self) -> None:
        """to_jsonld() returns the expected full structure."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        op = MultinomialOpinion(
            beliefs={"a": 0.2, "b": 0.3},
            uncertainty=0.5,
            base_rates={"a": 0.4, "b": 0.6},
        )
        d = op.to_jsonld()

        assert d == {
            "@type": "MultinomialOpinion",
            "beliefs": {"a": 0.2, "b": 0.3},
            "uncertainty": 0.5,
            "baseRates": {"a": 0.4, "b": 0.6},
        }


class TestMultinomialOpinionFromJsonLD:
    """Test from_jsonld() deserialization."""

    def test_from_jsonld_basic(self) -> None:
        """from_jsonld() reconstructs from JSON-LD dict."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        data = {
            "@type": "MultinomialOpinion",
            "beliefs": {"x1": 0.3, "x2": 0.3},
            "uncertainty": 0.4,
            "baseRates": {"x1": 0.5, "x2": 0.5},
        }
        op = MultinomialOpinion.from_jsonld(data)

        assert op.beliefs["x1"] == 0.3
        assert op.uncertainty == 0.4
        assert op.base_rates["x1"] == 0.5

    def test_from_jsonld_accepts_snake_case_fallback(self) -> None:
        """from_jsonld() accepts base_rates as fallback for robustness."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        data = {
            "@type": "MultinomialOpinion",
            "beliefs": {"x1": 0.3, "x2": 0.3},
            "uncertainty": 0.4,
            "base_rates": {"x1": 0.5, "x2": 0.5},
        }
        op = MultinomialOpinion.from_jsonld(data)

        assert op.base_rates["x1"] == 0.5

    def test_from_jsonld_prefers_camel_case(self) -> None:
        """When both baseRates and base_rates present, prefer baseRates."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        data = {
            "@type": "MultinomialOpinion",
            "beliefs": {"x1": 0.3, "x2": 0.3},
            "uncertainty": 0.4,
            "baseRates": {"x1": 0.6, "x2": 0.4},
            "base_rates": {"x1": 0.5, "x2": 0.5},  # should be ignored
        }
        op = MultinomialOpinion.from_jsonld(data)

        assert abs(op.base_rates["x1"] - 0.6) < 1e-9
        assert abs(op.base_rates["x2"] - 0.4) < 1e-9

    def test_round_trip_to_jsonld_from_jsonld(self) -> None:
        """to_jsonld() → from_jsonld() round-trip preserves values."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion(
            beliefs={"x1": 0.3, "x2": 0.2, "x3": 0.1},
            uncertainty=0.4,
            base_rates={"x1": 0.5, "x2": 0.3, "x3": 0.2},
        )
        reconstructed = MultinomialOpinion.from_jsonld(original.to_jsonld())
        assert reconstructed == original

    def test_round_trip_jsonld_10_state(self) -> None:
        """JSON-LD round-trip works for a 10-state domain."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        k = 10
        beliefs = {f"s{i}": 0.05 for i in range(k)}
        base_rates = {f"s{i}": 1.0 / k for i in range(k)}
        original = MultinomialOpinion(
            beliefs=beliefs,
            uncertainty=0.5,
            base_rates=base_rates,
        )
        reconstructed = MultinomialOpinion.from_jsonld(original.to_jsonld())
        assert reconstructed == original

    def test_round_trip_jsonld_from_evidence(self) -> None:
        """JSON-LD round-trip works for opinions created via from_evidence()."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion.from_evidence(
            evidence={"x1": 30, "x2": 50, "x3": 20},
        )
        reconstructed = MultinomialOpinion.from_jsonld(original.to_jsonld())
        assert reconstructed == original

    def test_round_trip_jsonld_vacuous(self) -> None:
        """JSON-LD round-trip works for vacuous opinion."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion(
            beliefs={"a": 0.0, "b": 0.0},
            uncertainty=1.0,
            base_rates={"a": 0.5, "b": 0.5},
        )
        reconstructed = MultinomialOpinion.from_jsonld(original.to_jsonld())
        assert reconstructed == original

    def test_round_trip_jsonld_dogmatic(self) -> None:
        """JSON-LD round-trip works for dogmatic opinion (u=0)."""
        from jsonld_ex.multinomial_algebra import MultinomialOpinion

        original = MultinomialOpinion(
            beliefs={"a": 0.6, "b": 0.4},
            uncertainty=0.0,
            base_rates={"a": 0.5, "b": 0.5},
        )
        reconstructed = MultinomialOpinion.from_jsonld(original.to_jsonld())
        assert reconstructed == original
