"""Tests for the Compliance Algebra module.

TDD RED PHASE — tests written FIRST, implementation does not yet exist.

Covers the full Compliance Algebra as formalized in compliance_algebra.md:
  §4  ComplianceOpinion (Definition 2)
  §5  Jurisdictional Meet (Definition 3, Theorem 1)
  §6  Compliance Propagation (Definition 6–7, Theorem 2)
  §7  Consent Assessment (Definitions 8–11, Theorem 3)
  §8  Temporal Decay triggers (Definitions 12–14, Theorem 4)
  §9  Erasure Propagation (Definitions 15–17, Theorem 5, Proposition 1)
  §10 Operator Interactions

Every test references the specific definition/theorem it validates.

Mathematical source of truth: compliance_algebra.md
Implementation source of truth: Jøsang (2016), Subjective Logic.
"""

import math
import pytest
from datetime import datetime, timezone

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay

from jsonld_ex.compliance_algebra import (
    # Class
    ComplianceOpinion,
    # Operator 1: Jurisdictional Meet (§5)
    jurisdictional_meet,
    # Operator 2: Compliance Propagation (§6)
    compliance_propagation,
    ProvenanceChain,
    # Operator 3: Consent Assessment (§7)
    consent_validity,
    withdrawal_override,
    # Operator 4: Temporal Triggers (§8)
    expiry_trigger,
    review_due_trigger,
    regulatory_change_trigger,
    # Operator 5: Erasure Propagation (§9)
    erasure_scope_opinion,
    residual_contamination,
)


# ── Tolerance for floating-point comparisons ───────────────────────
TOL = 1e-9


# ── Helper: assert valid compliance opinion ────────────────────────
def assert_valid_opinion(op, msg=""):
    """Assert that an opinion satisfies all SL constraints."""
    prefix = f"{msg}: " if msg else ""
    assert op.lawfulness >= -TOL, f"{prefix}lawfulness < 0: {op.lawfulness}"
    assert op.violation >= -TOL, f"{prefix}violation < 0: {op.violation}"
    assert op.uncertainty >= -TOL, f"{prefix}uncertainty < 0: {op.uncertainty}"
    assert 0.0 <= op.base_rate <= 1.0, f"{prefix}base_rate out of [0,1]: {op.base_rate}"
    total = op.lawfulness + op.violation + op.uncertainty
    assert abs(total - 1.0) < TOL, f"{prefix}l+v+u={total}, expected 1.0"


# ═══════════════════════════════════════════════════════════════════
# §4 — COMPLIANCE OPINION (Definition 2)
# ═══════════════════════════════════════════════════════════════════


class TestComplianceOpinionCreation:
    """Definition 2: ω = (l, v, u, a) with l + v + u = 1."""

    def test_create_basic(self):
        co = ComplianceOpinion.create(
            lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5
        )
        assert co.lawfulness == pytest.approx(0.7)
        assert co.violation == pytest.approx(0.1)
        assert co.uncertainty == pytest.approx(0.2)
        assert co.base_rate == pytest.approx(0.5)

    def test_create_default_base_rate(self):
        co = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2)
        assert co.base_rate == pytest.approx(0.5)

    def test_property_aliases_match_opinion_fields(self):
        """lawfulness = belief, violation = disbelief."""
        co = ComplianceOpinion.create(
            lawfulness=0.6, violation=0.15, uncertainty=0.25, base_rate=0.4
        )
        assert co.lawfulness == co.belief
        assert co.violation == co.disbelief

    def test_from_opinion_wraps_correctly(self):
        op = Opinion(belief=0.8, disbelief=0.05, uncertainty=0.15, base_rate=0.3)
        co = ComplianceOpinion.from_opinion(op)
        assert co.lawfulness == pytest.approx(0.8)
        assert co.violation == pytest.approx(0.05)
        assert co.uncertainty == pytest.approx(0.15)
        assert co.base_rate == pytest.approx(0.3)
        assert isinstance(co, ComplianceOpinion)

    def test_isinstance_opinion(self):
        """ComplianceOpinion must be accepted by existing SL operators."""
        co = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2)
        assert isinstance(co, Opinion)

    def test_projected_probability(self):
        """P(ω) = l + a·u (Definition 2)."""
        co = ComplianceOpinion.create(
            lawfulness=0.6, violation=0.1, uncertainty=0.3, base_rate=0.4
        )
        expected = 0.6 + 0.4 * 0.3  # 0.72
        assert co.projected_probability() == pytest.approx(expected)

    def test_constraint_violation_raises(self):
        """l + v + u must equal 1."""
        with pytest.raises(ValueError):
            ComplianceOpinion.create(lawfulness=0.5, violation=0.5, uncertainty=0.5)

    def test_negative_component_raises(self):
        with pytest.raises(ValueError):
            ComplianceOpinion.create(lawfulness=-0.1, violation=0.6, uncertainty=0.5)

    def test_eq_with_opinion(self):
        """ComplianceOpinion == Opinion when field values match."""
        co = ComplianceOpinion.create(
            lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5
        )
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        assert co == op
        assert op == co  # symmetry

    def test_eq_different_values(self):
        co = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        op = Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)
        assert co != op

    def test_hash_consistent_with_eq(self):
        co = ComplianceOpinion.create(
            lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5
        )
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        assert hash(co) == hash(op)

    def test_repr_uses_compliance_notation(self):
        co = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        r = repr(co)
        assert "ComplianceOpinion" in r
        assert "l=" in r
        assert "v=" in r

    def test_frozen_immutable(self):
        co = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        with pytest.raises(AttributeError):
            co.belief = 0.5  # type: ignore[misc]

    def test_vacuous_opinion(self):
        """Complete ignorance: (0, 0, 1, a)."""
        co = ComplianceOpinion.create(
            lawfulness=0.0, violation=0.0, uncertainty=1.0, base_rate=0.3
        )
        assert co.projected_probability() == pytest.approx(0.3)

    def test_dogmatic_compliant(self):
        """Certain compliance: (1, 0, 0, a)."""
        co = ComplianceOpinion.create(
            lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=0.5
        )
        assert co.projected_probability() == pytest.approx(1.0)

    def test_dogmatic_violation(self):
        """Certain violation: (0, 1, 0, a)."""
        co = ComplianceOpinion.create(
            lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.5
        )
        assert co.projected_probability() == pytest.approx(0.0)

    def test_works_with_existing_decay_operator(self):
        """ComplianceOpinion passes through existing SL operators."""
        co = ComplianceOpinion.create(
            lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5
        )
        decayed = decay_opinion(co, elapsed=10.0, half_life=10.0)
        # decay_opinion returns a plain Opinion — that's expected
        assert isinstance(decayed, Opinion)
        assert decayed.belief < co.belief  # lawfulness should decrease
        assert decayed.uncertainty > co.uncertainty


# ═══════════════════════════════════════════════════════════════════
# §5 — JURISDICTIONAL MEET (Definition 3, Theorem 1)
# ═══════════════════════════════════════════════════════════════════


class TestJurisdictionalMeetBinary:
    """Binary jurisdictional meet: J⊓(ω₁, ω₂)."""

    def test_formula_correctness(self):
        """Definition 3: l⊓=l₁·l₂, v⊓=v₁+v₂−v₁·v₂, u⊓=(1−v₁)(1−v₂)−l₁·l₂."""
        w1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        w2 = ComplianceOpinion.create(lawfulness=0.6, violation=0.1, uncertainty=0.3, base_rate=0.4)
        result = jurisdictional_meet(w1, w2)

        assert result.lawfulness == pytest.approx(0.8 * 0.6)
        assert result.violation == pytest.approx(0.05 + 0.1 - 0.05 * 0.1)
        assert result.uncertainty == pytest.approx(
            (1 - 0.05) * (1 - 0.1) - 0.8 * 0.6
        )
        assert result.base_rate == pytest.approx(0.5 * 0.4)

    def test_returns_compliance_opinion(self):
        w1 = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2)
        w2 = ComplianceOpinion.create(lawfulness=0.4, violation=0.2, uncertainty=0.4)
        result = jurisdictional_meet(w1, w2)
        assert isinstance(result, ComplianceOpinion)

    def test_accepts_plain_opinion(self):
        """Should accept Opinion, not just ComplianceOpinion."""
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        co = ComplianceOpinion.create(lawfulness=0.6, violation=0.2, uncertainty=0.2)
        result = jurisdictional_meet(op, co)
        assert isinstance(result, ComplianceOpinion)
        assert_valid_opinion(result)

    # ── Theorem 1 properties ──────────────────────────────────

    def test_theorem1a_constraint(self):
        """(a) l⊓ + v⊓ + u⊓ = 1."""
        w1 = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.6)
        w2 = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3, base_rate=0.5)
        result = jurisdictional_meet(w1, w2)
        assert_valid_opinion(result, "Theorem 1(a)")

    def test_theorem1b_non_negativity(self):
        """(b) l⊓, v⊓, u⊓ ≥ 0."""
        # Edge case: high violation in one jurisdiction
        w1 = ComplianceOpinion.create(lawfulness=0.1, violation=0.8, uncertainty=0.1, base_rate=0.3)
        w2 = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.7)
        result = jurisdictional_meet(w1, w2)
        assert result.lawfulness >= -TOL
        assert result.violation >= -TOL
        assert result.uncertainty >= -TOL

    def test_theorem1c_monotonic_restriction(self):
        """(c) l⊓ ≤ min(l₁, l₂)."""
        w1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15)
        w2 = ComplianceOpinion.create(lawfulness=0.6, violation=0.1, uncertainty=0.3)
        result = jurisdictional_meet(w1, w2)
        assert result.lawfulness <= min(w1.lawfulness, w2.lawfulness) + TOL

    def test_theorem1d_monotonic_violation(self):
        """(d) v⊓ ≥ max(v₁, v₂)."""
        w1 = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15)
        w2 = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3)
        result = jurisdictional_meet(w1, w2)
        assert result.violation >= max(w1.violation, w2.violation) - TOL

    def test_theorem1e_commutativity(self):
        """(e) J⊓(ω₁, ω₂) = J⊓(ω₂, ω₁)."""
        w1 = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.6)
        w2 = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3, base_rate=0.4)
        r12 = jurisdictional_meet(w1, w2)
        r21 = jurisdictional_meet(w2, w1)
        assert r12.lawfulness == pytest.approx(r21.lawfulness)
        assert r12.violation == pytest.approx(r21.violation)
        assert r12.uncertainty == pytest.approx(r21.uncertainty)
        assert r12.base_rate == pytest.approx(r21.base_rate)

    def test_theorem1f_associativity(self):
        """(f) J⊓(ω₁, J⊓(ω₂, ω₃)) = J⊓(J⊓(ω₁, ω₂), ω₃)."""
        w1 = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        w2 = ComplianceOpinion.create(lawfulness=0.5, violation=0.2, uncertainty=0.3, base_rate=0.6)
        w3 = ComplianceOpinion.create(lawfulness=0.6, violation=0.15, uncertainty=0.25, base_rate=0.4)
        left = jurisdictional_meet(jurisdictional_meet(w1, w2), w3)
        right = jurisdictional_meet(w1, jurisdictional_meet(w2, w3))
        assert left.lawfulness == pytest.approx(right.lawfulness, abs=TOL)
        assert left.violation == pytest.approx(right.violation, abs=TOL)
        assert left.uncertainty == pytest.approx(right.uncertainty, abs=TOL)
        assert left.base_rate == pytest.approx(right.base_rate, abs=TOL)

    def test_theorem1g_identity(self):
        """(g) ω∅ = (1, 0, 0, 1) is identity: J⊓(ω, ω∅) = ω."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.6)
        identity = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = jurisdictional_meet(w, identity)
        assert result.lawfulness == pytest.approx(w.lawfulness)
        assert result.violation == pytest.approx(w.violation)
        assert result.uncertainty == pytest.approx(w.uncertainty)
        assert result.base_rate == pytest.approx(w.base_rate)

    def test_theorem1h_annihilator(self):
        """(h) ω⊥ = (0, 1, 0, 0): J⊓(ω, ω⊥) = ω⊥."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.6)
        annihilator = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.0)
        result = jurisdictional_meet(w, annihilator)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)
        assert result.uncertainty == pytest.approx(0.0)
        assert result.base_rate == pytest.approx(0.0)

    def test_non_idempotent(self):
        """Remark: J⊓(ω, ω) ≠ ω unless l ∈ {0, 1}."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = jurisdictional_meet(w, w)
        assert result.lawfulness != pytest.approx(w.lawfulness)  # l² ≠ l for l=0.7

    def test_idempotent_at_extremes(self):
        """l=1: J⊓(ω, ω) = ω. l=0: J⊓(ω, ω) has l=0."""
        certain = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = jurisdictional_meet(certain, certain)
        assert result.lawfulness == pytest.approx(1.0)

    def test_eu_us_counterexample(self):
        """§5: EU/US example showing meet ≠ fusion.

        ω_EU = (0.8, 0.05, 0.15, 0.5), ω_US = (0.3, 0.4, 0.3, 0.5).
        Meet: l = 0.24 (correct: must satisfy BOTH).
        Fusion would give l ≈ 0.73 (incorrect: conflates propositions).
        """
        eu = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        us = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3, base_rate=0.5)
        result = jurisdictional_meet(eu, us)
        assert result.lawfulness == pytest.approx(0.24)
        assert result.lawfulness < min(eu.lawfulness, us.lawfulness)


class TestJurisdictionalMeetNary:
    """N-ary jurisdictional meet via left-fold (associativity proven)."""

    def test_single_opinion(self):
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = jurisdictional_meet(w)
        assert result == w

    def test_three_way(self):
        w1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        w2 = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.6)
        w3 = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.7)
        result = jurisdictional_meet(w1, w2, w3)
        assert_valid_opinion(result, "3-way meet")
        # l = 0.8 * 0.7 * 0.9 = 0.504
        assert result.lawfulness == pytest.approx(0.8 * 0.7 * 0.9)

    def test_five_way_constraint(self):
        opinions = [
            ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5),
            ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.6),
            ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.7),
            ComplianceOpinion.create(lawfulness=0.6, violation=0.2, uncertainty=0.2, base_rate=0.4),
            ComplianceOpinion.create(lawfulness=0.85, violation=0.08, uncertainty=0.07, base_rate=0.55),
        ]
        result = jurisdictional_meet(*opinions)
        assert_valid_opinion(result, "5-way meet")
        # l must be product of all lawfulness values
        expected_l = 0.8 * 0.7 * 0.9 * 0.6 * 0.85
        assert result.lawfulness == pytest.approx(expected_l, rel=1e-6)

    def test_nary_matches_iterated_binary(self):
        """N-ary via *args must equal left-fold of binary meets."""
        opinions = [
            ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5),
            ComplianceOpinion.create(lawfulness=0.6, violation=0.2, uncertainty=0.2, base_rate=0.4),
            ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.6),
        ]
        nary = jurisdictional_meet(*opinions)
        iterated = jurisdictional_meet(jurisdictional_meet(opinions[0], opinions[1]), opinions[2])
        assert nary.lawfulness == pytest.approx(iterated.lawfulness, abs=TOL)
        assert nary.violation == pytest.approx(iterated.violation, abs=TOL)
        assert nary.uncertainty == pytest.approx(iterated.uncertainty, abs=TOL)

    def test_empty_raises(self):
        with pytest.raises((ValueError, TypeError)):
            jurisdictional_meet()


# ═══════════════════════════════════════════════════════════════════
# §6 — COMPLIANCE PROPAGATION (Definitions 4–7, Theorem 2)
# ═══════════════════════════════════════════════════════════════════


class TestCompliancePropagation:
    """Definition 6: Prop(τ, π, ω_S) = J⊓(τ, π, ω_S)."""

    def test_formula_correctness(self):
        """l_D = t·p·l_S, v_D = 1−(1−f)(1−q)(1−v_S)."""
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.6)
        purpose = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        result = compliance_propagation(source, trust, purpose)

        t, f = trust.lawfulness, trust.violation
        p, q = purpose.lawfulness, purpose.violation
        l_s, v_s = source.lawfulness, source.violation

        assert result.lawfulness == pytest.approx(t * p * l_s)
        assert result.violation == pytest.approx(1 - (1 - f) * (1 - q) * (1 - v_s))
        assert_valid_opinion(result, "propagation")

    def test_returns_compliance_opinion(self):
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        purpose = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = compliance_propagation(source, trust, purpose)
        assert isinstance(result, ComplianceOpinion)

    # ── Theorem 2 properties ──────────────────────────────────

    def test_theorem2a_constraint(self):
        """(a) Constraint and non-negativity from J⊓."""
        source = ComplianceOpinion.create(lawfulness=0.6, violation=0.2, uncertainty=0.2, base_rate=0.5)
        trust = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.4)
        purpose = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2, base_rate=0.6)
        result = compliance_propagation(source, trust, purpose)
        assert_valid_opinion(result, "Theorem 2(a)")

    def test_theorem2b_degradation_monotonicity(self):
        """(b) l_D ≤ l_S; v_D ≥ v_S."""
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.5)
        purpose = ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.1, base_rate=0.5)
        result = compliance_propagation(source, trust, purpose)
        assert result.lawfulness <= source.lawfulness + TOL
        assert result.violation >= source.violation - TOL

    def test_theorem2c_identity_derivation(self):
        """(c) τ_id = π_id = (1,0,0,1) gives Prop = ω_S."""
        source = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15, base_rate=0.4)
        identity = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = compliance_propagation(source, identity, identity)
        assert result.lawfulness == pytest.approx(source.lawfulness)
        assert result.violation == pytest.approx(source.violation)
        assert result.uncertainty == pytest.approx(source.uncertainty)

    def test_theorem2d_violation_annihilation_via_trust(self):
        """(d) τ = (0,1,0,0) → result is (0,1,0,0)."""
        source = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        violating_trust = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.0)
        purpose = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        result = compliance_propagation(source, violating_trust, purpose)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)

    def test_theorem2d_violation_annihilation_via_purpose(self):
        """(d) π = (0,1,0,0) → result is (0,1,0,0)."""
        source = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        violating_purpose = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.0)
        result = compliance_propagation(source, trust, violating_purpose)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)

    def test_theorem2d_violation_annihilation_via_source(self):
        """(d) ω_S = (0,1,0,0) → result is (0,1,0,0)."""
        violating_source = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.0)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        purpose = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        result = compliance_propagation(violating_source, trust, purpose)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)

    def test_theorem2e_chain_associativity(self):
        """(e) Two-step chain equals single five-way meet."""
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5)
        t1 = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.5)
        p1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        t2 = ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.1, base_rate=0.5)
        p2 = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)

        # Step-by-step propagation
        d1 = compliance_propagation(source, t1, p1)
        d2 = compliance_propagation(d1, t2, p2)

        # Single five-way meet
        flat = jurisdictional_meet(source, t1, p1, t2, p2)

        assert d2.lawfulness == pytest.approx(flat.lawfulness, abs=TOL)
        assert d2.violation == pytest.approx(flat.violation, abs=TOL)

    def test_theorem2f_multiplicative_decay(self):
        """(f) l_{D_n} = l_S · ∏(t_i · p_i)."""
        source = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        steps = [
            (0.9, 0.85),  # (t_i, p_i) — lawfulness components of trust and purpose
            (0.8, 0.7),
            (0.95, 0.9),
        ]
        expected_l = 0.9
        current = source
        for t_val, p_val in steps:
            trust = ComplianceOpinion.create(lawfulness=t_val, violation=1.0 - t_val, uncertainty=0.0, base_rate=0.5)
            purpose = ComplianceOpinion.create(lawfulness=p_val, violation=1.0 - p_val, uncertainty=0.0, base_rate=0.5)
            current = compliance_propagation(current, trust, purpose)
            expected_l *= t_val * p_val

        assert current.lawfulness == pytest.approx(expected_l, rel=1e-6)


class TestProvenanceChain:
    """Definition 7: Ordered chain recording each derivation step."""

    def test_create_empty_chain(self):
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        chain = ProvenanceChain(source=source, source_timestamp=100.0)
        assert chain.source == source
        assert len(chain.steps) == 0

    def test_add_step(self):
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        chain = ProvenanceChain(source=source, source_timestamp=100.0)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        purpose = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        chain.add_step(trust=trust, purpose=purpose, timestamp=200.0)
        assert len(chain.steps) == 1

    def test_compute_matches_propagation(self):
        """chain.compute() must match iterative compliance_propagation."""
        source = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5)
        t1 = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.5)
        p1 = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)

        chain = ProvenanceChain(source=source, source_timestamp=100.0)
        chain.add_step(trust=t1, purpose=p1, timestamp=200.0)
        chain_result = chain.compute()

        direct_result = compliance_propagation(source, t1, p1)
        assert chain_result.lawfulness == pytest.approx(direct_result.lawfulness, abs=TOL)
        assert chain_result.violation == pytest.approx(direct_result.violation, abs=TOL)

    def test_multi_step_chain(self):
        source = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        chain = ProvenanceChain(source=source, source_timestamp=0.0)

        t1 = ComplianceOpinion.create(lawfulness=0.95, violation=0.01, uncertainty=0.04, base_rate=0.5)
        p1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        chain.add_step(trust=t1, purpose=p1, timestamp=100.0)

        t2 = ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.1, base_rate=0.5)
        p2 = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        chain.add_step(trust=t2, purpose=p2, timestamp=200.0)

        result = chain.compute()
        assert_valid_opinion(result, "multi-step chain")
        # l should degrade through chain
        assert result.lawfulness < source.lawfulness


# ═══════════════════════════════════════════════════════════════════
# §7 — CONSENT ASSESSMENT (Definitions 8–11, Theorem 3)
# ═══════════════════════════════════════════════════════════════════


class TestConsentValidity:
    """Definition 8: ω_c^P = J⊓(ω_free, ω_spec, ω_inf, ω_unamb, ω_demo, ω_dist)."""

    def test_six_conditions(self):
        """Six Art. 7 conditions composed via jurisdictional meet."""
        freely_given = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08)
        specific = ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.10)
        informed = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        unambiguous = ComplianceOpinion.create(lawfulness=0.75, violation=0.05, uncertainty=0.2)
        demonstrable = ComplianceOpinion.create(lawfulness=0.95, violation=0.01, uncertainty=0.04)
        distinguishable = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)

        result = consent_validity(
            freely_given=freely_given,
            specific=specific,
            informed=informed,
            unambiguous=unambiguous,
            demonstrable=demonstrable,
            distinguishable=distinguishable,
        )
        assert isinstance(result, ComplianceOpinion)
        assert_valid_opinion(result, "consent validity")

        # l should be product of all six lawfulness values
        expected_l = 0.9 * 0.85 * 0.8 * 0.75 * 0.95 * 0.7
        assert result.lawfulness == pytest.approx(expected_l, rel=1e-6)

    def test_single_weak_condition_dominates(self):
        """One weak condition should drag down composite lawfulness."""
        strong = ComplianceOpinion.create(lawfulness=0.95, violation=0.02, uncertainty=0.03)
        weak = ComplianceOpinion.create(lawfulness=0.2, violation=0.5, uncertainty=0.3)

        result = consent_validity(
            freely_given=strong,
            specific=strong,
            informed=strong,
            unambiguous=strong,
            demonstrable=strong,
            distinguishable=weak,
        )
        assert result.lawfulness < weak.lawfulness  # Product of all, so even less

    def test_all_certain_compliance(self):
        """All conditions certain → consent certain."""
        certain = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = consent_validity(
            freely_given=certain,
            specific=certain,
            informed=certain,
            unambiguous=certain,
            demonstrable=certain,
            distinguishable=certain,
        )
        assert result.lawfulness == pytest.approx(1.0)
        assert result.violation == pytest.approx(0.0)

    def test_one_violated_condition(self):
        """One certain violation → composite violation."""
        ok = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08, base_rate=0.5)
        violated = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0, base_rate=0.0)
        result = consent_validity(
            freely_given=ok,
            specific=ok,
            informed=ok,
            unambiguous=ok,
            demonstrable=ok,
            distinguishable=violated,
        )
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)

    def test_also_accepts_positional_args(self):
        """consent_validity should accept 6 positional opinions too."""
        conditions = [
            ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
            for _ in range(6)
        ]
        result = consent_validity(*conditions)
        assert_valid_opinion(result)


class TestWithdrawalOverride:
    """Definition 11: Proposition replacement at withdrawal time."""

    def test_pre_withdrawal_returns_consent(self):
        """t < t_w → return ω_c^P (Theorem 3(b))."""
        consent = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        withdrawal = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        result = withdrawal_override(
            consent_opinion=consent,
            withdrawal_opinion=withdrawal,
            assessment_time=50.0,
            withdrawal_time=100.0,
        )
        assert result == consent

    def test_post_withdrawal_returns_withdrawal(self):
        """t ≥ t_w → return ω_w^P (Theorem 3(a))."""
        consent = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15, base_rate=0.5)
        withdrawal = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2, base_rate=0.5)
        result = withdrawal_override(
            consent_opinion=consent,
            withdrawal_opinion=withdrawal,
            assessment_time=100.0,
            withdrawal_time=100.0,
        )
        assert result == withdrawal

    def test_exact_boundary(self):
        """t == t_w → post-withdrawal (≥ condition)."""
        consent = ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08)
        withdrawal = ComplianceOpinion.create(lawfulness=0.6, violation=0.15, uncertainty=0.25)
        result = withdrawal_override(
            consent_opinion=consent,
            withdrawal_opinion=withdrawal,
            assessment_time=100.0,
            withdrawal_time=100.0,
        )
        assert result == withdrawal

    def test_theorem3c_non_interference(self):
        """Withdrawal for purpose P₁ doesn't affect P₂.

        Verified by calling withdrawal_override independently per purpose.
        """
        consent_p1 = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        consent_p2 = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15)
        withdrawal_p1 = ComplianceOpinion.create(lawfulness=0.3, violation=0.5, uncertainty=0.2)

        # Withdraw P1 at t=100
        eff_p1 = withdrawal_override(consent_p1, withdrawal_p1, assessment_time=150.0, withdrawal_time=100.0)
        eff_p2 = withdrawal_override(consent_p2, consent_p2, assessment_time=150.0, withdrawal_time=200.0)

        # P1 is withdrawn
        assert eff_p1 == withdrawal_p1
        # P2 is unaffected (no withdrawal yet)
        assert eff_p2 == consent_p2

    def test_returns_compliance_opinion(self):
        consent = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        withdrawal = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2)
        result = withdrawal_override(consent, withdrawal, assessment_time=150.0, withdrawal_time=100.0)
        assert isinstance(result, ComplianceOpinion)

    def test_accepts_plain_opinions(self):
        """Should accept Opinion, not just ComplianceOpinion."""
        consent = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        withdrawal = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        result = withdrawal_override(consent, withdrawal, assessment_time=150.0, withdrawal_time=100.0)
        assert isinstance(result, ComplianceOpinion)


# ═══════════════════════════════════════════════════════════════════
# §8 — TEMPORAL DECAY TRIGGERS (Definitions 12–14, Theorem 4)
# ═══════════════════════════════════════════════════════════════════


class TestExpiryTrigger:
    """Definition 12: Asymmetric l → v transition at trigger time."""

    def test_pre_trigger_unchanged(self):
        """t < t_T → return opinion unchanged."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = expiry_trigger(w, assessment_time=50.0, trigger_time=100.0, residual_factor=0.0)
        assert result == w

    def test_post_trigger_transfers_l_to_v(self):
        """t ≥ t_T: l' = γ·l, v' = v + (1−γ)·l, u' = u."""
        w = ComplianceOpinion.create(lawfulness=0.6, violation=0.1, uncertainty=0.3, base_rate=0.5)
        gamma = 0.3
        result = expiry_trigger(w, assessment_time=150.0, trigger_time=100.0, residual_factor=gamma)

        assert result.lawfulness == pytest.approx(gamma * 0.6)
        assert result.violation == pytest.approx(0.1 + (1 - gamma) * 0.6)
        assert result.uncertainty == pytest.approx(0.3)  # unchanged
        assert_valid_opinion(result, "expiry trigger")

    def test_theorem4b_constraint_preservation(self):
        """(b) l' + v' + u' = 1."""
        w = ComplianceOpinion.create(lawfulness=0.5, violation=0.2, uncertainty=0.3)
        for gamma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = expiry_trigger(w, assessment_time=200.0, trigger_time=100.0, residual_factor=gamma)
            assert_valid_opinion(result, f"expiry γ={gamma}")

    def test_theorem4c_expiry_monotonicity(self):
        """(c) l' ≤ l, v' ≥ v, u' = u."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = expiry_trigger(w, assessment_time=200.0, trigger_time=100.0, residual_factor=0.5)
        assert result.lawfulness <= w.lawfulness + TOL
        assert result.violation >= w.violation - TOL
        assert result.uncertainty == pytest.approx(w.uncertainty)

    def test_theorem4d_hard_expiry(self):
        """(d) γ = 0: all lawfulness converts to violation."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = expiry_trigger(w, assessment_time=200.0, trigger_time=100.0, residual_factor=0.0)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(0.1 + 0.7)
        assert result.uncertainty == pytest.approx(0.2)

    def test_gamma_one_no_effect(self):
        """γ = 1: no change (expiry has no immediate impact)."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = expiry_trigger(w, assessment_time=200.0, trigger_time=100.0, residual_factor=1.0)
        assert result.lawfulness == pytest.approx(w.lawfulness)
        assert result.violation == pytest.approx(w.violation)
        assert result.uncertainty == pytest.approx(w.uncertainty)

    def test_returns_compliance_opinion(self):
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = expiry_trigger(w, assessment_time=200.0, trigger_time=100.0, residual_factor=0.5)
        assert isinstance(result, ComplianceOpinion)

    def test_accepts_plain_opinion(self):
        op = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        result = expiry_trigger(op, assessment_time=200.0, trigger_time=100.0, residual_factor=0.5)
        assert isinstance(result, ComplianceOpinion)


class TestReviewDueTrigger:
    """Definition 13: Accelerated decay toward vacuity post-trigger."""

    def test_pre_trigger_unchanged(self):
        """t < t_T → return opinion unchanged."""
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = review_due_trigger(
            w, assessment_time=50.0, trigger_time=100.0, accelerated_half_life=30.0
        )
        assert result == w

    def test_post_trigger_accelerated_decay(self):
        """t ≥ t_T → apply accelerated decay using elapsed since trigger."""
        w = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5)
        half_life = 50.0
        t_assess = 200.0
        t_trigger = 100.0
        elapsed_since = t_assess - t_trigger

        result = review_due_trigger(
            w, assessment_time=t_assess, trigger_time=t_trigger,
            accelerated_half_life=half_life,
        )

        # Should match decay_opinion with elapsed = t_assess - t_trigger
        expected = decay_opinion(w, elapsed=elapsed_since, half_life=half_life)
        assert result.lawfulness == pytest.approx(expected.belief, abs=TOL)
        assert result.violation == pytest.approx(expected.disbelief, abs=TOL)
        assert result.uncertainty == pytest.approx(expected.uncertainty, abs=TOL)

    def test_moves_toward_vacuity_not_violation(self):
        """Review-due moves toward u, NOT toward v (unlike expiry)."""
        w = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        result = review_due_trigger(w, assessment_time=200.0, trigger_time=100.0, accelerated_half_life=50.0)
        # Uncertainty should increase; violation should decrease proportionally with l
        assert result.uncertainty > w.uncertainty
        # The b/d ratio is preserved by decay_opinion
        if result.lawfulness > TOL and result.violation > TOL:
            assert (result.lawfulness / result.violation) == pytest.approx(
                w.lawfulness / w.violation, rel=0.01
            )

    def test_returns_compliance_opinion(self):
        w = ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2)
        result = review_due_trigger(w, assessment_time=200.0, trigger_time=100.0, accelerated_half_life=50.0)
        assert isinstance(result, ComplianceOpinion)


class TestRegulatoryChangeTrigger:
    """Definition 14: Proposition replacement at trigger time."""

    def test_pre_trigger_returns_original(self):
        """t < t_T → return ω."""
        original = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        new = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3)
        result = regulatory_change_trigger(
            original, assessment_time=50.0, trigger_time=100.0, new_opinion=new
        )
        assert result == original

    def test_post_trigger_returns_new(self):
        """t ≥ t_T → return ω_new."""
        original = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        new = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3)
        result = regulatory_change_trigger(
            original, assessment_time=150.0, trigger_time=100.0, new_opinion=new
        )
        assert result == new

    def test_exact_boundary(self):
        """t == t_T → post-trigger (≥ condition)."""
        original = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        new = ComplianceOpinion.create(lawfulness=0.3, violation=0.4, uncertainty=0.3)
        result = regulatory_change_trigger(
            original, assessment_time=100.0, trigger_time=100.0, new_opinion=new
        )
        assert result == new

    def test_returns_compliance_opinion(self):
        original = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        new = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3)
        result = regulatory_change_trigger(
            original, assessment_time=150.0, trigger_time=100.0, new_opinion=new
        )
        assert isinstance(result, ComplianceOpinion)

    def test_theorem4e_non_commutativity(self):
        """(e) Different trigger types in different order → different results.

        Expiry then regulatory change ≠ regulatory change then expiry.
        """
        w = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        new = ComplianceOpinion.create(lawfulness=0.5, violation=0.2, uncertainty=0.3)

        # Order 1: expiry at t=100, then reg change at t=200
        after_expiry = expiry_trigger(w, assessment_time=150.0, trigger_time=100.0, residual_factor=0.5)
        order1 = regulatory_change_trigger(
            after_expiry, assessment_time=250.0, trigger_time=200.0, new_opinion=new
        )

        # Order 2: reg change at t=100, then expiry at t=200
        after_regchange = regulatory_change_trigger(
            w, assessment_time=150.0, trigger_time=100.0, new_opinion=new
        )
        order2 = expiry_trigger(
            after_regchange, assessment_time=250.0, trigger_time=200.0, residual_factor=0.5
        )

        # These should differ
        assert order1.lawfulness != pytest.approx(order2.lawfulness, abs=0.01)


# ═══════════════════════════════════════════════════════════════════
# §9 — ERASURE PROPAGATION (Definitions 15–17, Theorem 5, Prop. 1)
# ═══════════════════════════════════════════════════════════════════


class TestErasureScopeOpinion:
    """§9.2: ω_E^R = J⊓(ω_e^i : D_i ∈ S) — composite erasure completeness."""

    def test_basic_two_nodes(self):
        """Two-node scope: conjunction of erasure opinions."""
        e1 = ComplianceOpinion.create(lawfulness=0.95, violation=0.02, uncertainty=0.03, base_rate=0.5)
        e2 = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05, base_rate=0.5)
        result = erasure_scope_opinion(e1, e2)
        assert_valid_opinion(result, "erasure scope 2-node")
        assert result.lawfulness == pytest.approx(0.95 * 0.9)

    def test_theorem5a_exponential_degradation(self):
        """(a) e_R = ∏ e_i. With uniform e_i = p, e_R = p^|S|."""
        p = 0.95
        m = 10
        nodes = [
            ComplianceOpinion.create(
                lawfulness=p, violation=1.0 - p, uncertainty=0.0, base_rate=0.5
            )
            for _ in range(m)
        ]
        result = erasure_scope_opinion(*nodes)
        assert result.lawfulness == pytest.approx(p**m, rel=1e-6)

    def test_theorem5a_table1_spot_checks(self):
        """Table 1 from §9.2: p^m for various (p, m) pairs."""
        test_cases = [
            (0.99, 5, 0.951),
            (0.95, 10, 0.599),
            (0.90, 20, 0.122),
            (0.95, 50, 0.077),
            (0.90, 50, 0.005),
        ]
        for p, m, expected in test_cases:
            nodes = [
                ComplianceOpinion.create(
                    lawfulness=p, violation=1.0 - p, uncertainty=0.0, base_rate=0.5
                )
                for _ in range(m)
            ]
            result = erasure_scope_opinion(*nodes)
            assert result.lawfulness == pytest.approx(expected, abs=0.002), (
                f"p={p}, m={m}: expected {expected}, got {result.lawfulness}"
            )

    def test_theorem5b_scope_monotonicity(self):
        """(b) Adding a node can only decrease e_R."""
        e1 = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        e2 = ComplianceOpinion.create(lawfulness=0.85, violation=0.1, uncertainty=0.05)
        e3 = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)

        r2 = erasure_scope_opinion(e1, e2)
        r3 = erasure_scope_opinion(e1, e2, e3)
        assert r3.lawfulness <= r2.lawfulness + TOL

    def test_theorem5c_exception_filtering(self):
        """(c) Removing a node increases e_R."""
        e1 = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        e2 = ComplianceOpinion.create(lawfulness=0.85, violation=0.1, uncertainty=0.05)
        e3 = ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2)  # weak node

        with_weak = erasure_scope_opinion(e1, e2, e3)
        without_weak = erasure_scope_opinion(e1, e2)
        assert without_weak.lawfulness >= with_weak.lawfulness - TOL

    def test_theorem5d_perfect_source(self):
        """(d) Perfect erasure at one node contributes no degradation."""
        perfect = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        other = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1, base_rate=0.5)
        result = erasure_scope_opinion(perfect, other)
        assert result.lawfulness == pytest.approx(other.lawfulness)

    def test_single_node(self):
        e = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        result = erasure_scope_opinion(e)
        assert result == e


class TestResidualContamination:
    """Definition 17, Proposition 1: Disjunctive contamination risk."""

    def test_formula_correctness(self):
        """r = 1−∏(1−ē_i), r̄ = ∏ e_i, u_r = ∏(1−ē_i)−∏ e_i.

        Where ē_i = disbelief (evidence of persistence),
              e_i = belief (evidence of erasure).
        """
        # Two ancestor opinions
        a1 = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        a2 = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)

        result = residual_contamination(a1, a2)

        e_bar_1 = a1.violation  # ē₁ = 0.05 (persistence evidence)
        e_bar_2 = a2.violation  # ē₂ = 0.10
        e_1 = a1.lawfulness    # e₁ = 0.9  (erasure evidence)
        e_2 = a2.lawfulness    # e₂ = 0.8

        expected_r = 1 - (1 - e_bar_1) * (1 - e_bar_2)
        expected_r_bar = e_1 * e_2
        expected_u = (1 - e_bar_1) * (1 - e_bar_2) - e_1 * e_2

        # In the contamination result:
        # violation = r (contamination risk)
        # lawfulness = r̄ (clean probability)
        # uncertainty = u_r
        assert result.violation == pytest.approx(expected_r)
        assert result.lawfulness == pytest.approx(expected_r_bar)
        assert result.uncertainty == pytest.approx(expected_u)

    def test_proposition1_constraint(self):
        """Proposition 1: r + r̄ + u_r = 1."""
        ancestors = [
            ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05),
            ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1),
            ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15),
        ]
        result = residual_contamination(*ancestors)
        assert_valid_opinion(result, "Proposition 1 constraint")

    def test_proposition1_non_negativity(self):
        """All components non-negative."""
        ancestors = [
            ComplianceOpinion.create(lawfulness=0.5, violation=0.3, uncertainty=0.2),
            ComplianceOpinion.create(lawfulness=0.6, violation=0.25, uncertainty=0.15),
        ]
        result = residual_contamination(*ancestors)
        assert result.lawfulness >= -TOL
        assert result.violation >= -TOL
        assert result.uncertainty >= -TOL

    def test_proposition1_monotonic_depth(self):
        """Deeper nodes face higher contamination risk (r non-decreasing in |A⁺|)."""
        base = ComplianceOpinion.create(lawfulness=0.85, violation=0.1, uncertainty=0.05)
        risks = []
        for depth in range(1, 6):
            ancestors = [base] * depth
            result = residual_contamination(*ancestors)
            risks.append(result.violation)

        for i in range(1, len(risks)):
            assert risks[i] >= risks[i - 1] - TOL, (
                f"Risk should increase with depth: {risks}"
            )

    def test_single_ancestor(self):
        """With one ancestor, risk = ē, clean = e."""
        a = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        result = residual_contamination(a)
        assert result.violation == pytest.approx(a.violation)
        assert result.lawfulness == pytest.approx(a.lawfulness)
        assert result.uncertainty == pytest.approx(a.uncertainty)

    def test_perfect_erasure_all_ancestors(self):
        """All ancestors perfectly erased → zero contamination."""
        perfect = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = residual_contamination(perfect, perfect, perfect)
        assert result.violation == pytest.approx(0.0)
        assert result.lawfulness == pytest.approx(1.0)

    def test_one_certain_persistence(self):
        """One ancestor with certain persistence → contamination = 1."""
        ok = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        persists = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0)
        result = residual_contamination(ok, persists)
        assert result.violation == pytest.approx(1.0)
        assert result.lawfulness == pytest.approx(0.0)

    def test_returns_compliance_opinion(self):
        a = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        result = residual_contamination(a, a)
        assert isinstance(result, ComplianceOpinion)

    def test_accepts_plain_opinions(self):
        a = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        result = residual_contamination(a, a)
        assert isinstance(result, ComplianceOpinion)


# ═══════════════════════════════════════════════════════════════════
# §10 — OPERATOR INTERACTIONS
# ═══════════════════════════════════════════════════════════════════


class TestOperatorInteractions:
    """§10: Operators compose to model complete compliance lifecycles."""

    def test_consent_to_propagation(self):
        """Consent output feeds into propagation as the lawful basis.

        Prop(τ, π, EffConsent(P, t))
        """
        # Assess consent
        conditions = [
            ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08)
            for _ in range(6)
        ]
        consent = consent_validity(*conditions)

        # Propagate through derivation
        trust = ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.1)
        purpose = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15)
        derived = compliance_propagation(consent, trust, purpose)

        assert_valid_opinion(derived)
        assert derived.lawfulness < consent.lawfulness  # Derivation degrades

    def test_withdrawal_to_erasure(self):
        """Consent withdrawal triggers erasure obligation.

        Withdrawal Override → Erasure Propagation
        """
        consent = ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15)
        withdrawal = ComplianceOpinion.create(lawfulness=0.1, violation=0.7, uncertainty=0.2)

        # Post-withdrawal
        effective = withdrawal_override(consent, withdrawal, assessment_time=200.0, withdrawal_time=100.0)

        # Erasure across 3 nodes
        erasure_nodes = [
            ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05),
            ComplianceOpinion.create(lawfulness=0.85, violation=0.08, uncertainty=0.07),
            ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15),
        ]
        erasure = erasure_scope_opinion(*erasure_nodes)

        # Compose: post-withdrawal compliance with erasure
        final = jurisdictional_meet(effective, erasure)
        assert_valid_opinion(final)
        assert final.lawfulness < effective.lawfulness

    def test_temporal_decay_applies_to_all(self):
        """All compliance opinions are subject to temporal decay."""
        consent = consent_validity(
            *[ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08) for _ in range(6)]
        )
        # Decay the consent opinion
        decayed = decay_opinion(consent, elapsed=365.0, half_life=365.0)
        decayed_co = ComplianceOpinion.from_opinion(decayed)
        assert decayed_co.lawfulness < consent.lawfulness
        assert decayed_co.uncertainty > consent.uncertainty

    def test_erasure_to_compliance_update(self):
        """Post-erasure: ω_updated = J⊓(ω_processing, ω_erasure)."""
        processing = ComplianceOpinion.create(lawfulness=0.7, violation=0.15, uncertainty=0.15)
        erasure = ComplianceOpinion.create(lawfulness=0.85, violation=0.08, uncertainty=0.07)
        updated = jurisdictional_meet(processing, erasure)
        assert_valid_opinion(updated)
        assert updated.lawfulness <= min(processing.lawfulness, erasure.lawfulness) + TOL

    def test_full_lifecycle(self):
        """Full chain: consent → propagation → temporal decay → erasure.

        This validates that all operators compose without errors and
        produce valid opinions throughout.
        """
        # 1. Consent assessment
        consent = consent_validity(
            freely_given=ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08),
            specific=ComplianceOpinion.create(lawfulness=0.85, violation=0.05, uncertainty=0.10),
            informed=ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1),
            unambiguous=ComplianceOpinion.create(lawfulness=0.9, violation=0.03, uncertainty=0.07),
            demonstrable=ComplianceOpinion.create(lawfulness=0.95, violation=0.01, uncertainty=0.04),
            distinguishable=ComplianceOpinion.create(lawfulness=0.7, violation=0.1, uncertainty=0.2),
        )
        assert_valid_opinion(consent, "lifecycle: consent")

        # 2. Propagation through derivation
        derived = compliance_propagation(
            source=consent,
            derivation_trust=ComplianceOpinion.create(lawfulness=0.9, violation=0.02, uncertainty=0.08),
            purpose_compat=ComplianceOpinion.create(lawfulness=0.8, violation=0.05, uncertainty=0.15),
        )
        assert_valid_opinion(derived, "lifecycle: propagation")
        assert derived.lawfulness < consent.lawfulness

        # 3. Temporal decay (1 year, half-life 2 years)
        decayed = decay_opinion(derived, elapsed=365.0, half_life=730.0)
        decayed_co = ComplianceOpinion.from_opinion(decayed)
        assert_valid_opinion(decayed_co, "lifecycle: decay")
        assert decayed_co.lawfulness < derived.lawfulness

        # 4. Expiry trigger (data retention period expired)
        expired = expiry_trigger(
            decayed_co, assessment_time=400.0, trigger_time=365.0, residual_factor=0.0
        )
        assert_valid_opinion(expired, "lifecycle: expiry")
        assert expired.lawfulness == pytest.approx(0.0)

        # 5. Erasure
        erasure = erasure_scope_opinion(
            ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05),
            ComplianceOpinion.create(lawfulness=0.85, violation=0.08, uncertainty=0.07),
        )
        assert_valid_opinion(erasure, "lifecycle: erasure")

        # 6. Final compliance update
        final = jurisdictional_meet(expired, erasure)
        assert_valid_opinion(final, "lifecycle: final")


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES AND ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases across all operators."""

    def test_meet_all_vacuous(self):
        """Meet of vacuous opinions → vacuous with product base rate."""
        vac1 = ComplianceOpinion.create(lawfulness=0.0, violation=0.0, uncertainty=1.0, base_rate=0.5)
        vac2 = ComplianceOpinion.create(lawfulness=0.0, violation=0.0, uncertainty=1.0, base_rate=0.4)
        result = jurisdictional_meet(vac1, vac2)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(0.0)
        assert result.uncertainty == pytest.approx(1.0)
        assert result.base_rate == pytest.approx(0.2)

    def test_meet_all_dogmatic_compliant(self):
        """Meet of certain compliance → certain compliance."""
        c1 = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        c2 = ComplianceOpinion.create(lawfulness=1.0, violation=0.0, uncertainty=0.0, base_rate=1.0)
        result = jurisdictional_meet(c1, c2)
        assert result.lawfulness == pytest.approx(1.0)

    def test_propagation_vacuous_source(self):
        """Vacuous source: complete ignorance propagates."""
        vacuous = ComplianceOpinion.create(lawfulness=0.0, violation=0.0, uncertainty=1.0, base_rate=0.5)
        trust = ComplianceOpinion.create(lawfulness=0.9, violation=0.05, uncertainty=0.05)
        purpose = ComplianceOpinion.create(lawfulness=0.8, violation=0.1, uncertainty=0.1)
        result = compliance_propagation(vacuous, trust, purpose)
        assert_valid_opinion(result)
        assert result.lawfulness == pytest.approx(0.0)  # 0 * anything = 0

    def test_contamination_all_vacuous(self):
        """All ancestors vacuous: contamination is vacuous."""
        vac = ComplianceOpinion.create(lawfulness=0.0, violation=0.0, uncertainty=1.0, base_rate=0.5)
        result = residual_contamination(vac, vac)
        assert_valid_opinion(result)
        assert result.lawfulness == pytest.approx(0.0)   # r̄ = ∏ e_i = 0
        assert result.violation == pytest.approx(0.0)     # r = 1 − ∏(1−ē_i) = 0
        assert result.uncertainty == pytest.approx(1.0)   # all uncertain

    def test_expiry_on_vacuous_opinion(self):
        """Expiry on vacuous: l=0 so no transfer, stays vacuous."""
        vac = ComplianceOpinion.create(lawfulness=0.0, violation=0.0, uncertainty=1.0)
        result = expiry_trigger(vac, assessment_time=200.0, trigger_time=100.0, residual_factor=0.0)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(0.0)
        assert result.uncertainty == pytest.approx(1.0)

    def test_expiry_on_certain_violation(self):
        """Expiry on (0,1,0): already violated, expiry changes nothing."""
        violated = ComplianceOpinion.create(lawfulness=0.0, violation=1.0, uncertainty=0.0)
        result = expiry_trigger(violated, assessment_time=200.0, trigger_time=100.0, residual_factor=0.0)
        assert result.lawfulness == pytest.approx(0.0)
        assert result.violation == pytest.approx(1.0)
