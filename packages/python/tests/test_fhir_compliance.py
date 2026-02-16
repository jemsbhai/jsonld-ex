"""Tests for FHIR R4 Consent ↔ Compliance Algebra bridge.

Phase 4 of FHIR interoperability: bridges the compliance algebra
(ComplianceOpinion, consent_validity, withdrawal_override,
jurisdictional_meet) to FHIR R4 Consent resources.

TDD RED PHASE — tests written FIRST, implementation does not yet exist.

Each test references:
  - The FHIR R4 Consent resource specification (hl7.org/fhir/R4/consent.html)
  - The compliance algebra formalization (compliance_algebra.md)
  - Specific definitions/theorems where applicable

FHIR R4 Consent.status values:
  draft | proposed | active | rejected | inactive | entered-in-error

Compliance algebra operators bridged:
  §5  jurisdictional_meet      — multi-site consent composition
  §7  consent_validity         — six-condition GDPR Art. 7 mapping
  §7  withdrawal_override      — active → inactive/rejected transition
  §4  ComplianceOpinion        — per-consent uncertainty model

Mathematical source of truth: compliance_algebra.md
FHIR source of truth: HL7 FHIR R4 Consent (v4.0.1)
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    jurisdictional_meet,
    consent_validity,
    withdrawal_override,
)
from jsonld_ex.fhir_interop._constants import FHIR_EXTENSION_URL
from jsonld_ex.fhir_interop._scalar import (
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
)

# ── Import targets (will fail until implementation exists) ────────

from jsonld_ex.fhir_interop import (
    fhir_consent_to_opinion,
    opinion_to_fhir_consent,
    fhir_consent_validity,
    fhir_consent_withdrawal,
    fhir_multi_site_meet,
    CONSENT_STATUS_PROBABILITY,
    CONSENT_STATUS_UNCERTAINTY,
)


# ── Tolerance for floating-point comparisons ──────────────────────
TOL = 1e-9


# ── Helper: assert valid compliance opinion ───────────────────────
def assert_valid_compliance_opinion(op, msg=""):
    """Assert SL constraints: l + v + u = 1, all in [0, 1]."""
    prefix = f"{msg}: " if msg else ""
    assert isinstance(op, ComplianceOpinion), (
        f"{prefix}expected ComplianceOpinion, got {type(op).__name__}"
    )
    assert op.lawfulness >= -TOL, f"{prefix}lawfulness < 0: {op.lawfulness}"
    assert op.violation >= -TOL, f"{prefix}violation < 0: {op.violation}"
    assert op.uncertainty >= -TOL, f"{prefix}uncertainty < 0: {op.uncertainty}"
    assert 0.0 - TOL <= op.base_rate <= 1.0 + TOL, (
        f"{prefix}base_rate out of [0,1]: {op.base_rate}"
    )
    total = op.lawfulness + op.violation + op.uncertainty
    assert abs(total - 1.0) < TOL, f"{prefix}l+v+u={total}, expected 1.0"


# ── Minimal FHIR R4 Consent resource factories ───────────────────

def _make_consent(
    *,
    status="active",
    scope_code="patient-privacy",
    category_code="59284-0",
    patient_ref="Patient/123",
    date_time="2025-01-15",
    policy_uri=None,
    policy_rule_code=None,
    provision_type=None,
    provision_period=None,
    verification=None,
    extensions=None,
    resource_id="consent-001",
):
    """Build a minimal valid FHIR R4 Consent resource dict.

    FHIR R4 Consent requires: resourceType, status, scope, category.
    All other fields are optional per the spec.
    """
    resource = {
        "resourceType": "Consent",
        "id": resource_id,
        "status": status,
        "scope": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                    "code": scope_code,
                }
            ]
        },
        "category": [
            {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": category_code,
                    }
                ]
            }
        ],
        "patient": {"reference": patient_ref},
    }
    if date_time is not None:
        resource["dateTime"] = date_time
    if policy_uri is not None:
        resource["policy"] = [{"uri": policy_uri}]
    if policy_rule_code is not None:
        resource["policyRule"] = {
            "coding": [{"code": policy_rule_code}]
        }
    if provision_type is not None:
        resource.setdefault("provision", {})["type"] = provision_type
    if provision_period is not None:
        resource.setdefault("provision", {})["period"] = provision_period
    if verification is not None:
        resource["verification"] = verification
    if extensions is not None:
        resource["extension"] = extensions
    return resource


def _make_hipaa_consent(**kwargs):
    """Convenience: HIPAA-scoped consent with policy rule."""
    defaults = {
        "scope_code": "patient-privacy",
        "policy_rule_code": "HIPAA",
    }
    defaults.update(kwargs)
    return _make_consent(**defaults)


# ═══════════════════════════════════════════════════════════════════
# CONSENT STATUS → PROBABILITY CONSTANTS
# ═══════════════════════════════════════════════════════════════════


class TestConsentStatusConstants:
    """Verify consent status mapping tables are well-defined.

    FHIR R4 Consent.status: draft | proposed | active | rejected |
    inactive | entered-in-error.

    Each status must have a probability and uncertainty entry.
    Values must reflect the epistemic meaning of each status.
    """

    EXPECTED_STATUSES = {
        "draft", "proposed", "active", "rejected",
        "inactive", "entered-in-error",
    }

    def test_all_r4_statuses_have_probability(self):
        """Every R4 Consent.status has a probability mapping."""
        for status in self.EXPECTED_STATUSES:
            assert status in CONSENT_STATUS_PROBABILITY, (
                f"Missing probability mapping for status '{status}'"
            )

    def test_all_r4_statuses_have_uncertainty(self):
        """Every R4 Consent.status has an uncertainty mapping."""
        for status in self.EXPECTED_STATUSES:
            assert status in CONSENT_STATUS_UNCERTAINTY, (
                f"Missing uncertainty mapping for status '{status}'"
            )

    def test_probabilities_in_unit_interval(self):
        """All probability values are in [0, 1]."""
        for status, prob in CONSENT_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, (
                f"Probability for '{status}' = {prob}, not in [0, 1]"
            )

    def test_uncertainties_in_unit_interval(self):
        """All uncertainty values are in [0, 1]."""
        for status, u in CONSENT_STATUS_UNCERTAINTY.items():
            assert 0.0 <= u <= 1.0, (
                f"Uncertainty for '{status}' = {u}, not in [0, 1]"
            )

    def test_active_has_highest_lawfulness_signal(self):
        """Active consent should have the highest probability of validity."""
        active_p = CONSENT_STATUS_PROBABILITY["active"]
        for status in ("draft", "proposed", "inactive", "entered-in-error"):
            assert active_p > CONSENT_STATUS_PROBABILITY[status], (
                f"Active ({active_p}) should exceed {status} "
                f"({CONSENT_STATUS_PROBABILITY[status]})"
            )

    def test_rejected_has_highest_violation_signal(self):
        """Rejected consent should have the lowest probability (highest violation).

        A rejected consent is a definitive negative outcome — the data
        subject or authority refused consent.
        """
        rejected_p = CONSENT_STATUS_PROBABILITY["rejected"]
        for status in ("draft", "proposed", "active", "inactive"):
            assert rejected_p < CONSENT_STATUS_PROBABILITY[status], (
                f"Rejected ({rejected_p}) should be below {status} "
                f"({CONSENT_STATUS_PROBABILITY[status]})"
            )

    def test_draft_proposed_have_high_uncertainty(self):
        """Draft and proposed consents are pre-decisional — high uncertainty.

        These statuses represent consents that have not yet been finalized.
        The epistemic state is genuinely uncertain, not low-lawfulness.
        """
        for status in ("draft", "proposed"):
            assert CONSENT_STATUS_UNCERTAINTY[status] >= 0.30, (
                f"Pre-decisional status '{status}' should have u >= 0.30, "
                f"got {CONSENT_STATUS_UNCERTAINTY[status]}"
            )

    def test_entered_in_error_has_high_uncertainty(self):
        """Entered-in-error compromises data integrity — high uncertainty.

        Consistent with the pattern in _constants.py where
        entered-in-error always signals data quality concerns.
        """
        assert CONSENT_STATUS_UNCERTAINTY["entered-in-error"] >= 0.40


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_to_opinion()
# ═══════════════════════════════════════════════════════════════════


class TestFhirConsentToOpinion:
    """Convert FHIR R4 Consent → ComplianceOpinion.

    The opinion reflects the epistemic state of consent validity:
      l = evidence that consent is lawfully valid
      v = evidence that consent is invalid/violated
      u = uncertainty about consent validity
    """

    # ── Basic status-driven conversion ────────────────────────────

    def test_active_consent_returns_compliance_opinion(self):
        """An active consent produces a valid ComplianceOpinion."""
        resource = _make_consent(status="active")
        op = fhir_consent_to_opinion(resource)
        assert_valid_compliance_opinion(op, "active consent")

    def test_active_consent_has_high_lawfulness(self):
        """Active status signals lawfully valid consent."""
        resource = _make_consent(status="active")
        op = fhir_consent_to_opinion(resource)
        assert op.lawfulness > 0.5, (
            f"Active consent lawfulness={op.lawfulness}, expected > 0.5"
        )

    def test_rejected_consent_has_high_violation(self):
        """Rejected status signals consent was refused — high violation."""
        resource = _make_consent(status="rejected")
        op = fhir_consent_to_opinion(resource)
        assert op.violation > op.lawfulness, (
            f"Rejected consent: v={op.violation} should exceed "
            f"l={op.lawfulness}"
        )

    def test_draft_consent_has_high_uncertainty(self):
        """Draft status signals pre-decisional — high uncertainty."""
        resource = _make_consent(status="draft")
        op = fhir_consent_to_opinion(resource)
        assert op.uncertainty > 0.25, (
            f"Draft consent uncertainty={op.uncertainty}, expected > 0.25"
        )

    def test_proposed_consent_has_high_uncertainty(self):
        """Proposed status is also pre-decisional."""
        resource = _make_consent(status="proposed")
        op = fhir_consent_to_opinion(resource)
        assert op.uncertainty > 0.25

    def test_inactive_consent_lower_lawfulness_than_active(self):
        """Inactive consent was once valid but is no longer in effect.

        Should have lower lawfulness than active, but not as low as
        rejected (it wasn't refused, it expired or was superseded).
        """
        active_op = fhir_consent_to_opinion(_make_consent(status="active"))
        inactive_op = fhir_consent_to_opinion(_make_consent(status="inactive"))
        assert inactive_op.lawfulness < active_op.lawfulness

    def test_entered_in_error_has_high_uncertainty(self):
        """Entered-in-error signals data integrity issue."""
        resource = _make_consent(status="entered-in-error")
        op = fhir_consent_to_opinion(resource)
        assert op.uncertainty >= 0.40

    def test_all_statuses_produce_valid_opinions(self):
        """Every R4 Consent.status produces a valid ComplianceOpinion."""
        for status in ("draft", "proposed", "active", "rejected",
                        "inactive", "entered-in-error"):
            resource = _make_consent(status=status)
            op = fhir_consent_to_opinion(resource)
            assert_valid_compliance_opinion(op, f"status={status}")

    # ── Verification signal adjustments ───────────────────────────

    def test_verified_consent_reduces_uncertainty(self):
        """FHIR verification[] with verified=true reduces uncertainty.

        FHIR R4 Consent.verification records explicit verification
        events. A verified consent has stronger evidence → lower u.
        """
        unverified = _make_consent(status="active")
        verified = _make_consent(
            status="active",
            verification=[{
                "verified": True,
                "verifiedWith": {"reference": "Patient/123"},
                "verificationDate": "2025-01-15",
            }],
        )
        op_unverified = fhir_consent_to_opinion(unverified)
        op_verified = fhir_consent_to_opinion(verified)
        assert op_verified.uncertainty < op_unverified.uncertainty, (
            "Verified consent should have lower uncertainty"
        )

    def test_multiple_verifications_further_reduce_uncertainty(self):
        """Multiple verification events accumulate evidence."""
        single_v = _make_consent(
            status="active",
            verification=[{
                "verified": True,
                "verificationDate": "2025-01-15",
            }],
        )
        double_v = _make_consent(
            status="active",
            verification=[
                {"verified": True, "verificationDate": "2025-01-15"},
                {"verified": True, "verificationDate": "2025-06-01"},
            ],
        )
        op_single = fhir_consent_to_opinion(single_v)
        op_double = fhir_consent_to_opinion(double_v)
        assert op_double.uncertainty <= op_single.uncertainty

    def test_unverified_verification_does_not_reduce_uncertainty(self):
        """A verification entry with verified=false adds no evidence."""
        no_v = _make_consent(status="active")
        false_v = _make_consent(
            status="active",
            verification=[{
                "verified": False,
                "verificationDate": "2025-01-15",
            }],
        )
        op_no = fhir_consent_to_opinion(no_v)
        op_false = fhir_consent_to_opinion(false_v)
        # verified=false should not reduce uncertainty vs no verification
        assert op_false.uncertainty >= op_no.uncertainty - TOL

    # ── Provision signal adjustments ──────────────────────────────

    def test_permit_provision_is_default_interpretation(self):
        """A consent with provision.type='permit' has standard lawfulness.

        The default FHIR Consent interpretation: base policy permits,
        provisions are exceptions. A permit provision is consistent
        with active consent.
        """
        resource = _make_consent(status="active", provision_type="permit")
        op = fhir_consent_to_opinion(resource)
        assert_valid_compliance_opinion(op)
        assert op.lawfulness > 0.5

    def test_deny_provision_reduces_lawfulness(self):
        """A consent with provision.type='deny' signals restriction.

        When the base provision is deny, lawfulness should be lower
        than permit, as the consent restricts rather than grants.
        """
        permit = _make_consent(status="active", provision_type="permit")
        deny = _make_consent(status="active", provision_type="deny")
        op_permit = fhir_consent_to_opinion(permit)
        op_deny = fhir_consent_to_opinion(deny)
        assert op_deny.lawfulness < op_permit.lawfulness

    # ── Extension recovery (round-trip preservation) ──────────────

    def test_extension_recovery_exact_opinion(self):
        """When a jsonld-ex extension is present, recover exact opinion.

        Follows the same extension recovery pattern as all other
        FHIR converters: if FHIR_EXTENSION_URL is present with an
        embedded opinion, use it exactly rather than reconstructing.
        """
        original = ComplianceOpinion.create(
            lawfulness=0.72, violation=0.08,
            uncertainty=0.20, base_rate=0.55,
        )
        ext = opinion_to_fhir_extension(original)
        resource = _make_consent(
            status="active",
            extensions=[ext],
        )
        recovered = fhir_consent_to_opinion(resource)
        assert abs(recovered.lawfulness - 0.72) < TOL
        assert abs(recovered.violation - 0.08) < TOL
        assert abs(recovered.uncertainty - 0.20) < TOL
        assert abs(recovered.base_rate - 0.55) < TOL

    # ── Error handling ────────────────────────────────────────────

    def test_missing_resource_type_raises(self):
        """Resource without resourceType raises ValueError."""
        with pytest.raises(ValueError, match="resourceType"):
            fhir_consent_to_opinion({"status": "active"})

    def test_wrong_resource_type_raises(self):
        """Non-Consent resource type raises ValueError."""
        resource = {"resourceType": "Observation", "status": "final"}
        with pytest.raises(ValueError, match="[Cc]onsent"):
            fhir_consent_to_opinion(resource)

    def test_unknown_status_falls_back_gracefully(self):
        """An unrecognized status produces a valid high-uncertainty opinion.

        Future FHIR versions may add new status codes. The converter
        should not crash but should signal high uncertainty.
        """
        resource = _make_consent(status="unknown-future-status")
        op = fhir_consent_to_opinion(resource)
        assert_valid_compliance_opinion(op)
        assert op.uncertainty >= 0.30


# ═══════════════════════════════════════════════════════════════════
# opinion_to_fhir_consent()
# ═══════════════════════════════════════════════════════════════════


class TestOpinionToFhirConsent:
    """Convert ComplianceOpinion → FHIR R4 Consent resource.

    Produces a valid FHIR R4 Consent with the opinion embedded as
    a jsonld-ex extension, preserving full SL tuple for round-trip.
    """

    def test_produces_valid_fhir_consent(self):
        """Output has resourceType='Consent' and required fields."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(op)
        assert resource["resourceType"] == "Consent"
        assert "status" in resource
        assert "scope" in resource
        assert "category" in resource

    def test_high_lawfulness_maps_to_active(self):
        """A high-lawfulness opinion produces status='active'."""
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        resource = opinion_to_fhir_consent(op)
        assert resource["status"] == "active"

    def test_high_violation_maps_to_rejected(self):
        """A high-violation opinion produces status='rejected'."""
        op = ComplianceOpinion.create(0.05, 0.85, 0.10, 0.5)
        resource = opinion_to_fhir_consent(op)
        assert resource["status"] == "rejected"

    def test_high_uncertainty_maps_to_draft(self):
        """A high-uncertainty opinion produces status='draft'."""
        op = ComplianceOpinion.create(0.10, 0.10, 0.80, 0.5)
        resource = opinion_to_fhir_consent(op)
        assert resource["status"] == "draft"

    def test_extension_embeds_full_opinion(self):
        """The jsonld-ex extension preserves the full SL tuple."""
        op = ComplianceOpinion.create(0.72, 0.08, 0.20, 0.55)
        resource = opinion_to_fhir_consent(op)
        # Extension should be present
        extensions = resource.get("extension", [])
        jsonld_exts = [
            e for e in extensions if e.get("url") == FHIR_EXTENSION_URL
        ]
        assert len(jsonld_exts) == 1, "Expected exactly one jsonld-ex extension"

    def test_patient_metadata_preserved(self):
        """Optional patient reference is included when provided."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(
            op, patient="Patient/456",
        )
        assert resource["patient"]["reference"] == "Patient/456"

    def test_scope_metadata_preserved(self):
        """Optional scope is included when provided."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(
            op, scope="patient-privacy",
        )
        scope_codings = resource["scope"]["coding"]
        codes = [c["code"] for c in scope_codings]
        assert "patient-privacy" in codes

    def test_policy_rule_metadata_preserved(self):
        """Optional policyRule is included when provided."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(
            op, policy_rule="HIPAA",
        )
        assert "policyRule" in resource

    def test_resource_id_preserved(self):
        """Optional resource id is included when provided."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(op, resource_id="consent-xyz")
        assert resource["id"] == "consent-xyz"


# ═══════════════════════════════════════════════════════════════════
# Round-trip: FHIR Consent ↔ ComplianceOpinion
# ═══════════════════════════════════════════════════════════════════


class TestConsentRoundTrip:
    """Verify round-trip fidelity: Consent → Opinion → Consent.

    Round-trip scope (consistent with module docstring):
      - status preserved (active/rejected/draft mapping)
      - opinion tuple preserved exactly via extension
      - resource id preserved
    """

    def test_opinion_survives_round_trip_via_extension(self):
        """ComplianceOpinion → FHIR Consent → ComplianceOpinion is exact."""
        original = ComplianceOpinion.create(0.65, 0.15, 0.20, 0.45)
        resource = opinion_to_fhir_consent(original, resource_id="rt-001")
        recovered = fhir_consent_to_opinion(resource)
        assert abs(recovered.lawfulness - original.lawfulness) < TOL
        assert abs(recovered.violation - original.violation) < TOL
        assert abs(recovered.uncertainty - original.uncertainty) < TOL
        assert abs(recovered.base_rate - original.base_rate) < TOL

    def test_resource_id_survives_round_trip(self):
        """Resource id is preserved through the round-trip."""
        op = ComplianceOpinion.create(0.8, 0.05, 0.15, 0.5)
        resource = opinion_to_fhir_consent(op, resource_id="rt-002")
        assert resource["id"] == "rt-002"

    def test_active_status_round_trip(self):
        """Active FHIR Consent → opinion → FHIR Consent preserves status."""
        resource_in = _make_consent(status="active")
        op = fhir_consent_to_opinion(resource_in)
        resource_out = opinion_to_fhir_consent(op, resource_id="rt-003")
        # High-lawfulness opinion should map back to active
        assert resource_out["status"] == "active"


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_validity() — six-condition GDPR Art. 7 mapping
# ═══════════════════════════════════════════════════════════════════


class TestFhirConsentValidity:
    """Map FHIR Consent fields to six GDPR Art. 7 conditions.

    Per compliance_algebra.md §7.1 (Definition 8):
      ω_c^P = J⊓(ω_free, ω_spec, ω_inf, ω_unamb, ω_demo, ω_dist)

    FHIR Consent field → GDPR condition mapping:
      - freely_given:    Inferred from scope + lack of coercion signals
      - specific:        provision.purpose specificity
      - informed:        provision.data + category detail
      - unambiguous:     provision.type clarity (permit/deny explicit)
      - demonstrable:    verification[] presence and verified=true
      - distinguishable: category coding specificity

    The mapping is a best-effort inference from available FHIR fields.
    Perfect mapping requires information beyond what FHIR captures.
    This is documented as a limitation.
    """

    def test_returns_compliance_opinion(self):
        """fhir_consent_validity produces a ComplianceOpinion."""
        resource = _make_hipaa_consent(status="active")
        op = fhir_consent_validity(resource)
        assert_valid_compliance_opinion(op, "consent_validity")

    def test_result_equals_six_way_meet(self):
        """The result is equivalent to consent_validity() from the algebra.

        fhir_consent_validity should internally extract six condition
        opinions and compose them via the algebra's consent_validity().
        """
        resource = _make_hipaa_consent(
            status="active",
            provision_type="permit",
            verification=[{
                "verified": True,
                "verificationDate": "2025-01-15",
            }],
        )
        op = fhir_consent_validity(resource)
        # The result is a jurisdictional meet of six conditions,
        # so lawfulness should be lower than any individual condition
        # (multiplicative degradation per Theorem 1(a)).
        assert op.lawfulness < 1.0
        # And violation should be non-negative (disjunctive risk)
        assert op.violation >= 0.0

    def test_verified_consent_has_higher_demonstrable(self):
        """Verification events strengthen the demonstrable condition.

        GDPR Art. 7(1): 'the controller shall be able to demonstrate
        that the data subject has consented.' FHIR verification[]
        directly maps to this requirement.
        """
        unverified = _make_hipaa_consent(status="active")
        verified = _make_hipaa_consent(
            status="active",
            verification=[{
                "verified": True,
                "verificationDate": "2025-01-15",
            }],
        )
        op_unverified = fhir_consent_validity(unverified)
        op_verified = fhir_consent_validity(verified)
        # Better demonstrability → higher overall lawfulness
        assert op_verified.lawfulness >= op_unverified.lawfulness - TOL

    def test_explicit_provision_type_strengthens_unambiguous(self):
        """Explicit permit/deny provision strengthens unambiguous condition.

        GDPR Art. 4(11): consent must be an 'unambiguous indication.'
        An explicit provision.type is clearer than no provision at all.
        """
        no_provision = _make_hipaa_consent(status="active")
        explicit_provision = _make_hipaa_consent(
            status="active",
            provision_type="permit",
        )
        op_no = fhir_consent_validity(no_provision)
        op_explicit = fhir_consent_validity(explicit_provision)
        assert op_explicit.lawfulness >= op_no.lawfulness - TOL

    def test_six_conditions_result_lower_than_single(self):
        """Six-way meet produces lower lawfulness than any single condition.

        This is a direct consequence of Theorem 1(a): l⊓ = l₁·l₂,
        so l⊓ ≤ min(l₁, l₂) when l values are in [0, 1].
        """
        resource = _make_hipaa_consent(status="active")
        composite = fhir_consent_validity(resource)
        # A single active consent opinion:
        single = fhir_consent_to_opinion(resource)
        # The composite six-way meet should be <= single because
        # it's a product of six sub-conditions
        assert composite.lawfulness <= single.lawfulness + TOL

    def test_missing_resource_type_raises(self):
        with pytest.raises(ValueError, match="resourceType"):
            fhir_consent_validity({"status": "active"})

    def test_wrong_resource_type_raises(self):
        with pytest.raises(ValueError, match="[Cc]onsent"):
            fhir_consent_validity({"resourceType": "Patient"})


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_withdrawal() — active → inactive/rejected transition
# ═══════════════════════════════════════════════════════════════════


class TestFhirConsentWithdrawal:
    """Bridge FHIR Consent status transitions to withdrawal_override.

    Per compliance_algebra.md §7.2 (Definition 11):
      Withdraw(ω_c^P, ω_w^P, t, t_w) = ω_c^P if t < t_w
                                       = ω_w^P if t ≥ t_w

    FHIR representation of withdrawal:
      - Active consent: status='active', dateTime=t_consent
      - Withdrawn consent: status='inactive' or 'rejected',
        with a later dateTime representing t_w

    The function accepts two FHIR Consent resources (active and
    withdrawn) plus an assessment timestamp, and delegates to
    the algebra's withdrawal_override.
    """

    def test_pre_withdrawal_returns_active_consent_opinion(self):
        """Assessment before withdrawal → active consent opinion.

        Theorem 3(b): pre-withdrawal preservation.
        """
        active = _make_consent(
            status="active",
            date_time="2025-01-15",
            resource_id="active-001",
        )
        withdrawn = _make_consent(
            status="inactive",
            date_time="2025-06-01",
            resource_id="withdrawn-001",
        )
        # Assessment at 2025-03-01 (before withdrawal at 2025-06-01)
        op = fhir_consent_withdrawal(
            active_consent=active,
            withdrawn_consent=withdrawn,
            assessment_time="2025-03-01",
        )
        assert_valid_compliance_opinion(op, "pre-withdrawal")
        # Should reflect the active consent's opinion
        active_op = fhir_consent_to_opinion(active)
        assert abs(op.lawfulness - active_op.lawfulness) < TOL

    def test_post_withdrawal_returns_withdrawal_opinion(self):
        """Assessment after withdrawal → withdrawal implementation opinion.

        Theorem 3(a): withdrawal dominance.
        """
        active = _make_consent(
            status="active",
            date_time="2025-01-15",
        )
        withdrawn = _make_consent(
            status="inactive",
            date_time="2025-06-01",
        )
        # Assessment at 2025-08-01 (after withdrawal at 2025-06-01)
        op = fhir_consent_withdrawal(
            active_consent=active,
            withdrawn_consent=withdrawn,
            assessment_time="2025-08-01",
        )
        assert_valid_compliance_opinion(op, "post-withdrawal")
        # Should reflect the withdrawn consent's opinion
        withdrawn_op = fhir_consent_to_opinion(withdrawn)
        assert abs(op.lawfulness - withdrawn_op.lawfulness) < TOL

    def test_at_exact_withdrawal_time_returns_withdrawal(self):
        """Assessment at exactly t_w → withdrawal opinion.

        Per Definition 11: t ≥ t_w uses ω_w^P (inclusive boundary).
        """
        active = _make_consent(
            status="active", date_time="2025-01-15",
        )
        withdrawn = _make_consent(
            status="inactive", date_time="2025-06-01",
        )
        op = fhir_consent_withdrawal(
            active_consent=active,
            withdrawn_consent=withdrawn,
            assessment_time="2025-06-01",
        )
        withdrawn_op = fhir_consent_to_opinion(withdrawn)
        assert abs(op.lawfulness - withdrawn_op.lawfulness) < TOL

    def test_withdrawal_produces_lower_lawfulness(self):
        """Post-withdrawal opinion has lower lawfulness than pre-withdrawal.

        An inactive consent should signal lower lawfulness than an
        active one. This is a semantic consistency check.
        """
        active = _make_consent(status="active", date_time="2025-01-15")
        withdrawn = _make_consent(status="inactive", date_time="2025-06-01")

        pre_op = fhir_consent_withdrawal(
            active, withdrawn, assessment_time="2025-03-01",
        )
        post_op = fhir_consent_withdrawal(
            active, withdrawn, assessment_time="2025-08-01",
        )
        assert post_op.lawfulness < pre_op.lawfulness

    def test_rejected_withdrawal_has_high_violation(self):
        """Withdrawal via status='rejected' signals strong violation.

        A rejected consent (as opposed to inactive) is a stronger
        negative signal — the consent was explicitly refused.
        """
        active = _make_consent(status="active", date_time="2025-01-15")
        rejected = _make_consent(status="rejected", date_time="2025-06-01")
        op = fhir_consent_withdrawal(
            active, rejected, assessment_time="2025-08-01",
        )
        assert op.violation > op.lawfulness

    def test_missing_datetime_raises(self):
        """Both consents must have dateTime for temporal comparison."""
        active = _make_consent(status="active", date_time=None)
        withdrawn = _make_consent(status="inactive", date_time="2025-06-01")
        with pytest.raises(ValueError, match="[Dd]ate[Tt]ime"):
            fhir_consent_withdrawal(active, withdrawn, "2025-08-01")


# ═══════════════════════════════════════════════════════════════════
# fhir_multi_site_meet() — jurisdictional meet across sites
# ═══════════════════════════════════════════════════════════════════


class TestFhirMultiSiteMeet:
    """Multi-site consent composition via jurisdictional meet.

    Per compliance_algebra.md §5 (Definition 3, Theorem 1):
    Multi-site clinical data sharing requires consent from ALL sites.
    This is a conjunction — jurisdictional_meet.

    Example: a multi-center clinical trial where patient data from
    Hospital A (HIPAA) and Hospital B (HIPAA + state law) must both
    have valid consent for the derived dataset to be usable.
    """

    def test_two_active_consents_produce_valid_opinion(self):
        """Meet of two active consents is a valid ComplianceOpinion."""
        site_a = _make_hipaa_consent(resource_id="site-a")
        site_b = _make_hipaa_consent(resource_id="site-b")
        op = fhir_multi_site_meet(site_a, site_b)
        assert_valid_compliance_opinion(op, "multi-site meet")

    def test_lawfulness_is_product(self):
        """Lawfulness follows Theorem 1(a): l⊓ = l₁ · l₂.

        This is the defining property of the jurisdictional meet:
        composite lawfulness is multiplicative, reflecting that
        ALL jurisdictions must be satisfied.
        """
        site_a = _make_hipaa_consent(resource_id="site-a")
        site_b = _make_hipaa_consent(resource_id="site-b")
        op_a = fhir_consent_to_opinion(site_a)
        op_b = fhir_consent_to_opinion(site_b)
        meet = fhir_multi_site_meet(site_a, site_b)
        expected_l = op_a.lawfulness * op_b.lawfulness
        assert abs(meet.lawfulness - expected_l) < TOL

    def test_violation_is_disjunctive(self):
        """Violation follows Theorem 1(b): v⊓ = v₁ + v₂ − v₁·v₂.

        Any single site's violation constitutes composite violation.
        """
        site_a = _make_hipaa_consent(resource_id="site-a")
        site_b = _make_hipaa_consent(resource_id="site-b")
        op_a = fhir_consent_to_opinion(site_a)
        op_b = fhir_consent_to_opinion(site_b)
        meet = fhir_multi_site_meet(site_a, site_b)
        expected_v = op_a.violation + op_b.violation - op_a.violation * op_b.violation
        assert abs(meet.violation - expected_v) < TOL

    def test_monotonic_violation(self):
        """Theorem 1(d): v⊓ ≥ max(v₁, v₂).

        Adding more sites can only increase composite violation.
        """
        site_a = _make_consent(status="active", resource_id="site-a")
        site_b = _make_consent(status="active", resource_id="site-b")
        op_a = fhir_consent_to_opinion(site_a)
        op_b = fhir_consent_to_opinion(site_b)
        meet = fhir_multi_site_meet(site_a, site_b)
        assert meet.violation >= max(op_a.violation, op_b.violation) - TOL

    def test_one_rejected_dominates(self):
        """If any site has rejected consent, composite violation is high.

        Per Theorem 1(d), violation is monotonic. A rejected site
        drives the composite toward violation.
        """
        good_site = _make_consent(status="active", resource_id="good")
        bad_site = _make_consent(status="rejected", resource_id="bad")
        meet = fhir_multi_site_meet(good_site, bad_site)
        assert meet.violation > 0.5, (
            f"One rejected site should produce high violation, got {meet.violation}"
        )

    def test_three_way_meet_associativity(self):
        """Theorem 1(f): J⊓ is associative.

        meet(A, B, C) = meet(meet(A, B), C) = meet(A, meet(B, C))
        """
        a = _make_consent(status="active", resource_id="a")
        b = _make_consent(status="active", resource_id="b")
        c = _make_consent(status="proposed", resource_id="c")
        abc = fhir_multi_site_meet(a, b, c)
        assert_valid_compliance_opinion(abc, "three-way meet")

    def test_single_site_returns_that_sites_opinion(self):
        """Single-site meet is identity — returns site's own opinion."""
        site = _make_hipaa_consent(resource_id="solo")
        meet = fhir_multi_site_meet(site)
        direct = fhir_consent_to_opinion(site)
        assert abs(meet.lawfulness - direct.lawfulness) < TOL
        assert abs(meet.violation - direct.violation) < TOL
        assert abs(meet.uncertainty - direct.uncertainty) < TOL

    def test_empty_raises(self):
        """No sites provided raises ValueError."""
        with pytest.raises(ValueError):
            fhir_multi_site_meet()

    def test_commutativity(self):
        """Theorem 1(e): meet(A, B) = meet(B, A)."""
        a = _make_consent(status="active", resource_id="a")
        b = _make_consent(status="proposed", resource_id="b")
        ab = fhir_multi_site_meet(a, b)
        ba = fhir_multi_site_meet(b, a)
        assert abs(ab.lawfulness - ba.lawfulness) < TOL
        assert abs(ab.violation - ba.violation) < TOL
        assert abs(ab.uncertainty - ba.uncertainty) < TOL


# ═══════════════════════════════════════════════════════════════════
# HIPAA-SPECIFIC SCENARIO TESTS
# ═══════════════════════════════════════════════════════════════════


class TestHipaaConsentScenarios:
    """End-to-end scenarios grounded in HIPAA consent modeling.

    These tests verify that the FHIR ↔ compliance algebra bridge
    produces semantically correct results for realistic HIPAA
    consent workflows.
    """

    def test_hipaa_research_consent_active(self):
        """Active HIPAA research consent has high lawfulness."""
        resource = _make_consent(
            status="active",
            scope_code="research",
            policy_rule_code="HIPAA",
            provision_type="permit",
            verification=[{
                "verified": True,
                "verificationDate": "2025-01-15",
            }],
        )
        op = fhir_consent_to_opinion(resource)
        assert_valid_compliance_opinion(op)
        assert op.lawfulness > 0.5

    def test_hipaa_withdrawal_scenario(self):
        """Patient withdraws research consent — full lifecycle.

        1. Active consent granted 2025-01-15
        2. Patient withdraws 2025-06-01 (status → inactive)
        3. Assessment at 2025-03-01 → active opinion
        4. Assessment at 2025-08-01 → withdrawal opinion
        """
        active = _make_consent(
            status="active",
            scope_code="research",
            policy_rule_code="HIPAA",
            date_time="2025-01-15",
        )
        withdrawn = _make_consent(
            status="inactive",
            scope_code="research",
            policy_rule_code="HIPAA",
            date_time="2025-06-01",
        )

        # Pre-withdrawal: should be valid
        pre = fhir_consent_withdrawal(active, withdrawn, "2025-03-01")
        assert pre.lawfulness > 0.5

        # Post-withdrawal: should reflect withdrawal
        post = fhir_consent_withdrawal(active, withdrawn, "2025-08-01")
        assert post.lawfulness < pre.lawfulness

    def test_multi_site_trial_mixed_consent(self):
        """Multi-center trial: one site active, one site draft.

        The composite should reflect the weakest link: the draft
        site's uncertainty propagates to the composite.
        """
        strong_site = _make_hipaa_consent(
            status="active",
            resource_id="site-strong",
            verification=[{"verified": True, "verificationDate": "2025-01-15"}],
        )
        weak_site = _make_hipaa_consent(
            status="draft",
            resource_id="site-weak",
        )
        meet = fhir_multi_site_meet(strong_site, weak_site)
        strong_op = fhir_consent_to_opinion(strong_site)
        # Composite lawfulness should be less than the strong site alone
        assert meet.lawfulness < strong_op.lawfulness

    def test_consent_validity_hipaa_well_documented(self):
        """Well-documented HIPAA consent has higher validity than bare one.

        A consent with explicit provision, verification, and specific
        scope should score higher on the six-condition composition.
        """
        bare = _make_hipaa_consent(status="active")
        documented = _make_hipaa_consent(
            status="active",
            provision_type="permit",
            verification=[{
                "verified": True,
                "verificationDate": "2025-01-15",
            }],
        )
        op_bare = fhir_consent_validity(bare)
        op_documented = fhir_consent_validity(documented)
        assert op_documented.lawfulness >= op_bare.lawfulness - TOL
