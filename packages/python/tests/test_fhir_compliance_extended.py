"""Extended tests for FHIR Consent ↔ Compliance Algebra bridge.

Phase 4b: consent expiry, regulatory change, and comprehensive
cross/multi-jurisdictional scenario coverage.

New functions tested:
  fhir_consent_expiry             — provision.period.end → expiry_trigger
  fhir_consent_regulatory_change  — policyRule transition → regulatory_change_trigger

Comprehensive scenario coverage:
  - Cross-jurisdictional: HIPAA + GDPR + state law
  - Operator composition chains (§10)
  - N-way multi-center trials (4+ sites)
  - Purpose-indexed consent across jurisdictions
  - Withdrawal within multi-site composites
  - Edge cases: all-rejected, all-draft, conflicting evidence

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
    compliance_propagation,
    expiry_trigger,
    regulatory_change_trigger,
)
from jsonld_ex.fhir_interop import (
    fhir_consent_to_opinion,
    opinion_to_fhir_consent,
    fhir_consent_validity,
    fhir_consent_withdrawal,
    fhir_multi_site_meet,
    fhir_consent_expiry,
    fhir_consent_regulatory_change,
    CONSENT_STATUS_PROBABILITY,
)


# ── Tolerance ─────────────────────────────────────────────────────
TOL = 1e-9


# ── Helpers ───────────────────────────────────────────────────────
def assert_valid_compliance_opinion(op, msg=""):
    prefix = f"{msg}: " if msg else ""
    assert isinstance(op, ComplianceOpinion), (
        f"{prefix}expected ComplianceOpinion, got {type(op).__name__}"
    )
    assert op.lawfulness >= -TOL, f"{prefix}l < 0: {op.lawfulness}"
    assert op.violation >= -TOL, f"{prefix}v < 0: {op.violation}"
    assert op.uncertainty >= -TOL, f"{prefix}u < 0: {op.uncertainty}"
    total = op.lawfulness + op.violation + op.uncertainty
    assert abs(total - 1.0) < TOL, f"{prefix}l+v+u={total}"


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
    provision_purpose=None,
    verification=None,
    extensions=None,
    resource_id="consent-001",
):
    """Build a minimal valid FHIR R4 Consent resource."""
    resource = {
        "resourceType": "Consent",
        "id": resource_id,
        "status": status,
        "scope": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                "code": scope_code,
            }]
        },
        "category": [{
            "coding": [{
                "system": "http://loinc.org",
                "code": category_code,
            }]
        }],
        "patient": {"reference": patient_ref},
    }
    if date_time is not None:
        resource["dateTime"] = date_time
    if policy_uri is not None:
        resource["policy"] = [{"uri": policy_uri}]
    if policy_rule_code is not None:
        resource["policyRule"] = {"coding": [{"code": policy_rule_code}]}
    if provision_type is not None or provision_period is not None or provision_purpose is not None:
        provision = resource.setdefault("provision", {})
        if provision_type is not None:
            provision["type"] = provision_type
        if provision_period is not None:
            provision["period"] = provision_period
        if provision_purpose is not None:
            provision["purpose"] = provision_purpose
    if verification is not None:
        resource["verification"] = verification
    if extensions is not None:
        resource["extension"] = extensions
    return resource


def _make_jurisdictional_consent(jurisdiction, **kwargs):
    """Build consent tagged with a specific jurisdiction/policy rule."""
    defaults = {
        "scope_code": "patient-privacy",
        "policy_rule_code": jurisdiction,
    }
    defaults.update(kwargs)
    return _make_consent(**defaults)


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_expiry() — provision.period.end → expiry_trigger
# ═══════════════════════════════════════════════════════════════════


class TestFhirConsentExpiry:
    """Bridge FHIR Consent provision.period to expiry_trigger.

    Per compliance_algebra.md §8.2 (Definition 12):
    At trigger time t_T, lawfulness transfers to violation (not
    uncertainty). An expired consent is a known fact.

    FHIR mapping:
      provision.period.end → trigger_time (t_T)
      assessment_time → evaluation time (t)

    Theorem 4 properties:
      (b) Constraint preservation: l' + v' + u' = 1
      (c) Monotonicity: l' ≤ l, v' ≥ v, u' = u
      (d) Hard expiry: γ=0 → l'=0, v'=v+l
    """

    def test_pre_expiry_returns_unchanged_opinion(self):
        """Assessment before provision.period.end → original opinion.

        The consent has not yet expired, so no l→v transfer occurs.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        op = fhir_consent_expiry(resource, assessment_time="2025-06-15")
        active_op = fhir_consent_to_opinion(resource)
        assert abs(op.lawfulness - active_op.lawfulness) < TOL
        assert abs(op.violation - active_op.violation) < TOL

    def test_post_expiry_transfers_lawfulness_to_violation(self):
        """Assessment after provision.period.end → l transferred to v.

        Per Definition 12: expired deadline is a known fact, so
        lawfulness moves to violation, not uncertainty.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        pre = fhir_consent_expiry(resource, assessment_time="2025-06-15")
        post = fhir_consent_expiry(resource, assessment_time="2026-03-01")
        assert post.lawfulness < pre.lawfulness
        assert post.violation > pre.violation

    def test_post_expiry_uncertainty_unchanged(self):
        """Theorem 4(c): u' = u after expiry trigger.

        Expiry is a known fact, so uncertainty does not change.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        pre = fhir_consent_expiry(resource, assessment_time="2025-06-15")
        post = fhir_consent_expiry(resource, assessment_time="2026-03-01")
        assert abs(post.uncertainty - pre.uncertainty) < TOL

    def test_hard_expiry_default(self):
        """Default residual_factor=0 → hard expiry (l'=0, v'=v+l).

        Theorem 4(d): complete transfer of lawfulness to violation.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        post = fhir_consent_expiry(resource, assessment_time="2026-03-01")
        assert post.lawfulness < TOL, (
            f"Hard expiry should zero lawfulness, got {post.lawfulness}"
        )

    def test_soft_expiry_with_residual(self):
        """residual_factor > 0 retains some lawfulness post-expiry.

        Models regulations that allow a grace period (e.g., data
        retention after consent expiry for audit purposes).
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        post = fhir_consent_expiry(
            resource, assessment_time="2026-03-01", residual_factor=0.3,
        )
        pre = fhir_consent_expiry(resource, assessment_time="2025-06-15")
        # Some lawfulness retained
        assert post.lawfulness > TOL
        # But less than pre-expiry
        assert post.lawfulness < pre.lawfulness

    def test_at_exact_expiry_date(self):
        """Assessment at exactly period.end → expired (inclusive).

        Consistent with withdrawal_override: t ≥ t_w uses post-trigger.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        op = fhir_consent_expiry(resource, assessment_time="2025-12-31")
        pre = fhir_consent_expiry(resource, assessment_time="2025-12-30")
        assert op.lawfulness < pre.lawfulness

    def test_constraint_preservation(self):
        """Theorem 4(b): l' + v' + u' = 1 always holds."""
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01", "end": "2025-12-31"},
        )
        for date in ("2025-06-15", "2025-12-31", "2026-06-01"):
            op = fhir_consent_expiry(resource, assessment_time=date)
            assert_valid_compliance_opinion(op, f"expiry at {date}")

    def test_no_period_returns_unchanged(self):
        """Consent without provision.period → no expiry, return as-is.

        Not all consents have an expiration date. In this case the
        function should return the consent's opinion unchanged.
        """
        resource = _make_consent(status="active")
        op = fhir_consent_expiry(resource, assessment_time="2030-01-01")
        active_op = fhir_consent_to_opinion(resource)
        assert abs(op.lawfulness - active_op.lawfulness) < TOL

    def test_no_period_end_returns_unchanged(self):
        """provision.period with start but no end → no expiry."""
        resource = _make_consent(
            status="active",
            provision_period={"start": "2025-01-01"},
        )
        op = fhir_consent_expiry(resource, assessment_time="2030-01-01")
        active_op = fhir_consent_to_opinion(resource)
        assert abs(op.lawfulness - active_op.lawfulness) < TOL

    def test_missing_resource_type_raises(self):
        with pytest.raises(ValueError, match="resourceType"):
            fhir_consent_expiry({"status": "active"}, "2025-06-01")

    def test_wrong_resource_type_raises(self):
        with pytest.raises(ValueError, match="[Cc]onsent"):
            fhir_consent_expiry(
                {"resourceType": "Patient"}, "2025-06-01",
            )


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_regulatory_change() — policyRule transition
# ═══════════════════════════════════════════════════════════════════


class TestFhirConsentRegulatoryChange:
    """Bridge FHIR Consent policyRule transition to regulatory_change_trigger.

    Per compliance_algebra.md §8.2 (Definition 14):
    At trigger time, the compliance opinion is replaced by a new
    assessment reflecting the changed legal framework.

    FHIR representation:
      - old_consent: original consent under previous regulation
      - new_consent: replacement consent under new regulation
      - The new consent's dateTime is the regulatory change time (t_T)

    Theorem 4(e): trigger ordering is non-commutative — the order
    of regulatory events matters.
    """

    def test_pre_change_returns_old_consent_opinion(self):
        """Assessment before regulatory change → old consent applies."""
        old = _make_jurisdictional_consent(
            "HIPAA-pre-2025",
            date_time="2024-01-01",
            resource_id="old-reg",
        )
        new = _make_jurisdictional_consent(
            "HIPAA-2025-amended",
            date_time="2025-07-01",
            resource_id="new-reg",
        )
        op = fhir_consent_regulatory_change(
            old_consent=old,
            new_consent=new,
            assessment_time="2025-03-01",
        )
        old_op = fhir_consent_to_opinion(old)
        assert abs(op.lawfulness - old_op.lawfulness) < TOL

    def test_post_change_returns_new_consent_opinion(self):
        """Assessment after regulatory change → new consent applies.

        The proposition under assessment changes: "compliance with
        old regulation" → "compliance with new regulation."
        """
        old = _make_jurisdictional_consent(
            "HIPAA-pre-2025",
            date_time="2024-01-01",
            resource_id="old-reg",
        )
        new = _make_jurisdictional_consent(
            "HIPAA-2025-amended",
            date_time="2025-07-01",
            resource_id="new-reg",
        )
        op = fhir_consent_regulatory_change(
            old_consent=old,
            new_consent=new,
            assessment_time="2025-09-01",
        )
        new_op = fhir_consent_to_opinion(new)
        assert abs(op.lawfulness - new_op.lawfulness) < TOL

    def test_at_exact_change_time(self):
        """Assessment at exactly t_T → new consent applies (inclusive)."""
        old = _make_jurisdictional_consent(
            "GDPR-pre", date_time="2023-01-01", resource_id="old",
        )
        new = _make_jurisdictional_consent(
            "GDPR-amended", date_time="2025-01-01", resource_id="new",
        )
        op = fhir_consent_regulatory_change(old, new, "2025-01-01")
        new_op = fhir_consent_to_opinion(new)
        assert abs(op.lawfulness - new_op.lawfulness) < TOL

    def test_draft_new_consent_produces_high_uncertainty(self):
        """New regulation with draft consent → high uncertainty.

        A regulatory change where the replacement consent is still
        in draft means we lack evidence under the new framework.
        """
        old = _make_jurisdictional_consent(
            "HIPAA", status="active",
            date_time="2024-01-01", resource_id="old",
        )
        new = _make_jurisdictional_consent(
            "HIPAA-amended", status="draft",
            date_time="2025-07-01", resource_id="new",
        )
        op = fhir_consent_regulatory_change(old, new, "2025-09-01")
        assert op.uncertainty > 0.25, (
            "Draft replacement consent should produce high uncertainty"
        )

    def test_constraint_preservation(self):
        """l + v + u = 1 before and after regulatory change."""
        old = _make_jurisdictional_consent(
            "GDPR", date_time="2024-01-01", resource_id="old",
        )
        new = _make_jurisdictional_consent(
            "GDPR-v2", date_time="2025-06-01", resource_id="new",
        )
        for date in ("2025-01-01", "2025-06-01", "2025-12-01"):
            op = fhir_consent_regulatory_change(old, new, date)
            assert_valid_compliance_opinion(op, f"reg-change at {date}")

    def test_missing_datetime_raises(self):
        """Both consents must have dateTime."""
        old = _make_jurisdictional_consent(
            "HIPAA", date_time=None, resource_id="old",
        )
        new = _make_jurisdictional_consent(
            "HIPAA-v2", date_time="2025-06-01", resource_id="new",
        )
        with pytest.raises(ValueError, match="[Dd]ate[Tt]ime"):
            fhir_consent_regulatory_change(old, new, "2025-09-01")


# ═══════════════════════════════════════════════════════════════════
# CROSS-JURISDICTIONAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════


class TestCrossJurisdictionalScenarios:
    """Realistic cross-jurisdictional consent composition.

    These tests model real-world multi-regulation scenarios where
    data processing must satisfy multiple legal frameworks simultaneously.
    Per §5 (Definition 3): composite compliance = J⊓ across jurisdictions.
    """

    def test_hipaa_plus_gdpr_meet(self):
        """HIPAA + GDPR: both must be satisfied for transatlantic data.

        A US hospital sharing data with an EU research partner must
        satisfy both HIPAA and GDPR. The composite lawfulness is
        multiplicative (Theorem 1(a)).
        """
        hipaa = _make_jurisdictional_consent(
            "HIPAA", resource_id="us-site",
        )
        gdpr = _make_jurisdictional_consent(
            "GDPR", resource_id="eu-site",
        )
        meet = fhir_multi_site_meet(hipaa, gdpr)
        assert_valid_compliance_opinion(meet, "HIPAA+GDPR")

        # Multiplicative: composite l < min of individual l values
        h_op = fhir_consent_to_opinion(hipaa)
        g_op = fhir_consent_to_opinion(gdpr)
        assert meet.lawfulness <= min(h_op.lawfulness, g_op.lawfulness) + TOL

    def test_hipaa_plus_gdpr_plus_state_law(self):
        """HIPAA + GDPR + California CCPA: three-way jurisdiction.

        Multi-regulation compliance: all three must be satisfied.
        Lawfulness degrades multiplicatively with each added jurisdiction.
        """
        hipaa = _make_jurisdictional_consent(
            "HIPAA", resource_id="hipaa",
        )
        gdpr = _make_jurisdictional_consent(
            "GDPR", resource_id="gdpr",
        )
        ccpa = _make_jurisdictional_consent(
            "CCPA", resource_id="ccpa",
        )
        two_way = fhir_multi_site_meet(hipaa, gdpr)
        three_way = fhir_multi_site_meet(hipaa, gdpr, ccpa)

        # Adding a jurisdiction can only decrease lawfulness
        assert three_way.lawfulness <= two_way.lawfulness + TOL
        # And can only increase violation (Theorem 1(d))
        assert three_way.violation >= two_way.violation - TOL

    def test_strong_and_weak_jurisdiction(self):
        """One strong + one weak jurisdiction → weak dominates.

        If one jurisdiction's consent is in draft (high u) while
        the other is active (high l), the composite should reflect
        the weakness of the uncertain jurisdiction.
        """
        strong = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="strong",
            verification=[{"verified": True, "verificationDate": "2025-01-15"}],
        )
        weak = _make_jurisdictional_consent(
            "GDPR", status="draft", resource_id="weak",
        )
        meet = fhir_multi_site_meet(strong, weak)
        strong_op = fhir_consent_to_opinion(strong)

        # Composite should be substantially weaker than the strong site
        assert meet.lawfulness < strong_op.lawfulness * 0.9

    def test_one_rejected_jurisdiction_dominates(self):
        """If any jurisdiction is rejected → high composite violation.

        Per Theorem 1(d): v⊓ ≥ max(v₁, v₂). A rejected consent
        in any jurisdiction poisons the composite.
        """
        active = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="ok",
        )
        rejected = _make_jurisdictional_consent(
            "GDPR", status="rejected", resource_id="rejected",
        )
        meet = fhir_multi_site_meet(active, rejected)
        rejected_op = fhir_consent_to_opinion(rejected)

        assert meet.violation >= rejected_op.violation - TOL
        assert meet.violation > meet.lawfulness

    def test_commutativity_across_jurisdictions(self):
        """Theorem 1(e): order of jurisdictions doesn't matter.

        meet(HIPAA, GDPR) = meet(GDPR, HIPAA)
        """
        hipaa = _make_jurisdictional_consent("HIPAA", resource_id="h")
        gdpr = _make_jurisdictional_consent("GDPR", resource_id="g")
        hg = fhir_multi_site_meet(hipaa, gdpr)
        gh = fhir_multi_site_meet(gdpr, hipaa)
        assert abs(hg.lawfulness - gh.lawfulness) < TOL
        assert abs(hg.violation - gh.violation) < TOL
        assert abs(hg.uncertainty - gh.uncertainty) < TOL

    def test_associativity_across_jurisdictions(self):
        """Theorem 1(f): grouping of jurisdictions doesn't matter.

        meet(HIPAA, meet(GDPR, CCPA)) = meet(meet(HIPAA, GDPR), CCPA)
        """
        h = _make_jurisdictional_consent("HIPAA", resource_id="h")
        g = _make_jurisdictional_consent("GDPR", resource_id="g")
        c = _make_jurisdictional_consent("CCPA", resource_id="c")

        # Left grouping: meet(meet(H, G), C)
        hg = fhir_multi_site_meet(h, g)
        hg_op = fhir_consent_to_opinion(
            opinion_to_fhir_consent(hg, resource_id="hg")
        )
        # We need to compose via algebra directly to avoid round-trip noise
        h_op = fhir_consent_to_opinion(h)
        g_op = fhir_consent_to_opinion(g)
        c_op = fhir_consent_to_opinion(c)

        left = jurisdictional_meet(jurisdictional_meet(h_op, g_op), c_op)
        right = jurisdictional_meet(h_op, jurisdictional_meet(g_op, c_op))
        flat = jurisdictional_meet(h_op, g_op, c_op)

        assert abs(left.lawfulness - right.lawfulness) < TOL
        assert abs(left.lawfulness - flat.lawfulness) < TOL
        assert abs(left.violation - flat.violation) < TOL


# ═══════════════════════════════════════════════════════════════════
# MULTI-CENTER CLINICAL TRIAL SCENARIOS
# ═══════════════════════════════════════════════════════════════════


class TestMultiCenterTrialScenarios:
    """Realistic multi-center clinical trial consent scenarios.

    Models a multi-site trial where patient data from N hospitals
    must all have valid consent for the aggregated dataset to be
    usable in research.
    """

    def test_four_site_trial_all_active(self):
        """4-site trial with all active consents → valid but degraded.

        Each additional site multiplicatively degrades lawfulness
        (Theorem 1(a): l⊓ = ∏lᵢ). Even with all sites active,
        the composite is less certain than any individual site.
        """
        sites = [
            _make_jurisdictional_consent(
                "HIPAA", status="active",
                resource_id=f"site-{i}",
            )
            for i in range(4)
        ]
        meet = fhir_multi_site_meet(*sites)
        assert_valid_compliance_opinion(meet, "4-site all-active")

        single_op = fhir_consent_to_opinion(sites[0])
        assert meet.lawfulness < single_op.lawfulness

    def test_four_site_trial_one_draft(self):
        """4-site trial with one draft → composite weakened significantly.

        The draft site has high uncertainty, which propagates to
        the composite via multiplicative lawfulness degradation.
        """
        sites = [
            _make_jurisdictional_consent(
                "HIPAA", status="active",
                resource_id=f"site-{i}",
            )
            for i in range(3)
        ]
        draft_site = _make_jurisdictional_consent(
            "HIPAA", status="draft", resource_id="site-3",
        )
        sites.append(draft_site)

        all_active = fhir_multi_site_meet(*sites[:3])
        with_draft = fhir_multi_site_meet(*sites)

        assert with_draft.lawfulness < all_active.lawfulness

    def test_five_site_trial_monotonic_violation(self):
        """Adding a 5th site can only increase composite violation.

        Theorem 1(d): v⊓ ≥ max(v₁, ..., vₙ).
        """
        sites_4 = [
            _make_jurisdictional_consent(
                "HIPAA", status="active", resource_id=f"s{i}",
            )
            for i in range(4)
        ]
        site_5 = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="s4",
        )
        meet_4 = fhir_multi_site_meet(*sites_4)
        meet_5 = fhir_multi_site_meet(*sites_4, site_5)

        assert meet_5.violation >= meet_4.violation - TOL

    def test_mixed_jurisdiction_trial(self):
        """Multi-center trial spanning US (HIPAA) and EU (GDPR).

        Real-world scenario: international clinical trial where
        sites operate under different regulatory frameworks.
        """
        us_sites = [
            _make_jurisdictional_consent(
                "HIPAA", status="active", resource_id=f"us-{i}",
            )
            for i in range(2)
        ]
        eu_sites = [
            _make_jurisdictional_consent(
                "GDPR", status="active", resource_id=f"eu-{i}",
            )
            for i in range(2)
        ]
        all_sites = us_sites + eu_sites
        meet = fhir_multi_site_meet(*all_sites)
        assert_valid_compliance_opinion(meet, "mixed HIPAA+GDPR trial")
        # 4 sites → significant multiplicative degradation
        single = fhir_consent_to_opinion(us_sites[0])
        assert meet.lawfulness < single.lawfulness


# ═══════════════════════════════════════════════════════════════════
# OPERATOR COMPOSITION CHAINS (§10)
# ═══════════════════════════════════════════════════════════════════


class TestOperatorCompositionChains:
    """Test operator composition per compliance_algebra.md §10.

    The five operators compose to model complete compliance lifecycles:
      1. Consent Assessment → feeds into Compliance Propagation
      2. Consent Withdrawal → may trigger Erasure
      3. Temporal Decay applies to all operators
      4. Expiry can invalidate previously valid consent
    """

    def test_consent_to_propagation_chain(self):
        """§10: Consent Assessment output feeds Compliance Propagation.

        Prop(τ, π, EffConsent(P, t)) — the effective consent becomes
        the source compliance for downstream derivation.
        """
        consent_resource = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="src",
            verification=[{"verified": True, "verificationDate": "2025-01-15"}],
        )
        consent_op = fhir_consent_to_opinion(consent_resource)

        # Derivation step: trusted process, compatible purpose
        trust = ComplianceOpinion.create(0.90, 0.02, 0.08, 0.5)
        purpose = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)

        derived = compliance_propagation(consent_op, trust, purpose)
        assert_valid_compliance_opinion(derived, "consent→propagation")
        # Derived must be weaker than source consent
        assert derived.lawfulness < consent_op.lawfulness

    def test_multi_site_consent_to_propagation(self):
        """Multi-site meet feeds into propagation as source.

        Real workflow: aggregate consent across sites, then assess
        derived dataset compliance.
        """
        site_a = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="a",
        )
        site_b = _make_jurisdictional_consent(
            "GDPR", status="active", resource_id="b",
        )
        composite_consent = fhir_multi_site_meet(site_a, site_b)

        trust = ComplianceOpinion.create(0.90, 0.02, 0.08, 0.5)
        purpose = ComplianceOpinion.create(0.80, 0.05, 0.15, 0.5)

        derived = compliance_propagation(composite_consent, trust, purpose)
        assert_valid_compliance_opinion(derived, "multi-site→propagation")
        assert derived.lawfulness < composite_consent.lawfulness

    def test_consent_validity_to_multi_site_meet(self):
        """fhir_consent_validity output composes with multi-site meet.

        Each site's six-condition validity is computed, then composed
        via jurisdictional meet across sites.
        """
        site_a = _make_jurisdictional_consent(
            "HIPAA", status="active", resource_id="a",
            provision_type="permit",
            verification=[{"verified": True, "verificationDate": "2025-01-15"}],
        )
        site_b = _make_jurisdictional_consent(
            "GDPR", status="active", resource_id="b",
            provision_type="permit",
        )
        validity_a = fhir_consent_validity(site_a)
        validity_b = fhir_consent_validity(site_b)

        composite = jurisdictional_meet(validity_a, validity_b)
        assert_valid_compliance_opinion(composite, "validity→meet")
        assert composite.lawfulness <= min(
            validity_a.lawfulness, validity_b.lawfulness
        ) + TOL

    def test_expiry_then_multi_site_meet(self):
        """One site's consent expires, then multi-site meet.

        If site A's consent has expired, the composite should reflect
        the expired state even if site B is still active.
        """
        site_a = _make_consent(
            status="active",
            provision_period={"start": "2024-01-01", "end": "2025-06-30"},
            resource_id="a",
        )
        site_b = _make_consent(
            status="active",
            resource_id="b",
        )
        # Site A expired as of 2025-09-01
        expired_a = fhir_consent_expiry(site_a, assessment_time="2025-09-01")
        active_b = fhir_consent_to_opinion(site_b)

        composite = jurisdictional_meet(expired_a, active_b)
        assert_valid_compliance_opinion(composite, "expired+active meet")
        # Expired site should drag composite lawfulness down
        assert composite.lawfulness < active_b.lawfulness
        # Expired site's violation propagates
        assert composite.violation >= expired_a.violation - TOL

    def test_withdrawal_then_multi_site_meet(self):
        """One site withdraws consent, then multi-site meet.

        Post-withdrawal, the withdrawn site's opinion is used in
        the composite. The result should reflect the weakness.
        """
        site_a_active = _make_consent(
            status="active", date_time="2025-01-01", resource_id="a-active",
        )
        site_a_withdrawn = _make_consent(
            status="inactive", date_time="2025-06-01", resource_id="a-withdrawn",
        )
        site_b = _make_consent(
            status="active", resource_id="b",
        )

        # Post-withdrawal opinion for site A
        withdrawn_op = fhir_consent_withdrawal(
            site_a_active, site_a_withdrawn, assessment_time="2025-09-01",
        )
        active_b_op = fhir_consent_to_opinion(site_b)

        composite = jurisdictional_meet(withdrawn_op, active_b_op)
        assert_valid_compliance_opinion(composite, "withdrawn+active meet")
        # Withdrawal weakens composite
        assert composite.lawfulness < active_b_op.lawfulness

    def test_full_lifecycle_consent_expiry_erasure(self):
        """Complete lifecycle: consent → expiry → compliance update.

        §10: Temporal Decay Applies to All Operators.
        Model a consent that expires and needs compliance reassessment.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2024-01-01", "end": "2025-12-31"},
            resource_id="lifecycle",
        )

        # Phase 1: valid consent
        valid = fhir_consent_expiry(resource, assessment_time="2025-06-01")
        assert valid.lawfulness > 0.5

        # Phase 2: consent expires
        expired = fhir_consent_expiry(resource, assessment_time="2026-03-01")
        assert expired.lawfulness < valid.lawfulness

        # Phase 3: compliance update (meet with processing assessment)
        processing_op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        updated = jurisdictional_meet(expired, processing_op)
        assert_valid_compliance_opinion(updated, "lifecycle final")
        # Expired consent drags everything down
        assert updated.lawfulness < processing_op.lawfulness


# ═══════════════════════════════════════════════════════════════════
# PURPOSE-INDEXED CONSENT SCENARIOS
# ═══════════════════════════════════════════════════════════════════


class TestPurposeIndexedConsent:
    """Purpose-indexed consent across jurisdictions.

    Per compliance_algebra.md §7.2: consent opinions are indexed by
    purpose. A data subject may have valid consent for purpose P₁
    but not P₂ (GDPR Art. 6(1)(a), Recital 32).

    FHIR mapping: provision.purpose codes distinguish purposes.
    """

    def test_different_purposes_independent_opinions(self):
        """Consent for treatment vs research → independent opinions.

        Theorem 3(c): non-interference. Purpose P₁ consent does
        not affect purpose P₂ assessment.
        """
        treatment = _make_consent(
            status="active",
            scope_code="treatment",
            provision_purpose=[{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
                "code": "TREAT",
            }],
            resource_id="treatment",
        )
        research = _make_consent(
            status="draft",
            scope_code="research",
            provision_purpose=[{
                "system": "http://terminology.hl7.org/CodeSystem/v3-ActReason",
                "code": "HRESCH",
            }],
            resource_id="research",
        )
        treat_op = fhir_consent_to_opinion(treatment)
        research_op = fhir_consent_to_opinion(research)

        # Treatment (active) should have higher lawfulness than research (draft)
        assert treat_op.lawfulness > research_op.lawfulness

    def test_purpose_specific_multi_site_meet(self):
        """Multi-site meet per-purpose: each purpose assessed independently.

        If site A has research consent and site B has research consent,
        the composite research consent is their meet. Treatment consent
        at each site would be a separate assessment.
        """
        site_a_research = _make_consent(
            status="active",
            scope_code="research",
            resource_id="a-research",
        )
        site_b_research = _make_consent(
            status="active",
            scope_code="research",
            resource_id="b-research",
        )
        research_meet = fhir_multi_site_meet(site_a_research, site_b_research)
        assert_valid_compliance_opinion(research_meet, "per-purpose meet")

    def test_withdrawal_purpose_specific(self):
        """Withdrawal for research does not affect treatment consent.

        Theorem 3(c): non-interference between purposes.
        """
        research_active = _make_consent(
            status="active", scope_code="research",
            date_time="2025-01-01", resource_id="research-active",
        )
        research_withdrawn = _make_consent(
            status="inactive", scope_code="research",
            date_time="2025-06-01", resource_id="research-withdrawn",
        )
        treatment_active = _make_consent(
            status="active", scope_code="treatment",
            date_time="2025-01-01", resource_id="treatment-active",
        )

        # Research consent withdrawn
        research_op = fhir_consent_withdrawal(
            research_active, research_withdrawn, "2025-09-01",
        )
        # Treatment consent unaffected
        treatment_op = fhir_consent_to_opinion(treatment_active)

        # Research should be weaker than treatment
        assert research_op.lawfulness < treatment_op.lawfulness


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestComplianceBridgeEdgeCases:
    """Edge cases for the FHIR compliance bridge."""

    def test_all_sites_rejected(self):
        """All sites rejected → very high composite violation."""
        sites = [
            _make_consent(status="rejected", resource_id=f"r-{i}")
            for i in range(3)
        ]
        meet = fhir_multi_site_meet(*sites)
        assert_valid_compliance_opinion(meet, "all-rejected")
        assert meet.violation > 0.9

    def test_all_sites_draft(self):
        """All sites draft → very high composite uncertainty.

        Draft consents are pre-decisional. The composite should
        reflect pervasive uncertainty, not violation.
        """
        sites = [
            _make_consent(status="draft", resource_id=f"d-{i}")
            for i in range(3)
        ]
        meet = fhir_multi_site_meet(*sites)
        assert_valid_compliance_opinion(meet, "all-draft")
        # Lawfulness should be very low (product of ~0.5 values)
        assert meet.lawfulness < 0.2

    def test_conflicting_evidence_across_sites(self):
        """Site A active, site B rejected → conflict in composite.

        The composite should have both l and v > 0, reflecting
        evidence of both compliance and violation across sites.
        """
        active = _make_consent(status="active", resource_id="a")
        rejected = _make_consent(status="rejected", resource_id="r")
        meet = fhir_multi_site_meet(active, rejected)
        assert_valid_compliance_opinion(meet, "conflicting")
        # Both components present
        assert meet.lawfulness > 0.0
        assert meet.violation > 0.0

    def test_single_site_identity(self):
        """Single-site meet is identity (Theorem 1(g) boundary).

        meet([ω]) = ω
        """
        resource = _make_consent(status="active", resource_id="solo")
        meet = fhir_multi_site_meet(resource)
        direct = fhir_consent_to_opinion(resource)
        assert abs(meet.lawfulness - direct.lawfulness) < TOL
        assert abs(meet.violation - direct.violation) < TOL
        assert abs(meet.uncertainty - direct.uncertainty) < TOL

    def test_expiry_on_already_rejected_consent(self):
        """Expiring a rejected consent → violation stays dominant.

        The consent was already invalid; expiry reinforces violation
        but the opinion is already violation-heavy.
        """
        resource = _make_consent(
            status="rejected",
            provision_period={"start": "2024-01-01", "end": "2025-06-30"},
            resource_id="rejected-expired",
        )
        op = fhir_consent_expiry(resource, assessment_time="2025-09-01")
        assert_valid_compliance_opinion(op, "rejected+expired")
        assert op.violation > op.lawfulness

    def test_regulatory_change_same_status(self):
        """Regulatory change where both consents are active.

        Models a regulation update that doesn't invalidate existing
        consent but changes the framework. The new opinion should
        simply replace the old one.
        """
        old = _make_jurisdictional_consent(
            "GDPR-v1", status="active",
            date_time="2024-01-01", resource_id="old",
        )
        new = _make_jurisdictional_consent(
            "GDPR-v2", status="active",
            date_time="2025-06-01", resource_id="new",
        )
        pre = fhir_consent_regulatory_change(old, new, "2025-03-01")
        post = fhir_consent_regulatory_change(old, new, "2025-09-01")
        # Both are active → similar lawfulness, but post uses new's opinion
        assert_valid_compliance_opinion(pre)
        assert_valid_compliance_opinion(post)

    def test_expiry_then_regulatory_change(self):
        """Consent expires, then regulation changes — non-commutative.

        Theorem 4(e): trigger ordering matters. The order of applying
        expiry vs regulatory change produces different results.
        """
        resource = _make_consent(
            status="active",
            provision_period={"start": "2024-01-01", "end": "2025-06-30"},
            resource_id="orig",
        )
        new_reg = _make_jurisdictional_consent(
            "HIPAA-v2", status="active",
            date_time="2025-09-01", resource_id="new",
        )

        # Path 1: expiry first (consent expired June 30), then reg change (Sep 1)
        expired = fhir_consent_expiry(resource, assessment_time="2025-08-01")
        # Now regulatory change at Sep 1, assessed at Oct 1
        path1 = fhir_consent_regulatory_change(
            # Need to wrap expired opinion as a consent resource
            opinion_to_fhir_consent(expired, resource_id="exp", date_time="2025-08-01"),
            new_reg,
            "2025-10-01",
        )

        # Path 2: assess directly under new regulation (no expiry applied)
        path2 = fhir_consent_to_opinion(new_reg)

        # Both should be valid but may differ
        assert_valid_compliance_opinion(path1, "expiry→reg-change")
        assert_valid_compliance_opinion(path2, "direct new reg")

    def test_multi_site_meet_preserves_base_rate_product(self):
        """Base rate follows Theorem 1: a⊓ = a₁ · a₂.

        The product of base rates ensures strict jurisdictions
        dominate the prior.
        """
        s1 = _make_consent(status="active", resource_id="s1")
        s2 = _make_consent(status="active", resource_id="s2")
        op1 = fhir_consent_to_opinion(s1)
        op2 = fhir_consent_to_opinion(s2)
        meet = fhir_multi_site_meet(s1, s2)
        expected_a = op1.base_rate * op2.base_rate
        assert abs(meet.base_rate - expected_a) < TOL
