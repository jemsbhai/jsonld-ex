"""
FHIR R4 Consent ↔ Compliance Algebra bridge.

Phase 4 of FHIR interoperability. Bridges the compliance algebra
(ComplianceOpinion, consent_validity, withdrawal_override,
jurisdictional_meet) to FHIR R4 Consent resources.

Public API:
    fhir_consent_to_opinion  — Consent → ComplianceOpinion
    opinion_to_fhir_consent  — ComplianceOpinion → Consent
    fhir_consent_validity    — Six-condition GDPR Art. 7 composition
    fhir_consent_withdrawal  — Active → withdrawn transition
    fhir_multi_site_meet     — Multi-site jurisdictional meet

FHIR R4 Consent spec: https://hl7.org/fhir/R4/consent.html
Compliance algebra spec: compliance_algebra.md (Syed et al. 2026)

Design rationale:
    The FHIR Consent resource captures consent *state* (status, scope,
    provision, verification) but not consent *uncertainty*. This module
    infers a ComplianceOpinion from available FHIR fields, and embeds
    the full SL tuple as a jsonld-ex extension for round-trip fidelity.

    The GDPR six-condition mapping (fhir_consent_validity) is a
    best-effort inference from available FHIR fields. Perfect mapping
    requires information beyond what FHIR captures — this is documented
    as a limitation in each condition's inference logic.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Optional, Union

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    consent_validity,
    expiry_trigger,
    jurisdictional_meet,
    regulatory_change_trigger,
    withdrawal_override,
)
from jsonld_ex.fhir_interop._constants import (
    CONSENT_STATUS_PROBABILITY,
    CONSENT_STATUS_UNCERTAINTY,
    FHIR_EXTENSION_URL,
)
from jsonld_ex.fhir_interop._scalar import (
    scalar_to_opinion,
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
)


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _validate_consent_resource(resource: dict[str, Any]) -> None:
    """Validate that resource is a FHIR Consent.

    Raises:
        ValueError: If resourceType is missing or not 'Consent'.
    """
    if "resourceType" not in resource:
        raise ValueError(
            "FHIR resource must contain a 'resourceType' field"
        )
    if resource["resourceType"] != "Consent":
        raise ValueError(
            f"Expected resourceType 'Consent', "
            f"got '{resource['resourceType']}'"
        )


def _try_recover_opinion_from_extensions(
    extensions: Optional[list[dict[str, Any]]],
) -> Optional[Opinion]:
    """Try to recover an exact Opinion from a jsonld-ex extension list."""
    if not extensions:
        return None
    for ext in extensions:
        if ext.get("url") == FHIR_EXTENSION_URL:
            return fhir_extension_to_opinion(ext)
    return None


def _count_verified(
    verifications: Optional[list[dict[str, Any]]],
) -> int:
    """Count verification entries where verified=true."""
    if not verifications:
        return 0
    return sum(
        1 for v in verifications
        if isinstance(v, dict) and v.get("verified") is True
    )


def _parse_date(value: Union[str, None]) -> Optional[date]:
    """Parse a FHIR date or dateTime string to a Python date.

    Handles: 'YYYY-MM-DD', 'YYYY-MM-DDThh:mm:ss...'.
    """
    if value is None:
        return None
    # Take only the date portion (first 10 chars)
    date_str = value[:10]
    return date.fromisoformat(date_str)


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_to_opinion()
# ═══════════════════════════════════════════════════════════════════


def fhir_consent_to_opinion(
    resource: dict[str, Any],
    *,
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Convert a FHIR R4 Consent resource to a ComplianceOpinion.

    Inference signals (in priority order):
      1. jsonld-ex extension — exact recovery if present
      2. Consent.status — primary signal (maps via CONSENT_STATUS_*)
      3. Consent.verification[] — verified=true reduces uncertainty
      4. Consent.provision.type — permit/deny adjusts lawfulness

    Args:
        resource: A FHIR R4 Consent resource dict.
        fhir_version: FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion reflecting consent validity.

    Raises:
        ValueError: If resource lacks resourceType or is not Consent.
    """
    _validate_consent_resource(resource)

    # Priority 1: exact recovery from extension
    extensions = resource.get("extension")
    recovered = _try_recover_opinion_from_extensions(extensions)
    if recovered is not None:
        return ComplianceOpinion.from_opinion(recovered)

    # Priority 2: status-driven reconstruction
    status = resource.get("status", "draft")
    prob = CONSENT_STATUS_PROBABILITY.get(status, 0.50)
    base_u = CONSENT_STATUS_UNCERTAINTY.get(status, 0.35)

    # Priority 3: verification signals
    verifications = resource.get("verification")
    verified_count = _count_verified(verifications)
    if verified_count > 0:
        # Each verification event reduces uncertainty (diminishing returns)
        # Factor: 0.75 per verification, floored at 0.3× of base
        factor = max(0.3, 0.75 ** verified_count)
        base_u *= factor
    elif verifications is not None:
        # Verification entries exist but none verified=true:
        # no evidence gained, slight increase for incomplete process
        base_u *= 1.05

    # Priority 4: provision type
    provision = resource.get("provision")
    if isinstance(provision, dict):
        prov_type = provision.get("type")
        if prov_type == "deny":
            # Deny provision: shift probability downward
            prob *= 0.7
        # permit is the default / expected case — no adjustment

    # Clamp uncertainty to valid range
    base_u = max(0.0, min(base_u, 0.99))

    # Build opinion via scalar_to_opinion (handles b/d/u distribution)
    opinion = scalar_to_opinion(prob, default_uncertainty=base_u)
    return ComplianceOpinion.from_opinion(opinion)


# ═══════════════════════════════════════════════════════════════════
# opinion_to_fhir_consent()
# ═══════════════════════════════════════════════════════════════════


def _opinion_to_status(op: ComplianceOpinion) -> str:
    """Map a ComplianceOpinion to the best FHIR Consent.status.

    Decision logic:
      - High uncertainty (u ≥ 0.5) → 'draft' (pre-decisional)
      - High violation (v > l and v > u) → 'rejected'
      - High lawfulness (l > v) → 'active'
      - Otherwise → 'draft'
    """
    l, v, u = op.lawfulness, op.violation, op.uncertainty
    if u >= 0.5:
        return "draft"
    if v > l and v > u:
        return "rejected"
    if l > v:
        return "active"
    return "draft"


def opinion_to_fhir_consent(
    opinion: ComplianceOpinion,
    *,
    patient: Optional[str] = None,
    scope: Optional[str] = None,
    policy_rule: Optional[str] = None,
    resource_id: Optional[str] = None,
    date_time: Optional[str] = None,
    fhir_version: str = "R4",
) -> dict[str, Any]:
    """Convert a ComplianceOpinion to a FHIR R4 Consent resource.

    Produces a minimal valid FHIR R4 Consent with the SL opinion
    embedded as a jsonld-ex extension for round-trip preservation.

    The Consent.status is inferred from the opinion's dominant component.

    Args:
        opinion: The ComplianceOpinion to export.
        patient: Optional patient reference (e.g. "Patient/123").
        scope: Optional scope code (e.g. "patient-privacy").
        policy_rule: Optional policy rule code (e.g. "HIPAA").
        resource_id: Optional resource id.
        date_time: Optional ISO date/dateTime string.
        fhir_version: FHIR version (currently only "R4").

    Returns:
        A FHIR R4 Consent resource dict.
    """
    co = (
        opinion if isinstance(opinion, ComplianceOpinion)
        else ComplianceOpinion.from_opinion(opinion)
    )

    status = _opinion_to_status(co)

    # Build the extension with the full opinion tuple
    ext = opinion_to_fhir_extension(co)

    resource: dict[str, Any] = {
        "resourceType": "Consent",
        "status": status,
        "scope": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/consentscope",
                "code": scope or "patient-privacy",
            }],
        },
        "category": [{
            "coding": [{
                "system": "http://loinc.org",
                "code": "59284-0",
            }],
        }],
        "extension": [ext],
    }

    if resource_id is not None:
        resource["id"] = resource_id
    if date_time is not None:
        resource["dateTime"] = date_time
    if patient is not None:
        resource["patient"] = {"reference": patient}
    if policy_rule is not None:
        resource["policyRule"] = {
            "coding": [{"code": policy_rule}],
        }

    return resource


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_validity() — six-condition GDPR Art. 7
# ═══════════════════════════════════════════════════════════════════


def _infer_freely_given(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'freely given' condition (GDPR Art. 7(4)).

    FHIR signal: scope code. Research or treatment scope with an
    active status is a positive signal. This is a weak inference —
    FHIR does not directly capture coercion/imbalance information.

    Limitation: true 'freely given' assessment requires contextual
    information (power imbalance, conditionality) not in FHIR.
    """
    status = resource.get("status", "draft")
    prob = CONSENT_STATUS_PROBABILITY.get(status, 0.50)
    # Moderate uncertainty — FHIR has limited information on this
    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(prob, default_uncertainty=0.25)
    )


def _infer_specific(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'specific' condition (GDPR Art. 4(11)).

    FHIR signal: provision.purpose presence and specificity.
    A consent with specific purpose codes is more specific.

    Limitation: specificity is relative to the processing context,
    which is not captured in a single Consent resource.
    """
    provision = resource.get("provision", {})
    purposes = provision.get("purpose", []) if isinstance(provision, dict) else []
    has_specific_purpose = len(purposes) > 0

    base_prob = 0.75 if has_specific_purpose else 0.60
    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(base_prob, default_uncertainty=0.25)
    )


def _infer_informed(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'informed' condition (GDPR Art. 4(11)).

    FHIR signal: provision.data presence, category specificity.
    A consent referencing specific data categories suggests the
    data subject was informed about what data is covered.

    Limitation: 'informed' requires evidence that information was
    provided to the data subject, which FHIR does not capture.
    """
    provision = resource.get("provision", {})
    data_refs = provision.get("data", []) if isinstance(provision, dict) else []
    categories = resource.get("category", [])

    has_data_refs = len(data_refs) > 0
    has_detailed_category = len(categories) > 1

    base_prob = 0.65
    if has_data_refs:
        base_prob += 0.10
    if has_detailed_category:
        base_prob += 0.05

    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(min(base_prob, 0.95), default_uncertainty=0.25)
    )


def _infer_unambiguous(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'unambiguous' condition (GDPR Art. 4(11)).

    FHIR signal: provision.type explicitness. An explicit 'permit'
    or 'deny' is clearer than no provision type at all.

    Limitation: unambiguity also depends on the consent ceremony
    (was it a clear affirmative act?) which FHIR does not capture.
    """
    provision = resource.get("provision", {})
    prov_type = provision.get("type") if isinstance(provision, dict) else None

    if prov_type in ("permit", "deny"):
        base_prob = 0.80
    else:
        base_prob = 0.60

    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(base_prob, default_uncertainty=0.25)
    )


def _infer_demonstrable(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'demonstrable' condition (GDPR Art. 7(1)).

    FHIR signal: verification[] with verified=true. This directly
    maps to the controller's ability to demonstrate consent.

    This is the strongest FHIR-to-GDPR mapping: FHIR verification
    records are explicit evidence of consent demonstration.
    """
    verifications = resource.get("verification")
    verified_count = _count_verified(verifications)

    if verified_count >= 2:
        base_prob = 0.90
        base_u = 0.10
    elif verified_count == 1:
        base_prob = 0.80
        base_u = 0.15
    else:
        base_prob = 0.50
        base_u = 0.35

    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(base_prob, default_uncertainty=base_u)
    )


def _infer_distinguishable(resource: dict[str, Any]) -> ComplianceOpinion:
    """Infer 'distinguishable' condition (GDPR Art. 7(2)).

    FHIR signal: category coding specificity. A consent with a
    specific LOINC or other standard category code suggests the
    consent is distinguishable from other matters.

    Limitation: distinguishability depends on presentation context
    (was consent separate from other agreements?) not captured in FHIR.
    """
    categories = resource.get("category", [])
    has_standard_coding = False
    for cat in categories:
        codings = cat.get("coding", []) if isinstance(cat, dict) else []
        for coding in codings:
            if coding.get("system") and coding.get("code"):
                has_standard_coding = True
                break

    base_prob = 0.75 if has_standard_coding else 0.55
    return ComplianceOpinion.from_opinion(
        scalar_to_opinion(base_prob, default_uncertainty=0.25)
    )


def fhir_consent_validity(
    resource: dict[str, Any],
    *,
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Map FHIR Consent fields to six GDPR Art. 7 conditions.

    Per compliance_algebra.md §7.1 (Definition 8):
      ω_c^P = J⊓(ω_free, ω_spec, ω_inf, ω_unamb, ω_demo, ω_dist)

    Each condition is inferred from available FHIR Consent fields.
    The six per-condition opinions are composed via consent_validity()
    from the compliance algebra.

    Limitation: FHIR Consent does not capture all six GDPR conditions
    directly. Each inference function documents what FHIR signal is
    used and what information is missing. The result should be
    interpreted as a lower-bound estimate of consent validity.

    Args:
        resource: A FHIR R4 Consent resource dict.
        fhir_version: FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion for composite consent validity.

    Raises:
        ValueError: If resource lacks resourceType or is not Consent.
    """
    _validate_consent_resource(resource)

    freely_given = _infer_freely_given(resource)
    specific = _infer_specific(resource)
    informed = _infer_informed(resource)
    unambiguous = _infer_unambiguous(resource)
    demonstrable = _infer_demonstrable(resource)
    distinguishable = _infer_distinguishable(resource)

    return consent_validity(
        freely_given=freely_given,
        specific=specific,
        informed=informed,
        unambiguous=unambiguous,
        demonstrable=demonstrable,
        distinguishable=distinguishable,
    )


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_withdrawal()
# ═══════════════════════════════════════════════════════════════════


def fhir_consent_withdrawal(
    active_consent: dict[str, Any],
    withdrawn_consent: dict[str, Any],
    assessment_time: str,
    *,
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Bridge FHIR Consent status transitions to withdrawal_override.

    Per compliance_algebra.md §7.2 (Definition 11):
      Withdraw(ω_c^P, ω_w^P, t, t_w) = ω_c^P if t < t_w
                                       = ω_w^P if t ≥ t_w

    The active consent's dateTime is the consent grant time.
    The withdrawn consent's dateTime is the withdrawal time (t_w).
    assessment_time is the current evaluation time (t).

    Args:
        active_consent:   FHIR Consent with status='active' + dateTime.
        withdrawn_consent: FHIR Consent with status='inactive'/'rejected'
                           + dateTime representing withdrawal.
        assessment_time:  ISO date string for assessment time.
        fhir_version:     FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion — active if t < t_w, withdrawal if t ≥ t_w.

    Raises:
        ValueError: If either consent lacks dateTime.
    """
    _validate_consent_resource(active_consent)
    _validate_consent_resource(withdrawn_consent)

    active_dt = active_consent.get("dateTime")
    withdrawn_dt = withdrawn_consent.get("dateTime")

    if active_dt is None:
        raise ValueError(
            "Active consent must have a 'dateTime' field"
        )
    if withdrawn_dt is None:
        raise ValueError(
            "Withdrawn consent must have a 'dateTime' field"
        )

    # Parse dates for ordinal comparison
    t_assess = _parse_date(assessment_time)
    t_w = _parse_date(withdrawn_dt)

    if t_assess is None:
        raise ValueError("assessment_time must be a valid ISO date string")
    if t_w is None:
        raise ValueError("Withdrawn consent dateTime must be a valid date")

    # Convert to ordinal for comparison (float timestamps for the algebra)
    t_assess_ord = float(t_assess.toordinal())
    t_w_ord = float(t_w.toordinal())

    consent_opinion = fhir_consent_to_opinion(active_consent)
    withdrawal_opinion = fhir_consent_to_opinion(withdrawn_consent)

    return withdrawal_override(
        consent_opinion=consent_opinion,
        withdrawal_opinion=withdrawal_opinion,
        assessment_time=t_assess_ord,
        withdrawal_time=t_w_ord,
    )


# ═══════════════════════════════════════════════════════════════════
# fhir_multi_site_meet()
# ═══════════════════════════════════════════════════════════════════


def fhir_multi_site_meet(
    *consent_resources: dict[str, Any],
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Multi-site consent composition via jurisdictional meet.

    Per compliance_algebra.md §5 (Definition 3, Theorem 1):
    Composite compliance across sites is a conjunction — ALL sites
    must have valid consent.

    Algebraic properties (from Theorem 1):
      (a) l⊓ = l₁ · l₂  (multiplicative lawfulness)
      (b) v⊓ = v₁ + v₂ − v₁·v₂  (disjunctive violation)
      (d) v⊓ ≥ max(v₁, v₂)  (monotonic violation)
      (e) Commutativity
      (f) Associativity

    Args:
        *consent_resources: One or more FHIR R4 Consent resource dicts.
        fhir_version: FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion for composite multi-site consent.

    Raises:
        ValueError: If no resources provided, or any is not a Consent.
    """
    if len(consent_resources) == 0:
        raise ValueError(
            "fhir_multi_site_meet requires at least one Consent resource"
        )

    opinions = []
    for res in consent_resources:
        _validate_consent_resource(res)
        opinions.append(fhir_consent_to_opinion(res))

    return jurisdictional_meet(*opinions)


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_expiry() — provision.period.end → expiry_trigger
# ═══════════════════════════════════════════════════════════════════


def fhir_consent_expiry(
    resource: dict[str, Any],
    assessment_time: str,
    *,
    residual_factor: float = 0.0,
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Bridge FHIR Consent provision.period.end to expiry_trigger.

    Per compliance_algebra.md §8.2 (Definition 12):
    At trigger time t_T, lawfulness transfers to violation (not
    uncertainty). An expired consent is a known fact.

    FHIR mapping:
      provision.period.end → trigger time (t_T)
      assessment_time      → evaluation time (t)

    If the consent has no provision.period or no period.end, the
    consent is treated as non-expiring and the opinion is returned
    unchanged.

    Theorem 4 properties:
      (b) Constraint preservation: l' + v' + u' = 1
      (c) Monotonicity: l' ≤ l, v' ≥ v, u' = u
      (d) Hard expiry: γ=0 → l'=0, v'=v+l

    Args:
        resource:        A FHIR R4 Consent resource dict.
        assessment_time: ISO date string for evaluation time.
        residual_factor: γ ∈ [0, 1], fraction of lawfulness retained
                         post-expiry. Default 0.0 (hard expiry).
        fhir_version:    FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion after applying expiry (if triggered).

    Raises:
        ValueError: If resource is not a Consent.
    """
    _validate_consent_resource(resource)

    consent_op = fhir_consent_to_opinion(resource)

    # Extract provision.period.end
    provision = resource.get("provision")
    if not isinstance(provision, dict):
        return consent_op

    period = provision.get("period")
    if not isinstance(period, dict):
        return consent_op

    end_str = period.get("end")
    if end_str is None:
        return consent_op

    t_assess = _parse_date(assessment_time)
    t_expiry = _parse_date(end_str)

    if t_assess is None or t_expiry is None:
        return consent_op

    return expiry_trigger(
        opinion=consent_op,
        assessment_time=float(t_assess.toordinal()),
        trigger_time=float(t_expiry.toordinal()),
        residual_factor=residual_factor,
    )


# ═══════════════════════════════════════════════════════════════════
# fhir_consent_regulatory_change()
# ═══════════════════════════════════════════════════════════════════


def fhir_consent_regulatory_change(
    old_consent: dict[str, Any],
    new_consent: dict[str, Any],
    assessment_time: str,
    *,
    fhir_version: str = "R4",
) -> ComplianceOpinion:
    """Bridge FHIR Consent policyRule transition to regulatory_change_trigger.

    Per compliance_algebra.md §8.2 (Definition 14):
    At trigger time, the compliance opinion is replaced by a new
    assessment reflecting the changed legal framework. Same
    proposition-replacement semantics as withdrawal_override.

    FHIR mapping:
      old_consent.dateTime  → original assessment time
      new_consent.dateTime  → regulatory change time (t_T)
      assessment_time       → evaluation time (t)

    Pre-trigger (t < t_T): old consent's opinion applies.
    Post-trigger (t ≥ t_T): new consent's opinion applies.

    Theorem 4(e): trigger ordering is non-commutative — the order
    of regulatory events matters.

    Args:
        old_consent:     FHIR Consent under previous regulation.
        new_consent:     FHIR Consent under new regulation, with
                         dateTime representing the change time.
        assessment_time: ISO date string for evaluation time.
        fhir_version:    FHIR version (currently only "R4").

    Returns:
        ComplianceOpinion — old if t < t_T, new if t ≥ t_T.

    Raises:
        ValueError: If either consent lacks dateTime or resourceType.
    """
    _validate_consent_resource(old_consent)
    _validate_consent_resource(new_consent)

    old_dt = old_consent.get("dateTime")
    new_dt = new_consent.get("dateTime")

    if old_dt is None:
        raise ValueError(
            "Old consent must have a 'dateTime' field"
        )
    if new_dt is None:
        raise ValueError(
            "New consent must have a 'dateTime' field"
        )

    t_assess = _parse_date(assessment_time)
    t_change = _parse_date(new_dt)

    if t_assess is None:
        raise ValueError("assessment_time must be a valid ISO date string")
    if t_change is None:
        raise ValueError("New consent dateTime must be a valid date")

    old_op = fhir_consent_to_opinion(old_consent)
    new_op = fhir_consent_to_opinion(new_consent)

    return regulatory_change_trigger(
        opinion=old_op,
        assessment_time=float(t_assess.toordinal()),
        trigger_time=float(t_change.toordinal()),
        new_opinion=new_op,
    )
