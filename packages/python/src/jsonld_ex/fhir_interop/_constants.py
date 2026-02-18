"""
Shared constants, mappings, and configuration for FHIR interoperability.

All categorical-to-probability mappings, uncertainty multipliers, and
namespace constants used across the FHIR sub-package are centralised
here to avoid circular imports and ensure consistency.
"""

from __future__ import annotations

# ── Namespace & Version Constants ──────────────────────────────────

FHIR_EXTENSION_URL = "https://jsonld-ex.github.io/ns/fhir/opinion"
"""FHIR extension URL for embedding SL opinion tuples in FHIR resources.

Follows FHIR's standard extension mechanism (R4 §2.4).  Systems that
do not understand this extension will ignore it; jsonld-ex-aware
systems can recover the full opinion for downstream reasoning.
"""

SUPPORTED_FHIR_VERSIONS = ("R4",)
"""FHIR versions with implemented conversion logic."""

FHIR = "http://hl7.org/fhir/"

# ── Status Uncertainty Multipliers ─────────────────────────────────
#
# These multipliers adjust the default uncertainty budget based on the
# FHIR resource status.  The rationale:
#
#   final / amended / corrected → reviewed, higher confidence
#   preliminary / registered    → early stage, lower confidence
#   entered-in-error            → data integrity compromised
#   cancelled                   → process abandoned
#
# Multipliers < 1.0 reduce uncertainty; > 1.0 increase it.
# Values are applied multiplicatively to the base uncertainty budget.

STATUS_UNCERTAINTY_MULTIPLIER: dict[str, float] = {
    "final": 0.5,
    "amended": 0.6,
    "corrected": 0.6,
    "appended": 0.7,
    "preliminary": 2.0,
    "registered": 1.5,
    "entered-in-error": 5.0,
    "cancelled": 3.0,
    "unknown": 1.2,
}


# ── Categorical-to-probability mappings ────────────────────────────

VERIFICATION_STATUS_PROBABILITY: dict[str, float] = {
    "confirmed": 0.95,
    "unconfirmed": 0.55,
    "provisional": 0.60,
    "differential": 0.40,
    "refuted": 0.05,
    "entered-in-error": 0.50,
}

VERIFICATION_STATUS_UNCERTAINTY: dict[str, float] = {
    "confirmed": 0.10,
    "unconfirmed": 0.35,
    "provisional": 0.35,
    "differential": 0.45,
    "refuted": 0.10,
    "entered-in-error": 0.70,
}

INTERPRETATION_PROBABILITY: dict[str, float] = {
    # Proposition: "clinically significant finding present"
    "H": 0.85,   # High
    "HH": 0.95,  # Critical high
    "L": 0.80,   # Low
    "LL": 0.95,  # Critical low
    "A": 0.85,   # Abnormal
    "AA": 0.95,  # Critical abnormal
    "N": 0.15,   # Normal (low probability of significant finding)
    # Susceptibility
    "S": 0.85,   # Susceptible
    "R": 0.85,   # Resistant
    "I": 0.50,   # Intermediate
}

# ── Phase 2: Criticality mappings (AllergyIntolerance) ─────────────

CRITICALITY_PROBABILITY: dict[str, float] = {
    "high": 0.90,
    "low": 0.30,
    "unable-to-assess": 0.50,
}

CRITICALITY_UNCERTAINTY: dict[str, float] = {
    "high": 0.15,
    "low": 0.15,
    "unable-to-assess": 0.45,
}

# ── Phase 2: MedicationStatement status → adherence confidence ────

MEDSTMT_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "completed": 0.90,
    "entered-in-error": 0.50,
    "intended": 0.60,
    "stopped": 0.10,
    "on-hold": 0.40,
    "unknown": 0.50,
    "not-taken": 0.05,
}

MEDSTMT_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "completed": 0.10,
    "entered-in-error": 0.70,
    "intended": 0.30,
    "stopped": 0.15,
    "on-hold": 0.25,
    "unknown": 0.45,
    "not-taken": 0.10,
}

# ── Phase 2: ClinicalImpression status multipliers ────────────────

CLINICAL_IMPRESSION_STATUS_MULTIPLIER: dict[str, float] = {
    "completed": 0.5,
    "in-progress": 2.0,
    "entered-in-error": 5.0,
}

# ── Phase 3: DetectedIssue severity → alert confidence ────────────

DETECTED_ISSUE_SEVERITY_PROBABILITY: dict[str, float] = {
    "high": 0.90,
    "moderate": 0.60,
    "low": 0.30,
}

DETECTED_ISSUE_SEVERITY_UNCERTAINTY: dict[str, float] = {
    "high": 0.15,
    "moderate": 0.25,
    "low": 0.20,
}

# ── Phase 3: Immunization status → seroconversion confidence ─────

IMMUNIZATION_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.85,
    "not-done": 0.05,
    "entered-in-error": 0.50,
}

IMMUNIZATION_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.15,
    "not-done": 0.10,
    "entered-in-error": 0.70,
}

# ── Phase 3: FamilyMemberHistory ──────────────────────────────────

FAMILY_HISTORY_DEFAULT_UNCERTAINTY: float = 0.30
"""Default uncertainty budget for FamilyMemberHistory opinions.

Higher than the standard 0.15 default to reflect the inherent
unreliability of patient-reported family health history.  Recall
bias, incomplete knowledge of family members' health, and
communication errors all contribute to elevated baseline uncertainty.
"""

FAMILY_HISTORY_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.70,
    "partial": 0.50,
    "health-unknown": 0.30,
    "entered-in-error": 0.50,
}

FAMILY_HISTORY_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.25,
    "partial": 0.40,
    "health-unknown": 0.55,
    "entered-in-error": 0.70,
}

# ── Phase 3: Procedure status → outcome confidence ───────────────

PROCEDURE_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.85,
    "in-progress": 0.60,
    "not-done": 0.05,
    "entered-in-error": 0.50,
    "preparation": 0.50,
    "on-hold": 0.40,
    "stopped": 0.20,
    "unknown": 0.50,
}

PROCEDURE_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.15,
    "in-progress": 0.35,
    "not-done": 0.10,
    "entered-in-error": 0.70,
    "preparation": 0.40,
    "on-hold": 0.30,
    "stopped": 0.20,
    "unknown": 0.45,
}


# ── Phase 4: Consent status → compliance opinion ─────────────────
#
# FHIR R4 Consent.status codes mapped to the probability that the
# consent is lawfully valid (the proposition under assessment).
#
# Epistemic rationale:
#   active     — consent is in force; strongest lawfulness signal.
#   inactive   — was valid, now expired/superseded; moderate lawfulness
#                (past validity is partial evidence, but no longer current).
#   draft      — pre-decisional; genuinely uncertain, not low-lawfulness.
#   proposed   — offered but not yet accepted; similar to draft.
#   rejected   — explicitly refused; strong violation signal.
#   entered-in-error — data integrity compromised; high uncertainty.

CONSENT_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.90,
    "inactive": 0.30,
    "draft": 0.50,
    "proposed": 0.50,
    "rejected": 0.05,
    "entered-in-error": 0.50,
}

CONSENT_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "inactive": 0.25,
    "draft": 0.40,
    "proposed": 0.35,
    "rejected": 0.10,
    "entered-in-error": 0.45,
}


# ── Phase 5: Observation status → validity confidence ─────────────
#
# When Observation lacks an interpretation code (very common in
# real-world data like Synthea), we fall back to a status-based
# opinion.  Proposition: "this observation is valid and reliable."

OBSERVATION_STATUS_PROBABILITY: dict[str, float] = {
    "final": 0.85,
    "amended": 0.85,
    "corrected": 0.85,
    "preliminary": 0.60,
    "registered": 0.50,
    "cancelled": 0.15,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

OBSERVATION_STATUS_UNCERTAINTY: dict[str, float] = {
    "final": 0.15,
    "amended": 0.15,
    "corrected": 0.15,
    "preliminary": 0.35,
    "registered": 0.40,
    "cancelled": 0.20,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}

# ── Phase 5: DiagnosticReport status → validity confidence ────────
#
# When DiagnosticReport lacks a conclusion text, we fall back to a
# status-based opinion.  Result count modulates uncertainty.
# Proposition: "this report is valid and reliable."

DIAGNOSTIC_REPORT_STATUS_PROBABILITY: dict[str, float] = {
    "final": 0.90,
    "amended": 0.88,
    "corrected": 0.88,
    "appended": 0.85,
    "preliminary": 0.60,
    "registered": 0.50,
    "cancelled": 0.15,
    "entered-in-error": 0.50,
    "unknown": 0.50,
    "partial": 0.55,
}

DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY: dict[str, float] = {
    "final": 0.12,
    "amended": 0.15,
    "corrected": 0.15,
    "appended": 0.18,
    "preliminary": 0.35,
    "registered": 0.40,
    "cancelled": 0.20,
    "entered-in-error": 0.70,
    "unknown": 0.45,
    "partial": 0.35,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Universal FHIR coverage — clinical workflow resources
# ═══════════════════════════════════════════════════════════════════

# ── Encounter status → visit validity confidence ──────────────────

ENCOUNTER_STATUS_PROBABILITY: dict[str, float] = {
    "finished": 0.90,
    "arrived": 0.70,
    "triaged": 0.75,
    "in-progress": 0.70,
    "onleave": 0.60,
    "planned": 0.50,
    "cancelled": 0.10,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

ENCOUNTER_STATUS_UNCERTAINTY: dict[str, float] = {
    "finished": 0.10,
    "arrived": 0.25,
    "triaged": 0.25,
    "in-progress": 0.30,
    "onleave": 0.30,
    "planned": 0.35,
    "cancelled": 0.15,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}

# ── MedicationRequest status → prescription validity ──────────────

MEDICATION_REQUEST_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "on-hold": 0.50,
    "cancelled": 0.10,
    "completed": 0.90,
    "entered-in-error": 0.50,
    "stopped": 0.15,
    "draft": 0.50,
    "unknown": 0.50,
}

MEDICATION_REQUEST_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "on-hold": 0.30,
    "cancelled": 0.15,
    "completed": 0.10,
    "entered-in-error": 0.70,
    "stopped": 0.15,
    "draft": 0.40,
    "unknown": 0.45,
}

# ── CarePlan status → plan validity/adherence ─────────────────────

CARE_PLAN_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "on-hold": 0.50,
    "revoked": 0.10,
    "completed": 0.90,
    "entered-in-error": 0.50,
    "draft": 0.50,
    "unknown": 0.50,
}

CARE_PLAN_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "on-hold": 0.30,
    "revoked": 0.15,
    "completed": 0.10,
    "entered-in-error": 0.70,
    "draft": 0.40,
    "unknown": 0.45,
}

# ── Goal lifecycleStatus → goal achievement confidence ────────────
#
# Goal uses lifecycleStatus (not status) and optionally
# achievementStatus for a second opinion dimension.

GOAL_LIFECYCLE_PROBABILITY: dict[str, float] = {
    "active": 0.75,
    "accepted": 0.70,
    "planned": 0.55,
    "proposed": 0.50,
    "on-hold": 0.40,
    "completed": 0.90,
    "cancelled": 0.10,
    "entered-in-error": 0.50,
    "rejected": 0.10,
}

GOAL_LIFECYCLE_UNCERTAINTY: dict[str, float] = {
    "active": 0.25,
    "accepted": 0.25,
    "planned": 0.35,
    "proposed": 0.40,
    "on-hold": 0.30,
    "completed": 0.10,
    "cancelled": 0.15,
    "entered-in-error": 0.70,
    "rejected": 0.15,
}

GOAL_ACHIEVEMENT_PROBABILITY: dict[str, float] = {
    "achieved": 0.95,
    "sustaining": 0.85,
    "in-progress": 0.60,
    "improving": 0.70,
    "worsening": 0.25,
    "no-change": 0.45,
    "not-achieved": 0.10,
    "no-progress": 0.20,
    "not-attainable": 0.05,
}

GOAL_ACHIEVEMENT_UNCERTAINTY: dict[str, float] = {
    "achieved": 0.10,
    "sustaining": 0.15,
    "in-progress": 0.30,
    "improving": 0.25,
    "worsening": 0.25,
    "no-change": 0.30,
    "not-achieved": 0.15,
    "no-progress": 0.25,
    "not-attainable": 0.15,
}

# ── CareTeam status → assignment validity ─────────────────────────

CARE_TEAM_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "proposed": 0.55,
    "suspended": 0.40,
    "inactive": 0.20,
    "entered-in-error": 0.50,
}

CARE_TEAM_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "proposed": 0.35,
    "suspended": 0.25,
    "inactive": 0.15,
    "entered-in-error": 0.70,
}

# ── ImagingStudy status → study validity ──────────────────────────

IMAGING_STUDY_STATUS_PROBABILITY: dict[str, float] = {
    "available": 0.90,
    "registered": 0.60,
    "cancelled": 0.10,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

IMAGING_STUDY_STATUS_UNCERTAINTY: dict[str, float] = {
    "available": 0.10,
    "registered": 0.35,
    "cancelled": 0.15,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Universal FHIR coverage — administrative resources
# ═══════════════════════════════════════════════════════════════════

# ── Patient data quality — no status field ────────────────────────
#
# Patient has no status; opinion based on data completeness.
# Base probability represents "this patient record is accurate."

PATIENT_BASE_PROBABILITY: float = 0.75
PATIENT_BASE_UNCERTAINTY: float = 0.25

# ── Organization / Practitioner — boolean active field ────────────

ACTIVE_BOOLEAN_PROBABILITY: dict[str, float] = {
    "true": 0.90,
    "false": 0.20,
}

ACTIVE_BOOLEAN_UNCERTAINTY: dict[str, float] = {
    "true": 0.10,
    "false": 0.15,
}

# ── Device status → device record validity ────────────────────────

DEVICE_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.90,
    "inactive": 0.25,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

DEVICE_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.10,
    "inactive": 0.15,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Universal FHIR coverage — financial resources
# ═══════════════════════════════════════════════════════════════════

# ── Claim status → claim validity ─────────────────────────────────

CLAIM_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "cancelled": 0.10,
    "draft": 0.50,
    "entered-in-error": 0.50,
}

CLAIM_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "cancelled": 0.15,
    "draft": 0.40,
    "entered-in-error": 0.70,
}

# ── ExplanationOfBenefit status → adjudication accuracy ───────────

EOB_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "cancelled": 0.10,
    "draft": 0.50,
    "entered-in-error": 0.50,
}

EOB_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "cancelled": 0.15,
    "draft": 0.40,
    "entered-in-error": 0.70,
}

EOB_OUTCOME_PROBABILITY: dict[str, float] = {
    "complete": 0.90,
    "queued": 0.50,
    "error": 0.15,
    "partial": 0.60,
}

EOB_OUTCOME_UNCERTAINTY: dict[str, float] = {
    "complete": 0.10,
    "queued": 0.40,
    "error": 0.20,
    "partial": 0.30,
}


# ── MedicationAdministration status → administration validity ─────

MED_ADMIN_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.90,
    "in-progress": 0.70,
    "not-done": 0.05,
    "on-hold": 0.40,
    "stopped": 0.15,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

MED_ADMIN_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.10,
    "in-progress": 0.30,
    "not-done": 0.10,
    "on-hold": 0.25,
    "stopped": 0.15,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 7A: High-value clinical expansion
# ═══════════════════════════════════════════════════════════════════

# ── ServiceRequest status → order validity confidence ─────────────
#
# Proposition: "this service request is valid and should be acted upon."
#
# ServiceRequest completes the diagnostic chain:
#   ServiceRequest → DiagnosticReport → Observation
#
# Status encodes order lifecycle validity.

SERVICE_REQUEST_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "completed": 0.90,
    "draft": 0.50,
    "on-hold": 0.50,
    "revoked": 0.10,
    "entered-in-error": 0.50,
    "unknown": 0.50,
}

SERVICE_REQUEST_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "completed": 0.10,
    "draft": 0.40,
    "on-hold": 0.30,
    "revoked": 0.15,
    "entered-in-error": 0.70,
    "unknown": 0.45,
}

# ── ServiceRequest intent → epistemic weight multiplier ───────────
#
# Intent encodes how deliberate the clinical decision is.
# Multipliers < 1.0 reduce uncertainty (stronger commitment);
# multipliers > 1.0 increase it (weaker commitment).
#
# Rationale:
#   order/original-order/instance-order (0.7) — formal clinical
#       decision, strongest evidence weight.
#   directive/filler-order (0.8) — protocol-driven or fulfillment,
#       slightly less deliberate than direct orders.
#   reflex-order (0.9) — automated response to another result,
#       less clinical deliberation involved.
#   plan (1.2) — planned but not yet ordered, moderate uncertainty.
#   proposal (1.4) — suggestion only, weakest commitment.
#   option (1.5) — informational, no commitment to act.

SERVICE_REQUEST_INTENT_MULTIPLIER: dict[str, float] = {
    "order": 0.7,
    "original-order": 0.7,
    "instance-order": 0.7,
    "directive": 0.8,
    "filler-order": 0.8,
    "reflex-order": 0.9,
    "plan": 1.2,
    "proposal": 1.4,
    "option": 1.5,
}

# ── ServiceRequest priority → clinical attention multiplier ───────
#
# Higher priority orders receive more clinical attention, which
# reduces uncertainty about their validity and appropriateness.
#
# Multipliers < 1.0 reduce uncertainty; 1.0 is the baseline.

SERVICE_REQUEST_PRIORITY_MULTIPLIER: dict[str, float] = {
    "routine": 1.0,
    "urgent": 0.85,
    "asap": 0.75,
    "stat": 0.65,
}


# ── QuestionnaireResponse status → response reliability ───────────
#
# Proposition: "this questionnaire response is complete and reliable."
#
# QuestionnaireResponse is fundamentally an epistemic resource:
# patient self-report introduces recall bias, social desirability
# bias, health literacy effects, and cognitive load.  Status encodes
# how complete and reviewed the response is.

QR_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.85,
    "amended": 0.80,
    "in-progress": 0.55,
    "stopped": 0.30,
    "entered-in-error": 0.50,
}

QR_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.15,
    "amended": 0.20,
    "in-progress": 0.35,
    "stopped": 0.25,
    "entered-in-error": 0.70,
}

# ── QuestionnaireResponse source → reporter reliability multiplier ─
#
# Who filled out the questionnaire directly affects epistemic quality.
# Practitioners have clinical training; patients introduce self-report
# biases; related persons are proxy reporters with indirect knowledge.
#
# Multipliers < 1.0 reduce uncertainty; > 1.0 increase it.

QR_SOURCE_RELIABILITY_MULTIPLIER: dict[str, float] = {
    "Practitioner": 0.7,
    "PractitionerRole": 0.7,
    "RelatedPerson": 1.2,
    "Patient": 1.3,
}

# ── QuestionnaireResponse item completeness thresholds ────────────
#
# The ratio of answered items to total items is a data quality signal.
# Thresholds define three bands: low, mid, high completeness.
# Each band applies a multiplicative uncertainty adjustment.
#
# All values are configurable via handler_config on from_fhir().

QR_COMPLETENESS_THRESHOLDS: dict[str, float] = {
    "low_threshold": 0.5,     # ratio < this → low completeness
    "high_threshold": 0.8,    # ratio > this → high completeness
    "low_multiplier": 1.3,    # low completeness raises uncertainty
    "mid_multiplier": 1.0,    # mid completeness is baseline
    "high_multiplier": 0.8,   # high completeness reduces uncertainty
}


# ── Specimen status → probability ──────────────────────────────────
#
# Proposition: "This specimen is suitable for reliable diagnostic analysis."
#
# available:        specimen ready for testing — high suitability
# unavailable:      logistic issue, may become available — moderate-low
# unsatisfactory:   quality-failed — lowest suitability
# entered-in-error: data integrity compromised — uncertain

SPECIMEN_STATUS_PROBABILITY: dict[str, float] = {
    "available": 0.85,
    "unavailable": 0.30,
    "unsatisfactory": 0.15,
    "entered-in-error": 0.50,
}

SPECIMEN_STATUS_UNCERTAINTY: dict[str, float] = {
    "available": 0.15,
    "unavailable": 0.40,
    "unsatisfactory": 0.25,
    "entered-in-error": 0.50,
}

# ── Specimen condition → uncertainty multiplier (v2 Table 0493) ───
#
# Each condition code applies an independent multiplicative adjustment
# to the base uncertainty.  Multiple conditions compound (e.g., a
# hemolyzed AND clotted specimen has u *= HEM_mult * CLOT_mult).
#
# Degradation codes (> 1.0) raise uncertainty:
#   HEM  — hemoglobin interference; most common pre-analytical error
#   CLOT — unusable for many coagulation tests
#   CON  — compromised specimen integrity
#   AUT  — tissue breakdown (autolysis), severe degradation
#   SNR  — sample not received, no specimen to analyze
#
# Neutral / positive codes (<= 1.0):
#   LIVE — live specimen, slight positive signal
#   COOL — proper cold-chain handling
#   FROZ — frozen state (test-dependent, neutral)
#   ROOM — room temperature; may be improper for some tests
#   CFU  — centrifuged, processed state
#
# Source: HL7 v2 Table 0493; clinical impact from Plebani (2006),
# Clin Chem Lab Med — pre-analytical errors account for ~70% of
# all lab testing errors.

SPECIMEN_CONDITION_MULTIPLIER: dict[str, float] = {
    "HEM": 1.4,
    "CLOT": 1.5,
    "CON": 1.5,
    "AUT": 1.6,
    "SNR": 2.0,
    "LIVE": 0.9,
    "COOL": 1.0,
    "FROZ": 1.0,
    "ROOM": 1.1,
    "CFU": 0.9,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 7B: US Core Completeness — Coverage
# ═══════════════════════════════════════════════════════════════════

# ── Coverage status → coverage validity confidence ────────────────
#
# Proposition: "This insurance coverage record is valid and currently
# in force."
#
# Completes the financial chain: Coverage → Claim → ExplanationOfBenefit
#
# FHIR R4 Coverage.status uses FinancialResourceStatusCodes (Required):
#   active           — coverage is in force; strongest validity signal
#   cancelled        — explicitly terminated; strong invalidity signal
#   draft            — pre-decisional; genuinely uncertain, not invalid
#   entered-in-error — data integrity compromised; high uncertainty

COVERAGE_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,
    "cancelled": 0.10,
    "draft": 0.50,
    "entered-in-error": 0.50,
}

COVERAGE_STATUS_UNCERTAINTY: dict[str, float] = {
    "active": 0.15,
    "cancelled": 0.15,
    "draft": 0.40,
    "entered-in-error": 0.70,
}


# ═══════════════════════════════════════════════════════════════════
# Phase 7B: US Core Completeness — DocumentReference
# ═══════════════════════════════════════════════════════════════════

# ── DocumentReference status → reference validity confidence ──────
#
# Proposition: "This document reference is valid and the underlying
# document is reliable."
#
# DocumentReference is unique in FHIR R4: it has TWO independent
# status dimensions (status + docStatus) producing genuinely
# different epistemic semantics.
#
# Signal 1 — ``status`` (DocumentReferenceStatus, Required binding):
#     The status of the *reference record* itself.
#     current=0.85 — active, valid reference (analogous to "active")
#     superseded=0.25 — replaced by newer version; this specific
#         reference should not be relied upon, but the document
#         it points to may still be valid (hence not as low as 0.10)
#     entered-in-error=0.50 — data integrity compromised

DOC_REF_STATUS_PROBABILITY: dict[str, float] = {
    "current": 0.85,
    "superseded": 0.25,
    "entered-in-error": 0.50,
}

DOC_REF_STATUS_UNCERTAINTY: dict[str, float] = {
    "current": 0.15,
    "superseded": 0.20,
    "entered-in-error": 0.70,
}

# ── DocumentReference docStatus → uncertainty multiplier ──────────
#
# Signal 2 — ``docStatus`` (CompositionStatus, Required binding):
#     The status of the *underlying document content*.
#     Multipliers modulate the base uncertainty from Signal 1.
#
#     final=×0.7 — reviewed and signed off; strongest maturity
#     amended=×0.8 — revised but less settled than final
#     preliminary=×1.4 — draft; may contain errors or be incomplete
#     entered-in-error=×2.0 — content itself is invalid
#
# The separation from status allows expressing states like
# "current reference to a preliminary document" — genuinely
# different from "current reference to a final document."

DOC_REF_DOC_STATUS_MULTIPLIER: dict[str, float] = {
    "final": 0.7,
    "amended": 0.8,
    "preliminary": 1.4,
    "entered-in-error": 2.0,
}


# ── Supported resource types ──────────────────────────────────────

SUPPORTED_RESOURCE_TYPES = frozenset({
    # Phase 1
    "RiskAssessment",
    "Observation",
    "DiagnosticReport",
    "Condition",
    # Phase 2
    "AllergyIntolerance",
    "MedicationStatement",
    "ClinicalImpression",
    # Phase 3
    "DetectedIssue",
    "Immunization",
    "FamilyMemberHistory",
    "Procedure",
    # Phase 4
    "Consent",
    "Provenance",
    # Phase 6 — universal coverage
    "Encounter",
    "MedicationRequest",
    "CarePlan",
    "Goal",
    "CareTeam",
    "ImagingStudy",
    "Patient",
    "Organization",
    "Practitioner",
    "Device",
    "Claim",
    "ExplanationOfBenefit",
    "MedicationAdministration",
    # Phase 7A — high-value clinical expansion
    "ServiceRequest",
    "QuestionnaireResponse",
    "Specimen",
    # Phase 7B — US Core completeness
    "DocumentReference",
    "Coverage",
})
