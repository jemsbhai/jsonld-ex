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
})
