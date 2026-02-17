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
})
