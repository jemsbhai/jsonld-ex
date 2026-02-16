"""
FHIR R4 Interoperability for JSON-LD-Ex.

Bidirectional mapping between FHIR R4 clinical resources and jsonld-ex's
Subjective Logic opinion model.  Provides mathematically grounded
uncertainty that composes, fuses, and propagates correctly — capabilities
that FHIR's scalar probability and categorical code model lack.

Supported FHIR R4 resources:
  Phase 1:
  - RiskAssessment      — prediction.probabilityDecimal → Opinion
  - Observation          — interpretation / valueQuantity → Opinion
  - DiagnosticReport     — aggregates Observations via fusion
  - Condition            — verificationStatus → continuous Opinion
  Phase 2:
  - AllergyIntolerance   — verificationStatus + criticality → dual Opinion
  - MedicationStatement  — status → adherence confidence Opinion
  - ClinicalImpression   — status + findings → assessment Opinion
  Phase 3:
  - DetectedIssue        — severity → alert confidence Opinion
  - Immunization         — status → seroconversion base confidence Opinion
  - FamilyMemberHistory  — status → reported-vs-confirmed evidence Opinion
  - Procedure            — status + outcome/complication/followUp → Opinion

Architecture notes:
  - All public functions accept a ``fhir_version`` parameter (default "R4")
    so that R5/R6 support can be added without breaking existing callers.
  - Version-specific logic is isolated in private ``_*_r4()`` functions.
  - The SL opinion is embedded in FHIR resources via the standard extension
    mechanism, ensuring zero breaking changes to existing FHIR infrastructure.

Round-trip fidelity scope:
  The ``from_fhir() → to_fhir()`` round-trip preserves:
    - Resource id, status, and type
    - Opinion tuples (exact when via extension, reconstructed otherwise)
    - Key metadata: vaccineCode, occurrenceDateTime, relationship,
      outcome text, statusReason, implicated references
  It does NOT preserve raw FHIR arrays that were used to *infer* the
  opinion (evidence[], protocolApplied[], condition[], complication[],
  followUp[], derivedFrom[], etc.).  This is by design: these arrays
  inform the uncertainty budget during ``from_fhir()`` and their
  evidential weight is encoded in the resulting opinion.  Systems
  needing the original FHIR arrays should retain the source resource.

References:
  - HL7 FHIR R4: https://hl7.org/fhir/R4/
  - Jøsang, A. (2016). Subjective Logic. Springer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    robust_fuse,
    pairwise_conflict,
)
from jsonld_ex.confidence_decay import decay_opinion
from jsonld_ex.owl_interop import ConversionReport

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

_STATUS_UNCERTAINTY_MULTIPLIER: dict[str, float] = {
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


# ═══════════════════════════════════════════════════════════════════
# SCALAR-TO-OPINION RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


def scalar_to_opinion(
    probability: float,
    *,
    status: Optional[str] = None,
    basis_count: Optional[int] = None,
    method: Optional[str] = None,
    default_uncertainty: float = 0.15,
    base_rate: Optional[float] = None,
) -> Opinion:
    """Reconstruct an SL opinion from a FHIR scalar probability.

    FHIR represents risk/likelihood as a bare decimal (e.g.
    ``RiskAssessment.prediction.probabilityDecimal = 0.87``).  This
    discards epistemic state: *how much* evidence supports that number?
    Do sources agree?  Is this preliminary or final?

    This function recovers that lost epistemic state by analysing
    metadata signals that FHIR already provides but does not
    structurally exploit:

    - **status** — resource lifecycle stage (final, preliminary, etc.)
    - **basis_count** — number of evidence references cited
    - **method** — whether an evaluation method is documented

    Each signal adjusts the uncertainty budget multiplicatively.
    The remaining mass ``(1 - u)`` is distributed between belief and
    disbelief proportionally to ``probability`` and ``1 - probability``.

    When ``base_rate`` is not explicitly provided, it defaults to the
    input ``probability``.  This ensures that the projected probability
    ``P(ω) = b + a·u`` exactly equals the original scalar, preserving
    semantic fidelity while adding uncertainty information.

    Args:
        probability: Scalar probability in [0, 1].
        status: FHIR resource status code (e.g. "final", "preliminary").
            ``None`` means no status signal — no adjustment applied.
        basis_count: Number of evidence references (``RiskAssessment.basis``,
            etc.).  ``None`` means unknown — no adjustment.  ``0`` means
            explicitly no evidence cited — increases uncertainty.
        method: Evaluation method name.  If truthy, indicates a documented
            methodology — reduces uncertainty.  ``None`` means absent.
        default_uncertainty: Base uncertainty budget before adjustments.
            Must be in [0, 1).  Default 0.15.
        base_rate: Prior probability for the Opinion.  Default ``None``
            means use ``probability`` as the base rate (preserves
            projected probability exactly).

    Returns:
        An ``Opinion`` with the reconstructed epistemic state.

    Raises:
        ValueError: If ``probability`` is outside [0, 1], or
            ``default_uncertainty`` is outside [0, 1), or
            ``basis_count`` is negative.
        TypeError: If ``probability`` is not numeric.
    """
    # ── Input validation ──────────────────────────────────────────
    if not isinstance(probability, (int, float)):
        raise TypeError(
            f"probability must be a number, got: {type(probability).__name__}"
        )
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got: {probability}")

    if not isinstance(default_uncertainty, (int, float)):
        raise TypeError(
            f"default_uncertainty must be a number, got: "
            f"{type(default_uncertainty).__name__}"
        )
    if default_uncertainty < 0.0:
        raise ValueError(
            f"default_uncertainty must be >= 0, got: {default_uncertainty}"
        )
    if default_uncertainty >= 1.0:
        raise ValueError(
            f"default_uncertainty must be < 1.0, got: {default_uncertainty}"
        )

    if basis_count is not None and basis_count < 0:
        raise ValueError(f"basis_count must be >= 0, got: {basis_count}")

    # ── Compute adjusted uncertainty ──────────────────────────────
    u = float(default_uncertainty)

    # Status signal
    if status is not None:
        multiplier = _STATUS_UNCERTAINTY_MULTIPLIER.get(status, 1.0)
        u *= multiplier

    # Basis count signal
    if basis_count is not None:
        if basis_count == 0:
            u *= 1.5
        elif basis_count >= 3:
            u *= 0.7
        else:
            # 1–2 evidence items: modest reduction
            u *= 0.9

    # Method signal
    if method:
        u *= 0.8

    # Clamp uncertainty to valid range [0, 1)
    # Must be strictly < 1.0 to leave mass for b and d.
    u = max(0.0, min(u, 0.99))

    # ── Distribute remaining mass ─────────────────────────────────
    #
    # The (1 - u) mass is split between belief and disbelief
    # proportionally to probability and (1 - probability).
    p = float(probability)
    remaining = 1.0 - u
    b = p * remaining
    d = (1.0 - p) * remaining

    # ── Determine base rate ───────────────────────────────────────
    #
    # Default: base_rate = probability.  This ensures P(ω) = b + a·u
    # exactly equals the original probability, because:
    #   P(ω) = p·(1-u) + p·u = p
    #
    # When a custom base_rate is provided, P(ω) may differ from p.
    # This is intentional — the caller is asserting a different prior.
    a = base_rate if base_rate is not None else p

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


# ═══════════════════════════════════════════════════════════════════
# FHIR EXTENSION SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def opinion_to_fhir_extension(opinion: Opinion) -> dict[str, Any]:
    """Serialize an SL opinion to a FHIR R4 complex extension.

    Produces a structure conforming to FHIR R4 §2.4 (Extensions):

    .. code-block:: json

        {
          "url": "https://jsonld-ex.github.io/ns/fhir/opinion",
          "extension": [
            {"url": "belief",      "valueDecimal": 0.65},
            {"url": "disbelief",   "valueDecimal": 0.15},
            {"url": "uncertainty", "valueDecimal": 0.20},
            {"url": "baseRate",    "valueDecimal": 0.50}
          ]
        }

    This extension can be attached to any FHIR element using the
    ``_<element>`` convention (e.g. ``_probabilityDecimal``).  Systems
    that do not recognise the extension will silently ignore it.

    Args:
        opinion: The SL opinion to serialize.

    Returns:
        A dict representing a valid FHIR R4 complex extension.
    """
    return {
        "url": FHIR_EXTENSION_URL,
        "extension": [
            {"url": "belief", "valueDecimal": opinion.belief},
            {"url": "disbelief", "valueDecimal": opinion.disbelief},
            {"url": "uncertainty", "valueDecimal": opinion.uncertainty},
            {"url": "baseRate", "valueDecimal": opinion.base_rate},
        ],
    }


def fhir_extension_to_opinion(extension: dict[str, Any]) -> Opinion:
    """Deserialize a FHIR R4 complex extension to an SL opinion.

    Reverses ``opinion_to_fhir_extension()``.  The extension must have
    the correct URL and contain sub-extensions for at least ``belief``,
    ``disbelief``, and ``uncertainty``.  ``baseRate`` defaults to 0.5
    if absent.

    Args:
        extension: A dict representing a FHIR R4 complex extension.

    Returns:
        The deserialized ``Opinion``.

    Raises:
        ValueError: If the extension URL does not match, or required
            sub-extensions (belief, disbelief, uncertainty) are missing.
    """
    url = extension.get("url")
    if url != FHIR_EXTENSION_URL:
        raise ValueError(
            f"Expected extension URL '{FHIR_EXTENSION_URL}', "
            f"got '{url}'"
        )

    # Parse sub-extensions into a lookup dict
    values: dict[str, float] = {}
    for sub in extension.get("extension", []):
        sub_url = sub.get("url")
        if sub_url and "valueDecimal" in sub:
            values[sub_url] = float(sub["valueDecimal"])

    # Validate required fields
    required = {"belief", "disbelief", "uncertainty"}
    missing = required - values.keys()
    if missing:
        raise ValueError(
            f"Missing required sub-extension(s): {sorted(missing)}"
        )

    return Opinion(
        belief=values["belief"],
        disbelief=values["disbelief"],
        uncertainty=values["uncertainty"],
        base_rate=values.get("baseRate", 0.5),
    )


# ═══════════════════════════════════════════════════════════════════
# FHIR → JSONLD-EX IMPORT  (from_fhir)
# ═══════════════════════════════════════════════════════════════════

# ── Categorical-to-probability mappings ────────────────────────────
#
# FHIR Condition.verificationStatus and Observation.interpretation
# use categorical codes, not probabilities.  We map each to a scalar
# that feeds into scalar_to_opinion(), with a deliberately wider
# default uncertainty budget to reflect the categorical origin.
#
# Rationale for each mapping is documented in the NeurIPS paper
# Section 5 (FHIR Interoperability).

_VERIFICATION_STATUS_PROBABILITY: dict[str, float] = {
    "confirmed": 0.95,
    "unconfirmed": 0.55,
    "provisional": 0.60,
    "differential": 0.40,
    "refuted": 0.05,
    "entered-in-error": 0.50,
}

_VERIFICATION_STATUS_UNCERTAINTY: dict[str, float] = {
    "confirmed": 0.10,
    "unconfirmed": 0.35,
    "provisional": 0.35,
    "differential": 0.45,
    "refuted": 0.10,
    "entered-in-error": 0.70,
}

_INTERPRETATION_PROBABILITY: dict[str, float] = {
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
#
# criticality codes represent the potential clinical harm of the
# allergy reaction.  We map to probability-of-severity:
#   high          → 0.90  (life-threatening potential)
#   low           → 0.30  (minor inconvenience)
#   unable-to-assess → 0.50  (maximum uncertainty)

_CRITICALITY_PROBABILITY: dict[str, float] = {
    "high": 0.90,
    "low": 0.30,
    "unable-to-assess": 0.50,
}

_CRITICALITY_UNCERTAINTY: dict[str, float] = {
    "high": 0.15,
    "low": 0.15,
    "unable-to-assess": 0.45,
}

# ── Phase 2: MedicationStatement status → adherence confidence ────
#
# MedicationStatement.status captures whether the patient is/was
# actually taking a medication.  This is inherently uncertain —
# the information may come from patient self-report, pharmacy
# fill data, or clinical observation, each with different reliability.

_MEDSTMT_STATUS_PROBABILITY: dict[str, float] = {
    "active": 0.85,           # currently taking
    "completed": 0.90,        # finished medication course
    "entered-in-error": 0.50, # data integrity compromised
    "intended": 0.60,         # planned but not yet started
    "stopped": 0.10,          # discontinued
    "on-hold": 0.40,          # temporarily paused
    "unknown": 0.50,          # no information
    "not-taken": 0.05,        # explicitly not taking
}

_MEDSTMT_STATUS_UNCERTAINTY: dict[str, float] = {
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
#
# ClinicalImpression.status reflects assessment completeness.
# We use the same multiplier pattern as the general status map.

_CLINICAL_IMPRESSION_STATUS_MULTIPLIER: dict[str, float] = {
    "completed": 0.5,         # assessment finalised
    "in-progress": 2.0,      # still evaluating
    "entered-in-error": 5.0, # data integrity compromised
}

# ── Phase 3: DetectedIssue severity → alert confidence ────────────
#
# severity codes represent how clinically significant the detected
# issue is.  We map to probability-of-clinical-significance:
#   high     → 0.90  (likely clinically significant, warrants action)
#   moderate → 0.60  (possibly significant, review recommended)
#   low      → 0.30  (unlikely significant, informational)

_DETECTED_ISSUE_SEVERITY_PROBABILITY: dict[str, float] = {
    "high": 0.90,
    "moderate": 0.60,
    "low": 0.30,
}

_DETECTED_ISSUE_SEVERITY_UNCERTAINTY: dict[str, float] = {
    "high": 0.15,
    "moderate": 0.25,
    "low": 0.20,
}

# ── Phase 3: Immunization status → seroconversion confidence ─────
#
# Immunization.status captures whether the vaccine was administered.
# The opinion proposition: "patient has been effectively immunized."
#   completed        → 0.85  (vaccine administered)
#   not-done         → 0.05  (vaccine not administered)
#   entered-in-error → 0.50  (data integrity compromised)

_IMMUNIZATION_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.85,
    "not-done": 0.05,
    "entered-in-error": 0.50,
}

_IMMUNIZATION_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.15,
    "not-done": 0.10,
    "entered-in-error": 0.70,
}

# ── Phase 3: FamilyMemberHistory ──────────────────────────────────
#
# Family history is inherently uncertain — patient-reported data
# subject to recall bias, incomplete knowledge, and communication
# errors.  The base uncertainty is intentionally higher than other
# resources to reflect this epistemic reality.

FAMILY_HISTORY_DEFAULT_UNCERTAINTY: float = 0.30
"""Default uncertainty budget for FamilyMemberHistory opinions.

Higher than the standard 0.15 default to reflect the inherent
unreliability of patient-reported family health history.  Recall
bias, incomplete knowledge of family members' health, and
communication errors all contribute to elevated baseline uncertainty.
"""

_FAMILY_HISTORY_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.70,        # full history reported
    "partial": 0.50,          # incomplete history
    "health-unknown": 0.30,   # family health not known
    "entered-in-error": 0.50, # data integrity compromised
}

_FAMILY_HISTORY_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.25,
    "partial": 0.40,
    "health-unknown": 0.55,
    "entered-in-error": 0.70,
}

# ── Phase 3: Procedure status → outcome confidence ───────────────
#
# Procedure.status captures the lifecycle of a clinical procedure.
# The opinion proposition: "the procedure was successfully performed."
#   completed        → 0.85  (procedure done)
#   in-progress      → 0.60  (still underway, outcome uncertain)
#   not-done         → 0.05  (procedure not performed)
#   entered-in-error → 0.50  (data integrity compromised)
#   preparation      → 0.50  (not yet started)
#   on-hold          → 0.40  (paused)
#   stopped          → 0.20  (discontinued before completion)
#   unknown          → 0.50  (no information)

_PROCEDURE_STATUS_PROBABILITY: dict[str, float] = {
    "completed": 0.85,
    "in-progress": 0.60,
    "not-done": 0.05,
    "entered-in-error": 0.50,
    "preparation": 0.50,
    "on-hold": 0.40,
    "stopped": 0.20,
    "unknown": 0.50,
}

_PROCEDURE_STATUS_UNCERTAINTY: dict[str, float] = {
    "completed": 0.15,
    "in-progress": 0.35,
    "not-done": 0.10,
    "entered-in-error": 0.70,
    "preparation": 0.40,
    "on-hold": 0.30,
    "stopped": 0.20,
    "unknown": 0.45,
}


_SUPPORTED_RESOURCE_TYPES = frozenset({
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
})


def from_fhir(
    resource: dict[str, Any],
    *,
    fhir_version: str = "R4",
) -> tuple[dict[str, Any], ConversionReport]:
    """Import a FHIR R4 resource into a jsonld-ex document with SL opinions.

    Extracts scalar probabilities or categorical codes from the resource,
    reconstructs Subjective Logic opinions from available metadata signals,
    and recovers exact opinions from jsonld-ex FHIR extensions when present.

    Supported resource types (Phase 1):
        - **RiskAssessment** — ``prediction.probabilityDecimal``
        - **Observation** — ``interpretation`` code
        - **DiagnosticReport** — ``conclusion`` with optional extension
        - **Condition** — ``verificationStatus`` code

    The returned document contains an ``opinions`` list where each entry
    has:
        - ``field`` — dotted path to the source FHIR field
        - ``value`` — the original scalar or categorical value
        - ``opinion`` — reconstructed or recovered ``Opinion``
        - ``source`` — ``"extension"`` if recovered from a jsonld-ex
          FHIR extension, ``"reconstructed"`` if inferred from metadata

    Args:
        resource: A FHIR R4 resource as a Python dict (JSON-parsed).
        fhir_version: FHIR version string.  Currently only ``"R4"``
            is supported.

    Returns:
        A tuple of ``(jsonld_ex_doc, ConversionReport)``.

    Raises:
        ValueError: If ``resource`` lacks ``resourceType``, or
            ``fhir_version`` is not supported.
    """
    # ── Validate inputs ───────────────────────────────────────────
    if "resourceType" not in resource:
        raise ValueError(
            "FHIR resource must contain a 'resourceType' field"
        )

    if fhir_version not in SUPPORTED_FHIR_VERSIONS:
        raise ValueError(
            f"Unsupported FHIR version '{fhir_version}'. "
            f"Supported: {', '.join(SUPPORTED_FHIR_VERSIONS)}"
        )

    resource_type = resource["resourceType"]

    # ── Dispatch to resource-specific handler ─────────────────────
    if resource_type not in _SUPPORTED_RESOURCE_TYPES:
        report = ConversionReport(
            success=True,
            nodes_converted=0,
            warnings=[
                f"Unsupported FHIR resource type '{resource_type}'. "
                f"Supported types: {', '.join(sorted(_SUPPORTED_RESOURCE_TYPES))}"
            ],
        )
        return {"@type": f"fhir:{resource_type}", "opinions": []}, report

    handler = _RESOURCE_HANDLERS[resource_type]
    return handler(resource)


# ── Extension recovery helper ─────────────────────────────────────


def _try_recover_opinion(
    element_extensions: Optional[dict[str, Any]],
) -> Optional[Opinion]:
    """Try to recover an Opinion from a FHIR element's extension list.

    Looks for a jsonld-ex opinion extension in the ``_<element>``
    extension structure.  Returns ``None`` if no matching extension
    is found.

    Args:
        element_extensions: The ``_<element>`` dict containing an
            ``"extension"`` list, or ``None``.

    Returns:
        Recovered ``Opinion`` or ``None``.
    """
    if not element_extensions:
        return None

    extensions = element_extensions.get("extension", [])
    for ext in extensions:
        if ext.get("url") == FHIR_EXTENSION_URL:
            return fhir_extension_to_opinion(ext)

    return None


# ── RiskAssessment handler (from_fhir) ────────────────────────────


def _from_risk_assessment_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 RiskAssessment → jsonld-ex document."""
    status = resource.get("status")
    method_text = None
    method_obj = resource.get("method")
    if isinstance(method_obj, dict):
        method_text = method_obj.get("text")

    basis = resource.get("basis")
    basis_count = len(basis) if basis is not None else None

    predictions = resource.get("prediction", [])
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    for i, pred in enumerate(predictions):
        prob = pred.get("probabilityDecimal")
        if prob is None:
            continue

        # Try extension recovery first
        pred_ext = pred.get("_probabilityDecimal")
        recovered = _try_recover_opinion(pred_ext)

        if recovered is not None:
            opinions.append({
                "field": f"prediction[{i}].probabilityDecimal",
                "value": prob,
                "opinion": recovered,
                "source": "extension",
            })
        else:
            op = scalar_to_opinion(
                prob,
                status=status,
                basis_count=basis_count,
                method=method_text,
            )
            opinions.append({
                "field": f"prediction[{i}].probabilityDecimal",
                "value": prob,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:RiskAssessment",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Observation handler (from_fhir) ──────────────────────────────


def _from_observation_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Observation → jsonld-ex document."""
    status = resource.get("status")
    interpretations = resource.get("interpretation", [])
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    for interp in interpretations:
        codings = interp.get("coding", [])
        for coding in codings:
            code = coding.get("code")
            if code is None:
                continue

            # Try extension recovery
            interp_ext = resource.get("_interpretation")
            recovered = _try_recover_opinion(interp_ext)

            if recovered is not None:
                opinions.append({
                    "field": "interpretation",
                    "value": code,
                    "opinion": recovered,
                    "source": "extension",
                })
            else:
                # Map categorical code to probability
                prob = _INTERPRETATION_PROBABILITY.get(code, 0.50)
                op = scalar_to_opinion(
                    prob,
                    status=status,
                    default_uncertainty=0.20,
                )
                opinions.append({
                    "field": "interpretation",
                    "value": code,
                    "opinion": op,
                    "source": "reconstructed",
                })
            nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:Observation",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    # Preserve valueQuantity for downstream use
    value_qty = resource.get("valueQuantity")
    if value_qty is not None:
        doc["value"] = value_qty

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── DiagnosticReport handler (from_fhir) ─────────────────────────


def _from_diagnostic_report_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 DiagnosticReport → jsonld-ex document."""
    status = resource.get("status")
    conclusion = resource.get("conclusion")
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Extract result references for downstream fusion
    results = resource.get("result", [])
    result_refs = [
        r.get("reference", "") for r in results if isinstance(r, dict)
    ]

    # Try extension on conclusion
    if conclusion is not None:
        conclusion_ext = resource.get("_conclusion")
        recovered = _try_recover_opinion(conclusion_ext)

        if recovered is not None:
            opinions.append({
                "field": "conclusion",
                "value": conclusion,
                "opinion": recovered,
                "source": "extension",
            })
            nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:DiagnosticReport",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
        "result_references": result_refs,
    }
    if conclusion is not None:
        doc["conclusion"] = conclusion

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Condition handler (from_fhir) ─────────────────────────────────


def _from_condition_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Condition → jsonld-ex document.

    Maps ``verificationStatus`` to a continuous SL Opinion.  FHIR's
    categorical codes (confirmed / provisional / differential / refuted)
    are mapped to scalar probabilities and then to opinions using
    ``scalar_to_opinion()``, with the evidence list influencing the
    uncertainty budget.
    """
    # Extract verificationStatus code
    vs_obj = resource.get("verificationStatus", {})
    vs_code = None
    for coding in vs_obj.get("coding", []):
        vs_code = coding.get("code")
        if vs_code:
            break

    # Extract clinicalStatus code
    cs_obj = resource.get("clinicalStatus", {})
    cs_code = None
    for coding in cs_obj.get("coding", []):
        cs_code = coding.get("code")
        if cs_code:
            break

    # Count evidence items (each item may have detail references)
    evidence = resource.get("evidence")
    evidence_count = len(evidence) if evidence is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    if vs_code is not None:
        # Try extension recovery
        vs_ext = resource.get("_verificationStatus")
        recovered = _try_recover_opinion(vs_ext)

        if recovered is not None:
            opinions.append({
                "field": "verificationStatus",
                "value": vs_code,
                "opinion": recovered,
                "source": "extension",
            })
        else:
            prob = _VERIFICATION_STATUS_PROBABILITY.get(vs_code, 0.50)
            default_u = _VERIFICATION_STATUS_UNCERTAINTY.get(vs_code, 0.30)
            op = scalar_to_opinion(
                prob,
                basis_count=evidence_count,
                default_uncertainty=default_u,
            )
            opinions.append({
                "field": "verificationStatus",
                "value": vs_code,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:Condition",
        "id": resource.get("id"),
        "clinicalStatus": cs_code,
        "opinions": opinions,
    }

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── AllergyIntolerance handler (from_fhir) ───────────────────────


def _from_allergy_intolerance_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 AllergyIntolerance → jsonld-ex document.

    AllergyIntolerance has two opinion-producing dimensions:

    1. **verificationStatus** — same categorical codes as Condition
       (confirmed / unconfirmed / presumed / refuted / entered-in-error).
       Reuses ``_VERIFICATION_STATUS_PROBABILITY`` mappings.

    2. **criticality** — the potential clinical harm if the patient is
       exposed (high / low / unable-to-assess).  This produces a
       *separate* opinion about severity, distinct from the
       verification opinion about *whether* the allergy exists.
    """
    # Extract verificationStatus code
    vs_obj = resource.get("verificationStatus", {})
    vs_code = None
    for coding in vs_obj.get("coding", []):
        vs_code = coding.get("code")
        if vs_code:
            break

    # Extract clinicalStatus code
    cs_obj = resource.get("clinicalStatus", {})
    cs_code = None
    for coding in cs_obj.get("coding", []):
        cs_code = coding.get("code")
        if cs_code:
            break

    criticality = resource.get("criticality")

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # ── verificationStatus opinion ────────────────────────────────
    if vs_code is not None:
        vs_ext = resource.get("_verificationStatus")
        recovered = _try_recover_opinion(vs_ext)

        if recovered is not None:
            opinions.append({
                "field": "verificationStatus",
                "value": vs_code,
                "opinion": recovered,
                "source": "extension",
            })
        else:
            prob = _VERIFICATION_STATUS_PROBABILITY.get(vs_code, 0.50)
            default_u = _VERIFICATION_STATUS_UNCERTAINTY.get(vs_code, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "verificationStatus",
                "value": vs_code,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    # ── criticality opinion ───────────────────────────────────────
    if criticality is not None:
        crit_prob = _CRITICALITY_PROBABILITY.get(criticality, 0.50)
        crit_u = _CRITICALITY_UNCERTAINTY.get(criticality, 0.30)
        crit_op = scalar_to_opinion(crit_prob, default_uncertainty=crit_u)
        opinions.append({
            "field": "criticality",
            "value": criticality,
            "opinion": crit_op,
            "source": "reconstructed",
        })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:AllergyIntolerance",
        "id": resource.get("id"),
        "clinicalStatus": cs_code,
        "opinions": opinions,
    }

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── MedicationStatement handler (from_fhir) ──────────────────────


def _from_medication_statement_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 MedicationStatement → jsonld-ex document.

    MedicationStatement.status maps to adherence confidence — the
    degree to which we believe the patient is/was actually taking
    the medication.  This is inherently uncertain because FHIR
    ``MedicationStatement`` data often comes from patient self-report.

    Additional metadata signals:

    - **informationSource** — Practitioner-reported data is more
      reliable than patient self-report (reduces uncertainty).
    - **derivedFrom** — supporting evidence (MedicationRequest,
      MedicationDispense, Observation) reduces uncertainty.
    """
    status = resource.get("status", "unknown")

    # Extract informationSource reference type
    info_source = resource.get("informationSource")
    source_type = None
    if isinstance(info_source, dict):
        ref = info_source.get("reference", "")
        source_type = ref.split("/")[0] if "/" in ref else None

    # Count derivedFrom references (supporting evidence)
    derived = resource.get("derivedFrom")
    derived_count = len(derived) if derived is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery
    status_ext = resource.get("_status")
    recovered = _try_recover_opinion(status_ext)

    if recovered is not None:
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        prob = _MEDSTMT_STATUS_PROBABILITY.get(status, 0.50)
        default_u = _MEDSTMT_STATUS_UNCERTAINTY.get(status, 0.30)

        # Information source signal
        # Practitioner/PractitionerRole → more reliable
        # Patient/RelatedPerson → less reliable (recall bias)
        if source_type in ("Practitioner", "PractitionerRole"):
            default_u *= 0.7
        elif source_type in ("Patient", "RelatedPerson"):
            default_u *= 1.3

        op = scalar_to_opinion(
            prob,
            basis_count=derived_count,
            default_uncertainty=default_u,
        )
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Extract medication text
    med_text = None
    med_cc = resource.get("medicationCodeableConcept")
    if isinstance(med_cc, dict):
        med_text = med_cc.get("text")

    doc: dict[str, Any] = {
        "@type": "fhir:MedicationStatement",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if med_text is not None:
        doc["medication"] = med_text

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── ClinicalImpression handler (from_fhir) ───────────────────────


def _from_clinical_impression_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 ClinicalImpression → jsonld-ex document.

    ClinicalImpression represents a practitioner's overall clinical
    assessment.  The opinion is derived from:

    - **status** — ``completed`` vs ``in-progress`` vs
      ``entered-in-error`` (modulates uncertainty).
    - **finding[]** — each finding (coded or reference) counts as
      supporting evidence, reducing uncertainty.
    - **summary** — preserved for downstream use and can carry
      an extension for opinion recovery.
    """
    status = resource.get("status", "in-progress")
    summary = resource.get("summary")
    findings = resource.get("finding", [])
    finding_count = len(findings) if findings else 0

    # Extract finding references for downstream
    finding_refs: list[str] = []
    for f in findings:
        item_ref = f.get("itemReference")
        if isinstance(item_ref, dict):
            ref = item_ref.get("reference")
            if ref:
                finding_refs.append(ref)

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Determine the opinion field — prefer summary if present
    opinion_field = "summary" if summary is not None else "status"

    # Try extension recovery on summary
    if summary is not None:
        summary_ext = resource.get("_summary")
        recovered = _try_recover_opinion(summary_ext)

        if recovered is not None:
            opinions.append({
                "field": "summary",
                "value": summary,
                "opinion": recovered,
                "source": "extension",
            })
            nodes_converted += 1

    # If no extension recovered, reconstruct from status + findings
    if not opinions:
        # Base uncertainty for a clinical assessment
        base_u = 0.20

        # Status multiplier
        multiplier = _CLINICAL_IMPRESSION_STATUS_MULTIPLIER.get(status, 1.0)
        adjusted_u = base_u * multiplier

        # Finding count reduces uncertainty (evidence signal)
        if finding_count == 0:
            adjusted_u *= 1.3
        elif finding_count >= 3:
            adjusted_u *= 0.7
        else:
            adjusted_u *= 0.9

        # Clamp
        adjusted_u = max(0.0, min(adjusted_u, 0.99))

        # A completed clinical impression with findings suggests
        # moderate-to-high belief in the assessment
        prob = 0.75
        op = scalar_to_opinion(prob, default_uncertainty=adjusted_u)
        opinions.append({
            "field": opinion_field,
            "value": summary if summary is not None else status,
            "opinion": op,
            "source": "reconstructed",
        })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:ClinicalImpression",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if summary is not None:
        doc["summary"] = summary
    if finding_refs:
        doc["finding_references"] = finding_refs

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── DetectedIssue handler (from_fhir) ─────────────────────────────


def _from_detected_issue_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 DetectedIssue → jsonld-ex document.

    DetectedIssue captures clinical alerts such as drug-drug
    interactions, duplicate therapy, and dose range violations.
    The opinion proposition is: "this detected issue is clinically
    significant and warrants action."

    Primary field: ``severity`` (high/moderate/low) maps to
    probability of clinical significance.  ``status`` modulates
    uncertainty via the standard multiplier.  ``evidence[]`` count
    provides an evidence signal.
    """
    status = resource.get("status")
    severity = resource.get("severity")

    # Count evidence items
    evidence = resource.get("evidence")
    evidence_count = len(evidence) if evidence is not None else None

    # Capture implicated references
    implicated = resource.get("implicated", [])
    implicated_refs = [
        r.get("reference", "") for r in implicated if isinstance(r, dict)
    ]

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery on severity
    if severity is not None:
        sev_ext = resource.get("_severity")
        recovered = _try_recover_opinion(sev_ext)

        if recovered is not None:
            opinions.append({
                "field": "severity",
                "value": severity,
                "opinion": recovered,
                "source": "extension",
            })
        else:
            prob = _DETECTED_ISSUE_SEVERITY_PROBABILITY.get(severity, 0.50)
            default_u = _DETECTED_ISSUE_SEVERITY_UNCERTAINTY.get(severity, 0.25)
            op = scalar_to_opinion(
                prob,
                status=status,
                basis_count=evidence_count,
                default_uncertainty=default_u,
            )
            opinions.append({
                "field": "severity",
                "value": severity,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1
    else:
        # No severity — produce a default opinion from status alone
        prob = 0.50
        default_u = 0.30
        op = scalar_to_opinion(
            prob,
            status=status,
            basis_count=evidence_count,
            default_uncertainty=default_u,
        )
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:DetectedIssue",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if implicated_refs:
        doc["implicated_references"] = implicated_refs

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Immunization handler (from_fhir) ──────────────────────────────


def _from_immunization_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Immunization → jsonld-ex document.

    Immunization captures vaccine administration events.  The opinion
    proposition is: "the patient has been effectively immunized."

    This produces the *base* seroconversion opinion only — temporal
    decay for waning immunity is applied separately via
    ``fhir_temporal_decay()``.  Some immunities are lifelong (e.g.
    MMR after full series), so decay is not baked in here.

    Primary field: ``status`` (completed/not-done/entered-in-error).
    ``protocolApplied[]`` dose count modulates uncertainty — more
    doses in a multi-dose series means stronger evidence of
    effective immunization.
    """
    status = resource.get("status", "completed")

    # Count protocol doses (evidence of series completion)
    protocol = resource.get("protocolApplied")
    dose_count = len(protocol) if protocol is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery
    status_ext = resource.get("_status")
    recovered = _try_recover_opinion(status_ext)

    if recovered is not None:
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        prob = _IMMUNIZATION_STATUS_PROBABILITY.get(status, 0.50)
        default_u = _IMMUNIZATION_STATUS_UNCERTAINTY.get(status, 0.30)

        # Protocol dose count signal: more doses → lower uncertainty
        if dose_count is not None:
            if dose_count == 0:
                default_u *= 1.3
            elif dose_count >= 3:
                default_u *= 0.7
            else:
                default_u *= 0.9

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Extract metadata for downstream use
    vaccine_code = None
    vc_obj = resource.get("vaccineCode")
    if isinstance(vc_obj, dict):
        vaccine_code = vc_obj.get("text")

    status_reason = None
    sr_obj = resource.get("statusReason")
    if isinstance(sr_obj, dict):
        status_reason = sr_obj.get("text")

    occurrence_dt = resource.get("occurrenceDateTime")

    doc: dict[str, Any] = {
        "@type": "fhir:Immunization",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if vaccine_code is not None:
        doc["vaccineCode"] = vaccine_code
    if status_reason is not None:
        doc["statusReason"] = status_reason
    if occurrence_dt is not None:
        doc["occurrenceDateTime"] = occurrence_dt

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── FamilyMemberHistory handler (from_fhir) ───────────────────────


def _from_family_member_history_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 FamilyMemberHistory → jsonld-ex document.

    FamilyMemberHistory is inherently uncertain — it captures
    patient-reported family health history.  Self-reported data is
    subject to recall bias, incomplete knowledge, and communication
    errors.

    The base uncertainty budget is ``FAMILY_HISTORY_DEFAULT_UNCERTAINTY``
    (0.30), intentionally higher than the standard 0.15 default.

    Primary field: ``status`` (completed/partial/health-unknown/
    entered-in-error).  ``condition[]`` count provides an evidence
    signal (more reported conditions → more complete picture).
    ``dataAbsentReason`` explicitly marks missing data and increases
    uncertainty.
    """
    status = resource.get("status", "completed")

    # Count conditions (evidence of completeness)
    conditions = resource.get("condition")
    condition_count = len(conditions) if conditions is not None else None

    # Check for dataAbsentReason
    data_absent = resource.get("dataAbsentReason")
    has_data_absent = data_absent is not None

    # Extract relationship
    rel_obj = resource.get("relationship", {})
    relationship_text = rel_obj.get("text") if isinstance(rel_obj, dict) else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery
    status_ext = resource.get("_status")
    recovered = _try_recover_opinion(status_ext)

    if recovered is not None:
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        prob = _FAMILY_HISTORY_STATUS_PROBABILITY.get(status, 0.50)
        default_u = _FAMILY_HISTORY_STATUS_UNCERTAINTY.get(
            status, FAMILY_HISTORY_DEFAULT_UNCERTAINTY
        )

        # Condition count signal
        if condition_count is not None:
            if condition_count == 0:
                default_u *= 1.3
            elif condition_count >= 3:
                default_u *= 0.7
            else:
                default_u *= 0.9

        # dataAbsentReason explicitly marks missing data
        if has_data_absent:
            default_u *= 1.3

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:FamilyMemberHistory",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if relationship_text is not None:
        doc["relationship"] = relationship_text

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Procedure handler (from_fhir) ─────────────────────────────────


def _from_procedure_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Procedure → jsonld-ex document.

    Procedure captures clinical procedures with outcome assessment.
    The opinion proposition is: "the procedure was successfully
    performed (or is being successfully performed)."

    Primary field: ``status`` (completed/in-progress/not-done/etc.).
    ``outcome`` presence reduces uncertainty (documented result).
    ``complication[]`` count increases uncertainty (ambiguous success).
    ``followUp[]`` count reduces uncertainty (better assessment
    quality from longer observation).
    """
    status = resource.get("status", "completed")

    # Outcome presence
    outcome_obj = resource.get("outcome")
    has_outcome = outcome_obj is not None
    outcome_text = None
    if isinstance(outcome_obj, dict):
        outcome_text = outcome_obj.get("text")

    # Complication count
    complications = resource.get("complication")
    complication_count = len(complications) if complications is not None else None

    # Follow-up count
    follow_ups = resource.get("followUp")
    followup_count = len(follow_ups) if follow_ups is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery
    status_ext = resource.get("_status")
    recovered = _try_recover_opinion(status_ext)

    if recovered is not None:
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        prob = _PROCEDURE_STATUS_PROBABILITY.get(status, 0.50)
        default_u = _PROCEDURE_STATUS_UNCERTAINTY.get(status, 0.30)

        # Outcome presence → documented result reduces uncertainty
        if has_outcome:
            default_u *= 0.7

        # Complication count → ambiguous success increases uncertainty.
        # We use discrete buckets rather than a continuous formula
        # (e.g., u×(1+0.15×n)) to avoid unbounded uncertainty growth:
        # a continuous formula with n=10 complications would yield
        # u×2.5, potentially pushing uncertainty above 1.0 before
        # clamping and distorting the opinion geometry.
        if complication_count is not None:
            if complication_count == 0:
                # Explicitly no complications → slight reduction
                default_u *= 0.9
            elif complication_count >= 2:
                default_u *= 1.5
            else:
                default_u *= 1.2

        # Follow-up count → better observation reduces uncertainty
        if followup_count is not None:
            if followup_count == 0:
                default_u *= 1.2
            elif followup_count >= 2:
                default_u *= 0.7
            else:
                default_u *= 0.9

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:Procedure",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if outcome_text is not None:
        doc["outcome"] = outcome_text

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── from_fhir handler dispatch table ──────────────────────────────

_RESOURCE_HANDLERS = {
    # Phase 1
    "RiskAssessment": _from_risk_assessment_r4,
    "Observation": _from_observation_r4,
    "DiagnosticReport": _from_diagnostic_report_r4,
    "Condition": _from_condition_r4,
    # Phase 2
    "AllergyIntolerance": _from_allergy_intolerance_r4,
    "MedicationStatement": _from_medication_statement_r4,
    "ClinicalImpression": _from_clinical_impression_r4,
    # Phase 3
    "DetectedIssue": _from_detected_issue_r4,
    "Immunization": _from_immunization_r4,
    "FamilyMemberHistory": _from_family_member_history_r4,
    "Procedure": _from_procedure_r4,
}


# ═══════════════════════════════════════════════════════════════════
# JSONLD-EX → FHIR EXPORT  (to_fhir)
# ═══════════════════════════════════════════════════════════════════


def to_fhir(
    doc: dict[str, Any],
    *,
    fhir_version: str = "R4",
) -> tuple[dict[str, Any], ConversionReport]:
    """Export a jsonld-ex document to a FHIR R4 resource.

    Reverses ``from_fhir()``.  Takes a jsonld-ex document (as produced
    by ``from_fhir()``) and produces a valid FHIR R4 resource with
    the SL opinions embedded as FHIR extensions.

    For resources with scalar probability fields (RiskAssessment),
    the opinion's projected probability ``P(ω) = b + a·u`` is used
    as the FHIR decimal value, ensuring that non-jsonld-ex-aware
    systems receive a usable probability.

    Args:
        doc: A jsonld-ex document dict with ``@type`` and ``opinions``.
        fhir_version: FHIR version string.  Currently only ``"R4"``
            is supported.

    Returns:
        A tuple of ``(fhir_resource, ConversionReport)``.

    Raises:
        ValueError: If ``doc`` lacks ``@type``, or the resource type
            is not supported, or ``fhir_version`` is not supported.
    """
    if "@type" not in doc:
        raise ValueError(
            "jsonld-ex document must contain an '@type' field"
        )

    if fhir_version not in SUPPORTED_FHIR_VERSIONS:
        raise ValueError(
            f"Unsupported FHIR version '{fhir_version}'. "
            f"Supported: {', '.join(SUPPORTED_FHIR_VERSIONS)}"
        )

    doc_type = doc["@type"]
    # Strip "fhir:" prefix
    if doc_type.startswith("fhir:"):
        resource_type = doc_type[5:]
    else:
        resource_type = doc_type

    if resource_type not in _TO_FHIR_HANDLERS:
        raise ValueError(
            f"Unsupported resource type '{resource_type}'. "
            f"Supported types: {', '.join(sorted(_TO_FHIR_HANDLERS))}"
        )

    handler = _TO_FHIR_HANDLERS[resource_type]
    return handler(doc)


# ── RiskAssessment handler (to_fhir) ─────────────────────────────


def _to_risk_assessment_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 RiskAssessment."""
    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    opinions = doc.get("opinions", [])
    predictions: list[dict[str, Any]] = []
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        pred: dict[str, Any] = {
            "probabilityDecimal": op.projected_probability(),
        }

        # Embed opinion as extension
        ext = opinion_to_fhir_extension(op)
        pred["_probabilityDecimal"] = {"extension": [ext]}

        predictions.append(pred)
        nodes_converted += 1

    resource["prediction"] = predictions

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── Observation handler (to_fhir) ────────────────────────────────


def _to_observation_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Observation."""
    resource: dict[str, Any] = {
        "resourceType": "Observation",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore valueQuantity
    if doc.get("value"):
        resource["valueQuantity"] = doc["value"]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        code = entry.get("value")

        # Rebuild interpretation array
        if code is not None:
            resource.setdefault("interpretation", []).append({
                "coding": [{"code": code}],
            })

            # Embed opinion as extension on _interpretation
            ext = opinion_to_fhir_extension(op)
            resource["_interpretation"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── DiagnosticReport handler (to_fhir) ───────────────────────────


def _to_diagnostic_report_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 DiagnosticReport."""
    resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]
    if doc.get("conclusion"):
        resource["conclusion"] = doc["conclusion"]

    # Restore result references
    result_refs = doc.get("result_references", [])
    if result_refs:
        resource["result"] = [
            {"reference": ref} for ref in result_refs
        ]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "conclusion":
            ext = opinion_to_fhir_extension(op)
            resource["_conclusion"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── Condition handler (to_fhir) ──────────────────────────────────


def _to_condition_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Condition."""
    resource: dict[str, Any] = {
        "resourceType": "Condition",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]

    # Restore clinicalStatus
    cs = doc.get("clinicalStatus")
    if cs:
        resource["clinicalStatus"] = {"coding": [{"code": cs}]}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")
        code = entry.get("value")

        if field == "verificationStatus" and code is not None:
            resource["verificationStatus"] = {
                "coding": [{"code": code}],
            }

            ext = opinion_to_fhir_extension(op)
            resource["_verificationStatus"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── AllergyIntolerance handler (to_fhir) ─────────────────────


def _to_allergy_intolerance_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 AllergyIntolerance."""
    resource: dict[str, Any] = {
        "resourceType": "AllergyIntolerance",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]

    # Restore clinicalStatus
    cs = doc.get("clinicalStatus")
    if cs:
        resource["clinicalStatus"] = {"coding": [{"code": cs}]}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")
        code = entry.get("value")

        if field_name == "verificationStatus" and code is not None:
            resource["verificationStatus"] = {
                "coding": [{"code": code}],
            }
            ext = opinion_to_fhir_extension(op)
            resource["_verificationStatus"] = {"extension": [ext]}
            nodes_converted += 1

        elif field_name == "criticality" and code is not None:
            resource["criticality"] = code
            ext = opinion_to_fhir_extension(op)
            resource["_criticality"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── MedicationStatement handler (to_fhir) ────────────────────


def _to_medication_statement_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 MedicationStatement."""
    resource: dict[str, Any] = {
        "resourceType": "MedicationStatement",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore medication text
    med_text = doc.get("medication")
    if med_text is not None:
        resource["medicationCodeableConcept"] = {"text": med_text}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── ClinicalImpression handler (to_fhir) ─────────────────────


def _to_clinical_impression_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 ClinicalImpression."""
    resource: dict[str, Any] = {
        "resourceType": "ClinicalImpression",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]
    if doc.get("summary"):
        resource["summary"] = doc["summary"]

    # Restore finding references
    finding_refs = doc.get("finding_references", [])
    if finding_refs:
        resource["finding"] = [
            {"itemReference": {"reference": ref}} for ref in finding_refs
        ]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "summary":
            ext = opinion_to_fhir_extension(op)
            resource["_summary"] = {"extension": [ext]}
            nodes_converted += 1
        elif field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── DetectedIssue handler (to_fhir) ─────────────────────────────


def _to_detected_issue_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 DetectedIssue."""
    resource: dict[str, Any] = {
        "resourceType": "DetectedIssue",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore implicated references
    implicated_refs = doc.get("implicated_references", [])
    if implicated_refs:
        resource["implicated"] = [
            {"reference": ref} for ref in implicated_refs
        ]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")
        code = entry.get("value")

        if field_name == "severity" and code is not None:
            resource["severity"] = code
            ext = opinion_to_fhir_extension(op)
            resource["_severity"] = {"extension": [ext]}
            nodes_converted += 1
        elif field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── Immunization handler (to_fhir) ──────────────────────────────


def _to_immunization_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Immunization."""
    resource: dict[str, Any] = {
        "resourceType": "Immunization",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore vaccineCode
    vaccine_code = doc.get("vaccineCode")
    if vaccine_code is not None:
        resource["vaccineCode"] = {"text": vaccine_code}

    # Restore occurrenceDateTime
    occurrence_dt = doc.get("occurrenceDateTime")
    if occurrence_dt is not None:
        resource["occurrenceDateTime"] = occurrence_dt

    # Restore statusReason
    status_reason = doc.get("statusReason")
    if status_reason is not None:
        resource["statusReason"] = {"text": status_reason}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── FamilyMemberHistory handler (to_fhir) ───────────────────────


def _to_family_member_history_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 FamilyMemberHistory."""
    resource: dict[str, Any] = {
        "resourceType": "FamilyMemberHistory",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore relationship
    relationship = doc.get("relationship")
    if relationship is not None:
        resource["relationship"] = {"text": relationship}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── Procedure handler (to_fhir) ─────────────────────────────────


def _to_procedure_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Procedure."""
    resource: dict[str, Any] = {
        "resourceType": "Procedure",
    }
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    # Restore outcome
    outcome = doc.get("outcome")
    if outcome is not None:
        resource["outcome"] = {"text": outcome}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return resource, report


# ── to_fhir handler dispatch table ───────────────────────────────

_TO_FHIR_HANDLERS = {
    # Phase 1
    "RiskAssessment": _to_risk_assessment_r4,
    "Observation": _to_observation_r4,
    "DiagnosticReport": _to_diagnostic_report_r4,
    "Condition": _to_condition_r4,
    # Phase 2
    "AllergyIntolerance": _to_allergy_intolerance_r4,
    "MedicationStatement": _to_medication_statement_r4,
    "ClinicalImpression": _to_clinical_impression_r4,
    # Phase 3
    "DetectedIssue": _to_detected_issue_r4,
    "Immunization": _to_immunization_r4,
    "FamilyMemberHistory": _to_family_member_history_r4,
    "Procedure": _to_procedure_r4,
}


# ═══════════════════════════════════════════════════════════════════
# CLINICAL FUSION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class FusionReport:
    """Report from fhir_clinical_fuse() describing the fusion outcome.

    Attributes:
        input_count:     Number of input documents provided.
        opinions_fused:  Number of opinions actually fused (after
                         skipping empty documents).
        method:          Fusion method used ("cumulative", "averaging",
                         or "robust").
        conflict_scores: Pairwise conflict scores between opinions.
                         One entry per unique pair (i, j) with i < j.
        warnings:        Any issues encountered (e.g. empty documents).
    """

    input_count: int = 0
    opinions_fused: int = 0
    method: str = "cumulative"
    conflict_scores: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_FUSION_METHODS = {"cumulative", "averaging", "robust"}


def fhir_clinical_fuse(
    docs: Sequence[dict[str, Any]],
    *,
    method: str = "cumulative",
) -> tuple[Opinion, FusionReport]:
    """Fuse opinions from multiple FHIR-derived jsonld-ex documents.

    This is the key capability that FHIR lacks: mathematically
    grounded combination of uncertain evidence from multiple clinical
    sources into a single opinion with properly propagated uncertainty.

    The function extracts the first opinion from each document,
    computes pairwise conflict scores, then fuses using the selected
    Subjective Logic operator:

    - **cumulative** (default): For independent sources.  Each new
      source reduces uncertainty — equivalent to accumulating evidence.
      Use when lab results, imaging, and clinical assessments are
      independently produced.

    - **averaging**: For potentially correlated sources.  Avoids
      double-counting evidence.  Use when sources may share underlying
      data (e.g. two risk scores derived from the same lab panel).

    - **robust**: Byzantine-resistant fusion.  Detects and removes
      outlier opinions that conflict with the majority before fusing.
      Use when data integrity is uncertain (e.g. multi-site studies).

    Args:
        docs:   List of jsonld-ex documents as produced by
                :func:`from_fhir`.  Each must have an ``"opinions"``
                list; documents with empty opinions are skipped with
                a warning.
        method: Fusion method — one of ``"cumulative"``,
                ``"averaging"``, or ``"robust"``.

    Returns:
        Tuple of (fused_opinion, FusionReport).

    Raises:
        ValueError: If ``docs`` is empty, all documents lack opinions,
            or ``method`` is not recognized.
    """
    if method not in _FUSION_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported: {', '.join(sorted(_FUSION_METHODS))}"
        )

    if not docs:
        raise ValueError("No documents provided; at least one is required.")

    # ── Extract opinions from documents ───────────────────────────
    opinions: list[Opinion] = []
    warnings: list[str] = []

    for doc in docs:
        doc_opinions = doc.get("opinions", [])
        if not doc_opinions:
            doc_id = doc.get("id", "unknown")
            doc_type = doc.get("@type", "unknown")
            warnings.append(
                f"Skipped {doc_type} '{doc_id}': no opinions to fuse"
            )
            continue
        # Take the first opinion from each document
        opinions.append(doc_opinions[0]["opinion"])

    if not opinions:
        raise ValueError(
            "No fusable opinions found; all documents had empty opinion lists."
        )

    # ── Compute pairwise conflict scores ──────────────────────────
    conflict_scores: list[float] = []
    for i in range(len(opinions)):
        for j in range(i + 1, len(opinions)):
            score = pairwise_conflict(opinions[i], opinions[j])
            conflict_scores.append(score)

    # ── Fuse ──────────────────────────────────────────────────────
    if len(opinions) == 1:
        fused = opinions[0]
    elif method == "cumulative":
        fused = cumulative_fuse(*opinions)
    elif method == "averaging":
        fused = averaging_fuse(*opinions)
    elif method == "robust":
        fused, _removed = robust_fuse(opinions)
    else:
        # Should not reach here due to validation above
        raise ValueError(f"Unsupported method: {method}")

    report = FusionReport(
        input_count=len(docs),
        opinions_fused=len(opinions),
        method=method,
        conflict_scores=conflict_scores,
        warnings=warnings,
    )

    return fused, report


# ═════════════════════════════════════════════════════════════════
# Phase 3: Clinical Decision Support — Temporal Decay
# ═════════════════════════════════════════════════════════════════

# Timestamp field priority: we search these keys in order to find
# the most clinically relevant timestamp for a FHIR-derived document.
_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "effectiveDateTime",   # Observation, DiagnosticReport
    "occurrenceDateTime",  # Immunization, Procedure
    "onsetDateTime",       # Condition, AllergyIntolerance
    "recordedDate",        # Condition, AllergyIntolerance (fallback)
    "date",                # ClinicalImpression, DetectedIssue
    "assertedDate",        # AllergyIntolerance
)


def _extract_timestamp(doc: dict[str, Any]) -> datetime | None:
    """Extract the best available timestamp from a jsonld-ex document.

    Searches ``_TIMESTAMP_FIELDS`` in priority order and returns the
    first successfully parsed ISO-8601 datetime.  Returns ``None``
    if no parseable timestamp is found.
    """
    for field_name in _TIMESTAMP_FIELDS:
        raw = doc.get(field_name)
        if raw is None:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            continue
    return None


def fhir_temporal_decay(
    doc: dict[str, Any],
    *,
    reference_time: str | None = None,
    half_life_days: float = 365.0,
) -> tuple[dict[str, Any], ConversionReport]:
    """Apply temporal decay to all opinions in a FHIR-derived document.

    Clinical evidence ages: a lab result from last week is more
    relevant than one from two years ago.  This function bridges
    the jsonld-ex ``decay_opinion()`` operator to FHIR resource
    timestamps, producing a new document whose opinions reflect
    evidential aging.

    The decay model is exponential by default: after one half-life,
    belief and disbelief are halved, with the freed mass migrating
    to uncertainty.  The belief/disbelief *ratio* is preserved —
    we forget *how much* evidence we had, not *which direction* it
    pointed.

    Args:
        doc:             A jsonld-ex document (output of ``from_fhir()``).
        reference_time:  ISO-8601 datetime string for "now".
                         Defaults to ``datetime.now(timezone.utc)``.
        half_life_days:  Time (in days) for belief/disbelief to halve.
                         Default 365 days (1 year).  Shorter values
                         model fast-changing evidence (e.g., vital
                         signs → hours); longer values model stable
                         evidence (e.g., genetic tests → decades).

    Returns:
        ``(decayed_doc, report)`` where ``decayed_doc`` is a deep
        copy with all opinions decayed, and ``report`` contains
        success status and any warnings.

    Notes:
        - If no timestamp is found, the document is returned
          unchanged with a warning.
        - If the timestamp is in the future (relative to
          ``reference_time``), the document is returned unchanged
          with a warning.
        - The original document is never mutated.
    """
    # Parse reference time
    if reference_time is not None:
        ref_dt = datetime.fromisoformat(
            reference_time.replace("Z", "+00:00")
        )
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=timezone.utc)
    else:
        ref_dt = datetime.now(timezone.utc)

    # Deep-copy to avoid mutating the original
    result = copy.deepcopy(doc)
    warnings: list[str] = []

    # Extract timestamp
    doc_dt = _extract_timestamp(result)

    if doc_dt is None:
        warnings.append(
            "No parseable timestamp found in document; "
            "opinions returned unchanged."
        )
        report = ConversionReport(
            success=True,
            warnings=warnings,
        )
        return result, report

    # Compute elapsed time in days
    elapsed_seconds = (ref_dt - doc_dt).total_seconds()

    if elapsed_seconds < 0:
        warnings.append(
            f"Document timestamp ({doc_dt.isoformat()}) is in the "
            f"future relative to reference_time "
            f"({ref_dt.isoformat()}); opinions returned unchanged."
        )
        report = ConversionReport(
            success=True,
            warnings=warnings,
        )
        return result, report

    elapsed_days = elapsed_seconds / 86400.0

    # Apply decay to each opinion
    opinions = result.get("opinions", [])
    for entry in opinions:
        op = entry["opinion"]
        decayed_op = decay_opinion(
            op,
            elapsed=elapsed_days,
            half_life=half_life_days,
        )
        entry["opinion"] = decayed_op

    report = ConversionReport(
        success=True,
        nodes_converted=len(opinions),
        warnings=warnings,
    )
    return result, report


# ═════════════════════════════════════════════════════════════════
# Phase 3: Clinical Decision Support — Escalation Policy
# ═════════════════════════════════════════════════════════════════


def _primary_opinion(doc: dict[str, Any]) -> Opinion | None:
    """Extract the first opinion from a jsonld-ex document, or None."""
    opinions = doc.get("opinions", [])
    if not opinions:
        return None
    return opinions[0].get("opinion")


def fhir_escalation_policy(
    resources: list[dict[str, Any]],
    *,
    escalation_threshold: float = 0.7,
    conflict_threshold: float = 0.3,
) -> dict[str, list[dict[str, Any]]]:
    """Categorise jsonld-ex documents into clinical triage buckets.

    Implements confidence-based triage for clinical decision support.
    Each document is routed to exactly one bucket based on its
    primary opinion's belief/disbelief/uncertainty profile.

    Buckets:
        ``auto_accept``
            High belief (≥ escalation_threshold) and low disbelief.
            Safe to act on without clinician review.
        ``human_review``
            Moderate confidence.  Needs clinician review before
            action.  This is the default bucket for ambiguous cases.
        ``reject``
            High disbelief (≥ escalation_threshold).  Evidence
            contradicts the claim; should not be acted on.
        ``escalate``
            High pairwise conflict between any two documents
            (≥ conflict_threshold).  Contradictory evidence requires
            specialist review.

    Args:
        resources:            List of jsonld-ex documents.
        escalation_threshold: Belief/disbelief threshold for
                              auto_accept/reject (default 0.7).
        conflict_threshold:   Pairwise conflict threshold for
                              escalation (default 0.3).

    Returns:
        Dict with keys ``auto_accept``, ``human_review``, ``reject``,
        ``escalate``.  Each value is a list of documents.
    """
    result: dict[str, list[dict[str, Any]]] = {
        "auto_accept": [],
        "human_review": [],
        "reject": [],
        "escalate": [],
    }

    if not resources:
        return result

    # First pass: detect pairwise conflicts for escalation
    escalated_indices: set[int] = set()
    opinions_list: list[Opinion | None] = [
        _primary_opinion(doc) for doc in resources
    ]

    for i in range(len(resources)):
        for j in range(i + 1, len(resources)):
            op_i = opinions_list[i]
            op_j = opinions_list[j]
            if op_i is None or op_j is None:
                continue
            conflict = pairwise_conflict(op_i, op_j)
            if conflict >= conflict_threshold:
                escalated_indices.add(i)
                escalated_indices.add(j)

    # Second pass: categorise each document
    for idx, doc in enumerate(resources):
        if idx in escalated_indices:
            result["escalate"].append(doc)
            continue

        op = opinions_list[idx]
        if op is None:
            # No opinion → human review by default
            result["human_review"].append(doc)
            continue

        if op.disbelief >= escalation_threshold:
            result["reject"].append(doc)
        elif op.belief >= escalation_threshold:
            result["auto_accept"].append(doc)
        else:
            result["human_review"].append(doc)

    return result
