"""
FHIR R4 Interoperability for JSON-LD-Ex.

Bidirectional mapping between FHIR R4 clinical resources and jsonld-ex's
Subjective Logic opinion model.  Provides mathematically grounded
uncertainty that composes, fuses, and propagates correctly — capabilities
that FHIR's scalar probability and categorical code model lack.

Supported FHIR R4 resources (Phase 1):
  - RiskAssessment   — prediction.probabilityDecimal → Opinion
  - Observation       — interpretation / valueQuantity → Opinion
  - DiagnosticReport  — aggregates Observations via fusion
  - Condition         — verificationStatus → continuous Opinion

Architecture notes:
  - All public functions accept a ``fhir_version`` parameter (default "R4")
    so that R5/R6 support can be added without breaking existing callers.
  - Version-specific logic is isolated in private ``_*_r4()`` functions.
  - The SL opinion is embedded in FHIR resources via the standard extension
    mechanism, ensuring zero breaking changes to existing FHIR infrastructure.

References:
  - HL7 FHIR R4: https://hl7.org/fhir/R4/
  - Jøsang, A. (2016). Subjective Logic. Springer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    robust_fuse,
    pairwise_conflict,
)
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

_SUPPORTED_RESOURCE_TYPES = frozenset({
    "RiskAssessment",
    "Observation",
    "DiagnosticReport",
    "Condition",
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


# ── from_fhir handler dispatch table ──────────────────────────────

_RESOURCE_HANDLERS = {
    "RiskAssessment": _from_risk_assessment_r4,
    "Observation": _from_observation_r4,
    "DiagnosticReport": _from_diagnostic_report_r4,
    "Condition": _from_condition_r4,
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

    if resource_type not in _SUPPORTED_RESOURCE_TYPES:
        raise ValueError(
            f"Unsupported resource type '{resource_type}'. "
            f"Supported types: {', '.join(sorted(_SUPPORTED_RESOURCE_TYPES))}"
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


# ── to_fhir handler dispatch table ───────────────────────────────

_TO_FHIR_HANDLERS = {
    "RiskAssessment": _to_risk_assessment_r4,
    "Observation": _to_observation_r4,
    "DiagnosticReport": _to_diagnostic_report_r4,
    "Condition": _to_condition_r4,
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
