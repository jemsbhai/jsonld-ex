"""
FHIR R4 ↔ jsonld-ex bidirectional converters.

Contains ``from_fhir()`` and ``to_fhir()`` public functions plus all
resource-specific private handlers (``_from_*_r4`` / ``_to_*_r4``).
"""

from __future__ import annotations

from typing import Any, Optional

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.owl_interop import ConversionReport

from jsonld_ex.fhir_interop._constants import (
    FHIR_EXTENSION_URL,
    SUPPORTED_FHIR_VERSIONS,
    SUPPORTED_RESOURCE_TYPES,
    VERIFICATION_STATUS_PROBABILITY,
    VERIFICATION_STATUS_UNCERTAINTY,
    INTERPRETATION_PROBABILITY,
    CRITICALITY_PROBABILITY,
    CRITICALITY_UNCERTAINTY,
    MEDSTMT_STATUS_PROBABILITY,
    MEDSTMT_STATUS_UNCERTAINTY,
    CLINICAL_IMPRESSION_STATUS_MULTIPLIER,
    DETECTED_ISSUE_SEVERITY_PROBABILITY,
    DETECTED_ISSUE_SEVERITY_UNCERTAINTY,
    IMMUNIZATION_STATUS_PROBABILITY,
    IMMUNIZATION_STATUS_UNCERTAINTY,
    FAMILY_HISTORY_STATUS_PROBABILITY,
    FAMILY_HISTORY_STATUS_UNCERTAINTY,
    FAMILY_HISTORY_DEFAULT_UNCERTAINTY,
    PROCEDURE_STATUS_PROBABILITY,
    PROCEDURE_STATUS_UNCERTAINTY,
    CONSENT_STATUS_PROBABILITY,
    CONSENT_STATUS_UNCERTAINTY,
    OBSERVATION_STATUS_PROBABILITY,
    OBSERVATION_STATUS_UNCERTAINTY,
    DIAGNOSTIC_REPORT_STATUS_PROBABILITY,
    DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY,
    # Phase 6 — universal coverage
    ENCOUNTER_STATUS_PROBABILITY,
    ENCOUNTER_STATUS_UNCERTAINTY,
    MEDICATION_REQUEST_STATUS_PROBABILITY,
    MEDICATION_REQUEST_STATUS_UNCERTAINTY,
    CARE_PLAN_STATUS_PROBABILITY,
    CARE_PLAN_STATUS_UNCERTAINTY,
    GOAL_LIFECYCLE_PROBABILITY,
    GOAL_LIFECYCLE_UNCERTAINTY,
    GOAL_ACHIEVEMENT_PROBABILITY,
    GOAL_ACHIEVEMENT_UNCERTAINTY,
    CARE_TEAM_STATUS_PROBABILITY,
    CARE_TEAM_STATUS_UNCERTAINTY,
    IMAGING_STUDY_STATUS_PROBABILITY,
    IMAGING_STUDY_STATUS_UNCERTAINTY,
    PATIENT_BASE_PROBABILITY,
    PATIENT_BASE_UNCERTAINTY,
    ACTIVE_BOOLEAN_PROBABILITY,
    ACTIVE_BOOLEAN_UNCERTAINTY,
    DEVICE_STATUS_PROBABILITY,
    DEVICE_STATUS_UNCERTAINTY,
    CLAIM_STATUS_PROBABILITY,
    CLAIM_STATUS_UNCERTAINTY,
    EOB_STATUS_PROBABILITY,
    EOB_STATUS_UNCERTAINTY,
    EOB_OUTCOME_PROBABILITY,
    EOB_OUTCOME_UNCERTAINTY,
    MED_ADMIN_STATUS_PROBABILITY,
    MED_ADMIN_STATUS_UNCERTAINTY,
    # Phase 7A
    SERVICE_REQUEST_STATUS_PROBABILITY,
    SERVICE_REQUEST_STATUS_UNCERTAINTY,
    SERVICE_REQUEST_INTENT_MULTIPLIER,
    SERVICE_REQUEST_PRIORITY_MULTIPLIER,
    # Phase 7A — QuestionnaireResponse
    QR_STATUS_PROBABILITY,
    QR_STATUS_UNCERTAINTY,
    QR_SOURCE_RELIABILITY_MULTIPLIER,
    QR_COMPLETENESS_THRESHOLDS,
    # Phase 7A — Specimen
    SPECIMEN_STATUS_PROBABILITY,
    SPECIMEN_STATUS_UNCERTAINTY,
    SPECIMEN_CONDITION_MULTIPLIER,
    # Phase 7B — DocumentReference
    DOC_REF_STATUS_PROBABILITY,
    DOC_REF_STATUS_UNCERTAINTY,
    DOC_REF_DOC_STATUS_MULTIPLIER,
    # Phase 7B — Coverage
    COVERAGE_STATUS_PROBABILITY,
    COVERAGE_STATUS_UNCERTAINTY,
)
from jsonld_ex.fhir_interop._scalar import (
    scalar_to_opinion,
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
)


# ═══════════════════════════════════════════════════════════════════
# FHIR → JSONLD-EX IMPORT  (from_fhir)
# ═══════════════════════════════════════════════════════════════════


def from_fhir(
    resource: dict[str, Any],
    *,
    fhir_version: str = "R4",
    handler_config: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], ConversionReport]:
    """Import a FHIR R4 resource into a jsonld-ex document with SL opinions.

    Extracts scalar probabilities or categorical codes from the resource,
    reconstructs Subjective Logic opinions from available metadata signals,
    and recovers exact opinions from jsonld-ex FHIR extensions when present.

    Args:
        resource: A FHIR R4 resource as a Python dict (JSON-parsed).
        fhir_version: FHIR version string.  Currently only ``"R4"``
            is supported.
        handler_config: Optional dict of resource-specific parameter
            overrides.  Currently supported by QuestionnaireResponse
            handler.  Keys:

            - ``status_probability``: dict mapping status codes to
              probability values (merged with defaults).
            - ``status_uncertainty``: dict mapping status codes to
              uncertainty values (merged with defaults).
            - ``source_reliability_multiplier``: dict mapping source
              types to uncertainty multipliers (merged with defaults).
            - ``completeness_thresholds``: dict with keys
              ``low_threshold``, ``high_threshold``,
              ``low_multiplier``, ``mid_multiplier``,
              ``high_multiplier`` (replaces defaults).

            Handlers that do not support configurability silently
            ignore this parameter.  ``None`` or ``{}`` use defaults.

    Returns:
        A tuple of ``(jsonld_ex_doc, ConversionReport)``.

    Raises:
        ValueError: If ``resource`` lacks ``resourceType``, or
            ``fhir_version`` is not supported.
    """
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

    if resource_type not in SUPPORTED_RESOURCE_TYPES:
        report = ConversionReport(
            success=True,
            nodes_converted=0,
            warnings=[
                f"Unsupported FHIR resource type '{resource_type}'. "
                f"Supported types: {', '.join(sorted(SUPPORTED_RESOURCE_TYPES))}"
            ],
        )
        return {"@type": f"fhir:{resource_type}", "opinions": []}, report

    handler = _RESOURCE_HANDLERS[resource_type]

    # Configurable handlers accept handler_config; others ignore it.
    if resource_type in _CONFIGURABLE_HANDLERS:
        return handler(resource, handler_config=handler_config)
    return handler(resource)


# ── Extension recovery helper ─────────────────────────────────────


def _try_recover_opinion(
    element_extensions: Optional[dict[str, Any]],
) -> Optional[Opinion]:
    """Try to recover an Opinion from a FHIR element's extension list."""
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
                prob = INTERPRETATION_PROBABILITY.get(code, 0.50)
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

    # ── Status-based fallback when no interpretation is present ────
    if not opinions:
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
            prob = OBSERVATION_STATUS_PROBABILITY.get(status, 0.50)
            default_u = OBSERVATION_STATUS_UNCERTAINTY.get(status, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "status",
                "value": status,
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

    effective_dt = resource.get("effectiveDateTime")
    if effective_dt is not None:
        doc["effectiveDateTime"] = effective_dt

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

    results = resource.get("result", [])
    result_refs = [
        r.get("reference", "") for r in results if isinstance(r, dict)
    ]

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
        else:
            # Reconstruct opinion from status signal; the conclusion
            # text itself is categorical, so we use a moderate base
            # probability and let status modulate uncertainty.
            prob = DIAGNOSTIC_REPORT_STATUS_PROBABILITY.get(status, 0.50)
            default_u = DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY.get(status, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "conclusion",
                "value": conclusion,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    # ── Status-based fallback when no conclusion is present ────────
    if not opinions:
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
            prob = DIAGNOSTIC_REPORT_STATUS_PROBABILITY.get(status, 0.50)
            default_u = DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY.get(status, 0.30)

            # Result count modulates uncertainty: more linked
            # observations = more supporting evidence.
            result_count = len(result_refs)
            if result_count == 0:
                default_u *= 1.3
            elif result_count >= 3:
                default_u *= 0.7
            else:
                default_u *= 0.9

            default_u = max(0.0, min(default_u, 0.99))

            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "status",
                "value": status,
                "opinion": op,
                "source": "reconstructed",
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

    effective_dt = resource.get("effectiveDateTime")
    if effective_dt is not None:
        doc["effectiveDateTime"] = effective_dt

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Condition handler (from_fhir) ─────────────────────────────────


def _from_condition_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Condition → jsonld-ex document."""
    vs_obj = resource.get("verificationStatus", {})
    vs_code = None
    for coding in vs_obj.get("coding", []):
        vs_code = coding.get("code")
        if vs_code:
            break

    cs_obj = resource.get("clinicalStatus", {})
    cs_code = None
    for coding in cs_obj.get("coding", []):
        cs_code = coding.get("code")
        if cs_code:
            break

    evidence = resource.get("evidence")
    evidence_count = len(evidence) if evidence is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
            prob = VERIFICATION_STATUS_PROBABILITY.get(vs_code, 0.50)
            default_u = VERIFICATION_STATUS_UNCERTAINTY.get(vs_code, 0.30)
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

    onset_dt = resource.get("onsetDateTime")
    if onset_dt is not None:
        doc["onsetDateTime"] = onset_dt
    recorded_date = resource.get("recordedDate")
    if recorded_date is not None:
        doc["recordedDate"] = recorded_date

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

    Preserves ALL clinically relevant fields from the FHIR resource:
      - code (substance identity: text + coding)
      - patient reference
      - category (food/medication/environment/biologic)
      - type (allergy vs intolerance)
      - recordedDate, onsetDateTime, lastOccurrence
      - asserter reference
      - reaction array (substance, manifestation, severity)
      - note array
    """
    vs_obj = resource.get("verificationStatus", {})
    vs_code = None
    for coding in vs_obj.get("coding", []):
        vs_code = coding.get("code")
        if vs_code:
            break

    cs_obj = resource.get("clinicalStatus", {})
    cs_code = None
    for coding in cs_obj.get("coding", []):
        cs_code = coding.get("code")
        if cs_code:
            break

    criticality = resource.get("criticality")

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
            prob = VERIFICATION_STATUS_PROBABILITY.get(vs_code, 0.50)
            default_u = VERIFICATION_STATUS_UNCERTAINTY.get(vs_code, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "verificationStatus",
                "value": vs_code,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    if criticality is not None:
        crit_prob = CRITICALITY_PROBABILITY.get(criticality, 0.50)
        crit_u = CRITICALITY_UNCERTAINTY.get(criticality, 0.30)
        crit_op = scalar_to_opinion(crit_prob, default_uncertainty=crit_u)
        opinions.append({
            "field": "criticality",
            "value": criticality,
            "opinion": crit_op,
            "source": "reconstructed",
        })
        nodes_converted += 1

    # --- Code (substance identity) -----------------------------------
    code_obj = resource.get("code")
    code_out: Optional[dict[str, Any]] = None
    if code_obj is not None:
        code_out = {}
        if "text" in code_obj:
            code_out["text"] = code_obj["text"]
        if "coding" in code_obj:
            code_out["coding"] = code_obj["coding"]
        if not code_out:
            code_out = None  # empty CodeableConcept → treat as absent

    # --- Patient reference -------------------------------------------
    patient_ref = resource.get("patient", {})
    patient = patient_ref.get("reference") if isinstance(patient_ref, dict) else None

    # --- Category ----------------------------------------------------
    category = resource.get("category")  # already a list in FHIR

    # --- Type (allergy vs intolerance) -------------------------------
    allergy_type = resource.get("type")

    # --- Temporal fields ---------------------------------------------
    recorded_date = resource.get("recordedDate")
    onset_dt = resource.get("onsetDateTime")
    last_occurrence = resource.get("lastOccurrence")

    # --- Asserter reference ------------------------------------------
    asserter_obj = resource.get("asserter", {})
    asserter = asserter_obj.get("reference") if isinstance(asserter_obj, dict) else None

    # --- Reaction array ----------------------------------------------
    raw_reactions = resource.get("reaction")
    reactions: Optional[list[dict[str, Any]]] = None
    if raw_reactions is not None:
        reactions = []
        for rxn in raw_reactions:
            entry: dict[str, Any] = {}
            # substance (CodeableConcept — preserve fully)
            rxn_substance = rxn.get("substance")
            if rxn_substance is not None:
                sub_out: dict[str, Any] = {}
                if "text" in rxn_substance:
                    sub_out["text"] = rxn_substance["text"]
                if "coding" in rxn_substance:
                    sub_out["coding"] = rxn_substance["coding"]
                entry["substance"] = sub_out if sub_out else None
            else:
                entry["substance"] = None
            # manifestation (list of CodeableConcepts)
            raw_manif = rxn.get("manifestation", [])
            entry["manifestation"] = [
                {k: v for k, v in m.items()} for m in raw_manif
            ]
            # severity
            entry["severity"] = rxn.get("severity")
            reactions.append(entry)

    # --- Note array --------------------------------------------------
    raw_notes = resource.get("note")
    notes: Optional[list[str]] = None
    if raw_notes is not None:
        notes = [n.get("text", "") for n in raw_notes]

    # --- Assemble document -------------------------------------------
    doc: dict[str, Any] = {
        "@type": "fhir:AllergyIntolerance",
        "id": resource.get("id"),
        "clinicalStatus": cs_code,
        "opinions": opinions,
    }

    # Attach non-None metadata fields
    if code_out is not None:
        doc["code"] = code_out
    if patient is not None:
        doc["patient"] = patient
    if category is not None:
        doc["category"] = category
    if allergy_type is not None:
        doc["type"] = allergy_type
    if recorded_date is not None:
        doc["recordedDate"] = recorded_date
    if onset_dt is not None:
        doc["onsetDateTime"] = onset_dt
    if last_occurrence is not None:
        doc["lastOccurrence"] = last_occurrence
    if asserter is not None:
        doc["asserter"] = asserter
    if reactions is not None:
        doc["reaction"] = reactions
    if notes is not None:
        doc["note"] = notes

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── MedicationStatement handler (from_fhir) ──────────────────────


def _from_medication_statement_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 MedicationStatement → jsonld-ex document."""
    status = resource.get("status", "unknown")

    info_source = resource.get("informationSource")
    source_type = None
    if isinstance(info_source, dict):
        ref = info_source.get("reference", "")
        source_type = ref.split("/")[0] if "/" in ref else None

    derived = resource.get("derivedFrom")
    derived_count = len(derived) if derived is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = MEDSTMT_STATUS_PROBABILITY.get(status, 0.50)
        default_u = MEDSTMT_STATUS_UNCERTAINTY.get(status, 0.30)

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
    """Convert FHIR R4 ClinicalImpression → jsonld-ex document."""
    status = resource.get("status", "in-progress")
    summary = resource.get("summary")
    findings = resource.get("finding", [])
    finding_count = len(findings) if findings else 0

    finding_refs: list[str] = []
    for f in findings:
        item_ref = f.get("itemReference")
        if isinstance(item_ref, dict):
            ref = item_ref.get("reference")
            if ref:
                finding_refs.append(ref)

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    opinion_field = "summary" if summary is not None else "status"

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

    if not opinions:
        base_u = 0.20
        multiplier = CLINICAL_IMPRESSION_STATUS_MULTIPLIER.get(status, 1.0)
        adjusted_u = base_u * multiplier

        if finding_count == 0:
            adjusted_u *= 1.3
        elif finding_count >= 3:
            adjusted_u *= 0.7
        else:
            adjusted_u *= 0.9

        adjusted_u = max(0.0, min(adjusted_u, 0.99))

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
    """Convert FHIR R4 DetectedIssue → jsonld-ex document."""
    status = resource.get("status")
    severity = resource.get("severity")

    evidence = resource.get("evidence")
    evidence_count = len(evidence) if evidence is not None else None

    implicated = resource.get("implicated", [])
    implicated_refs = [
        r.get("reference", "") for r in implicated if isinstance(r, dict)
    ]

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
            prob = DETECTED_ISSUE_SEVERITY_PROBABILITY.get(severity, 0.50)
            default_u = DETECTED_ISSUE_SEVERITY_UNCERTAINTY.get(severity, 0.25)
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
    """Convert FHIR R4 Immunization → jsonld-ex document."""
    status = resource.get("status", "completed")

    protocol = resource.get("protocolApplied")
    dose_count = len(protocol) if protocol is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = IMMUNIZATION_STATUS_PROBABILITY.get(status, 0.50)
        default_u = IMMUNIZATION_STATUS_UNCERTAINTY.get(status, 0.30)

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
    """Convert FHIR R4 FamilyMemberHistory → jsonld-ex document."""
    status = resource.get("status", "completed")

    conditions = resource.get("condition")
    condition_count = len(conditions) if conditions is not None else None

    data_absent = resource.get("dataAbsentReason")
    has_data_absent = data_absent is not None

    rel_obj = resource.get("relationship", {})
    relationship_text = rel_obj.get("text") if isinstance(rel_obj, dict) else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = FAMILY_HISTORY_STATUS_PROBABILITY.get(status, 0.50)
        default_u = FAMILY_HISTORY_STATUS_UNCERTAINTY.get(
            status, FAMILY_HISTORY_DEFAULT_UNCERTAINTY
        )

        if condition_count is not None:
            if condition_count == 0:
                default_u *= 1.3
            elif condition_count >= 3:
                default_u *= 0.7
            else:
                default_u *= 0.9

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
    """Convert FHIR R4 Procedure → jsonld-ex document."""
    status = resource.get("status", "completed")

    outcome_obj = resource.get("outcome")
    has_outcome = outcome_obj is not None
    outcome_text = None
    if isinstance(outcome_obj, dict):
        outcome_text = outcome_obj.get("text")

    complications = resource.get("complication")
    complication_count = len(complications) if complications is not None else None

    follow_ups = resource.get("followUp")
    followup_count = len(follow_ups) if follow_ups is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = PROCEDURE_STATUS_PROBABILITY.get(status, 0.50)
        default_u = PROCEDURE_STATUS_UNCERTAINTY.get(status, 0.30)

        if has_outcome:
            default_u *= 0.7

        if complication_count is not None:
            if complication_count == 0:
                default_u *= 0.9
            elif complication_count >= 2:
                default_u *= 1.5
            else:
                default_u *= 1.2

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


# ── Consent handler (from_fhir) ────────────────────────────────────


def _from_consent_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Consent → jsonld-ex document.

    Uses CONSENT_STATUS_PROBABILITY / CONSENT_STATUS_UNCERTAINTY to
    reconstruct an SL opinion from the Consent status code.  This
    integrates Consent into the uniform from_fhir() / to_fhir() API
    while the richer compliance algebra functions (fhir_consent_to_opinion,
    fhir_consent_validity, etc.) remain available for advanced use cases.
    """
    status = resource.get("status", "draft")

    scope_code = None
    scope_obj = resource.get("scope")
    if isinstance(scope_obj, dict):
        for coding in scope_obj.get("coding", []):
            scope_code = coding.get("code")
            if scope_code:
                break

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = CONSENT_STATUS_PROBABILITY.get(status, 0.50)
        default_u = CONSENT_STATUS_UNCERTAINTY.get(status, 0.35)
        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:Consent",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if scope_code is not None:
        doc["scope"] = scope_code

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ── Provenance handler (from_fhir) ─────────────────────────────────


def _from_provenance_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Provenance → jsonld-ex document.

    Proposition under assessment: "this provenance record is reliable
    and complete."  The SL opinion is reconstructed from metadata
    signals that indicate the quality and completeness of the
    provenance chain.

    Uncertainty-lowering signals:
      - ``recorded`` timestamp present (documented when)
      - Multiple agents with distinct roles (corroboration)
      - Agents with ``who`` references (identifiable actors)
      - Entities with explicit ``role`` values (complete chain)

    Uncertainty-raising signals:
      - Missing ``recorded`` timestamp
      - Empty or no agent list
      - Agents without ``who`` (anonymous actors)

    Base probability: 0.75 (a provenance record exists because
    something was documented — moderate positive prior).

    Default uncertainty: 0.20 (higher than clinical resources to
    reflect the inherent incompleteness of provenance metadata).

    All FHIR Provenance fields are preserved in the output doc —
    no data is excluded.
    """
    recorded = resource.get("recorded")

    # ── Extract and flatten agents ────────────────────────────────
    raw_agents = resource.get("agent", [])
    agents: list[dict[str, Any]] = []
    identified_count = 0
    for ag in raw_agents:
        flat: dict[str, Any] = {}

        # Extract type code from CodeableConcept
        type_obj = ag.get("type")
        if isinstance(type_obj, dict):
            for coding in type_obj.get("coding", []):
                code = coding.get("code")
                if code:
                    flat["type"] = code
                    break

        # Extract who reference
        who_obj = ag.get("who")
        if isinstance(who_obj, dict):
            ref = who_obj.get("reference")
            if ref:
                flat["who"] = ref
                identified_count += 1

        # Extract onBehalfOf (delegation)
        obh_obj = ag.get("onBehalfOf")
        if isinstance(obh_obj, dict):
            obh_ref = obh_obj.get("reference")
            if obh_ref:
                flat["onBehalfOf"] = obh_ref

        agents.append(flat)

    agent_count = len(agents)

    # ── Extract and flatten entities ──────────────────────────────
    raw_entities = resource.get("entity", [])
    entities: list[dict[str, Any]] = []
    roles_present = 0
    for ent in raw_entities:
        flat_ent: dict[str, Any] = {}
        role = ent.get("role")
        if role is not None:
            flat_ent["role"] = role
            roles_present += 1
        what_obj = ent.get("what")
        if isinstance(what_obj, dict):
            ref = what_obj.get("reference")
            if ref:
                flat_ent["what"] = ref
        entities.append(flat_ent)

    # ── Extract targets ───────────────────────────────────────────
    raw_targets = resource.get("target", [])
    targets: list[dict[str, Any]] = []
    for t in raw_targets:
        if isinstance(t, dict):
            targets.append(t)  # preserve as-is (contains "reference")

    # ── Extract activity code ─────────────────────────────────────
    activity = None
    activity_obj = resource.get("activity")
    if isinstance(activity_obj, dict):
        for coding in activity_obj.get("coding", []):
            code = coding.get("code")
            if code:
                activity = code
                break

    # ── Extract reason codes ──────────────────────────────────────
    reason_list: list[str] = []
    for reason_obj in resource.get("reason", []):
        if isinstance(reason_obj, dict):
            for coding in reason_obj.get("coding", []):
                code = coding.get("code")
                if code:
                    reason_list.append(code)
                    break

    # ── Extract policy, location, period ──────────────────────────
    policy = resource.get("policy")

    location_ref = None
    location_obj = resource.get("location")
    if isinstance(location_obj, dict):
        location_ref = location_obj.get("reference")

    period = resource.get("period")  # preserve as-is

    # ── Reconstruct SL opinion ────────────────────────────────────
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery on _recorded
    recorded_ext = resource.get("_recorded")
    recovered = _try_recover_opinion(recorded_ext)

    if recovered is not None:
        opinion_field = "recorded" if recorded else "provenance"
        opinions.append({
            "field": opinion_field,
            "value": recorded if recorded else None,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        # Reconstruct from metadata signals
        base_prob = 0.75
        u = 0.20

        # Signal: recorded timestamp
        if recorded is not None:
            u *= 0.8
        else:
            u *= 1.4

        # Signal: agent count and identification
        if agent_count == 0:
            u *= 1.5
        elif agent_count >= 2:
            u *= 0.7
        # else: single agent, baseline

        # Signal: agents with 'who' (identifiable)
        if agent_count > 0:
            anon_count = agent_count - identified_count
            if anon_count > 0:
                # Proportion of anonymous agents raises uncertainty
                u *= 1.0 + 0.3 * (anon_count / agent_count)

        # Signal: entities with explicit roles
        if roles_present > 0:
            u *= 0.8

        # Clamp to valid range [0, 1)
        u = max(0.0, min(u, 0.99))

        op = scalar_to_opinion(base_prob, default_uncertainty=u)

        opinion_field = "recorded" if recorded else "provenance"
        opinions.append({
            "field": opinion_field,
            "value": recorded if recorded else None,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # ── Build jsonld-ex doc ───────────────────────────────────────
    doc: dict[str, Any] = {
        "@type": "fhir:Provenance",
        "id": resource.get("id"),
        "opinions": opinions,
        "targets": targets,
        "agents": agents,
        "entities": entities,
    }

    if recorded is not None:
        doc["recorded"] = recorded
    if activity is not None:
        doc["activity"] = activity
    if policy is not None:
        doc["policy"] = policy
    if reason_list:
        doc["reason"] = reason_list
    if location_ref is not None:
        doc["location"] = location_ref
    if period is not None:
        doc["period"] = period

    report = ConversionReport(
        success=True,
        nodes_converted=nodes_converted,
    )
    return doc, report


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Universal FHIR coverage — from_fhir handlers
# ═══════════════════════════════════════════════════════════════════


def _make_status_handler(
    resource_type: str,
    prob_map: dict[str, float],
    uncertainty_map: dict[str, float],
    *,
    status_field: str = "status",
    passthrough_fields: tuple[str, ...] = (),
):
    """Factory for simple status-based from_fhir handlers.

    Creates a handler that:
    1. Checks for an extension on ``_<status_field>`` (exact recovery).
    2. Falls back to a reconstructed opinion from the status code.
    3. Returns a jsonld-ex document with ``@type``, ``id``, ``status``,
       ``opinions``, and any fields listed in *passthrough_fields*
       copied verbatim from the FHIR resource (for temporal decay, etc.).
    """

    def handler(
        resource: dict[str, Any],
    ) -> tuple[dict[str, Any], ConversionReport]:
        status = resource.get(status_field)
        opinions: list[dict[str, Any]] = []

        status_ext = resource.get(f"_{status_field}")
        recovered = _try_recover_opinion(status_ext)

        if recovered is not None:
            opinions.append({
                "field": status_field,
                "value": status,
                "opinion": recovered,
                "source": "extension",
            })
        else:
            prob = prob_map.get(status, 0.50)
            default_u = uncertainty_map.get(status, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": status_field,
                "value": status,
                "opinion": op,
                "source": "reconstructed",
            })

        doc: dict[str, Any] = {
            "@type": f"fhir:{resource_type}",
            "id": resource.get("id"),
            status_field: status,
            "opinions": opinions,
        }

        # Copy timestamp / metadata fields for temporal decay
        for field_name in passthrough_fields:
            val = resource.get(field_name)
            if val is not None:
                doc[field_name] = val

        report = ConversionReport(success=True, nodes_converted=1)
        return doc, report

    return handler


# ── Batch 1: Clinical workflow (simple status-based) ─────────────

_from_encounter_r4 = _make_status_handler(
    "Encounter", ENCOUNTER_STATUS_PROBABILITY, ENCOUNTER_STATUS_UNCERTAINTY,
    passthrough_fields=("period",),
)

_from_medication_request_r4 = _make_status_handler(
    "MedicationRequest",
    MEDICATION_REQUEST_STATUS_PROBABILITY,
    MEDICATION_REQUEST_STATUS_UNCERTAINTY,
    passthrough_fields=("authoredOn",),
)

_from_care_plan_r4 = _make_status_handler(
    "CarePlan", CARE_PLAN_STATUS_PROBABILITY, CARE_PLAN_STATUS_UNCERTAINTY,
    passthrough_fields=("period",),
)

_from_care_team_r4 = _make_status_handler(
    "CareTeam", CARE_TEAM_STATUS_PROBABILITY, CARE_TEAM_STATUS_UNCERTAINTY,
    passthrough_fields=("period",),
)

_from_imaging_study_r4 = _make_status_handler(
    "ImagingStudy",
    IMAGING_STUDY_STATUS_PROBABILITY,
    IMAGING_STUDY_STATUS_UNCERTAINTY,
    passthrough_fields=("started",),
)

_from_medication_administration_r4 = _make_status_handler(
    "MedicationAdministration",
    MED_ADMIN_STATUS_PROBABILITY,
    MED_ADMIN_STATUS_UNCERTAINTY,
    passthrough_fields=("effectiveDateTime",),
)

# ── Batch 2: Administrative ─────────────────────────────────────

_from_device_r4 = _make_status_handler(
    "Device", DEVICE_STATUS_PROBABILITY, DEVICE_STATUS_UNCERTAINTY,
    passthrough_fields=("manufactureDate",),
)


def _from_patient_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Patient → jsonld-ex document.

    Patient has no status field.  Opinion is based on data
    completeness: more populated demographic fields → lower
    uncertainty (more confident the record is accurate).
    """
    # Count populated data-quality fields.
    quality_fields = (
        "name", "gender", "birthDate", "telecom", "address",
        "identifier", "maritalStatus", "communication",
    )
    populated = sum(1 for f in quality_fields if resource.get(f))
    # Scale uncertainty: 0 fields → base, 8 fields → base * 0.4
    completeness_ratio = populated / len(quality_fields)
    adjusted_u = PATIENT_BASE_UNCERTAINTY * (1.0 - 0.6 * completeness_ratio)

    op = scalar_to_opinion(
        PATIENT_BASE_PROBABILITY, default_uncertainty=adjusted_u,
    )

    doc: dict[str, Any] = {
        "@type": "fhir:Patient",
        "id": resource.get("id"),
        "opinions": [{
            "field": "data_quality",
            "value": f"{populated}/{len(quality_fields)}",
            "opinion": op,
            "source": "reconstructed",
        }],
    }

    report = ConversionReport(success=True, nodes_converted=1)
    return doc, report


def _from_active_boolean_r4(
    resource_type: str,
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 resource with boolean ``active`` field."""
    active = resource.get("active")
    key = str(active).lower() if active is not None else "true"

    status_ext = resource.get("_active")
    recovered = _try_recover_opinion(status_ext)

    if recovered is not None:
        opinion_entry = {
            "field": "active",
            "value": active,
            "opinion": recovered,
            "source": "extension",
        }
    else:
        prob = ACTIVE_BOOLEAN_PROBABILITY.get(key, 0.50)
        default_u = ACTIVE_BOOLEAN_UNCERTAINTY.get(key, 0.25)
        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinion_entry = {
            "field": "active",
            "value": active,
            "opinion": op,
            "source": "reconstructed",
        }

    doc: dict[str, Any] = {
        "@type": f"fhir:{resource_type}",
        "id": resource.get("id"),
        "active": active,
        "opinions": [opinion_entry],
    }

    report = ConversionReport(success=True, nodes_converted=1)
    return doc, report


def _from_organization_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    return _from_active_boolean_r4("Organization", resource)


def _from_practitioner_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    return _from_active_boolean_r4("Practitioner", resource)


# ── Goal handler (from_fhir) ─ special: lifecycleStatus + achievementStatus


def _from_goal_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Goal → jsonld-ex document.

    Goal uses ``lifecycleStatus`` (not ``status``).  If
    ``achievementStatus`` is present, a second opinion is created.
    """
    lifecycle = resource.get("lifecycleStatus")
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Primary: lifecycleStatus
    lifecycle_ext = resource.get("_lifecycleStatus")
    recovered = _try_recover_opinion(lifecycle_ext)

    if recovered is not None:
        opinions.append({
            "field": "lifecycleStatus",
            "value": lifecycle,
            "opinion": recovered,
            "source": "extension",
        })
    else:
        prob = GOAL_LIFECYCLE_PROBABILITY.get(lifecycle, 0.50)
        default_u = GOAL_LIFECYCLE_UNCERTAINTY.get(lifecycle, 0.35)
        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "lifecycleStatus",
            "value": lifecycle,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Secondary: achievementStatus (optional)
    achievement_obj = resource.get("achievementStatus")
    if achievement_obj is not None:
        codings = achievement_obj.get("coding", [])
        for coding in codings:
            code = coding.get("code")
            if code is None:
                continue

            ach_ext = resource.get("_achievementStatus")
            recovered_ach = _try_recover_opinion(ach_ext)

            if recovered_ach is not None:
                opinions.append({
                    "field": "achievementStatus",
                    "value": code,
                    "opinion": recovered_ach,
                    "source": "extension",
                })
            else:
                prob = GOAL_ACHIEVEMENT_PROBABILITY.get(code, 0.50)
                default_u = GOAL_ACHIEVEMENT_UNCERTAINTY.get(code, 0.30)
                op = scalar_to_opinion(prob, default_uncertainty=default_u)
                opinions.append({
                    "field": "achievementStatus",
                    "value": code,
                    "opinion": op,
                    "source": "reconstructed",
                })
            nodes_converted += 1
            break  # Only first coding

    doc: dict[str, Any] = {
        "@type": "fhir:Goal",
        "id": resource.get("id"),
        "lifecycleStatus": lifecycle,
        "opinions": opinions,
    }

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# ── Batch 3: Financial ───────────────────────────────────────────

_from_claim_r4 = _make_status_handler(
    "Claim", CLAIM_STATUS_PROBABILITY, CLAIM_STATUS_UNCERTAINTY,
    passthrough_fields=("created",),
)


def _from_eob_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 ExplanationOfBenefit → jsonld-ex document.

    Status-based primary opinion.  If ``outcome`` is present, a
    second opinion is created for adjudication accuracy.
    """
    status = resource.get("status")
    outcome = resource.get("outcome")
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Primary: status
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
        prob = EOB_STATUS_PROBABILITY.get(status, 0.50)
        default_u = EOB_STATUS_UNCERTAINTY.get(status, 0.30)
        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Secondary: outcome (optional)
    if outcome is not None:
        outcome_ext = resource.get("_outcome")
        recovered_out = _try_recover_opinion(outcome_ext)

        if recovered_out is not None:
            opinions.append({
                "field": "outcome",
                "value": outcome,
                "opinion": recovered_out,
                "source": "extension",
            })
        else:
            prob = EOB_OUTCOME_PROBABILITY.get(outcome, 0.50)
            default_u = EOB_OUTCOME_UNCERTAINTY.get(outcome, 0.30)
            op = scalar_to_opinion(prob, default_uncertainty=default_u)
            opinions.append({
                "field": "outcome",
                "value": outcome,
                "opinion": op,
                "source": "reconstructed",
            })
        nodes_converted += 1

    doc: dict[str, Any] = {
        "@type": "fhir:ExplanationOfBenefit",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }
    if outcome is not None:
        doc["outcome"] = outcome
    created = resource.get("created")
    if created is not None:
        doc["created"] = created

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# ── from_fhir handler dispatch table ──────────────────────────────

def _from_service_request_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 ServiceRequest -> jsonld-ex document.

    ServiceRequest completes the diagnostic chain:
        ServiceRequest -> DiagnosticReport -> Observation

    Proposition: "this service request is valid and should be acted upon."

    Uncertainty is modulated by four multiplicative signals:
        1. status -- base probability and uncertainty from order lifecycle
        2. intent -- epistemic weight (order vs proposal vs option)
        3. priority -- clinical attention level (stat vs routine)
        4. reasonReference count -- evidence basis for ordering

    This is a custom handler (not ``_make_status_handler``) because
    the multiplicative interaction of intent x priority x evidence
    produces genuinely different epistemic semantics from simple
    status-only resources.
    """
    status = resource.get("status")
    intent = resource.get("intent")
    priority = resource.get("priority")

    # Extract reasonReference count for evidence basis
    reason_refs_raw = resource.get("reasonReference")
    reason_refs: list[str] = []
    if reason_refs_raw is not None:
        for ref_obj in reason_refs_raw:
            if isinstance(ref_obj, dict):
                ref = ref_obj.get("reference", "")
                if ref:
                    reason_refs.append(ref)
    reason_count = len(reason_refs) if reason_refs_raw is not None else None

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery on _status
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
        prob = SERVICE_REQUEST_STATUS_PROBABILITY.get(status, 0.50)
        default_u = SERVICE_REQUEST_STATUS_UNCERTAINTY.get(status, 0.30)

        # Signal 1: intent modulates uncertainty
        intent_mult = SERVICE_REQUEST_INTENT_MULTIPLIER.get(intent, 1.0)
        default_u *= intent_mult

        # Signal 2: priority modulates uncertainty
        priority_mult = SERVICE_REQUEST_PRIORITY_MULTIPLIER.get(priority, 1.0)
        default_u *= priority_mult

        # Signal 3: reasonReference count (evidence basis)
        if reason_count is not None:
            if reason_count == 0:
                default_u *= 1.3
            elif reason_count >= 3:
                default_u *= 0.7
            else:
                default_u *= 0.9

        # Clamp to valid range (0, 1)
        default_u = max(0.01, min(default_u, 0.99))

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Extract requester reference
    requester_ref = None
    requester_obj = resource.get("requester")
    if isinstance(requester_obj, dict):
        requester_ref = requester_obj.get("reference")

    # Build jsonld-ex document
    doc: dict[str, Any] = {
        "@type": "fhir:ServiceRequest",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    if intent is not None:
        doc["intent"] = intent
    if priority is not None:
        doc["priority"] = priority
    if reason_refs:
        doc["reason_references"] = reason_refs
    if requester_ref is not None:
        doc["requester"] = requester_ref

    authored_on = resource.get("authoredOn")
    if authored_on is not None:
        doc["authoredOn"] = authored_on

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# ── QuestionnaireResponse handler (from_fhir) ─────────────────────


def _count_leaf_items(
    items: list[dict[str, Any]],
) -> tuple[int, int]:
    """Count total and answered leaf items, recursing into sub-items.

    Group items (those with nested ``item`` arrays but no ``answer``)
    are not counted themselves; only their leaf descendants are counted.
    An item with both ``answer`` and nested ``item`` is counted as a
    leaf (the answer is on this item) and its children are also counted.

    Returns:
        ``(total_leaf_items, answered_leaf_items)``
    """
    total = 0
    answered = 0
    for item in items:
        sub_items = item.get("item")
        has_answer = bool(item.get("answer"))

        if sub_items:
            # Recurse into nested items
            sub_total, sub_answered = _count_leaf_items(sub_items)
            total += sub_total
            answered += sub_answered
            # If this group item itself has an answer, count it too
            if has_answer:
                total += 1
                answered += 1
        else:
            # Leaf item
            total += 1
            if has_answer:
                answered += 1

    return total, answered


def _from_questionnaire_response_r4(
    resource: dict[str, Any],
    *,
    handler_config: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 QuestionnaireResponse → jsonld-ex document.

    Proposition: "this questionnaire response is complete and reliable."

    QuestionnaireResponse is the strongest SL use case among remaining
    FHIR types.  Patient self-report introduces recall bias, social
    desirability bias, health literacy effects, and cognitive load.

    Uncertainty is modulated by three multiplicative signals:
        1. status — response completeness (completed/amended/in-progress/
           stopped/entered-in-error)
        2. source — reporter type (Practitioner < RelatedPerson < Patient)
        3. item completeness — ratio of answered items to total items

    All epistemic parameters are configurable via ``handler_config``
    to support tuning for specific instruments (PHQ-9, GAD-7, etc.)
    and populations.

    Args:
        resource: FHIR R4 QuestionnaireResponse dict.
        handler_config: Optional parameter overrides.  Supported keys:
            ``status_probability``, ``status_uncertainty``,
            ``source_reliability_multiplier``, ``completeness_thresholds``.
            Dicts are merged with defaults (overrides win); ``None``/``{}``
            use defaults.
    """
    cfg = handler_config or {}

    # Merge configurable parameters with defaults
    status_prob = {**QR_STATUS_PROBABILITY, **cfg.get("status_probability", {})}
    status_unc = {**QR_STATUS_UNCERTAINTY, **cfg.get("status_uncertainty", {})}
    source_mult = {
        **QR_SOURCE_RELIABILITY_MULTIPLIER,
        **cfg.get("source_reliability_multiplier", {}),
    }
    completeness_cfg = {
        **QR_COMPLETENESS_THRESHOLDS,
        **cfg.get("completeness_thresholds", {}),
    }

    status = resource.get("status")

    # ── Extract source reference and type ──────────────────────────
    source_obj = resource.get("source")
    source_ref: Optional[str] = None
    source_type: Optional[str] = None
    if isinstance(source_obj, dict):
        source_ref = source_obj.get("reference")
        if source_ref and "/" in source_ref:
            source_type = source_ref.split("/")[0]

    # ── Count leaf items for completeness signal ─────────────────
    raw_items = resource.get("item")
    item_total = 0
    item_answered = 0
    if raw_items:
        item_total, item_answered = _count_leaf_items(raw_items)

    # ── Reconstruct SL opinion ──────────────────────────────────
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = status_prob.get(status, 0.50)
        default_u = status_unc.get(status, 0.30)

        # Signal 1: source type modulates uncertainty
        if source_type is not None:
            src_mult = source_mult.get(source_type, 1.0)
            default_u *= src_mult

        # Signal 2: item completeness modulates uncertainty
        if item_total > 0:
            ratio = item_answered / item_total
            low_t = completeness_cfg["low_threshold"]
            high_t = completeness_cfg["high_threshold"]
            if ratio < low_t:
                default_u *= completeness_cfg["low_multiplier"]
            elif ratio > high_t:
                default_u *= completeness_cfg["high_multiplier"]
            else:
                default_u *= completeness_cfg["mid_multiplier"]

        # Clamp to valid range (0, 1)
        default_u = max(0.01, min(default_u, 0.99))

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # ── Build jsonld-ex document ─────────────────────────────────
    doc: dict[str, Any] = {
        "@type": "fhir:QuestionnaireResponse",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    # Metadata passthrough
    authored = resource.get("authored")
    if authored is not None:
        doc["authored"] = authored

    questionnaire = resource.get("questionnaire")
    if questionnaire is not None:
        doc["questionnaire"] = questionnaire

    if source_ref is not None:
        doc["source"] = source_ref

    # Item count metadata for downstream analysis
    if raw_items and item_total > 0:
        doc["item_count"] = {
            "total": item_total,
            "answered": item_answered,
        }

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# ── Specimen handler (from_fhir) ─────────────────────────────────


def _from_specimen_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Specimen → jsonld-ex document.

    Specimen links pre-analytical quality to the diagnostic chain:
        ServiceRequest → **Specimen** → DiagnosticReport → Observation

    Proposition: "this specimen is suitable for reliable diagnostic analysis."

    Uncertainty is modulated by five multiplicative signals:
        1. status — base probability and uncertainty from availability
        2. condition[] — v2 Table 0493 degradation codes compound
           independently (each HEM/CLOT/CON/AUT raises u)
        3. processing[] count — chain of custody documentation
        4. collection.collector presence — accountability signal
        5. collection.quantity presence — sample adequacy documentation

    This is a custom handler because pre-analytical quality involves
    multiple independent degradation signals that interact
    multiplicatively.  Pre-analytical errors cause ~70% of lab
    testing errors (Plebani 2006), making condition codes the most
    impactful signal.
    """
    status = resource.get("status")

    # ── Extract condition codes (0..*) ────────────────────────────
    raw_conditions = resource.get("condition")
    condition_codes: list[str] = []
    if raw_conditions is not None:
        for cond in raw_conditions:
            codings = cond.get("coding", [])
            for coding in codings:
                code = coding.get("code")
                if code:
                    condition_codes.append(code)
                    break  # first code per CodeableConcept

    # ── Extract processing count ───────────────────────────────
    raw_processing = resource.get("processing")
    processing_count = len(raw_processing) if raw_processing is not None else 0

    # ── Extract collection backbone signals ─────────────────────
    collection = resource.get("collection", {})
    if not isinstance(collection, dict):
        collection = {}

    collector_obj = collection.get("collector")
    collector_ref: str | None = None
    if isinstance(collector_obj, dict):
        collector_ref = collector_obj.get("reference")

    quantity_obj = collection.get("quantity")
    has_quantity = quantity_obj is not None and isinstance(quantity_obj, dict)

    collected_dt = collection.get("collectedDateTime")
    method_obj = collection.get("method")
    method_text = method_obj.get("text") if isinstance(method_obj, dict) else None
    body_site_obj = collection.get("bodySite")
    body_site_text = body_site_obj.get("text") if isinstance(body_site_obj, dict) else None

    # ── Extract other metadata ─────────────────────────────────
    type_obj = resource.get("type")
    type_text = type_obj.get("text") if isinstance(type_obj, dict) else None

    subject_obj = resource.get("subject")
    subject_ref = subject_obj.get("reference") if isinstance(subject_obj, dict) else None

    received_time = resource.get("receivedTime")

    raw_requests = resource.get("request")
    request_refs: list[str] = []
    if raw_requests is not None:
        for req in raw_requests:
            if isinstance(req, dict):
                ref = req.get("reference")
                if ref:
                    request_refs.append(ref)

    raw_parents = resource.get("parent")
    parent_refs: list[str] = []
    if raw_parents is not None:
        for par in raw_parents:
            if isinstance(par, dict):
                ref = par.get("reference")
                if ref:
                    parent_refs.append(ref)

    raw_notes = resource.get("note")
    notes: list[str] | None = None
    if raw_notes is not None:
        notes = [n.get("text", "") for n in raw_notes]

    # ── Reconstruct SL opinion ─────────────────────────────────
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = SPECIMEN_STATUS_PROBABILITY.get(status, 0.50)
        default_u = SPECIMEN_STATUS_UNCERTAINTY.get(status, 0.30)

        # Signal 1: condition codes (multiplicative compounding)
        for code in condition_codes:
            mult = SPECIMEN_CONDITION_MULTIPLIER.get(code, 1.0)
            default_u *= mult

        # Signal 2: processing chain count
        if processing_count == 0:
            default_u *= 1.3
        elif processing_count >= 3:
            default_u *= 0.8
        # else 1–2 steps: baseline (1.0)

        # Signal 3: collector identification
        if collector_ref is not None:
            default_u *= 0.9
        else:
            default_u *= 1.1

        # Signal 4: collection quantity documentation
        if has_quantity:
            default_u *= 0.9
        # else: baseline (1.0) — absence is not penalised

        # Clamp to valid range (0, 1)
        default_u = max(0.01, min(default_u, 0.99))

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # ── Build jsonld-ex document ───────────────────────────────
    doc: dict[str, Any] = {
        "@type": "fhir:Specimen",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    # Metadata passthrough — no data excluded
    if type_text is not None:
        doc["type"] = type_text
    if subject_ref is not None:
        doc["subject"] = subject_ref
    if received_time is not None:
        doc["receivedTime"] = received_time
    if request_refs:
        doc["request_references"] = request_refs
    if collector_ref is not None:
        doc["collector"] = collector_ref
    if collected_dt is not None:
        doc["collectedDateTime"] = collected_dt
    if has_quantity:
        doc["collection_quantity"] = quantity_obj
    if method_text is not None:
        doc["collection_method"] = method_text
    if body_site_text is not None:
        doc["collection_bodySite"] = body_site_text
    if condition_codes:
        doc["condition_codes"] = condition_codes
    doc["processing_count"] = processing_count
    if notes is not None:
        doc["note"] = notes
    if parent_refs:
        doc["parent_references"] = parent_refs

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# -- DocumentReference handler (from_fhir) -------------------------


def _from_document_reference_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 DocumentReference -> jsonld-ex document.

    Proposition: "this document reference is valid and the underlying
    document is reliable."

    DocumentReference is unique in FHIR R4: it has **two independent
    status dimensions** (status + docStatus) producing genuinely
    different epistemic semantics.

    Uncertainty is modulated by five multiplicative signals:
        1. status -- base probability and uncertainty from reference
           lifecycle (current / superseded / entered-in-error)
        2. docStatus -- document content maturity multiplier
           (final / amended / preliminary / entered-in-error)
        3. authenticator presence -- verification signal (x0.8)
        4. author[] count -- accountability (0=x1.2, 1=x1.0, 2+=x0.9)
        5. content[] count -- document availability (0=x1.3, 1=x1.0, 2+=x0.9)

    This is a custom handler (not ``_make_status_handler``) because
    the dual-status model and multiplicative signal interactions
    produce genuinely different epistemic semantics.
    """
    status = resource.get("status")
    doc_status = resource.get("docStatus")

    # Extract authenticator reference
    authenticator_ref = None
    auth_obj = resource.get("authenticator")
    if isinstance(auth_obj, dict):
        authenticator_ref = auth_obj.get("reference")

    # Extract author references
    author_refs: list[str] = []
    author_raw = resource.get("author")
    has_author_field = author_raw is not None
    if author_raw is not None:
        for ref_obj in author_raw:
            if isinstance(ref_obj, dict):
                ref = ref_obj.get("reference", "")
                if ref:
                    author_refs.append(ref)

    # Count content elements
    content_raw = resource.get("content")
    content_count = len(content_raw) if content_raw is not None else 0

    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

    # Try extension recovery on _status
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
        prob = DOC_REF_STATUS_PROBABILITY.get(status, 0.50)
        default_u = DOC_REF_STATUS_UNCERTAINTY.get(status, 0.30)

        # Signal 2: docStatus modulates uncertainty
        if doc_status is not None:
            doc_status_mult = DOC_REF_DOC_STATUS_MULTIPLIER.get(
                doc_status, 1.0
            )
            default_u *= doc_status_mult

        # Signal 3: authenticator presence reduces uncertainty
        if authenticator_ref is not None:
            default_u *= 0.8

        # Signal 4: author count (accountability)
        if has_author_field:
            author_count = len(author_refs)
            if author_count == 0:
                default_u *= 1.2
            elif author_count >= 2:
                default_u *= 0.9
            # author_count == 1 -> x1.0 (baseline)

        # Signal 5: content count (document availability)
        if content_count == 0:
            default_u *= 1.3
        elif content_count >= 2:
            default_u *= 0.9
        # content_count == 1 -> x1.0 (baseline)

        # Clamp to valid range (0, 1)
        default_u = max(0.01, min(default_u, 0.99))

        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # Extract metadata for passthrough
    # Type text
    type_text = None
    type_obj = resource.get("type")
    if isinstance(type_obj, dict):
        type_text = type_obj.get("text")

    # Subject reference
    subject_ref = None
    subject_obj = resource.get("subject")
    if isinstance(subject_obj, dict):
        subject_ref = subject_obj.get("reference")

    # Custodian reference
    custodian_ref = None
    custodian_obj = resource.get("custodian")
    if isinstance(custodian_obj, dict):
        custodian_ref = custodian_obj.get("reference")

    # SecurityLabel codes
    security_label_codes: list[str] = []
    for sl in resource.get("securityLabel", []):
        if isinstance(sl, dict):
            codings = sl.get("coding", [])
            for c in codings:
                if isinstance(c, dict) and c.get("code"):
                    security_label_codes.append(c["code"])

    # RelatesTo
    relates_to: list[dict[str, str]] = []
    for rt in resource.get("relatesTo", []):
        if isinstance(rt, dict):
            code = rt.get("code", "")
            target_ref = ""
            target_obj = rt.get("target")
            if isinstance(target_obj, dict):
                target_ref = target_obj.get("reference", "")
            if code or target_ref:
                relates_to.append({"code": code, "target": target_ref})

    # Context
    context_encounter_refs: list[str] = []
    context_period = None
    context_obj = resource.get("context")
    if isinstance(context_obj, dict):
        for enc in context_obj.get("encounter", []):
            if isinstance(enc, dict):
                ref = enc.get("reference", "")
                if ref:
                    context_encounter_refs.append(ref)
        period_obj = context_obj.get("period")
        if isinstance(period_obj, dict):
            context_period = period_obj

    # Category
    category_raw = resource.get("category")

    # Build jsonld-ex document
    doc: dict[str, Any] = {
        "@type": "fhir:DocumentReference",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    if doc_status is not None:
        doc["docStatus"] = doc_status
    if type_text is not None:
        doc["type"] = type_text
    if subject_ref is not None:
        doc["subject"] = subject_ref
    if authenticator_ref is not None:
        doc["authenticator"] = authenticator_ref
    if custodian_ref is not None:
        doc["custodian"] = custodian_ref
    if author_refs:
        doc["author_references"] = author_refs
    if content_count > 0:
        doc["content_count"] = content_count
    if security_label_codes:
        doc["securityLabel_codes"] = security_label_codes
    if relates_to:
        doc["relatesTo"] = relates_to
    if context_encounter_refs:
        doc["context_encounter_references"] = context_encounter_refs
    if context_period is not None:
        doc["context_period"] = context_period
    if category_raw is not None:
        doc["category"] = category_raw

    date = resource.get("date")
    if date is not None:
        doc["date"] = date

    description = resource.get("description")
    if description is not None:
        doc["description"] = description

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


# ── Configurable handler registry ───────────────────────────────
#
# Resource types whose handlers accept a ``handler_config`` kwarg.
# Used by from_fhir() to decide whether to pass configuration through.

# ── Coverage handler (from_fhir) ───────────────────────────────────────


def _from_coverage_r4(
    resource: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert FHIR R4 Coverage → jsonld-ex document.

    Proposition: "This insurance coverage record is valid and currently
    in force."

    Completes the financial chain:
        **Coverage** → Claim → ExplanationOfBenefit

    Coverage is a status-based resource with rich metadata that links
    the insured patient to their insurer and benefit plan.  The status
    field uses FinancialResourceStatusCodes (active / cancelled / draft /
    entered-in-error).  Period start/end is passed through for temporal
    decay support.

    This is a custom handler (not ``_make_status_handler``) because
    FHIR Reference fields (beneficiary, subscriber, payor) and
    CodeableConcept fields (type, relationship) require proper
    extraction for round-trip fidelity — consistent with how
    ServiceRequest, Specimen, and DocumentReference handle references.
    """
    status = resource.get("status")

    # ── Extract period for temporal decay ───────────────────────
    period = resource.get("period")

    # ── Extract type text (CodeableConcept → text or first code) ────
    type_text = None
    type_obj = resource.get("type")
    if isinstance(type_obj, dict):
        type_text = type_obj.get("text")
        if type_text is None:
            # Fallback to first coding code
            for coding in type_obj.get("coding", []):
                if isinstance(coding, dict) and coding.get("code"):
                    type_text = coding["code"]
                    break

    # ── Extract beneficiary reference ───────────────────────────
    beneficiary_ref = None
    beneficiary_obj = resource.get("beneficiary")
    if isinstance(beneficiary_obj, dict):
        beneficiary_ref = beneficiary_obj.get("reference")

    # ── Extract subscriber reference ────────────────────────────
    subscriber_ref = None
    subscriber_obj = resource.get("subscriber")
    if isinstance(subscriber_obj, dict):
        subscriber_ref = subscriber_obj.get("reference")

    # ── Extract payor references (0..*) ─────────────────────────
    payor_refs: list[str] = []
    payor_raw = resource.get("payor")
    if payor_raw is not None:
        for ref_obj in payor_raw:
            if isinstance(ref_obj, dict):
                ref = ref_obj.get("reference", "")
                if ref:
                    payor_refs.append(ref)

    # ── Extract relationship code (CodeableConcept) ──────────────
    relationship_code = None
    rel_obj = resource.get("relationship")
    if isinstance(rel_obj, dict):
        for coding in rel_obj.get("coding", []):
            if isinstance(coding, dict):
                code = coding.get("code")
                if code:
                    relationship_code = code
                    break

    # ── Extract simple fields ──────────────────────────────────
    dependent = resource.get("dependent")
    order = resource.get("order")
    network = resource.get("network")

    # ── Reconstruct SL opinion ─────────────────────────────────
    opinions: list[dict[str, Any]] = []
    nodes_converted = 0

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
        prob = COVERAGE_STATUS_PROBABILITY.get(status, 0.50)
        default_u = COVERAGE_STATUS_UNCERTAINTY.get(status, 0.30)
        op = scalar_to_opinion(prob, default_uncertainty=default_u)
        opinions.append({
            "field": "status",
            "value": status,
            "opinion": op,
            "source": "reconstructed",
        })
    nodes_converted += 1

    # ── Build jsonld-ex document ───────────────────────────────
    doc: dict[str, Any] = {
        "@type": "fhir:Coverage",
        "id": resource.get("id"),
        "status": status,
        "opinions": opinions,
    }

    # Metadata passthrough — no data excluded
    if period is not None:
        doc["period"] = period
    if type_text is not None:
        doc["type"] = type_text
    if beneficiary_ref is not None:
        doc["beneficiary"] = beneficiary_ref
    if subscriber_ref is not None:
        doc["subscriber"] = subscriber_ref
    if payor_refs:
        doc["payor_references"] = payor_refs
    if dependent is not None:
        doc["dependent"] = dependent
    if relationship_code is not None:
        doc["relationship"] = relationship_code
    if order is not None:
        doc["order"] = order
    if network is not None:
        doc["network"] = network

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return doc, report


_CONFIGURABLE_HANDLERS: frozenset[str] = frozenset({
    "QuestionnaireResponse",
})


# ── from_fhir handler dispatch table ────────────────────────────

_RESOURCE_HANDLERS = {
    "RiskAssessment": _from_risk_assessment_r4,
    "Observation": _from_observation_r4,
    "DiagnosticReport": _from_diagnostic_report_r4,
    "Condition": _from_condition_r4,
    "AllergyIntolerance": _from_allergy_intolerance_r4,
    "MedicationStatement": _from_medication_statement_r4,
    "ClinicalImpression": _from_clinical_impression_r4,
    "DetectedIssue": _from_detected_issue_r4,
    "Immunization": _from_immunization_r4,
    "FamilyMemberHistory": _from_family_member_history_r4,
    "Procedure": _from_procedure_r4,
    "Consent": _from_consent_r4,
    "Provenance": _from_provenance_r4,
    # Phase 6 — universal coverage
    "Encounter": _from_encounter_r4,
    "MedicationRequest": _from_medication_request_r4,
    "CarePlan": _from_care_plan_r4,
    "Goal": _from_goal_r4,
    "CareTeam": _from_care_team_r4,
    "ImagingStudy": _from_imaging_study_r4,
    "Patient": _from_patient_r4,
    "Organization": _from_organization_r4,
    "Practitioner": _from_practitioner_r4,
    "Device": _from_device_r4,
    "Claim": _from_claim_r4,
    "ExplanationOfBenefit": _from_eob_r4,
    "MedicationAdministration": _from_medication_administration_r4,
    # Phase 7A — high-value clinical expansion
    "ServiceRequest": _from_service_request_r4,
    "QuestionnaireResponse": _from_questionnaire_response_r4,
    "Specimen": _from_specimen_r4,
    # Phase 7B — US Core completeness
    "DocumentReference": _from_document_reference_r4,
    "Coverage": _from_coverage_r4,
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

    Reverses ``from_fhir()``.  Takes a jsonld-ex document and produces
    a valid FHIR R4 resource with SL opinions embedded as extensions.

    Args:
        doc: A jsonld-ex document dict with ``@type`` and ``opinions``.
        fhir_version: FHIR version string.  Currently only ``"R4"``.

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
    resource: dict[str, Any] = {"resourceType": "RiskAssessment"}
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
        ext = opinion_to_fhir_extension(op)
        pred["_probabilityDecimal"] = {"extension": [ext]}
        predictions.append(pred)
        nodes_converted += 1

    resource["prediction"] = predictions

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Observation handler (to_fhir) ────────────────────────────────


def _to_observation_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Observation."""
    resource: dict[str, Any] = {"resourceType": "Observation"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]
    if doc.get("value"):
        resource["valueQuantity"] = doc["value"]

    effective_dt = doc.get("effectiveDateTime")
    if effective_dt is not None:
        resource["effectiveDateTime"] = effective_dt

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")
        code = entry.get("value")

        if field_name == "interpretation" and code is not None:
            resource.setdefault("interpretation", []).append({
                "coding": [{"code": code}],
            })
            ext = opinion_to_fhir_extension(op)
            resource["_interpretation"] = {"extension": [ext]}
            nodes_converted += 1
        elif field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── DiagnosticReport handler (to_fhir) ───────────────────────────


def _to_diagnostic_report_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 DiagnosticReport."""
    resource: dict[str, Any] = {"resourceType": "DiagnosticReport"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]
    if doc.get("conclusion"):
        resource["conclusion"] = doc["conclusion"]

    effective_dt = doc.get("effectiveDateTime")
    if effective_dt is not None:
        resource["effectiveDateTime"] = effective_dt

    result_refs = doc.get("result_references", [])
    if result_refs:
        resource["result"] = [{"reference": ref} for ref in result_refs]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "conclusion":
            ext = opinion_to_fhir_extension(op)
            resource["_conclusion"] = {"extension": [ext]}
            nodes_converted += 1
        elif field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Condition handler (to_fhir) ──────────────────────────────────


def _to_condition_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Condition."""
    resource: dict[str, Any] = {"resourceType": "Condition"}
    if doc.get("id"):
        resource["id"] = doc["id"]

    cs = doc.get("clinicalStatus")
    if cs:
        resource["clinicalStatus"] = {"coding": [{"code": cs}]}

    onset_dt = doc.get("onsetDateTime")
    if onset_dt is not None:
        resource["onsetDateTime"] = onset_dt

    recorded_date = doc.get("recordedDate")
    if recorded_date is not None:
        resource["recordedDate"] = recorded_date

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")
        code = entry.get("value")

        if field == "verificationStatus" and code is not None:
            resource["verificationStatus"] = {"coding": [{"code": code}]}
            ext = opinion_to_fhir_extension(op)
            resource["_verificationStatus"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── AllergyIntolerance handler (to_fhir) ─────────────────────────


def _to_allergy_intolerance_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 AllergyIntolerance.

    Round-trips ALL fields preserved by _from_allergy_intolerance_r4:
    code, patient, category, type, temporal fields, asserter,
    reaction array, and note array.
    """
    resource: dict[str, Any] = {"resourceType": "AllergyIntolerance"}
    if doc.get("id"):
        resource["id"] = doc["id"]

    cs = doc.get("clinicalStatus")
    if cs:
        resource["clinicalStatus"] = {"coding": [{"code": cs}]}

    # --- Code (substance identity) -----------------------------------
    code_obj = doc.get("code")
    if code_obj is not None:
        fhir_code: dict[str, Any] = {}
        if "text" in code_obj:
            fhir_code["text"] = code_obj["text"]
        if "coding" in code_obj:
            fhir_code["coding"] = code_obj["coding"]
        if fhir_code:
            resource["code"] = fhir_code

    # --- Patient reference -------------------------------------------
    patient = doc.get("patient")
    if patient is not None:
        resource["patient"] = {"reference": patient}

    # --- Category ----------------------------------------------------
    category = doc.get("category")
    if category is not None:
        resource["category"] = category

    # --- Type --------------------------------------------------------
    allergy_type = doc.get("type")
    if allergy_type is not None:
        resource["type"] = allergy_type

    # --- Temporal fields ---------------------------------------------
    recorded_date = doc.get("recordedDate")
    if recorded_date is not None:
        resource["recordedDate"] = recorded_date

    onset_dt = doc.get("onsetDateTime")
    if onset_dt is not None:
        resource["onsetDateTime"] = onset_dt

    last_occurrence = doc.get("lastOccurrence")
    if last_occurrence is not None:
        resource["lastOccurrence"] = last_occurrence

    # --- Asserter reference ------------------------------------------
    asserter = doc.get("asserter")
    if asserter is not None:
        resource["asserter"] = {"reference": asserter}

    # --- Reaction array ----------------------------------------------
    reactions = doc.get("reaction")
    if reactions is not None:
        fhir_reactions: list[dict[str, Any]] = []
        for rxn in reactions:
            fhir_rxn: dict[str, Any] = {}
            # substance
            sub = rxn.get("substance")
            if sub is not None:
                fhir_sub: dict[str, Any] = {}
                if "text" in sub:
                    fhir_sub["text"] = sub["text"]
                if "coding" in sub:
                    fhir_sub["coding"] = sub["coding"]
                if fhir_sub:
                    fhir_rxn["substance"] = fhir_sub
            # manifestation
            manif = rxn.get("manifestation", [])
            if manif:
                fhir_rxn["manifestation"] = [
                    {k: v for k, v in m.items()} for m in manif
                ]
            # severity
            sev = rxn.get("severity")
            if sev is not None:
                fhir_rxn["severity"] = sev
            fhir_reactions.append(fhir_rxn)
        if fhir_reactions:
            resource["reaction"] = fhir_reactions

    # --- Note array --------------------------------------------------
    notes = doc.get("note")
    if notes is not None:
        resource["note"] = [{"text": t} for t in notes]

    # --- Opinions → FHIR extensions ----------------------------------
    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")
        code = entry.get("value")

        if field_name == "verificationStatus" and code is not None:
            resource["verificationStatus"] = {"coding": [{"code": code}]}
            ext = opinion_to_fhir_extension(op)
            resource["_verificationStatus"] = {"extension": [ext]}
            nodes_converted += 1
        elif field_name == "criticality" and code is not None:
            resource["criticality"] = code
            ext = opinion_to_fhir_extension(op)
            resource["_criticality"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── MedicationStatement handler (to_fhir) ────────────────────────


def _to_medication_statement_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 MedicationStatement."""
    resource: dict[str, Any] = {"resourceType": "MedicationStatement"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── ClinicalImpression handler (to_fhir) ─────────────────────────


def _to_clinical_impression_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 ClinicalImpression."""
    resource: dict[str, Any] = {"resourceType": "ClinicalImpression"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]
    if doc.get("summary"):
        resource["summary"] = doc["summary"]

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── DetectedIssue handler (to_fhir) ─────────────────────────────


def _to_detected_issue_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 DetectedIssue."""
    resource: dict[str, Any] = {"resourceType": "DetectedIssue"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Immunization handler (to_fhir) ──────────────────────────────


def _to_immunization_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Immunization."""
    resource: dict[str, Any] = {"resourceType": "Immunization"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    vaccine_code = doc.get("vaccineCode")
    if vaccine_code is not None:
        resource["vaccineCode"] = {"text": vaccine_code}

    occurrence_dt = doc.get("occurrenceDateTime")
    if occurrence_dt is not None:
        resource["occurrenceDateTime"] = occurrence_dt

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── FamilyMemberHistory handler (to_fhir) ───────────────────────


def _to_family_member_history_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 FamilyMemberHistory."""
    resource: dict[str, Any] = {"resourceType": "FamilyMemberHistory"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Procedure handler (to_fhir) ─────────────────────────────────


def _to_procedure_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Procedure."""
    resource: dict[str, Any] = {"resourceType": "Procedure"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

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

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Consent handler (to_fhir) ─────────────────────────────────────


def _to_consent_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Consent."""
    resource: dict[str, Any] = {"resourceType": "Consent"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("status"):
        resource["status"] = doc["status"]

    scope = doc.get("scope")
    if scope is not None:
        resource["scope"] = {"coding": [{"code": scope}]}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field_name = entry.get("field", "")

        if field_name == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Provenance handler (to_fhir) ────────────────────────────────


def _to_provenance_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex document → FHIR R4 Provenance.

    Reverses ``_from_provenance_r4()``.  Reconstructs the full FHIR
    Provenance structure including CodeableConcept wrappers for agent
    type, Reference wrappers for who/what/location/onBehalfOf, and
    embeds the SL opinion as an extension on ``_recorded``.
    """
    resource: dict[str, Any] = {"resourceType": "Provenance"}
    if doc.get("id"):
        resource["id"] = doc["id"]
    if doc.get("recorded"):
        resource["recorded"] = doc["recorded"]

    # ── Targets ──────────────────────────────────────────────────
    targets = doc.get("targets", [])
    if targets:
        resource["target"] = targets  # already in FHIR Reference format

    # ── Agents ───────────────────────────────────────────────────
    agents = doc.get("agents", [])
    if agents:
        fhir_agents: list[dict[str, Any]] = []
        for ag in agents:
            fhir_ag: dict[str, Any] = {}

            # type → CodeableConcept
            agent_type = ag.get("type")
            if agent_type is not None:
                fhir_ag["type"] = {"coding": [{"code": agent_type}]}

            # who → Reference
            who = ag.get("who")
            if who is not None:
                fhir_ag["who"] = {"reference": who}

            # onBehalfOf → Reference
            obh = ag.get("onBehalfOf")
            if obh is not None:
                fhir_ag["onBehalfOf"] = {"reference": obh}

            fhir_agents.append(fhir_ag)
        resource["agent"] = fhir_agents

    # ── Entities ─────────────────────────────────────────────────
    entities = doc.get("entities", [])
    if entities:
        fhir_entities: list[dict[str, Any]] = []
        for ent in entities:
            fhir_ent: dict[str, Any] = {}
            role = ent.get("role")
            if role is not None:
                fhir_ent["role"] = role
            what = ent.get("what")
            if what is not None:
                fhir_ent["what"] = {"reference": what}
            fhir_entities.append(fhir_ent)
        resource["entity"] = fhir_entities

    # ── Activity → CodeableConcept ───────────────────────────────
    activity = doc.get("activity")
    if activity is not None:
        resource["activity"] = {"coding": [{"code": activity}]}

    # ── Reason → list[CodeableConcept] ────────────────────────────
    reason = doc.get("reason")
    if reason:
        resource["reason"] = [
            {"coding": [{"code": r}]} for r in reason
        ]

    # ── Policy (list[uri]) ───────────────────────────────────────
    policy = doc.get("policy")
    if policy is not None:
        resource["policy"] = policy

    # ── Location → Reference ─────────────────────────────────────
    location = doc.get("location")
    if location is not None:
        resource["location"] = {"reference": location}

    # ── Period (preserve as-is) ──────────────────────────────────
    period = doc.get("period")
    if period is not None:
        resource["period"] = period

    # ── Embed SL opinion as extension on _recorded ───────────────
    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        ext = opinion_to_fhir_extension(op)
        resource["_recorded"] = {"extension": [ext]}
        nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ═══════════════════════════════════════════════════════════════════
# Phase 6: Universal FHIR coverage — to_fhir handlers
# ═══════════════════════════════════════════════════════════════════


def _make_to_status_handler(
    resource_type: str,
    *,
    status_field: str = "status",
    passthrough_fields: tuple[str, ...] = (),
):
    """Factory for simple status-based to_fhir handlers.

    Reconstructs a minimal FHIR resource with the status field
    and embeds SL opinion as a FHIR extension on ``_<status_field>``.
    Fields listed in *passthrough_fields* are copied from the
    jsonld-ex document back into the FHIR resource (mirrors the
    symmetric parameter on ``_make_status_handler``).
    """

    def handler(
        doc: dict[str, Any],
    ) -> tuple[dict[str, Any], ConversionReport]:
        resource: dict[str, Any] = {
            "resourceType": resource_type,
            "id": doc.get("id"),
        }
        status = doc.get(status_field)
        if status is not None:
            resource[status_field] = status

        # Restore timestamp / metadata fields for round-trip fidelity
        for field_name in passthrough_fields:
            val = doc.get(field_name)
            if val is not None:
                resource[field_name] = val

        opinions = doc.get("opinions", [])
        nodes_converted = 0

        for entry in opinions:
            op: Opinion = entry["opinion"]
            field = entry.get("field", "")

            if field == status_field:
                ext = opinion_to_fhir_extension(op)
                resource[f"_{status_field}"] = {"extension": [ext]}
                nodes_converted += 1

        report = ConversionReport(success=True, nodes_converted=nodes_converted)
        return resource, report

    return handler


# ── Batch 1: Clinical workflow (simple status-based) ─────────────

_to_encounter_r4 = _make_to_status_handler(
    "Encounter", passthrough_fields=("period",),
)
_to_medication_request_r4 = _make_to_status_handler(
    "MedicationRequest", passthrough_fields=("authoredOn",),
)
_to_care_plan_r4 = _make_to_status_handler(
    "CarePlan", passthrough_fields=("period",),
)
_to_care_team_r4 = _make_to_status_handler(
    "CareTeam", passthrough_fields=("period",),
)
_to_imaging_study_r4 = _make_to_status_handler(
    "ImagingStudy", passthrough_fields=("started",),
)

# ── Batch 2: Administrative ─────────────────────────────────────

_to_device_r4 = _make_to_status_handler(
    "Device", passthrough_fields=("manufactureDate",),
)


def _to_patient_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex Patient → FHIR R4."""
    resource: dict[str, Any] = {
        "resourceType": "Patient",
        "id": doc.get("id"),
    }

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")
        if field == "data_quality":
            ext = opinion_to_fhir_extension(op)
            resource["_data_quality"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_active_boolean_r4(
    resource_type: str,
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex resource with boolean ``active`` → FHIR R4."""
    resource: dict[str, Any] = {
        "resourceType": resource_type,
        "id": doc.get("id"),
    }
    active = doc.get("active")
    if active is not None:
        resource["active"] = active

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")
        if field == "active":
            ext = opinion_to_fhir_extension(op)
            resource["_active"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_organization_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    return _to_active_boolean_r4("Organization", doc)


def _to_practitioner_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    return _to_active_boolean_r4("Practitioner", doc)


# ── Goal (to_fhir) ─ special: lifecycleStatus + achievementStatus ──


def _to_goal_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex Goal → FHIR R4."""
    resource: dict[str, Any] = {
        "resourceType": "Goal",
        "id": doc.get("id"),
    }
    lifecycle = doc.get("lifecycleStatus")
    if lifecycle is not None:
        resource["lifecycleStatus"] = lifecycle

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "lifecycleStatus":
            ext = opinion_to_fhir_extension(op)
            resource["_lifecycleStatus"] = {"extension": [ext]}
            nodes_converted += 1
        elif field == "achievementStatus":
            ext = opinion_to_fhir_extension(op)
            resource["_achievementStatus"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── Batch 3: Financial ───────────────────────────────────────────

_to_claim_r4 = _make_to_status_handler(
    "Claim", passthrough_fields=("created",),
)
_to_medication_administration_r4 = _make_to_status_handler(
    "MedicationAdministration", passthrough_fields=("effectiveDateTime",),
)


def _to_eob_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex ExplanationOfBenefit → FHIR R4."""
    resource: dict[str, Any] = {
        "resourceType": "ExplanationOfBenefit",
        "id": doc.get("id"),
    }
    status = doc.get("status")
    if status is not None:
        resource["status"] = status
    outcome = doc.get("outcome")
    if outcome is not None:
        resource["outcome"] = outcome
    created = doc.get("created")
    if created is not None:
        resource["created"] = created

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1
        elif field == "outcome":
            ext = opinion_to_fhir_extension(op)
            resource["_outcome"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


# ── to_fhir handler dispatch table ───────────────────────────────

def _to_service_request_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex ServiceRequest -> FHIR R4.

    Reconstructs the FHIR R4 ServiceRequest structure including
    intent, priority, requester Reference, and reasonReference
    array.  Embeds SL opinion as extension on ``_status``.
    """
    resource: dict[str, Any] = {
        "resourceType": "ServiceRequest",
        "id": doc.get("id"),
    }

    status = doc.get("status")
    if status is not None:
        resource["status"] = status

    intent = doc.get("intent")
    if intent is not None:
        resource["intent"] = intent

    priority = doc.get("priority")
    if priority is not None:
        resource["priority"] = priority

    authored_on = doc.get("authoredOn")
    if authored_on is not None:
        resource["authoredOn"] = authored_on

    requester = doc.get("requester")
    if requester is not None:
        resource["requester"] = {"reference": requester}

    reason_refs = doc.get("reason_references", [])
    if reason_refs:
        resource["reasonReference"] = [
            {"reference": ref} for ref in reason_refs
        ]

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_questionnaire_response_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex QuestionnaireResponse → FHIR R4.

    Reconstructs the FHIR R4 QuestionnaireResponse structure including
    source Reference, questionnaire canonical URL, and authored timestamp.
    Embeds SL opinion as extension on ``_status``.
    """
    resource: dict[str, Any] = {
        "resourceType": "QuestionnaireResponse",
        "id": doc.get("id"),
    }

    status = doc.get("status")
    if status is not None:
        resource["status"] = status

    authored = doc.get("authored")
    if authored is not None:
        resource["authored"] = authored

    questionnaire = doc.get("questionnaire")
    if questionnaire is not None:
        resource["questionnaire"] = questionnaire

    source = doc.get("source")
    if source is not None:
        resource["source"] = {"reference": source}

    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_specimen_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex Specimen → FHIR R4.

    Reconstructs the FHIR R4 Specimen structure including collection
    backbone (collector, quantity, collectedDateTime, method, bodySite),
    condition CodeableConcept array, type, subject, receivedTime,
    request/parent References, processing count, and note annotations.
    Embeds SL opinion as extension on ``_status``.
    """
    resource: dict[str, Any] = {
        "resourceType": "Specimen",
        "id": doc.get("id"),
    }

    status = doc.get("status")
    if status is not None:
        resource["status"] = status

    # Type
    type_text = doc.get("type")
    if type_text is not None:
        resource["type"] = {"text": type_text}

    # Subject
    subject_ref = doc.get("subject")
    if subject_ref is not None:
        resource["subject"] = {"reference": subject_ref}

    # ReceivedTime
    received_time = doc.get("receivedTime")
    if received_time is not None:
        resource["receivedTime"] = received_time

    # Request references
    request_refs = doc.get("request_references", [])
    if request_refs:
        resource["request"] = [{"reference": r} for r in request_refs]

    # Parent references
    parent_refs = doc.get("parent_references", [])
    if parent_refs:
        resource["parent"] = [{"reference": r} for r in parent_refs]

    # Collection backbone
    collection: dict[str, Any] = {}
    collector = doc.get("collector")
    if collector is not None:
        collection["collector"] = {"reference": collector}
    collected_dt = doc.get("collectedDateTime")
    if collected_dt is not None:
        collection["collectedDateTime"] = collected_dt
    qty = doc.get("collection_quantity")
    if qty is not None:
        collection["quantity"] = qty
    method_text = doc.get("collection_method")
    if method_text is not None:
        collection["method"] = {"text": method_text}
    body_site_text = doc.get("collection_bodySite")
    if body_site_text is not None:
        collection["bodySite"] = {"text": body_site_text}
    if collection:
        resource["collection"] = collection

    # Condition codes
    condition_codes = doc.get("condition_codes", [])
    if condition_codes:
        resource["condition"] = [
            {"coding": [{"code": c}]} for c in condition_codes
        ]

    # Processing (reconstruct from count)
    processing_count = doc.get("processing_count", 0)
    if processing_count > 0:
        resource["processing"] = [
            {"description": f"Step {i + 1}"} for i in range(processing_count)
        ]

    # Notes
    notes = doc.get("note")
    if notes is not None:
        resource["note"] = [{"text": t} for t in notes]

    # Embed SL opinion as extension on _status
    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_document_reference_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex DocumentReference -> FHIR R4.

    Reconstructs the FHIR R4 DocumentReference structure including
    dual status (status + docStatus), type CodeableConcept, subject/
    authenticator/custodian References, author References, relatesTo
    array, securityLabel CodeableConcept array, context backbone
    (encounter, period), date, and description.
    Embeds SL opinion as extension on ``_status``.
    """
    resource: dict[str, Any] = {
        "resourceType": "DocumentReference",
        "id": doc.get("id"),
    }

    status = doc.get("status")
    if status is not None:
        resource["status"] = status

    doc_status = doc.get("docStatus")
    if doc_status is not None:
        resource["docStatus"] = doc_status

    # Type as CodeableConcept
    type_text = doc.get("type")
    if type_text is not None:
        resource["type"] = {"text": type_text}

    # Subject reference
    subject_ref = doc.get("subject")
    if subject_ref is not None:
        resource["subject"] = {"reference": subject_ref}

    # Authenticator reference
    authenticator_ref = doc.get("authenticator")
    if authenticator_ref is not None:
        resource["authenticator"] = {"reference": authenticator_ref}

    # Custodian reference
    custodian_ref = doc.get("custodian")
    if custodian_ref is not None:
        resource["custodian"] = {"reference": custodian_ref}

    # Author references
    author_refs = doc.get("author_references", [])
    if author_refs:
        resource["author"] = [{"reference": r} for r in author_refs]

    # Date
    date = doc.get("date")
    if date is not None:
        resource["date"] = date

    # Description
    description = doc.get("description")
    if description is not None:
        resource["description"] = description

    # SecurityLabel codes
    security_label_codes = doc.get("securityLabel_codes", [])
    if security_label_codes:
        resource["securityLabel"] = [
            {"coding": [{"code": c}]} for c in security_label_codes
        ]

    # RelatesTo
    relates_to = doc.get("relatesTo", [])
    if relates_to:
        resource["relatesTo"] = [
            {
                "code": rt.get("code", ""),
                "target": {"reference": rt.get("target", "")},
            }
            for rt in relates_to
        ]

    # Context backbone
    context: dict[str, Any] = {}
    context_enc_refs = doc.get("context_encounter_references", [])
    if context_enc_refs:
        context["encounter"] = [{"reference": r} for r in context_enc_refs]
    context_period = doc.get("context_period")
    if context_period is not None:
        context["period"] = context_period
    if context:
        resource["context"] = context

    # Embed SL opinion as extension on _status
    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


def _to_coverage_r4(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Export jsonld-ex Coverage → FHIR R4.

    Reconstructs the FHIR R4 Coverage structure including beneficiary,
    subscriber, and payor References; type and relationship
    CodeableConcepts; period, dependent, order, and network scalars.
    Embeds SL opinion as extension on ``_status``.
    """
    resource: dict[str, Any] = {
        "resourceType": "Coverage",
        "id": doc.get("id"),
    }

    status = doc.get("status")
    if status is not None:
        resource["status"] = status

    # Period (pass-through)
    period = doc.get("period")
    if period is not None:
        resource["period"] = period

    # Type as CodeableConcept
    type_text = doc.get("type")
    if type_text is not None:
        resource["type"] = {"text": type_text}

    # Beneficiary reference
    beneficiary_ref = doc.get("beneficiary")
    if beneficiary_ref is not None:
        resource["beneficiary"] = {"reference": beneficiary_ref}

    # Subscriber reference
    subscriber_ref = doc.get("subscriber")
    if subscriber_ref is not None:
        resource["subscriber"] = {"reference": subscriber_ref}

    # Payor references (0..*)
    payor_refs = doc.get("payor_references", [])
    if payor_refs:
        resource["payor"] = [{"reference": r} for r in payor_refs]

    # Dependent
    dependent = doc.get("dependent")
    if dependent is not None:
        resource["dependent"] = dependent

    # Relationship as CodeableConcept
    relationship_code = doc.get("relationship")
    if relationship_code is not None:
        resource["relationship"] = {"coding": [{"code": relationship_code}]}

    # Order
    order = doc.get("order")
    if order is not None:
        resource["order"] = order

    # Network
    network = doc.get("network")
    if network is not None:
        resource["network"] = network

    # Embed SL opinion as extension on _status
    opinions = doc.get("opinions", [])
    nodes_converted = 0

    for entry in opinions:
        op: Opinion = entry["opinion"]
        field = entry.get("field", "")

        if field == "status":
            ext = opinion_to_fhir_extension(op)
            resource["_status"] = {"extension": [ext]}
            nodes_converted += 1

    report = ConversionReport(success=True, nodes_converted=nodes_converted)
    return resource, report


_TO_FHIR_HANDLERS = {
    "RiskAssessment": _to_risk_assessment_r4,
    "Observation": _to_observation_r4,
    "DiagnosticReport": _to_diagnostic_report_r4,
    "Condition": _to_condition_r4,
    "AllergyIntolerance": _to_allergy_intolerance_r4,
    "MedicationStatement": _to_medication_statement_r4,
    "ClinicalImpression": _to_clinical_impression_r4,
    "DetectedIssue": _to_detected_issue_r4,
    "Immunization": _to_immunization_r4,
    "FamilyMemberHistory": _to_family_member_history_r4,
    "Procedure": _to_procedure_r4,
    "Consent": _to_consent_r4,
    "Provenance": _to_provenance_r4,
    # Phase 6 — universal coverage
    "Encounter": _to_encounter_r4,
    "MedicationRequest": _to_medication_request_r4,
    "CarePlan": _to_care_plan_r4,
    "Goal": _to_goal_r4,
    "CareTeam": _to_care_team_r4,
    "ImagingStudy": _to_imaging_study_r4,
    "Patient": _to_patient_r4,
    "Organization": _to_organization_r4,
    "Practitioner": _to_practitioner_r4,
    "Device": _to_device_r4,
    "Claim": _to_claim_r4,
    "ExplanationOfBenefit": _to_eob_r4,
    "MedicationAdministration": _to_medication_administration_r4,
    # Phase 7A — high-value clinical expansion
    "ServiceRequest": _to_service_request_r4,
    "QuestionnaireResponse": _to_questionnaire_response_r4,
    "Specimen": _to_specimen_r4,
    # Phase 7B — US Core completeness
    "DocumentReference": _to_document_reference_r4,
    "Coverage": _to_coverage_r4,
}
