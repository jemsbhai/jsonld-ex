"""Tests for Phase 7A: QuestionnaireResponse FHIR R4 resource support.

RED PHASE: Tests written before implementation.

QuestionnaireResponse is the strongest SL use case of any remaining type.
Patient self-report is fundamentally an epistemic problem: recall bias,
social desirability bias, health literacy, cognitive load.  Validated
instruments have known reliability coefficients (Cronbach's α, test-retest)
that map directly to SL uncertainty budgets.

Proposition under assessment:
    "This questionnaire response is complete and reliable."

Epistemic signals extracted (multiplicative):
    1. status: in-progress/completed/amended/entered-in-error/stopped
    2. source: reporter type (Patient vs Practitioner vs RelatedPerson)
    3. item completeness: ratio of answered items to total items
    4. authored: timestamp passthrough for temporal decay

This is a CUSTOM handler (not a simple status handler) because:
    1. Source type modulates uncertainty (patient ≠ practitioner)
    2. Item completeness modulates uncertainty (partial ≠ complete)
    3. These signals interact multiplicatively with status

All epistemic parameters are CONFIGURABLE via ``handler_config`` on
``from_fhir()``, enabling researchers to tune for their specific
datasets, clinical contexts, and instrument reliability profiles.

Design decisions documented here for scientific rigor:

    Status probability mappings encode "response completeness and reliability":
        completed=0.85       — finished, reviewed, highest completeness
        amended=0.80         — corrected after completion, slightly lower
        in-progress=0.55     — partially filled, genuinely uncertain
        stopped=0.30         — abandoned, low reliability
        entered-in-error=0.50 — data integrity compromised

    Status uncertainty mappings:
        completed=0.15       — low uncertainty
        amended=0.20         — slightly higher (correction implies original error)
        in-progress=0.35     — substantial uncertainty about final form
        stopped=0.25         — moderate (we know it's abandoned)
        entered-in-error=0.70 — very high

    Source reliability multipliers encode reporter trustworthiness:
        Practitioner=0.7     — clinical expertise, lowest uncertainty
        PractitionerRole=0.7 — same as Practitioner (role-attributed)
        RelatedPerson=1.2    — proxy reporter, moderate uncertainty increase
        Patient=1.3          — self-report, highest uncertainty (bias, literacy)

    Item completeness thresholds and multipliers:
        ratio < 0.5  → ×1.3 (low completeness, significant data gaps)
        0.5 ≤ ratio ≤ 0.8 → ×1.0 (moderate completeness, baseline)
        ratio > 0.8  → ×0.8 (high completeness, good data quality)

    Configurability rationale:
        Different screening instruments have different reliability profiles.
        PHQ-9 (depression) may have different optimal parameters than
        GAD-7 (anxiety) or AUDIT (alcohol).  Researchers must be able to
        tune source multipliers, completeness thresholds, and status
        mappings for their specific instrument and population.

References:
    - HL7 FHIR R4 QuestionnaireResponse:
      https://hl7.org/fhir/R4/questionnaireresponse.html
    - Jøsang, A. (2016). Subjective Logic. Springer.
    - Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001).
      The PHQ-9: validity of a brief depression severity measure.
      Journal of General Internal Medicine, 16(9), 606-613.
"""

import pytest
from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    FHIR_EXTENSION_URL,
    SUPPORTED_RESOURCE_TYPES,
    opinion_to_fhir_extension,
)
from jsonld_ex.fhir_interop._constants import (
    QR_STATUS_PROBABILITY,
    QR_STATUS_UNCERTAINTY,
    QR_SOURCE_RELIABILITY_MULTIPLIER,
    QR_COMPLETENESS_THRESHOLDS,
)


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════


def _questionnaire_response(
    *,
    status="completed",
    source_type=None,
    source_ref=None,
    items=None,
    authored=None,
    id_="qr-1",
    questionnaire=None,
):
    """Build a minimal valid FHIR R4 QuestionnaireResponse.

    Args:
        status: FHIR status code.
        source_type: Resource type for the source reference
            (e.g. "Patient", "Practitioner", "RelatedPerson").
        source_ref: Full reference string; if None and source_type
            is set, auto-generates one.
        items: List of item dicts (FHIR QuestionnaireResponse.item).
            If None, no items are included.
        authored: ISO datetime string for when the response was authored.
        id_: Resource id.
        questionnaire: Canonical URL of the questionnaire definition.
    """
    resource = {
        "resourceType": "QuestionnaireResponse",
        "id": id_,
        "status": status,
        "subject": {"reference": "Patient/p1"},
    }
    if source_type is not None:
        ref = source_ref or f"{source_type}/src-1"
        resource["source"] = {"reference": ref}
    if items is not None:
        resource["item"] = items
    if authored is not None:
        resource["authored"] = authored
    if questionnaire is not None:
        resource["questionnaire"] = questionnaire
    return resource


def _make_items(answered: int, total: int):
    """Build a list of FHIR QuestionnaireResponse items.

    Creates ``total`` items, the first ``answered`` of which have
    an ``answer`` array.  The rest have no answer (unanswered).
    """
    items = []
    for i in range(total):
        item = {"linkId": f"q{i + 1}", "text": f"Question {i + 1}"}
        if i < answered:
            item["answer"] = [{"valueString": f"Answer {i + 1}"}]
        items.append(item)
    return items


# ═══════════════════════════════════════════════════════════════════
# Registration: QuestionnaireResponse in SUPPORTED_RESOURCE_TYPES
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseRegistration:
    """QuestionnaireResponse must be registered as a supported type."""

    def test_in_supported_types(self):
        assert "QuestionnaireResponse" in SUPPORTED_RESOURCE_TYPES


# ═══════════════════════════════════════════════════════════════════
# Constants: Mapping tables exist and are well-formed
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseConstants:
    """Verify constant mapping tables exist and have valid values."""

    # ── Status probability map ────────────────────────────────────

    def test_status_probability_covers_all_fhir_codes(self):
        """All FHIR R4 QuestionnaireResponse.status codes must be mapped."""
        expected = {
            "in-progress", "completed", "amended",
            "entered-in-error", "stopped",
        }
        assert set(QR_STATUS_PROBABILITY.keys()) == expected

    def test_status_probability_values_in_range(self):
        for code, prob in QR_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, f"status '{code}' prob={prob} out of range"

    def test_status_uncertainty_covers_all_fhir_codes(self):
        expected = {
            "in-progress", "completed", "amended",
            "entered-in-error", "stopped",
        }
        assert set(QR_STATUS_UNCERTAINTY.keys()) == expected

    def test_status_uncertainty_values_in_range(self):
        for code, u in QR_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, f"status '{code}' uncertainty={u} out of range"

    # ── Source reliability multiplier map ─────────────────────────

    def test_source_multiplier_covers_key_types(self):
        """At minimum: Patient, Practitioner, RelatedPerson, PractitionerRole."""
        required = {"Patient", "Practitioner", "RelatedPerson", "PractitionerRole"}
        assert required.issubset(set(QR_SOURCE_RELIABILITY_MULTIPLIER.keys()))

    def test_source_multiplier_values_positive(self):
        for src, mult in QR_SOURCE_RELIABILITY_MULTIPLIER.items():
            assert mult > 0.0, f"source '{src}' multiplier={mult} must be positive"

    def test_practitioner_lower_than_patient(self):
        """Practitioner reports should produce less uncertainty than patient."""
        assert (
            QR_SOURCE_RELIABILITY_MULTIPLIER["Practitioner"]
            < QR_SOURCE_RELIABILITY_MULTIPLIER["Patient"]
        )

    def test_related_person_between_practitioner_and_patient(self):
        """RelatedPerson should fall between Practitioner and Patient."""
        pract = QR_SOURCE_RELIABILITY_MULTIPLIER["Practitioner"]
        related = QR_SOURCE_RELIABILITY_MULTIPLIER["RelatedPerson"]
        patient = QR_SOURCE_RELIABILITY_MULTIPLIER["Patient"]
        assert pract < related < patient

    def test_practitioner_role_same_as_practitioner(self):
        """PractitionerRole should have same multiplier as Practitioner."""
        assert (
            QR_SOURCE_RELIABILITY_MULTIPLIER["PractitionerRole"]
            == QR_SOURCE_RELIABILITY_MULTIPLIER["Practitioner"]
        )

    # ── Completeness thresholds ───────────────────────────────────

    def test_completeness_thresholds_exist(self):
        required_keys = {
            "low_threshold", "high_threshold",
            "low_multiplier", "mid_multiplier", "high_multiplier",
        }
        assert required_keys.issubset(set(QR_COMPLETENESS_THRESHOLDS.keys()))

    def test_thresholds_ordered(self):
        """low_threshold < high_threshold."""
        assert (
            QR_COMPLETENESS_THRESHOLDS["low_threshold"]
            < QR_COMPLETENESS_THRESHOLDS["high_threshold"]
        )

    def test_thresholds_in_unit_interval(self):
        assert 0.0 < QR_COMPLETENESS_THRESHOLDS["low_threshold"] < 1.0
        assert 0.0 < QR_COMPLETENESS_THRESHOLDS["high_threshold"] <= 1.0

    def test_low_multiplier_increases_uncertainty(self):
        """Low completeness should raise uncertainty (multiplier > 1.0)."""
        assert QR_COMPLETENESS_THRESHOLDS["low_multiplier"] > 1.0

    def test_high_multiplier_decreases_uncertainty(self):
        """High completeness should reduce uncertainty (multiplier < 1.0)."""
        assert QR_COMPLETENESS_THRESHOLDS["high_multiplier"] < 1.0

    def test_mid_multiplier_is_baseline(self):
        """Mid completeness should be baseline (multiplier = 1.0)."""
        assert QR_COMPLETENESS_THRESHOLDS["mid_multiplier"] == 1.0

    # ── Epistemic ordering within status map ──────────────────────

    def test_completed_higher_prob_than_in_progress(self):
        assert QR_STATUS_PROBABILITY["completed"] > QR_STATUS_PROBABILITY["in-progress"]

    def test_completed_higher_prob_than_stopped(self):
        assert QR_STATUS_PROBABILITY["completed"] > QR_STATUS_PROBABILITY["stopped"]

    def test_amended_between_completed_and_in_progress(self):
        assert (
            QR_STATUS_PROBABILITY["in-progress"]
            < QR_STATUS_PROBABILITY["amended"]
            <= QR_STATUS_PROBABILITY["completed"]
        )

    def test_entered_in_error_high_uncertainty(self):
        assert QR_STATUS_UNCERTAINTY["entered-in-error"] >= 0.50


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Basic status-based conversion
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseFromFhir:
    """Test from_fhir() for QuestionnaireResponse resources."""

    def test_basic_completed(self):
        """Completed response should produce a valid opinion."""
        doc, report = from_fhir(_questionnaire_response(status="completed"))
        assert report.success
        assert report.nodes_converted >= 1
        assert doc["@type"] == "fhir:QuestionnaireResponse"
        assert doc["id"] == "qr-1"
        assert doc["status"] == "completed"
        assert len(doc["opinions"]) >= 1

    def test_opinion_is_valid_sl_triple(self):
        """Opinion must satisfy b + d + u = 1, all in [0, 1]."""
        doc, _ = from_fhir(_questionnaire_response())
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert 0.0 <= op.belief <= 1.0
        assert 0.0 <= op.disbelief <= 1.0
        assert 0.0 <= op.uncertainty <= 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    @pytest.mark.parametrize("status", [
        "in-progress", "completed", "amended",
        "entered-in-error", "stopped",
    ])
    def test_all_status_codes_produce_valid_opinion(self, status):
        """Every FHIR R4 QuestionnaireResponse.status code must produce valid opinion."""
        doc, report = from_fhir(_questionnaire_response(status=status))
        assert report.success
        assert len(doc["opinions"]) >= 1
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_status_field_preserved(self):
        doc, _ = from_fhir(_questionnaire_response(status="amended"))
        assert doc["status"] == "amended"

    def test_opinion_source_is_reconstructed(self):
        """Without extension, source should be 'reconstructed'."""
        doc, _ = from_fhir(_questionnaire_response())
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_opinion_field_is_status(self):
        """Primary opinion should be on the 'status' field."""
        doc, _ = from_fhir(_questionnaire_response())
        assert doc["opinions"][0]["field"] == "status"

    def test_completed_higher_belief_than_in_progress(self):
        """Completed should have higher belief than in-progress."""
        doc_completed, _ = from_fhir(
            _questionnaire_response(status="completed")
        )
        doc_inprog, _ = from_fhir(
            _questionnaire_response(status="in-progress")
        )
        b_completed = doc_completed["opinions"][0]["opinion"].belief
        b_inprog = doc_inprog["opinions"][0]["opinion"].belief
        assert b_completed > b_inprog

    def test_stopped_low_belief(self):
        """Stopped responses should have low belief."""
        doc, _ = from_fhir(_questionnaire_response(status="stopped"))
        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.40

    def test_entered_in_error_high_uncertainty(self):
        """Entered-in-error should have elevated uncertainty."""
        doc, _ = from_fhir(
            _questionnaire_response(status="entered-in-error")
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty >= 0.35


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Source reliability modulation
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseSourceModulation:
    """Source type should modulate uncertainty: Practitioner < RelatedPerson < Patient."""

    def test_practitioner_less_uncertain_than_patient(self):
        """Practitioner-reported should have less uncertainty than patient-reported."""
        doc_pract, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Practitioner")
        )
        doc_patient, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Patient")
        )
        u_pract = doc_pract["opinions"][0]["opinion"].uncertainty
        u_patient = doc_patient["opinions"][0]["opinion"].uncertainty
        assert u_pract < u_patient

    def test_related_person_between_practitioner_and_patient(self):
        """RelatedPerson should fall between Practitioner and Patient."""
        doc_pract, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Practitioner")
        )
        doc_related, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="RelatedPerson")
        )
        doc_patient, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Patient")
        )
        u_pract = doc_pract["opinions"][0]["opinion"].uncertainty
        u_related = doc_related["opinions"][0]["opinion"].uncertainty
        u_patient = doc_patient["opinions"][0]["opinion"].uncertainty
        assert u_pract < u_related < u_patient

    @pytest.mark.parametrize("source_type", [
        "Patient", "Practitioner", "RelatedPerson", "PractitionerRole",
    ])
    def test_all_source_types_produce_valid_opinion(self, source_type):
        doc, report = from_fhir(
            _questionnaire_response(status="completed", source_type=source_type)
        )
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_no_source_uses_baseline(self):
        """Omitting source should use baseline (multiplier 1.0)."""
        doc_no_src, _ = from_fhir(
            _questionnaire_response(status="completed", source_type=None)
        )
        # No source → no multiplier adjustment → baseline uncertainty
        op = doc_no_src["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_unknown_source_type_uses_baseline(self):
        """An unrecognised source type should use multiplier 1.0."""
        doc_unknown, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Device")
        )
        doc_no_src, _ = from_fhir(
            _questionnaire_response(status="completed", source_type=None)
        )
        u_unknown = doc_unknown["opinions"][0]["opinion"].uncertainty
        u_no_src = doc_no_src["opinions"][0]["opinion"].uncertainty
        assert abs(u_unknown - u_no_src) < 1e-9

    def test_practitioner_role_same_as_practitioner(self):
        """PractitionerRole should produce same uncertainty as Practitioner."""
        doc_pract, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Practitioner")
        )
        doc_role, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="PractitionerRole")
        )
        u_pract = doc_pract["opinions"][0]["opinion"].uncertainty
        u_role = doc_role["opinions"][0]["opinion"].uncertainty
        assert abs(u_pract - u_role) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Item completeness modulation
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseCompletenessModulation:
    """Item completeness ratio modulates uncertainty."""

    def test_high_completeness_lower_uncertainty(self):
        """90% answered items → lower uncertainty than 30% answered."""
        items_high = _make_items(answered=9, total=10)
        items_low = _make_items(answered=3, total=10)

        doc_high, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_high)
        )
        doc_low, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_low)
        )
        u_high = doc_high["opinions"][0]["opinion"].uncertainty
        u_low = doc_low["opinions"][0]["opinion"].uncertainty
        assert u_high < u_low

    def test_all_answered_lowest_uncertainty(self):
        """100% completion should give lowest uncertainty for a given status."""
        items_all = _make_items(answered=10, total=10)
        items_mid = _make_items(answered=6, total=10)

        doc_all, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_all)
        )
        doc_mid, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_mid)
        )
        u_all = doc_all["opinions"][0]["opinion"].uncertainty
        u_mid = doc_mid["opinions"][0]["opinion"].uncertainty
        assert u_all < u_mid

    def test_no_items_uses_baseline(self):
        """No items present → no completeness adjustment (baseline)."""
        doc, _ = from_fhir(
            _questionnaire_response(status="completed", items=None)
        )
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_empty_items_list_uses_baseline(self):
        """Empty items list → no items to assess, use baseline."""
        doc, _ = from_fhir(
            _questionnaire_response(status="completed", items=[])
        )
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_single_answered_item(self):
        """Single answered item out of one → 100% completeness."""
        items = _make_items(answered=1, total=1)
        doc, report = from_fhir(
            _questionnaire_response(status="completed", items=items)
        )
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_zero_answered_highest_uncertainty(self):
        """0 out of 10 answered → highest completeness penalty."""
        items_zero = _make_items(answered=0, total=10)
        items_some = _make_items(answered=5, total=10)

        doc_zero, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_zero)
        )
        doc_some, _ = from_fhir(
            _questionnaire_response(status="completed", items=items_some)
        )
        u_zero = doc_zero["opinions"][0]["opinion"].uncertainty
        u_some = doc_some["opinions"][0]["opinion"].uncertainty
        assert u_zero > u_some


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Signal interactions
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseSignalInteractions:
    """Test multiplicative interaction of status × source × completeness."""

    def test_best_case_lowest_uncertainty(self):
        """completed + practitioner + 100% items → minimum uncertainty."""
        items = _make_items(answered=10, total=10)
        doc, _ = from_fhir(
            _questionnaire_response(
                status="completed",
                source_type="Practitioner",
                items=items,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 0.12

    def test_worst_case_highest_uncertainty(self):
        """in-progress + patient + 20% items → high uncertainty."""
        items = _make_items(answered=2, total=10)
        doc, _ = from_fhir(
            _questionnaire_response(
                status="in-progress",
                source_type="Patient",
                items=items,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.30

    def test_uncertainty_always_clamped_below_one(self):
        """Even worst-case signal combination must clamp u < 1.0."""
        items = _make_items(answered=0, total=10)
        doc, _ = from_fhir(
            _questionnaire_response(
                status="entered-in-error",
                source_type="Patient",
                items=items,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_uncertainty_always_positive(self):
        """Even best-case signal combination must have u > 0."""
        items = _make_items(answered=20, total=20)
        doc, _ = from_fhir(
            _questionnaire_response(
                status="completed",
                source_type="Practitioner",
                items=items,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.0


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Extension recovery (exact round-trip)
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseExtensionRecovery:
    """When a jsonld-ex FHIR extension is present, recover the exact opinion."""

    def test_extension_recovery_overrides_reconstruction(self):
        """An embedded extension should be recovered exactly."""
        original = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        resource = _questionnaire_response(status="completed")
        resource["_status"] = {"extension": [ext]}

        doc, report = from_fhir(resource)
        assert report.success
        recovered = doc["opinions"][0]["opinion"]
        assert doc["opinions"][0]["source"] == "extension"
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9

    def test_extension_ignores_metadata_signals(self):
        """When extension is present, source/completeness don't affect opinion."""
        original = Opinion(belief=0.30, disbelief=0.50, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        items = _make_items(answered=10, total=10)
        resource = _questionnaire_response(
            status="completed",
            source_type="Practitioner",
            items=items,
        )
        resource["_status"] = {"extension": [ext]}

        doc, _ = from_fhir(resource)
        recovered = doc["opinions"][0]["opinion"]
        assert abs(recovered.belief - 0.30) < 1e-9
        assert abs(recovered.disbelief - 0.50) < 1e-9

    def test_wrong_extension_url_triggers_reconstruction(self):
        """An extension with a different URL should not be recovered."""
        resource = _questionnaire_response(status="completed")
        resource["_status"] = {
            "extension": [{
                "url": "https://other-system.example/extension",
                "valueString": "not-our-extension",
            }]
        }

        doc, _ = from_fhir(resource)
        assert doc["opinions"][0]["source"] == "reconstructed"


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Metadata passthrough
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseMetadataPassthrough:
    """Key metadata fields should be preserved in the jsonld-ex document."""

    def test_authored_preserved(self):
        doc, _ = from_fhir(
            _questionnaire_response(authored="2024-06-15T14:30:00Z")
        )
        assert doc["authored"] == "2024-06-15T14:30:00Z"

    def test_authored_absent_when_not_set(self):
        doc, _ = from_fhir(_questionnaire_response(authored=None))
        assert "authored" not in doc

    def test_questionnaire_preserved(self):
        doc, _ = from_fhir(
            _questionnaire_response(
                questionnaire="http://example.org/Questionnaire/PHQ-9"
            )
        )
        assert doc["questionnaire"] == "http://example.org/Questionnaire/PHQ-9"

    def test_source_reference_preserved(self):
        doc, _ = from_fhir(
            _questionnaire_response(
                source_type="Patient",
                source_ref="Patient/p42",
            )
        )
        assert doc["source"] == "Patient/p42"

    def test_source_absent_when_not_set(self):
        doc, _ = from_fhir(_questionnaire_response(source_type=None))
        assert "source" not in doc

    def test_item_count_preserved(self):
        """Item count metadata should be available for downstream analysis."""
        items = _make_items(answered=7, total=10)
        doc, _ = from_fhir(
            _questionnaire_response(status="completed", items=items)
        )
        assert "item_count" in doc
        assert doc["item_count"]["total"] == 10
        assert doc["item_count"]["answered"] == 7


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Configurability via handler_config
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseConfigurability:
    """All epistemic parameters should be tunable via handler_config."""

    def test_handler_config_accepted(self):
        """from_fhir() should accept handler_config kwarg without error."""
        doc, report = from_fhir(
            _questionnaire_response(),
            handler_config={},
        )
        assert report.success

    def test_none_config_uses_defaults(self):
        """handler_config=None should behave identically to no config."""
        doc_none, _ = from_fhir(
            _questionnaire_response(), handler_config=None,
        )
        doc_default, _ = from_fhir(
            _questionnaire_response(),
        )
        op_none = doc_none["opinions"][0]["opinion"]
        op_default = doc_default["opinions"][0]["opinion"]
        assert abs(op_none.belief - op_default.belief) < 1e-9
        assert abs(op_none.uncertainty - op_default.uncertainty) < 1e-9

    def test_empty_config_uses_defaults(self):
        """handler_config={} should behave identically to no config."""
        doc_empty, _ = from_fhir(
            _questionnaire_response(), handler_config={},
        )
        doc_default, _ = from_fhir(
            _questionnaire_response(),
        )
        op_empty = doc_empty["opinions"][0]["opinion"]
        op_default = doc_default["opinions"][0]["opinion"]
        assert abs(op_empty.belief - op_default.belief) < 1e-9
        assert abs(op_empty.uncertainty - op_default.uncertainty) < 1e-9

    def test_override_source_reliability_multiplier(self):
        """Custom source multiplier should change uncertainty."""
        # Default: Patient=1.3.  Override: Patient=2.0 (much more uncertain)
        config = {
            "source_reliability_multiplier": {"Patient": 2.0},
        }
        doc_default, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Patient"),
        )
        doc_custom, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Patient"),
            handler_config=config,
        )
        u_default = doc_default["opinions"][0]["opinion"].uncertainty
        u_custom = doc_custom["opinions"][0]["opinion"].uncertainty
        assert u_custom > u_default

    def test_override_completeness_thresholds(self):
        """Custom completeness thresholds should change classification."""
        # Default: high_threshold=0.8.  Override: high_threshold=0.5
        # Now 60% completion qualifies as "high" and gets reduction
        items = _make_items(answered=6, total=10)  # 60% completion
        config = {
            "completeness_thresholds": {
                "low_threshold": 0.3,
                "high_threshold": 0.5,
                "low_multiplier": 1.3,
                "mid_multiplier": 1.0,
                "high_multiplier": 0.8,
            },
        }
        doc_default, _ = from_fhir(
            _questionnaire_response(status="completed", items=items),
        )
        doc_custom, _ = from_fhir(
            _questionnaire_response(status="completed", items=items),
            handler_config=config,
        )
        u_default = doc_default["opinions"][0]["opinion"].uncertainty
        u_custom = doc_custom["opinions"][0]["opinion"].uncertainty
        # With custom config, 60% is "high" → multiplier 0.8 → less uncertain
        assert u_custom < u_default

    def test_override_status_probability(self):
        """Custom status probability should change belief."""
        config = {
            "status_probability": {"completed": 0.99},
        }
        doc_default, _ = from_fhir(
            _questionnaire_response(status="completed"),
        )
        doc_custom, _ = from_fhir(
            _questionnaire_response(status="completed"),
            handler_config=config,
        )
        b_default = doc_default["opinions"][0]["opinion"].belief
        b_custom = doc_custom["opinions"][0]["opinion"].belief
        assert b_custom > b_default

    def test_override_status_uncertainty(self):
        """Custom status uncertainty should change base uncertainty."""
        config = {
            "status_uncertainty": {"completed": 0.05},
        }
        doc_default, _ = from_fhir(
            _questionnaire_response(status="completed"),
        )
        doc_custom, _ = from_fhir(
            _questionnaire_response(status="completed"),
            handler_config=config,
        )
        u_default = doc_default["opinions"][0]["opinion"].uncertainty
        u_custom = doc_custom["opinions"][0]["opinion"].uncertainty
        assert u_custom < u_default

    def test_partial_override_merges_with_defaults(self):
        """Overriding one source type should not affect others."""
        config = {
            "source_reliability_multiplier": {"Patient": 2.0},
        }
        # Practitioner should still use default multiplier
        doc_pract, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Practitioner"),
            handler_config=config,
        )
        doc_pract_default, _ = from_fhir(
            _questionnaire_response(status="completed", source_type="Practitioner"),
        )
        u_pract = doc_pract["opinions"][0]["opinion"].uncertainty
        u_pract_default = doc_pract_default["opinions"][0]["opinion"].uncertainty
        assert abs(u_pract - u_pract_default) < 1e-9

    def test_config_does_not_affect_other_resource_types(self):
        """handler_config should be silently ignored by non-QR handlers."""
        config = {
            "source_reliability_multiplier": {"Patient": 2.0},
        }
        resource = {
            "resourceType": "Observation",
            "id": "obs-1",
            "status": "final",
        }
        doc_default, _ = from_fhir(resource)
        doc_config, _ = from_fhir(resource, handler_config=config)
        op_default = doc_default["opinions"][0]["opinion"]
        op_config = doc_config["opinions"][0]["opinion"]
        assert abs(op_default.belief - op_config.belief) < 1e-9

    def test_config_backward_compatible(self):
        """Existing from_fhir() calls without handler_config must still work."""
        # This tests that all existing resource types work unchanged
        for rtype in ["Observation", "Condition", "ServiceRequest"]:
            resource = {
                "resourceType": rtype,
                "id": f"{rtype.lower()}-1",
                "status": "active" if rtype != "Condition" else None,
            }
            if rtype == "Condition":
                resource["verificationStatus"] = {
                    "coding": [{"code": "confirmed"}]
                }
                del resource["status"]
            if rtype == "ServiceRequest":
                resource["intent"] = "order"
            doc, report = from_fhir(resource)
            assert report.success


# ═══════════════════════════════════════════════════════════════════
# to_fhir(): Export to FHIR R4
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseToFhir:
    """Test to_fhir() export for QuestionnaireResponse."""

    def test_basic_export(self):
        """A jsonld-ex QuestionnaireResponse should export to valid FHIR."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)
        assert report.success
        assert resource["resourceType"] == "QuestionnaireResponse"
        assert resource["id"] == "qr-1"
        assert resource["status"] == "completed"

    def test_opinion_embedded_as_extension(self):
        """SL opinion should be embedded as a FHIR extension on _status."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert "_status" in resource
        extensions = resource["_status"]["extension"]
        assert len(extensions) >= 1
        assert extensions[0]["url"] == FHIR_EXTENSION_URL

    def test_authored_exported(self):
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "authored": "2024-06-15T14:30:00Z",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["authored"] == "2024-06-15T14:30:00Z"

    def test_questionnaire_exported(self):
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "questionnaire": "http://example.org/Questionnaire/PHQ-9",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["questionnaire"] == "http://example.org/Questionnaire/PHQ-9"

    def test_source_exported_as_reference(self):
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "source": "Patient/p42",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["source"] == {"reference": "Patient/p42"}

    def test_source_absent_when_not_in_doc(self):
        doc = {
            "@type": "fhir:QuestionnaireResponse",
            "id": "qr-1",
            "status": "completed",
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert "source" not in resource


# ═══════════════════════════════════════════════════════════════════
# Round-trip fidelity: from_fhir() → to_fhir() → from_fhir()
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseRoundTrip:
    """Round-trip through from_fhir → to_fhir should preserve opinions exactly."""

    def test_opinion_preserved_via_extension(self):
        """After from_fhir → to_fhir → from_fhir, opinion should be exact."""
        items = _make_items(answered=8, total=10)
        original = _questionnaire_response(
            status="completed",
            source_type="Patient",
            items=items,
            authored="2024-06-15T14:30:00Z",
        )
        # Forward pass
        doc1, _ = from_fhir(original)
        op1 = doc1["opinions"][0]["opinion"]

        # Export back to FHIR (embeds extension)
        fhir_resource, _ = to_fhir(doc1)

        # Second import (should recover from extension)
        doc2, _ = from_fhir(fhir_resource)
        op2 = doc2["opinions"][0]["opinion"]

        assert doc2["opinions"][0]["source"] == "extension"
        assert abs(op1.belief - op2.belief) < 1e-9
        assert abs(op1.disbelief - op2.disbelief) < 1e-9
        assert abs(op1.uncertainty - op2.uncertainty) < 1e-9

    def test_resource_type_preserved(self):
        original = _questionnaire_response()
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["resourceType"] == "QuestionnaireResponse"

    def test_id_preserved(self):
        original = _questionnaire_response(id_="qr-roundtrip-99")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["id"] == "qr-roundtrip-99"

    def test_status_preserved(self):
        original = _questionnaire_response(status="amended")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["status"] == "amended"

    def test_authored_preserved_through_round_trip(self):
        original = _questionnaire_response(authored="2024-01-01T00:00:00Z")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["authored"] == "2024-01-01T00:00:00Z"

    def test_questionnaire_preserved_through_round_trip(self):
        original = _questionnaire_response(
            questionnaire="http://example.org/Questionnaire/GAD-7"
        )
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["questionnaire"] == "http://example.org/Questionnaire/GAD-7"

    def test_source_preserved_through_round_trip(self):
        original = _questionnaire_response(
            source_type="Practitioner",
            source_ref="Practitioner/dr-1",
        )
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["source"] == {"reference": "Practitioner/dr-1"}


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestQuestionnaireResponseEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_missing_id(self):
        """Resource without id should not crash."""
        resource = _questionnaire_response()
        del resource["id"]
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["id"] is None

    def test_missing_status_uses_default(self):
        """If status is absent, should use a safe default."""
        resource = _questionnaire_response()
        del resource["status"]
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_novel_status_code_handled(self):
        """A future/unknown status code should not crash."""
        resource = _questionnaire_response()
        resource["status"] = "future-status-code"
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_items_with_nested_items(self):
        """Items containing sub-items should be counted correctly."""
        items = [
            {
                "linkId": "group1",
                "text": "Group 1",
                "item": [
                    {
                        "linkId": "q1.1",
                        "text": "Sub-question 1",
                        "answer": [{"valueString": "Yes"}],
                    },
                    {
                        "linkId": "q1.2",
                        "text": "Sub-question 2",
                        # No answer — unanswered
                    },
                ],
            },
            {
                "linkId": "q2",
                "text": "Question 2",
                "answer": [{"valueInteger": 5}],
            },
        ]
        doc, report = from_fhir(
            _questionnaire_response(status="completed", items=items)
        )
        assert report.success
        # Should count leaf items: q1.1 (answered), q1.2 (unanswered), q2 (answered)
        # Completeness: 2/3 ≈ 0.67
        assert doc["item_count"]["total"] == 3
        assert doc["item_count"]["answered"] == 2

    def test_source_without_reference_handled(self):
        """Source with display but no reference should not crash."""
        resource = _questionnaire_response()
        resource["source"] = {"display": "Dr. Smith"}
        doc, report = from_fhir(resource)
        assert report.success

    def test_malformed_source_reference_handled(self):
        """Source with reference lacking '/' should not crash."""
        resource = _questionnaire_response()
        resource["source"] = {"reference": "malformed-ref"}
        doc, report = from_fhir(resource)
        assert report.success
