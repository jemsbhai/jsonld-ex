"""
Tests for AllergyIntolerance converter completeness.

TDD Red Phase: Extends the converter to preserve ALL clinically relevant
fields from FHIR R4 AllergyIntolerance, closing the data gap that blocked
fhir_allergy_reconcile() implementation.

Fields being added to from_fhir() output:
  - code         (substance: text + coding system/code/display)
  - patient      (reference string)
  - category     (list of category codes)
  - type         (allergy | intolerance)
  - recordedDate (ISO datetime string)
  - onsetDateTime (ISO datetime string, simplified from onset[x])
  - asserter     (reference string)
  - lastOccurrence (ISO datetime string)
  - reaction     (list of {substance, manifestation[], severity})
  - note         (list of annotation text strings)

Regression: existing opinion semantics MUST remain unchanged.
"""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    opinion_to_fhir_extension,
)


# ---------------------------------------------------------------------------
# Enhanced fixture builder
# ---------------------------------------------------------------------------

def _allergy_full(
    *,
    clinical_status="active",
    verification_status="confirmed",
    criticality=None,
    code_text="Penicillin",
    code_coding=None,
    category=None,
    allergy_type=None,
    patient="Patient/pat-1",
    recorded_date=None,
    onset_date_time=None,
    asserter=None,
    last_occurrence=None,
    reaction=None,
    note=None,
    extensions=None,
) -> dict:
    """Build a comprehensive FHIR R4 AllergyIntolerance resource.

    Extends the minimal fixture to exercise every field the converter
    should preserve.
    """
    resource: dict = {
        "resourceType": "AllergyIntolerance",
        "id": "allergy-complete-1",
        "clinicalStatus": {
            "coding": [{"code": clinical_status}],
        },
        "verificationStatus": {
            "coding": [{"code": verification_status}],
        },
        "patient": {"reference": patient},
    }

    # code (substance)
    code_obj: dict = {}
    if code_text is not None:
        code_obj["text"] = code_text
    if code_coding is not None:
        code_obj["coding"] = code_coding
    if code_obj:
        resource["code"] = code_obj

    if criticality is not None:
        resource["criticality"] = criticality
    if category is not None:
        resource["category"] = category if isinstance(category, list) else [category]
    if allergy_type is not None:
        resource["type"] = allergy_type
    if recorded_date is not None:
        resource["recordedDate"] = recorded_date
    if onset_date_time is not None:
        resource["onsetDateTime"] = onset_date_time
    if asserter is not None:
        resource["asserter"] = {"reference": asserter}
    if last_occurrence is not None:
        resource["lastOccurrence"] = last_occurrence
    if reaction is not None:
        resource["reaction"] = reaction
    if note is not None:
        resource["note"] = [{"text": t} for t in note] if isinstance(note, list) else [{"text": note}]
    if extensions is not None:
        resource["_verificationStatus"] = {"extension": extensions}

    return resource


# ===================================================================
# 1. CODE (SUBSTANCE) PRESERVATION
# ===================================================================


class TestCodePreservation:
    """The substance/allergen code is the IDENTITY of the allergy."""

    def test_code_text_preserved(self):
        """code.text must appear in the jsonld-ex doc."""
        resource = _allergy_full(code_text="Penicillin")
        doc, _ = from_fhir(resource)
        assert doc.get("code") is not None
        assert doc["code"]["text"] == "Penicillin"

    def test_code_coding_preserved(self):
        """code.coding with system/code/display must be preserved."""
        coding = [
            {
                "system": "http://snomed.info/sct",
                "code": "91936005",
                "display": "Allergy to penicillin",
            }
        ]
        resource = _allergy_full(code_text="Penicillin", code_coding=coding)
        doc, _ = from_fhir(resource)

        assert doc["code"]["text"] == "Penicillin"
        assert len(doc["code"]["coding"]) == 1
        assert doc["code"]["coding"][0]["system"] == "http://snomed.info/sct"
        assert doc["code"]["coding"][0]["code"] == "91936005"
        assert doc["code"]["coding"][0]["display"] == "Allergy to penicillin"

    def test_code_text_only_no_coding(self):
        """code with only text (no coding) is still preserved."""
        resource = _allergy_full(code_text="Peanuts", code_coding=None)
        doc, _ = from_fhir(resource)
        assert doc["code"]["text"] == "Peanuts"
        assert "coding" not in doc["code"] or doc["code"].get("coding") is None

    def test_code_coding_only_no_text(self):
        """code with only coding (no text) preserves the coding."""
        coding = [{"system": "http://snomed.info/sct", "code": "91936005"}]
        resource = _allergy_full(code_text=None, code_coding=coding)
        doc, _ = from_fhir(resource)
        assert doc["code"]["coding"][0]["code"] == "91936005"

    def test_no_code_field(self):
        """Missing code field → code is None in doc."""
        resource = _allergy_full(code_text=None, code_coding=None)
        doc, _ = from_fhir(resource)
        assert doc.get("code") is None

    def test_multiple_codings_preserved(self):
        """Multiple coding entries (e.g., SNOMED + local) all preserved."""
        coding = [
            {"system": "http://snomed.info/sct", "code": "91936005", "display": "Allergy to penicillin"},
            {"system": "http://hospital.local/codes", "code": "PEN-ALLERGY", "display": "Penicillin Allergy"},
        ]
        resource = _allergy_full(code_text="Penicillin", code_coding=coding)
        doc, _ = from_fhir(resource)
        assert len(doc["code"]["coding"]) == 2


# ===================================================================
# 2. PATIENT REFERENCE
# ===================================================================


class TestPatientPreservation:
    """Patient reference is needed to verify same patient across systems."""

    def test_patient_reference_preserved(self):
        """patient.reference string is captured in the doc."""
        resource = _allergy_full(patient="Patient/12345")
        doc, _ = from_fhir(resource)
        assert doc.get("patient") == "Patient/12345"

    def test_no_patient_field(self):
        """Missing patient field → patient is None."""
        resource = _allergy_full()
        del resource["patient"]
        doc, _ = from_fhir(resource)
        assert doc.get("patient") is None


# ===================================================================
# 3. CATEGORY
# ===================================================================


class TestCategoryPreservation:
    """Category (food/medication/environment/biologic) for grouping."""

    def test_single_category_preserved(self):
        resource = _allergy_full(category=["food"])
        doc, _ = from_fhir(resource)
        assert doc.get("category") == ["food"]

    def test_multiple_categories_preserved(self):
        resource = _allergy_full(category=["food", "environment"])
        doc, _ = from_fhir(resource)
        assert doc.get("category") == ["food", "environment"]

    def test_no_category(self):
        resource = _allergy_full(category=None)
        doc, _ = from_fhir(resource)
        assert doc.get("category") is None


# ===================================================================
# 4. TYPE (allergy vs intolerance)
# ===================================================================


class TestTypePreservation:
    """Type distinguishes allergy from intolerance."""

    def test_type_allergy(self):
        resource = _allergy_full(allergy_type="allergy")
        doc, _ = from_fhir(resource)
        assert doc.get("type") == "allergy"

    def test_type_intolerance(self):
        resource = _allergy_full(allergy_type="intolerance")
        doc, _ = from_fhir(resource)
        assert doc.get("type") == "intolerance"

    def test_no_type(self):
        resource = _allergy_full(allergy_type=None)
        doc, _ = from_fhir(resource)
        assert doc.get("type") is None


# ===================================================================
# 5. TEMPORAL FIELDS
# ===================================================================


class TestTemporalFieldsPreservation:
    """Dates are essential for temporal reasoning and reconciliation."""

    def test_recorded_date_preserved(self):
        resource = _allergy_full(recorded_date="2024-03-15T10:00:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("recordedDate") == "2024-03-15T10:00:00Z"

    def test_onset_date_time_preserved(self):
        resource = _allergy_full(onset_date_time="2020-06-01")
        doc, _ = from_fhir(resource)
        assert doc.get("onsetDateTime") == "2020-06-01"

    def test_last_occurrence_preserved(self):
        resource = _allergy_full(last_occurrence="2024-01-10T08:30:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("lastOccurrence") == "2024-01-10T08:30:00Z"

    def test_no_temporal_fields(self):
        resource = _allergy_full()
        doc, _ = from_fhir(resource)
        assert doc.get("recordedDate") is None
        assert doc.get("onsetDateTime") is None
        assert doc.get("lastOccurrence") is None


# ===================================================================
# 6. ASSERTER
# ===================================================================


class TestAsserterPreservation:
    """Who asserted the allergy — useful for trust assessment."""

    def test_asserter_reference_preserved(self):
        resource = _allergy_full(asserter="Practitioner/dr-smith")
        doc, _ = from_fhir(resource)
        assert doc.get("asserter") == "Practitioner/dr-smith"

    def test_patient_as_asserter(self):
        """Patient self-report as asserter."""
        resource = _allergy_full(asserter="Patient/pat-1")
        doc, _ = from_fhir(resource)
        assert doc.get("asserter") == "Patient/pat-1"

    def test_no_asserter(self):
        resource = _allergy_full(asserter=None)
        doc, _ = from_fhir(resource)
        assert doc.get("asserter") is None


# ===================================================================
# 7. REACTION ARRAY
# ===================================================================


class TestReactionPreservation:
    """Reaction events are clinical evidence supporting the allergy."""

    def test_single_reaction_preserved(self):
        reactions = [
            {
                "substance": {
                    "coding": [{"system": "http://snomed.info/sct", "code": "764146007"}],
                    "text": "Penicillin",
                },
                "manifestation": [
                    {"coding": [{"code": "39579001"}], "text": "Anaphylaxis"},
                ],
                "severity": "severe",
            }
        ]
        resource = _allergy_full(reaction=reactions)
        doc, _ = from_fhir(resource)

        assert len(doc.get("reaction", [])) == 1
        rxn = doc["reaction"][0]
        assert rxn["substance"]["text"] == "Penicillin"
        assert rxn["severity"] == "severe"
        assert len(rxn["manifestation"]) == 1
        assert rxn["manifestation"][0]["text"] == "Anaphylaxis"

    def test_multiple_reactions_preserved(self):
        reactions = [
            {
                "manifestation": [{"text": "Hives"}],
                "severity": "mild",
            },
            {
                "manifestation": [{"text": "Anaphylaxis"}],
                "severity": "severe",
            },
        ]
        resource = _allergy_full(reaction=reactions)
        doc, _ = from_fhir(resource)
        assert len(doc["reaction"]) == 2

    def test_reaction_with_multiple_manifestations(self):
        reactions = [
            {
                "manifestation": [
                    {"text": "Hives"},
                    {"text": "Difficulty breathing"},
                    {"text": "Swelling"},
                ],
                "severity": "moderate",
            }
        ]
        resource = _allergy_full(reaction=reactions)
        doc, _ = from_fhir(resource)
        assert len(doc["reaction"][0]["manifestation"]) == 3

    def test_reaction_minimal_no_substance_no_severity(self):
        """Reaction with only manifestation (substance and severity optional)."""
        reactions = [
            {
                "manifestation": [{"text": "Rash"}],
            }
        ]
        resource = _allergy_full(reaction=reactions)
        doc, _ = from_fhir(resource)
        rxn = doc["reaction"][0]
        assert rxn.get("substance") is None
        assert rxn.get("severity") is None
        assert rxn["manifestation"][0]["text"] == "Rash"

    def test_no_reactions(self):
        resource = _allergy_full(reaction=None)
        doc, _ = from_fhir(resource)
        assert doc.get("reaction") is None

    def test_reaction_substance_coding_preserved(self):
        """Reaction substance coding is preserved (not just text)."""
        reactions = [
            {
                "substance": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "764146007",
                            "display": "Penicillin",
                        }
                    ],
                    "text": "Penicillin",
                },
                "manifestation": [{"text": "Rash"}],
            }
        ]
        resource = _allergy_full(reaction=reactions)
        doc, _ = from_fhir(resource)
        coding = doc["reaction"][0]["substance"]["coding"]
        assert coding[0]["system"] == "http://snomed.info/sct"
        assert coding[0]["code"] == "764146007"


# ===================================================================
# 8. NOTE
# ===================================================================


class TestNotePreservation:
    """Additional narrative text."""

    def test_single_note_preserved(self):
        resource = _allergy_full(note=["Patient reports severe reaction in 2019."])
        doc, _ = from_fhir(resource)
        assert doc.get("note") == ["Patient reports severe reaction in 2019."]

    def test_multiple_notes_preserved(self):
        resource = _allergy_full(note=["Note 1.", "Note 2."])
        doc, _ = from_fhir(resource)
        assert doc["note"] == ["Note 1.", "Note 2."]

    def test_no_notes(self):
        resource = _allergy_full(note=None)
        doc, _ = from_fhir(resource)
        assert doc.get("note") is None


# ===================================================================
# 9. ROUND-TRIP: from_fhir() → to_fhir()
# ===================================================================


class TestAllergyRoundTripCompleteness:
    """Verify all new fields survive the from_fhir → to_fhir round-trip."""

    def _make_full_resource(self) -> dict:
        """A maximally populated AllergyIntolerance resource."""
        return _allergy_full(
            clinical_status="active",
            verification_status="confirmed",
            criticality="high",
            code_text="Penicillin",
            code_coding=[
                {
                    "system": "http://snomed.info/sct",
                    "code": "91936005",
                    "display": "Allergy to penicillin",
                }
            ],
            category=["medication"],
            allergy_type="allergy",
            patient="Patient/pat-1",
            recorded_date="2024-03-15T10:00:00Z",
            onset_date_time="2020-06-01",
            asserter="Practitioner/dr-smith",
            last_occurrence="2024-01-10T08:30:00Z",
            reaction=[
                {
                    "substance": {"text": "Penicillin"},
                    "manifestation": [{"text": "Anaphylaxis"}],
                    "severity": "severe",
                }
            ],
            note=["Confirmed by allergy testing."],
        )

    def test_round_trip_code(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["code"]["text"] == "Penicillin"
        assert fhir_out["code"]["coding"][0]["code"] == "91936005"

    def test_round_trip_patient(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["patient"]["reference"] == "Patient/pat-1"

    def test_round_trip_category(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["category"] == ["medication"]

    def test_round_trip_type(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["type"] == "allergy"

    def test_round_trip_recorded_date(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["recordedDate"] == "2024-03-15T10:00:00Z"

    def test_round_trip_onset(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["onsetDateTime"] == "2020-06-01"

    def test_round_trip_asserter(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["asserter"]["reference"] == "Practitioner/dr-smith"

    def test_round_trip_last_occurrence(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["lastOccurrence"] == "2024-01-10T08:30:00Z"

    def test_round_trip_reaction(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert len(fhir_out["reaction"]) == 1
        assert fhir_out["reaction"][0]["severity"] == "severe"
        assert fhir_out["reaction"][0]["manifestation"][0]["text"] == "Anaphylaxis"

    def test_round_trip_note(self):
        resource = self._make_full_resource()
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["note"][0]["text"] == "Confirmed by allergy testing."


# ===================================================================
# 10. REGRESSION — EXISTING OPINION BEHAVIOR UNCHANGED
# ===================================================================


class TestAllergyRegressionOpinionUnchanged:
    """Adding fields must NOT alter existing opinion semantics."""

    def test_verification_status_opinion_unchanged(self):
        """Adding code/patient/etc must not change the verificationStatus opinion."""
        # Minimal resource (old behavior)
        minimal = {
            "resourceType": "AllergyIntolerance",
            "id": "allergy-regression",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed"}]},
            "code": {"text": "Penicillin"},
        }
        # Full resource (new fields)
        full = _allergy_full(
            verification_status="confirmed",
            code_text="Penicillin",
            category=["medication"],
            allergy_type="allergy",
            patient="Patient/pat-1",
            recorded_date="2024-03-15",
            reaction=[{"manifestation": [{"text": "Rash"}]}],
        )

        doc_min, _ = from_fhir(minimal)
        doc_full, _ = from_fhir(full)

        vs_min = [o for o in doc_min["opinions"] if o["field"] == "verificationStatus"][0]
        vs_full = [o for o in doc_full["opinions"] if o["field"] == "verificationStatus"][0]

        assert vs_min["opinion"].belief == pytest.approx(vs_full["opinion"].belief, abs=1e-12)
        assert vs_min["opinion"].disbelief == pytest.approx(vs_full["opinion"].disbelief, abs=1e-12)
        assert vs_min["opinion"].uncertainty == pytest.approx(vs_full["opinion"].uncertainty, abs=1e-12)

    def test_criticality_opinion_unchanged(self):
        """Criticality opinion is not affected by new fields."""
        minimal = {
            "resourceType": "AllergyIntolerance",
            "id": "allergy-regression",
            "clinicalStatus": {"coding": [{"code": "active"}]},
            "verificationStatus": {"coding": [{"code": "confirmed"}]},
            "code": {"text": "Penicillin"},
            "criticality": "high",
        }
        full = _allergy_full(
            verification_status="confirmed",
            criticality="high",
            category=["medication"],
            patient="Patient/pat-1",
        )

        doc_min, _ = from_fhir(minimal)
        doc_full, _ = from_fhir(full)

        crit_min = [o for o in doc_min["opinions"] if o["field"] == "criticality"][0]
        crit_full = [o for o in doc_full["opinions"] if o["field"] == "criticality"][0]

        assert crit_min["opinion"].belief == pytest.approx(crit_full["opinion"].belief, abs=1e-12)
        assert crit_min["opinion"].uncertainty == pytest.approx(crit_full["opinion"].uncertainty, abs=1e-12)

    def test_extension_recovery_unchanged(self):
        """Extension-based opinion recovery still takes precedence."""
        exact_op = Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20, base_rate=0.50)
        ext = opinion_to_fhir_extension(exact_op)
        resource = _allergy_full(
            extensions=[ext],
            code_text="Penicillin",
            category=["medication"],
            patient="Patient/pat-1",
        )
        doc, _ = from_fhir(resource)

        vs_op = [o for o in doc["opinions"] if o["field"] == "verificationStatus"][0]
        assert vs_op["source"] == "extension"
        assert vs_op["opinion"].belief == pytest.approx(0.55, abs=1e-9)
