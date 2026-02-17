"""Tests for temporal decay coverage across all 26 FHIR resource types.

RED PHASE: Verifies that fhir_temporal_decay() correctly extracts
timestamps and applies decay for Phase 6 resource types that use
non-standard timestamp fields (period.start, created, started,
authoredOn, manufactureDate).
"""

import pytest
from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import from_fhir, fhir_temporal_decay


REF_TIME = "2025-06-01T00:00:00Z"


# ═══════════════════════════════════════════════════════════════════
# Existing types that already work (sanity checks)
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayExistingTypes:
    """Sanity: types that already propagate timestamps still work."""

    def test_observation_effective_datetime(self):
        res = {
            "resourceType": "Observation",
            "id": "obs-1",
            "status": "final",
            "effectiveDateTime": "2024-06-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        # 1 year elapsed → significant decay
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b
        assert report.nodes_converted >= 1

    def test_immunization_occurrence_datetime(self):
        res = {
            "resourceType": "Immunization",
            "id": "imm-1",
            "status": "completed",
            "vaccineCode": {"text": "Flu"},
            "occurrenceDateTime": "2024-06-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b


# ═══════════════════════════════════════════════════════════════════
# Phase 6 types with period.start
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayPeriodStart:
    """Encounter, CarePlan, CareTeam use period.start."""

    def test_encounter_period_start(self):
        res = {
            "resourceType": "Encounter",
            "id": "enc-1",
            "status": "finished",
            "period": {"start": "2024-01-15T10:00:00Z"},
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b
        assert report.nodes_converted >= 1

    def test_care_plan_period_start(self):
        res = {
            "resourceType": "CarePlan",
            "id": "cp-1",
            "status": "active",
            "intent": "plan",
            "period": {"start": "2023-06-01T00:00:00Z"},
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b

    def test_care_team_period_start(self):
        res = {
            "resourceType": "CareTeam",
            "id": "ct-1",
            "status": "active",
            "period": {"start": "2024-03-01T00:00:00Z"},
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b


# ═══════════════════════════════════════════════════════════════════
# Phase 6 types with 'created'
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayCreated:
    """Claim and ExplanationOfBenefit use 'created'."""

    def test_claim_created(self):
        res = {
            "resourceType": "Claim",
            "id": "claim-1",
            "status": "active",
            "created": "2024-09-15T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b

    def test_eob_created(self):
        res = {
            "resourceType": "ExplanationOfBenefit",
            "id": "eob-1",
            "status": "active",
            "created": "2024-09-15T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b


# ═══════════════════════════════════════════════════════════════════
# Phase 6 types with other timestamp fields
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayOtherFields:
    """ImagingStudy.started, MedicationRequest.authoredOn, Device.manufactureDate."""

    def test_imaging_study_started(self):
        res = {
            "resourceType": "ImagingStudy",
            "id": "img-1",
            "status": "available",
            "started": "2024-06-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b

    def test_medication_request_authored_on(self):
        res = {
            "resourceType": "MedicationRequest",
            "id": "rx-1",
            "status": "active",
            "intent": "order",
            "authoredOn": "2024-11-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b

    def test_device_manufacture_date(self):
        res = {
            "resourceType": "Device",
            "id": "dev-1",
            "status": "active",
            "manufactureDate": "2020-01-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b

    def test_medication_administration_effective_datetime(self):
        res = {
            "resourceType": "MedicationAdministration",
            "id": "ma-1",
            "status": "completed",
            "effectiveDateTime": "2024-08-01T00:00:00Z",
        }
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b


# ═══════════════════════════════════════════════════════════════════
# Edge: types with no natural timestamp
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayNoTimestamp:
    """Patient, Organization, Practitioner have no natural timestamp."""

    def test_patient_no_timestamp_warning(self):
        res = {"resourceType": "Patient", "id": "p-1", "name": [{"family": "Smith"}]}
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        assert len(report.warnings) > 0
        assert "No parseable timestamp" in report.warnings[0]
        # Opinions unchanged
        orig_u = doc["opinions"][0]["opinion"].uncertainty
        dec_u = decayed["opinions"][0]["opinion"].uncertainty
        assert abs(orig_u - dec_u) < 1e-9

    def test_organization_no_timestamp_warning(self):
        res = {"resourceType": "Organization", "id": "org-1", "active": True}
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        assert len(report.warnings) > 0

    def test_practitioner_no_timestamp_warning(self):
        res = {"resourceType": "Practitioner", "id": "pract-1", "active": True}
        doc, _ = from_fhir(res)
        decayed, report = fhir_temporal_decay(doc, reference_time=REF_TIME)
        assert len(report.warnings) > 0


# ═══════════════════════════════════════════════════════════════════
# Comprehensive: all 26 types with timestamps produce decay
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDecayComprehensive:
    """Every type that CAN have a timestamp gets decayed."""

    RESOURCE_WITH_TIMESTAMPS = [
        # (resource_dict, expected_timestamp_field)
        ({"resourceType": "Observation", "id": "t1", "status": "final",
          "effectiveDateTime": "2024-01-01T00:00:00Z"}, "effectiveDateTime"),
        ({"resourceType": "Immunization", "id": "t2", "status": "completed",
          "vaccineCode": {"text": "X"}, "occurrenceDateTime": "2024-01-01T00:00:00Z"}, "occurrenceDateTime"),
        ({"resourceType": "Condition", "id": "t3", "verificationStatus": {
          "coding": [{"code": "confirmed"}]}, "onsetDateTime": "2024-01-01T00:00:00Z"}, "onsetDateTime"),
        ({"resourceType": "Encounter", "id": "t4", "status": "finished",
          "period": {"start": "2024-01-01T00:00:00Z"}}, "period.start"),
        ({"resourceType": "CarePlan", "id": "t5", "status": "active", "intent": "plan",
          "period": {"start": "2024-01-01T00:00:00Z"}}, "period.start"),
        ({"resourceType": "CareTeam", "id": "t6", "status": "active",
          "period": {"start": "2024-01-01T00:00:00Z"}}, "period.start"),
        ({"resourceType": "MedicationRequest", "id": "t7", "status": "active",
          "intent": "order", "authoredOn": "2024-01-01T00:00:00Z"}, "authoredOn"),
        ({"resourceType": "MedicationAdministration", "id": "t8", "status": "completed",
          "effectiveDateTime": "2024-01-01T00:00:00Z"}, "effectiveDateTime"),
        ({"resourceType": "ImagingStudy", "id": "t9", "status": "available",
          "started": "2024-01-01T00:00:00Z"}, "started"),
        ({"resourceType": "Device", "id": "t10", "status": "active",
          "manufactureDate": "2024-01-01T00:00:00Z"}, "manufactureDate"),
        ({"resourceType": "Claim", "id": "t11", "status": "active",
          "created": "2024-01-01T00:00:00Z"}, "created"),
        ({"resourceType": "ExplanationOfBenefit", "id": "t12", "status": "active",
          "created": "2024-01-01T00:00:00Z"}, "created"),
    ]

    @pytest.mark.parametrize("resource,ts_field", RESOURCE_WITH_TIMESTAMPS,
                             ids=[r[0]["resourceType"] for r in RESOURCE_WITH_TIMESTAMPS])
    def test_decayed_with_timestamp(self, resource, ts_field):
        """Resource with timestamp → decay applied, belief decreases."""
        doc, _ = from_fhir(resource)
        decayed, report = fhir_temporal_decay(
            doc, reference_time=REF_TIME, half_life_days=365.0,
        )
        orig_b = doc["opinions"][0]["opinion"].belief
        dec_b = decayed["opinions"][0]["opinion"].belief
        assert dec_b < orig_b, (
            f"{resource['resourceType']}.{ts_field}: belief not decayed "
            f"(orig={orig_b:.4f}, decayed={dec_b:.4f})"
        )
        assert report.nodes_converted >= 1
