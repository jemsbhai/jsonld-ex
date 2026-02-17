"""Tests for passthrough-field preservation in from_fhir → to_fhir round-trips.

RED PHASE: These tests verify that timestamp and metadata fields copied
into the jsonld-ex document during from_fhir() survive the to_fhir()
export.  Without this, temporal-decay pipelines lose the timestamps they
depend on after a round-trip through FHIR.

Affected types (all use _make_status_handler with passthrough_fields):
  Encounter            → period
  MedicationRequest    → authoredOn
  CarePlan             → period
  CareTeam             → period
  ImagingStudy         → started
  MedicationAdministration → effectiveDateTime
  Device               → manufactureDate
"""

import pytest
from jsonld_ex.fhir_interop import from_fhir, to_fhir


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers — include timestamp fields
# ═══════════════════════════════════════════════════════════════════


def _encounter_with_period():
    return {
        "resourceType": "Encounter",
        "id": "enc-rt-1",
        "status": "finished",
        "period": {
            "start": "2025-01-15T08:30:00Z",
            "end": "2025-01-15T09:45:00Z",
        },
    }


def _medication_request_with_authored_on():
    return {
        "resourceType": "MedicationRequest",
        "id": "medrx-rt-1",
        "status": "active",
        "intent": "order",
        "authoredOn": "2025-02-10T14:00:00Z",
    }


def _care_plan_with_period():
    return {
        "resourceType": "CarePlan",
        "id": "cp-rt-1",
        "status": "active",
        "intent": "plan",
        "period": {
            "start": "2025-03-01T00:00:00Z",
            "end": "2025-06-01T00:00:00Z",
        },
    }


def _care_team_with_period():
    return {
        "resourceType": "CareTeam",
        "id": "ct-rt-1",
        "status": "active",
        "period": {
            "start": "2025-04-01T00:00:00Z",
        },
    }


def _imaging_study_with_started():
    return {
        "resourceType": "ImagingStudy",
        "id": "img-rt-1",
        "status": "available",
        "started": "2025-05-20T10:15:00Z",
    }


def _medication_administration_with_effective():
    return {
        "resourceType": "MedicationAdministration",
        "id": "ma-rt-1",
        "status": "completed",
        "effectiveDateTime": "2025-06-12T07:30:00Z",
    }


def _device_with_manufacture_date():
    return {
        "resourceType": "Device",
        "id": "dev-rt-1",
        "status": "active",
        "manufactureDate": "2024-11-01",
    }


# ═══════════════════════════════════════════════════════════════════
# Round-trip passthrough preservation tests
# ═══════════════════════════════════════════════════════════════════


class TestEncounterPassthroughRoundTrip:
    """Encounter.period must survive from_fhir → to_fhir."""

    def test_period_present_in_doc(self):
        """from_fhir copies period into the jsonld-ex document."""
        resource = _encounter_with_period()
        doc, _ = from_fhir(resource)
        assert "period" in doc
        assert doc["period"]["start"] == "2025-01-15T08:30:00Z"
        assert doc["period"]["end"] == "2025-01-15T09:45:00Z"

    def test_period_survives_round_trip(self):
        """period is written back to the FHIR resource by to_fhir."""
        resource_in = _encounter_with_period()
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert "period" in resource_out
        assert resource_out["period"]["start"] == "2025-01-15T08:30:00Z"
        assert resource_out["period"]["end"] == "2025-01-15T09:45:00Z"

    def test_missing_period_no_error(self):
        """Encounter without period round-trips without error."""
        resource_in = {"resourceType": "Encounter", "id": "enc-no-period", "status": "finished"}
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)
        assert report.success is True
        assert "period" not in resource_out


class TestMedicationRequestPassthroughRoundTrip:
    """MedicationRequest.authoredOn must survive from_fhir → to_fhir."""

    def test_authored_on_present_in_doc(self):
        """from_fhir copies authoredOn into the jsonld-ex document."""
        resource = _medication_request_with_authored_on()
        doc, _ = from_fhir(resource)
        assert doc["authoredOn"] == "2025-02-10T14:00:00Z"

    def test_authored_on_survives_round_trip(self):
        """authoredOn is written back to the FHIR resource by to_fhir."""
        resource_in = _medication_request_with_authored_on()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("authoredOn") == "2025-02-10T14:00:00Z"

    def test_missing_authored_on_no_error(self):
        """MedicationRequest without authoredOn round-trips without error."""
        resource_in = {"resourceType": "MedicationRequest", "id": "rx-no-date", "status": "active"}
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "authoredOn" not in resource_out


class TestCarePlanPassthroughRoundTrip:
    """CarePlan.period must survive from_fhir → to_fhir."""

    def test_period_present_in_doc(self):
        """from_fhir copies period into the jsonld-ex document."""
        resource = _care_plan_with_period()
        doc, _ = from_fhir(resource)
        assert doc["period"]["start"] == "2025-03-01T00:00:00Z"

    def test_period_survives_round_trip(self):
        """period is written back to the FHIR resource by to_fhir."""
        resource_in = _care_plan_with_period()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert "period" in resource_out
        assert resource_out["period"]["start"] == "2025-03-01T00:00:00Z"
        assert resource_out["period"]["end"] == "2025-06-01T00:00:00Z"


class TestCareTeamPassthroughRoundTrip:
    """CareTeam.period must survive from_fhir → to_fhir."""

    def test_period_present_in_doc(self):
        """from_fhir copies period into the jsonld-ex document."""
        resource = _care_team_with_period()
        doc, _ = from_fhir(resource)
        assert doc["period"]["start"] == "2025-04-01T00:00:00Z"

    def test_period_survives_round_trip(self):
        """period is written back to the FHIR resource by to_fhir."""
        resource_in = _care_team_with_period()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert "period" in resource_out
        assert resource_out["period"]["start"] == "2025-04-01T00:00:00Z"


class TestImagingStudyPassthroughRoundTrip:
    """ImagingStudy.started must survive from_fhir → to_fhir."""

    def test_started_present_in_doc(self):
        """from_fhir copies started into the jsonld-ex document."""
        resource = _imaging_study_with_started()
        doc, _ = from_fhir(resource)
        assert doc["started"] == "2025-05-20T10:15:00Z"

    def test_started_survives_round_trip(self):
        """started is written back to the FHIR resource by to_fhir."""
        resource_in = _imaging_study_with_started()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("started") == "2025-05-20T10:15:00Z"

    def test_missing_started_no_error(self):
        """ImagingStudy without started round-trips without error."""
        resource_in = {"resourceType": "ImagingStudy", "id": "img-no-start", "status": "available"}
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "started" not in resource_out


class TestMedicationAdministrationPassthroughRoundTrip:
    """MedicationAdministration.effectiveDateTime must survive round-trip."""

    def test_effective_present_in_doc(self):
        """from_fhir copies effectiveDateTime into the jsonld-ex document."""
        resource = _medication_administration_with_effective()
        doc, _ = from_fhir(resource)
        assert doc["effectiveDateTime"] == "2025-06-12T07:30:00Z"

    def test_effective_survives_round_trip(self):
        """effectiveDateTime is written back to the FHIR resource by to_fhir."""
        resource_in = _medication_administration_with_effective()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("effectiveDateTime") == "2025-06-12T07:30:00Z"

    def test_missing_effective_no_error(self):
        """MedicationAdministration without effectiveDateTime round-trips OK."""
        resource_in = {"resourceType": "MedicationAdministration", "id": "ma-no-dt", "status": "completed"}
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "effectiveDateTime" not in resource_out


class TestDevicePassthroughRoundTrip:
    """Device.manufactureDate must survive from_fhir → to_fhir."""

    def test_manufacture_date_present_in_doc(self):
        """from_fhir copies manufactureDate into the jsonld-ex document."""
        resource = _device_with_manufacture_date()
        doc, _ = from_fhir(resource)
        assert doc["manufactureDate"] == "2024-11-01"

    def test_manufacture_date_survives_round_trip(self):
        """manufactureDate is written back to the FHIR resource by to_fhir."""
        resource_in = _device_with_manufacture_date()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("manufactureDate") == "2024-11-01"

    def test_missing_manufacture_date_no_error(self):
        """Device without manufactureDate round-trips without error."""
        resource_in = {"resourceType": "Device", "id": "dev-no-date", "status": "active"}
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "manufactureDate" not in resource_out


# ═══════════════════════════════════════════════════════════════════
# Cross-cutting: all 7 types in a single parametrized check
# ═══════════════════════════════════════════════════════════════════


_PASSTHROUGH_CASES = [
    ("Encounter", _encounter_with_period, "period"),
    ("MedicationRequest", _medication_request_with_authored_on, "authoredOn"),
    ("CarePlan", _care_plan_with_period, "period"),
    ("CareTeam", _care_team_with_period, "period"),
    ("ImagingStudy", _imaging_study_with_started, "started"),
    ("MedicationAdministration", _medication_administration_with_effective, "effectiveDateTime"),
    ("Device", _device_with_manufacture_date, "manufactureDate"),
]


class TestAllPassthroughFieldsRoundTrip:
    """Parametrized: every passthrough field survives the full round-trip."""

    @pytest.mark.parametrize(
        "resource_type, fixture_fn, field_name",
        _PASSTHROUGH_CASES,
        ids=[c[0] for c in _PASSTHROUGH_CASES],
    )
    def test_field_survives_round_trip(self, resource_type, fixture_fn, field_name):
        """Passthrough field is present in FHIR output after round-trip."""
        resource_in = fixture_fn()
        doc, _ = from_fhir(resource_in)

        # Field must be in the intermediate jsonld-ex doc
        assert field_name in doc, (
            f"{resource_type}: '{field_name}' missing from jsonld-ex doc"
        )

        resource_out, report = to_fhir(doc)
        assert report.success is True

        # Field must survive into the output FHIR resource
        assert field_name in resource_out, (
            f"{resource_type}: '{field_name}' lost during to_fhir() round-trip"
        )

        # Value must match exactly
        assert resource_out[field_name] == resource_in[field_name], (
            f"{resource_type}: '{field_name}' value changed during round-trip"
        )
