"""Tests for Phase 1–4 manual handler round-trip data preservation.

RED PHASE: These tests verify that timestamp and metadata fields
stored by from_fhir() in custom (non-factory) handlers survive the
to_fhir() export.

Affected types and fields:
  Observation            → effectiveDateTime
  DiagnosticReport       → effectiveDateTime
  Condition              → onsetDateTime, recordedDate
  ExplanationOfBenefit   → created
  Claim                  → created  (factory-based, passthrough missing)

These fields are critical for:
  - Temporal decay pipelines (fhir_temporal_decay relies on timestamps)
  - Provenance tracking (recordedDate documents when data entered system)
  - Clinical timeline reconstruction (onsetDateTime, effectiveDateTime)
  - Financial audit trails (created)
"""

import pytest
from jsonld_ex.fhir_interop import from_fhir, to_fhir


# ═══════════════════════════════════════════════════════════════════
# Observation — effectiveDateTime
# ═══════════════════════════════════════════════════════════════════


class TestObservationEffectiveDateTimeRoundTrip:
    """Observation.effectiveDateTime must survive from_fhir → to_fhir."""

    def _observation(self, *, effective_dt=None, status="final", code="H"):
        resource = {
            "resourceType": "Observation",
            "id": "obs-rt-1",
            "status": status,
            "interpretation": [{"coding": [{"code": code}]}],
        }
        if effective_dt is not None:
            resource["effectiveDateTime"] = effective_dt
        return resource

    def test_effective_present_in_doc(self):
        """from_fhir copies effectiveDateTime into jsonld-ex document."""
        resource = self._observation(effective_dt="2025-03-15T10:30:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("effectiveDateTime") == "2025-03-15T10:30:00Z"

    def test_effective_survives_round_trip(self):
        """effectiveDateTime written back to FHIR resource by to_fhir."""
        resource_in = self._observation(effective_dt="2025-03-15T10:30:00Z")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out.get("effectiveDateTime") == "2025-03-15T10:30:00Z"

    def test_effective_date_only(self):
        """Date-only effectiveDateTime survives round-trip."""
        resource_in = self._observation(effective_dt="2025-03-15")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("effectiveDateTime") == "2025-03-15"

    def test_missing_effective_no_error(self):
        """Observation without effectiveDateTime round-trips without error."""
        resource_in = self._observation()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "effectiveDateTime" not in resource_out

    def test_effective_with_status_fallback(self):
        """effectiveDateTime preserved even when opinion comes from status fallback."""
        resource_in = {
            "resourceType": "Observation",
            "id": "obs-status-fb",
            "status": "final",
            "effectiveDateTime": "2025-06-01T08:00:00Z",
            # No interpretation → triggers status fallback
        }
        doc, _ = from_fhir(resource_in)
        assert doc.get("effectiveDateTime") == "2025-06-01T08:00:00Z"
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("effectiveDateTime") == "2025-06-01T08:00:00Z"


# ═══════════════════════════════════════════════════════════════════
# DiagnosticReport — effectiveDateTime
# ═══════════════════════════════════════════════════════════════════


class TestDiagnosticReportEffectiveDateTimeRoundTrip:
    """DiagnosticReport.effectiveDateTime must survive from_fhir → to_fhir."""

    def _diagnostic_report(self, *, effective_dt=None, status="final",
                           conclusion=None):
        resource = {
            "resourceType": "DiagnosticReport",
            "id": "dr-rt-1",
            "status": status,
        }
        if conclusion is not None:
            resource["conclusion"] = conclusion
        if effective_dt is not None:
            resource["effectiveDateTime"] = effective_dt
        return resource

    def test_effective_present_in_doc(self):
        """from_fhir copies effectiveDateTime into jsonld-ex document."""
        resource = self._diagnostic_report(effective_dt="2025-04-20T14:00:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("effectiveDateTime") == "2025-04-20T14:00:00Z"

    def test_effective_survives_round_trip(self):
        """effectiveDateTime written back to FHIR resource by to_fhir."""
        resource_in = self._diagnostic_report(
            effective_dt="2025-04-20T14:00:00Z",
            conclusion="Normal findings",
        )
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out.get("effectiveDateTime") == "2025-04-20T14:00:00Z"

    def test_effective_with_status_fallback(self):
        """effectiveDateTime preserved when opinion from status fallback."""
        resource_in = self._diagnostic_report(
            effective_dt="2025-04-20T14:00:00Z",
            # No conclusion → status fallback
        )
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("effectiveDateTime") == "2025-04-20T14:00:00Z"

    def test_missing_effective_no_error(self):
        """DiagnosticReport without effectiveDateTime round-trips OK."""
        resource_in = self._diagnostic_report(conclusion="Normal")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "effectiveDateTime" not in resource_out


# ═══════════════════════════════════════════════════════════════════
# Condition — onsetDateTime and recordedDate
# ═══════════════════════════════════════════════════════════════════


class TestConditionTemporalFieldsRoundTrip:
    """Condition.onsetDateTime and .recordedDate must survive round-trip."""

    def _condition(self, *, vs_code="confirmed", onset_dt=None,
                   recorded_date=None):
        resource = {
            "resourceType": "Condition",
            "id": "cond-rt-1",
            "verificationStatus": {"coding": [{"code": vs_code}]},
        }
        if onset_dt is not None:
            resource["onsetDateTime"] = onset_dt
        if recorded_date is not None:
            resource["recordedDate"] = recorded_date
        return resource

    # ── onsetDateTime ──

    def test_onset_present_in_doc(self):
        """from_fhir copies onsetDateTime into jsonld-ex document."""
        resource = self._condition(onset_dt="2024-12-01T00:00:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("onsetDateTime") == "2024-12-01T00:00:00Z"

    def test_onset_survives_round_trip(self):
        """onsetDateTime written back to FHIR resource by to_fhir."""
        resource_in = self._condition(onset_dt="2024-12-01T00:00:00Z")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out.get("onsetDateTime") == "2024-12-01T00:00:00Z"

    def test_onset_date_only(self):
        """Date-only onsetDateTime survives round-trip."""
        resource_in = self._condition(onset_dt="2024-12-01")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("onsetDateTime") == "2024-12-01"

    def test_missing_onset_no_error(self):
        """Condition without onsetDateTime round-trips without error."""
        resource_in = self._condition()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "onsetDateTime" not in resource_out

    # ── recordedDate ──

    def test_recorded_date_present_in_doc(self):
        """from_fhir copies recordedDate into jsonld-ex document."""
        resource = self._condition(recorded_date="2025-01-10")
        doc, _ = from_fhir(resource)
        assert doc.get("recordedDate") == "2025-01-10"

    def test_recorded_date_survives_round_trip(self):
        """recordedDate written back to FHIR resource by to_fhir."""
        resource_in = self._condition(recorded_date="2025-01-10")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("recordedDate") == "2025-01-10"

    def test_missing_recorded_date_no_error(self):
        """Condition without recordedDate round-trips without error."""
        resource_in = self._condition()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "recordedDate" not in resource_out

    # ── Both together ──

    def test_both_temporal_fields_survive_round_trip(self):
        """Both onsetDateTime and recordedDate survive together."""
        resource_in = self._condition(
            onset_dt="2024-12-01T00:00:00Z",
            recorded_date="2025-01-10",
        )
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("onsetDateTime") == "2024-12-01T00:00:00Z"
        assert resource_out.get("recordedDate") == "2025-01-10"


# ═══════════════════════════════════════════════════════════════════
# ExplanationOfBenefit — created
# ═══════════════════════════════════════════════════════════════════


class TestEobCreatedRoundTrip:
    """ExplanationOfBenefit.created must survive from_fhir → to_fhir."""

    def _eob(self, *, status="active", outcome="complete", created=None):
        resource = {
            "resourceType": "ExplanationOfBenefit",
            "id": "eob-rt-1",
            "status": status,
            "outcome": outcome,
        }
        if created is not None:
            resource["created"] = created
        return resource

    def test_created_present_in_doc(self):
        """from_fhir copies created into jsonld-ex document."""
        resource = self._eob(created="2025-02-01T12:00:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("created") == "2025-02-01T12:00:00Z"

    def test_created_survives_round_trip(self):
        """created written back to FHIR resource by to_fhir."""
        resource_in = self._eob(created="2025-02-01T12:00:00Z")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out.get("created") == "2025-02-01T12:00:00Z"

    def test_created_date_only(self):
        """Date-only created survives round-trip."""
        resource_in = self._eob(created="2025-02-01")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("created") == "2025-02-01"

    def test_missing_created_no_error(self):
        """EoB without created round-trips without error."""
        resource_in = self._eob()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "created" not in resource_out

    def test_created_coexists_with_outcome(self):
        """Both created and outcome survive round-trip."""
        resource_in = self._eob(
            created="2025-02-01T12:00:00Z",
            outcome="complete",
        )
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("created") == "2025-02-01T12:00:00Z"
        assert resource_out.get("outcome") == "complete"


# ═══════════════════════════════════════════════════════════════════
# Claim — created (factory-based)
# ═══════════════════════════════════════════════════════════════════


class TestClaimCreatedRoundTrip:
    """Claim.created must survive from_fhir → to_fhir."""

    def _claim(self, *, status="active", created=None):
        resource = {
            "resourceType": "Claim",
            "id": "claim-rt-1",
            "status": status,
        }
        if created is not None:
            resource["created"] = created
        return resource

    def test_created_present_in_doc(self):
        """from_fhir copies created into jsonld-ex document."""
        resource = self._claim(created="2025-05-10T09:00:00Z")
        doc, _ = from_fhir(resource)
        assert doc.get("created") == "2025-05-10T09:00:00Z"

    def test_created_survives_round_trip(self):
        """created written back to FHIR resource by to_fhir."""
        resource_in = self._claim(created="2025-05-10T09:00:00Z")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out.get("created") == "2025-05-10T09:00:00Z"

    def test_created_date_only(self):
        """Date-only created survives round-trip."""
        resource_in = self._claim(created="2025-05-10")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert resource_out.get("created") == "2025-05-10"

    def test_missing_created_no_error(self):
        """Claim without created round-trips without error."""
        resource_in = self._claim()
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)
        assert "created" not in resource_out


# ═══════════════════════════════════════════════════════════════════
# Cross-cutting parametrized check — all 6 fields
# ═══════════════════════════════════════════════════════════════════


def _obs_with_effective():
    return {
        "resourceType": "Observation", "id": "obs-x",
        "status": "final",
        "interpretation": [{"coding": [{"code": "N"}]}],
        "effectiveDateTime": "2025-01-01T00:00:00Z",
    }


def _dr_with_effective():
    return {
        "resourceType": "DiagnosticReport", "id": "dr-x",
        "status": "final",
        "conclusion": "Normal",
        "effectiveDateTime": "2025-02-01T00:00:00Z",
    }


def _cond_with_onset():
    return {
        "resourceType": "Condition", "id": "cond-x1",
        "verificationStatus": {"coding": [{"code": "confirmed"}]},
        "onsetDateTime": "2025-03-01T00:00:00Z",
    }


def _cond_with_recorded():
    return {
        "resourceType": "Condition", "id": "cond-x2",
        "verificationStatus": {"coding": [{"code": "confirmed"}]},
        "recordedDate": "2025-03-15",
    }


def _eob_with_created():
    return {
        "resourceType": "ExplanationOfBenefit", "id": "eob-x",
        "status": "active", "outcome": "complete",
        "created": "2025-04-01T00:00:00Z",
    }


def _claim_with_created():
    return {
        "resourceType": "Claim", "id": "claim-x",
        "status": "active",
        "created": "2025-05-01T00:00:00Z",
    }


_MANUAL_HANDLER_PASSTHROUGH_CASES = [
    ("Observation", _obs_with_effective, "effectiveDateTime"),
    ("DiagnosticReport", _dr_with_effective, "effectiveDateTime"),
    ("Condition (onset)", _cond_with_onset, "onsetDateTime"),
    ("Condition (recorded)", _cond_with_recorded, "recordedDate"),
    ("ExplanationOfBenefit", _eob_with_created, "created"),
    ("Claim", _claim_with_created, "created"),
]


class TestAllManualHandlerPassthroughRoundTrip:
    """Parametrized: every from_fhir passthrough field survives to_fhir."""

    @pytest.mark.parametrize(
        "label, fixture_fn, field_name",
        _MANUAL_HANDLER_PASSTHROUGH_CASES,
        ids=[c[0] for c in _MANUAL_HANDLER_PASSTHROUGH_CASES],
    )
    def test_field_present_in_doc(self, label, fixture_fn, field_name):
        """Field stored by from_fhir in the intermediate document."""
        resource_in = fixture_fn()
        doc, _ = from_fhir(resource_in)
        assert field_name in doc, (
            f"{label}: '{field_name}' missing from jsonld-ex doc"
        )

    @pytest.mark.parametrize(
        "label, fixture_fn, field_name",
        _MANUAL_HANDLER_PASSTHROUGH_CASES,
        ids=[c[0] for c in _MANUAL_HANDLER_PASSTHROUGH_CASES],
    )
    def test_field_survives_round_trip(self, label, fixture_fn, field_name):
        """Field present in FHIR output after round-trip."""
        resource_in = fixture_fn()
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert field_name in resource_out, (
            f"{label}: '{field_name}' lost during to_fhir() round-trip"
        )
        assert resource_out[field_name] == resource_in[field_name], (
            f"{label}: '{field_name}' value changed during round-trip"
        )
