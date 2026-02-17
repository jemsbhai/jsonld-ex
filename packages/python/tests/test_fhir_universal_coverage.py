"""Tests for Phase 6: Universal FHIR resource coverage.

RED PHASE: Tests for 12 new FHIR resource types to achieve 100%
coverage of the Synthea corpus.

Batch 1 — Clinical workflow:
  Encounter, MedicationRequest, CarePlan, Goal, CareTeam, ImagingStudy

Batch 2 — Administrative identity:
  Patient, Organization, Practitioner, Device

Batch 3 — Financial:
  Claim, ExplanationOfBenefit
"""

import pytest
from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    FHIR_EXTENSION_URL,
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
)


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════


def _encounter(*, status="finished"):
    return {
        "resourceType": "Encounter",
        "id": "encounter-1",
        "status": status,
        "class": {"code": "AMB"},
        "subject": {"reference": "Patient/p1"},
    }


def _medication_request(*, status="active", intent="order"):
    return {
        "resourceType": "MedicationRequest",
        "id": "medrx-1",
        "status": status,
        "intent": intent,
        "medicationCodeableConcept": {"text": "Atorvastatin 20mg"},
        "subject": {"reference": "Patient/p1"},
    }


def _care_plan(*, status="active", intent="plan"):
    return {
        "resourceType": "CarePlan",
        "id": "careplan-1",
        "status": status,
        "intent": intent,
        "subject": {"reference": "Patient/p1"},
    }


def _goal(*, lifecycle_status="active", achievement_status=None):
    resource = {
        "resourceType": "Goal",
        "id": "goal-1",
        "lifecycleStatus": lifecycle_status,
        "description": {"text": "Reduce HbA1c to <7%"},
        "subject": {"reference": "Patient/p1"},
    }
    if achievement_status is not None:
        resource["achievementStatus"] = {
            "coding": [{"code": achievement_status}],
        }
    return resource


def _care_team(*, status="active"):
    return {
        "resourceType": "CareTeam",
        "id": "careteam-1",
        "status": status,
        "subject": {"reference": "Patient/p1"},
    }


def _imaging_study(*, status="available"):
    return {
        "resourceType": "ImagingStudy",
        "id": "imaging-1",
        "status": status,
        "subject": {"reference": "Patient/p1"},
    }


def _patient(*, extras=None):
    resource = {
        "resourceType": "Patient",
        "id": "patient-1",
        "name": [{"family": "Smith", "given": ["John"]}],
        "gender": "male",
        "birthDate": "1980-01-01",
    }
    if extras:
        resource.update(extras)
    return resource


def _organization(*, active=True):
    return {
        "resourceType": "Organization",
        "id": "org-1",
        "active": active,
        "name": "General Hospital",
    }


def _practitioner(*, active=True):
    return {
        "resourceType": "Practitioner",
        "id": "pract-1",
        "active": active,
        "name": [{"family": "Jones", "given": ["Jane"]}],
    }


def _device(*, status="active"):
    return {
        "resourceType": "Device",
        "id": "device-1",
        "status": status,
        "type": {"text": "Cardiac pacemaker"},
        "patient": {"reference": "Patient/p1"},
    }


def _claim(*, status="active"):
    return {
        "resourceType": "Claim",
        "id": "claim-1",
        "status": status,
        "type": {"coding": [{"code": "professional"}]},
        "use": "claim",
        "patient": {"reference": "Patient/p1"},
        "provider": {"reference": "Organization/org-1"},
    }


def _eob(*, status="active", outcome=None):
    resource = {
        "resourceType": "ExplanationOfBenefit",
        "id": "eob-1",
        "status": status,
        "type": {"coding": [{"code": "professional"}]},
        "use": "claim",
        "patient": {"reference": "Patient/p1"},
        "insurer": {"reference": "Organization/ins-1"},
        "provider": {"reference": "Organization/org-1"},
    }
    if outcome is not None:
        resource["outcome"] = outcome
    return resource


# ═══════════════════════════════════════════════════════════════════
# Batch 1: Clinical workflow
# ═══════════════════════════════════════════════════════════════════


class TestFromFhirEncounter:
    """Encounter: 'this encounter occurred as documented.'"""

    def test_basic_finished(self):
        doc, report = from_fhir(_encounter(status="finished"))
        assert report.success is True
        assert doc["@type"] == "fhir:Encounter"
        assert len(doc["opinions"]) >= 1
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_finished_high_belief(self):
        doc, _ = from_fhir(_encounter(status="finished"))
        assert doc["opinions"][0]["opinion"].belief > 0.5

    def test_planned_lower_belief_than_finished(self):
        doc_fin, _ = from_fhir(_encounter(status="finished"))
        doc_plan, _ = from_fhir(_encounter(status="planned"))
        assert doc_fin["opinions"][0]["opinion"].belief > doc_plan["opinions"][0]["opinion"].belief

    def test_cancelled_low_belief(self):
        doc, _ = from_fhir(_encounter(status="cancelled"))
        assert doc["opinions"][0]["opinion"].belief < 0.3

    def test_entered_in_error_high_uncertainty(self):
        doc, _ = from_fhir(_encounter(status="entered-in-error"))
        assert doc["opinions"][0]["opinion"].uncertainty > 0.4

    def test_status_preserved(self):
        doc, _ = from_fhir(_encounter(status="in-progress"))
        assert doc["status"] == "in-progress"

    def test_id_preserved(self):
        doc, _ = from_fhir(_encounter())
        assert doc["id"] == "encounter-1"

    def test_extension_recovery(self):
        ext = opinion_to_fhir_extension(Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15))
        res = _encounter()
        res["_status"] = {"extension": [ext]}
        doc, _ = from_fhir(res)
        assert doc["opinions"][0]["source"] == "extension"
        assert abs(doc["opinions"][0]["opinion"].belief - 0.80) < 1e-9

    def test_additivity_all_statuses(self):
        for s in ("finished", "arrived", "triaged", "in-progress",
                   "onleave", "planned", "cancelled", "entered-in-error"):
            doc, _ = from_fhir(_encounter(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirMedicationRequest:
    """MedicationRequest: 'this prescription is valid and appropriate.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_medication_request(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:MedicationRequest"
        assert len(doc["opinions"]) >= 1

    def test_completed_higher_belief_than_stopped(self):
        doc_comp, _ = from_fhir(_medication_request(status="completed"))
        doc_stop, _ = from_fhir(_medication_request(status="stopped"))
        assert doc_comp["opinions"][0]["opinion"].belief > doc_stop["opinions"][0]["opinion"].belief

    def test_cancelled_low_belief(self):
        doc, _ = from_fhir(_medication_request(status="cancelled"))
        assert doc["opinions"][0]["opinion"].belief < 0.3

    def test_draft_high_uncertainty(self):
        doc, _ = from_fhir(_medication_request(status="draft"))
        assert doc["opinions"][0]["opinion"].uncertainty > 0.25

    def test_additivity(self):
        for s in ("active", "on-hold", "cancelled", "completed",
                   "entered-in-error", "stopped", "draft"):
            doc, _ = from_fhir(_medication_request(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirCarePlan:
    """CarePlan: 'this care plan is active and being followed.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_care_plan(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:CarePlan"
        assert len(doc["opinions"]) >= 1

    def test_completed_higher_belief_than_revoked(self):
        doc_comp, _ = from_fhir(_care_plan(status="completed"))
        doc_rev, _ = from_fhir(_care_plan(status="revoked"))
        assert doc_comp["opinions"][0]["opinion"].belief > doc_rev["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("active", "on-hold", "revoked", "completed",
                   "entered-in-error", "draft"):
            doc, _ = from_fhir(_care_plan(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirGoal:
    """Goal: uses lifecycleStatus, not status. Optional achievementStatus."""

    def test_basic_active(self):
        doc, report = from_fhir(_goal(lifecycle_status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:Goal"
        assert len(doc["opinions"]) >= 1

    def test_field_is_lifecycle_status(self):
        doc, _ = from_fhir(_goal(lifecycle_status="active"))
        assert doc["opinions"][0]["field"] == "lifecycleStatus"
        assert doc["opinions"][0]["value"] == "active"

    def test_completed_higher_belief_than_cancelled(self):
        doc_comp, _ = from_fhir(_goal(lifecycle_status="completed"))
        doc_canc, _ = from_fhir(_goal(lifecycle_status="cancelled"))
        assert doc_comp["opinions"][0]["opinion"].belief > doc_canc["opinions"][0]["opinion"].belief

    def test_achievement_status_produces_second_opinion(self):
        doc, _ = from_fhir(_goal(
            lifecycle_status="active",
            achievement_status="in-progress",
        ))
        achievement_ops = [o for o in doc["opinions"] if o["field"] == "achievementStatus"]
        assert len(achievement_ops) == 1
        op = achievement_ops[0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_achieved_higher_belief_than_worsening(self):
        doc_ach, _ = from_fhir(_goal(lifecycle_status="active", achievement_status="achieved"))
        doc_wor, _ = from_fhir(_goal(lifecycle_status="active", achievement_status="worsening"))
        ach_op = [o for o in doc_ach["opinions"] if o["field"] == "achievementStatus"][0]
        wor_op = [o for o in doc_wor["opinions"] if o["field"] == "achievementStatus"][0]
        assert ach_op["opinion"].belief > wor_op["opinion"].belief

    def test_no_achievement_status_single_opinion(self):
        doc, _ = from_fhir(_goal(lifecycle_status="active"))
        assert len(doc["opinions"]) == 1

    def test_additivity_all_lifecycle(self):
        for s in ("active", "accepted", "planned", "proposed", "on-hold",
                   "completed", "cancelled", "entered-in-error", "rejected"):
            doc, _ = from_fhir(_goal(lifecycle_status=s))
            for entry in doc["opinions"]:
                op = entry["opinion"]
                assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirCareTeam:
    """CareTeam: 'this care team assignment is active.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_care_team(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:CareTeam"
        assert len(doc["opinions"]) >= 1

    def test_active_higher_belief_than_inactive(self):
        doc_act, _ = from_fhir(_care_team(status="active"))
        doc_inact, _ = from_fhir(_care_team(status="inactive"))
        assert doc_act["opinions"][0]["opinion"].belief > doc_inact["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("active", "proposed", "suspended", "inactive", "entered-in-error"):
            doc, _ = from_fhir(_care_team(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirImagingStudy:
    """ImagingStudy: 'this imaging study is complete and valid.'"""

    def test_basic_available(self):
        doc, report = from_fhir(_imaging_study(status="available"))
        assert report.success is True
        assert doc["@type"] == "fhir:ImagingStudy"
        assert len(doc["opinions"]) >= 1

    def test_available_higher_belief_than_cancelled(self):
        doc_avail, _ = from_fhir(_imaging_study(status="available"))
        doc_canc, _ = from_fhir(_imaging_study(status="cancelled"))
        assert doc_avail["opinions"][0]["opinion"].belief > doc_canc["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("available", "registered", "cancelled", "entered-in-error"):
            doc, _ = from_fhir(_imaging_study(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Batch 2: Administrative identity
# ═══════════════════════════════════════════════════════════════════


class TestFromFhirPatient:
    """Patient: 'this patient record is accurate.' No status field."""

    def test_basic_patient(self):
        doc, report = from_fhir(_patient())
        assert report.success is True
        assert doc["@type"] == "fhir:Patient"
        assert len(doc["opinions"]) >= 1

    def test_opinion_is_valid(self):
        doc, _ = from_fhir(_patient())
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_field_is_data_quality(self):
        """Patient opinion field should reflect data quality assessment."""
        doc, _ = from_fhir(_patient())
        assert doc["opinions"][0]["field"] in ("active", "data_quality")

    def test_id_preserved(self):
        doc, _ = from_fhir(_patient())
        assert doc["id"] == "patient-1"

    def test_more_data_lower_uncertainty(self):
        """Patient with more fields → lower uncertainty than minimal patient."""
        minimal = {"resourceType": "Patient", "id": "p-min"}
        rich = _patient(extras={
            "telecom": [{"value": "555-1234"}],
            "address": [{"city": "Boston"}],
            "maritalStatus": {"text": "Married"},
            "identifier": [{"value": "MRN-12345"}],
        })
        doc_min, _ = from_fhir(minimal)
        doc_rich, _ = from_fhir(rich)
        u_min = doc_min["opinions"][0]["opinion"].uncertainty
        u_rich = doc_rich["opinions"][0]["opinion"].uncertainty
        assert u_rich < u_min


class TestFromFhirOrganization:
    """Organization: 'this organization record is valid.' Uses boolean active."""

    def test_basic_active(self):
        doc, report = from_fhir(_organization(active=True))
        assert report.success is True
        assert doc["@type"] == "fhir:Organization"
        assert len(doc["opinions"]) >= 1

    def test_active_true_high_belief(self):
        doc, _ = from_fhir(_organization(active=True))
        assert doc["opinions"][0]["opinion"].belief > 0.5

    def test_active_false_low_belief(self):
        doc, _ = from_fhir(_organization(active=False))
        assert doc["opinions"][0]["opinion"].belief < 0.4

    def test_active_true_vs_false(self):
        doc_t, _ = from_fhir(_organization(active=True))
        doc_f, _ = from_fhir(_organization(active=False))
        assert doc_t["opinions"][0]["opinion"].belief > doc_f["opinions"][0]["opinion"].belief

    def test_no_active_field_still_produces_opinion(self):
        res = {"resourceType": "Organization", "id": "org-no-active", "name": "Test"}
        doc, report = from_fhir(res)
        assert report.success is True
        assert len(doc["opinions"]) >= 1

    def test_additivity(self):
        for active in (True, False):
            doc, _ = from_fhir(_organization(active=active))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirPractitioner:
    """Practitioner: 'this practitioner record is valid.' Uses boolean active."""

    def test_basic_active(self):
        doc, report = from_fhir(_practitioner(active=True))
        assert report.success is True
        assert doc["@type"] == "fhir:Practitioner"
        assert len(doc["opinions"]) >= 1

    def test_active_true_vs_false(self):
        doc_t, _ = from_fhir(_practitioner(active=True))
        doc_f, _ = from_fhir(_practitioner(active=False))
        assert doc_t["opinions"][0]["opinion"].belief > doc_f["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for active in (True, False):
            doc, _ = from_fhir(_practitioner(active=active))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirDevice:
    """Device: 'this device record is valid.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_device(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:Device"
        assert len(doc["opinions"]) >= 1

    def test_active_higher_belief_than_inactive(self):
        doc_act, _ = from_fhir(_device(status="active"))
        doc_inact, _ = from_fhir(_device(status="inactive"))
        assert doc_act["opinions"][0]["opinion"].belief > doc_inact["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("active", "inactive", "entered-in-error"):
            doc, _ = from_fhir(_device(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Batch 3: Financial
# ═══════════════════════════════════════════════════════════════════


class TestFromFhirClaim:
    """Claim: 'this claim is valid and justified.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_claim(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:Claim"
        assert len(doc["opinions"]) >= 1

    def test_active_higher_belief_than_cancelled(self):
        doc_act, _ = from_fhir(_claim(status="active"))
        doc_canc, _ = from_fhir(_claim(status="cancelled"))
        assert doc_act["opinions"][0]["opinion"].belief > doc_canc["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("active", "cancelled", "draft", "entered-in-error"):
            doc, _ = from_fhir(_claim(status=s))
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestFromFhirExplanationOfBenefit:
    """ExplanationOfBenefit: 'this adjudication is accurate.'"""

    def test_basic_active(self):
        doc, report = from_fhir(_eob(status="active"))
        assert report.success is True
        assert doc["@type"] == "fhir:ExplanationOfBenefit"
        assert len(doc["opinions"]) >= 1

    def test_outcome_complete_produces_second_opinion(self):
        doc, _ = from_fhir(_eob(status="active", outcome="complete"))
        outcome_ops = [o for o in doc["opinions"] if o["field"] == "outcome"]
        assert len(outcome_ops) == 1
        op = outcome_ops[0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_outcome_complete_higher_belief_than_error(self):
        doc_comp, _ = from_fhir(_eob(status="active", outcome="complete"))
        doc_err, _ = from_fhir(_eob(status="active", outcome="error"))
        comp_op = [o for o in doc_comp["opinions"] if o["field"] == "outcome"][0]
        err_op = [o for o in doc_err["opinions"] if o["field"] == "outcome"][0]
        assert comp_op["opinion"].belief > err_op["opinion"].belief

    def test_no_outcome_single_opinion(self):
        doc, _ = from_fhir(_eob(status="active"))
        assert len(doc["opinions"]) == 1

    def test_additivity(self):
        for s in ("active", "cancelled", "draft", "entered-in-error"):
            doc, _ = from_fhir(_eob(status=s))
            for entry in doc["opinions"]:
                op = entry["opinion"]
                assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# to_fhir() + round-trip tests for all new types
# ═══════════════════════════════════════════════════════════════════


class TestFromFhirMedicationAdministration:
    """MedicationAdministration: 'this medication was administered as documented.'"""

    def test_basic_completed(self):
        res = {"resourceType": "MedicationAdministration", "id": "ma-1", "status": "completed"}
        doc, report = from_fhir(res)
        assert report.success is True
        assert doc["@type"] == "fhir:MedicationAdministration"
        assert len(doc["opinions"]) >= 1

    def test_completed_higher_belief_than_not_done(self):
        doc_comp, _ = from_fhir({"resourceType": "MedicationAdministration", "id": "ma-1", "status": "completed"})
        doc_nd, _ = from_fhir({"resourceType": "MedicationAdministration", "id": "ma-2", "status": "not-done"})
        assert doc_comp["opinions"][0]["opinion"].belief > doc_nd["opinions"][0]["opinion"].belief

    def test_additivity(self):
        for s in ("completed", "in-progress", "not-done", "on-hold", "stopped", "entered-in-error"):
            doc, _ = from_fhir({"resourceType": "MedicationAdministration", "id": "ma-1", "status": s})
            op = doc["opinions"][0]["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


class TestToFhirUniversal:
    """to_fhir() and round-trip for all Phase 6 resource types."""

    @pytest.mark.parametrize("fixture,status_field", [
        (_encounter, "status"),
        (_medication_request, "status"),
        (_care_plan, "status"),
        (_care_team, "status"),
        (_imaging_study, "status"),
        (_device, "status"),
        (_claim, "status"),
    ])
    def test_from_to_round_trip_status_types(self, fixture, status_field):
        """Status-based resources survive from_fhir → to_fhir round-trip."""
        resource_in = fixture()
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == resource_in["resourceType"]
        assert resource_out["id"] == resource_in["id"]
        assert "_status" in resource_out

    def test_goal_round_trip(self):
        """Goal survives round-trip with lifecycleStatus."""
        resource_in = _goal(lifecycle_status="active")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Goal"
        assert resource_out.get("lifecycleStatus") == "active"

    def test_patient_round_trip(self):
        """Patient survives round-trip."""
        resource_in = _patient()
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Patient"

    def test_organization_round_trip(self):
        """Organization survives round-trip."""
        resource_in = _organization(active=True)
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Organization"
        assert resource_out.get("active") is True

    def test_practitioner_round_trip(self):
        """Practitioner survives round-trip."""
        resource_in = _practitioner(active=True)
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Practitioner"
        assert resource_out.get("active") is True

    def test_eob_round_trip(self):
        """ExplanationOfBenefit survives round-trip."""
        resource_in = _eob(status="active", outcome="complete")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "ExplanationOfBenefit"

    def test_round_trip_opinion_fidelity(self):
        """Opinion values survive from_fhir → to_fhir → from_fhir."""
        resource_in = _encounter(status="finished")
        doc1, _ = from_fhir(resource_in)
        resource_mid, _ = to_fhir(doc1)
        doc2, _ = from_fhir(resource_mid)

        op1 = doc1["opinions"][0]["opinion"]
        op2 = doc2["opinions"][0]["opinion"]
        assert abs(op1.belief - op2.belief) < 1e-9
        assert abs(op1.disbelief - op2.disbelief) < 1e-9
        assert abs(op1.uncertainty - op2.uncertainty) < 1e-9
        assert doc2["opinions"][0]["source"] == "extension"


# ═══════════════════════════════════════════════════════════════════
# Cross-cutting: all 12 types integrate with existing pipeline
# ═══════════════════════════════════════════════════════════════════


class TestPhase6Integration:
    """Phase 6 resources integrate with existing fhir_clinical_fuse."""

    def test_all_13_types_handled(self):
        """All 13 new resource types are handled by from_fhir()."""
        resources = [
            _encounter(), _medication_request(), _care_plan(),
            _goal(), _care_team(), _imaging_study(),
            _patient(), _organization(), _practitioner(),
            _device(), _claim(), _eob(),
            {"resourceType": "MedicationAdministration", "id": "ma-1", "status": "completed"},
        ]
        for res in resources:
            doc, report = from_fhir(res)
            assert report.success is True, f"{res['resourceType']} failed"
            assert len(doc["opinions"]) >= 1, f"{res['resourceType']} no opinions"

    def test_phase6_fusable_with_phase1(self):
        """Phase 6 documents can fuse with Phase 1 RiskAssessment."""
        from jsonld_ex.fhir_interop import fhir_clinical_fuse

        risk = {
            "resourceType": "RiskAssessment",
            "id": "risk-1",
            "status": "final",
            "prediction": [{"probabilityDecimal": 0.80, "outcome": {"text": "MI"}}],
        }
        doc_risk, _ = from_fhir(risk)
        doc_enc, _ = from_fhir(_encounter(status="finished"))

        fused, report = fhir_clinical_fuse([doc_risk, doc_enc])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9
