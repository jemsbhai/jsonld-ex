"""Tests for Observation and DiagnosticReport status-based fallback opinions.

RED PHASE: These tests define the expected behavior for when Observation
and DiagnosticReport resources lack their primary opinion-producing fields
(interpretation and conclusion respectively) but DO have a status field.

Problem: Real-world FHIR data (e.g., Synthea) produces Observations with
status="final" and valueQuantity but no interpretation code. Our current
converter produces 0 opinions for these resources, causing 187/187
Observations and 15/15 DiagnosticReports in a Synthea bundle to be
effectively invisible to the opinion pipeline.

Fix: When the primary field is absent, fall back to a status-based opinion.
The proposition is "this [observation/report] is valid and reliable."
This follows the same pattern already used by Immunization, Procedure,
Consent, etc.

Design decisions:
  - interpretation (Observation) and conclusion (DiagnosticReport) opinions
    take PRIORITY when present — status fallback is only used when they
    are absent.
  - For DiagnosticReport, the number of linked result[] references
    modulates uncertainty (more results = more evidence = lower uncertainty).
  - New constants OBSERVATION_STATUS_PROBABILITY/_UNCERTAINTY and
    DIAGNOSTIC_REPORT_STATUS_PROBABILITY/_UNCERTAINTY are added to
    _constants.py following the existing pattern.
  - to_fhir() must handle the new "status" field opinion for both
    resource types, attaching the extension to _status.
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


def _observation_no_interp(
    *,
    status="final",
    value_quantity=None,
    category=None,
):
    """Build a Synthea-style Observation: status + valueQuantity, no interpretation."""
    resource = {
        "resourceType": "Observation",
        "id": "obs-synthea-1",
        "status": status,
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "85354-9"}],
            "text": "Blood Pressure",
        },
    }
    if value_quantity is not None:
        resource["valueQuantity"] = value_quantity
    if category is not None:
        resource["category"] = category
    return resource


def _diagnostic_report_no_conclusion(
    *,
    status="final",
    results=None,
):
    """Build a Synthea-style DiagnosticReport: status + result[], no conclusion."""
    resource = {
        "resourceType": "DiagnosticReport",
        "id": "report-synthea-1",
        "status": status,
        "code": {"text": "Lipid Panel"},
    }
    if results is not None:
        resource["result"] = [
            {"reference": f"Observation/{r}"} for r in results
        ]
    return resource


# ═══════════════════════════════════════════════════════════════════
# Observation: status-based fallback tests
# ═══════════════════════════════════════════════════════════════════


class TestObservationStatusFallback:
    """Observation without interpretation → status-based fallback opinion."""

    def test_no_interp_final_produces_opinion(self):
        """Observation with status='final' but no interpretation → produces opinion."""
        resource = _observation_no_interp(status="final")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:Observation"
        assert len(doc["opinions"]) >= 1, (
            "Observation with status='final' should produce a status-based "
            "fallback opinion even without interpretation"
        )

    def test_fallback_opinion_is_valid(self):
        """Fallback opinion satisfies b + d + u = 1."""
        resource = _observation_no_interp(status="final")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        total = op.belief + op.disbelief + op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_fallback_field_is_status(self):
        """Fallback opinion has field='status' to distinguish from interpretation."""
        resource = _observation_no_interp(status="final")
        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["field"] == "status"
        assert entry["value"] == "final"
        assert entry["source"] == "reconstructed"

    def test_fallback_final_high_belief(self):
        """status='final' → high belief (observation is valid and reliable)."""
        resource = _observation_no_interp(status="final")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief > 0.5

    def test_fallback_preliminary_lower_belief_than_final(self):
        """status='preliminary' → lower belief than 'final'."""
        res_final = _observation_no_interp(status="final")
        res_prelim = _observation_no_interp(status="preliminary")

        doc_final, _ = from_fhir(res_final)
        doc_prelim, _ = from_fhir(res_prelim)

        b_final = doc_final["opinions"][0]["opinion"].belief
        b_prelim = doc_prelim["opinions"][0]["opinion"].belief
        assert b_final > b_prelim

    def test_fallback_preliminary_higher_uncertainty_than_final(self):
        """status='preliminary' → higher uncertainty than 'final'."""
        res_final = _observation_no_interp(status="final")
        res_prelim = _observation_no_interp(status="preliminary")

        doc_final, _ = from_fhir(res_final)
        doc_prelim, _ = from_fhir(res_prelim)

        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_prelim > u_final

    def test_fallback_entered_in_error_high_uncertainty(self):
        """status='entered-in-error' → very high uncertainty."""
        resource = _observation_no_interp(status="entered-in-error")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.4

    def test_fallback_cancelled_low_belief(self):
        """status='cancelled' → low belief."""
        resource = _observation_no_interp(status="cancelled")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.4

    def test_interpretation_takes_priority(self):
        """When interpretation IS present, it takes priority over status fallback."""
        resource = {
            "resourceType": "Observation",
            "id": "obs-with-interp",
            "status": "final",
            "interpretation": [{"coding": [{"code": "H"}]}],
        }
        doc, _ = from_fhir(resource)

        # Should have interpretation-based opinion, NOT status-based
        interp_opinions = [o for o in doc["opinions"] if o["field"] == "interpretation"]
        status_opinions = [o for o in doc["opinions"] if o["field"] == "status"]
        assert len(interp_opinions) >= 1
        assert len(status_opinions) == 0

    def test_fallback_value_quantity_still_captured(self):
        """valueQuantity is preserved alongside status-based fallback opinion."""
        resource = _observation_no_interp(
            status="final",
            value_quantity={"value": 120, "unit": "mmHg"},
        )
        doc, _ = from_fhir(resource)

        assert doc.get("value") == {"value": 120, "unit": "mmHg"}
        assert len(doc["opinions"]) >= 1

    def test_fallback_all_statuses_valid(self):
        """Every known Observation status produces a valid fallback opinion."""
        for status in ("final", "preliminary", "registered", "amended",
                       "corrected", "cancelled", "entered-in-error", "unknown"):
            resource = _observation_no_interp(status=status)
            doc, report = from_fhir(resource)
            assert report.success is True
            assert len(doc["opinions"]) >= 1, (
                f"status='{status}' should produce a fallback opinion"
            )
            op = doc["opinions"][0]["opinion"]
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"Additivity violated for status='{status}': {total}"
            )

    def test_fallback_extension_recovery_on_status(self):
        """Extension on _status overrides status-based reconstruction."""
        injected = Opinion(belief=0.90, disbelief=0.02, uncertainty=0.08)
        ext = opinion_to_fhir_extension(injected)
        resource = _observation_no_interp(status="final")
        resource["_status"] = {"extension": [ext]}

        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.90) < 1e-9

    def test_nodes_converted_counts_fallback(self):
        """ConversionReport counts the status-based fallback as a converted node."""
        resource = _observation_no_interp(status="final")
        _, report = from_fhir(resource)
        assert report.nodes_converted >= 1


# ═══════════════════════════════════════════════════════════════════
# DiagnosticReport: status-based fallback tests
# ═══════════════════════════════════════════════════════════════════


class TestDiagnosticReportStatusFallback:
    """DiagnosticReport without conclusion → status-based fallback opinion."""

    def test_no_conclusion_final_produces_opinion(self):
        """DiagnosticReport with status='final' but no conclusion → produces opinion."""
        resource = _diagnostic_report_no_conclusion(status="final")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:DiagnosticReport"
        assert len(doc["opinions"]) >= 1, (
            "DiagnosticReport with status='final' should produce a status-based "
            "fallback opinion even without conclusion"
        )

    def test_fallback_opinion_is_valid(self):
        """Fallback opinion satisfies b + d + u = 1."""
        resource = _diagnostic_report_no_conclusion(status="final")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        total = op.belief + op.disbelief + op.uncertainty
        assert abs(total - 1.0) < 1e-9

    def test_fallback_field_is_status(self):
        """Fallback opinion has field='status'."""
        resource = _diagnostic_report_no_conclusion(status="final")
        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["field"] == "status"
        assert entry["value"] == "final"
        assert entry["source"] == "reconstructed"

    def test_fallback_final_high_belief(self):
        """status='final' → high belief."""
        resource = _diagnostic_report_no_conclusion(status="final")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief > 0.5

    def test_fallback_preliminary_higher_uncertainty_than_final(self):
        """status='preliminary' → higher uncertainty than 'final'."""
        res_final = _diagnostic_report_no_conclusion(status="final")
        res_prelim = _diagnostic_report_no_conclusion(status="preliminary")

        doc_final, _ = from_fhir(res_final)
        doc_prelim, _ = from_fhir(res_prelim)

        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_prelim > u_final

    def test_fallback_entered_in_error_high_uncertainty(self):
        """status='entered-in-error' → very high uncertainty."""
        resource = _diagnostic_report_no_conclusion(status="entered-in-error")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.4

    def test_result_count_reduces_uncertainty(self):
        """More result[] references → lower uncertainty (more evidence)."""
        res_no_results = _diagnostic_report_no_conclusion(
            status="final", results=[],
        )
        res_many_results = _diagnostic_report_no_conclusion(
            status="final",
            results=["obs-ldl", "obs-hdl", "obs-trig", "obs-chol"],
        )

        doc_no, _ = from_fhir(res_no_results)
        doc_many, _ = from_fhir(res_many_results)

        u_no = doc_no["opinions"][0]["opinion"].uncertainty
        u_many = doc_many["opinions"][0]["opinion"].uncertainty
        assert u_many < u_no

    def test_conclusion_takes_priority(self):
        """When conclusion IS present, it takes priority over status fallback."""
        resource = {
            "resourceType": "DiagnosticReport",
            "id": "report-with-conclusion",
            "status": "final",
            "code": {"text": "Lipid Panel"},
            "conclusion": "Normal lipid levels",
        }
        doc, _ = from_fhir(resource)

        # Should have conclusion-based opinion, NOT status-based
        conclusion_opinions = [o for o in doc["opinions"] if o["field"] == "conclusion"]
        status_opinions = [o for o in doc["opinions"] if o["field"] == "status"]
        assert len(conclusion_opinions) >= 1
        assert len(status_opinions) == 0

    def test_result_references_still_captured(self):
        """result_references are preserved alongside status-based fallback."""
        resource = _diagnostic_report_no_conclusion(
            status="final",
            results=["obs-1", "obs-2"],
        )
        doc, _ = from_fhir(resource)

        assert len(doc["result_references"]) == 2
        assert len(doc["opinions"]) >= 1

    def test_fallback_all_statuses_valid(self):
        """Every known DiagnosticReport status produces a valid fallback opinion."""
        for status in ("final", "preliminary", "registered", "amended",
                       "corrected", "cancelled", "entered-in-error",
                       "appended", "unknown"):
            resource = _diagnostic_report_no_conclusion(status=status)
            doc, report = from_fhir(resource)
            assert report.success is True
            assert len(doc["opinions"]) >= 1, (
                f"status='{status}' should produce a fallback opinion"
            )
            op = doc["opinions"][0]["opinion"]
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"Additivity violated for status='{status}': {total}"
            )

    def test_fallback_extension_recovery_on_status(self):
        """Extension on _status overrides status-based reconstruction."""
        injected = Opinion(belief=0.88, disbelief=0.04, uncertainty=0.08)
        ext = opinion_to_fhir_extension(injected)
        resource = _diagnostic_report_no_conclusion(status="final")
        resource["_status"] = {"extension": [ext]}

        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.88) < 1e-9

    def test_nodes_converted_counts_fallback(self):
        """ConversionReport counts the status-based fallback as a converted node."""
        resource = _diagnostic_report_no_conclusion(status="final")
        _, report = from_fhir(resource)
        assert report.nodes_converted >= 1


# ═══════════════════════════════════════════════════════════════════
# to_fhir() round-trip for status-based fallback opinions
# ═══════════════════════════════════════════════════════════════════


class TestStatusFallbackToFhir:
    """to_fhir() correctly exports status-based fallback opinions."""

    def test_observation_status_fallback_exports(self):
        """Observation with status-field opinion → _status extension in FHIR."""
        op = Opinion(belief=0.72, disbelief=0.13, uncertainty=0.15)
        doc = {
            "@type": "fhir:Observation",
            "id": "obs-fb-1",
            "status": "final",
            "opinions": [{
                "field": "status",
                "value": "final",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "Observation"
        assert resource["status"] == "final"
        assert "_status" in resource
        exts = resource["_status"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1

    def test_diagnostic_report_status_fallback_exports(self):
        """DiagnosticReport with status-field opinion → _status extension in FHIR."""
        op = Opinion(belief=0.72, disbelief=0.13, uncertainty=0.15)
        doc = {
            "@type": "fhir:DiagnosticReport",
            "id": "report-fb-1",
            "status": "final",
            "result_references": ["Observation/obs-1"],
            "opinions": [{
                "field": "status",
                "value": "final",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "DiagnosticReport"
        assert resource["status"] == "final"
        assert "_status" in resource
        exts = resource["_status"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1

    def test_observation_round_trip_status_fallback(self):
        """Observation: from_fhir → to_fhir round-trip preserves status fallback."""
        resource_in = _observation_no_interp(status="final")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["resourceType"] == "Observation"
        assert resource_out["status"] == "final"
        assert "_status" in resource_out

    def test_diagnostic_report_round_trip_status_fallback(self):
        """DiagnosticReport: from_fhir → to_fhir round-trip preserves status fallback."""
        resource_in = _diagnostic_report_no_conclusion(
            status="final", results=["obs-1"],
        )
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["resourceType"] == "DiagnosticReport"
        assert resource_out["status"] == "final"
        assert "_status" in resource_out

    def test_observation_round_trip_opinion_fidelity(self):
        """Observation status fallback opinion survives full round-trip via extension."""
        resource_in = _observation_no_interp(status="final")
        doc1, _ = from_fhir(resource_in)
        resource_mid, _ = to_fhir(doc1)
        doc2, _ = from_fhir(resource_mid)

        op1 = doc1["opinions"][0]["opinion"]
        op2 = doc2["opinions"][0]["opinion"]
        assert abs(op1.belief - op2.belief) < 1e-9
        assert abs(op1.disbelief - op2.disbelief) < 1e-9
        assert abs(op1.uncertainty - op2.uncertainty) < 1e-9
        assert doc2["opinions"][0]["source"] == "extension"

    def test_diagnostic_report_round_trip_opinion_fidelity(self):
        """DiagnosticReport status fallback opinion survives full round-trip."""
        resource_in = _diagnostic_report_no_conclusion(status="final")
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
# Constants: verify new mappings are importable and well-formed
# ═══════════════════════════════════════════════════════════════════


class TestStatusFallbackConstants:
    """New constants for Observation and DiagnosticReport status fallback."""

    def test_observation_status_probability_importable(self):
        """OBSERVATION_STATUS_PROBABILITY is importable from _constants."""
        from jsonld_ex.fhir_interop._constants import OBSERVATION_STATUS_PROBABILITY
        assert isinstance(OBSERVATION_STATUS_PROBABILITY, dict)
        assert "final" in OBSERVATION_STATUS_PROBABILITY
        assert "preliminary" in OBSERVATION_STATUS_PROBABILITY
        assert "entered-in-error" in OBSERVATION_STATUS_PROBABILITY

    def test_observation_status_uncertainty_importable(self):
        """OBSERVATION_STATUS_UNCERTAINTY is importable from _constants."""
        from jsonld_ex.fhir_interop._constants import OBSERVATION_STATUS_UNCERTAINTY
        assert isinstance(OBSERVATION_STATUS_UNCERTAINTY, dict)
        assert "final" in OBSERVATION_STATUS_UNCERTAINTY
        assert "preliminary" in OBSERVATION_STATUS_UNCERTAINTY

    def test_diagnostic_report_status_probability_importable(self):
        """DIAGNOSTIC_REPORT_STATUS_PROBABILITY is importable from _constants."""
        from jsonld_ex.fhir_interop._constants import DIAGNOSTIC_REPORT_STATUS_PROBABILITY
        assert isinstance(DIAGNOSTIC_REPORT_STATUS_PROBABILITY, dict)
        assert "final" in DIAGNOSTIC_REPORT_STATUS_PROBABILITY
        assert "preliminary" in DIAGNOSTIC_REPORT_STATUS_PROBABILITY

    def test_diagnostic_report_status_uncertainty_importable(self):
        """DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY is importable from _constants."""
        from jsonld_ex.fhir_interop._constants import DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY
        assert isinstance(DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY, dict)
        assert "final" in DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY
        assert "preliminary" in DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY

    def test_observation_final_higher_prob_than_preliminary(self):
        """'final' should map to higher probability than 'preliminary'."""
        from jsonld_ex.fhir_interop._constants import OBSERVATION_STATUS_PROBABILITY
        assert OBSERVATION_STATUS_PROBABILITY["final"] > OBSERVATION_STATUS_PROBABILITY["preliminary"]

    def test_observation_final_lower_uncertainty_than_preliminary(self):
        """'final' should map to lower uncertainty than 'preliminary'."""
        from jsonld_ex.fhir_interop._constants import OBSERVATION_STATUS_UNCERTAINTY
        assert OBSERVATION_STATUS_UNCERTAINTY["final"] < OBSERVATION_STATUS_UNCERTAINTY["preliminary"]

    def test_diagnostic_report_final_higher_prob_than_preliminary(self):
        """'final' should map to higher probability than 'preliminary'."""
        from jsonld_ex.fhir_interop._constants import DIAGNOSTIC_REPORT_STATUS_PROBABILITY
        assert DIAGNOSTIC_REPORT_STATUS_PROBABILITY["final"] > DIAGNOSTIC_REPORT_STATUS_PROBABILITY["preliminary"]

    def test_all_probabilities_in_valid_range(self):
        """All probability values are in [0, 1]."""
        from jsonld_ex.fhir_interop._constants import (
            OBSERVATION_STATUS_PROBABILITY,
            DIAGNOSTIC_REPORT_STATUS_PROBABILITY,
        )
        for mapping in (OBSERVATION_STATUS_PROBABILITY, DIAGNOSTIC_REPORT_STATUS_PROBABILITY):
            for key, val in mapping.items():
                assert 0.0 <= val <= 1.0, f"{key} → {val} out of range"

    def test_all_uncertainties_in_valid_range(self):
        """All uncertainty values are in [0, 1)."""
        from jsonld_ex.fhir_interop._constants import (
            OBSERVATION_STATUS_UNCERTAINTY,
            DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY,
        )
        for mapping in (OBSERVATION_STATUS_UNCERTAINTY, DIAGNOSTIC_REPORT_STATUS_UNCERTAINTY):
            for key, val in mapping.items():
                assert 0.0 <= val < 1.0, f"{key} → {val} out of range"
