"""
Tests for FHIR R4 Coverage ↔ jsonld-ex bidirectional conversion.

Coverage completes the financial chain:
    **Coverage** → Claim → ExplanationOfBenefit

Proposition under assessment: "This insurance coverage record is valid
and currently in force."

Coverage uses a custom handler (not ``_make_status_handler``) because
FHIR Reference fields (beneficiary, subscriber, payor) and
CodeableConcept fields (type, relationship) require proper extraction
and reconstruction for round-trip fidelity — consistent with how
ServiceRequest, Specimen, and DocumentReference handle references.

FHIR R4 Coverage.status uses FinancialResourceStatusCodes (Required):
    active, cancelled, draft, entered-in-error

Test organisation:
    1. Constants — probability/uncertainty maps cover all R4 codes
    2. Status-based opinion reconstruction — each status code
    3. Extension recovery — exact opinion from jsonld-ex extension
    4. Metadata passthrough — period, type, beneficiary, payor, etc.
    5. Round-trip — from_fhir → to_fhir preserves all fields
    6. Edge cases — missing fields, novel status, empty arrays
    7. Integration — Coverage + Claim + EOB financial chain fusion
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    FHIR_EXTENSION_URL,
)
from jsonld_ex.fhir_interop._constants import (
    COVERAGE_STATUS_PROBABILITY,
    COVERAGE_STATUS_UNCERTAINTY,
    SUPPORTED_RESOURCE_TYPES,
)
from jsonld_ex.fhir_interop._scalar import opinion_to_fhir_extension


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _coverage(
    *,
    status: str = "active",
    coverage_id: str = "cov-1",
    period: dict | None = None,
    type_text: str | None = None,
    type_coding: list | None = None,
    subscriber_ref: str | None = None,
    beneficiary_ref: str | None = None,
    payor_refs: list[str] | None = None,
    dependent: str | None = None,
    relationship_code: str | None = None,
    order: int | None = None,
    network: str | None = None,
) -> dict:
    """Build a minimal FHIR R4 Coverage resource for testing."""
    resource: dict = {
        "resourceType": "Coverage",
        "id": coverage_id,
        "status": status,
    }
    if period is not None:
        resource["period"] = period
    if type_text is not None or type_coding is not None:
        type_obj: dict = {}
        if type_text is not None:
            type_obj["text"] = type_text
        if type_coding is not None:
            type_obj["coding"] = type_coding
        resource["type"] = type_obj
    if subscriber_ref is not None:
        resource["subscriber"] = {"reference": subscriber_ref}
    if beneficiary_ref is not None:
        resource["beneficiary"] = {"reference": beneficiary_ref}
    if payor_refs is not None:
        resource["payor"] = [{"reference": r} for r in payor_refs]
    if dependent is not None:
        resource["dependent"] = dependent
    if relationship_code is not None:
        resource["relationship"] = {
            "coding": [{"code": relationship_code}],
        }
    if order is not None:
        resource["order"] = order
    if network is not None:
        resource["network"] = network
    return resource


def _inject_extension(resource: dict, opinion: Opinion) -> dict:
    """Inject a jsonld-ex opinion extension onto _status."""
    ext = opinion_to_fhir_extension(opinion)
    resource["_status"] = {"extension": [ext]}
    return resource


# ═══════════════════════════════════════════════════════════════════
# 1. Constants
# ═══════════════════════════════════════════════════════════════════


class TestCoverageConstants:
    """Constants cover all FHIR R4 FinancialResourceStatusCodes."""

    EXPECTED_CODES = {"active", "cancelled", "draft", "entered-in-error"}

    def test_probability_keys_complete(self):
        """Probability map covers all 4 R4 status codes."""
        assert set(COVERAGE_STATUS_PROBABILITY.keys()) == self.EXPECTED_CODES

    def test_uncertainty_keys_complete(self):
        """Uncertainty map covers all 4 R4 status codes."""
        assert set(COVERAGE_STATUS_UNCERTAINTY.keys()) == self.EXPECTED_CODES

    def test_probabilities_in_valid_range(self):
        """All probability values are in (0, 1)."""
        for code, prob in COVERAGE_STATUS_PROBABILITY.items():
            assert 0.0 < prob < 1.0, f"{code}: {prob}"

    def test_uncertainties_in_valid_range(self):
        """All uncertainty values are in (0, 1)."""
        for code, u in COVERAGE_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, f"{code}: {u}"

    def test_coverage_in_supported_types(self):
        """Coverage is registered in SUPPORTED_RESOURCE_TYPES."""
        assert "Coverage" in SUPPORTED_RESOURCE_TYPES

    def test_active_has_highest_probability(self):
        """Active coverage has the highest probability."""
        assert COVERAGE_STATUS_PROBABILITY["active"] == max(
            COVERAGE_STATUS_PROBABILITY.values()
        )

    def test_cancelled_has_lowest_probability(self):
        """Cancelled coverage has the lowest probability."""
        assert COVERAGE_STATUS_PROBABILITY["cancelled"] == min(
            COVERAGE_STATUS_PROBABILITY.values()
        )

    def test_entered_in_error_has_highest_uncertainty(self):
        """Entered-in-error has the highest uncertainty."""
        assert COVERAGE_STATUS_UNCERTAINTY["entered-in-error"] == max(
            COVERAGE_STATUS_UNCERTAINTY.values()
        )


# ═══════════════════════════════════════════════════════════════════
# 2. Status-based opinion reconstruction
# ═══════════════════════════════════════════════════════════════════


class TestCoverageStatusReconstruction:
    """Each status code produces a valid reconstructed opinion."""

    @pytest.mark.parametrize("status", [
        "active", "cancelled", "draft", "entered-in-error",
    ])
    def test_each_status_produces_opinion(self, status):
        """from_fhir produces a valid SL opinion for each status code."""
        resource = _coverage(status=status)
        doc, report = from_fhir(resource)

        assert report.success is True
        assert report.nodes_converted == 1
        assert doc["@type"] == "fhir:Coverage"
        assert doc["id"] == "cov-1"
        assert doc["status"] == status
        assert len(doc["opinions"]) == 1

        entry = doc["opinions"][0]
        assert entry["field"] == "status"
        assert entry["value"] == status
        assert entry["source"] == "reconstructed"

        op: Opinion = entry["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_active_high_belief(self):
        """Active coverage → high belief, low uncertainty."""
        doc, _ = from_fhir(_coverage(status="active"))
        op = doc["opinions"][0]["opinion"]
        assert op.belief > 0.5
        assert op.uncertainty < 0.3

    def test_cancelled_low_belief(self):
        """Cancelled coverage → low belief."""
        doc, _ = from_fhir(_coverage(status="cancelled"))
        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.3

    def test_draft_moderate_uncertainty(self):
        """Draft coverage → moderate uncertainty (genuinely uncertain)."""
        doc, _ = from_fhir(_coverage(status="draft"))
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.2

    def test_entered_in_error_high_uncertainty(self):
        """Entered-in-error → high uncertainty (data integrity compromised)."""
        doc, _ = from_fhir(_coverage(status="entered-in-error"))
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.4

    def test_unknown_status_fallback(self):
        """Unknown status code uses default probability 0.50."""
        doc, _ = from_fhir(_coverage(status="novel-status"))
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 3. Extension recovery
# ═══════════════════════════════════════════════════════════════════


class TestCoverageExtensionRecovery:
    """Exact SL opinion recovered from jsonld-ex FHIR extension."""

    def test_extension_recovery_exact(self):
        """Opinion from extension matches exactly (not reconstructed)."""
        original_op = Opinion(0.70, 0.10, 0.20)
        resource = _inject_extension(
            _coverage(status="active"), original_op,
        )
        doc, report = from_fhir(resource)

        assert report.success is True
        entry = doc["opinions"][0]
        assert entry["source"] == "extension"

        recovered: Opinion = entry["opinion"]
        assert abs(recovered.belief - 0.70) < 1e-9
        assert abs(recovered.disbelief - 0.10) < 1e-9
        assert abs(recovered.uncertainty - 0.20) < 1e-9

    def test_extension_overrides_reconstruction(self):
        """Extension opinion takes precedence over status reconstruction."""
        # Use "cancelled" status but inject a high-belief extension
        custom_op = Opinion(0.90, 0.05, 0.05)
        resource = _inject_extension(
            _coverage(status="cancelled"), custom_op,
        )
        doc, _ = from_fhir(resource)

        recovered = doc["opinions"][0]["opinion"]
        # Should match the extension, not the "cancelled" reconstruction
        assert recovered.belief > 0.8

    def test_no_extension_falls_back_to_reconstruction(self):
        """Without extension, opinion is reconstructed from status."""
        doc, _ = from_fhir(_coverage(status="active"))
        assert doc["opinions"][0]["source"] == "reconstructed"


# ═══════════════════════════════════════════════════════════════════
# 4. Metadata passthrough
# ═══════════════════════════════════════════════════════════════════


class TestCoverageMetadataPassthrough:
    """All clinically relevant metadata is preserved in the jsonld-ex doc."""

    def test_period_passthrough(self):
        """Period start/end preserved for temporal decay."""
        period = {"start": "2024-01-01", "end": "2024-12-31"}
        doc, _ = from_fhir(_coverage(period=period))
        assert doc["period"] == period

    def test_type_text_passthrough(self):
        """Coverage type text preserved."""
        doc, _ = from_fhir(_coverage(type_text="Medicare Part A"))
        assert doc["type"] == "Medicare Part A"

    def test_beneficiary_ref_passthrough(self):
        """Beneficiary reference extracted as string."""
        doc, _ = from_fhir(_coverage(beneficiary_ref="Patient/p-1"))
        assert doc["beneficiary"] == "Patient/p-1"

    def test_subscriber_ref_passthrough(self):
        """Subscriber reference extracted as string."""
        doc, _ = from_fhir(_coverage(subscriber_ref="Patient/sub-1"))
        assert doc["subscriber"] == "Patient/sub-1"

    def test_payor_refs_passthrough(self):
        """Payor references extracted as list of strings."""
        doc, _ = from_fhir(_coverage(
            payor_refs=["Organization/ins-1", "Organization/ins-2"],
        ))
        assert doc["payor_references"] == [
            "Organization/ins-1", "Organization/ins-2",
        ]

    def test_single_payor_passthrough(self):
        """Single payor still results in a list."""
        doc, _ = from_fhir(_coverage(payor_refs=["Organization/ins-1"]))
        assert doc["payor_references"] == ["Organization/ins-1"]

    def test_dependent_passthrough(self):
        """Dependent string preserved."""
        doc, _ = from_fhir(_coverage(dependent="1"))
        assert doc["dependent"] == "1"

    def test_relationship_code_passthrough(self):
        """Relationship code extracted from CodeableConcept."""
        doc, _ = from_fhir(_coverage(relationship_code="self"))
        assert doc["relationship"] == "self"

    def test_order_passthrough(self):
        """Coverage order (priority) preserved."""
        doc, _ = from_fhir(_coverage(order=1))
        assert doc["order"] == 1

    def test_network_passthrough(self):
        """Network string preserved."""
        doc, _ = from_fhir(_coverage(network="PPO"))
        assert doc["network"] == "PPO"

    def test_absent_metadata_omitted(self):
        """Missing optional fields are not present in the doc."""
        doc, _ = from_fhir(_coverage())
        assert "period" not in doc
        assert "type" not in doc
        assert "beneficiary" not in doc
        assert "subscriber" not in doc
        assert "payor_references" not in doc
        assert "dependent" not in doc
        assert "relationship" not in doc
        assert "order" not in doc
        assert "network" not in doc

    def test_full_metadata_all_present(self):
        """All metadata fields present when populated."""
        doc, _ = from_fhir(_coverage(
            period={"start": "2024-01-01"},
            type_text="HMO",
            beneficiary_ref="Patient/p-1",
            subscriber_ref="Patient/sub-1",
            payor_refs=["Organization/ins-1"],
            dependent="0",
            relationship_code="self",
            order=1,
            network="PPO",
        ))
        assert doc["period"] == {"start": "2024-01-01"}
        assert doc["type"] == "HMO"
        assert doc["beneficiary"] == "Patient/p-1"
        assert doc["subscriber"] == "Patient/sub-1"
        assert doc["payor_references"] == ["Organization/ins-1"]
        assert doc["dependent"] == "0"
        assert doc["relationship"] == "self"
        assert doc["order"] == 1
        assert doc["network"] == "PPO"


# ═══════════════════════════════════════════════════════════════════
# 5. Round-trip fidelity: from_fhir → to_fhir
# ═══════════════════════════════════════════════════════════════════


class TestCoverageRoundTrip:
    """from_fhir → to_fhir preserves structure and opinion."""

    def test_basic_round_trip(self):
        """Minimal Coverage survives round-trip."""
        resource_in = _coverage(status="active")
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Coverage"
        assert resource_out["id"] == "cov-1"
        assert resource_out["status"] == "active"
        assert "_status" in resource_out

    def test_opinion_fidelity_through_round_trip(self):
        """Opinion values survive from_fhir → to_fhir → from_fhir."""
        resource_in = _coverage(status="active")
        doc1, _ = from_fhir(resource_in)
        resource_mid, _ = to_fhir(doc1)
        doc2, _ = from_fhir(resource_mid)

        op1 = doc1["opinions"][0]["opinion"]
        op2 = doc2["opinions"][0]["opinion"]
        assert abs(op1.belief - op2.belief) < 1e-9
        assert abs(op1.disbelief - op2.disbelief) < 1e-9
        assert abs(op1.uncertainty - op2.uncertainty) < 1e-9
        assert doc2["opinions"][0]["source"] == "extension"

    @pytest.mark.parametrize("status", [
        "active", "cancelled", "draft", "entered-in-error",
    ])
    def test_all_statuses_round_trip(self, status):
        """All 4 status codes survive round-trip."""
        resource_in = _coverage(status=status)
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["status"] == status

    def test_period_round_trip(self):
        """Period survives round-trip."""
        period = {"start": "2024-01-01", "end": "2024-12-31"}
        resource_in = _coverage(period=period)
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out.get("period") == period

    def test_beneficiary_round_trip(self):
        """Beneficiary reference survives round-trip (string → Reference)."""
        resource_in = _coverage(beneficiary_ref="Patient/p-1")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["beneficiary"] == {"reference": "Patient/p-1"}

    def test_subscriber_round_trip(self):
        """Subscriber reference survives round-trip."""
        resource_in = _coverage(subscriber_ref="Patient/sub-1")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["subscriber"] == {"reference": "Patient/sub-1"}

    def test_payor_refs_round_trip(self):
        """Payor references survive round-trip (strings → References)."""
        resource_in = _coverage(
            payor_refs=["Organization/ins-1", "Organization/ins-2"],
        )
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["payor"] == [
            {"reference": "Organization/ins-1"},
            {"reference": "Organization/ins-2"},
        ]

    def test_type_round_trip(self):
        """Coverage type text survives round-trip (string → CodeableConcept)."""
        resource_in = _coverage(type_text="Medicare Part A")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["type"] == {"text": "Medicare Part A"}

    def test_relationship_round_trip(self):
        """Relationship code survives round-trip (string → CodeableConcept)."""
        resource_in = _coverage(relationship_code="self")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["relationship"] == {"coding": [{"code": "self"}]}

    def test_dependent_round_trip(self):
        """Dependent string survives round-trip."""
        resource_in = _coverage(dependent="1")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["dependent"] == "1"

    def test_order_round_trip(self):
        """Coverage order survives round-trip."""
        resource_in = _coverage(order=2)
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["order"] == 2

    def test_network_round_trip(self):
        """Network string survives round-trip."""
        resource_in = _coverage(network="PPO")
        doc, _ = from_fhir(resource_in)
        resource_out, _ = to_fhir(doc)

        assert resource_out["network"] == "PPO"

    def test_full_resource_round_trip(self):
        """Fully populated Coverage survives round-trip."""
        resource_in = _coverage(
            status="active",
            period={"start": "2024-01-01", "end": "2024-12-31"},
            type_text="HMO",
            beneficiary_ref="Patient/p-1",
            subscriber_ref="Patient/sub-1",
            payor_refs=["Organization/ins-1"],
            dependent="0",
            relationship_code="self",
            order=1,
            network="PPO",
        )
        doc, _ = from_fhir(resource_in)
        resource_out, report = to_fhir(doc)

        assert report.success is True
        assert resource_out["resourceType"] == "Coverage"
        assert resource_out["status"] == "active"
        assert resource_out["period"] == {"start": "2024-01-01", "end": "2024-12-31"}
        assert resource_out["type"] == {"text": "HMO"}
        assert resource_out["beneficiary"] == {"reference": "Patient/p-1"}
        assert resource_out["subscriber"] == {"reference": "Patient/sub-1"}
        assert resource_out["payor"] == [{"reference": "Organization/ins-1"}]
        assert resource_out["dependent"] == "0"
        assert resource_out["relationship"] == {"coding": [{"code": "self"}]}
        assert resource_out["order"] == 1
        assert resource_out["network"] == "PPO"


# ═══════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestCoverageEdgeCases:
    """Defensive handling of malformed or incomplete resources."""

    def test_minimal_resource(self):
        """Minimal resource with only resourceType and status."""
        resource = {"resourceType": "Coverage", "status": "active"}
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc["id"] is None
        assert len(doc["opinions"]) == 1

    def test_missing_status(self):
        """Missing status field uses None as value, default opinion."""
        resource = {"resourceType": "Coverage", "id": "cov-x"}
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc["opinions"][0]["value"] is None

    def test_empty_payor_array(self):
        """Empty payor array → no payor_references in doc."""
        resource = _coverage()
        resource["payor"] = []
        doc, _ = from_fhir(resource)
        assert "payor_references" not in doc

    def test_payor_without_reference_key(self):
        """Payor objects missing 'reference' key are skipped."""
        resource = _coverage()
        resource["payor"] = [{"display": "Some Insurer"}]
        doc, _ = from_fhir(resource)
        assert "payor_references" not in doc

    def test_beneficiary_not_dict(self):
        """Non-dict beneficiary is ignored gracefully."""
        resource = _coverage()
        resource["beneficiary"] = "not-a-dict"
        doc, _ = from_fhir(resource)
        assert "beneficiary" not in doc

    def test_subscriber_not_dict(self):
        """Non-dict subscriber is ignored gracefully."""
        resource = _coverage()
        resource["subscriber"] = 42
        doc, _ = from_fhir(resource)
        assert "subscriber" not in doc

    def test_type_without_text(self):
        """Type CodeableConcept without text but with coding → first code."""
        resource = _coverage()
        resource["type"] = {"coding": [{"code": "HIP", "display": "Health"}]}
        doc, _ = from_fhir(resource)
        # Type text extraction: text preferred, coding[0].code fallback
        assert "type" in doc

    def test_relationship_empty_coding(self):
        """Relationship with empty coding → not present in doc."""
        resource = _coverage()
        resource["relationship"] = {"coding": []}
        doc, _ = from_fhir(resource)
        assert "relationship" not in doc

    def test_relationship_not_dict(self):
        """Non-dict relationship is ignored gracefully."""
        resource = _coverage()
        resource["relationship"] = "self"
        doc, _ = from_fhir(resource)
        assert "relationship" not in doc

    def test_to_fhir_missing_optional_fields(self):
        """to_fhir with minimal doc produces valid FHIR resource."""
        doc = {
            "@type": "fhir:Coverage",
            "id": "cov-min",
            "status": "active",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(0.70, 0.15, 0.15),
                "source": "reconstructed",
            }],
        }
        resource_out, report = to_fhir(doc)
        assert report.success is True
        assert resource_out["resourceType"] == "Coverage"
        assert resource_out["status"] == "active"
        assert "_status" in resource_out
        # Optional fields should not be present
        assert "period" not in resource_out
        assert "beneficiary" not in resource_out
        assert "subscriber" not in resource_out
        assert "payor" not in resource_out


# ═══════════════════════════════════════════════════════════════════
# 7. Integration — financial chain
# ═══════════════════════════════════════════════════════════════════


class TestCoverageFinancialChainIntegration:
    """Coverage integrates with existing Claim and EOB in the
    financial chain: Coverage → Claim → ExplanationOfBenefit."""

    def test_coverage_fusable_with_claim(self):
        """Coverage and Claim opinions can be fused."""
        from jsonld_ex.fhir_interop import fhir_clinical_fuse

        doc_cov, _ = from_fhir(_coverage(status="active"))
        doc_claim, _ = from_fhir({
            "resourceType": "Claim",
            "id": "claim-1",
            "status": "active",
        })

        fused, report = fhir_clinical_fuse([doc_cov, doc_claim])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_coverage_fusable_with_eob(self):
        """Coverage and ExplanationOfBenefit opinions can be fused."""
        from jsonld_ex.fhir_interop import fhir_clinical_fuse

        doc_cov, _ = from_fhir(_coverage(status="active"))
        doc_eob, _ = from_fhir({
            "resourceType": "ExplanationOfBenefit",
            "id": "eob-1",
            "status": "active",
            "outcome": "complete",
        })

        fused, report = fhir_clinical_fuse([doc_cov, doc_eob])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused >= 2

    def test_full_financial_chain_fusion(self):
        """Full financial chain: Coverage + Claim + EOB fuses correctly."""
        from jsonld_ex.fhir_interop import fhir_clinical_fuse

        doc_cov, _ = from_fhir(_coverage(status="active"))
        doc_claim, _ = from_fhir({
            "resourceType": "Claim",
            "id": "claim-1",
            "status": "active",
        })
        doc_eob, _ = from_fhir({
            "resourceType": "ExplanationOfBenefit",
            "id": "eob-1",
            "status": "active",
            "outcome": "complete",
        })

        fused, report = fhir_clinical_fuse([doc_cov, doc_claim, doc_eob])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused >= 3
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9
