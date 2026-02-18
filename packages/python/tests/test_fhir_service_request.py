"""Tests for Phase 7A: ServiceRequest FHIR R4 resource support.

RED PHASE: Tests written before implementation.

ServiceRequest completes the diagnostic chain:
    ServiceRequest → DiagnosticReport → Observation

It is the highest-priority expansion type (Score 44/50) because it
closes the workflow gap between clinical ordering and results.

Proposition under assessment:
    "This service request is valid and should be acted upon."

Epistemic signals extracted:
    - status: draft/active/on-hold/revoked/completed/entered-in-error/unknown
    - intent: proposal/plan/directive/order/original-order/reflex-order/
              filler-order/instance-order/option
    - priority: routine/urgent/asap/stat
    - reasonReference[] count: evidence basis for ordering

This is a CUSTOM handler (not a simple status handler) because:
    1. Intent modulates uncertainty (proposal ≠ order)
    2. Priority modulates uncertainty (stat ≠ routine)
    3. reasonReference count provides evidence basis
    4. These signals interact multiplicatively

Design decisions documented here for scientific rigor:

    Status probability mappings encode "validity and actionability":
        active=0.85   — order is live, high validity
        completed=0.90 — fulfilled, strongest signal
        draft=0.50     — pre-decisional, genuinely uncertain
        on-hold=0.50   — paused, uncertain continuation
        revoked=0.10   — explicitly cancelled, strong invalidity
        entered-in-error=0.50 — data integrity compromised
        unknown=0.50   — maximum ignorance

    Intent uncertainty multipliers encode epistemic weight:
        order/original-order/instance-order=0.7 — formal orders carry
            strongest evidence weight (deliberate clinical decision)
        directive/filler-order=0.8 — protocol-driven or fulfillment,
            slightly less deliberate than direct orders
        reflex-order=0.9 — automated response to another result,
            less clinical deliberation
        plan=1.2 — planned but not yet ordered, moderate uncertainty
        proposal=1.4 — weakest: suggestion only
        option=1.5 — informational, no commitment

    Priority uncertainty multipliers encode clinical attention:
        routine=1.0 — baseline
        urgent=0.85 — increased attention reduces uncertainty
        asap=0.75   — significant attention
        stat=0.65   — maximum clinical focus, immediate action

    reasonReference count follows existing basis_count pattern:
        0 reasons   → u *= 1.3 (no documented rationale)
        1–2 reasons → u *= 0.9 (some evidence)
        3+ reasons  → u *= 0.7 (strong evidence basis)

References:
    - HL7 FHIR R4 ServiceRequest: https://hl7.org/fhir/R4/servicerequest.html
    - Jøsang, A. (2016). Subjective Logic. Springer.
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
    SERVICE_REQUEST_STATUS_PROBABILITY,
    SERVICE_REQUEST_STATUS_UNCERTAINTY,
    SERVICE_REQUEST_INTENT_MULTIPLIER,
    SERVICE_REQUEST_PRIORITY_MULTIPLIER,
)


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════


def _service_request(
    *,
    status="active",
    intent="order",
    priority=None,
    reason_references=None,
    id_="sr-1",
):
    """Build a minimal valid FHIR R4 ServiceRequest."""
    resource = {
        "resourceType": "ServiceRequest",
        "id": id_,
        "status": status,
        "intent": intent,
        "code": {
            "coding": [{"system": "http://loinc.org", "code": "24323-8"}],
            "text": "Comprehensive metabolic panel",
        },
        "subject": {"reference": "Patient/p1"},
    }
    if priority is not None:
        resource["priority"] = priority
    if reason_references is not None:
        resource["reasonReference"] = [
            {"reference": ref} for ref in reason_references
        ]
    return resource


# ═══════════════════════════════════════════════════════════════════
# Registration: ServiceRequest in SUPPORTED_RESOURCE_TYPES
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestRegistration:
    """ServiceRequest must be registered as a supported type."""

    def test_in_supported_types(self):
        assert "ServiceRequest" in SUPPORTED_RESOURCE_TYPES


# ═══════════════════════════════════════════════════════════════════
# Constants: Mapping tables exist and are well-formed
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestConstants:
    """Verify constant mapping tables exist and have valid values."""

    # ── Status probability map ────────────────────────────────────

    def test_status_probability_covers_all_fhir_codes(self):
        """All FHIR R4 ServiceRequest.status codes must be mapped."""
        expected = {
            "draft", "active", "on-hold", "revoked",
            "completed", "entered-in-error", "unknown",
        }
        assert set(SERVICE_REQUEST_STATUS_PROBABILITY.keys()) == expected

    def test_status_probability_values_in_range(self):
        for code, prob in SERVICE_REQUEST_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, f"status '{code}' prob={prob} out of range"

    def test_status_uncertainty_covers_all_fhir_codes(self):
        expected = {
            "draft", "active", "on-hold", "revoked",
            "completed", "entered-in-error", "unknown",
        }
        assert set(SERVICE_REQUEST_STATUS_UNCERTAINTY.keys()) == expected

    def test_status_uncertainty_values_in_range(self):
        for code, u in SERVICE_REQUEST_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, f"status '{code}' uncertainty={u} out of range"

    # ── Intent multiplier map ─────────────────────────────────────

    def test_intent_multiplier_covers_all_fhir_codes(self):
        """All FHIR R4 ServiceRequest.intent codes must be mapped."""
        expected = {
            "proposal", "plan", "directive", "order",
            "original-order", "reflex-order", "filler-order",
            "instance-order", "option",
        }
        assert set(SERVICE_REQUEST_INTENT_MULTIPLIER.keys()) == expected

    def test_intent_multiplier_values_positive(self):
        for code, mult in SERVICE_REQUEST_INTENT_MULTIPLIER.items():
            assert mult > 0.0, f"intent '{code}' multiplier={mult} must be positive"

    def test_intent_order_strongest(self):
        """Formal orders should have the lowest uncertainty multiplier."""
        order_mult = SERVICE_REQUEST_INTENT_MULTIPLIER["order"]
        proposal_mult = SERVICE_REQUEST_INTENT_MULTIPLIER["proposal"]
        assert order_mult < proposal_mult, (
            "Order intent should produce less uncertainty than proposal"
        )

    def test_intent_proposal_weakest(self):
        """Proposal should have the highest (or near-highest) multiplier."""
        proposal_mult = SERVICE_REQUEST_INTENT_MULTIPLIER["proposal"]
        for code, mult in SERVICE_REQUEST_INTENT_MULTIPLIER.items():
            if code == "option":
                continue  # option may be even weaker
            assert mult <= proposal_mult, (
                f"intent '{code}' multiplier {mult} > proposal {proposal_mult}"
            )

    # ── Priority multiplier map ───────────────────────────────────

    def test_priority_multiplier_covers_all_fhir_codes(self):
        """All FHIR R4 ServiceRequest.priority codes must be mapped."""
        expected = {"routine", "urgent", "asap", "stat"}
        assert set(SERVICE_REQUEST_PRIORITY_MULTIPLIER.keys()) == expected

    def test_priority_multiplier_values_positive(self):
        for code, mult in SERVICE_REQUEST_PRIORITY_MULTIPLIER.items():
            assert mult > 0.0, f"priority '{code}' multiplier={mult} must be positive"

    def test_priority_routine_is_baseline(self):
        """Routine priority should be the baseline (multiplier = 1.0)."""
        assert SERVICE_REQUEST_PRIORITY_MULTIPLIER["routine"] == 1.0

    def test_priority_stat_lowest_multiplier(self):
        """Stat (emergency) should have the lowest uncertainty multiplier."""
        stat_mult = SERVICE_REQUEST_PRIORITY_MULTIPLIER["stat"]
        for code, mult in SERVICE_REQUEST_PRIORITY_MULTIPLIER.items():
            assert stat_mult <= mult, (
                f"stat multiplier {stat_mult} > {code} multiplier {mult}"
            )

    def test_priority_monotonically_decreasing(self):
        """Uncertainty multiplier should decrease: routine > urgent > asap > stat."""
        m = SERVICE_REQUEST_PRIORITY_MULTIPLIER
        assert m["routine"] > m["urgent"] > m["asap"] > m["stat"]

    # ── Epistemic ordering within status map ──────────────────────

    def test_completed_higher_prob_than_draft(self):
        assert (
            SERVICE_REQUEST_STATUS_PROBABILITY["completed"]
            > SERVICE_REQUEST_STATUS_PROBABILITY["draft"]
        )

    def test_active_higher_prob_than_revoked(self):
        assert (
            SERVICE_REQUEST_STATUS_PROBABILITY["active"]
            > SERVICE_REQUEST_STATUS_PROBABILITY["revoked"]
        )

    def test_entered_in_error_high_uncertainty(self):
        assert SERVICE_REQUEST_STATUS_UNCERTAINTY["entered-in-error"] >= 0.50


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Basic status-based conversion
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestFromFhir:
    """Test from_fhir() for ServiceRequest resources."""

    def test_basic_active_order(self):
        """Active order should produce a valid opinion."""
        doc, report = from_fhir(_service_request(status="active", intent="order"))
        assert report.success
        assert report.nodes_converted >= 1
        assert doc["@type"] == "fhir:ServiceRequest"
        assert doc["id"] == "sr-1"
        assert doc["status"] == "active"
        assert len(doc["opinions"]) >= 1

    def test_opinion_is_valid_sl_triple(self):
        """Opinion must satisfy b + d + u = 1, all in [0, 1]."""
        doc, _ = from_fhir(_service_request())
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert 0.0 <= op.belief <= 1.0
        assert 0.0 <= op.disbelief <= 1.0
        assert 0.0 <= op.uncertainty <= 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    @pytest.mark.parametrize("status", [
        "draft", "active", "on-hold", "revoked",
        "completed", "entered-in-error", "unknown",
    ])
    def test_all_status_codes_produce_valid_opinion(self, status):
        """Every FHIR R4 ServiceRequest.status code must produce a valid opinion."""
        doc, report = from_fhir(_service_request(status=status))
        assert report.success
        assert len(doc["opinions"]) >= 1
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_status_field_preserved(self):
        doc, _ = from_fhir(_service_request(status="completed"))
        assert doc["status"] == "completed"

    def test_opinion_source_is_reconstructed(self):
        """Without extension, source should be 'reconstructed'."""
        doc, _ = from_fhir(_service_request())
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_opinion_field_is_status(self):
        """Primary opinion should be on the 'status' field."""
        doc, _ = from_fhir(_service_request())
        assert doc["opinions"][0]["field"] == "status"

    def test_completed_higher_belief_than_draft(self):
        """Completed orders should have higher belief than drafts."""
        doc_completed, _ = from_fhir(
            _service_request(status="completed", intent="order")
        )
        doc_draft, _ = from_fhir(
            _service_request(status="draft", intent="order")
        )
        b_completed = doc_completed["opinions"][0]["opinion"].belief
        b_draft = doc_draft["opinions"][0]["opinion"].belief
        assert b_completed > b_draft

    def test_revoked_low_belief(self):
        """Revoked orders should have low belief."""
        doc, _ = from_fhir(_service_request(status="revoked", intent="order"))
        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.30

    def test_entered_in_error_high_uncertainty(self):
        """Entered-in-error should have elevated uncertainty."""
        doc, _ = from_fhir(
            _service_request(status="entered-in-error", intent="order")
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty >= 0.35

    def test_unknown_status_falls_back(self):
        """Unknown/unmapped status should produce a default opinion."""
        doc, report = from_fhir(
            _service_request(status="unknown", intent="order")
        )
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Intent modulation
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestIntentModulation:
    """Intent should modulate uncertainty: order < plan < proposal."""

    def test_order_less_uncertain_than_proposal(self):
        """A formal order should have less uncertainty than a proposal."""
        doc_order, _ = from_fhir(
            _service_request(status="active", intent="order")
        )
        doc_proposal, _ = from_fhir(
            _service_request(status="active", intent="proposal")
        )
        u_order = doc_order["opinions"][0]["opinion"].uncertainty
        u_proposal = doc_proposal["opinions"][0]["opinion"].uncertainty
        assert u_order < u_proposal

    def test_plan_between_order_and_proposal(self):
        """Plan intent should fall between order and proposal in uncertainty."""
        doc_order, _ = from_fhir(
            _service_request(status="active", intent="order")
        )
        doc_plan, _ = from_fhir(
            _service_request(status="active", intent="plan")
        )
        doc_proposal, _ = from_fhir(
            _service_request(status="active", intent="proposal")
        )
        u_order = doc_order["opinions"][0]["opinion"].uncertainty
        u_plan = doc_plan["opinions"][0]["opinion"].uncertainty
        u_proposal = doc_proposal["opinions"][0]["opinion"].uncertainty
        assert u_order < u_plan < u_proposal

    @pytest.mark.parametrize("intent", [
        "proposal", "plan", "directive", "order",
        "original-order", "reflex-order", "filler-order",
        "instance-order", "option",
    ])
    def test_all_intent_codes_produce_valid_opinion(self, intent):
        """Every FHIR R4 ServiceRequest.intent code must produce a valid opinion."""
        doc, report = from_fhir(_service_request(intent=intent))
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_original_order_same_as_order(self):
        """original-order should have the same multiplier as order."""
        doc_order, _ = from_fhir(
            _service_request(status="active", intent="order")
        )
        doc_orig, _ = from_fhir(
            _service_request(status="active", intent="original-order")
        )
        u_order = doc_order["opinions"][0]["opinion"].uncertainty
        u_orig = doc_orig["opinions"][0]["opinion"].uncertainty
        assert abs(u_order - u_orig) < 1e-9

    def test_unknown_intent_uses_baseline(self):
        """An unrecognised intent code should use multiplier 1.0 (baseline)."""
        doc_baseline, _ = from_fhir(
            _service_request(status="active", intent="order")
        )
        # Unknown intent: no multiplier advantage
        resource = _service_request(status="active", intent="order")
        resource["intent"] = "some-future-code"
        doc_unknown, _ = from_fhir(resource)

        u_baseline = doc_baseline["opinions"][0]["opinion"].uncertainty
        u_unknown = doc_unknown["opinions"][0]["opinion"].uncertainty
        # Unknown should have higher uncertainty than a formal order
        assert u_unknown >= u_baseline


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Priority modulation
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestPriorityModulation:
    """Priority should modulate uncertainty: stat < asap < urgent < routine."""

    def test_stat_less_uncertain_than_routine(self):
        """Stat priority should produce less uncertainty than routine."""
        doc_stat, _ = from_fhir(
            _service_request(status="active", intent="order", priority="stat")
        )
        doc_routine, _ = from_fhir(
            _service_request(status="active", intent="order", priority="routine")
        )
        u_stat = doc_stat["opinions"][0]["opinion"].uncertainty
        u_routine = doc_routine["opinions"][0]["opinion"].uncertainty
        assert u_stat < u_routine

    def test_priority_monotonic_ordering(self):
        """Uncertainty should decrease: routine > urgent > asap > stat."""
        results = {}
        for prio in ["routine", "urgent", "asap", "stat"]:
            doc, _ = from_fhir(
                _service_request(status="active", intent="order", priority=prio)
            )
            results[prio] = doc["opinions"][0]["opinion"].uncertainty

        assert results["routine"] > results["urgent"]
        assert results["urgent"] > results["asap"]
        assert results["asap"] > results["stat"]

    def test_no_priority_uses_baseline(self):
        """Omitting priority should use baseline (same as routine or no modifier)."""
        doc_none, _ = from_fhir(
            _service_request(status="active", intent="order", priority=None)
        )
        doc_routine, _ = from_fhir(
            _service_request(status="active", intent="order", priority="routine")
        )
        u_none = doc_none["opinions"][0]["opinion"].uncertainty
        u_routine = doc_routine["opinions"][0]["opinion"].uncertainty
        assert abs(u_none - u_routine) < 1e-9

    @pytest.mark.parametrize("priority", ["routine", "urgent", "asap", "stat"])
    def test_all_priority_codes_produce_valid_opinion(self, priority):
        doc, report = from_fhir(
            _service_request(status="active", intent="order", priority=priority)
        )
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): reasonReference evidence basis
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestReasonReference:
    """reasonReference count modulates uncertainty as evidence basis."""

    def test_no_reasons_increases_uncertainty(self):
        """No reasonReferences → higher uncertainty than with reasons."""
        doc_no_reason, _ = from_fhir(
            _service_request(status="active", intent="order", reason_references=None)
        )
        doc_reasons, _ = from_fhir(
            _service_request(
                status="active", intent="order",
                reason_references=["Condition/c1", "Condition/c2", "Condition/c3"],
            )
        )
        u_none = doc_no_reason["opinions"][0]["opinion"].uncertainty
        u_many = doc_reasons["opinions"][0]["opinion"].uncertainty
        assert u_none > u_many

    def test_empty_reasons_list_increases_uncertainty(self):
        """Empty reasonReference list should increase uncertainty."""
        doc_empty, _ = from_fhir(
            _service_request(status="active", intent="order", reason_references=[])
        )
        doc_some, _ = from_fhir(
            _service_request(
                status="active", intent="order",
                reason_references=["Condition/c1", "Condition/c2"],
            )
        )
        u_empty = doc_empty["opinions"][0]["opinion"].uncertainty
        u_some = doc_some["opinions"][0]["opinion"].uncertainty
        assert u_empty > u_some

    def test_many_reasons_decreases_uncertainty(self):
        """3+ reasonReferences → significantly lower uncertainty."""
        doc_one, _ = from_fhir(
            _service_request(
                status="active", intent="order",
                reason_references=["Condition/c1"],
            )
        )
        doc_many, _ = from_fhir(
            _service_request(
                status="active", intent="order",
                reason_references=["Condition/c1", "Condition/c2", "Condition/c3"],
            )
        )
        u_one = doc_one["opinions"][0]["opinion"].uncertainty
        u_many = doc_many["opinions"][0]["opinion"].uncertainty
        assert u_many < u_one


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Signal interactions
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestSignalInteractions:
    """Test multiplicative interaction of status × intent × priority × evidence."""

    def test_best_case_lowest_uncertainty(self):
        """completed + order + stat + many reasons → minimum uncertainty."""
        doc, _ = from_fhir(
            _service_request(
                status="completed",
                intent="order",
                priority="stat",
                reason_references=["Condition/c1", "Condition/c2", "Condition/c3"],
            )
        )
        op = doc["opinions"][0]["opinion"]
        # Should have very low uncertainty
        assert op.uncertainty < 0.10

    def test_worst_case_highest_uncertainty(self):
        """draft + proposal + no priority + no reasons → high uncertainty."""
        doc, _ = from_fhir(
            _service_request(
                status="draft",
                intent="proposal",
                priority=None,
                reason_references=[],
            )
        )
        op = doc["opinions"][0]["opinion"]
        # Should have significantly elevated uncertainty
        assert op.uncertainty > 0.30

    def test_uncertainty_always_clamped_below_one(self):
        """Even worst-case signal combination must clamp u < 1.0."""
        doc, _ = from_fhir(
            _service_request(
                status="entered-in-error",
                intent="option",
                priority=None,
                reason_references=[],
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_uncertainty_always_positive(self):
        """Even best-case signal combination must have u > 0."""
        doc, _ = from_fhir(
            _service_request(
                status="completed",
                intent="order",
                priority="stat",
                reason_references=[f"Condition/c{i}" for i in range(10)],
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.0


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Extension recovery (exact round-trip)
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestExtensionRecovery:
    """When a jsonld-ex FHIR extension is present, recover the exact opinion."""

    def test_extension_recovery_overrides_reconstruction(self):
        """An embedded extension should be recovered exactly, not reconstructed."""
        original = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        resource = _service_request(status="active", intent="order")
        resource["_status"] = {"extension": [ext]}

        doc, report = from_fhir(resource)
        assert report.success
        recovered = doc["opinions"][0]["opinion"]
        assert doc["opinions"][0]["source"] == "extension"
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9

    def test_extension_ignores_metadata_signals(self):
        """When extension is present, intent/priority/reasons don't affect opinion."""
        original = Opinion(belief=0.30, disbelief=0.50, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        resource = _service_request(
            status="completed",
            intent="order",
            priority="stat",
            reason_references=["Condition/c1", "Condition/c2", "Condition/c3"],
        )
        resource["_status"] = {"extension": [ext]}

        doc, _ = from_fhir(resource)
        recovered = doc["opinions"][0]["opinion"]
        # Should match original exactly, not be affected by favorable signals
        assert abs(recovered.belief - 0.30) < 1e-9
        assert abs(recovered.disbelief - 0.50) < 1e-9

    def test_wrong_extension_url_triggers_reconstruction(self):
        """An extension with a different URL should not be recovered."""
        resource = _service_request(status="active", intent="order")
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


class TestServiceRequestMetadataPassthrough:
    """Key metadata fields should be preserved in the jsonld-ex document."""

    def test_intent_preserved(self):
        doc, _ = from_fhir(_service_request(intent="plan"))
        assert doc["intent"] == "plan"

    def test_priority_preserved(self):
        doc, _ = from_fhir(_service_request(priority="urgent"))
        assert doc["priority"] == "urgent"

    def test_priority_absent_when_not_set(self):
        doc, _ = from_fhir(_service_request(priority=None))
        assert "priority" not in doc

    def test_authored_on_preserved(self):
        resource = _service_request()
        resource["authoredOn"] = "2024-03-15T10:00:00Z"
        doc, _ = from_fhir(resource)
        assert doc["authoredOn"] == "2024-03-15T10:00:00Z"

    def test_requester_reference_preserved(self):
        resource = _service_request()
        resource["requester"] = {"reference": "Practitioner/dr-smith"}
        doc, _ = from_fhir(resource)
        assert doc["requester"] == "Practitioner/dr-smith"

    def test_reason_reference_refs_preserved(self):
        """reasonReference references should be extractable."""
        doc, _ = from_fhir(
            _service_request(
                reason_references=["Condition/c1", "Observation/o1"]
            )
        )
        assert "reason_references" in doc
        assert "Condition/c1" in doc["reason_references"]
        assert "Observation/o1" in doc["reason_references"]


# ═══════════════════════════════════════════════════════════════════
# to_fhir(): Export to FHIR R4
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestToFhir:
    """Test to_fhir() export for ServiceRequest."""

    def test_basic_export(self):
        """A jsonld-ex ServiceRequest should export to a valid FHIR resource."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)
        assert report.success
        assert resource["resourceType"] == "ServiceRequest"
        assert resource["id"] == "sr-1"
        assert resource["status"] == "active"
        assert resource["intent"] == "order"

    def test_opinion_embedded_as_extension(self):
        """SL opinion should be embedded as a FHIR extension on _status."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert "_status" in resource
        extensions = resource["_status"]["extension"]
        assert len(extensions) >= 1
        assert extensions[0]["url"] == FHIR_EXTENSION_URL

    def test_intent_exported(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "plan",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["intent"] == "plan"

    def test_priority_exported_when_present(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "priority": "stat",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["priority"] == "stat"

    def test_priority_absent_when_not_in_doc(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert "priority" not in resource

    def test_authored_on_exported(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "authoredOn": "2024-03-15T10:00:00Z",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["authoredOn"] == "2024-03-15T10:00:00Z"

    def test_requester_exported_as_reference(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "requester": "Practitioner/dr-smith",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource["requester"] == {"reference": "Practitioner/dr-smith"}

    def test_reason_references_exported(self):
        doc = {
            "@type": "fhir:ServiceRequest",
            "id": "sr-1",
            "status": "active",
            "intent": "order",
            "reason_references": ["Condition/c1", "Observation/o1"],
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert "reasonReference" in resource
        refs = [r["reference"] for r in resource["reasonReference"]]
        assert "Condition/c1" in refs
        assert "Observation/o1" in refs


# ═══════════════════════════════════════════════════════════════════
# Round-trip fidelity: from_fhir() → to_fhir() → from_fhir()
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestRoundTrip:
    """Round-trip through from_fhir → to_fhir should preserve opinions exactly."""

    def test_opinion_preserved_via_extension(self):
        """After from_fhir → to_fhir → from_fhir, opinion should be exact."""
        original = _service_request(
            status="active",
            intent="order",
            priority="urgent",
            reason_references=["Condition/c1"],
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
        original = _service_request()
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["resourceType"] == "ServiceRequest"

    def test_id_preserved(self):
        original = _service_request(id_="sr-roundtrip-42")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["id"] == "sr-roundtrip-42"

    def test_status_preserved(self):
        original = _service_request(status="completed")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["status"] == "completed"

    def test_intent_preserved_through_round_trip(self):
        original = _service_request(intent="reflex-order")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["intent"] == "reflex-order"

    def test_priority_preserved_through_round_trip(self):
        original = _service_request(priority="asap")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["priority"] == "asap"


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestServiceRequestEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_missing_id(self):
        """Resource without id should not crash."""
        resource = _service_request()
        del resource["id"]
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["id"] is None

    def test_missing_intent_uses_baseline(self):
        """If intent is absent (shouldn't happen in valid FHIR, but defensive)."""
        resource = _service_request()
        del resource["intent"]
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_missing_status_uses_default(self):
        """If status is absent, should use a safe default."""
        resource = _service_request()
        del resource["status"]
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_novel_status_code_handled(self):
        """A future/unknown status code should not crash."""
        resource = _service_request()
        resource["status"] = "future-status-code"
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_novel_intent_code_handled(self):
        """A future/unknown intent code should not crash."""
        resource = _service_request()
        resource["intent"] = "future-intent-code"
        doc, report = from_fhir(resource)
        assert report.success

    def test_novel_priority_code_handled(self):
        """A future/unknown priority code should not crash."""
        resource = _service_request(priority="future-priority-code")
        doc, report = from_fhir(resource)
        assert report.success
