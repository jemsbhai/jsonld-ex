"""Tests for Phase 7A: Specimen FHIR R4 resource support.

RED PHASE: Tests written before implementation.

Specimen links the diagnostic chain to pre-analytical quality:
    ServiceRequest → **Specimen** → DiagnosticReport → Observation

It enables quality-adjusted opinion propagation: a hemolyzed blood
sample reduces confidence in downstream Observation results, which
is invisible in standard FHIR but captured by SL opinions.

Proposition under assessment:
    "This specimen is suitable for reliable diagnostic analysis."

Epistemic signals extracted (5 multiplicative signals):
    1. status: available | unavailable | unsatisfactory | entered-in-error
    2. condition[]: quality degradation codes (v2 table 0493)
    3. processing[] count: chain of custody documentation
    4. collection.collector presence: accountability (identified collector)
    5. collection.quantity presence: sample adequacy documentation

This is a CUSTOM handler (not a simple status handler) because:
    1. Condition codes compound multiplicatively — each degradation
       flag independently raises uncertainty
    2. Pre-analytical errors cause ~70% of lab testing errors, making
       condition the most impactful signal
    3. Processing chain length, collector identity, and quantity presence
       interact with status to produce genuinely different epistemic
       semantics from simple status-only resources

Design decisions documented here for scientific rigor:

    Status probability mappings encode specimen suitability:
        available=0.85     — specimen ready for testing, high suitability
        unavailable=0.30   — not available, low suitability
        unsatisfactory=0.15 — explicitly quality-failed
        entered-in-error=0.50 — data integrity compromised

    Status uncertainty mappings:
        available=0.15     — low uncertainty, known good state
        unavailable=0.40   — moderate-high, may become available
        unsatisfactory=0.25 — low uncertainty about poor quality
                              (we're quite certain it's bad)
        entered-in-error=0.50 — maximum uncertainty, data unreliable

    Condition degradation multipliers (each compounds independently):
        HEM (hemolyzed)=1.4    — hemoglobin interference affects many
                                  analytes; most common pre-analytical error
        CLOT (clotted)=1.5     — unusable for many coagulation tests
        CON (contaminated)=1.5 — compromised specimen integrity
        AUT (autolyzed)=1.6    — tissue breakdown, severe degradation
        SNR (sample not received)=2.0 — no specimen to analyze
        LIVE=0.9               — live specimen, slight positive signal
        COOL=1.0               — proper cold-chain, neutral
        FROZ=1.0               — frozen state, neutral (test-dependent)
        ROOM=1.1               — room temperature may indicate improper
                                  handling for temperature-sensitive tests
        CFU (centrifuged)=0.9  — processed, slight positive signal

    Processing chain count (documentation completeness):
        0 steps   → u *= 1.3  (no documented processing — unknown handling)
        1-2 steps → u *= 1.0  (baseline documentation)
        3+ steps  → u *= 0.8  (thorough chain of custody)

    Collector identification (accountability signal):
        collector present → u *= 0.9  (identifiable responsible person)
        collector absent  → u *= 1.1  (anonymous collection)

    Collection quantity (documentation signal):
        quantity present → u *= 0.9   (sample volume/amount documented)
        quantity absent  → u *= 1.0   (baseline — no information)

    Note on quantity: We use presence/absence rather than threshold-based
    adequacy because adequacy thresholds are test-specific (e.g., 3 mL
    for CBC vs 10 mL for blood culture). Presence of documentation is
    itself an epistemic signal about collection quality.

References:
    - HL7 FHIR R4 Specimen: https://hl7.org/fhir/R4/specimen.html
    - HL7 v2 Table 0493 (Specimen Condition): http://hl7.org/fhir/R4/v2/0493/
    - Pre-analytical error rates: Plebani (2006), Clin Chem Lab Med
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
    SPECIMEN_STATUS_PROBABILITY,
    SPECIMEN_STATUS_UNCERTAINTY,
    SPECIMEN_CONDITION_MULTIPLIER,
)


# ═══════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════

def _make_specimen(
    *,
    specimen_id: str = "spec-001",
    status: str = "available",
    condition_codes: list[str] | None = None,
    processing_count: int = 0,
    collector_ref: str | None = None,
    quantity_value: float | None = None,
    quantity_unit: str = "mL",
    type_text: str | None = None,
    subject_ref: str | None = None,
    received_time: str | None = None,
    request_refs: list[str] | None = None,
    collected_dt: str | None = None,
    method_text: str | None = None,
    body_site_text: str | None = None,
    note_texts: list[str] | None = None,
    parent_refs: list[str] | None = None,
    status_extension: dict | None = None,
) -> dict:
    """Build a minimal FHIR R4 Specimen resource for testing."""
    resource: dict = {
        "resourceType": "Specimen",
        "id": specimen_id,
        "status": status,
    }

    # Condition codes (0..*)
    if condition_codes is not None:
        resource["condition"] = [
            {"coding": [{"code": code}]} for code in condition_codes
        ]

    # Processing steps (0..*)
    if processing_count > 0:
        resource["processing"] = [
            {"description": f"Step {i + 1}"} for i in range(processing_count)
        ]

    # Collection backbone
    collection: dict = {}
    if collector_ref is not None:
        collection["collector"] = {"reference": collector_ref}
    if quantity_value is not None:
        collection["quantity"] = {"value": quantity_value, "unit": quantity_unit}
    if collected_dt is not None:
        collection["collectedDateTime"] = collected_dt
    if method_text is not None:
        collection["method"] = {"text": method_text}
    if body_site_text is not None:
        collection["bodySite"] = {"text": body_site_text}
    if collection:
        resource["collection"] = collection

    # Type
    if type_text is not None:
        resource["type"] = {"text": type_text}

    # Subject
    if subject_ref is not None:
        resource["subject"] = {"reference": subject_ref}

    # ReceivedTime
    if received_time is not None:
        resource["receivedTime"] = received_time

    # Request references (links to ServiceRequest)
    if request_refs is not None:
        resource["request"] = [{"reference": r} for r in request_refs]

    # Parent specimens
    if parent_refs is not None:
        resource["parent"] = [{"reference": r} for r in parent_refs]

    # Notes
    if note_texts is not None:
        resource["note"] = [{"text": t} for t in note_texts]

    # Extension for exact opinion recovery
    if status_extension is not None:
        resource["_status"] = status_extension

    return resource


def _get_opinion(doc: dict) -> Opinion:
    """Extract the first opinion from a jsonld-ex document."""
    opinions = doc.get("opinions", [])
    assert len(opinions) >= 1, "Expected at least one opinion"
    return opinions[0]["opinion"]


# ═══════════════════════════════════════════════════════════════════
#  1. Registration & dispatch
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenRegistration:
    """Verify Specimen is registered in all dispatch infrastructure."""

    def test_in_supported_resource_types(self):
        assert "Specimen" in SUPPORTED_RESOURCE_TYPES

    def test_from_fhir_dispatches(self):
        resource = _make_specimen()
        doc, report = from_fhir(resource)
        assert doc["@type"] == "fhir:Specimen"
        assert report.success is True

    def test_to_fhir_dispatches(self):
        resource = _make_specimen()
        doc, _ = from_fhir(resource)
        fhir_out, report = to_fhir(doc)
        assert fhir_out["resourceType"] == "Specimen"
        assert report.success is True


# ═══════════════════════════════════════════════════════════════════
#  2. Constants validation
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenConstants:
    """Verify constant tables are complete and well-formed."""

    def test_status_probability_covers_all_codes(self):
        """All FHIR R4 SpecimenStatus codes must be mapped."""
        expected = {"available", "unavailable", "unsatisfactory", "entered-in-error"}
        assert expected == set(SPECIMEN_STATUS_PROBABILITY.keys())

    def test_status_uncertainty_covers_all_codes(self):
        expected = {"available", "unavailable", "unsatisfactory", "entered-in-error"}
        assert expected == set(SPECIMEN_STATUS_UNCERTAINTY.keys())

    def test_status_probabilities_in_valid_range(self):
        for code, prob in SPECIMEN_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, f"{code}: {prob}"

    def test_status_uncertainties_in_valid_range(self):
        for code, u in SPECIMEN_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, f"{code}: {u}"

    def test_condition_multiplier_covers_v2_0493_codes(self):
        """All v2 Table 0493 specimen condition codes must be mapped."""
        expected = {"HEM", "CLOT", "CON", "AUT", "SNR", "LIVE", "COOL", "FROZ", "ROOM", "CFU"}
        assert expected == set(SPECIMEN_CONDITION_MULTIPLIER.keys())

    def test_condition_multipliers_are_positive(self):
        for code, mult in SPECIMEN_CONDITION_MULTIPLIER.items():
            assert mult > 0.0, f"{code}: {mult}"

    def test_degradation_codes_raise_uncertainty(self):
        """HEM, CLOT, CON, AUT, SNR must have multiplier > 1.0."""
        degradation_codes = {"HEM", "CLOT", "CON", "AUT", "SNR"}
        for code in degradation_codes:
            assert SPECIMEN_CONDITION_MULTIPLIER[code] > 1.0, (
                f"{code} should raise uncertainty (multiplier > 1.0)"
            )


# ═══════════════════════════════════════════════════════════════════
#  3. Status signal modulation
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenStatusSignal:
    """Test that status codes produce distinct epistemic states."""

    @pytest.mark.parametrize("status", [
        "available", "unavailable", "unsatisfactory", "entered-in-error",
    ])
    def test_all_status_codes_produce_valid_opinion(self, status):
        resource = _make_specimen(status=status)
        doc, report = from_fhir(resource)
        op = _get_opinion(doc)
        assert isinstance(op, Opinion)
        assert report.success is True
        assert report.nodes_converted >= 1

    def test_available_has_highest_probability(self):
        doc_avail, _ = from_fhir(_make_specimen(status="available"))
        doc_unsat, _ = from_fhir(_make_specimen(status="unsatisfactory"))
        op_avail = _get_opinion(doc_avail)
        op_unsat = _get_opinion(doc_unsat)
        assert op_avail.projected_probability() > op_unsat.projected_probability()

    def test_unsatisfactory_has_lower_probability_than_unavailable(self):
        """Unsatisfactory is a quality judgment; unavailable is logistic."""
        doc_unsat, _ = from_fhir(_make_specimen(status="unsatisfactory"))
        doc_unavail, _ = from_fhir(_make_specimen(status="unavailable"))
        op_unsat = _get_opinion(doc_unsat)
        op_unavail = _get_opinion(doc_unavail)
        assert op_unsat.projected_probability() < op_unavail.projected_probability()

    def test_available_has_lowest_uncertainty(self):
        doc_avail, _ = from_fhir(_make_specimen(status="available"))
        doc_unavail, _ = from_fhir(_make_specimen(status="unavailable"))
        op_avail = _get_opinion(doc_avail)
        op_unavail = _get_opinion(doc_unavail)
        assert op_avail.uncertainty < op_unavail.uncertainty

    def test_entered_in_error_has_high_uncertainty(self):
        doc_err, _ = from_fhir(_make_specimen(status="entered-in-error"))
        doc_avail, _ = from_fhir(_make_specimen(status="available"))
        op_err = _get_opinion(doc_err)
        op_avail = _get_opinion(doc_avail)
        assert op_err.uncertainty > op_avail.uncertainty

    def test_opinion_field_is_status(self):
        doc, _ = from_fhir(_make_specimen())
        assert doc["opinions"][0]["field"] == "status"
        assert doc["opinions"][0]["value"] == "available"
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_unknown_status_uses_fallback(self):
        """Unknown status codes should produce a valid opinion with defaults."""
        resource = _make_specimen()
        resource["status"] = "some-unknown-code"
        doc, report = from_fhir(resource)
        op = _get_opinion(doc)
        assert isinstance(op, Opinion)
        assert report.success is True


# ═══════════════════════════════════════════════════════════════════
#  4. Condition degradation signal
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenConditionSignal:
    """Test that condition codes modulate uncertainty."""

    def test_hemolyzed_raises_uncertainty(self):
        doc_clean, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_hem, _ = from_fhir(_make_specimen(condition_codes=["HEM"]))
        op_clean = _get_opinion(doc_clean)
        op_hem = _get_opinion(doc_hem)
        assert op_hem.uncertainty > op_clean.uncertainty

    def test_clotted_raises_uncertainty(self):
        doc_clean, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_clot, _ = from_fhir(_make_specimen(condition_codes=["CLOT"]))
        assert _get_opinion(doc_clot).uncertainty > _get_opinion(doc_clean).uncertainty

    def test_contaminated_raises_uncertainty(self):
        doc_clean, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_con, _ = from_fhir(_make_specimen(condition_codes=["CON"]))
        assert _get_opinion(doc_con).uncertainty > _get_opinion(doc_clean).uncertainty

    def test_autolyzed_raises_uncertainty(self):
        doc_clean, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_aut, _ = from_fhir(_make_specimen(condition_codes=["AUT"]))
        assert _get_opinion(doc_aut).uncertainty > _get_opinion(doc_clean).uncertainty

    def test_snr_raises_uncertainty_most(self):
        """Sample not received should raise uncertainty the most."""
        doc_snr, _ = from_fhir(_make_specimen(condition_codes=["SNR"]))
        doc_hem, _ = from_fhir(_make_specimen(condition_codes=["HEM"]))
        assert _get_opinion(doc_snr).uncertainty > _get_opinion(doc_hem).uncertainty

    def test_multiple_conditions_compound(self):
        """Multiple degradation codes should compound multiplicatively."""
        doc_one, _ = from_fhir(_make_specimen(condition_codes=["HEM"]))
        doc_two, _ = from_fhir(_make_specimen(condition_codes=["HEM", "CLOT"]))
        assert _get_opinion(doc_two).uncertainty > _get_opinion(doc_one).uncertainty

    def test_three_conditions_compound_further(self):
        doc_two, _ = from_fhir(_make_specimen(condition_codes=["HEM", "CLOT"]))
        doc_three, _ = from_fhir(_make_specimen(
            condition_codes=["HEM", "CLOT", "CON"]
        ))
        assert _get_opinion(doc_three).uncertainty > _get_opinion(doc_two).uncertainty

    def test_live_reduces_uncertainty(self):
        """LIVE specimen is a positive signal (multiplier < 1.0)."""
        doc_none, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_live, _ = from_fhir(_make_specimen(condition_codes=["LIVE"]))
        assert _get_opinion(doc_live).uncertainty < _get_opinion(doc_none).uncertainty

    def test_neutral_conditions_dont_change_much(self):
        """COOL and FROZ should have multiplier == 1.0 (neutral)."""
        doc_none, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_cool, _ = from_fhir(_make_specimen(condition_codes=["COOL"]))
        op_none = _get_opinion(doc_none)
        op_cool = _get_opinion(doc_cool)
        # Allow tiny floating point differences
        assert abs(op_cool.uncertainty - op_none.uncertainty) < 0.01

    def test_empty_condition_list_is_baseline(self):
        """Empty condition list should not change uncertainty."""
        doc_none, _ = from_fhir(_make_specimen())  # no condition key
        doc_empty, _ = from_fhir(_make_specimen(condition_codes=[]))
        op_none = _get_opinion(doc_none)
        op_empty = _get_opinion(doc_empty)
        assert abs(op_none.uncertainty - op_empty.uncertainty) < 0.001

    def test_unknown_condition_code_is_neutral(self):
        """Unrecognized condition codes should default to multiplier 1.0."""
        doc_none, _ = from_fhir(_make_specimen(condition_codes=[]))
        doc_unknown, _ = from_fhir(_make_specimen(condition_codes=["UNKNOWN_CODE"]))
        op_none = _get_opinion(doc_none)
        op_unknown = _get_opinion(doc_unknown)
        assert abs(op_none.uncertainty - op_unknown.uncertainty) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  5. Processing chain signal
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenProcessingSignal:
    """Test that processing step count modulates uncertainty."""

    def test_no_processing_raises_uncertainty(self):
        doc_none, _ = from_fhir(_make_specimen(processing_count=0))
        doc_one, _ = from_fhir(_make_specimen(processing_count=1))
        assert _get_opinion(doc_none).uncertainty > _get_opinion(doc_one).uncertainty

    def test_thorough_processing_reduces_uncertainty(self):
        doc_one, _ = from_fhir(_make_specimen(processing_count=1))
        doc_many, _ = from_fhir(_make_specimen(processing_count=4))
        assert _get_opinion(doc_many).uncertainty < _get_opinion(doc_one).uncertainty

    def test_moderate_processing_is_baseline(self):
        """1-2 processing steps should use baseline multiplier (1.0)."""
        doc_one, _ = from_fhir(_make_specimen(processing_count=1))
        doc_two, _ = from_fhir(_make_specimen(processing_count=2))
        op_one = _get_opinion(doc_one)
        op_two = _get_opinion(doc_two)
        # Both in baseline band, should be very close
        assert abs(op_one.uncertainty - op_two.uncertainty) < 0.01


# ═══════════════════════════════════════════════════════════════════
#  6. Collector identification signal
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenCollectorSignal:
    """Test that collector presence modulates uncertainty."""

    def test_identified_collector_reduces_uncertainty(self):
        doc_no_coll, _ = from_fhir(_make_specimen())
        doc_with_coll, _ = from_fhir(_make_specimen(
            collector_ref="Practitioner/dr-jones"
        ))
        assert _get_opinion(doc_with_coll).uncertainty < _get_opinion(doc_no_coll).uncertainty

    def test_collector_signal_is_modest(self):
        """Collector presence is a supporting signal, not dominant."""
        doc_no_coll, _ = from_fhir(_make_specimen())
        doc_with_coll, _ = from_fhir(_make_specimen(
            collector_ref="Practitioner/dr-jones"
        ))
        op_no = _get_opinion(doc_no_coll)
        op_yes = _get_opinion(doc_with_coll)
        # Difference should be noticeable but not huge
        diff = op_no.uncertainty - op_yes.uncertainty
        assert 0.0 < diff < 0.10


# ═══════════════════════════════════════════════════════════════════
#  7. Collection quantity signal
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenQuantitySignal:
    """Test that collection quantity presence modulates uncertainty."""

    def test_quantity_present_reduces_uncertainty(self):
        doc_no_qty, _ = from_fhir(_make_specimen())
        doc_with_qty, _ = from_fhir(_make_specimen(quantity_value=5.0))
        assert _get_opinion(doc_with_qty).uncertainty < _get_opinion(doc_no_qty).uncertainty

    def test_quantity_signal_is_documentation_based(self):
        """Different quantity values should produce same uncertainty
        (we use presence, not threshold)."""
        doc_small, _ = from_fhir(_make_specimen(quantity_value=1.0))
        doc_large, _ = from_fhir(_make_specimen(quantity_value=100.0))
        op_small = _get_opinion(doc_small)
        op_large = _get_opinion(doc_large)
        assert abs(op_small.uncertainty - op_large.uncertainty) < 0.001


# ═══════════════════════════════════════════════════════════════════
#  8. Signal interactions (multiplicative)
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenSignalInteractions:
    """Test that signals interact multiplicatively."""

    def test_all_positive_signals_minimize_uncertainty(self):
        """Available + no degradation + processing + collector + quantity
        should produce lowest uncertainty."""
        doc_minimal, _ = from_fhir(_make_specimen())
        doc_full, _ = from_fhir(_make_specimen(
            status="available",
            condition_codes=["LIVE"],
            processing_count=4,
            collector_ref="Practitioner/dr-jones",
            quantity_value=10.0,
        ))
        assert _get_opinion(doc_full).uncertainty < _get_opinion(doc_minimal).uncertainty

    def test_all_negative_signals_maximize_uncertainty(self):
        """Unsatisfactory + degradation + no processing + no collector
        should produce high uncertainty."""
        doc_good, _ = from_fhir(_make_specimen(status="available"))
        doc_bad, _ = from_fhir(_make_specimen(
            status="unsatisfactory",
            condition_codes=["HEM", "CLOT", "CON"],
            processing_count=0,
        ))
        assert _get_opinion(doc_bad).uncertainty > _get_opinion(doc_good).uncertainty

    def test_condition_dominates_over_collector(self):
        """Hemolyzed specimen with collector identified should still
        have higher uncertainty than clean specimen without collector."""
        doc_clean_anon, _ = from_fhir(_make_specimen(
            condition_codes=[],
        ))
        doc_hem_identified, _ = from_fhir(_make_specimen(
            condition_codes=["HEM"],
            collector_ref="Practitioner/dr-jones",
        ))
        assert _get_opinion(doc_hem_identified).uncertainty > _get_opinion(doc_clean_anon).uncertainty

    def test_uncertainty_is_clamped(self):
        """Even with extreme degradation, uncertainty must stay < 1.0."""
        doc, _ = from_fhir(_make_specimen(
            status="unsatisfactory",
            condition_codes=["HEM", "CLOT", "CON", "AUT", "SNR"],
            processing_count=0,
        ))
        op = _get_opinion(doc)
        assert 0.0 < op.uncertainty < 1.0

    def test_opinion_components_sum_to_one(self):
        """SL invariant: b + d + u must equal 1.0."""
        doc, _ = from_fhir(_make_specimen(
            status="available",
            condition_codes=["HEM"],
            processing_count=2,
            collector_ref="Practitioner/dr-jones",
            quantity_value=5.0,
        ))
        op = _get_opinion(doc)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-10


# ═══════════════════════════════════════════════════════════════════
#  9. Extension recovery (exact opinion preservation)
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenExtensionRecovery:
    """Test that exact opinions embedded via FHIR extensions are recovered."""

    def test_extension_recovery_returns_exact_opinion(self):
        exact_op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.5)
        ext = opinion_to_fhir_extension(exact_op)
        resource = _make_specimen(
            status_extension={"extension": [ext]},
        )
        doc, _ = from_fhir(resource)
        recovered = _get_opinion(doc)
        assert abs(recovered.belief - exact_op.belief) < 1e-10
        assert abs(recovered.disbelief - exact_op.disbelief) < 1e-10
        assert abs(recovered.uncertainty - exact_op.uncertainty) < 1e-10

    def test_extension_recovery_source_is_extension(self):
        exact_op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.5)
        ext = opinion_to_fhir_extension(exact_op)
        resource = _make_specimen(
            status_extension={"extension": [ext]},
        )
        doc, _ = from_fhir(resource)
        assert doc["opinions"][0]["source"] == "extension"

    def test_extension_takes_precedence_over_reconstruction(self):
        """When extension is present, ignore status/condition signals."""
        exact_op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.5)
        ext = opinion_to_fhir_extension(exact_op)
        resource = _make_specimen(
            status="unsatisfactory",
            condition_codes=["HEM", "CLOT"],
            status_extension={"extension": [ext]},
        )
        doc, _ = from_fhir(resource)
        recovered = _get_opinion(doc)
        # Must match extension, not reconstruction
        assert abs(recovered.uncertainty - 0.20) < 1e-10


# ═══════════════════════════════════════════════════════════════════
#  10. Metadata passthrough (no data excluded)
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenMetadataPassthrough:
    """Verify all FHIR fields survive from_fhir → jsonld-ex doc."""

    def test_type_preserved(self):
        doc, _ = from_fhir(_make_specimen(type_text="Venous blood"))
        assert doc.get("type") == "Venous blood"

    def test_subject_preserved(self):
        doc, _ = from_fhir(_make_specimen(subject_ref="Patient/p-001"))
        assert doc.get("subject") == "Patient/p-001"

    def test_received_time_preserved(self):
        doc, _ = from_fhir(_make_specimen(received_time="2024-06-15T10:30:00Z"))
        assert doc.get("receivedTime") == "2024-06-15T10:30:00Z"

    def test_request_references_preserved(self):
        refs = ["ServiceRequest/sr-1", "ServiceRequest/sr-2"]
        doc, _ = from_fhir(_make_specimen(request_refs=refs))
        assert doc.get("request_references") == refs

    def test_collection_collector_preserved(self):
        doc, _ = from_fhir(_make_specimen(collector_ref="Practitioner/dr-jones"))
        assert doc.get("collector") == "Practitioner/dr-jones"

    def test_collection_datetime_preserved(self):
        doc, _ = from_fhir(_make_specimen(collected_dt="2024-06-15T08:00:00Z"))
        assert doc.get("collectedDateTime") == "2024-06-15T08:00:00Z"

    def test_collection_quantity_preserved(self):
        doc, _ = from_fhir(_make_specimen(quantity_value=5.0, quantity_unit="mL"))
        qty = doc.get("collection_quantity")
        assert qty is not None
        assert qty["value"] == 5.0
        assert qty["unit"] == "mL"

    def test_collection_method_preserved(self):
        doc, _ = from_fhir(_make_specimen(method_text="Venipuncture"))
        assert doc.get("collection_method") == "Venipuncture"

    def test_collection_body_site_preserved(self):
        doc, _ = from_fhir(_make_specimen(body_site_text="Left antecubital fossa"))
        assert doc.get("collection_bodySite") == "Left antecubital fossa"

    def test_condition_codes_preserved(self):
        doc, _ = from_fhir(_make_specimen(condition_codes=["HEM", "CLOT"]))
        assert doc.get("condition_codes") == ["HEM", "CLOT"]

    def test_processing_count_preserved(self):
        doc, _ = from_fhir(_make_specimen(processing_count=3))
        assert doc.get("processing_count") == 3

    def test_note_preserved(self):
        notes = ["Slightly hemolyzed", "Recollect if possible"]
        doc, _ = from_fhir(_make_specimen(note_texts=notes))
        assert doc.get("note") == notes

    def test_parent_references_preserved(self):
        parents = ["Specimen/parent-001"]
        doc, _ = from_fhir(_make_specimen(parent_refs=parents))
        assert doc.get("parent_references") == parents

    def test_id_preserved(self):
        doc, _ = from_fhir(_make_specimen(specimen_id="my-specimen"))
        assert doc["id"] == "my-specimen"

    def test_doc_type_is_fhir_specimen(self):
        doc, _ = from_fhir(_make_specimen())
        assert doc["@type"] == "fhir:Specimen"


# ═══════════════════════════════════════════════════════════════════
#  11. Round-trip fidelity (from_fhir → to_fhir)
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenRoundTrip:
    """Test from_fhir → to_fhir preserves all data."""

    def test_basic_round_trip(self):
        resource = _make_specimen(status="available")
        doc, _ = from_fhir(resource)
        fhir_out, report = to_fhir(doc)
        assert fhir_out["resourceType"] == "Specimen"
        assert fhir_out["id"] == "spec-001"
        assert fhir_out["status"] == "available"
        assert report.success is True

    def test_round_trip_preserves_opinion_as_extension(self):
        resource = _make_specimen(status="available")
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        # Opinion should be embedded as extension on _status
        assert "_status" in fhir_out
        exts = fhir_out["_status"]["extension"]
        assert len(exts) == 1
        assert exts[0]["url"] == FHIR_EXTENSION_URL

    def test_round_trip_condition_codes(self):
        resource = _make_specimen(condition_codes=["HEM", "CLOT"])
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        # Condition should be restored as CodeableConcept array
        conditions = fhir_out.get("condition", [])
        codes = []
        for cond in conditions:
            for coding in cond.get("coding", []):
                codes.append(coding.get("code"))
        assert "HEM" in codes
        assert "CLOT" in codes

    def test_round_trip_collection_data(self):
        resource = _make_specimen(
            collector_ref="Practitioner/dr-jones",
            collected_dt="2024-06-15T08:00:00Z",
            quantity_value=5.0,
            quantity_unit="mL",
            method_text="Venipuncture",
            body_site_text="Left antecubital fossa",
        )
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        coll = fhir_out.get("collection", {})
        assert coll.get("collector", {}).get("reference") == "Practitioner/dr-jones"
        assert coll.get("collectedDateTime") == "2024-06-15T08:00:00Z"
        assert coll.get("quantity", {}).get("value") == 5.0
        assert coll.get("method", {}).get("text") == "Venipuncture"
        assert coll.get("bodySite", {}).get("text") == "Left antecubital fossa"

    def test_round_trip_type(self):
        resource = _make_specimen(type_text="Venous blood")
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out.get("type", {}).get("text") == "Venous blood"

    def test_round_trip_subject(self):
        resource = _make_specimen(subject_ref="Patient/p-001")
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out.get("subject", {}).get("reference") == "Patient/p-001"

    def test_round_trip_received_time(self):
        resource = _make_specimen(received_time="2024-06-15T10:30:00Z")
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out.get("receivedTime") == "2024-06-15T10:30:00Z"

    def test_round_trip_request_references(self):
        refs = ["ServiceRequest/sr-1", "ServiceRequest/sr-2"]
        resource = _make_specimen(request_refs=refs)
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        out_refs = [r["reference"] for r in fhir_out.get("request", [])]
        assert out_refs == refs

    def test_round_trip_note(self):
        notes = ["Slightly hemolyzed", "Recollect if possible"]
        resource = _make_specimen(note_texts=notes)
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        out_notes = [n["text"] for n in fhir_out.get("note", [])]
        assert out_notes == notes

    def test_round_trip_parent_references(self):
        parents = ["Specimen/parent-001"]
        resource = _make_specimen(parent_refs=parents)
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        out_parents = [p["reference"] for p in fhir_out.get("parent", [])]
        assert out_parents == parents

    def test_round_trip_preserves_extension_exactly(self):
        """Extension round-trip: from_fhir(ext) → to_fhir → from_fhir → same."""
        exact_op = Opinion(belief=0.65, disbelief=0.15, uncertainty=0.20, base_rate=0.5)
        ext = opinion_to_fhir_extension(exact_op)
        resource = _make_specimen(status_extension={"extension": [ext]})
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        doc2, _ = from_fhir(fhir_out)
        op2 = _get_opinion(doc2)
        assert abs(op2.belief - exact_op.belief) < 1e-10
        assert abs(op2.disbelief - exact_op.disbelief) < 1e-10
        assert abs(op2.uncertainty - exact_op.uncertainty) < 1e-10


# ═══════════════════════════════════════════════════════════════════
#  12. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestSpecimenEdgeCases:
    """Edge cases and boundary conditions."""

    def test_missing_status_uses_fallback(self):
        """Resource with no status field should still produce a valid opinion."""
        resource = {"resourceType": "Specimen", "id": "no-status"}
        doc, report = from_fhir(resource)
        op = _get_opinion(doc)
        assert isinstance(op, Opinion)
        assert report.success is True

    def test_empty_collection_backbone(self):
        """Empty collection object should not crash."""
        resource = _make_specimen()
        resource["collection"] = {}
        doc, report = from_fhir(resource)
        assert report.success is True

    def test_no_collection_key(self):
        """Absence of collection key should not crash."""
        resource = _make_specimen()
        # Default _make_specimen with no collector/quantity has no collection key
        doc, report = from_fhir(resource)
        assert report.success is True

    def test_condition_with_empty_coding(self):
        """Condition CodeableConcept with no coding should be handled."""
        resource = _make_specimen()
        resource["condition"] = [{"text": "slightly hemolyzed"}]
        doc, report = from_fhir(resource)
        assert report.success is True

    def test_condition_with_multiple_codings(self):
        """Condition with multiple coding entries should use first code."""
        resource = _make_specimen()
        resource["condition"] = [{
            "coding": [
                {"code": "HEM", "display": "Hemolyzed"},
                {"code": "OTHER", "display": "Other"},
            ]
        }]
        doc, report = from_fhir(resource)
        # Should still process — HEM is the first code
        assert report.success is True

    def test_processing_with_no_description(self):
        """Processing steps without description should still count."""
        resource = _make_specimen()
        resource["processing"] = [{}, {}, {}]
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc.get("processing_count") == 3

    def test_quantity_zero_is_still_present(self):
        """Quantity value of 0 should still count as 'present'."""
        doc_zero, _ = from_fhir(_make_specimen(quantity_value=0.0))
        doc_none, _ = from_fhir(_make_specimen())
        # Both should produce valid opinions
        assert isinstance(_get_opinion(doc_zero), Opinion)
        assert isinstance(_get_opinion(doc_none), Opinion)
        # Zero quantity IS documented, so it should reduce uncertainty
        # compared to absent quantity
        assert _get_opinion(doc_zero).uncertainty < _get_opinion(doc_none).uncertainty
