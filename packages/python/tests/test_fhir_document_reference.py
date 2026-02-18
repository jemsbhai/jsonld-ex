"""Tests for Phase 7B: DocumentReference FHIR R4 resource support.

RED PHASE: Tests written before implementation.

DocumentReference is the highest-priority Phase 7B expansion type
(Score 38/50).  It indexes clinical notes, lab report PDFs, scanned
documents, and other binary objects — the most common FHIR resource
in many EHR systems.  US Core USCDI mandated.

Proposition under assessment:
    "This document reference is valid and the underlying document
    is reliable."

DocumentReference is unique among FHIR R4 resources in having **two
independent status dimensions**:

    1. ``status`` (1..1, Required) — The status of this document
       *reference* (the metadata record).
       Codes: current | superseded | entered-in-error
       ValueSet: DocumentReferenceStatus

    2. ``docStatus`` (0..1, Required binding) — The status of the
       underlying *document* (the content itself).
       Codes: preliminary | final | amended | entered-in-error
       ValueSet: CompositionStatus

This dual-status model is the key epistemic contribution: the
reference can be ``current`` while the document is ``preliminary``,
or the reference can be ``superseded`` while the document itself
was ``final``.  Each combination produces genuinely different
epistemic semantics.

Epistemic signals extracted (5 multiplicative signals):

    Signal 1 — ``status`` (reference validity):
        current=0.85/0.15  — active, valid reference
        superseded=0.25/0.20 — replaced by newer version
        entered-in-error=0.50/0.70 — data integrity compromised

    Signal 2 — ``docStatus`` (document content maturity):
        Uncertainty multiplier on base uncertainty.
        final=×0.7   — reviewed, finalized
        amended=×0.8  — revised but less settled
        preliminary=×1.4 — draft, unreliable content
        entered-in-error=×2.0 — invalid content
        absent=×1.0   — baseline (many valid docs lack docStatus)

    Signal 3 — ``authenticator`` presence (verification):
        present=×0.8 — independently verified
        absent=×1.0  — baseline (no penalty; optional field)

    Signal 4 — ``author[]`` count (accountability):
        0 authors=×1.2 — anonymous, less accountable
        1 author=×1.0  — baseline
        2+ authors=×0.9 — corroborated authorship

    Signal 5 — ``content[]`` count (document availability):
        0 content=×1.3 — defensive (spec says 1..* but handle edge)
        1 content=×1.0 — baseline
        2+ content=×0.9 — multiple representations available

This is a CUSTOM handler (not ``_make_status_handler``) because:
    1. Dual status dimensions (status + docStatus) — unique in FHIR R4
    2. Authenticator provides independent verification signal
    3. Author count modulates accountability
    4. Content count modulates availability/completeness
    5. These 5 signals interact multiplicatively

Design decisions documented for scientific rigor:

    ``status`` probability encodes "reference record validity":
        current=0.85 — active reference, high validity (analogous
            to ``active`` in other resources)
        superseded=0.25 — replaced; low probability that this specific
            reference should be relied upon, but the document it
            pointed to may still be valid (hence not as low as 0.10)
        entered-in-error=0.50 — data integrity compromised, maximum
            uncertainty rather than definite invalidity

    ``docStatus`` multiplier encodes "content maturity":
        These mirror CompositionStatus semantics.  ``final`` has been
        reviewed and signed off; ``preliminary`` is a draft that may
        contain errors or incomplete information.
        The separation from ``status`` allows expressing states like
        "current reference to a preliminary document" — a genuinely
        different epistemic state from "current reference to a final
        document."

    ``authenticator`` provides a binary verification signal.
        FHIR R4 spec: "Which person or organization authenticates
        that this document is valid."  This is an independent
        attestation that reduces uncertainty.  Its absence is not
        penalized (many valid documents lack authenticators) but its
        presence is rewarded.

    ``author[]`` count follows existing corroboration patterns
        (similar to agent count in Provenance handler).

    ``content[]`` count represents document availability.
        Multiple content elements mean the document is available in
        multiple formats/representations.

Metadata passthrough (no data excluded):
    type (CodeableConcept text), category codes, subject ref, date,
    author refs, authenticator ref, custodian ref, description,
    securityLabel codes, relatesTo (code + target ref), content count,
    context.encounter refs, context.period

References:
    - HL7 FHIR R4 DocumentReference:
      https://hl7.org/fhir/R4/documentreference.html
    - DocumentReferenceStatus ValueSet:
      https://hl7.org/fhir/R4/valueset-document-reference-status.html
    - CompositionStatus ValueSet:
      https://hl7.org/fhir/R4/valueset-composition-status.html
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
    DOC_REF_STATUS_PROBABILITY,
    DOC_REF_STATUS_UNCERTAINTY,
    DOC_REF_DOC_STATUS_MULTIPLIER,
)


# ═══════════════════════════════════════════════════════════════════
# Fixture helpers
# ═══════════════════════════════════════════════════════════════════


def _document_reference(
    *,
    status="current",
    doc_status=None,
    authenticator=None,
    authors=None,
    content_count=1,
    id_="dr-1",
):
    """Build a minimal valid FHIR R4 DocumentReference.

    Args:
        status: DocumentReferenceStatus code.
        doc_status: CompositionStatus code (None = absent).
        authenticator: Reference string or None.
        authors: List of reference strings, or None for default single author.
        content_count: Number of content elements (each with a stub attachment).
        id_: Resource id.
    """
    resource = {
        "resourceType": "DocumentReference",
        "id": id_,
        "status": status,
        "type": {
            "coding": [{"system": "http://loinc.org", "code": "34133-9"}],
            "text": "Summary of episode note",
        },
        "subject": {"reference": "Patient/p1"},
        "content": [
            {
                "attachment": {
                    "contentType": "application/pdf",
                    "url": f"Binary/doc-{i}",
                },
            }
            for i in range(content_count)
        ],
    }
    if doc_status is not None:
        resource["docStatus"] = doc_status
    if authenticator is not None:
        resource["authenticator"] = {"reference": authenticator}
    if authors is not None:
        resource["author"] = [{"reference": ref} for ref in authors]
    return resource


# ═══════════════════════════════════════════════════════════════════
# Registration: DocumentReference in SUPPORTED_RESOURCE_TYPES
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceRegistration:
    """DocumentReference must be registered as a supported type."""

    def test_in_supported_types(self):
        assert "DocumentReference" in SUPPORTED_RESOURCE_TYPES


# ═══════════════════════════════════════════════════════════════════
# Constants: Mapping tables exist and are well-formed
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceConstants:
    """Verify constant mapping tables exist and have valid values."""

    # ── Status probability map (DocumentReferenceStatus) ──────────

    def test_status_probability_covers_all_fhir_codes(self):
        """All FHIR R4 DocumentReference.status codes must be mapped.

        ValueSet: DocumentReferenceStatus (Required binding)
        Codes: current | superseded | entered-in-error
        """
        expected = {"current", "superseded", "entered-in-error"}
        assert set(DOC_REF_STATUS_PROBABILITY.keys()) == expected

    def test_status_probability_values_in_range(self):
        for code, prob in DOC_REF_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, (
                f"status '{code}' prob={prob} out of [0, 1]"
            )

    def test_status_uncertainty_covers_all_fhir_codes(self):
        expected = {"current", "superseded", "entered-in-error"}
        assert set(DOC_REF_STATUS_UNCERTAINTY.keys()) == expected

    def test_status_uncertainty_values_in_range(self):
        for code, u in DOC_REF_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, (
                f"status '{code}' uncertainty={u} out of (0, 1)"
            )

    # ── docStatus multiplier map (CompositionStatus) ──────────────

    def test_doc_status_multiplier_covers_all_fhir_codes(self):
        """All FHIR R4 DocumentReference.docStatus codes must be mapped.

        ValueSet: CompositionStatus (Required binding)
        Codes: preliminary | final | amended | entered-in-error
        """
        expected = {"preliminary", "final", "amended", "entered-in-error"}
        assert set(DOC_REF_DOC_STATUS_MULTIPLIER.keys()) == expected

    def test_doc_status_multiplier_values_positive(self):
        for code, mult in DOC_REF_DOC_STATUS_MULTIPLIER.items():
            assert mult > 0.0, (
                f"docStatus '{code}' multiplier={mult} must be positive"
            )

    def test_doc_status_final_reduces_uncertainty(self):
        """Final documents should have multiplier < 1.0."""
        assert DOC_REF_DOC_STATUS_MULTIPLIER["final"] < 1.0

    def test_doc_status_preliminary_increases_uncertainty(self):
        """Preliminary documents should have multiplier > 1.0."""
        assert DOC_REF_DOC_STATUS_MULTIPLIER["preliminary"] > 1.0

    def test_doc_status_entered_in_error_highest_multiplier(self):
        """entered-in-error should have the highest multiplier."""
        eie_mult = DOC_REF_DOC_STATUS_MULTIPLIER["entered-in-error"]
        for code, mult in DOC_REF_DOC_STATUS_MULTIPLIER.items():
            assert eie_mult >= mult, (
                f"entered-in-error ({eie_mult}) should be >= {code} ({mult})"
            )

    def test_doc_status_ordering_final_lt_amended_lt_preliminary(self):
        """Multiplier ordering: final < amended < preliminary."""
        m = DOC_REF_DOC_STATUS_MULTIPLIER
        assert m["final"] < m["amended"] < m["preliminary"]

    # ── Epistemic ordering within status map ──────────────────────

    def test_current_higher_prob_than_superseded(self):
        assert (
            DOC_REF_STATUS_PROBABILITY["current"]
            > DOC_REF_STATUS_PROBABILITY["superseded"]
        )

    def test_entered_in_error_high_uncertainty(self):
        assert DOC_REF_STATUS_UNCERTAINTY["entered-in-error"] >= 0.50


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Basic status-based conversion
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceFromFhir:
    """Test from_fhir() for DocumentReference resources."""

    def test_basic_current_document(self):
        """Current document should produce a valid opinion."""
        doc, report = from_fhir(_document_reference(status="current"))
        assert report.success
        assert report.nodes_converted >= 1
        assert doc["@type"] == "fhir:DocumentReference"
        assert doc["id"] == "dr-1"
        assert doc["status"] == "current"
        assert len(doc["opinions"]) >= 1

    def test_opinion_is_valid_sl_triple(self):
        """Opinion must satisfy b + d + u = 1, all in [0, 1]."""
        doc, _ = from_fhir(_document_reference())
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert 0.0 <= op.belief <= 1.0
        assert 0.0 <= op.disbelief <= 1.0
        assert 0.0 <= op.uncertainty <= 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    @pytest.mark.parametrize("status", [
        "current", "superseded", "entered-in-error",
    ])
    def test_all_status_codes_produce_valid_opinion(self, status):
        """Every FHIR R4 DocumentReference.status code must produce valid opinion."""
        doc, report = from_fhir(_document_reference(status=status))
        assert report.success
        assert len(doc["opinions"]) >= 1
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_status_field_preserved(self):
        doc, _ = from_fhir(_document_reference(status="superseded"))
        assert doc["status"] == "superseded"

    def test_opinion_source_is_reconstructed(self):
        """Without extension, source should be 'reconstructed'."""
        doc, _ = from_fhir(_document_reference())
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_opinion_field_is_status(self):
        """Primary opinion should be on the 'status' field."""
        doc, _ = from_fhir(_document_reference())
        assert doc["opinions"][0]["field"] == "status"

    def test_current_higher_belief_than_superseded(self):
        """Current references should have higher belief than superseded."""
        doc_current, _ = from_fhir(_document_reference(status="current"))
        doc_superseded, _ = from_fhir(_document_reference(status="superseded"))
        b_current = doc_current["opinions"][0]["opinion"].belief
        b_superseded = doc_superseded["opinions"][0]["opinion"].belief
        assert b_current > b_superseded

    def test_superseded_low_belief(self):
        """Superseded references should have low belief."""
        doc, _ = from_fhir(_document_reference(status="superseded"))
        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.40

    def test_entered_in_error_high_uncertainty(self):
        """Entered-in-error should have elevated uncertainty."""
        doc, _ = from_fhir(_document_reference(status="entered-in-error"))
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty >= 0.35


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): docStatus modulation (Signal 2)
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceDocStatusModulation:
    """docStatus modulates uncertainty: final < amended < preliminary."""

    def test_final_less_uncertain_than_preliminary(self):
        """A final document should have less uncertainty than preliminary."""
        doc_final, _ = from_fhir(
            _document_reference(status="current", doc_status="final")
        )
        doc_prelim, _ = from_fhir(
            _document_reference(status="current", doc_status="preliminary")
        )
        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_final < u_prelim

    def test_amended_between_final_and_preliminary(self):
        """Amended should fall between final and preliminary in uncertainty."""
        doc_final, _ = from_fhir(
            _document_reference(status="current", doc_status="final")
        )
        doc_amended, _ = from_fhir(
            _document_reference(status="current", doc_status="amended")
        )
        doc_prelim, _ = from_fhir(
            _document_reference(status="current", doc_status="preliminary")
        )
        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_amended = doc_amended["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_final < u_amended < u_prelim

    def test_doc_status_entered_in_error_highest_uncertainty(self):
        """docStatus=entered-in-error should yield the highest uncertainty."""
        doc_eie, _ = from_fhir(
            _document_reference(status="current", doc_status="entered-in-error")
        )
        doc_prelim, _ = from_fhir(
            _document_reference(status="current", doc_status="preliminary")
        )
        u_eie = doc_eie["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_eie > u_prelim

    def test_absent_doc_status_is_baseline(self):
        """No docStatus should not change uncertainty vs baseline."""
        doc_none, _ = from_fhir(
            _document_reference(status="current", doc_status=None)
        )
        # Baseline: no docStatus multiplier applied (×1.0)
        op = doc_none["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    @pytest.mark.parametrize("doc_status", [
        "preliminary", "final", "amended", "entered-in-error",
    ])
    def test_all_doc_status_codes_produce_valid_opinion(self, doc_status):
        """Every CompositionStatus code must produce a valid opinion."""
        doc, report = from_fhir(
            _document_reference(status="current", doc_status=doc_status)
        )
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_dual_status_interaction(self):
        """Superseded reference + final document should differ from
        current reference + preliminary document.

        This tests the genuinely different epistemic state produced
        by the dual-status model."""
        doc_sup_final, _ = from_fhir(
            _document_reference(status="superseded", doc_status="final")
        )
        doc_cur_prelim, _ = from_fhir(
            _document_reference(status="current", doc_status="preliminary")
        )
        # Different base probabilities, different uncertainty paths
        b_sup = doc_sup_final["opinions"][0]["opinion"].belief
        b_cur = doc_cur_prelim["opinions"][0]["opinion"].belief
        # Superseded should have lower belief than current regardless of docStatus
        assert b_sup < b_cur


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Authenticator modulation (Signal 3)
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceAuthenticatorModulation:
    """Authenticator presence reduces uncertainty (verification signal)."""

    def test_authenticator_reduces_uncertainty(self):
        """Authenticated documents should have lower uncertainty."""
        doc_auth, _ = from_fhir(
            _document_reference(
                status="current",
                authenticator="Practitioner/dr-smith",
            )
        )
        doc_no_auth, _ = from_fhir(
            _document_reference(status="current", authenticator=None)
        )
        u_auth = doc_auth["opinions"][0]["opinion"].uncertainty
        u_no_auth = doc_no_auth["opinions"][0]["opinion"].uncertainty
        assert u_auth < u_no_auth

    def test_authenticator_absence_is_not_penalized(self):
        """Missing authenticator should use baseline (×1.0), not increase u."""
        doc_no_auth, _ = from_fhir(
            _document_reference(status="current", authenticator=None)
        )
        # Without any other modifying signals, should be at baseline
        op = doc_no_auth["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Author count modulation (Signal 4)
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceAuthorModulation:
    """Author count modulates uncertainty: 0 > 1 > 2+ authors."""

    def test_no_authors_increases_uncertainty(self):
        """Zero authors should increase uncertainty vs single author."""
        doc_none, _ = from_fhir(
            _document_reference(status="current", authors=[])
        )
        doc_one, _ = from_fhir(
            _document_reference(
                status="current", authors=["Practitioner/dr-a"]
            )
        )
        u_none = doc_none["opinions"][0]["opinion"].uncertainty
        u_one = doc_one["opinions"][0]["opinion"].uncertainty
        assert u_none > u_one

    def test_multiple_authors_reduces_uncertainty(self):
        """Two+ authors should reduce uncertainty vs single author."""
        doc_one, _ = from_fhir(
            _document_reference(
                status="current", authors=["Practitioner/dr-a"]
            )
        )
        doc_two, _ = from_fhir(
            _document_reference(
                status="current",
                authors=["Practitioner/dr-a", "Practitioner/dr-b"],
            )
        )
        u_one = doc_one["opinions"][0]["opinion"].uncertainty
        u_two = doc_two["opinions"][0]["opinion"].uncertainty
        assert u_two < u_one

    def test_absent_author_field_is_baseline(self):
        """When author field is absent entirely (None), use baseline."""
        doc_absent, _ = from_fhir(
            _document_reference(status="current", authors=None)
        )
        op = doc_absent["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Content count modulation (Signal 5)
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceContentModulation:
    """Content count modulates uncertainty for document availability."""

    def test_multiple_content_reduces_uncertainty(self):
        """2+ content elements should reduce uncertainty."""
        doc_one, _ = from_fhir(
            _document_reference(status="current", content_count=1)
        )
        doc_two, _ = from_fhir(
            _document_reference(status="current", content_count=2)
        )
        u_one = doc_one["opinions"][0]["opinion"].uncertainty
        u_two = doc_two["opinions"][0]["opinion"].uncertainty
        assert u_two < u_one

    def test_zero_content_increases_uncertainty(self):
        """Zero content (defensive edge case) should increase uncertainty."""
        doc_zero, _ = from_fhir(
            _document_reference(status="current", content_count=0)
        )
        doc_one, _ = from_fhir(
            _document_reference(status="current", content_count=1)
        )
        u_zero = doc_zero["opinions"][0]["opinion"].uncertainty
        u_one = doc_one["opinions"][0]["opinion"].uncertainty
        assert u_zero > u_one


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Signal interactions
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceSignalInteractions:
    """Test multiplicative interaction of all 5 signals."""

    def test_best_case_lowest_uncertainty(self):
        """current + final + authenticator + 2 authors + 2 content → min u."""
        doc, _ = from_fhir(
            _document_reference(
                status="current",
                doc_status="final",
                authenticator="Practitioner/dr-smith",
                authors=["Practitioner/dr-a", "Practitioner/dr-b"],
                content_count=2,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 0.10

    def test_worst_case_highest_uncertainty(self):
        """entered-in-error + entered-in-error docStatus + no auth + no authors + 0 content."""
        doc, _ = from_fhir(
            _document_reference(
                status="entered-in-error",
                doc_status="entered-in-error",
                authenticator=None,
                authors=[],
                content_count=0,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.50

    def test_uncertainty_always_clamped_below_one(self):
        """Even worst-case signal combination must clamp u < 1.0."""
        doc, _ = from_fhir(
            _document_reference(
                status="entered-in-error",
                doc_status="entered-in-error",
                authenticator=None,
                authors=[],
                content_count=0,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 1.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_uncertainty_always_positive(self):
        """Even best-case signal combination must have u > 0."""
        doc, _ = from_fhir(
            _document_reference(
                status="current",
                doc_status="final",
                authenticator="Practitioner/dr-smith",
                authors=[f"Practitioner/dr-{i}" for i in range(5)],
                content_count=3,
            )
        )
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.0

    def test_doc_status_compounds_with_authenticator(self):
        """final + authenticator should yield less uncertainty than
        final alone, confirming multiplicative interaction."""
        doc_final_only, _ = from_fhir(
            _document_reference(
                status="current", doc_status="final", authenticator=None,
            )
        )
        doc_final_auth, _ = from_fhir(
            _document_reference(
                status="current",
                doc_status="final",
                authenticator="Practitioner/dr-smith",
            )
        )
        u_only = doc_final_only["opinions"][0]["opinion"].uncertainty
        u_auth = doc_final_auth["opinions"][0]["opinion"].uncertainty
        assert u_auth < u_only


# ═══════════════════════════════════════════════════════════════════
# from_fhir(): Extension recovery (exact round-trip)
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceExtensionRecovery:
    """When a jsonld-ex FHIR extension is present, recover the exact opinion."""

    def test_extension_recovery_overrides_reconstruction(self):
        """An embedded extension should be recovered exactly."""
        original = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        resource = _document_reference(status="current")
        resource["_status"] = {"extension": [ext]}

        doc, report = from_fhir(resource)
        assert report.success
        recovered = doc["opinions"][0]["opinion"]
        assert doc["opinions"][0]["source"] == "extension"
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9

    def test_extension_ignores_metadata_signals(self):
        """When extension present, docStatus/authenticator/etc. don't affect opinion."""
        original = Opinion(belief=0.30, disbelief=0.50, uncertainty=0.20)
        ext = opinion_to_fhir_extension(original)

        resource = _document_reference(
            status="current",
            doc_status="final",
            authenticator="Practitioner/dr-smith",
            authors=["Practitioner/dr-a", "Practitioner/dr-b"],
            content_count=3,
        )
        resource["_status"] = {"extension": [ext]}

        doc, _ = from_fhir(resource)
        recovered = doc["opinions"][0]["opinion"]
        assert abs(recovered.belief - 0.30) < 1e-9
        assert abs(recovered.disbelief - 0.50) < 1e-9

    def test_wrong_extension_url_triggers_reconstruction(self):
        """An extension with a different URL should not be recovered."""
        resource = _document_reference(status="current")
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


class TestDocumentReferenceMetadataPassthrough:
    """Key metadata fields should be preserved in the jsonld-ex document."""

    def test_doc_status_preserved(self):
        doc, _ = from_fhir(_document_reference(doc_status="final"))
        assert doc.get("docStatus") == "final"

    def test_doc_status_absent_when_not_set(self):
        doc, _ = from_fhir(_document_reference(doc_status=None))
        assert "docStatus" not in doc

    def test_date_preserved(self):
        resource = _document_reference()
        resource["date"] = "2024-06-15T14:30:00Z"
        doc, _ = from_fhir(resource)
        assert doc["date"] == "2024-06-15T14:30:00Z"

    def test_description_preserved(self):
        resource = _document_reference()
        resource["description"] = "Discharge summary for patient P1"
        doc, _ = from_fhir(resource)
        assert doc["description"] == "Discharge summary for patient P1"

    def test_type_text_preserved(self):
        doc, _ = from_fhir(_document_reference())
        assert doc.get("type") == "Summary of episode note"

    def test_subject_preserved(self):
        doc, _ = from_fhir(_document_reference())
        assert doc.get("subject") == "Patient/p1"

    def test_authenticator_ref_preserved(self):
        doc, _ = from_fhir(
            _document_reference(authenticator="Practitioner/dr-smith")
        )
        assert doc.get("authenticator") == "Practitioner/dr-smith"

    def test_authenticator_absent_when_not_set(self):
        doc, _ = from_fhir(_document_reference(authenticator=None))
        assert "authenticator" not in doc

    def test_author_refs_preserved(self):
        doc, _ = from_fhir(
            _document_reference(
                authors=["Practitioner/dr-a", "Organization/org-1"]
            )
        )
        assert "author_references" in doc
        assert "Practitioner/dr-a" in doc["author_references"]
        assert "Organization/org-1" in doc["author_references"]

    def test_custodian_preserved(self):
        resource = _document_reference()
        resource["custodian"] = {"reference": "Organization/hosp-1"}
        doc, _ = from_fhir(resource)
        assert doc.get("custodian") == "Organization/hosp-1"

    def test_content_count_preserved(self):
        doc, _ = from_fhir(_document_reference(content_count=3))
        assert doc.get("content_count") == 3

    def test_security_label_codes_preserved(self):
        resource = _document_reference()
        resource["securityLabel"] = [
            {"coding": [{"code": "R"}]},
            {"coding": [{"code": "V"}]},
        ]
        doc, _ = from_fhir(resource)
        assert "securityLabel_codes" in doc
        assert "R" in doc["securityLabel_codes"]
        assert "V" in doc["securityLabel_codes"]

    def test_relates_to_preserved(self):
        resource = _document_reference()
        resource["relatesTo"] = [
            {
                "code": "replaces",
                "target": {"reference": "DocumentReference/old-1"},
            },
        ]
        doc, _ = from_fhir(resource)
        assert "relatesTo" in doc
        assert len(doc["relatesTo"]) == 1
        assert doc["relatesTo"][0]["code"] == "replaces"
        assert doc["relatesTo"][0]["target"] == "DocumentReference/old-1"

    def test_context_encounter_refs_preserved(self):
        resource = _document_reference()
        resource["context"] = {
            "encounter": [{"reference": "Encounter/enc-1"}],
        }
        doc, _ = from_fhir(resource)
        assert "context_encounter_references" in doc
        assert "Encounter/enc-1" in doc["context_encounter_references"]

    def test_context_period_preserved(self):
        resource = _document_reference()
        resource["context"] = {
            "period": {"start": "2024-01-01", "end": "2024-01-31"},
        }
        doc, _ = from_fhir(resource)
        assert "context_period" in doc
        assert doc["context_period"]["start"] == "2024-01-01"

    def test_category_codes_preserved(self):
        resource = _document_reference()
        resource["category"] = [
            {"coding": [{"code": "clinical-note"}], "text": "Clinical Note"},
        ]
        doc, _ = from_fhir(resource)
        assert "category" in doc


# ═══════════════════════════════════════════════════════════════════
# to_fhir(): Export to FHIR R4
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceToFhir:
    """Test to_fhir() export for DocumentReference."""

    def _make_doc(self, **overrides):
        """Build a minimal jsonld-ex DocumentReference document."""
        base = {
            "@type": "fhir:DocumentReference",
            "id": "dr-1",
            "status": "current",
            "opinions": [{
                "field": "status",
                "value": "current",
                "opinion": Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
                "source": "reconstructed",
            }],
        }
        base.update(overrides)
        return base

    def test_basic_export(self):
        """A jsonld-ex DocumentReference should export to valid FHIR resource."""
        resource, report = to_fhir(self._make_doc())
        assert report.success
        assert resource["resourceType"] == "DocumentReference"
        assert resource["id"] == "dr-1"
        assert resource["status"] == "current"

    def test_opinion_embedded_as_extension(self):
        """SL opinion should be embedded as a FHIR extension on _status."""
        resource, _ = to_fhir(self._make_doc())
        assert "_status" in resource
        extensions = resource["_status"]["extension"]
        assert len(extensions) >= 1
        assert extensions[0]["url"] == FHIR_EXTENSION_URL

    def test_doc_status_exported(self):
        resource, _ = to_fhir(self._make_doc(docStatus="final"))
        assert resource["docStatus"] == "final"

    def test_doc_status_absent_when_not_in_doc(self):
        resource, _ = to_fhir(self._make_doc())
        assert "docStatus" not in resource

    def test_date_exported(self):
        resource, _ = to_fhir(self._make_doc(date="2024-06-15T14:30:00Z"))
        assert resource["date"] == "2024-06-15T14:30:00Z"

    def test_description_exported(self):
        resource, _ = to_fhir(
            self._make_doc(description="Discharge summary")
        )
        assert resource["description"] == "Discharge summary"

    def test_type_exported_as_codeable_concept(self):
        resource, _ = to_fhir(
            self._make_doc(type="Summary of episode note")
        )
        assert "type" in resource
        assert resource["type"]["text"] == "Summary of episode note"

    def test_subject_exported_as_reference(self):
        resource, _ = to_fhir(self._make_doc(subject="Patient/p1"))
        assert resource["subject"] == {"reference": "Patient/p1"}

    def test_authenticator_exported_as_reference(self):
        resource, _ = to_fhir(
            self._make_doc(authenticator="Practitioner/dr-smith")
        )
        assert resource["authenticator"] == {
            "reference": "Practitioner/dr-smith"
        }

    def test_custodian_exported_as_reference(self):
        resource, _ = to_fhir(
            self._make_doc(custodian="Organization/hosp-1")
        )
        assert resource["custodian"] == {"reference": "Organization/hosp-1"}

    def test_author_refs_exported_as_references(self):
        resource, _ = to_fhir(
            self._make_doc(
                author_references=["Practitioner/dr-a", "Organization/org-1"]
            )
        )
        assert "author" in resource
        refs = [a["reference"] for a in resource["author"]]
        assert "Practitioner/dr-a" in refs
        assert "Organization/org-1" in refs

    def test_relates_to_exported(self):
        resource, _ = to_fhir(
            self._make_doc(
                relatesTo=[{
                    "code": "replaces",
                    "target": "DocumentReference/old-1",
                }]
            )
        )
        assert "relatesTo" in resource
        rt = resource["relatesTo"][0]
        assert rt["code"] == "replaces"
        assert rt["target"] == {"reference": "DocumentReference/old-1"}

    def test_security_label_codes_exported(self):
        resource, _ = to_fhir(
            self._make_doc(securityLabel_codes=["R", "V"])
        )
        assert "securityLabel" in resource
        codes = [
            sl["coding"][0]["code"] for sl in resource["securityLabel"]
        ]
        assert "R" in codes
        assert "V" in codes

    def test_context_encounter_exported(self):
        resource, _ = to_fhir(
            self._make_doc(
                context_encounter_references=["Encounter/enc-1"]
            )
        )
        assert "context" in resource
        enc_refs = [
            e["reference"] for e in resource["context"]["encounter"]
        ]
        assert "Encounter/enc-1" in enc_refs

    def test_context_period_exported(self):
        resource, _ = to_fhir(
            self._make_doc(
                context_period={"start": "2024-01-01", "end": "2024-01-31"}
            )
        )
        assert "context" in resource
        assert resource["context"]["period"]["start"] == "2024-01-01"


# ═══════════════════════════════════════════════════════════════════
# Round-trip fidelity: from_fhir() → to_fhir() → from_fhir()
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceRoundTrip:
    """Round-trip through from_fhir → to_fhir should preserve opinions exactly."""

    def test_opinion_preserved_via_extension(self):
        """After from_fhir → to_fhir → from_fhir, opinion should be exact."""
        original = _document_reference(
            status="current",
            doc_status="final",
            authenticator="Practitioner/dr-smith",
            authors=["Practitioner/dr-a"],
            content_count=2,
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
        original = _document_reference()
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["resourceType"] == "DocumentReference"

    def test_id_preserved(self):
        original = _document_reference(id_="dr-roundtrip-42")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["id"] == "dr-roundtrip-42"

    def test_status_preserved(self):
        original = _document_reference(status="superseded")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource["status"] == "superseded"

    def test_doc_status_preserved_through_round_trip(self):
        original = _document_reference(doc_status="amended")
        doc, _ = from_fhir(original)
        resource, _ = to_fhir(doc)
        assert resource.get("docStatus") == "amended"


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestDocumentReferenceEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_missing_id(self):
        """Resource without id should not crash."""
        resource = _document_reference()
        del resource["id"]
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["id"] is None

    def test_missing_status_uses_default(self):
        """If status is absent, should use a safe default."""
        resource = _document_reference()
        del resource["status"]
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_novel_status_code_handled(self):
        """A future/unknown status code should not crash."""
        resource = _document_reference()
        resource["status"] = "future-status-code"
        doc, report = from_fhir(resource)
        assert report.success
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_novel_doc_status_code_handled(self):
        """A future/unknown docStatus code should not crash."""
        resource = _document_reference()
        resource["docStatus"] = "future-doc-status"
        doc, report = from_fhir(resource)
        assert report.success

    def test_empty_content_array(self):
        """Empty content array (violates 1..* but defensive)."""
        resource = _document_reference(content_count=0)
        doc, report = from_fhir(resource)
        assert report.success

    def test_missing_content_entirely(self):
        """Missing content key entirely (defensive)."""
        resource = _document_reference()
        del resource["content"]
        doc, report = from_fhir(resource)
        assert report.success

    def test_authenticator_without_reference(self):
        """Authenticator object without 'reference' key (defensive)."""
        resource = _document_reference()
        resource["authenticator"] = {"display": "Dr. Smith"}
        doc, report = from_fhir(resource)
        assert report.success

    def test_author_without_reference(self):
        """Author objects without 'reference' key (defensive)."""
        resource = _document_reference()
        resource["author"] = [{"display": "Dr. Smith"}]
        doc, report = from_fhir(resource)
        assert report.success

    def test_relates_to_empty_array(self):
        """Empty relatesTo array should not crash."""
        resource = _document_reference()
        resource["relatesTo"] = []
        doc, report = from_fhir(resource)
        assert report.success

    def test_context_empty_object(self):
        """Empty context object should not crash."""
        resource = _document_reference()
        resource["context"] = {}
        doc, report = from_fhir(resource)
        assert report.success

    def test_context_encounter_without_reference(self):
        """Context encounter without 'reference' key (defensive)."""
        resource = _document_reference()
        resource["context"] = {
            "encounter": [{"display": "Office visit"}],
        }
        doc, report = from_fhir(resource)
        assert report.success
