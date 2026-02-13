"""Tests for dpv_interop module — Phase 3: DPV v2.2 Bidirectional Interop.

TDD: Write tests FIRST (red phase), then implement (green phase).

Tests the mapping between jsonld-ex data protection annotations and
W3C Data Privacy Vocabulary (DPV) v2.2 concepts.
"""

import pytest

from jsonld_ex.data_protection import annotate_protection, create_consent_record
from jsonld_ex.owl_interop import ConversionReport, VerbosityComparison

# Module under test — doesn't exist yet (red phase).
from jsonld_ex.dpv_interop import (
    to_dpv,
    from_dpv,
    compare_with_dpv,
    # Namespace constants
    DPV,
    EU_GDPR,
    DPV_LOC,
    DPV_PD,
    # Mapping constants
    LEGAL_BASIS_TO_DPV,
    CATEGORY_TO_DPV,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_protected_doc():
    """A realistic document with data protection annotations."""
    return {
        "@context": {"ex": "http://example.org/"},
        "@id": "ex:record/1",
        "@type": "Person",
        "name": annotate_protection(
            "Alice Smith",
            personal_data_category="regular",
            legal_basis="consent",
            data_controller="https://example.org/corp/acme",
            data_processor="https://example.org/cloud/provider",
            data_subject="https://example.org/subjects/alice",
            jurisdiction="EU",
            retention_until="2027-01-01T00:00:00Z",
            access_level="restricted",
            consent=create_consent_record(
                given_at="2025-06-15T10:00:00Z",
                scope=["Marketing", "Analytics"],
                granularity="specific",
            ),
        ),
        "diagnosis": annotate_protection(
            "common cold",
            personal_data_category="sensitive",
            legal_basis="vital_interest",
            data_subject="https://example.org/subjects/alice",
            data_controller="https://example.org/hospital",
        ),
    }


def _make_erasure_doc():
    """A document with erasure request annotations."""
    return {
        "@context": {"ex": "http://example.org/"},
        "@id": "ex:record/2",
        "@type": "Person",
        "email": annotate_protection(
            "alice@example.com",
            personal_data_category="regular",
            legal_basis="consent",
            data_subject="https://example.org/subjects/alice",
            erasure_requested=True,
            erasure_requested_at="2026-02-13T10:00:00Z",
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# Namespace constants
# ═══════════════════════════════════════════════════════════════════


class TestNamespaces:
    """Verify DPV namespace URIs are correct."""

    def test_dpv_namespace(self):
        assert DPV == "https://w3id.org/dpv#"

    def test_eu_gdpr_namespace(self):
        assert EU_GDPR == "https://w3id.org/dpv/legal/eu/gdpr#"

    def test_dpv_loc_namespace(self):
        assert DPV_LOC == "https://w3id.org/dpv/loc#"

    def test_dpv_pd_namespace(self):
        assert DPV_PD == "https://w3id.org/dpv/pd#"


# ═══════════════════════════════════════════════════════════════════
# Mapping constants
# ═══════════════════════════════════════════════════════════════════


class TestMappingConstants:
    """Verify legal basis and category mapping tables."""

    def test_all_legal_bases_mapped(self):
        expected_keys = {
            "consent", "contract", "legal_obligation",
            "vital_interest", "public_task", "legitimate_interest",
        }
        assert set(LEGAL_BASIS_TO_DPV.keys()) == expected_keys

    def test_legal_basis_consent_maps_to_a6_1_a(self):
        assert LEGAL_BASIS_TO_DPV["consent"] == f"{EU_GDPR}A6-1-a"

    def test_legal_basis_contract_maps_to_a6_1_b(self):
        assert LEGAL_BASIS_TO_DPV["contract"] == f"{EU_GDPR}A6-1-b"

    def test_legal_basis_legal_obligation_maps_to_a6_1_c(self):
        assert LEGAL_BASIS_TO_DPV["legal_obligation"] == f"{EU_GDPR}A6-1-c"

    def test_legal_basis_vital_interest_maps_to_a6_1_d(self):
        assert LEGAL_BASIS_TO_DPV["vital_interest"] == f"{EU_GDPR}A6-1-d"

    def test_legal_basis_public_task_maps_to_a6_1_e(self):
        assert LEGAL_BASIS_TO_DPV["public_task"] == f"{EU_GDPR}A6-1-e"

    def test_legal_basis_legitimate_interest_maps_to_a6_1_f(self):
        assert LEGAL_BASIS_TO_DPV["legitimate_interest"] == f"{EU_GDPR}A6-1-f"

    def test_category_sensitive_mapped(self):
        assert "sensitive" in CATEGORY_TO_DPV
        assert CATEGORY_TO_DPV["sensitive"] == f"{DPV_PD}SensitivePersonalData"

    def test_category_special_mapped(self):
        assert CATEGORY_TO_DPV["special_category"] == f"{DPV_PD}SpecialCategoryPersonalData"

    def test_category_regular_mapped(self):
        assert "regular" in CATEGORY_TO_DPV

    def test_category_pseudonymized_mapped(self):
        assert "pseudonymized" in CATEGORY_TO_DPV


# ═══════════════════════════════════════════════════════════════════
# to_dpv()
# ═══════════════════════════════════════════════════════════════════


class TestToDpv:
    """Tests for to_dpv() — jsonld-ex → DPV conversion."""

    def test_returns_tuple(self):
        doc = _make_protected_doc()
        result = to_dpv(doc)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_conversion_report(self):
        doc = _make_protected_doc()
        dpv_doc, report = to_dpv(doc)
        assert isinstance(report, ConversionReport)
        assert report.success is True

    def test_output_has_dpv_context(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        ctx = dpv_doc.get("@context", {})
        assert "dpv" in ctx
        assert ctx["dpv"] == DPV

    def test_maps_legal_basis_to_gdpr_article(self):
        """@legalBasis=consent → eu-gdpr:A6-1-a."""
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        graph = dpv_doc.get("@graph", [])
        # Find a node with legal basis
        legal_bases = [
            n for n in graph
            if n.get("@type") in (f"{EU_GDPR}A6-1-a", "eu-gdpr:A6-1-a")
            or n.get("dpv:hasLegalBasis", {}).get("@id", "").endswith("A6-1-a")
            or any(
                "A6-1-a" in str(v)
                for v in _flatten_values(n, "dpv:hasLegalBasis")
            )
        ]
        # At least one reference to consent legal basis
        all_text = str(dpv_doc)
        assert "A6-1-a" in all_text

    def test_maps_data_controller(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        assert "DataController" in all_text
        assert "https://example.org/corp/acme" in all_text

    def test_maps_data_processor(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        assert "DataProcessor" in all_text
        assert "https://example.org/cloud/provider" in all_text

    def test_maps_sensitive_category(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        assert "SensitivePersonalData" in all_text

    def test_maps_consent_record(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        # Should reference consent and provision time
        assert "Consent" in all_text
        assert "2025-06-15" in all_text

    def test_maps_retention(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        assert "2027-01-01" in all_text

    def test_maps_jurisdiction(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        assert "EU" in all_text

    def test_maps_erasure_request(self):
        doc = _make_erasure_doc()
        dpv_doc, _ = to_dpv(doc)
        all_text = str(dpv_doc)
        # GDPR Art. 17 right to erasure
        assert "A17" in all_text

    def test_report_counts_nodes(self):
        doc = _make_protected_doc()
        _, report = to_dpv(doc)
        assert report.nodes_converted >= 1

    def test_report_counts_triples(self):
        doc = _make_protected_doc()
        _, report = to_dpv(doc)
        assert report.triples_output > 0

    def test_empty_doc(self):
        dpv_doc, report = to_dpv({})
        assert report.success is True
        assert report.nodes_converted == 0


# ═══════════════════════════════════════════════════════════════════
# from_dpv()
# ═══════════════════════════════════════════════════════════════════


class TestFromDpv:
    """Tests for from_dpv() — DPV → jsonld-ex round-trip."""

    def test_round_trip_legal_basis(self):
        """consent → to_dpv → from_dpv → consent."""
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        restored, report = from_dpv(dpv_doc)
        assert report.success is True
        # Find a property with legal basis
        all_text = str(restored)
        assert "consent" in all_text

    def test_round_trip_category(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        restored, _ = from_dpv(dpv_doc)
        all_text = str(restored)
        assert "sensitive" in all_text

    def test_round_trip_data_controller(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        restored, _ = from_dpv(dpv_doc)
        all_text = str(restored)
        assert "https://example.org/corp/acme" in all_text

    def test_round_trip_erasure(self):
        doc = _make_erasure_doc()
        dpv_doc, _ = to_dpv(doc)
        restored, _ = from_dpv(dpv_doc)
        all_text = str(restored)
        assert "erasure" in all_text.lower() or "A17" in all_text

    def test_empty_dpv_doc(self):
        restored, report = from_dpv({})
        assert report.success is True
        assert report.nodes_converted == 0

    def test_report_success(self):
        doc = _make_protected_doc()
        dpv_doc, _ = to_dpv(doc)
        _, report = from_dpv(dpv_doc)
        assert isinstance(report, ConversionReport)
        assert report.success is True


# ═══════════════════════════════════════════════════════════════════
# compare_with_dpv()
# ═══════════════════════════════════════════════════════════════════


class TestCompareWithDpv:
    """Tests for compare_with_dpv() — verbosity comparison."""

    def test_returns_verbosity_comparison(self):
        doc = _make_protected_doc()
        comp = compare_with_dpv(doc)
        assert isinstance(comp, VerbosityComparison)

    def test_alternative_name(self):
        doc = _make_protected_doc()
        comp = compare_with_dpv(doc)
        assert comp.alternative_name == "DPV"

    def test_dpv_is_more_verbose(self):
        """DPV representation should be more verbose than jsonld-ex."""
        doc = _make_protected_doc()
        comp = compare_with_dpv(doc)
        assert comp.alternative_bytes >= comp.jsonld_ex_bytes

    def test_triple_counts_positive(self):
        doc = _make_protected_doc()
        comp = compare_with_dpv(doc)
        assert comp.jsonld_ex_triples > 0
        assert comp.alternative_triples > 0

    def test_empty_doc(self):
        comp = compare_with_dpv({})
        assert comp.jsonld_ex_triples == 0
        assert comp.alternative_triples == 0


# ── Helper for flexible graph searching ───────────────────────────

def _flatten_values(node: dict, key: str) -> list:
    """Get all values for a key, whether single or list."""
    val = node.get(key)
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]
