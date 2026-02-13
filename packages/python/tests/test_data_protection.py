"""Tests for data_protection module — Phase 1: Core Annotations & Data Classification.

TDD: Write tests FIRST, then implement to make them pass.
"""

import pytest
from datetime import datetime, timezone

from jsonld_ex.data_protection import (
    # Core annotation function
    annotate_protection,
    # Extraction
    get_protection_metadata,
    DataProtectionMetadata,
    # Consent lifecycle
    ConsentRecord,
    create_consent_record,
    is_consent_active,
    # Classification helpers
    is_personal_data,
    is_sensitive_data,
    # Graph filtering
    filter_by_jurisdiction,
    filter_personal_data,
    # Constants
    LEGAL_BASES,
    PERSONAL_DATA_CATEGORIES,
    CONSENT_GRANULARITIES,
    ACCESS_LEVELS,
)


# ═══════════════════════════════════════════════════════════════════
# annotate_protection() — Core annotation function
# ═══════════════════════════════════════════════════════════════════


class TestAnnotateProtection:
    """Tests for annotate_protection()."""

    def test_minimal_annotation(self):
        """Annotate with just a value — no protection fields."""
        result = annotate_protection("John Doe")
        assert result["@value"] == "John Doe"
        # No protection fields added when none specified
        assert "@personalDataCategory" not in result

    def test_personal_data_category(self):
        result = annotate_protection(
            "John Doe",
            personal_data_category="regular",
        )
        assert result["@value"] == "John Doe"
        assert result["@personalDataCategory"] == "regular"

    def test_all_personal_data_categories(self):
        """All valid categories accepted."""
        for cat in PERSONAL_DATA_CATEGORIES:
            result = annotate_protection("x", personal_data_category=cat)
            assert result["@personalDataCategory"] == cat

    def test_invalid_personal_data_category(self):
        with pytest.raises(ValueError, match="Invalid personal data category"):
            annotate_protection("x", personal_data_category="invalid")

    def test_legal_basis(self):
        result = annotate_protection(
            "John Doe",
            legal_basis="consent",
        )
        assert result["@legalBasis"] == "consent"

    def test_all_legal_bases(self):
        for basis in LEGAL_BASES:
            result = annotate_protection("x", legal_basis=basis)
            assert result["@legalBasis"] == basis

    def test_invalid_legal_basis(self):
        with pytest.raises(ValueError, match="Invalid legal basis"):
            annotate_protection("x", legal_basis="because_i_want_to")

    def test_processing_purpose_string(self):
        result = annotate_protection(
            "John Doe",
            processing_purpose="Marketing communications",
        )
        assert result["@processingPurpose"] == "Marketing communications"

    def test_processing_purpose_iri(self):
        result = annotate_protection(
            "John Doe",
            processing_purpose="https://w3id.org/dpv#ServiceProvision",
        )
        assert result["@processingPurpose"] == "https://w3id.org/dpv#ServiceProvision"

    def test_processing_purpose_list(self):
        """Multiple purposes."""
        result = annotate_protection(
            "John Doe",
            processing_purpose=["Marketing", "Analytics"],
        )
        assert result["@processingPurpose"] == ["Marketing", "Analytics"]

    def test_data_controller(self):
        result = annotate_protection(
            "John Doe",
            data_controller="https://example.org/company",
        )
        assert result["@dataController"] == "https://example.org/company"

    def test_data_processor(self):
        result = annotate_protection(
            "John Doe",
            data_processor="https://cloud-provider.example.org",
        )
        assert result["@dataProcessor"] == "https://cloud-provider.example.org"

    def test_data_subject(self):
        result = annotate_protection(
            "John Doe",
            data_subject="https://example.org/users/123",
        )
        assert result["@dataSubject"] == "https://example.org/users/123"

    def test_retention_until(self):
        result = annotate_protection(
            "John Doe",
            retention_until="2027-01-01T00:00:00Z",
        )
        assert result["@retentionUntil"] == "2027-01-01T00:00:00Z"

    def test_jurisdiction(self):
        result = annotate_protection(
            "John Doe",
            jurisdiction="EU",
        )
        assert result["@jurisdiction"] == "EU"

    def test_jurisdiction_iso_code(self):
        result = annotate_protection("x", jurisdiction="DE")
        assert result["@jurisdiction"] == "DE"

    def test_access_level(self):
        result = annotate_protection("x", access_level="restricted")
        assert result["@accessLevel"] == "restricted"

    def test_all_access_levels(self):
        for level in ACCESS_LEVELS:
            result = annotate_protection("x", access_level=level)
            assert result["@accessLevel"] == level

    def test_invalid_access_level(self):
        with pytest.raises(ValueError, match="Invalid access level"):
            annotate_protection("x", access_level="top_secret_ultra")

    def test_all_fields_together(self):
        """All Phase 1 fields set at once."""
        result = annotate_protection(
            "John Doe",
            personal_data_category="sensitive",
            legal_basis="consent",
            processing_purpose="Healthcare provision",
            data_controller="https://hospital.example.org",
            data_processor="https://cloud.example.org",
            data_subject="https://example.org/patients/456",
            retention_until="2030-12-31T23:59:59Z",
            jurisdiction="EU",
            access_level="confidential",
        )
        assert result["@value"] == "John Doe"
        assert result["@personalDataCategory"] == "sensitive"
        assert result["@legalBasis"] == "consent"
        assert result["@processingPurpose"] == "Healthcare provision"
        assert result["@dataController"] == "https://hospital.example.org"
        assert result["@dataProcessor"] == "https://cloud.example.org"
        assert result["@dataSubject"] == "https://example.org/patients/456"
        assert result["@retentionUntil"] == "2030-12-31T23:59:59Z"
        assert result["@jurisdiction"] == "EU"
        assert result["@accessLevel"] == "confidential"

    def test_dict_value_passthrough(self):
        """Works with dict values (existing annotated nodes)."""
        result = annotate_protection(
            {"@value": "John Doe", "@confidence": 0.95},
            personal_data_category="regular",
        )
        # Should wrap the dict, preserving existing fields
        assert result["@value"] == {"@value": "John Doe", "@confidence": 0.95}
        assert result["@personalDataCategory"] == "regular"

    def test_none_value(self):
        """None value is allowed (marking structure, not a literal)."""
        result = annotate_protection(None, personal_data_category="regular")
        assert result["@value"] is None
        assert result["@personalDataCategory"] == "regular"

    def test_numeric_value(self):
        result = annotate_protection(42, personal_data_category="regular")
        assert result["@value"] == 42

    def test_no_fields_returns_bare_value(self):
        """If no protection fields specified, still returns @value wrapper."""
        result = annotate_protection("test")
        assert result == {"@value": "test"}


# ═══════════════════════════════════════════════════════════════════
# Consent lifecycle
# ═══════════════════════════════════════════════════════════════════


class TestConsentRecord:
    """Tests for ConsentRecord and consent lifecycle functions."""

    def test_create_consent_record_minimal(self):
        record = create_consent_record(
            given_at="2025-01-15T10:00:00Z",
            scope=["Marketing"],
        )
        assert record["@consentGivenAt"] == "2025-01-15T10:00:00Z"
        assert record["@consentScope"] == ["Marketing"]
        assert "@consentWithdrawnAt" not in record
        assert "@consentGranularity" not in record

    def test_create_consent_record_full(self):
        record = create_consent_record(
            given_at="2025-01-15T10:00:00Z",
            scope=["Marketing", "Analytics"],
            granularity="specific",
            withdrawn_at=None,
        )
        assert record["@consentGivenAt"] == "2025-01-15T10:00:00Z"
        assert record["@consentScope"] == ["Marketing", "Analytics"]
        assert record["@consentGranularity"] == "specific"
        assert "@consentWithdrawnAt" not in record

    def test_create_consent_record_withdrawn(self):
        record = create_consent_record(
            given_at="2025-01-15T10:00:00Z",
            scope=["Marketing"],
            withdrawn_at="2025-06-01T12:00:00Z",
        )
        assert record["@consentWithdrawnAt"] == "2025-06-01T12:00:00Z"

    def test_all_consent_granularities(self):
        for g in CONSENT_GRANULARITIES:
            record = create_consent_record(
                given_at="2025-01-01T00:00:00Z",
                scope=["test"],
                granularity=g,
            )
            assert record["@consentGranularity"] == g

    def test_invalid_consent_granularity(self):
        with pytest.raises(ValueError, match="Invalid consent granularity"):
            create_consent_record(
                given_at="2025-01-01T00:00:00Z",
                scope=["test"],
                granularity="ultra_specific",
            )

    def test_empty_scope_raises(self):
        with pytest.raises(ValueError, match="Consent scope must not be empty"):
            create_consent_record(
                given_at="2025-01-01T00:00:00Z",
                scope=[],
            )

    def test_scope_single_string(self):
        """Single string scope gets wrapped in list."""
        record = create_consent_record(
            given_at="2025-01-01T00:00:00Z",
            scope="Marketing",
        )
        assert record["@consentScope"] == ["Marketing"]

    def test_annotate_with_consent(self):
        """Consent record can be attached via annotate_protection."""
        consent = create_consent_record(
            given_at="2025-01-15T10:00:00Z",
            scope=["Marketing"],
        )
        result = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            legal_basis="consent",
            consent=consent,
        )
        assert result["@consent"] == consent


class TestIsConsentActive:
    """Tests for is_consent_active()."""

    def test_active_consent_no_withdrawal(self):
        record = {
            "@consentGivenAt": "2025-01-15T10:00:00Z",
            "@consentScope": ["Marketing"],
        }
        assert is_consent_active(record) is True

    def test_withdrawn_consent(self):
        record = {
            "@consentGivenAt": "2025-01-15T10:00:00Z",
            "@consentScope": ["Marketing"],
            "@consentWithdrawnAt": "2025-06-01T12:00:00Z",
        }
        assert is_consent_active(record) is False

    def test_active_at_specific_time_before_withdrawal(self):
        record = {
            "@consentGivenAt": "2025-01-15T10:00:00Z",
            "@consentScope": ["Marketing"],
            "@consentWithdrawnAt": "2025-06-01T12:00:00Z",
        }
        # Check at a time before withdrawal
        assert is_consent_active(record, at_time="2025-03-01T00:00:00Z") is True

    def test_inactive_at_specific_time_after_withdrawal(self):
        record = {
            "@consentGivenAt": "2025-01-15T10:00:00Z",
            "@consentScope": ["Marketing"],
            "@consentWithdrawnAt": "2025-06-01T12:00:00Z",
        }
        assert is_consent_active(record, at_time="2025-07-01T00:00:00Z") is False

    def test_inactive_before_consent_given(self):
        record = {
            "@consentGivenAt": "2025-01-15T10:00:00Z",
            "@consentScope": ["Marketing"],
        }
        assert is_consent_active(record, at_time="2024-12-01T00:00:00Z") is False

    def test_none_record(self):
        assert is_consent_active(None) is False

    def test_empty_dict(self):
        assert is_consent_active({}) is False

    def test_missing_given_at(self):
        record = {"@consentScope": ["Marketing"]}
        assert is_consent_active(record) is False


# ═══════════════════════════════════════════════════════════════════
# get_protection_metadata() — Extraction
# ═══════════════════════════════════════════════════════════════════


class TestGetProtectionMetadata:
    """Tests for get_protection_metadata()."""

    def test_full_extraction(self):
        node = {
            "@value": "John Doe",
            "@personalDataCategory": "sensitive",
            "@legalBasis": "consent",
            "@processingPurpose": "Healthcare",
            "@dataController": "https://hospital.example.org",
            "@dataProcessor": "https://cloud.example.org",
            "@dataSubject": "https://example.org/patients/456",
            "@retentionUntil": "2030-12-31T23:59:59Z",
            "@jurisdiction": "EU",
            "@accessLevel": "confidential",
            "@consent": {
                "@consentGivenAt": "2025-01-15T10:00:00Z",
                "@consentScope": ["Healthcare"],
            },
        }
        meta = get_protection_metadata(node)
        assert isinstance(meta, DataProtectionMetadata)
        assert meta.personal_data_category == "sensitive"
        assert meta.legal_basis == "consent"
        assert meta.processing_purpose == "Healthcare"
        assert meta.data_controller == "https://hospital.example.org"
        assert meta.data_processor == "https://cloud.example.org"
        assert meta.data_subject == "https://example.org/patients/456"
        assert meta.retention_until == "2030-12-31T23:59:59Z"
        assert meta.jurisdiction == "EU"
        assert meta.access_level == "confidential"
        assert meta.consent is not None
        assert meta.consent["@consentGivenAt"] == "2025-01-15T10:00:00Z"

    def test_empty_node(self):
        meta = get_protection_metadata({})
        assert meta.personal_data_category is None
        assert meta.legal_basis is None
        assert meta.processing_purpose is None
        assert meta.data_controller is None
        assert meta.consent is None

    def test_none_node(self):
        meta = get_protection_metadata(None)
        assert isinstance(meta, DataProtectionMetadata)
        assert meta.personal_data_category is None

    def test_non_dict_node(self):
        meta = get_protection_metadata("not a dict")
        assert isinstance(meta, DataProtectionMetadata)
        assert meta.personal_data_category is None

    def test_partial_extraction(self):
        node = {
            "@value": "data",
            "@personalDataCategory": "regular",
            "@jurisdiction": "US",
        }
        meta = get_protection_metadata(node)
        assert meta.personal_data_category == "regular"
        assert meta.jurisdiction == "US"
        assert meta.legal_basis is None
        assert meta.data_controller is None


# ═══════════════════════════════════════════════════════════════════
# Classification helpers
# ═══════════════════════════════════════════════════════════════════


class TestIsPersonalData:
    """Tests for is_personal_data()."""

    def test_regular_personal_data(self):
        node = {"@value": "John", "@personalDataCategory": "regular"}
        assert is_personal_data(node) is True

    def test_sensitive_personal_data(self):
        node = {"@value": "diagnosis", "@personalDataCategory": "sensitive"}
        assert is_personal_data(node) is True

    def test_special_category(self):
        node = {"@value": "ethnicity", "@personalDataCategory": "special_category"}
        assert is_personal_data(node) is True

    def test_pseudonymized_is_personal(self):
        """Pseudonymized data is still personal data under GDPR."""
        node = {"@value": "hash-abc", "@personalDataCategory": "pseudonymized"}
        assert is_personal_data(node) is True

    def test_anonymized_is_not_personal(self):
        """Anonymized data is NOT personal data under GDPR."""
        node = {"@value": "stats", "@personalDataCategory": "anonymized"}
        assert is_personal_data(node) is False

    def test_synthetic_is_not_personal(self):
        node = {"@value": "generated", "@personalDataCategory": "synthetic"}
        assert is_personal_data(node) is False

    def test_non_personal(self):
        node = {"@value": "weather", "@personalDataCategory": "non_personal"}
        assert is_personal_data(node) is False

    def test_no_category_returns_false(self):
        node = {"@value": "unknown"}
        assert is_personal_data(node) is False

    def test_none_node(self):
        assert is_personal_data(None) is False

    def test_non_dict(self):
        assert is_personal_data("string") is False


class TestIsSensitiveData:
    """Tests for is_sensitive_data()."""

    def test_sensitive(self):
        node = {"@value": "x", "@personalDataCategory": "sensitive"}
        assert is_sensitive_data(node) is True

    def test_special_category(self):
        node = {"@value": "x", "@personalDataCategory": "special_category"}
        assert is_sensitive_data(node) is True

    def test_regular_is_not_sensitive(self):
        node = {"@value": "x", "@personalDataCategory": "regular"}
        assert is_sensitive_data(node) is False

    def test_no_category(self):
        assert is_sensitive_data({"@value": "x"}) is False

    def test_none(self):
        assert is_sensitive_data(None) is False


# ═══════════════════════════════════════════════════════════════════
# Graph filtering
# ═══════════════════════════════════════════════════════════════════


class TestFilterByJurisdiction:
    """Tests for filter_by_jurisdiction()."""

    def test_filter_eu(self):
        graph = [
            {"@id": "n1", "name": annotate_protection("Alice", jurisdiction="EU")},
            {"@id": "n2", "name": annotate_protection("Bob", jurisdiction="US")},
            {"@id": "n3", "name": annotate_protection("Carol", jurisdiction="EU")},
        ]
        results = filter_by_jurisdiction(graph, "name", "EU")
        assert len(results) == 2
        assert results[0]["@id"] == "n1"
        assert results[1]["@id"] == "n3"

    def test_filter_no_matches(self):
        graph = [
            {"@id": "n1", "name": annotate_protection("Alice", jurisdiction="US")},
        ]
        results = filter_by_jurisdiction(graph, "name", "EU")
        assert len(results) == 0

    def test_filter_empty_graph(self):
        results = filter_by_jurisdiction([], "name", "EU")
        assert results == []

    def test_filter_missing_property(self):
        graph = [
            {"@id": "n1", "age": 30},
        ]
        results = filter_by_jurisdiction(graph, "name", "EU")
        assert len(results) == 0

    def test_filter_list_values(self):
        """Property with list of annotated values."""
        graph = [
            {
                "@id": "n1",
                "names": [
                    annotate_protection("Alice", jurisdiction="EU"),
                    annotate_protection("Alicia", jurisdiction="ES"),
                ],
            },
        ]
        # Node has at least one value in EU jurisdiction
        results = filter_by_jurisdiction(graph, "names", "EU")
        assert len(results) == 1


class TestFilterPersonalData:
    """Tests for filter_personal_data()."""

    def test_filter_personal(self):
        graph = [
            {"@id": "n1", "name": annotate_protection("Alice", personal_data_category="regular")},
            {"@id": "n2", "temp": annotate_protection(22.5, personal_data_category="non_personal")},
            {"@id": "n3", "diagnosis": annotate_protection("flu", personal_data_category="sensitive")},
        ]
        results = filter_personal_data(graph, "name")
        assert len(results) == 1
        assert results[0]["@id"] == "n1"

    def test_filter_across_all_properties(self):
        """When property_name is None, check all properties."""
        graph = [
            {"@id": "n1", "name": annotate_protection("Alice", personal_data_category="regular")},
            {"@id": "n2", "temp": annotate_protection(22.5, personal_data_category="non_personal")},
            {"@id": "n3", "diagnosis": annotate_protection("flu", personal_data_category="sensitive")},
        ]
        results = filter_personal_data(graph)
        assert len(results) == 2  # n1 (regular) and n3 (sensitive)
        ids = [r["@id"] for r in results]
        assert "n1" in ids
        assert "n3" in ids

    def test_filter_empty_graph(self):
        assert filter_personal_data([]) == []

    def test_filter_no_personal_data(self):
        graph = [
            {"@id": "n1", "temp": annotate_protection(22.5, personal_data_category="anonymized")},
        ]
        results = filter_personal_data(graph)
        assert len(results) == 0


# ═══════════════════════════════════════════════════════════════════
# Composability with existing annotate()
# ═══════════════════════════════════════════════════════════════════


class TestComposability:
    """Verify that data protection annotations compose with ai_ml.annotate()."""

    def test_protection_output_is_plain_dict(self):
        """Output is a plain dict, compatible with JSON serialization."""
        result = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            legal_basis="consent",
        )
        assert isinstance(result, dict)
        assert "@value" in result

    def test_merge_with_ai_ml_annotate(self):
        """Protection dict can be merged with ai_ml annotate output."""
        from jsonld_ex.ai_ml import annotate

        provenance = annotate("John Doe", confidence=0.95, source="model-v1")
        protection = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            legal_basis="consent",
        )
        # Merge: provenance fields + protection fields
        merged = {**provenance, **protection}
        assert merged["@value"] == "John Doe"
        assert merged["@confidence"] == 0.95
        assert merged["@source"] == "model-v1"
        assert merged["@personalDataCategory"] == "regular"
        assert merged["@legalBasis"] == "consent"

    def test_get_protection_from_merged_node(self):
        """get_protection_metadata works on merged provenance+protection nodes."""
        merged = {
            "@value": "John Doe",
            "@confidence": 0.95,
            "@source": "model-v1",
            "@personalDataCategory": "regular",
            "@legalBasis": "consent",
            "@jurisdiction": "EU",
        }
        meta = get_protection_metadata(merged)
        assert meta.personal_data_category == "regular"
        assert meta.legal_basis == "consent"
        assert meta.jurisdiction == "EU"

    def test_ai_ml_get_provenance_ignores_protection_fields(self):
        """Existing get_provenance() is unaffected by protection fields."""
        from jsonld_ex.ai_ml import get_provenance

        merged = {
            "@value": "John Doe",
            "@confidence": 0.95,
            "@personalDataCategory": "regular",
            "@legalBasis": "consent",
        }
        prov = get_provenance(merged)
        assert prov.confidence == 0.95
        # Protection fields don't interfere
        assert prov.source is None


# ═══════════════════════════════════════════════════════════════════
# DataProtectionMetadata dataclass
# ═══════════════════════════════════════════════════════════════════


class TestDataProtectionMetadata:
    """Tests for DataProtectionMetadata dataclass."""

    def test_default_all_none(self):
        meta = DataProtectionMetadata()
        assert meta.personal_data_category is None
        assert meta.legal_basis is None
        assert meta.processing_purpose is None
        assert meta.data_controller is None
        assert meta.data_processor is None
        assert meta.data_subject is None
        assert meta.retention_until is None
        assert meta.jurisdiction is None
        assert meta.access_level is None
        assert meta.consent is None

    def test_equality(self):
        a = DataProtectionMetadata(personal_data_category="regular", jurisdiction="EU")
        b = DataProtectionMetadata(personal_data_category="regular", jurisdiction="EU")
        assert a == b

    def test_inequality(self):
        a = DataProtectionMetadata(personal_data_category="regular")
        b = DataProtectionMetadata(personal_data_category="sensitive")
        assert a != b


# ═══════════════════════════════════════════════════════════════════
# Constants validation
# ═══════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify expected constant values."""

    def test_legal_bases_complete(self):
        expected = {
            "consent", "contract", "legal_obligation",
            "vital_interest", "public_task", "legitimate_interest",
        }
        assert set(LEGAL_BASES) == expected

    def test_personal_data_categories_complete(self):
        expected = {
            "regular", "sensitive", "special_category",
            "anonymized", "pseudonymized", "synthetic", "non_personal",
        }
        assert set(PERSONAL_DATA_CATEGORIES) == expected

    def test_consent_granularities_complete(self):
        expected = {"broad", "specific", "granular"}
        assert set(CONSENT_GRANULARITIES) == expected

    def test_access_levels_complete(self):
        expected = {"public", "internal", "restricted", "confidential", "secret"}
        assert set(ACCESS_LEVELS) == expected


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Data Subject Rights — Annotation Field Extensions
# ═══════════════════════════════════════════════════════════════════


class TestAnnotateProtectionPhase2Fields:
    """Tests for Phase 2 annotation fields on annotate_protection()."""

    # ── Erasure fields ────────────────────────────────────────────

    def test_erasure_requested_true(self):
        result = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            erasure_requested=True,
        )
        assert result["@erasureRequested"] is True

    def test_erasure_requested_false(self):
        result = annotate_protection("x", erasure_requested=False)
        assert result["@erasureRequested"] is False

    def test_erasure_requested_at(self):
        result = annotate_protection(
            "John Doe",
            erasure_requested=True,
            erasure_requested_at="2026-02-13T10:00:00Z",
        )
        assert result["@erasureRequestedAt"] == "2026-02-13T10:00:00Z"

    def test_erasure_completed_at(self):
        result = annotate_protection(
            "John Doe",
            erasure_requested=True,
            erasure_completed_at="2026-02-20T10:00:00Z",
        )
        assert result["@erasureCompletedAt"] == "2026-02-20T10:00:00Z"

    # ── Restriction fields ────────────────────────────────────────

    def test_restrict_processing_true(self):
        result = annotate_protection(
            "John Doe",
            restrict_processing=True,
        )
        assert result["@restrictProcessing"] is True

    def test_restriction_reason(self):
        result = annotate_protection(
            "John Doe",
            restrict_processing=True,
            restriction_reason="Accuracy contested by data subject",
        )
        assert result["@restrictionReason"] == "Accuracy contested by data subject"

    def test_processing_restrictions_list(self):
        result = annotate_protection(
            "John Doe",
            processing_restrictions=["profiling", "automated_decision"],
        )
        assert result["@processingRestrictions"] == ["profiling", "automated_decision"]

    def test_processing_restrictions_single_string(self):
        """Single string is accepted (not auto-wrapped — user must pass list)."""
        result = annotate_protection(
            "John Doe",
            processing_restrictions=["profiling"],
        )
        assert result["@processingRestrictions"] == ["profiling"]

    # ── Portability field ─────────────────────────────────────────

    def test_portability_format(self):
        result = annotate_protection(
            "John Doe",
            portability_format="application/json",
        )
        assert result["@portabilityFormat"] == "application/json"

    def test_portability_format_csv(self):
        result = annotate_protection("x", portability_format="text/csv")
        assert result["@portabilityFormat"] == "text/csv"

    # ── Rectification fields ──────────────────────────────────────

    def test_rectified_at(self):
        result = annotate_protection(
            "Jane Doe",
            rectified_at="2026-02-13T12:00:00Z",
        )
        assert result["@rectifiedAt"] == "2026-02-13T12:00:00Z"

    def test_rectification_note(self):
        result = annotate_protection(
            "Jane Doe",
            rectified_at="2026-02-13T12:00:00Z",
            rectification_note="Corrected spelling of surname",
        )
        assert result["@rectificationNote"] == "Corrected spelling of surname"

    # ── All Phase 2 fields together ───────────────────────────────

    def test_all_phase2_fields(self):
        result = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            legal_basis="consent",
            erasure_requested=True,
            erasure_requested_at="2026-02-13T10:00:00Z",
            erasure_completed_at="2026-02-20T10:00:00Z",
            restrict_processing=True,
            restriction_reason="Under review",
            processing_restrictions=["profiling", "marketing"],
            portability_format="application/json",
            rectified_at="2026-01-15T09:00:00Z",
            rectification_note="Name corrected",
        )
        assert result["@erasureRequested"] is True
        assert result["@erasureRequestedAt"] == "2026-02-13T10:00:00Z"
        assert result["@erasureCompletedAt"] == "2026-02-20T10:00:00Z"
        assert result["@restrictProcessing"] is True
        assert result["@restrictionReason"] == "Under review"
        assert result["@processingRestrictions"] == ["profiling", "marketing"]
        assert result["@portabilityFormat"] == "application/json"
        assert result["@rectifiedAt"] == "2026-01-15T09:00:00Z"
        assert result["@rectificationNote"] == "Name corrected"
        # Phase 1 fields still present
        assert result["@personalDataCategory"] == "regular"
        assert result["@legalBasis"] == "consent"

    def test_none_phase2_fields_excluded(self):
        """Phase 2 fields not present when not specified."""
        result = annotate_protection("x", personal_data_category="regular")
        assert "@erasureRequested" not in result
        assert "@erasureRequestedAt" not in result
        assert "@erasureCompletedAt" not in result
        assert "@restrictProcessing" not in result
        assert "@restrictionReason" not in result
        assert "@processingRestrictions" not in result
        assert "@portabilityFormat" not in result
        assert "@rectifiedAt" not in result
        assert "@rectificationNote" not in result


class TestGetProtectionMetadataPhase2:
    """Tests for Phase 2 fields in get_protection_metadata()."""

    def test_extract_phase2_fields(self):
        node = {
            "@value": "John Doe",
            "@personalDataCategory": "regular",
            "@erasureRequested": True,
            "@erasureRequestedAt": "2026-02-13T10:00:00Z",
            "@erasureCompletedAt": "2026-02-20T10:00:00Z",
            "@restrictProcessing": True,
            "@restrictionReason": "Under review",
            "@processingRestrictions": ["profiling"],
            "@portabilityFormat": "application/json",
            "@rectifiedAt": "2026-01-15T09:00:00Z",
            "@rectificationNote": "Name corrected",
        }
        meta = get_protection_metadata(node)
        assert meta.erasure_requested is True
        assert meta.erasure_requested_at == "2026-02-13T10:00:00Z"
        assert meta.erasure_completed_at == "2026-02-20T10:00:00Z"
        assert meta.restrict_processing is True
        assert meta.restriction_reason == "Under review"
        assert meta.processing_restrictions == ["profiling"]
        assert meta.portability_format == "application/json"
        assert meta.rectified_at == "2026-01-15T09:00:00Z"
        assert meta.rectification_note == "Name corrected"

    def test_phase2_fields_default_none(self):
        meta = get_protection_metadata({})
        assert meta.erasure_requested is None
        assert meta.erasure_requested_at is None
        assert meta.erasure_completed_at is None
        assert meta.restrict_processing is None
        assert meta.restriction_reason is None
        assert meta.processing_restrictions is None
        assert meta.portability_format is None
        assert meta.rectified_at is None
        assert meta.rectification_note is None

    def test_mixed_phase1_and_phase2(self):
        """Both Phase 1 and Phase 2 fields extracted together."""
        node = {
            "@value": "x",
            "@personalDataCategory": "sensitive",
            "@legalBasis": "consent",
            "@jurisdiction": "EU",
            "@erasureRequested": False,
            "@restrictProcessing": True,
        }
        meta = get_protection_metadata(node)
        # Phase 1
        assert meta.personal_data_category == "sensitive"
        assert meta.legal_basis == "consent"
        assert meta.jurisdiction == "EU"
        # Phase 2
        assert meta.erasure_requested is False
        assert meta.restrict_processing is True
