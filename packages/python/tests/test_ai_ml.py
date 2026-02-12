"""Tests for AI/ML extensions."""

import pytest
from jsonld_ex.ai_ml import (
    annotate, get_confidence, get_provenance,
    filter_by_confidence, aggregate_confidence,
)


class TestAnnotate:
    def test_basic_confidence(self):
        result = annotate("John Smith", confidence=0.95)
        assert result == {"@value": "John Smith", "@confidence": 0.95}

    def test_full_provenance(self):
        result = annotate(
            "John Smith",
            confidence=0.95,
            source="https://model.example.org/ner-v2",
            extracted_at="2026-01-15T10:30:00Z",
            method="NER",
            human_verified=False,
        )
        assert result["@confidence"] == 0.95
        assert result["@source"] == "https://model.example.org/ner-v2"
        assert result["@method"] == "NER"
        assert result["@humanVerified"] is False

    def test_rejects_invalid_confidence(self):
        with pytest.raises(ValueError):
            annotate("x", confidence=1.5)
        with pytest.raises(ValueError):
            annotate("x", confidence=-0.1)

    def test_numeric_value(self):
        result = annotate(42, confidence=0.8)
        assert result["@value"] == 42


class TestGetConfidence:
    def test_compact_form(self):
        assert get_confidence({"@value": "test", "@confidence": 0.9}) == 0.9

    def test_expanded_form(self):
        node = {
            "http://www.w3.org/ns/jsonld-ex/confidence": [{"@value": 0.85}]
        }
        assert get_confidence(node) == 0.85

    def test_missing(self):
        assert get_confidence({"@value": "test"}) is None
        assert get_confidence(None) is None


class TestGetProvenance:
    def test_extracts_all(self):
        node = {"@confidence": 0.9, "@source": "https://x.org/v1", "@method": "NER"}
        prov = get_provenance(node)
        assert prov.confidence == 0.9
        assert prov.source == "https://x.org/v1"
        assert prov.method == "NER"


class TestFilterByConfidence:
    graph = [
        {"@id": "#a", "name": {"@value": "Alice", "@confidence": 0.95}},
        {"@id": "#b", "name": {"@value": "Bob", "@confidence": 0.6}},
        {"@id": "#c", "name": {"@value": "Charlie", "@confidence": 0.3}},
    ]

    def test_filters_above_threshold(self):
        result = filter_by_confidence(self.graph, "name", 0.5)
        assert len(result) == 2
        assert result[0]["@id"] == "#a"

    def test_high_threshold(self):
        result = filter_by_confidence(self.graph, "name", 0.99)
        assert len(result) == 0


class TestAggregateConfidence:
    def test_mean(self):
        assert abs(aggregate_confidence([0.8, 0.6, 0.4]) - 0.6) < 1e-9

    def test_max(self):
        assert aggregate_confidence([0.8, 0.6, 0.4], "max") == 0.8

    def test_min(self):
        assert aggregate_confidence([0.8, 0.6, 0.4], "min") == 0.4

    def test_weighted(self):
        result = aggregate_confidence([0.9, 0.5], "weighted", [3, 1])
        assert abs(result - 0.8) < 1e-9

    def test_empty(self):
        assert aggregate_confidence([]) == 0.0

    def test_weighted_zero_weights_raises(self):
        with pytest.raises(ValueError, match="greater than zero"):
            aggregate_confidence([0.5, 0.5], "weighted", [0, 0])

    def test_weighted_negative_weight_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            aggregate_confidence([0.5, 0.5], "weighted", [-1, 2])

    def test_weighted_bool_weight_raises(self):
        with pytest.raises(TypeError, match="number"):
            aggregate_confidence([0.5, 0.5], "weighted", [True, 1])

    def test_weighted_mismatched_length(self):
        with pytest.raises(ValueError, match="match"):
            aggregate_confidence([0.5], "weighted", [1, 2])


class TestConfidenceEdgeCases:
    def test_nan_confidence_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            annotate("x", confidence=float("nan"))

    def test_inf_confidence_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            annotate("x", confidence=float("inf"))

    def test_neg_inf_confidence_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            annotate("x", confidence=float("-inf"))

    def test_bool_confidence_rejected(self):
        with pytest.raises(TypeError, match="number"):
            annotate("x", confidence=True)

    def test_string_confidence_rejected(self):
        with pytest.raises(TypeError, match="number"):
            annotate("x", confidence="0.5")

    def test_boundary_0(self):
        result = annotate("x", confidence=0.0)
        assert result["@confidence"] == 0.0

    def test_boundary_1(self):
        result = annotate("x", confidence=1.0)
        assert result["@confidence"] == 1.0

    def test_integer_confidence_accepted(self):
        result = annotate("x", confidence=1)
        assert result["@confidence"] == 1

    def test_get_confidence_non_dict(self):
        assert get_confidence(42) is None
        assert get_confidence("string") is None
        assert get_confidence([]) is None

    def test_get_provenance_non_dict(self):
        prov = get_provenance(42)
        assert prov.confidence is None

    def test_filter_empty_graph(self):
        assert filter_by_confidence([], "name", 0.5) == []

    def test_filter_missing_property(self):
        graph = [{"@id": "#a", "other": "value"}]
        assert filter_by_confidence(graph, "name", 0.5) == []

    def test_annotate_none_value(self):
        result = annotate(None, confidence=0.5)
        assert result["@value"] is None


# -- GAP-MM1: Multimodal annotations -----------------------------------------


class TestMultimodalAnnotations:
    """@mediaType, @contentUrl, @contentHash for non-text data."""

    def test_annotate_with_media_type(self):
        result = annotate(None, confidence=0.85, media_type="image/png")
        assert result["@mediaType"] == "image/png"
        assert result["@confidence"] == 0.85

    def test_annotate_with_content_url(self):
        result = annotate(None, confidence=0.9, content_url="s3://bucket/img.png")
        assert result["@contentUrl"] == "s3://bucket/img.png"

    def test_annotate_with_content_hash(self):
        result = annotate(
            None,
            confidence=0.95,
            content_hash="sha256-abc123def456",
        )
        assert result["@contentHash"] == "sha256-abc123def456"

    def test_annotate_full_multimodal(self):
        result = annotate(
            None,
            confidence=0.88,
            media_type="image/jpeg",
            content_url="https://cdn.example.org/photo.jpg",
            content_hash="sha256-deadbeef",
            source="https://model.example.org/v3",
            method="object-detection",
        )
        assert result["@mediaType"] == "image/jpeg"
        assert result["@contentUrl"] == "https://cdn.example.org/photo.jpg"
        assert result["@contentHash"] == "sha256-deadbeef"
        assert result["@source"] == "https://model.example.org/v3"
        assert result["@method"] == "object-detection"

    def test_get_provenance_multimodal(self):
        node = {
            "@value": None,
            "@confidence": 0.85,
            "@mediaType": "image/png",
            "@contentUrl": "s3://bucket/img.png",
            "@contentHash": "sha256-abc",
        }
        prov = get_provenance(node)
        assert prov.media_type == "image/png"
        assert prov.content_url == "s3://bucket/img.png"
        assert prov.content_hash == "sha256-abc"


# -- GAP-ML1: Language-specific confidence ------------------------------------


class TestLanguageConfidence:
    """annotate() works with @language-tagged values."""

    def test_annotate_preserves_language_dict(self):
        """When value is a dict with @value/@language, annotate merges."""
        lang_val = {"@value": "\u6771\u4eac\u30bf\u30ef\u30fc", "@language": "ja"}
        result = annotate(lang_val, confidence=0.70)
        # The value is the language dict itself
        assert result["@value"] == lang_val
        assert result["@confidence"] == 0.70

    def test_annotate_string_with_language_params(self):
        """Simple string annotation with confidence."""
        result = annotate("Tokyo Tower", confidence=0.95)
        assert result["@value"] == "Tokyo Tower"
        assert result["@confidence"] == 0.95


# -- GAP-ML2: Translation provenance ------------------------------------------


class TestTranslationProvenance:
    """@translatedFrom, @translationModel for translation tracking."""

    def test_annotate_with_translated_from(self):
        result = annotate(
            "Tokyo Tower",
            confidence=0.92,
            translated_from="ja",
        )
        assert result["@translatedFrom"] == "ja"

    def test_annotate_with_translation_model(self):
        result = annotate(
            "Tokyo Tower",
            confidence=0.92,
            translated_from="ja",
            translation_model="gpt-4-turbo",
        )
        assert result["@translatedFrom"] == "ja"
        assert result["@translationModel"] == "gpt-4-turbo"

    def test_get_provenance_translation(self):
        node = {
            "@value": "Tokyo Tower",
            "@confidence": 0.92,
            "@translatedFrom": "ja",
            "@translationModel": "gpt-4-turbo",
        }
        prov = get_provenance(node)
        assert prov.translated_from == "ja"
        assert prov.translation_model == "gpt-4-turbo"


# -- GAP-IOT1: Measurement uncertainty ----------------------------------------


class TestMeasurementUncertainty:
    """@measurementUncertainty and @unit for sensor/IoT data."""

    def test_annotate_with_uncertainty(self):
        result = annotate(
            23.5,
            confidence=0.99,
            measurement_uncertainty=0.5,
            unit="celsius",
        )
        assert result["@value"] == 23.5
        assert result["@measurementUncertainty"] == 0.5
        assert result["@unit"] == "celsius"

    def test_uncertainty_distinct_from_confidence(self):
        """@measurementUncertainty is numeric same-unit; @confidence is 0-1."""
        result = annotate(
            100.0,
            confidence=0.95,
            measurement_uncertainty=2.3,
            unit="kPa",
        )
        assert result["@confidence"] == 0.95
        assert result["@measurementUncertainty"] == 2.3
        assert result["@unit"] == "kPa"

    def test_get_provenance_uncertainty(self):
        node = {
            "@value": 23.5,
            "@confidence": 0.99,
            "@measurementUncertainty": 0.5,
            "@unit": "celsius",
        }
        prov = get_provenance(node)
        assert prov.measurement_uncertainty == 0.5
        assert prov.unit == "celsius"

    def test_uncertainty_without_confidence(self):
        result = annotate(23.5, measurement_uncertainty=0.5, unit="celsius")
        assert "@confidence" not in result
        assert result["@measurementUncertainty"] == 0.5
        assert result["@unit"] == "celsius"


# -- GAP-P2: Derivation tracking (@derivedFrom) ------------------------------


class TestDerivedFrom:
    """@derivedFrom for lineage/derivation tracking."""

    def test_annotate_single_source(self):
        result = annotate(
            "merged entity",
            confidence=0.85,
            derived_from="https://example.org/entity/42",
        )
        assert result["@derivedFrom"] == "https://example.org/entity/42"
        assert result["@confidence"] == 0.85

    def test_annotate_multiple_sources(self):
        sources = [
            "https://example.org/entity/1",
            "https://example.org/entity/2",
            "https://example.org/entity/3",
        ]
        result = annotate("fused value", confidence=0.90, derived_from=sources)
        assert result["@derivedFrom"] == sources

    def test_annotate_derived_from_without_confidence(self):
        result = annotate("derived value", derived_from="https://example.org/src")
        assert result["@derivedFrom"] == "https://example.org/src"
        assert "@confidence" not in result

    def test_annotate_derived_from_with_full_provenance(self):
        result = annotate(
            "John Smith",
            confidence=0.95,
            source="https://model.example.org/ner-v2",
            method="entity-resolution",
            derived_from=[
                "https://source-a.org/record/123",
                "https://source-b.org/record/456",
            ],
        )
        assert result["@confidence"] == 0.95
        assert result["@source"] == "https://model.example.org/ner-v2"
        assert result["@method"] == "entity-resolution"
        assert len(result["@derivedFrom"]) == 2

    def test_get_provenance_derived_from_single(self):
        node = {
            "@value": "test",
            "@confidence": 0.8,
            "@derivedFrom": "https://example.org/src",
        }
        prov = get_provenance(node)
        assert prov.derived_from == "https://example.org/src"

    def test_get_provenance_derived_from_list(self):
        sources = ["https://example.org/a", "https://example.org/b"]
        node = {
            "@value": "test",
            "@derivedFrom": sources,
        }
        prov = get_provenance(node)
        assert prov.derived_from == sources

    def test_get_provenance_derived_from_absent(self):
        node = {"@value": "test", "@confidence": 0.5}
        prov = get_provenance(node)
        assert prov.derived_from is None


# -- GAP-IOT4: Aggregation metadata -------------------------------------------


class TestAggregationMetadata:
    """@aggregationMethod, @aggregationWindow, @aggregationCount."""

    def test_annotate_with_aggregation_method(self):
        result = annotate(23.5, confidence=0.99, aggregation_method="mean")
        assert result["@aggregationMethod"] == "mean"

    def test_annotate_with_aggregation_window(self):
        result = annotate(23.5, aggregation_window="PT5M")
        assert result["@aggregationWindow"] == "PT5M"

    def test_annotate_with_aggregation_count(self):
        result = annotate(23.5, aggregation_count=100)
        assert result["@aggregationCount"] == 100

    def test_annotate_full_aggregation(self):
        result = annotate(
            23.5,
            confidence=0.99,
            measurement_uncertainty=0.5,
            unit="celsius",
            aggregation_method="mean",
            aggregation_window="PT5M",
            aggregation_count=100,
        )
        assert result["@value"] == 23.5
        assert result["@aggregationMethod"] == "mean"
        assert result["@aggregationWindow"] == "PT5M"
        assert result["@aggregationCount"] == 100
        assert result["@measurementUncertainty"] == 0.5
        assert result["@unit"] == "celsius"

    def test_aggregation_without_other_annotations(self):
        result = annotate(72.0, aggregation_method="p95", aggregation_count=500)
        assert result == {
            "@value": 72.0,
            "@aggregationMethod": "p95",
            "@aggregationCount": 500,
        }

    def test_get_provenance_aggregation(self):
        node = {
            "@value": 23.5,
            "@aggregationMethod": "median",
            "@aggregationWindow": "PT10M",
            "@aggregationCount": 200,
        }
        prov = get_provenance(node)
        assert prov.aggregation_method == "median"
        assert prov.aggregation_window == "PT10M"
        assert prov.aggregation_count == 200

    def test_get_provenance_aggregation_absent(self):
        node = {"@value": 23.5, "@confidence": 0.9}
        prov = get_provenance(node)
        assert prov.aggregation_method is None
        assert prov.aggregation_window is None
        assert prov.aggregation_count is None


# -- GAP-IOT2: Calibration metadata -------------------------------------------


class TestCalibrationMetadata:
    """@calibratedAt, @calibrationMethod, @calibrationAuthority."""

    def test_annotate_with_calibrated_at(self):
        result = annotate(23.5, calibrated_at="2025-06-01T09:00:00Z")
        assert result["@calibratedAt"] == "2025-06-01T09:00:00Z"

    def test_annotate_with_calibration_method(self):
        result = annotate(23.5, calibration_method="NIST-traceable two-point")
        assert result["@calibrationMethod"] == "NIST-traceable two-point"

    def test_annotate_with_calibration_authority(self):
        result = annotate(23.5, calibration_authority="https://nist.gov")
        assert result["@calibrationAuthority"] == "https://nist.gov"

    def test_annotate_full_calibration(self):
        result = annotate(
            23.5,
            confidence=0.99,
            measurement_uncertainty=0.5,
            unit="celsius",
            calibrated_at="2025-06-01T09:00:00Z",
            calibration_method="NIST-traceable two-point",
            calibration_authority="https://nist.gov",
        )
        assert result["@calibratedAt"] == "2025-06-01T09:00:00Z"
        assert result["@calibrationMethod"] == "NIST-traceable two-point"
        assert result["@calibrationAuthority"] == "https://nist.gov"
        assert result["@measurementUncertainty"] == 0.5
        assert result["@unit"] == "celsius"

    def test_calibration_composes_with_aggregation(self):
        result = annotate(
            23.5,
            measurement_uncertainty=0.5,
            unit="celsius",
            calibrated_at="2025-06-01T09:00:00Z",
            calibration_method="NIST-traceable two-point",
            aggregation_method="mean",
            aggregation_count=100,
        )
        assert result["@calibratedAt"] == "2025-06-01T09:00:00Z"
        assert result["@aggregationMethod"] == "mean"
        assert result["@aggregationCount"] == 100

    def test_get_provenance_calibration(self):
        node = {
            "@value": 23.5,
            "@calibratedAt": "2025-06-01T09:00:00Z",
            "@calibrationMethod": "NIST-traceable two-point",
            "@calibrationAuthority": "https://nist.gov",
        }
        prov = get_provenance(node)
        assert prov.calibrated_at == "2025-06-01T09:00:00Z"
        assert prov.calibration_method == "NIST-traceable two-point"
        assert prov.calibration_authority == "https://nist.gov"

    def test_get_provenance_calibration_absent(self):
        node = {"@value": 23.5, "@confidence": 0.9}
        prov = get_provenance(node)
        assert prov.calibrated_at is None
        assert prov.calibration_method is None
        assert prov.calibration_authority is None


# -- GAP-P1: Delegation chains ------------------------------------------------


class TestDelegationChains:
    """@delegatedBy for multi-agent provenance pipelines."""

    def test_annotate_with_delegated_by(self):
        result = annotate(
            "extracted entity",
            confidence=0.9,
            source="https://model.example.org/ner-v3",
            delegated_by="https://pipeline.example.org/etl-v2",
        )
        assert result["@delegatedBy"] == "https://pipeline.example.org/etl-v2"
        assert result["@source"] == "https://model.example.org/ner-v3"

    def test_annotate_delegation_chain_list(self):
        result = annotate(
            "value",
            source="https://model.example.org/v1",
            delegated_by=[
                "https://pipeline.example.org/step-2",
                "https://user.example.org/alice",
            ],
        )
        assert isinstance(result["@delegatedBy"], list)
        assert len(result["@delegatedBy"]) == 2

    def test_annotate_delegated_by_without_source(self):
        result = annotate("value", delegated_by="https://orchestrator.example.org")
        assert result["@delegatedBy"] == "https://orchestrator.example.org"
        assert "@source" not in result

    def test_get_provenance_delegated_by(self):
        node = {
            "@value": "test",
            "@source": "https://model.example.org/v1",
            "@delegatedBy": "https://pipeline.example.org/etl",
        }
        prov = get_provenance(node)
        assert prov.delegated_by == "https://pipeline.example.org/etl"

    def test_get_provenance_delegated_by_list(self):
        node = {
            "@value": "test",
            "@delegatedBy": ["https://a.org", "https://b.org"],
        }
        prov = get_provenance(node)
        assert prov.delegated_by == ["https://a.org", "https://b.org"]

    def test_get_provenance_delegated_by_absent(self):
        node = {"@value": "test", "@confidence": 0.5}
        prov = get_provenance(node)
        assert prov.delegated_by is None


# -- GAP-MM3: Content addressing (@contentHash) --------------------------------


class TestContentHash:
    """@contentHash for content-addressable annotations."""

    def test_annotate_with_content_hash(self):
        result = annotate(
            "Alice",
            confidence=0.95,
            content_hash="sha256:abc123def456",
        )
        assert result["@contentHash"] == "sha256:abc123def456"

    def test_annotate_content_hash_without_confidence(self):
        result = annotate("data", content_hash="sha256:deadbeef")
        assert result["@contentHash"] == "sha256:deadbeef"
        assert "@confidence" not in result

    def test_get_provenance_content_hash(self):
        node = {
            "@value": "Alice",
            "@confidence": 0.9,
            "@contentHash": "sha256:abc123",
        }
        prov = get_provenance(node)
        assert prov.content_hash == "sha256:abc123"

    def test_get_provenance_content_hash_absent(self):
        node = {"@value": "Alice", "@confidence": 0.9}
        prov = get_provenance(node)
        assert prov.content_hash is None

    def test_content_hash_composes_with_provenance(self):
        result = annotate(
            "fact",
            confidence=0.99,
            source="https://model.example.org/v3",
            content_hash="sha256:fedcba987654",
        )
        assert result["@contentHash"] == "sha256:fedcba987654"
        assert result["@source"] == "https://model.example.org/v3"
        assert result["@confidence"] == 0.99


# -- GAP-P3: Invalidation / retraction ----------------------------------------


class TestInvalidation:
    """@invalidatedAt / @invalidationReason for retracted assertions."""

    def test_annotate_with_invalidated_at(self):
        result = annotate(
            "old value",
            confidence=0.3,
            invalidated_at="2025-06-01T00:00:00Z",
        )
        assert result["@invalidatedAt"] == "2025-06-01T00:00:00Z"

    def test_annotate_with_invalidation_reason(self):
        result = annotate(
            "old value",
            confidence=0.3,
            invalidation_reason="Superseded by updated extraction",
        )
        assert result["@invalidationReason"] == "Superseded by updated extraction"

    def test_annotate_full_invalidation(self):
        result = annotate(
            "retracted claim",
            confidence=0.1,
            source="https://model.example.org/v1",
            invalidated_at="2025-07-01T12:00:00Z",
            invalidation_reason="Model v1 deprecated",
        )
        assert result["@invalidatedAt"] == "2025-07-01T12:00:00Z"
        assert result["@invalidationReason"] == "Model v1 deprecated"
        assert result["@source"] == "https://model.example.org/v1"

    def test_get_provenance_invalidation(self):
        node = {
            "@value": "old",
            "@confidence": 0.2,
            "@invalidatedAt": "2025-06-01T00:00:00Z",
            "@invalidationReason": "Corrected",
        }
        prov = get_provenance(node)
        assert prov.invalidated_at == "2025-06-01T00:00:00Z"
        assert prov.invalidation_reason == "Corrected"

    def test_get_provenance_invalidation_absent(self):
        node = {"@value": "current", "@confidence": 0.9}
        prov = get_provenance(node)
        assert prov.invalidated_at is None
        assert prov.invalidation_reason is None


class TestFilterByConfidenceExcludeInvalidated:
    """exclude_invalidated parameter on filter_by_confidence."""

    def _make_graph(self):
        return [
            {"@id": "a", "name": annotate("Alice", confidence=0.9)},
            {"@id": "b", "name": annotate(
                "Bob", confidence=0.8,
                invalidated_at="2025-06-01T00:00:00Z",
            )},
            {"@id": "c", "name": annotate("Carol", confidence=0.7)},
        ]

    def test_default_includes_invalidated(self):
        """By default, invalidated nodes are NOT excluded (backward compat)."""
        graph = self._make_graph()
        results = filter_by_confidence(graph, "name", 0.5)
        assert len(results) == 3

    def test_exclude_invalidated_true(self):
        graph = self._make_graph()
        results = filter_by_confidence(graph, "name", 0.5, exclude_invalidated=True)
        ids = [r["@id"] for r in results]
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_exclude_invalidated_false_explicit(self):
        graph = self._make_graph()
        results = filter_by_confidence(graph, "name", 0.5, exclude_invalidated=False)
        assert len(results) == 3

    def test_exclude_invalidated_no_invalidated_nodes(self):
        graph = [
            {"@id": "a", "name": annotate("Alice", confidence=0.9)},
            {"@id": "b", "name": annotate("Bob", confidence=0.8)},
        ]
        results = filter_by_confidence(graph, "name", 0.5, exclude_invalidated=True)
        assert len(results) == 2
