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
