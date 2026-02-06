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
