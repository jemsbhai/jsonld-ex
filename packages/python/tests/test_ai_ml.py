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
