"""Tests for batch API (GAP-API1)."""

import pytest
from jsonld_ex.batch import annotate_batch, validate_batch, filter_by_confidence_batch
from jsonld_ex.validation import ValidationResult


# -- annotate_batch -----------------------------------------------------------


class TestAnnotateBatch:
    """annotate_batch applies the same annotation params to a list of values."""

    def test_basic_batch(self):
        values = ["Alice", "Bob", "Carol"]
        results = annotate_batch(values, confidence=0.9, source="model-v1")
        assert len(results) == 3
        for i, r in enumerate(results):
            assert r["@value"] == values[i]
            assert r["@confidence"] == 0.9
            assert r["@source"] == "model-v1"

    def test_empty_list(self):
        assert annotate_batch([], confidence=0.5) == []

    def test_mixed_types(self):
        values = ["text", 42, True, None]
        results = annotate_batch(values, confidence=0.8)
        assert len(results) == 4
        assert results[0]["@value"] == "text"
        assert results[1]["@value"] == 42
        assert results[2]["@value"] is True
        assert results[3]["@value"] is None

    def test_with_multimodal_params(self):
        values = [None, None]
        results = annotate_batch(
            values,
            confidence=0.85,
            media_type="image/png",
            content_url="s3://bucket/img.png",
        )
        for r in results:
            assert r["@mediaType"] == "image/png"
            assert r["@contentUrl"] == "s3://bucket/img.png"

    def test_per_item_overrides(self):
        """Each item can be a (value, overrides_dict) tuple."""
        items = [
            ("Alice", {"confidence": 0.95}),
            ("Bob", {"confidence": 0.70}),
            ("Carol", {}),  # uses defaults
        ]
        results = annotate_batch(items, confidence=0.5, source="model-v1")
        assert results[0]["@confidence"] == 0.95
        assert results[1]["@confidence"] == 0.70
        assert results[2]["@confidence"] == 0.5
        # All share the default source
        for r in results:
            assert r["@source"] == "model-v1"

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError):
            annotate_batch(["a"], confidence=1.5)

    def test_large_batch(self):
        """Verify 10K items process without issue."""
        values = list(range(10_000))
        results = annotate_batch(values, confidence=0.5)
        assert len(results) == 10_000
        assert results[9999]["@value"] == 9999


# -- validate_batch -----------------------------------------------------------


class TestValidateBatch:
    """validate_batch validates a list of nodes against a shape."""

    def test_all_valid(self):
        shape = {
            "@type": "Person",
            "name": {"@required": True, "@type": "xsd:string"},
        }
        nodes = [
            {"@type": "Person", "name": "Alice"},
            {"@type": "Person", "name": "Bob"},
        ]
        results = validate_batch(nodes, shape)
        assert len(results) == 2
        assert all(r.valid for r in results)

    def test_some_invalid(self):
        shape = {
            "@type": "Person",
            "name": {"@required": True},
        }
        nodes = [
            {"@type": "Person", "name": "Alice"},
            {"@type": "Person"},  # missing name
            {"@type": "Person", "name": "Carol"},
        ]
        results = validate_batch(nodes, shape)
        assert results[0].valid
        assert not results[1].valid
        assert results[2].valid

    def test_empty_list(self):
        shape = {"@type": "Person", "name": {"@required": True}}
        assert validate_batch([], shape) == []

    def test_returns_validation_results(self):
        shape = {"@type": "Person"}
        nodes = [{"@type": "Person"}]
        results = validate_batch(nodes, shape)
        assert isinstance(results[0], ValidationResult)

    def test_large_batch(self):
        shape = {"@type": "Thing", "val": {"@type": "xsd:integer"}}
        nodes = [{"@type": "Thing", "val": i} for i in range(10_000)]
        results = validate_batch(nodes, shape)
        assert len(results) == 10_000
        assert all(r.valid for r in results)


# -- filter_by_confidence_batch -----------------------------------------------


class TestFilterByConfidenceBatch:
    """filter_by_confidence_batch filters across multiple properties."""

    def test_basic_filter(self):
        nodes = [
            {"@type": "Thing", "name": {"@value": "A", "@confidence": 0.9}},
            {"@type": "Thing", "name": {"@value": "B", "@confidence": 0.3}},
            {"@type": "Thing", "name": {"@value": "C", "@confidence": 0.7}},
        ]
        results = filter_by_confidence_batch(nodes, "name", 0.5)
        assert len(results) == 2
        assert results[0]["name"]["@value"] == "A"
        assert results[1]["name"]["@value"] == "C"

    def test_empty_list(self):
        assert filter_by_confidence_batch([], "name", 0.5) == []

    def test_no_matches(self):
        nodes = [
            {"@type": "Thing", "name": {"@value": "A", "@confidence": 0.1}},
        ]
        assert filter_by_confidence_batch(nodes, "name", 0.5) == []

    def test_multi_property_filter(self):
        """Filter with multiple (property, threshold) pairs."""
        nodes = [
            {
                "@type": "Thing",
                "name": {"@value": "A", "@confidence": 0.9},
                "desc": {"@value": "good", "@confidence": 0.8},
            },
            {
                "@type": "Thing",
                "name": {"@value": "B", "@confidence": 0.9},
                "desc": {"@value": "bad", "@confidence": 0.2},
            },
        ]
        # Filter: name >= 0.5 AND desc >= 0.5
        results = filter_by_confidence_batch(
            nodes, [("name", 0.5), ("desc", 0.5)]
        )
        assert len(results) == 1
        assert results[0]["name"]["@value"] == "A"

    def test_large_batch(self):
        nodes = [
            {"@type": "Thing", "val": {"@value": i, "@confidence": i / 10_000}}
            for i in range(10_000)
        ]
        results = filter_by_confidence_batch(nodes, "val", 0.5)
        assert len(results) == 5_000
