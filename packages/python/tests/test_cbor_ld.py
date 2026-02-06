"""Tests for CBOR-LD serialization."""

import json
import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for CBOR-LD tests")

from jsonld_ex.cbor_ld import (
    to_cbor,
    from_cbor,
    payload_stats,
    PayloadStats,
    DEFAULT_CONTEXT_REGISTRY,
)


# ═══════════════════════════════════════════════════════════════════
# Round-trip
# ═══════════════════════════════════════════════════════════════════


class TestRoundTrip:
    def test_simple_document(self):
        doc = {
            "@context": "https://schema.org/",
            "@type": "Person",
            "name": "Alice",
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["@type"] == "Person"
        assert restored["name"] == "Alice"
        # Context restored from registry (http/https both map to ID 1)
        assert "schema.org" in restored["@context"]

    def test_annotated_values_preserved(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
                "@source": "model-v1",
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["name"]["@value"] == "Alice"
        assert restored["name"]["@confidence"] == 0.95
        assert restored["name"]["@source"] == "model-v1"

    def test_graph_array(self):
        doc = {
            "@context": "http://schema.org/",
            "@graph": [
                {"@id": "ex:a", "@type": "Person", "name": "A"},
                {"@id": "ex:b", "@type": "Person", "name": "B"},
            ],
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert len(restored["@graph"]) == 2

    def test_unknown_context_preserved(self):
        doc = {
            "@context": "http://custom.example.org/v1",
            "@type": "Widget",
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        # Unknown context URL should pass through as-is
        assert restored["@context"] == "http://custom.example.org/v1"

    def test_array_context(self):
        doc = {
            "@context": ["http://schema.org/", "http://custom.example.org/v1"],
            "@type": "Person",
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert len(restored["@context"]) == 2

    def test_nested_objects(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "address": {
                "@type": "PostalAddress",
                "city": "Melbourne",
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["address"]["city"] == "Melbourne"

    def test_temporal_annotations_preserved(self):
        doc = {
            "@context": "http://schema.org/",
            "jobTitle": {
                "@value": "Engineer",
                "@confidence": 0.9,
                "@validFrom": "2020-01-01",
                "@validUntil": "2024-12-31",
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["jobTitle"]["@validFrom"] == "2020-01-01"
        assert restored["jobTitle"]["@validUntil"] == "2024-12-31"


# ═══════════════════════════════════════════════════════════════════
# Context compression
# ═══════════════════════════════════════════════════════════════════


class TestContextCompression:
    def test_known_context_compressed(self):
        doc = {"@context": "http://schema.org/", "name": "test"}
        data = to_cbor(doc)
        # Decode raw CBOR to check the integer is there
        raw = cbor2.loads(data)
        assert raw["@context"] == 1  # schema.org → 1

    def test_custom_registry(self):
        registry = {"http://my.org/ctx": 99}
        doc = {"@context": "http://my.org/ctx", "x": 1}
        data = to_cbor(doc, context_registry=registry)
        raw = cbor2.loads(data)
        assert raw["@context"] == 99
        restored = from_cbor(data, context_registry=registry)
        assert restored["@context"] == "http://my.org/ctx"


# ═══════════════════════════════════════════════════════════════════
# Payload stats
# ═══════════════════════════════════════════════════════════════════


class TestPayloadStats:
    def test_cbor_smaller_than_json(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "name": {"@value": "Alice", "@confidence": 0.95},
            "email": "alice@example.com",
            "jobTitle": {"@value": "Engineer", "@confidence": 0.8},
        }
        stats = payload_stats(doc)
        assert stats.cbor_bytes < stats.json_bytes
        assert 0.0 < stats.cbor_ratio < 1.0

    def test_gzip_cbor_smallest(self):
        doc = {
            "@context": "http://schema.org/",
            "@graph": [
                {"@id": f"ex:{i}", "@type": "Person", "name": f"Person {i}"}
                for i in range(20)
            ],
        }
        stats = payload_stats(doc)
        assert stats.gzip_cbor_bytes <= stats.gzip_json_bytes
        assert stats.gzip_cbor_bytes < stats.json_bytes

    def test_empty_doc(self):
        stats = payload_stats({})
        assert stats.json_bytes > 0
        assert stats.cbor_bytes > 0
