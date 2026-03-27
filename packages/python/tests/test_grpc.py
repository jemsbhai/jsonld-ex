"""Tests for gRPC transport module.

Verifies gRPC metadata derivation from JSON-LD annotations and
.proto schema suggestion generation. Unlike other transport modules,
gRPC uses compiled Protobuf schemas — this module bridges the gap
by mapping JSON-LD metadata to gRPC metadata keys and generating
proto schema suggestions from document structure.

References:
    gRPC Metadata: https://grpc.io/docs/guides/metadata/
    Protocol Buffers: https://protobuf.dev/
    gRPC Python: https://grpc.github.io/grpc/python/
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

from jsonld_ex.grpc import (
    # Metadata derivation
    derive_grpc_metadata,
    # Proto schema suggestion
    suggest_proto_schema,
    # Serialization helpers
    to_grpc_json,
    from_grpc_json,
    # Constants
    METADATA_KEY_TYPE,
    METADATA_KEY_CONFIDENCE,
    METADATA_KEY_SOURCE,
    METADATA_KEY_ID,
    METADATA_KEY_CONTENT_TYPE,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sensor_doc():
    return {
        "@context": "https://schema.org/",
        "@type": "SensorReading",
        "@id": "urn:sensor:imu-001",
        "@confidence": 0.95,
        "@source": "https://model.example.org/temp-v2",
        "temperature": {
            "@value": 36.7,
            "@confidence": 0.88,
        },
        "humidity": {
            "@value": 45.2,
        },
    }


@pytest.fixture
def product_doc():
    return {
        "@context": "https://schema.org/",
        "@type": "Product",
        "@id": "urn:product:widget-42",
        "name": "Widget Pro",
        "price": 29.99,
        "inStock": True,
        "tags": ["electronics", "gadget"],
        "embedding": {
            "@container": "@vector",
            "@dimensions": 4,
            "@value": [0.1, 0.2, 0.3, 0.4],
        },
    }


@pytest.fixture
def minimal_doc():
    return {
        "@context": "https://schema.org/",
        "@type": "Person",
        "name": "Alice",
    }


# ═══════════════════════════════════════════════════════════════════
# gRPC metadata derivation
# ═══════════════════════════════════════════════════════════════════


class TestGRPCMetadata:
    """gRPC metadata (key-value pairs) from JSON-LD annotations."""

    def test_type_metadata(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_TYPE] == "SensorReading"

    def test_confidence_metadata(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_CONFIDENCE] == "0.95"

    def test_source_metadata(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_SOURCE] == "https://model.example.org/temp-v2"

    def test_id_metadata(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_ID] == "urn:sensor:imu-001"

    def test_content_type_metadata(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_CONTENT_TYPE] == "application/ld+json"

    def test_minimal_doc_sparse(self, minimal_doc):
        meta = derive_grpc_metadata(minimal_doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_TYPE] == "Person"
        assert METADATA_KEY_CONFIDENCE not in meta_dict
        assert METADATA_KEY_SOURCE not in meta_dict

    def test_metadata_keys_lowercase(self, sensor_doc):
        """gRPC metadata keys must be lowercase ASCII (gRPC spec)."""
        meta = derive_grpc_metadata(sensor_doc)
        for key, _ in meta:
            assert key == key.lower(), f"Key {key!r} not lowercase"
            assert key.isascii(), f"Key {key!r} not ASCII"

    def test_metadata_values_are_strings(self, sensor_doc):
        """gRPC text metadata values must be strings."""
        meta = derive_grpc_metadata(sensor_doc)
        for key, val in meta:
            if not key.endswith("-bin"):
                assert isinstance(val, str), f"Value for {key!r} not str"

    def test_returns_list_of_tuples(self, sensor_doc):
        meta = derive_grpc_metadata(sensor_doc)
        assert isinstance(meta, list)
        for item in meta:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_array_type(self):
        doc = {"@type": ["SensorReading", "Observation"]}
        meta = derive_grpc_metadata(doc)
        meta_dict = dict(meta)
        assert meta_dict[METADATA_KEY_TYPE] == "SensorReading"

    def test_empty_document(self):
        meta = derive_grpc_metadata({})
        meta_dict = dict(meta)
        assert METADATA_KEY_CONTENT_TYPE in meta_dict
        assert METADATA_KEY_TYPE not in meta_dict


# ═══════════════════════════════════════════════════════════════════
# JSON serialization (for gRPC JSON transcoding)
# ═══════════════════════════════════════════════════════════════════


class TestGRPCJson:
    """JSON serialization for gRPC JSON transcoding (grpc-gateway)."""

    def test_round_trip(self, sensor_doc):
        payload = to_grpc_json(sensor_doc)
        assert isinstance(payload, str)
        restored = from_grpc_json(payload)
        assert restored["@type"] == "SensorReading"

    def test_context_reattach(self):
        doc = {"@type": "Sensor", "value": 42}
        payload = to_grpc_json(doc)
        restored = from_grpc_json(payload, context="https://schema.org/")
        assert restored["@context"] == "https://schema.org/"

    def test_compact_json(self, sensor_doc):
        """JSON should be compact (no extra whitespace)."""
        payload = to_grpc_json(sensor_doc)
        assert "\n" not in payload
        assert "  " not in payload


# ═══════════════════════════════════════════════════════════════════
# Proto schema suggestion
# ═══════════════════════════════════════════════════════════════════


class TestProtoSuggestion:
    """Generate .proto schema suggestions from JSON-LD structure."""

    def test_message_name_from_type(self, sensor_doc):
        """@type → proto message name."""
        proto = suggest_proto_schema(sensor_doc)
        assert "message SensorReading" in proto

    def test_string_field(self, minimal_doc):
        """String property → string field."""
        proto = suggest_proto_schema(minimal_doc)
        assert "string name" in proto

    def test_float_field(self, sensor_doc):
        """Nested @value float → double field."""
        proto = suggest_proto_schema(sensor_doc)
        assert "double temperature" in proto

    def test_bool_field(self, product_doc):
        """Boolean property → bool field."""
        proto = suggest_proto_schema(product_doc)
        assert "bool in_stock" in proto or "bool inStock" in proto

    def test_repeated_field(self, product_doc):
        """Array property → repeated field."""
        proto = suggest_proto_schema(product_doc)
        assert "repeated string tags" in proto

    def test_vector_field(self, product_doc):
        """@vector container → repeated double/float field."""
        proto = suggest_proto_schema(product_doc)
        assert "repeated" in proto
        # embedding should appear somewhere
        assert "embedding" in proto

    def test_syntax_proto3(self, sensor_doc):
        """Output uses proto3 syntax."""
        proto = suggest_proto_schema(sensor_doc)
        assert 'syntax = "proto3"' in proto

    def test_confidence_field(self, sensor_doc):
        """@confidence at document level → confidence field."""
        proto = suggest_proto_schema(sensor_doc)
        assert "double confidence" in proto

    def test_missing_type_uses_default(self):
        """No @type → message name 'JsonLdDocument'."""
        proto = suggest_proto_schema({"name": "test"})
        assert "message JsonLdDocument" in proto

    def test_proto_is_string(self, sensor_doc):
        proto = suggest_proto_schema(sensor_doc)
        assert isinstance(proto, str)

    def test_field_numbers_sequential(self, sensor_doc):
        """Proto field numbers should be sequential starting from 1."""
        proto = suggest_proto_schema(sensor_doc)
        assert "= 1;" in proto
        assert "= 2;" in proto

    def test_id_field(self, sensor_doc):
        """@id → string id field."""
        proto = suggest_proto_schema(sensor_doc)
        assert "string id" in proto


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestGRPCEdgeCases:

    def test_empty_document_metadata(self):
        meta = derive_grpc_metadata({})
        assert isinstance(meta, list)

    def test_empty_document_proto(self):
        proto = suggest_proto_schema({})
        assert "message JsonLdDocument" in proto

    def test_nested_annotated_value(self):
        """Annotated value with @value extracts the inner type."""
        doc = {
            "@type": "Reading",
            "value": {"@value": 42, "@confidence": 0.9},
        }
        proto = suggest_proto_schema(doc)
        # value should be numeric, not a sub-message
        assert "int" in proto.lower() or "double" in proto.lower()
