"""Tests for WebSocket transport module.

Verifies WebSocket frame metadata derivation from JSON-LD,
subprotocol negotiation, and per-message metadata for use with
any WebSocket library (websockets, aiohttp, Socket.IO).

References:
    RFC 6455: The WebSocket Protocol
    RFC 7692: Compression Extensions for WebSocket (permessage-deflate)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for WebSocket tests")

from jsonld_ex.websocket import (
    # Payload serialization
    to_ws_message,
    from_ws_message,
    # Metadata derivation
    derive_ws_metadata,
    derive_ws_subprotocols,
    # Constants
    SUBPROTOCOL_JSONLD,
    SUBPROTOCOL_CBOR,
    WS_OPCODE_TEXT,
    WS_OPCODE_BINARY,
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
            "@validUntil": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
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
# Payload serialization
# ═══════════════════════════════════════════════════════════════════


class TestWSPayload:
    """WebSocket message serialization (text for JSON, binary for CBOR)."""

    def test_json_produces_str(self, sensor_doc):
        """JSON mode → str message (text frame)."""
        msg = to_ws_message(sensor_doc, compress=False)
        assert isinstance(msg, str)

    def test_cbor_produces_bytes(self, sensor_doc):
        """CBOR mode → bytes message (binary frame)."""
        msg = to_ws_message(sensor_doc, compress=True)
        assert isinstance(msg, bytes)

    def test_json_round_trip(self, sensor_doc):
        msg = to_ws_message(sensor_doc, compress=False)
        restored = from_ws_message(msg, compressed=False)
        assert restored["@type"] == "SensorReading"
        assert restored["temperature"]["@value"] == 36.7

    def test_cbor_round_trip(self, sensor_doc):
        msg = to_ws_message(sensor_doc, compress=True)
        restored = from_ws_message(msg, compressed=True)
        assert restored["@type"] == "SensorReading"
        assert restored["temperature"]["@value"] == 36.7

    def test_context_reattach(self):
        doc = {"@type": "Sensor", "value": 42}
        msg = to_ws_message(doc, compress=False)
        restored = from_ws_message(msg, compressed=False, context="https://schema.org/")
        assert restored["@context"] == "https://schema.org/"

    def test_cbor_smaller_than_json(self, sensor_doc):
        """CBOR binary frame should be smaller than JSON text frame."""
        json_msg = to_ws_message(sensor_doc, compress=False)
        cbor_msg = to_ws_message(sensor_doc, compress=True)
        assert len(cbor_msg) < len(json_msg.encode("utf-8"))


# ═══════════════════════════════════════════════════════════════════
# Subprotocol negotiation
# ═══════════════════════════════════════════════════════════════════


class TestWSSubprotocols:
    """WebSocket Sec-WebSocket-Protocol negotiation."""

    def test_json_subprotocol(self):
        """JSON mode → jsonld subprotocol."""
        protos = derive_ws_subprotocols(compress=False)
        assert SUBPROTOCOL_JSONLD in protos

    def test_cbor_subprotocol(self):
        """CBOR mode → cbor subprotocol preferred."""
        protos = derive_ws_subprotocols(compress=True)
        assert SUBPROTOCOL_CBOR in protos

    def test_returns_list(self):
        """Subprotocols returned as a list (for handshake header)."""
        protos = derive_ws_subprotocols(compress=True)
        assert isinstance(protos, list)
        assert len(protos) >= 1

    def test_cbor_includes_jsonld_fallback(self):
        """CBOR mode still includes jsonld as fallback."""
        protos = derive_ws_subprotocols(compress=True)
        assert SUBPROTOCOL_CBOR in protos
        assert SUBPROTOCOL_JSONLD in protos
        # CBOR should be first (preferred)
        assert protos.index(SUBPROTOCOL_CBOR) < protos.index(SUBPROTOCOL_JSONLD)


# ═══════════════════════════════════════════════════════════════════
# Per-message metadata
# ═══════════════════════════════════════════════════════════════════


class TestWSMetadata:
    """Per-message metadata dict for application-level framing."""

    def test_opcode_text_for_json(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["opcode"] == WS_OPCODE_TEXT

    def test_opcode_binary_for_cbor(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=True)
        assert meta["opcode"] == WS_OPCODE_BINARY

    def test_type_present(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["jsonld_type"] == "SensorReading"

    def test_id_present(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["jsonld_id"] == "urn:sensor:imu-001"

    def test_confidence_present(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["jsonld_confidence"] == 0.95

    def test_source_present(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["jsonld_source"] == "https://model.example.org/temp-v2"

    def test_minimal_doc_sparse_metadata(self, minimal_doc):
        meta = derive_ws_metadata(minimal_doc, compress=False)
        assert meta["jsonld_type"] == "Person"
        assert "jsonld_confidence" not in meta
        assert "jsonld_source" not in meta
        assert "jsonld_id" not in meta

    def test_content_type_json(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert meta["content_type"] == "application/ld+json"

    def test_content_type_cbor(self, sensor_doc):
        meta = derive_ws_metadata(sensor_doc, compress=True)
        assert meta["content_type"] == "application/cbor"

    def test_ttl_from_valid_until(self, sensor_doc):
        """@validUntil → ttl_seconds in metadata."""
        meta = derive_ws_metadata(sensor_doc, compress=False)
        assert "ttl_seconds" in meta
        assert 3500 <= meta["ttl_seconds"] <= 3700

    def test_no_ttl_without_valid_until(self, minimal_doc):
        meta = derive_ws_metadata(minimal_doc, compress=False)
        assert "ttl_seconds" not in meta


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestWSEdgeCases:

    def test_empty_document_json(self):
        msg = to_ws_message({}, compress=False)
        restored = from_ws_message(msg, compressed=False)
        assert restored == {}

    def test_empty_document_cbor(self):
        msg = to_ws_message({}, compress=True)
        restored = from_ws_message(msg, compressed=True)
        assert restored == {}

    def test_empty_document_metadata(self):
        meta = derive_ws_metadata({}, compress=False)
        assert meta["opcode"] == WS_OPCODE_TEXT
        assert "jsonld_type" not in meta

    def test_array_type(self):
        doc = {"@type": ["SensorReading", "Observation"]}
        meta = derive_ws_metadata(doc, compress=False)
        assert meta["jsonld_type"] == "SensorReading"

    def test_expired_valid_until_no_ttl(self):
        doc = {
            "reading": {"@value": 42, "@validUntil": "2020-01-01T00:00:00Z"},
        }
        meta = derive_ws_metadata(doc, compress=False)
        assert "ttl_seconds" not in meta
