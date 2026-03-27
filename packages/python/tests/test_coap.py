"""Tests for CoAP transport module.

Verifies CoAP option derivation from JSON-LD metadata, payload
serialization/deserialization, and message type mapping.

References:
    RFC 7252: The Constrained Application Protocol (CoAP)
    RFC 7641: Observing Resources in CoAP
    RFC 7959: Block-Wise Transfers in CoAP
"""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for CoAP tests")

from jsonld_ex.coap import (
    # Payload serialization
    to_coap_payload,
    from_coap_payload,
    # Option derivation
    derive_coap_options,
    derive_coap_uri_path,
    derive_coap_message_type,
    # Content format constants
    CONTENT_FORMAT_CBOR,
    CONTENT_FORMAT_JSON,
    CONTENT_FORMAT_JSONLD,
    # Message type constants
    MESSAGE_TYPE_CON,
    MESSAGE_TYPE_NON,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sensor_doc():
    """Typical IoT sensor reading with confidence and temporal metadata."""
    return {
        "@context": "https://schema.org/",
        "@type": "SensorReading",
        "@id": "urn:sensor:imu-001",
        "@confidence": 0.95,
        "temperature": {
            "@value": 36.7,
            "@confidence": 0.88,
            "@validUntil": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        },
    }


@pytest.fixture
def verified_doc():
    """Document with @humanVerified and @integrity."""
    return {
        "@context": "https://schema.org/",
        "@type": "ClinicalObservation",
        "@id": "urn:clinical:obs-42",
        "@confidence": 0.99,
        "@humanVerified": True,
        "@integrity": "sha256-abc123def456",
        "diagnosis": {
            "@value": "Type 2 Diabetes",
            "@confidence": 0.97,
        },
    }


@pytest.fixture
def low_confidence_doc():
    """Noisy telemetry with low confidence."""
    return {
        "@context": "https://schema.org/",
        "@type": "AccelerometerSample",
        "@id": "urn:sensor:accel-007",
        "@confidence": 0.3,
        "x": 0.012,
        "y": -0.005,
        "z": 9.81,
    }


@pytest.fixture
def no_metadata_doc():
    """Document with no confidence or temporal metadata."""
    return {
        "@context": "https://schema.org/",
        "@type": "Person",
        "name": "Alice",
    }


# ═══════════════════════════════════════════════════════════════════
# Payload serialization round-trip
# ═══════════════════════════════════════════════════════════════════


class TestCoAPPayload:
    """CBOR and JSON payload serialization for CoAP."""

    def test_cbor_round_trip(self, sensor_doc):
        """CBOR-compressed payload round-trips correctly."""
        payload = to_coap_payload(sensor_doc, compress=True)
        assert isinstance(payload, bytes)
        restored = from_coap_payload(payload, compressed=True)
        assert restored["@type"] == "SensorReading"
        assert restored["temperature"]["@value"] == 36.7

    def test_json_round_trip(self, sensor_doc):
        """JSON payload round-trips correctly."""
        payload = to_coap_payload(sensor_doc, compress=False)
        assert isinstance(payload, bytes)
        restored = from_coap_payload(payload, compressed=False)
        assert restored["@type"] == "SensorReading"

    def test_cbor_smaller_than_json(self, sensor_doc):
        """CBOR payload should be smaller than JSON."""
        cbor_payload = to_coap_payload(sensor_doc, compress=True)
        json_payload = to_coap_payload(sensor_doc, compress=False)
        assert len(cbor_payload) < len(json_payload)

    def test_context_reattach(self):
        """Context stripped during send can be reattached on receive."""
        doc = {"@type": "Sensor", "value": 42}
        payload = to_coap_payload(doc, compress=True)
        restored = from_coap_payload(
            payload, compressed=True,
            context="https://schema.org/"
        )
        assert restored["@context"] == "https://schema.org/"

    def test_max_payload_enforcement(self):
        """Payload exceeding max size raises ValueError."""
        # CoAP over UDP: practical limit ~1024 bytes for constrained networks
        large_doc = {
            "@context": "https://schema.org/",
            "data": "x" * 2000,
        }
        with pytest.raises(ValueError, match="[Pp]ayload.*exceed|[Ss]ize"):
            to_coap_payload(large_doc, compress=False, max_payload=1024)


# ═══════════════════════════════════════════════════════════════════
# URI path derivation
# ═══════════════════════════════════════════════════════════════════


class TestCoAPUriPath:
    """URI path derivation from @type and @id."""

    def test_basic_path(self, sensor_doc):
        """Standard path: /{prefix}/{type}/{id_fragment}."""
        segments = derive_coap_uri_path(sensor_doc)
        assert segments == ["ld", "SensorReading", "imu-001"]

    def test_custom_prefix(self, sensor_doc):
        """Custom prefix replaces default 'ld'."""
        segments = derive_coap_uri_path(sensor_doc, prefix="sensors")
        assert segments[0] == "sensors"

    def test_full_iri_type(self):
        """Full IRI @type extracts local name."""
        doc = {
            "@type": "https://schema.org/Person",
            "@id": "https://example.org/people/alice",
        }
        segments = derive_coap_uri_path(doc)
        assert segments == ["ld", "Person", "alice"]

    def test_urn_id(self):
        """URN @id extracts last segment."""
        doc = {"@type": "Sensor", "@id": "urn:device:temp:sensor-42"}
        segments = derive_coap_uri_path(doc)
        assert segments[2] == "sensor-42"

    def test_missing_type_and_id(self, no_metadata_doc):
        """Missing @id defaults to 'unknown'."""
        doc = {"name": "test"}  # no @type, no @id
        segments = derive_coap_uri_path(doc)
        assert segments == ["ld", "unknown", "unknown"]

    def test_array_type(self):
        """Array @type uses first element."""
        doc = {"@type": ["SensorReading", "Observation"], "@id": "urn:x:1"}
        segments = derive_coap_uri_path(doc)
        assert segments[1] == "SensorReading"


# ═══════════════════════════════════════════════════════════════════
# Message type derivation
# ═══════════════════════════════════════════════════════════════════


class TestCoAPMessageType:
    """CON/NON message type from confidence metadata."""

    def test_high_confidence_is_con(self, sensor_doc):
        """High confidence (>= 0.9) → CON (confirmable)."""
        msg_type = derive_coap_message_type(sensor_doc)
        assert msg_type == MESSAGE_TYPE_CON

    def test_low_confidence_is_non(self, low_confidence_doc):
        """Low confidence (< 0.5) → NON (non-confirmable)."""
        msg_type = derive_coap_message_type(low_confidence_doc)
        assert msg_type == MESSAGE_TYPE_NON

    def test_human_verified_is_con(self, verified_doc):
        """@humanVerified = true → CON regardless of confidence."""
        msg_type = derive_coap_message_type(verified_doc)
        assert msg_type == MESSAGE_TYPE_CON

    def test_no_metadata_defaults_con(self, no_metadata_doc):
        """No confidence metadata → CON (safe default for constrained)."""
        msg_type = derive_coap_message_type(no_metadata_doc)
        assert msg_type == MESSAGE_TYPE_CON

    def test_medium_confidence_is_con(self):
        """Medium confidence (0.5-0.9) → CON."""
        doc = {"@confidence": 0.7}
        msg_type = derive_coap_message_type(doc)
        assert msg_type == MESSAGE_TYPE_CON


# ═══════════════════════════════════════════════════════════════════
# Full option derivation
# ═══════════════════════════════════════════════════════════════════


class TestCoAPOptions:
    """Derive CoAP options from JSON-LD metadata."""

    def test_content_format_cbor(self, sensor_doc):
        """CBOR compression → Content-Format = 60."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert options["content_format"] == CONTENT_FORMAT_CBOR

    def test_content_format_json(self, sensor_doc):
        """No compression → Content-Format = application/json."""
        options = derive_coap_options(sensor_doc, compress=False)
        assert options["content_format"] in (CONTENT_FORMAT_JSON, CONTENT_FORMAT_JSONLD)

    def test_max_age_from_valid_until(self, sensor_doc):
        """Max-Age derived from @validUntil (seconds remaining)."""
        options = derive_coap_options(sensor_doc, compress=True)
        # validUntil is ~1 hour from now, so max_age should be ~3600
        assert "max_age" in options
        assert 3500 <= options["max_age"] <= 3700

    def test_max_age_absent_when_no_valid_until(self, low_confidence_doc):
        """No @validUntil → max_age absent (CoAP default 60s applies)."""
        options = derive_coap_options(low_confidence_doc, compress=True)
        assert "max_age" not in options

    def test_max_age_expired_omitted(self):
        """Already-expired @validUntil → max_age omitted."""
        doc = {
            "@context": "https://schema.org/",
            "reading": {
                "@value": 42,
                "@validUntil": "2020-01-01T00:00:00Z",  # past
            },
        }
        options = derive_coap_options(doc, compress=True)
        assert "max_age" not in options

    def test_etag_from_integrity(self, verified_doc):
        """ETag derived from @integrity hash, truncated to <= 8 bytes."""
        options = derive_coap_options(verified_doc, compress=True)
        assert "etag" in options
        assert isinstance(options["etag"], bytes)
        assert 1 <= len(options["etag"]) <= 8

    def test_etag_absent_when_no_integrity(self, sensor_doc):
        """No @integrity → etag absent."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert "etag" not in options

    def test_uri_path_included(self, sensor_doc):
        """URI path segments included in options."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert options["uri_path"] == ["ld", "SensorReading", "imu-001"]

    def test_message_type_included(self, sensor_doc):
        """Message type (CON/NON) included in options."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert options["message_type"] == MESSAGE_TYPE_CON

    def test_size1_from_payload(self, sensor_doc):
        """Size1 option set from serialized payload size."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert "size1" in options
        assert isinstance(options["size1"], int)
        assert options["size1"] > 0

    def test_block_recommendation_for_large_payload(self):
        """Large payload triggers block transfer recommendation."""
        import random
        rng = random.Random(42)
        # 768-dim float vector = large payload
        doc = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "embedding": {
                "@container": "@vector",
                "@dimensions": 768,
                "@value": [rng.gauss(0, 1) for _ in range(768)],
            },
        }
        options = derive_coap_options(doc, compress=True)
        # Payload > 1024 bytes should recommend block transfer
        if options["size1"] > 1024:
            assert options.get("block_szx") is not None

    def test_observe_flag_for_temporal_doc(self, sensor_doc):
        """Documents with @validUntil suggest Observe capability."""
        options = derive_coap_options(sensor_doc, compress=True)
        assert "observable" in options
        assert options["observable"] is True

    def test_no_observe_without_temporal(self, no_metadata_doc):
        """No temporal metadata → observable not set."""
        options = derive_coap_options(no_metadata_doc, compress=True)
        assert options.get("observable", False) is False


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestCoAPEdgeCases:
    """Edge cases for CoAP transport."""

    def test_empty_document(self):
        """Empty document serializes and derives options without error."""
        doc = {}
        payload = to_coap_payload(doc, compress=True)
        assert len(payload) > 0
        options = derive_coap_options(doc, compress=True)
        assert options["uri_path"] == ["ld", "unknown", "unknown"]

    def test_max_age_uint32_clamped(self):
        """Max-Age clamped to uint32 max (like MQTT message expiry)."""
        far_future = (
            datetime.now(timezone.utc) + timedelta(days=200 * 365)
        ).isoformat()
        doc = {
            "reading": {"@value": 1, "@validUntil": far_future},
        }
        options = derive_coap_options(doc, compress=True)
        if "max_age" in options:
            # CoAP Max-Age is uint, max ~136 years = 4,294,967,295 seconds
            assert options["max_age"] <= 0xFFFFFFFF

    def test_nested_valid_until(self):
        """@validUntil found in nested property value."""
        future = (
            datetime.now(timezone.utc) + timedelta(minutes=30)
        ).isoformat()
        doc = {
            "@type": "Reading",
            "temperature": {
                "@value": 25.0,
                "@validUntil": future,
            },
        }
        options = derive_coap_options(doc, compress=True)
        assert "max_age" in options
        assert 1700 <= options["max_age"] <= 1900
