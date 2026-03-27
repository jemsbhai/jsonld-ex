"""Tests for AMQP transport module.

Verifies AMQP message property derivation from JSON-LD metadata,
routing key generation, and payload serialization for use with
RabbitMQ, Azure Service Bus, and other AMQP 0-9-1 / 1.0 brokers.

References:
    AMQP 0-9-1: https://www.rabbitmq.com/amqp-0-9-1-reference
    AMQP 1.0: https://docs.oasis-open.org/amqp/core/v1.0/amqp-core-overview-v1.0.html
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for AMQP tests")

from jsonld_ex.amqp import (
    # Payload serialization
    to_amqp_payload,
    from_amqp_payload,
    # Property derivation
    derive_amqp_properties,
    derive_routing_key,
    derive_amqp_priority,
    derive_amqp_headers,
    # Constants
    DELIVERY_MODE_TRANSIENT,
    DELIVERY_MODE_PERSISTENT,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def sensor_doc():
    """IoT sensor reading with full metadata."""
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
def verified_doc():
    """High-confidence, human-verified document."""
    return {
        "@context": "https://schema.org/",
        "@type": "ClinicalAlert",
        "@id": "urn:alert:critical-42",
        "@confidence": 0.99,
        "@humanVerified": True,
        "@source": "https://ehr.example.org/alerts",
        "severity": "critical",
    }


@pytest.fixture
def low_confidence_doc():
    """Noisy telemetry."""
    return {
        "@context": "https://schema.org/",
        "@type": "AccelerometerSample",
        "@id": "urn:sensor:accel-007",
        "@confidence": 0.3,
        "x": 0.012,
    }


@pytest.fixture
def minimal_doc():
    """No jsonld-ex metadata."""
    return {
        "@context": "https://schema.org/",
        "@type": "Person",
        "name": "Alice",
    }


# ═══════════════════════════════════════════════════════════════════
# Payload serialization
# ═══════════════════════════════════════════════════════════════════


class TestAMQPPayload:
    """CBOR and JSON payload serialization for AMQP."""

    def test_cbor_round_trip(self, sensor_doc):
        payload = to_amqp_payload(sensor_doc, compress=True)
        assert isinstance(payload, bytes)
        restored = from_amqp_payload(payload, compressed=True)
        assert restored["@type"] == "SensorReading"

    def test_json_round_trip(self, sensor_doc):
        payload = to_amqp_payload(sensor_doc, compress=False)
        restored = from_amqp_payload(payload, compressed=False)
        assert restored["@type"] == "SensorReading"

    def test_context_reattach(self):
        doc = {"@type": "Sensor", "value": 42}
        payload = to_amqp_payload(doc, compress=True)
        restored = from_amqp_payload(
            payload, compressed=True, context="https://schema.org/"
        )
        assert restored["@context"] == "https://schema.org/"


# ═══════════════════════════════════════════════════════════════════
# Routing key derivation
# ═══════════════════════════════════════════════════════════════════


class TestRoutingKey:
    """AMQP routing key from @type and @id."""

    def test_basic_routing_key(self, sensor_doc):
        """Dot-separated routing key: prefix.type.id_fragment."""
        key = derive_routing_key(sensor_doc)
        assert key == "ld.SensorReading.imu-001"

    def test_custom_prefix(self, sensor_doc):
        key = derive_routing_key(sensor_doc, prefix="sensors")
        assert key.startswith("sensors.")

    def test_full_iri_type(self):
        doc = {
            "@type": "https://schema.org/Person",
            "@id": "https://example.org/people/alice",
        }
        key = derive_routing_key(doc)
        assert key == "ld.Person.alice"

    def test_missing_type_and_id(self):
        key = derive_routing_key({})
        assert key == "ld.unknown.unknown"

    def test_array_type(self):
        doc = {"@type": ["SensorReading", "Observation"], "@id": "urn:x:1"}
        key = derive_routing_key(doc)
        assert key.split(".")[1] == "SensorReading"

    def test_dots_in_segments_preserved(self):
        """Dots in type/id are valid in AMQP routing keys (unlike MQTT)."""
        doc = {"@type": "v1.SensorReading", "@id": "urn:sensor:v2.imu"}
        key = derive_routing_key(doc)
        # Dots from the original are part of the local name extraction,
        # not from our delimiter
        assert "ld." in key


# ═══════════════════════════════════════════════════════════════════
# Priority derivation
# ═══════════════════════════════════════════════════════════════════


class TestAMQPPriority:
    """AMQP message priority (0-9) from @confidence."""

    def test_high_confidence_high_priority(self, verified_doc):
        """@confidence >= 0.9 → priority 9."""
        priority = derive_amqp_priority(verified_doc)
        assert priority == 9

    def test_medium_confidence_medium_priority(self):
        doc = {"@confidence": 0.7}
        priority = derive_amqp_priority(doc)
        assert 5 <= priority <= 7

    def test_low_confidence_low_priority(self, low_confidence_doc):
        """@confidence < 0.5 → priority 0-4."""
        priority = derive_amqp_priority(low_confidence_doc)
        assert priority <= 4

    def test_no_confidence_default(self, minimal_doc):
        """No confidence → priority 4 (middle)."""
        priority = derive_amqp_priority(minimal_doc)
        assert priority == 4

    def test_priority_range(self):
        """Priority always in [0, 9]."""
        for conf in [0.0, 0.1, 0.5, 0.9, 1.0]:
            doc = {"@confidence": conf}
            p = derive_amqp_priority(doc)
            assert 0 <= p <= 9


# ═══════════════════════════════════════════════════════════════════
# Headers derivation
# ═══════════════════════════════════════════════════════════════════


class TestAMQPHeaders:
    """AMQP message headers from JSON-LD metadata."""

    def test_type_header(self, sensor_doc):
        headers = derive_amqp_headers(sensor_doc)
        assert headers["x-jsonld-type"] == "SensorReading"

    def test_confidence_header(self, sensor_doc):
        headers = derive_amqp_headers(sensor_doc)
        assert headers["x-jsonld-confidence"] == "0.95"

    def test_source_header(self, sensor_doc):
        headers = derive_amqp_headers(sensor_doc)
        assert headers["x-jsonld-source"] == "https://model.example.org/temp-v2"

    def test_id_header(self, sensor_doc):
        headers = derive_amqp_headers(sensor_doc)
        assert headers["x-jsonld-id"] == "urn:sensor:imu-001"

    def test_minimal_doc_sparse_headers(self, minimal_doc):
        headers = derive_amqp_headers(minimal_doc)
        assert headers["x-jsonld-type"] == "Person"
        assert "x-jsonld-confidence" not in headers
        assert "x-jsonld-source" not in headers

    def test_header_values_are_strings(self, sensor_doc):
        headers = derive_amqp_headers(sensor_doc)
        for k, v in headers.items():
            assert isinstance(v, str), f"Header {k} value is not a string: {v!r}"


# ═══════════════════════════════════════════════════════════════════
# Full property derivation
# ═══════════════════════════════════════════════════════════════════


class TestAMQPProperties:
    """Full AMQP message properties dict."""

    def test_content_type_cbor(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert props["content_type"] == "application/cbor"

    def test_content_type_jsonld(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=False)
        assert props["content_type"] == "application/ld+json"

    def test_delivery_mode_persistent_high_conf(self, verified_doc):
        """High confidence → persistent delivery."""
        props = derive_amqp_properties(verified_doc, compress=True)
        assert props["delivery_mode"] == DELIVERY_MODE_PERSISTENT

    def test_delivery_mode_transient_low_conf(self, low_confidence_doc):
        """Low confidence → transient delivery."""
        props = derive_amqp_properties(low_confidence_doc, compress=True)
        assert props["delivery_mode"] == DELIVERY_MODE_TRANSIENT

    def test_delivery_mode_persistent_default(self, minimal_doc):
        """No confidence → persistent (safe default)."""
        props = derive_amqp_properties(minimal_doc, compress=True)
        assert props["delivery_mode"] == DELIVERY_MODE_PERSISTENT

    def test_expiration_from_valid_until(self, sensor_doc):
        """@validUntil → expiration in milliseconds (AMQP convention)."""
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert "expiration" in props
        exp_ms = int(props["expiration"])
        # ~1 hour = ~3,600,000 ms
        assert 3_500_000 <= exp_ms <= 3_700_000

    def test_no_expiration_without_valid_until(self, minimal_doc):
        props = derive_amqp_properties(minimal_doc, compress=True)
        assert "expiration" not in props

    def test_message_id_from_doc_id(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert props["message_id"] == "urn:sensor:imu-001"

    def test_no_message_id_without_doc_id(self):
        props = derive_amqp_properties({"@type": "Thing"}, compress=True)
        assert "message_id" not in props

    def test_priority_included(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert "priority" in props
        assert 0 <= props["priority"] <= 9

    def test_headers_included(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert "headers" in props
        assert isinstance(props["headers"], dict)

    def test_routing_key_included(self, sensor_doc):
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert "routing_key" in props
        assert props["routing_key"] == "ld.SensorReading.imu-001"

    def test_timestamp_present(self, sensor_doc):
        """Timestamp should be set to current time (epoch seconds)."""
        import time
        before = int(time.time())
        props = derive_amqp_properties(sensor_doc, compress=True)
        after = int(time.time())
        assert "timestamp" in props
        assert before <= props["timestamp"] <= after


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestAMQPEdgeCases:

    def test_empty_document(self):
        props = derive_amqp_properties({}, compress=True)
        assert props["routing_key"] == "ld.unknown.unknown"
        assert props["delivery_mode"] == DELIVERY_MODE_PERSISTENT

    def test_expired_valid_until_no_expiration(self):
        doc = {
            "reading": {"@value": 42, "@validUntil": "2020-01-01T00:00:00Z"},
        }
        props = derive_amqp_properties(doc, compress=True)
        assert "expiration" not in props

    def test_all_property_values_correct_types(self, sensor_doc):
        """AMQP properties must match expected types for client libraries."""
        props = derive_amqp_properties(sensor_doc, compress=True)
        assert isinstance(props["content_type"], str)
        assert isinstance(props["delivery_mode"], int)
        assert isinstance(props["priority"], int)
        assert isinstance(props["routing_key"], str)
        assert isinstance(props["timestamp"], int)
        if "expiration" in props:
            # AMQP 0-9-1 expiration is a string of milliseconds
            assert isinstance(props["expiration"], str)
        if "message_id" in props:
            assert isinstance(props["message_id"], str)
