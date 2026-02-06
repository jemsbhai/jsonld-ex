"""Tests for MQTT transport optimization."""

import json
import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for MQTT tests")

from jsonld_ex.mqtt import (
    to_mqtt_payload,
    from_mqtt_payload,
    derive_mqtt_topic,
    derive_mqtt_qos,
)


# ═══════════════════════════════════════════════════════════════════
# Payload round-trip
# ═══════════════════════════════════════════════════════════════════


class TestMqttPayloadRoundTrip:
    def test_compressed_round_trip(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "SensorReading",
            "@id": "urn:sensor:imu-001",
            "value": {"@value": 42.5, "@confidence": 0.9},
        }
        payload = to_mqtt_payload(doc, compress=True)
        restored = from_mqtt_payload(payload, compressed=True)
        assert restored["@type"] == "SensorReading"
        assert restored["value"]["@value"] == 42.5
        assert restored["value"]["@confidence"] == 0.9

    def test_uncompressed_round_trip(self):
        doc = {"@type": "Event", "name": "test"}
        payload = to_mqtt_payload(doc, compress=False)
        restored = from_mqtt_payload(payload, compressed=False)
        assert restored == doc

    def test_context_reattach(self):
        doc = {"@type": "Person", "name": "Alice"}
        payload = to_mqtt_payload(doc, compress=False)
        restored = from_mqtt_payload(
            payload, compressed=False, context="http://schema.org/"
        )
        assert restored["@context"] == "http://schema.org/"

    def test_compressed_smaller_than_json(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "SensorReading",
            "value": {"@value": 42.5, "@confidence": 0.9},
            "timestamp": "2025-01-15T10:30:00Z",
        }
        compressed = to_mqtt_payload(doc, compress=True)
        uncompressed = to_mqtt_payload(doc, compress=False)
        assert len(compressed) < len(uncompressed)


class TestMqttPayloadLimits:
    def test_exceeds_max_payload_raises(self):
        doc = {"data": "x" * 1000}
        with pytest.raises(ValueError, match="exceeds max_payload"):
            to_mqtt_payload(doc, compress=False, max_payload=100)

    def test_within_max_payload_ok(self):
        doc = {"data": "small"}
        payload = to_mqtt_payload(doc, compress=False, max_payload=100_000)
        assert len(payload) < 100_000


# ═══════════════════════════════════════════════════════════════════
# Topic derivation
# ═══════════════════════════════════════════════════════════════════


class TestDeriveTopicBasic:
    def test_basic(self):
        doc = {"@type": "SensorReading", "@id": "urn:sensor:imu-001"}
        assert derive_mqtt_topic(doc) == "ld/SensorReading/imu-001"

    def test_custom_prefix(self):
        doc = {"@type": "SensorReading", "@id": "urn:sensor:imu-001"}
        assert derive_mqtt_topic(doc, prefix="devices") == "devices/SensorReading/imu-001"

    def test_http_id(self):
        doc = {"@type": "Person", "@id": "http://example.org/people/alice"}
        assert derive_mqtt_topic(doc) == "ld/Person/alice"

    def test_fragment_id(self):
        doc = {"@type": "Widget", "@id": "http://example.org/things#widget-5"}
        assert derive_mqtt_topic(doc) == "ld/Widget/widget-5"

    def test_missing_type(self):
        doc = {"@id": "urn:x:1"}
        topic = derive_mqtt_topic(doc)
        assert topic == "ld/unknown/1"

    def test_missing_id(self):
        doc = {"@type": "Event"}
        topic = derive_mqtt_topic(doc)
        assert topic == "ld/Event/unknown"

    def test_type_array(self):
        doc = {"@type": ["SensorReading", "Observation"], "@id": "urn:s:1"}
        topic = derive_mqtt_topic(doc)
        assert topic == "ld/SensorReading/1"

    def test_namespaced_type(self):
        doc = {"@type": "http://schema.org/Person", "@id": "ex:alice"}
        topic = derive_mqtt_topic(doc)
        assert topic == "ld/Person/alice"


class TestDeriveTopicSanitisation:
    def test_wildcard_hash_replaced(self):
        doc = {"@type": "Type#Sub", "@id": "urn:x:1"}
        topic = derive_mqtt_topic(doc)
        # The # should be sanitised or the local name extracted
        assert "#" not in topic

    def test_wildcard_plus_replaced(self):
        doc = {"@type": "A+B", "@id": "urn:x:1"}
        topic = derive_mqtt_topic(doc)
        assert "+" not in topic


# ═══════════════════════════════════════════════════════════════════
# QoS derivation
# ═══════════════════════════════════════════════════════════════════


class TestDeriveQos:
    def test_high_confidence_qos2(self):
        doc = {"@confidence": 0.95, "@type": "Alert"}
        assert derive_mqtt_qos(doc) == 2

    def test_medium_confidence_qos1(self):
        doc = {"@confidence": 0.7, "@type": "Reading"}
        assert derive_mqtt_qos(doc) == 1

    def test_low_confidence_qos0(self):
        doc = {"@confidence": 0.3, "@type": "Noise"}
        assert derive_mqtt_qos(doc) == 0

    def test_human_verified_qos2(self):
        doc = {"@humanVerified": True, "@type": "Diagnosis"}
        assert derive_mqtt_qos(doc) == 2

    def test_no_confidence_defaults_qos1(self):
        doc = {"@type": "Event", "name": "test"}
        assert derive_mqtt_qos(doc) == 1

    def test_property_level_confidence(self):
        """Falls back to first annotated property."""
        doc = {
            "@type": "Reading",
            "value": {"@value": 42, "@confidence": 0.95},
        }
        assert derive_mqtt_qos(doc) == 2

    def test_property_level_human_verified(self):
        doc = {
            "@type": "Reading",
            "value": {"@value": 42, "@humanVerified": True},
        }
        assert derive_mqtt_qos(doc) == 2

    def test_boundary_0_5_is_qos1(self):
        doc = {"@confidence": 0.5}
        assert derive_mqtt_qos(doc) == 1

    def test_boundary_0_9_is_qos2(self):
        doc = {"@confidence": 0.9}
        assert derive_mqtt_qos(doc) == 2

    def test_boundary_below_0_5_is_qos0(self):
        doc = {"@confidence": 0.49}
        assert derive_mqtt_qos(doc) == 0
