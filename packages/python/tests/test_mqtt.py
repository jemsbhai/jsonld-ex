"""Tests for MQTT transport optimization."""

import json
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for MQTT tests")

from jsonld_ex.mqtt import (
    to_mqtt_payload,
    from_mqtt_payload,
    derive_mqtt_topic,
    derive_mqtt_qos,
    derive_mqtt5_properties,
    _sanitise_topic_segment,
    _derive_expiry_seconds,
    _MAX_TOPIC_BYTES,
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

    def test_dollar_prefix_stripped(self):
        """Leading $ is reserved for broker system topics ($SYS/)."""
        doc = {"@type": "$SYS", "@id": "urn:x:1"}
        topic = derive_mqtt_topic(doc)
        # Leading $ must be stripped; $SYS → SYS
        assert not topic.split("/")[1].startswith("$")
        assert topic == "ld/SYS/1"

    def test_dollar_in_middle_preserved(self):
        """$ in the middle of a segment is fine per MQTT spec."""
        doc = {"@type": "My$Type", "@id": "urn:x:1"}
        topic = derive_mqtt_topic(doc)
        assert topic == "ld/My$Type/1"

    def test_dollar_only_type_becomes_unknown(self):
        """Type that is only '$' should fall back to 'unknown'."""
        result = _sanitise_topic_segment("$")
        assert result == "unknown"

    def test_multiple_leading_dollars(self):
        result = _sanitise_topic_segment("$$SYS")
        assert result == "SYS"

    def test_null_char_replaced(self):
        result = _sanitise_topic_segment("hello\x00world")
        assert "\x00" not in result
        assert result == "hello_world"


class TestDeriveTopicLengthValidation:
    """MQTT spec: topic name MUST NOT exceed 65,535 bytes (UTF-8)."""

    def test_normal_topic_within_limit(self):
        doc = {"@type": "SensorReading", "@id": "urn:sensor:imu-001"}
        # Should not raise
        topic = derive_mqtt_topic(doc)
        assert len(topic.encode("utf-8")) < _MAX_TOPIC_BYTES

    def test_oversized_topic_raises(self):
        """Topic exceeding 65,535 bytes must raise ValueError."""
        # Create a document whose @id will produce a huge topic segment
        huge_id = "urn:x:" + "a" * 70_000
        doc = {"@type": "T", "@id": huge_id}
        with pytest.raises(ValueError, match="MQTT spec limit"):
            derive_mqtt_topic(doc)

    def test_exactly_at_limit_ok(self):
        """Topic at exactly 65,535 bytes should be accepted."""
        # prefix "ld" + "/" + "T" + "/" = 5 bytes overhead
        needed = _MAX_TOPIC_BYTES - 5  # bytes for the id segment
        doc = {"@type": "T", "@id": "urn:x:" + "a" * needed}
        # _local_name will extract the 'a...' part after last ':'
        topic = derive_mqtt_topic(doc)
        assert len(topic.encode("utf-8")) == _MAX_TOPIC_BYTES

    def test_multibyte_utf8_counted_correctly(self):
        """Ensure byte count uses UTF-8, not character count."""
        # Each emoji is 4 bytes in UTF-8
        emoji_count = 16_384  # 4 * 16384 = 65,536 bytes just for emojis
        huge_id = "urn:x:" + "😀" * emoji_count
        doc = {"@type": "T", "@id": huge_id}
        with pytest.raises(ValueError, match="MQTT spec limit"):
            derive_mqtt_topic(doc)


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


# ═══════════════════════════════════════════════════════════════════
# MQTT 5.0 PUBLISH properties derivation
# ═══════════════════════════════════════════════════════════════════


class TestDeriveMqtt5PropertiesPayloadFormat:
    """§3.3.2.3.2 — Payload Format Indicator."""

    def test_compressed_is_binary(self):
        props = derive_mqtt5_properties({"@type": "T"}, compress=True)
        assert props["payload_format_indicator"] == 0

    def test_uncompressed_is_utf8(self):
        props = derive_mqtt5_properties({"@type": "T"}, compress=False)
        assert props["payload_format_indicator"] == 1


class TestDeriveMqtt5PropertiesContentType:
    """§3.3.2.3.9 — Content Type (MIME)."""

    def test_compressed_content_type(self):
        props = derive_mqtt5_properties({"@type": "T"}, compress=True)
        assert props["content_type"] == "application/cbor"

    def test_uncompressed_content_type(self):
        props = derive_mqtt5_properties({"@type": "T"}, compress=False)
        assert props["content_type"] == "application/ld+json"


class TestDeriveMqtt5PropertiesExpiry:
    """§3.3.2.3.3 — Message Expiry Interval."""

    def test_no_valid_until_no_expiry(self):
        props = derive_mqtt5_properties({"@type": "T"})
        assert "message_expiry_interval" not in props

    def test_future_valid_until(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        doc = {"@type": "T", "@validUntil": future}
        props = derive_mqtt5_properties(doc)
        # Should be approximately 3600 seconds (±5s tolerance)
        assert "message_expiry_interval" in props
        assert 3595 <= props["message_expiry_interval"] <= 3605

    def test_past_valid_until_excluded(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        doc = {"@type": "T", "@validUntil": past}
        props = derive_mqtt5_properties(doc)
        assert "message_expiry_interval" not in props

    def test_z_suffix_parsed(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        doc = {"@type": "T", "@validUntil": future}
        props = derive_mqtt5_properties(doc)
        assert "message_expiry_interval" in props
        assert 1795 <= props["message_expiry_interval"] <= 1805

    def test_property_level_valid_until(self):
        """@validUntil on a property value is also detected."""
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        doc = {
            "@type": "Reading",
            "value": {"@value": 42, "@validUntil": future},
        }
        props = derive_mqtt5_properties(doc)
        assert "message_expiry_interval" in props
        assert 7195 <= props["message_expiry_interval"] <= 7205

    def test_invalid_date_string_ignored(self):
        doc = {"@type": "T", "@validUntil": "not-a-date"}
        props = derive_mqtt5_properties(doc)
        assert "message_expiry_interval" not in props

    def test_non_string_valid_until_ignored(self):
        doc = {"@type": "T", "@validUntil": 12345}
        props = derive_mqtt5_properties(doc)
        assert "message_expiry_interval" not in props

    def test_expiry_is_integer(self):
        """MQTT spec requires uint32, so result must be int."""
        future = (datetime.now(timezone.utc) + timedelta(seconds=100)).isoformat()
        doc = {"@type": "T", "@validUntil": future}
        props = derive_mqtt5_properties(doc)
        assert isinstance(props["message_expiry_interval"], int)

    def test_expiry_ceiled(self):
        """Fractional seconds should be ceiled, not floored."""
        # 0.5 seconds in the future should give expiry of 1, not 0
        future = (datetime.now(timezone.utc) + timedelta(milliseconds=500)).isoformat()
        doc = {"@type": "T", "@validUntil": future}
        props = derive_mqtt5_properties(doc)
        if "message_expiry_interval" in props:
            assert props["message_expiry_interval"] >= 1


class TestDeriveMqtt5PropertiesUserProperties:
    """§3.3.2.3.7 — User Properties (key-value pairs)."""

    def test_type_included(self):
        props = derive_mqtt5_properties({"@type": "SensorReading"})
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_type"] == "SensorReading"

    def test_confidence_included(self):
        props = derive_mqtt5_properties(
            {"@type": "T", "@confidence": 0.95}
        )
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_confidence"] == "0.95"

    def test_source_included(self):
        props = derive_mqtt5_properties(
            {"@type": "T", "@source": "https://model.example.org/v2"}
        )
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_source"] == "https://model.example.org/v2"

    def test_id_included(self):
        props = derive_mqtt5_properties(
            {"@type": "T", "@id": "urn:sensor:imu-001"}
        )
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_id"] == "urn:sensor:imu-001"

    def test_no_metadata_minimal_user_props(self):
        """Doc with only @type still gets type user property."""
        props = derive_mqtt5_properties({"@type": "T"})
        assert "user_properties" in props
        user_props = dict(props["user_properties"])
        assert "jsonld_type" in user_props

    def test_empty_doc_no_user_props(self):
        props = derive_mqtt5_properties({})
        assert "user_properties" not in props

    def test_type_array_uses_first(self):
        props = derive_mqtt5_properties(
            {"@type": ["SensorReading", "Observation"]}
        )
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_type"] == "SensorReading"

    def test_user_properties_are_string_pairs(self):
        """MQTT 5.0 user properties must be UTF-8 string key-value pairs."""
        props = derive_mqtt5_properties(
            {"@type": "T", "@confidence": 0.5, "@id": "urn:x:1"}
        )
        for key, val in props["user_properties"]:
            assert isinstance(key, str)
            assert isinstance(val, str)


class TestDeriveMqtt5PropertiesCombined:
    """Integration tests combining multiple property derivations."""

    def test_full_iot_document(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
        doc = {
            "@type": "SensorReading",
            "@id": "urn:sensor:imu-001",
            "@confidence": 0.92,
            "@source": "https://model.example.org/v3",
            "@validUntil": future,
            "value": 42.5,
        }
        props = derive_mqtt5_properties(doc, compress=True)

        assert props["payload_format_indicator"] == 0
        assert props["content_type"] == "application/cbor"
        assert 295 <= props["message_expiry_interval"] <= 305
        user_props = dict(props["user_properties"])
        assert user_props["jsonld_type"] == "SensorReading"
        assert user_props["jsonld_confidence"] == "0.92"
        assert user_props["jsonld_source"] == "https://model.example.org/v3"
        assert user_props["jsonld_id"] == "urn:sensor:imu-001"

    def test_minimal_document(self):
        """Bare-minimum doc still returns format + content type."""
        props = derive_mqtt5_properties({}, compress=False)
        assert props["payload_format_indicator"] == 1
        assert props["content_type"] == "application/ld+json"
        assert "message_expiry_interval" not in props
        assert "user_properties" not in props


# ═══════════════════════════════════════════════════════════════════
# Internal helper edge cases
# ═══════════════════════════════════════════════════════════════════


class TestDeriveExpirySeconds:
    def test_returns_none_for_empty_doc(self):
        assert _derive_expiry_seconds({}) is None

    def test_returns_none_for_non_string(self):
        assert _derive_expiry_seconds({"@validUntil": True}) is None

    def test_capped_at_uint32_max(self):
        """MQTT Message Expiry Interval is uint32 — max 4,294,967,295."""
        far_future = (
            datetime.now(timezone.utc) + timedelta(days=365 * 200)
        ).isoformat()
        result = _derive_expiry_seconds({"@validUntil": far_future})
        assert result is not None
        assert result <= 0xFFFFFFFF
