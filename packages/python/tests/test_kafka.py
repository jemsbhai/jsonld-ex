"""Tests for Kafka transport module.

Verifies Kafka producer record derivation from JSON-LD metadata,
including partition key, headers, and timestamp mapping for use
with Apache Kafka, Confluent, and Redpanda.

References:
    Apache Kafka Protocol: https://kafka.apache.org/protocol
    Confluent Python Client: https://docs.confluent.io/kafka-clients/python/current/overview.html
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for Kafka tests")

from jsonld_ex.kafka import (
    # Payload serialization
    to_kafka_value,
    from_kafka_value,
    # Record derivation
    derive_kafka_record,
    derive_kafka_key,
    derive_kafka_headers,
    derive_kafka_timestamp,
    derive_kafka_topic,
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
        "@extractedAt": "2026-03-27T10:00:00Z",
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


@pytest.fixture
def multi_type_doc():
    return {
        "@type": ["SensorReading", "Observation"],
        "@id": "urn:obs:42",
        "@confidence": 0.7,
    }


# ═══════════════════════════════════════════════════════════════════
# Payload serialization
# ═══════════════════════════════════════════════════════════════════


class TestKafkaPayload:

    def test_cbor_round_trip(self, sensor_doc):
        value = to_kafka_value(sensor_doc, compress=True)
        assert isinstance(value, bytes)
        restored = from_kafka_value(value, compressed=True)
        assert restored["@type"] == "SensorReading"

    def test_json_round_trip(self, sensor_doc):
        value = to_kafka_value(sensor_doc, compress=False)
        restored = from_kafka_value(value, compressed=False)
        assert restored["@type"] == "SensorReading"

    def test_context_reattach(self):
        doc = {"@type": "Sensor", "value": 42}
        value = to_kafka_value(doc, compress=True)
        restored = from_kafka_value(
            value, compressed=True, context="https://schema.org/"
        )
        assert restored["@context"] == "https://schema.org/"


# ═══════════════════════════════════════════════════════════════════
# Key derivation (partition key)
# ═══════════════════════════════════════════════════════════════════


class TestKafkaKey:
    """Kafka record key from @id (determines partitioning)."""

    def test_key_from_id(self, sensor_doc):
        """@id → key bytes (UTF-8 encoded)."""
        key = derive_kafka_key(sensor_doc)
        assert key == b"urn:sensor:imu-001"

    def test_no_key_without_id(self, minimal_doc):
        """No @id → None (round-robin partitioning)."""
        key = derive_kafka_key(minimal_doc)
        assert key is None

    def test_key_is_bytes(self, sensor_doc):
        key = derive_kafka_key(sensor_doc)
        assert isinstance(key, bytes)


# ═══════════════════════════════════════════════════════════════════
# Topic derivation
# ═══════════════════════════════════════════════════════════════════


class TestKafkaTopic:
    """Kafka topic from @type."""

    def test_basic_topic(self, sensor_doc):
        """prefix.type_local as topic name."""
        topic = derive_kafka_topic(sensor_doc)
        assert topic == "ld.SensorReading"

    def test_custom_prefix(self, sensor_doc):
        topic = derive_kafka_topic(sensor_doc, prefix="events")
        assert topic == "events.SensorReading"

    def test_full_iri_type(self):
        doc = {"@type": "https://schema.org/Person"}
        topic = derive_kafka_topic(doc)
        assert topic == "ld.Person"

    def test_array_type(self, multi_type_doc):
        topic = derive_kafka_topic(multi_type_doc)
        assert topic == "ld.SensorReading"

    def test_missing_type(self):
        topic = derive_kafka_topic({})
        assert topic == "ld.unknown"

    def test_topic_sanitized(self):
        """Kafka topic names cannot contain certain characters."""
        doc = {"@type": "My#Special+Type"}
        topic = derive_kafka_topic(doc)
        assert "#" not in topic
        assert "+" not in topic


# ═══════════════════════════════════════════════════════════════════
# Headers derivation
# ═══════════════════════════════════════════════════════════════════


class TestKafkaHeaders:
    """Kafka record headers from JSON-LD metadata."""

    def test_type_header(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc)
        header_dict = dict(headers)
        assert header_dict["x-jsonld-type"] == b"SensorReading"

    def test_confidence_header(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc)
        header_dict = dict(headers)
        assert header_dict["x-jsonld-confidence"] == b"0.95"

    def test_source_header(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc)
        header_dict = dict(headers)
        assert header_dict["x-jsonld-source"] == b"https://model.example.org/temp-v2"

    def test_id_header(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc)
        header_dict = dict(headers)
        assert header_dict["x-jsonld-id"] == b"urn:sensor:imu-001"

    def test_content_type_header_cbor(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc, compress=True)
        header_dict = dict(headers)
        assert header_dict["content-type"] == b"application/cbor"

    def test_content_type_header_json(self, sensor_doc):
        headers = derive_kafka_headers(sensor_doc, compress=False)
        header_dict = dict(headers)
        assert header_dict["content-type"] == b"application/ld+json"

    def test_minimal_doc_sparse_headers(self, minimal_doc):
        headers = derive_kafka_headers(minimal_doc)
        header_dict = dict(headers)
        assert header_dict["x-jsonld-type"] == b"Person"
        assert "x-jsonld-confidence" not in header_dict

    def test_headers_are_tuples_of_str_bytes(self, sensor_doc):
        """Kafka headers are list of (str, bytes) tuples."""
        headers = derive_kafka_headers(sensor_doc)
        assert isinstance(headers, list)
        for key, val in headers:
            assert isinstance(key, str), f"Key {key!r} not str"
            assert isinstance(val, bytes), f"Val for {key!r} not bytes: {val!r}"


# ═══════════════════════════════════════════════════════════════════
# Timestamp derivation
# ═══════════════════════════════════════════════════════════════════


class TestKafkaTimestamp:
    """Kafka record timestamp from @extractedAt."""

    def test_timestamp_from_extracted_at(self, sensor_doc):
        """@extractedAt → epoch milliseconds."""
        ts = derive_kafka_timestamp(sensor_doc)
        assert ts is not None
        # 2026-03-27T10:00:00Z in ms
        assert ts > 1_700_000_000_000  # sanity: after 2023

    def test_no_timestamp_without_extracted_at(self, minimal_doc):
        """No @extractedAt → None (broker assigns timestamp)."""
        ts = derive_kafka_timestamp(minimal_doc)
        assert ts is None

    def test_timestamp_is_milliseconds(self, sensor_doc):
        """Kafka timestamps are epoch milliseconds, not seconds."""
        ts = derive_kafka_timestamp(sensor_doc)
        # Must be > 1e12 (milliseconds) not ~1e9 (seconds)
        assert ts > 1_000_000_000_000


# ═══════════════════════════════════════════════════════════════════
# Full record derivation
# ═══════════════════════════════════════════════════════════════════


class TestKafkaRecord:
    """Full Kafka producer record dict."""

    def test_topic_present(self, sensor_doc):
        record = derive_kafka_record(sensor_doc, compress=True)
        assert "topic" in record
        assert record["topic"] == "ld.SensorReading"

    def test_key_present(self, sensor_doc):
        record = derive_kafka_record(sensor_doc, compress=True)
        assert record["key"] == b"urn:sensor:imu-001"

    def test_value_is_bytes(self, sensor_doc):
        record = derive_kafka_record(sensor_doc, compress=True)
        assert isinstance(record["value"], bytes)

    def test_headers_present(self, sensor_doc):
        record = derive_kafka_record(sensor_doc, compress=True)
        assert "headers" in record
        assert isinstance(record["headers"], list)

    def test_timestamp_present(self, sensor_doc):
        record = derive_kafka_record(sensor_doc, compress=True)
        assert "timestamp" in record

    def test_no_key_for_doc_without_id(self, minimal_doc):
        record = derive_kafka_record(minimal_doc, compress=True)
        assert record["key"] is None

    def test_round_trip_from_record(self, sensor_doc):
        """Value from record can be deserialized back."""
        record = derive_kafka_record(sensor_doc, compress=True)
        restored = from_kafka_value(record["value"], compressed=True)
        assert restored["@type"] == "SensorReading"
        assert restored["temperature"]["@value"] == 36.7


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestKafkaEdgeCases:

    def test_empty_document(self):
        record = derive_kafka_record({}, compress=True)
        assert record["topic"] == "ld.unknown"
        assert record["key"] is None

    def test_invalid_extracted_at(self):
        """Non-parseable @extractedAt → None timestamp."""
        doc = {"@extractedAt": "not-a-date"}
        ts = derive_kafka_timestamp(doc)
        assert ts is None

    def test_numeric_extracted_at_ignored(self):
        """Non-string @extractedAt → None."""
        doc = {"@extractedAt": 12345}
        ts = derive_kafka_timestamp(doc)
        assert ts is None
