"""
Kafka Transport Optimization for JSON-LD-Ex.

Derives Apache Kafka producer record fields from JSON-LD metadata,
enabling semantic-aware partitioning, filtering, and timestamping
in ML data pipelines and event streaming architectures.

Metadata mappings:

- **Topic**: ``prefix.type_local`` from ``@type``, mapping each
  document type to a dedicated Kafka topic.
- **Key**: ``@id`` as UTF-8 bytes — determines partition assignment
  (same key → same partition → ordering guarantee).
- **Headers**: List of ``(str, bytes)`` tuples carrying
  ``x-jsonld-type``, ``x-jsonld-confidence``, ``x-jsonld-source``,
  ``x-jsonld-id``, and ``content-type`` for consumer-side filtering
  without deserialization.
- **Timestamp**: Epoch milliseconds from ``@extractedAt`` — aligns
  Kafka's log timestamp with the data extraction time rather than
  ingestion time, enabling accurate time-windowed processing.
- **Value**: CBOR or JSON serialized payload.

No Kafka client library is required — this module returns plain
dicts and byte values suitable for any Kafka producer
(``confluent-kafka``, ``kafka-python``, ``aiokafka``).

References:
    Apache Kafka Protocol: https://kafka.apache.org/protocol
    Confluent Platform: https://docs.confluent.io/
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

from jsonld_ex._transport_common import (
    extract_type_local,
    find_valid_until,
    seconds_remaining,
)

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False


# ═══════════════════════════════════════════════════════════════════
# PAYLOAD SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_kafka_value(
    doc: dict[str, Any],
    compress: bool = True,
    context_registry: Optional[dict[str, int]] = None,
) -> bytes:
    """Serialize a JSON-LD document as a Kafka record value.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON (False).
        context_registry: For CBOR context compression.

    Returns:
        Encoded bytes for the Kafka record value.

    Raises:
        ImportError: If compress=True but ``cbor2`` is not installed.
    """
    if compress:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR Kafka payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        return to_cbor(doc, context_registry)
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def from_kafka_value(
    value: bytes,
    compressed: bool = True,
    context: Optional[Any] = None,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize a Kafka record value back to a JSON-LD document.

    Args:
        value: Raw bytes from Kafka consumer record.
        compressed: CBOR (True) or JSON (False).
        context: Optional ``@context`` to reattach.
        context_registry: For CBOR context decompression.

    Returns:
        Restored JSON-LD document.
    """
    if compressed:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR Kafka payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        doc = from_cbor(value, context_registry)
    else:
        doc = json.loads(value.decode("utf-8"))

    if context is not None and "@context" not in doc:
        doc["@context"] = context

    return doc


# ═══════════════════════════════════════════════════════════════════
# KEY DERIVATION (partition key)
# ═══════════════════════════════════════════════════════════════════


def derive_kafka_key(doc: dict[str, Any]) -> Optional[bytes]:
    """Derive Kafka record key from ``@id``.

    The key determines partition assignment: records with the same
    key are guaranteed to land in the same partition, preserving
    ordering for a given entity.

    Args:
        doc: JSON-LD document.

    Returns:
        ``@id`` as UTF-8 bytes, or None if no ``@id`` (round-robin
        partitioning).
    """
    doc_id = doc.get("@id")
    if doc_id is None:
        return None
    return str(doc_id).encode("utf-8")


# ═══════════════════════════════════════════════════════════════════
# TOPIC DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_kafka_topic(
    doc: dict[str, Any],
    prefix: str = "ld",
) -> str:
    """Derive Kafka topic name from ``@type``.

    Pattern: ``prefix.type_local``

    Unlike MQTT/CoAP (which include ``@id`` in the routing path),
    Kafka uses topics for broad categories and keys for entity-level
    routing.  Including ``@id`` in the topic would create excessive
    topic proliferation.

    Args:
        doc: JSON-LD document.
        prefix: Topic prefix (default ``"ld"``).

    Returns:
        Topic name string.

    Examples::

        >>> derive_kafka_topic({"@type": "SensorReading"})
        'ld.SensorReading'
    """
    type_str = extract_type_local(doc)
    return f"{prefix}.{type_str}"


# ═══════════════════════════════════════════════════════════════════
# HEADERS DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_kafka_headers(
    doc: dict[str, Any],
    compress: bool = True,
) -> List[Tuple[str, bytes]]:
    """Derive Kafka record headers from JSON-LD metadata.

    Kafka headers are a list of ``(key, value)`` tuples where keys
    are strings and values are bytes.  They enable consumer-side
    filtering and routing without deserializing the record value.

    Args:
        doc: JSON-LD document.
        compress: Whether the value is CBOR (True) or JSON (False).

    Returns:
        List of ``(str, bytes)`` header tuples.
    """
    headers: List[Tuple[str, bytes]] = []

    # Content-Type
    ct = "application/cbor" if compress else "application/ld+json"
    headers.append(("content-type", ct.encode("utf-8")))

    # @type
    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            headers.append(("x-jsonld-type", str(type_val).encode("utf-8")))

    # @confidence
    conf = doc.get("@confidence")
    if conf is not None:
        headers.append(("x-jsonld-confidence", str(conf).encode("utf-8")))

    # @source
    source = doc.get("@source")
    if source is not None:
        headers.append(("x-jsonld-source", str(source).encode("utf-8")))

    # @id
    doc_id = doc.get("@id")
    if doc_id is not None:
        headers.append(("x-jsonld-id", str(doc_id).encode("utf-8")))

    return headers


# ═══════════════════════════════════════════════════════════════════
# TIMESTAMP DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_kafka_timestamp(doc: dict[str, Any]) -> Optional[int]:
    """Derive Kafka record timestamp from ``@extractedAt``.

    Kafka timestamps are epoch milliseconds.  Using ``@extractedAt``
    aligns the record's log-append time with the actual data
    extraction time, enabling accurate time-windowed processing
    (e.g., Kafka Streams, ksqlDB) even when ingestion is delayed.

    Args:
        doc: JSON-LD document.

    Returns:
        Epoch milliseconds as int, or None if no ``@extractedAt``
        (broker assigns its own timestamp).
    """
    extracted_at = doc.get("@extractedAt")
    if extracted_at is None or not isinstance(extracted_at, str):
        return None

    try:
        dt_str = extracted_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except (ValueError, TypeError, OverflowError):
        return None


# ═══════════════════════════════════════════════════════════════════
# FULL RECORD DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_kafka_record(
    doc: dict[str, Any],
    compress: bool = True,
    prefix: str = "ld",
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Derive a complete Kafka producer record from a JSON-LD document.

    Returns a dict compatible with ``confluent_kafka.Producer.produce()``
    and ``kafka.KafkaProducer.send()`` keyword arguments.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON (False).
        prefix: Topic prefix.
        context_registry: For CBOR context compression.

    Returns:
        Dict with keys: ``topic``, ``key``, ``value``, ``headers``,
        ``timestamp``.
    """
    return {
        "topic": derive_kafka_topic(doc, prefix=prefix),
        "key": derive_kafka_key(doc),
        "value": to_kafka_value(doc, compress=compress, context_registry=context_registry),
        "headers": derive_kafka_headers(doc, compress=compress),
        "timestamp": derive_kafka_timestamp(doc),
    }
