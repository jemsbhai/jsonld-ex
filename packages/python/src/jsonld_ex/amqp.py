"""
AMQP Transport Optimization for JSON-LD-Ex.

Derives AMQP 0-9-1 message properties from JSON-LD metadata,
enabling semantic-aware routing, priority, and TTL on enterprise
message brokers (RabbitMQ, Azure Service Bus, Apache Qpid).

Metadata mappings:

- **Routing key**: Dot-separated ``prefix.type.id_fragment`` from
  ``@type`` and ``@id``, suitable for AMQP topic exchanges.
- **Priority** (0–9): Linearly mapped from ``@confidence`` (0.0–1.0).
- **Delivery mode**: Persistent (2) for high-confidence or verified
  data; transient (1) for low-confidence telemetry.
- **Expiration**: Milliseconds remaining from ``@validUntil``
  (AMQP 0-9-1 convention: string of milliseconds).
- **Content-Type**: ``application/cbor`` or ``application/ld+json``.
- **Message-ID**: From ``@id`` when present.
- **Timestamp**: Current epoch seconds at derivation time.
- **Headers**: ``x-jsonld-type``, ``x-jsonld-confidence``,
  ``x-jsonld-source``, ``x-jsonld-id`` for exchange-level routing
  via header bindings.

No AMQP client library is required — this module returns plain
dicts suitable for any AMQP client (``pika``, ``aio-pika``,
``kombu``, ``azure-servicebus``).

References:
    AMQP 0-9-1: https://www.rabbitmq.com/amqp-0-9-1-reference
    AMQP 1.0: OASIS Standard.
    RabbitMQ: https://www.rabbitmq.com/
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from jsonld_ex._transport_common import (
    extract_type_local,
    extract_id_fragment,
    find_valid_until,
    seconds_remaining,
    scan_confidence,
)

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False


# ── AMQP delivery modes (AMQP 0-9-1 §4.2.4.2) ────────────────────

DELIVERY_MODE_TRANSIENT: int = 1
"""Non-persistent message — may be lost on broker restart."""

DELIVERY_MODE_PERSISTENT: int = 2
"""Persistent message — survives broker restart."""


# ═══════════════════════════════════════════════════════════════════
# PAYLOAD SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_amqp_payload(
    doc: dict[str, Any],
    compress: bool = True,
    context_registry: Optional[dict[str, int]] = None,
) -> bytes:
    """Serialize a JSON-LD document for AMQP transmission.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON (False).
        context_registry: For CBOR context compression.

    Returns:
        Encoded bytes.

    Raises:
        ImportError: If compress=True but ``cbor2`` is not installed.
    """
    if compress:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR AMQP payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        return to_cbor(doc, context_registry)
    return json.dumps(doc, separators=(",", ":")).encode("utf-8")


def from_amqp_payload(
    payload: bytes,
    compressed: bool = True,
    context: Optional[Any] = None,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize an AMQP payload back to a JSON-LD document.

    Args:
        payload: Raw bytes from AMQP message body.
        compressed: CBOR (True) or JSON (False).
        context: Optional ``@context`` to reattach.
        context_registry: For CBOR context decompression.

    Returns:
        Restored JSON-LD document.
    """
    if compressed:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR AMQP payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        doc = from_cbor(payload, context_registry)
    else:
        doc = json.loads(payload.decode("utf-8"))

    if context is not None and "@context" not in doc:
        doc["@context"] = context

    return doc


# ═══════════════════════════════════════════════════════════════════
# ROUTING KEY DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_routing_key(
    doc: dict[str, Any],
    prefix: str = "ld",
) -> str:
    """Derive an AMQP routing key from JSON-LD metadata.

    Pattern: ``prefix.type_local.id_fragment``

    Uses dot separators (AMQP topic exchange convention) rather
    than slashes (MQTT/CoAP convention).

    Args:
        doc: JSON-LD document.
        prefix: First segment (default ``"ld"``).

    Returns:
        Dot-separated routing key string.

    Examples::

        >>> derive_routing_key({"@type": "SensorReading", "@id": "urn:sensor:imu-001"})
        'ld.SensorReading.imu-001'
    """
    type_str = extract_type_local(doc)
    id_str = extract_id_fragment(doc)
    return f"{prefix}.{type_str}.{id_str}"


# ═══════════════════════════════════════════════════════════════════
# PRIORITY DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_amqp_priority(doc: dict[str, Any]) -> int:
    """Map document confidence to AMQP priority (0–9).

    Linear mapping: ``priority = round(confidence * 9)``.

    AMQP 0-9-1 supports priority 0–9 (§4.2.4.2). Higher values
    indicate higher priority.

    Falls back to 4 (middle) if no confidence metadata is found.

    Args:
        doc: JSON-LD document.

    Returns:
        Integer priority in [0, 9].
    """
    conf, _, _ = scan_confidence(doc)

    if conf is None:
        return 4  # Middle priority as safe default

    # Linear map [0.0, 1.0] → [0, 9]
    return max(0, min(9, round(conf * 9)))


# ═══════════════════════════════════════════════════════════════════
# HEADERS DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_amqp_headers(doc: dict[str, Any]) -> dict[str, str]:
    """Derive AMQP message headers from JSON-LD metadata.

    Headers are key-value string pairs carried in the AMQP
    ``headers`` property table.  They enable routing via AMQP
    headers exchanges (binding on header values).

    Args:
        doc: JSON-LD document.

    Returns:
        Dict of header name → string value.
    """
    headers: dict[str, str] = {}

    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            headers["x-jsonld-type"] = str(type_val)

    conf = doc.get("@confidence")
    if conf is not None:
        headers["x-jsonld-confidence"] = str(conf)

    source = doc.get("@source")
    if source is not None:
        headers["x-jsonld-source"] = str(source)

    doc_id = doc.get("@id")
    if doc_id is not None:
        headers["x-jsonld-id"] = str(doc_id)

    return headers


# ═══════════════════════════════════════════════════════════════════
# FULL PROPERTY DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_amqp_properties(
    doc: dict[str, Any],
    compress: bool = True,
    prefix: str = "ld",
) -> dict[str, Any]:
    """Derive AMQP 0-9-1 message properties from JSON-LD metadata.

    Returns a dict compatible with ``pika.BasicProperties`` and
    similar AMQP client APIs.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON (False).
        prefix: Routing key prefix.

    Returns:
        Dict with AMQP property names as keys.
    """
    props: dict[str, Any] = {}

    # -- Content-Type --
    props["content_type"] = (
        "application/cbor" if compress else "application/ld+json"
    )

    # -- Routing Key --
    props["routing_key"] = derive_routing_key(doc, prefix=prefix)

    # -- Priority (0-9) --
    props["priority"] = derive_amqp_priority(doc)

    # -- Delivery Mode --
    conf, human_verified, _ = scan_confidence(doc)
    if human_verified or conf is None or conf >= 0.5:
        props["delivery_mode"] = DELIVERY_MODE_PERSISTENT
    else:
        props["delivery_mode"] = DELIVERY_MODE_TRANSIENT

    # -- Timestamp (epoch seconds) --
    props["timestamp"] = int(time.time())

    # -- Message ID from @id --
    doc_id = doc.get("@id")
    if doc_id is not None:
        props["message_id"] = str(doc_id)

    # -- Expiration from @validUntil --
    # AMQP 0-9-1 expiration is a string of milliseconds
    valid_until = find_valid_until(doc)
    if valid_until is not None:
        remaining = seconds_remaining(valid_until)
        if remaining is not None and remaining > 0:
            props["expiration"] = str(int(remaining * 1000))

    # -- Headers --
    headers = derive_amqp_headers(doc)
    if headers:
        props["headers"] = headers

    return props
