"""
MQTT Transport Optimization for JSON-LD-Ex.

Optimises jsonld-ex documents for IoT pub/sub via MQTT, with:
  - CBOR or JSON payload serialization
  - Automatic MQTT topic derivation from ``@type`` and ``@id``
  - QoS level mapping from ``@confidence``
  - MQTT 5.0 PUBLISH property derivation

Requires the ``cbor2`` package for compressed payloads::

    pip install jsonld-ex[iot]

The optional ``paho-mqtt`` package is NOT required by this module —
it provides serialization/deserialization and metadata derivation that
can be used with any MQTT client.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

from jsonld_ex.ai_ml import get_confidence
from jsonld_ex._transport_common import (
    local_name,
    sanitise_segment,
    extract_type_local,
    extract_id_fragment,
    derive_expiry_seconds,
    scan_confidence,
)

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False

# MQTT spec: topic name MUST NOT exceed 65,535 bytes (UTF-8 encoded).
_MAX_TOPIC_BYTES = 65_535


# ═══════════════════════════════════════════════════════════════════
# PAYLOAD SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_mqtt_payload(
    doc: dict[str, Any],
    compress: bool = True,
    max_payload: int = 256_000,
    context_registry: Optional[dict[str, int]] = None,
) -> bytes:
    """Serialize a jsonld-ex document for MQTT transmission.

    When *compress* is True, uses CBOR encoding with context
    compression for minimal payload size.  Falls back to compact
    JSON if ``cbor2`` is not installed.

    The ``@context`` can optionally be stripped (replaced with a
    registry reference) to save bytes when sender and receiver share
    a context registry.

    Args:
        doc: JSON-LD document with jsonld-ex extensions.
        compress: Use CBOR encoding (True) or JSON (False).
        max_payload: Maximum payload size in bytes (MQTT broker limit).
            Defaults to 256 KB (MQTT v3.1.1 default).
        context_registry: Context URL → integer mapping for CBOR
            compression.

    Returns:
        Encoded bytes ready for MQTT publish.

    Raises:
        ValueError: If the serialized payload exceeds *max_payload*.
        ImportError: If compress=True but ``cbor2`` is not installed.
    """
    if compress:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for compressed MQTT payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        payload = to_cbor(doc, context_registry)
    else:
        payload = json.dumps(doc, separators=(",", ":")).encode("utf-8")

    if len(payload) > max_payload:
        raise ValueError(
            f"Payload size {len(payload)} bytes exceeds max_payload "
            f"({max_payload} bytes)"
        )

    return payload


def from_mqtt_payload(
    payload: bytes,
    context: Optional[Any] = None,
    compressed: bool = True,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize an MQTT payload back to a jsonld-ex document.

    Args:
        payload: Raw bytes received from MQTT.
        context: Optional ``@context`` to reattach (if it was stripped
            during serialization).
        compressed: Whether the payload is CBOR (True) or JSON (False).
        context_registry: Registry used during serialization, for CBOR
            context decompression.

    Returns:
        Restored JSON-LD document.
    """
    if compressed:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for compressed MQTT payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        doc = from_cbor(payload, context_registry)
    else:
        doc = json.loads(payload.decode("utf-8"))

    if context is not None and "@context" not in doc:
        doc["@context"] = context

    return doc


# ═══════════════════════════════════════════════════════════════════
# TOPIC DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_mqtt_topic(
    doc: dict[str, Any],
    prefix: str = "ld",
) -> str:
    """Generate an MQTT topic from JSON-LD document metadata.

    Pattern: ``{prefix}/{@type}/{@id_fragment}``

    The ``@id`` is reduced to its last path segment or fragment to
    keep topics short.  If ``@type`` or ``@id`` are missing, the
    corresponding segment is ``"unknown"``.

    Per MQTT spec (v3.1.1 §4.7, v5.0 §4.7):

    - Topic names are UTF-8 encoded strings, max 65,535 bytes.
    - Wildcard characters ``#`` and ``+`` are forbidden in PUBLISH
      topic names.
    - Null character (``\\x00``) is forbidden.
    - Topics starting with ``$`` are reserved for broker system use.

    Args:
        doc: JSON-LD document or node.
        prefix: Topic prefix (default ``"ld"``).

    Returns:
        MQTT topic string.

    Raises:
        ValueError: If the generated topic exceeds 65,535 bytes
            (MQTT spec limit).

    Examples::

        >>> derive_mqtt_topic({"@type": "SensorReading", "@id": "urn:sensor:imu-001"})
        'ld/SensorReading/imu-001'
    """
    type_str = extract_type_local(doc)
    id_str = extract_id_fragment(doc)

    topic = f"{prefix}/{type_str}/{id_str}"

    # MQTT spec: topic name MUST NOT exceed 65,535 bytes (UTF-8).
    topic_bytes = len(topic.encode("utf-8"))
    if topic_bytes > _MAX_TOPIC_BYTES:
        raise ValueError(
            f"Generated MQTT topic is {topic_bytes} bytes, exceeding "
            f"the MQTT spec limit of {_MAX_TOPIC_BYTES} bytes."
        )

    return topic


# ═══════════════════════════════════════════════════════════════════
# QOS DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_mqtt_qos(doc: dict[str, Any]) -> int:
    """Map document confidence to MQTT QoS level.

    Heuristic mapping:
        - QoS 0 (at most once):  ``@confidence < 0.5``
            Low-priority or noisy sensor data.
        - QoS 1 (at least once): ``0.5 <= @confidence < 0.9``
            Normal priority.
        - QoS 2 (exactly once):  ``@confidence >= 0.9`` or
            ``@humanVerified = True``
            Critical/verified data.

    If no confidence is found at the document level, inspects the
    first property value.  If still not found, defaults to QoS 1.

    Args:
        doc: JSON-LD document or node.

    Returns:
        MQTT QoS level: 0, 1, or 2.
    """
    return derive_mqtt_qos_detailed(doc)["qos"]


def derive_mqtt_qos_detailed(doc: dict[str, Any]) -> dict[str, Any]:
    """Map document confidence to MQTT QoS level with reasoning.

    Same heuristic as :func:`derive_mqtt_qos` but returns a dict
    with the QoS level, human-readable reasoning, the confidence
    value used (if any), the source location, and whether
    ``@humanVerified`` was set.

    Returns:
        Dict with keys ``qos``, ``reasoning``, ``confidence_used``,
        ``human_verified``.
    """
    conf = get_confidence(doc)
    human_verified = doc.get("@humanVerified", False) is True
    source = "document-level"

    # Document-level humanVerified
    if human_verified:
        return {
            "qos": 2,
            "reasoning": f"@humanVerified=True ({source}) \u2192 QoS 2 (exactly once)",
            "confidence_used": conf,
            "human_verified": True,
        }

    # If no document-level confidence, scan first annotated property
    if conf is None:
        for key, val in doc.items():
            if key.startswith("@"):
                continue
            if isinstance(val, dict):
                prop_conf = get_confidence(val)
                prop_hv = val.get("@humanVerified", False) is True
                if prop_hv:
                    return {
                        "qos": 2,
                        "reasoning": (
                            f"@humanVerified=True (property '{key}') "
                            "\u2192 QoS 2 (exactly once)"
                        ),
                        "confidence_used": prop_conf,
                        "human_verified": True,
                    }
                if prop_conf is not None:
                    conf = prop_conf
                    source = f"property '{key}'"
                    break

    if conf is None:
        return {
            "qos": 1,
            "reasoning": "No confidence metadata found \u2192 QoS 1 (default)",
            "confidence_used": None,
            "human_verified": False,
        }

    if conf >= 0.9:
        qos = 2
        reasoning = f"@confidence={conf} \u2265 0.9 ({source}) \u2192 QoS 2 (exactly once)"
    elif conf >= 0.5:
        qos = 1
        reasoning = f"0.5 \u2264 @confidence={conf} < 0.9 ({source}) \u2192 QoS 1 (at least once)"
    else:
        qos = 0
        reasoning = f"@confidence={conf} < 0.5 ({source}) \u2192 QoS 0 (at most once)"

    return {
        "qos": qos,
        "reasoning": reasoning,
        "confidence_used": conf,
        "human_verified": False,
    }


# ═══════════════════════════════════════════════════════════════════
# MQTT 5.0 PUBLISH PROPERTIES
# ═══════════════════════════════════════════════════════════════════


def derive_mqtt5_properties(
    doc: dict[str, Any],
    compress: bool = True,
) -> dict[str, Any]:
    """Derive MQTT 5.0 PUBLISH packet properties from a JSON-LD document.

    MQTT 5.0 (OASIS Standard, §3.3.2.3) introduced several PUBLISH
    properties that map naturally to JSON-LD-Ex metadata:

    - **Payload Format Indicator** (byte): ``0`` = unspecified bytes
      (CBOR), ``1`` = UTF-8 character data (JSON).
    - **Content Type** (UTF-8 string): MIME type of the payload.
      ``"application/cbor"`` for compressed, ``"application/ld+json"``
      for uncompressed.
    - **Message Expiry Interval** (uint32, seconds): Derived from
      ``@validUntil`` temporal annotation — seconds remaining until
      the statement expires.  ``None`` if no temporal bound.
    - **User Properties** (key-value pairs): Extracted from JSON-LD
      metadata — ``@type``, ``@source``, ``@confidence``.

    These properties are returned as a plain dict suitable for passing
    to any MQTT 5.0 client library (e.g. ``paho-mqtt``, ``gmqtt``).

    Args:
        doc: JSON-LD document or node.
        compress: Whether the payload will be CBOR-compressed (True)
            or JSON (False).  Determines format indicator and content
            type.

    Returns:
        Dict with MQTT 5.0 property names as keys.  Only properties
        that can be derived are included.

    Examples::

        >>> props = derive_mqtt5_properties(
        ...     {"@type": "SensorReading", "@confidence": 0.95},
        ...     compress=True,
        ... )
        >>> props["payload_format_indicator"]
        0
        >>> props["content_type"]
        'application/cbor'
    """
    props: dict[str, Any] = {}

    # --- Payload Format Indicator (§3.3.2.3.2) ---
    # 0 = unspecified byte stream (CBOR), 1 = UTF-8 (JSON)
    props["payload_format_indicator"] = 0 if compress else 1

    # --- Content Type (§3.3.2.3.9) ---
    props["content_type"] = (
        "application/cbor" if compress else "application/ld+json"
    )

    # --- Message Expiry Interval (§3.3.2.3.3) ---
    expiry = derive_expiry_seconds(doc)
    if expiry is not None:
        props["message_expiry_interval"] = expiry

    # --- User Properties (§3.3.2.3.7) ---
    user_props: list[tuple[str, str]] = []

    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            user_props.append(("jsonld_type", str(type_val)))

    conf = get_confidence(doc)
    if conf is not None:
        user_props.append(("jsonld_confidence", str(conf)))

    source = doc.get("@source")
    if source is not None:
        user_props.append(("jsonld_source", str(source)))

    doc_id = doc.get("@id")
    if doc_id is not None:
        user_props.append(("jsonld_id", str(doc_id)))

    if user_props:
        props["user_properties"] = user_props

    return props


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


# Backward-compatible aliases for private helpers that were
# extracted to _transport_common.  Preserves imports in existing
# test code and any downstream users of these private names.
_local_name = local_name
_sanitise_topic_segment = sanitise_segment
_derive_expiry_seconds = derive_expiry_seconds
