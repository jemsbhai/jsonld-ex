"""
MQTT Transport Optimization for JSON-LD-Ex.

Optimises jsonld-ex documents for IoT pub/sub via MQTT, with:
  - CBOR or JSON payload serialization
  - Automatic MQTT topic derivation from ``@type`` and ``@id``
  - QoS level mapping from ``@confidence``

Requires the ``cbor2`` package for compressed payloads::

    pip install jsonld-ex[iot]

The optional ``paho-mqtt`` package is NOT required by this module —
it provides serialization/deserialization and metadata derivation that
can be used with any MQTT client.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from jsonld_ex.ai_ml import get_confidence

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False


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

    Args:
        doc: JSON-LD document or node.
        prefix: Topic prefix (default ``"ld"``).

    Returns:
        MQTT topic string.

    Examples::

        >>> derive_mqtt_topic({"@type": "SensorReading", "@id": "urn:sensor:imu-001"})
        'ld/SensorReading/imu-001'
    """
    # Extract type
    type_val = doc.get("@type", "unknown")
    if isinstance(type_val, list):
        type_val = type_val[0] if type_val else "unknown"
    # Strip namespace prefix
    type_str = _local_name(str(type_val))

    # Extract id fragment
    id_val = doc.get("@id", "unknown")
    id_str = _local_name(str(id_val))

    # Sanitise for MQTT topic (no #, +, or null)
    type_str = _sanitise_topic_segment(type_str)
    id_str = _sanitise_topic_segment(id_str)

    return f"{prefix}/{type_str}/{id_str}"


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
    # Check document-level confidence
    conf = get_confidence(doc)

    # Check if human-verified at document level
    if doc.get("@humanVerified") is True:
        return 2

    # If no document-level confidence, scan first annotated property
    if conf is None:
        for key, val in doc.items():
            if key.startswith("@"):
                continue
            if isinstance(val, dict):
                conf = get_confidence(val)
                if val.get("@humanVerified") is True:
                    return 2
                if conf is not None:
                    break

    if conf is None:
        return 1  # default

    if conf >= 0.9:
        return 2
    if conf >= 0.5:
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _local_name(iri: str) -> str:
    """Extract the local/fragment part of an IRI or URN."""
    # Try fragment first
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    # Try last path segment
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    # Try URN
    if ":" in iri:
        return iri.rsplit(":", 1)[-1]
    return iri


def _sanitise_topic_segment(segment: str) -> str:
    """Remove MQTT-illegal characters from a topic segment."""
    # MQTT wildcards # and + are not allowed in published topics
    return re.sub(r"[#+\x00]", "_", segment)
