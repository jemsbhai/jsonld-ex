"""
CoAP Transport Optimization for JSON-LD-Ex.

Derives Constrained Application Protocol (RFC 7252) options from
JSON-LD metadata, enabling semantic-aware transport on constrained
IoT networks.  Complements the MQTT module — together they cover
pub/sub (MQTT) and request/response (CoAP) IoT patterns.

Metadata mappings:

- **Content-Format** (option 12): ``60`` for CBOR, ``50`` for JSON,
  ``11050`` for ``application/ld+json``.
- **ETag** (option 4): Derived from ``@integrity`` hash, truncated
  to <= 8 bytes per RFC 7252 §5.10.6.
- **Max-Age** (option 14): Seconds remaining from ``@validUntil``.
  Default 60 s per RFC 7252 §5.10.5.
- **Uri-Path** (option 11): Segments derived from ``@type`` and
  ``@id``, analogous to MQTT topic derivation.
- **Message type** (header): CON (confirmable) for high-confidence
  or critical data; NON (non-confirmable) for low-confidence
  telemetry.
- **Size1** (option 60): Payload byte length.
- **Block-wise** (RFC 7959): Block size recommendation when payload
  exceeds 1024 bytes (typical constrained MTU).
- **Observe** (RFC 7641): Flagged when ``@validUntil`` is present,
  indicating the resource has temporal semantics suitable for
  observation.

Requires the ``cbor2`` package for compressed payloads::

    pip install jsonld-ex[iot]

No CoAP client library is required — this module provides option
derivation and payload serialization that can be used with any
CoAP implementation (e.g. ``aiocoap``, ``CoAPthon3``).

References:
    RFC 7252: The Constrained Application Protocol (CoAP).
    RFC 7641: Observing Resources in CoAP.
    RFC 7959: Block-Wise Transfers in CoAP.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

from jsonld_ex.ai_ml import get_confidence

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False


# ── CoAP Content-Format IDs (IANA CoRE Parameters registry) ───────

CONTENT_FORMAT_CBOR: int = 60
"""Content-Format ID for ``application/cbor`` (RFC 8949)."""

CONTENT_FORMAT_JSON: int = 50
"""Content-Format ID for ``application/json`` (RFC 7159)."""

CONTENT_FORMAT_JSONLD: int = 11050
"""Content-Format ID for ``application/ld+json``.

Note: As of 2026, ``application/ld+json`` does not have an official
IANA CoAP Content-Format assignment.  The value 11050 is in the
First Come First Served range (10000–64999) and is used here as
an illustrative placeholder.  Implementers SHOULD register an
official Content-Format ID with IANA for production use.
"""

# ── CoAP message types (RFC 7252 §4.2–4.3) ────────────────────────

MESSAGE_TYPE_CON: int = 0
"""Confirmable message — requires acknowledgement."""

MESSAGE_TYPE_NON: int = 1
"""Non-confirmable message — fire-and-forget."""

# ── Block size exponents (RFC 7959 §2.2) ──────────────────────────
# SZX = 0..6, block_size = 2^(SZX+4) = 16..1024 bytes

_BLOCK_THRESHOLD = 1024  # recommend block transfer above this


# ═══════════════════════════════════════════════════════════════════
# PAYLOAD SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_coap_payload(
    doc: dict[str, Any],
    compress: bool = True,
    max_payload: int = 1024,
    context_registry: Optional[dict[str, int]] = None,
) -> bytes:
    """Serialize a JSON-LD document for CoAP transmission.

    Args:
        doc: JSON-LD document.
        compress: Use CBOR encoding (True) or JSON (False).
        max_payload: Maximum payload in bytes.  Default 1024 for
            constrained networks (single UDP datagram).  Set higher
            for block-wise transfer.
        context_registry: Context URL → integer mapping for CBOR.

    Returns:
        Encoded bytes.

    Raises:
        ValueError: If payload exceeds *max_payload*.
        ImportError: If compress=True but ``cbor2`` is not installed.
    """
    if compress:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR CoAP payloads. "
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


def from_coap_payload(
    payload: bytes,
    compressed: bool = True,
    context: Optional[Any] = None,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize a CoAP payload back to a JSON-LD document.

    Args:
        payload: Raw bytes from CoAP response/notification.
        compressed: CBOR (True) or JSON (False).
        context: Optional ``@context`` to reattach.
        context_registry: Registry for CBOR context decompression.

    Returns:
        Restored JSON-LD document.
    """
    if compressed:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR CoAP payloads. "
                "Install with: pip install jsonld-ex[iot]"
            )
        doc = from_cbor(payload, context_registry)
    else:
        doc = json.loads(payload.decode("utf-8"))

    if context is not None and "@context" not in doc:
        doc["@context"] = context

    return doc


# ═══════════════════════════════════════════════════════════════════
# URI-PATH DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_coap_uri_path(
    doc: dict[str, Any],
    prefix: str = "ld",
) -> list[str]:
    """Derive CoAP Uri-Path option segments from JSON-LD metadata.

    Pattern: ``[prefix, @type_local, @id_fragment]``

    Each segment becomes a separate Uri-Path option (RFC 7252
    §5.10.1), enabling CoAP proxies to route by type and identity.

    Args:
        doc: JSON-LD document.
        prefix: First path segment (default ``"ld"``).

    Returns:
        List of URI path segments.

    Examples::

        >>> derive_coap_uri_path({"@type": "SensorReading", "@id": "urn:sensor:imu-001"})
        ['ld', 'SensorReading', 'imu-001']
    """
    # Extract type
    type_val = doc.get("@type", "unknown")
    if isinstance(type_val, list):
        type_val = type_val[0] if type_val else "unknown"
    type_str = _local_name(str(type_val))

    # Extract id fragment
    id_val = doc.get("@id", "unknown")
    id_str = _local_name(str(id_val))

    # Sanitise (reuse MQTT-compatible sanitisation)
    type_str = _sanitise_segment(type_str)
    id_str = _sanitise_segment(id_str)

    return [prefix, type_str, id_str]


# ═══════════════════════════════════════════════════════════════════
# MESSAGE TYPE DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_coap_message_type(doc: dict[str, Any]) -> int:
    """Map document confidence to CoAP message type.

    Heuristic:
        - ``@humanVerified = True`` → CON (confirmable)
        - ``@confidence >= 0.5``    → CON (reliable delivery)
        - ``@confidence < 0.5``     → NON (fire-and-forget)
        - No metadata               → CON (safe default)

    Unlike MQTT's 3 QoS levels, CoAP has only CON vs NON for
    reliability.  The mapping is deliberately conservative: only
    clearly low-confidence telemetry gets NON.

    Args:
        doc: JSON-LD document.

    Returns:
        :data:`MESSAGE_TYPE_CON` or :data:`MESSAGE_TYPE_NON`.
    """
    # Check @humanVerified first
    if doc.get("@humanVerified", False) is True:
        return MESSAGE_TYPE_CON

    # Check document-level confidence
    conf = get_confidence(doc)

    # Scan properties if no document-level confidence
    if conf is None:
        for key, val in doc.items():
            if key.startswith("@"):
                continue
            if isinstance(val, dict):
                if val.get("@humanVerified", False) is True:
                    return MESSAGE_TYPE_CON
                prop_conf = get_confidence(val)
                if prop_conf is not None:
                    conf = prop_conf
                    break

    # No confidence found → safe default is CON
    if conf is None:
        return MESSAGE_TYPE_CON

    # Low confidence → NON (fire-and-forget)
    if conf < 0.5:
        return MESSAGE_TYPE_NON

    return MESSAGE_TYPE_CON


# ═══════════════════════════════════════════════════════════════════
# FULL OPTION DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_coap_options(
    doc: dict[str, Any],
    compress: bool = True,
    prefix: str = "ld",
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Derive CoAP options from JSON-LD metadata.

    Combines all derivation functions into a single options dict
    suitable for passing to any CoAP client library.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON (False).
        prefix: Uri-Path prefix.
        context_registry: For CBOR context compression.

    Returns:
        Dict with CoAP options:

        - ``content_format`` (int): Content-Format option value.
        - ``uri_path`` (list[str]): Uri-Path segments.
        - ``message_type`` (int): CON or NON.
        - ``size1`` (int): Payload byte length.
        - ``etag`` (bytes, optional): From ``@integrity``.
        - ``max_age`` (int, optional): From ``@validUntil``.
        - ``block_szx`` (int, optional): Block size exponent if
          payload > 1024 bytes.
        - ``observable`` (bool, optional): True if temporal metadata
          suggests the resource should support Observe (RFC 7641).
    """
    options: dict[str, Any] = {}

    # -- Content-Format (§5.10.3) --
    if compress:
        options["content_format"] = CONTENT_FORMAT_CBOR
    else:
        options["content_format"] = CONTENT_FORMAT_JSONLD

    # -- Uri-Path (§5.10.1) --
    options["uri_path"] = derive_coap_uri_path(doc, prefix=prefix)

    # -- Message Type (§4.2–4.3) --
    options["message_type"] = derive_coap_message_type(doc)

    # -- Size1 (§5.10.9) --
    # Compute actual payload size for the Size1 option
    try:
        if compress and _HAS_CBOR:
            payload = to_cbor(doc, context_registry)
        else:
            payload = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        payload_size = len(payload)
    except Exception:
        payload_size = 0

    options["size1"] = payload_size

    # -- Block SZX (RFC 7959 §2.2) --
    if payload_size > _BLOCK_THRESHOLD:
        # Recommend the largest standard block size (SZX=6 → 1024 bytes)
        options["block_szx"] = 6

    # -- ETag (§5.10.6) --
    etag = _derive_etag(doc)
    if etag is not None:
        options["etag"] = etag

    # -- Max-Age (§5.10.5) --
    max_age = _derive_max_age(doc)
    if max_age is not None:
        options["max_age"] = max_age

    # -- Observable (RFC 7641) --
    has_temporal = _has_valid_until(doc)
    if has_temporal:
        options["observable"] = True

    return options


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _local_name(iri: str) -> str:
    """Extract the local/fragment part of an IRI or URN."""
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    if ":" in iri:
        return iri.rsplit(":", 1)[-1]
    return iri


def _sanitise_segment(segment: str) -> str:
    """Clean a URI path segment.

    Removes characters problematic for CoAP URI paths (null bytes,
    control characters).  Similar to MQTT topic sanitisation.
    """
    sanitised = re.sub(r"[#+\x00]", "_", segment)
    sanitised = sanitised.lstrip("$")
    return sanitised or "unknown"


def _derive_etag(doc: dict[str, Any]) -> Optional[bytes]:
    """Derive ETag from ``@integrity`` metadata.

    The ``@integrity`` value (e.g. ``"sha256-abc123..."```) is hashed
    to produce a deterministic ETag.  Per RFC 7252 §5.10.6, ETag is
    an opaque sequence of 1–8 bytes.  We take the first 8 bytes of
    the SHA-256 hash of the integrity string.

    Returns None if no ``@integrity`` is present.
    """
    integrity = doc.get("@integrity")
    if integrity is None:
        return None

    # Hash the integrity string to get a deterministic 8-byte ETag
    h = hashlib.sha256(str(integrity).encode("utf-8")).digest()
    return h[:8]  # Truncate to 8 bytes (CoAP ETag max)


def _derive_max_age(doc: dict[str, Any]) -> Optional[int]:
    """Derive Max-Age from ``@validUntil`` (seconds remaining).

    Scans document-level and property-level ``@validUntil``.
    Returns None if not found or already expired.
    """
    valid_until = doc.get("@validUntil")

    # Fall back to property-level
    if valid_until is None:
        for key, val in doc.items():
            if key.startswith("@"):
                continue
            if isinstance(val, dict) and "@validUntil" in val:
                valid_until = val["@validUntil"]
                break

    if valid_until is None:
        return None

    try:
        if isinstance(valid_until, str):
            dt_str = valid_until.replace("Z", "+00:00")
            expiry_dt = datetime.fromisoformat(dt_str)
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
        else:
            return None

        now = datetime.now(timezone.utc)
        remaining = (expiry_dt - now).total_seconds()

        if remaining <= 0:
            return None

        # CoAP Max-Age is uint32 (same as MQTT Message Expiry)
        return min(int(math.ceil(remaining)), 0xFFFFFFFF)

    except (ValueError, TypeError, OverflowError):
        return None


def _has_valid_until(doc: dict[str, Any]) -> bool:
    """Check whether the document has any ``@validUntil`` annotation."""
    if "@validUntil" in doc:
        return True
    for key, val in doc.items():
        if key.startswith("@"):
            continue
        if isinstance(val, dict) and "@validUntil" in val:
            return True
    return False
