"""
WebSocket Transport Optimization for JSON-LD-Ex.

Derives WebSocket framing metadata and subprotocol negotiation
from JSON-LD annotations, enabling semantic-aware real-time
streaming of knowledge graph updates.

Metadata mappings:

- **Message type**: Text frame (opcode 0x1) for JSON-LD, binary
  frame (opcode 0x2) for CBOR — determined by compression flag.
- **Subprotocol**: ``jsonld-ex.cbor`` or ``jsonld-ex.jsonld`` for
  ``Sec-WebSocket-Protocol`` handshake negotiation.
- **Per-message metadata**: Application-level dict with
  ``jsonld_type``, ``jsonld_id``, ``jsonld_confidence``,
  ``jsonld_source``, ``content_type``, ``ttl_seconds`` — designed
  to be sent as a companion control message or envelope wrapper.

WebSocket (RFC 6455) has no native per-message header mechanism
like MQTT 5.0 or Kafka.  The metadata dict is provided for
application-level framing protocols (e.g., a JSON envelope with
``{"meta": {...}, "payload": ...}`` or a separate control message
on a multiplexed channel).

No WebSocket library is required — this module returns plain
strings, bytes, and dicts suitable for any WebSocket implementation
(``websockets``, ``aiohttp``, ``Socket.IO``, ``FastAPI WebSocket``).

References:
    RFC 6455: The WebSocket Protocol.
    RFC 7692: Compression Extensions for WebSocket.
"""

from __future__ import annotations

import json
import math
from typing import Any, List, Optional, Union

from jsonld_ex._transport_common import (
    find_valid_until,
    seconds_remaining,
)

try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor

    _HAS_CBOR = True
except ImportError:
    _HAS_CBOR = False


# ── Subprotocol identifiers ───────────────────────────────────────

SUBPROTOCOL_JSONLD: str = "jsonld-ex.jsonld"
"""WebSocket subprotocol for JSON-LD text frames."""

SUBPROTOCOL_CBOR: str = "jsonld-ex.cbor"
"""WebSocket subprotocol for CBOR binary frames."""

# ── WebSocket opcodes (RFC 6455 §5.2) ─────────────────────────────

WS_OPCODE_TEXT: int = 0x1
"""Text frame opcode (UTF-8 payload)."""

WS_OPCODE_BINARY: int = 0x2
"""Binary frame opcode (arbitrary bytes)."""


# ═══════════════════════════════════════════════════════════════════
# MESSAGE SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_ws_message(
    doc: dict[str, Any],
    compress: bool = True,
    context_registry: Optional[dict[str, int]] = None,
) -> Union[str, bytes]:
    """Serialize a JSON-LD document for WebSocket transmission.

    JSON mode returns a ``str`` (for text frames); CBOR mode returns
    ``bytes`` (for binary frames).  Most WebSocket libraries
    automatically select the frame type based on the Python type.

    Args:
        doc: JSON-LD document.
        compress: CBOR binary frame (True) or JSON text frame (False).
        context_registry: For CBOR context compression.

    Returns:
        ``str`` for JSON (text frame) or ``bytes`` for CBOR (binary
        frame).

    Raises:
        ImportError: If compress=True but ``cbor2`` is not installed.
    """
    if compress:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR WebSocket messages. "
                "Install with: pip install jsonld-ex[iot]"
            )
        return to_cbor(doc, context_registry)
    return json.dumps(doc, separators=(",", ":"))


def from_ws_message(
    message: Union[str, bytes],
    compressed: bool = True,
    context: Optional[Any] = None,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize a WebSocket message back to a JSON-LD document.

    Args:
        message: ``str`` from text frame or ``bytes`` from binary frame.
        compressed: CBOR (True) or JSON (False).
        context: Optional ``@context`` to reattach.
        context_registry: For CBOR context decompression.

    Returns:
        Restored JSON-LD document.
    """
    if compressed:
        if not _HAS_CBOR:
            raise ImportError(
                "cbor2 is required for CBOR WebSocket messages. "
                "Install with: pip install jsonld-ex[iot]"
            )
        data = message if isinstance(message, bytes) else message.encode("utf-8")
        doc = from_cbor(data, context_registry)
    else:
        text = message if isinstance(message, str) else message.decode("utf-8")
        doc = json.loads(text)

    if context is not None and "@context" not in doc:
        doc["@context"] = context

    return doc


# ═══════════════════════════════════════════════════════════════════
# SUBPROTOCOL NEGOTIATION
# ═══════════════════════════════════════════════════════════════════


def derive_ws_subprotocols(compress: bool = True) -> List[str]:
    """Derive WebSocket subprotocol list for handshake negotiation.

    The client sends a list of acceptable subprotocols in the
    ``Sec-WebSocket-Protocol`` header during the opening handshake
    (RFC 6455 §4.2.1).  The server selects one.

    The preferred subprotocol is listed first.

    Args:
        compress: Prefer CBOR (True) or JSON-LD (False).

    Returns:
        Ordered list of subprotocol strings.
    """
    if compress:
        return [SUBPROTOCOL_CBOR, SUBPROTOCOL_JSONLD]
    return [SUBPROTOCOL_JSONLD, SUBPROTOCOL_CBOR]


# ═══════════════════════════════════════════════════════════════════
# PER-MESSAGE METADATA
# ═══════════════════════════════════════════════════════════════════


def derive_ws_metadata(
    doc: dict[str, Any],
    compress: bool = True,
) -> dict[str, Any]:
    """Derive per-message metadata from JSON-LD annotations.

    WebSocket (RFC 6455) has no native per-message header mechanism.
    This metadata dict is provided for application-level framing —
    e.g., sent as a JSON envelope, a companion control message, or
    used internally by the application to make routing decisions.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON-LD (False).

    Returns:
        Dict with available metadata fields.  Only fields that can
        be derived are included.
    """
    meta: dict[str, Any] = {}

    # -- Frame type --
    meta["opcode"] = WS_OPCODE_BINARY if compress else WS_OPCODE_TEXT

    # -- Content type --
    meta["content_type"] = (
        "application/cbor" if compress else "application/ld+json"
    )

    # -- @type --
    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            meta["jsonld_type"] = str(type_val)

    # -- @id --
    doc_id = doc.get("@id")
    if doc_id is not None:
        meta["jsonld_id"] = str(doc_id)

    # -- @confidence --
    conf = doc.get("@confidence")
    if conf is not None:
        meta["jsonld_confidence"] = conf

    # -- @source --
    source = doc.get("@source")
    if source is not None:
        meta["jsonld_source"] = str(source)

    # -- TTL from @validUntil --
    valid_until = find_valid_until(doc)
    if valid_until is not None:
        remaining = seconds_remaining(valid_until)
        if remaining is not None and remaining > 0:
            meta["ttl_seconds"] = int(math.ceil(remaining))

    return meta
