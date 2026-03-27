"""
gRPC Transport Optimization for JSON-LD-Ex.

Derives gRPC metadata and generates Protocol Buffer schema suggestions
from JSON-LD annotations, bridging the semantic web and RPC worlds.

gRPC is fundamentally different from the other transport modules:
it uses compiled Protobuf schemas rather than schema-free serialization.
This module addresses that gap with two approaches:

1. **Metadata derivation**: Maps JSON-LD annotations to gRPC metadata
   key-value pairs (analogous to HTTP headers), enabling server-side
   interceptors to route, filter, or log based on semantic metadata
   without deserializing the payload.

2. **Proto schema suggestion**: Generates a ``.proto`` file skeleton
   from a JSON-LD document's structure, helping developers bootstrap
   a typed gRPC service definition from their existing JSON-LD data
   model.  The output is a *suggestion* — not a compiler, not a
   runtime serializer.

Metadata mappings:

- ``x-jsonld-type``: From ``@type`` (first element if array).
- ``x-jsonld-confidence``: From ``@confidence``.
- ``x-jsonld-source``: From ``@source``.
- ``x-jsonld-id``: From ``@id``.
- ``x-jsonld-content-type``: Always ``application/ld+json``.

No gRPC library is required — this module returns plain lists and
strings suitable for any gRPC implementation (``grpcio``, ``grpclib``,
``betterproto``).

References:
    gRPC Metadata: https://grpc.io/docs/guides/metadata/
    Protocol Buffers v3: https://protobuf.dev/programming-guides/proto3/
    gRPC Python: https://grpc.github.io/grpc/python/

Caveats:
    - The proto schema suggestion is heuristic: it infers types from
      Python values in a single document instance, not from a schema.
      It cannot discover optional fields not present in the sample,
      enum types, or oneof variants.
    - gRPC metadata keys must be lowercase ASCII.  Keys ending in
      ``-bin`` carry binary values; all others carry ASCII text.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Tuple

from jsonld_ex._transport_common import (
    extract_type_local,
    local_name,
)


# ── Metadata key constants ─────────────────────────────────────────

METADATA_KEY_TYPE: str = "x-jsonld-type"
"""gRPC metadata key for ``@type``."""

METADATA_KEY_CONFIDENCE: str = "x-jsonld-confidence"
"""gRPC metadata key for ``@confidence``."""

METADATA_KEY_SOURCE: str = "x-jsonld-source"
"""gRPC metadata key for ``@source``."""

METADATA_KEY_ID: str = "x-jsonld-id"
"""gRPC metadata key for ``@id``."""

METADATA_KEY_CONTENT_TYPE: str = "x-jsonld-content-type"
"""gRPC metadata key for the JSON-LD content type."""


# ═══════════════════════════════════════════════════════════════════
# gRPC METADATA DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_grpc_metadata(
    doc: dict[str, Any],
) -> List[Tuple[str, str]]:
    """Derive gRPC metadata from JSON-LD annotations.

    gRPC metadata is a list of ``(key, value)`` tuples where keys
    are lowercase ASCII strings and values are strings (text metadata)
    or bytes (binary metadata, key ending in ``-bin``).

    These metadata entries can be attached to gRPC calls as initial
    or trailing metadata, enabling server interceptors to inspect
    semantic properties without deserializing the Protobuf payload.

    Args:
        doc: JSON-LD document.

    Returns:
        List of ``(str, str)`` metadata tuples.
    """
    meta: List[Tuple[str, str]] = []

    # -- Content type (always present) --
    meta.append((METADATA_KEY_CONTENT_TYPE, "application/ld+json"))

    # -- @type --
    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            meta.append((METADATA_KEY_TYPE, str(type_val)))

    # -- @confidence --
    conf = doc.get("@confidence")
    if conf is not None:
        meta.append((METADATA_KEY_CONFIDENCE, str(conf)))

    # -- @source --
    source = doc.get("@source")
    if source is not None:
        meta.append((METADATA_KEY_SOURCE, str(source)))

    # -- @id --
    doc_id = doc.get("@id")
    if doc_id is not None:
        meta.append((METADATA_KEY_ID, str(doc_id)))

    return meta


# ═══════════════════════════════════════════════════════════════════
# JSON SERIALIZATION (for gRPC JSON transcoding)
# ═══════════════════════════════════════════════════════════════════


def to_grpc_json(doc: dict[str, Any]) -> str:
    """Serialize a JSON-LD document as compact JSON for gRPC transcoding.

    gRPC JSON transcoding (grpc-gateway, Envoy) allows HTTP/JSON
    clients to call gRPC services.  This function produces compact
    JSON suitable for the JSON body of a transcoded request.

    Args:
        doc: JSON-LD document.

    Returns:
        Compact JSON string.
    """
    return json.dumps(doc, separators=(",", ":"))


def from_grpc_json(
    payload: str,
    context: Optional[Any] = None,
) -> dict[str, Any]:
    """Deserialize a gRPC JSON transcoding response.

    Args:
        payload: JSON string from transcoded gRPC response.
        context: Optional ``@context`` to reattach.

    Returns:
        Restored JSON-LD document.
    """
    doc = json.loads(payload)
    if context is not None and "@context" not in doc:
        doc["@context"] = context
    return doc


# ═══════════════════════════════════════════════════════════════════
# PROTO SCHEMA SUGGESTION
# ═══════════════════════════════════════════════════════════════════


def suggest_proto_schema(doc: dict[str, Any]) -> str:
    """Generate a .proto schema suggestion from JSON-LD document structure.

    Inspects the document's properties and their Python types to
    produce a proto3 message definition.  This is a *heuristic
    suggestion*, not a definitive schema — it infers types from a
    single document instance.

    Limitations (reported honestly):
        - Cannot discover optional fields absent from the sample.
        - Cannot infer enum types or oneof variants.
        - Nested objects become sub-messages only one level deep.
        - Array element types are inferred from the first element.

    Args:
        doc: JSON-LD document.

    Returns:
        Proto3 schema string.
    """
    # -- Message name from @type --
    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
    if type_val is not None:
        msg_name = local_name(str(type_val))
    else:
        msg_name = "JsonLdDocument"

    # -- Collect fields --
    fields: List[Tuple[str, str, str]] = []  # (proto_type, field_name, comment)
    field_num = 1

    # @id → string id
    if "@id" in doc:
        fields.append(("string", "id", "@id"))
        field_num += 1

    # @confidence → double confidence
    if "@confidence" in doc:
        fields.append(("double", "confidence", "@confidence"))
        field_num += 1

    # @source → string source
    if "@source" in doc:
        fields.append(("string", "source", "@source"))
        field_num += 1

    # Data properties
    for key, val in doc.items():
        if key.startswith("@"):
            continue

        proto_name = _to_snake_case(key)
        proto_type, comment = _infer_proto_type(val)
        fields.append((proto_type, proto_name, comment))

    # -- Build proto file --
    lines = [
        'syntax = "proto3";',
        "",
        f"// Auto-suggested from JSON-LD @type: {type_val or 'unknown'}",
        f"// Generated by jsonld-ex proto schema suggestion.",
        f"// This is a heuristic — review and adjust before use.",
        "",
        f"message {msg_name} {{",
    ]

    for i, (ptype, pname, comment) in enumerate(fields, start=1):
        lines.append(f"  {ptype} {pname} = {i}; // {comment}")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _to_snake_case(name: str) -> str:
    """Convert camelCase or PascalCase to snake_case for proto fields."""
    # Insert underscore before uppercase letters
    s1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s2 = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s1)
    return s2.lower().replace("-", "_")


def _infer_proto_type(val: Any) -> Tuple[str, str]:
    """Infer a proto3 field type from a Python value.

    Returns (proto_type, comment) tuple.
    """
    # Annotated value: extract @value
    if isinstance(val, dict):
        if "@value" in val:
            return _infer_proto_type(val["@value"])
        if "@container" in val and val.get("@container") == "@vector":
            dims = val.get("@dimensions", "?")
            return f"repeated double", f"@vector ({dims}-dim)"
        # Generic nested object
        return "bytes", "nested object (define sub-message)"

    if isinstance(val, bool):
        return "bool", type(val).__name__
    if isinstance(val, int):
        return "int64", type(val).__name__
    if isinstance(val, float):
        return "double", type(val).__name__
    if isinstance(val, str):
        return "string", type(val).__name__

    if isinstance(val, list):
        if not val:
            return "repeated string", "empty array (type unknown)"
        first = val[0]
        inner, _ = _infer_proto_type(first)
        return f"repeated {inner}", f"array of {type(first).__name__}"

    return "bytes", f"unknown type: {type(val).__name__}"
