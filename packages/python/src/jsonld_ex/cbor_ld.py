"""
CBOR-LD Serialization for JSON-LD-Ex.

Binary-efficient serialization of JSON-LD documents using CBOR
(RFC 8949).  Provides significant payload reduction over JSON,
especially for IoT and bandwidth-constrained environments.

Supports context compression via a registry that maps full context
URLs to short integer IDs, mirroring the CBOR-LD specification's
approach to reducing repetitive context references.

Requires the ``cbor2`` package::

    pip install jsonld-ex[iot]
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from typing import Any, Optional

try:
    import cbor2

    _HAS_CBOR2 = True
except ImportError:
    _HAS_CBOR2 = False


def _require_cbor2() -> None:
    if not _HAS_CBOR2:
        raise ImportError(
            "cbor2 is required for CBOR-LD serialization. "
            "Install it with: pip install jsonld-ex[iot]"
        )


# ── Default context registry ──────────────────────────────────────

# Maps well-known context URLs to compact integer IDs.
# Users can extend this with custom registries.
DEFAULT_CONTEXT_REGISTRY: dict[str, int] = {
    "http://schema.org/": 1,
    "https://schema.org/": 1,
    "https://www.w3.org/ns/activitystreams": 2,
    "https://w3id.org/security/v2": 3,
    "https://www.w3.org/2018/credentials/v1": 4,
    "http://www.w3.org/ns/prov#": 5,
}

_REVERSE_DEFAULT: dict[int, str] = {v: k for k, v in DEFAULT_CONTEXT_REGISTRY.items()}


# ── Data Structures ────────────────────────────────────────────────


@dataclass
class PayloadStats:
    """Comparison of serialization sizes for a document."""

    json_bytes: int
    cbor_bytes: int
    gzip_json_bytes: int
    gzip_cbor_bytes: int

    @property
    def cbor_ratio(self) -> float:
        """CBOR size as a fraction of JSON size (lower = better)."""
        if self.json_bytes == 0:
            return 0.0
        return self.cbor_bytes / self.json_bytes

    @property
    def gzip_cbor_ratio(self) -> float:
        """Gzipped CBOR as a fraction of JSON (the headline number)."""
        if self.json_bytes == 0:
            return 0.0
        return self.gzip_cbor_bytes / self.json_bytes


# ═══════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════


def to_cbor(
    doc: dict[str, Any],
    context_registry: Optional[dict[str, int]] = None,
) -> bytes:
    """Serialize a JSON-LD document to CBOR with context compression.

    Context URLs found in the registry are replaced with their compact
    integer IDs.  All jsonld-ex extension keywords are preserved.

    Args:
        doc: JSON-LD document (Python dict).
        context_registry: Mapping of context URL → integer ID.
            Defaults to :data:`DEFAULT_CONTEXT_REGISTRY`.

    Returns:
        CBOR-encoded bytes.

    Raises:
        ImportError: If ``cbor2`` is not installed.
    """
    _require_cbor2()
    registry = context_registry or DEFAULT_CONTEXT_REGISTRY
    compressed = _compress_contexts(doc, registry)
    return cbor2.dumps(compressed)


def from_cbor(
    data: bytes,
    context_registry: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    """Deserialize CBOR bytes back to a JSON-LD document.

    Restores compressed context integer IDs to their full URLs using
    the provided (or default) registry.

    Args:
        data: CBOR-encoded bytes.
        context_registry: Same registry used during serialization.

    Returns:
        Restored JSON-LD document.

    Raises:
        ImportError: If ``cbor2`` is not installed.
    """
    _require_cbor2()
    registry = context_registry or DEFAULT_CONTEXT_REGISTRY
    reverse = {v: k for k, v in registry.items()}
    decoded = cbor2.loads(data)
    return _decompress_contexts(decoded, reverse)


# ═══════════════════════════════════════════════════════════════════
# PAYLOAD STATISTICS
# ═══════════════════════════════════════════════════════════════════


def payload_stats(
    doc: dict[str, Any],
    context_registry: Optional[dict[str, int]] = None,
) -> PayloadStats:
    """Compare serialization sizes for a document.

    Computes JSON, CBOR, gzipped JSON, and gzipped CBOR sizes.
    Useful for benchmarking payload reduction.

    Args:
        doc: JSON-LD document.
        context_registry: Optional context registry for CBOR compression.

    Returns:
        PayloadStats with all four sizes and derived ratios.
    """
    _require_cbor2()
    json_bytes = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    cbor_bytes = to_cbor(doc, context_registry)
    gzip_json = gzip.compress(json_bytes)
    gzip_cbor = gzip.compress(cbor_bytes)

    return PayloadStats(
        json_bytes=len(json_bytes),
        cbor_bytes=len(cbor_bytes),
        gzip_json_bytes=len(gzip_json),
        gzip_cbor_bytes=len(gzip_cbor),
    )


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _compress_contexts(obj: Any, registry: dict[str, int]) -> Any:
    """Recursively replace context URLs with registry IDs."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "@context":
                result[k] = _compress_context_value(v, registry)
            else:
                result[k] = _compress_contexts(v, registry)
        return result
    if isinstance(obj, list):
        return [_compress_contexts(item, registry) for item in obj]
    return obj


def _compress_context_value(ctx: Any, registry: dict[str, int]) -> Any:
    """Compress a single @context value."""
    if isinstance(ctx, str):
        return registry.get(ctx, ctx)
    if isinstance(ctx, list):
        return [_compress_context_value(item, registry) for item in ctx]
    if isinstance(ctx, dict):
        # Inline context definition — don't compress, but recurse
        return {k: _compress_context_value(v, registry) if k == "@import" else v
                for k, v in ctx.items()}
    return ctx


def _decompress_contexts(obj: Any, reverse: dict[int, str]) -> Any:
    """Recursively restore context URLs from registry IDs."""
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if k == "@context":
                result[k] = _decompress_context_value(v, reverse)
            else:
                result[k] = _decompress_contexts(v, reverse)
        return result
    if isinstance(obj, list):
        return [_decompress_contexts(item, reverse) for item in obj]
    return obj


def _decompress_context_value(ctx: Any, reverse: dict[int, str]) -> Any:
    """Restore a single @context value."""
    if isinstance(ctx, int):
        return reverse.get(ctx, ctx)
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, list):
        return [_decompress_context_value(item, reverse) for item in ctx]
    if isinstance(ctx, dict):
        return ctx
    return ctx
