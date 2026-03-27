"""
HTTP Header Derivation for JSON-LD-Ex.

Derives HTTP response and request headers from JSON-LD metadata,
bridging semantic annotations and standard HTTP caching, content
negotiation, and conditional request mechanisms.

Header mappings:

**Response headers (derived from document metadata):**

- **Content-Type**: ``application/ld+json`` for JSON-LD,
  ``application/cbor`` for CBOR-LD.
- **ETag**: Deterministic quoted string derived from ``@integrity``
  (RFC 9110 §8.8.3).  Enables conditional requests (``304 Not
  Modified``) for documents with integrity hashes.
- **Cache-Control**: ``max-age=N`` derived from ``@validUntil``
  (seconds remaining).  ``no-cache`` if already expired.
- **Link**: Context discovery header per W3C JSON-LD 1.1 §4.1 —
  ``<url>; rel="http://www.w3.org/ns/json-ld#context"``.
- **X-JsonLD-Confidence**: Custom header exposing ``@confidence``
  for proxy/CDN routing decisions.
- **X-JsonLD-Source**: Custom header exposing ``@source``.
- **X-JsonLD-Type**: Custom header exposing ``@type``.

**Request headers (for content negotiation and conditional requests):**

- **Accept**: ``application/ld+json`` or ``application/cbor``.
- **If-None-Match**: Conditional GET using a previously received ETag.

No HTTP library is required — this module returns plain dicts
suitable for any HTTP framework (Flask, FastAPI, aiohttp, requests).

References:
    RFC 9110: HTTP Semantics (Content-Type, ETag, Cache-Control).
    RFC 8288: Web Linking (Link header).
    W3C JSON-LD 1.1 §4.1: Interpreting JSON as JSON-LD.
"""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Any, Optional

from jsonld_ex._transport_common import (
    find_valid_until,
    seconds_remaining,
)


# ── Media types ────────────────────────────────────────────────────

MEDIA_TYPE_JSONLD: str = "application/ld+json"
"""IANA media type for JSON-LD (W3C JSON-LD 1.1 §8)."""

MEDIA_TYPE_CBOR: str = "application/cbor"
"""IANA media type for CBOR (RFC 8949)."""

MEDIA_TYPE_JSON: str = "application/json"
"""IANA media type for plain JSON (RFC 8259)."""

# JSON-LD context discovery link relation
_JSONLD_CONTEXT_REL = "http://www.w3.org/ns/json-ld#context"


# ═══════════════════════════════════════════════════════════════════
# CONTENT-TYPE
# ═══════════════════════════════════════════════════════════════════


def derive_content_type(compress: bool = False) -> str:
    """Derive the Content-Type header value.

    Args:
        compress: True for CBOR, False for JSON-LD.

    Returns:
        Media type string.
    """
    if compress:
        return MEDIA_TYPE_CBOR
    return MEDIA_TYPE_JSONLD


# ═══════════════════════════════════════════════════════════════════
# ETAG
# ═══════════════════════════════════════════════════════════════════


def derive_etag(doc: dict[str, Any]) -> Optional[str]:
    """Derive an ETag header from ``@integrity`` metadata.

    The ``@integrity`` value is hashed (SHA-256) and hex-encoded to
    produce a deterministic, opaque entity tag.  The result is
    returned as a quoted string per RFC 9110 §8.8.3.

    Args:
        doc: JSON-LD document.

    Returns:
        Quoted ETag string (e.g. ``'"a1b2c3..."'``), or None if
        no ``@integrity`` is present.
    """
    integrity = doc.get("@integrity")
    if integrity is None:
        return None

    # Hash to produce a fixed-length, deterministic tag
    h = hashlib.sha256(str(integrity).encode("utf-8")).hexdigest()
    # Use first 32 hex chars (128 bits) — strong enough, compact
    return f'"{h[:32]}"'


# ═══════════════════════════════════════════════════════════════════
# CACHE-CONTROL
# ═══════════════════════════════════════════════════════════════════


def derive_cache_control(doc: dict[str, Any]) -> Optional[str]:
    """Derive Cache-Control header from ``@validUntil``.

    Scans document-level and property-level ``@validUntil``:

    - Future ``@validUntil`` → ``max-age=N`` (seconds remaining).
    - Past ``@validUntil`` → ``no-cache`` (assertion expired).
    - No ``@validUntil`` → None (no caching directive).

    Args:
        doc: JSON-LD document.

    Returns:
        Cache-Control header value string, or None.
    """
    valid_until_str = find_valid_until(doc)
    if valid_until_str is None:
        return None

    remaining = seconds_remaining(valid_until_str)
    if remaining is None:
        # Parse failed — no directive
        return None

    if remaining <= 0:
        return "no-cache"

    max_age = min(int(math.ceil(remaining)), 0xFFFFFFFF)
    return f"max-age={max_age}"


# ═══════════════════════════════════════════════════════════════════
# LINK HEADER
# ═══════════════════════════════════════════════════════════════════


def derive_link_header(doc: dict[str, Any]) -> Optional[str]:
    """Derive Link header for JSON-LD context discovery.

    Per W3C JSON-LD 1.1 §4.1, a JSON document can be interpreted as
    JSON-LD if a Link header provides the context URL with relation
    type ``http://www.w3.org/ns/json-ld#context``.

    Only string context URLs are included — inline context objects
    (dicts) are excluded since they cannot be referenced by URL.

    Args:
        doc: JSON-LD document.

    Returns:
        Link header value string with one or more entries, or None
        if no URL-based ``@context`` is present.
    """
    context = doc.get("@context")
    if context is None:
        return None

    urls: list[str] = []

    if isinstance(context, str):
        urls.append(context)
    elif isinstance(context, list):
        for item in context:
            if isinstance(item, str):
                urls.append(item)
            # Inline context objects (dicts) are excluded
    # Single inline context (dict) → no Link header
    elif isinstance(context, dict):
        return None

    if not urls:
        return None

    # Build Link header entries
    entries = [
        f'<{url}>; rel="{_JSONLD_CONTEXT_REL}"'
        for url in urls
    ]
    return ", ".join(entries)


# ═══════════════════════════════════════════════════════════════════
# RESPONSE HEADERS
# ═══════════════════════════════════════════════════════════════════


def derive_response_headers(
    doc: dict[str, Any],
    compress: bool = False,
) -> dict[str, str]:
    """Derive HTTP response headers from JSON-LD metadata.

    Combines all derivation functions into a single headers dict
    ready for any HTTP framework.

    Args:
        doc: JSON-LD document.
        compress: CBOR (True) or JSON-LD (False).

    Returns:
        Dict of HTTP header name → value (all strings).
    """
    headers: dict[str, str] = {}

    # -- Content-Type (always present) --
    headers["Content-Type"] = derive_content_type(compress)

    # -- ETag --
    etag = derive_etag(doc)
    if etag is not None:
        headers["ETag"] = etag

    # -- Cache-Control --
    cc = derive_cache_control(doc)
    if cc is not None:
        headers["Cache-Control"] = cc

    # -- Link --
    link = derive_link_header(doc)
    if link is not None:
        headers["Link"] = link

    # -- Custom metadata headers --
    # These use X- prefix per convention for application-specific headers.
    # While RFC 6648 deprecated X- prefix creation, these are explicitly
    # jsonld-ex-specific and unlikely to become standard HTTP headers.

    conf = doc.get("@confidence")
    if conf is not None:
        headers["X-JsonLD-Confidence"] = str(conf)

    source = doc.get("@source")
    if source is not None:
        headers["X-JsonLD-Source"] = str(source)

    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else None
        if type_val is not None:
            headers["X-JsonLD-Type"] = str(type_val)

    return headers


# ═══════════════════════════════════════════════════════════════════
# REQUEST HEADERS
# ═══════════════════════════════════════════════════════════════════


def derive_request_headers(
    compress: bool = False,
    etag: Optional[str] = None,
) -> dict[str, str]:
    """Derive HTTP request headers for content negotiation.

    Args:
        compress: Prefer CBOR (True) or JSON-LD (False).
        etag: Previously received ETag for conditional GET
            (``If-None-Match``).

    Returns:
        Dict of HTTP header name → value.
    """
    headers: dict[str, str] = {}

    # -- Accept --
    if compress:
        headers["Accept"] = f"{MEDIA_TYPE_CBOR}, {MEDIA_TYPE_JSONLD};q=0.9"
    else:
        headers["Accept"] = f"{MEDIA_TYPE_JSONLD}, {MEDIA_TYPE_JSON};q=0.8"

    # -- If-None-Match (conditional GET) --
    if etag is not None:
        headers["If-None-Match"] = etag

    return headers


# ═══════════════════════════════════════════════════════════════════

# Internal helpers (_find_valid_until, _seconds_remaining) replaced
# by shared versions in jsonld_ex._transport_common
