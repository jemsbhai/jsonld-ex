"""Shared internal helpers for transport modules.

Common utilities used by CoAP, MQTT, HTTP, and future transport
modules.  Extracts duplicated logic for metadata scanning, IRI
parsing, and temporal option derivation.

This is an internal module — not part of the public API.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Optional

from jsonld_ex.ai_ml import get_confidence


# ═══════════════════════════════════════════════════════════════════
# IRI / URN parsing
# ═══════════════════════════════════════════════════════════════════


def local_name(iri: str) -> str:
    """Extract the local/fragment part of an IRI or URN.

    Resolution order:
    1. Fragment: ``http://example.org/ns#Thing`` → ``Thing``
    2. Path: ``http://example.org/ns/Thing`` → ``Thing``
    3. URN: ``urn:sensor:imu-001`` → ``imu-001``
    4. Fallback: return the full string.
    """
    if "#" in iri:
        return iri.rsplit("#", 1)[-1]
    if "/" in iri:
        return iri.rsplit("/", 1)[-1]
    if ":" in iri:
        return iri.rsplit(":", 1)[-1]
    return iri


def sanitise_segment(segment: str) -> str:
    """Clean a topic/path segment for transport protocols.

    Removes characters problematic for MQTT topics and CoAP URIs:
    ``#``, ``+``, null bytes.  Strips leading ``$`` (reserved for
    broker system topics in MQTT).

    Returns ``"unknown"`` if the result would be empty.
    """
    sanitised = re.sub(r"[#+\x00]", "_", segment)
    sanitised = sanitised.lstrip("$")
    return sanitised or "unknown"


# ═══════════════════════════════════════════════════════════════════
# @type / @id extraction
# ═══════════════════════════════════════════════════════════════════


def extract_type_local(doc: dict[str, Any]) -> str:
    """Extract a sanitised local name from ``@type``."""
    type_val = doc.get("@type", "unknown")
    if isinstance(type_val, list):
        type_val = type_val[0] if type_val else "unknown"
    return sanitise_segment(local_name(str(type_val)))


def extract_id_fragment(doc: dict[str, Any]) -> str:
    """Extract a sanitised local name from ``@id``."""
    id_val = doc.get("@id", "unknown")
    return sanitise_segment(local_name(str(id_val)))


# ═══════════════════════════════════════════════════════════════════
# Temporal metadata
# ═══════════════════════════════════════════════════════════════════


def find_valid_until(doc: dict[str, Any]) -> Optional[str]:
    """Find the first ``@validUntil`` in document or property values.

    Scans document-level first, then non-``@``-prefixed property
    values (dicts only).
    """
    valid_until = doc.get("@validUntil")
    if valid_until is not None:
        return valid_until

    for key, val in doc.items():
        if key.startswith("@"):
            continue
        if isinstance(val, dict) and "@validUntil" in val:
            return val["@validUntil"]

    return None


def seconds_remaining(valid_until_str: Any) -> Optional[float]:
    """Parse ISO 8601 datetime and return seconds from now.

    Returns None if *valid_until_str* is not a string or if parsing
    fails.  Returns negative values for past datetimes (caller
    decides how to handle).
    """
    if not isinstance(valid_until_str, str):
        return None
    try:
        dt_str = valid_until_str.replace("Z", "+00:00")
        expiry_dt = datetime.fromisoformat(dt_str)
        if expiry_dt.tzinfo is None:
            expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        return (expiry_dt - now).total_seconds()

    except (ValueError, TypeError, OverflowError):
        return None


def derive_expiry_seconds(doc: dict[str, Any]) -> Optional[int]:
    """Compute expiry interval (seconds remaining) from ``@validUntil``.

    Returns None if no ``@validUntil`` is found or if it has already
    passed.  Result is clamped to uint32 max (4,294,967,295).
    """
    valid_until = find_valid_until(doc)
    if valid_until is None:
        return None

    remaining = seconds_remaining(valid_until)
    if remaining is None or remaining <= 0:
        return None

    return min(int(math.ceil(remaining)), 0xFFFFFFFF)


# ═══════════════════════════════════════════════════════════════════
# Confidence scanning
# ═══════════════════════════════════════════════════════════════════


def scan_confidence(doc: dict[str, Any]) -> tuple[Optional[float], bool, str]:
    """Scan document for confidence and humanVerified metadata.

    Checks document-level first, then scans property values.

    Returns:
        ``(confidence, human_verified, source_description)`` where
        *source_description* is e.g. ``"document-level"`` or
        ``"property 'temperature'"``.
    """
    conf = get_confidence(doc)
    human_verified = doc.get("@humanVerified", False) is True
    source = "document-level"

    if human_verified:
        return (conf, True, source)

    if conf is not None:
        return (conf, False, source)

    # Scan properties
    for key, val in doc.items():
        if key.startswith("@"):
            continue
        if isinstance(val, dict):
            prop_hv = val.get("@humanVerified", False) is True
            if prop_hv:
                return (get_confidence(val), True, f"property '{key}'")
            prop_conf = get_confidence(val)
            if prop_conf is not None:
                return (prop_conf, False, f"property '{key}'")

    return (None, False, "none")
