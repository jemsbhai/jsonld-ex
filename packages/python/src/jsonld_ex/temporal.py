"""
Temporal Extensions for JSON-LD-Ex.

Adds time-aware assertions to JSON-LD values, enabling knowledge graph
versioning and point-in-time queries.

Extension keywords:
    ``@validFrom``  — ISO 8601 timestamp when an assertion becomes true.
    ``@validUntil`` — ISO 8601 timestamp when an assertion ceases to be true.
    ``@asOf``       — Point in time the assertion was known/observed to hold.

These compose naturally with existing jsonld-ex annotations
(``@confidence``, ``@source``, etc.) on the same value object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Sequence

from jsonld_ex.ai_ml import get_confidence

# ── ISO 8601 Parsing ───────────────────────────────────────────────

# Supported formats (subset of ISO 8601 that covers common usage):
#   2025-01-15
#   2025-01-15T10:30:00Z
#   2025-01-15T10:30:00+00:00
#   2025-01-15T10:30:00.123Z

_ISO_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
]


def _parse_timestamp(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string into a datetime."""
    if not isinstance(ts, str):
        raise TypeError(f"Timestamp must be a string, got: {type(ts).__name__}")
    # Normalise trailing Z to +00:00 for consistent parsing
    normalised = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    for fmt in _ISO_FORMATS:
        try:
            return datetime.strptime(normalised, fmt)
        except ValueError:
            continue
    # Fallback: try fromisoformat (Python 3.11+)
    try:
        return datetime.fromisoformat(normalised)
    except (ValueError, AttributeError):
        pass
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


# ── Annotation helpers ─────────────────────────────────────────────


def add_temporal(
    value: Any,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    as_of: Optional[str] = None,
) -> dict[str, Any]:
    """Add temporal qualifiers to a value.

    If *value* is already an annotated dict (has ``@value``), the
    temporal keys are added in place (on a copy).  Otherwise a new
    annotated-value wrapper is created.

    Args:
        value: The value to annotate (plain or already annotated).
        valid_from:  ISO 8601 string — when the assertion becomes true.
        valid_until: ISO 8601 string — when the assertion expires.
        as_of:       ISO 8601 string — observation timestamp.

    Returns:
        An annotated-value dict with temporal qualifiers.

    Raises:
        TypeError:  If timestamps are not strings.
        ValueError: If timestamps cannot be parsed, or if
            ``valid_from`` > ``valid_until``.
    """
    if valid_from is None and valid_until is None and as_of is None:
        raise ValueError("At least one temporal qualifier must be provided")

    # Validate timestamps parse correctly
    parsed_from = _parse_timestamp(valid_from) if valid_from else None
    parsed_until = _parse_timestamp(valid_until) if valid_until else None
    if as_of is not None:
        _parse_timestamp(as_of)

    if parsed_from and parsed_until and parsed_from > parsed_until:
        raise ValueError(
            f"@validFrom ({valid_from}) must not be after @validUntil ({valid_until})"
        )

    # Build result
    if isinstance(value, dict) and "@value" in value:
        result = dict(value)  # shallow copy
    else:
        result = {"@value": value}

    if valid_from is not None:
        result["@validFrom"] = valid_from
    if valid_until is not None:
        result["@validUntil"] = valid_until
    if as_of is not None:
        result["@asOf"] = as_of

    return result


# ── Time-slice query ───────────────────────────────────────────────


def query_at_time(
    graph: Sequence[dict[str, Any]],
    timestamp: str,
    property_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Return the graph state as of a given timestamp.

    For each node, retains only the properties whose temporal bounds
    include *timestamp* (or that have no temporal bounds — treated as
    always-valid).

    Args:
        graph: List of JSON-LD nodes (typically from ``doc["@graph"]``).
        timestamp: ISO 8601 timestamp to query at.
        property_name: If given, only filter this property; pass through
            all others unchanged.

    Returns:
        Filtered list of nodes.  Nodes with no matching properties
        are omitted entirely.
    """
    ts = _parse_timestamp(timestamp)
    result: list[dict[str, Any]] = []

    for node in graph:
        filtered = _filter_node_at_time(node, ts, property_name)
        if filtered is not None:
            result.append(filtered)

    return result


def _filter_node_at_time(
    node: dict[str, Any],
    ts: datetime,
    property_name: Optional[str],
) -> Optional[dict[str, Any]]:
    """Filter a single node's properties by temporal validity."""
    out: dict[str, Any] = {}
    has_any_data = False

    for key, value in node.items():
        # Identity keys always pass through
        if key in ("@id", "@type", "@context"):
            out[key] = value
            continue

        # If filtering by a specific property, pass others unchanged
        if property_name is not None and key != property_name:
            out[key] = value
            has_any_data = True
            continue

        # Check temporal validity
        if isinstance(value, list):
            kept = [v for v in value if _is_valid_at(v, ts)]
            if kept:
                out[key] = kept if len(kept) > 1 else kept[0]
                has_any_data = True
        elif _is_valid_at(value, ts):
            out[key] = value
            has_any_data = True

    if not has_any_data:
        return None
    return out


def _is_valid_at(value: Any, ts: datetime) -> bool:
    """Check if a value is temporally valid at the given timestamp."""
    if not isinstance(value, dict):
        return True  # no temporal metadata → always valid

    vf = value.get("@validFrom")
    vu = value.get("@validUntil")

    # No temporal bounds → always valid
    if vf is None and vu is None:
        return True

    if vf is not None:
        from_dt = _parse_timestamp(vf)
        if ts < from_dt:
            return False
    if vu is not None:
        until_dt = _parse_timestamp(vu)
        if ts > until_dt:
            return False

    return True


# ── Temporal diff ──────────────────────────────────────────────────


@dataclass
class TemporalDiffResult:
    """Result of comparing a graph at two points in time."""

    added: list[dict[str, Any]] = field(default_factory=list)
    removed: list[dict[str, Any]] = field(default_factory=list)
    modified: list[dict[str, Any]] = field(default_factory=list)
    unchanged: list[dict[str, Any]] = field(default_factory=list)


def temporal_diff(
    graph: Sequence[dict[str, Any]],
    t1: str,
    t2: str,
) -> TemporalDiffResult:
    """Compute what changed between two points in time.

    Snapshots the graph at *t1* and *t2*, then compares property
    values (stripped of annotations) for each ``@id``.

    Args:
        graph: List of JSON-LD nodes with temporal annotations.
        t1: Earlier ISO 8601 timestamp.
        t2: Later ISO 8601 timestamp.

    Returns:
        TemporalDiffResult with added, removed, modified, unchanged.
    """
    snap1 = {n["@id"]: n for n in query_at_time(graph, t1) if "@id" in n}
    snap2 = {n["@id"]: n for n in query_at_time(graph, t2) if "@id" in n}

    result = TemporalDiffResult()
    all_ids = set(snap1.keys()) | set(snap2.keys())

    for nid in sorted(all_ids):
        n1 = snap1.get(nid)
        n2 = snap2.get(nid)

        if n1 is None and n2 is not None:
            result.added.append({"@id": nid, "state": n2})
            continue
        if n1 is not None and n2 is None:
            result.removed.append({"@id": nid, "state": n1})
            continue

        # Both present — compare properties
        props1 = _data_props(n1)  # type: ignore[arg-type]
        props2 = _data_props(n2)  # type: ignore[arg-type]

        for prop in set(props1.keys()) | set(props2.keys()):
            v1 = props1.get(prop)
            v2 = props2.get(prop)
            bare1 = _bare(v1)
            bare2 = _bare(v2)

            if v1 is None and v2 is not None:
                result.added.append({"@id": nid, "property": prop, "value": v2})
            elif v1 is not None and v2 is None:
                result.removed.append({"@id": nid, "property": prop, "value": v1})
            elif bare1 != bare2:
                result.modified.append({
                    "@id": nid, "property": prop,
                    "value_at_t1": v1, "value_at_t2": v2,
                })
            else:
                result.unchanged.append({"@id": nid, "property": prop, "value": bare1})

    return result


def _data_props(node: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in node.items() if k not in ("@id", "@type", "@context")}


def _bare(val: Any) -> Any:
    if isinstance(val, dict) and "@value" in val:
        return val["@value"]
    return val
