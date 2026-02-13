"""
Data Subject Rights & Compliance Operations for JSON-LD

Phase 2: Graph-level operations implementing GDPR Articles 15-20.
These functions transform graphs, produce reports, and create audit trails.

Complements the annotation-level fields in data_protection.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from datetime import datetime, timezone

from jsonld_ex.data_protection import (
    _parse_iso,
    is_personal_data,
)


# ── Data Structures ───────────────────────────────────────────────

@dataclass
class ErasurePlan:
    """Result of request_erasure(): identifies what will be erased."""
    data_subject: str
    affected_node_ids: list[str] = field(default_factory=list)
    affected_property_count: int = 0


@dataclass
class ErasureAudit:
    """Result of execute_erasure(): records what was erased."""
    data_subject: str
    erased_node_ids: list[str] = field(default_factory=list)
    erased_property_count: int = 0


@dataclass
class RestrictionResult:
    """Result of request_restriction()."""
    data_subject: str
    restricted_property_count: int = 0


@dataclass
class PortableExport:
    """Result of export_portable(): data subject's data in portable form."""
    data_subject: str
    format: str
    records: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AccessReport:
    """Result of right_of_access_report(): structured GDPR Art. 15 report."""
    data_subject: str
    records: list[dict[str, Any]] = field(default_factory=list)
    total_property_count: int = 0
    categories: set[str] = field(default_factory=set)
    controllers: set[str] = field(default_factory=set)
    legal_bases: set[str] = field(default_factory=set)
    jurisdictions: set[str] = field(default_factory=set)


@dataclass
class RetentionViolation:
    """A single retention deadline violation."""
    node_id: str
    property_name: str
    retention_until: str


@dataclass
class AuditEntry:
    """A single entry in the audit trail for a data subject."""
    node_id: str
    property_name: str
    data_subject: str
    value: Any = None
    personal_data_category: Optional[str] = None
    legal_basis: Optional[str] = None
    data_controller: Optional[str] = None
    erasure_requested: Optional[bool] = None
    restrict_processing: Optional[bool] = None


# ── Internal Helpers ──────────────────────────────────────────────

def _iter_subject_properties(
    graph: Sequence[dict[str, Any]],
    data_subject: str,
) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    """Yield (node, property_name, property_value) for all properties
    belonging to *data_subject*.

    Only considers property values that are dicts with ``@dataSubject``
    matching the given subject.
    """
    results: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for node in graph:
        for key, prop in node.items():
            if key.startswith("@"):
                continue
            values = prop if isinstance(prop, list) else [prop]
            for v in values:
                if (
                    isinstance(v, dict)
                    and v.get("@dataSubject") == data_subject
                ):
                    results.append((node, key, v))
    return results


# ── Operations ────────────────────────────────────────────────────

def request_erasure(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
    requested_at: Optional[str] = None,
) -> ErasurePlan:
    """Mark all properties belonging to a data subject for erasure (GDPR Art. 17).

    Mutates the graph in place by setting ``@erasureRequested=True`` on
    matching property values.

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject requesting erasure.
        requested_at: Optional ISO 8601 timestamp of the request.

    Returns:
        An :class:`ErasurePlan` summarising what was marked.
    """
    plan = ErasurePlan(data_subject=data_subject)
    seen_nodes: set[str] = set()

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        prop_val["@erasureRequested"] = True
        if requested_at is not None:
            prop_val["@erasureRequestedAt"] = requested_at
        plan.affected_property_count += 1
        node_id = node.get("@id", "")
        if node_id and node_id not in seen_nodes:
            seen_nodes.add(node_id)
            plan.affected_node_ids.append(node_id)

    return plan


def execute_erasure(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
    completed_at: Optional[str] = None,
) -> ErasureAudit:
    """Execute erasure on properties previously marked with ``@erasureRequested``.

    Sets ``@value`` to None and records ``@erasureCompletedAt``.
    Only affects properties where ``@erasureRequested is True`` AND
    ``@dataSubject`` matches.

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject.
        completed_at: Optional ISO 8601 timestamp of completion.

    Returns:
        An :class:`ErasureAudit` recording what was erased.
    """
    audit = ErasureAudit(data_subject=data_subject)
    seen_nodes: set[str] = set()

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        if prop_val.get("@erasureRequested") is not True:
            continue
        prop_val["@value"] = None
        if completed_at is not None:
            prop_val["@erasureCompletedAt"] = completed_at
        audit.erased_property_count += 1
        node_id = node.get("@id", "")
        if node_id and node_id not in seen_nodes:
            seen_nodes.add(node_id)
            audit.erased_node_ids.append(node_id)

    return audit


def request_restriction(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
    reason: str,
    processing_restrictions: Optional[list[str]] = None,
) -> RestrictionResult:
    """Mark all properties of a data subject for restricted processing (GDPR Art. 18).

    Mutates the graph in place.

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject.
        reason: Free-text reason for the restriction.
        processing_restrictions: Optional list of disallowed processing operations.

    Returns:
        A :class:`RestrictionResult` summarising what was restricted.
    """
    result = RestrictionResult(data_subject=data_subject)

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        prop_val["@restrictProcessing"] = True
        prop_val["@restrictionReason"] = reason
        if processing_restrictions is not None:
            prop_val["@processingRestrictions"] = processing_restrictions
        result.restricted_property_count += 1

    return result


def export_portable(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
    format: str = "json",
) -> PortableExport:
    """Export all personal data for a data subject in a portable format (GDPR Art. 20).

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject.
        format: Target format identifier (e.g. ``"json"``, ``"text/csv"``).

    Returns:
        A :class:`PortableExport` containing the extracted records.
    """
    export = PortableExport(data_subject=data_subject, format=format)
    # Group by node
    node_props: dict[str, dict[str, Any]] = {}

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        node_id = node.get("@id", "")
        if node_id not in node_props:
            node_props[node_id] = {
                "node_id": node_id,
                "type": node.get("@type"),
                "properties": {},
            }
        node_props[node_id]["properties"][prop_name] = prop_val.get("@value")

    export.records = list(node_props.values())
    return export


def rectify_data(
    node: dict[str, Any],
    *,
    new_value: Any,
    note: str,
    rectified_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create a rectified copy of an annotated node (GDPR Art. 16).

    Returns a **new dict** — does not mutate the original.

    Args:
        node: An annotated node dict (with ``@value``).
        new_value: The corrected value.
        note: Description of what was corrected.
        rectified_at: Optional ISO 8601 timestamp. If None, not set.

    Returns:
        A new dict with the corrected value and rectification metadata.
    """
    result = dict(node)
    result["@value"] = new_value
    result["@rectificationNote"] = note
    if rectified_at is not None:
        result["@rectifiedAt"] = rectified_at
    return result


def right_of_access_report(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
) -> AccessReport:
    """Generate a structured report of all data held about a subject (GDPR Art. 15).

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject.

    Returns:
        An :class:`AccessReport` summarising all data held.
    """
    report = AccessReport(data_subject=data_subject)
    node_data: dict[str, dict[str, Any]] = {}

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        node_id = node.get("@id", "")
        if node_id not in node_data:
            node_data[node_id] = {
                "node_id": node_id,
                "type": node.get("@type"),
                "properties": {},
            }
        node_data[node_id]["properties"][prop_name] = prop_val.get("@value")
        report.total_property_count += 1

        # Collect metadata for summary
        cat = prop_val.get("@personalDataCategory")
        if cat:
            report.categories.add(cat)
        ctrl = prop_val.get("@dataController")
        if ctrl:
            report.controllers.add(ctrl)
        basis = prop_val.get("@legalBasis")
        if basis:
            report.legal_bases.add(basis)
        jur = prop_val.get("@jurisdiction")
        if jur:
            report.jurisdictions.add(jur)

    report.records = list(node_data.values())
    return report


def validate_retention(
    graph: Sequence[dict[str, Any]],
    *,
    as_of: str,
) -> list[RetentionViolation]:
    """Find all properties whose retention deadline has passed.

    Args:
        graph: A list of JSON-LD node dicts.
        as_of: ISO 8601 datetime to check against.

    Returns:
        A list of :class:`RetentionViolation` for each expired property.
    """
    check_time = _parse_iso(as_of)
    violations: list[RetentionViolation] = []

    for node in graph:
        node_id = node.get("@id", "")
        for key, prop in node.items():
            if key.startswith("@"):
                continue
            values = prop if isinstance(prop, list) else [prop]
            for v in values:
                if not isinstance(v, dict):
                    continue
                retention = v.get("@retentionUntil")
                if retention is None:
                    continue
                if _parse_iso(retention) < check_time:
                    violations.append(RetentionViolation(
                        node_id=node_id,
                        property_name=key,
                        retention_until=retention,
                    ))

    return violations


def audit_trail(
    graph: Sequence[dict[str, Any]],
    *,
    data_subject: str,
) -> list[AuditEntry]:
    """Build a complete audit trail for a data subject.

    Returns one :class:`AuditEntry` per property value belonging to the subject,
    capturing the current state of all annotation fields.

    Args:
        graph: A list of JSON-LD node dicts.
        data_subject: IRI of the data subject.

    Returns:
        A list of :class:`AuditEntry` instances.
    """
    entries: list[AuditEntry] = []

    for node, prop_name, prop_val in _iter_subject_properties(graph, data_subject):
        entries.append(AuditEntry(
            node_id=node.get("@id", ""),
            property_name=prop_name,
            data_subject=data_subject,
            value=prop_val.get("@value"),
            personal_data_category=prop_val.get("@personalDataCategory"),
            legal_basis=prop_val.get("@legalBasis"),
            data_controller=prop_val.get("@dataController"),
            erasure_requested=prop_val.get("@erasureRequested"),
            restrict_processing=prop_val.get("@restrictProcessing"),
        ))

    return entries
