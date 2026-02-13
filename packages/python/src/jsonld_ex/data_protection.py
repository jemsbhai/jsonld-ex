"""
Data Protection Extensions for JSON-LD

Phase 1: Core annotations, data classification, consent lifecycle,
and graph filtering for GDPR/privacy compliance metadata.

Designed to compose with ai_ml.annotate() — both produce compatible
@value dicts that can be merged. This module is fully additive and
does not modify any existing modules.

Maps to W3C Data Privacy Vocabulary (DPV) v2.2 concepts.
DPV interop (to_dpv/from_dpv) is Phase 3.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Sequence
from datetime import datetime, timezone


# ── Constants ──────────────────────────────────────────────────────

LEGAL_BASES: tuple[str, ...] = (
    "consent",
    "contract",
    "legal_obligation",
    "vital_interest",
    "public_task",
    "legitimate_interest",
)

PERSONAL_DATA_CATEGORIES: tuple[str, ...] = (
    "regular",
    "sensitive",
    "special_category",
    "anonymized",
    "pseudonymized",
    "synthetic",
    "non_personal",
)

# Categories that count as personal data under GDPR.
# Anonymized, synthetic, and non_personal are excluded.
_PERSONAL_CATEGORIES: frozenset[str] = frozenset({
    "regular",
    "sensitive",
    "special_category",
    "pseudonymized",
})

_SENSITIVE_CATEGORIES: frozenset[str] = frozenset({
    "sensitive",
    "special_category",
})

CONSENT_GRANULARITIES: tuple[str, ...] = (
    "broad",
    "specific",
    "granular",
)

ACCESS_LEVELS: tuple[str, ...] = (
    "public",
    "internal",
    "restricted",
    "confidential",
    "secret",
)


# ── Data Structures ───────────────────────────────────────────────

@dataclass
class DataProtectionMetadata:
    """Data protection metadata extracted from a JSON-LD node."""
    personal_data_category: Optional[str] = None
    legal_basis: Optional[str] = None
    processing_purpose: Optional[str | list[str]] = None
    data_controller: Optional[str] = None
    data_processor: Optional[str] = None
    data_subject: Optional[str] = None
    retention_until: Optional[str] = None
    jurisdiction: Optional[str] = None
    access_level: Optional[str] = None
    consent: Optional[dict[str, Any]] = None


@dataclass
class ConsentRecord:
    """Structured consent record."""
    given_at: str
    scope: list[str]
    granularity: Optional[str] = None
    withdrawn_at: Optional[str] = None


# ── Core Annotation Function ─────────────────────────────────────

def annotate_protection(
    value: Any,
    *,
    personal_data_category: Optional[str] = None,
    legal_basis: Optional[str] = None,
    processing_purpose: Optional[str | list[str]] = None,
    data_controller: Optional[str] = None,
    data_processor: Optional[str] = None,
    data_subject: Optional[str] = None,
    retention_until: Optional[str] = None,
    jurisdiction: Optional[str] = None,
    access_level: Optional[str] = None,
    consent: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a JSON-LD value annotated with data protection metadata.

    Produces a dict with ``@value`` plus any specified protection fields.
    The output is compatible with ``ai_ml.annotate()`` — both can be
    merged with ``{**provenance, **protection}`` to combine provenance
    and data protection metadata on a single node.

    Args:
        value: The data value to annotate.
        personal_data_category: One of :data:`PERSONAL_DATA_CATEGORIES`.
        legal_basis: One of :data:`LEGAL_BASES` (maps to GDPR Art. 6).
        processing_purpose: Free-text, IRI, or list describing why data is processed.
        data_controller: IRI of the entity determining purposes and means.
        data_processor: IRI of the entity processing data on behalf of controller.
        data_subject: IRI or category of the individual the data relates to.
        retention_until: ISO 8601 datetime for mandatory deletion deadline.
        jurisdiction: ISO 3166 code or IRI for applicable legal jurisdiction.
        access_level: One of :data:`ACCESS_LEVELS`.
        consent: A consent record dict (from :func:`create_consent_record`).

    Returns:
        A dict with ``@value`` and any specified ``@``-prefixed protection fields.

    Raises:
        ValueError: If an enum field receives an invalid value.
    """
    # Validate enum fields
    if personal_data_category is not None and personal_data_category not in PERSONAL_DATA_CATEGORIES:
        raise ValueError(
            f"Invalid personal data category: {personal_data_category!r}. "
            f"Must be one of: {', '.join(PERSONAL_DATA_CATEGORIES)}"
        )
    if legal_basis is not None and legal_basis not in LEGAL_BASES:
        raise ValueError(
            f"Invalid legal basis: {legal_basis!r}. "
            f"Must be one of: {', '.join(LEGAL_BASES)}"
        )
    if access_level is not None and access_level not in ACCESS_LEVELS:
        raise ValueError(
            f"Invalid access level: {access_level!r}. "
            f"Must be one of: {', '.join(ACCESS_LEVELS)}"
        )

    result: dict[str, Any] = {"@value": value}

    if personal_data_category is not None:
        result["@personalDataCategory"] = personal_data_category
    if legal_basis is not None:
        result["@legalBasis"] = legal_basis
    if processing_purpose is not None:
        result["@processingPurpose"] = processing_purpose
    if data_controller is not None:
        result["@dataController"] = data_controller
    if data_processor is not None:
        result["@dataProcessor"] = data_processor
    if data_subject is not None:
        result["@dataSubject"] = data_subject
    if retention_until is not None:
        result["@retentionUntil"] = retention_until
    if jurisdiction is not None:
        result["@jurisdiction"] = jurisdiction
    if access_level is not None:
        result["@accessLevel"] = access_level
    if consent is not None:
        result["@consent"] = consent

    return result


# ── Consent Lifecycle ─────────────────────────────────────────────

def create_consent_record(
    *,
    given_at: str,
    scope: str | list[str],
    granularity: Optional[str] = None,
    withdrawn_at: Optional[str] = None,
) -> dict[str, Any]:
    """Create a structured consent record.

    Args:
        given_at: ISO 8601 datetime when consent was given.
        scope: Purpose(s) for which consent was given. A single string
            is automatically wrapped in a list.
        granularity: One of :data:`CONSENT_GRANULARITIES`.
        withdrawn_at: ISO 8601 datetime when consent was withdrawn, or
            None if consent is still active.

    Returns:
        A dict with ``@consent*`` fields suitable for embedding in an
        annotated node via :func:`annotate_protection`.

    Raises:
        ValueError: If granularity is invalid or scope is empty.
    """
    if granularity is not None and granularity not in CONSENT_GRANULARITIES:
        raise ValueError(
            f"Invalid consent granularity: {granularity!r}. "
            f"Must be one of: {', '.join(CONSENT_GRANULARITIES)}"
        )

    # Normalize scope to list
    if isinstance(scope, str):
        scope = [scope]

    if len(scope) == 0:
        raise ValueError("Consent scope must not be empty")

    record: dict[str, Any] = {
        "@consentGivenAt": given_at,
        "@consentScope": scope,
    }

    if granularity is not None:
        record["@consentGranularity"] = granularity
    if withdrawn_at is not None:
        record["@consentWithdrawnAt"] = withdrawn_at

    return record


def is_consent_active(
    record: Optional[dict[str, Any]],
    at_time: Optional[str] = None,
) -> bool:
    """Check whether a consent record is active.

    Args:
        record: A consent record dict (from :func:`create_consent_record`).
        at_time: Optional ISO 8601 datetime to check consent status at a
            specific point in time. If None, checks current status
            (active = not withdrawn).

    Returns:
        True if consent is active, False otherwise.
    """
    if record is None or not isinstance(record, dict):
        return False

    given_at = record.get("@consentGivenAt")
    if given_at is None:
        return False

    withdrawn_at = record.get("@consentWithdrawnAt")

    if at_time is not None:
        # Parse timestamps for comparison
        check = _parse_iso(at_time)
        given = _parse_iso(given_at)

        # Not yet given at the check time
        if check < given:
            return False

        # Withdrawn before the check time
        if withdrawn_at is not None:
            withdrawn = _parse_iso(withdrawn_at)
            if check >= withdrawn:
                return False

        return True

    # No specific time — just check if withdrawn
    return withdrawn_at is None


# ── Extraction ────────────────────────────────────────────────────

def get_protection_metadata(node: Any) -> DataProtectionMetadata:
    """Extract data protection metadata from a JSON-LD node.

    Mirrors :func:`ai_ml.get_provenance` in pattern. Returns a
    :class:`DataProtectionMetadata` with all fields set to None
    if the node contains no protection annotations.

    Args:
        node: A JSON-LD node dict, or any other value.

    Returns:
        A :class:`DataProtectionMetadata` instance.
    """
    if node is None or not isinstance(node, dict):
        return DataProtectionMetadata()

    return DataProtectionMetadata(
        personal_data_category=node.get("@personalDataCategory"),
        legal_basis=node.get("@legalBasis"),
        processing_purpose=node.get("@processingPurpose"),
        data_controller=node.get("@dataController"),
        data_processor=node.get("@dataProcessor"),
        data_subject=node.get("@dataSubject"),
        retention_until=node.get("@retentionUntil"),
        jurisdiction=node.get("@jurisdiction"),
        access_level=node.get("@accessLevel"),
        consent=node.get("@consent"),
    )


# ── Classification Helpers ────────────────────────────────────────

def is_personal_data(node: Any) -> bool:
    """Check if a node is classified as personal data.

    Under GDPR, personal data includes regular, sensitive,
    special_category, and pseudonymized data. Anonymized,
    synthetic, and non_personal data are NOT personal data.

    Args:
        node: A JSON-LD node dict.

    Returns:
        True if the node is classified as personal data.
    """
    if node is None or not isinstance(node, dict):
        return False
    category = node.get("@personalDataCategory")
    return category in _PERSONAL_CATEGORIES


def is_sensitive_data(node: Any) -> bool:
    """Check if a node is classified as sensitive or special category data.

    Args:
        node: A JSON-LD node dict.

    Returns:
        True if the node is sensitive or special_category.
    """
    if node is None or not isinstance(node, dict):
        return False
    category = node.get("@personalDataCategory")
    return category in _SENSITIVE_CATEGORIES


# ── Graph Filtering ───────────────────────────────────────────────

def filter_by_jurisdiction(
    graph: Sequence[dict[str, Any]],
    property_name: str,
    jurisdiction: str,
) -> list[dict[str, Any]]:
    """Filter graph nodes where a property has a specific jurisdiction.

    Args:
        graph: A list of JSON-LD node dicts.
        property_name: The property to inspect for jurisdiction metadata.
        jurisdiction: The jurisdiction code to match.

    Returns:
        Nodes where at least one value of the property matches the jurisdiction.
    """
    results: list[dict[str, Any]] = []
    for node in graph:
        prop = node.get(property_name)
        if prop is None:
            continue
        values = prop if isinstance(prop, list) else [prop]
        for v in values:
            if isinstance(v, dict) and v.get("@jurisdiction") == jurisdiction:
                results.append(node)
                break
    return results


def filter_personal_data(
    graph: Sequence[dict[str, Any]],
    property_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Filter graph nodes that contain personal data.

    Args:
        graph: A list of JSON-LD node dicts.
        property_name: If specified, only check this property.
            If None, check all properties on each node.

    Returns:
        Nodes that contain at least one personal data value.
    """
    results: list[dict[str, Any]] = []
    for node in graph:
        if property_name is not None:
            prop = node.get(property_name)
            if prop is None:
                continue
            values = prop if isinstance(prop, list) else [prop]
            if any(is_personal_data(v) for v in values):
                results.append(node)
        else:
            # Check all properties
            found = False
            for key, prop in node.items():
                if key.startswith("@"):
                    # Skip JSON-LD keywords on the node itself
                    # (we only inspect property values, not @id/@type)
                    continue
                values = prop if isinstance(prop, list) else [prop]
                if any(is_personal_data(v) for v in values):
                    found = True
                    break
            if found:
                results.append(node)
    return results


# ── Internal Helpers ──────────────────────────────────────────────

def _parse_iso(s: str) -> datetime:
    """Parse an ISO 8601 datetime string to a timezone-aware datetime."""
    # Python 3.11+ fromisoformat handles 'Z' suffix
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    # If no timezone info, assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt
