"""Context Versioning and Diff Extensions for JSON-LD (GAP-CTX1).

Provides ``@contextVersion`` support and utilities for comparing
JSON-LD context definitions to detect additions, removals, and
breaking changes between versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TermChange:
    """Describes how a single term changed between context versions."""

    term: str
    old_value: Any = None
    new_value: Any = None


@dataclass
class ContextDiff:
    """Result of comparing two JSON-LD contexts."""

    added: dict[str, Any] = field(default_factory=dict)
    removed: dict[str, Any] = field(default_factory=dict)
    changed: dict[str, TermChange] = field(default_factory=dict)
    old_version: Optional[str] = None
    new_version: Optional[str] = None


@dataclass
class BreakingChange:
    """A single breaking or non-breaking change."""

    term: str
    change_type: str  # "removed", "changed-mapping", "changed-type", "added", etc.
    detail: str = ""


@dataclass
class CompatibilityResult:
    """Result of compatibility check between two contexts."""

    compatible: bool
    breaking: list[BreakingChange] = field(default_factory=list)
    non_breaking: list[BreakingChange] = field(default_factory=list)


def context_diff(
    old_ctx: dict[str, Any],
    new_ctx: dict[str, Any],
) -> ContextDiff:
    """Compare two JSON-LD context dicts and return a structured diff.

    Args:
        old_ctx: The previous context definition.
        new_ctx: The new context definition.

    Returns:
        A :class:`ContextDiff` with added, removed, and changed terms.
    """
    result = ContextDiff(
        old_version=old_ctx.get("@contextVersion"),
        new_version=new_ctx.get("@contextVersion"),
    )

    old_keys = set(old_ctx.keys()) - {"@contextVersion"}
    new_keys = set(new_ctx.keys()) - {"@contextVersion"}

    # Added terms
    for key in new_keys - old_keys:
        result.added[key] = new_ctx[key]

    # Removed terms
    for key in old_keys - new_keys:
        result.removed[key] = old_ctx[key]

    # Changed terms (present in both)
    for key in old_keys & new_keys:
        old_val = old_ctx[key]
        new_val = new_ctx[key]
        if old_val != new_val:
            result.changed[key] = TermChange(
                term=key, old_value=old_val, new_value=new_val,
            )

    return result


def check_compatibility(
    old_ctx: dict[str, Any],
    new_ctx: dict[str, Any],
) -> CompatibilityResult:
    """Check backward compatibility between two context versions.

    Breaking changes:
        - Removed terms
        - Changed IRI mapping (term maps to different IRI)
        - Changed or added ``@type`` coercion
        - Changed ``@vocab`` or ``@base``

    Non-breaking changes:
        - Added terms

    Args:
        old_ctx: The previous context definition.
        new_ctx: The new context definition.

    Returns:
        A :class:`CompatibilityResult` indicating compatibility.
    """
    diff = context_diff(old_ctx, new_ctx)
    breaking: list[BreakingChange] = []
    non_breaking: list[BreakingChange] = []

    # Removals are always breaking
    for term, val in diff.removed.items():
        breaking.append(BreakingChange(
            term=term,
            change_type="removed",
            detail=f"Term '{term}' was removed",
        ))

    # Additions are non-breaking
    for term, val in diff.added.items():
        non_breaking.append(BreakingChange(
            term=term,
            change_type="added",
            detail=f"Term '{term}' was added",
        ))

    # Changes require classification
    for term, change in diff.changed.items():
        # Special context keywords — changes to these are breaking
        if term.startswith("@"):
            breaking.append(BreakingChange(
                term=term,
                change_type=f"changed-{term[1:]}",
                detail=f"{term} changed from {change.old_value!r} to {change.new_value!r}",
            ))
            continue

        old_iri = _extract_iri(change.old_value)
        new_iri = _extract_iri(change.new_value)

        # IRI mapping changed
        if old_iri != new_iri:
            breaking.append(BreakingChange(
                term=term,
                change_type="changed-mapping",
                detail=f"IRI changed from {old_iri!r} to {new_iri!r}",
            ))
            continue

        # Same IRI but definition structure changed (type coercion, etc.)
        old_type = _extract_type(change.old_value)
        new_type = _extract_type(change.new_value)

        if old_type != new_type:
            breaking.append(BreakingChange(
                term=term,
                change_type="changed-type",
                detail=f"@type coercion changed from {old_type!r} to {new_type!r}",
            ))
            continue

        # Other structural change (e.g. @container) — treat as breaking
        breaking.append(BreakingChange(
            term=term,
            change_type="changed-definition",
            detail=f"Definition changed for term '{term}'",
        ))

    return CompatibilityResult(
        compatible=len(breaking) == 0,
        breaking=breaking,
        non_breaking=non_breaking,
    )


# -- Helpers ------------------------------------------------------------------


def _extract_iri(value: Any) -> Optional[str]:
    """Extract the IRI from a context term value (string or expanded def)."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("@id")
    return None


def _extract_type(value: Any) -> Optional[str]:
    """Extract @type coercion from a context term value."""
    if isinstance(value, str):
        return None  # Simple string mapping has no type coercion
    if isinstance(value, dict):
        return value.get("@type")
    return None
