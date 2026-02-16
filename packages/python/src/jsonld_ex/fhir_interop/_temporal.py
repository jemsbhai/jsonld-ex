"""
Temporal decay for clinical evidence.

Bridges jsonld-ex ``decay_opinion()`` to FHIR resource timestamps,
producing documents whose opinions reflect evidential aging.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any

from jsonld_ex.confidence_decay import decay_opinion
from jsonld_ex.owl_interop import ConversionReport


# Timestamp field priority: we search these keys in order to find
# the most clinically relevant timestamp for a FHIR-derived document.
_TIMESTAMP_FIELDS: tuple[str, ...] = (
    "effectiveDateTime",   # Observation, DiagnosticReport
    "occurrenceDateTime",  # Immunization, Procedure
    "onsetDateTime",       # Condition, AllergyIntolerance
    "recordedDate",        # Condition, AllergyIntolerance (fallback)
    "date",                # ClinicalImpression, DetectedIssue
    "assertedDate",        # AllergyIntolerance
)


def _extract_timestamp(doc: dict[str, Any]) -> datetime | None:
    """Extract the best available timestamp from a jsonld-ex document.

    Searches ``_TIMESTAMP_FIELDS`` in priority order and returns the
    first successfully parsed ISO-8601 datetime.  Returns ``None``
    if no parseable timestamp is found.
    """
    for field_name in _TIMESTAMP_FIELDS:
        raw = doc.get(field_name)
        if raw is None:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, AttributeError):
            continue
    return None


def fhir_temporal_decay(
    doc: dict[str, Any],
    *,
    reference_time: str | None = None,
    half_life_days: float = 365.0,
) -> tuple[dict[str, Any], ConversionReport]:
    """Apply temporal decay to all opinions in a FHIR-derived document.

    Clinical evidence ages: a lab result from last week is more
    relevant than one from two years ago.  This function bridges
    the jsonld-ex ``decay_opinion()`` operator to FHIR resource
    timestamps, producing a new document whose opinions reflect
    evidential aging.

    The decay model is exponential by default: after one half-life,
    belief and disbelief are halved, with the freed mass migrating
    to uncertainty.

    Args:
        doc:             A jsonld-ex document (output of ``from_fhir()``).
        reference_time:  ISO-8601 datetime string for "now".
                         Defaults to ``datetime.now(timezone.utc)``.
        half_life_days:  Time (in days) for belief/disbelief to halve.
                         Default 365 days (1 year).

    Returns:
        ``(decayed_doc, report)`` where ``decayed_doc`` is a deep
        copy with all opinions decayed.

    Notes:
        - If no timestamp is found, the document is returned
          unchanged with a warning.
        - If the timestamp is in the future, the document is returned
          unchanged with a warning.
        - The original document is never mutated.
    """
    if reference_time is not None:
        ref_dt = datetime.fromisoformat(
            reference_time.replace("Z", "+00:00")
        )
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=timezone.utc)
    else:
        ref_dt = datetime.now(timezone.utc)

    result = copy.deepcopy(doc)
    warnings: list[str] = []

    doc_dt = _extract_timestamp(result)

    if doc_dt is None:
        warnings.append(
            "No parseable timestamp found in document; "
            "opinions returned unchanged."
        )
        report = ConversionReport(
            success=True,
            warnings=warnings,
        )
        return result, report

    elapsed_seconds = (ref_dt - doc_dt).total_seconds()

    if elapsed_seconds < 0:
        warnings.append(
            f"Document timestamp ({doc_dt.isoformat()}) is in the "
            f"future relative to reference_time "
            f"({ref_dt.isoformat()}); opinions returned unchanged."
        )
        report = ConversionReport(
            success=True,
            warnings=warnings,
        )
        return result, report

    elapsed_days = elapsed_seconds / 86400.0

    opinions = result.get("opinions", [])
    for entry in opinions:
        op = entry["opinion"]
        decayed_op = decay_opinion(
            op,
            elapsed=elapsed_days,
            half_life=half_life_days,
        )
        entry["opinion"] = decayed_op

    report = ConversionReport(
        success=True,
        nodes_converted=len(opinions),
        warnings=warnings,
    )
    return result, report
