"""
Multi-EHR allergy list reconciliation via Subjective Logic fusion.

Provides ``fhir_allergy_reconcile()`` — the killer use case for SL in
clinical interoperability: when a patient transfers between hospitals,
each system may have overlapping, conflicting, or unique allergy entries.
Reconciliation matches allergies by substance code, fuses opinions for
matched entries using the selected SL operator, and preserves unique
entries — producing a single reconciled allergy list with conflict
detection.

Matching key hierarchy:
  1. code.coding: same ``system`` + ``code`` (e.g., SNOMED 91936005)
  2. code.text: case-insensitive string match (fallback)
  3. No match → preserved as unique entry
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    robust_fuse,
    pairwise_conflict,
)


@dataclass
class ReconciliationReport:
    """Audit trail for allergy list reconciliation.

    Attributes:
        input_list_count:    Number of source allergy lists.
        total_input_entries: Total allergy entries across all lists.
        matched_groups:      Number of substance groups where entries
                             from 2+ lists were fused.
        unique_entries:      Entries appearing in only one list.
        conflicts:           Per-group conflict information.  Each entry
                             is a dict with ``match_key`` (the substance
                             identity used for grouping) and ``scores``
                             (pairwise conflict scores for the
                             verificationStatus opinions in that group).
        output_count:        Total entries in the reconciled list.
        warnings:            Any issues encountered during processing.
    """

    input_list_count: int = 0
    total_input_entries: int = 0
    matched_groups: int = 0
    unique_entries: int = 0
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    output_count: int = 0
    warnings: list[str] = field(default_factory=list)


_FUSION_METHODS = {"cumulative", "averaging", "robust"}


def _extract_match_key(doc: dict[str, Any]) -> str | None:
    """Extract a normalised matching key from a jsonld-ex allergy doc.

    Key hierarchy:
      1. First ``(system, code)`` pair from ``code.coding`` — normalised
         as ``"system|code"`` for unambiguous identity.
      2. ``code.text`` lowercased — fallback when no structured coding
         is available.
      3. ``None`` — no code field at all; entry cannot be matched.
    """
    code_obj = doc.get("code")
    if code_obj is None:
        return None

    # Try structured coding first
    codings = code_obj.get("coding")
    if codings:
        for c in codings:
            system = c.get("system")
            code = c.get("code")
            if system and code:
                return f"{system}|{code}"

    # Fall back to text
    text = code_obj.get("text")
    if text:
        return f"text|{text.strip().lower()}"

    return None


def _fuse_opinion_lists(
    opinions_by_field: dict[str, list[Opinion]],
    method: str,
) -> list[dict[str, Any]]:
    """Fuse grouped opinions per field using the selected SL operator.

    For each field (e.g., "verificationStatus", "criticality"), if
    multiple opinions exist they are fused; if only one exists it is
    preserved as-is.

    Returns a list of opinion entries in standard jsonld-ex format.
    """
    result: list[dict[str, Any]] = []

    for field_name, op_list in opinions_by_field.items():
        if len(op_list) == 1:
            fused = op_list[0]
            source = "original"
        elif method == "cumulative":
            fused = cumulative_fuse(*op_list)
            source = "fused"
        elif method == "averaging":
            fused = averaging_fuse(*op_list)
            source = "fused"
        elif method == "robust":
            fused, _removed = robust_fuse(op_list)
            source = "fused"
        else:
            raise ValueError(f"Unsupported method: {method}")

        result.append({
            "field": field_name,
            "value": None,  # populated below
            "opinion": fused,
            "source": source,
        })

    return result


def fhir_allergy_reconcile(
    lists: Sequence[Sequence[dict[str, Any]]],
    *,
    method: str = "cumulative",
) -> tuple[list[dict[str, Any]], ReconciliationReport]:
    """Reconcile allergy lists from multiple EHR systems.

    Matches AllergyIntolerance entries across systems by substance
    code, fuses Subjective Logic opinions for matched entries, detects
    conflicts, and preserves unique entries — producing a single
    reconciled allergy list.

    Both ``verificationStatus`` and ``criticality`` opinions are fused
    independently when a match is found.  Each represents a distinct
    clinical proposition and must be handled separately.

    Args:
        lists:  Sequence of allergy lists.  Each inner sequence
                contains jsonld-ex documents as produced by
                :func:`from_fhir` for AllergyIntolerance resources.
        method: Fusion method — ``"cumulative"`` (default),
                ``"averaging"``, or ``"robust"``.

    Returns:
        Tuple of ``(reconciled_list, ReconciliationReport)``.

    Raises:
        ValueError: If *lists* is empty or *method* is not recognized.
    """
    if not lists:
        raise ValueError("Empty input: at least one allergy list is required.")

    if method not in _FUSION_METHODS:
        raise ValueError(
            f"Unsupported fusion method '{method}'. "
            f"Supported: {', '.join(sorted(_FUSION_METHODS))}"
        )

    # --- Count inputs ------------------------------------------------
    total_entries = sum(len(lst) for lst in lists)
    warnings: list[str] = []

    # --- Group by match key ------------------------------------------
    #  key → list of (doc, list_index) tuples
    groups: dict[str, list[tuple[dict[str, Any], int]]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []

    for list_idx, allergy_list in enumerate(lists):
        for doc in allergy_list:
            key = _extract_match_key(doc)
            if key is None:
                unmatched.append(doc)
            else:
                groups[key].append((doc, list_idx))

    # --- Process each group ------------------------------------------
    reconciled: list[dict[str, Any]] = []
    matched_group_count = 0
    unique_count = 0
    conflict_entries: list[dict[str, Any]] = []

    for match_key, members in groups.items():
        # Determine which distinct source lists contributed
        source_lists = set(idx for _, idx in members)

        if len(members) == 1:
            # Only one entry for this substance → unique, pass through
            reconciled.append(members[0][0])
            unique_count += 1
            continue

        # Multiple entries → need to fuse
        matched_group_count += 1

        # Collect opinions by field across all members
        opinions_by_field: dict[str, list[Opinion]] = defaultdict(list)
        values_by_field: dict[str, str | None] = {}

        for doc, _ in members:
            for op_entry in doc.get("opinions", []):
                fname = op_entry["field"]
                opinions_by_field[fname].append(op_entry["opinion"])
                # Keep the first non-None value for the field
                if fname not in values_by_field:
                    values_by_field[fname] = op_entry.get("value")

        # Fuse opinions per field
        fused_opinions = _fuse_opinion_lists(opinions_by_field, method)

        # Attach the original value labels
        for op_entry in fused_opinions:
            fname = op_entry["field"]
            op_entry["value"] = values_by_field.get(fname)

        # Compute pairwise conflict scores for verificationStatus
        vs_opinions = opinions_by_field.get("verificationStatus", [])
        scores: list[float] = []
        if len(vs_opinions) >= 2:
            for i in range(len(vs_opinions)):
                for j in range(i + 1, len(vs_opinions)):
                    scores.append(pairwise_conflict(vs_opinions[i], vs_opinions[j]))

        conflict_entries.append({
            "match_key": match_key,
            "scores": scores,
            "member_count": len(members),
        })

        # Build the reconciled doc — use the first member as the
        # structural template (preserving code, patient, category, etc.)
        template = deepcopy(members[0][0])
        template["opinions"] = fused_opinions

        reconciled.append(template)

    # Add unmatched entries (no code → always unique)
    for doc in unmatched:
        reconciled.append(doc)
        unique_count += 1

    report = ReconciliationReport(
        input_list_count=len(lists),
        total_input_entries=total_entries,
        matched_groups=matched_group_count,
        unique_entries=unique_count,
        conflicts=conflict_entries,
        output_count=len(reconciled),
        warnings=warnings,
    )

    return reconciled, report
