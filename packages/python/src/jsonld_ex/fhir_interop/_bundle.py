"""
FHIR R4 Bundle processing for JSON-LD-Ex.

Provides two key capabilities that FHIR lacks:

- ``fhir_bundle_annotate()`` — Batch-annotate all supported resources
  in a FHIR Bundle with Subjective Logic opinions.
- ``fhir_bundle_fuse()`` — Fuse multiple Bundles from different sources
  (e.g., patient data from multiple hospitals) with confidence-aware
  conflict resolution.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    robust_fuse,
    pairwise_conflict,
)
from jsonld_ex.fhir_interop._converters import from_fhir


_FUSION_METHODS = {"cumulative", "averaging", "robust"}


@dataclass
class BundleReport:
    """Report from Bundle processing operations.

    Attributes:
        total_entries:   Total entry count in the source Bundle(s).
        annotated:       Entries successfully annotated with opinions.
        skipped:         Entries skipped (unsupported type, missing
                         resource, or no opinions produced).
        warnings:        Human-readable warnings for skipped entries.
        bundle_id:       Source Bundle id (annotate) or None (fuse).
        bundle_type:     Source Bundle type (annotate) or None (fuse).
        groups_fused:    Number of resource groups that were fused
                         (i.e., same type+id appeared in >1 bundle).
        conflict_scores: Pairwise conflict scores from fusion.
        fusion_method:   Fusion method used (None for annotate).
    """

    total_entries: int = 0
    annotated: int = 0
    skipped: int = 0
    warnings: list[str] = field(default_factory=list)
    bundle_id: str | None = None
    bundle_type: str | None = None
    groups_fused: int = 0
    conflict_scores: list[float] = field(default_factory=list)
    fusion_method: str | None = None


# ── Public API ────────────────────────────────────────────────────


def fhir_bundle_annotate(
    bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], BundleReport]:
    """Annotate all supported resources in a FHIR Bundle with SL opinions.

    Iterates ``bundle.entry[].resource``, calls :func:`from_fhir` on
    each supported resource, and collects the resulting jsonld-ex
    documents.  Unsupported resource types and malformed entries are
    skipped with warnings.

    Args:
        bundle: A FHIR R4 Bundle resource (JSON-parsed dict).

    Returns:
        Tuple of ``(annotated_docs, BundleReport)``.

    Raises:
        ValueError: If *bundle* is not a FHIR Bundle resource.
    """
    _validate_bundle(bundle)

    entries = bundle.get("entry", [])
    report = BundleReport(
        total_entries=len(entries),
        bundle_id=bundle.get("id"),
        bundle_type=bundle.get("type"),
    )

    docs: list[dict[str, Any]] = []

    for i, entry in enumerate(entries):
        resource = entry.get("resource")
        if resource is None or not isinstance(resource, dict):
            report.skipped += 1
            report.warnings.append(
                f"Entry [{i}]: missing or invalid 'resource' field"
            )
            continue

        if "resourceType" not in resource:
            report.skipped += 1
            report.warnings.append(
                f"Entry [{i}]: resource missing 'resourceType'"
            )
            continue

        doc, conv_report = from_fhir(resource)

        if not doc.get("opinions"):
            # from_fhir returned a doc with empty opinions →
            # unsupported resource type
            report.skipped += 1
            report.warnings.extend(conv_report.warnings)
        else:
            report.annotated += 1

        docs.append(doc)

    return docs, report


def fhir_bundle_fuse(
    bundles: Sequence[dict[str, Any]],
    *,
    method: str = "cumulative",
) -> tuple[list[dict[str, Any]], BundleReport]:
    """Fuse multiple FHIR Bundles from different sources.

    Groups resources by ``(resourceType, id)`` across all bundles,
    then fuses opinions for groups with multiple entries using the
    specified Subjective Logic fusion operator.  Single-entry groups
    pass through unchanged.

    The production use case: patient data from N hospitals arrives
    as N separate Bundles.  This function produces a single set of
    confidence-annotated documents with conflict detection on
    overlapping resources.

    Args:
        bundles: List of FHIR R4 Bundle resources.
        method:  Fusion method — ``"cumulative"`` (default),
                 ``"averaging"``, or ``"robust"``.

    Returns:
        Tuple of ``(fused_docs, BundleReport)``.

    Raises:
        ValueError: If *bundles* is empty, contains non-Bundle
            resources, or *method* is invalid.
    """
    if not bundles:
        raise ValueError("At least one Bundle is required.")

    if method not in _FUSION_METHODS:
        raise ValueError(
            f"Unsupported fusion method '{method}'. "
            f"Supported: {', '.join(sorted(_FUSION_METHODS))}"
        )

    for i, b in enumerate(bundles):
        _validate_bundle(b)

    # Phase 1: Annotate all bundles
    all_docs: list[dict[str, Any]] = []
    total_entries = 0
    total_skipped = 0
    all_warnings: list[str] = []

    for b in bundles:
        docs, report = fhir_bundle_annotate(b)
        for doc in docs:
            if doc.get("opinions"):
                all_docs.append(doc)
            else:
                total_skipped += 1
                # Warnings already captured in annotate report
        total_entries += report.total_entries
        total_skipped += report.skipped
        all_warnings.extend(report.warnings)

    # Phase 2: Group by (type, id)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in all_docs:
        key = f"{doc.get('@type', '')}::{doc.get('id', '')}"
        groups[key].append(doc)

    # Phase 3: Fuse or pass through
    fused_docs: list[dict[str, Any]] = []
    groups_fused = 0
    all_conflict_scores: list[float] = []

    for key, group_docs in groups.items():
        if len(group_docs) == 1:
            # Single entry — no fusion needed
            fused_docs.append(group_docs[0])
        else:
            # Multiple entries — fuse opinions
            fused_doc, conflict_scores = _fuse_group(
                group_docs, method=method,
            )
            fused_docs.append(fused_doc)
            groups_fused += 1
            all_conflict_scores.extend(conflict_scores)

    fuse_report = BundleReport(
        total_entries=total_entries,
        annotated=len(all_docs),
        skipped=total_skipped,
        warnings=all_warnings,
        groups_fused=groups_fused,
        conflict_scores=all_conflict_scores,
        fusion_method=method,
    )

    return fused_docs, fuse_report


# ── Private helpers ───────────────────────────────────────────────


def _validate_bundle(bundle: dict[str, Any]) -> None:
    """Raise ValueError if *bundle* is not a FHIR Bundle."""
    if not isinstance(bundle, dict):
        raise ValueError("Bundle must be a dict.")

    rt = bundle.get("resourceType")
    if rt is None:
        raise ValueError(
            "Bundle must contain a 'resourceType' field."
        )
    if rt != "Bundle":
        raise ValueError(
            f"Expected resourceType 'Bundle', got '{rt}'."
        )


def _fuse_group(
    docs: list[dict[str, Any]],
    *,
    method: str,
) -> tuple[dict[str, Any], list[float]]:
    """Fuse opinions from a group of docs sharing the same type+id.

    Returns the fused doc and a list of pairwise conflict scores.
    """
    # Extract first opinion from each doc
    opinions: list[Opinion] = []
    for doc in docs:
        doc_opinions = doc.get("opinions", [])
        if doc_opinions:
            opinions.append(doc_opinions[0]["opinion"])

    # Compute pairwise conflict scores
    conflict_scores: list[float] = []
    for i in range(len(opinions)):
        for j in range(i + 1, len(opinions)):
            score = pairwise_conflict(opinions[i], opinions[j])
            conflict_scores.append(score)

    # Fuse
    if len(opinions) == 1:
        fused_opinion = opinions[0]
    elif method == "cumulative":
        fused_opinion = cumulative_fuse(*opinions)
    elif method == "averaging":
        fused_opinion = averaging_fuse(*opinions)
    elif method == "robust":
        fused_opinion, _removed = robust_fuse(opinions)
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Build fused doc from the first doc as template
    template = docs[0]
    fused_doc: dict[str, Any] = {
        "@type": template.get("@type"),
        "id": template.get("id"),
    }

    # Preserve status from template
    if "status" in template:
        fused_doc["status"] = template["status"]

    # Copy any non-opinion metadata from template
    for key in template:
        if key not in ("@type", "id", "status", "opinions"):
            fused_doc[key] = template[key]

    # Set fused opinion
    first_opinion_entry = template.get("opinions", [{}])[0]
    fused_doc["opinions"] = [{
        "field": first_opinion_entry.get("field", ""),
        "value": first_opinion_entry.get("value", ""),
        "opinion": fused_opinion,
        "source": "fused",
    }]

    return fused_doc, conflict_scores
