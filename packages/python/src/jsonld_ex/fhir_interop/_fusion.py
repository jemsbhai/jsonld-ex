"""
Clinical fusion engine for FHIR-derived opinions.

Provides ``fhir_clinical_fuse()`` — the key capability that FHIR lacks:
mathematically grounded combination of uncertain evidence from multiple
clinical sources into a single opinion with properly propagated uncertainty.
"""

from __future__ import annotations

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
class FusionReport:
    """Report from fhir_clinical_fuse() describing the fusion outcome.

    Attributes:
        input_count:     Number of input documents provided.
        opinions_fused:  Number of opinions actually fused (after
                         skipping empty documents).
        method:          Fusion method used ("cumulative", "averaging",
                         or "robust").
        conflict_scores: Pairwise conflict scores between opinions.
                         One entry per unique pair (i, j) with i < j.
        warnings:        Any issues encountered (e.g. empty documents).
    """

    input_count: int = 0
    opinions_fused: int = 0
    method: str = "cumulative"
    conflict_scores: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


_FUSION_METHODS = {"cumulative", "averaging", "robust"}


def fhir_clinical_fuse(
    docs: Sequence[dict[str, Any]],
    *,
    method: str = "cumulative",
) -> tuple[Opinion, FusionReport]:
    """Fuse opinions from multiple FHIR-derived jsonld-ex documents.

    This is the key capability that FHIR lacks: mathematically
    grounded combination of uncertain evidence from multiple clinical
    sources into a single opinion with properly propagated uncertainty.

    The function extracts the first opinion from each document,
    computes pairwise conflict scores, then fuses using the selected
    Subjective Logic operator:

    - **cumulative** (default): For independent sources.
    - **averaging**: For potentially correlated sources.
    - **robust**: Byzantine-resistant fusion.

    Args:
        docs:   List of jsonld-ex documents as produced by
                :func:`from_fhir`.
        method: Fusion method — one of ``"cumulative"``,
                ``"averaging"``, or ``"robust"``.

    Returns:
        Tuple of (fused_opinion, FusionReport).

    Raises:
        ValueError: If ``docs`` is empty, all documents lack opinions,
            or ``method`` is not recognized.
    """
    if method not in _FUSION_METHODS:
        raise ValueError(
            f"Unsupported method '{method}'. "
            f"Supported: {', '.join(sorted(_FUSION_METHODS))}"
        )

    if not docs:
        raise ValueError("No documents provided; at least one is required.")

    opinions: list[Opinion] = []
    warnings: list[str] = []

    for doc in docs:
        doc_opinions = doc.get("opinions", [])
        if not doc_opinions:
            doc_id = doc.get("id", "unknown")
            doc_type = doc.get("@type", "unknown")
            warnings.append(
                f"Skipped {doc_type} '{doc_id}': no opinions to fuse"
            )
            continue
        opinions.append(doc_opinions[0]["opinion"])

    if not opinions:
        raise ValueError(
            "No fusable opinions found; all documents had empty opinion lists."
        )

    conflict_scores: list[float] = []
    for i in range(len(opinions)):
        for j in range(i + 1, len(opinions)):
            score = pairwise_conflict(opinions[i], opinions[j])
            conflict_scores.append(score)

    if len(opinions) == 1:
        fused = opinions[0]
    elif method == "cumulative":
        fused = cumulative_fuse(*opinions)
    elif method == "averaging":
        fused = averaging_fuse(*opinions)
    elif method == "robust":
        fused, _removed = robust_fuse(opinions)
    else:
        raise ValueError(f"Unsupported method: {method}")

    report = FusionReport(
        input_count=len(docs),
        opinions_fused=len(opinions),
        method=method,
        conflict_scores=conflict_scores,
        warnings=warnings,
    )

    return fused, report
