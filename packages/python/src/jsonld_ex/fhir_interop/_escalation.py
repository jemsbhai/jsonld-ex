"""
Confidence-based clinical escalation policy.

Categorises jsonld-ex documents into clinical triage buckets based
on opinion belief/disbelief/uncertainty profiles and pairwise conflict.
"""

from __future__ import annotations

from typing import Any

from jsonld_ex.confidence_algebra import Opinion, pairwise_conflict


def _primary_opinion(doc: dict[str, Any]) -> Opinion | None:
    """Extract the first opinion from a jsonld-ex document, or None."""
    opinions = doc.get("opinions", [])
    if not opinions:
        return None
    return opinions[0].get("opinion")


def fhir_escalation_policy(
    resources: list[dict[str, Any]],
    *,
    escalation_threshold: float = 0.7,
    conflict_threshold: float = 0.3,
) -> dict[str, list[dict[str, Any]]]:
    """Categorise jsonld-ex documents into clinical triage buckets.

    Implements confidence-based triage for clinical decision support.
    Each document is routed to exactly one bucket based on its
    primary opinion's belief/disbelief/uncertainty profile.

    Buckets:
        ``auto_accept``
            High belief (>= escalation_threshold) and low disbelief.
        ``human_review``
            Moderate confidence.  Needs clinician review.
        ``reject``
            High disbelief (>= escalation_threshold).
        ``escalate``
            High pairwise conflict (>= conflict_threshold).

    Args:
        resources:            List of jsonld-ex documents.
        escalation_threshold: Belief/disbelief threshold for
                              auto_accept/reject (default 0.7).
        conflict_threshold:   Pairwise conflict threshold for
                              escalation (default 0.3).

    Returns:
        Dict with keys ``auto_accept``, ``human_review``, ``reject``,
        ``escalate``.  Each value is a list of documents.
    """
    result: dict[str, list[dict[str, Any]]] = {
        "auto_accept": [],
        "human_review": [],
        "reject": [],
        "escalate": [],
    }

    if not resources:
        return result

    # First pass: detect pairwise conflicts for escalation
    escalated_indices: set[int] = set()
    opinions_list: list[Opinion | None] = [
        _primary_opinion(doc) for doc in resources
    ]

    for i in range(len(resources)):
        for j in range(i + 1, len(resources)):
            op_i = opinions_list[i]
            op_j = opinions_list[j]
            if op_i is None or op_j is None:
                continue
            conflict = pairwise_conflict(op_i, op_j)
            if conflict >= conflict_threshold:
                escalated_indices.add(i)
                escalated_indices.add(j)

    # Second pass: categorise each document
    for idx, doc in enumerate(resources):
        if idx in escalated_indices:
            result["escalate"].append(doc)
            continue

        op = opinions_list[idx]
        if op is None:
            result["human_review"].append(doc)
            continue

        if op.disbelief >= escalation_threshold:
            result["reject"].append(doc)
        elif op.belief >= escalation_threshold:
            result["auto_accept"].append(doc)
        else:
            result["human_review"].append(doc)

    return result
