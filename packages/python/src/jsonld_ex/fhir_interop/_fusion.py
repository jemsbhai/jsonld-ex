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
    trust_discount,
)


@dataclass
class TrustChainReport:
    """Report from fhir_trust_chain() describing each hop in the chain.

    Provides a faithful audit trail of how the originating opinion
    degrades through successive trust discounts.  Every intermediate
    state is recorded so that downstream consumers (or auditors) can
    verify the algebraic result from the provenance chain — matching
    the accountability requirement described in Definition 7 of the
    compliance algebra specification.

    Attributes:
        chain_length:      Number of documents in the referral chain.
        per_hop_opinions:  Opinion after each discount step.  Index 0
                           is the original opinion (no discount);
                           index *i* is the opinion after *i* trust
                           discount operations.
        per_hop_projected: Projected probability at each hop, computed
                           as ``opinion.projected_probability``.
        warnings:          Any issues encountered during processing.
    """

    chain_length: int = 0
    per_hop_opinions: list[Opinion] = field(default_factory=list)
    per_hop_projected: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


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


def fhir_trust_chain(
    chain: Sequence[dict[str, Any]],
    trust_levels: Sequence[Opinion],
) -> tuple[Opinion, TrustChainReport]:
    """Apply cascading trust discount through a clinical referral chain.

    Models how epistemic confidence degrades as an assessment passes
    through intermediaries with varying trustworthiness.  For example,
    a PCP’s observation forwarded through a specialist to an AI model
    loses confidence at each hop proportional to the trust placed in
    the intermediary.

    The chain starts with ``chain[0]``’s opinion (the originating
    assessor) and applies ``trust_discount(trust_levels[i], current)``
    at each subsequent hop.  The final result represents the opinion
    as seen from the perspective of the last entity in the chain.

    Mathematical properties (from Jøsang 2016 §14.3):

    - **Full-trust identity:** If every trust opinion is dogmatic
      belief ``(1, 0, 0, ·)``, the output equals the input opinion.
    - **Zero-trust vacuous:** If any trust opinion has ``b = 0``,
      the result from that hop onward is vacuous (``u = 1``).
    - **Monotonic degradation:** Projected probability moves toward
      the base rate with each partial-trust hop; it cannot increase
      above the originating opinion’s projected probability.
    - **Uncertainty accumulation:** Uncertainty is non-decreasing
      through partial-trust hops.

    Args:
        chain:        Ordered sequence of jsonld-ex documents (as
                      produced by :func:`from_fhir`), representing
                      the referral chain from originator to final
                      receiver.  Each document must contain at least
                      one opinion entry.
        trust_levels: Trust opinions between consecutive hops.
                      ``trust_levels[i]`` is the trust that the
                      link between ``chain[i]`` and ``chain[i+1]``
                      carries.  Must have exactly
                      ``len(chain) - 1`` elements.

    Returns:
        Tuple of ``(final_opinion, TrustChainReport)``.

    Raises:
        ValueError: If *chain* is empty, *trust_levels* has the wrong
            length, or any document in the chain lacks opinions.
    """
    if not chain:
        raise ValueError(
            "Empty chain: at least one document is required."
        )

    expected_trust_count = len(chain) - 1
    if len(trust_levels) != expected_trust_count:
        raise ValueError(
            f"Trust levels length mismatch: expected {expected_trust_count} "
            f"(len(chain) - 1), got {len(trust_levels)}."
        )

    # --- Extract the originating opinion from chain[0] ---------------
    per_hop_opinions: list[Opinion] = []
    warnings: list[str] = []

    first_doc = chain[0]
    first_opinions = first_doc.get("opinions", [])
    if not first_opinions:
        doc_id = first_doc.get("id", "unknown")
        doc_type = first_doc.get("@type", "unknown")
        raise ValueError(
            f"No opinion in chain[0] ({doc_type} '{doc_id}'): "
            f"cannot start a trust chain without an originating opinion."
        )

    current = first_opinions[0]["opinion"]
    per_hop_opinions.append(current)

    # --- Validate remaining docs have opinions -----------------------
    for idx in range(1, len(chain)):
        doc = chain[idx]
        doc_opinions = doc.get("opinions", [])
        if not doc_opinions:
            doc_id = doc.get("id", "unknown")
            doc_type = doc.get("@type", "unknown")
            raise ValueError(
                f"No opinion in chain[{idx}] ({doc_type} '{doc_id}'): "
                f"a broken chain link is semantically meaningless."
            )

    # --- Apply cascading trust discount ------------------------------
    for i, trust_opinion in enumerate(trust_levels):
        current = trust_discount(trust_opinion, current)
        per_hop_opinions.append(current)

    per_hop_projected = [
        op.projected_probability for op in per_hop_opinions
    ]

    report = TrustChainReport(
        chain_length=len(chain),
        per_hop_opinions=per_hop_opinions,
        per_hop_projected=per_hop_projected,
        warnings=warnings,
    )

    return current, report
