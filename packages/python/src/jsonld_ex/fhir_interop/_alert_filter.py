"""
Confidence-based alert fatigue reduction using Subjective Logic.

Provides ``fhir_filter_alerts()`` — filters DetectedIssue alerts by
projected probability with an optional uncertainty ceiling that restores
the multi-dimensional advantage of Subjective Logic opinions.

Clinical motivation
-------------------
Drug-drug interaction alerts fire constantly with >90% override rates in
most EHR systems.  Clinicians develop "alert fatigue" and miss genuinely
dangerous interactions.  Using the SL projected probability as a filter
lets systems suppress low-confidence alerts while preserving high-
confidence ones.

The scalar confidence trap
--------------------------
Filtering purely on projected probability ``P = b + a·u`` collapses the
opinion back to a scalar — exactly the problem jsonld-ex is designed to
solve.  Two alerts can have identical P values with very different
evidence quality:

  - Well-evidenced: b=0.40, d=0.35, u=0.25 → P=0.525
  - Thin evidence:  b=0.10, d=0.05, u=0.85 → P=0.525

The ``uncertainty_ceiling`` parameter addresses this: alerts with
``u >= uncertainty_ceiling`` are always kept regardless of projected
probability, because the evidence base is too thin to justify
suppression.  This is both epistemically sound (we acknowledge what we
don't know) and clinically safe (fail-safe: never suppress what you
cannot assess).

Recommended configuration
~~~~~~~~~~~~~~~~~~~~~~~~~
Use both ``threshold`` and ``uncertainty_ceiling`` together for
production deployments.  ``threshold`` alone is a pragmatic
simplification suitable for research baselines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass
class AlertFilterReport:
    """Audit trail for alert filtering.

    Attributes:
        total_input:          Number of alerts received.
        kept_count:           Number of alerts kept (above threshold or
                              rescued by uncertainty ceiling).
        suppressed_count:     Number of alerts suppressed.
        kept_projected:       Projected probabilities of kept alerts,
                              in input order.
        suppressed_projected: Projected probabilities of suppressed
                              alerts, in input order.
        warnings:             Issues encountered (e.g., docs with no
                              opinions that were kept by fail-safe).
    """

    total_input: int = 0
    kept_count: int = 0
    suppressed_count: int = 0
    kept_projected: list[float] = field(default_factory=list)
    suppressed_projected: list[float] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def fhir_filter_alerts(
    docs: Sequence[dict[str, Any]],
    *,
    threshold: float = 0.5,
    uncertainty_ceiling: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], AlertFilterReport]:
    """Filter DetectedIssue alerts by SL projected probability.

    Partitions *docs* into **kept** and **suppressed** lists.  No data
    is lost — suppressed alerts are returned separately for optional
    clinician review.

    Decision logic for each alert:

    1. If the doc has no opinions → **kept** (fail-safe) with a warning.
    2. If ``uncertainty_ceiling`` is set and the opinion's uncertainty
       ``u >= uncertainty_ceiling`` → **kept** (evidence too thin to
       suppress).
    3. If the opinion's projected probability ``P >= threshold`` →
       **kept**.
    4. Otherwise → **suppressed**.

    Args:
        docs:                 Sequence of jsonld-ex documents (typically
                              DetectedIssue, but works with any doc that
                              has an ``opinions`` list with Opinion objects).
        threshold:            Projected probability cutoff.  Alerts with
                              ``P >= threshold`` are kept.  Must be in
                              ``[0.0, 1.0]``.
        uncertainty_ceiling:  If set, alerts with ``u >= ceiling`` are
                              always kept regardless of projected
                              probability.  Must be in ``[0.0, 1.0]``
                              or ``None`` (disabled).

    Returns:
        Tuple of ``(kept, suppressed, AlertFilterReport)``.

    Raises:
        ValueError: If *threshold* or *uncertainty_ceiling* is outside
                    ``[0.0, 1.0]``.
    """
    # --- Parameter validation ----------------------------------------
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(
            f"Threshold must be in [0.0, 1.0], got {threshold}"
        )
    if uncertainty_ceiling is not None and not (0.0 <= uncertainty_ceiling <= 1.0):
        raise ValueError(
            f"Uncertainty ceiling must be in [0.0, 1.0], got {uncertainty_ceiling}"
        )

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    kept_projected: list[float] = []
    suppressed_projected: list[float] = []
    warnings: list[str] = []

    for doc in docs:
        opinions = doc.get("opinions", [])

        # Fail-safe: no opinions → keep with warning
        if not opinions:
            kept.append(doc)
            warnings.append(
                f"Doc '{doc.get('id', '<unknown>')}' has no opinions; "
                f"kept by fail-safe (cannot assess for suppression)."
            )
            continue

        # Use the first opinion's projected probability and uncertainty
        # (DetectedIssue typically has exactly one opinion on severity
        # or status; if multiple exist, the first is the primary.)
        opinion = opinions[0]["opinion"]
        proj_prob = opinion.projected_probability()
        uncertainty = opinion.uncertainty

        # Decision logic
        if uncertainty_ceiling is not None and uncertainty >= uncertainty_ceiling:
            # Evidence too thin to justify suppression
            kept.append(doc)
            kept_projected.append(proj_prob)
        elif proj_prob >= threshold:
            kept.append(doc)
            kept_projected.append(proj_prob)
        else:
            suppressed.append(doc)
            suppressed_projected.append(proj_prob)

    report = AlertFilterReport(
        total_input=len(docs),
        kept_count=len(kept),
        suppressed_count=len(suppressed),
        kept_projected=kept_projected,
        suppressed_projected=suppressed_projected,
        warnings=warnings,
    )

    return kept, suppressed, report
