"""Batch API for high-throughput annotation, validation, and filtering (GAP-API1).

All functions accept lists and process with minimal per-call overhead,
targeting 100K+ nodes for ML pipeline scalability.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence, Union

from jsonld_ex.ai_ml import annotate, get_confidence
from jsonld_ex.validation import ValidationResult, validate_node


def annotate_batch(
    items: Sequence[Any],
    *,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    extracted_at: Optional[str] = None,
    method: Optional[str] = None,
    human_verified: Optional[bool] = None,
    media_type: Optional[str] = None,
    content_url: Optional[str] = None,
    content_hash: Optional[str] = None,
    translated_from: Optional[str] = None,
    translation_model: Optional[str] = None,
    measurement_uncertainty: Optional[float] = None,
    unit: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Annotate a list of values with shared provenance metadata.

    Each element of *items* can be either:
    - A plain value (applied with the shared keyword arguments), or
    - A ``(value, overrides)`` tuple where *overrides* is a dict of
      keyword arguments that override the shared defaults for that item.

    Returns a list of annotated JSON-LD value nodes.
    """
    # Build shared kwargs dict once (skip None values for efficiency)
    shared: dict[str, Any] = {}
    if confidence is not None:
        shared["confidence"] = confidence
    if source is not None:
        shared["source"] = source
    if extracted_at is not None:
        shared["extracted_at"] = extracted_at
    if method is not None:
        shared["method"] = method
    if human_verified is not None:
        shared["human_verified"] = human_verified
    if media_type is not None:
        shared["media_type"] = media_type
    if content_url is not None:
        shared["content_url"] = content_url
    if content_hash is not None:
        shared["content_hash"] = content_hash
    if translated_from is not None:
        shared["translated_from"] = translated_from
    if translation_model is not None:
        shared["translation_model"] = translation_model
    if measurement_uncertainty is not None:
        shared["measurement_uncertainty"] = measurement_uncertainty
    if unit is not None:
        shared["unit"] = unit

    results: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, tuple) and len(item) == 2:
            value, overrides = item
            kwargs = {**shared, **overrides}
        else:
            value = item
            kwargs = shared
        results.append(annotate(value, **kwargs))
    return results


def validate_batch(
    nodes: Sequence[dict[str, Any]],
    shape: dict[str, Any],
) -> list[ValidationResult]:
    """Validate a list of nodes against a single shape.

    Returns one :class:`ValidationResult` per node, in order.
    """
    return [validate_node(node, shape) for node in nodes]


def filter_by_confidence_batch(
    nodes: Sequence[dict[str, Any]],
    criteria: Union[str, Sequence[tuple[str, float]]],
    min_confidence: float = 0.0,
) -> list[dict[str, Any]]:
    """Filter nodes by confidence threshold on one or more properties.

    Parameters
    ----------
    nodes:
        List of JSON-LD nodes to filter.
    criteria:
        Either a single property name (used with *min_confidence*),
        or a list of ``(property_name, threshold)`` pairs for multi-property
        filtering (node must satisfy ALL thresholds).
    min_confidence:
        Threshold when *criteria* is a single property name.
    """
    if isinstance(criteria, str):
        pairs = [(criteria, min_confidence)]
    else:
        pairs = list(criteria)

    results: list[dict[str, Any]] = []
    for node in nodes:
        if _passes_all(node, pairs):
            results.append(node)
    return results


# -- Internal -----------------------------------------------------------------


def _passes_all(
    node: dict[str, Any],
    pairs: list[tuple[str, float]],
) -> bool:
    """Return True if *node* meets all (property, threshold) criteria."""
    for prop, threshold in pairs:
        prop_value = node.get(prop)
        if prop_value is None:
            return False
        values = prop_value if isinstance(prop_value, list) else [prop_value]
        if not any(
            (c := get_confidence(v)) is not None and c >= threshold
            for v in values
        ):
            return False
    return True
