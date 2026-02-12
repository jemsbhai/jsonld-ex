"""
AI/ML Extensions for JSON-LD

Provides @confidence, @source, @extractedAt, @method, @humanVerified
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Literal
import math

JSONLD_EX_NAMESPACE = "http://www.w3.org/ns/jsonld-ex/"


@dataclass
class ProvenanceMetadata:
    """AI/ML provenance metadata attached to a value."""
    confidence: Optional[float] = None
    source: Optional[str] = None
    extracted_at: Optional[str] = None
    method: Optional[str] = None
    human_verified: Optional[bool] = None
    # Multimodal (GAP-MM1)
    media_type: Optional[str] = None
    content_url: Optional[str] = None
    content_hash: Optional[str] = None
    # Translation provenance (GAP-ML2)
    translated_from: Optional[str] = None
    translation_model: Optional[str] = None
    # Measurement uncertainty (GAP-IOT1)
    measurement_uncertainty: Optional[float] = None
    unit: Optional[str] = None


def annotate(
    value: Any,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    extracted_at: Optional[str] = None,
    method: Optional[str] = None,
    human_verified: Optional[bool] = None,
    # Multimodal (GAP-MM1)
    media_type: Optional[str] = None,
    content_url: Optional[str] = None,
    content_hash: Optional[str] = None,
    # Translation provenance (GAP-ML2)
    translated_from: Optional[str] = None,
    translation_model: Optional[str] = None,
    # Measurement uncertainty (GAP-IOT1)
    measurement_uncertainty: Optional[float] = None,
    unit: Optional[str] = None,
) -> dict[str, Any]:
    """Create an annotated JSON-LD value with provenance metadata."""
    result: dict[str, Any] = {"@value": value}

    if confidence is not None:
        _validate_confidence(confidence)
        result["@confidence"] = confidence
    if source is not None:
        result["@source"] = source
    if extracted_at is not None:
        result["@extractedAt"] = extracted_at
    if method is not None:
        result["@method"] = method
    if human_verified is not None:
        result["@humanVerified"] = human_verified
    # Multimodal
    if media_type is not None:
        result["@mediaType"] = media_type
    if content_url is not None:
        result["@contentUrl"] = content_url
    if content_hash is not None:
        result["@contentHash"] = content_hash
    # Translation
    if translated_from is not None:
        result["@translatedFrom"] = translated_from
    if translation_model is not None:
        result["@translationModel"] = translation_model
    # Measurement
    if measurement_uncertainty is not None:
        result["@measurementUncertainty"] = measurement_uncertainty
    if unit is not None:
        result["@unit"] = unit

    return result


def get_confidence(node: Any) -> Optional[float]:
    """Extract confidence score from a node or annotated value."""
    if node is None or not isinstance(node, dict):
        return None

    # Compact form
    if "@confidence" in node:
        return node["@confidence"]

    # Expanded form
    key = f"{JSONLD_EX_NAMESPACE}confidence"
    if key in node:
        val = node[key]
        if isinstance(val, list) and len(val) > 0:
            item = val[0]
            return item.get("@value", item) if isinstance(item, dict) else item
        if isinstance(val, dict):
            return val.get("@value", val)
        return val

    return None


def get_provenance(node: Any) -> ProvenanceMetadata:
    """Extract all provenance metadata from a node."""
    if node is None or not isinstance(node, dict):
        return ProvenanceMetadata()

    return ProvenanceMetadata(
        confidence=_extract_field(node, "confidence", "@confidence"),
        source=_extract_field(node, "source", "@source"),
        extracted_at=_extract_field(node, "extractedAt", "@extractedAt"),
        method=_extract_field(node, "method", "@method"),
        human_verified=_extract_field(node, "humanVerified", "@humanVerified"),
        # Multimodal
        media_type=_extract_field(node, "mediaType", "@mediaType"),
        content_url=_extract_field(node, "contentUrl", "@contentUrl"),
        content_hash=_extract_field(node, "contentHash", "@contentHash"),
        # Translation
        translated_from=_extract_field(node, "translatedFrom", "@translatedFrom"),
        translation_model=_extract_field(node, "translationModel", "@translationModel"),
        # Measurement
        measurement_uncertainty=_extract_field(node, "measurementUncertainty", "@measurementUncertainty"),
        unit=_extract_field(node, "unit", "@unit"),
    )


def filter_by_confidence(
    graph: Sequence[dict[str, Any]],
    property_name: str,
    min_confidence: float,
) -> list[dict[str, Any]]:
    """Filter graph nodes by minimum confidence on a property."""
    _validate_confidence(min_confidence)
    results = []
    for node in graph:
        prop_value = node.get(property_name)
        if prop_value is None:
            continue
        values = prop_value if isinstance(prop_value, list) else [prop_value]
        if any(
            (c := get_confidence(v)) is not None and c >= min_confidence
            for v in values
        ):
            results.append(node)
    return results


def aggregate_confidence(
    scores: Sequence[float],
    strategy: Literal["mean", "max", "min", "weighted"] = "mean",
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Aggregate multiple confidence scores."""
    if len(scores) == 0:
        return 0.0
    for s in scores:
        _validate_confidence(s)

    if strategy == "max":
        return max(scores)
    elif strategy == "min":
        return min(scores)
    elif strategy == "weighted":
        if weights is None or len(weights) != len(scores):
            raise ValueError("Weights must match scores length")
        for w in weights:
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise TypeError(f"Weight must be a number, got: {type(w).__name__}")
            if w < 0:
                raise ValueError(f"Weight must be non-negative, got: {w}")
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight must be greater than zero")
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:  # mean
        return sum(scores) / len(scores)


# ── Internal ───────────────────────────────────────────────────────

def _validate_confidence(score: float) -> None:
    if not isinstance(score, (int, float)) or isinstance(score, bool):
        raise TypeError(f"@confidence must be a number, got: {type(score).__name__}")
    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"@confidence must be finite, got: {score}")
    if score < 0 or score > 1:
        raise ValueError(f"@confidence must be between 0.0 and 1.0, got: {score}")


def _extract_field(node: dict, compact_name: str, keyword: str) -> Any:
    if keyword in node:
        return node[keyword]
    if compact_name in node:
        return node[compact_name]
    expanded_key = f"{JSONLD_EX_NAMESPACE}{compact_name}"
    if expanded_key in node:
        val = node[expanded_key]
        if isinstance(val, list) and len(val) > 0:
            item = val[0]
            return item.get("@value", item) if isinstance(item, dict) else item
        if isinstance(val, dict):
            return val.get("@value", val)
        return val
    return None
