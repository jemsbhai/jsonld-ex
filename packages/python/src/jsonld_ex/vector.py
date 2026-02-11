"""Vector Embedding Extensions for JSON-LD."""

from __future__ import annotations
import math
from typing import Any, Optional


def vector_term_definition(
    term_name: str, iri: str, dimensions: Optional[int] = None
) -> dict[str, Any]:
    """Create a context term definition for a vector embedding property."""
    defn: dict[str, Any] = {"@id": iri, "@container": "@vector"}
    if dimensions is not None:
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise ValueError(f"@dimensions must be a positive integer, got: {dimensions}")
        defn["@dimensions"] = dimensions
    return {term_name: defn}


def validate_vector(
    vector: Any, expected_dimensions: Optional[int] = None
) -> tuple[bool, list[str]]:
    """Validate a vector embedding. Returns (valid, errors)."""
    errors: list[str] = []
    if not isinstance(vector, (list, tuple)):
        errors.append(f"Vector must be a list, got: {type(vector).__name__}")
        return False, errors
    if len(vector) == 0:
        errors.append("Vector must not be empty")
        return False, errors
    for i, v in enumerate(vector):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            errors.append(f"Vector element [{i}] must be a finite number, got: {v}")
    if expected_dimensions is not None and len(vector) != expected_dimensions:
        errors.append(
            f"Vector dimension mismatch: expected {expected_dimensions}, got {len(vector)}"
        )
    return len(errors) == 0, errors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Raises ``ValueError`` when either vector has zero magnitude,
    since cosine similarity is mathematically undefined (0/0) in that case.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be empty")
    for i, (x, y) in enumerate(zip(a, b)):
        for label, v in (("a", x), ("b", y)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"Vector {label}[{i}] must be a number, got: {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Vector {label}[{i}] must be finite, got: {v}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero-magnitude vector")
    return dot / (norm_a * norm_b)


def extract_vectors(
    node: dict[str, Any], vector_properties: list[str]
) -> dict[str, list[float]]:
    """Extract vector embeddings from a JSON-LD node."""
    vectors: dict[str, list[float]] = {}
    if not isinstance(node, dict):
        return vectors
    for prop in vector_properties:
        value = node.get(prop)
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            vectors[prop] = value
    return vectors


def strip_vectors_for_rdf(doc: Any, vector_properties: list[str]) -> Any:
    """Remove vector embeddings before RDF conversion."""
    if isinstance(doc, list):
        return [strip_vectors_for_rdf(item, vector_properties) for item in doc]
    if not isinstance(doc, dict):
        return doc
    return {
        k: strip_vectors_for_rdf(v, vector_properties)
        for k, v in doc.items()
        if k not in vector_properties
    }
