"""
JsonLdEx — Extended JSON-LD Processor (Python)

Wraps PyLD with backward-compatible extensions for AI/ML,
security, and validation.
"""

from __future__ import annotations
from typing import Any, Optional

from pyld import jsonld

from jsonld_ex.ai_ml import (
    annotate, get_confidence, get_provenance,
    filter_by_confidence, aggregate_confidence, ProvenanceMetadata,
)
from jsonld_ex.vector import (
    vector_term_definition, validate_vector, cosine_similarity,
    extract_vectors, strip_vectors_for_rdf,
)
from jsonld_ex.security import (
    compute_integrity, verify_integrity, integrity_context,
    is_context_allowed, enforce_resource_limits, DEFAULT_RESOURCE_LIMITS,
)
from jsonld_ex.validation import validate_node, validate_document, ValidationResult


class JsonLdEx:
    """Extended JSON-LD processor wrapping PyLD."""

    def __init__(
        self,
        resource_limits: Optional[dict[str, int]] = None,
        context_allowlist: Optional[dict[str, Any]] = None,
    ):
        self._limits = {**DEFAULT_RESOURCE_LIMITS, **(resource_limits or {})}
        self._allowlist = context_allowlist

    # ── Core Operations ──────────────────────────────────────────

    def expand(self, doc: dict[str, Any], **kwargs: Any) -> list[dict[str, Any]]:
        """Expand a JSON-LD document with resource limit enforcement."""
        enforce_resource_limits(doc, self._limits)
        return jsonld.expand(doc, kwargs)

    def compact(self, doc: dict[str, Any], ctx: Any, **kwargs: Any) -> dict[str, Any]:
        """Compact a JSON-LD document."""
        enforce_resource_limits(doc, self._limits)
        return jsonld.compact(doc, ctx, kwargs)

    def flatten(self, doc: dict[str, Any], ctx: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Flatten a JSON-LD document."""
        enforce_resource_limits(doc, self._limits)
        return jsonld.flatten(doc, ctx, kwargs)

    def to_rdf(self, doc: dict[str, Any], **kwargs: Any) -> str:
        """Convert to N-Quads."""
        enforce_resource_limits(doc, self._limits)
        return jsonld.to_rdf(doc, {**kwargs, "format": "application/n-quads"})

    def from_rdf(self, nquads: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Convert N-Quads to JSON-LD."""
        return jsonld.from_rdf(nquads, kwargs)

    # ── AI/ML Extensions ─────────────────────────────────────────

    annotate = staticmethod(annotate)
    get_confidence = staticmethod(get_confidence)
    get_provenance = staticmethod(get_provenance)
    filter_by_confidence = staticmethod(filter_by_confidence)
    aggregate_confidence = staticmethod(aggregate_confidence)

    # ── Vector Extensions ────────────────────────────────────────

    vector_term_definition = staticmethod(vector_term_definition)
    validate_vector = staticmethod(validate_vector)
    cosine_similarity = staticmethod(cosine_similarity)
    extract_vectors = staticmethod(extract_vectors)
    strip_vectors_for_rdf = staticmethod(strip_vectors_for_rdf)

    # ── Security Extensions ──────────────────────────────────────

    compute_integrity = staticmethod(compute_integrity)
    verify_integrity = staticmethod(verify_integrity)
    integrity_context = staticmethod(integrity_context)
    is_context_allowed = staticmethod(is_context_allowed)

    # ── Validation Extensions ────────────────────────────────────

    validate_node = staticmethod(validate_node)
    validate_document = staticmethod(validate_document)
