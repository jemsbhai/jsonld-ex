"""
jsonld-ex: JSON-LD 1.2 Extensions for AI/ML, Security, and Validation

Reference implementation of proposed JSON-LD 1.2 extensions.
Wraps PyLD for core JSON-LD processing and adds extension layers.
"""

__version__ = "0.1.3"

from jsonld_ex.processor import JsonLdEx
from jsonld_ex.ai_ml import annotate, get_confidence, get_provenance, filter_by_confidence
from jsonld_ex.vector import validate_vector, cosine_similarity, vector_term_definition
from jsonld_ex.security import compute_integrity, verify_integrity, is_context_allowed
from jsonld_ex.validation import validate_node, validate_document

__all__ = [
    "JsonLdEx",
    "annotate",
    "get_confidence",
    "get_provenance",
    "filter_by_confidence",
    "validate_vector",
    "cosine_similarity",
    "vector_term_definition",
    "compute_integrity",
    "verify_integrity",
    "is_context_allowed",
    "validate_node",
    "validate_document",
]
