"""
jsonld-ex: JSON-LD 1.2 Extensions for AI/ML, Security, and Validation

Reference implementation of proposed JSON-LD 1.2 extensions.
Wraps PyLD for core JSON-LD processing and adds extension layers.
"""

__version__ = "0.2.0-dev"

from jsonld_ex.processor import JsonLdEx
from jsonld_ex.ai_ml import annotate, get_confidence, get_provenance, filter_by_confidence
from jsonld_ex.vector import validate_vector, cosine_similarity, vector_term_definition
from jsonld_ex.security import compute_integrity, verify_integrity, is_context_allowed
from jsonld_ex.validation import validate_node, validate_document
from jsonld_ex.owl_interop import (
    to_prov_o,
    from_prov_o,
    shape_to_shacl,
    shacl_to_shape,
    shape_to_owl_restrictions,
    to_rdf_star_ntriples,
    compare_with_prov_o,
    compare_with_shacl,
)
from jsonld_ex.inference import (
    propagate_confidence,
    combine_sources,
    resolve_conflict,
    propagate_graph_confidence,
    PropagationResult,
    ConflictReport,
)

__all__ = [
    "JsonLdEx",
    # AI/ML annotations
    "annotate",
    "get_confidence",
    "get_provenance",
    "filter_by_confidence",
    # Vector operations
    "validate_vector",
    "cosine_similarity",
    "vector_term_definition",
    # Security
    "compute_integrity",
    "verify_integrity",
    "is_context_allowed",
    # Validation
    "validate_node",
    "validate_document",
    # OWL/RDF interoperability
    "to_prov_o",
    "from_prov_o",
    "shape_to_shacl",
    "shacl_to_shape",
    "shape_to_owl_restrictions",
    "to_rdf_star_ntriples",
    "compare_with_prov_o",
    "compare_with_shacl",
    # Confidence propagation & inference
    "propagate_confidence",
    "combine_sources",
    "resolve_conflict",
    "propagate_graph_confidence",
    "PropagationResult",
    "ConflictReport",
]
