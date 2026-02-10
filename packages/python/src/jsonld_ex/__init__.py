"""
jsonld-ex: JSON-LD 1.2 Extensions for AI/ML, Security, and Validation

Reference implementation of proposed JSON-LD 1.2 extensions.
Wraps PyLD for core JSON-LD processing and adds extension layers.
"""

__version__ = "0.2.0"

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
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
)
from jsonld_ex.confidence_bridge import (
    combine_opinions_from_scalars,
    propagate_opinions_from_scalars,
)
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)
from jsonld_ex.merge import (
    merge_graphs,
    diff_graphs,
    MergeReport,
    MergeConflict,
)
from jsonld_ex.temporal import (
    add_temporal,
    query_at_time,
    temporal_diff,
    TemporalDiffResult,
)

# Optional modules — import only if dependencies are available
try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor, payload_stats, PayloadStats
except ImportError:
    pass

try:
    from jsonld_ex.mqtt import (
        to_mqtt_payload, from_mqtt_payload,
        derive_mqtt_topic, derive_mqtt_qos,
    )
except ImportError:
    pass

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
    # Formal confidence algebra (Subjective Logic)
    "Opinion",
    "cumulative_fuse",
    "averaging_fuse",
    "trust_discount",
    "deduce",
    "combine_opinions_from_scalars",
    "propagate_opinions_from_scalars",
    # Temporal decay
    "decay_opinion",
    "exponential_decay",
    "linear_decay",
    "step_decay",
    # Graph merging
    "merge_graphs",
    "diff_graphs",
    "MergeReport",
    "MergeConflict",
    # Temporal extensions
    "add_temporal",
    "query_at_time",
    "temporal_diff",
    "TemporalDiffResult",
    # CBOR-LD serialization (requires cbor2)
    "to_cbor",
    "from_cbor",
    "payload_stats",
    "PayloadStats",
    # MQTT transport (requires cbor2)
    "to_mqtt_payload",
    "from_mqtt_payload",
    "derive_mqtt_topic",
    "derive_mqtt_qos",
]
