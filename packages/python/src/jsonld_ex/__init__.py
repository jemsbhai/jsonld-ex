"""
jsonld-ex: JSON-LD 1.2 Extensions for AI/ML, Security, and Validation

Reference implementation of proposed JSON-LD 1.2 extensions.
Wraps PyLD for core JSON-LD processing and adds extension layers.
"""

__version__ = "0.6.5"

from jsonld_ex.processor import JsonLdEx
from jsonld_ex.ai_ml import annotate, get_confidence, get_provenance, filter_by_confidence
from jsonld_ex.vector import validate_vector, cosine_similarity, vector_term_definition
from jsonld_ex.similarity import (
    similarity,
    compare_metrics,
    analyze_vectors,
    recommend_metric,
    evaluate_metrics,
    euclidean_distance,
    dot_product,
    manhattan_distance,
    chebyshev_distance,
    hamming_distance,
    jaccard_similarity,
    register_similarity_metric,
    get_similarity_metric,
    list_similarity_metrics,
    unregister_similarity_metric,
    BUILTIN_METRIC_NAMES,
    MetricProperties,
    VectorProperties,
    HeuristicRecommender,
    get_metric_properties,
    get_all_metric_properties,
)
from jsonld_ex.security import compute_integrity, verify_integrity, is_context_allowed
from jsonld_ex.validation import validate_node, validate_document
from jsonld_ex.owl_interop import (
    ConversionReport,
    VerbosityComparison,
    to_prov_o,
    to_prov_o_graph,
    from_prov_o,
    shape_to_shacl,
    shacl_to_shape,
    shape_to_owl_restrictions,
    owl_to_shape,
    to_rdf_star_ntriples,
    from_rdf_star_ntriples,
    to_rdf_star_turtle,
    to_ssn,
    from_ssn,
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
    pairwise_conflict,
    conflict_metric,
    robust_fuse,
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
from jsonld_ex.batch import annotate_batch, validate_batch, filter_by_confidence_batch
from jsonld_ex.dataset import (
    create_dataset_metadata,
    validate_dataset_metadata,
    add_distribution,
    add_file_set,
    add_record_set,
    create_field,
    to_croissant,
    from_croissant,
    DATASET_CONTEXT,
    CROISSANT_CONTEXT,
    DATASET_SHAPE,
)
from jsonld_ex.context import (
    context_diff,
    check_compatibility,
    ContextDiff,
    CompatibilityResult,
)
from jsonld_ex.data_protection import (
    annotate_protection,
    get_protection_metadata,
    DataProtectionMetadata,
    ConsentRecord,
    create_consent_record,
    is_consent_active,
    is_personal_data,
    is_sensitive_data,
    filter_by_jurisdiction,
    filter_personal_data,
    LEGAL_BASES,
    PERSONAL_DATA_CATEGORIES,
    CONSENT_GRANULARITIES,
    ACCESS_LEVELS,
)
from jsonld_ex.data_rights import (
    request_erasure,
    execute_erasure,
    request_restriction,
    export_portable,
    rectify_data,
    right_of_access_report,
    validate_retention,
    audit_trail,
    ErasurePlan,
    ErasureAudit,
    RestrictionResult,
    PortableExport,
    AccessReport,
    RetentionViolation,
    AuditEntry,
)

from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    jurisdictional_meet,
    compliance_propagation,
    ProvenanceChain,
    consent_validity,
    withdrawal_override,
    expiry_trigger,
    review_due_trigger,
    regulatory_change_trigger,
    erasure_scope_opinion,
    residual_contamination,
)

from jsonld_ex.dpv_interop import (
    to_dpv,
    from_dpv,
    compare_with_dpv,
    DPV,
    EU_GDPR,
    DPV_LOC,
    DPV_PD,
    LEGAL_BASIS_TO_DPV,
    CATEGORY_TO_DPV,
)
from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    jurisdictional_meet,
    compliance_propagation,
    ProvenanceChain,
    consent_validity,
    withdrawal_override,
    expiry_trigger,
    review_due_trigger,
    regulatory_change_trigger,
    erasure_scope_opinion,
    residual_contamination,
)

# Optional modules — import only if dependencies are available
try:
    from jsonld_ex.cbor_ld import to_cbor, from_cbor, payload_stats, PayloadStats
except ImportError:
    pass

try:
    from jsonld_ex.mqtt import (
        to_mqtt_payload, from_mqtt_payload,
        derive_mqtt_topic, derive_mqtt_qos, derive_mqtt_qos_detailed,
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
    # Similarity metrics & registry
    "similarity",
    "euclidean_distance",
    "dot_product",
    "manhattan_distance",
    "chebyshev_distance",
    "hamming_distance",
    "jaccard_similarity",
    "register_similarity_metric",
    "get_similarity_metric",
    "list_similarity_metrics",
    "unregister_similarity_metric",
    "BUILTIN_METRIC_NAMES",
    # Metric selection advisory
    "MetricProperties",
    "VectorProperties",
    "HeuristicRecommender",
    "compare_metrics",
    "analyze_vectors",
    "recommend_metric",
    "evaluate_metrics",
    "get_metric_properties",
    "get_all_metric_properties",
    # Security
    "compute_integrity",
    "verify_integrity",
    "is_context_allowed",
    # Validation
    "validate_node",
    "validate_document",
    # OWL/RDF interoperability
    "to_prov_o",
    "to_prov_o_graph",
    "from_prov_o",
    "shape_to_shacl",
    "shacl_to_shape",
    "shape_to_owl_restrictions",
    "owl_to_shape",
    "to_rdf_star_ntriples",
    "from_rdf_star_ntriples",
    "to_rdf_star_turtle",
    "to_ssn",
    "from_ssn",
    "ConversionReport",
    "VerbosityComparison",
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
    "pairwise_conflict",
    "conflict_metric",
    "robust_fuse",
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
    # Dataset metadata (Croissant interop)
    "create_dataset_metadata",
    "validate_dataset_metadata",
    "add_distribution",
    "add_file_set",
    "add_record_set",
    "create_field",
    "to_croissant",
    "from_croissant",
    "DATASET_CONTEXT",
    "CROISSANT_CONTEXT",
    "DATASET_SHAPE",
    # Batch API
    "annotate_batch",
    "validate_batch",
    "filter_by_confidence_batch",
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
    "derive_mqtt_qos_detailed",
    # Context versioning
    "context_diff",
    "check_compatibility",
    "ContextDiff",
    "CompatibilityResult",
    # Data protection (GDPR/privacy compliance)
    "annotate_protection",
    "get_protection_metadata",
    "DataProtectionMetadata",
    "ConsentRecord",
    "create_consent_record",
    "is_consent_active",
    "is_personal_data",
    "is_sensitive_data",
    "filter_by_jurisdiction",
    "filter_personal_data",
    "LEGAL_BASES",
    "PERSONAL_DATA_CATEGORIES",
    "CONSENT_GRANULARITIES",
    "ACCESS_LEVELS",
    # Data subject rights (GDPR Articles 15-20)
    "request_erasure",
    "execute_erasure",
    "request_restriction",
    "export_portable",
    "rectify_data",
    "right_of_access_report",
    "validate_retention",
    "audit_trail",
    "ErasurePlan",
    "ErasureAudit",
    "RestrictionResult",
    "PortableExport",
    "AccessReport",
    "RetentionViolation",
    "AuditEntry",
    # DPV v2.2 interoperability
    "to_dpv",
    "from_dpv",
    "compare_with_dpv",
    "DPV",
    "EU_GDPR",
    "DPV_LOC",
    "DPV_PD",
    "LEGAL_BASIS_TO_DPV",
    "CATEGORY_TO_DPV",
    # Compliance algebra (Subjective Logic)
    "ComplianceOpinion",
    "jurisdictional_meet",
    "compliance_propagation",
    "ProvenanceChain",
    "consent_validity",
    "withdrawal_override",
    "expiry_trigger",
    "review_due_trigger",
    "regulatory_change_trigger",
    "erasure_scope_opinion",
    "residual_contamination",
]
