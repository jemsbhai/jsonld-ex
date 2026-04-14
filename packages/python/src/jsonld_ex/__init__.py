"""
jsonld-ex: JSON-LD 1.2 Extensions for AI/ML, Security, and Validation

Reference implementation of proposed JSON-LD 1.2 extensions.
Wraps PyLD for core JSON-LD processing and adds extension layers.
"""

__version__ = "0.7.2"

# JsonLdEx requires pyld, which may not be compatible with all Python versions.
# Use lazy import to avoid breaking the entire package when pyld has issues.
# Access via: from jsonld_ex.processor import JsonLdEx
# Or:         jsonld_ex.JsonLdEx (triggers lazy load)

from jsonld_ex.ai_ml import annotate, get_confidence, get_provenance, filter_by_confidence
from jsonld_ex.vector import (
    validate_vector,
    cosine_similarity,
    vector_term_definition,
    quantization_descriptor,
    validate_quantization_descriptor,
)
from jsonld_ex.quantization_bridge import (
    DISTORTION_CONSTANTS,
    quantization_distortion,
    distortion_to_uncertainty,
    similarity_to_confidence,
    quantization_to_opinion,
)
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
from jsonld_ex.confidence_byzantine import (
    ByzantineStrategy,
    ByzantineConfig,
    AgentRemoval,
    ByzantineFusionReport,
    DistanceMetric,
    euclidean_opinion_distance,
    manhattan_opinion_distance,
    jsd_opinion_distance,
    hellinger_opinion_distance,
    opinion_distance,
    byzantine_fuse,
    build_conflict_matrix,
    cohesion_score,
)
from jsonld_ex.confidence_temporal_fusion import (
    TimestampedOpinion,
    TemporalFusionConfig,
    TemporalFusionReport,
    temporal_fuse,
    temporal_fuse_weighted,
    temporal_byzantine_fuse,
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
    RAI_NS,
    RAI_SPEC_VERSION,
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

try:
    from jsonld_ex.coap import (
        to_coap_payload, from_coap_payload,
        derive_coap_options, derive_coap_uri_path, derive_coap_message_type,
        CONTENT_FORMAT_CBOR, CONTENT_FORMAT_JSON, CONTENT_FORMAT_JSONLD,
        MESSAGE_TYPE_CON, MESSAGE_TYPE_NON,
    )
except ImportError:
    pass

from jsonld_ex.http_headers import (
    derive_response_headers, derive_request_headers,
    derive_etag, derive_cache_control, derive_link_header, derive_content_type,
    MEDIA_TYPE_JSONLD, MEDIA_TYPE_CBOR, MEDIA_TYPE_JSON,
)

try:
    from jsonld_ex.amqp import (
        to_amqp_payload, from_amqp_payload,
        derive_amqp_properties, derive_routing_key, derive_amqp_priority,
        derive_amqp_headers,
        DELIVERY_MODE_TRANSIENT, DELIVERY_MODE_PERSISTENT,
    )
except ImportError:
    pass

try:
    from jsonld_ex.kafka import (
        to_kafka_value, from_kafka_value,
        derive_kafka_record, derive_kafka_key, derive_kafka_headers,
        derive_kafka_timestamp, derive_kafka_topic,
    )
except ImportError:
    pass

from jsonld_ex.websocket import (
    to_ws_message, from_ws_message,
    derive_ws_metadata, derive_ws_subprotocols,
    SUBPROTOCOL_JSONLD, SUBPROTOCOL_CBOR,
    WS_OPCODE_TEXT, WS_OPCODE_BINARY,
)

from jsonld_ex.grpc import (
    derive_grpc_metadata, suggest_proto_schema,
    to_grpc_json, from_grpc_json,
    METADATA_KEY_TYPE, METADATA_KEY_CONFIDENCE,
    METADATA_KEY_SOURCE, METADATA_KEY_ID, METADATA_KEY_CONTENT_TYPE,
)

from jsonld_ex.lwm2m import (
    derive_lwm2m_objects, derive_lwm2m_registration, derive_lwm2m_links,
    extract_lwm2m_resources, IPSO_OBJECT_REGISTRY,
    BINDING_MODE_UDP, BINDING_MODE_TCP,
)

try:
    from jsonld_ex.fhir_interop import (
        # Core conversion
        scalar_to_opinion,
        opinion_to_fhir_extension,
        fhir_extension_to_opinion,
        from_fhir,
        to_fhir,
        # Fusion & trust chains
        fhir_clinical_fuse,
        fhir_trust_chain,
        FusionReport,
        TrustChainReport,
        # Temporal & escalation
        fhir_temporal_decay,
        fhir_escalation_policy,
        # Bundle processing
        fhir_bundle_annotate,
        fhir_bundle_fuse,
        BundleReport,
        # Compliance algebra bridge
        fhir_consent_to_opinion,
        opinion_to_fhir_consent,
        fhir_consent_validity,
        fhir_consent_withdrawal,
        fhir_multi_site_meet,
        fhir_consent_expiry,
        fhir_consent_regulatory_change,
        # Provenance bridge
        fhir_provenance_to_prov_o,
        # Allergy reconciliation
        fhir_allergy_reconcile,
        ReconciliationReport,
        # Alert filtering
        fhir_filter_alerts,
        AlertFilterReport,
        # Constants
        FHIR_EXTENSION_URL,
        SUPPORTED_FHIR_VERSIONS,
        FAMILY_HISTORY_DEFAULT_UNCERTAINTY,
        CONSENT_STATUS_PROBABILITY,
        CONSENT_STATUS_UNCERTAINTY,
        SUPPORTED_RESOURCE_TYPES,
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
    "quantization_descriptor",
    "validate_quantization_descriptor",
    # Quantization-SL bridge
    "DISTORTION_CONSTANTS",
    "quantization_distortion",
    "distortion_to_uncertainty",
    "similarity_to_confidence",
    "quantization_to_opinion",
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
    "RAI_NS",
    "RAI_SPEC_VERSION",
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
    # CoAP transport (requires cbor2)
    "to_coap_payload",
    "from_coap_payload",
    "derive_coap_options",
    "derive_coap_uri_path",
    "derive_coap_message_type",
    "CONTENT_FORMAT_CBOR",
    "CONTENT_FORMAT_JSON",
    "CONTENT_FORMAT_JSONLD",
    "MESSAGE_TYPE_CON",
    "MESSAGE_TYPE_NON",
    # HTTP header derivation
    "derive_response_headers",
    "derive_request_headers",
    "derive_etag",
    "derive_cache_control",
    "derive_link_header",
    "derive_content_type",
    "MEDIA_TYPE_JSONLD",
    "MEDIA_TYPE_CBOR",
    "MEDIA_TYPE_JSON",
    # AMQP transport (requires cbor2)
    "to_amqp_payload",
    "from_amqp_payload",
    "derive_amqp_properties",
    "derive_routing_key",
    "derive_amqp_priority",
    "derive_amqp_headers",
    "DELIVERY_MODE_TRANSIENT",
    "DELIVERY_MODE_PERSISTENT",
    # Kafka transport (requires cbor2)
    "to_kafka_value",
    "from_kafka_value",
    "derive_kafka_record",
    "derive_kafka_key",
    "derive_kafka_headers",
    "derive_kafka_timestamp",
    "derive_kafka_topic",
    # WebSocket transport
    "to_ws_message",
    "from_ws_message",
    "derive_ws_metadata",
    "derive_ws_subprotocols",
    "SUBPROTOCOL_JSONLD",
    "SUBPROTOCOL_CBOR",
    "WS_OPCODE_TEXT",
    "WS_OPCODE_BINARY",
    # gRPC transport
    "derive_grpc_metadata",
    "suggest_proto_schema",
    "to_grpc_json",
    "from_grpc_json",
    "METADATA_KEY_TYPE",
    "METADATA_KEY_CONFIDENCE",
    "METADATA_KEY_SOURCE",
    "METADATA_KEY_ID",
    "METADATA_KEY_CONTENT_TYPE",
    # LwM2M transport
    "derive_lwm2m_objects",
    "derive_lwm2m_registration",
    "derive_lwm2m_links",
    "extract_lwm2m_resources",
    "IPSO_OBJECT_REGISTRY",
    "BINDING_MODE_UDP",
    "BINDING_MODE_TCP",
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
    # FHIR R4 interoperability — core conversion
    "scalar_to_opinion",
    "opinion_to_fhir_extension",
    "fhir_extension_to_opinion",
    "from_fhir",
    "to_fhir",
    # FHIR R4 — fusion & trust chains
    "fhir_clinical_fuse",
    "fhir_trust_chain",
    "FusionReport",
    "TrustChainReport",
    # FHIR R4 — temporal & escalation
    "fhir_temporal_decay",
    "fhir_escalation_policy",
    # FHIR R4 — bundle processing
    "fhir_bundle_annotate",
    "fhir_bundle_fuse",
    "BundleReport",
    # FHIR R4 — compliance algebra bridge
    "fhir_consent_to_opinion",
    "opinion_to_fhir_consent",
    "fhir_consent_validity",
    "fhir_consent_withdrawal",
    "fhir_multi_site_meet",
    "fhir_consent_expiry",
    "fhir_consent_regulatory_change",
    # FHIR R4 — provenance bridge
    "fhir_provenance_to_prov_o",
    # FHIR R4 — allergy reconciliation
    "fhir_allergy_reconcile",
    "ReconciliationReport",
    # FHIR R4 — alert filtering
    "fhir_filter_alerts",
    "AlertFilterReport",
    # FHIR R4 — constants
    "FHIR_EXTENSION_URL",
    "SUPPORTED_FHIR_VERSIONS",
    "FAMILY_HISTORY_DEFAULT_UNCERTAINTY",
    "CONSENT_STATUS_PROBABILITY",
    "CONSENT_STATUS_UNCERTAINTY",
    "SUPPORTED_RESOURCE_TYPES",
    # Enhanced Byzantine fusion
    "ByzantineStrategy",
    "ByzantineConfig",
    "AgentRemoval",
    "ByzantineFusionReport",
    "DistanceMetric",
    "euclidean_opinion_distance",
    "manhattan_opinion_distance",
    "jsd_opinion_distance",
    "hellinger_opinion_distance",
    "opinion_distance",
    "byzantine_fuse",
    "build_conflict_matrix",
    "cohesion_score",
    # Temporal fusion
    "TimestampedOpinion",
    "TemporalFusionConfig",
    "TemporalFusionReport",
    "temporal_fuse",
    "temporal_fuse_weighted",
    "temporal_byzantine_fuse",
]


def __getattr__(name: str):
    """Lazy import for JsonLdEx (requires pyld, which may not support all Python versions)."""
    if name == "JsonLdEx":
        from jsonld_ex.processor import JsonLdEx
        return JsonLdEx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
