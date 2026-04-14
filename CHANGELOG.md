# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

**CoAP Transport Module** — `coap` module

- `to_coap_payload` / `from_coap_payload`: CBOR/JSON serialization for CoAP (RFC 7252)
- `derive_coap_options`: full option derivation from JSON-LD metadata (Content-Format, ETag, Max-Age, Uri-Path, Size1, Block SZX, Observable)
- `derive_coap_uri_path`: URI path segments from `@type` and `@id`
- `derive_coap_message_type`: CON/NON from `@confidence` metadata
- Constants: `CONTENT_FORMAT_CBOR`, `CONTENT_FORMAT_JSON`, `CONTENT_FORMAT_JSONLD`, `MESSAGE_TYPE_CON`, `MESSAGE_TYPE_NON`
- ETag derived from `@integrity` (SHA-256, truncated to 8 bytes per RFC 7252 §5.10.6)
- Max-Age derived from `@validUntil` (seconds remaining)
- Observable flag for resources with temporal annotations (RFC 7641)
- Block-wise transfer recommendation for payloads > 1024 bytes (RFC 7959)

**HTTP Header Derivation Module** — `http_headers` module

- `derive_response_headers`: Content-Type, ETag, Cache-Control, Link, X-JsonLD-Confidence/Source/Type
- `derive_request_headers`: Accept content negotiation, If-None-Match conditional GET
- `derive_etag`: quoted ETag from `@integrity` (SHA-256, first 32 hex chars)
- `derive_cache_control`: `max-age=N` from `@validUntil`, `no-cache` for expired
- `derive_link_header`: JSON-LD context discovery per W3C JSON-LD 1.1 §4.1
- `derive_content_type`: `application/ld+json` or `application/cbor`
- Constants: `MEDIA_TYPE_JSONLD`, `MEDIA_TYPE_CBOR`, `MEDIA_TYPE_JSON`

**AMQP Transport Module** — `amqp` module

- `to_amqp_payload` / `from_amqp_payload`: CBOR/JSON serialization
- `derive_amqp_properties`: content_type, routing_key, priority (0-9), delivery_mode, expiration, message_id, timestamp, headers
- `derive_routing_key`: dot-separated `prefix.type.id_fragment` for topic exchanges
- `derive_amqp_priority`: linear map `@confidence` → 0-9
- `derive_amqp_headers`: `x-jsonld-*` headers for header exchange routing
- Delivery mode: persistent for high-confidence/verified, transient for low-confidence

**Kafka Transport Module** — `kafka` module

- `to_kafka_value` / `from_kafka_value`: CBOR/JSON serialization
- `derive_kafka_record`: full producer record (topic, key, value, headers, timestamp)
- `derive_kafka_topic`: `prefix.type_local` from `@type`
- `derive_kafka_key`: `@id` as bytes for partition assignment
- `derive_kafka_headers`: list of `(str, bytes)` tuples for consumer filtering
- `derive_kafka_timestamp`: epoch milliseconds from `@extractedAt`

**WebSocket Transport Module** — `websocket` module

- `to_ws_message` / `from_ws_message`: str (text frame) for JSON, bytes (binary frame) for CBOR
- `derive_ws_subprotocols`: `jsonld-ex.cbor` / `jsonld-ex.jsonld` for handshake negotiation
- `derive_ws_metadata`: per-message metadata dict (opcode, content_type, jsonld_type/id/confidence/source, ttl_seconds)

**gRPC Transport Module** — `grpc` module

- `derive_grpc_metadata`: list of `(str, str)` tuples for gRPC initial/trailing metadata
- `to_grpc_json` / `from_grpc_json`: compact JSON for gRPC transcoding (grpc-gateway)
- `suggest_proto_schema`: heuristic `.proto` file generation from JSON-LD document structure

**LwM2M Transport Module** — `lwm2m` module

- `derive_lwm2m_objects`: map `@type` to IPSO Smart Object IDs (8 sensor types registered)
- `extract_lwm2m_resources`: property values to numbered resources (5700=value, 5701=units, etc.)
- `derive_lwm2m_registration`: endpoint from `@id`, lifetime from `@validUntil`
- `derive_lwm2m_links`: RFC 6690 CoRE Link Format for discovery
- `IPSO_OBJECT_REGISTRY`: TemperatureSensor, HumiditySensor, Barometer, Accelerometer, Illuminance, DigitalOutput, AnalogInput, GenericSensor

**Shared Transport Helpers** — `_transport_common` internal module

- Extracted shared logic from `mqtt.py`, `coap.py`, `http_headers.py`
- `local_name`, `sanitise_segment`, `extract_type_local`, `extract_id_fragment`
- `find_valid_until`, `seconds_remaining`, `derive_expiry_seconds`
- `scan_confidence`: unified confidence/humanVerified scanning

**EN8.5 Experiment** — CBOR-LD compact transport with TurboQuant enhancement

- 5 document variants × 6 serialization formats
- Payload bytes, compression ratio, throughput, round-trip fidelity
- Quantization byte savings analysis (128-dim and 768-dim)

### Changed
- `mqtt.py` refactored to use `_transport_common` shared helpers (backward-compatible aliases preserved)

### Fixed
- CoAP `CONTENT_FORMAT_JSONLD` corrected from 11050 (collides with `application/json` deflate) to 11100 (unassigned range)

## [0.7.2] - 2026-04-14

### Added

**Validation Framework: 3 New Constraint Types (GAP-V8, V9, V10)**

- `@class` — Instance-of check (maps to `sh:class`): constrains a property value to be a node whose `@type` includes the specified class IRI. Useful for ML schemas that require typed nested nodes (e.g., "author must be a Person").
- `@qualifiedShape` / `@qualifiedMinCount` / `@qualifiedMaxCount` — Qualified cardinality (maps to `sh:qualifiedValueShape`): constrains how many items in a list must conform to a given shape. Enables ML quality gates like "at least 2 annotations must have confidence > 0.9".
- `@uniqueLang` — Unique language tags (maps to `sh:uniqueLang`): ensures no two values in a list share the same `@language` tag (case-insensitive per RDF semantics).

**SHACL Interop for New Constraints**

- `shape_to_shacl()` now emits `sh:class`, `sh:qualifiedValueShape`/`sh:qualifiedMinCount`/`sh:qualifiedMaxCount`, and `sh:uniqueLang`
- `shacl_to_shape()` now round-trips all three back to `@class`, `@qualifiedShape`, `@uniqueLang`
- Removed `sh:class`, `sh:qualifiedValueShape`, and `sh:uniqueLang` from the "unsupported SHACL features" warning list
- Only `sh:sparql`, `sh:node`, and `sh:hasValue` remain as genuinely unsupported SHACL features

## [0.7.0] — 2026-03-03

### Added

**Enhanced Byzantine-Resistant Fusion** — `confidence_byzantine` module

- `ByzantineConfig` frozen dataclass: threshold, max_removals, strategy, trust_weights, min_agents
- `AgentRemoval` frozen dataclass: records index, opinion, discord_score, and human-readable reason for each removal
- `ByzantineFusionReport` frozen dataclass: fused opinion, removal list, full conflict matrix, cohesion score, surviving indices
- Three removal strategies via `ByzantineStrategy` literal type:
  - `"most_conflicting"` — remove agent with highest mean pairwise discord (default)
  - `"least_trusted"` — remove lowest-trust agent (requires trust_weights)
  - `"combined"` — remove by discord × (1 − trust), prioritizing untrusted rogues
- `byzantine_fuse(opinions, config)` → `ByzantineFusionReport`: configurable Byzantine filtering with rich reporting
- `build_conflict_matrix(opinions)` → symmetric n×n pairwise conflict matrix
- `cohesion_score(opinions, distance_fn=None)` → scalar group agreement metric using pluggable distance (default: Euclidean)
- `opinion_distance(a, b, distance_fn=None)` → configurable distance dispatcher with Euclidean default
- Pluggable distance metrics on the opinion simplex (`DistanceMetric` type alias):
  - `euclidean_opinion_distance` — L2/sqrt(2), uniform sensitivity, simple default
  - `manhattan_opinion_distance` — L1/2, more robust to single-dimension outliers
  - `jsd_opinion_distance` — sqrt(Jensen-Shannon divergence), information-theoretic, boundary-sensitive
  - `hellinger_opinion_distance` — Hellinger distance, boundary-sensitive, numerically stable
- All four metrics are proper metrics (non-negativity, identity, symmetry, triangle inequality) normalized to [0, 1]
- Cohesion uses opinion distance (not Josang's pairwise_conflict) so identical opinions always yield 1.0
- Backward compatible: original `robust_fuse` in `confidence_algebra` is unchanged

**Temporal Fusion** — `confidence_temporal_fusion` module

- `TimestampedOpinion` frozen dataclass: wraps Opinion + datetime + optional source_id
- `TemporalFusionConfig` frozen dataclass: half_life, decay_fn, fusion_method, reference_time
- `TemporalFusionReport` frozen dataclass: fused opinion, list of decayed opinions, reference time
- `temporal_fuse(opinions, config)` → decay all opinions by age then fuse (cumulative or averaging)
- `temporal_fuse_weighted(opinions, half_life_map, ...)` → per-source half-lives (e.g., academic sources decay slower than social media)
- `temporal_byzantine_fuse(opinions, temporal_config, byzantine_config)` → full pipeline: decay → Byzantine filter → fuse
- Composes `confidence_decay`, `confidence_algebra`, and `confidence_byzantine` without modifying any existing module

### Changed
- Version bumped to 0.7.0

## [0.6.5] — 2026-02-15

### Added

**Metric Selection Advisory System** — Four-phase feature for intelligent metric selection

- `MetricProperties` frozen dataclass: 11 fields capturing mathematical facts about metrics (kind, range, boundedness, metric space, symmetry, normalization sensitivity, zero-vector behavior, computational complexity, best-for domains)
- Pre-defined properties for all 7 built-in metrics and 10 example metrics, each mathematically defensible with documented justifications
- `get_metric_properties()` and `get_all_metric_properties()` registry API
- `register_similarity_metric()` now accepts optional `properties` kwarg with name-match validation
- `compare_metrics(a, b)`: compute all (or selected) registered metrics on a single vector pair with structured output (score, kind, error per metric)
- `VectorProperties` frozen dataclass: deterministic statistical properties of vector samples (binary detection, sparsity, unit normalization, non-negativity, magnitude coefficient of variation)
- `analyze_vectors(vectors)`: single-pass data property detection with input validation
- `HeuristicRecommender`: rule-based recommendation engine with 6 rules grounded in mathematical properties, documented thresholds, and academic citations (Aggarwal et al., 2001)
- `recommend_metric(vectors, engine=...)`: pluggable recommendation with `RecommendationEngine` protocol for custom engines (ML-based, domain-specific, LLM-backed)
- `evaluate_metrics(labeled_pairs)`: empirical evaluation on labeled vector pairs with three mathematically rigorous measures:
  - Exact ROC-AUC via Mann–Whitney U statistic (not trapezoidal approximation)
  - Spearman rank correlation with average-rank tie handling
  - Direction-corrected mean separation
- Three-tier direction resolution for custom metrics: explicit override → MetricProperties.kind → default with warning
- All measures direction-corrected: positive always means better separation regardless of distance vs similarity
- Undefined measures (single class, constant scores) return `None`, never fabricated values
- 141 new tests across 13 test classes

### Changed
- Version bumped to 0.6.5

### Gap Analysis Status
**All 28 gaps from the competitive gap analysis are now closed:**
- CRITICAL (4/4): GAP-V1 (@minCount/@maxCount), GAP-V2 (@in/@enum), GAP-D1 (dataset metadata), GAP-D3 (Croissant interop)
- HIGH (8/8): GAP-V3 (logical combinators), GAP-V4 (cross-property), GAP-MM1 (multimodal), GAP-ML1 (language confidence), GAP-ML2 (translation provenance), GAP-IOT1 (measurement uncertainty), GAP-D2 (splits/distributions), GAP-API1 (batch API)
- MEDIUM (13/13): GAP-V5 (nested shapes), GAP-V6 (severity), GAP-V7 (conditional constraints), GAP-MM2 (multi-embedding), GAP-MM3 (content addressing), GAP-IOT2 (calibration metadata), GAP-IOT3 (SSN/SOSA interop), GAP-IOT4 (aggregation metadata), GAP-P1 (delegation chains), GAP-P2 (@derivedFrom), GAP-P3 (invalidation/retraction), GAP-OWL1 (shape inheritance), GAP-CTX1 (context versioning)

### Migration from 0.6.0
No breaking changes. All new functions are additive exports. `register_similarity_metric()` gains an optional `properties` keyword argument with no effect on existing calls.

## [0.6.0] — 2026-02-14

### Added

**Similarity Metric Registry** (`similarity`) — Extensible metric system for vector embeddings
- 7 built-in metrics: cosine, euclidean, dot_product, manhattan, chebyshev, hamming, jaccard
- Full registry API: `register_similarity_metric()`, `get_similarity_metric()`, `list_similarity_metrics()`, `unregister_similarity_metric()`
- `similarity(a, b)` dispatcher resolving metric from explicit name, `@similarity` in term definition, or cosine default
- Shared input validation via `_validate_vector_pair()` for DRY consistency
- Built-in protection: cannot overwrite built-ins without `force=True`
- `BUILTIN_METRIC_NAMES` frozenset for programmatic access

**Example Metrics** (`similarity_examples`) — 10 domain-specific metric recipes
- Ecology/composition: Canberra distance, Bray-Curtis dissimilarity
- Geographic: Haversine distance
- Time series: Dynamic Time Warping (DTW)
- Correlated features: Mahalanobis distance
- Semantic text: Soft Cosine similarity
- Distributions: KL divergence, Wasserstein (Earth Mover's) distance
- Ordinal/rank: Spearman correlation, Kendall Tau correlation
- All with graceful skip when numpy/scipy not available

**Vector Term Definition Enhancement**
- `vector_term_definition()` gains keyword-only `similarity` parameter for `@similarity` metadata

### Changed
- Version bumped to 0.6.0

### Migration from 0.5.0
No breaking changes. `vector_term_definition()` new parameter is keyword-only with no default change.

## [0.5.0] — 2026-02-14

### Added

**Compliance Algebra** (`compliance_algebra`) — Regulatory uncertainty modeling with Subjective Logic
- New `ComplianceOpinion` class extending `Opinion` with domain semantics (lawfulness, violation, uncertainty, base_rate)
- Jurisdictional Meet (`jurisdictional_meet`): conjunction of compliance across jurisdictions (GDPR Art. 44–49). Binary and n-ary. Proven: constraint, non-negativity, monotonicity, commutativity, associativity, identity, annihilator.
- Compliance Propagation (`compliance_propagation`): uncertainty propagation through data derivation chains (Art. 5, 6, 25). Multiplicative lawfulness decay.
- Provenance Chain (`ProvenanceChain`): ordered audit trail with iterative computation (Art. 30, Art. 5(2)).
- Consent Assessment (`consent_validity`): six-condition GDPR Art. 7 composition via keyword or positional arguments.
- Withdrawal Override (`withdrawal_override`): novel proposition-replacement operator at consent withdrawal (Art. 7(3)).
- Expiry Trigger (`expiry_trigger`): asymmetric l→v transition modeling hard/soft deadline expiry (Art. 5(1)(e)).
- Review-Due Trigger (`review_due_trigger`): accelerated decay toward vacuity for missed reviews (Art. 45(3), 35(11)).
- Regulatory Change Trigger (`regulatory_change_trigger`): proposition replacement for discrete legal events.
- Erasure Scope (`erasure_scope_opinion`): composite erasure completeness across data lineage graphs (Art. 17).
- Residual Contamination (`residual_contamination`): disjunctive contamination risk at individual nodes (Art. 17).
- 103 new tests covering all operators, theorem properties, edge cases, and operator interactions.
- Mathematical formalization: compliance_algebra.md (all definitions, theorems, proofs).

### Migration from 0.4.0
No breaking changes. The `compliance_algebra` module is entirely additive. No existing modules were modified.

## [0.4.0] — 2026-02-13

### Added

**Data Protection & Privacy Compliance** (`data_protection`) — Phase 1
- New `annotate_protection()` function with 10 annotation fields mapping to W3C DPV v2.2 concepts
- Personal data classification: `@personalDataCategory` (regular, sensitive, special_category, anonymized, pseudonymized, synthetic, non_personal)
- Legal basis tracking: `@legalBasis` (consent, contract, legal_obligation, vital_interest, public_task, legitimate_interest — GDPR Art. 6)
- Processing metadata: `@processingPurpose`, `@dataController`, `@dataProcessor`, `@dataSubject`
- Retention management: `@retentionUntil` (semantically distinct from `@validUntil`)
- Jurisdiction and access control: `@jurisdiction`, `@accessLevel`
- Consent lifecycle: `create_consent_record()`, `is_consent_active()` with time-aware status checking
- GDPR-correct classification helpers: `is_personal_data()`, `is_sensitive_data()`
- Graph filtering: `filter_by_jurisdiction()`, `filter_personal_data()`
- Composes with existing `ai_ml.annotate()` via dict merge — both produce compatible `@value` dicts
- 54 new tests

### Changed
- Version bumped to 0.4.0

### Migration from 0.3.x
No breaking changes. The `data_protection` module is entirely additive. No existing modules were modified. All new parameters use keyword-only arguments to prevent accidental positional usage.

## [0.3.5] — 2026-02-12

### Added

**IoT Sensor Metadata**
- Aggregation metadata: `@aggregationMethod`, `@aggregationWindow`, `@aggregationCount`
- Calibration metadata: `@calibratedAt`, `@calibrationMethod`, `@calibrationAuthority`

**Provenance Extensions**
- Delegation chains: `@delegatedBy` with PROV-O `prov:actedOnBehalfOf` bidirectional mapping + RDF-star
- Invalidation/retraction: `@invalidatedAt`, `@invalidationReason` with PROV-O `prov:wasInvalidatedBy` bidirectional mapping + RDF-star
- `filter_by_confidence(exclude_invalidated=True)` parameter for filtering invalidated assertions

**Test Coverage**
- Multi-embedding document tests (GAP-MM2)
- Content addressing `@contentHash` tests (GAP-MM3)

### Migration from 0.3.0
No breaking changes. All new parameters have backward-compatible defaults. `filter_by_confidence()` defaults to `exclude_invalidated=False` preserving existing behavior.

## [0.3.0] — 2026-02-11

### Added

**MCP Server Integration** (`jsonld_ex.mcp`) — *requires `mcp>=1.7`*
- Model Context Protocol server exposing jsonld-ex as 16 MCP tools
- 6 tool groups: AI/ML annotation, confidence algebra, security, vectors, graph ops, interop
- 3 MCP resources: AI/ML context, security context, opinion JSON Schema
- 2 MCP prompts: annotate_tool_results, trust_chain_analysis
- Entry point: `python -m jsonld_ex.mcp` (stdio/HTTP transport)
- Install via `pip install jsonld-ex[mcp]`
- All tools are read-only and stateless

### Changed
- Added `mcp` optional dependency group in pyproject.toml
- Version bumped to 0.3.0

### Migration from 0.2.x
No breaking changes. The MCP module is entirely optional and additive. Existing public API is fully preserved. Users who do not install the `mcp` extra are completely unaffected.

## [0.2.0] — 2026-02-06

### Added

**OWL/RDF Interoperability** (`owl_interop`)
- Bidirectional PROV-O conversion (`to_prov_o`, `from_prov_o`)
- Bidirectional SHACL mapping (`shape_to_shacl`, `shacl_to_shape`)
- OWL class restriction generation (`shape_to_owl_restrictions`)
- RDF-star N-Triples export (`to_rdf_star_ntriples`)
- Verbosity comparison utilities (`compare_with_prov_o`, `compare_with_shacl`)

**Confidence Propagation** (`inference`)
- Chain propagation: multiply, bayesian, min, dampened methods
- Multi-source combination: average, max, noisy-OR, Dempster–Shafer
- Conflict resolution: highest, weighted_vote, recency strategies
- Graph-level propagation along property chains

**Graph Merging** (`merge`)
- Confidence-aware merge of multiple JSON-LD graphs
- Conflict strategies: highest, weighted_vote, recency, union
- Semantic diff between two graphs (`diff_graphs`)
- Full audit trail via `MergeReport`

**Temporal Extensions** (`temporal`)
- `@validFrom`, `@validUntil`, `@asOf` annotation helpers
- Point-in-time graph queries (`query_at_time`)
- Temporal diff between two timestamps (`temporal_diff`)

**CBOR-LD Serialization** (`cbor_ld`) — *requires `cbor2`*
- Binary serialization with context compression (`to_cbor`, `from_cbor`)
- Payload size comparison (`payload_stats`)

**MQTT Transport** (`mqtt`) — *requires `cbor2`*
- MQTT payload serialization (`to_mqtt_payload`, `from_mqtt_payload`)
- Topic derivation from `@type`/`@id` (`derive_mqtt_topic`)
- QoS mapping from `@confidence` (`derive_mqtt_qos`)

### Changed
- Version bumped to 0.2.0
- Added `iot` and `mqtt` optional dependency groups in pyproject.toml
- Added `pytest-benchmark` to dev dependencies

### Migration from 0.1.x
No breaking changes. All existing public API is preserved. New modules are purely additive. CBOR-LD and MQTT modules are optional — they gracefully skip if `cbor2` is not installed.

## [0.1.3] — 2026-01-20

### Added
- Core AI/ML extensions: `@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`
- Vector extensions: `@vector` container, cosine similarity, dimension validation
- Security extensions: `@integrity` context verification, context allowlists, resource limits
- Validation extensions: `@shape` native validation framework
- `JsonLdEx` processor wrapping PyLD
