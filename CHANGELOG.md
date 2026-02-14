# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

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
