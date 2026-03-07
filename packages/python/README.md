# jsonld-ex

[![PyPI version](https://img.shields.io/pypi/v/jsonld-ex)](https://pypi.org/project/jsonld-ex/)
[![Python](https://img.shields.io/pypi/pyversions/jsonld-ex)](https://pypi.org/project/jsonld-ex/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-4100%2B%20passing-brightgreen)]()

**JSON-LD 1.2 Extensions for AI/ML Data Exchange, Security, and Validation**

Reference implementation of proposed JSON-LD 1.2 extensions that address critical gaps in the current specification for machine learning workflows. Wraps [PyLD](https://github.com/digitalbazaar/pyld) for core JSON-LD processing and adds extension layers for confidence tracking (Subjective Logic), provenance, security hardening, native validation, vector embeddings, temporal modeling, graph-structured inference (SLNetwork), trust propagation, FHIR R4 clinical interoperability, and standards interoperability.

## Installation

```bash
# Core library
pip install jsonld-ex

# With FHIR R4 interoperability
pip install jsonld-ex[fhir]

# With graph visualization (NetworkX + DOT/Graphviz)
pip install jsonld-ex[viz]

# With MCP server (requires Python 3.10+)
pip install jsonld-ex[mcp]

# With IoT/CBOR-LD support
pip install jsonld-ex[iot]

# With MQTT transport
pip install jsonld-ex[mqtt]

# Everything for development
pip install jsonld-ex[dev]
```

## Quick Start

```python
from jsonld_ex import annotate, validate_node, Opinion, cumulative_fuse

# Annotate a value with AI/ML provenance
name = annotate(
    "John Smith",
    confidence=0.95,
    source="https://ml-model.example.org/ner-v2",
    method="NER",
)
# {'@value': 'John Smith', '@confidence': 0.95, '@source': '...', '@method': 'NER'}

# Validate against a shape
shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string"},
    "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
}
result = validate_node({"@type": "Person", "name": "John", "age": 30}, shape)
assert result.valid

# Formal confidence algebra (Subjective Logic)
sensor_a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
sensor_b = Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25)
fused = cumulative_fuse(sensor_a, sensor_b)
print(f"Fused: b={fused.belief:.3f}, u={fused.uncertainty:.3f}")
```

## Features

jsonld-ex provides **28 modules** organized into eight extension categories:

| Category | Modules | Purpose |
|----------|---------|---------|
| **AI/ML Data Modeling** | ai_ml, confidence_algebra, confidence_bridge, confidence_decay, confidence_byzantine, confidence_temporal_fusion, inference, vector, similarity | Subjective Logic opinions, provenance, temporal decay, Byzantine-resistant fusion, temporal fusion, embeddings, 17 similarity metrics, advisory system |
| **Graph Inference** | sl_network (types, network, inference, trust, counterfactuals) | DAG-structured Subjective Logic inference engine with trust propagation, attestation discounting, and counterfactual strategies |
| **Compliance & Privacy** | compliance_algebra, compliance_adapters, data_protection, data_rights, dpv_interop | GDPR regulatory algebra (5 operators, formal proofs), consent lifecycle, data subject rights (Art. 15-20), W3C DPV v2.2 |
| **Security & Validation** | security, validation | Context integrity, allowlists, resource limits, native shapes with `@if`/`@then`, `@extends` |
| **Interoperability** | owl_interop, dataset, context, fhir_interop | PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant, FHIR R4 (32 resource types), context versioning |
| **Graph & Temporal** | merge, temporal, batch | Graph merge/diff, time-aware queries, batch operations |
| **Transport** | cbor_ld, mqtt, processor | CBOR-LD binary serialization, MQTT topic/QoS derivation, core processing |
| **Visualization** | visualization (dot, nx_export) | Zero-dependency DOT/Graphviz generation, NetworkX graph export |

---

## AI/ML Annotations

Tag any extracted value with provenance metadata using the `@confidence`, `@source`, `@extractedAt`, `@method`, and `@humanVerified` extension keywords.

```python
from jsonld_ex import annotate, get_confidence, get_provenance, filter_by_confidence

# Annotate with full provenance chain
value = annotate(
    "San Francisco",
    confidence=0.92,
    source="https://model.example.org/geo-v3",
    extracted_at="2025-06-01T12:00:00Z",
    method="geocoding",
    human_verified=False,
)

# Extract confidence from an annotated node
score = get_confidence(value)  # 0.92

# Extract all provenance metadata
prov = get_provenance(value)
# ProvenanceMetadata(confidence=0.92, source='...', extracted_at='...', method='geocoding', ...)

# Filter a graph's nodes by minimum confidence on a property
graph = [{"@type": "Person", "name": value}]
filtered = filter_by_confidence(graph, "name", min_confidence=0.8)
```

## Confidence Algebra (Subjective Logic)

A complete implementation of Josang's Subjective Logic framework, where an **opinion** w = (b, d, u, a) distinguishes evidence-for, evidence-against, and absence-of-evidence, unlike a scalar confidence score.

All algebraic properties are validated by property-based tests (Hypothesis) with thousands of random inputs per property.

```python
from jsonld_ex import (
    Opinion, cumulative_fuse, averaging_fuse, trust_discount,
    deduce, pairwise_conflict, conflict_metric, robust_fuse,
)

# Create opinions
sensor = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
model  = Opinion(belief=0.6, disbelief=0.0, uncertainty=0.4)

# Cumulative fusion -- independent sources, reduces uncertainty
fused = cumulative_fuse(sensor, model)

# Averaging fusion -- correlated/dependent sources
avg = averaging_fuse(sensor, model)

# Trust discounting -- propagate through a trust chain
trust_in_sensor = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
discounted = trust_discount(trust_in_sensor, sensor)

# Deduction -- Subjective Logic modus ponens (Josang Def. 12.6)
opinion_x = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
y_given_x = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
y_given_not_x = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)
opinion_y = deduce(opinion_x, y_given_x, y_given_not_x)

# Conflict detection
conflict = pairwise_conflict(sensor, model)  # con(A,B) = b_A*d_B + d_A*b_B
internal = conflict_metric(sensor)  # 1 - |b-d| - u

# Byzantine-resistant fusion -- removes outliers before fusing
agents = [sensor, model, Opinion(belief=0.01, disbelief=0.98, uncertainty=0.01)]
fused, removed = robust_fuse(agents)
```

### Operators Summary

| Operator | Use Case | Associative | Commutative |
|----------|----------|:-----------:|:-----------:|
| `cumulative_fuse` | Independent sources | Yes | Yes |
| `averaging_fuse` | Correlated sources | No | Yes |
| `robust_fuse` | Adversarial environments | -- | Yes |
| `trust_discount` | Trust chain propagation | Yes | No |
| `deduce` | Conditional reasoning | -- | -- |
| `pairwise_conflict` | Source disagreement | -- | Yes |
| `conflict_metric` | Internal conflict | -- | -- |

## Byzantine-Resistant Fusion

Advanced Byzantine fault tolerance for multi-agent opinion fusion, extending the basic `robust_fuse` with configurable detection strategies and distance metrics.

```python
from jsonld_ex import (
    ByzantineStrategy, ByzantineConfig, byzantine_fuse,
    build_conflict_matrix, cohesion_score,
    euclidean_opinion_distance, jsd_opinion_distance,
)

# Configure Byzantine detection
config = ByzantineConfig(
    strategy=ByzantineStrategy.MEDIAN_DISTANCE,
    threshold=2.0,  # standard deviations
)

# Fuse with Byzantine detection
opinions = [
    Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
    Opinion(belief=0.75, disbelief=0.15, uncertainty=0.1),
    Opinion(belief=0.01, disbelief=0.98, uncertainty=0.01),  # adversarial
]
report = byzantine_fuse(opinions, config)
# report.fused_opinion, report.removed (indices), report.strategy_used

# Build a pairwise conflict matrix
matrix = build_conflict_matrix(opinions)

# Measure group cohesion (0 = perfect agreement, 1 = total disagreement)
score = cohesion_score(opinions)
```

### Distance Metrics

| Metric | Function | Properties |
|--------|----------|------------|
| Euclidean | `euclidean_opinion_distance` | L2 norm in (b,d,u) space |
| Manhattan | `manhattan_opinion_distance` | L1 norm in (b,d,u) space |
| Jensen-Shannon | `jsd_opinion_distance` | Information-theoretic, bounded [0,1] |
| Hellinger | `hellinger_opinion_distance` | Bounded [0,1], sqrt of divergence |

## Temporal Fusion

Timestamped opinion fusion that applies temporal decay before combining evidence, giving more weight to recent observations.

```python
from jsonld_ex import (
    TimestampedOpinion, TemporalFusionConfig, TemporalFusionReport,
    temporal_fuse, temporal_fuse_weighted, temporal_byzantine_fuse,
)
from datetime import datetime, timezone

# Create timestamped opinions
now = datetime.now(timezone.utc)
opinions = [
    TimestampedOpinion(
        opinion=Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
        timestamp=now,
    ),
    TimestampedOpinion(
        opinion=Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),  # older
    ),
]

# Temporal fusion with configurable half-life
config = TemporalFusionConfig(half_life_seconds=86400 * 30)  # 30-day half-life
report = temporal_fuse(opinions, config, reference_time=now)
# report.fused_opinion -- recent evidence weighted more heavily

# Temporal fusion with Byzantine detection
report = temporal_byzantine_fuse(opinions, config, reference_time=now)
```

## Confidence Bridge

Bridge legacy scalar confidence scores to the formal Subjective Logic algebra and back.

```python
from jsonld_ex import combine_opinions_from_scalars, propagate_opinions_from_scalars

# Combine scalar scores via formal algebra
fused = combine_opinions_from_scalars(
    [0.9, 0.85, 0.7],
    fusion="cumulative",  # or "averaging"
)

# Propagate through a trust chain from scalars
# With defaults (uncertainty=0, base_rate=0), produces the exact
# same result as scalar multiplication, proving equivalence.
propagated = propagate_opinions_from_scalars([0.9, 0.8, 0.95])
```

## Temporal Decay

Model evidence aging: as time passes, belief and disbelief migrate toward uncertainty, reflecting that old evidence is less reliable.

```python
from jsonld_ex import Opinion, decay_opinion, exponential_decay, linear_decay, step_decay

opinion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

# Exponential decay (default) -- smooth half-life model
decayed = decay_opinion(opinion, elapsed_seconds=3600, half_life_seconds=7200)

# Linear decay -- constant decay rate
decayed = decay_opinion(opinion, elapsed_seconds=3600, half_life_seconds=7200,
                        decay_fn=linear_decay)

# Step decay -- binary cutoff at half-life
decayed = decay_opinion(opinion, elapsed_seconds=3600, half_life_seconds=7200,
                        decay_fn=step_decay)
```

## Inference Engine

Propagate confidence through multi-hop inference chains and combine evidence from multiple sources with auditable conflict resolution.

```python
from jsonld_ex import (
    propagate_confidence, combine_sources,
    resolve_conflict, propagate_graph_confidence,
)

# Chain propagation -- confidence across inference steps
result = propagate_confidence([0.95, 0.90, 0.85], method="multiply")
# Also: "bayesian", "min", "dampened" (product^(1/sqrt(n)))

# Source combination -- multiple sources assert the same fact
result = combine_sources([0.8, 0.75, 0.9], method="noisy_or")
# Also: "average", "max", "dempster_shafer"

# Conflict resolution -- pick a winner among disagreeing sources
assertions = [
    {"@value": "New York", "@confidence": 0.9},
    {"@value": "NYC", "@confidence": 0.85},
    {"@value": "Boston", "@confidence": 0.3},
]
report = resolve_conflict(assertions, strategy="weighted_vote")
# Also: "highest", "recency"

# Graph propagation -- trace confidence along a property chain
result = propagate_graph_confidence(document, ["author", "affiliation", "country"])
```

## Security

Protect against context injection, DNS poisoning, and resource exhaustion attacks.

```python
from jsonld_ex import compute_integrity, verify_integrity, is_context_allowed

# Compute SRI-style integrity hash for a context
integrity = compute_integrity('{"@context": {"name": "http://schema.org/name"}}')
# "sha256-abc123..."

# Verify context hasn't been tampered with
is_valid = verify_integrity(context_json, integrity)

# Check context URL against an allowlist
allowed = is_context_allowed("https://schema.org/", {
    "allowed": ["https://schema.org/"],
    "patterns": ["https://w3id.org/*"],
    "block_remote_contexts": False,
})
```

### Resource Limits

Enforce configurable resource limits to prevent denial-of-service:

| Limit | Default | Description |
|-------|---------|-------------|
| `max_context_depth` | 10 | Maximum nested context chain |
| `max_graph_depth` | 100 | Maximum @graph nesting |
| `max_document_size` | 10 MB | Maximum input size |
| `max_expansion_time` | 30 s | Processing timeout |

## Validation

Native `@shape` validation framework that maps bidirectionally to SHACL, requiring no external tools.

```python
from jsonld_ex import validate_node

shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
    "email": {"@pattern": "^[^@]+@[^@]+$"},
    "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
}

result = validate_node(
    {"@type": "Person", "name": "Alice", "age": 200},
    shape,
)
# result.valid == False
# result.errors[0].message -> age exceeds @maximum
```

Supported constraints: `@required`, `@type`, `@minimum`, `@maximum`, `@minLength`, `@maxLength`, `@pattern`, `@minCount`, `@maxCount`, `@in`/`@enum`, `@and`/`@or`/`@not` (logical combinators), `@if`/`@then`/`@else` (conditional), `@extends` (shape inheritance), nested shapes, and configurable severity levels.

## Data Protection and Privacy Compliance

GDPR/privacy compliance metadata for ML data pipelines. Annotations map to [W3C Data Privacy Vocabulary (DPV) v2.2](https://w3id.org/dpv) concepts. Composes with `annotate()`: both produce compatible `@value` dicts that can be merged.

```python
from jsonld_ex import (
    annotate_protection, get_protection_metadata,
    create_consent_record, is_consent_active,
    is_personal_data, is_sensitive_data,
    filter_personal_data, filter_by_jurisdiction,
)

# Annotate a value with data protection metadata
name = annotate_protection(
    "John Doe",
    personal_data_category="regular",      # regular, sensitive, special_category,
                                            # anonymized, pseudonymized, synthetic, non_personal
    legal_basis="consent",                  # Maps to GDPR Art. 6
    processing_purpose="Healthcare provision",
    data_controller="https://hospital.example.org",
    retention_until="2030-12-31T23:59:59Z", # Mandatory deletion deadline
    jurisdiction="EU",
    access_level="confidential",
)

# Compose with AI/ML provenance (both produce @value dicts)
from jsonld_ex import annotate
provenance = annotate("John Doe", confidence=0.95, source="ner-model-v2")
protection = annotate_protection("John Doe", personal_data_category="regular", legal_basis="consent")
merged = {**provenance, **protection}  # All fields coexist

# Consent lifecycle tracking
consent = create_consent_record(
    given_at="2025-01-15T10:00:00Z",
    scope=["Marketing", "Analytics"],
    granularity="specific",
)
is_consent_active(consent)  # True
is_consent_active(consent, at_time="2024-12-01T00:00:00Z")  # False (before given)

# GDPR-correct classification
is_personal_data({"@value": "John", "@personalDataCategory": "pseudonymized"})  # True
is_personal_data({"@value": "stats", "@personalDataCategory": "anonymized"})    # False
is_sensitive_data({"@value": "diagnosis", "@personalDataCategory": "sensitive"}) # True

# Filter graphs for personal data or by jurisdiction
personal_nodes = filter_personal_data(graph)
eu_nodes = filter_by_jurisdiction(graph, "name", "EU")
```

## Vector Embeddings

`@vector` container type for storing vector embeddings alongside symbolic data, with dimension validation and similarity computation.

```python
from jsonld_ex import validate_vector, cosine_similarity, vector_term_definition

# Validate an embedding node
valid, errors = validate_vector([0.1, -0.2, 0.3], expected_dimensions=3)

# Cosine similarity between embeddings
sim = cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])  # 0.0

# Generate a JSON-LD term definition for a vector property
term_def = vector_term_definition("embedding", "http://example.org/embedding", dimensions=768)
```

> **Note:** `cosine_similarity` raises `ValueError` on zero-magnitude vectors. Cosine similarity is mathematically undefined (0/0) for the zero vector, and silently returning 0.0 would mask an error.

## Graph Operations

Merge and diff JSON-LD graphs with confidence-aware conflict resolution.

```python
from jsonld_ex import merge_graphs, diff_graphs

# Merge two graphs -- boosts confidence where sources agree (noisy-OR)
merged, report = merge_graphs(
    [graph_a, graph_b],
    conflict_strategy="highest",  # or "weighted_vote", "union", "recency"
)
# report.conflicts -> list of resolved conflicts with audit trail

# Semantic diff between two graphs
diff = diff_graphs(graph_a, graph_b)
# diff keys: "added", "removed", "modified", "unchanged"
```

## Temporal Extensions

Temporal annotations for time-varying data with point-in-time queries and temporal differencing.

```python
from jsonld_ex import add_temporal, query_at_time, temporal_diff

# Add temporal bounds to a value
value = add_temporal("Engineer", valid_from="2020-01-01", valid_until="2024-12-31")
# {'@value': 'Engineer', '@validFrom': '2020-01-01', '@validUntil': '2024-12-31'}

# Query the graph state at a point in time
snapshot = query_at_time(nodes, "2022-06-15")

# Compute what changed between two timestamps
diff = temporal_diff(nodes, t1="2020-01-01", t2="2024-01-01")
# TemporalDiffResult with .added, .removed, .modified, .unchanged
```

## Standards Interoperability

Bidirectional conversion between jsonld-ex extensions and established W3C standards, with verbosity comparison metrics.

### PROV-O (W3C Provenance Ontology)

```python
from jsonld_ex import to_prov_o, from_prov_o, compare_with_prov_o

# Convert to PROV-O
prov_doc, report = to_prov_o(annotated_doc)

# Convert back -- full round-trip
recovered, report = from_prov_o(prov_doc)

# Measure verbosity reduction vs PROV-O
comparison = compare_with_prov_o(annotated_doc)
# comparison.triple_reduction_pct -> e.g. 60% fewer triples
```

### SHACL (Shapes Constraint Language)

```python
from jsonld_ex import shape_to_shacl, shacl_to_shape, compare_with_shacl

# Convert @shape -> SHACL
shacl_doc = shape_to_shacl(shape, target_class="http://schema.org/Person")

# Convert SHACL -> @shape -- full round-trip
recovered_shape, warnings = shacl_to_shape(shacl_doc)

# Measure verbosity reduction vs SHACL
comparison = compare_with_shacl(shape)
```

### OWL and RDF-Star

```python
from jsonld_ex import shape_to_owl_restrictions, to_rdf_star_ntriples

# Convert @shape -> OWL class restrictions
owl_doc = shape_to_owl_restrictions(shape, class_iri="http://example.org/Person")

# Export annotations as RDF-Star N-Triples
ntriples, report = to_rdf_star_ntriples(annotated_doc)
```

## CBOR-LD Serialization

Binary serialization for bandwidth-constrained environments (requires `pip install jsonld-ex[iot]`).

```python
from jsonld_ex import to_cbor, from_cbor, payload_stats

# Serialize to CBOR
cbor_bytes = to_cbor(document)

# Deserialize
document = from_cbor(cbor_bytes)

# Compression statistics
stats = payload_stats(document)
# PayloadStats with .json_bytes, .cbor_bytes, .compression_ratio
```

## MQTT Transport

IoT transport optimization with confidence-aware QoS mapping (requires `pip install jsonld-ex[mqtt]`).

```python
from jsonld_ex import (
    to_mqtt_payload, from_mqtt_payload,
    derive_mqtt_topic, derive_mqtt_qos_detailed,
)

# Encode for MQTT transmission (CBOR-compressed by default)
payload = to_mqtt_payload(document, compress=True, max_payload=256_000)

# Decode back
document = from_mqtt_payload(payload, compressed=True)

# Derive hierarchical MQTT topic from document metadata
topic = derive_mqtt_topic(document, prefix="ld")
# e.g. "ld/SensorReading/sensor-42"

# Map confidence to MQTT QoS level
qos_info = derive_mqtt_qos_detailed(document)
# {"qos": 2, "reasoning": "...", "confidence_used": 0.95}
# QoS 0: confidence < 0.5 | QoS 1: 0.5 <= c < 0.9 | QoS 2: c >= 0.9 or @humanVerified
```

---

## SLNetwork: Subjective Logic Inference Engine

SLNetwork is a directed acyclic graph (DAG) inference engine for Subjective Logic. It extends jsonld-ex from flat operator calls into graph-structured reasoning with topological inference, transitive trust propagation, and counterfactual strategies.

SLNetwork composes the existing confidence algebra primitives (Opinion, cumulative_fuse, deduce, trust_discount, decay_opinion). It never reimplements them.

Reference: Josang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.

### Content Network (Tier 1)

Build a DAG of Subjective Logic opinions and infer derived beliefs through conditional deduction.

```python
from jsonld_ex.sl_network import (
    SLNetwork, SLNode, SLEdge, infer_node, infer_all,
    vacuous_counterfactual,
)
from jsonld_ex import Opinion

# Build a content network
net = SLNetwork()
net.add_node(SLNode(id="rain", opinion=Opinion(0.7, 0.1, 0.2)))
net.add_node(SLNode(id="wet_grass"))

# Add a conditional edge: P(wet_grass | rain) and P(wet_grass | ~rain)
net.add_edge(SLEdge(
    parent="rain",
    child="wet_grass",
    conditional_positive=Opinion(0.9, 0.05, 0.05),
    conditional_negative=Opinion(0.1, 0.7, 0.2),
))

# Infer the opinion for wet_grass via topological deduction
result = infer_node(net, "wet_grass")
print(f"Wet grass: b={result.opinion.belief:.3f}, u={result.opinion.uncertainty:.3f}")
# result.steps -- full audit trail of each deduction step

# Infer all nodes in topological order
all_results = infer_all(net)
```

### Counterfactual Strategies

When a node has multiple parents but not all provide edges, a counterfactual function fills the missing conditional. Three strategies are provided:

| Strategy | Function | Behavior |
|----------|----------|----------|
| `vacuous` | `vacuous_counterfactual` | Returns (0, 0, 1, 0.5): maximum ignorance |
| `adversarial` | `adversarial_counterfactual` | Returns (0, 1, 0, 0.5): worst case |
| `prior` | `prior_counterfactual` | Returns the child node's prior opinion |

### Trust Networks (Tier 2)

Layer a trust network on top of the content DAG. Agents hold opinions about other agents (trust edges) and attest to content node opinions (attestation edges). Trust propagation follows all simple paths from source to target and fuses derived trust.

```python
from jsonld_ex.sl_network import (
    SLNetwork, TrustEdge, AttestationEdge,
    propagate_trust, infer_with_trust,
)
from jsonld_ex import Opinion

net = SLNetwork()

# Add agents
net.add_agent("alice")
net.add_agent("bob")
net.add_agent("charlie")

# Trust edges: alice trusts bob, bob trusts charlie
net.add_trust_edge(TrustEdge(
    source="alice", target="bob",
    opinion=Opinion(0.9, 0.05, 0.05),
))
net.add_trust_edge(TrustEdge(
    source="bob", target="charlie",
    opinion=Opinion(0.8, 0.1, 0.1),
))

# Content node with attestations from bob and charlie
net.add_node(SLNode(id="claim_x"))
net.add_attestation(AttestationEdge(
    agent="bob", node="claim_x",
    opinion=Opinion(0.85, 0.05, 0.1),
))
net.add_attestation(AttestationEdge(
    agent="charlie", node="claim_x",
    opinion=Opinion(0.7, 0.1, 0.2),
))

# Propagate trust from alice's perspective
trust_result = propagate_trust(net, source="alice", target="charlie")
# trust_result.derived_trust -- the transitive trust opinion
# trust_result.paths -- all simple paths found

# Full pipeline: trust propagation -> attestation discounting -> deduction
result = infer_with_trust(net, "claim_x", perspective="alice")
# result.opinion -- alice's trust-weighted view of claim_x
```

### Serialization

SLNetwork supports full round-trip serialization to Python dicts and JSON-LD.

```python
# Round-trip via dict
data = net.to_dict()
restored = SLNetwork.from_dict(data)

# Round-trip via JSON-LD
jsonld_doc = net.to_jsonld()
restored = SLNetwork.from_jsonld(jsonld_doc)
```

### Visualization

Visualize SLNetwork graphs as DOT strings (zero dependencies) or as NetworkX DiGraphs (requires `pip install jsonld-ex[viz]`).

```python
from jsonld_ex.visualization import to_dot

# Generate DOT string for Graphviz rendering
dot_string = to_dot(net)
# Content nodes: ellipse, color-coded by belief
# Agent nodes: box, blue
# Deduction edges: solid
# Trust edges: dashed
# Attestation edges: dotted

# Export to NetworkX (requires networkx)
from jsonld_ex.visualization import to_networkx
G = to_networkx(net)
# G is a networkx.DiGraph with rich node/edge attributes
```

---

## FHIR R4 Interoperability

Bidirectional mapping between HL7 FHIR R4 clinical resources and jsonld-ex's Subjective Logic opinion model. Provides mathematically grounded uncertainty that composes, fuses, and propagates correctly, addressing limitations of FHIR's scalar probability and categorical code model.

Supports 32 FHIR R4 resource types across four epistemic tiers:

| Tier | Resources | Epistemic Model |
|------|-----------|-----------------|
| **1. Clinical assertions** | RiskAssessment, Observation, DiagnosticReport, Condition, AllergyIntolerance, MedicationStatement, ClinicalImpression, DetectedIssue, Immunization, FamilyMemberHistory, Procedure, Consent, Provenance, QuestionnaireResponse, Specimen | Domain-specific mappings with rich signal extraction |
| **2. Clinical workflow** | ServiceRequest, Encounter, MedicationRequest, MedicationAdministration, CarePlan, Goal, CareTeam, ImagingStudy, DocumentReference | Status-based event validity |
| **3. Administrative identity** | Patient, Organization, Practitioner, Device, Location | Record accuracy and completeness |
| **4. Financial** | Claim, ExplanationOfBenefit, Coverage | Claim and adjudication validity |

```python
from jsonld_ex.fhir_interop import from_fhir, to_fhir, fhir_clinical_fuse

# Convert a FHIR R4 Observation to a Subjective Logic opinion
fhir_obs = {
    "resourceType": "Observation",
    "id": "bp-reading",
    "status": "final",
    "interpretation": [{"coding": [{"code": "H"}]}],
}
result = from_fhir(fhir_obs)
# result["opinion"] -- Opinion derived from status + interpretation

# Round-trip back to FHIR
fhir_out = to_fhir(result)

# Fuse multiple clinical observations
fused = fhir_clinical_fuse([obs_a, obs_b], fusion="cumulative")
```

Additional FHIR capabilities include trust chain analysis (`fhir_trust_chain`), temporal decay on clinical opinions (`fhir_temporal_decay`), escalation policies (`fhir_escalation_policy`), bundle processing (`fhir_bundle_annotate`, `fhir_bundle_fuse`), GDPR consent bridging (`fhir_consent_to_opinion`, `fhir_consent_validity`), allergy reconciliation (`fhir_allergy_reconcile`), and alert filtering (`fhir_filter_alerts`).

The SL opinion is embedded in FHIR resources via the standard extension mechanism, ensuring zero breaking changes to existing FHIR infrastructure.

---

## MCP Server

jsonld-ex includes a [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes all library capabilities as **53 tools** for LLM agents. The server is stateless and read-only, safe for autonomous agent use.

### Setup

```bash
# Install with MCP support (requires Python 3.10+)
pip install jsonld-ex[mcp]

# Run with stdio transport (default -- for Claude Desktop, Cursor, etc.)
python -m jsonld_ex.mcp

# Run with streamable HTTP transport
python -m jsonld_ex.mcp --http
```

**Claude Desktop configuration** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "jsonld-ex": {
      "command": "python",
      "args": ["-m", "jsonld_ex.mcp"]
    }
  }
}
```

### Tool Overview

| # | Group | Tools | Description |
|---|-------|-------|-------------|
| 1 | AI/ML Annotation | 4 | Annotate values, extract confidence/provenance |
| 2 | Confidence Algebra | 7 | Subjective Logic: create, fuse, discount, deduce, conflict |
| 3 | Confidence Bridge | 2 | Scalar-to-opinion conversion and fusion |
| 4 | Inference | 4 | Chain propagation, source combination, conflict resolution |
| 5 | Security | 5 | Integrity hashing, allowlists, validation, resource limits |
| 6 | Vector / Similarity | 4 | Cosine similarity, vector validation, multi-metric comparison, metric listing |
| 7 | Graph Operations | 2 | Merge and diff JSON-LD graphs |
| 8 | Temporal | 3 | Point-in-time queries, annotations, temporal diff |
| 9 | Interop / Standards | 8 | PROV-O, SHACL, OWL, RDF-Star conversion and comparison |
| 10 | MQTT / IoT | 4 | Encode, decode, topic derivation, QoS mapping |
| 11 | Compliance Algebra | 10 | GDPR: jurisdictional meet, consent, propagation, triggers, erasure |
| | **Total** | **53** | |

### Tool Details

#### AI/ML Annotation (4 tools)

- **`annotate_value`** -- Create an annotated JSON-LD value with confidence, source, method, and provenance metadata.
- **`get_confidence_score`** -- Extract the @confidence score from an annotated value node.
- **`filter_by_confidence`** -- Filter a document's @graph nodes by minimum confidence threshold.
- **`get_provenance`** -- Extract all provenance metadata (confidence, source, method, timestamps) from a node.

#### Confidence Algebra (7 tools)

- **`create_opinion`** -- Create a Subjective Logic opinion w = (b, d, u, a) per Josang (2016).
- **`fuse_opinions`** -- Fuse multiple opinions using cumulative, averaging, or robust (Byzantine-resistant) fusion.
- **`discount_opinion`** -- Discount an opinion through a trust chain (Josang S14.3).
- **`decay_opinion`** -- Apply temporal decay (exponential, linear, or step) to an opinion.
- **`deduce_opinion`** -- Subjective Logic deduction, the modus ponens analogue (Josang Def. 12.6).
- **`measure_pairwise_conflict`** -- Measure pairwise conflict between two opinions: con(A,B) = b_A*d_B + d_A*b_B.
- **`measure_conflict`** -- Measure internal conflict within a single opinion.

#### Confidence Bridge (2 tools)

- **`combine_opinions_from_scalars`** -- Lift scalar confidence scores to opinions and fuse them.
- **`propagate_opinions_from_scalars`** -- Propagate scalar scores through a trust chain via iterated discount.

#### Inference (4 tools)

- **`propagate_confidence`** -- Propagate confidence through an inference chain (multiply, bayesian, min, dampened).
- **`combine_sources`** -- Combine confidence from multiple independent sources (noisy-OR, average, max, Dempster-Shafer).
- **`resolve_conflict`** -- Resolve conflicting assertions with auditable strategy (highest, weighted_vote, recency).
- **`propagate_graph_confidence`** -- Propagate confidence along a property chain in a JSON-LD graph.

#### Security (5 tools)

- **`compute_integrity`** -- Compute an SRI-style cryptographic hash (SHA-256/384/512) for a context.
- **`verify_integrity`** -- Verify a context against its declared integrity hash.
- **`validate_document`** -- Validate a document against a @shape definition with constraint checking.
- **`check_context_allowed`** -- Check if a context URL is permitted by an allowlist configuration.
- **`enforce_resource_limits`** -- Validate a document against configurable resource limits.

#### Vector Operations (2 tools)

- **`cosine_similarity`** -- Compute cosine similarity between two embedding vectors.
- **`validate_vector`** -- Validate vector dimensions and data integrity.

#### Graph Operations (2 tools)

- **`merge_graphs`** -- Merge two JSON-LD graphs with confidence-aware conflict resolution.
- **`diff_graphs`** -- Compute a semantic diff between two JSON-LD graphs.

#### Temporal (3 tools)

- **`query_at_time`** -- Query a document for its state at a specific point in time.
- **`add_temporal_annotation`** -- Add @validFrom, @validUntil, and @asOf temporal qualifiers to a value.
- **`temporal_diff`** -- Compute what changed between two points in time.

#### Interop / Standards (8 tools)

- **`to_prov_o`** -- Convert jsonld-ex annotations to W3C PROV-O provenance graph.
- **`from_prov_o`** -- Convert W3C PROV-O back to jsonld-ex annotations (round-trip).
- **`shape_to_shacl`** -- Convert a @shape definition to W3C SHACL constraints.
- **`shacl_to_shape`** -- Convert SHACL constraints back to @shape (round-trip).
- **`shape_to_owl`** -- Convert a @shape to OWL class restrictions.
- **`to_rdf_star`** -- Export annotations as RDF-Star N-Triples.
- **`compare_prov_o_verbosity`** -- Measure jsonld-ex vs PROV-O triple count and payload reduction.
- **`compare_shacl_verbosity`** -- Measure jsonld-ex @shape vs SHACL triple count and payload reduction.

#### MQTT / IoT (4 tools)

- **`mqtt_encode`** -- Serialize a JSON-LD document for MQTT transmission (CBOR or JSON, with MQTT 5.0 properties).
- **`mqtt_decode`** -- Deserialize an MQTT payload back to a JSON-LD document.
- **`mqtt_derive_topic`** -- Derive a hierarchical MQTT topic from document metadata.
- **`mqtt_derive_qos`** -- Map confidence metadata to MQTT QoS level (0/1/2).

#### Compliance Algebra (10 tools)

- **`create_compliance_opinion`** -- Create a compliance opinion w = (l, v, u, a) modeling regulatory compliance as uncertain state.
- **`jurisdictional_meet`** -- Conjunctive composition across multiple regulatory jurisdictions (e.g., GDPR + CCPA).
- **`compliance_propagation`** -- Propagate compliance through data derivation steps with multiplicative degradation.
- **`consent_validity`** -- Assess GDPR Art. 7 consent validity via six-condition composition.
- **`withdrawal_override`** -- Apply consent withdrawal override with proposition replacement (Art. 7(3)).
- **`expiry_trigger`** -- Asymmetric lawfulness-to-violation transition at expiry (retention deadlines, consent expiry).
- **`review_due_trigger`** -- Accelerated decay toward vacuity for missed mandatory reviews (Art. 35(11)).
- **`regulatory_change_trigger`** -- Proposition replacement at regulatory change events.
- **`erasure_scope_opinion`** -- Composite erasure completeness across data lineage (Art. 17).
- **`residual_contamination`** -- Contamination risk from incomplete erasure in ancestor nodes.

### Resources (5)

| URI | Description |
|-----|-------------|
| `jsonld-ex://context/ai-ml` | JSON-LD context for AI/ML annotation extensions |
| `jsonld-ex://context/security` | JSON-LD context for security extensions |
| `jsonld-ex://context/compliance` | JSON-LD context for compliance algebra extensions |
| `jsonld-ex://schema/opinion` | JSON Schema for a Subjective Logic opinion object |
| `jsonld-ex://schema/compliance-opinion` | JSON Schema for a Compliance Algebra opinion object |

### Prompts (4)

| Prompt | Description |
|--------|-------------|
| `annotate_tool_results` | Guided workflow for adding provenance annotations to any MCP tool output |
| `trust_chain_analysis` | Step-by-step workflow for multi-hop trust propagation analysis |
| `gdpr_compliance_assessment` | Multi-jurisdictional GDPR compliance assessment workflow |
| `consent_lifecycle` | Full consent lifecycle management for a processing purpose |

---

## Project Context

jsonld-ex is a research project targeting W3C standardization. It addresses gaps identified in JSON-LD 1.1 for machine learning data exchange:

- **Security hardening.** Context integrity verification, allowlists, and resource limits not present in JSON-LD 1.1.
- **AI data modeling.** No standard way to express confidence, provenance, or vector embeddings in JSON-LD. Subjective Logic provides a formally grounded alternative to scalar confidence scores, distinguishing evidence-for, evidence-against, and absence-of-evidence.
- **Graph-structured inference.** SLNetwork extends flat opinion operators into DAG-structured deduction with trust propagation, enabling reasoning over networks of uncertain assertions.
- **Validation.** JSON-LD lacks native validation; current options (SHACL, ShEx) require separate RDF tooling.
- **Data protection.** No existing ML data format has built-in GDPR/privacy compliance metadata with W3C DPV interop.
- **Clinical interoperability.** FHIR R4 resources lack composable uncertainty; scalar probabilities and categorical codes cannot be fused or propagated through trust chains.

## Links

- **Repository:** [github.com/jemsbhai/jsonld-ex](https://github.com/jemsbhai/jsonld-ex)
- **PyPI:** [pypi.org/project/jsonld-ex](https://pypi.org/project/jsonld-ex/)
- **JSON-LD 1.1 Specification:** [w3.org/TR/json-ld11](https://www.w3.org/TR/json-ld11/)
- **Model Context Protocol:** [modelcontextprotocol.io](https://modelcontextprotocol.io/)

## License

MIT
