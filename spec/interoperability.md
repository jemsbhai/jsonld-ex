# Interoperability Extensions

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-13  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/interoperability](https://jsonld-ex.github.io/ns/interoperability)

---

## 1. Introduction

This document specifies the bidirectional interoperability mappings between jsonld-ex and six established standards in the semantic web and machine learning ecosystems. Each mapping defines a forward algorithm (jsonld-ex → target standard), a reverse algorithm (target standard → jsonld-ex), round-trip guarantees, and an honest accounting of limitations.

The formal property definitions for all annotation and validation keywords are in the [Vocabulary](vocabulary.md) specification. This document defines the mapping algorithms, normative tables, and conversion semantics that enable jsonld-ex documents to interoperate with existing toolchains.

### 1.1 Motivation

JSON-LD extensions for AI/ML metadata are useful only to the extent that they integrate with the broader linked data ecosystem. Organizations adopting jsonld-ex need assurance that their annotated data can:

1. **Round-trip through existing infrastructure.** A jsonld-ex document converted to PROV-O for archival and later converted back MUST preserve all annotation semantics.
2. **Interoperate with established tools.** SHACL validators, OWL reasoners, RDF-star triple stores, and SSN/SOSA-aware IoT platforms MUST be able to consume jsonld-ex data in their native formats.
3. **Coexist with dataset-level metadata.** jsonld-ex assertion-level annotations MUST complement — not conflict with — dataset-level formats such as Croissant.

The six target standards were selected because they cover the primary integration points for AI/ML metadata in practice: provenance (PROV-O), validation (SHACL), ontological reasoning (OWL), statement-level annotation (RDF-star), sensor and IoT pipelines (SSN/SOSA), and ML dataset discovery (Croissant).

### 1.2 Scope

This specification covers the following mappings:

| Standard | Scope | Direction |
|----------|-------|-----------|
| **PROV-O** | Provenance annotation keywords → PROV-O entities, agents, activities | Bidirectional |
| **SHACL** | Validation shape keywords → SHACL node shapes and property shapes | Bidirectional |
| **OWL** | Validation constraint keywords → OWL class restrictions and datatype restrictions | Bidirectional |
| **RDF-star** | All 22 annotation keywords → RDF-star embedded triples (N-Triples and Turtle) | Bidirectional |
| **SSN/SOSA** | IoT and sensor annotation keywords → SOSA observations and SSN capabilities | Bidirectional |
| **Croissant** | Context-level interoperability with MLCommons dataset metadata | Bidirectional |

### 1.3 Conformance

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

A conforming implementation MUST implement the forward and reverse algorithms for at least one of the six target standards. An implementation that claims conformance with a specific standard mapping MUST implement both the forward and reverse directions for that mapping.

### 1.4 Terminology

- **Forward mapping** — conversion from jsonld-ex representation to a target standard's representation.
- **Reverse mapping** — conversion from a target standard's representation back to jsonld-ex.
- **Round-trip** — a forward mapping followed by a reverse mapping (or vice versa). A round-trip is *lossless* if the output is semantically equivalent to the input.
- **Graceful degradation** — the behavior when a jsonld-ex construct has no equivalent in the target standard. The construct MUST be preserved using the `jex:` namespace rather than silently discarded.
- **Conversion report** — a structured record of the conversion outcome, including success status, node counts, triple counts, and any warnings or errors (see §10).
- **Verbosity comparison** — a quantitative measurement of the representational overhead between jsonld-ex and a target standard for equivalent content (see §9).

### 1.5 Namespace Prefixes

The following namespace prefixes are used throughout this document:

| Prefix | Namespace IRI |
|--------|---------------|
| `jex:` | `https://w3id.org/jsonld-ex/` |
| `prov:` | `http://www.w3.org/ns/prov#` |
| `sh:` | `http://www.w3.org/ns/shacl#` |
| `owl:` | `http://www.w3.org/2002/07/owl#` |
| `xsd:` | `http://www.w3.org/2001/XMLSchema#` |
| `rdfs:` | `http://www.w3.org/2000/01/rdf-schema#` |
| `rdf:` | `http://www.w3.org/1999/02/22-rdf-syntax-ns#` |
| `sosa:` | `http://www.w3.org/ns/sosa/` |
| `ssn:` | `http://www.w3.org/ns/ssn/` |
| `ssn-system:` | `http://www.w3.org/ns/ssn/systems/` |
| `qudt:` | `http://qudt.org/schema/qudt/` |

---

## 2. Design Principles

The interoperability mappings are governed by five principles, listed in priority order.

### 2.1 Bidirectionality

Every mapping MUST define both a forward and a reverse direction. A standard that can only receive jsonld-ex data but not produce it — or vice versa — does not meet the interoperability requirement. Bidirectionality ensures that jsonld-ex is not a write-only format and that data can flow freely between ecosystems.

### 2.2 Lossless Round-Trip

A forward mapping followed by a reverse mapping MUST produce output that is semantically equivalent to the input. Specifically:

- All annotation keyword values present in the input MUST be recoverable from the output.
- The structural relationship between a value and its annotations MUST be preserved.
- Annotation keywords that have no equivalent in the target standard MUST be preserved in the `jex:` namespace on the target representation, so that the reverse mapping can restore them.

Semantic equivalence does not require syntactic identity. Blank node identifiers, key ordering, and whitespace MAY differ between the input and the round-tripped output.

### 2.3 Verbosity Reduction

jsonld-ex SHOULD require fewer triples and fewer bytes than the equivalent representation in each target standard for common annotation patterns. This is not a hard requirement — some standards may be more compact for specific edge cases — but the general expectation is that jsonld-ex's inline annotation syntax is more concise than the graph-based patterns required by PROV-O, SHACL, and SSN/SOSA. Verbosity metrics are defined in §9.

### 2.4 Graceful Degradation

When a jsonld-ex construct has no equivalent in a target standard, the mapping MUST NOT silently discard information. Instead:

1. The construct MUST be preserved as an annotation in the `jex:` namespace on the nearest appropriate node in the target representation.
2. A warning MUST be recorded in the conversion report (§10).
3. The reverse mapping MUST restore the original construct from the `jex:` namespace annotation.

**Example:** The `@confidence` annotation has no PROV-O equivalent. When converting to PROV-O, the confidence value is preserved as `jex:confidence` on the `prov:Entity` node. The reverse mapping reads this annotation and restores the `@confidence` keyword on the value object.

### 2.5 Complementarity

jsonld-ex operates at the **assertion level**: it attaches metadata (confidence, provenance, temporal validity) to individual values within a JSON-LD document. The target standards operate at different granularity levels:

| Standard | Granularity | Relationship to jsonld-ex |
|----------|-------------|---------------------------|
| **PROV-O** | Entity/activity level | jsonld-ex provides a compact inline syntax for common provenance patterns that PROV-O expresses as multi-node graphs |
| **SHACL** | Graph/shape level | jsonld-ex provides a JSON-LD-native constraint syntax; SHACL provides the full power of RDF-based validation |
| **OWL** | Class/ontology level | jsonld-ex constraint keywords map to OWL restrictions where applicable; OWL provides description logic reasoning |
| **RDF-star** | Statement level | jsonld-ex annotations serialize naturally as RDF-star embedded triples |
| **SSN/SOSA** | Observation level | jsonld-ex provides compact sensor annotations that SSN/SOSA expresses as observation graphs |
| **Croissant** | Dataset level | jsonld-ex assertion-level metadata complements Croissant's dataset-level discoverability metadata |

The mappings are designed to enable coexistence, not competition. An organization MAY use jsonld-ex for inline annotation during data production, convert to PROV-O for long-term archival, convert to SSN/SOSA for IoT platform integration, and convert to Croissant for dataset publication — all from the same source document.

---

## 3. PROV-O Mapping

[PROV-O](https://www.w3.org/TR/prov-o/) is the W3C Recommendation for representing provenance information as an OWL ontology. It defines core concepts — entities, activities, and agents — that describe how data was produced, by whom, and through what processes.

jsonld-ex provenance annotations express the same information inline on value objects. The PROV-O mapping converts between these two representations: the compact inline form (jsonld-ex) and the multi-node graph form (PROV-O).

### 3.1 Forward Algorithm

The forward algorithm converts a jsonld-ex annotated document to a PROV-O graph. For each annotated value object in the document, the algorithm creates one or more PROV-O nodes and links them according to the PROV-O ontology.

**Input:** A JSON-LD document (compact form) containing value objects with jsonld-ex provenance annotations.

**Output:** A PROV-O JSON-LD document with `@graph` containing entity, agent, and activity nodes. A `ConversionReport` (§10).

**Procedure:**

1. Initialize an empty output graph and a conversion report.
2. Construct a PROV-O context containing the `prov:`, `xsd:`, `rdfs:`, and `jex:` namespace bindings. Preserve any namespace bindings from the input context that do not conflict with these.
3. Walk the input document recursively. For each property whose value is a dict containing `@value`:
   a. Extract the `ProvenanceMetadata` from the value object.
   b. If no provenance annotations are present, preserve the value unchanged and continue.
   c. If any annotation keyword is present (`@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`, `@derivedFrom`, `@delegatedBy`, `@invalidatedAt`, `@invalidationReason`):
      - Create a `prov:Entity` node with a fresh blank node identifier.
      - Set `prov:value` on the entity to the literal from `@value`.
      - Map each annotation keyword to PROV-O constructs as specified in the mapping table (§3.3).
      - Replace the annotated value on the main node with a reference (`{"@id": entity_id}`) to the newly created entity.
      - Append all created nodes (entity, agents, activities) to the output graph.
      - Increment the conversion report counters.
4. For nested node objects (dicts without `@value`), recurse into step 3.
5. For list values, apply step 3 to each element that is an annotated value object.
6. Assemble the output document: `{"@context": prov_context, "@graph": [main_node, ...entity_nodes, ...agent_nodes, ...activity_nodes]}`.
7. Return the PROV-O document and the conversion report.

**Batch variant:** The `to_prov_o_graph` function accepts a document with a `@graph` array and applies the forward algorithm to each node in the array, producing a single unified PROV-O graph.

### 3.2 Reverse Algorithm

The reverse algorithm converts a PROV-O graph back to jsonld-ex inline annotations.

**Input:** A JSON-LD document with `@graph` containing PROV-O entities, agents, and activities.

**Output:** A jsonld-ex annotated document with inline provenance annotations on value objects. A `ConversionReport`.

**Procedure:**

1. Index all nodes in the `@graph` by their `@id`.
2. Identify all `prov:Entity` nodes (nodes whose `@type` includes `prov:Entity`).
3. Identify the main node: the first node in the graph whose `@type` does not include any `prov:` type. If no main node is found, report an error and return the input unchanged.
4. For each property on the main node whose value is a reference (`{"@id": ...}`) to a `prov:Entity`:
   a. Read `prov:value` from the entity to obtain the literal value.
   b. Construct a jsonld-ex value object `{"@value": literal}`.
   c. Reverse-map each PROV-O construct on the entity back to annotation keywords:
      - `jex:confidence` on the entity → `@confidence`.
      - `prov:wasAttributedTo` referencing a `prov:SoftwareAgent` → `@source` (using the agent's `@id`).
      - `prov:wasAttributedTo` referencing a `prov:Person` → `@humanVerified: true`.
      - `prov:generatedAtTime` → `@extractedAt`.
      - `prov:wasGeneratedBy` referencing a `prov:Activity` → `@method` (using the activity's `rdfs:label`).
      - `prov:wasDerivedFrom` → `@derivedFrom` (single value or list).
      - `prov:actedOnBehalfOf` on the `prov:SoftwareAgent` → `@delegatedBy` (single value or list).
      - `prov:wasInvalidatedBy` referencing a `prov:Activity` → `@invalidatedAt` (from `prov:atTime` on the activity) and `@invalidationReason` (from `rdfs:label` on the activity).
   d. Replace the reference on the main node with the reconstructed value object.
5. Return the reconstructed jsonld-ex document and the conversion report.

### 3.3 Mapping Table

The following table defines the complete mapping between jsonld-ex annotation keywords and PROV-O constructs:

| jsonld-ex Keyword | PROV-O Equivalent | Direction | Notes |
|-------------------|-------------------|-----------|-------|
| `@value` | `prov:value` on `prov:Entity` | Both | The annotated literal becomes the entity's value |
| `@source` | `prov:SoftwareAgent` + `prov:wasAttributedTo` | Both | The source IRI becomes the agent's `@id` |
| `@extractedAt` | `prov:generatedAtTime` | Both | Typed as `xsd:dateTime` in PROV-O |
| `@method` | `prov:Activity` + `prov:wasGeneratedBy` | Both | Method string becomes `rdfs:label` on the activity |
| `@confidence` | `jex:confidence` on `prov:Entity` | Both | **No PROV-O equivalent.** Preserved in `jex:` namespace |
| `@humanVerified` | `prov:Person` + `prov:wasAttributedTo` | Both | `true` creates a Person agent; `false` produces no node |
| `@derivedFrom` | `prov:wasDerivedFrom` | Both | Supports single IRI or list of IRIs |
| `@delegatedBy` | `prov:actedOnBehalfOf` on `prov:SoftwareAgent` | Both | Supports single IRI or list of IRIs |
| `@invalidatedAt` | `prov:wasInvalidatedBy` → `prov:Activity` + `prov:atTime` | Both | Invalidation time placed on an invalidation activity |
| `@invalidationReason` | `rdfs:label` on the invalidation `prov:Activity` | Both | Reason string placed as the activity's label |
| `@validFrom` | *(none)* | N/A | Temporal validity has no PROV-O equivalent. See [Temporal](temporal.md) §9.2 |
| `@validUntil` | *(none)* | N/A | Temporal validity has no PROV-O equivalent. See [Temporal](temporal.md) §9.2 |

### 3.4 Forward Mapping Detail: Node Construction

The forward mapping creates distinct PROV-O node types depending on which annotation keywords are present:

**Entity node** (always created for an annotated value):

```json
{
  "@id": "_:entity-a1b2c3d4",
  "@type": "prov:Entity",
  "prov:value": "Jane Doe",
  "jex:confidence": 0.95,
  "prov:generatedAtTime": {
    "@value": "2026-01-15T10:30:00Z",
    "@type": "xsd:dateTime"
  },
  "prov:wasAttributedTo": {"@id": "https://model.example.org/ner-v4"},
  "prov:wasGeneratedBy": {"@id": "_:activity-e5f6a7b8"}
}
```

**SoftwareAgent node** (created when `@source` is present):

```json
{
  "@id": "https://model.example.org/ner-v4",
  "@type": "prov:SoftwareAgent"
}
```

When `@delegatedBy` is also present, the agent includes delegation:

```json
{
  "@id": "https://model.example.org/ner-v4",
  "@type": "prov:SoftwareAgent",
  "prov:actedOnBehalfOf": {"@id": "https://org.example.org/research-lab"}
}
```

**Activity node** (created when `@method` is present):

```json
{
  "@id": "_:activity-e5f6a7b8",
  "@type": "prov:Activity",
  "rdfs:label": "NER",
  "prov:wasAssociatedWith": {"@id": "https://model.example.org/ner-v4"}
}
```

When both `@method` and `@source` are present, the activity is linked to the agent via `prov:wasAssociatedWith`.

**Person node** (created when `@humanVerified` is `true`):

```json
{
  "@id": "_:human-verifier-c9d0e1f2",
  "@type": "prov:Person",
  "rdfs:label": "Human Verifier"
}
```

**Invalidation activity node** (created when `@invalidatedAt` and/or `@invalidationReason` are present):

```json
{
  "@id": "_:invalidation-a3b4c5d6",
  "@type": "prov:Activity",
  "prov:atTime": {
    "@value": "2026-02-01T00:00:00Z",
    "@type": "xsd:dateTime"
  },
  "rdfs:label": "Superseded by updated model"
}
```

### 3.5 Worked Example

**Input** (jsonld-ex):

```json
{
  "@context": ["http://schema.org/", "https://w3id.org/jsonld-ex/context/v1.jsonld"],
  "@type": "Person",
  "@id": "http://example.org/alice",
  "name": {
    "@value": "Alice Smith",
    "@confidence": 0.98,
    "@source": "https://model.example.org/ner-v4",
    "@extractedAt": "2026-01-15T10:30:00Z",
    "@method": "NER"
  }
}
```

**Output** (PROV-O):

```json
{
  "@context": {
    "prov": "http://www.w3.org/ns/prov#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "jsonld-ex": "http://www.w3.org/ns/jsonld-ex/"
  },
  "@graph": [
    {
      "@id": "http://example.org/alice",
      "@type": "Person",
      "name": {"@id": "_:entity-a1b2c3d4"}
    },
    {
      "@id": "_:entity-a1b2c3d4",
      "@type": "prov:Entity",
      "prov:value": "Alice Smith",
      "jsonld-ex:confidence": 0.98,
      "prov:generatedAtTime": {
        "@value": "2026-01-15T10:30:00Z",
        "@type": "xsd:dateTime"
      },
      "prov:wasAttributedTo": {"@id": "https://model.example.org/ner-v4"},
      "prov:wasGeneratedBy": {"@id": "_:activity-e5f6a7b8"}
    },
    {
      "@id": "https://model.example.org/ner-v4",
      "@type": "prov:SoftwareAgent"
    },
    {
      "@id": "_:activity-e5f6a7b8",
      "@type": "prov:Activity",
      "rdfs:label": "NER",
      "prov:wasAssociatedWith": {"@id": "https://model.example.org/ner-v4"}
    }
  ]
}
```

The jsonld-ex input is 8 lines of annotation content. The PROV-O output requires 4 graph nodes and 11 triples to express the same provenance information.

### 3.6 Round-Trip Guarantees

The following invariants hold for PROV-O round-trips:

1. **Forward-then-reverse.** For any jsonld-ex document `D`, let `P = to_prov_o(D)` and `D' = from_prov_o(P)`. Then:
   - Every annotation keyword present on a value object in `D` MUST be present with the same value on the corresponding value object in `D'`.
   - The `@value` literal MUST be identical.
   - Annotation keywords that have no PROV-O equivalent (`@confidence`) MUST be preserved via the `jex:` namespace in `P` and restored in `D'`.

2. **Reverse-then-forward.** For any PROV-O document `P` that was produced by `to_prov_o`, let `D = from_prov_o(P)` and `P' = to_prov_o(D)`. Then:
   - Every `prov:Entity` in `P` MUST have a corresponding entity in `P'` with the same `prov:value`, `prov:generatedAtTime`, `prov:wasAttributedTo`, and `prov:wasGeneratedBy` relationships.
   - `jex:confidence` values MUST be preserved.

3. **Blank node identifiers.** Blank node identifiers (`_:entity-...`, `_:activity-...`) are generated fresh on each conversion. Round-trip equivalence is semantic, not syntactic — blank node identifiers MAY differ between the input and output.

4. **Context preservation.** Non-PROV-O namespace bindings from the input context MUST be preserved in the output context under keys that do not conflict with `prov:`, `xsd:`, `rdfs:`, or `jex:`.

### 3.7 Verbosity Comparison

The `compare_with_prov_o` function computes the representational overhead of PROV-O relative to jsonld-ex for equivalent provenance content. It operates by:

1. Converting the jsonld-ex document to PROV-O using `to_prov_o`.
2. Counting triples in the jsonld-ex input (approximate: one triple per annotated value plus one per annotation field).
3. Counting triples in the PROV-O output (reported by the conversion report).
4. Measuring serialization sizes in bytes (JSON with 2-space indentation, UTF-8 encoded).

The result is a `VerbosityComparison` (§9.1) with triple reduction and byte reduction percentages. In typical scenarios with fully annotated values, jsonld-ex requires 40–60% fewer triples than the equivalent PROV-O graph, because PROV-O requires separate nodes for entities, agents, and activities that jsonld-ex expresses as inline keywords on a single value object.

### 3.8 Limitations

The PROV-O mapping has the following known limitations:

1. **No PROV-O equivalent for `@confidence`.** PROV-O does not model belief strength. The confidence value is preserved in the `jex:` namespace, but a PROV-O consumer that does not recognize the `jex:` namespace will ignore it.

2. **No PROV-O equivalent for temporal validity.** The `@validFrom` and `@validUntil` annotations express assertion validity, not entity lifecycle. PROV-O's `prov:generatedAtTime` and `prov:invalidatedAtTime` model entity creation and invalidation, which are related but semantically distinct concepts. See [Temporal](temporal.md) §9.2.

3. **Agent identity simplification.** In jsonld-ex, `@source` is a single IRI identifying the software agent. In PROV-O, agents can have rich descriptions (names, types, organizational affiliations). The forward mapping creates a minimal `prov:SoftwareAgent` with only the `@id`; additional agent metadata is not generated.

4. **Activity identity.** The forward mapping creates a fresh blank node for each `prov:Activity`. If the same method string appears on multiple values, each produces a distinct activity node. PROV-O consumers that need activity deduplication MUST perform their own merging.

5. **Annotation keywords not mapped.** The following jsonld-ex annotation keywords have no PROV-O mapping and are not preserved during conversion: `@mediaType`, `@contentUrl`, `@contentHash`, `@translatedFrom`, `@translationModel`, `@measurementUncertainty`, `@unit`, `@aggregationMethod`, `@aggregationWindow`, `@aggregationCount`, `@calibratedAt`, `@calibrationMethod`, `@calibrationAuthority`. These keywords are specific to IoT and content metadata use cases and are covered by the SSN/SOSA mapping (§7) and RDF-star mapping (§6).

---

## 4. SHACL Mapping

[SHACL](https://www.w3.org/TR/shacl/) (Shapes Constraint Language) is the W3C Recommendation for validating RDF graphs against a set of conditions called "shapes." SHACL defines node shapes and property shapes that constrain the structure and values of RDF nodes.

jsonld-ex validation shapes express equivalent constraints inline using JSON-LD-native `@`-prefixed keywords. The SHACL mapping converts between these two representations: the compact inline form (jsonld-ex `@shape`) and the RDF graph form (SHACL `sh:NodeShape` with `sh:property` constraints).

The complete constraint-level mapping table is defined in [Validation](validation.md) §15.4. This section specifies the document-level conversion algorithms, conditional encoding patterns, round-trip guarantees, and limitations.

### 4.1 Forward Algorithm

The forward algorithm converts a jsonld-ex `@shape` definition to a SHACL shape graph serialized as JSON-LD.

**Input:** A jsonld-ex `@shape` definition dict. An optional target class IRI (defaults to the shape's `@type`). An optional shape IRI for the generated SHACL node shape.

**Output:** A SHACL shape graph as JSON-LD, containing one `sh:NodeShape` with `sh:property` entries.

**Procedure:**

1. Extract the target class from `@type` on the shape, or from the `target_class` parameter. If neither is provided, the algorithm MUST raise an error.
2. Generate a shape IRI. If not provided, a fresh blank node identifier is used.
3. Construct a SHACL context containing the `sh:`, `xsd:`, and `rdfs:` namespace bindings.
4. Initialize an empty list of property shapes.
5. For each property in the shape (keys that do not start with `@` and whose values are dicts):
   a. Create a property shape with `sh:path` set to the property name.
   b. **Cardinality.** If `@minCount` is present, set `sh:minCount` to its value (takes precedence over `@required`). Otherwise, if `@required` is `true`, set `sh:minCount` to `1`. If `@maxCount` is present, set `sh:maxCount`.
   c. **Datatype.** If `@type` is present, resolve any `xsd:` prefix to the full XSD namespace IRI and set `sh:datatype`.
   d. **Numeric range.** If `@minimum` is present, set `sh:minInclusive`. If `@maximum` is present, set `sh:maxInclusive`.
   e. **String length.** If `@minLength` is present, set `sh:minLength`. If `@maxLength` is present, set `sh:maxLength`.
   f. **Enumeration.** If `@in` is present, set `sh:in` as an RDF list (`{"@list": values}`).
   g. **Pattern.** If `@pattern` is present, set `sh:pattern`.
   h. **Logical combinators.** If `@or` is present, set `sh:or` as an RDF list of SHACL property shape fragments, each produced by recursively converting the branch constraint dict. If `@and` is present, set `sh:and` analogously. If `@not` is present, set `sh:not` to the converted branch.
   i. **Conditional constraints.** If `@if` is present, encode using the conditional patterns described in §4.4.
   j. **Cross-property constraints.** If `@lessThan`, `@lessThanOrEquals`, `@equals`, or `@disjoint` is present, set the corresponding SHACL property (`sh:lessThan`, `sh:lessThanOrEquals`, `sh:equals`, `sh:disjoint`) with the target property IRI.
   k. Append the property shape to the list.
6. Construct the `sh:NodeShape` node with `@id`, `@type` of `sh:NodeShape`, `sh:targetClass` referencing the target class, and `sh:property` containing the list of property shapes.
7. **Shape inheritance.** If `@extends` is present on the shape, convert each parent shape recursively. Set `sh:node` on the child shape to reference the parent shape's `@id`. Preserve the inheritance relationship as `jex:extends` for round-trip fidelity.
8. Return the SHACL document: `{"@context": shacl_context, "@graph": [shape_node, ...parent_shapes]}`.

### 4.2 Reverse Algorithm

The reverse algorithm converts a SHACL shape graph back to a jsonld-ex `@shape` definition.

**Input:** A SHACL shape graph as JSON-LD, containing one or more `sh:NodeShape` nodes.

**Output:** A jsonld-ex `@shape` definition dict. A list of warning strings for unsupported SHACL features.

**Procedure:**

1. Extract all `sh:NodeShape` nodes from the `@graph`. If none are found, return an empty shape with a warning.
2. If multiple node shapes are present, convert the first and emit a warning. Multi-shape support is deferred.
3. Extract `sh:targetClass` and set `@type` on the output shape.
4. For each entry in `sh:property`:
   a. Extract `sh:path` to determine the property name. If absent, skip with a warning.
   b. **Cardinality.** If `sh:minCount` is `1`, set `@required: true`. If `sh:minCount` is greater than `1`, set `@minCount`. If `sh:maxCount` is present, set `@maxCount`.
   c. **Datatype.** If `sh:datatype` is present, convert the IRI to `xsd:` prefixed form and set `@type`.
   d. **Numeric range.** Map `sh:minInclusive` → `@minimum`, `sh:maxInclusive` → `@maximum`.
   e. **String length.** Map `sh:minLength` → `@minLength`, `sh:maxLength` → `@maxLength`.
   f. **Pattern.** Map `sh:pattern` → `@pattern`.
   g. **Enumeration.** If `sh:in` is present, extract the `@list` and set `@in`.
   h. **Conditional detection.** If `sh:or` is present and contains a `jex:conditionalType` annotation, decode the conditional pattern (§4.4) to reconstruct `@if`/`@then`/`@else`.
   i. **Logical combinators.** If `sh:or` is present without a conditional marker, recursively convert each branch back to a constraint dict and set `@or`. Similarly for `sh:and` → `@and` and `sh:not` → `@not`.
   j. **Cross-property constraints.** Map `sh:lessThan` → `@lessThan`, `sh:lessThanOrEquals` → `@lessThanOrEquals`, `sh:equals` → `@equals`, `sh:disjoint` → `@disjoint`.
   k. **Unsupported features.** If any of the SHACL properties listed in §4.7 are present, emit a warning.
5. **Shape inheritance.** If the node shape has a `jex:extends` annotation, locate the referenced parent shape(s) in the graph, recursively convert them, and set `@extends` on the output shape.
6. Return the jsonld-ex shape and the list of warnings.

### 4.3 Constraint Mapping Reference

The complete keyword-level mapping between jsonld-ex validation keywords and SHACL constraint components is specified in [Validation](validation.md) §15.4. That table is normative and is not duplicated here. Implementers MUST consult validation §15.4 for the authoritative constraint correspondence.

The algorithms in §4.1 and §4.2 define how those individual constraint mappings are composed into complete SHACL shape graphs at the document level.

### 4.4 Conditional Encoding

jsonld-ex supports conditional validation via `@if`/`@then`/`@else` (see [Validation](validation.md) §6). SHACL has no native conditional construct. The forward mapping encodes conditionals using logical combinator patterns with a `jex:conditionalType` marker to enable lossless round-tripping.

#### 4.4.1 If-Then (without Else)

The conditional `@if: P, @then: Q` is equivalent to the material implication ¬P ∨ Q. The forward mapping encodes this as:

```json
{
  "sh:or": {
    "@list": [
      {"sh:not": <SHACL encoding of P>},
      <SHACL encoding of Q>
    ],
    "jex:conditionalType": "if-then"
  }
}
```

The `jex:conditionalType` annotation on the `sh:or` node distinguishes this encoding from a regular disjunction.

#### 4.4.2 If-Then-Else

The conditional `@if: P, @then: Q, @else: R` is equivalent to (P ∧ Q) ∨ (¬P ∧ R). The forward mapping encodes this as:

```json
{
  "sh:or": {
    "@list": [
      {"sh:and": {"@list": [<SHACL(P)>, <SHACL(Q)>]}},
      {"sh:and": {"@list": [{"sh:not": <SHACL(P)>}, <SHACL(R)>]}}
    ],
    "jex:conditionalType": "if-then-else"
  }
}
```

#### 4.4.3 Round-Trip Reconstruction

The reverse algorithm detects the `jex:conditionalType` marker on an `sh:or` node and reconstructs the original conditional:

- `"if-then"`: Branch 0 is `sh:not(P)` — extract `P` as `@if`. Branch 1 is `Q` — extract as `@then`.
- `"if-then-else"`: Branch 0 is `sh:and([P, Q])` — extract `P` as `@if`, `Q` as `@then`. Branch 1 is `sh:and([sh:not(P), R])` — extract `R` as `@else`.

Without the `jex:conditionalType` marker, the reverse algorithm treats the `sh:or` node as a regular disjunction and maps it to `@or`.

### 4.5 Round-Trip Guarantees

The following invariants hold for SHACL round-trips:

1. **Forward-then-reverse.** For any jsonld-ex shape `S`, let `H = shape_to_shacl(S)` and `S' = shacl_to_shape(H)`. Then:
   - Every constraint keyword present in `S` MUST be present with the same value in `S'`.
   - `@type` MUST be identical.
   - `@extends` references MUST be preserved via the `jex:extends` annotation and reconstructed.
   - Conditional constraints (`@if`/`@then`/`@else`) MUST be preserved via the `jex:conditionalType` marker and reconstructed.

2. **Reverse-then-forward.** For any SHACL document `H` that was produced by `shape_to_shacl`, let `S = shacl_to_shape(H)` and `H' = shape_to_shacl(S)`. Then:
   - Every `sh:property` constraint in `H` MUST have a corresponding constraint in `H'`.
   - `sh:targetClass` MUST be identical.
   - `sh:node` references for inherited shapes MUST be preserved.

3. **Shape identifier stability.** Shape identifiers (`@id` on `sh:NodeShape`) are generated as blank nodes by default. Round-trip equivalence is semantic — shape identifiers MAY differ between the input and output.

4. **SHACL-native documents.** A SHACL document that was not produced by `shape_to_shacl` can be converted to jsonld-ex only for the SHACL features that have jsonld-ex equivalents. Unsupported features (§4.7) are reported as warnings and discarded. The reverse-then-forward round-trip for such documents is therefore lossy for unsupported features.

### 4.6 Verbosity Comparison

The `compare_with_shacl` function computes the representational overhead of SHACL relative to jsonld-ex for equivalent validation content. It operates by:

1. Converting the jsonld-ex shape to SHACL using `shape_to_shacl`.
2. Counting constraint entries in the jsonld-ex shape (approximate: one per constraint keyword per property).
3. Counting triples in the SHACL output (including the `sh:NodeShape` type triple, `sh:targetClass`, and all `sh:property` entries with their nested constraint triples).
4. Measuring serialization sizes in bytes (JSON with 2-space indentation, UTF-8 encoded).

The result is a `VerbosityComparison` (§9.1) with triple reduction and byte reduction percentages. In typical shapes with multiple properties and mixed constraints, jsonld-ex requires 50–70% fewer triples than the equivalent SHACL graph, because SHACL requires separate graph nodes for each property shape and nested RDF lists for logical combinators and enumerations.

### 4.7 Unsupported SHACL Features

The following SHACL features have no jsonld-ex `@shape` equivalent. When encountered during reverse mapping (`shacl_to_shape`), a warning is emitted and the constraint is discarded:

| SHACL Feature | Reason for Non-Support |
|---------------|------------------------|
| `sh:sparql` | SPARQL-based constraints require an RDF query engine. jsonld-ex validation operates on JSON objects without RDF materialization. |
| `sh:qualifiedValueShape` | Qualified cardinality constraints (e.g., "at least 2 values that match shape X") require shape-scoped counting that is beyond the jsonld-ex validation model. |
| `sh:class` | Class-based node constraints require type hierarchy reasoning. jsonld-ex validates against explicit `@type` values only. |
| `sh:hasValue` | Requires checking for a specific value in a property's value set. jsonld-ex provides `@in` for enumeration but not exact-value-presence checking. |
| `sh:uniqueLang` | Requires checking uniqueness of language tags across a property's values. This is an RDF-specific concern with no JSON-LD-native equivalent. |
| `sh:node` (standalone) | Referencing an external shape for validation is handled by `@extends` for inheritance but not for arbitrary shape references on property values (which jsonld-ex handles via nested `@shape`). |
| `sh:xone` | Exclusive-or is not supported by jsonld-ex logical combinators. `@or` maps to inclusive disjunction. |

These limitations are documented honestly. Organizations that require the full power of SHACL SHOULD use SHACL directly; the jsonld-ex `@shape` system is designed for the practical subset of validation needs that covers the majority of JSON-LD use cases without requiring RDF knowledge.

---

## 5. OWL Mapping

[OWL 2](https://www.w3.org/TR/owl2-syntax/) (Web Ontology Language) is the W3C Recommendation for defining ontologies with formal description logic semantics. OWL class restrictions and datatype restrictions provide a mechanism for constraining the values that instances of a class may take for a given property.

jsonld-ex validation shapes express equivalent constraints using `@`-prefixed keywords on property definitions. The OWL mapping converts between these two representations: the compact inline form (jsonld-ex `@shape`) and the OWL axiom form (class restrictions expressed as `rdfs:subClassOf` entries on an `owl:Class`).

Unlike the SHACL mapping, which targets validation toolchains, the OWL mapping targets ontological reasoning. OWL reasoners can use the generated restrictions for class consistency checking, subsumption reasoning, and instance classification.

### 5.1 Forward Algorithm

The forward algorithm converts a jsonld-ex `@shape` definition to OWL class restrictions serialized as JSON-LD.

**Input:** A jsonld-ex `@shape` definition dict. An optional class IRI (defaults to the shape's `@type`).

**Output:** An OWL axiom document as JSON-LD, containing one `owl:Class` node with `rdfs:subClassOf` entries.

**Procedure:**

1. Extract the target class IRI from `@type` on the shape, or from the `class_iri` parameter. If neither is provided, the algorithm MUST raise an error.
2. Construct an OWL context containing the `owl:`, `xsd:`, `rdfs:`, and `rdf:` namespace bindings.
3. Initialize an empty list of OWL restrictions and an empty list of unmappable annotations.
4. For each property in the shape (keys that do not start with `@` and whose values are dicts):
   a. **Cardinality.** If `@minCount` is present, create an `owl:Restriction` with `owl:onProperty` set to the property IRI and `owl:minCardinality` set to the value (typed as `xsd:nonNegativeInteger`). `@minCount` takes precedence over `@required`. If `@minCount` is absent and `@required` is `true`, create a restriction with `owl:minCardinality` of `1`. If `@maxCount` is present, create a separate restriction with `owl:maxCardinality`.
   b. **Datatype and facets.** Collect all XSD constraining facets present on the property: `@minimum` → `xsd:minInclusive`, `@maximum` → `xsd:maxInclusive`, `@minLength` → `xsd:minLength`, `@maxLength` → `xsd:maxLength`, `@pattern` → `xsd:pattern`. Then:
      - If `@type` is present **and** facets are present, create an `owl:Restriction` with `owl:onProperty` and `owl:allValuesFrom` containing an OWL 2 DatatypeRestriction (§5.4).
      - If `@type` is present **without** facets, create a simple `owl:Restriction` with `owl:allValuesFrom` referencing the datatype IRI directly.
      - If facets are present **without** `@type`, create a DatatypeRestriction using a default base datatype (§5.4.1).
   c. **Enumeration.** If `@in` is present, create an `owl:Restriction` with `owl:allValuesFrom` containing `owl:oneOf` as an RDF list of the enumerated values.
   d. **Logical combinators.** If `@or` is present, recursively convert each branch to an OWL DataRange (§5.5) and create an `owl:Restriction` with `owl:allValuesFrom` containing `owl:unionOf`. If `@and` is present, use `owl:intersectionOf`. If `@not` is present, use `owl:datatypeComplementOf`.
   e. **Unmappable constraints.** If `@lessThan`, `@lessThanOrEquals`, `@equals`, `@disjoint`, or `@severity` is present, record the constraint as an unmappable annotation in the `jex:` namespace (§5.6). If `@if`/`@then`/`@else` is present, record the entire conditional as `jex:conditional`.
   f. Append all generated restrictions to the list.
5. **Shape inheritance.** If `@extends` is present, add each parent class IRI as a plain `rdfs:subClassOf` entry (not wrapped in a restriction).
6. Construct the `owl:Class` node with `@id` set to the target class IRI, `@type` of `owl:Class`, and `rdfs:subClassOf` containing all restrictions (as a single entry if only one, or as a list if multiple).
7. Attach unmappable annotations as class-level properties in the `jex:` namespace.
8. Return the OWL document: `{"@context": owl_context, "@graph": [owl_class]}`.

### 5.2 Reverse Algorithm

The reverse algorithm converts OWL class restrictions back to a jsonld-ex `@shape` definition.

**Input:** An OWL axiom document as JSON-LD (as produced by `shape_to_owl_restrictions`).

**Output:** A jsonld-ex `@shape` definition dict.

**Procedure:**

1. Extract the first node from `@graph`. This is the `owl:Class` node. Extract its `@id` as the class IRI and set `@type` on the output shape.
2. Collect all `rdfs:subClassOf` entries. Normalize to a list if a single entry.
3. For each entry:
   a. If the entry has `@type` of `owl:Restriction`, it is a property restriction — proceed to step 4.
   b. If the entry is a plain IRI reference (has `@id` but no `@type` of `owl:Restriction`), it is an `@extends` parent — collect the IRI.
4. For each `owl:Restriction`:
   a. Extract `owl:onProperty` to determine the property IRI. Ensure a constraint dict exists for that property.
   b. **Cardinality.** If `owl:minCardinality` is present, extract the value (unwrapping `@value` if typed). If the value is `1`, set `@required: true`. If greater than `1`, set `@minCount`. If `owl:maxCardinality` is present, set `@maxCount`.
   c. **allValuesFrom.** If `owl:allValuesFrom` is present, delegate to the `allValuesFrom` parser (§5.2.1).
5. Set `@extends` on the output shape: a single IRI if one parent, a list if multiple.
6. Restore `jex:` namespace annotations from the OWL class node (§5.6.1).
7. Return the jsonld-ex shape.

#### 5.2.1 allValuesFrom Parser

The `owl:allValuesFrom` value is parsed according to its structure:

1. **Simple IRI reference** (`{"@id": iri}` without `@type`): Set `@type` on the property constraint, converting the full XSD IRI to `xsd:` prefixed form.
2. **`owl:oneOf`**: Extract the `@list` and set `@in`.
3. **`owl:unionOf`**: Recursively convert each member to a jsonld-ex constraint dict and set `@or`.
4. **`owl:intersectionOf`**: Recursively convert each member and set `@and`.
5. **`owl:datatypeComplementOf`**: Recursively convert the complement and set `@not`.
6. **DatatypeRestriction** (`owl:onDatatype` + `owl:withRestrictions`): Extract the base datatype IRI and each facet from the restrictions list. Map each facet to the corresponding jsonld-ex keyword using the facet table (§5.3). If the base datatype is a default (§5.4.1), omit `@type` to match the original jsonld-ex convention.

### 5.3 Complete Mapping Table

The following table defines the complete mapping between jsonld-ex constraint keywords and OWL constructs:

| jsonld-ex Keyword | OWL Equivalent | Direction | Notes |
|-------------------|----------------|-----------|-------|
| `@required: true` | `owl:Restriction` + `owl:minCardinality 1` | Both | Typed as `xsd:nonNegativeInteger` |
| `@minCount` | `owl:Restriction` + `owl:minCardinality` | Both | Takes precedence over `@required` |
| `@maxCount` | `owl:Restriction` + `owl:maxCardinality` | Both | Typed as `xsd:nonNegativeInteger` |
| `@type` (datatype only) | `owl:Restriction` + `owl:allValuesFrom` (simple IRI) | Both | e.g., `owl:allValuesFrom xsd:string` |
| `@type` + facets | `owl:Restriction` + `owl:allValuesFrom` (DatatypeRestriction) | Both | §5.4 |
| `@minimum` | `xsd:minInclusive` facet in `owl:withRestrictions` | Both | Within DatatypeRestriction |
| `@maximum` | `xsd:maxInclusive` facet in `owl:withRestrictions` | Both | Within DatatypeRestriction |
| `@minLength` | `xsd:minLength` facet in `owl:withRestrictions` | Both | Within DatatypeRestriction |
| `@maxLength` | `xsd:maxLength` facet in `owl:withRestrictions` | Both | Within DatatypeRestriction |
| `@pattern` | `xsd:pattern` facet in `owl:withRestrictions` | Both | Within DatatypeRestriction |
| `@in` | `owl:Restriction` + `owl:allValuesFrom` + `owl:oneOf` | Both | Enumerated DataRange |
| `@or` | `owl:allValuesFrom` + `owl:unionOf` | Both | Recursive DataRange (§5.5) |
| `@and` | `owl:allValuesFrom` + `owl:intersectionOf` | Both | Recursive DataRange (§5.5) |
| `@not` | `owl:allValuesFrom` + `owl:datatypeComplementOf` | Both | Recursive DataRange (§5.5) |
| `@extends` | `rdfs:subClassOf` (plain IRI) | Both | Not an `owl:Restriction` node |
| `@lessThan` | *(none)* | Forward only | Preserved as `jex:lessThan` (§5.6) |
| `@lessThanOrEquals` | *(none)* | Forward only | Preserved as `jex:lessThanOrEquals` (§5.6) |
| `@equals` | *(none)* | Forward only | Preserved as `jex:equals` (§5.6) |
| `@disjoint` | *(none)* | Forward only | Preserved as `jex:disjoint` (§5.6) |
| `@severity` | *(none)* | Forward only | Preserved as `jex:severity` (§5.6) |
| `@if`/`@then`/`@else` | *(none)* | Forward only | Preserved as `jex:conditional` (§5.6) |

### 5.4 DatatypeRestriction Handling

OWL 2 DatatypeRestrictions (§7.5 of the [OWL 2 Structural Specification](https://www.w3.org/TR/owl2-syntax/)) constrain a base datatype with XSD facets. The forward mapping generates DatatypeRestrictions when a jsonld-ex property has both a datatype and constraining facets, or when it has facets without an explicit datatype.

**Forward structure:**

```json
{
  "@type": "owl:Restriction",
  "owl:onProperty": {"@id": "http://schema.org/age"},
  "owl:allValuesFrom": {
    "@type": "rdfs:Datatype",
    "owl:onDatatype": {"@id": "xsd:integer"},
    "owl:withRestrictions": {
      "@list": [
        {"xsd:minInclusive": 0},
        {"xsd:maxInclusive": 150}
      ]
    }
  }
}
```

This corresponds to the jsonld-ex shape:

```json
{
  "age": {
    "@type": "xsd:integer",
    "@minimum": 0,
    "@maximum": 150
  }
}
```

**Facet mapping:**

| jsonld-ex Keyword | XSD Facet IRI | Facet Category |
|-------------------|---------------|----------------|
| `@minimum` | `xsd:minInclusive` | Numeric |
| `@maximum` | `xsd:maxInclusive` | Numeric |
| `@minLength` | `xsd:minLength` | String |
| `@maxLength` | `xsd:maxLength` | String |
| `@pattern` | `xsd:pattern` | String |

#### 5.4.1 Default Datatype Logic

When facets are present without an explicit `@type`, the forward mapping infers a default base datatype:

- If any string facet (`@minLength`, `@maxLength`, `@pattern`) is present, the default base datatype is `xsd:string`.
- If only numeric facets (`@minimum`, `@maximum`) are present, the default base datatype is `xsd:decimal`.

The reverse mapping detects these defaults and omits `@type` from the reconstructed shape when the base datatype matches the expected default. This ensures that a shape that originally had no explicit `@type` round-trips without acquiring one.

**Default detection logic (reverse):**

1. If the DatatypeRestriction has string facets and the base datatype is `xsd:string`, omit `@type`.
2. If the DatatypeRestriction has only numeric facets and the base datatype is `xsd:decimal`, omit `@type`.
3. Otherwise, set `@type` to the `xsd:` prefixed form of the base datatype.

### 5.5 Logical Combinators

jsonld-ex logical combinators (`@or`, `@and`, `@not`) map to OWL 2 DataRange expressions (§7 of the OWL 2 Structural Specification). Each branch of a combinator is recursively converted to a DataRange.

#### 5.5.1 Union (`@or` → `owl:unionOf`)

```json
// jsonld-ex
{"@or": [{"@type": "xsd:string"}, {"@type": "xsd:integer"}]}

// OWL
{
  "owl:allValuesFrom": {
    "@type": "rdfs:Datatype",
    "owl:unionOf": {
      "@list": [
        {"@id": "xsd:string"},
        {"@id": "xsd:integer"}
      ]
    }
  }
}
```

#### 5.5.2 Intersection (`@and` → `owl:intersectionOf`)

```json
// jsonld-ex
{"@and": [{"@minimum": 0}, {"@maximum": 100}]}

// OWL
{
  "owl:allValuesFrom": {
    "@type": "rdfs:Datatype",
    "owl:intersectionOf": {
      "@list": [
        {
          "@type": "rdfs:Datatype",
          "owl:onDatatype": {"@id": "xsd:decimal"},
          "owl:withRestrictions": {"@list": [{"xsd:minInclusive": 0}]}
        },
        {
          "@type": "rdfs:Datatype",
          "owl:onDatatype": {"@id": "xsd:decimal"},
          "owl:withRestrictions": {"@list": [{"xsd:maxInclusive": 100}]}
        }
      ]
    }
  }
}
```

#### 5.5.3 Complement (`@not` → `owl:datatypeComplementOf`)

```json
// jsonld-ex
{"@not": {"@type": "xsd:string"}}

// OWL
{
  "owl:allValuesFrom": {
    "@type": "rdfs:Datatype",
    "owl:datatypeComplementOf": {"@id": "xsd:string"}
  }
}
```

#### 5.5.4 Recursive Nesting

Combinators may be nested arbitrarily. Each level is processed by the recursive `_constraint_to_owl_datarange` (forward) and `_owl_datarange_to_constraint` (reverse) functions. For example, `@or` branches may themselves contain `@and` or `@not` expressions, and these are converted to nested `owl:unionOf`/`owl:intersectionOf`/`owl:datatypeComplementOf` structures.

### 5.6 Unmappable Constraints

Several jsonld-ex constraint keywords have no OWL property restriction equivalent. These constraints are preserved as class-level annotations in the `jex:` namespace on the `owl:Class` node, ensuring that the reverse mapping can restore them.

**Unmappable scalar constraints:**

| jsonld-ex Keyword | Preserved As | Reason |
|-------------------|-------------|--------|
| `@lessThan` | `jex:lessThan` | Cross-property comparison is not expressible as an OWL property restriction. |
| `@lessThanOrEquals` | `jex:lessThanOrEquals` | Same as above. |
| `@equals` | `jex:equals` | Same as above. |
| `@disjoint` | `jex:disjoint` | Same as above. |
| `@severity` | `jex:severity` | OWL has no concept of validation severity. |

**Unmappable structural constraints:**

| jsonld-ex Keyword | Preserved As | Reason |
|-------------------|-------------|--------|
| `@if`/`@then`/`@else` | `jex:conditional` (dict containing the full conditional) | Conditional validation has no OWL equivalent. OWL class expressions are unconditional. |

#### 5.6.1 Restoration on Reverse Mapping

The reverse algorithm scans all properties on the `owl:Class` node whose keys start with the `jex:` namespace IRI. For each:

1. **Known scalar annotations** (`lessThan`, `lessThanOrEquals`, `equals`, `disjoint`, `severity`): The value is placed on the output shape under the corresponding `@`-prefixed key.
2. **Conditional** (`jex:conditional`): The stored dict is unpacked and its `@if`, `@then`, and `@else` keys are placed directly on the output shape.
3. **Unknown `jex:` annotations**: Preserved as-is on the output shape (forward compatibility).

### 5.7 Round-Trip Guarantees

The following invariants hold for OWL round-trips:

1. **Forward-then-reverse.** For any jsonld-ex shape `S`, let `O = shape_to_owl_restrictions(S)` and `S' = owl_to_shape(O)`. Then:
   - Every constraint keyword present in `S` MUST be present with the same value in `S'`.
   - `@type` (the class IRI) MUST be identical.
   - `@extends` references MUST be preserved as `rdfs:subClassOf` plain IRIs and reconstructed.
   - Unmappable constraints MUST be preserved via `jex:` namespace annotations and restored.
   - DatatypeRestrictions with default base datatypes MUST round-trip without acquiring an explicit `@type`.

2. **Reverse-then-forward.** For any OWL document `O` that was produced by `shape_to_owl_restrictions`, let `S = owl_to_shape(O)` and `O' = shape_to_owl_restrictions(S)`. Then:
   - Every `owl:Restriction` in `O` MUST have a corresponding restriction in `O'` with the same `owl:onProperty`, cardinality values, and `owl:allValuesFrom` structure.
   - `jex:` namespace annotations MUST be preserved.

3. **Cardinality precedence.** The `@minCount` keyword takes precedence over `@required` in both directions. The forward mapping generates `owl:minCardinality` from whichever is present; the reverse mapping sets `@required: true` only when the cardinality value is exactly `1` and `@minCount` was not explicitly specified.

4. **OWL-native documents.** An OWL document that was not produced by `shape_to_owl_restrictions` can be converted to jsonld-ex only for the OWL constructs that the reverse algorithm recognizes. Unrecognized OWL axiom types (e.g., `owl:hasKey`, `owl:disjointWith`, `owl:equivalentClass`) are not processed and are silently ignored. The reverse-then-forward round-trip for such documents is lossy for unrecognized constructs.

### 5.8 Limitations

The OWL mapping has the following known limitations:

1. **No cross-property restriction mapping.** OWL property restrictions operate on individual properties. Cross-property comparisons (`@lessThan`, `@equals`, `@disjoint`) have no OWL equivalent. These are preserved in the `jex:` namespace but cannot be interpreted by standard OWL reasoners.

2. **No conditional mapping.** OWL class expressions are unconditional. The `@if`/`@then`/`@else` construct is preserved in the `jex:` namespace but has no effect on OWL reasoning.

3. **Property-level restriction only.** The mapping produces `owl:Restriction` nodes with `owl:onProperty`. OWL's richer class expressions (e.g., `owl:hasKey`, `owl:disjointWith`, general class intersection/union at the class level) are not generated by the forward mapping.

4. **No cardinality qualification.** OWL supports qualified cardinality restrictions (`owl:minQualifiedCardinality` with `owl:onDataRange`). jsonld-ex `@minCount`/`@maxCount` map to unqualified cardinality restrictions only.

5. **Datatype restriction scope.** Only XSD constraining facets defined in the facet table (§5.4) are supported. OWL 2 allows arbitrary XSD facets; facets not in the table are not generated or parsed.

6. **Single class output.** The forward mapping produces a single `owl:Class` node. It does not generate OWL ontology-level constructs (e.g., `owl:Ontology` declarations, import statements). Consumers that require a complete OWL ontology document MUST wrap the output in appropriate ontology-level boilerplate.

7. **No severity mapping.** OWL has no concept of validation severity. The `@severity` annotation is preserved in the `jex:` namespace but has no effect on OWL reasoning or class consistency checking.

---

## 6. RDF-star Mapping

[RDF 1.2](https://www.w3.org/TR/rdf12-concepts/) introduces embedded triples (formerly "RDF-star"), enabling statements about statements without verbose reification. An embedded triple `<< s p o >>` can serve as the subject of an annotation triple, attaching metadata directly to the statement it describes.

jsonld-ex annotation keywords express the same kind of statement-level metadata — confidence, provenance, measurement context — inline on value objects. The RDF-star mapping serializes these inline annotations as embedded-triple annotations in both N-Triples and Turtle surface syntaxes. This is the most natural mapping of the six, because jsonld-ex and RDF-star address the same problem at the same granularity: metadata about individual assertions.

The mapping supports all 22 annotation keywords defined by the jsonld-ex vocabulary.

### 6.1 Forward Algorithm (N-Triples)

The forward algorithm converts a jsonld-ex annotated document to RDF 1.2 N-Triples with embedded triple annotations.

**Input:** A JSON-LD document (compact form) containing value objects with jsonld-ex annotations. An optional base subject IRI (defaults to the document's `@id` or `_:subject`).

**Output:** An N-Triples-star string. A `ConversionReport` (§10).

**Procedure:**

1. Determine the document subject. If `base_subject` is provided, use it. Otherwise, use the document's `@id`. If neither is available, use the blank node `_:subject`. If the subject is an IRI (not a blank node), wrap it in angle brackets.
2. Initialize an empty list of output lines and a conversion report.
3. For each property on the document (keys that do not start with `@`):
   a. If the value is a dict containing `@value`:
      - Extract the `ProvenanceMetadata` from the value object.
      - Collect all non-None annotation fields using the shared annotation field descriptor table (§6.4). Each field produces a tuple of `(jex_local_name, python_value, value_kind)`.
      - If no annotation fields are present, emit a plain base triple `subject <predicate> literal .` and continue.
      - If annotations are present:
        1. Format the `@value` literal as an N-Triples term (applying typed literal syntax for integers, doubles, and booleans; plain string syntax otherwise).
        2. Emit the base triple: `subject <predicate> literal .`
        3. Construct the embedded triple: `<< subject <predicate> literal >>`.
        4. For each annotation tuple, format the annotation value according to its `value_kind` (§6.4.1) using full IRIs (as required by N-Triples) and emit: `<< subject <predicate> literal >> <jex:local_name> formatted_value .`
        5. Increment the conversion report counters.
   b. If the value is a plain string, number, or boolean, emit a standard triple with appropriate typed literal formatting.
4. Join all output lines with newlines.
5. Return the N-Triples string and the conversion report.

### 6.2 Forward Algorithm (Turtle)

The Turtle forward algorithm produces a human-readable alternative to the N-Triples output, using `@prefix` declarations and semicolon grouping.

**Input:** Same as §6.1.

**Output:** A Turtle-star string with prefix declarations. A `ConversionReport`.

**Procedure:**

1. Determine the document subject (same as §6.1, step 1).
2. Initialize an empty set of used prefixes, an empty list of body lines, and a conversion report.
3. For each property on the document (keys that do not start with `@`):
   a. If the value is a dict containing `@value`:
      - Collect annotation tuples as in §6.1.
      - If no annotations are present, emit a plain triple and continue.
      - If annotations are present:
        1. Emit the base triple.
        2. Construct the embedded triple.
        3. Add `jex` to the used prefixes set.
        4. For each annotation tuple, format the annotation value using prefixed names (e.g., `jex:confidence`, `xsd:double`) and track which prefixes are referenced. Group all annotations for the same embedded triple using Turtle's `;` continuation syntax:
           ```
           << subject <predicate> literal >>
               jex:confidence "0.95"^^xsd:double ;
               jex:source <https://model.example.org/v1> .
           ```
        5. Increment the conversion report counters.
   b. Non-annotated values are handled as in §6.1.
4. Assemble the output: emit `@prefix` declarations only for prefixes that are actually referenced in the body, followed by a blank line, followed by the body lines.
5. Return the Turtle string and the conversion report.

The Turtle output is semantically identical to the N-Triples output. The differences are purely syntactic: prefixed names instead of full IRIs, semicolon grouping instead of repeated embedded triples, and prefix declarations at the top.

### 6.3 Reverse Algorithm

The reverse algorithm parses RDF-star N-Triples and reconstructs a jsonld-ex annotated document.

**Input:** An N-Triples-star string (as produced by `to_rdf_star_ntriples`).

**Output:** A reconstructed jsonld-ex document with inline annotations on value objects. A `ConversionReport`.

**Procedure:**

The algorithm operates in three phases.

**Phase 1 — Parse.** For each non-empty, non-comment line in the input:

1. If the line starts with `<<`, it is an annotation triple:
   a. Locate the closing `>>` to extract the embedded triple content.
   b. Parse the three terms of the embedded triple (subject, predicate, object) using the N-Triples term parser. The parser handles IRIs (angle-bracket delimited), typed literals (`"value"^^<type>`), language-tagged literals (`"value"@lang`), plain string literals, and blank nodes.
   c. Parse the annotation predicate and value from the outer portion (after `>>`).
   d. Look up the annotation predicate IRI in the predicate-to-keyword mapping (§6.4). If not found, emit a warning and skip.
   e. Store the annotation keyed by `(subject, predicate, raw_object_form)` — the raw form is used as a grouping key so that base triples and their annotations can be matched even when typed literal syntax varies.
2. If the line does not start with `<<`, it is a base triple:
   a. Parse subject, predicate, and object.
   b. Store as a base triple with subject IRI, predicate IRI, Python value, and raw object form.

**Phase 2 — Reconstruct.** For each base triple:

1. Look up the grouping key `(subject, predicate, raw_object)` in the annotations index.
2. If annotations exist:
   a. Create an annotated value object `{"@value": python_value}`.
   b. For each annotation `(keyword, value)`: if the keyword is a multi-value keyword (§6.5), append the value to a list under that keyword. Otherwise, set the keyword directly.
   c. Attach the annotated value to the subject's property dict.
3. If no annotations exist, attach the plain Python value to the subject's property dict.

**Phase 3 — Assemble.** If a single subject is present, return it as a flat document with `@id`. If multiple subjects are present, wrap them in a `@graph` array.

### 6.4 Complete Annotation Mapping Table

The following table lists all 22 annotation keywords supported by the RDF-star mapping, derived from the shared `_ANNOTATION_FIELDS` descriptor and the `_RDF_STAR_PREDICATE_TO_KEYWORD` reverse lookup.

| # | jsonld-ex Keyword | jex: Local Name | Predicate IRI | Value Kind | Multi-Value |
|---|-------------------|-----------------|---------------|------------|-------------|
| 1 | `@confidence` | `confidence` | `jex:confidence` | `xsd:double` | No |
| 2 | `@source` | `source` | `jex:source` | IRI | No |
| 3 | `@extractedAt` | `extractedAt` | `jex:extractedAt` | `xsd:dateTime` | No |
| 4 | `@method` | `method` | `jex:method` | `xsd:string` | No |
| 5 | `@humanVerified` | `humanVerified` | `jex:humanVerified` | `xsd:boolean` | No |
| 6 | `@mediaType` | `mediaType` | `jex:mediaType` | `xsd:string` | No |
| 7 | `@contentUrl` | `contentUrl` | `jex:contentUrl` | IRI | No |
| 8 | `@contentHash` | `contentHash` | `jex:contentHash` | `xsd:string` | No |
| 9 | `@translatedFrom` | `translatedFrom` | `jex:translatedFrom` | `xsd:string` | No |
| 10 | `@translationModel` | `translationModel` | `jex:translationModel` | `xsd:string` | No |
| 11 | `@measurementUncertainty` | `measurementUncertainty` | `jex:measurementUncertainty` | `xsd:double` | No |
| 12 | `@unit` | `unit` | `jex:unit` | `xsd:string` | No |
| 13 | `@derivedFrom` | `derivedFrom` | `jex:derivedFrom` | IRI | **Yes** |
| 14 | `@delegatedBy` | `delegatedBy` | `jex:delegatedBy` | IRI | **Yes** |
| 15 | `@aggregationMethod` | `aggregationMethod` | `jex:aggregationMethod` | `xsd:string` | No |
| 16 | `@aggregationWindow` | `aggregationWindow` | `jex:aggregationWindow` | `xsd:string` | No |
| 17 | `@aggregationCount` | `aggregationCount` | `jex:aggregationCount` | `xsd:integer` | No |
| 18 | `@calibratedAt` | `calibratedAt` | `jex:calibratedAt` | `xsd:dateTime` | No |
| 19 | `@calibrationMethod` | `calibrationMethod` | `jex:calibrationMethod` | `xsd:string` | No |
| 20 | `@calibrationAuthority` | `calibrationAuthority` | `jex:calibrationAuthority` | `xsd:string` | No |
| 21 | `@invalidatedAt` | `invalidatedAt` | `jex:invalidatedAt` | `xsd:dateTime` | No |
| 22 | `@invalidationReason` | `invalidationReason` | `jex:invalidationReason` | `xsd:string` | No |

#### 6.4.1 Value Kind Formatting

The `Value Kind` column determines how the annotation value is serialized:

| Value Kind | N-Triples Format | Turtle Format | Python Type |
|------------|------------------|---------------|-------------|
| `xsd:double` | `"0.95"^^<http://www.w3.org/2001/XMLSchema#double>` | `"0.95"^^xsd:double` | `float` |
| `xsd:integer` | `"42"^^<http://www.w3.org/2001/XMLSchema#integer>` | `"42"^^xsd:integer` | `int` |
| `xsd:boolean` | `"true"^^<http://www.w3.org/2001/XMLSchema#boolean>` | `"true"^^xsd:boolean` | `bool` |
| `xsd:dateTime` | `"2026-01-15T10:30:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime>` | `"2026-01-15T10:30:00Z"^^xsd:dateTime` | `str` |
| IRI | `<https://example.org/resource>` | `<https://example.org/resource>` | `str` |
| `xsd:string` | `"plain text"` | `"plain text"` | `str` |

The reverse algorithm casts typed literals back to the appropriate Python type: `xsd:double` → `float`, `xsd:integer` → `int`, `xsd:boolean` → `bool`, all others (including `xsd:dateTime`) → `str`.

### 6.5 Multi-Value Keywords

Two annotation keywords — `@derivedFrom` and `@delegatedBy` — accept multiple values. In jsonld-ex, these are represented as lists:

```json
{
  "@value": "Jane Doe",
  "@derivedFrom": [
    "https://source.example.org/db1",
    "https://source.example.org/db2"
  ]
}
```

In the RDF-star serialization, each list item produces a separate annotation triple:

```
<< <http://example.org/person> <http://schema.org/name> "Jane Doe" >> <jex:derivedFrom> <https://source.example.org/db1> .
<< <http://example.org/person> <http://schema.org/name> "Jane Doe" >> <jex:derivedFrom> <https://source.example.org/db2> .
```

The reverse algorithm detects multi-value keywords by consulting the `_RDF_STAR_MULTI_VALUE_KEYWORDS` set. When multiple annotation triples share the same embedded triple and the same multi-value keyword, their values are accumulated into a list rather than overwriting each other.

All other keywords are single-valued. If multiple annotation triples for the same embedded triple carry the same single-valued keyword, the last value encountered takes precedence.

### 6.6 Worked Example

**Input** (jsonld-ex):

```json
{
  "@id": "http://example.org/sensor-reading",
  "http://schema.org/temperature": {
    "@value": 22.5,
    "@confidence": 0.95,
    "@source": "https://sensor.example.org/thermostat-1",
    "@extractedAt": "2026-01-15T10:30:00Z",
    "@unit": "celsius"
  }
}
```

**Output (N-Triples-star):**

```
<http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^<http://www.w3.org/2001/XMLSchema#double> .
<< <http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^<http://www.w3.org/2001/XMLSchema#double> >> <http://www.w3.org/ns/jsonld-ex/confidence> "0.95"^^<http://www.w3.org/2001/XMLSchema#double> .
<< <http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^<http://www.w3.org/2001/XMLSchema#double> >> <http://www.w3.org/ns/jsonld-ex/source> <https://sensor.example.org/thermostat-1> .
<< <http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^<http://www.w3.org/2001/XMLSchema#double> >> <http://www.w3.org/ns/jsonld-ex/extractedAt> "2026-01-15T10:30:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> .
<< <http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^<http://www.w3.org/2001/XMLSchema#double> >> <http://www.w3.org/ns/jsonld-ex/unit> "celsius" .
```

**Output (Turtle-star):**

```turtle
@prefix jex: <http://www.w3.org/ns/jsonld-ex/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^xsd:double .
<< <http://example.org/sensor-reading> <http://schema.org/temperature> "22.5"^^xsd:double >>
    jex:confidence "0.95"^^xsd:double ;
    jex:source <https://sensor.example.org/thermostat-1> ;
    jex:extractedAt "2026-01-15T10:30:00Z"^^xsd:dateTime ;
    jex:unit "celsius" .
```

The N-Triples output requires 5 lines (1 base triple + 4 annotation triples). The Turtle output groups the 4 annotations under a single embedded triple subject using semicolons, improving readability. Both are semantically identical.

The jsonld-ex input expresses the same information in 7 lines of JSON. The RDF-star representation is more verbose but integrates directly with RDF 1.2 triple stores and SPARQL 1.2 query engines.

### 6.7 Round-Trip Guarantees

The following invariants hold for RDF-star round-trips:

1. **Forward-then-reverse (N-Triples).** For any jsonld-ex document `D`, let `N = to_rdf_star_ntriples(D)` and `D' = from_rdf_star_ntriples(N)`. Then:
   - Every annotation keyword present on a value object in `D` MUST be present with the same value on the corresponding value object in `D'`.
   - The `@value` literal MUST be identical after type casting (integers, floats, booleans, and strings are preserved through typed literal serialization and deserialization).
   - Multi-value keywords (`@derivedFrom`, `@delegatedBy`) MUST be restored as lists with the same elements (order MAY differ).
   - The document `@id` MUST be preserved.

2. **All 22 keywords.** Every annotation keyword in the mapping table (§6.4) is serialized and deserialized without loss. There are no keywords that are "forward only" or "reverse only" — the RDF-star mapping is fully symmetric.

3. **Typed literal fidelity.** Numeric values round-trip through their XSD typed literal representations: `float` → `xsd:double` → `float`, `int` → `xsd:integer` → `int`, `bool` → `xsd:boolean` → `bool`. String values (including `xsd:dateTime`) round-trip as plain or typed string literals.

4. **Turtle equivalence.** The `to_rdf_star_turtle` output is semantically identical to the `to_rdf_star_ntriples` output. A reverse parser for Turtle-star would produce the same jsonld-ex document. The current reference implementation provides a reverse parser for N-Triples only; Turtle reverse parsing is deferred.

5. **Subject identity.** The subject IRI (from `@id` or the `base_subject` parameter) is preserved through the round-trip. Blank node subjects (`_:subject`) are generated deterministically when no `@id` is present.

### 6.8 Limitations

The RDF-star mapping has the following known limitations:

1. **N-Triples reverse only.** The reference implementation provides `from_rdf_star_ntriples` for reverse mapping but no `from_rdf_star_turtle`. Turtle parsing requires handling prefix expansion, multiline literals, and `;`/`,` grouping syntax, which is significantly more complex than N-Triples line parsing. Applications that need Turtle reverse mapping SHOULD convert Turtle to N-Triples using a standard RDF library before calling the reverse algorithm.

2. **Flat document structure.** The forward mapping processes top-level properties on a single document node. Nested node objects (nodes within nodes) and `@graph` arrays are not traversed. To serialize a multi-node document, each node MUST be converted separately.

3. **No named graph support.** The mapping produces triples in the default graph. Named graph assignment (N-Quads) is not supported.

4. **Annotation-on-annotation.** RDF-star supports nested embedded triples (annotations on annotations). The jsonld-ex mapping does not generate or parse nested annotations — each annotation triple has the embedded base triple as its subject, not another annotation triple.

5. **String escaping scope.** The N-Triples serializer escapes backslash, double quote, newline, carriage return, and tab characters. Unicode escape sequences (`\uXXXX`, `\UXXXXXXXX`) are not generated. Input strings containing characters outside the ASCII printable range are serialized as UTF-8 without escaping, which is valid N-Triples but may not be handled by all parsers.

6. **No RDF-star reification vocabulary.** The mapping uses the embedded triple syntax (`<< s p o >>`) directly. It does not generate the RDF 1.2 reification vocabulary terms (`rdf:reifies`, `rdf:ReificationProperty`) that some systems may use as an alternative representation of the same semantics.

---

## 7. SSN/SOSA Mapping

[SSN](https://www.w3.org/TR/vocab-ssn/) (Semantic Sensor Network Ontology) and its lightweight core [SOSA](https://www.w3.org/TR/vocab-ssn/#SOSAOntology) (Sensor, Observation, Sample, and Actuator) are W3C Recommendations for describing sensors, their observations, and the features they observe. SSN/SOSA is the standard vocabulary for IoT platforms, environmental monitoring systems, and scientific instrumentation.

jsonld-ex IoT annotations express sensor metadata — measurement values, units, uncertainties, calibration records, and aggregation parameters — inline on value objects. The SSN/SOSA mapping converts between these two representations: the compact inline form (jsonld-ex annotations) and the multi-node observation graph form (SOSA observations with associated sensors, procedures, results, and capabilities).

Of the six mappings, SSN/SOSA produces the richest graph structure: a single annotated jsonld-ex value can expand into five or more SOSA/SSN nodes (Observation, Sensor, Procedure, Result, SystemCapability, Accuracy, FeatureOfInterest, ObservableProperty).

### 7.1 Forward Algorithm

The forward algorithm converts a jsonld-ex annotated document to an SSN/SOSA observation graph.

**Input:** A JSON-LD document (compact form) containing value objects with jsonld-ex IoT annotations.

**Output:** An SSN/SOSA JSON-LD document with `@graph` containing observation, sensor, procedure, result, feature-of-interest, and capability nodes. A `ConversionReport` (§10).

**Procedure:**

1. Initialize an empty output graph, a conversion report, and a sensor deduplication index (keyed by `@source` IRI).
2. Construct an SSN/SOSA context containing the `sosa:`, `ssn:`, `ssn-system:`, `qudt:`, `xsd:`, `rdfs:`, and `jex:` namespace bindings. Preserve any namespace bindings from the input context that do not conflict with these.
3. Create a `sosa:FeatureOfInterest` node from the document's `@id` and `@type`. If the document has an existing `@type`, the FeatureOfInterest node carries both `sosa:FeatureOfInterest` and the original type(s) in its `@type` array.
4. Initialize an empty main node dict for non-annotated properties.
5. For each property on the document (keys that do not start with `@`):
   a. If the value is not a dict containing `@value`, preserve it on the main node unchanged and continue.
   b. Extract the `ProvenanceMetadata` from the value object.
   c. If no IoT-relevant annotations are present (`@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`, `@measurementUncertainty`, `@unit`, `@calibratedAt`, `@calibrationMethod`, `@calibrationAuthority`, `@aggregationMethod`, `@aggregationWindow`, `@aggregationCount`), preserve the value on the main node unchanged and continue.
   d. Create a `sosa:Observation` node with a fresh blank node identifier.
   e. Link the observation to the FeatureOfInterest: `sosa:hasFeatureOfInterest → {"@id": foi_id}`.
   f. Create a `sosa:ObservableProperty` node from the property key and link the observation to it: `sosa:observedProperty → {"@id": property_key}`.
   g. **Result.** If `@unit` is present, create a structured `sosa:Result` node with `qudt:numericValue` and `qudt:unit`, and link the observation via `sosa:hasResult`. If `@unit` is absent, set `sosa:hasSimpleResult` directly on the observation.
   h. **Result time.** If `@extractedAt` is present, set `sosa:resultTime` on the observation (typed as `xsd:dateTime`).
   i. **Confidence.** If `@confidence` is present, set `jex:confidence` on the observation. SOSA has no native confidence concept.
   j. **Sensor.** If `@source` is present, set `sosa:madeBySensor → {"@id": source_iri}` on the observation. Create a `sosa:Sensor` node for the source IRI if one has not already been created (sensor deduplication — see §7.4.2).
   k. **Procedure.** If `@aggregationMethod` or `@method` is present (aggregation takes precedence), create a `sosa:Procedure` node with `rdfs:label` set to the method string. If `@aggregationWindow` or `@aggregationCount` is also present, attach them as `jex:aggregationWindow` and `jex:aggregationCount` on the procedure node. Link the observation via `sosa:usedProcedure`.
   l. **System capability.** If any of `@measurementUncertainty`, `@calibratedAt`, `@calibrationMethod`, or `@calibrationAuthority` is present, and a sensor node exists:
      - Create an `ssn-system:SystemCapability` node.
      - If `@measurementUncertainty` is present, create an `ssn-system:Accuracy` node with `jex:value` set to the uncertainty value, and link it to the capability via `ssn-system:hasSystemProperty`.
      - If calibration properties are present, attach them as `jex:calibratedAt`, `jex:calibrationMethod`, and `jex:calibrationAuthority` on the capability node.
      - Link the capability to the sensor via `ssn-system:hasSystemCapability`.
   m. Append the observation and all created nodes to the output graph. Increment the conversion report counters.
6. Append all deduplicated sensor nodes to the output graph.
7. If any observations were created, append the FeatureOfInterest node to the output graph.
8. Insert the main node (non-annotated properties) at the beginning of the output graph.
9. Return the SSN/SOSA document (`{"@context": ssn_context, "@graph": graph_nodes}`) and the conversion report.

### 7.2 Reverse Algorithm

The reverse algorithm converts an SSN/SOSA observation graph back to jsonld-ex inline annotations.

**Input:** A JSON-LD document with `@graph` containing SSN/SOSA nodes.

**Output:** A jsonld-ex annotated document with inline annotations on value objects. A `ConversionReport`.

**Procedure:**

1. Index all nodes in the `@graph` by their `@id`.
2. Identify all `sosa:Observation` nodes (nodes whose `@type` includes `sosa:Observation`).
3. If no observations are found, report an error and return the input unchanged.
4. Determine the parent node from the `sosa:hasFeatureOfInterest` reference on the first observation. Set `@id` on the output document to the FeatureOfInterest IRI. Recover the original `@type` by extracting non-SOSA types from the FeatureOfInterest node's `@type` array.
5. For each `sosa:Observation`, delegate to the observation-to-annotation converter (§7.2.1).
6. Warn on unsupported SOSA node types present in the graph (§7.7).
7. Return the reconstructed jsonld-ex document and the conversion report.

#### 7.2.1 Observation-to-Annotation Conversion

For each `sosa:Observation` node:

1. **Property key.** Extract `sosa:observedProperty` to determine the property key. If absent, emit a warning and skip.
2. **Result value.** Check `sosa:hasResult` first (structured result): resolve the Result node and extract `qudt:numericValue` as the `@value` and `qudt:unit` as `@unit`. If no structured result, check `sosa:hasSimpleResult` for a plain value. If no result is found, emit a warning and skip.
3. **Confidence.** Read `jex:confidence` from the observation.
4. **Source.** Read `sosa:madeBySensor` to obtain the sensor IRI, and set `@source`. If the sensor node has an `ssn-system:hasSystemCapability`, resolve the capability node:
   - If the capability has `ssn-system:hasSystemProperty` referencing an `ssn-system:Accuracy` node, read `jex:value` from the Accuracy node and set `@measurementUncertainty`.
5. **Extracted at.** Read `sosa:resultTime` and set `@extractedAt`.
6. **Method.** Read `sosa:usedProcedure`, resolve the Procedure node, and extract `rdfs:label` as `@method`.
7. Return the `(property_key, annotated_value)` pair.

### 7.3 Mapping Table

The following table defines the complete mapping between jsonld-ex annotation keywords and SSN/SOSA constructs:

| jsonld-ex Keyword | SSN/SOSA Equivalent | Direction | Notes |
|-------------------|---------------------|-----------|-------|
| `@value` (with `@unit`) | `sosa:hasResult` → `sosa:Result` + `qudt:numericValue` | Both | Structured result with QUDT unit |
| `@value` (without `@unit`) | `sosa:hasSimpleResult` | Both | Plain literal result |
| `@source` | `sosa:madeBySensor` → `sosa:Sensor` | Both | Sensor IRI becomes the `@id` of the Sensor node |
| `@extractedAt` | `sosa:resultTime` | Both | Typed as `xsd:dateTime` |
| `@method` | `sosa:usedProcedure` → `sosa:Procedure` + `rdfs:label` | Both | Method string becomes the procedure's label |
| `@confidence` | `jex:confidence` on `sosa:Observation` | Both | **No SOSA equivalent.** Preserved in `jex:` namespace |
| `@unit` | `qudt:unit` on `sosa:Result` | Both | QUDT unit identifier |
| `@measurementUncertainty` | `ssn-system:Accuracy` + `jex:value` on `ssn-system:SystemCapability` | Both | Accuracy node linked to Sensor via capability |
| `@calibratedAt` | `jex:calibratedAt` on `ssn-system:SystemCapability` | Both | **No native SSN property.** Preserved in `jex:` namespace |
| `@calibrationMethod` | `jex:calibrationMethod` on `ssn-system:SystemCapability` | Both | **No native SSN property.** Preserved in `jex:` namespace |
| `@calibrationAuthority` | `jex:calibrationAuthority` on `ssn-system:SystemCapability` | Both | **No native SSN property.** Preserved in `jex:` namespace |
| `@aggregationMethod` | `sosa:usedProcedure` → `sosa:Procedure` + `rdfs:label` | Forward | Takes precedence over `@method` for procedure label |
| `@aggregationWindow` | `jex:aggregationWindow` on `sosa:Procedure` | Forward | **No native SOSA property.** Preserved in `jex:` namespace |
| `@aggregationCount` | `jex:aggregationCount` on `sosa:Procedure` | Forward | **No native SOSA property.** Preserved in `jex:` namespace |
| `@humanVerified` | *(not mapped)* | N/A | Human verification is a provenance concept, not an observation concept. Use the PROV-O mapping (§3) |
| parent `@id`/`@type` | `sosa:hasFeatureOfInterest` → `sosa:FeatureOfInterest` | Both | The document's subject becomes the feature being observed |
| property key | `sosa:observedProperty` → `sosa:ObservableProperty` | Both | The JSON-LD property key becomes the observable property IRI |

### 7.4 Node Construction Detail

The forward mapping creates up to eight distinct SSN/SOSA node types from a single annotated jsonld-ex value. This section describes each node type and the conditions under which it is generated.

#### 7.4.1 sosa:Observation

Always created for each annotated value. Serves as the central node linking the feature of interest, observable property, result, sensor, and procedure.

```json
{
  "@id": "_:obs-a1b2c3d4",
  "@type": "sosa:Observation",
  "sosa:hasFeatureOfInterest": {"@id": "http://example.org/building-1"},
  "sosa:observedProperty": {"@id": "http://schema.org/temperature"},
  "sosa:hasSimpleResult": 22.5,
  "sosa:resultTime": {"@value": "2026-01-15T10:30:00Z", "@type": "xsd:dateTime"},
  "sosa:madeBySensor": {"@id": "https://sensor.example.org/thermostat-1"},
  "sosa:usedProcedure": {"@id": "_:proc-e5f6a7b8"},
  "jex:confidence": 0.95
}
```

#### 7.4.2 sosa:Sensor

Created when `@source` is present. The source IRI becomes the sensor's `@id`. Sensors are **deduplicated**: if multiple annotated values reference the same `@source` IRI, only one Sensor node is created. This reflects the real-world scenario where multiple observations are made by the same physical sensor.

```json
{
  "@id": "https://sensor.example.org/thermostat-1",
  "@type": "sosa:Sensor"
}
```

When system capability annotations are present, the sensor node is extended with a capability link (§7.4.6).

#### 7.4.3 sosa:Procedure

Created when `@method` or `@aggregationMethod` is present. If `@aggregationMethod` is present, it takes precedence over `@method` as the procedure's label. Aggregation parameters (`@aggregationWindow`, `@aggregationCount`) are attached to the procedure node in the `jex:` namespace.

```json
{
  "@id": "_:proc-e5f6a7b8",
  "@type": "sosa:Procedure",
  "rdfs:label": "5-minute moving average",
  "jex:aggregationWindow": "PT5M",
  "jex:aggregationCount": 300
}
```

#### 7.4.4 sosa:Result

Created when `@unit` is present. The result node carries the numeric value via `qudt:numericValue` and the unit via `qudt:unit`. When `@unit` is absent, the value is attached directly to the observation as `sosa:hasSimpleResult` and no Result node is created.

```json
{
  "@id": "_:result-c9d0e1f2",
  "@type": "sosa:Result",
  "qudt:numericValue": 22.5,
  "qudt:unit": "http://qudt.org/vocab/unit/DEG_C"
}
```

#### 7.4.5 sosa:FeatureOfInterest and sosa:ObservableProperty

The FeatureOfInterest is created from the document's `@id` and `@type`. It represents the real-world entity being observed (e.g., a building, a patient, a machine). The ObservableProperty is created from the property key of the annotated value (e.g., `http://schema.org/temperature`).

```json
{
  "@id": "http://example.org/building-1",
  "@type": ["sosa:FeatureOfInterest", "http://schema.org/Place"]
}
```

```json
{
  "@id": "http://schema.org/temperature",
  "@type": "sosa:ObservableProperty"
}
```

#### 7.4.6 ssn-system:SystemCapability and ssn-system:Accuracy

Created when measurement uncertainty or calibration metadata is present **and** a sensor node exists. The SystemCapability node groups instrument-level metadata and is linked to the sensor.

```json
{
  "@id": "_:cap-d3e4f5a6",
  "@type": "ssn-system:SystemCapability",
  "ssn-system:hasSystemProperty": {"@id": "_:acc-b7c8d9e0"},
  "jex:calibratedAt": "2026-01-01T00:00:00Z",
  "jex:calibrationMethod": "NIST traceable",
  "jex:calibrationAuthority": "https://www.nist.gov"
}
```

```json
{
  "@id": "_:acc-b7c8d9e0",
  "@type": "ssn-system:Accuracy",
  "jex:value": 0.5
}
```

The Accuracy node models the measurement uncertainty. Its `jex:value` carries the numeric uncertainty value from `@measurementUncertainty`. The SSN ontology defines `ssn-system:Accuracy` as a system property but does not prescribe a specific property for the numeric uncertainty value, so the `jex:` namespace is used.

### 7.5 Worked Example

**Input** (jsonld-ex — IoT sensor scenario):

```json
{
  "@context": ["http://schema.org/", "https://w3id.org/jsonld-ex/context/v1.jsonld"],
  "@type": "Place",
  "@id": "http://example.org/building-1",
  "http://schema.org/temperature": {
    "@value": 22.5,
    "@confidence": 0.95,
    "@source": "https://sensor.example.org/thermostat-1",
    "@extractedAt": "2026-01-15T10:30:00Z",
    "@method": "digital thermometer",
    "@unit": "http://qudt.org/vocab/unit/DEG_C",
    "@measurementUncertainty": 0.5,
    "@calibratedAt": "2026-01-01T00:00:00Z",
    "@calibrationMethod": "NIST traceable"
  }
}
```

**Output** (SSN/SOSA):

```json
{
  "@context": {
    "sosa": "http://www.w3.org/ns/sosa/",
    "ssn": "http://www.w3.org/ns/ssn/",
    "ssn-system": "http://www.w3.org/ns/ssn/systems/",
    "qudt": "http://qudt.org/schema/qudt/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "jsonld-ex": "http://www.w3.org/ns/jsonld-ex/"
  },
  "@graph": [
    {
      "@id": "http://example.org/building-1",
      "@type": "Place"
    },
    {
      "@id": "http://schema.org/temperature",
      "@type": "sosa:ObservableProperty"
    },
    {
      "@id": "_:result-c9d0e1f2",
      "@type": "sosa:Result",
      "qudt:numericValue": 22.5,
      "qudt:unit": "http://qudt.org/vocab/unit/DEG_C"
    },
    {
      "@id": "_:proc-e5f6a7b8",
      "@type": "sosa:Procedure",
      "rdfs:label": "digital thermometer"
    },
    {
      "@id": "_:acc-b7c8d9e0",
      "@type": "ssn-system:Accuracy",
      "jsonld-ex:value": 0.5
    },
    {
      "@id": "_:cap-d3e4f5a6",
      "@type": "ssn-system:SystemCapability",
      "ssn-system:hasSystemProperty": {"@id": "_:acc-b7c8d9e0"},
      "jsonld-ex:calibratedAt": "2026-01-01T00:00:00Z",
      "jsonld-ex:calibrationMethod": "NIST traceable"
    },
    {
      "@id": "https://sensor.example.org/thermostat-1",
      "@type": "sosa:Sensor",
      "ssn-system:hasSystemCapability": {"@id": "_:cap-d3e4f5a6"}
    },
    {
      "@id": "_:obs-a1b2c3d4",
      "@type": "sosa:Observation",
      "sosa:hasFeatureOfInterest": {"@id": "http://example.org/building-1"},
      "sosa:observedProperty": {"@id": "http://schema.org/temperature"},
      "sosa:hasResult": {"@id": "_:result-c9d0e1f2"},
      "sosa:resultTime": {"@value": "2026-01-15T10:30:00Z", "@type": "xsd:dateTime"},
      "sosa:madeBySensor": {"@id": "https://sensor.example.org/thermostat-1"},
      "sosa:usedProcedure": {"@id": "_:proc-e5f6a7b8"},
      "jsonld-ex:confidence": 0.95
    },
    {
      "@id": "http://example.org/building-1",
      "@type": ["sosa:FeatureOfInterest", "Place"]
    }
  ]
}
```

The jsonld-ex input is 11 lines of annotation content on a single value object. The SSN/SOSA output requires 8 graph nodes and approximately 24 triples to express the same sensor metadata. This expansion ratio (~3×) reflects the SSN/SOSA ontology's design: each aspect of an observation (the sensor, the procedure, the result, the capability, the accuracy) is modeled as a separate resource with explicit relationships.

### 7.6 Cross-Reference to Temporal Extensions

Sensor observations are inherently temporal. The `@extractedAt` annotation on a jsonld-ex value corresponds to `sosa:resultTime` on the observation. For richer temporal modeling:

- **Temporal validity** (`@validFrom`, `@validUntil`) can be combined with sensor annotations on the same value object. See [Temporal](temporal.md) §7.1 for composition rules.
- **PROV-O temporal properties** (`prov:generatedAtTime`, `prov:invalidatedAtTime`) provide entity lifecycle timestamps that are related but semantically distinct from observation result times. See [Temporal](temporal.md) §9.2.
- **OWL-Time** provides comprehensive temporal modeling (instants, intervals, durations, Allen's interval algebra) for applications that require more than simple validity intervals. See [Temporal](temporal.md) §9.1.

The SSN/SOSA ontology itself defines `sosa:phenomenonTime` (the time at which the observation applies to the feature of interest) in addition to `sosa:resultTime` (the time at which the result was produced). The current jsonld-ex mapping uses `@extractedAt` for `sosa:resultTime` only. Future extensions MAY add a dedicated keyword for `sosa:phenomenonTime` or map `@asOf` to it (see [Temporal](temporal.md) §3).

### 7.7 Unsupported SOSA Types

The following SOSA/SSN types are defined by the SSN ontology but have no jsonld-ex equivalent. When encountered during reverse mapping (`from_ssn`), nodes of these types are logged as warnings and dropped:

| SOSA Type | Reason for Non-Support |
|-----------|------------------------|
| `sosa:Platform` | Platforms describe the physical hosting of sensors (e.g., a weather station hosting multiple sensors). jsonld-ex models individual sensor annotations, not deployment topology. |
| `sosa:Actuator` | Actuators perform actions on features of interest (e.g., adjusting a thermostat). jsonld-ex models observations (read operations), not actuations (write operations). |
| `sosa:Actuation` | The event of an actuator performing an action. Not applicable to observational metadata. |
| `sosa:Sample` | Physical or statistical samples taken from a feature of interest. jsonld-ex does not model sampling methodology. |
| `sosa:Sampling` | The event of taking a sample. Not applicable to inline annotation metadata. |
| `sosa:Sampler` | The device or agent that performs sampling. Not applicable to inline annotation metadata. |

These limitations are inherent to the scope difference between jsonld-ex (assertion-level annotation) and SSN/SOSA (full sensor network modeling). Organizations that require platform modeling, actuation control, or sampling workflows SHOULD use SSN/SOSA directly; the jsonld-ex mapping targets the common observation-and-measurement subset that covers the majority of IoT data annotation use cases.

### 7.8 Round-Trip Guarantees

The following invariants hold for SSN/SOSA round-trips:

1. **Forward-then-reverse.** For any jsonld-ex document `D`, let `S = to_ssn(D)` and `D' = from_ssn(S)`. Then:
   - The `@value` literal MUST be identical.
   - `@source`, `@extractedAt`, and `@method` MUST be preserved through the Sensor, resultTime, and Procedure nodes respectively.
   - `@confidence` MUST be preserved via the `jex:confidence` annotation on the Observation node.
   - `@unit` MUST be preserved through the QUDT-typed Result node.
   - `@measurementUncertainty` MUST be preserved through the SystemCapability → Accuracy chain.
   - The document's `@id` and non-SOSA `@type` MUST be recoverable from the FeatureOfInterest node.

2. **Reverse-then-forward.** For any SSN/SOSA document `S` that was produced by `to_ssn`, let `D = from_ssn(S)` and `S' = to_ssn(D)`. Then:
   - Every `sosa:Observation` in `S` MUST have a corresponding observation in `S'` with equivalent `sosa:observedProperty`, result value, `sosa:resultTime`, and `sosa:madeBySensor`.
   - `jex:confidence` values MUST be preserved.

3. **Sensor deduplication stability.** If multiple annotated values in `D` reference the same `@source` IRI, the forward mapping creates a single `sosa:Sensor` node. The reverse mapping sets `@source` to the sensor's `@id` on each reconstructed value. The sensor identity is preserved, but the number of Sensor nodes in `S'` MAY differ from `S` if the intermediate document `D` was modified.

4. **Blank node identifiers.** Observation, procedure, result, capability, and accuracy nodes use generated blank node identifiers. Round-trip equivalence is semantic — blank node identifiers MAY differ between the input and output.

5. **Calibration metadata.** The `@calibratedAt`, `@calibrationMethod`, and `@calibrationAuthority` annotations are preserved in the `jex:` namespace on the SystemCapability node. The current reverse mapping does not extract calibration properties from capability nodes. This is a known asymmetry: calibration metadata survives a forward-then-reverse round-trip only if the reverse mapping is extended to read `jex:` properties from capability nodes.

6. **Aggregation metadata.** The `@aggregationMethod` becomes the procedure label, while `@aggregationWindow` and `@aggregationCount` are preserved as `jex:` properties on the Procedure node. The current reverse mapping extracts only the procedure label as `@method` and does not reconstruct the aggregation-specific keywords. This is a known asymmetry documented in §7.9.

### 7.9 Limitations

The SSN/SOSA mapping has the following known limitations:

1. **No SOSA equivalent for `@confidence`.** SOSA models measurement quality through system capabilities (accuracy, precision, sensitivity), not through belief strength. The confidence value is preserved in the `jex:` namespace on the Observation node, but a SOSA consumer that does not recognize the `jex:` namespace will ignore it.

2. **Calibration round-trip asymmetry.** The forward mapping places calibration metadata (`@calibratedAt`, `@calibrationMethod`, `@calibrationAuthority`) on the SystemCapability node using `jex:` properties. The reverse mapping does not currently extract these properties from capability nodes. A forward-then-reverse round-trip will lose calibration metadata. This is a known gap in the reference implementation.

3. **Aggregation round-trip asymmetry.** The forward mapping uses `@aggregationMethod` as the Procedure label and attaches `@aggregationWindow`/`@aggregationCount` as `jex:` properties on the Procedure node. The reverse mapping reconstructs only `@method` from the label and does not distinguish aggregation procedures from other procedures. A forward-then-reverse round-trip will convert `@aggregationMethod` to `@method` and lose `@aggregationWindow`/`@aggregationCount`.

4. **No `sosa:phenomenonTime` mapping.** SOSA distinguishes between `sosa:resultTime` (when the result was produced) and `sosa:phenomenonTime` (when the observation applies). jsonld-ex maps `@extractedAt` to `sosa:resultTime` only. The `@asOf` annotation from the temporal extensions ([Temporal](temporal.md) §3) is a candidate for future mapping to `sosa:phenomenonTime`, but this is not currently implemented.

5. **No `@humanVerified` mapping.** Human verification is a provenance concept (`prov:Person` attribution) rather than a sensor observation concept. It is not included in the SSN/SOSA mapping. Use the PROV-O mapping (§3) for human verification metadata.

6. **Single document scope.** The forward mapping processes a single document node. Documents with `@graph` arrays require per-node conversion. A batch variant analogous to `to_prov_o_graph` is not currently provided for SSN/SOSA.

7. **QUDT unit validation.** The forward mapping copies the `@unit` value to `qudt:unit` without validating that it is a recognized QUDT unit IRI. Invalid unit identifiers will be preserved faithfully but will not be resolvable against the QUDT ontology.

8. **Annotation keywords not mapped.** The following jsonld-ex annotation keywords have no SSN/SOSA mapping and are not preserved during conversion: `@derivedFrom`, `@delegatedBy`, `@invalidatedAt`, `@invalidationReason`, `@mediaType`, `@contentUrl`, `@contentHash`, `@translatedFrom`, `@translationModel`. These keywords are specific to provenance and content metadata use cases and are covered by the PROV-O mapping (§3) and RDF-star mapping (§6).

---

## 8. Croissant Mapping

The Croissant specification ([MLCommons Croissant 1.0](https://mlcommons.org/croissant/)) defines dataset-level metadata for ML datasets: distributions (FileObject, FileSet), record sets, fields, and data sources. jsonld-ex operates at a different level — assertion-level annotations on individual values within a document. The two formats are **complementary**, not competing: Croissant describes *what a dataset contains and how to access it*, while jsonld-ex describes *how much to trust each value and where it came from*.

This section specifies the bidirectional mapping between jsonld-ex dataset documents and Croissant-compatible JSON-LD. Unlike the preceding sections (§3–§7), which map between structurally different representations, the Croissant mapping is a **context swap** — the underlying JSON structure is preserved and only the `@context` and `conformsTo` declaration change.

### 8.1 Namespace Prefixes

The following namespace prefixes are used in this section in addition to those defined in §1.5:

| Prefix | IRI | Specification |
|--------|-----|---------------|
| `sc:` | `https://schema.org/` | [Schema.org](https://schema.org/) |
| `cr:` | `http://mlcommons.org/croissant/` | [Croissant 1.0](https://mlcommons.org/croissant/) |
| `dct:` | `http://purl.org/dc/terms/` | [Dublin Core Terms](http://purl.org/dc/terms/) |
| `rai:` | `http://mlcommons.org/croissant/RAI/` | Croissant Responsible AI vocabulary |

### 8.2 Complementarity Model

jsonld-ex and Croissant occupy distinct layers of the metadata stack:

| Concern | Croissant | jsonld-ex |
|---------|-----------|----------|
| Scope | Dataset / distribution / record set | Individual assertion / value |
| Primary question | "What does this dataset contain?" | "How confident is this value?" |
| Provenance granularity | Dataset-level (`creator`, `datePublished`) | Value-level (`@source`, `@extractedAt`) |
| Validation | Schema-level (`dataType` on fields) | Assertion-level (`@shape` on properties) |
| Trust model | None (assumes dataset is authoritative) | Confidence algebra (Subjective Logic) |
| Target consumer | Dataset search engines, ML platforms | Inference pipelines, knowledge fusion |

A single dataset document MAY carry both Croissant dataset-level metadata and jsonld-ex assertion-level annotations. The forward and reverse mappings defined below handle the `@context` and `conformsTo` declarations that distinguish the two formats.

### 8.3 Context Definitions

The jsonld-ex dataset context (`DATASET_CONTEXT`) maps a minimal set of Croissant terms alongside Schema.org:

```json
{
  "@vocab": "https://schema.org/",
  "sc": "https://schema.org/",
  "cr": "http://mlcommons.org/croissant/",
  "dct": "http://purl.org/dc/terms/",
  "citeAs": "cr:citeAs",
  "conformsTo": "dct:conformsTo",
  "recordSet": "cr:recordSet",
  "field": "cr:field",
  "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
  "source": "cr:source",
  "extract": "cr:extract",
  "fileObject": "cr:fileObject",
  "fileSet": "cr:fileSet",
  "isLiveDataset": "cr:isLiveDataset",
  "@language": "en"
}
```

The Croissant context (`CROISSANT_CONTEXT`) is a superset that includes the full Croissant vocabulary:

```json
{
  "@language": "en",
  "@vocab": "https://schema.org/",
  "sc": "https://schema.org/",
  "cr": "http://mlcommons.org/croissant/",
  "dct": "http://purl.org/dc/terms/",
  "rai": "http://mlcommons.org/croissant/RAI/",
  "citeAs": "cr:citeAs",
  "column": "cr:column",
  "conformsTo": "dct:conformsTo",
  "data": {"@id": "cr:data", "@type": "@json"},
  "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
  "examples": {"@id": "cr:examples", "@type": "@json"},
  "extract": "cr:extract",
  "field": "cr:field",
  "fileProperty": "cr:fileProperty",
  "fileObject": "cr:fileObject",
  "fileSet": "cr:fileSet",
  "format": "cr:format",
  "includes": "cr:includes",
  "isLiveDataset": "cr:isLiveDataset",
  "jsonPath": "cr:jsonPath",
  "key": "cr:key",
  "md5": "cr:md5",
  "parentField": "cr:parentField",
  "path": "cr:path",
  "recordSet": "cr:recordSet",
  "references": "cr:references",
  "regex": "cr:regex",
  "repeated": "cr:repeated",
  "replace": "cr:replace",
  "separator": "cr:separator",
  "source": "cr:source",
  "subField": "cr:subField",
  "transform": "cr:transform"
}
```

Because both contexts share the same `@vocab` (`https://schema.org/`) and map Croissant terms to the same IRIs, the expanded form of any dataset document is identical under either context for the subset of terms that both contexts define.

### 8.4 Forward Mapping (jsonld-ex → Croissant)

Given a jsonld-ex dataset document `D`, the forward mapping produces a Croissant-compatible document `C`:

1. Let `C` be a deep copy of `D`.
2. Replace the `@context` of `C` with the full `CROISSANT_CONTEXT`.
3. Set `C["conformsTo"]` to `"http://mlcommons.org/croissant/1.0"`.
4. Return `C`.

All other properties — `@type`, `name`, `description`, `distribution`, `recordSet`, and any jsonld-ex assertion-level annotations — are preserved without modification. A Croissant consumer that does not recognize jsonld-ex annotation keywords (e.g., `@confidence`, `@source`) will ignore them per the JSON-LD 1.1 processing rules for unknown keywords.

### 8.5 Reverse Mapping (Croissant → jsonld-ex)

Given a Croissant document `C`, the reverse mapping produces a jsonld-ex dataset document `D`:

1. Let `D` be a deep copy of `C`.
2. Replace the `@context` of `D` with `DATASET_CONTEXT`.
3. Remove the `conformsTo` property from `D` (this is a Croissant-specific declaration).
4. Return `D`.

Croissant terms that are present in `CROISSANT_CONTEXT` but absent from `DATASET_CONTEXT` (e.g., `column`, `jsonPath`, `regex`, `transform`, `md5`, `parentField`, `subField`, `references`, `format`, `replace`, `separator`, `repeated`, `key`, `fileProperty`, `data`, `examples`, `includes`, `path`) will lose their compact-form mappings in the resulting document. These terms will still be present in the JSON structure, but a JSON-LD processor expanding the document under `DATASET_CONTEXT` will not resolve them to their Croissant IRIs. See §8.8 for the implications of this asymmetry.

### 8.6 Worked Example

**jsonld-ex dataset document:**

```json
{
  "@context": { "...DATASET_CONTEXT..." },
  "@type": "sc:Dataset",
  "name": "ImageNet-Annotated",
  "description": "ImageNet subset with confidence annotations",
  "distribution": [
    {
      "@type": "cr:FileObject",
      "@id": "images.tar.gz",
      "name": "images.tar.gz",
      "contentUrl": "https://example.org/images.tar.gz",
      "encodingFormat": "application/x-tar"
    }
  ],
  "recordSet": [
    {
      "@type": "cr:RecordSet",
      "@id": "annotations",
      "name": "annotations",
      "field": [
        {
          "@type": "cr:Field",
          "@id": "annotations/label",
          "name": "label",
          "dataType": "sc:Text"
        }
      ]
    }
  ]
}
```

**After `to_croissant()`:**

```json
{
  "@context": { "...CROISSANT_CONTEXT..." },
  "@type": "sc:Dataset",
  "name": "ImageNet-Annotated",
  "description": "ImageNet subset with confidence annotations",
  "conformsTo": "http://mlcommons.org/croissant/1.0",
  "distribution": [ "...unchanged..." ],
  "recordSet": [ "...unchanged..." ]
}
```

The only structural changes are the `@context` replacement and the addition of `conformsTo`. All dataset content is preserved verbatim.

### 8.7 Migration Path

Organizations with existing Croissant datasets can adopt jsonld-ex incrementally:

1. **Import.** Call `from_croissant(existing_dataset)` to obtain a jsonld-ex dataset document.
2. **Annotate.** Add assertion-level annotations to individual values using `annotate()` and related functions.
3. **Export.** Call `to_croissant(annotated_dataset)` to produce a Croissant-compatible document that retains the assertion-level annotations as opaque JSON properties.

Croissant consumers that do not implement jsonld-ex will ignore the annotation keywords. jsonld-ex consumers will process both the dataset-level and assertion-level metadata.

### 8.8 Round-Trip Guarantees

1. **Forward-then-reverse.** For any jsonld-ex dataset document `D`, let `C = to_croissant(D)` and `D' = from_croissant(C)`. Then `D'` is semantically equivalent to `D`: all properties, values, and annotations are preserved. The `@context` of `D'` will be `DATASET_CONTEXT` (matching `D`), and the `conformsTo` property added by the forward mapping will be removed.

2. **Reverse-then-forward.** For any Croissant document `C`, let `D = from_croissant(C)` and `C' = to_croissant(D)`. Then `C'` preserves all properties from `C` and restores the `conformsTo` declaration. However, the `@context` of `C'` will be the jsonld-ex `CROISSANT_CONTEXT` constant, which MAY differ from the original `@context` of `C` if `C` used a custom or versioned Croissant context.

3. **Annotation preservation.** jsonld-ex annotation keywords (`@confidence`, `@source`, `@extractedAt`, etc.) are JSON properties that pass through both directions without modification. They are not consumed or transformed by either mapping function.

### 8.9 Limitations

The Croissant mapping has the following known limitations:

1. **Context-only mapping.** The forward and reverse mappings perform a context swap and `conformsTo` toggle. They do not perform structural transformation, semantic validation, or term resolution. This is by design — the two formats share the same underlying JSON-LD data model.

2. **Croissant-only terms lose context.** Croissant terms that are defined in `CROISSANT_CONTEXT` but not in `DATASET_CONTEXT` (e.g., `column`, `jsonPath`, `regex`, `transform`) will not be resolvable to their Croissant IRIs when the document is processed under `DATASET_CONTEXT`. The JSON properties remain in the document, but their semantic meaning is lost to a strict JSON-LD processor. To preserve full Croissant semantics after a reverse mapping, consumers SHOULD use the Croissant context or an extended context that includes the missing term definitions.

3. **Custom Croissant contexts.** If a Croissant document uses a context other than the standard Croissant context (e.g., a versioned or extended context), the reverse-then-forward round-trip will replace it with the jsonld-ex `CROISSANT_CONTEXT` constant. The original context is not preserved.

4. **No Responsible AI (RAI) mapping.** The Croissant RAI vocabulary (`rai:` prefix) defines terms for documenting ethical considerations, bias assessments, and use restrictions. jsonld-ex does not currently map these terms to its own vocabulary. RAI properties are preserved as opaque JSON during conversion but have no jsonld-ex semantic equivalent.

5. **No cross-layer validation.** The mapping does not verify consistency between Croissant dataset-level metadata (e.g., `dataType` on a field) and jsonld-ex assertion-level annotations (e.g., `@shape` constraints on individual values). Cross-layer validation is left to the consuming application.

---

## 9. Verbosity Metrics Framework

A central claim of jsonld-ex is that inline annotations reduce verbosity compared to the equivalent representation in each target standard. This section specifies the measurement framework used to substantiate that claim. All verbosity comparisons in §3–§7 are computed using the data model and algorithms defined here.

### 9.1 VerbosityComparison Data Model

A verbosity comparison is represented as a record with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `jsonld_ex_triples` | integer | Approximate triple count for the jsonld-ex representation. |
| `alternative_triples` | integer | Triple count for the alternative representation (e.g., PROV-O, SHACL). |
| `jsonld_ex_bytes` | integer | Byte count of the jsonld-ex representation serialized as indented JSON (UTF-8). |
| `alternative_bytes` | integer | Byte count of the alternative representation serialized as indented JSON (UTF-8). |
| `triple_reduction_pct` | float | Percentage reduction in triple count: `(alternative_triples - jsonld_ex_triples) / alternative_triples × 100`. Positive values indicate jsonld-ex uses fewer triples. |
| `byte_reduction_pct` | float | Percentage reduction in byte count: `(alternative_bytes - jsonld_ex_bytes) / alternative_bytes × 100`. Positive values indicate jsonld-ex uses fewer bytes. |
| `alternative_name` | string | Name of the alternative standard (e.g., `"PROV-O"`, `"SHACL"`). |

If the alternative triple count or byte count is zero, the corresponding reduction percentage MUST be `0.0`.

### 9.2 Triple Counting Methodology

Triple counts are approximations of the number of RDF triples that a JSON-LD processor would produce from the expanded form of the document. The counting algorithms are intentionally simple — they provide a consistent basis for relative comparison, not an exact RDF serialization count.

#### 9.2.1 jsonld-ex Triple Counting

For a jsonld-ex document, the triple count is computed as follows:

1. For each key-value pair in the document (excluding `@context`, `@id`, and JSON-LD structural keywords other than `@type`):
   a. If the key is `@type`, count 1 triple (the `rdf:type` statement).
   b. If the value is a dictionary containing `@value`, count 1 triple for the base assertion. Annotation keywords (`@confidence`, `@source`, `@extractedAt`, etc.) within the value object are **not** counted as separate triples — this reflects the design principle that inline annotations do not increase the triple count.
   c. If the value is a list, count 1 triple per list element.
   d. Otherwise, count 1 triple.

This counting method captures the key advantage of jsonld-ex: annotation metadata is encoded as properties of the value node rather than as separate triples.

#### 9.2.2 SHACL Triple Counting

For a SHACL shapes graph, the triple count is computed by iterating over the `@graph` array:

1. For each node in the graph:
   a. Skip `@id` (not a triple).
   b. If the key is `@type`, count 1 triple.
   c. If the value is a list, count each element: dictionaries contribute 1 triple per key (excluding `@id`); scalars contribute 1 triple each.
   d. Otherwise, count 1 triple.

SHACL shapes are inherently verbose because each constraint (minCount, maxCount, datatype, pattern, etc.) is a separate triple on a property shape blank node.

### 9.3 Byte Counting Methodology

Byte counts are computed by serializing the document as JSON with 2-space indentation and encoding the result as UTF-8:

```
bytes = len(json.dumps(document, indent=2).encode("utf-8"))
```

Indented serialization is used rather than compact serialization because it reflects the format developers actually read and store. The 2-space indent is a fixed convention across all measurements to ensure comparability.

### 9.4 Comparison Algorithms

#### 9.4.1 compare_with_prov_o

Given a jsonld-ex annotated document `D`:

1. Convert `D` to PROV-O using `to_prov_o(D)` to obtain `(P, report)`.
2. Compute `jsonld_ex_triples` using the jsonld-ex triple counting algorithm (§9.2.1) on `D`.
3. Set `alternative_triples` to `report.triples_output` (the triple count reported by the PROV-O conversion).
4. Compute `jsonld_ex_bytes` and `alternative_bytes` using the byte counting methodology (§9.3) on `D` and `P` respectively.
5. Compute reduction percentages per the formulas in §9.1.
6. Return a `VerbosityComparison` with `alternative_name` set to `"PROV-O"`.

#### 9.4.2 compare_with_shacl

Given a jsonld-ex `@shape` definition `S`:

1. Convert `S` to a SHACL shapes graph using `shape_to_shacl(S)` to obtain `H`.
2. Compute `jsonld_ex_triples` as the sum of constraint counts across all properties in `S`: for each property value that is a dictionary (excluding the `@type` entry of the shape itself), count the number of keys in that dictionary.
3. Compute `alternative_triples` using the SHACL triple counting algorithm (§9.2.2) on `H`.
4. Compute `jsonld_ex_bytes` on `{"@shape": S}` and `alternative_bytes` on `H` using the byte counting methodology (§9.3).
5. Compute reduction percentages per the formulas in §9.1.
6. Return a `VerbosityComparison` with `alternative_name` set to `"SHACL"`.

### 9.5 Summary of Verbosity Comparisons

The following table summarizes the verbosity characteristics across all six interoperability targets. Entries marked "N/A" indicate that the mapping does not lend itself to meaningful triple-count comparison (e.g., context swap mappings or format-level mappings).

| Standard | Mapping type | Triple comparison | Byte comparison | Measurement function |
|----------|-------------|-------------------|-----------------|---------------------|
| PROV-O | Structural | jsonld-ex uses fewer triples (annotations are inline vs. separate entities) | jsonld-ex uses fewer bytes (no agent/activity/entity boilerplate) | `compare_with_prov_o()` |
| SHACL | Structural | jsonld-ex uses fewer triples (constraints are inline vs. separate property shape nodes) | jsonld-ex uses fewer bytes (no blank node overhead) | `compare_with_shacl()` |
| OWL | Structural | Varies by constraint complexity; simple type constraints are comparable, complex combinators (union/intersection) expand significantly in OWL | jsonld-ex uses fewer bytes for typical validation constraints | Not provided (OWL restrictions are not full documents) |
| RDF-star | Serialization | jsonld-ex inline annotations map 1:1 to RDF-star annotations; triple counts are comparable | N-Triples serialization is significantly more verbose than JSON-LD; Turtle is comparable | Not provided (comparison is format-level, not semantic) |
| SSN/SOSA | Structural | jsonld-ex uses fewer triples (one annotated value vs. Observation + Result + Sensor + Procedure subgraph) | jsonld-ex uses fewer bytes (no observation boilerplate) | Not provided (SSN/SOSA output is a graph, not a single document) |
| Croissant | Context swap | N/A (same structure, different context) | N/A (byte difference is only the context and conformsTo overhead) | Not provided (comparison is not meaningful) |

### 9.6 Limitations

1. **Approximate counts.** Triple counts are approximations. They do not account for blank node triples generated during JSON-LD expansion, RDF list encoding overhead, or datatype triples for typed literals. The counts are suitable for relative comparison but SHOULD NOT be cited as exact RDF triple counts.

2. **Two standards measured.** The reference implementation provides `compare_with_prov_o()` and `compare_with_shacl()` as automated measurement functions. Verbosity comparisons for OWL, RDF-star, SSN/SOSA, and Croissant are documented qualitatively in their respective sections but do not have dedicated measurement functions.

3. **Serialization dependence.** Byte counts depend on JSON serialization conventions (indent size, key ordering). All measurements use `json.dumps(doc, indent=2)` with default key ordering for consistency, but comparison with non-JSON serializations (Turtle, N-Triples) requires format-specific measurement.

---

## 10. Conversion Report Model

All structural mapping functions (§3–§5, §7) return a `ConversionReport` alongside the converted document. The report provides metadata about the conversion process for auditing, debugging, and quality assessment.

### 10.1 Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | `true` if the conversion completed without fatal errors. |
| `nodes_converted` | integer | Number of document nodes (annotated values, shape properties, etc.) that were mapped to the target representation. Default: `0`. |
| `triples_input` | integer | Approximate triple count of the input representation. Default: `0`. |
| `triples_output` | integer | Approximate triple count of the output representation. Default: `0`. |
| `warnings` | list of strings | Non-fatal issues encountered during conversion (e.g., unmappable keywords preserved in the `jex:` namespace). Default: empty list. |
| `errors` | list of strings | Fatal issues that prevented complete conversion. If non-empty, `success` SHOULD be `false`. Default: empty list. |

### 10.2 Derived Properties

The `compression_ratio` is a derived property computed as:

```
compression_ratio = triples_output / triples_input
```

If `triples_input` is `0`, the compression ratio MUST be `0.0`. A compression ratio greater than `1.0` indicates that the output is more verbose than the input (expansion); a ratio less than `1.0` indicates compression.

### 10.3 Usage Across Mappings

The following mapping functions return a `ConversionReport` as the second element of their return tuple:

| Function | Section | Notes |
|----------|---------|-------|
| `to_prov_o()` | §3 | Reports annotated values converted to PROV-O entities. |
| `to_prov_o_graph()` | §3 | Reports nodes across a `@graph` array. |
| `from_prov_o()` | §3 | Reports PROV-O entities converted back to annotated values. |
| `shape_to_shacl()` | §4 | Returns the SHACL graph only (no report). |
| `shacl_to_shape()` | §4 | Returns the shape only (no report). |
| `shape_to_owl_restrictions()` | §5 | Returns the OWL restriction only (no report). |
| `owl_to_shape()` | §5 | Returns the shape only (no report). |
| `to_rdf_star_ntriples()` | §6 | Returns a string (no report). |
| `from_rdf_star_ntriples()` | §6 | Returns a document (no report). |
| `to_ssn()` | §7 | Reports annotated values converted to SSN/SOSA observations. |
| `from_ssn()` | §7 | Reports SSN/SOSA observations converted back to annotated values. |

The SHACL, OWL, and RDF-star mapping functions do not return conversion reports because they operate on individual shapes or serialization units rather than full documents with countable node populations. Future versions of the specification MAY extend these functions to return reports.

---

## 11. Cross-References

The interoperability mappings defined in this document interact with definitions in other jsonld-ex specifications. This section provides a consolidated cross-reference index.

### 11.1 Validation Specification

- [Validation](validation.md) §15.1 defines the relationship between jsonld-ex `@shape` validation and SHACL. The SHACL mapping (§4 of this document) implements the bidirectional conversion described there.
- [Validation](validation.md) §15.4 contains the SHACL constraint mapping table. The SHACL mapping (§4) cross-references this table rather than duplicating it.

### 11.2 Temporal Specification

- [Temporal](temporal.md) §9.1 documents the relationship between jsonld-ex temporal extensions (`@validFrom`, `@validUntil`, `@asOf`) and OWL Time. The SSN/SOSA mapping (§7.6) cross-references this for the `@asOf` → `sosa:phenomenonTime` candidate mapping.
- [Temporal](temporal.md) §9.2 documents the relationship between jsonld-ex temporal annotations and PROV-O temporal properties (`prov:generatedAtTime`, `prov:invalidatedAtTime`). The PROV-O mapping (§3) implements the `@extractedAt` → `prov:generatedAtTime` conversion.
- [Temporal](temporal.md) §9.3 documents the relationship between jsonld-ex temporal annotations and Schema.org temporal properties.

### 11.3 Confidence Algebra Specification

- [Confidence Algebra](confidence-algebra.md) §13 proves bridge theorems between scalar confidence values and Subjective Logic opinions. The PROV-O mapping (§3) preserves `@confidence` as a `jex:confidence` annotation because PROV-O has no native trust model; the confidence algebra specification defines the formal semantics of the preserved value.

### 11.4 Vocabulary Specification

- [Vocabulary](vocabulary.md) defines all annotation keywords (`@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`, etc.) and their formal semantics. The mapping tables in §3–§7 reference these keyword definitions.

---

## 12. Backward Compatibility

All interoperability mappings are designed to degrade gracefully when processed by JSON-LD 1.1 processors that do not implement the jsonld-ex extensions.

### 12.1 Non-Extended Processors

A standard JSON-LD 1.1 processor encountering a jsonld-ex annotated document will:

1. **Ignore unknown keywords.** Keywords beginning with `@` that are not defined in the JSON-LD 1.1 specification (e.g., `@confidence`, `@source`, `@extractedAt`) are ignored during expansion. The `@value` of an annotated value node is preserved.
2. **Process known keywords normally.** `@context`, `@type`, `@id`, `@value`, and all standard JSON-LD keywords are processed according to the JSON-LD 1.1 specification.
3. **Produce valid RDF.** The expanded and flattened output of a jsonld-ex document under a standard processor is valid RDF. Annotation metadata is silently discarded, but the base assertions are preserved.

### 12.2 Non-Extended Consumers of Converted Output

The output of each forward mapping is designed for consumption by tools that implement the target standard:

| Target | Consumer behavior |
|--------|------------------|
| PROV-O | Any PROV-O consumer (e.g., ProvStore, PROV Toolbox) can process the output. The `jex:confidence` annotation on Entity nodes is a standard RDF property and will be preserved but may not be understood semantically. |
| SHACL | Any SHACL validator (e.g., pySHACL, TopBraid) can process the output shapes graph. Constraints use standard SHACL vocabulary. |
| OWL | Any OWL reasoner (e.g., HermiT, Pellet) can process the output class restrictions. Unmappable constraints preserved in the `jex:` namespace will be treated as unknown annotation properties. |
| RDF-star | Any RDF-star-aware triple store (e.g., Oxigraph, Blazegraph) can process the output. Annotation keywords are mapped to standard `jex:` namespace IRIs. |
| SSN/SOSA | Any SSN/SOSA consumer (e.g., FIWARE, SensorThings) can process the output observations. The `jex:confidence` annotation on Observation nodes uses a standard RDF property. |
| Croissant | Any Croissant consumer (e.g., Hugging Face Datasets, TensorFlow Datasets) can process the output. jsonld-ex annotation keywords are opaque JSON properties that are ignored. |

### 12.3 Versioning

The interoperability mappings defined in this document are versioned with the specification. The mapping algorithms are stable across patch versions (e.g., v0.1.0 → v0.1.1) and SHOULD remain backward compatible across minor versions (e.g., v0.1.x → v0.2.x). Breaking changes to mapping algorithms MUST be documented in the specification changelog and MUST increment the minor or major version number.

---

## 13. Reference Implementation

The reference implementation is the `jsonld-ex` Python package ([PyPI](https://pypi.org/project/jsonld-ex/), [GitHub](https://github.com/jemsbhai/jsonld-ex)). All interoperability functions are exported from the top-level `jsonld_ex` module.

### 13.1 Interoperability API

| Function | Module | Section | Description |
|----------|--------|---------|-------------|
| `to_prov_o(doc)` | `owl_interop` | §3 | Convert annotated document to PROV-O graph. Returns `(doc, ConversionReport)`. |
| `to_prov_o_graph(doc)` | `owl_interop` | §3 | Convert document with `@graph` array to PROV-O. Returns `(doc, ConversionReport)`. |
| `from_prov_o(prov_doc)` | `owl_interop` | §3 | Convert PROV-O graph to annotated document. Returns `(doc, ConversionReport)`. |
| `shape_to_shacl(shape)` | `owl_interop` | §4 | Convert `@shape` to SHACL shapes graph. Returns `dict`. |
| `shacl_to_shape(shacl_doc)` | `owl_interop` | §4 | Convert SHACL shapes graph to `@shape`. Returns `dict`. |
| `shape_to_owl_restrictions(shape)` | `owl_interop` | §5 | Convert `@shape` to OWL class restrictions. Returns `dict`. |
| `owl_to_shape(owl_doc)` | `owl_interop` | §5 | Convert OWL class restrictions to `@shape`. Returns `dict`. |
| `to_rdf_star_ntriples(doc)` | `owl_interop` | §6 | Serialize annotated document as RDF-star N-Triples. Returns `str`. |
| `from_rdf_star_ntriples(text)` | `owl_interop` | §6 | Parse RDF-star N-Triples to annotated document. Returns `dict`. |
| `to_rdf_star_turtle(doc)` | `owl_interop` | §6 | Serialize annotated document as RDF-star Turtle. Returns `str`. |
| `to_ssn(doc)` | `owl_interop` | §7 | Convert annotated document to SSN/SOSA observations. Returns `(doc, ConversionReport)`. |
| `from_ssn(ssn_doc)` | `owl_interop` | §7 | Convert SSN/SOSA observations to annotated document. Returns `(doc, ConversionReport)`. |
| `to_croissant(dataset)` | `dataset` | §8 | Convert jsonld-ex dataset to Croissant format. Returns `dict`. |
| `from_croissant(croissant_doc)` | `dataset` | §8 | Convert Croissant document to jsonld-ex format. Returns `dict`. |

### 13.2 Metrics API

| Function | Module | Section | Description |
|----------|--------|---------|-------------|
| `compare_with_prov_o(doc)` | `owl_interop` | §9 | Measure verbosity of jsonld-ex vs. PROV-O. Returns `VerbosityComparison`. |
| `compare_with_shacl(shape)` | `owl_interop` | §9 | Measure verbosity of jsonld-ex `@shape` vs. SHACL. Returns `VerbosityComparison`. |

### 13.3 Data Structures

| Class | Module | Section | Description |
|-------|--------|---------|-------------|
| `ConversionReport` | `owl_interop` | §10 | Report from a conversion operation. |
| `VerbosityComparison` | `owl_interop` | §9 | Verbosity comparison metrics. |

### 13.4 Dataset Metadata API

| Function | Module | Description |
|----------|--------|-------------|
| `create_dataset_metadata(name, ...)` | `dataset` | Create a jsonld-ex dataset document with Schema.org/Croissant metadata. |
| `validate_dataset_metadata(doc)` | `dataset` | Validate a dataset document against the dataset shape. |
| `add_distribution(dataset, ...)` | `dataset` | Add a `cr:FileObject` to the dataset. |
| `add_file_set(dataset, ...)` | `dataset` | Add a `cr:FileSet` to the dataset. |
| `add_record_set(dataset, ...)` | `dataset` | Add a `cr:RecordSet` with fields to the dataset. |
| `create_field(name, ...)` | `dataset` | Create a `cr:Field` definition. |

### 13.5 Context Constants

| Constant | Module | Description |
|----------|--------|-------------|
| `DATASET_CONTEXT` | `dataset` | jsonld-ex dataset context (Schema.org + minimal Croissant terms). |
| `CROISSANT_CONTEXT` | `dataset` | Full Croissant context (all Croissant vocabulary terms). |
| `DATASET_SHAPE` | `dataset` | Validation shape for dataset metadata documents. |

---

## References

The following normative and informative references are cited in this specification.

### Normative References

- **[JSON-LD 1.1]** M. Sporny, D. Longley, G. Kellogg, M. Lanthaler, P.-A. Champin. *JSON-LD 1.1: A JSON-based Serialization for Linked Data.* W3C Recommendation, 16 July 2020. https://www.w3.org/TR/json-ld11/

- **[PROV-O]** T. Lebo, S. Sahoo, D. McGuinness. *PROV-O: The PROV Ontology.* W3C Recommendation, 30 April 2013. https://www.w3.org/TR/prov-o/

- **[SHACL]** H. Knublauch, D. Kontokostas. *Shapes Constraint Language (SHACL).* W3C Recommendation, 20 July 2017. https://www.w3.org/TR/shacl/

- **[OWL 2]** W3C OWL Working Group. *OWL 2 Web Ontology Language Document Overview (Second Edition).* W3C Recommendation, 11 December 2012. https://www.w3.org/TR/owl2-overview/

- **[RDF 1.2 Concepts]** R. Cyganiak, D. Wood, M. Lanthaler. *RDF 1.2 Concepts and Abstract Syntax.* W3C Working Draft. https://www.w3.org/TR/rdf12-concepts/

- **[SSN/SOSA]** A. Haller, K. Janowicz, S. Cox, D. Le Phuoc, K. Taylor, M. Lefrançois. *Semantic Sensor Network Ontology.* W3C Recommendation, 19 October 2017. https://www.w3.org/TR/vocab-ssn/

- **[RFC 2119]** S. Bradner. *Key words for use in RFCs to Indicate Requirement Levels.* IETF, March 1997. https://www.rfc-editor.org/rfc/rfc2119

- **[RFC 8174]** B. Leiba. *Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words.* IETF, May 2017. https://www.rfc-editor.org/rfc/rfc8174

### Informative References

- **[Croissant]** MLCommons. *Croissant: A Metadata Format for ML-Ready Datasets.* Version 1.0. https://mlcommons.org/croissant/

- **[QUDT]** QUDT.org. *Quantities, Units, Dimensions, and Types Ontology.* http://qudt.org/

- **[Jøsang 2016]** A. Jøsang. *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer, 2016. ISBN 978-3-319-42335-7.

- **[Schema.org]** Schema.org Community Group. *Schema.org Vocabulary.* https://schema.org/

- **[Dublin Core]** Dublin Core Metadata Initiative. *DCMI Metadata Terms.* https://www.dublincore.org/specifications/dublin-core/dcmi-terms/

- **[OWL-Time]** S. Cox, C. Little. *Time Ontology in OWL.* W3C Recommendation, 19 October 2017. https://www.w3.org/TR/owl-time/

