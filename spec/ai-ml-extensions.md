# JSON-LD Extensions for AI/ML — Specification Overview

**Status:** Draft  
**Version:** 0.2.0  
**Authors:** Muntaser Syed, Marius Silaghi, Sheikh Abujar, Rwaida Alssadi  
**Affiliation:** Florida Institute of Technology  
**Date:** 2026-02-14  
**Canonical URL:** [https://jsonld-ex.github.io/ns/](https://jsonld-ex.github.io/ns/)  
**Reference Implementation:** [PyPI](https://pypi.org/project/jsonld-ex/) · [GitHub](https://github.com/jemsbhai/jsonld-ex)

---

## 1. Introduction

This specification defines backward-compatible extensions to JSON-LD 1.1 that enable AI/ML metadata to be embedded alongside symbolic linked data. The extensions address five critical gaps in the current JSON-LD ecosystem:

1. **Confidence & Uncertainty** — No standard way to express prediction uncertainty, distinguish informed confidence from ignorance, or compose confidence across sources.
2. **Provenance Tracking** — No standard inline attribution for AI-generated assertions, including source identity, extraction timestamps, methods, and human verification.
3. **Temporal Validity** — No mechanism for time-bounded assertions, point-in-time queries, or temporal differencing of knowledge graphs.
4. **Security** — No defense against context injection attacks, resource exhaustion, or unauthorized remote context loading.
5. **Validation** — No JSON-LD-native constraint language for validating structure, types, cardinalities, and relationships without external tools.

The extensions are organized as a suite of companion specifications, each addressing a distinct concern. This document provides the overview, defines the core annotation keywords, and serves as the entry point for the specification suite.

### 1.1 Design Principles

The extension suite is governed by four principles:

**Backward compatibility.** All extensions use the `@`-keyword convention. Standard JSON-LD 1.1 processors treat unrecognized `@`-keywords as opaque properties, preserving the `@value` literal. No existing JSON-LD documents are affected.

**Minimal overhead.** Annotation keywords are inline on value objects — no separate graph nodes, no reification boilerplate. A single annotated value is one JSON object, not five.

**Mathematical rigor.** The confidence model is grounded in Jøsang's Subjective Logic, with formal proofs of algebraic properties and bridge theorems to classical methods. Claims are backed by reproducible benchmarks.

**Interoperability.** Bidirectional mappings to six W3C/community standards (PROV-O, SHACL, OWL, RDF-star, SSN/SOSA, Croissant) ensure that jsonld-ex data integrates with existing semantic web and ML toolchains.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

---

## 2. Specification Suite

The full specification is organized into the following companion documents:

| Document | Scope | Status |
|----------|-------|--------|
| **[Vocabulary](vocabulary.md)** | Complete vocabulary: all classes, properties, IRIs, types, ranges, and formal definitions for every jsonld-ex term. | Draft v0.1.0 |
| **[Confidence Algebra](confidence-algebra.md)** | Formal confidence algebra grounded in Subjective Logic: Opinion space, fusion operators, trust discount, deduction, conflict metrics, robust fusion, temporal decay, and bridge theorems. | Draft v0.1.0 |
| **[Temporal Extensions](temporal.md)** | `@validFrom`, `@validUntil`, `@asOf` — time-bounded assertions, point-in-time queries, temporal differencing, and interaction with confidence decay. | Draft v0.1.0 |
| **[Validation Extensions](validation.md)** | `@shape` — JSON-LD-native constraint language with 22 keywords covering types, cardinalities, ranges, patterns, enumerations, logical combinators, conditionals, cross-property constraints, and shape inheritance. | Draft v0.1.0 |
| **[Security Extensions](security.md)** | `@integrity` — context integrity verification via cryptographic hashes, context allowlists, and resource limits defending against context injection and resource exhaustion. | Draft v0.1.0 |
| **[Interoperability](interoperability.md)** | Bidirectional mappings to PROV-O, SHACL, OWL, RDF-star, SSN/SOSA, and Croissant with round-trip guarantees, verbosity metrics, and honest limitation accounting. | Draft v0.1.0 |
| **[Transport Extensions](transport.md)** | CBOR-LD binary serialization and MQTT transport optimization for IoT and bandwidth-constrained networks. | Draft v0.1.0 |

Each companion document is self-contained but cross-references related specifications where semantics interact (e.g., temporal extensions provide timestamps for the confidence algebra's decay operators).

---

## 3. Core Annotation Keywords

This section defines the five core annotation keywords. The complete vocabulary of all jsonld-ex terms (22 annotation keywords, 22 validation keywords, Opinion properties, and data protection terms) is specified in the [Vocabulary](vocabulary.md) document.

### 3.1 @confidence

**IRI:** `https://w3id.org/jsonld-ex/confidence`  
**Type:** `xsd:double`  
**Range:** `[0.0, 1.0]`  
**Applies to:** Value objects (`@value`)

Indicates the confidence or certainty of an assertion. A value of `1.0` indicates absolute certainty; `0.0` indicates no confidence. When a full Subjective Logic Opinion is available, the scalar confidence is the projected probability P(ω) = b + a·u. See the [Confidence Algebra](confidence-algebra.md) specification for the formal treatment.

```json
{
  "name": {
    "@value": "John Smith",
    "@confidence": 0.95
  }
}
```

**Processing rules:**

- Processors MUST validate that `@confidence` is a number in `[0.0, 1.0]`.
- `@confidence` MUST only appear on value objects containing `@value`.
- During expansion, `@confidence` maps to `https://w3id.org/jsonld-ex/confidence`.
- During RDF conversion, confidence is preserved as a reified annotation (see [RDF-star mapping](interoperability.md) §6).
- `@confidence` MAY alternatively hold an Opinion object (`{"@type": "Opinion", "belief": ..., "disbelief": ..., "uncertainty": ..., "baseRate": ...}`) for full uncertainty decomposition.

### 3.2 @source

**IRI:** `https://w3id.org/jsonld-ex/source`  
**Type:** `@id` (IRI)  
**Applies to:** Value objects

Identifies the system, model, or agent that produced the assertion.

```json
{
  "name": {
    "@value": "John Smith",
    "@confidence": 0.95,
    "@source": "https://ml-model.example.org/ner-v2"
  }
}
```

### 3.3 @extractedAt

**IRI:** `https://w3id.org/jsonld-ex/extractedAt`  
**Type:** `xsd:dateTime`  
**Applies to:** Value objects

ISO 8601 timestamp indicating when the value was extracted or generated. This serves as the reference time for the confidence algebra's temporal decay operators (see [Confidence Algebra](confidence-algebra.md) §12) and maps to `prov:generatedAtTime` in the PROV-O interoperability mapping (see [Interoperability](interoperability.md) §3).

### 3.4 @method

**IRI:** `https://w3id.org/jsonld-ex/method`  
**Type:** `xsd:string`  
**Applies to:** Value objects

Describes the extraction method (e.g., `"NER"`, `"classification"`, `"regression"`).

### 3.5 @humanVerified

**IRI:** `https://w3id.org/jsonld-ex/humanVerified`  
**Type:** `xsd:boolean`  
**Applies to:** Value objects

Indicates whether a human has reviewed and verified the assertion.

---

## 4. Vector Embedding Container

### 4.1 @vector Container Type

A new container type `@vector` indicates that a property holds a dense vector embedding. Vectors are treated as opaque annotations and are NOT converted to RDF triples.

```json
{
  "@context": {
    "embedding": {
      "@id": "http://example.org/embedding",
      "@container": "@vector",
      "@dimensions": 768
    }
  },
  "@type": "Product",
  "name": "Widget",
  "embedding": [0.123, -0.456, 0.789, "..."]
}
```

### 4.2 @dimensions

**Type:** `xsd:integer`  
**Applies to:** Term definitions with `@container: @vector`

Optional dimensionality constraint. Processors SHOULD validate vector length against `@dimensions` during expansion.

### 4.3 Processing Rules

- During expansion: vector values are preserved as JSON arrays.
- During compaction: vector values are preserved without modification.
- During RDF conversion: vector properties are EXCLUDED (annotation-only).
- During CBOR-LD encoding: vectors use efficient binary float arrays (see [Transport](transport.md)).

---

## 5. Context Definition

The extensions are defined under the proposed namespace:

| Property | Value |
|----------|-------|
| **Proposed Namespace IRI** | `https://w3id.org/jsonld-ex/` |
| **Preferred prefix** | `jex:` |
| **Current hosting** | `https://jsonld-ex.github.io/ns/` |

The `w3id.org` permanent identifier has not yet been registered. Until registration, the namespace resolves via the GitHub Pages URL above.

Documents using these extensions SHOULD include the context:

```json
{
  "@context": [
    "https://schema.org/",
    "https://jsonld-ex.github.io/ns/context/v1.jsonld"
  ]
}
```

---

## 6. Backward Compatibility

These extensions are fully backward compatible with JSON-LD 1.1:

- Standard processors treat unrecognized `@`-keywords as opaque properties, preserving the `@value` literal.
- Extended processors interpret annotation keywords with the defined semantics, gaining access to confidence decomposition, temporal validity, and validation.
- No existing JSON-LD documents are affected.
- The extensions use the established `@`-keyword convention per JSON-LD 1.1 §4.1.

Scalar `@confidence` values are a degenerate case of the full Opinion model (u = 0). Non-extended processors can use scalar confidence directly; extended processors can upgrade to Opinion objects for richer uncertainty modeling. See [Confidence Algebra](confidence-algebra.md) §14.5 for the backward compatibility design.

---

## 7. Use Cases

### 7.1 Healthcare Wearable Data

```json
{
  "@context": ["https://schema.org/", "https://jsonld-ex.github.io/ns/context/v1.jsonld"],
  "@type": "MedicalObservation",
  "posture": {
    "@value": "forward-head",
    "@confidence": 0.87,
    "@source": "https://device.example.org/imu-classifier-v3",
    "@extractedAt": "2026-01-15T14:30:00Z",
    "@method": "IMU-6axis-classification"
  }
}
```

### 7.2 Knowledge Graph Extraction

```json
{
  "@context": ["https://schema.org/", "https://jsonld-ex.github.io/ns/context/v1.jsonld"],
  "@type": "Person",
  "name": {
    "@value": "Jane Doe",
    "@confidence": 0.98,
    "@source": "https://model.example.org/ner-v4",
    "@humanVerified": true
  },
  "worksFor": {
    "@id": "http://example.org/AcmeCorp",
    "@confidence": 0.72,
    "@source": "https://model.example.org/rel-extract-v2"
  }
}
```

### 7.3 Multi-Source Fusion with Uncertainty

This example demonstrates the confidence algebra's value. Two NER models extract the same entity with different confidence profiles. A scalar system would average them (0.83); the algebra fuses their evidence, reducing uncertainty:

```json
{
  "@context": ["https://schema.org/", "https://jsonld-ex.github.io/ns/context/v1.jsonld"],
  "@type": "Person",
  "name": {
    "@value": "Jane Doe",
    "@confidence": {
      "@type": "Opinion",
      "belief": 0.811,
      "disbelief": 0.108,
      "uncertainty": 0.081,
      "baseRate": 0.50
    },
    "@source": "https://model.example.org/ensemble-ner",
    "@method": "cumulative-fusion"
  }
}
```

See [Confidence Algebra](confidence-algebra.md) §5 for the cumulative fusion operator that produces this result.

### 7.4 Temporally Bounded Assertion

```json
{
  "@context": ["https://schema.org/", "https://jsonld-ex.github.io/ns/context/v1.jsonld"],
  "@type": "Organization",
  "ceo": {
    "@value": "Jane Smith",
    "@confidence": 0.95,
    "@source": "https://model.example.org/rel-extract-v3",
    "@validFrom": "2024-03-01T00:00:00Z",
    "@validUntil": "2027-03-01T00:00:00Z"
  }
}
```

See [Temporal Extensions](temporal.md) for validity semantics and point-in-time queries.

---

## 8. References

### Normative References

- **[JSON-LD 1.1]** M. Sporny, D. Longley, G. Kellogg, M. Lanthaler, P.-A. Champin. *JSON-LD 1.1: A JSON-based Serialization for Linked Data.* W3C Recommendation, 16 July 2020. https://www.w3.org/TR/json-ld11/

- **[RFC 2119]** S. Bradner. *Key words for use in RFCs to Indicate Requirement Levels.* IETF, March 1997. https://www.rfc-editor.org/rfc/rfc2119

### Informative References

- **[Jøsang 2016]** A. Jøsang. *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer, 2016. ISBN 978-3-319-42335-7.

- **[PROV-O]** T. Lebo, S. Sahoo, D. McGuinness. *PROV-O: The PROV Ontology.* W3C Recommendation, 30 April 2013. https://www.w3.org/TR/prov-o/

- **[SHACL]** H. Knublauch, D. Kontokostas. *Shapes Constraint Language (SHACL).* W3C Recommendation, 20 July 2017. https://www.w3.org/TR/shacl/

- **[OWL 2]** W3C OWL Working Group. *OWL 2 Web Ontology Language Document Overview.* W3C Recommendation, 11 December 2012. https://www.w3.org/TR/owl2-overview/

- **[SSN/SOSA]** A. Haller et al. *Semantic Sensor Network Ontology.* W3C Recommendation, 19 October 2017. https://www.w3.org/TR/vocab-ssn/

- **[Croissant]** MLCommons. *Croissant: A Metadata Format for ML-Ready Datasets.* Version 1.0. https://mlcommons.org/croissant/
