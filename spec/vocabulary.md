# jsonld-ex Vocabulary Specification

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/vocabulary](https://jsonld-ex.github.io/ns/vocabulary)

---

## 1. Introduction

This document defines the complete vocabulary of the jsonld-ex extension. Every class and property listed here is implemented in the [reference implementation](https://pypi.org/project/jsonld-ex/) and tested against 832+ unit and integration tests.

Terms are organized into functional categories. For each term, the specification provides:

- **IRI** — the full Internationalized Resource Identifier
- **Compact form** — the `@`-prefixed keyword used in JSON-LD documents
- **Type** — the expected XSD datatype or structural type
- **Domain** — what kind of JSON-LD construct the term applies to
- **Range** — the set of allowed values
- **Description** — normative prose defining the term's semantics
- **Processing rules** — requirements on conforming processors (using RFC 2119 keywords)
- **Example** — a minimal JSON-LD snippet

### 1.1 Namespace

All terms in this vocabulary are defined under the namespace:

```
https://w3id.org/jsonld-ex/
```

The preferred prefix is `jex:`. For example, the full IRI of the `confidence` property is `https://w3id.org/jsonld-ex/confidence`.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

A **conforming processor** is a JSON-LD 1.1 processor that additionally interprets the terms defined in this specification according to the processing rules stated herein.

A **non-extended processor** is a standard JSON-LD 1.1 processor. Such processors MUST treat jsonld-ex terms as ordinary properties mapped by the context file, preserving them without semantic interpretation.

### 1.3 Notation

In the definitions below:

- "Value object" refers to a JSON-LD node of the form `{"@value": ..., ...}`.
- "Node object" refers to a JSON-LD node that may contain `@id`, `@type`, and properties.
- XSD datatypes use the prefix `xsd:` for `http://www.w3.org/2001/XMLSchema#`.

---

## 2. Classes

### 2.1 jex:Opinion

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/Opinion` |
| **Subclass of** | `rdfs:Resource` |

An opinion ω = (b, d, u, a) per Jøsang's Subjective Logic (2016), representing a nuanced belief state that distinguishes between evidence for a proposition, evidence against it, and absence of evidence.

An Opinion node MUST contain exactly the properties `jex:belief`, `jex:disbelief`, `jex:uncertainty`, and `jex:baseRate`, satisfying the constraint b + d + u = 1.

See the [Confidence Algebra](confidence-algebra.md) specification for the full formal treatment.

**Example:**

```json
{
  "@type": "jex:Opinion",
  "jex:belief": 0.70,
  "jex:disbelief": 0.10,
  "jex:uncertainty": 0.20,
  "jex:baseRate": 0.50
}
```

---

## 3. Core Annotation Properties

These properties attach AI/ML provenance metadata to value objects. They are the most commonly used jsonld-ex terms.

### 3.1 jex:confidence

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/confidence` |
| **Compact form** | `@confidence` |
| **Type** | `xsd:double` |
| **Domain** | Value object |
| **Range** | `[0.0, 1.0]` |

The scalar confidence or certainty of an assertion. A value of `1.0` indicates absolute certainty; `0.0` indicates no confidence.

This is a **lossy projection** of the richer Opinion representation. When an Opinion is available, the projected probability P(ω) = b + a·u provides the equivalent scalar. See the [Confidence Algebra](confidence-algebra.md) specification for the relationship between scalar confidence and opinions.

**Processing rules:**

- Processors MUST validate that `@confidence` is a finite number in `[0.0, 1.0]`.
- Processors MUST reject `NaN`, `Infinity`, `-Infinity`, booleans, and non-numeric values.
- `@confidence` SHOULD appear on value objects (nodes containing `@value`).
- During expansion, `@confidence` maps to `https://w3id.org/jsonld-ex/confidence`.

**Example:**

```json
{
  "name": {
    "@value": "Jane Doe",
    "@confidence": 0.95
  }
}
```

### 3.2 jex:source

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/source` |
| **Compact form** | `@source` |
| **Type** | `@id` (IRI) |
| **Domain** | Value object |
| **Range** | IRI identifying a system, model, or agent |

Identifies the system, model, or agent that produced the assertion. This is the primary provenance link, analogous to `prov:wasAttributedTo` in PROV-O.

**Processing rules:**

- The value SHOULD be a dereferenceable IRI identifying the producing system.
- When converting to PROV-O, `@source` maps to a `prov:SoftwareAgent` via `prov:wasAttributedTo`.

**Example:**

```json
{
  "name": {
    "@value": "Jane Doe",
    "@confidence": 0.95,
    "@source": "https://model.example.org/ner-v4"
  }
}
```

### 3.3 jex:extractedAt

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/extractedAt` |
| **Compact form** | `@extractedAt` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object |
| **Range** | ISO 8601 timestamp |

The timestamp at which the value was extracted or generated. Corresponds to `prov:generatedAtTime` in PROV-O.

**Processing rules:**

- The value MUST be a valid ISO 8601 date-time string.
- Processors SHOULD accept both `Z` and `±HH:MM` timezone suffixes.
- Date-only values (`YYYY-MM-DD`) are permitted and interpreted as midnight UTC.

**Example:**

```json
{
  "name": {
    "@value": "Jane Doe",
    "@extractedAt": "2026-01-15T10:30:00Z"
  }
}
```

### 3.4 jex:method

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/method` |
| **Compact form** | `@method` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Free-text description of the extraction method |

Describes the method used to produce the assertion (e.g., "NER", "classification", "regression", "IMU-6axis-classification").

**Processing rules:**

- The value MUST be a non-empty string.
- Processors MAY use this value for display, filtering, or provenance reporting.

**Example:**

```json
{
  "name": {
    "@value": "Jane Doe",
    "@method": "NER"
  }
}
```

### 3.5 jex:humanVerified

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/humanVerified` |
| **Compact form** | `@humanVerified` |
| **Type** | `xsd:boolean` |
| **Domain** | Value object |
| **Range** | `true` or `false` |

Indicates whether a human has reviewed and verified the assertion. This is critical for human-in-the-loop ML pipelines where automated extractions require manual validation.

**Processing rules:**

- The value MUST be a JSON boolean (`true` or `false`).
- Processors MUST NOT coerce strings or numbers to booleans.

**Example:**

```json
{
  "name": {
    "@value": "Jane Doe",
    "@confidence": 0.98,
    "@humanVerified": true
  }
}
```

---

## 4. Multimodal Properties

These properties support provenance tracking for multimodal content (images, audio, video, documents).

### 4.1 jex:mediaType

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/mediaType` |
| **Compact form** | `@mediaType` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | IANA media type (e.g., `image/png`, `audio/wav`) |

The IANA media type of the source content from which the value was derived.

**Processing rules:**

- The value SHOULD conform to the IANA media type registry format (`type/subtype`).

**Example:**

```json
{
  "caption": {
    "@value": "A cat sitting on a windowsill",
    "@mediaType": "image/jpeg",
    "@contentUrl": "https://example.org/images/cat.jpg"
  }
}
```

### 4.2 jex:contentUrl

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/contentUrl` |
| **Compact form** | `@contentUrl` |
| **Type** | `@id` (IRI) |
| **Domain** | Value object |
| **Range** | IRI of the source content |

A dereferenceable IRI pointing to the original content from which the value was derived.

### 4.3 jex:contentHash

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/contentHash` |
| **Compact form** | `@contentHash` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Hash string in `algorithm-base64` format |

A cryptographic hash of the source content, enabling integrity verification. Uses the same `algorithm-base64` format as `@integrity` (see [Security](security.md)).

**Example:**

```json
{
  "caption": {
    "@value": "A cat sitting on a windowsill",
    "@contentHash": "sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
  }
}
```

---

## 5. Translation Properties

These properties track provenance for machine-translated content.

### 5.1 jex:translatedFrom

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/translatedFrom` |
| **Compact form** | `@translatedFrom` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | BCP 47 language tag (e.g., `en`, `zh-Hans`, `fr-FR`) |

The language of the original text before translation.

**Example:**

```json
{
  "description": {
    "@value": "Ein rotes Auto",
    "@language": "de",
    "@translatedFrom": "en",
    "@translationModel": "https://model.example.org/nmt-en-de-v3"
  }
}
```

### 5.2 jex:translationModel

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/translationModel` |
| **Compact form** | `@translationModel` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Identifier or IRI of the translation model |

Identifies the machine translation model or service used.

---

## 6. IoT and Sensor Properties

These properties support sensor data provenance and measurement metadata, designed for interoperability with the W3C [SSN/SOSA](https://www.w3.org/TR/vocab-ssn/) ontology.

### 6.1 jex:measurementUncertainty

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/measurementUncertainty` |
| **Compact form** | `@measurementUncertainty` |
| **Type** | `xsd:double` |
| **Domain** | Value object |
| **Range** | Non-negative number |

The measurement uncertainty (error margin) of a sensor reading. This is distinct from `@confidence`, which represents epistemic certainty about an assertion; `@measurementUncertainty` represents the physical precision of a measurement instrument.

**Processing rules:**

- The value MUST be a non-negative finite number.
- The units of uncertainty match the units of the measured value.
- Maps to `sosa:resultQuality` in SSN/SOSA.

**Example:**

```json
{
  "temperature": {
    "@value": 36.7,
    "@measurementUncertainty": 0.2,
    "@unit": "celsius"
  }
}
```

### 6.2 jex:unit

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/unit` |
| **Compact form** | `@unit` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Unit identifier (e.g., `celsius`, `kg`, `m/s`) |

The unit of measurement for a numeric value. Processors SHOULD use standardized unit identifiers (e.g., UCUM codes) where possible.

---

## 7. Derivation Properties

### 7.1 jex:derivedFrom

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/derivedFrom` |
| **Compact form** | `@derivedFrom` |
| **Type** | `@id` or array of `@id` |
| **Domain** | Value object |
| **Range** | IRI(s) of source entities or assertions |

Links a derived assertion to its source(s). Corresponds to `prov:wasDerivedFrom` in PROV-O. When multiple sources contributed, the value is an array of IRIs.

**Example:**

```json
{
  "summary": {
    "@value": "Patient shows improvement in mobility.",
    "@derivedFrom": [
      "https://example.org/observation/001",
      "https://example.org/observation/002"
    ],
    "@method": "LLM-summarization"
  }
}
```

---

## 8. Aggregation Properties

These properties describe how a value was produced by aggregating multiple observations.

### 8.1 jex:aggregationMethod

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/aggregationMethod` |
| **Compact form** | `@aggregationMethod` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Aggregation method identifier (e.g., `mean`, `median`, `max`, `min`, `sum`, `count`) |

The statistical method used to aggregate multiple observations into this value.

### 8.2 jex:aggregationWindow

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/aggregationWindow` |
| **Compact form** | `@aggregationWindow` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | ISO 8601 duration (e.g., `PT1H`, `P1D`) or descriptive string |

The time window over which observations were aggregated.

### 8.3 jex:aggregationCount

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/aggregationCount` |
| **Compact form** | `@aggregationCount` |
| **Type** | `xsd:integer` |
| **Domain** | Value object |
| **Range** | Positive integer |

The number of individual observations that were aggregated to produce this value.

**Example:**

```json
{
  "averageHeartRate": {
    "@value": 72,
    "@aggregationMethod": "mean",
    "@aggregationWindow": "PT1H",
    "@aggregationCount": 360,
    "@unit": "bpm"
  }
}
```

---

## 9. Calibration Properties

These properties record the calibration state of a sensor or measurement instrument.

### 9.1 jex:calibratedAt

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/calibratedAt` |
| **Compact form** | `@calibratedAt` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object |
| **Range** | ISO 8601 timestamp |

The timestamp of the most recent calibration of the instrument that produced this value.

### 9.2 jex:calibrationMethod

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/calibrationMethod` |
| **Compact form** | `@calibrationMethod` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Free-text description of the calibration procedure |

### 9.3 jex:calibrationAuthority

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/calibrationAuthority` |
| **Compact form** | `@calibrationAuthority` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Name or IRI of the calibrating entity |

The organization or entity that performed the calibration.

**Example:**

```json
{
  "temperature": {
    "@value": 36.7,
    "@calibratedAt": "2025-12-01T09:00:00Z",
    "@calibrationMethod": "two-point water bath",
    "@calibrationAuthority": "NIST"
  }
}
```

---

## 10. Delegation Properties

### 10.1 jex:delegatedBy

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/delegatedBy` |
| **Compact form** | `@delegatedBy` |
| **Type** | `@id` or array of `@id` |
| **Domain** | Value object |
| **Range** | IRI(s) identifying delegating agents |

Records the chain of delegation for an assertion. When agent A delegates to agent B which produces the assertion, the delegation chain is recorded as `["agentA-iri", "agentB-iri"]`. Corresponds to `prov:actedOnBehalfOf` in PROV-O.

**Example:**

```json
{
  "diagnosis": {
    "@value": "hypertension",
    "@confidence": 0.82,
    "@source": "https://model.example.org/clinical-v2",
    "@delegatedBy": "https://hospital.example.org/dr-smith"
  }
}
```

---

## 11. Invalidation Properties

These properties mark an assertion as no longer valid.

### 11.1 jex:invalidatedAt

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/invalidatedAt` |
| **Compact form** | `@invalidatedAt` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object |
| **Range** | ISO 8601 timestamp |

The timestamp at which the assertion was invalidated. Corresponds to `prov:invalidatedAtTime` in PROV-O.

**Processing rules:**

- When `@invalidatedAt` is present, processors SHOULD treat the assertion as retracted.
- The `filter_by_confidence` function supports an `exclude_invalidated` parameter that skips nodes with this property.

### 11.2 jex:invalidationReason

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/invalidationReason` |
| **Compact form** | `@invalidationReason` |
| **Type** | `xsd:string` |
| **Domain** | Value object |
| **Range** | Free-text explanation |

A human-readable explanation of why the assertion was invalidated.

**Example:**

```json
{
  "diagnosis": {
    "@value": "hypertension",
    "@confidence": 0.82,
    "@invalidatedAt": "2026-02-01T14:00:00Z",
    "@invalidationReason": "Superseded by updated lab results"
  }
}
```

---

## 12. Temporal Properties

These properties enable time-aware assertions. See the [Temporal](temporal.md) specification for full query and diff semantics.

### 12.1 jex:validFrom

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/validFrom` |
| **Compact form** | `@validFrom` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object or node object |
| **Range** | ISO 8601 timestamp |

The timestamp at which the assertion becomes true. Before this time, the assertion is not yet in effect.

**Processing rules:**

- When both `@validFrom` and `@validUntil` are present, processors MUST validate that `@validFrom ≤ @validUntil`.

### 12.2 jex:validUntil

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/validUntil` |
| **Compact form** | `@validUntil` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object or node object |
| **Range** | ISO 8601 timestamp |

The timestamp at which the assertion ceases to be true. After this time, the assertion has expired.

### 12.3 jex:asOf

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/asOf` |
| **Compact form** | `@asOf` |
| **Type** | `xsd:dateTime` |
| **Domain** | Value object or node object |
| **Range** | ISO 8601 timestamp |

The point in time at which the assertion was known or observed to hold. This is distinct from `@extractedAt` (when the value was produced) and `@validFrom` (when the assertion becomes effective). `@asOf` records the observation time — "as of this moment, this was true."

**Example:**

```json
{
  "jobTitle": {
    "@value": "Senior Engineer",
    "@validFrom": "2024-01-01",
    "@validUntil": "2025-12-31",
    "@asOf": "2025-06-15T12:00:00Z"
  }
}
```

---

## 13. Opinion Properties

These properties represent the components of a Subjective Logic opinion. They appear on nodes of type `jex:Opinion`. See the [Confidence Algebra](confidence-algebra.md) specification for the complete formal treatment.

### 13.1 jex:belief

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/belief` |
| **Compact form** | `belief` (within Opinion context) |
| **Type** | `xsd:double` |
| **Domain** | `jex:Opinion` |
| **Range** | `[0.0, 1.0]` |

The degree of belief (evidence FOR the proposition). Denoted _b_ in the opinion tuple ω = (b, d, u, a).

### 13.2 jex:disbelief

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/disbelief` |
| **Compact form** | `disbelief` (within Opinion context) |
| **Type** | `xsd:double` |
| **Domain** | `jex:Opinion` |
| **Range** | `[0.0, 1.0]` |

The degree of disbelief (evidence AGAINST the proposition). Denoted _d_ in the opinion tuple.

### 13.3 jex:uncertainty

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/uncertainty` |
| **Compact form** | `uncertainty` (within Opinion context) |
| **Type** | `xsd:double` |
| **Domain** | `jex:Opinion` |
| **Range** | `[0.0, 1.0]` |

The degree of uncertainty (absence of evidence). Denoted _u_ in the opinion tuple.

**Invariant:** For any valid opinion, `belief + disbelief + uncertainty = 1`.

### 13.4 jex:baseRate

| Property | Value |
|----------|-------|
| **IRI** | `https://w3id.org/jsonld-ex/baseRate` |
| **Compact form** | `baseRate` (within Opinion context) |
| **Type** | `xsd:double` |
| **Domain** | `jex:Opinion` |
| **Range** | `[0.0, 1.0]` |
| **Default** | `0.5` |

The prior probability (base rate) of the proposition. Denoted _a_ in the opinion tuple. Used to compute the projected probability: P(ω) = b + a·u.

---

## 14. Vector Properties

These properties support dense vector embeddings alongside symbolic data.

### 14.1 @vector (Container Type)

| Property | Value |
|----------|-------|
| **Keyword** | `@vector` |
| **Used in** | Context term definitions (`@container`) |

A container type indicating that a property holds a dense vector embedding (an ordered array of floating-point numbers).

**Processing rules:**

- During expansion, vector values MUST be preserved as JSON arrays of numbers.
- During compaction, vector values MUST be preserved without modification.
- During RDF conversion, properties with `@container: @vector` MUST be excluded. Vectors are annotation-only and do not participate in the RDF graph.
- During CBOR-LD encoding, vectors SHOULD use efficient binary float array encoding.

### 14.2 @dimensions

| Property | Value |
|----------|-------|
| **Keyword** | `@dimensions` |
| **Used in** | Context term definitions (alongside `@container: @vector`) |
| **Type** | `xsd:integer` |
| **Range** | Positive integer |

Specifies the expected dimensionality of a vector property. Processors SHOULD validate that the vector length matches `@dimensions` during expansion.

**Example:**

```json
{
  "@context": {
    "embedding": {
      "@id": "https://w3id.org/jsonld-ex/embedding",
      "@container": "@vector",
      "@dimensions": 768
    }
  },
  "@type": "Product",
  "name": "Widget",
  "embedding": [0.123, -0.456, 0.789]
}
```

---

## 15. Security Properties

See the [Security](security.md) specification for the complete threat model and processing algorithms.

### 15.1 @integrity

| Property | Value |
|----------|-------|
| **Keyword** | `@integrity` |
| **Used in** | Context references |
| **Type** | `xsd:string` |
| **Format** | `algorithm-base64hash` |

A cryptographic hash of a context document, enabling integrity verification before use. Prevents context injection attacks via DNS poisoning or man-in-the-middle modification.

**Supported algorithms:** `sha256`, `sha384`, `sha512`.

**Processing rules:**

- The format MUST be `algorithm-base64hash` where `algorithm` is one of the supported algorithms and `base64hash` is the Base64-encoded digest of the context content.
- Processors MUST compute the hash of the retrieved context and compare it to the declared value.
- If the hashes do not match, processors MUST reject the context (fail-closed behavior).
- Processors MUST NOT fall back to using the context without verification when `@integrity` is declared.

**Example:**

```json
{
  "@context": {
    "@id": "https://schema.org/",
    "@integrity": "sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
  }
}
```

---

## 16. Validation Properties

These properties define shape-based constraints for JSON-LD documents. See the [Validation](validation.md) specification for the complete formal semantics, inheritance resolution, and relationship to SHACL.

### 16.1 @shape

| Property | Value |
|----------|-------|
| **Keyword** | `@shape` |
| **Type** | Object (shape definition) or nested shape reference |

A shape definition constraining the structure and values of a JSON-LD node. Shapes contain property constraints using the keywords defined below. When used as a property constraint (nested), validates that the property value conforms to the inner shape.

### 16.2 @required

| Property | Value |
|----------|-------|
| **Keyword** | `@required` |
| **Type** | `xsd:boolean` |
| **Default** | `false` |

When `true`, the property MUST be present in the node and MUST NOT have a null or absent value.

**SHACL equivalent:** `sh:minCount 1`

### 16.3 @extends

| Property | Value |
|----------|-------|
| **Keyword** | `@extends` |
| **Type** | String (shape name), inline shape object, or array thereof |

Specifies one or more parent shapes from which to inherit constraints. Child constraints override parent constraints for the same property. Resolution uses the `shape_registry` parameter.

**Processing rules:**

- Circular inheritance MUST be detected and prevented.
- Multiple parents are merged left-to-right, with later parents overriding earlier ones.
- The child shape's own constraints override all inherited constraints.

### 16.4 @type (in validation context)

When used within a shape property constraint, `@type` specifies the expected XSD datatype of the value. Supported types: `xsd:string`, `xsd:integer`, `xsd:double`, `xsd:float`, `xsd:decimal`, `xsd:boolean`.

**SHACL equivalent:** `sh:datatype`

### 16.5 @minimum

| Property | Value |
|----------|-------|
| **Keyword** | `@minimum` |
| **Type** | Numeric |

The value MUST be greater than or equal to this number. Applies only to numeric values (booleans excluded).

**SHACL equivalent:** `sh:minInclusive`

### 16.6 @maximum

| Property | Value |
|----------|-------|
| **Keyword** | `@maximum` |
| **Type** | Numeric |

The value MUST be less than or equal to this number.

**SHACL equivalent:** `sh:maxInclusive`

### 16.7 @minLength

| Property | Value |
|----------|-------|
| **Keyword** | `@minLength` |
| **Type** | `xsd:integer` |

The string value MUST have at least this many characters.

**SHACL equivalent:** `sh:minLength`

### 16.8 @maxLength

| Property | Value |
|----------|-------|
| **Keyword** | `@maxLength` |
| **Type** | `xsd:integer` |

The string value MUST have at most this many characters.

**SHACL equivalent:** `sh:maxLength`

### 16.9 @pattern

| Property | Value |
|----------|-------|
| **Keyword** | `@pattern` |
| **Type** | `xsd:string` (regular expression) |

The string value MUST match this regular expression (searched, not full-match). Uses standard regex syntax.

**SHACL equivalent:** `sh:pattern`

### 16.10 @in

| Property | Value |
|----------|-------|
| **Keyword** | `@in` |
| **Type** | Array |

The value MUST be a member of the specified array (enumeration constraint).

**SHACL equivalent:** `sh:in`

### 16.11 @minCount

| Property | Value |
|----------|-------|
| **Keyword** | `@minCount` |
| **Type** | `xsd:integer` |

The property MUST have at least this many values. For absent properties, the count is 0; for a single value, the count is 1; for an array, the count is the array length.

**SHACL equivalent:** `sh:minCount`

### 16.12 @maxCount

| Property | Value |
|----------|-------|
| **Keyword** | `@maxCount` |
| **Type** | `xsd:integer` |

The property MUST have at most this many values.

**SHACL equivalent:** `sh:maxCount`

### 16.13 @severity

| Property | Value |
|----------|-------|
| **Keyword** | `@severity` |
| **Type** | `xsd:string` |
| **Range** | `"error"`, `"warning"`, `"info"` |
| **Default** | `"error"` |

Controls the severity of a constraint violation. Violations at `"warning"` or `"info"` severity are reported as warnings rather than errors and do not cause validation failure.

**SHACL equivalent:** `sh:severity`

### 16.14 @or

| Property | Value |
|----------|-------|
| **Keyword** | `@or` |
| **Type** | Array of constraint objects |

The value MUST satisfy at least one of the constraint branches (logical disjunction).

**SHACL equivalent:** `sh:or`

### 16.15 @and

| Property | Value |
|----------|-------|
| **Keyword** | `@and` |
| **Type** | Array of constraint objects |

The value MUST satisfy all of the constraint branches (logical conjunction).

**SHACL equivalent:** `sh:and`

### 16.16 @not

| Property | Value |
|----------|-------|
| **Keyword** | `@not` |
| **Type** | Constraint object |

The value MUST NOT satisfy the inner constraint (logical negation).

**SHACL equivalent:** `sh:not`

### 16.17 @if / @then / @else

| Property | Value |
|----------|-------|
| **Keywords** | `@if`, `@then`, `@else` |
| **Type** | Constraint objects |

Conditional constraint evaluation. If the value satisfies `@if`, it MUST also satisfy `@then`. If it does not satisfy `@if`, it MUST satisfy `@else` (if present). When `@else` is absent and `@if` is not met, the constraint is vacuously true.

**Example:**

```json
{
  "age": {
    "@if": {"@minimum": 18},
    "@then": {"@in": ["adult", "senior"]},
    "@else": {"@in": ["minor"]}
  }
}
```

### 16.18 @lessThan

| Property | Value |
|----------|-------|
| **Keyword** | `@lessThan` |
| **Type** | `xsd:string` (property name) |

Cross-property constraint: the value MUST be strictly less than the value of the named sibling property.

**SHACL equivalent:** `sh:lessThan`

### 16.19 @lessThanOrEquals

| Property | Value |
|----------|-------|
| **Keyword** | `@lessThanOrEquals` |
| **Type** | `xsd:string` (property name) |

Cross-property constraint: the value MUST be less than or equal to the value of the named sibling property.

**SHACL equivalent:** `sh:lessThanOrEquals`

### 16.20 @equals

| Property | Value |
|----------|-------|
| **Keyword** | `@equals` |
| **Type** | `xsd:string` (property name) |

Cross-property constraint: the value MUST be equal to the value of the named sibling property.

**SHACL equivalent:** `sh:equals`

### 16.21 @disjoint

| Property | Value |
|----------|-------|
| **Keyword** | `@disjoint` |
| **Type** | `xsd:string` (property name) |

Cross-property constraint: the value MUST differ from the value of the named sibling property.

**SHACL equivalent:** `sh:disjoint`

### 16.22 Validation Shape Example

```json
{
  "@shape": {
    "@type": "Person",
    "name": {
      "@required": true,
      "@type": "xsd:string",
      "@minLength": 1
    },
    "email": {
      "@pattern": "^[^@]+@[^@]+$"
    },
    "age": {
      "@type": "xsd:integer",
      "@minimum": 0,
      "@maximum": 150
    },
    "startDate": {
      "@lessThan": "endDate"
    }
  }
}
```

---

## 17. Summary Table

| # | Term | Compact Form | Type | Category |
|---|------|-------------|------|----------|
| 1 | `jex:confidence` | `@confidence` | `xsd:double` | Core |
| 2 | `jex:source` | `@source` | `@id` | Core |
| 3 | `jex:extractedAt` | `@extractedAt` | `xsd:dateTime` | Core |
| 4 | `jex:method` | `@method` | `xsd:string` | Core |
| 5 | `jex:humanVerified` | `@humanVerified` | `xsd:boolean` | Core |
| 6 | `jex:mediaType` | `@mediaType` | `xsd:string` | Multimodal |
| 7 | `jex:contentUrl` | `@contentUrl` | `@id` | Multimodal |
| 8 | `jex:contentHash` | `@contentHash` | `xsd:string` | Multimodal |
| 9 | `jex:translatedFrom` | `@translatedFrom` | `xsd:string` | Translation |
| 10 | `jex:translationModel` | `@translationModel` | `xsd:string` | Translation |
| 11 | `jex:measurementUncertainty` | `@measurementUncertainty` | `xsd:double` | IoT/Sensor |
| 12 | `jex:unit` | `@unit` | `xsd:string` | IoT/Sensor |
| 13 | `jex:derivedFrom` | `@derivedFrom` | `@id` / array | Derivation |
| 14 | `jex:aggregationMethod` | `@aggregationMethod` | `xsd:string` | Aggregation |
| 15 | `jex:aggregationWindow` | `@aggregationWindow` | `xsd:string` | Aggregation |
| 16 | `jex:aggregationCount` | `@aggregationCount` | `xsd:integer` | Aggregation |
| 17 | `jex:calibratedAt` | `@calibratedAt` | `xsd:dateTime` | Calibration |
| 18 | `jex:calibrationMethod` | `@calibrationMethod` | `xsd:string` | Calibration |
| 19 | `jex:calibrationAuthority` | `@calibrationAuthority` | `xsd:string` | Calibration |
| 20 | `jex:delegatedBy` | `@delegatedBy` | `@id` / array | Delegation |
| 21 | `jex:invalidatedAt` | `@invalidatedAt` | `xsd:dateTime` | Invalidation |
| 22 | `jex:invalidationReason` | `@invalidationReason` | `xsd:string` | Invalidation |
| 23 | `jex:validFrom` | `@validFrom` | `xsd:dateTime` | Temporal |
| 24 | `jex:validUntil` | `@validUntil` | `xsd:dateTime` | Temporal |
| 25 | `jex:asOf` | `@asOf` | `xsd:dateTime` | Temporal |
| 26 | `jex:belief` | `belief` | `xsd:double` | Opinion |
| 27 | `jex:disbelief` | `disbelief` | `xsd:double` | Opinion |
| 28 | `jex:uncertainty` | `uncertainty` | `xsd:double` | Opinion |
| 29 | `jex:baseRate` | `baseRate` | `xsd:double` | Opinion |
| 30 | `jex:Opinion` | `Opinion` | Class | Opinion |
| 31 | `@vector` | `@vector` | Container type | Vector |
| 32 | `@dimensions` | `@dimensions` | `xsd:integer` | Vector |
| 33 | `@integrity` | `@integrity` | `xsd:string` | Security |
| 34 | `@shape` | `@shape` | Object | Validation |
| 35 | `@required` | `@required` | `xsd:boolean` | Validation |
| 36 | `@extends` | `@extends` | String / Object | Validation |
| 37 | `@minimum` | `@minimum` | Numeric | Validation |
| 38 | `@maximum` | `@maximum` | Numeric | Validation |
| 39 | `@minLength` | `@minLength` | `xsd:integer` | Validation |
| 40 | `@maxLength` | `@maxLength` | `xsd:integer` | Validation |
| 41 | `@pattern` | `@pattern` | `xsd:string` | Validation |
| 42 | `@in` | `@in` | Array | Validation |
| 43 | `@minCount` | `@minCount` | `xsd:integer` | Validation |
| 44 | `@maxCount` | `@maxCount` | `xsd:integer` | Validation |
| 45 | `@severity` | `@severity` | `xsd:string` | Validation |
| 46 | `@or` | `@or` | Array | Validation |
| 47 | `@and` | `@and` | Array | Validation |
| 48 | `@not` | `@not` | Object | Validation |
| 49 | `@if` / `@then` / `@else` | `@if` / `@then` / `@else` | Object | Validation |
| 50 | `@lessThan` | `@lessThan` | `xsd:string` | Validation |
| 51 | `@lessThanOrEquals` | `@lessThanOrEquals` | `xsd:string` | Validation |
| 52 | `@equals` | `@equals` | `xsd:string` | Validation |
| 53 | `@disjoint` | `@disjoint` | `xsd:string` | Validation |

---

## 18. References

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.
- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- W3C. (2017). PROV-O: The PROV Ontology. W3C Recommendation. https://www.w3.org/TR/prov-o/
- W3C. (2017). SHACL: Shapes Constraint Language. W3C Recommendation. https://www.w3.org/TR/shacl/
- W3C. (2017). SSN/SOSA: Semantic Sensor Network Ontology. W3C Recommendation. https://www.w3.org/TR/vocab-ssn/
- IETF. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. https://www.rfc-editor.org/rfc/rfc2119
