# JSON-LD AI/ML Extensions Specification

**Status:** Draft  
**Version:** 0.1.0  
**Authors:** Muntaser Aljabry, Marius Silaghi  
**Date:** 2026-01-15

## 1. Introduction

This specification defines backward-compatible extensions to JSON-LD 1.1
that enable AI/ML metadata to be embedded alongside symbolic linked data.
The extensions address three critical gaps:

1. **Confidence Scores** — No standard way to express prediction uncertainty
2. **Provenance Tracking** — No standard attribution for AI-generated assertions
3. **Vector Embeddings** — No mechanism for dense vectors alongside symbolic data

## 2. Extension Keywords

### 2.1 @confidence

**Type:** `xsd:double`  
**Range:** `[0.0, 1.0]`  
**Applies to:** Value objects (`@value`)

Indicates the confidence or certainty of an assertion. A value of `1.0`
indicates absolute certainty; `0.0` indicates no confidence.

```json
{
  "name": {
    "@value": "John Smith",
    "@confidence": 0.95
  }
}
```

**Processing rules:**
- Processors MUST validate that `@confidence` is a number in `[0.0, 1.0]`
- `@confidence` MUST only appear on value objects containing `@value`
- During expansion, `@confidence` maps to `<http://www.w3.org/ns/jsonld-ex/confidence>`
- During RDF conversion, confidence is preserved as a reified annotation

### 2.2 @source

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

### 2.3 @extractedAt

**Type:** `xsd:dateTime`  
**Applies to:** Value objects

ISO 8601 timestamp indicating when the value was extracted or generated.

### 2.4 @method

**Type:** `xsd:string`  
**Applies to:** Value objects

Describes the extraction method (e.g., "NER", "classification", "regression").

### 2.5 @humanVerified

**Type:** `xsd:boolean`  
**Applies to:** Value objects

Indicates whether a human has reviewed and verified the assertion.

## 3. Vector Embedding Container

### 3.1 @vector Container Type

A new container type `@vector` indicates that a property holds a
dense vector embedding. Vectors are treated as opaque annotations
and are NOT converted to RDF triples.

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
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

### 3.2 @dimensions

**Type:** `xsd:integer`  
**Applies to:** Term definitions with `@container: @vector`

Optional dimensionality constraint. Processors SHOULD validate
vector length against `@dimensions` during expansion.

### 3.3 Processing Rules

1. During expansion: vector values are preserved as JSON arrays
2. During compaction: vector values are preserved without modification
3. During RDF conversion: vector properties are EXCLUDED (annotation-only)
4. During CBOR-LD encoding: vectors use efficient binary float arrays

## 4. Context Definition

The extensions are defined under the namespace:

```
http://www.w3.org/ns/jsonld-ex/
```

Documents using these extensions SHOULD include the context:

```json
{
  "@context": [
    "http://schema.org/",
    "http://www.w3.org/ns/jsonld-ex/"
  ]
}
```

## 5. Backward Compatibility

These extensions are fully backward compatible with JSON-LD 1.1:

- Standard processors will treat extension keywords as regular properties
- Extended processors will interpret them with the defined semantics
- No existing JSON-LD documents are affected
- The extensions use the established `@`-keyword convention

## 6. Use Cases

### 6.1 Healthcare Wearable Data

```json
{
  "@context": ["http://schema.org/", "http://www.w3.org/ns/jsonld-ex/"],
  "@type": "MedicalObservation",
  "posture": {
    "@value": "forward-head",
    "@confidence": 0.87,
    "@source": "https://device.example.org/imu-classifier-v3",
    "@extractedAt": "2026-01-15T14:30:00Z",
    "@method": "IMU-6axis-classification"
  },
  "sensorData": {
    "@id": "http://example.org/sensor-embedding",
    "@container": "@vector",
    "@dimensions": 128
  }
}
```

### 6.2 Knowledge Graph Extraction

```json
{
  "@context": ["http://schema.org/", "http://www.w3.org/ns/jsonld-ex/"],
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
