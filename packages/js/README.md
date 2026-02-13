# @jsonld-ex/core

> JavaScript/TypeScript implementation of JSON-LD 1.2 extensions for AI/ML data exchange, security hardening, and validation.

[![npm](https://img.shields.io/npm/v/@jsonld-ex/core)](https://www.npmjs.com/package/@jsonld-ex/core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

## Status

**Feature Complete (v0.1.1)** — All extensions, including Subjective Logic, AI/ML Provenance, GDPR Compliance, and Interop modules are fully implemented and tested, achieving 100% feature parity with the Python reference implementation.

### Implemented Modules

| Module | Features | Description |
|--------|-----------|-------------|
| **AI/ML** | `annotate`, `provenance` | Confidence scores, data lineage, method tracking. |
| **Data Protection** | `annotateProtection` | GDPR metadata, consent records, and graph filtering. |
| **Confidence** | `Opinion`, `fuse`, `discount` | Subjective Logic algebra for uncertainty propagation. |
| **Logic** | `decay`, `deduce` | Temporal belief decay and conditional reasoning. |
| **Inference** | `merge`, `diff`, `conflict` | Graph merging with confidence-aware conflict resolution. |
| **Temporal** | `validFrom`, `asOf` | Bitemporal versioning and time-slice queries. |
| **Security** | `integrity`, `allowlist` | Context hashing and resource limits. |
| **Validation** | `@shape` | Native structure validation without SHACL. |
| **Vector** | `@vector` | Embeddings and cosine similarity. |
| **Interop** | `prov`, `shacl`, `owl` | Bidirectional conversion to W3C standards. |
| **Transport** | `mqtt` | MQTT QoS mapping and topic derivation. |
| **Batch** | `batch` | High-throughput processing for large datasets. |
| **Dataset** | `dataset`, `croissant` | MLCommons Croissant metadata interoperability. |
| **MCP** | `server` | Model Context Protocol server for AI agents. |

## Installation

```bash
npm install @jsonld-ex/core
```

## Quick Start

The **Client API** (`JsonLdExClient`) provides a unified, validated interface for all features.

```typescript
import client from '@jsonld-ex/core';

// 1. Annotate data with confidence and provenance
const fact = client.annotate("Sky is blue", {
  confidence: 0.99,
  source: "https://sensor.example.org/cam-1",
  extractedAt: "2024-03-15T10:00:00Z"
});

// 2. Merge conflicting knowledge graphs
const graphA = [{ "@id": "node1", "status": "active" }];
const graphB = [{ "@id": "node1", "status": "inactive" }];

const { merged, report } = client.merge([graphA, graphB], {
  conflictStrategy: "highest_confidence"
});

// 3. Propagate confidence through a chain of reasoning
const result = client.propagate([0.9, 0.8, 0.95], 'multiply');
// result.score ≈ 0.68
```

## Feature Deep Dive

### 1. Data Protection & GDPR

Manage regulatory compliance with metadata for legal basis, consent, and personal data categories.

```typescript
import { annotateProtection, createConsentRecord } from '@jsonld-ex/core';

const profile = annotateProtection("John Doe", {
  personalDataCategory: "regular",
  legalBasis: "consent",
  jurisdiction: "EU",
  consent: createConsentRecord("2024-01-01T00:00:00Z", ["marketing", "analytics"])
});
```

### 2. Subjective Logic (Uncertainty)

Work directly with Subjective Logic opinions (Belief, Disbelief, Uncertainty).

```typescript
import { Opinion, cumulativeFuse, trustDiscount } from '@jsonld-ex/core';

// Agent A trusts Agent B (trust metric)
const trustAB = Opinion.fromConfidence(0.9);

// Agent B believes proposition X
const opinionBX = Opinion.fromConfidence(0.8);

// Agent A's derived opinion on X (Trust Discount)
const opinionAX = trustDiscount(trustAB, opinionBX);
console.log(opinionAX.toConfidence()); // ~0.72

// Combine independent opinions (Cumulative Fusion)
const opinionC = Opinion.fromConfidence(0.6);
const fused = cumulativeFuse(opinionAX, opinionC);
```

### 3. Standards Interoperability

Convert between `jsonld-ex` extensions and established W3C standards.

```typescript
import { toProvO, shapeToShacl, shapeToOwlRestrictions } from '@jsonld-ex/core';

// Export provenance as PROV-O (RDF)
const { provGraph } = toProvO(annotatedDoc);

// Convert @shape definitions to SHACL shapes
const shaclShapes = shapeToShacl(myShape);

// Convert to OWL Class Restrictions
const owlClasses = shapeToOwlRestrictions(myShape);
```

### 4. MQTT & IoT Transport

Optimize payloads for bandwidth-constrained environments.

```typescript
import { toMqttPayload, deriveMqttQos } from '@jsonld-ex/core';

// Serialize to CBOR-compressed MQTT payload
const payload = toMqttPayload(doc, { compress: true });

// Map confidence to MQTT QoS level (0, 1, or 2)
const qos = deriveMqttQos(doc);
```

### 5. Batch Processing

High-performance API for processing annotation and validation in bulk.

```typescript
import { annotateBatch, validateBatch } from '@jsonld-ex/core';

const items = ["A", "B", "C"];
const annotated = annotateBatch(items, { confidence: 0.9 });
```

### 6. Dataset (Croissant)

Interoperability with MLCommons Croissant metadata for datasets.

```typescript
import { toCroissant, createDatasetMetadata } from '@jsonld-ex/core';

const metadata = createDatasetMetadata("My Dataset");
const croissant = toCroissant(metadata);
```

### 7. MCP Server

Run `jsonld-ex` as a Model Context Protocol (MCP) server to give AI agents access to these tools.

```bash
# Run directly
npx @jsonld-ex/core

# Or via the binary
./bin/mcp-server.js
```

## Architecture

```
src/
├── client.ts             # High-level Façade API
├── types.ts              # Unified Type Definitions
├── schemas.ts            # Zod Validation Schemas
├── processor.ts          # JsonLdEx Processor (Legacy Wrapper)
├── data-protection.ts    # GDPR & Privacy (New)
├── batch.ts              # Batch Processing (New)
├── dataset.ts            # Croissant Support (New)
├── mqtt.ts               # IoT Transport (New)
├── owl.ts                # Interop (PROV/SHACL/OWL) (New)
├── mcp/
│   └── server.ts         # MCP Server Implementation
├── confidence/
│   ├── algebra.ts        # Subjective Logic Math
│   ├── decay.ts          # Temporal Decay Functions
│   └── bridge.ts         # Scalar <-> Opinion Bridge
├── extensions/
│   ├── ai-ml.ts          # Provenance & Annotations
│   ├── vector.ts         # Vector Embeddings
│   ├── security.ts       # Integrity & Limits
│   └── validation.ts     # Shape Validation
└── ...
```

## Development

This project uses **TypeScript** and **ES Modules** (NodeNext).

```bash
# Install
npm install

# Test (Jest with generic ESM support)
npm test

# Build (tsc)
npm run build
```

## License

MIT
