# @jsonld-ex/core

> JavaScript/TypeScript implementation of JSON-LD 1.2 extensions for AI/ML data exchange, security hardening, and validation.

[![npm](https://img.shields.io/npm/v/@jsonld-ex/core)](https://www.npmjs.com/package/@jsonld-ex/core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

## Status

**Early development (v0.1.0)** — core extension modules are implemented and tested. Feature parity with the [Python package](../python/README.md) is planned but not yet complete.

### Implemented Modules

| Module | Functions | Description |
|--------|-----------|-------------|
| `ai-ml` | `annotate`, `getConfidence`, `getProvenance`, `filterByConfidence`, `aggregateConfidence` | `@confidence`, `@source`, provenance tracking |
| `security` | `computeIntegrity`, `verifyIntegrity`, `isContextAllowed`, `createSecureDocumentLoader`, `enforceResourceLimits` | `@integrity` verification, context allowlists, resource limits |
| `validation` | `validateNode`, `validateDocument` | `@shape` native validation framework |
| `vector` | `vectorTermDefinition`, `validateVector`, `cosineSimilarity`, `extractVectors`, `stripVectorsForRdf` | `@vector` container support |

### Not Yet Implemented

The following modules are available in the Python package but not yet ported:

- Confidence Algebra (Subjective Logic framework)
- Confidence Bridge / Decay
- Inference engine
- Graph merge / diff
- Temporal extensions
- OWL / PROV-O / SHACL interop
- CBOR-LD / MQTT transport
- MCP server

## Installation

```bash
npm install @jsonld-ex/core
```

## Quick Start

```typescript
import { JsonLdEx } from '@jsonld-ex/core';

const jex = new JsonLdEx();

// Annotate a value with AI/ML provenance
const name = jex.annotate("John Smith", {
  confidence: 0.95,
  source: "https://ml-model.example.org/ner-v2",
  method: "NER",
});
// { "@value": "John Smith", "@confidence": 0.95, "@source": "...", "@method": "NER" }
```

## Usage

### AI/ML Annotations

```typescript
import { annotate, getConfidence, getProvenance, filterByConfidence } from '@jsonld-ex/core';

// Annotate values with provenance metadata
const value = annotate("Jane Doe", {
  confidence: 0.92,
  source: "https://model.example.org/v3",
  extractedAt: "2026-01-15T10:30:00Z",
  method: "NER",
  humanVerified: false,
});

// Extract confidence from annotated values
const conf = getConfidence(value); // 0.92

// Extract full provenance
const prov = getProvenance(value);
// { confidence: 0.92, source: "...", extractedAt: "...", method: "NER", humanVerified: false }

// Filter graph nodes by minimum confidence
const graph = [
  { "@type": "Person", name: annotate("Alice", { confidence: 0.9 }) },
  { "@type": "Person", name: annotate("Bob", { confidence: 0.3 }) },
];
const highConf = filterByConfidence(graph, "name", 0.8);
// [{ "@type": "Person", name: { "@value": "Alice", "@confidence": 0.9 } }]
```

### Security

```typescript
import { computeIntegrity, verifyIntegrity, isContextAllowed } from '@jsonld-ex/core';

// Compute and verify context integrity
const ctx = '{"@vocab": "http://schema.org/"}';
const hash = computeIntegrity(ctx); // "sha256-..."
const valid = verifyIntegrity(ctx, hash); // true

// Context allowlists
const allowed = isContextAllowed("https://schema.org/", {
  allowed: ["https://schema.org/"],
  blockRemoteContexts: false,
}); // true
```

### Validation

```typescript
import { validateNode } from '@jsonld-ex/core';

const shape = {
  "@type": "Person",
  "name": { "@required": true, "@type": "xsd:string", "@minLength": 1 },
  "email": { "@pattern": "^[^@]+@[^@]+$" },
  "age": { "@type": "xsd:integer", "@minimum": 0, "@maximum": 150 },
};

const result = validateNode(
  { "@type": "Person", "name": "Alice", "age": 30 },
  shape
);
// { valid: true, errors: [], warnings: [] }
```

### Vector Embeddings

```typescript
import { vectorTermDefinition, validateVector, cosineSimilarity } from '@jsonld-ex/core';

// Define a vector property in context
const ctx = vectorTermDefinition("embedding", "http://example.org/embedding", 768);
// { embedding: { "@id": "...", "@container": "@vector", "@dimensions": 768 } }

// Validate vector values
const check = validateVector([0.1, -0.2, 0.3], 3);
// { valid: true, errors: [] }

// Compute similarity
const sim = cosineSimilarity([1, 0, 0], [0, 1, 0]); // 0.0
```

### Processor Class

The `JsonLdEx` class wraps [jsonld.js](https://github.com/digitalbazaar/jsonld.js) with security enforcement:

```typescript
import { JsonLdEx } from '@jsonld-ex/core';

const jex = new JsonLdEx({
  resourceLimits: {
    maxDocumentSize: 5 * 1024 * 1024, // 5 MB
    maxGraphDepth: 50,
    maxExpansionTime: 10_000, // 10s timeout
  },
  contextAllowlist: {
    allowed: ["https://schema.org/", "https://w3id.org/security/v2"],
    blockRemoteContexts: false,
  },
});

// All standard JSON-LD operations enforced with limits
const expanded = await jex.expand(doc);
const compacted = await jex.compact(doc, ctx);
const nquads = await jex.toRdf(doc);
```

## Development

```bash
# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

## Architecture

```
src/
├── index.ts              # Public API exports
├── processor.ts          # JsonLdEx class (wraps jsonld.js)
├── types.ts              # TypeScript interfaces
├── keywords.ts           # Extension keyword constants
├── jsonld.d.ts           # Type declarations for jsonld.js
└── extensions/
    ├── ai-ml.ts          # @confidence, @source, provenance
    ├── security.ts       # @integrity, allowlists, resource limits
    ├── validation.ts     # @shape validation framework
    └── vector.ts         # @vector container, similarity
```

## Related

- [Python package](../python/README.md) — full-featured reference implementation (14 modules, 832+ tests)
- [Root project](../../README.md) — project overview and specifications
- [Extension specs](../../spec/) — formal specification documents

## License

MIT
