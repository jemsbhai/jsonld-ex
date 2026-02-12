# @jsonld-ex/core

> JavaScript/TypeScript implementation of JSON-LD 1.2 extensions for AI/ML data exchange, security hardening, and validation.

[![npm](https://img.shields.io/npm/v/@jsonld-ex/core)](https://www.npmjs.com/package/@jsonld-ex/core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](../../LICENSE)

## Status

**Feature Complete (v0.1.0)** — Core extensions, Subjective Logic, and MCP Server are fully implemented and tested, achieving parity with the Python reference implementation.

### Implemented Modules

| Module | Features | Description |
|--------|-----------|-------------|
| **AI/ML** | `annotate`, `provenance` | Confidence scores, data lineage, method tracking. |
| **Confidence** | `Opinion`, `fuse`, `discount` | Subjective Logic algebra for uncertainty propagation. |
| **Logic** | `decay`, `deduce` | Temporal belief decay and conditional reasoning. |
| **Inference** | `merge`, `diff`, `conflict` | Graph merging with confidence-aware conflict resolution. |
| **Temporal** | `validFrom`, `asOf` | Bitemporal versioning and time-slice queries. |
| **Security** | `integrity`, `allowlist` | Context hashing and resource limits. |
| **Validation** | `@shape` | Native structure validation without SHACL. |
| **Vector** | `@vector` | Embeddings and cosine similarity. |
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

## Advanced Usage

### Subjective Logic (Uncertainty)

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

### Temporal Queries

Manage knowledge over time using bitemporal assertions.

```typescript
import client from '@jsonld-ex/core';

// Add temporal validity
const assertion = client.addTemporal({ "status": "open" }, {
  validFrom: "2024-01-01T00:00:00Z",
  validUntil: "2024-12-31T23:59:59Z"
});

// Query graph at specific point in time
const snapshot = client.queryAtTime(historyGraph, "2024-06-01T00:00:00Z");
```

### Graph Merging & Diffing

Merge graphs with fine-grained conflict resolution.

```typescript
import client from '@jsonld-ex/core';

// Calculate semantic difference
const diff = client.diff(graphV1, graphV2);
console.log(`Changed: ${diff.modified.length}, Added: ${diff.added.length}`);

// Merge with report
const { merged, report } = client.merge([sourceA, sourceB], {
  conflictStrategy: 'weighted_vote',
  confidenceCombination: 'average'
});
```

### MCP Server

Run `jsonld-ex` as a Model Context Protocol (MCP) server to give AI agents access to these tools.

```bash
# Run directly
npx @jsonld-ex/core

# Or via the binary
./bin/mcp-server.js
```

**Available Tools:**
- `jsonld_annotate`: Create annotated values.
- `jsonld_merge`: Merge knowledge graphs.
- `jsonld_diff`: Compare graphs.
- `jsonld_propagate`: Calculate derived confidence.
- `jsonld_temporal`: Add temporal qualifiers.

## Architecture

```
src/
├── client.ts             # High-level Façade API
├── types.ts              # Unified Type Definitions
├── schemas.ts            # Zod Validation Schemas
├── processor.ts          # JsonLdEx Processor (Legacy Wrapper)
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
