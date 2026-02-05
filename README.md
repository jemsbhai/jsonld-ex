# jsonld-ex — JSON-LD 1.2 Extensions

**Reference implementation of proposed JSON-LD 1.2 extensions for AI/ML data exchange, security hardening, and validation.**

> Companion implementation for: *"Extending JSON-LD for Modern AI: Addressing Security, Data Modeling, and Implementation Gaps"* — FLAIRS-39 (2026)

## Overview

`jsonld-ex` extends the existing JSON-LD ecosystem with backward-compatible extensions that address critical gaps in:

1. **AI/ML Data Modeling** — `@confidence`, `@source`, `@vector` container, provenance tracking
2. **Security Hardening** — `@integrity` context verification, context allowlists, resource limits
3. **Validation** — `@shape` native validation framework
4. **Performance** — Context caching, CBOR-LD benchmarks

## Packages

| Package | Language | Registry | Status |
|---------|----------|----------|--------|
| `@jsonld-ex/core` | TypeScript/JS | npm | 🚧 In Development |
| `jsonld-ex` | Python | PyPI | 🚧 In Development |

## Architecture

Both packages wrap proven base libraries and add extension processing:

- **JavaScript**: Wraps [jsonld.js](https://github.com/digitalbazaar/jsonld.js) (Digital Bazaar)
- **Python**: Wraps [PyLD](https://github.com/digitalbazaar/pyld) (Digital Bazaar)

```
┌─────────────────────────────────────────┐
│           Application Layer             │
├─────────────────────────────────────────┤
│         jsonld-ex Extensions            │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ AI/ML    │ │ Security │ │ Validate│ │
│  │@confidence│ │@integrity│ │ @shape  │ │
│  │@vector   │ │allowlist │ │         │ │
│  │@source   │ │limits    │ │         │ │
│  └──────────┘ └──────────┘ └─────────┘ │
├─────────────────────────────────────────┤
│     jsonld.js / PyLD (Core Processing)  │
├─────────────────────────────────────────┤
│           JSON-LD 1.1 Spec             │
└─────────────────────────────────────────┘
```

## Quick Start

### JavaScript / TypeScript

```bash
npm install @jsonld-ex/core
```

```typescript
import { JsonLdEx } from '@jsonld-ex/core';

const doc = {
  "@context": [
    "http://schema.org/",
    "http://www.w3.org/ns/jsonld-ex/"
  ],
  "@type": "Person",
  "name": {
    "@value": "John Smith",
    "@confidence": 0.95,
    "@source": "https://ml-model.example.org/ner-v2",
    "@extractedAt": "2026-01-15T10:30:00Z"
  }
};

const processor = new JsonLdEx();
const result = await processor.expand(doc);
const confidence = processor.getConfidence(result, "name"); // 0.95
```

### Python

```bash
pip install jsonld-ex
```

```python
from jsonld_ex import JsonLdEx

doc = {
    "@context": [
        "http://schema.org/",
        "http://www.w3.org/ns/jsonld-ex/"
    ],
    "@type": "Person",
    "name": {
        "@value": "John Smith",
        "@confidence": 0.95,
        "@source": "https://ml-model.example.org/ner-v2",
        "@extractedAt": "2026-01-15T10:30:00Z"
    }
}

processor = JsonLdEx()
result = processor.expand(doc)
confidence = processor.get_confidence(result, "name")  # 0.95
```

## Extension Specifications

Formal specifications for each extension are in [`/spec`](./spec/):

- [AI/ML Extensions](./spec/ai-ml-extensions.md) — Confidence, provenance, vector embeddings
- [Security Extensions](./spec/security-extensions.md) — Integrity, allowlists, resource limits
- [Validation Extensions](./spec/validation-extensions.md) — Shape-based validation

## Benchmarks

Performance comparisons against baseline JSON-LD processing are in [`/benchmarks`](./benchmarks/).

## Contributing

This is a research implementation accompanying an academic publication. Contributions welcome via issues and PRs.

## License

MIT

## Citation

```bibtex
@inproceedings{jsonld-ex-2026,
  title={Extending JSON-LD for Modern AI: Addressing Security, Data Modeling, and Implementation Gaps},
  author={Syed, Muntaser and Silaghi, Marius and Abujar, Sheikh and Alssadi, Rwaida},
  booktitle={Proceedings of the 39th International FLAIRS Conference},
  year={2026}
}
```
