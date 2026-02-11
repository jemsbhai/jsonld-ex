# jsonld-ex — JSON-LD 1.2 Extensions

**Reference implementation of proposed JSON-LD 1.2 extensions for AI/ML data exchange, security hardening, and validation.**

> Companion implementation for: *"Extending JSON-LD for Modern AI: Addressing Security, Data Modeling, and Implementation Gaps"* — FLAIRS-39 (2026)

[![PyPI](https://img.shields.io/pypi/v/jsonld-ex)](https://pypi.org/project/jsonld-ex/)
[![Tests](https://img.shields.io/badge/tests-832%2B%20passing-brightgreen)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

`jsonld-ex` extends the existing JSON-LD ecosystem with backward-compatible extensions that address critical gaps in:

1. **AI/ML Data Modeling** — `@confidence`, `@source`, `@vector` container, provenance tracking
2. **Confidence Algebra** — Full Subjective Logic framework (Jøsang 2016): opinions, cumulative/averaging fusion, trust discount, deduction, conflict detection, Byzantine-resistant fusion
3. **Security Hardening** — `@integrity` context verification, context allowlists, resource limits
4. **Validation** — `@shape` native validation framework
5. **Inference** — Confidence propagation through inference chains, multi-source combination (noisy-OR, Dempster–Shafer)
6. **Graph Operations** — Confidence-aware merging, semantic diff, conflict resolution
7. **Temporal Modeling** — `@validFrom`, `@validUntil`, `@asOf` for time-aware assertions
8. **IoT Transport** — CBOR-LD binary serialization, MQTT topic/QoS derivation
9. **MCP Server** — 41 tools exposing all library capabilities to LLM agents via the [Model Context Protocol](https://modelcontextprotocol.io/)

## Ecosystem Interoperability

jsonld-ex does not replace existing standards — it bridges them:

| Standard | Relationship |
|----------|-------------|
| **PROV-O** | Bidirectional conversion via `to_prov_o` / `from_prov_o` (60–75% fewer triples) |
| **SHACL** | Bidirectional mapping via `shape_to_shacl` / `shacl_to_shape` |
| **OWL** | `@shape` → OWL class restrictions via `shape_to_owl_restrictions` |
| **RDF-star** | Export annotated values as RDF-star N-Triples |
| **CBOR-LD** | Binary serialization with context compression |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────────────────────────────────────────────┤
│                  jsonld-ex Extensions                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐ │
│  │ AI/ML    │ │ Security │ │Validate │ │ Inference      │ │
│  │@confidence│ │@integrity│ │ @shape  │ │ propagation    │ │
│  │@vector   │ │allowlist │ │         │ │ combination    │ │
│  │@source   │ │limits    │ │         │ │ conflict res.  │ │
│  └──────────┘ └──────────┘ └─────────┘ └────────────────┘ │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Confidence Algebra (Subjective Logic)                │  │
│  │ Opinion, cumulative/averaging fusion, trust discount │  │
│  │ deduction, conflict detection, Byzantine-resistant   │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐ │
│  │ Temporal  │ │  Merge   │ │ OWL/RDF │ │ IoT Transport  │ │
│  │@validFrom │ │ graphs   │ │ interop │ │ CBOR-LD, MQTT  │ │
│  │@validUntil│ │ diff     │ │ PROV-O  │ │ topic, QoS     │ │
│  │@asOf     │ │ conflict │ │ SHACL   │ │                │ │
│  └──────────┘ └──────────┘ └─────────┘ └────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│          jsonld.js / PyLD (Core Processing)                 │
├─────────────────────────────────────────────────────────────┤
│                    JSON-LD 1.1 Spec                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Core (all features except IoT transport)
pip install jsonld-ex

# With IoT transport (CBOR-LD + MQTT helpers)
pip install jsonld-ex[iot]
```

### Annotate Values with Confidence and Provenance

```python
from jsonld_ex import annotate, get_confidence

doc = {
    "@context": "http://schema.org/",
    "@type": "Person",
    "name": annotate(
        "John Smith",
        confidence=0.95,
        source="https://ml-model.example.org/ner-v2",
        extracted_at="2026-01-15T10:30:00Z",
        method="NER",
    ),
}

get_confidence(doc["name"])  # 0.95
```

### Propagate Confidence Through Inference Chains

```python
from jsonld_ex import propagate_confidence, combine_sources

# Source (0.9 conf) → Rule (0.8 conf) → Conclusion
result = propagate_confidence([0.9, 0.8], method="dampened")
result.score  # 0.849 (less aggressive than naive 0.72)

# Two sources independently say the same thing
combined = combine_sources([0.8, 0.7], method="noisy_or")
combined.score  # 0.94
```

### Merge Graphs from Multiple Sources

```python
from jsonld_ex import merge_graphs

graph_a = {"@context": "http://schema.org/", "@graph": [
    {"@id": "ex:alice", "@type": "Person",
     "name": {"@value": "Alice", "@confidence": 0.8, "@source": "model-A"}}
]}
graph_b = {"@context": "http://schema.org/", "@graph": [
    {"@id": "ex:alice", "@type": "Person",
     "name": {"@value": "Alice", "@confidence": 0.7, "@source": "model-B"}}
]}

merged, report = merge_graphs([graph_a, graph_b])
# Agreement → confidence boosted via noisy-OR: 0.94
# report.properties_agreed == 1, report.properties_conflicted == 0
```

### Time-Aware Assertions

```python
from jsonld_ex import add_temporal, query_at_time

nodes = [
    {"@id": "ex:alice", "jobTitle": add_temporal(
        {"@value": "Engineer", "@confidence": 0.9},
        valid_from="2020-01-01", valid_until="2023-12-31",
    )},
    {"@id": "ex:alice", "jobTitle": add_temporal(
        {"@value": "Manager", "@confidence": 0.85},
        valid_from="2024-01-01",
    )},
]

query_at_time(nodes, "2022-06-15")  # → Engineer
query_at_time(nodes, "2025-01-01")  # → Manager
```

### CBOR-LD Payload Optimization

```python
from jsonld_ex import to_cbor, from_cbor, payload_stats

doc = {"@context": "http://schema.org/", "@type": "SensorReading",
       "value": {"@value": 42.5, "@confidence": 0.9}}

stats = payload_stats(doc)
# stats.cbor_ratio ≈ 0.65 (35% smaller than JSON)
# stats.gzip_cbor_ratio ≈ 0.45 (55% smaller than JSON)

payload = to_cbor(doc)          # bytes for wire transmission
restored = from_cbor(payload)   # back to dict
```

### Convert to/from PROV-O

```python
from jsonld_ex import to_prov_o, from_prov_o

doc = {
    "@context": "http://schema.org/",
    "@type": "Person",
    "name": {"@value": "Alice", "@confidence": 0.95,
             "@source": "https://model.example.org/v2",
             "@method": "NER"},
}

prov_doc, report = to_prov_o(doc)
# Full PROV-O graph with Entity, Activity, Agent nodes
# report.compression_ratio shows jsonld-ex is 3-5x more compact

round_tripped = from_prov_o(prov_doc)
# Back to inline annotations — lossless round-trip
```

## Module Reference

| Module | Import | Description |
|--------|--------|-------------|
| `ai_ml` | `annotate`, `get_confidence`, `get_provenance`, `filter_by_confidence` | Core annotation and extraction |
| `vector` | `validate_vector`, `cosine_similarity`, `vector_term_definition` | `@vector` container support |
| `security` | `compute_integrity`, `verify_integrity`, `is_context_allowed` | `@integrity` and allowlists |
| `validation` | `validate_node`, `validate_document` | `@shape` validation framework |
| `owl_interop` | `to_prov_o`, `from_prov_o`, `shape_to_shacl`, `shacl_to_shape`, `shape_to_owl_restrictions`, `to_rdf_star_ntriples` | Bidirectional standards mapping |
| `confidence_algebra` | `Opinion`, `cumulative_fuse`, `averaging_fuse`, `trust_discount`, `deduce`, `robust_fuse` | Subjective Logic framework (Jøsang 2016) |
| `confidence_bridge` | `combine_opinions_from_scalars`, `propagate_opinions_from_scalars` | Scalar-to-opinion bridge |
| `confidence_decay` | `decay_opinion`, `exponential_decay`, `linear_decay`, `step_decay` | Temporal decay of evidence |
| `inference` | `propagate_confidence`, `combine_sources`, `resolve_conflict` | Confidence propagation and combination |
| `merge` | `merge_graphs`, `diff_graphs` | Graph merging and diff |
| `temporal` | `add_temporal`, `query_at_time`, `temporal_diff` | Time-aware assertions |
| `cbor_ld` | `to_cbor`, `from_cbor`, `payload_stats` | Binary serialization *(requires `cbor2`)* |
| `mqtt` | `to_mqtt_payload`, `from_mqtt_payload`, `derive_mqtt_topic`, `derive_mqtt_qos` | IoT transport *(requires `cbor2`)* |
| `mcp` | MCP server (41 tools) | LLM agent integration *(requires `mcp`, Python 3.10+)* |

## Packages

Detailed documentation, usage examples, and API reference for each language implementation:

| Package | Path | Status |
|---------|------|--------|
| **Python** | [`packages/python/README.md`](./packages/python/README.md) | ✅ Published on [PyPI](https://pypi.org/project/jsonld-ex/) — 14 modules, 41 MCP tools, 832+ tests |
| **JavaScript/TypeScript** | [`packages/js/README.md`](./packages/js/README.md) | 🚧 Early development (v0.1.0) — 4 core modules (ai-ml, security, validation, vector) |

## Extension Specifications

Formal specifications for each extension are in [`/spec`](./spec/):

- [AI/ML Extensions](./spec/ai-ml-extensions.md) — Confidence, provenance, vector embeddings

## Contributing

This is a research implementation accompanying an academic publication. Contributions welcome via issues and PRs.

## License

MIT

## Citation

```bibtex
@inproceedings{jsonld-ex-flairs-2026,
  title={Extending JSON-LD for Modern AI: Addressing Security, Data Modeling, and Implementation Gaps},
  author={Syed, Muntaser and Silaghi, Marius and Abujar, Sheikh and Alssadi, Rwaida},
  booktitle={Proceedings of the 39th International FLAIRS Conference},
  year={2026}
}
```

A follow-up paper targeting **NeurIPS 2026 Datasets & Benchmarks** is in preparation, covering the formal confidence algebra, comprehensive benchmarks, and extended evaluation.
