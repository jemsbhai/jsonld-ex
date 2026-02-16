# Documentation Roadmap

**Created:** 2026-02-15  
**Status:** Planning  
**Goal:** Comprehensive, modular documentation suitable for NeurIPS reviewers, ML engineers, and W3C standards audience.

## Current State

- `README.md` (root) — Project overview, outdated (references v0.3.0-era numbers)
- `packages/python/README.md` — Python quickstart + API overview, partially outdated
- `CHANGELOG.md` — Up to date through v0.6.5
- `spec/ai-ml-extensions.md` — W3C-style formal specification (static)
- `docs/neurips_experiment_roadmap.md` — Internal experiment planning
- `docs/experiment_collation.md` — Internal experiment tracking

## Target Structure

```
docs/
├── guides/
│   ├── confidence-algebra.md      # Subjective Logic: opinions, fusion, trust, decay
│   ├── compliance-algebra.md      # GDPR operators, consent lifecycle, erasure
│   ├── similarity-metrics.md      # Registry, built-ins, examples, advisory system
│   ├── data-protection.md         # Protection annotations, consent, data rights, DPV
│   ├── interoperability.md        # PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant
│   ├── validation.md              # Shapes, nested, conditional, severity, inheritance
│   ├── transport.md               # CBOR-LD, MQTT, batch API
│   ├── temporal.md                # Time-aware assertions, queries, diff
│   ├── vector-embeddings.md       # @vector container, similarity, term definitions
│   └── mcp-server.md              # All tools, resources, prompts with examples
├── api/
│   ├── ai_ml.md                   # annotate, get_confidence, get_provenance, filter
│   ├── confidence_algebra.md      # Opinion, fuse, discount, deduce, conflict
│   ├── compliance_algebra.md      # ComplianceOpinion, operators, triggers
│   ├── similarity.md              # Registry, metrics, advisory system
│   ├── data_protection.md         # Protection, consent, classification
│   ├── data_rights.md             # Erasure, restriction, portability, access
│   ├── validation.md              # validate_node, validate_document
│   ├── security.md                # Integrity, allowlists, resource limits
│   ├── owl_interop.md             # All interop functions
│   ├── dpv_interop.md             # DPV v2.2 conversion
│   ├── inference.md               # Propagation, combination, conflict resolution
│   ├── merge.md                   # Graph merge, diff
│   ├── temporal.md                # Temporal annotations, queries, diff
│   ├── vector.md                  # Vector validation, cosine, term definitions
│   ├── batch.md                   # Batch annotation, validation, filtering
│   ├── context.md                 # Context diff, compatibility checking
│   ├── cbor_ld.md                 # CBOR serialization
│   └── mqtt.md                    # MQTT transport
├── internal/                      # Not for public consumption
│   ├── neurips_experiment_roadmap.md
│   └── experiment_collation.md
└── tutorials/
    ├── quickstart.md              # 5-minute getting started
    ├── rag-pipeline.md            # Confidence-aware RAG with jsonld-ex
    ├── iot-sensor-pipeline.md     # SSN/SOSA + MQTT end-to-end
    └── gdpr-compliance.md         # Data protection annotation workflow
```

## Work Packages

### WP1: Surgical README Updates (Priority: NOW) ✅ DONE
- [x] Root README.md — Fixed numbers (2025+ tests, 53 tools, 23 modules), added 14 feature categories, updated architecture diagram, updated module table, updated interop table
- [x] packages/python/README.md — Fixed numbers, 6 categories/23 modules, 53 MCP tools, 5 resources, 4 prompts, compliance algebra tools section, expanded validation constraints

### WP2: Key Guides (Priority: HIGH — needed for NeurIPS) ✅ DONE
- [x] docs/guides/confidence-algebra.md — Opinions, fusion, trust, decay, conflict, scalar bridge
- [x] docs/guides/similarity-metrics.md — Registry, built-ins, custom, examples, advisory system
- [x] docs/guides/interoperability.md — PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant, DPV

### WP3: Remaining Guides (Priority: MEDIUM — improves submission)
- [ ] docs/guides/compliance-algebra.md
- [ ] docs/guides/data-protection.md
- [ ] docs/guides/validation.md
- [ ] docs/guides/transport.md
- [ ] docs/guides/temporal.md
- [ ] docs/guides/vector-embeddings.md
- [ ] docs/guides/mcp-server.md

### WP4: API Reference (Priority: LOW — can auto-generate)
- [ ] One file per module with full function signatures, parameters, return types, examples
- [ ] Consider auto-generation from docstrings (sphinx/pdoc)

### WP5: Tutorials (Priority: LOW — after experiments)
- [ ] Quickstart tutorial
- [ ] RAG pipeline tutorial (ties to E3.1 experiment)
- [ ] IoT sensor pipeline tutorial (ties to E8.2 experiment)
- [ ] GDPR compliance tutorial (ties to compliance algebra)

## Principles

1. **No duplication** — READMEs link to guides, guides link to API reference
2. **Examples first** — Every guide starts with a working code example
3. **Honest about limitations** — Document what doesn't work, not just what does
4. **Testable examples** — All code examples should be extractable and runnable
5. **Audience-aware** — Guides target ML engineers; API docs target developers; spec targets W3C
