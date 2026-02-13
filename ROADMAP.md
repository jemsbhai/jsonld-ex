# jsonld-ex Roadmap

This document adheres to the progress of the `jsonld-ex` specifications and implementations. The project aims to provide a robust, production-ready extension to JSON-LD 1.2 for AI/ML, Security, and Interoperability.

## Implementation Status

Both the **Python** reference implementation and the **TypeScript/JavaScript** port are now at **feature parity**.

| Feature Family | Module | Python (Reference) | TypeScript (Parity) | Description |
|---|---|:---:|:---:|---|
| **AI/ML** | `ai_ml` | ✅ | ✅ | Provenance, Confidence, Methods, Attribution |
| **Logic** | `confidence` | ✅ | ✅ | Subjective Logic Algebra, Decay, Bridge |
| **Inference** | `inference` | ✅ | ✅ | Propagation, Combination, Conflict Res |
| **Vector** | `vector` | ✅ | ✅ | Embeddings, Cosine Sim, Validation |
| **Security** | `security` | ✅ | ✅ | Integrity (`@integrity`), Allowlists, Limits |
| **Validation** | `validation` | ✅ | ✅ | Native `@shape` validation, SHACL mapping |
| **Temporal** | `temporal` | ✅ | ✅ | Time-travel queries (`queryAtTime`), `@validFrom`/`Until` |
| **Interop** | `owl` | ✅ | ✅ | PROV-O, OWL, RDF-Star conversions |
| **Transport** | `mqtt` | ✅ | ✅ | QoS mapping, Topic derivation |
| **Serialization**| `cbor` | ✅ | ✅ | CBOR-LD binary format (`.cbor`) |
| **Data Ops** | `merge`/`diff`| ✅ | ✅ | Graph merging and semantic difference |
| **Dataset** | `dataset` | ✅ | ✅ | Croissant / MLAgent interoperability |
| **Batch** | `batch` | ✅ | ✅ | High-throughput batch processing |
| **API** | `mcp` | ✅ | ✅ | Model Context Protocol Server (41+ tools) |

## Version History

### Python Reference Implementation (`packages/python`)
- **v0.1.0** - Initial release
- **v0.2.0** - Confidence Algebra & Security
- **v0.3.0** - Interop & Vectors
- **v0.3.5** - Current Stable (Full Feature Set)

### TypeScript / JavaScript Implementation (`packages/js`)
- **v0.1.0** - Initial port (Core, Logic, AI/ML)
- **v0.1.1** - **Parity Release** (Feb 2026)
    - Added OWL/SHACL Interop
    - Added MQTT & CBOR-LD
    - Added Batch & Dataset
    - Added full AI/ML Provenance (Delegation, Derivation, etc.)
    - Full MCP Server support

## Future Roadmap

### Phase 3: Standardization & Adoption
- [ ] Submit `jsonld-ex` vocabulary to W3C Community Group
- [ ] Formalize Subjective Logic JSON Schema
- [ ] Browser extension for verified credential display
- [ ] Standardize the `@shape` syntax as a lightweight SHACL profile

### Phase 4: Extended Ecosystem
- [ ] **Rust** implementation (`jsonld-ex-rs`) for high-performance edges
- [ ] **Go** implementation for kubernetes controllers
- [ ] Integration with LangChain / Haystack / LlamaIndex
- [ ] Native support in vector databases (e.g., Qdrant, Weaviate) via plugins
