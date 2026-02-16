# jsonld-ex Roadmap

## Current Status

**jsonld-ex** is a Python reference implementation of proposed JSON-LD 1.2 extensions for AI/ML applications. The library is published on [PyPI](https://pypi.org/project/jsonld-ex/) and under active development.

### Implemented

- Assertion-level confidence algebra grounded in Jøsang's Subjective Logic
- Compliance algebra for modeling regulatory compliance as uncertain epistemic states
- Bidirectional interoperability with 8 W3C standards (PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant, DPV v2.2)
- Native inline validation (`@shape`) with SHACL-equivalent expressiveness
- Data protection annotations, consent lifecycle, and data subject rights operations
- Similarity metric registry with built-in and extensible custom metrics
- CBOR-LD serialization and MQTT transport for IoT applications
- Model Context Protocol (MCP) server for LLM agent integration
- Temporal validity, graph merge/diff, batch processing, and context versioning

## Standardization Path

- Submit jsonld-ex vocabulary and extensions to W3C Community Group
- Formalize Subjective Logic opinion schema for JSON-LD contexts
- Propose `@shape` syntax as a lightweight SHACL profile
- Propose `@confidence`, `@source`, and provenance fields for JSON-LD 1.2

## Future Ecosystem

- Additional language implementations (TypeScript/JavaScript, Rust, Go)
- Integration with ML frameworks (LangChain, Haystack, LlamaIndex)
- Vector database plugins (Qdrant, Weaviate)
- Browser extension for verified credential display

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Issues and PRs welcome.
