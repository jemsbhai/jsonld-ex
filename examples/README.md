# JSON-LD Extensions — Python Examples

Comprehensive examples demonstrating all features of the `jsonld-ex` library.

## Setup

```bash
cd packages/python
pip install -e ".[dev]"
```

## Running Examples

```bash
python examples/01_confidence_basics.py
python examples/02_provenance_tracking.py
# ... etc
```

## Index

| # | Example | Features |
|---|---------|----------|
| 01 | Confidence Basics | `annotate()`, `get_confidence()`, `filter_by_confidence()`, `aggregate_confidence()` |
| 02 | Provenance Tracking | Full provenance metadata, `get_provenance()`, multi-source merging |
| 03 | Vector Embeddings | `vector_term_definition()`, `validate_vector()`, `cosine_similarity()`, `strip_vectors_for_rdf()` |
| 04 | Context Integrity | `compute_integrity()`, `verify_integrity()`, tamper detection |
| 05 | Context Allowlist | Allowlist config, pattern matching, blocking remote contexts |
| 06 | Resource Limits | Size limits, depth limits, enforcement |
| 07 | Shape Validation | All constraint types: required, type, min/max, length, pattern |
| 08 | Document Validation | Multi-node graphs, multiple shapes, error reporting |
| 09 | Healthcare Wearable | End-to-end posture monitoring use case (paper validation scenario) |
| 10 | Knowledge Graph RAG | AI extraction pipeline with confidence, vectors, and validation |
