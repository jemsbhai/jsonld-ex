# jsonld-ex Examples

Both JavaScript/TypeScript and Python examples are provided, covering identical functionality.

## Examples

| # | Topic | JS | Python |
|---|-------|----|--------|
| 01 | Confidence Basics | [js/01_confidence_basics.ts](js/01_confidence_basics.ts) | [python/01_confidence_basics.py](python/01_confidence_basics.py) |
| 02 | Provenance Tracking | [js/02_provenance_tracking.ts](js/02_provenance_tracking.ts) | [python/02_provenance_tracking.py](python/02_provenance_tracking.py) |
| 03 | Vector Embeddings | [js/03_vector_embeddings.ts](js/03_vector_embeddings.ts) | [python/03_vector_embeddings.py](python/03_vector_embeddings.py) |
| 04 | Context Integrity | [js/04_context_integrity.ts](js/04_context_integrity.ts) | [python/04_context_integrity.py](python/04_context_integrity.py) |
| 05 | Context Allowlist | [js/05_context_allowlist.ts](js/05_context_allowlist.ts) | [python/05_context_allowlist.py](python/05_context_allowlist.py) |
| 06 | Resource Limits | [js/06_resource_limits.ts](js/06_resource_limits.ts) | [python/06_resource_limits.py](python/06_resource_limits.py) |
| 07 | Shape Validation | [js/07_shape_validation.ts](js/07_shape_validation.ts) | [python/07_shape_validation.py](python/07_shape_validation.py) |
| 08 | Document Validation | [js/08_document_validation.ts](js/08_document_validation.ts) | [python/08_document_validation.py](python/08_document_validation.py) |
| 09 | Healthcare Wearable (E2E) | [js/09_healthcare_wearable.ts](js/09_healthcare_wearable.ts) | [python/09_healthcare_wearable.py](python/09_healthcare_wearable.py) |
| 10 | Knowledge Graph RAG Pipeline | [js/10_knowledge_graph_rag.ts](js/10_knowledge_graph_rag.ts) | [python/10_knowledge_graph_rag.py](python/10_knowledge_graph_rag.py) |

## Running

### JavaScript/TypeScript

From the repository root:

```bash
cd packages/js
npm install
npm run build
cd ../../examples/js
npx ts-node 01_confidence_basics.ts
```

### Python

```bash
pip install jsonld-ex
cd examples/python
python 01_confidence_basics.py
```
