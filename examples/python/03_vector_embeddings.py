"""
Example 03: Vector Embeddings
==============================

Demonstrates the @vector container type for storing dense vector
embeddings alongside symbolic linked data.

Use case: Product catalog with semantic search embeddings.
"""

import json
from jsonld_ex import JsonLdEx, validate_vector, cosine_similarity
from jsonld_ex.vector import vector_term_definition, extract_vectors, strip_vectors_for_rdf

processor = JsonLdEx()

# ── 1. Defining vector properties in context ─────────────────────

print("=== 1. Vector Term Definition ===\n")

# Define a 4-dimensional embedding (small for demo; real would be 768/1536)
embedding_def = vector_term_definition("embedding", "http://example.org/embedding", 4)
print(f"Term definition: {json.dumps(embedding_def, indent=2)}")

# ── 2. Creating documents with embeddings ────────────────────────

print("\n=== 2. Documents with Embeddings ===\n")

products = [
    {
        "@type": "Product",
        "@id": "#laptop",
        "name": "UltraBook Pro 15",
        "category": "Electronics",
        "embedding": [0.82, -0.15, 0.63, 0.41],
    },
    {
        "@type": "Product",
        "@id": "#tablet",
        "name": "TabletAir 10",
        "category": "Electronics",
        "embedding": [0.79, -0.12, 0.58, 0.45],
    },
    {
        "@type": "Product",
        "@id": "#chair",
        "name": "Ergonomic Office Chair",
        "category": "Furniture",
        "embedding": [-0.31, 0.72, -0.18, 0.55],
    },
    {
        "@type": "Product",
        "@id": "#desk",
        "name": "Standing Desk Pro",
        "category": "Furniture",
        "embedding": [-0.28, 0.68, -0.22, 0.61],
    },
]

for p in products:
    print(f"  {p['@id']}: {p['name']} → embedding={p['embedding']}")

# ── 3. Validating vectors ────────────────────────────────────────

print("\n=== 3. Vector Validation ===\n")

# Valid vector
valid, errors = validate_vector([0.1, -0.2, 0.3, 0.4], expected_dimensions=4)
print(f"Valid 4D vector: valid={valid}")

# Wrong dimensions
valid, errors = validate_vector([0.1, -0.2, 0.3], expected_dimensions=4)
print(f"Wrong dimensions: valid={valid}, errors={errors}")

# Invalid elements
valid, errors = validate_vector([0.1, "not_a_number", 0.3, 0.4])
print(f"Invalid element: valid={valid}, errors={errors}")

# Empty vector
valid, errors = validate_vector([])
print(f"Empty vector: valid={valid}, errors={errors}")

# ── 4. Semantic similarity search ────────────────────────────────

print("\n=== 4. Semantic Similarity Search ===\n")

# Query: "portable computer" embedding
query_embedding = [0.80, -0.14, 0.60, 0.43]

print(f"Query embedding: {query_embedding}")
print(f"Results ranked by cosine similarity:\n")

similarities = []
for product in products:
    sim = cosine_similarity(query_embedding, product["embedding"])
    similarities.append((product, sim))

# Sort by similarity descending
similarities.sort(key=lambda x: x[1], reverse=True)

for product, sim in similarities:
    print(f"  {sim:.4f}  {product['name']} ({product['category']})")

# ── 5. Extracting vectors from nodes ─────────────────────────────

print("\n=== 5. Extracting Vectors ===\n")

vectors = extract_vectors(products[0], ["embedding"])
for prop, vec in vectors.items():
    print(f"  {prop}: {vec} (dimensions={len(vec)})")

# ── 6. Stripping vectors for RDF conversion ──────────────────────

print("\n=== 6. Stripping Vectors for RDF ===\n")

# Vectors are annotation-only — they should not become RDF triples
original = products[0].copy()
rdf_ready = strip_vectors_for_rdf(original, ["embedding"])

print("Original keys:", list(original.keys()))
print("RDF-ready keys:", list(rdf_ready.keys()))
print("Embedding removed:", "embedding" not in rdf_ready)

# ── 7. Full document with context ────────────────────────────────

print("\n=== 7. Full Document ===\n")

document = {
    "@context": {
        "@vocab": "http://schema.org/",
        **vector_term_definition("embedding", "http://example.org/embedding", 4),
    },
    "@graph": products,
}

print(json.dumps(document, indent=2))
