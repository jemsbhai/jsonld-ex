"""
Example 10: Knowledge Graph RAG Pipeline
==========================================

End-to-end use case demonstrating how jsonld-ex extensions support
Retrieval-Augmented Generation (RAG) pipelines that build, validate,
and query knowledge graphs extracted by LLMs.

Pipeline:  Source documents → LLM extraction → JSON-LD with provenance
           → Validation → Embedding → Similarity retrieval → Answer
"""

import json
import math
from datetime import datetime, timezone
from jsonld_ex import (
    JsonLdEx,
    annotate,
    get_confidence,
    get_provenance,
    filter_by_confidence,
    validate_node,
    validate_document,
    validate_vector,
    cosine_similarity,
    compute_integrity,
    verify_integrity,
)
from jsonld_ex.vector import vector_term_definition, extract_vectors, strip_vectors_for_rdf
from jsonld_ex.ai_ml import aggregate_confidence

processor = JsonLdEx()

# ══════════════════════════════════════════════════════════════════
# Stage 1: LLM-Based Entity Extraction
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("STAGE 1: LLM Entity Extraction")
print("=" * 60 + "\n")

# Simulate entities extracted by an LLM from research papers
MODEL_V1 = "https://models.example.org/gpt-extract-v1"
MODEL_V2 = "https://models.example.org/llama-extract-v2"
NOW = datetime.now(timezone.utc).isoformat()

# Two models extract entities from the same document
extractions_model1 = [
    {
        "@type": "Person",
        "@id": "#geoffrey-hinton",
        "name": annotate("Geoffrey Hinton", confidence=0.98,
                         source=MODEL_V1, method="NER", extracted_at=NOW),
        "affiliation": annotate("University of Toronto", confidence=0.92,
                                source=MODEL_V1, method="relation-extraction"),
        "field": annotate("Deep Learning", confidence=0.96,
                          source=MODEL_V1, method="classification"),
    },
    {
        "@type": "ScholarlyArticle",
        "@id": "#attention-paper",
        "name": annotate("Attention Is All You Need", confidence=0.99,
                         source=MODEL_V1, method="title-extraction"),
        "author": annotate("Vaswani et al.", confidence=0.95,
                           source=MODEL_V1, method="NER"),
        "year": annotate("2017", confidence=0.97,
                         source=MODEL_V1, method="date-extraction"),
        "topic": annotate("Transformer Architecture", confidence=0.91,
                          source=MODEL_V1, method="topic-classification"),
    },
    {
        "@type": "Organization",
        "@id": "#google-brain",
        "name": annotate("Google Brain", confidence=0.94,
                         source=MODEL_V1, method="NER"),
        "parentOrganization": annotate("Google", confidence=0.88,
                                       source=MODEL_V1, method="relation-extraction"),
    },
]

extractions_model2 = [
    {
        "@type": "Person",
        "@id": "#geoffrey-hinton",
        "name": annotate("Geoffrey E. Hinton", confidence=0.95,
                         source=MODEL_V2, method="NER", extracted_at=NOW),
        "affiliation": annotate("University of Toronto", confidence=0.89,
                                source=MODEL_V2, method="relation-extraction"),
        "field": annotate("Neural Networks", confidence=0.90,
                          source=MODEL_V2, method="classification"),
    },
]

print(f"Model 1 ({MODEL_V1}): {len(extractions_model1)} entities")
print(f"Model 2 ({MODEL_V2}): {len(extractions_model2)} entities")

# ══════════════════════════════════════════════════════════════════
# Stage 2: Cross-Model Confidence Aggregation
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 2: Cross-Model Confidence Aggregation")
print("=" * 60 + "\n")

# Both models extracted Geoffrey Hinton — aggregate confidence
m1_name = get_confidence(extractions_model1[0]["name"])
m2_name = get_confidence(extractions_model2[0]["name"])

combined = aggregate_confidence([m1_name, m2_name], "mean")
print(f"Hinton name confidence — Model 1: {m1_name}, Model 2: {m2_name}")
print(f"Aggregated (mean): {combined:.4f}")

m1_affil = get_confidence(extractions_model1[0]["affiliation"])
m2_affil = get_confidence(extractions_model2[0]["affiliation"])
combined_affil = aggregate_confidence([m1_affil, m2_affil], "mean")
print(f"\nAffiliation — Model 1: {m1_affil}, Model 2: {m2_affil}")
print(f"Aggregated (mean): {combined_affil:.4f}")

# ══════════════════════════════════════════════════════════════════
# Stage 3: Validation
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 3: Schema Validation")
print("=" * 60 + "\n")

person_shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
    "affiliation": {"@type": "xsd:string"},
}

article_shape = {
    "@type": "ScholarlyArticle",
    "name": {"@required": True, "@type": "xsd:string"},
    "author": {"@required": True, "@type": "xsd:string"},
    "year": {"@required": True, "@type": "xsd:string", "@pattern": r"^\d{4}$"},
}

org_shape = {
    "@type": "Organization",
    "name": {"@required": True, "@type": "xsd:string"},
}

shapes = [person_shape, article_shape, org_shape]

kg_doc = {"@graph": extractions_model1}
result = validate_document(kg_doc, shapes)

print(f"Validation result: {'✓ PASS' if result.valid else '✗ FAIL'}")
print(f"Errors: {len(result.errors)}")
for err in result.errors:
    print(f"  ✗ {err.path}: {err.message}")

# ══════════════════════════════════════════════════════════════════
# Stage 4: Add Vector Embeddings
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 4: Vector Embeddings")
print("=" * 60 + "\n")

# Simulated embeddings (4D for demo; real would be 768+)
entity_embeddings = {
    "#geoffrey-hinton":  [0.82, 0.45, -0.31, 0.67],
    "#attention-paper":  [0.71, 0.52, -0.18, 0.73],
    "#google-brain":     [0.55, 0.38, 0.12, 0.80],
}

# Attach embeddings to entities
for entity in extractions_model1:
    eid = entity["@id"]
    if eid in entity_embeddings:
        vec = entity_embeddings[eid]
        valid, errors = validate_vector(vec, expected_dimensions=4)
        if valid:
            entity["embedding"] = vec
            print(f"  ✓ Attached 4D embedding to {eid}")
        else:
            print(f"  ✗ Invalid embedding for {eid}: {errors}")

# ══════════════════════════════════════════════════════════════════
# Stage 5: Similarity-Based Retrieval (RAG Query)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 5: RAG Query — Similarity Retrieval")
print("=" * 60 + "\n")

# User asks: "Who works on deep learning architectures?"
# Query is embedded by the same model
query_embedding = [0.78, 0.50, -0.25, 0.70]
print(f"Query embedding: {query_embedding}")
print(f"Query: 'Who works on deep learning architectures?'\n")

results = []
for entity in extractions_model1:
    if "embedding" in entity:
        sim = cosine_similarity(query_embedding, entity["embedding"])
        name_val = entity["name"]["@value"] if isinstance(entity["name"], dict) else entity["name"]
        conf = get_confidence(entity["name"])
        results.append({
            "id": entity["@id"],
            "type": entity["@type"],
            "name": name_val,
            "similarity": sim,
            "confidence": conf,
        })

# Rank by similarity
results.sort(key=lambda x: x["similarity"], reverse=True)

print("Retrieval results (ranked by cosine similarity):\n")
for i, r in enumerate(results, 1):
    print(f"  {i}. {r['name']} ({r['type']})")
    print(f"     Similarity: {r['similarity']:.4f}, Confidence: {r['confidence']}")

# ══════════════════════════════════════════════════════════════════
# Stage 6: Confidence-Filtered Answer
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 6: Confidence-Filtered Answer Generation")
print("=" * 60 + "\n")

SIMILARITY_THRESHOLD = 0.95
CONFIDENCE_THRESHOLD = 0.85

print(f"Thresholds — Similarity >= {SIMILARITY_THRESHOLD}, Confidence >= {CONFIDENCE_THRESHOLD}\n")

relevant = [r for r in results
            if r["similarity"] >= SIMILARITY_THRESHOLD
            and r["confidence"] >= CONFIDENCE_THRESHOLD]

if relevant:
    print(f"Found {len(relevant)} high-quality result(s):\n")
    for r in relevant:
        # Get full provenance for attribution
        entity = next(e for e in extractions_model1 if e["@id"] == r["id"])
        prov = get_provenance(entity["name"])

        print(f"  Entity: {r['name']}")
        print(f"  Type: {r['type']}")
        print(f"  Similarity: {r['similarity']:.4f}")
        print(f"  Confidence: {r['confidence']}")
        print(f"  Source model: {prov.source}")
        print(f"  Extraction method: {prov.method}")
else:
    print("No results meet both thresholds.")
    low_sim = [r for r in results if r["similarity"] < SIMILARITY_THRESHOLD]
    if low_sim:
        print(f"  Best match was {low_sim[0]['name']} (sim={low_sim[0]['similarity']:.4f})")

# ══════════════════════════════════════════════════════════════════
# Stage 7: Context Integrity for Knowledge Graph Export
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 7: Knowledge Graph Export with Integrity")
print("=" * 60 + "\n")

export_context = {
    "@vocab": "http://schema.org/",
    "ex": "http://www.w3.org/ns/jsonld-ex/",
    **vector_term_definition("embedding", "http://example.org/embedding", 4),
}

context_hash = compute_integrity(export_context)

final_document = {
    "@context": {
        **export_context,
        "@integrity": context_hash,
    },
    "@graph": extractions_model1,
}

print(f"Context hash: {context_hash[:50]}...")
print(f"Entities: {len(final_document['@graph'])}")
print(f"Integrity verified: {verify_integrity(export_context, context_hash)}")

# ══════════════════════════════════════════════════════════════════
# Stage 8: RDF-Ready Export (vectors stripped)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("STAGE 8: RDF-Ready Export")
print("=" * 60 + "\n")

rdf_entities = []
for entity in extractions_model1:
    rdf_entity = strip_vectors_for_rdf(entity, ["embedding"])
    rdf_entities.append(rdf_entity)

print(f"Original entity keys:  {list(extractions_model1[0].keys())}")
print(f"RDF-ready entity keys: {list(rdf_entities[0].keys())}")
print(f"Embeddings stripped:   {'embedding' not in rdf_entities[0]}")

# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PIPELINE SUMMARY")
print("=" * 60 + "\n")

all_confs = []
for entity in extractions_model1:
    for key, val in entity.items():
        if isinstance(val, dict) and "@confidence" in val:
            all_confs.append(val["@confidence"])

print(f"  Total entities extracted:    {len(extractions_model1)}")
print(f"  Extraction models used:      2 ({MODEL_V1.split('/')[-1]}, {MODEL_V2.split('/')[-1]})")
print(f"  Average confidence:          {sum(all_confs)/len(all_confs):.4f}")
print(f"  Schema validation:           {'PASS' if result.valid else 'FAIL'}")
print(f"  Entities with embeddings:    {sum(1 for e in extractions_model1 if 'embedding' in e)}")
print(f"  Context integrity verified:  True")
print(f"  RAG results above threshold: {len(relevant)}")
print(f"\n  Extensions used: @confidence, @source, @method, @extractedAt,")
print(f"                   @vector, @integrity, @shape")
