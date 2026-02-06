"""
Example 02: Provenance Tracking
================================

Demonstrates full provenance metadata: @source, @extractedAt,
@method, and @humanVerified for AI-generated knowledge graphs.

Use case: Tracking which ML model produced each assertion, when,
and whether a human has verified it.
"""

import json
from datetime import datetime, timezone
from jsonld_ex import annotate, JsonLdEx
from jsonld_ex.ai_ml import get_provenance

processor = JsonLdEx()

# ── 1. Full provenance annotation ────────────────────────────────

print("=== 1. Full Provenance Annotation ===\n")

# NER model extracts a person name
name = annotate(
    "Dr. Sarah Chen",
    confidence=0.93,
    source="https://models.example.org/ner-bert-large-v4",
    extracted_at=datetime.now(timezone.utc).isoformat(),
    method="NER",
    human_verified=False,
)
print("Annotated name:")
print(json.dumps(name, indent=2))

# Relation extraction model identifies employer
employer = annotate(
    "MIT",
    confidence=0.76,
    source="https://models.example.org/rel-extract-v2",
    extracted_at=datetime.now(timezone.utc).isoformat(),
    method="relation-extraction",
    human_verified=False,
)

# ── 2. Extracting provenance metadata ────────────────────────────

print("\n=== 2. Extracting Provenance ===\n")

prov = get_provenance(name)
print(f"Confidence:     {prov.confidence}")
print(f"Source model:   {prov.source}")
print(f"Extracted at:   {prov.extracted_at}")
print(f"Method:         {prov.method}")
print(f"Human verified: {prov.human_verified}")

# ── 3. Multi-model provenance pipeline ───────────────────────────

print("\n=== 3. Multi-Model Pipeline ===\n")

# Simulate a pipeline: NER → Entity Linking → Sentiment
entity_pipeline = {
    "@context": "http://schema.org/",
    "@type": "Person",
    "@id": "http://example.org/entity/sarah-chen",
    "name": annotate(
        "Dr. Sarah Chen",
        confidence=0.93,
        source="https://models.example.org/ner-v4",
        method="NER",
    ),
    "sameAs": annotate(
        "https://www.wikidata.org/wiki/Q12345",
        confidence=0.81,
        source="https://models.example.org/entity-linker-v2",
        method="entity-linking",
    ),
    "description": annotate(
        "Renowned AI researcher known for work in NLP",
        confidence=0.67,
        source="https://models.example.org/summarizer-v1",
        method="abstractive-summarization",
    ),
}

print("Multi-model knowledge graph node:")
print(json.dumps(entity_pipeline, indent=2))

# ── 4. Human verification workflow ───────────────────────────────

print("\n=== 4. Human Verification Workflow ===\n")

# Before verification
unverified = annotate(
    "Machine Learning Engineer",
    confidence=0.72,
    source="https://models.example.org/role-classifier-v1",
    method="classification",
    human_verified=False,
)
print(f"Before review: confidence={unverified['@confidence']}, "
      f"verified={unverified['@humanVerified']}")

# After human verifies and corrects
verified = annotate(
    "Senior ML Engineer",  # Human corrected the title
    confidence=1.0,        # Human-verified = full confidence
    source="https://models.example.org/role-classifier-v1",
    method="classification",
    human_verified=True,
)
print(f"After review:  confidence={verified['@confidence']}, "
      f"verified={verified['@humanVerified']}")

# ── 5. Comparing provenance across sources ───────────────────────

print("\n=== 5. Comparing Sources ===\n")

# Same fact extracted by different models
extractions = [
    annotate("MIT", confidence=0.91, source="model-A", method="NER"),
    annotate("MIT", confidence=0.85, source="model-B", method="NER"),
    annotate("Massachusetts Institute of Technology", confidence=0.78,
             source="model-C", method="entity-resolution"),
]

for i, ext in enumerate(extractions):
    prov = get_provenance(ext)
    print(f"  Source {i+1}: '{ext['@value']}' "
          f"(conf={prov.confidence}, model={prov.source}, method={prov.method})")

# Aggregate confidence from all models
scores = [get_provenance(e).confidence for e in extractions]
combined = processor.aggregate_confidence(scores, "mean")
print(f"\n  Combined confidence: {combined:.4f}")
