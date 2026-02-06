"""
Example 01: Confidence Basics
=============================

Demonstrates how to annotate JSON-LD values with confidence scores,
extract them, filter graphs by confidence, and aggregate scores.

Use case: An NER model extracts person names from text with varying certainty.
"""

from jsonld_ex import annotate, JsonLdEx

processor = JsonLdEx()

# ── 1. Annotating values with confidence ─────────────────────────

print("=== 1. Annotating Values ===\n")

# High-confidence extraction
name_high = annotate("John Smith", confidence=0.95)
print(f"High confidence: {name_high}")
# {'@value': 'John Smith', '@confidence': 0.95}

# Low-confidence extraction
name_low = annotate("J. Smith", confidence=0.45)
print(f"Low confidence:  {name_low}")

# Perfect confidence (human-entered data)
name_exact = annotate("Jane Doe", confidence=1.0)
print(f"Exact value:     {name_exact}")

# ── 2. Extracting confidence from nodes ──────────────────────────

print("\n=== 2. Extracting Confidence ===\n")

node = {
    "@type": "Person",
    "name": {"@value": "John Smith", "@confidence": 0.95},
    "email": {"@value": "john@example.com", "@confidence": 0.72},
    "age": 30,  # No confidence — human-entered
}

name_conf = processor.get_confidence(node["name"])
email_conf = processor.get_confidence(node["email"])
age_conf = processor.get_confidence(node.get("age"))

print(f"Name confidence:  {name_conf}")   # 0.95
print(f"Email confidence: {email_conf}")  # 0.72
print(f"Age confidence:   {age_conf}")    # None (no annotation)

# ── 3. Filtering a graph by confidence threshold ─────────────────

print("\n=== 3. Filtering by Confidence ===\n")

knowledge_graph = [
    {"@id": "#person1", "name": annotate("Alice Johnson", confidence=0.97)},
    {"@id": "#person2", "name": annotate("Bob Smith", confidence=0.62)},
    {"@id": "#person3", "name": annotate("C. Williams", confidence=0.31)},
    {"@id": "#person4", "name": annotate("Diana Prince", confidence=0.88)},
    {"@id": "#person5", "name": annotate("E. Unknown", confidence=0.15)},
]

# Keep only high-confidence extractions (>= 0.7)
high_conf = processor.filter_by_confidence(knowledge_graph, "name", 0.7)
print(f"Nodes with confidence >= 0.7: {len(high_conf)}")
for node in high_conf:
    print(f"  {node['@id']}: {node['name']['@value']}")

# Medium confidence threshold (>= 0.5)
medium_conf = processor.filter_by_confidence(knowledge_graph, "name", 0.5)
print(f"\nNodes with confidence >= 0.5: {len(medium_conf)}")

# ── 4. Aggregating confidence scores ─────────────────────────────

print("\n=== 4. Aggregating Confidence ===\n")

# Multiple models extracted the same entity with different confidences
model_scores = [0.92, 0.87, 0.78]

mean_conf = processor.aggregate_confidence(model_scores, "mean")
max_conf = processor.aggregate_confidence(model_scores, "max")
min_conf = processor.aggregate_confidence(model_scores, "min")

print(f"Mean confidence:  {mean_conf:.4f}")  # Average of all models
print(f"Max confidence:   {max_conf:.4f}")   # Best model's score
print(f"Min confidence:   {min_conf:.4f}")   # Worst model's score

# Weighted aggregation (trust some models more)
# Model 1 is the best, model 3 is weakest
weighted = processor.aggregate_confidence(
    model_scores, "weighted", weights=[3.0, 2.0, 1.0]
)
print(f"Weighted (3:2:1): {weighted:.4f}")

# ── 5. Confidence in a full JSON-LD document ─────────────────────

print("\n=== 5. Full Document Example ===\n")

document = {
    "@context": "http://schema.org/",
    "@type": "Person",
    "@id": "http://example.org/person/12345",
    "name": annotate("John Smith", confidence=0.95),
    "jobTitle": annotate("Software Engineer", confidence=0.78),
    "worksFor": {
        "@type": "Organization",
        "name": annotate("Acme Corp", confidence=0.85),
    },
}

print("Document:")
import json
print(json.dumps(document, indent=2))
