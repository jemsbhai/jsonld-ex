"""
Example 08: Document Validation
=================================

Demonstrates validating entire JSON-LD graphs containing multiple
node types against multiple shape definitions.

Use case: Validating an organization's knowledge graph before import.
"""

import json
from jsonld_ex import validate_node, validate_document

# ── 1. Define multiple shapes ────────────────────────────────────

print("=== 1. Multiple Shape Definitions ===\n")

person_shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
    "email": {"@required": True, "@pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$"},
}

organization_shape = {
    "@type": "Organization",
    "name": {"@required": True, "@type": "xsd:string"},
    "url": {"@required": True, "@type": "xsd:string", "@pattern": r"^https?://"},
    "employeeCount": {"@type": "xsd:integer", "@minimum": 1},
}

product_shape = {
    "@type": "Product",
    "name": {"@required": True, "@type": "xsd:string"},
    "price": {"@type": "xsd:double", "@minimum": 0},
    "sku": {"@required": True, "@type": "xsd:string", "@pattern": r"^[A-Z]{2}-\d{4}$"},
}

shapes = [person_shape, organization_shape, product_shape]
print(f"Defined {len(shapes)} shapes: Person, Organization, Product")

# ── 2. Valid document ────────────────────────────────────────────

print("\n=== 2. Valid Document ===\n")

valid_doc = {
    "@graph": [
        {
            "@id": "#alice",
            "@type": "Person",
            "name": "Alice Johnson",
            "email": "alice@acme.com",
        },
        {
            "@id": "#acme",
            "@type": "Organization",
            "name": "Acme Corp",
            "url": "https://acme.example.com",
            "employeeCount": 500,
        },
        {
            "@id": "#widget",
            "@type": "Product",
            "name": "Super Widget",
            "price": 29.99,
            "sku": "SW-1234",
        },
    ]
}

result = validate_document(valid_doc, shapes)
print(f"Valid: {result.valid}")
print(f"Errors: {len(result.errors)}")

# ── 3. Document with errors across node types ────────────────────

print("\n=== 3. Document with Errors ===\n")

invalid_doc = {
    "@graph": [
        {
            "@id": "#bob",
            "@type": "Person",
            "name": "Bob",
            # Missing required email
        },
        {
            "@id": "#badcorp",
            "@type": "Organization",
            "name": "Bad Corp",
            "url": "not-a-url",        # Doesn't match URL pattern
            "employeeCount": 0,         # Below minimum
        },
        {
            "@id": "#product1",
            "@type": "Product",
            "name": "Gadget",
            "price": -5.00,             # Negative price
            "sku": "invalid-sku",       # Doesn't match SKU pattern
        },
        {
            "@id": "#validperson",
            "@type": "Person",
            "name": "Charlie Davis",
            "email": "charlie@example.com",
            # This one is valid
        },
    ]
}

result = validate_document(invalid_doc, shapes)
print(f"Valid: {result.valid}")
print(f"Total errors: {len(result.errors)}\n")

for err in result.errors:
    print(f"  ✗ {err.path}")
    print(f"    [{err.constraint}] {err.message}\n")

# ── 4. Selective validation (only some types) ────────────────────

print("=== 4. Selective Validation ===\n")

# Only validate Person nodes
result = validate_document(invalid_doc, [person_shape])
print(f"Person-only validation:")
print(f"  Errors: {len(result.errors)}")
for err in result.errors:
    print(f"  ✗ {err.path}: {err.message}")

# ── 5. Mixed valid and invalid with summary ─────────────────────

print("\n=== 5. Validation Summary Report ===\n")

large_doc = {
    "@graph": [
        {"@type": "Person", "@id": f"#p{i}", "name": f"Person {i}", "email": f"p{i}@x.com"}
        for i in range(5)
    ] + [
        {"@type": "Person", "@id": "#bad1", "name": ""},         # Empty name
        {"@type": "Person", "@id": "#bad2", "email": "x@y.com"}, # Missing name
        {"@type": "Person", "@id": "#bad3", "name": "Valid", "email": "invalid"},
    ]
}

result = validate_document(large_doc, [person_shape])

total_nodes = len(large_doc["@graph"])
error_nodes = len(set(e.path.split("/")[0] for e in result.errors))

print(f"  Total nodes:   {total_nodes}")
print(f"  Valid nodes:   {total_nodes - error_nodes}")
print(f"  Invalid nodes: {error_nodes}")
print(f"  Total errors:  {len(result.errors)}")
print(f"  Pass rate:     {(total_nodes - error_nodes) / total_nodes * 100:.0f}%")

if result.errors:
    print(f"\n  Error breakdown:")
    from collections import Counter
    constraint_counts = Counter(e.constraint for e in result.errors)
    for constraint, count in constraint_counts.most_common():
        print(f"    {constraint}: {count}")
