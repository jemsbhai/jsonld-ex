"""
Example 06: Resource Limits
============================

Demonstrates resource limit enforcement to prevent denial-of-service
attacks via deeply nested documents or oversized payloads.

Use case: API gateway protecting JSON-LD processing endpoints.
"""

import json
from jsonld_ex import JsonLdEx
from jsonld_ex.security import enforce_resource_limits, DEFAULT_RESOURCE_LIMITS

# ── 1. Default resource limits ───────────────────────────────────

print("=== 1. Default Resource Limits ===\n")

for key, value in DEFAULT_RESOURCE_LIMITS.items():
    unit = "bytes" if "size" in key else "levels" if "depth" in key else "seconds"
    print(f"  {key}: {value:,} {unit}")

# ── 2. Document size limit ───────────────────────────────────────

print("\n=== 2. Document Size Limit ===\n")

# Normal document — passes
small_doc = {"@type": "Person", "name": "Alice"}
try:
    enforce_resource_limits(small_doc, {"max_document_size": 1024})
    print(f"  Small document ({len(json.dumps(small_doc))} bytes): ✓ Passed")
except ValueError as e:
    print(f"  Small document: ✗ {e}")

# Oversized document — blocked
large_doc = {"@type": "Person", "name": "A" * 2000}
try:
    enforce_resource_limits(large_doc, {"max_document_size": 1024})
    print(f"  Large document: ✓ Passed")
except ValueError as e:
    print(f"  Large document ({len(json.dumps(large_doc))} bytes): ✗ Blocked")
    print(f"    Reason: {e}")

# ── 3. Nesting depth limit ──────────────────────────────────────

print("\n=== 3. Nesting Depth Limit ===\n")

# Build a deeply nested document
def build_nested(depth: int) -> dict:
    doc = {"@type": "Leaf", "value": "data"}
    for i in range(depth):
        doc = {"@type": f"Level{depth - i}", "@graph": [doc]}
    return doc

# Shallow nesting — passes
shallow = build_nested(5)
try:
    enforce_resource_limits(shallow, {"max_graph_depth": 20})
    print(f"  Depth 5: ✓ Passed (limit=20)")
except ValueError as e:
    print(f"  Depth 5: ✗ {e}")

# Deep nesting — blocked
deep = build_nested(50)
try:
    enforce_resource_limits(deep, {"max_graph_depth": 20})
    print(f"  Depth 50: ✓ Passed")
except ValueError as e:
    print(f"  Depth 50: ✗ Blocked (limit=20)")
    print(f"    Reason: {e}")

# ── 4. Custom limits for different environments ──────────────────

print("\n=== 4. Environment-Specific Limits ===\n")

environments = {
    "IoT / Edge": {
        "max_document_size": 64 * 1024,      # 64 KB
        "max_graph_depth": 10,
        "max_context_depth": 3,
        "max_expansion_time": 5,
    },
    "Web API": {
        "max_document_size": 1 * 1024 * 1024, # 1 MB
        "max_graph_depth": 50,
        "max_context_depth": 10,
        "max_expansion_time": 15,
    },
    "Batch Processing": {
        "max_document_size": 50 * 1024 * 1024, # 50 MB
        "max_graph_depth": 200,
        "max_context_depth": 20,
        "max_expansion_time": 120,
    },
}

for env_name, limits in environments.items():
    print(f"  {env_name}:")
    for key, value in limits.items():
        unit = "bytes" if "size" in key else "levels" if "depth" in key else "seconds"
        print(f"    {key}: {value:,} {unit}")
    print()

# ── 5. Using limits with the processor ───────────────────────────

print("=== 5. Processor with Resource Limits ===\n")

# Create a processor with IoT limits
iot_processor = JsonLdEx(resource_limits={
    "max_document_size": 64 * 1024,
    "max_graph_depth": 10,
})

# Normal IoT reading — passes
sensor_reading = {
    "@context": "http://schema.org/",
    "@type": "Observation",
    "value": 23.5,
    "unit": "celsius",
}

try:
    # Note: expand() will fail without network access to resolve context,
    # but resource limits are checked first
    enforce_resource_limits(sensor_reading, {"max_document_size": 64 * 1024})
    print("  IoT sensor reading: ✓ Within limits")
except ValueError as e:
    print(f"  IoT sensor reading: ✗ {e}")

print("\n  Resource limits protect against:")
print("    • Billion laughs attack (deep nesting)")
print("    • Payload bombs (oversized documents)")
print("    • Circular context references (context depth)")
print("    • Slow-loris attacks (expansion timeout)")
