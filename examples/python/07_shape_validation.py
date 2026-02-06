"""
Example 07: Shape Validation
==============================

Demonstrates @shape for native JSON-LD validation with all
constraint types: required, type, min/max, length, and pattern.

Use case: API input validation for a user registration endpoint.
"""

import json
from jsonld_ex import validate_node
from jsonld_ex.validation import ValidationResult

# ── 1. Defining shapes ──────────────────────────────────────────

print("=== 1. Shape Definitions ===\n")

person_shape = {
    "@type": "Person",
    "name": {
        "@required": True,
        "@type": "xsd:string",
        "@minLength": 1,
        "@maxLength": 100,
    },
    "email": {
        "@required": True,
        "@type": "xsd:string",
        "@pattern": r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
    },
    "age": {
        "@type": "xsd:integer",
        "@minimum": 0,
        "@maximum": 150,
    },
    "bio": {
        "@type": "xsd:string",
        "@maxLength": 500,
    },
}

print("Person shape constraints:")
for prop, constraint in person_shape.items():
    if prop == "@type":
        print(f"  Type: {constraint}")
    elif isinstance(constraint, dict):
        parts = [f"{k}={v}" for k, v in constraint.items()]
        print(f"  {prop}: {', '.join(parts)}")

# ── 2. Valid node ────────────────────────────────────────────────

print("\n=== 2. Valid Node ===\n")

valid_person = {
    "@type": "Person",
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "age": 30,
    "bio": "Software engineer who loves open source.",
}

result = validate_node(valid_person, person_shape)
print(f"Valid: {result.valid}")
print(f"Errors: {len(result.errors)}")

# ── 3. Required field missing ────────────────────────────────────

print("\n=== 3. Required Field Missing ===\n")

missing_email = {
    "@type": "Person",
    "name": "Bob Smith",
    "age": 25,
}

result = validate_node(missing_email, person_shape)
print(f"Valid: {result.valid}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 4. Type mismatches ──────────────────────────────────────────

print("\n=== 4. Type Mismatches ===\n")

bad_types = {
    "@type": "Person",
    "name": 12345,          # Should be string
    "email": "a@b.com",
    "age": "thirty",        # Should be integer
}

result = validate_node(bad_types, person_shape)
print(f"Valid: {result.valid}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 5. Range violations ─────────────────────────────────────────

print("\n=== 5. Range Violations ===\n")

out_of_range = {
    "@type": "Person",
    "name": "Charlie",
    "email": "c@d.com",
    "age": -5,    # Below minimum
}

result = validate_node(out_of_range, person_shape)
print(f"Valid: {result.valid}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# Over maximum
too_old = {
    "@type": "Person",
    "name": "Methuselah",
    "email": "m@bible.org",
    "age": 969,   # Above maximum
}

result = validate_node(too_old, person_shape)
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 6. String length violations ──────────────────────────────────

print("\n=== 6. String Length Violations ===\n")

length_issues = {
    "@type": "Person",
    "name": "",               # Below minLength
    "email": "a@b.com",
    "bio": "x" * 600,        # Exceeds maxLength
}

result = validate_node(length_issues, person_shape)
print(f"Valid: {result.valid}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 7. Pattern violations ───────────────────────────────────────

print("\n=== 7. Pattern Violations ===\n")

bad_patterns = {
    "@type": "Person",
    "name": "Diana",
    "email": "not-an-email",   # Doesn't match email pattern
}

result = validate_node(bad_patterns, person_shape)
print(f"Valid: {result.valid}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 8. Multiple errors at once ───────────────────────────────────

print("\n=== 8. Multiple Errors ===\n")

everything_wrong = {
    "@type": "Organization",  # Wrong type
    "age": -10,               # Below minimum, name missing, email missing
}

result = validate_node(everything_wrong, person_shape)
print(f"Valid: {result.valid}")
print(f"Total errors: {len(result.errors)}")
for err in result.errors:
    print(f"  ✗ [{err.constraint}] {err.path}: {err.message}")

# ── 9. JSON-LD @value wrapped values ────────────────────────────

print("\n=== 9. @value Wrapped Values ===\n")

wrapped_node = {
    "@type": "Person",
    "name": {"@value": "Eve Nakamura"},
    "email": {"@value": "eve@example.com"},
    "age": {"@value": 28},
}

result = validate_node(wrapped_node, person_shape)
print(f"Valid: {result.valid}")
print("  @shape validation correctly unwraps @value objects.")
