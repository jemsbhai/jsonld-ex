"""
Example 04: Context Integrity Verification
============================================

Demonstrates @integrity for cryptographic hash verification of
JSON-LD contexts, preventing context injection attacks.

Use case: Financial API ensuring context hasn't been tampered with.
"""

import json
from jsonld_ex import compute_integrity, verify_integrity
from jsonld_ex.security import integrity_context

# ── 1. Computing integrity hashes ────────────────────────────────

print("=== 1. Computing Integrity Hashes ===\n")

# A trusted context definition
schema_context = {
    "@vocab": "http://schema.org/",
    "name": "http://schema.org/name",
    "amount": {"@id": "http://schema.org/amount", "@type": "xsd:decimal"},
    "currency": "http://schema.org/currency",
}

# Compute hash using different algorithms
sha256_hash = compute_integrity(schema_context, "sha256")
sha384_hash = compute_integrity(schema_context, "sha384")
sha512_hash = compute_integrity(schema_context, "sha512")

print(f"SHA-256: {sha256_hash}")
print(f"SHA-384: {sha384_hash}")
print(f"SHA-512: {sha512_hash}")

# ── 2. Verifying integrity ───────────────────────────────────────

print("\n=== 2. Verifying Integrity ===\n")

# Verify the context matches its hash
is_valid = verify_integrity(schema_context, sha256_hash)
print(f"Original context valid: {is_valid}")  # True

# Simulate a tampered context (attacker swaps field mappings)
tampered_context = {
    **schema_context,
    "source": "http://schema.org/amount",       # Swapped!
    "destination": "http://schema.org/currency", # Swapped!
}

is_valid_tampered = verify_integrity(tampered_context, sha256_hash)
print(f"Tampered context valid: {is_valid_tampered}")  # False — attack detected!

# ── 3. Creating integrity-protected context references ───────────

print("\n=== 3. Integrity-Protected Context ===\n")

protected_ref = integrity_context(
    "https://api.bank.example.org/contexts/payment-v2",
    schema_context,
    "sha256",
)
print(f"Protected reference: {json.dumps(protected_ref, indent=2)}")

# ── 4. Using integrity in a document ─────────────────────────────

print("\n=== 4. Document with Integrity ===\n")

document = {
    "@context": {
        "@id": "https://api.bank.example.org/contexts/payment-v2",
        "@integrity": sha256_hash,
    },
    "@type": "MoneyTransfer",
    "source": "https://bank.example.org/account/alice",
    "destination": "https://bank.example.org/account/bob",
    "amount": "500.00",
    "currency": "USD",
}

print(json.dumps(document, indent=2))

# ── 5. Attack scenario demonstration ─────────────────────────────

print("\n=== 5. Attack Scenario ===\n")

print("Scenario: Attacker performs DNS poisoning to redirect context URL")
print("  to a malicious context that swaps 'source' and 'destination'.\n")

# The legitimate context
legitimate = {"source": "http://schema.org/sender", "destination": "http://schema.org/recipient"}
legitimate_hash = compute_integrity(legitimate)
print(f"Legitimate hash: {legitimate_hash}")

# The attacker's context (swapped mappings)
malicious = {"source": "http://schema.org/recipient", "destination": "http://schema.org/sender"}
malicious_hash = compute_integrity(malicious)
print(f"Malicious hash:  {malicious_hash}")

# Verification catches the attack
attack_detected = not verify_integrity(malicious, legitimate_hash)
print(f"\nAttack detected: {attack_detected}")
print("Result: Processor rejects the tampered context → transaction blocked.")
