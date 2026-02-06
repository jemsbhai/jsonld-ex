"""
Example 05: Context Allowlist
==============================

Demonstrates restricting which remote contexts a processor can load,
preventing unauthorized context injection.

Use case: Enterprise API that only trusts specific context providers.
"""

from jsonld_ex.security import is_context_allowed

# ── 1. Exact URL allowlist ───────────────────────────────────────

print("=== 1. Exact URL Allowlist ===\n")

config = {
    "allowed": [
        "https://schema.org/",
        "https://w3id.org/security/v2",
        "https://www.w3.org/ns/credentials/v2",
    ],
}

test_urls = [
    "https://schema.org/",
    "https://w3id.org/security/v2",
    "https://malicious.example.org/fake-context",
    "http://schema.org/",  # HTTP, not HTTPS
]

for url in test_urls:
    allowed = is_context_allowed(url, config)
    status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
    print(f"  {status}: {url}")

# ── 2. Pattern-based allowlist ───────────────────────────────────

print("\n=== 2. Pattern-Based Allowlist ===\n")

config_patterns = {
    "allowed": ["https://schema.org/"],
    "patterns": [
        "https://api.company.example.org/contexts/*",
        "https://w3id.org/*",
    ],
}

test_urls = [
    "https://schema.org/",
    "https://api.company.example.org/contexts/payment-v2",
    "https://api.company.example.org/contexts/user-profile",
    "https://w3id.org/security/v2",
    "https://w3id.org/vc/status-list/2021",
    "https://evil.example.org/contexts/steal-data",
]

for url in test_urls:
    allowed = is_context_allowed(url, config_patterns)
    status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
    print(f"  {status}: {url}")

# ── 3. Block all remote contexts ─────────────────────────────────

print("\n=== 3. Block All Remote Contexts ===\n")

config_blocked = {"block_remote_contexts": True}

test_urls = [
    "https://schema.org/",
    "https://trusted.example.org/context",
]

for url in test_urls:
    allowed = is_context_allowed(url, config_blocked)
    status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
    print(f"  {status}: {url}")

print("\n  Use case: Offline processing or air-gapped environments.")
print("  All contexts must be provided inline or from local cache.")

# ── 4. No allowlist (permissive mode) ────────────────────────────

print("\n=== 4. No Allowlist (Permissive) ===\n")

config_open = {}  # No restrictions

test_urls = [
    "https://schema.org/",
    "https://anything.example.org/any-context",
]

for url in test_urls:
    allowed = is_context_allowed(url, config_open)
    status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
    print(f"  {status}: {url}")

print("\n  Warning: Permissive mode trusts all remote contexts.")
print("  Recommended only for development, not production.")

# ── 5. Enterprise security profile ──────────────────────────────

print("\n=== 5. Enterprise Security Profile ===\n")

enterprise_config = {
    "allowed": [
        "https://schema.org/",
        "https://www.w3.org/ns/credentials/v2",
    ],
    "patterns": [
        "https://api.acmecorp.com/contexts/*",
        "https://internal.acmecorp.com/ontology/*",
    ],
}

scenarios = [
    ("Public schema.org context", "https://schema.org/"),
    ("Internal payment context", "https://api.acmecorp.com/contexts/payment"),
    ("Internal ontology", "https://internal.acmecorp.com/ontology/hr-v3"),
    ("Phishing attempt", "https://api.acmecorp.com.evil.org/contexts/payment"),
    ("Competitor's context", "https://api.competitor.com/contexts/product"),
]

for desc, url in scenarios:
    allowed = is_context_allowed(url, enterprise_config)
    status = "✓ ALLOWED" if allowed else "✗ BLOCKED"
    print(f"  {status}: {desc}")
    print(f"           {url}\n")
