# Security Extensions

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/security](https://jsonld-ex.github.io/ns/security)

---

## 1. Introduction

This document specifies the security extensions for jsonld-ex. Three mechanisms — context integrity verification, context allowlists, and resource limits — defend JSON-LD processing against context injection, resource exhaustion, and unauthorized remote context loading.

The formal property definition for `@integrity` (IRI, type, format) is in the [Vocabulary](vocabulary.md) specification, §15. This document defines the threat model, processing algorithms, configuration semantics, and enforcement behavior.

### 1.1 Motivation

JSON-LD's power derives from remote contexts: a single `@context` URL maps compact terms to full IRIs, enabling interoperable data exchange. However, remote context loading introduces an attack surface. A compromised or malicious context can silently redefine the meaning of every term in a document, turning a valid financial transfer into a fraudulent one or a benign medical record into a harmful instruction.

JSON-LD 1.1 acknowledges these risks in its security considerations but provides no standardized defense mechanisms. Processors are left to implement ad hoc protections — or none at all. The jsonld-ex security extensions fill this gap with three composable, backward-compatible mechanisms that can be adopted incrementally.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.3 Terminology

- **Context** — a JSON-LD context document (a JSON object or string URL) that maps terms to IRIs.
- **Remote context** — a context identified by a URL that must be fetched over the network.
- **Integrity string** — a value of the form `algorithm-base64hash` (e.g., `sha256-n4bQgY...`) that cryptographically identifies the expected content of a context.
- **Allowlist** — a configuration object specifying which context URLs a processor is permitted to load.
- **Resource limits** — numeric bounds on document size, nesting depth, context chain depth, and processing time.
- **Fail-closed** — a security posture where verification failure causes rejection rather than fallback to unverified processing.

---

## 2. Threat Model

This section describes the threats that the jsonld-ex security extensions are designed to mitigate. Each threat is paired with the mechanism that addresses it.

### 2.1 Context Injection via DNS Poisoning or MITM

**Threat:** An attacker compromises the resolution of a context URL — through DNS poisoning, BGP hijacking, or man-in-the-middle interception — and serves a malicious context that redefines term mappings.

**Example:** A document declares `@context: "https://payments.example.org/v1"`. The legitimate context maps `"source"` to `schema:sender` and `"destination"` to `schema:recipient`. An attacker-controlled context swaps these mappings. A processor using the poisoned context interprets the document with reversed sender and recipient, potentially misdirecting a financial transfer.

**Mitigation:** Context integrity verification (§3). The document author embeds a cryptographic hash of the expected context content. Any modification — including semantically targeted term swaps — changes the hash, causing the processor to reject the context.

### 2.2 Recursion Bombs (Deeply Nested Documents)

**Threat:** An attacker crafts a document with extreme nesting depth — deeply nested `@graph` containers, arrays of arrays, or recursive object structures — to exhaust stack space or memory and crash the processor.

**Example:**

```json
{"@graph": {"@graph": {"@graph": {"@graph": "... hundreds of levels deep ..."}}}}
```

**Mitigation:** Resource limits (§5). The `max_graph_depth` parameter bounds the nesting depth of any document. Documents exceeding the limit are rejected before full processing begins.

### 2.3 Document Size Attacks

**Threat:** An attacker submits an extremely large document to exhaust memory or disk space during parsing. This is especially relevant for services that accept JSON-LD input from untrusted sources (e.g., APIs, data pipelines, IoT gateways).

**Mitigation:** Resource limits (§5). The `max_document_size` parameter bounds the byte size of the serialized document.

### 2.4 Context Chain Depth Attacks

**Threat:** A context references another context, which references another, forming a long or circular chain. This can exhaust network connections, memory, or processing time.

**Mitigation:** Resource limits (§5). The `max_context_depth` parameter bounds the depth of transitive context references.

### 2.5 Unauthorized Remote Context Loading

**Threat:** A document from an untrusted source references an attacker-controlled context URL. Loading that URL may leak information (via the HTTP request itself), introduce malicious term mappings, or expose the processor to server-side attacks (e.g., slow responses causing resource exhaustion).

**Mitigation:** Context allowlists (§4). The processor is configured to load only contexts from approved URLs or URL patterns.

---

## 3. Context Integrity Verification

Context integrity verification enables a document author to declare the expected cryptographic hash of a context document. A conforming processor MUST verify the hash before using the context, and MUST reject the context if verification fails.

### 3.1 Integrity String Format

An integrity string has the format:

```
algorithm-base64hash
```

Where:

- `algorithm` is one of the supported hash algorithms (see §3.2).
- The literal `-` character separates the algorithm name from the hash.
- `base64hash` is the Base64-encoded (standard alphabet, with padding) digest of the context content.

**Examples:**

```
sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg=
sha384-OLBgp1GsljhM2TJ+sbHjaiH9txEUvgdDTAzHv2P24donTt6/529l+9Ua0vFImLlb
sha512-MJ7MSJwS1utMxA9QyQLytNDtd+5RGnx6m808qG1M2G+YndNbxf9JlnDaNCVbRbDP2DDoH2Bdz33FVC6TrpzXbw==
```

### 3.2 Supported Algorithms

| Algorithm | Hash Length (bits) | Base64 Length (chars) |
|-----------|-------------------|----------------------|
| `sha256` | 256 | 44 |
| `sha384` | 384 | 64 |
| `sha512` | 512 | 88 |

Processors MUST support all three algorithms. `sha256` is RECOMMENDED as the default.

Processors MUST reject integrity strings that specify an unsupported algorithm.

### 3.3 Computing an Integrity Hash

To compute the integrity hash of a context:

**Input:** A context value (string or JSON object) and an algorithm identifier.

**Procedure:**

1. If the context is a JSON object (dict), serialize it to a JSON string using canonical key ordering (keys sorted lexicographically). This ensures that semantically identical contexts produce identical hashes regardless of key order in the source document.
2. If the context is already a string, use it as-is.
3. Encode the string as UTF-8.
4. Compute the cryptographic hash using the specified algorithm.
5. Base64-encode the hash digest using the standard Base64 alphabet (RFC 4648 §4) with padding.
6. Return the string `algorithm-base64hash`.

**Example:** Given the context `{"name": "http://schema.org/name"}`:

1. Canonical serialization: `{"name": "http://schema.org/name"}` (already sorted).
2. UTF-8 encoding produces the byte sequence.
3. SHA-256 hash of those bytes produces a 32-byte digest.
4. Base64 encoding produces a 44-character string.
5. Result: `sha256-<base64hash>`.

### 3.4 Declaring Integrity in a Document

A context reference with integrity verification uses an object with `@id` and `@integrity`:

```json
{
  "@context": {
    "@id": "https://schema.org/",
    "@integrity": "sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
  }
}
```

The `@id` identifies the context URL. The `@integrity` declares the expected hash.

When used in a context array, the integrity-verified context appears as an object element:

```json
{
  "@context": [
    {
      "@id": "https://schema.org/",
      "@integrity": "sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
    },
    {
      "@id": "https://example.org/custom-context",
      "@integrity": "sha384-OLBgp1GsljhM2TJ+sbHjaiH9txEUvgdDTAzHv2P24donTt6/529l+9Ua0vFImLlb"
    }
  ]
}
```

### 3.5 Verification Algorithm

**Input:** A context reference (object with `@id` and `@integrity`) and the retrieved context content.

**Procedure:**

1. Parse the integrity string by splitting on the first `-` character.
2. Validate that the result has exactly two parts. If not, raise an error.
3. Validate that the algorithm part is one of the supported algorithms (`sha256`, `sha384`, `sha512`). If not, raise an error.
4. Compute the integrity hash of the retrieved context content using the declared algorithm (§3.3).
5. Compare the computed integrity string to the declared integrity string.
6. If the strings are equal, verification succeeds. The context MAY be used.
7. If the strings differ, verification fails. The processor MUST reject the context.

### 3.6 Fail-Closed Behavior

When `@integrity` is declared on a context reference:

- Processors MUST NOT use the context if verification fails.
- Processors MUST NOT fall back to using the context without verification.
- Processors MUST NOT silently ignore the `@integrity` keyword.
- A verification failure SHOULD produce a clear error indicating the expected and computed hashes, enabling diagnosis of unintentional changes (e.g., a context update that the document author has not yet accounted for).

### 3.7 Creating Integrity-Verified Context References

The `integrity_context` convenience function produces a context reference with a computed integrity hash:

**Input:** A context URL (string), the context content (string or JSON object), and an optional algorithm (default: `sha256`).

**Output:** A JSON object with `@id` and `@integrity`.

**Example:**

```json
// Input: url="https://schema.org/", content=<schema.org context>, algorithm="sha256"
// Output:
{
  "@id": "https://schema.org/",
  "@integrity": "sha256-n4bQgYhMfWWaL+qgxVrQFaO/TxsrC4Is0V1sFbDwCgg="
}
```

---

## 4. Context Allowlists

Context allowlists restrict which remote context URLs a processor is permitted to load. This prevents documents from untrusted sources from introducing arbitrary remote contexts.

### 4.1 Configuration Model

An allowlist configuration is a JSON object with the following properties:

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `allowed` | Array of strings | `[]` | Exact context URLs that are permitted |
| `patterns` | Array of strings | `[]` | URL patterns with wildcards |
| `block_remote_contexts` | Boolean | `false` | If `true`, reject ALL remote contexts |

**Example configuration:**

```json
{
  "allowed": [
    "https://schema.org/",
    "https://w3id.org/security/v2"
  ],
  "patterns": [
    "https://example.org/contexts/*"
  ],
  "block_remote_contexts": false
}
```

### 4.2 Pattern Syntax

Patterns use two wildcard characters:

| Wildcard | Meaning | Example |
|----------|---------|---------|
| `*` | Matches zero or more characters | `https://example.org/*` matches `https://example.org/v1`, `https://example.org/v2/context` |
| `?` | Matches exactly one character | `https://example.org/v?` matches `https://example.org/v1`, `https://example.org/v2` but not `https://example.org/v10` |

Patterns are matched against the entire URL. A pattern `https://example.org/*` does NOT match `https://other.org/`.

All other characters in a pattern are matched literally. Special regex characters in the URL or pattern (e.g., `.`, `+`, `(`) are escaped — they carry no special meaning.

### 4.3 Matching Algorithm

**Input:** A context URL (string) and an allowlist configuration (object).

**Procedure:**

1. If `block_remote_contexts` is `true`, return **denied**.
2. If the URL appears in the `allowed` array (exact string match), return **permitted**.
3. For each pattern in the `patterns` array:
   a. Convert the pattern to a regular expression by escaping all regex-special characters, then replacing escaped `\*` with `.*` and escaped `\?` with `.`.
   b. Anchor the regex with `^` and `$`.
   c. If the URL matches the regex, return **permitted**.
4. If the `allowed` array is non-empty OR the `patterns` array is non-empty (i.e., an allowlist is actively configured), return **denied**. The URL was not matched by any rule.
5. If both `allowed` and `patterns` are empty and `block_remote_contexts` is `false`, return **permitted**. No allowlist is configured, so all URLs are allowed by default.

### 4.4 Open-by-Default Semantics

When no allowlist is configured (empty `allowed`, empty `patterns`, `block_remote_contexts` is `false`), the processor permits all remote contexts. This preserves backward compatibility with standard JSON-LD 1.1 processing.

Allowlist enforcement is **opt-in**: it activates only when at least one of `allowed`, `patterns`, or `block_remote_contexts` is explicitly set.

### 4.5 Complete Blocking Mode

Setting `block_remote_contexts` to `true` rejects all remote context URLs, regardless of the contents of `allowed` and `patterns`. This is the most restrictive configuration, suitable for environments where all contexts are embedded inline or pre-loaded.

---

## 5. Resource Limits

Resource limits bound the size and complexity of JSON-LD documents before full processing begins. They defend against resource exhaustion attacks from oversized or deeply nested input.

### 5.1 Limit Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_context_depth` | Integer | 10 | Maximum depth of transitive context chains (context A references B which references C, etc.) |
| `max_graph_depth` | Integer | 100 | Maximum nesting depth of the document structure (nested objects, arrays, `@graph` containers) |
| `max_document_size` | Integer (bytes) | 10,485,760 (10 MB) | Maximum byte size of the serialized document |
| `max_expansion_time` | Integer (seconds) | 30 | Maximum wall-clock time for expansion processing |

### 5.2 Default Values

The defaults are designed to accommodate the vast majority of legitimate JSON-LD documents while providing meaningful protection:

- **`max_context_depth = 10`**: Real-world context chains rarely exceed 3–4 levels. A limit of 10 provides generous headroom while preventing unbounded chains.
- **`max_graph_depth = 100`**: Legitimate documents occasionally reach 20–30 levels of nesting (e.g., deeply nested organizational hierarchies). A limit of 100 provides ample margin.
- **`max_document_size = 10 MB`**: The schema.org context is approximately 1.5 MB. A 10 MB limit accommodates large but legitimate documents while preventing memory exhaustion from multi-gigabyte payloads.
- **`max_expansion_time = 30 seconds`**: Complex expansion with multiple remote context fetches may take several seconds. A 30-second limit prevents indefinite hangs while allowing for network latency.

Processors SHOULD use these defaults when no explicit limits are provided. Processors MAY allow callers to override any or all parameters.

### 5.3 Enforcement Algorithm

**Input:** A document (string, dict, or list) and an optional limits configuration (object with any subset of the four parameters).

**Procedure:**

1. Merge the provided limits with the defaults. Provided values override defaults; missing values use defaults.
2. **Type validation:** The document MUST be a string, dict, or list. Reject other types with a type error.
3. **Size check:**
   a. If the document is a string, measure its byte length.
   b. If the document is a dict or list, serialize it to JSON and measure the byte length of the serialized form.
   c. If the byte length exceeds `max_document_size`, reject the document with a size error.
4. **Parse:** If the document is a string, parse it as JSON. If parsing fails, reject with a parse error.
5. **Depth check:** Measure the nesting depth of the parsed structure (see §5.4). If the depth exceeds `max_graph_depth`, reject the document with a depth error.
6. If all checks pass, the document is accepted for further processing.

Note: `max_context_depth` and `max_expansion_time` are enforced during context resolution and expansion respectively, not during the initial document validation step. The enforcement algorithm above covers the pre-processing checks (`max_document_size` and `max_graph_depth`).

### 5.4 Depth Measurement

The nesting depth of a JSON structure is measured recursively:

**Input:** A JSON value and a current depth counter (initially 0).

**Procedure:**

1. If the current depth exceeds the safety cap (500), return the current depth. This prevents stack overflow in the measurement function itself when processing adversarial input.
2. If the value is `null`, a string, a number, or a boolean, return the current depth.
3. If the value is an object (dict), recurse into each value with `current depth + 1`. Return the maximum depth found.
4. If the value is an array (list), recurse into each element with `current depth + 1`. Return the maximum depth found.

**Example:** The document `{"a": {"b": {"c": 1}}}` has depth 3. The document `{"a": [{"b": 1}, {"c": 2}]}` has depth 2 (the array adds one level, each element adds one more, and the maximum is 2).

### 5.5 Error Reporting

When a resource limit is exceeded, the error SHOULD include:

- Which limit was exceeded (e.g., `max_document_size`, `max_graph_depth`).
- The measured value (e.g., document size in bytes, nesting depth).
- The configured limit.

**Example error message:**

```
Document size 15728640 exceeds limit 10485760
```

```
Document depth 150 exceeds limit 100
```

---

## 6. Processing Model

When all three security mechanisms are in use, they are applied in the following order:

1. **Allowlist check** — Before fetching any remote context, verify the URL against the allowlist configuration. If denied, reject the document without making a network request.
2. **Integrity verification** — After fetching a context (if permitted by the allowlist), verify its content against the declared `@integrity` hash. If verification fails, reject the context.
3. **Resource limit enforcement** — Before expanding the document, validate it against the configured resource limits (size, depth). If any limit is exceeded, reject the document.
4. **Standard processing** — If all checks pass, proceed with normal JSON-LD expansion, compaction, or other operations.

This ordering is designed to minimize unnecessary work: cheap checks (allowlist, size) run before expensive ones (hashing, depth traversal, expansion).

Each mechanism is independently optional. A processor MAY implement any combination:

- Integrity verification only (detect context tampering).
- Allowlists only (restrict context sources).
- Resource limits only (prevent resource exhaustion).
- Any combination of two or all three.

---

## 7. Backward Compatibility

### 7.1 Non-Extended Processors

A standard JSON-LD 1.1 processor that does not implement the jsonld-ex security extensions will:

- **`@integrity`**: Treat the context reference object `{"@id": "...", "@integrity": "..."}` as a context identified by the `@id` value. The `@integrity` property is ignored. The context is loaded and used without verification. This is semantically correct but not security-hardened — the document is processed normally, just without the tamper-detection guarantee.
- **Allowlists and resource limits**: These are processor configuration, not document-level keywords. Non-extended processors simply do not have these configurations and process all documents without restriction, which is the current JSON-LD 1.1 behavior.

### 7.2 Graceful Degradation

The security extensions degrade gracefully:

- Documents with `@integrity` are valid JSON-LD with or without integrity verification.
- No new keywords are required in the document for allowlists or resource limits — they are processor-side configurations.
- A processor that implements only a subset of the security mechanisms still provides value for the mechanisms it does implement.

### 7.3 Migration Path

Adopting the security extensions requires no changes to existing JSON-LD documents. Authors can add `@integrity` to context references incrementally, one context at a time. Processor operators can enable allowlists and resource limits independently.

---

## 8. Relationship to Existing Standards

### 8.1 W3C Subresource Integrity (SRI)

[Subresource Integrity](https://www.w3.org/TR/SRI/) (W3C Recommendation, 2016) defines integrity verification for web resources loaded via HTML `<script>` and `<link>` elements. The jsonld-ex `@integrity` mechanism is directly inspired by SRI:

| Aspect | SRI | jsonld-ex `@integrity` |
|--------|-----|----------------------|
| **Scope** | HTML subresources (`<script>`, `<link>`) | JSON-LD context references |
| **Format** | `algorithm-base64hash` | `algorithm-base64hash` (identical) |
| **Algorithms** | sha256, sha384, sha512 | sha256, sha384, sha512 (identical) |
| **Multiple hashes** | Supported (space-separated) | Single hash per context reference |
| **Failure behavior** | Block resource loading | Reject context (fail-closed) |

The deliberate alignment with SRI's format means that tools and libraries for computing SRI hashes can be reused for jsonld-ex integrity strings. The only difference is the application context: SRI protects browser subresources; jsonld-ex `@integrity` protects JSON-LD contexts.

### 8.2 Content Security Policy (CSP)

[Content Security Policy](https://www.w3.org/TR/CSP3/) restricts which sources a browser may load resources from. The jsonld-ex context allowlist serves an analogous role for JSON-LD processors:

| Aspect | CSP | jsonld-ex Allowlists |
|--------|-----|---------------------|
| **Scope** | Browser resource loading | JSON-LD context loading |
| **Granularity** | Per-resource-type (script-src, style-src, etc.) | Context URLs only |
| **Wildcards** | `*` in domain position | `*` and `?` in URL patterns |
| **Block-all mode** | `default-src 'none'` | `block_remote_contexts: true` |
| **Configuration** | HTTP header or `<meta>` tag | Processor configuration object |

### 8.3 JSON-LD 1.1 Security Considerations

The JSON-LD 1.1 specification (§5, Security Considerations) identifies the risks of remote context loading and recommends that implementations "provide mechanisms to limit" context loading. The jsonld-ex security extensions provide concrete, standardized implementations of these recommendations.

---

## 9. Reference Implementation

The reference implementation is in the `jsonld_ex.security` module of the [jsonld-ex Python package](https://pypi.org/project/jsonld-ex/).

| Function | Spec Section |
|----------|-------------|
| `compute_integrity(context, algorithm)` | §3.3 |
| `verify_integrity(context, declared)` | §3.5 |
| `integrity_context(url, content, algorithm)` | §3.7 |
| `is_context_allowed(url, config)` | §4.3 |
| `enforce_resource_limits(document, limits)` | §5.3 |

---

## References

- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- W3C. (2016). Subresource Integrity. W3C Recommendation. https://www.w3.org/TR/SRI/
- W3C. (2023). Content Security Policy Level 3. W3C Working Draft. https://www.w3.org/TR/CSP3/
- Bradner, S. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. https://www.rfc-editor.org/rfc/rfc2119
