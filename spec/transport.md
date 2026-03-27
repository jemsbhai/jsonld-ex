# Transport Extensions

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/transport](https://jsonld-ex.github.io/ns/transport)

---

## 1. Introduction

This document specifies the transport extensions for jsonld-ex. Two modules — CBOR-LD serialization and MQTT transport optimization — enable efficient transmission of JSON-LD documents over bandwidth-constrained and publish/subscribe networks, with particular attention to IoT and sensor data pipelines.

The CBOR-LD module provides binary serialization with context compression, reducing payload size for repeated context references. The MQTT module builds on CBOR-LD to provide automatic topic derivation, quality-of-service mapping from confidence metadata, and MQTT 5.0 PUBLISH property generation — bridging JSON-LD semantics and MQTT transport semantics.

### 1.1 Motivation

IoT and edge computing environments impose constraints that standard JSON-LD processing does not address:

- **Bandwidth**: Cellular, LoRa, and satellite links have limited throughput. A JSON-LD document with a full schema.org context reference can be several hundred bytes before any data is included. Binary serialization with context compression can reduce payloads by 40–70%.
- **Publish/subscribe routing**: MQTT brokers route messages by topic. Deriving topics from JSON-LD metadata (`@type`, `@id`) enables semantic routing without application-level dispatching.
- **Quality of service**: MQTT defines three QoS levels (0, 1, 2) with increasing delivery guarantees and overhead. Mapping QoS to the confidence or verification status of the data enables automatic resource allocation — high-confidence or human-verified data gets reliable delivery; noisy sensor telemetry gets best-effort.
- **Message expiry**: MQTT 5.0 supports message expiry intervals. Temporal annotations (`@validUntil`) provide a natural source for this metadata — a message whose assertion has expired is not worth delivering.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.3 Terminology

- **CBOR** — Concise Binary Object Representation ([RFC 8949](https://www.rfc-editor.org/rfc/rfc8949)).
- **CBOR-LD** — A method for compressing JSON-LD documents using CBOR, including context compression via integer registry IDs.
- **MQTT** — Message Queuing Telemetry Transport, a lightweight publish/subscribe protocol ([OASIS MQTT v3.1.1](http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html), [MQTT v5.0](https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html)).
- **QoS** — Quality of Service. MQTT defines three levels: 0 (at most once), 1 (at least once), 2 (exactly once).
- **Topic** — An MQTT routing string. Brokers deliver messages to subscribers whose topic filters match the published topic.
- **Context registry** — A mapping from context URLs to compact integer identifiers, shared between sender and receiver.

---

## 2. CBOR-LD Serialization

CBOR-LD serialization encodes JSON-LD documents as CBOR binary data, with optional context compression that replaces repeated context URLs with short integer identifiers.

### 2.1 Encoding

To serialize a JSON-LD document to CBOR:

**Input:** A JSON-LD document (dict) and an optional context registry (dict mapping context URL strings to integer IDs).

**Procedure:**

1. If a context registry is provided, use it. Otherwise, use the default context registry (§2.3).
2. Recursively traverse the document. For each `@context` value:
   a. If the value is a string and appears as a key in the registry, replace it with the corresponding integer ID.
   b. If the value is an array, apply this replacement to each string element.
   c. If the value is an inline context object, preserve it without compression.
3. Encode the resulting structure as CBOR using [RFC 8949](https://www.rfc-editor.org/rfc/rfc8949) encoding rules.
4. Return the CBOR bytes.

All jsonld-ex extension keywords (`@confidence`, `@source`, `@validFrom`, etc.) are preserved as-is in the CBOR encoding. They are standard JSON-LD properties and serialize naturally to CBOR.

### 2.2 Decoding

To deserialize CBOR bytes back to a JSON-LD document:

**Input:** CBOR-encoded bytes and an optional context registry (the same registry used during encoding).

**Procedure:**

1. Decode the CBOR bytes to a Python/JSON structure.
2. Build a reverse registry (integer ID → context URL) from the provided or default registry.
3. Recursively traverse the decoded structure. For each `@context` value:
   a. If the value is an integer and appears in the reverse registry, replace it with the corresponding URL string.
   b. If the value is an array, apply this replacement to each integer element.
   c. String and object values are preserved unchanged.
4. Return the restored JSON-LD document.

### 2.3 Default Context Registry

The reference implementation provides a default registry for well-known contexts:

| Context URL | Integer ID |
|-------------|-----------|
| `http://schema.org/` | 1 |
| `https://schema.org/` | 1 |
| `https://www.w3.org/ns/activitystreams` | 2 |
| `https://w3id.org/security/v2` | 3 |
| `https://www.w3.org/2018/credentials/v1` | 4 |
| `http://www.w3.org/ns/prov#` | 5 |

Note that both `http://` and `https://` variants of schema.org map to the same ID, reflecting real-world usage where both schemes are encountered.

Processors MAY extend or replace this registry with application-specific mappings. Sender and receiver MUST agree on the registry in use — a mismatch will produce incorrect context restoration.

### 2.4 Payload Statistics

The `payload_stats` function computes four serialization sizes for a document, enabling empirical comparison:

| Metric | Description |
|--------|-------------|
| `json_bytes` | Compact JSON serialization (no whitespace) |
| `cbor_bytes` | CBOR with context compression |
| `gzip_json_bytes` | Gzip-compressed JSON |
| `gzip_cbor_bytes` | Gzip-compressed CBOR |

Two derived ratios are provided:

- **`cbor_ratio`** = `cbor_bytes / json_bytes` — measures CBOR compression alone.
- **`gzip_cbor_ratio`** = `gzip_cbor_bytes / json_bytes` — the headline number for maximum compression (CBOR + gzip vs. raw JSON).

**Example:**

```json
{
  "@context": "https://schema.org/",
  "@type": "SensorReading",
  "temperature": {
    "@value": 36.7,
    "@confidence": 0.95,
    "@unit": "celsius"
  }
}
```

Typical results for a document like this: JSON ~180 bytes, CBOR ~120 bytes (0.67 ratio), gzip CBOR ~95 bytes (0.53 ratio).

---

## 3. MQTT Payload Serialization

The MQTT payload serialization functions wrap CBOR-LD (or JSON) encoding with payload size enforcement suitable for MQTT transmission.

### 3.1 Encoding

**Input:** A JSON-LD document (dict), a compression flag (boolean, default `true`), a maximum payload size (integer, default 256,000 bytes), and an optional context registry.

**Procedure:**

1. If compression is enabled:
   a. Serialize the document to CBOR using the CBOR-LD encoding (§2.1).
   b. If the `cbor2` library is not available, raise an error indicating the dependency requirement.
2. If compression is disabled:
   a. Serialize the document to compact JSON (no whitespace) and encode as UTF-8 bytes.
3. Measure the byte length of the serialized payload.
4. If the byte length exceeds `max_payload`, raise a size error.
5. Return the encoded bytes.

The default maximum payload of 256,000 bytes (256 KB) corresponds to the MQTT v3.1.1 default maximum packet size. Processors MAY adjust this limit based on broker configuration — MQTT v5.0 brokers can advertise their maximum packet size via the CONNACK `Maximum Packet Size` property.

### 3.2 Decoding

**Input:** Raw bytes received from MQTT, an optional `@context` to reattach, a compression flag (boolean, default `true`), and an optional context registry.

**Procedure:**

1. If compression is enabled, decode using CBOR-LD decoding (§2.2).
2. If compression is disabled, decode the bytes as UTF-8 and parse as JSON.
3. If a `@context` value is provided and the decoded document does not already contain `@context`, attach the provided context.
4. Return the restored JSON-LD document.

The context reattachment in step 3 enables a common optimization: when sender and receiver agree on a shared context, the sender strips `@context` before transmission to save bytes, and the receiver reattaches it on receipt.

---

## 4. Topic Derivation

MQTT topics determine message routing. The topic derivation function generates structured, semantically meaningful topics from JSON-LD metadata.

### 4.1 Topic Pattern

```
{prefix}/{@type}/{@id_fragment}
```

| Segment | Source | Default |
|---------|--------|---------|
| `prefix` | Caller-provided | `"ld"` |
| `@type` | Document `@type` property, local name extracted | `"unknown"` |
| `@id_fragment` | Document `@id` property, last path/fragment segment | `"unknown"` |

**Examples:**

| Document | Derived Topic |
|----------|--------------|
| `{"@type": "SensorReading", "@id": "urn:sensor:imu-001"}` | `ld/SensorReading/imu-001` |
| `{"@type": "https://schema.org/Person", "@id": "https://example.org/people/alice"}` | `ld/Person/alice` |
| `{"@type": ["SensorReading", "Observation"], "@id": "urn:obs:42"}` | `ld/SensorReading/42` |
| `{}` | `ld/unknown/unknown` |

### 4.2 Local Name Extraction

When `@type` or `@id` contains a full IRI, the local name is extracted:

1. If the IRI contains `#`, return the substring after the last `#`.
2. Otherwise, if the IRI contains `/`, return the substring after the last `/`.
3. Otherwise, if the IRI contains `:` (e.g., a URN), return the substring after the last `:`.
4. Otherwise, return the full string.

When `@type` is an array, the first element is used.

### 4.3 Topic Sanitization

Per MQTT specification (v3.1.1 §4.7, v5.0 §4.7), PUBLISH topic names have the following constraints:

| Character | Constraint | Handling |
|-----------|-----------|----------|
| `#` | Wildcard, forbidden in PUBLISH topics | Replaced with `_` |
| `+` | Wildcard, forbidden in PUBLISH topics | Replaced with `_` |
| `\x00` (null) | Forbidden | Replaced with `_` |
| `$` (leading) | Reserved for broker system topics (e.g., `$SYS/`) | Stripped from start of segment |

After sanitization, if a segment is empty, it is replaced with `"unknown"`.

### 4.4 Topic Length Limit

The MQTT specification limits topic names to 65,535 bytes when UTF-8 encoded. If the generated topic exceeds this limit, the derivation function MUST raise an error. In practice, topics derived from JSON-LD metadata are far shorter than this limit.

---

## 5. QoS Derivation

The QoS derivation function maps jsonld-ex confidence metadata to MQTT Quality of Service levels, enabling automatic resource allocation based on data reliability.

### 5.1 Mapping Table

| Condition | QoS Level | MQTT Semantics |
|-----------|-----------|----------------|
| `@humanVerified = true` | 2 | Exactly once |
| `@confidence ≥ 0.9` | 2 | Exactly once |
| `0.5 ≤ @confidence < 0.9` | 1 | At least once |
| `@confidence < 0.5` | 0 | At most once |
| No confidence metadata | 1 | At least once (default) |

### 5.2 Evaluation Order

1. Check for `@humanVerified` at the document level. If `true`, return QoS 2.
2. Check for `@confidence` at the document level. If present, apply the mapping table.
3. If no document-level metadata is found, scan property values (non-`@`-prefixed keys):
   a. For each property whose value is a dict, check for `@humanVerified`. If `true`, return QoS 2.
   b. For each property whose value is a dict, check for `@confidence`. If found, apply the mapping table using the first match and stop scanning.
4. If no confidence metadata is found anywhere, return QoS 1 (default).

### 5.3 Rationale

The mapping reflects a practical heuristic for IoT data pipelines:

- **QoS 0 (at most once)** — suitable for high-frequency, low-confidence telemetry (e.g., raw accelerometer samples at 100 Hz). Losing individual messages is acceptable; the next sample arrives shortly.
- **QoS 1 (at least once)** — the safe default. Messages may be duplicated but not lost. Suitable for most sensor readings and ML inference results.
- **QoS 2 (exactly once)** — the most expensive QoS level, requiring a four-step handshake. Reserved for high-confidence or human-verified data where duplication would cause problems (e.g., confirmed clinical alerts, verified financial events).

### 5.4 Detailed Response

The `derive_mqtt_qos_detailed` function returns additional diagnostic information alongside the QoS level:

| Field | Type | Description |
|-------|------|-------------|
| `qos` | Integer (0, 1, 2) | The derived QoS level |
| `reasoning` | String | Human-readable explanation of the derivation |
| `confidence_used` | Float or null | The confidence value that drove the decision |
| `human_verified` | Boolean | Whether `@humanVerified` was the deciding factor |

**Example response:**

```json
{
  "qos": 2,
  "reasoning": "@confidence=0.95 ≥ 0.9 (document-level) → QoS 2 (exactly once)",
  "confidence_used": 0.95,
  "human_verified": false
}
```

---

## 6. MQTT 5.0 PUBLISH Properties

MQTT 5.0 ([OASIS Standard, §3.3.2.3](https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html)) introduced structured PUBLISH packet properties. The jsonld-ex transport module derives these properties from JSON-LD document metadata.

### 6.1 Derived Properties

| MQTT 5.0 Property | MQTT Spec Section | Source | Value |
|-------------------|-------------------|--------|-------|
| Payload Format Indicator | §3.3.2.3.2 | Compression flag | `0` (unspecified bytes) for CBOR, `1` (UTF-8) for JSON |
| Content Type | §3.3.2.3.9 | Compression flag | `"application/cbor"` for CBOR, `"application/ld+json"` for JSON |
| Message Expiry Interval | §3.3.2.3.3 | `@validUntil` | Seconds remaining until expiry (see §6.2) |
| User Properties | §3.3.2.3.7 | Document metadata | Key-value pairs (see §6.3) |

### 6.2 Message Expiry Interval

The Message Expiry Interval is derived from the `@validUntil` temporal annotation:

**Procedure:**

1. Check for `@validUntil` at the document level.
2. If not found, scan property values (non-`@`-prefixed keys) for the first dict containing `@validUntil`.
3. If no `@validUntil` is found, omit the Message Expiry Interval (the message does not expire).
4. Parse the `@validUntil` value as an ISO 8601 datetime.
5. Compute the number of seconds remaining: `ceil(validUntil - now)`.
6. If the result is zero or negative (the assertion has already expired), omit the Message Expiry Interval.
7. Clamp the result to the MQTT uint32 maximum (4,294,967,295 seconds, approximately 136 years).
8. Return the clamped integer.

This mapping means that MQTT brokers will automatically discard messages whose temporal assertions have expired, without requiring application-level expiry logic.

### 6.3 User Properties

User Properties are key-value string pairs attached to MQTT 5.0 PUBLISH packets. The following JSON-LD metadata fields are extracted when present:

| User Property Key | Source | Description |
|-------------------|--------|-------------|
| `jsonld_type` | `@type` | The document type (first element if array) |
| `jsonld_confidence` | `@confidence` | Confidence score as string |
| `jsonld_source` | `@source` | Source/model IRI |
| `jsonld_id` | `@id` | Document identifier |

User Properties enable MQTT 5.0 subscribers to filter or route messages based on JSON-LD metadata without deserializing the payload. For example, a subscriber could filter for messages where `jsonld_type = "CriticalAlert"` at the broker level.

### 6.4 Example

Given the document:

```json
{
  "@type": "SensorReading",
  "@id": "urn:sensor:imu-001",
  "@confidence": 0.95,
  "@source": "https://model.example.org/imu-classifier-v2",
  "temperature": {
    "@value": 36.7,
    "@validUntil": "2026-02-12T18:00:00Z"
  }
}
```

The derived MQTT 5.0 properties (assuming CBOR compression):

```json
{
  "payload_format_indicator": 0,
  "content_type": "application/cbor",
  "message_expiry_interval": 3600,
  "user_properties": [
    ["jsonld_type", "SensorReading"],
    ["jsonld_confidence", "0.95"],
    ["jsonld_source", "https://model.example.org/imu-classifier-v2"],
    ["jsonld_id", "urn:sensor:imu-001"]
  ]
}
```

---

## 7. Backward Compatibility

### 7.1 Non-Extended Processors

The transport extensions are entirely processor-side functionality. They do not introduce new keywords into JSON-LD documents. A standard JSON-LD 1.1 processor can:

- Process documents that have been serialized and deserialized via the CBOR-LD path, because the restored document is valid JSON-LD.
- Ignore topic, QoS, and MQTT 5.0 property derivation, which are transport-layer concerns outside JSON-LD processing.

### 7.2 MQTT Version Compatibility

The transport extensions support both MQTT v3.1.1 and MQTT v5.0:

| Feature | MQTT v3.1.1 | MQTT v5.0 |
|---------|-------------|-----------|
| Payload serialization | ✓ (CBOR or JSON) | ✓ (CBOR or JSON) |
| Topic derivation | ✓ | ✓ |
| QoS derivation | ✓ | ✓ |
| PUBLISH properties | Not available | ✓ (§6) |
| Message expiry | Not available | ✓ (§6.2) |

When operating with an MQTT v3.1.1 broker, the MQTT 5.0 properties are simply not used. All other features remain functional.

### 7.3 Optional Dependencies

The CBOR-LD serialization requires the `cbor2` Python package. When `cbor2` is not installed:

- Compressed MQTT payloads (`compress=True`) raise an `ImportError` with installation instructions.
- JSON-mode payloads (`compress=False`) work without any additional dependencies.
- All other MQTT functions (topic derivation, QoS derivation, MQTT 5.0 properties) work without `cbor2`.

The `cbor2` dependency is available via the `iot` optional extra: `pip install jsonld-ex[iot]`.

---

## 8. Relationship to Existing Standards

### 8.1 MQTT v3.1.1 and v5.0

The MQTT transport extensions operate within the constraints of the MQTT specification:

| Constraint | MQTT Spec Reference | jsonld-ex Compliance |
|-----------|-------------------|---------------------|
| Topic max length: 65,535 bytes | v3.1.1 §4.7, v5.0 §4.7 | Enforced in topic derivation (§4.4) |
| No wildcards (`#`, `+`) in PUBLISH topics | v3.1.1 §4.7, v5.0 §4.7 | Sanitized (§4.3) |
| No null character in topics | v3.1.1 §4.7, v5.0 §4.7 | Sanitized (§4.3) |
| `$`-prefixed topics reserved for broker | v3.1.1 §4.7, v5.0 §4.7 | Leading `$` stripped (§4.3) |
| QoS levels 0, 1, 2 | v3.1.1 §4.3, v5.0 §4.3 | Mapped from confidence (§5) |
| Payload Format Indicator values 0, 1 | v5.0 §3.3.2.3.2 | Set from compression mode (§6.1) |
| Message Expiry Interval: uint32 seconds | v5.0 §3.3.2.3.3 | Clamped to uint32 max (§6.2) |

### 8.2 CBOR (RFC 8949)

The CBOR-LD serialization uses standard CBOR encoding as defined in [RFC 8949](https://www.rfc-editor.org/rfc/rfc8949). The context compression scheme (replacing URL strings with integer IDs) is applied before CBOR encoding — the CBOR layer receives a standard JSON-compatible data structure and encodes it without modification. This means any compliant CBOR decoder can read the bytes, though context restoration requires knowledge of the registry.

### 8.3 CBOR-LD Draft Specification

The W3C [CBOR-LD](https://json-ld.github.io/cbor-ld-spec/) draft specification defines a more comprehensive compression scheme that assigns integer codes to JSON-LD keywords and vocabulary terms. The jsonld-ex CBOR-LD module implements a simplified subset focused on context URL compression. The registry-based approach is compatible with the CBOR-LD draft's context compression mechanism, but does not implement full term-level compression.

### 8.4 JSON-LD 1.1 Content Type

When using JSON serialization (not CBOR), the Content Type is `application/ld+json` as defined in the [JSON-LD 1.1 specification](https://www.w3.org/TR/json-ld11/) §8 (IANA Considerations). When using CBOR serialization, the Content Type is `application/cbor` as registered in the IANA media type registry.

---

## 9. Reference Implementation

The reference implementation spans two modules in the [jsonld-ex Python package](https://pypi.org/project/jsonld-ex/).

### CBOR-LD Module (`jsonld_ex.cbor_ld`)

| Function | Spec Section |
|----------|-------------|
| `to_cbor(doc, context_registry)` | §2.1 |
| `from_cbor(data, context_registry)` | §2.2 |
| `payload_stats(doc, context_registry)` | §2.4 |
| `DEFAULT_CONTEXT_REGISTRY` | §2.3 |

### MQTT Module (`jsonld_ex.mqtt`)

| Function | Spec Section |
|----------|-------------|
| `to_mqtt_payload(doc, compress, max_payload, context_registry)` | §3.1 |
| `from_mqtt_payload(payload, context, compressed, context_registry)` | §3.2 |
| `derive_mqtt_topic(doc, prefix)` | §4 |
| `derive_mqtt_qos(doc)` | §5 |
| `derive_mqtt_qos_detailed(doc)` | §5.4 |
| `derive_mqtt5_properties(doc, compress)` | §6 |

---

### CoAP Module (`jsonld_ex.coap`)

| Function | Spec Section |
|----------|-------------|
| `to_coap_payload(doc, compress, max_payload, context_registry)` | §10.1 |
| `from_coap_payload(payload, compressed, context, context_registry)` | §10.1 |
| `derive_coap_options(doc, compress, prefix, context_registry)` | §10.2 |
| `derive_coap_uri_path(doc, prefix)` | §10.2 |
| `derive_coap_message_type(doc)` | §10.3 |

### HTTP Module (`jsonld_ex.http_headers`)

| Function | Spec Section |
|----------|-------------|
| `derive_response_headers(doc, compress)` | §11.1 |
| `derive_request_headers(compress, etag)` | §11.5 |
| `derive_etag(doc)` | §11.2 |
| `derive_cache_control(doc)` | §11.3 |
| `derive_link_header(doc)` | §11.4 |
| `derive_content_type(compress)` | §11.1 |

---

## 10. CoAP Transport

The Constrained Application Protocol (RFC 7252) is a RESTful protocol for constrained IoT devices. The CoAP transport module derives CoAP options from JSON-LD metadata, enabling semantic-aware caching, conditional requests, and resource observation on constrained networks.

### 10.1 Payload Serialization

CoAP payloads use the same CBOR-LD (§2) or JSON serialization as MQTT (§3). The default maximum payload is 1024 bytes (single UDP datagram for constrained networks). Block-wise transfer (RFC 7959) is recommended for larger payloads.

### 10.2 Option Derivation

CoAP options are derived from JSON-LD metadata as follows:

| CoAP Option | RFC 7252 Section | jsonld-ex Source | Mapping |
|-------------|-----------------|-----------------|---------|
| Content-Format (12) | §5.10.3 | Compression flag | `60` (CBOR) or `11100` (JSON-LD, placeholder — see §10.6) |
| ETag (4) | §5.10.6 | `@integrity` | SHA-256 of integrity string, truncated to 8 bytes |
| Max-Age (14) | §5.10.5 | `@validUntil` | Seconds remaining until expiry |
| Uri-Path (11) | §5.10.1 | `@type`, `@id` | Path segments: `[prefix, type_local, id_fragment]` |
| Size1 (60) | §5.10.9 | Payload | Byte length of serialized payload |

### 10.3 Message Type Derivation

CoAP message types (CON vs NON) are derived from confidence metadata:

| Condition | Message Type | Semantics |
|-----------|-------------|-----------|
| `@humanVerified = true` | CON | Confirmable (requires ACK) |
| `@confidence >= 0.5` | CON | Reliable delivery |
| `@confidence < 0.5` | NON | Non-confirmable (fire-and-forget) |
| No confidence metadata | CON | Safe default |

Unlike MQTT's three QoS levels, CoAP has only CON vs NON. The mapping is conservative: only clearly low-confidence telemetry gets NON.

### 10.4 Block-Wise Transfer

When the serialized payload exceeds 1024 bytes (e.g., documents with large vector embeddings), the module recommends block-wise transfer (RFC 7959) with `block_szx = 6` (1024-byte blocks).

### 10.5 Observe

Resources with `@validUntil` annotations are flagged as observable (RFC 7641), indicating temporal semantics suitable for the CoAP Observe pattern.

### 10.6 Content-Format Registration

As of 2026, `application/ld+json` does not have an official IANA CoAP Content-Format assignment. The value `11100` is used as an illustrative placeholder from the unassigned range 11061–11541, avoiding collision with existing assignments (notably 11050 = `application/json` with `deflate` encoding; 11542–11544 = OMA LwM2M formats). Implementers MUST NOT use this value in production without registering an official Content-Format ID with IANA per RFC 7252 §12.3 and RFC 9876. Until registration, using Content-Format `50` (`application/json`) is the safest interoperable choice.

---

## 11. HTTP Header Derivation

The HTTP header derivation module bridges JSON-LD metadata and standard HTTP caching, content negotiation, and conditional request mechanisms (RFC 9110).

### 11.1 Response Headers

HTTP response headers are derived from document metadata:

| HTTP Header | RFC Section | jsonld-ex Source | Value |
|------------|-------------|-----------------|-------|
| Content-Type | RFC 9110 §8.3 | Compression flag | `application/ld+json` or `application/cbor` |
| ETag | RFC 9110 §8.8.3 | `@integrity` | Quoted SHA-256 hash (first 32 hex chars) |
| Cache-Control | RFC 9110 §5.2 | `@validUntil` | `max-age=N` (seconds remaining) or `no-cache` (expired) |
| Link | RFC 8288 | `@context` | `<url>; rel="http://www.w3.org/ns/json-ld#context"` |
| X-JsonLD-Confidence | Custom | `@confidence` | Confidence score as string |
| X-JsonLD-Source | Custom | `@source` | Source/model IRI |
| X-JsonLD-Type | Custom | `@type` | Document type (first element if array) |

### 11.2 ETag Derivation

The ETag is derived from `@integrity` by hashing the integrity string with SHA-256 and taking the first 32 hex characters. The result is returned as a quoted string per RFC 9110 §8.8.3, enabling conditional requests (`If-None-Match` → `304 Not Modified`).

### 11.3 Cache-Control Derivation

Cache-Control is derived from `@validUntil`:

| Condition | Cache-Control Value |
|-----------|-------------------|
| `@validUntil` in future | `max-age=N` (seconds remaining) |
| `@validUntil` in past | `no-cache` |
| No `@validUntil` | Not included |

### 11.4 Link Header for Context Discovery

Per W3C JSON-LD 1.1 §4.1, a JSON document can be interpreted as JSON-LD if a Link header provides the context URL. The module generates Link headers from `@context`:

- String context → single Link entry
- Array context → one Link entry per URL (inline context objects excluded)
- No context → no Link header

### 11.5 Request Headers

Request headers support content negotiation and conditional requests:

| HTTP Header | Source | Value |
|------------|--------|-------|
| Accept | Compression preference | `application/ld+json` or `application/cbor` |
| If-None-Match | Previously received ETag | Quoted ETag string |

### 11.6 Custom Headers

The `X-JsonLD-*` headers enable proxy and CDN routing decisions based on JSON-LD metadata without deserializing the payload. These use the `X-` prefix per convention. While RFC 6648 deprecated creating new `X-` headers, these are explicitly jsonld-ex-specific and unlikely to become standard HTTP headers.

---

### AMQP Module (`jsonld_ex.amqp`)

| Function | Spec Section |
|----------|-------------|
| `to_amqp_payload(doc, compress, context_registry)` | §12.1 |
| `from_amqp_payload(payload, compressed, context, context_registry)` | §12.1 |
| `derive_amqp_properties(doc, compress, prefix)` | §12.1 |
| `derive_routing_key(doc, prefix)` | §12.1 |
| `derive_amqp_priority(doc)` | §12.1 |
| `derive_amqp_headers(doc)` | §12.1 |

### Kafka Module (`jsonld_ex.kafka`)

| Function | Spec Section |
|----------|-------------|
| `to_kafka_value(doc, compress, context_registry)` | §13.1 |
| `from_kafka_value(value, compressed, context, context_registry)` | §13.1 |
| `derive_kafka_record(doc, compress, prefix, context_registry)` | §13.1 |
| `derive_kafka_topic(doc, prefix)` | §13.1 |
| `derive_kafka_key(doc)` | §13.1 |
| `derive_kafka_headers(doc, compress)` | §13.1 |
| `derive_kafka_timestamp(doc)` | §13.1 |

### WebSocket Module (`jsonld_ex.websocket`)

| Function | Spec Section |
|----------|-------------|
| `to_ws_message(doc, compress, context_registry)` | §14.1 |
| `from_ws_message(message, compressed, context, context_registry)` | §14.1 |
| `derive_ws_subprotocols(compress)` | §14.2 |
| `derive_ws_metadata(doc, compress)` | §14.3 |

### gRPC Module (`jsonld_ex.grpc`)

| Function | Spec Section |
|----------|-------------|
| `derive_grpc_metadata(doc)` | §15.1 |
| `suggest_proto_schema(doc)` | §15.2 |
| `to_grpc_json(doc)` | §15.3 |
| `from_grpc_json(payload, context)` | §15.3 |

### LwM2M Module (`jsonld_ex.lwm2m`)

| Function | Spec Section |
|----------|-------------|
| `derive_lwm2m_objects(doc)` | §16.1 |
| `extract_lwm2m_resources(doc, resource_map)` | §16.1 |
| `derive_lwm2m_registration(doc, binding)` | §16.2 |
| `derive_lwm2m_links(doc)` | §16.3 |

---

## 12. AMQP Transport

The AMQP transport module derives AMQP 0-9-1 message properties from JSON-LD metadata for enterprise messaging (RabbitMQ, Azure Service Bus).

### 12.1 Property Derivation

| AMQP Property | jsonld-ex Source | Value |
|--------------|-----------------|-------|
| Content-Type | Compression flag | `application/cbor` or `application/ld+json` |
| Routing Key | `@type`, `@id` | `prefix.type_local.id_fragment` (dot-separated for topic exchanges) |
| Priority | `@confidence` | Linear map: `round(confidence * 9)` → 0-9 |
| Delivery Mode | `@confidence`, `@humanVerified` | 2 (persistent) for ≥0.5 or verified; 1 (transient) for <0.5 |
| Expiration | `@validUntil` | Milliseconds remaining (string, AMQP 0-9-1 convention) |
| Message ID | `@id` | Document identifier |
| Timestamp | Current time | Epoch seconds |
| Headers | Metadata | `x-jsonld-type`, `x-jsonld-confidence`, `x-jsonld-source`, `x-jsonld-id` |

---

## 13. Kafka Transport

The Kafka transport module derives Apache Kafka producer record fields from JSON-LD metadata for ML data pipelines and event streaming.

### 13.1 Record Derivation

| Kafka Field | jsonld-ex Source | Value |
|------------|-----------------|-------|
| Topic | `@type` | `prefix.type_local` (type-per-topic pattern) |
| Key | `@id` | UTF-8 bytes (determines partition assignment) |
| Headers | Metadata | List of `(str, bytes)`: `content-type`, `x-jsonld-type`, `x-jsonld-confidence`, `x-jsonld-source`, `x-jsonld-id` |
| Timestamp | `@extractedAt` | Epoch milliseconds (aligns log timestamp with extraction time) |
| Value | Document | CBOR or JSON bytes |

### 13.2 Design Rationale

Unlike MQTT and CoAP (which include `@id` in the routing path), Kafka uses topics for broad categories and keys for entity-level ordering. Including `@id` in the topic would cause excessive topic proliferation. The `@extractedAt` timestamp enables accurate time-windowed processing even when ingestion is delayed.

---

## 14. WebSocket Transport

The WebSocket transport module derives framing metadata and subprotocol negotiation from JSON-LD annotations for real-time streaming.

### 14.1 Message Serialization

| Mode | Python Type | Frame Type | Content |
|------|-----------|------------|---------|
| JSON | `str` | Text (opcode 0x1) | Compact JSON |
| CBOR | `bytes` | Binary (opcode 0x2) | CBOR-LD |

### 14.2 Subprotocol Negotiation

Subprotocol identifiers for `Sec-WebSocket-Protocol` handshake:

| Subprotocol | Description |
|------------|-------------|
| `jsonld-ex.cbor` | CBOR binary frames |
| `jsonld-ex.jsonld` | JSON-LD text frames |

### 14.3 Per-Message Metadata

WebSocket (RFC 6455) has no native per-message header mechanism. The metadata dict is provided for application-level framing protocols:

| Field | Source | Type |
|-------|--------|------|
| `opcode` | Compression flag | int (0x1 or 0x2) |
| `content_type` | Compression flag | str |
| `jsonld_type` | `@type` | str |
| `jsonld_id` | `@id` | str |
| `jsonld_confidence` | `@confidence` | float |
| `jsonld_source` | `@source` | str |
| `ttl_seconds` | `@validUntil` | int |

---

## 15. gRPC Transport

The gRPC transport module bridges JSON-LD's property-based model and gRPC's compiled Protobuf schema model via metadata derivation and proto schema suggestion.

### 15.1 Metadata Derivation

gRPC metadata keys must be lowercase ASCII (gRPC specification):

| Metadata Key | Source |
|-------------|--------|
| `x-jsonld-content-type` | Always `application/ld+json` |
| `x-jsonld-type` | `@type` |
| `x-jsonld-confidence` | `@confidence` |
| `x-jsonld-source` | `@source` |
| `x-jsonld-id` | `@id` |

### 15.2 Proto Schema Suggestion

The `suggest_proto_schema()` function generates a `.proto` file skeleton from a JSON-LD document's structure. This is a *heuristic suggestion* — it infers types from a single document instance.

Limitations (documented honestly): cannot discover optional fields absent from the sample, cannot infer enum types or oneof variants, nested objects become sub-messages only one level deep.

### 15.3 JSON Transcoding

`to_grpc_json` / `from_grpc_json` provide compact JSON serialization for gRPC JSON transcoding (grpc-gateway, Envoy).

---

## 16. LwM2M Transport

The LwM2M (Lightweight Machine to Machine) transport module maps JSON-LD documents to the OMA LwM2M object/resource model for constrained IoT device management.

### 16.1 IPSO Smart Object Mapping

| JSON-LD `@type` | IPSO Object ID | Primary Resource |
|-----------------|---------------|-----------------|
| TemperatureSensor | 3303 | 5700 (Sensor Value) |
| HumiditySensor | 3304 | 5700 |
| Barometer | 3315 | 5700 |
| Accelerometer | 3313 | 5702/5703/5704 (X/Y/Z) |
| Illuminance | 3301 | 5700 |
| GenericSensor | 3300 | 5700 |
| DigitalOutput | 3201 | 5550 |
| AnalogInput | 3202 | 5600 |

Unknown `@type` values are assigned custom object IDs in the 26241+ range (OMA reserved for custom objects).

### 16.2 Registration Parameters

| Parameter | Source | Default |
|-----------|--------|---------|
| Endpoint | `@id` | `"unknown"` |
| Lifetime | `@validUntil` | 86400 s (24 hours) |
| Binding | Caller-specified | `"U"` (UDP) |
| Objects | Derived from document | IPSO or custom objects |

### 16.3 CoRE Link Format

`derive_lwm2m_links()` generates RFC 6690 link-format strings for LwM2M registration and `.well-known/core` discovery (e.g. `</3303/0>,</3304/0>`).

---

## References

- Bormann, C., and Hoffman, P. (2020). RFC 8949: Concise Binary Object Representation (CBOR). IETF. https://www.rfc-editor.org/rfc/rfc8949
- OASIS. (2014). MQTT Version 3.1.1. OASIS Standard. http://docs.oasis-open.org/mqtt/mqtt/v3.1.1/os/mqtt-v3.1.1-os.html
- OASIS. (2019). MQTT Version 5.0. OASIS Standard. https://docs.oasis-open.org/mqtt/mqtt/v5.0/os/mqtt-v5.0-os.html
- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- W3C. CBOR-LD. Draft Specification. https://json-ld.github.io/cbor-ld-spec/
- Bradner, S. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. https://www.rfc-editor.org/rfc/rfc2119
- Shelby, Z., Hartke, K., and Bormann, C. (2014). RFC 7252: The Constrained Application Protocol (CoAP). IETF. https://www.rfc-editor.org/rfc/rfc7252
- Hartke, K. (2015). RFC 7641: Observing Resources in the Constrained Application Protocol (CoAP). IETF. https://www.rfc-editor.org/rfc/rfc7641
- Bormann, C., and Shelby, Z. (2016). RFC 7959: Block-Wise Transfers in the Constrained Application Protocol (CoAP). IETF. https://www.rfc-editor.org/rfc/rfc7959
- Fielding, R., et al. (2022). RFC 9110: HTTP Semantics. IETF. https://www.rfc-editor.org/rfc/rfc9110
- Nottingham, M. (2017). RFC 8288: Web Linking. IETF. https://www.rfc-editor.org/rfc/rfc8288
