# Temporal Extensions

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/temporal](https://jsonld-ex.github.io/ns/temporal)

---

## 1. Introduction

This document specifies the semantics of temporal extensions for jsonld-ex. Three annotation keywords — `@validFrom`, `@validUntil`, and `@asOf` — enable time-aware assertions in JSON-LD documents, supporting knowledge graph versioning, point-in-time queries, and temporal differencing.

The formal property definitions (IRI, type, domain, range) are in the [Vocabulary](vocabulary.md) specification, §12. This document defines the processing semantics: how timestamps are parsed, how temporal validity is evaluated, and how graphs are queried and compared across time.

### 1.1 Motivation

Many real-world assertions are time-bounded. An employee's job title changes, a sensor calibration expires, a model prediction is valid only for a monitoring window. Without temporal metadata, knowledge graphs can only represent "current truth" — there is no standard way to express when a statement was true, or to retrieve the state of the graph at an earlier point in time.

The jsonld-ex temporal extensions address this by attaching validity intervals and observation timestamps directly to value objects using standard JSON-LD annotation mechanisms.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.3 Terminology

- **Value object** — a JSON-LD node of the form `{"@value": ..., ...}`.
- **Temporally bounded value** — a value object containing at least one of `@validFrom`, `@validUntil`, or `@asOf`.
- **Unbounded value** — a value (plain or annotated) with no temporal qualifiers. Unbounded values are treated as always valid.
- **Validity interval** — the time range `[@validFrom, @validUntil]` during which an assertion holds. The interval is closed on both endpoints.

---

## 2. Timestamp Format

### 2.1 Supported Formats

Temporal keywords accept strings in a subset of [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Conforming processors MUST support the following formats:

| Format | Example | Notes |
|--------|---------|-------|
| Date only | `2025-01-15` | Midnight UTC implied |
| Date-time with Z | `2025-01-15T10:30:00Z` | UTC |
| Date-time with offset | `2025-01-15T10:30:00+00:00` | Explicit offset |
| Fractional seconds with Z | `2025-01-15T10:30:00.123Z` | Millisecond precision |
| Fractional seconds with offset | `2025-01-15T10:30:00.123+05:30` | Millisecond precision, offset |
| Date-time without offset | `2025-01-15T10:30:00` | Local time (no timezone) |

### 2.2 Parsing Rules

- Processors MUST accept the trailing `Z` suffix as equivalent to the `+00:00` offset.
- Processors SHOULD accept at least millisecond precision in fractional seconds.
- Processors MUST reject values that are not strings. If a temporal keyword contains a non-string value, the processor MUST raise a `TypeError`.
- Processors MUST reject strings that do not match any supported format with a `ValueError`.

---

## 3. Annotation Semantics

### 3.1 Adding Temporal Qualifiers

Temporal qualifiers attach to a value by wrapping it in a value object (if not already wrapped) and inserting the temporal keywords.

**Plain value → temporally annotated value:**

```json
"Senior Engineer"
```

becomes:

```json
{
  "@value": "Senior Engineer",
  "@validFrom": "2024-01-01",
  "@validUntil": "2025-12-31"
}
```

**Already-annotated value → composed annotation:**

```json
{
  "@value": "Kyphotic posture detected",
  "@confidence": 0.87,
  "@source": "model:posture-classifier-v2"
}
```

becomes:

```json
{
  "@value": "Kyphotic posture detected",
  "@confidence": 0.87,
  "@source": "model:posture-classifier-v2",
  "@validFrom": "2026-01-15T10:00:00Z",
  "@validUntil": "2026-01-15T10:05:00Z",
  "@asOf": "2026-01-15T10:02:30Z"
}
```

### 3.2 Requirements

- At least one temporal qualifier (`@validFrom`, `@validUntil`, or `@asOf`) MUST be provided when adding temporal annotations. Providing none is an error.
- All temporal keywords are optional individually. A value MAY have only `@validFrom`, only `@validUntil`, only `@asOf`, or any combination.

### 3.3 Keyword Semantics

| Keyword | Meaning |
|---------|---------|
| `@validFrom` | The assertion becomes true at this time. Before this time, the assertion is not yet in effect. |
| `@validUntil` | The assertion ceases to be true after this time. After this time, the assertion has expired. |
| `@asOf` | The point in time at which the assertion was known or observed to hold. |

**Distinguishing `@asOf` from `@extractedAt`:** The `@extractedAt` annotation (defined in the [Vocabulary](vocabulary.md), §3.3) records when the value was *produced* (e.g., when an NLP model ran). The `@asOf` annotation records the *observation time* — the moment in the real world to which the assertion refers. These may differ: a model may run at 10:30 AM to classify data that was observed at 10:00 AM.

---

## 4. Validity Rules

### 4.1 Interval Constraint

When both `@validFrom` and `@validUntil` are present on the same value object, processors MUST validate that:

$$
\text{@validFrom} \leq \text{@validUntil}
$$

If `@validFrom` is strictly after `@validUntil`, the processor MUST raise a `ValueError`. Equal values are permitted (representing an instantaneous assertion).

### 4.2 Open-Ended Intervals

- If only `@validFrom` is present: the assertion is valid from that time *onward* with no defined expiration (open-ended future).
- If only `@validUntil` is present: the assertion is valid from the beginning of time *until* the specified moment (open-ended past).
- If neither is present: the value is **unbounded** — valid at all times.

### 4.3 Interval Boundaries

The validity interval is **closed on both endpoints**. A value with `@validFrom = t₁` and `@validUntil = t₂` is valid at times *t* where:

$$
t_1 \leq t \leq t_2
$$

---

## 5. Point-in-Time Query

### 5.1 Overview

The point-in-time query operation retrieves the state of a graph at a specific timestamp. For each node, only properties whose temporal bounds include the query timestamp are retained.

### 5.2 Algorithm: `query_at_time`

**Input:**
- `graph` — a list of JSON-LD nodes (typically from `doc["@graph"]`)
- `timestamp` — an ISO 8601 string specifying the query time
- `property_name` (optional) — if provided, only filter this property; pass all others unchanged

**Output:**
- A filtered list of nodes. Nodes with no matching data properties are omitted entirely.

**Procedure:**

For each node in `graph`:
1. Identity keys (`@id`, `@type`, `@context`) always pass through.
2. If `property_name` is specified and the current key does not match it, the property passes through unchanged.
3. For each candidate property value:
   - If the value is a **list**, retain only those list items that are valid at the query timestamp. If no items remain, the property is dropped.
   - If the value is a **single value**, retain it if valid, drop it otherwise.
4. If a node has no remaining data properties (only identity keys), it is omitted from the result.

### 5.3 Validity Check

A value is **valid at time *t*** if and only if:

1. The value has no temporal bounds (no `@validFrom` and no `@validUntil`), OR
2. The value is not a dict (plain literal — always valid), OR
3. Both of the following hold:
   - If `@validFrom` is present: *t* ≥ `@validFrom`
   - If `@validUntil` is present: *t* ≤ `@validUntil`

Values without temporal metadata are treated as **always valid** — they are never filtered out by temporal queries. This ensures backward compatibility: adding temporal support to a graph does not invalidate existing unbounded assertions.

### 5.4 Example

Given a graph representing employment history:

```json
{
  "@graph": [
    {
      "@id": "ex:alice",
      "@type": "Person",
      "jobTitle": [
        {
          "@value": "Junior Engineer",
          "@validFrom": "2020-01-01",
          "@validUntil": "2022-12-31"
        },
        {
          "@value": "Senior Engineer",
          "@validFrom": "2023-01-01",
          "@validUntil": "2025-12-31"
        },
        {
          "@value": "Staff Engineer",
          "@validFrom": "2026-01-01"
        }
      ],
      "name": "Alice Smith"
    }
  ]
}
```

Querying at `2024-06-15`:

```json
[
  {
    "@id": "ex:alice",
    "@type": "Person",
    "jobTitle": {
      "@value": "Senior Engineer",
      "@validFrom": "2023-01-01",
      "@validUntil": "2025-12-31"
    },
    "name": "Alice Smith"
  }
]
```

Note that `name` (unbounded) is retained, only the matching `jobTitle` entry survives, and it is unwrapped from the list since only one item remains.

---

## 6. Temporal Diff

### 6.1 Overview

The temporal diff operation compares the state of a graph at two points in time, identifying what was added, removed, modified, or unchanged. This supports change detection, audit logging, and incremental updates in knowledge graph systems.

### 6.2 Algorithm: `temporal_diff`

**Input:**
- `graph` — a list of JSON-LD nodes with temporal annotations
- `t1` — earlier ISO 8601 timestamp
- `t2` — later ISO 8601 timestamp

**Output:**
- A `TemporalDiffResult` containing four lists: `added`, `removed`, `modified`, `unchanged`.

**Procedure:**

1. Compute snapshot at *t₁*: `snap1 = query_at_time(graph, t1)`, indexed by `@id`.
2. Compute snapshot at *t₂*: `snap2 = query_at_time(graph, t2)`, indexed by `@id`.
3. For each `@id` present in either snapshot:
   - If the node exists only in *snap2*: classify as **added** (entire node).
   - If the node exists only in *snap1*: classify as **removed** (entire node).
   - If the node exists in both: compare data properties (excluding `@id`, `@type`, `@context`):
     - Property present only in *snap2*: **added** (property level).
     - Property present only in *snap1*: **removed** (property level).
     - Property present in both with different bare values: **modified**.
     - Property present in both with identical bare values: **unchanged**.

### 6.3 Bare Value Comparison

When comparing property values across snapshots, the comparison strips annotation wrappers. The **bare value** of a value object `{"@value": v, ...}` is `v`. The bare value of a plain value is itself.

This means that changes in annotations (e.g., `@confidence` increasing from 0.80 to 0.95) without a change in the underlying `@value` are classified as **unchanged**. The diff focuses on factual changes, not metadata changes.

### 6.4 Result Structure

The `TemporalDiffResult` contains:

| Field | Type | Description |
|-------|------|-------------|
| `added` | list | Nodes or properties that exist at *t₂* but not at *t₁* |
| `removed` | list | Nodes or properties that exist at *t₁* but not at *t₂* |
| `modified` | list | Properties whose bare value changed between *t₁* and *t₂* |
| `unchanged` | list | Properties whose bare value is identical at both times |

Each entry in the lists contains:
- `@id` — the node identifier
- For property-level changes: `property` (the property key) and `value` or `value_at_t1`/`value_at_t2`

### 6.5 Example

Using the employment history from §5.4, diffing between `2022-06-15` and `2024-06-15`:

```json
{
  "added": [],
  "removed": [],
  "modified": [
    {
      "@id": "ex:alice",
      "property": "jobTitle",
      "value_at_t1": {
        "@value": "Junior Engineer",
        "@validFrom": "2020-01-01",
        "@validUntil": "2022-12-31"
      },
      "value_at_t2": {
        "@value": "Senior Engineer",
        "@validFrom": "2023-01-01",
        "@validUntil": "2025-12-31"
      }
    }
  ],
  "unchanged": [
    {
      "@id": "ex:alice",
      "property": "name",
      "value": "Alice Smith"
    }
  ]
}
```

---

## 7. Composition with Other Annotations

### 7.1 General Principle

Temporal qualifiers compose freely with all other jsonld-ex annotations on the same value object. A single value may carry `@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`, and temporal qualifiers simultaneously:

```json
{
  "@value": "Forward head posture",
  "@confidence": 0.92,
  "@source": "model:posture-v3",
  "@extractedAt": "2026-01-15T10:30:00Z",
  "@validFrom": "2026-01-15T10:00:00Z",
  "@validUntil": "2026-01-15T10:05:00Z",
  "@asOf": "2026-01-15T10:02:30Z"
}
```

There is no ordering requirement among annotation keywords within the value object.

### 7.2 Interaction with Confidence Decay

The [Confidence Algebra](confidence-algebra.md) specification defines temporal decay operators that model the degradation of belief over time (§10). Temporal extensions provide the timestamps needed to compute decay:

- `@extractedAt` or `@asOf` supplies the **reference time** (when the opinion was formed).
- The **query time** (e.g., from `query_at_time`) supplies the current time.
- The elapsed interval drives the decay function (exponential, linear, or step).

Temporal validity and confidence decay are complementary: `@validUntil` defines a hard expiration boundary, while decay models the gradual degradation of belief within the validity window.

### 7.3 Interaction with Invalidation

If a value carries both temporal qualifiers and `@invalidatedAt` (see [Vocabulary](vocabulary.md), §11.1), the invalidation takes precedence. A value that has been explicitly invalidated SHOULD be treated as no longer valid regardless of its `@validUntil` bound.

---

## 8. Backward Compatibility

### 8.1 Non-Extended Processors

A standard JSON-LD 1.1 processor treats `@validFrom`, `@validUntil`, and `@asOf` as ordinary properties mapped by the jsonld-ex context. The annotations are preserved through expansion, compaction, and flattening without semantic interpretation. No errors are raised and no data is lost.

### 8.2 Round-Trip Safety

Temporal annotations survive round-trip processing through standard JSON-LD operations (expansion → compaction, serialization → deserialization). The temporal metadata is preserved as regular property values on the value object.

---

## 9. Relationship to Existing Standards

### 9.1 OWL Time

The [OWL-Time ontology](https://www.w3.org/TR/owl-time/) provides a comprehensive model for temporal entities including instants, intervals, durations, and temporal relations. jsonld-ex temporal extensions are deliberately simpler: they cover the common case of validity intervals and observation timestamps without requiring the full OWL-Time machinery. For applications needing complex temporal reasoning (Allen's interval algebra, recurring events, calendar systems), OWL-Time remains the appropriate tool.

### 9.2 PROV-O Temporal Properties

[PROV-O](https://www.w3.org/TR/prov-o/) defines `prov:generatedAtTime` and `prov:invalidatedAtTime` for provenance records. The jsonld-ex `@extractedAt` maps to `prov:generatedAtTime`, and `@invalidatedAt` maps to `prov:wasInvalidatedBy`. The temporal extensions (`@validFrom`, `@validUntil`) have no direct PROV-O equivalent — they express *assertion validity*, not *entity lifecycle*. See the [Interoperability](interoperability.md) specification for full mapping tables.

### 9.3 Schema.org Temporal Properties

Schema.org defines `schema:validFrom` and `schema:validThrough` on types like `Offer` and `Permit`. The jsonld-ex temporal qualifiers generalize this pattern to any value in any JSON-LD document, not limited to specific schema.org types.

---

## 10. Reference Implementation

The reference implementation is in the `jsonld_ex.temporal` module of the [jsonld-ex Python package](https://pypi.org/project/jsonld-ex/).

| Function | Spec Section |
|----------|-------------|
| `add_temporal(value, valid_from, valid_until, as_of)` | §3 |
| `query_at_time(graph, timestamp, property_name)` | §5 |
| `temporal_diff(graph, t1, t2)` | §6 |
| `TemporalDiffResult` | §6.4 |

---

## References

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.
- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- W3C. (2017). OWL-Time. W3C Recommendation. https://www.w3.org/TR/owl-time/
- W3C. (2013). PROV-O: The PROV Ontology. W3C Recommendation. https://www.w3.org/TR/prov-o/
- Bradner, S. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. https://www.rfc-editor.org/rfc/rfc2119
