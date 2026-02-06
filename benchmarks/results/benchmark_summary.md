# jsonld-ex Benchmark Results

**Date:** 2026-02-06T23:10:22.039615+00:00  
**Version:** 0.2.0  
**Python:** 3.12.2  
**Total Time:** 3.36s

---

## Domain 1: OWL/RDF Ecosystem Interoperability

### PROV-O Verbosity Ratio

| Scale | jsonld-ex (bytes) | PROV-O (bytes) | Byte Ratio | Node Expansion | Triple Expansion |
|-------|-------------------|----------------|------------|----------------|------------------|
| n=10 | 3,563 | 16,902 | 0.211 | 7.0x | 7.0x |
| n=100 | 35,490 | 169,181 | 0.21 | 7.0x | 7.0x |
| n=1000 | 355,000 | 1,691,272 | 0.21 | 7.0x | 7.0x |

### SHACL Verbosity Ratio

| Complexity | @shape (bytes) | SHACL (bytes) | Ratio | Round-trip |
|------------|----------------|---------------|-------|------------|
| simple (2 props) | 128 | 646 | 0.198 | ✓ |
| medium (5 props) | 272 | 1205 | 0.226 | ✓ |
| complex (10 props) | 430 | 1943 | 0.221 | ✓ |

### Round-trip Fidelity: 100.0% (300/300 properties)

### Conversion Throughput

| Scale | to_prov_o (nodes/s) | from_prov_o (nodes/s) |
|-------|--------------------|-----------------------|
| n=10 | 148,588 | 5,617,935 |
| n=100 | 172,724 | 59,524,181 |
| n=1000 | 179,362 | 568,177,492 |

---

## Domain 2: Multi-Agent KG Construction

### Merge Throughput

| Scale | Avg (ms) | Nodes/sec |
|-------|----------|-----------|
| n=10 | 0.3 | 36,020 |
| n=50 | 1.2 | 42,117 |
| n=100 | 2.7 | 37,219 |
| n=500 | 11.8 | 42,363 |
| n=1000 | 23.8 | 41,987 |

### Merge by Conflict Rate

| Rate | Avg (ms) | Agreed | Conflicted |
|------|----------|--------|------------|
| rate=0.0 | 1.4 | 300 | 0 |
| rate=0.1 | 1.8 | 233 | 67 |
| rate=0.3 | 2.1 | 152 | 148 |
| rate=0.5 | 2.5 | 52 | 248 |
| rate=0.8 | 2.7 | 19 | 281 |
| rate=1.0 | 2.8 | 9 | 291 |

### Propagation Overhead (μs/call)

| Chain Length | multiply | bayesian | min | dampened |
|-------------|----------|----------|-----|----------|
| 2 | 2.47 | 1.24 | 0.72 | 0.84 |
| 5 | 1.14 | 1.88 | 0.99 | 1.17 |
| 10 | 1.65 | 3.23 | 1.64 | 1.83 |
| 20 | 3.08 | 6.02 | 2.89 | 3.33 |
| 50 | 6.89 | 12.81 | 6.28 | 6.84 |
| 100 | 12.44 | 26.95 | 12.13 | 12.71 |

---

## Domain 3: Healthcare IoT Pipeline

### Payload Sizes

| Batch | JSON | CBOR | gzip+CBOR | Savings |
|-------|------|------|-----------|---------|
| n=1 | 333 | 258 | 240 | 27.9% |
| n=10 | 2,946 | 2,409 | 565 | 80.8% |
| n=100 | 29,078 | 23,920 | 2,934 | 89.9% |
| n=1000 | 290,309 | 239,021 | 22,979 | 92.1% |

### Pipeline Throughput (n=1000)

| Phase | Time (ms) |
|-------|-----------|
| Annotate | 0.29 |
| Validate | 3.3 |
| Serialize (CBOR) | 1.9 |
| **Total** | **5.49** |
| **Throughput** | **182,149 readings/sec** |

### MQTT Overhead (per message)

| Operation | μs/msg |
|-----------|--------|
| Topic derivation | 0.9 |
| QoS derivation | 0.46 |
| Full roundtrip | 7.36 |

---

## Domain 4: RAG Pipeline & Temporal Queries

### Temporal Query Performance

| Scale | Avg (ms) | Nodes/sec |
|-------|----------|-----------|
| n=100 | 4.5 | 22,288 |
| n=500 | 21.1 | 23,723 |
| n=1000 | 43.2 | 23,142 |
| n=5000 | 213.3 | 23,436 |

### RAG Pipeline (n=500, 3 sources)

| Phase | Time (ms) |
|-------|-----------|
| Merge (3 sources) | 12.85 |
| Confidence filter (≥0.7) | 0.2 |
| **Total** | **13.05** |
| Nodes after filter | 499 |
| **Effective throughput** | **38,314 nodes/sec** |
