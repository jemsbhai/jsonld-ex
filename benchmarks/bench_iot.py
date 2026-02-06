"""
Benchmark Domain 3: Healthcare IoT Pipeline

Measures:
  - Payload sizes: JSON vs CBOR-LD vs gzip variants
  - End-to-end pipeline: annotate → validate → serialize
  - MQTT topic/QoS derivation overhead
  - Batch sensor processing throughput
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex import (
    annotate,
    validate_node,
    to_cbor,
    from_cbor,
    payload_stats,
    PayloadStats,
)
from jsonld_ex.mqtt import (
    to_mqtt_payload,
    from_mqtt_payload,
    derive_mqtt_topic,
    derive_mqtt_qos,
)

from data_generators import make_sensor_reading, make_sensor_batch


SENSOR_SHAPE = {
    "@type": "SensorReading",
    "value": {"@required": True, "@type": "xsd:double"},
    "unit": {"@required": True, "@type": "xsd:string"},
    "axis": {"@type": "xsd:string"},
}


@dataclass
class IoTResults:
    payload_sizes: dict[str, Any] = field(default_factory=dict)
    pipeline_throughput: dict[str, Any] = field(default_factory=dict)
    mqtt_overhead: dict[str, Any] = field(default_factory=dict)
    batch_scaling: dict[str, Any] = field(default_factory=dict)


def bench_payload_sizes(sizes: list[int] = [1, 10, 100, 1000]) -> dict[str, Any]:
    """Compare JSON, CBOR, gzip+JSON, gzip+CBOR for sensor batches."""
    results = {}
    for n in sizes:
        batch = make_sensor_batch(n)
        doc = {"@context": "http://schema.org/", "@graph": batch}

        stats = payload_stats(doc)
        results[f"n={n}"] = {
            "json_bytes": stats.json_bytes,
            "cbor_bytes": stats.cbor_bytes,
            "gzip_json_bytes": stats.gzip_json_bytes,
            "gzip_cbor_bytes": stats.gzip_cbor_bytes,
            "cbor_ratio": round(stats.cbor_ratio, 3),
            "gzip_cbor_ratio": round(stats.gzip_cbor_ratio, 3),
            "savings_pct": round((1 - stats.gzip_cbor_ratio) * 100, 1),
        }
    return results


def bench_pipeline_throughput(
    n_readings: int = 1000,
    iterations: int = 5,
) -> dict[str, Any]:
    """End-to-end: generate → annotate → validate → serialize → deserialize."""
    readings = make_sensor_batch(n_readings)

    # Time each phase
    results: dict[str, Any] = {"n_readings": n_readings}

    # Phase 1: Annotation (already annotated by generator, but simulate re-annotation)
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for r in readings:
            r["value"] = annotate(
                r["value"]["@value"],
                confidence=r["value"]["@confidence"],
                source=r["value"]["@source"],
            )
        times.append(time.perf_counter() - start)
    results["annotate_avg_ms"] = round(sum(times) / len(times) * 1000, 2)

    # Phase 2: Validation
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for r in readings:
            validate_node(r, SENSOR_SHAPE)
        times.append(time.perf_counter() - start)
    results["validate_avg_ms"] = round(sum(times) / len(times) * 1000, 2)

    # Phase 3: CBOR serialization
    doc = {"@context": "http://schema.org/", "@graph": readings}
    times_ser = []
    times_de = []
    for _ in range(iterations):
        start = time.perf_counter()
        payload = to_cbor(doc)
        times_ser.append(time.perf_counter() - start)

        start = time.perf_counter()
        from_cbor(payload)
        times_de.append(time.perf_counter() - start)

    results["serialize_avg_ms"] = round(sum(times_ser) / len(times_ser) * 1000, 2)
    results["deserialize_avg_ms"] = round(sum(times_de) / len(times_de) * 1000, 2)
    results["total_avg_ms"] = round(
        results["annotate_avg_ms"]
        + results["validate_avg_ms"]
        + results["serialize_avg_ms"],
        2,
    )
    results["readings_per_sec"] = round(
        n_readings / (results["total_avg_ms"] / 1000), 0
    )

    return results


def bench_mqtt_overhead(
    n: int = 1000,
    iterations: int = 5,
) -> dict[str, Any]:
    """Overhead of topic derivation and QoS mapping per message."""
    readings = make_sensor_batch(n)
    results: dict[str, Any] = {"n": n}

    # Topic derivation
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for r in readings:
            derive_mqtt_topic(r)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    results["topic_derivation_us_per_msg"] = round(avg / n * 1_000_000, 2)

    # QoS derivation
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for r in readings:
            derive_mqtt_qos(r)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    results["qos_derivation_us_per_msg"] = round(avg / n * 1_000_000, 2)

    # Full MQTT payload round-trip
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for r in readings:
            payload = to_mqtt_payload(r, compress=True)
            from_mqtt_payload(payload, compressed=True)
        times.append(time.perf_counter() - start)
    avg = sum(times) / len(times)
    results["mqtt_roundtrip_us_per_msg"] = round(avg / n * 1_000_000, 2)

    return results


def bench_batch_scaling(
    sizes: list[int] = [10, 100, 500, 1000, 5000, 10000],
    iterations: int = 3,
) -> dict[str, Any]:
    """Full pipeline throughput at varying batch sizes."""
    results = {}
    for n in sizes:
        readings = make_sensor_batch(n)
        doc = {"@context": "http://schema.org/", "@graph": readings}

        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            # annotate
            for r in readings:
                annotate(r["value"]["@value"], confidence=r["value"]["@confidence"])
            # validate
            for r in readings:
                validate_node(r, SENSOR_SHAPE)
            # serialize
            to_cbor(doc)
            times.append(time.perf_counter() - start)

        avg = sum(times) / len(times)
        results[f"n={n}"] = {
            "avg_sec": round(avg, 4),
            "readings_per_sec": round(n / avg, 0) if avg > 0 else 0,
        }
    return results


def run_all() -> IoTResults:
    results = IoTResults()
    print("=== Domain 3: Healthcare IoT Pipeline ===\n")

    print("3.1  Payload sizes...")
    results.payload_sizes = bench_payload_sizes()

    print("3.2  Pipeline throughput...")
    results.pipeline_throughput = bench_pipeline_throughput()

    print("3.3  MQTT overhead...")
    results.mqtt_overhead = bench_mqtt_overhead()

    print("3.4  Batch scaling...")
    results.batch_scaling = bench_batch_scaling()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n--- Payload Sizes ---")
    for k, v in r.payload_sizes.items():
        print(f"  {k}: JSON {v['json_bytes']}B → gzip+CBOR {v['gzip_cbor_bytes']}B "
              f"({v['savings_pct']}% smaller)")

    print(f"\n--- Pipeline Throughput (n={r.pipeline_throughput['n_readings']}) ---")
    p = r.pipeline_throughput
    print(f"  annotate:    {p['annotate_avg_ms']:.1f}ms")
    print(f"  validate:    {p['validate_avg_ms']:.1f}ms")
    print(f"  serialize:   {p['serialize_avg_ms']:.1f}ms")
    print(f"  total:       {p['total_avg_ms']:.1f}ms ({p['readings_per_sec']:.0f} readings/s)")

    print(f"\n--- MQTT Overhead ---")
    m = r.mqtt_overhead
    print(f"  topic derivation:  {m['topic_derivation_us_per_msg']:.1f} μs/msg")
    print(f"  QoS derivation:    {m['qos_derivation_us_per_msg']:.1f} μs/msg")
    print(f"  full roundtrip:    {m['mqtt_roundtrip_us_per_msg']:.1f} μs/msg")

    print(f"\n--- Batch Scaling ---")
    for k, v in r.batch_scaling.items():
        print(f"  {k}: {v['readings_per_sec']:.0f} readings/s")
