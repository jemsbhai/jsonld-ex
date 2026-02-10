"""
Benchmark Domain 3: Healthcare IoT Pipeline

Measures:
  - Payload sizes: JSON vs CBOR-LD vs gzip variants
  - End-to-end pipeline: annotate → validate → serialize
  - MQTT topic/QoS derivation overhead
  - Batch sensor processing throughput
  All with stddev, 95% CI, and n=30 trials.
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
from bench_utils import timed_trials, timed_trials_us, DEFAULT_TRIALS


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
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """End-to-end: generate → annotate → validate → serialize → deserialize."""
    readings = make_sensor_batch(n_readings)

    results: dict[str, Any] = {"n_readings": n_readings}

    # Phase 1: Annotation
    def do_annotate():
        for r in readings:
            r["value"] = annotate(
                r["value"]["@value"],
                confidence=r["value"]["@confidence"],
                source=r["value"]["@source"],
            )
    stats_ann = timed_trials(do_annotate, n=n_trials)
    results["annotate"] = stats_ann.to_dict()
    results["annotate_avg_ms"] = stats_ann.mean_ms()
    results["annotate_std_ms"] = stats_ann.std_ms()

    # Phase 2: Validation
    def do_validate():
        for r in readings:
            validate_node(r, SENSOR_SHAPE)
    stats_val = timed_trials(do_validate, n=n_trials)
    results["validate"] = stats_val.to_dict()
    results["validate_avg_ms"] = stats_val.mean_ms()
    results["validate_std_ms"] = stats_val.std_ms()

    # Phase 3: CBOR serialization
    doc = {"@context": "http://schema.org/", "@graph": readings}
    stats_ser = timed_trials(lambda: to_cbor(doc), n=n_trials)
    results["serialize"] = stats_ser.to_dict()
    results["serialize_avg_ms"] = stats_ser.mean_ms()
    results["serialize_std_ms"] = stats_ser.std_ms()

    # Total
    total_mean = stats_ann.mean + stats_val.mean + stats_ser.mean
    total_std = (stats_ann.std**2 + stats_val.std**2 + stats_ser.std**2) ** 0.5
    results["total_avg_ms"] = round(total_mean * 1000, 2)
    results["total_std_ms"] = round(total_std * 1000, 2)
    results["readings_per_sec"] = round(n_readings / total_mean, 0) if total_mean > 0 else 0
    results["n_trials"] = n_trials

    return results


def bench_mqtt_overhead(
    n: int = 1000,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Overhead of topic derivation and QoS mapping per message."""
    readings = make_sensor_batch(n)
    results: dict[str, Any] = {"n": n}

    # Topic derivation
    def do_topic():
        for r in readings:
            derive_mqtt_topic(r)
    stats_topic = timed_trials(do_topic, n=n_trials)
    results["topic_derivation_us_per_msg"] = round(stats_topic.mean / n * 1_000_000, 2)
    results["topic_derivation_std_us"] = round(stats_topic.std / n * 1_000_000, 2)

    # QoS derivation
    def do_qos():
        for r in readings:
            derive_mqtt_qos(r)
    stats_qos = timed_trials(do_qos, n=n_trials)
    results["qos_derivation_us_per_msg"] = round(stats_qos.mean / n * 1_000_000, 2)
    results["qos_derivation_std_us"] = round(stats_qos.std / n * 1_000_000, 2)

    # Full MQTT payload round-trip
    def do_roundtrip():
        for r in readings:
            payload = to_mqtt_payload(r, compress=True)
            from_mqtt_payload(payload, compressed=True)
    stats_rt = timed_trials(do_roundtrip, n=n_trials)
    results["mqtt_roundtrip_us_per_msg"] = round(stats_rt.mean / n * 1_000_000, 2)
    results["mqtt_roundtrip_std_us"] = round(stats_rt.std / n * 1_000_000, 2)
    results["n_trials"] = n_trials

    return results


def bench_batch_scaling(
    sizes: list[int] = [10, 100, 500, 1000, 5000, 10000],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Full pipeline throughput at varying batch sizes."""
    results = {}
    for n in sizes:
        readings = make_sensor_batch(n)
        doc = {"@context": "http://schema.org/", "@graph": readings}

        def pipeline():
            for r in readings:
                annotate(r["value"]["@value"], confidence=r["value"]["@confidence"])
            for r in readings:
                validate_node(r, SENSOR_SHAPE)
            to_cbor(doc)

        stats = timed_trials(pipeline, n=n_trials)
        results[f"n={n}"] = {
            **stats.to_dict(),
            "readings_per_sec": round(n / stats.mean, 0) if stats.mean > 0 else 0,
            "readings_per_sec_ci95": [
                round(n / stats.ci95_high, 0) if stats.ci95_high > 0 else 0,
                round(n / stats.ci95_low, 0) if stats.ci95_low > 0 else 0,
            ],
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
    print(f"  annotate:    {p['annotate_avg_ms']:.1f} ± {p['annotate_std_ms']:.2f}ms")
    print(f"  validate:    {p['validate_avg_ms']:.1f} ± {p['validate_std_ms']:.2f}ms")
    print(f"  serialize:   {p['serialize_avg_ms']:.1f} ± {p['serialize_std_ms']:.2f}ms")
    print(f"  total:       {p['total_avg_ms']:.1f} ± {p['total_std_ms']:.2f}ms "
          f"({p['readings_per_sec']:.0f} readings/s, n={p['n_trials']})")
