"""
Benchmark Domain 1: OWL/RDF Ecosystem Interoperability

Measures:
  - PROV-O verbosity ratio (triples, bytes)
  - SHACL verbosity ratio
  - Round-trip fidelity (to_prov_o → from_prov_o)
  - Conversion throughput (nodes/sec)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex import (
    to_prov_o,
    from_prov_o,
    shape_to_shacl,
    shacl_to_shape,
    compare_with_prov_o,
    compare_with_shacl,
    get_confidence,
)

from data_generators import make_annotated_graph


@dataclass
class InteropResults:
    prov_o_verbosity: dict[str, Any] = field(default_factory=dict)
    shacl_verbosity: dict[str, Any] = field(default_factory=dict)
    round_trip_fidelity: dict[str, Any] = field(default_factory=dict)
    conversion_throughput: dict[str, Any] = field(default_factory=dict)


def bench_prov_o_verbosity(sizes: list[int] = [10, 100, 1000]) -> dict[str, Any]:
    """Measure triple count and byte size: jsonld-ex vs PROV-O.

    Converts each node individually (to_prov_o handles single documents,
    not @graph arrays), then sums the results.
    """
    results = {}
    for n in sizes:
        doc = make_annotated_graph(n)
        doc_bytes = len(json.dumps(doc, separators=(",", ":")))

        # Convert each node individually and sum PROV-O output
        total_prov_nodes = 0
        total_prov_bytes = 0
        total_triples_in = 0
        total_triples_out = 0
        for node in doc["@graph"]:
            single = {"@context": "http://schema.org/"}
            single.update(node)
            prov_doc, report = to_prov_o(single)
            total_prov_nodes += len(prov_doc.get("@graph", []))
            total_prov_bytes += len(json.dumps(prov_doc, separators=(",", ":")))
            total_triples_in += report.triples_input
            total_triples_out += report.triples_output

        results[f"n={n}"] = {
            "jsonld_ex_bytes": doc_bytes,
            "prov_o_bytes": total_prov_bytes,
            "byte_ratio": round(doc_bytes / total_prov_bytes, 3) if total_prov_bytes else 0,
            "jsonld_ex_nodes": n,
            "prov_o_graph_nodes": total_prov_nodes,
            "node_expansion_factor": round(total_prov_nodes / n, 1) if n else 0,
            "triples_in": total_triples_in,
            "triples_out": total_triples_out,
            "triple_expansion": round(total_triples_out / total_triples_in, 1) if total_triples_in else 0,
        }
    return results


def bench_shacl_verbosity() -> dict[str, Any]:
    """Measure verbosity of @shape vs equivalent SHACL."""
    shapes = [
        {
            "label": "simple (2 props)",
            "shape": {
                "@type": "Person",
                "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
                "email": {"@pattern": "^[^@]+@[^@]+$"},
            },
        },
        {
            "label": "medium (5 props)",
            "shape": {
                "@type": "Person",
                "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
                "email": {"@pattern": "^[^@]+@[^@]+$"},
                "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
                "phone": {"@type": "xsd:string", "@pattern": "^\\+?[0-9]+$"},
                "url": {"@type": "xsd:string"},
            },
        },
        {
            "label": "complex (10 props)",
            "shape": {
                "@type": "MedicalObservation",
                "posture": {"@required": True, "@type": "xsd:string"},
                "confidence": {"@type": "xsd:double", "@minimum": 0.0, "@maximum": 1.0},
                "sensorId": {"@required": True, "@type": "xsd:string"},
                "timestamp": {"@required": True, "@type": "xsd:dateTime"},
                "ax": {"@type": "xsd:double"},
                "ay": {"@type": "xsd:double"},
                "az": {"@type": "xsd:double"},
                "gx": {"@type": "xsd:double"},
                "gy": {"@type": "xsd:double"},
                "gz": {"@type": "xsd:double"},
            },
        },
    ]

    results = {}
    for case in shapes:
        shape = case["shape"]
        shape_bytes = len(json.dumps({"@shape": shape}, separators=(",", ":")))
        shacl = shape_to_shacl(shape)
        shacl_bytes = len(json.dumps(shacl, separators=(",", ":")))

        # Round-trip
        rt, _warnings = shacl_to_shape(shacl)
        # Check structural match (ignoring key order)
        n_props = len([k for k in shape if k != "@type"])
        rt_props = len([k for k in rt if k != "@type"])

        results[case["label"]] = {
            "shape_bytes": shape_bytes,
            "shacl_bytes": shacl_bytes,
            "byte_ratio": round(shape_bytes / shacl_bytes, 3) if shacl_bytes else 0,
            "round_trip_properties_preserved": rt_props == n_props,
        }
    return results


def bench_round_trip_fidelity(n: int = 100) -> dict[str, Any]:
    """Verify lossless round-trip: jsonld-ex → PROV-O → jsonld-ex (per node)."""
    doc = make_annotated_graph(n)

    total_props = 0
    preserved = 0

    # Test round-trip per individual node (from_prov_o recovers one main node)
    for orig_node in doc["@graph"]:
        single_doc = {"@context": "http://schema.org/"}
        single_doc.update(orig_node)

        prov_doc, _ = to_prov_o(single_doc)
        restored, _ = from_prov_o(prov_doc)

        for key in ("name", "worksFor", "location"):
            if key in orig_node and isinstance(orig_node[key], dict):
                total_props += 1
                orig_conf = orig_node[key].get("@confidence")
                rest_conf = None
                if key in restored and isinstance(restored[key], dict):
                    rest_conf = restored[key].get("@confidence")
                if orig_conf is not None and rest_conf is not None:
                    if abs(orig_conf - rest_conf) < 0.001:
                        preserved += 1

    return {
        "total_annotated_properties": total_props,
        "confidence_preserved": preserved,
        "fidelity": round(preserved / total_props, 4) if total_props else 1.0,
    }


def bench_conversion_throughput(
    sizes: list[int] = [10, 100, 1000],
    iterations: int = 5,
) -> dict[str, Any]:
    """Measure to_prov_o and shape_to_shacl throughput."""
    results = {}
    for n in sizes:
        doc = make_annotated_graph(n)

        # to_prov_o
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            to_prov_o(doc)
            times.append(time.perf_counter() - start)
        avg_prov = sum(times) / len(times)

        # from_prov_o
        prov_doc, _ = to_prov_o(doc)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            from_prov_o(prov_doc)
            times.append(time.perf_counter() - start)
        avg_from = sum(times) / len(times)

        results[f"n={n}"] = {
            "to_prov_o_avg_sec": round(avg_prov, 6),
            "from_prov_o_avg_sec": round(avg_from, 6),
            "to_prov_o_nodes_per_sec": round(n / avg_prov, 1) if avg_prov > 0 else 0,
            "from_prov_o_nodes_per_sec": round(n / avg_from, 1) if avg_from > 0 else 0,
        }
    return results


def run_all() -> InteropResults:
    results = InteropResults()
    print("=== Domain 1: OWL/RDF Ecosystem ===\n")

    print("1.1  PROV-O verbosity ratio...")
    results.prov_o_verbosity = bench_prov_o_verbosity()

    print("1.2  SHACL verbosity ratio...")
    results.shacl_verbosity = bench_shacl_verbosity()

    print("1.3  Round-trip fidelity...")
    results.round_trip_fidelity = bench_round_trip_fidelity()

    print("1.4  Conversion throughput...")
    results.conversion_throughput = bench_conversion_throughput()

    return results


if __name__ == "__main__":
    r = run_all()
    print("\n--- PROV-O Verbosity ---")
    for k, v in r.prov_o_verbosity.items():
        print(f"  {k}: jsonld-ex {v['jsonld_ex_bytes']}B vs PROV-O {v['prov_o_bytes']}B "
              f"(ratio {v['byte_ratio']}, compression {v['compression_ratio']:.2f})")

    print("\n--- SHACL Verbosity ---")
    for k, v in r.shacl_verbosity.items():
        print(f"  {k}: @shape {v['shape_bytes']}B vs SHACL {v['shacl_bytes']}B "
              f"(ratio {v['byte_ratio']})")

    print(f"\n--- Round-trip Fidelity ---")
    rt = r.round_trip_fidelity
    print(f"  {rt['confidence_preserved']}/{rt['total_annotated_properties']} "
          f"properties preserved ({rt['fidelity']:.1%})")

    print("\n--- Conversion Throughput ---")
    for k, v in r.conversion_throughput.items():
        print(f"  {k}: to_prov_o {v['to_prov_o_nodes_per_sec']:.0f} nodes/s, "
              f"from_prov_o {v['from_prov_o_nodes_per_sec']:.0f} nodes/s")
