"""
jsonld-ex Benchmark Suite — Unified Runner

Runs all four evaluation domains and outputs combined results as
JSON (machine-readable) and a Markdown summary (paper-ready).

Usage:
    cd benchmarks
    python run_all.py
    # or for a specific domain:
    python bench_owl_rdf.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

# Add parent to path so we can import jsonld_ex from the source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "python", "src"))

import bench_owl_rdf
import bench_multi_agent
import bench_iot
import bench_rag


def main() -> None:
    print("=" * 60)
    print("jsonld-ex Benchmark Suite")
    print(f"Date: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 60)
    print()

    overall_start = time.perf_counter()

    # Domain 1
    d1 = bench_owl_rdf.run_all()

    # Domain 2
    d2 = bench_multi_agent.run_all()

    # Domain 3
    d3 = bench_iot.run_all()

    # Domain 4
    d4 = bench_rag.run_all()

    total_sec = time.perf_counter() - overall_start

    # ── Assemble results ──────────────────────────────────────

    results = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_seconds": round(total_sec, 2),
            "jsonld_ex_version": _get_version(),
            "python_version": sys.version.split()[0],
        },
        "domain_1_owl_rdf": {
            "prov_o_verbosity": d1.prov_o_verbosity,
            "shacl_verbosity": d1.shacl_verbosity,
            "round_trip_fidelity": d1.round_trip_fidelity,
            "conversion_throughput": d1.conversion_throughput,
        },
        "domain_2_multi_agent": {
            "merge_throughput": d2.merge_throughput,
            "merge_by_conflict_rate": d2.merge_by_conflict_rate,
            "propagation_overhead": d2.propagation_overhead,
            "combination_comparison": d2.combination_comparison,
            "diff_throughput": d2.diff_throughput,
        },
        "domain_3_iot": {
            "payload_sizes": d3.payload_sizes,
            "pipeline_throughput": d3.pipeline_throughput,
            "mqtt_overhead": d3.mqtt_overhead,
            "batch_scaling": d3.batch_scaling,
        },
        "domain_4_rag": {
            "confidence_filter": d4.confidence_filter,
            "temporal_query": d4.temporal_query,
            "temporal_diff": d4.temporal_diff_bench,
            "rag_pipeline": d4.rag_pipeline,
        },
    }

    # ── Save JSON ─────────────────────────────────────────────

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_path}")

    # ── Generate Markdown summary ─────────────────────────────

    md = _generate_markdown(results, d1, d2, d3, d4)
    md_path = os.path.join(out_dir, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown summary saved to: {md_path}")

    print(f"\nTotal benchmark time: {total_sec:.1f}s")


def _get_version() -> str:
    try:
        import jsonld_ex
        return jsonld_ex.__version__
    except Exception:
        return "unknown"


def _generate_markdown(results, d1, d2, d3, d4) -> str:
    lines = [
        "# jsonld-ex Benchmark Results",
        "",
        f"**Date:** {results['metadata']['timestamp']}  ",
        f"**Version:** {results['metadata']['jsonld_ex_version']}  ",
        f"**Python:** {results['metadata']['python_version']}  ",
        f"**Total Time:** {results['metadata']['total_seconds']}s",
        "",
        "---",
        "",
        "## Domain 1: OWL/RDF Ecosystem Interoperability",
        "",
        "### PROV-O Verbosity Ratio",
        "",
        "| Scale | jsonld-ex (bytes) | PROV-O (bytes) | Byte Ratio | Node Expansion | Triple Expansion |",
        "|-------|-------------------|----------------|------------|----------------|------------------|",
    ]

    for k, v in d1.prov_o_verbosity.items():
        lines.append(
            f"| {k} | {v['jsonld_ex_bytes']:,} | {v['prov_o_bytes']:,} | "
            f"{v['byte_ratio']} | {v['node_expansion_factor']}x | {v['triple_expansion']}x |"
        )

    lines += [
        "",
        "### SHACL Verbosity Ratio",
        "",
        "| Complexity | @shape (bytes) | SHACL (bytes) | Ratio | Round-trip |",
        "|------------|----------------|---------------|-------|------------|",
    ]
    for k, v in d1.shacl_verbosity.items():
        lines.append(
            f"| {k} | {v['shape_bytes']} | {v['shacl_bytes']} | "
            f"{v['byte_ratio']} | {'✓' if v['round_trip_properties_preserved'] else '✗'} |"
        )

    rt = d1.round_trip_fidelity
    lines += [
        "",
        f"### Round-trip Fidelity: {rt['fidelity']:.1%} "
        f"({rt['confidence_preserved']}/{rt['total_annotated_properties']} properties)",
        "",
        "### Conversion Throughput",
        "",
        "| Scale | to_prov_o (nodes/s) | from_prov_o (nodes/s) |",
        "|-------|--------------------|-----------------------|",
    ]
    for k, v in d1.conversion_throughput.items():
        lines.append(
            f"| {k} | {v['to_prov_o_nodes_per_sec']:,.0f} | "
            f"{v['from_prov_o_nodes_per_sec']:,.0f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Domain 2: Multi-Agent KG Construction",
        "",
        "### Merge Throughput",
        "",
        "| Scale | Avg (ms) | Nodes/sec |",
        "|-------|----------|-----------|",
    ]
    for k, v in d2.merge_throughput.items():
        lines.append(f"| {k} | {v['avg_sec']*1000:.1f} | {v['nodes_per_sec']:,.0f} |")

    lines += [
        "",
        "### Merge by Conflict Rate",
        "",
        "| Rate | Avg (ms) | Agreed | Conflicted |",
        "|------|----------|--------|------------|",
    ]
    for k, v in d2.merge_by_conflict_rate.items():
        lines.append(
            f"| {k} | {v['avg_sec']*1000:.1f} | {v['properties_agreed']} | "
            f"{v['properties_conflicted']} |"
        )

    lines += [
        "",
        "### Propagation Overhead (μs/call)",
        "",
        "| Chain Length | multiply | bayesian | min | dampened |",
        "|-------------|----------|----------|-----|----------|",
    ]
    for k, v in d2.propagation_overhead.items():
        lines.append(
            f"| {v['chain_length']} | {v['multiply_us']} | {v['bayesian_us']} | "
            f"{v['min_us']} | {v['dampened_us']} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Domain 3: Healthcare IoT Pipeline",
        "",
        "### Payload Sizes",
        "",
        "| Batch | JSON | CBOR | gzip+CBOR | Savings |",
        "|-------|------|------|-----------|---------|",
    ]
    for k, v in d3.payload_sizes.items():
        lines.append(
            f"| {k} | {v['json_bytes']:,} | {v['cbor_bytes']:,} | "
            f"{v['gzip_cbor_bytes']:,} | {v['savings_pct']}% |"
        )

    p = d3.pipeline_throughput
    lines += [
        "",
        f"### Pipeline Throughput (n={p['n_readings']})",
        "",
        f"| Phase | Time (ms) |",
        f"|-------|-----------|",
        f"| Annotate | {p['annotate_avg_ms']} |",
        f"| Validate | {p['validate_avg_ms']} |",
        f"| Serialize (CBOR) | {p['serialize_avg_ms']} |",
        f"| **Total** | **{p['total_avg_ms']}** |",
        f"| **Throughput** | **{p['readings_per_sec']:,.0f} readings/sec** |",
    ]

    m = d3.mqtt_overhead
    lines += [
        "",
        "### MQTT Overhead (per message)",
        "",
        f"| Operation | μs/msg |",
        f"|-----------|--------|",
        f"| Topic derivation | {m['topic_derivation_us_per_msg']} |",
        f"| QoS derivation | {m['qos_derivation_us_per_msg']} |",
        f"| Full roundtrip | {m['mqtt_roundtrip_us_per_msg']} |",
    ]

    lines += [
        "",
        "---",
        "",
        "## Domain 4: RAG Pipeline & Temporal Queries",
        "",
        "### Temporal Query Performance",
        "",
        "| Scale | Avg (ms) | Nodes/sec |",
        "|-------|----------|-----------|",
    ]
    for k, v in d4.temporal_query.items():
        lines.append(f"| {k} | {v['avg_ms']:.1f} | {v['nodes_per_sec']:,.0f} |")

    rp = d4.rag_pipeline
    lines += [
        "",
        f"### RAG Pipeline (n={rp['n_nodes']}, {rp['n_sources']} sources)",
        "",
        f"| Phase | Time (ms) |",
        f"|-------|-----------|",
        f"| Merge ({rp['n_sources']} sources) | {rp['merge_avg_ms']} |",
        f"| Confidence filter (≥{rp['threshold']}) | {rp['filter_avg_ms']} |",
        f"| **Total** | **{rp['total_avg_ms']}** |",
        f"| Nodes after filter | {rp['nodes_after_filter']} |",
        f"| **Effective throughput** | **{rp['effective_nodes_per_sec']:,.0f} nodes/sec** |",
    ]

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
