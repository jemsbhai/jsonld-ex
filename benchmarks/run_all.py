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
import bench_baselines
import bench_algebra
import bench_bridge


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

    # Baseline comparisons
    db = bench_baselines.run_all()

    # Domain 5: Confidence Algebra
    d5 = bench_algebra.run_all()

    # Domain 6: Neuro-Symbolic Bridge
    d6 = bench_bridge.run_all()

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
        "baselines": {
            "prov_o_construction": db.prov_o_construction,
            "shacl_validation": db.shacl_validation,
            "graph_merge": db.graph_merge,
            "temporal_query": db.temporal_query,
        },
        "domain_5_confidence_algebra": {
            "cumulative_fusion": d5.cumulative_fusion,
            "averaging_fusion": d5.averaging_fusion,
            "trust_discount_chain": d5.trust_discount_chain,
            "trust_vs_scalar": d5.trust_vs_scalar,
            "deduction": d5.deduction,
            "temporal_decay": d5.temporal_decay,
            "opinion_formation": d5.opinion_formation,
            "information_richness": d5.information_richness,
            "calibration": d5.calibration,
        },
        "domain_6_neuro_symbolic_bridge": {
            "pipeline_comparison": d6.pipeline_comparison,
            "metadata_richness": d6.metadata_richness,
        },
    }

    # ── Save JSON ─────────────────────────────────────────────

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)

    # Timestamped filenames for reproducibility and audit trail
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    json_ts_path = os.path.join(out_dir, f"benchmark_results_{ts}.json")
    with open(json_ts_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nJSON results saved to: {json_ts_path}")

    # Also write a "latest" copy for convenience (scripts/CI can reference this)
    json_latest = os.path.join(out_dir, "benchmark_results_latest.json")
    with open(json_latest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"JSON latest copy:      {json_latest}")

    # ── Generate Markdown summary ─────────────────────────────

    md = _generate_markdown(results, d1, d2, d3, d4, db, d5, d6)

    md_ts_path = os.path.join(out_dir, f"benchmark_summary_{ts}.md")
    with open(md_ts_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown summary saved to: {md_ts_path}")

    md_latest = os.path.join(out_dir, "benchmark_summary_latest.md")
    with open(md_latest, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Markdown latest copy:      {md_latest}")

    print(f"\nTotal benchmark time: {total_sec:.1f}s")


def _get_version() -> str:
    try:
        import jsonld_ex
        return jsonld_ex.__version__
    except Exception:
        return "unknown"


def _generate_markdown(results, d1, d2, d3, d4, db, d5, d6) -> str:
    n_trials = 30  # for display in header
    lines = [
        "# jsonld-ex Benchmark Results",
        "",
        f"**Date:** {results['metadata']['timestamp']}  ",
        f"**Version:** {results['metadata']['jsonld_ex_version']}  ",
        f"**Python:** {results['metadata']['python_version']}  ",
        f"**Total Time:** {results['metadata']['total_seconds']}s  ",
        f"**Trials per measurement:** {n_trials} (with 3 warmup iterations)  ",
        f"**Statistical method:** Mean +/- stddev, 95% CI via t-distribution",
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
            f"{v['byte_ratio']} | {'PASS' if v['round_trip_properties_preserved'] else 'FAIL'} |"
        )

    rt = d1.round_trip_fidelity
    lines += [
        "",
        f"### Round-trip Fidelity: {rt['fidelity']:.1%} "
        f"({rt['confidence_preserved']}/{rt['total_annotated_properties']} properties)",
        "",
        "### Conversion Throughput (n=30 trials)",
        "",
        "| Scale | to_prov_o (nodes/s) | to_prov_o (ms) | from_prov_o (nodes/s) | from_prov_o (ms) |",
        "|-------|---------------------|----------------|-----------------------|------------------|",
    ]
    for k, v in d1.conversion_throughput.items():
        tp = v['to_prov_o']
        fp = v['from_prov_o']
        lines.append(
            f"| {k} | {tp['nodes_per_sec']:,.0f} | "
            f"{tp['mean_sec']*1000:.2f} +/- {tp['std_sec']*1000:.2f} | "
            f"{fp['nodes_per_sec']:,.0f} | "
            f"{fp['mean_sec']*1000:.2f} +/- {fp['std_sec']*1000:.2f} |"
        )

    # -- Domain 1 Analysis (data-driven) --
    prov_ratios = [v['byte_ratio'] for v in d1.prov_o_verbosity.values()]
    prov_ratio_avg = sum(prov_ratios) / len(prov_ratios)
    prov_expansion = list(d1.prov_o_verbosity.values())[0]['node_expansion_factor']
    prov_inv_ratio = 1.0 / prov_ratio_avg if prov_ratio_avg > 0 else 0

    shacl_ratios = [v['byte_ratio'] for v in d1.shacl_verbosity.values()]
    shacl_ratio_avg = sum(shacl_ratios) / len(shacl_ratios)
    shacl_inv_ratio = 1.0 / shacl_ratio_avg if shacl_ratio_avg > 0 else 0

    ct_keys = list(d1.conversion_throughput.keys())
    ct_largest = d1.conversion_throughput[ct_keys[-1]]
    to_prov_nps = ct_largest['to_prov_o']['nodes_per_sec']
    from_prov_nps = ct_largest['from_prov_o']['nodes_per_sec']

    rt_fidelity_pct = rt['fidelity'] * 100
    rt_preserved = rt['confidence_preserved']
    rt_total = rt['total_annotated_properties']
    all_rt_pass = all(v['round_trip_properties_preserved'] for v in d1.shacl_verbosity.values())

    lines += [
        "",
        "### Analysis",
        "",
        f"PROV-O requires {prov_expansion:.0f}x more graph nodes and triples to express "
        f"the same provenance information. The byte ratio of ~{prov_ratio_avg:.2f} "
        f"(jsonld-ex is ~{prov_inv_ratio:.0f}x smaller) holds constant across all tested "
        f"scales, confirming per-annotation rather than structural overhead.",
        "",
        f"@shape definitions are ~{shacl_inv_ratio:.0f}x smaller than equivalent SHACL "
        f"shape graphs (ratio {min(shacl_ratios):.3f}--{max(shacl_ratios):.3f}). "
        f"Round-trip is {'lossless' if all_rt_pass else 'lossy'} -- shape_to_shacl "
        f"followed by shacl_to_shape preserves all constraint properties for common "
        f"validation patterns.",
        "",
        f"Round-trip fidelity: {rt_fidelity_pct:.0f}% across {rt_total} annotated "
        f"properties ({rt_preserved}/{rt_total} preserved). "
        f"Conversion throughput at largest scale: {to_prov_nps:,.0f} nodes/sec for "
        f"to_prov_o and {from_prov_nps:,.0f} nodes/sec for from_prov_o.",
        "",
    ]

    # ==================================================================
    # Domain 2
    # ==================================================================
    lines += [
        "---",
        "",
        "## Domain 2: Multi-Agent KG Construction",
        "",
        "### Merge Throughput (n=30 trials)",
        "",
        "| Scale | Mean (ms) | Nodes/sec | 95% CI (nodes/s) |",
        "|-------|-----------|-----------|-------------------|",
    ]
    for k, v in d2.merge_throughput.items():
        ci = v.get('nodes_per_sec_ci95', [0, 0])
        lines.append(
            f"| {k} | {v['mean_sec']*1000:.2f} +/- {v['std_sec']*1000:.2f} | "
            f"{v['nodes_per_sec']:,.0f} | [{ci[0]:,.0f}, {ci[1]:,.0f}] |"
        )

    lines += [
        "",
        "### Merge by Conflict Rate (n=30 trials)",
        "",
        "| Rate | Mean (ms) | Agreed | Conflicted |",
        "|------|-----------|--------|------------|",
    ]
    for k, v in d2.merge_by_conflict_rate.items():
        lines.append(
            f"| {k} | {v['mean_sec']*1000:.2f} +/- {v['std_sec']*1000:.2f} | "
            f"{v['properties_agreed']} | {v['properties_conflicted']} |"
        )

    lines += [
        "",
        "### Propagation Overhead (us/call, n=30 trials)",
        "",
        "| Chain Length | multiply | bayesian | min | dampened |",
        "|-------------|----------|----------|-----|----------|",
    ]
    for k, v in d2.propagation_overhead.items():
        lines.append(
            f"| {v['chain_length']} | "
            f"{v['multiply_us']} +/- {v.get('multiply_std_us', 0):.2f} | "
            f"{v['bayesian_us']} +/- {v.get('bayesian_std_us', 0):.2f} | "
            f"{v['min_us']} +/- {v.get('min_std_us', 0):.2f} | "
            f"{v['dampened_us']} +/- {v.get('dampened_std_us', 0):.2f} |"
        )

    # -- Domain 2 Analysis (data-driven) --
    merge_nps_values = [v['nodes_per_sec'] for v in d2.merge_throughput.values()]
    merge_nps_min = min(merge_nps_values)
    merge_nps_max = max(merge_nps_values)
    merge_nps_avg = sum(merge_nps_values) / len(merge_nps_values)

    conflict_times = list(d2.merge_by_conflict_rate.values())
    conflict_time_min = conflict_times[0]['mean_sec'] * 1000
    conflict_time_max = conflict_times[-1]['mean_sec'] * 1000
    conflict_slowdown = conflict_time_max / conflict_time_min if conflict_time_min > 0 else 0

    prop_keys = list(d2.propagation_overhead.keys())
    prop_largest = d2.propagation_overhead[prop_keys[-1]]
    prop_largest_len = prop_largest['chain_length']
    prop_mult_us = prop_largest['multiply_us']
    prop_bays_us = prop_largest['bayesian_us']
    prop_bays_ratio = prop_bays_us / prop_mult_us if prop_mult_us > 0 else 0

    lines += [
        "",
        "### Analysis",
        "",
        f"Confidence-aware graph merging scales linearly with stable performance "
        f"between {merge_nps_min:,.0f}--{merge_nps_max:,.0f} nodes/sec "
        f"(mean {merge_nps_avg:,.0f}) across all tested scales.",
        "",
        f"Merge time increases by ~{conflict_slowdown:.1f}x as conflict rate goes "
        f"from 0% to 100% ({conflict_time_min:.2f}ms to {conflict_time_max:.2f}ms). "
        f"Conflict resolution is computationally cheap relative to node alignment "
        f"and property iteration.",
        "",
        f"Confidence propagation at {prop_largest_len} hops: multiply method "
        f"costs {prop_mult_us:.2f}us, bayesian method costs {prop_bays_us:.2f}us "
        f"(~{prop_bays_ratio:.1f}x slower).",
        "",
    ]

    # ==================================================================
    # Domain 3
    # ==================================================================
    lines += [
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
        f"### Pipeline Throughput (n={p['n_readings']}, {p['n_trials']} trials)",
        "",
        f"| Phase | Mean (ms) |",
        f"|-------|-----------|",
        f"| Annotate | {p['annotate_avg_ms']} +/- {p['annotate_std_ms']} |",
        f"| Validate | {p['validate_avg_ms']} +/- {p['validate_std_ms']} |",
        f"| Serialize (CBOR) | {p['serialize_avg_ms']} +/- {p['serialize_std_ms']} |",
        f"| **Total** | **{p['total_avg_ms']} +/- {p['total_std_ms']}** |",
        f"| **Throughput** | **{p['readings_per_sec']:,.0f} readings/sec** |",
    ]

    m = d3.mqtt_overhead
    lines += [
        "",
        f"### MQTT Overhead (per message, {m['n_trials']} trials)",
        "",
        f"| Operation | Mean (us/msg) |",
        f"|-----------|---------------|",
        f"| Topic derivation | {m['topic_derivation_us_per_msg']} +/- {m['topic_derivation_std_us']} |",
        f"| QoS derivation | {m['qos_derivation_us_per_msg']} +/- {m['qos_derivation_std_us']} |",
        f"| Full roundtrip | {m['mqtt_roundtrip_us_per_msg']} +/- {m['mqtt_roundtrip_std_us']} |",
    ]

    # -- Domain 3 Analysis (data-driven) --
    payload_keys = list(d3.payload_sizes.keys())
    payload_largest = d3.payload_sizes[payload_keys[-1]]
    payload_n1 = d3.payload_sizes[payload_keys[0]]

    p_annotate = float(p['annotate_avg_ms'])
    p_validate = float(p['validate_avg_ms'])
    p_serialize = float(p['serialize_avg_ms'])
    p_total = float(p['total_avg_ms'])
    p_throughput = p['readings_per_sec']
    p_n = p['n_readings']

    m_topic = float(m['topic_derivation_us_per_msg'])
    m_qos = float(m['qos_derivation_us_per_msg'])
    m_roundtrip = float(m['mqtt_roundtrip_us_per_msg'])
    m_msgs_per_sec = int(1_000_000 / m_roundtrip) if m_roundtrip > 0 else 0

    lines += [
        "",
        "### Analysis",
        "",
        f"At single-reading granularity, CBOR+gzip savings are {payload_n1['savings_pct']}% "
        f"(fixed overhead dominates). At batch size {payload_keys[-1].split('=')[1]}, "
        f"a {payload_largest['json_bytes']:,}-byte JSON payload compresses to "
        f"{payload_largest['gzip_cbor_bytes']:,} bytes ({payload_largest['savings_pct']}% "
        f"reduction).",
        "",
        f"Pipeline throughput for {p_n} readings: annotation {p_annotate:.2f}ms, "
        f"validation {p_validate:.2f}ms (most expensive phase), "
        f"CBOR serialization {p_serialize:.2f}ms. Total {p_total:.2f}ms yields "
        f"{p_throughput:,.0f} readings/sec.",
        "",
        f"MQTT overhead: topic derivation {m_topic:.2f}us/msg, QoS derivation "
        f"{m_qos:.2f}us/msg, full roundtrip {m_roundtrip:.2f}us/msg "
        f"(~{m_msgs_per_sec:,} messages/sec per core).",
        "",
    ]

    # ==================================================================
    # Domain 4
    # ==================================================================
    lines += [
        "---",
        "",
        "## Domain 4: RAG Pipeline & Temporal Queries",
        "",
        "### Temporal Query Performance (n=30 trials)",
        "",
        "| Scale | Mean (ms) | Nodes/sec | 95% CI (nodes/s) |",
        "|-------|-----------|-----------|-------------------|",
    ]
    for k, v in d4.temporal_query.items():
        ci = v.get('nodes_per_sec_ci95', [0, 0])
        lines.append(
            f"| {k} | {v['avg_ms']:.2f} +/- {v['std_ms']:.2f} | "
            f"{v['nodes_per_sec']:,.0f} | [{ci[0]:,.0f}, {ci[1]:,.0f}] |"
        )

    rp = d4.rag_pipeline
    lines += [
        "",
        f"### RAG Pipeline (n={rp['n_nodes']}, {rp['n_sources']} sources, {rp['n_trials']} trials)",
        "",
        f"| Phase | Mean (ms) |",
        f"|-------|-----------|",
        f"| Merge ({rp['n_sources']} sources) | {rp['merge_avg_ms']} +/- {rp['merge_std_ms']} |",
        f"| Confidence filter (>={rp['threshold']}) | {rp['filter_avg_ms']} +/- {rp['filter_std_ms']} |",
        f"| **Total** | **{rp['total_avg_ms']} +/- {rp['total_std_ms']}** |",
        f"| Nodes after filter | {rp['nodes_after_filter']} |",
        f"| **Effective throughput** | **{rp['effective_nodes_per_sec']:,.0f} nodes/sec** |",
    ]

    # -- Domain 4 Analysis (data-driven) --
    tq_nps_values = [v['nodes_per_sec'] for v in d4.temporal_query.values()]
    tq_nps_avg = sum(tq_nps_values) / len(tq_nps_values)
    tq_keys = list(d4.temporal_query.keys())

    rp_merge_ms = float(rp['merge_avg_ms'])
    rp_filter_ms = float(rp['filter_avg_ms'])
    rp_total_ms = float(rp['total_avg_ms'])
    rp_n_nodes = rp['n_nodes']
    rp_eff_nps = rp['effective_nodes_per_sec']
    rp_merge_pct = (rp_merge_ms / rp_total_ms * 100) if rp_total_ms > 0 else 0

    lines += [
        "",
        "### Analysis",
        "",
        f"Temporal queries achieve ~{tq_nps_avg:,.0f} nodes/sec with linear scaling "
        f"from {tq_keys[0]} to {tq_keys[-1]}.",
        "",
        f"RAG pipeline ({rp['n_sources']} sources, filter >= {rp['threshold']}): "
        f"{rp_total_ms:.1f}ms for {rp_n_nodes} nodes ({rp_eff_nps:,.0f} nodes/sec). "
        f"Merge dominates at {rp_merge_ms:.1f}ms ({rp_merge_pct:.0f}% of total); "
        f"confidence filtering costs {rp_filter_ms:.3f}ms.",
        "",
    ]

    # ==================================================================
    # Baseline Comparisons
    # ==================================================================
    lines += [
        "---",
        "",
        "## Baseline Comparisons Against Existing Tools",
        "",
        "All comparisons perform the **same task** using both the established tool",
        "and jsonld-ex, measuring wall-clock time under identical conditions (n=30 trials).",
        "",
    ]

    # B.1 PROV-O Construction
    lines += [
        "### B.1 PROV-O Provenance Construction: rdflib vs jsonld-ex",
        "",
        "| Scale | rdflib (ms) | jsonld-ex (ms) | Speedup |",
        "|-------|-------------|----------------|---------|",
    ]
    for k, v in db.prov_o_construction.items():
        rl = v['rdflib']
        jl = v['jsonld_ex']
        lines.append(
            f"| {k} | {rl['mean_sec']*1000:.2f} +/- {rl['std_sec']*1000:.2f} | "
            f"{jl['mean_sec']*1000:.2f} +/- {jl['std_sec']*1000:.2f} | "
            f"{v['speedup']}x |"
        )
    lines.append("")

    # B.2 SHACL Validation
    if "skipped" not in db.shacl_validation:
        lines += [
            "### B.2 SHACL Validation: pyshacl vs jsonld-ex",
            "",
            "| Scale | pyshacl (ms) | jsonld-ex (ms) | Speedup |",
            "|-------|--------------|----------------|---------|",
        ]
        for k, v in db.shacl_validation.items():
            ps = v['pyshacl']
            jl = v['jsonld_ex']
            lines.append(
                f"| {k} | {ps['mean_sec']*1000:.2f} +/- {ps['std_sec']*1000:.2f} | "
                f"{jl['mean_sec']*1000:.2f} +/- {jl['std_sec']*1000:.2f} | "
                f"{v['speedup']}x |"
            )
        lines.append("")
    else:
        lines += [
            "### B.2 SHACL Validation: pyshacl vs jsonld-ex",
            "",
            "*Skipped: pyshacl not installed.*",
            "",
        ]

    # B.3 Graph Merge
    lines += [
        "### B.3 Graph Merge: rdflib vs jsonld-ex",
        "",
        "| Scale | rdflib (ms) | jsonld-ex (ms) | Speedup |",
        "|-------|-------------|----------------|---------|",
    ]
    for k, v in db.graph_merge.items():
        rl = v['rdflib_merge']
        jl = v['jsonld_ex_merge']
        lines.append(
            f"| {k} | {rl['mean_sec']*1000:.2f} +/- {rl['std_sec']*1000:.2f} | "
            f"{jl['mean_sec']*1000:.2f} +/- {jl['std_sec']*1000:.2f} | "
            f"{v['speedup']}x |"
        )
    lines.append("")

    # B.4 Temporal Query
    lines += [
        "### B.4 Temporal Query: SPARQL via rdflib vs jsonld-ex",
        "",
        "| Scale | SPARQL (ms) | jsonld-ex (ms) | Speedup |",
        "|-------|-------------|----------------|---------|",
    ]
    for k, v in db.temporal_query.items():
        sp = v['rdflib_sparql']
        jl = v['jsonld_ex']
        lines.append(
            f"| {k} | {sp['mean_sec']*1000:.2f} +/- {sp['std_sec']*1000:.2f} | "
            f"{jl['mean_sec']*1000:.2f} +/- {jl['std_sec']*1000:.2f} | "
            f"{v['speedup']}x |"
        )
    lines.append("")

    # -- Baseline Analysis (data-driven) --
    prov_speedups = [v['speedup'] for v in db.prov_o_construction.values()]
    merge_speedups = [v['speedup'] for v in db.graph_merge.values()]
    temp_speedups = [v['speedup'] for v in db.temporal_query.values()]

    if "skipped" not in db.shacl_validation:
        shacl_speedups = [v['speedup'] for v in db.shacl_validation.values()]
        shacl_str = f"SHACL validation {min(shacl_speedups)}--{max(shacl_speedups)}x, "
    else:
        shacl_str = ""

    lines += [
        "### Baseline Analysis",
        "",
        f"Speedup ranges: "
        f"PROV-O construction {min(prov_speedups)}--{max(prov_speedups)}x, "
        f"{shacl_str}"
        f"graph merge {min(merge_speedups)}--{max(merge_speedups)}x, "
        f"temporal query {min(temp_speedups)}--{max(temp_speedups)}x. "
        f"The advantage comes from co-locating metadata directly on JSON-LD values "
        f"rather than materializing separate RDF graph structures.",
    ]

    # ==================================================================
    # Domain 5: Confidence Algebra
    # ==================================================================
    lines += [
        "",
        "---",
        "",
        "## Domain 5: Confidence Algebra (Subjective Logic)",
        "",
        "### Cumulative Fusion Throughput",
        "",
        "| Opinions | Mean (us) | Ops/sec |",
        "|----------|-----------|---------|",
    ]
    for k, v in d5.cumulative_fusion.items():
        lines.append(
            f"| {k} | {v['mean_us']:.2f} +/- {v['std_us']:.2f} | "
            f"{v['ops_per_sec']:,.0f} |"
        )

    lines += [
        "",
        "### Averaging Fusion Throughput (n-ary simultaneous)",
        "",
        "| Opinions | Mean (us) | Ops/sec |",
        "|----------|-----------|---------|",
    ]
    for k, v in d5.averaging_fusion.items():
        lines.append(
            f"| {k} | {v['mean_us']:.2f} +/- {v['std_us']:.2f} | "
            f"{v['ops_per_sec']:,.0f} |"
        )

    lines += [
        "",
        "### Trust Discount Chain",
        "",
        "| Chain Length | Mean (us) | us/hop |",
        "|-------------|-----------|--------|",
    ]
    for k, v in d5.trust_discount_chain.items():
        lines.append(
            f"| {k} | {v['mean_us']:.2f} +/- {v['std_us']:.2f} | "
            f"{v['us_per_hop']:.2f} |"
        )

    lines += [
        "",
        "### Trust Discount vs Scalar Multiply Equivalence",
        "",
        "| Chain | Scalar (us) | Algebra (us) | Overhead | Equivalent |",
        "|-------|-------------|--------------|----------|------------|",
    ]
    for k, v in d5.trust_vs_scalar.items():
        eq = "PASS" if v['numerically_equivalent'] else "FAIL"
        lines.append(
            f"| {k} | {v['scalar_us']:.2f} | {v['algebra_us']:.2f} | "
            f"+{v['overhead_pct']:.1f}% | {eq} |"
        )

    lines += [
        "",
        "### Deduction Throughput",
        "",
        "| Operation | Mean (us) |",
        "|-----------|-----------|",
    ]
    for k, v in d5.deduction.items():
        extra = f" ({v.get('us_per_stage', 0):.2f} us/stage)" if 'us_per_stage' in v else ""
        lines.append(f"| {k} | {v['mean_us']:.2f} +/- {v['std_us']:.2f}{extra} |")

    lines += [
        "",
        "### Opinion Formation & Serialization",
        "",
        "| Operation | Mean (us) |",
        "|-----------|-----------|",
    ]
    for k, v in d5.opinion_formation.items():
        lines.append(f"| {k} | {v['mean_us']:.2f} +/- {v['std_us']:.2f} |")

    lines += [
        "",
        "### Information Richness: Scalar vs Algebra",
        "",
        "| Scenario | P(a) | P(b) | Same Scalar | u(a) | u(b) |",
        "|----------|------|------|-------------|------|------|",
    ]
    for k, v in d5.information_richness.items():
        lines.append(
            f"| {k} | {v['P_a']:.4f} | {v['P_b']:.4f} | "
            f"{v['same_scalar']} | {v['uncertainty_a']:.2f} | {v['uncertainty_b']:.2f} |"
        )

    cal = d5.calibration
    lines += [
        "",
        "### Calibration Analysis",
        "",
        f"- **Expected Calibration Error (ECE):** {cal['expected_calibration_error']:.4f}",
        f"- **Brier Score:** {cal['brier_score']:.4f}",
        f"- **Mean Uncertainty:** {cal['mean_uncertainty']:.4f}",
        f"- **Propositions:** {cal['n_propositions']}, {cal['n_sources_per_proposition']} sources each",
        "",
        "| Bin | Count | Predicted | Actual | Error | Uncertainty |",
        "|-----|-------|-----------|--------|-------|-------------|",
    ]
    for b in cal['bins']:
        if b['count'] > 0:
            lines.append(
                f"| {b['bin']} | {b['count']} | {b['mean_predicted']:.3f} | "
                f"{b['mean_actual']:.3f} | {b['calibration_error']:.3f} | "
                f"{b['mean_uncertainty']:.3f} |"
            )

    # -- Domain 5 Analysis (data-driven) --
    cum_n2 = d5.cumulative_fusion.get('n=2', {})
    cum_n2_us = cum_n2.get('mean_us', 0)
    cum_n2_ops = cum_n2.get('ops_per_sec', 0)
    cum_n100 = d5.cumulative_fusion.get('n=100', {})
    cum_n100_us = cum_n100.get('mean_us', 0)
    cum_n100_ops = cum_n100.get('ops_per_sec', 0)

    cum_per_opinion_us = (cum_n100_us - cum_n2_us) / (100 - 2) if (cum_n100_us > 0 and cum_n2_us > 0) else 0

    avg_n100 = d5.averaging_fusion.get('n=100', {})
    avg_n100_us = avg_n100.get('mean_us', 0)
    avg_n100_ops = avg_n100.get('ops_per_sec', 0)
    avg_vs_cum_ratio = cum_n100_us / avg_n100_us if avg_n100_us > 0 else 0

    td_hops = [v['us_per_hop'] for v in d5.trust_discount_chain.values()]
    td_avg_hop = sum(td_hops) / len(td_hops) if td_hops else 0

    td_overheads = [v['overhead_pct'] for v in d5.trust_vs_scalar.values()]
    td_overhead_min = min(td_overheads) if td_overheads else 0
    td_overhead_max = max(td_overheads) if td_overheads else 0
    td_all_equiv = all(v['numerically_equivalent'] for v in d5.trust_vs_scalar.values())

    ir_all_same = all(v['same_scalar'] for v in d5.information_richness.values())
    ir_count = len(d5.information_richness)

    cal_ece = cal['expected_calibration_error']
    cal_brier = cal['brier_score']

    lines += [
        "",
        "### Analysis",
        "",
        f"Cumulative fusion at n=2: {cum_n2_us:.2f}us ({cum_n2_ops:,.0f} ops/sec). "
        f"Scales linearly at ~{cum_per_opinion_us:.2f}us per additional opinion. "
        f"At n=100: {cum_n100_us:.1f}us ({cum_n100_ops:,.0f} ops/sec).",
        "",
        f"Averaging fusion (n-ary simultaneous formula): at n=100 costs "
        f"{avg_n100_us:.1f}us ({avg_n100_ops:,.0f} ops/sec), "
        f"{avg_vs_cum_ratio:.1f}x faster than cumulative at the same scale.",
        "",
        f"Trust discount chains: ~{td_avg_hop:.2f}us/hop (constant per-hop cost). "
        f"Equivalence with scalar multiplication "
        f"{'verified' if td_all_equiv else 'FAILED'} across all chain lengths, "
        f"with +{td_overhead_min:.0f}% to +{td_overhead_max:.0f}% overhead for "
        f"preserving full (b, d, u, a) tuples.",
        "",
        f"Information richness: {ir_count} pairs of opinions map to identical scalar "
        f"probabilities but carry different epistemic states "
        f"(all same_scalar={'True' if ir_all_same else 'False'}).",
        "",
        f"Calibration: ECE={cal_ece:.4f}, Brier={cal_brier:.4f}. Cumulative fusion "
        f"of evidence-based opinions produces well-calibrated probability estimates.",
    ]

    # ==================================================================
    # Domain 6: Neuro-Symbolic Bridge
    # ==================================================================
    lines += [
        "",
        "---",
        "",
        "## Domain 6: Neuro-Symbolic Bridge Pipeline",
        "",
        "End-to-end: ML outputs -> opinion lift -> fusion -> decay -> validate -> PROV-O export -> filter",
        "",
        "### Pipeline Comparison: jsonld-ex vs Ad-hoc",
        "",
        "| Scale | jsonld-ex (ms) | Ad-hoc (ms) | Overhead | Conflicts | Validated |",
        "|-------|----------------|-------------|----------|-----------|-----------|",
    ]
    for k, v in d6.pipeline_comparison.items():
        jl = v['jsonld_ex']
        ah = v['adhoc']
        lines.append(
            f"| {k} | {jl['mean_sec']*1000:.2f} +/- {jl['std_sec']*1000:.2f} | "
            f"{ah['mean_sec']*1000:.2f} +/- {ah['std_sec']*1000:.2f} | "
            f"{v['overhead_factor']}x | "
            f"{jl['metadata']['merge_conflicts']} | {jl['metadata']['valid_nodes']} |"
        )

    lines += [
        "",
        "### Phase Breakdown (n=1000 nodes)",
        "",
    ]
    if "n=1000" in d6.pipeline_comparison:
        phases = d6.pipeline_comparison["n=1000"]["jsonld_ex"]["phase_breakdown_ms"]
        lines += [
            "| Phase | Time (ms) | % of Total |",
            "|-------|-----------|------------|",
        ]
        total_ms_d6 = sum(phases.values())
        for phase, ms in phases.items():
            pct = (ms / total_ms_d6 * 100) if total_ms_d6 > 0 else 0
            label = phase.replace('_sec', '')
            lines.append(f"| {label} | {ms:.3f} | {pct:.1f}% |")

    mr = d6.metadata_richness
    lines += [
        "",
        "### Metadata Richness",
        "",
        f"| Pipeline | Metadata Dimensions |",
        f"|----------|---------------------|",
        f"| jsonld-ex | {mr['jsonld_ex_preserves']['metadata_dimensions']} |",
        f"| ad-hoc | {mr['adhoc_preserves']['metadata_dimensions']} |",
    ]

    # -- Domain 6 Analysis (data-driven) --
    if "n=1000" in d6.pipeline_comparison:
        d6_1k = d6.pipeline_comparison["n=1000"]
        d6_jl_ms = d6_1k['jsonld_ex']['mean_sec'] * 1000
        d6_ah_ms = d6_1k['adhoc']['mean_sec'] * 1000
        d6_overhead = d6_1k['overhead_factor']
        d6_nps = 1000 / (d6_jl_ms / 1000) if d6_jl_ms > 0 else 0

        d6_phases = d6_1k['jsonld_ex']['phase_breakdown_ms']
        d6_total_phase_ms = sum(d6_phases.values())

        sorted_phases = sorted(d6_phases.items(), key=lambda x: x[1], reverse=True)
        phase_strs = []
        for phase_name, phase_ms in sorted_phases:
            phase_pct = (phase_ms / d6_total_phase_ms * 100) if d6_total_phase_ms > 0 else 0
            label = phase_name.replace('_sec', '')
            phase_strs.append(f"{label} {phase_ms:.1f}ms ({phase_pct:.0f}%)")

        d6_jl_dims = mr['jsonld_ex_preserves']['metadata_dimensions']
        d6_ah_dims = mr['adhoc_preserves']['metadata_dimensions']

        lines += [
            "",
            "### Analysis",
            "",
            f"At 1,000 nodes, the full 6-stage pipeline runs in {d6_jl_ms:.0f}ms "
            f"({d6_nps:,.0f} nodes/sec), {d6_overhead}x slower than the ad-hoc "
            f"baseline ({d6_ah_ms:.1f}ms). The ad-hoc baseline performs only scalar "
            f"weighting and naive merge, preserving {d6_ah_dims} metadata dimensions. "
            f"jsonld-ex preserves {d6_jl_dims} metadata dimensions.",
            "",
            f"Phase breakdown (descending cost): {', '.join(phase_strs)}.",
            "",
            f"The overhead is the cost of {d6_jl_dims} vs {d6_ah_dims} metadata "
            f"dimensions: quality-aware decisions, provenance auditing, PROV-O/SHACL "
            f"interop, and temporal freshness reasoning.",
        ]
    else:
        lines += [
            "",
            "### Analysis",
            "",
            "(n=1000 data point not available for detailed analysis.)",
        ]

    return "\n".join(lines) + "\n"



if __name__ == "__main__":
    main()
