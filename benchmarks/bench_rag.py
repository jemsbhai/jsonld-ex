"""
Benchmark Domain 4: RAG Pipeline & Temporal Queries

Measures:
  - Confidence-filtered retrieval throughput
  - Temporal query (query_at_time) performance
  - temporal_diff overhead at scale
  - End-to-end RAG-style pipeline: filter → merge → query
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex import (
    get_confidence,
    merge_graphs,
    query_at_time,
    temporal_diff,
    add_temporal,
)

from data_generators import (
    make_annotated_graph,
    make_temporal_graph,
    make_conflicting_graphs,
)


@dataclass
class RAGResults:
    confidence_filter: dict[str, Any] = field(default_factory=dict)
    temporal_query: dict[str, Any] = field(default_factory=dict)
    temporal_diff_bench: dict[str, Any] = field(default_factory=dict)
    rag_pipeline: dict[str, Any] = field(default_factory=dict)


def _filter_graph_by_confidence(doc: dict, min_conf: float) -> dict:
    """Filter all nodes in a graph keeping only those with any property >= min_conf."""
    nodes = doc.get("@graph", [])
    kept = []
    for node in nodes:
        dominated = False
        for key, val in node.items():
            if key.startswith("@"):
                continue
            targets = val if isinstance(val, list) else [val]
            for t in targets:
                c = get_confidence(t)
                if c is not None and c >= min_conf:
                    dominated = True
                    break
            if dominated:
                break
        if dominated:
            kept.append(node)
    return {"@context": doc.get("@context", {}), "@graph": kept}


def bench_confidence_filter(
    sizes: list[int] = [100, 1000, 10000],
    thresholds: list[float] = [0.5, 0.7, 0.9],
    iterations: int = 5,
) -> dict[str, Any]:
    """Confidence-based graph filtering throughput and selectivity."""
    results = {}
    for n in sizes:
        doc = make_annotated_graph(n)
        for thresh in thresholds:
            times = []
            kept = 0
            for _ in range(iterations):
                start = time.perf_counter()
                filtered = _filter_graph_by_confidence(doc, thresh)
                times.append(time.perf_counter() - start)
                kept = len(filtered.get("@graph", []))
            avg = sum(times) / len(times)
            results[f"n={n},t={thresh}"] = {
                "avg_ms": round(avg * 1000, 2),
                "nodes_in": n,
                "nodes_kept": kept,
                "selectivity": round(kept / n, 3),
                "nodes_per_sec": round(n / avg, 0) if avg > 0 else 0,
            }
    return results


def bench_temporal_query(
    node_counts: list[int] = [100, 500, 1000, 5000],
    versions_per_node: int = 4,
    iterations: int = 5,
) -> dict[str, Any]:
    """query_at_time throughput at varying scale."""
    results = {}
    for n in node_counts:
        graph = make_temporal_graph(n, versions_per_node)
        times = []
        result_count = 0
        for _ in range(iterations):
            start = time.perf_counter()
            snapshot = query_at_time(graph, "2022-06-15T00:00:00Z")
            times.append(time.perf_counter() - start)
            result_count = len(snapshot)
        avg = sum(times) / len(times)
        results[f"n={n}"] = {
            "avg_ms": round(avg * 1000, 2),
            "nodes_returned": result_count,
            "nodes_per_sec": round(n / avg, 0) if avg > 0 else 0,
        }
    return results


def bench_temporal_diff(
    node_counts: list[int] = [100, 500, 1000],
    iterations: int = 5,
) -> dict[str, Any]:
    """temporal_diff throughput."""
    results = {}
    for n in node_counts:
        graph = make_temporal_graph(n, versions_per_node=4)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            temporal_diff(graph, "2020-06-01T00:00:00Z", "2024-06-01T00:00:00Z")
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)
        results[f"n={n}"] = {
            "avg_ms": round(avg * 1000, 2),
            "nodes_per_sec": round(n / avg, 0) if avg > 0 else 0,
        }
    return results


def bench_rag_pipeline(
    n_nodes: int = 500,
    n_sources: int = 3,
    confidence_threshold: float = 0.7,
    iterations: int = 5,
) -> dict[str, Any]:
    """Simulated RAG pipeline: multi-source merge → filter → temporal query."""
    graphs = make_conflicting_graphs(n_nodes, n_sources, conflict_rate=0.3)

    results: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_sources": n_sources,
        "threshold": confidence_threshold,
    }

    # Phase 1: Merge
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        merged, report = merge_graphs(graphs)
        times.append(time.perf_counter() - start)
    results["merge_avg_ms"] = round(sum(times) / len(times) * 1000, 2)

    # Phase 2: Confidence filter
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        filtered = _filter_graph_by_confidence(merged, confidence_threshold)
        times.append(time.perf_counter() - start)
    results["filter_avg_ms"] = round(sum(times) / len(times) * 1000, 2)
    results["nodes_after_filter"] = len(filtered.get("@graph", []))

    # Total
    results["total_avg_ms"] = round(
        results["merge_avg_ms"] + results["filter_avg_ms"], 2
    )
    results["effective_nodes_per_sec"] = round(
        n_nodes / (results["total_avg_ms"] / 1000), 0
    ) if results["total_avg_ms"] > 0 else 0

    return results


def run_all() -> RAGResults:
    results = RAGResults()
    print("=== Domain 4: RAG Pipeline & Temporal ===\n")

    print("4.1  Confidence-filtered retrieval...")
    results.confidence_filter = bench_confidence_filter()

    print("4.2  Temporal query performance...")
    results.temporal_query = bench_temporal_query()

    print("4.3  Temporal diff...")
    results.temporal_diff_bench = bench_temporal_diff()

    print("4.4  RAG pipeline (merge → filter)...")
    results.rag_pipeline = bench_rag_pipeline()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n--- Confidence Filter ---")
    for k, v in r.confidence_filter.items():
        print(f"  {k}: {v['nodes_per_sec']:.0f} nodes/s, "
              f"selectivity {v['selectivity']:.1%}")

    print("\n--- Temporal Query ---")
    for k, v in r.temporal_query.items():
        print(f"  {k}: {v['avg_ms']:.1f}ms ({v['nodes_per_sec']:.0f} nodes/s)")

    print("\n--- Temporal Diff ---")
    for k, v in r.temporal_diff_bench.items():
        print(f"  {k}: {v['avg_ms']:.1f}ms ({v['nodes_per_sec']:.0f} nodes/s)")

    print(f"\n--- RAG Pipeline ---")
    p = r.rag_pipeline
    print(f"  merge:  {p['merge_avg_ms']:.1f}ms")
    print(f"  filter: {p['filter_avg_ms']:.1f}ms")
    print(f"  total:  {p['total_avg_ms']:.1f}ms "
          f"({p['effective_nodes_per_sec']:.0f} nodes/s)")
