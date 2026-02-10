"""
Benchmark Domain 4: RAG Pipeline & Temporal Queries

Measures:
  - Confidence-filtered retrieval throughput
  - Temporal query (query_at_time) performance
  - temporal_diff overhead at scale
  - End-to-end RAG-style pipeline: merge → filter
  All with stddev, 95% CI, and n=30 trials.
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
from bench_utils import timed_trials, DEFAULT_TRIALS


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
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Confidence-based graph filtering throughput and selectivity."""
    results = {}
    for n in sizes:
        doc = make_annotated_graph(n)
        for thresh in thresholds:
            stats = timed_trials(
                lambda: _filter_graph_by_confidence(doc, thresh),
                n=n_trials,
            )
            # Get selectivity from one run
            filtered = _filter_graph_by_confidence(doc, thresh)
            kept = len(filtered.get("@graph", []))

            results[f"n={n},t={thresh}"] = {
                **stats.to_dict(),
                "nodes_in": n,
                "nodes_kept": kept,
                "selectivity": round(kept / n, 3),
                "nodes_per_sec": round(n / stats.mean, 0) if stats.mean > 0 else 0,
                "nodes_per_sec_ci95": [
                    round(n / stats.ci95_high, 0) if stats.ci95_high > 0 else 0,
                    round(n / stats.ci95_low, 0) if stats.ci95_low > 0 else 0,
                ],
            }
    return results


def bench_temporal_query(
    node_counts: list[int] = [100, 500, 1000, 5000],
    versions_per_node: int = 4,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """query_at_time throughput at varying scale."""
    results = {}
    for n in node_counts:
        graph = make_temporal_graph(n, versions_per_node)

        stats = timed_trials(
            lambda: query_at_time(graph, "2022-06-15T00:00:00Z"),
            n=n_trials,
        )
        # Get result count from one run
        snapshot = query_at_time(graph, "2022-06-15T00:00:00Z")
        result_count = len(snapshot)

        results[f"n={n}"] = {
            **stats.to_dict(),
            "avg_ms": stats.mean_ms(),
            "std_ms": stats.std_ms(),
            "nodes_returned": result_count,
            "nodes_per_sec": round(n / stats.mean, 0) if stats.mean > 0 else 0,
            "nodes_per_sec_ci95": [
                round(n / stats.ci95_high, 0) if stats.ci95_high > 0 else 0,
                round(n / stats.ci95_low, 0) if stats.ci95_low > 0 else 0,
            ],
        }
    return results


def bench_temporal_diff(
    node_counts: list[int] = [100, 500, 1000],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """temporal_diff throughput."""
    results = {}
    for n in node_counts:
        graph = make_temporal_graph(n, versions_per_node=4)

        stats = timed_trials(
            lambda: temporal_diff(graph, "2020-06-01T00:00:00Z", "2024-06-01T00:00:00Z"),
            n=n_trials,
        )
        results[f"n={n}"] = {
            **stats.to_dict(),
            "avg_ms": stats.mean_ms(),
            "std_ms": stats.std_ms(),
            "nodes_per_sec": round(n / stats.mean, 0) if stats.mean > 0 else 0,
        }
    return results


def bench_rag_pipeline(
    n_nodes: int = 500,
    n_sources: int = 3,
    confidence_threshold: float = 0.7,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Simulated RAG pipeline: multi-source merge → confidence filter."""
    graphs = make_conflicting_graphs(n_nodes, n_sources, conflict_rate=0.3)

    results: dict[str, Any] = {
        "n_nodes": n_nodes,
        "n_sources": n_sources,
        "threshold": confidence_threshold,
    }

    # Phase 1: Merge
    stats_merge = timed_trials(lambda: merge_graphs(graphs), n=n_trials)
    results["merge"] = stats_merge.to_dict()
    results["merge_avg_ms"] = stats_merge.mean_ms()
    results["merge_std_ms"] = stats_merge.std_ms()

    # Phase 2: Confidence filter
    merged, _ = merge_graphs(graphs)
    stats_filter = timed_trials(
        lambda: _filter_graph_by_confidence(merged, confidence_threshold),
        n=n_trials,
    )
    results["filter"] = stats_filter.to_dict()
    results["filter_avg_ms"] = stats_filter.mean_ms()
    results["filter_std_ms"] = stats_filter.std_ms()

    filtered = _filter_graph_by_confidence(merged, confidence_threshold)
    results["nodes_after_filter"] = len(filtered.get("@graph", []))

    # Total (propagate uncertainty via quadrature)
    total_mean = stats_merge.mean + stats_filter.mean
    total_std = (stats_merge.std**2 + stats_filter.std**2) ** 0.5
    results["total_avg_ms"] = round(total_mean * 1000, 2)
    results["total_std_ms"] = round(total_std * 1000, 2)
    results["effective_nodes_per_sec"] = round(
        n_nodes / total_mean, 0
    ) if total_mean > 0 else 0
    results["n_trials"] = n_trials

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
        print(f"  {k}: {v['nodes_per_sec']:.0f} nodes/s "
              f"(mean {v['mean_sec']*1000:.1f}ms ± {v['std_sec']*1000:.2f}ms)")

    print("\n--- Temporal Query ---")
    for k, v in r.temporal_query.items():
        print(f"  {k}: {v['avg_ms']:.1f} ± {v['std_ms']:.2f}ms "
              f"({v['nodes_per_sec']:.0f} nodes/s, n={v['n_trials']})")
