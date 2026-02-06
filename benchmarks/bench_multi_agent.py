"""
Benchmark Domain 2: Multi-Agent Knowledge Graph Construction

Measures:
  - merge_graphs throughput at varying scale and conflict rates
  - Confidence propagation overhead per chain length
  - Conflict resolution accuracy and throughput
  - combine_sources comparison across methods
"""

from __future__ import annotations

import time
import json
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex import (
    merge_graphs,
    diff_graphs,
    propagate_confidence,
    combine_sources,
    resolve_conflict,
)

from data_generators import make_conflicting_graphs, make_annotated_graph


@dataclass
class MultiAgentResults:
    merge_throughput: dict[str, Any] = field(default_factory=dict)
    merge_by_conflict_rate: dict[str, Any] = field(default_factory=dict)
    propagation_overhead: dict[str, Any] = field(default_factory=dict)
    combination_comparison: dict[str, Any] = field(default_factory=dict)
    diff_throughput: dict[str, Any] = field(default_factory=dict)


def bench_merge_throughput(
    node_counts: list[int] = [10, 50, 100, 500, 1000],
    n_sources: int = 3,
    iterations: int = 5,
) -> dict[str, Any]:
    """Merge throughput: nodes per second at varying scale."""
    results = {}
    for n in node_counts:
        graphs = make_conflicting_graphs(n, n_sources, conflict_rate=0.3)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            merge_graphs(graphs)
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)
        results[f"n={n}"] = {
            "avg_sec": round(avg, 6),
            "nodes_per_sec": round(n / avg, 1) if avg > 0 else 0,
            "total_input_nodes": n * n_sources,
        }
    return results


def bench_merge_by_conflict_rate(
    n_nodes: int = 100,
    n_sources: int = 3,
    rates: list[float] = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0],
    iterations: int = 5,
) -> dict[str, Any]:
    """How conflict rate affects merge time and report contents."""
    results = {}
    for rate in rates:
        graphs = make_conflicting_graphs(n_nodes, n_sources, conflict_rate=rate)
        times = []
        reports = []
        for _ in range(iterations):
            start = time.perf_counter()
            _, report = merge_graphs(graphs)
            times.append(time.perf_counter() - start)
            reports.append(report)
        avg = sum(times) / len(times)
        last = reports[-1]
        results[f"rate={rate}"] = {
            "avg_sec": round(avg, 6),
            "properties_agreed": last.properties_agreed,
            "properties_conflicted": last.properties_conflicted,
            "nodes_merged": last.nodes_merged,
        }
    return results


def bench_propagation_overhead(
    chain_lengths: list[int] = [2, 5, 10, 20, 50, 100],
    methods: list[str] = ["multiply", "bayesian", "min", "dampened"],
    iterations: int = 1000,
) -> dict[str, Any]:
    """Time per propagation call at varying chain length."""
    results = {}
    for length in chain_lengths:
        chain = [0.9] * length  # uniform chain for consistency
        row: dict[str, Any] = {"chain_length": length}
        for method in methods:
            start = time.perf_counter()
            for _ in range(iterations):
                propagate_confidence(chain, method=method)  # type: ignore
            elapsed = time.perf_counter() - start
            row[f"{method}_us"] = round(elapsed / iterations * 1_000_000, 2)
        results[f"len={length}"] = row
    return results


def bench_combination_comparison(
    source_counts: list[int] = [2, 3, 5, 10, 20],
    methods: list[str] = ["average", "max", "noisy_or", "dempster_shafer"],
    iterations: int = 1000,
) -> dict[str, Any]:
    """Compare combine_sources methods: output score and timing."""
    import random
    rng = random.Random(42)

    results = {}
    for n in source_counts:
        scores = [round(rng.uniform(0.5, 0.95), 3) for _ in range(n)]
        row: dict[str, Any] = {"n_sources": n, "input_scores": scores}
        for method in methods:
            start = time.perf_counter()
            for _ in range(iterations):
                r = combine_sources(scores, method=method)  # type: ignore
            elapsed = time.perf_counter() - start
            row[f"{method}_score"] = round(r.score, 4)
            row[f"{method}_us"] = round(elapsed / iterations * 1_000_000, 2)
        results[f"n={n}"] = row
    return results


def bench_diff_throughput(
    sizes: list[int] = [10, 100, 500, 1000],
    iterations: int = 5,
) -> dict[str, Any]:
    """diff_graphs throughput at varying scale."""
    results = {}
    for n in sizes:
        graphs = make_conflicting_graphs(n, 2, conflict_rate=0.3)
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            diff_graphs(graphs[0], graphs[1])
            times.append(time.perf_counter() - start)
        avg = sum(times) / len(times)
        results[f"n={n}"] = {
            "avg_sec": round(avg, 6),
            "nodes_per_sec": round(n / avg, 1) if avg > 0 else 0,
        }
    return results


def run_all() -> MultiAgentResults:
    results = MultiAgentResults()
    print("=== Domain 2: Multi-Agent KG Construction ===\n")

    print("2.1  Merge throughput by scale...")
    results.merge_throughput = bench_merge_throughput()

    print("2.2  Merge by conflict rate...")
    results.merge_by_conflict_rate = bench_merge_by_conflict_rate()

    print("2.3  Propagation overhead...")
    results.propagation_overhead = bench_propagation_overhead()

    print("2.4  Combination method comparison...")
    results.combination_comparison = bench_combination_comparison()

    print("2.5  Diff throughput...")
    results.diff_throughput = bench_diff_throughput()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n--- Merge Throughput ---")
    for k, v in r.merge_throughput.items():
        print(f"  {k}: {v['nodes_per_sec']:.0f} nodes/s ({v['avg_sec']*1000:.1f}ms)")

    print("\n--- Merge by Conflict Rate ---")
    for k, v in r.merge_by_conflict_rate.items():
        print(f"  {k}: {v['avg_sec']*1000:.1f}ms, "
              f"agreed={v['properties_agreed']}, conflicted={v['properties_conflicted']}")

    print("\n--- Propagation Overhead (μs/call) ---")
    for k, v in r.propagation_overhead.items():
        methods = {m: v[f"{m}_us"] for m in ["multiply", "bayesian", "min", "dampened"]}
        print(f"  {k}: {methods}")

    print("\n--- Combination Comparison ---")
    for k, v in r.combination_comparison.items():
        scores = {m: v[f"{m}_score"] for m in ["average", "max", "noisy_or", "dempster_shafer"]}
        print(f"  {k}: {scores}")

    print("\n--- Diff Throughput ---")
    for k, v in r.diff_throughput.items():
        print(f"  {k}: {v['nodes_per_sec']:.0f} nodes/s")
