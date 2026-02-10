"""
Benchmark Domain 2: Multi-Agent Knowledge Graph Construction

Measures:
  - merge_graphs throughput at varying scale and conflict rates
  - Confidence propagation overhead per chain length
  - combine_sources comparison across methods
  - diff_graphs throughput
  All with stddev, 95% CI, and n=30 trials.
"""

from __future__ import annotations

import time
import json
import random
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
from bench_utils import timed_trials, timed_trials_us, DEFAULT_TRIALS


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
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Merge throughput: nodes per second at varying scale."""
    results = {}
    for n in node_counts:
        graphs = make_conflicting_graphs(n, n_sources, conflict_rate=0.3)
        stats = timed_trials(lambda: merge_graphs(graphs), n=n_trials)
        results[f"n={n}"] = {
            **stats.to_dict(),
            "nodes_per_sec": round(n / stats.mean, 1) if stats.mean > 0 else 0,
            "nodes_per_sec_ci95": [
                round(n / stats.ci95_high, 1) if stats.ci95_high > 0 else 0,
                round(n / stats.ci95_low, 1) if stats.ci95_low > 0 else 0,
            ],
            "total_input_nodes": n * n_sources,
        }
    return results


def bench_merge_by_conflict_rate(
    n_nodes: int = 100,
    n_sources: int = 3,
    rates: list[float] = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """How conflict rate affects merge time and report contents."""
    results = {}
    for rate in rates:
        graphs = make_conflicting_graphs(n_nodes, n_sources, conflict_rate=rate)

        # Capture report from one run for property counts
        _, sample_report = merge_graphs(graphs)

        stats = timed_trials(lambda: merge_graphs(graphs), n=n_trials)
        results[f"rate={rate}"] = {
            **stats.to_dict(),
            "properties_agreed": sample_report.properties_agreed,
            "properties_conflicted": sample_report.properties_conflicted,
            "nodes_merged": sample_report.nodes_merged,
        }
    return results


def bench_propagation_overhead(
    chain_lengths: list[int] = [2, 5, 10, 20, 50, 100],
    methods: list[str] = ["multiply", "bayesian", "min", "dampened"],
    inner_iterations: int = 1000,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Time per propagation call at varying chain length."""
    results = {}
    for length in chain_lengths:
        chain = [0.9] * length
        row: dict[str, Any] = {"chain_length": length}
        for method in methods:
            stats = timed_trials_us(
                lambda m=method: propagate_confidence(chain, method=m),
                inner_iterations=inner_iterations,
                n=n_trials,
            )
            row[f"{method}_us"] = round(stats.mean * 1_000_000, 2)
            row[f"{method}_std_us"] = round(stats.std * 1_000_000, 2)
            row[f"{method}_n"] = stats.n
        results[f"len={length}"] = row
    return results


def bench_combination_comparison(
    source_counts: list[int] = [2, 3, 5, 10, 20],
    methods: list[str] = ["average", "max", "noisy_or", "dempster_shafer"],
    inner_iterations: int = 1000,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare combine_sources methods: output score and timing."""
    rng = random.Random(42)

    results = {}
    for n in source_counts:
        scores = [round(rng.uniform(0.5, 0.95), 3) for _ in range(n)]
        row: dict[str, Any] = {"n_sources": n, "input_scores": scores}
        for method in methods:
            # Get the output score
            r = combine_sources(scores, method=method)
            row[f"{method}_score"] = round(r.score, 4)

            # Time it
            stats = timed_trials_us(
                lambda m=method: combine_sources(scores, method=m),
                inner_iterations=inner_iterations,
                n=n_trials,
            )
            row[f"{method}_us"] = round(stats.mean * 1_000_000, 2)
            row[f"{method}_std_us"] = round(stats.std * 1_000_000, 2)
        results[f"n={n}"] = row
    return results


def bench_diff_throughput(
    sizes: list[int] = [10, 100, 500, 1000],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """diff_graphs throughput at varying scale."""
    results = {}
    for n in sizes:
        graphs = make_conflicting_graphs(n, 2, conflict_rate=0.3)
        stats = timed_trials(
            lambda: diff_graphs(graphs[0], graphs[1]),
            n=n_trials,
        )
        results[f"n={n}"] = {
            **stats.to_dict(),
            "nodes_per_sec": round(n / stats.mean, 1) if stats.mean > 0 else 0,
            "nodes_per_sec_ci95": [
                round(n / stats.ci95_high, 1) if stats.ci95_high > 0 else 0,
                round(n / stats.ci95_low, 1) if stats.ci95_low > 0 else 0,
            ],
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
        print(f"  {k}: {v['nodes_per_sec']:.0f} nodes/s "
              f"(mean {v['mean_sec']*1000:.1f}ms ± {v['std_sec']*1000:.2f}ms, n={v['n_trials']})")

    print("\n--- Merge by Conflict Rate ---")
    for k, v in r.merge_by_conflict_rate.items():
        print(f"  {k}: {v['mean_sec']*1000:.1f}ms ± {v['std_sec']*1000:.2f}ms, "
              f"agreed={v['properties_agreed']}, conflicted={v['properties_conflicted']}")
