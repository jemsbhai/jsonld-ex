"""
EN4.1 / EN4.3 / EN4.4: Scalability Benchmarks for jsonld-ex

Standalone script measuring wall-clock time and peak memory (tracemalloc)
for core operations at 1K, 10K, 100K, and 1M node scales.

Operations tested:
  S1: annotate_batch   -- annotate N raw values with provenance metadata
  S2: filter_by_confidence_batch -- filter N annotated nodes by threshold
  S3: merge_graphs     -- merge two N-node annotated graphs
  S4: validate_batch   -- validate N annotated nodes against a shape

Also includes EN4.4: per-item vs batch API overhead comparison at 10K items.

Usage:
    cd benchmarks
    python bench_scaling.py              # full run (1K-1M)
    python bench_scaling.py --quick      # quick run (1K-100K only)

Results saved to benchmarks/results/scaling_*.json and scaling_*.md

Statistical method: timed_trials with t-distribution 95% CI.
Trial counts scale inversely with N to keep total runtime reasonable:
  1K: 30 trials, 10K: 20 trials, 100K: 10 trials, 1M: 5 trials.
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Ensure benchmarks/ is on path for bench_utils
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from bench_utils import timed_trials, TrialStats

from jsonld_ex.ai_ml import annotate
from jsonld_ex.batch import annotate_batch, validate_batch, filter_by_confidence_batch
from jsonld_ex.merge import merge_graphs
from jsonld_ex.validation import validate_node


# ======================================================================
# Configuration
# ======================================================================

SCALES = [1_000, 10_000, 100_000, 1_000_000]
QUICK_SCALES = [1_000, 10_000, 100_000]

# Trials per scale -- fewer at larger N to keep runtime sane
TRIALS_BY_SCALE = {
    1_000: 30,
    10_000: 20,
    100_000: 10,
    1_000_000: 5,
}

WARMUP_BY_SCALE = {
    1_000: 3,
    10_000: 2,
    100_000: 1,
    1_000_000: 0,  # too expensive for warmup at 1M
}

PERSON_SHAPE = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string"},
    "worksFor": {"@type": "xsd:string"},
    "location": {"@type": "xsd:string"},
}

NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
         "Ivan", "Judy", "Karl", "Lily", "Mike", "Nina", "Oscar", "Paula"]
ORGS = ["Acme Corp", "Globex", "Initech", "Umbrella", "Stark Industries",
        "Wayne Enterprises", "Cyberdyne", "Aperture Science"]
CITIES = ["Melbourne", "New York", "London", "Tokyo", "Berlin", "Sydney",
          "Paris", "Toronto", "Seoul", "Mumbai"]


# ======================================================================
# Data generation
# ======================================================================


def make_raw_values(n: int, seed: int = 42) -> list[str]:
    """Generate n raw string values for annotation benchmarks."""
    rng = random.Random(seed)
    return [rng.choice(NAMES) for _ in range(n)]


def make_annotated_nodes(n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Generate n annotated Person nodes for filter/validate benchmarks."""
    rng = random.Random(seed)
    nodes = []
    for i in range(n):
        conf = round(rng.uniform(0.1, 0.99), 3)
        nodes.append({
            "@id": f"ex:person-{i}",
            "@type": "Person",
            "name": {
                "@value": rng.choice(NAMES),
                "@confidence": conf,
                "@source": "https://models.example.org/ner-v4",
            },
            "worksFor": {
                "@value": rng.choice(ORGS),
                "@confidence": round(rng.uniform(0.2, 0.95), 3),
                "@source": "https://models.example.org/rel-extract-v2",
            },
            "location": {
                "@value": rng.choice(CITIES),
                "@confidence": round(rng.uniform(0.3, 0.98), 3),
                "@source": "https://models.example.org/classifier-v1",
            },
        })
    return nodes


def make_graph_document(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap nodes in a JSON-LD @graph document."""
    return {
        "@context": "http://schema.org/",
        "@graph": nodes,
    }


def make_conflicting_graph(n: int, source_idx: int, seed: int = 42) -> dict[str, Any]:
    """Generate an annotated graph with a specific source tag.

    Two graphs with the same node IDs but different sources simulate
    a multi-source merge scenario.
    """
    rng = random.Random(seed + source_idx * 7919)
    nodes = []
    for i in range(n):
        nodes.append({
            "@id": f"ex:person-{i}",
            "@type": "Person",
            "name": {
                "@value": rng.choice(NAMES),
                "@confidence": round(rng.uniform(0.3, 0.99), 3),
                "@source": f"https://models.example.org/source-{source_idx}",
            },
            "worksFor": {
                "@value": rng.choice(ORGS),
                "@confidence": round(rng.uniform(0.2, 0.95), 3),
                "@source": f"https://models.example.org/source-{source_idx}",
            },
            "location": {
                "@value": rng.choice(CITIES),
                "@confidence": round(rng.uniform(0.3, 0.98), 3),
                "@source": f"https://models.example.org/source-{source_idx}",
            },
        })
    return {
        "@context": "http://schema.org/",
        "@graph": nodes,
    }


# ======================================================================
# Memory-profiled timing
# ======================================================================


@dataclass
class ScaleResult:
    """Result for one operation at one scale."""
    operation: str
    n: int
    mean_sec: float
    std_sec: float
    ci95_low_sec: float
    ci95_high_sec: float
    min_sec: float
    max_sec: float
    n_trials: int
    peak_memory_mb: float
    throughput_nodes_per_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "n": self.n,
            "mean_sec": round(self.mean_sec, 6),
            "std_sec": round(self.std_sec, 6),
            "ci95_low_sec": round(self.ci95_low_sec, 6),
            "ci95_high_sec": round(self.ci95_high_sec, 6),
            "min_sec": round(self.min_sec, 6),
            "max_sec": round(self.max_sec, 6),
            "n_trials": self.n_trials,
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "throughput_nodes_per_sec": round(self.throughput_nodes_per_sec, 0),
        }


def measure_with_memory(
    fn: Callable[[], Any],
    n: int,
    operation: str,
    trials: int,
    warmup: int,
) -> ScaleResult:
    """Run timed_trials AND measure peak memory via tracemalloc."""
    # Force GC before measurement
    gc.collect()

    # Measure peak memory on a single run
    tracemalloc.start()
    fn()
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    # Force GC again before timing
    gc.collect()

    # Timed trials
    stats = timed_trials(fn, n=trials, warmup=warmup)

    throughput = n / stats.mean if stats.mean > 0 else 0

    return ScaleResult(
        operation=operation,
        n=n,
        mean_sec=stats.mean,
        std_sec=stats.std,
        ci95_low_sec=stats.ci95_low,
        ci95_high_sec=stats.ci95_high,
        min_sec=stats.min,
        max_sec=stats.max,
        n_trials=stats.n,
        peak_memory_mb=peak_mb,
        throughput_nodes_per_sec=throughput,
    )


# ======================================================================
# S1: Annotation scaling
# ======================================================================


def bench_annotate_scaling(n: int, trials: int, warmup: int) -> ScaleResult:
    """Measure annotate_batch throughput at scale n."""
    raw_values = make_raw_values(n)

    def do_annotate():
        return annotate_batch(
            raw_values,
            confidence=0.85,
            source="https://models.example.org/ner-v4",
            method="NER",
        )

    return measure_with_memory(do_annotate, n, "annotate_batch", trials, warmup)


# ======================================================================
# S2: Confidence filtering scaling
# ======================================================================


def bench_filter_scaling(n: int, trials: int, warmup: int) -> ScaleResult:
    """Measure filter_by_confidence_batch throughput at scale n."""
    nodes = make_annotated_nodes(n)

    def do_filter():
        return filter_by_confidence_batch(nodes, "name", min_confidence=0.5)

    return measure_with_memory(do_filter, n, "filter_by_confidence", trials, warmup)


# ======================================================================
# S3: Graph merge scaling
# ======================================================================


def bench_merge_scaling(n: int, trials: int, warmup: int) -> ScaleResult:
    """Measure merge_graphs throughput at scale n (two n-node graphs)."""
    graph_a = make_conflicting_graph(n, source_idx=0)
    graph_b = make_conflicting_graph(n, source_idx=1)

    def do_merge():
        return merge_graphs([graph_a, graph_b])

    return measure_with_memory(do_merge, n, "merge_graphs", trials, warmup)


# ======================================================================
# S4: Validation scaling
# ======================================================================


def bench_validate_scaling(n: int, trials: int, warmup: int) -> ScaleResult:
    """Measure validate_batch throughput at scale n."""
    nodes = make_annotated_nodes(n)

    def do_validate():
        return validate_batch(nodes, PERSON_SHAPE)

    return measure_with_memory(do_validate, n, "validate_batch", trials, warmup)


# ======================================================================
# EN4.4: Per-item vs batch API overhead
# ======================================================================


def bench_batch_overhead(n: int = 10_000) -> dict[str, Any]:
    """Compare per-item API calls vs batch API for n items.

    Three operations: annotate, validate, filter_by_confidence.
    """
    results = {}

    # --- Annotate ---
    raw_values = make_raw_values(n)
    nodes = make_annotated_nodes(n)

    # Per-item annotate
    stats_per = timed_trials(
        lambda: [annotate(v, confidence=0.85, source="src", method="NER") for v in raw_values],
        n=10,
        warmup=1,
    )

    # Batch annotate
    stats_batch = timed_trials(
        lambda: annotate_batch(raw_values, confidence=0.85, source="src", method="NER"),
        n=10,
        warmup=1,
    )

    results["annotate"] = {
        "n": n,
        "per_item_sec": round(stats_per.mean, 6),
        "batch_sec": round(stats_batch.mean, 6),
        "speedup": round(stats_per.mean / stats_batch.mean, 2) if stats_batch.mean > 0 else 0,
    }

    # --- Validate ---
    stats_per = timed_trials(
        lambda: [validate_node(node, PERSON_SHAPE) for node in nodes],
        n=10,
        warmup=1,
    )

    stats_batch = timed_trials(
        lambda: validate_batch(nodes, PERSON_SHAPE),
        n=10,
        warmup=1,
    )

    results["validate"] = {
        "n": n,
        "per_item_sec": round(stats_per.mean, 6),
        "batch_sec": round(stats_batch.mean, 6),
        "speedup": round(stats_per.mean / stats_batch.mean, 2) if stats_batch.mean > 0 else 0,
    }

    # --- Filter ---
    stats_per = timed_trials(
        lambda: [node for node in nodes
                 if (node.get("name", {}).get("@confidence", 0) or 0) >= 0.5],
        n=10,
        warmup=1,
    )

    stats_batch = timed_trials(
        lambda: filter_by_confidence_batch(nodes, "name", min_confidence=0.5),
        n=10,
        warmup=1,
    )

    results["filter"] = {
        "n": n,
        "per_item_sec": round(stats_per.mean, 6),
        "batch_sec": round(stats_batch.mean, 6),
        "speedup": round(stats_per.mean / stats_batch.mean, 2) if stats_batch.mean > 0 else 0,
    }

    return results


# ======================================================================
# Main runner
# ======================================================================


def run_scaling(scales: list[int]) -> dict[str, Any]:
    """Run all scaling benchmarks with checkpointing."""
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

    all_results: dict[str, list[dict]] = {
        "annotate_batch": [],
        "filter_by_confidence": [],
        "merge_graphs": [],
        "validate_batch": [],
    }

    total_ops = len(scales) * 4
    completed = 0

    for n in scales:
        trials = TRIALS_BY_SCALE.get(n, 10)
        warmup = WARMUP_BY_SCALE.get(n, 1)

        print(f"\n{'='*60}")
        print(f"  Scale: N = {n:,} nodes  ({trials} trials, {warmup} warmup)")
        print(f"{'='*60}")

        # S1: Annotate
        completed += 1
        print(f"\n  [{completed}/{total_ops}] S1: annotate_batch (n={n:,})...")
        t0 = time.perf_counter()
        r = bench_annotate_scaling(n, trials, warmup)
        elapsed = time.perf_counter() - t0
        all_results["annotate_batch"].append(r.to_dict())
        print(f"    -> {r.mean_sec*1000:.1f}ms +/- {r.std_sec*1000:.1f}ms, "
              f"peak={r.peak_memory_mb:.1f}MB, "
              f"{r.throughput_nodes_per_sec:,.0f} nodes/sec  "
              f"[wall: {elapsed:.1f}s]")

        # S2: Filter
        completed += 1
        print(f"\n  [{completed}/{total_ops}] S2: filter_by_confidence (n={n:,})...")
        t0 = time.perf_counter()
        r = bench_filter_scaling(n, trials, warmup)
        elapsed = time.perf_counter() - t0
        all_results["filter_by_confidence"].append(r.to_dict())
        print(f"    -> {r.mean_sec*1000:.1f}ms +/- {r.std_sec*1000:.1f}ms, "
              f"peak={r.peak_memory_mb:.1f}MB, "
              f"{r.throughput_nodes_per_sec:,.0f} nodes/sec  "
              f"[wall: {elapsed:.1f}s]")

        # S3: Merge
        completed += 1
        print(f"\n  [{completed}/{total_ops}] S3: merge_graphs (n={n:,})...")
        t0 = time.perf_counter()
        r = bench_merge_scaling(n, trials, warmup)
        elapsed = time.perf_counter() - t0
        all_results["merge_graphs"].append(r.to_dict())
        print(f"    -> {r.mean_sec*1000:.1f}ms +/- {r.std_sec*1000:.1f}ms, "
              f"peak={r.peak_memory_mb:.1f}MB, "
              f"{r.throughput_nodes_per_sec:,.0f} nodes/sec  "
              f"[wall: {elapsed:.1f}s]")

        # S4: Validate
        completed += 1
        print(f"\n  [{completed}/{total_ops}] S4: validate_batch (n={n:,})...")
        t0 = time.perf_counter()
        r = bench_validate_scaling(n, trials, warmup)
        elapsed = time.perf_counter() - t0
        all_results["validate_batch"].append(r.to_dict())
        print(f"    -> {r.mean_sec*1000:.1f}ms +/- {r.std_sec*1000:.1f}ms, "
              f"peak={r.peak_memory_mb:.1f}MB, "
              f"{r.throughput_nodes_per_sec:,.0f} nodes/sec  "
              f"[wall: {elapsed:.1f}s]")

        # Checkpoint after each scale
        checkpoint = {
            "timestamp": ts,
            "completed_scales": [s for s in scales if s <= n],
            "scaling": all_results,
        }
        ckpt_path = out_dir / f"scaling_checkpoint_{ts}.json"
        with open(ckpt_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        print(f"\n  [checkpoint saved: {ckpt_path.name}]")

        # Force GC between scales
        gc.collect()

    # EN4.4: Batch overhead comparison
    print(f"\n{'='*60}")
    print(f"  EN4.4: Per-item vs Batch API overhead (n=10,000)")
    print(f"{'='*60}")
    batch_overhead = bench_batch_overhead(n=10_000)
    for op, v in batch_overhead.items():
        print(f"    {op}: per-item={v['per_item_sec']*1000:.1f}ms, "
              f"batch={v['batch_sec']*1000:.1f}ms, "
              f"speedup={v['speedup']:.2f}x")

    # Assemble final results
    final = {
        "timestamp": ts,
        "scales": scales,
        "scaling": all_results,
        "batch_overhead_en44": batch_overhead,
    }

    # Check scaling linearity
    linearity = {}
    for op, entries in all_results.items():
        if len(entries) >= 2:
            # Compare throughput at smallest and largest scale
            t_small = entries[0]["throughput_nodes_per_sec"]
            t_large = entries[-1]["throughput_nodes_per_sec"]
            ratio = t_large / t_small if t_small > 0 else 0
            linearity[op] = {
                "smallest_scale": entries[0]["n"],
                "largest_scale": entries[-1]["n"],
                "throughput_smallest": round(t_small, 0),
                "throughput_largest": round(t_large, 0),
                "throughput_ratio": round(ratio, 3),
                "is_linear": ratio > 0.5,  # throughput should stay within 2x
            }
    final["linearity_check"] = linearity

    return final


def save_results(results: dict[str, Any]) -> tuple[str, str]:
    """Save results as JSON and Markdown."""
    out_dir = _SCRIPT_DIR / "results"
    out_dir.mkdir(exist_ok=True)
    ts = results["timestamp"]

    # JSON
    json_path = out_dir / f"scaling_results_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown
    md_lines = [
        "# EN4.1 / EN4.3 Scalability Benchmarks",
        "",
        f"Generated: {ts}",
        "",
        "## Scaling Results",
        "",
    ]

    for op in ["annotate_batch", "filter_by_confidence", "merge_graphs", "validate_batch"]:
        entries = results["scaling"].get(op, [])
        if not entries:
            continue

        md_lines += [
            f"### {op}",
            "",
            "| N | Mean (ms) | Std (ms) | 95% CI (ms) | Peak Mem (MB) | Throughput (nodes/sec) |",
            "|---|-----------|----------|-------------|---------------|----------------------|",
        ]
        for e in entries:
            ci = f"[{e['ci95_low_sec']*1000:.1f}, {e['ci95_high_sec']*1000:.1f}]"
            md_lines.append(
                f"| {e['n']:,} | {e['mean_sec']*1000:.1f} | {e['std_sec']*1000:.1f} | "
                f"{ci} | {e['peak_memory_mb']:.1f} | {e['throughput_nodes_per_sec']:,.0f} |"
            )
        md_lines.append("")

    # Linearity
    md_lines += [
        "## Linearity Check",
        "",
        "| Operation | Smallest N | Largest N | Throughput (small) | Throughput (large) | Ratio | Linear? |",
        "|-----------|-----------|----------|-------------------|-------------------|-------|---------|",
    ]
    for op, lc in results.get("linearity_check", {}).items():
        md_lines.append(
            f"| {op} | {lc['smallest_scale']:,} | {lc['largest_scale']:,} | "
            f"{lc['throughput_smallest']:,.0f} | {lc['throughput_largest']:,.0f} | "
            f"{lc['throughput_ratio']:.3f} | {'YES' if lc['is_linear'] else 'NO'} |"
        )
    md_lines.append("")

    # EN4.4 batch overhead
    md_lines += [
        "## EN4.4: Batch vs Per-Item API Overhead (n=10,000)",
        "",
        "| Operation | Per-item (ms) | Batch (ms) | Speedup |",
        "|-----------|---------------|------------|---------|",
    ]
    for op, v in results.get("batch_overhead_en44", {}).items():
        md_lines.append(
            f"| {op} | {v['per_item_sec']*1000:.1f} | {v['batch_sec']*1000:.1f} | "
            f"{v['speedup']:.2f}x |"
        )
    md_lines.append("")

    # Analysis
    md_lines += [
        "## Analysis",
        "",
    ]

    # Data-driven analysis
    lin = results.get("linearity_check", {})
    all_linear = all(v.get("is_linear", False) for v in lin.values())
    md_lines.append(
        f"All operations {'demonstrate' if all_linear else 'do NOT all demonstrate'} "
        f"linear scaling from {results['scales'][0]:,} to {results['scales'][-1]:,} nodes "
        f"(throughput ratio > 0.5 between smallest and largest scale)."
    )
    md_lines.append("")

    for op, lc in lin.items():
        md_lines.append(
            f"- **{op}**: {lc['throughput_smallest']:,.0f} -> {lc['throughput_largest']:,.0f} "
            f"nodes/sec (ratio {lc['throughput_ratio']:.3f})"
        )
    md_lines.append("")

    bo = results.get("batch_overhead_en44", {})
    if bo:
        speedups = [v["speedup"] for v in bo.values()]
        avg_speedup = sum(speedups) / len(speedups) if speedups else 0
        md_lines.append(
            f"Batch API provides {avg_speedup:.2f}x average speedup over per-item calls at n=10,000."
        )
        md_lines.append("")

    md_path = out_dir / f"scaling_results_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return str(json_path), str(md_path)


# ======================================================================
# Entry point
# ======================================================================


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    scales = QUICK_SCALES if quick else SCALES

    print("=" * 60)
    print("  EN4.1 / EN4.3 / EN4.4 SCALABILITY BENCHMARKS")
    print(f"  Scales: {', '.join(f'{s:,}' for s in scales)}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print("=" * 60)

    start_time = time.perf_counter()
    results = run_scaling(scales)
    total_time = time.perf_counter() - start_time

    json_path, md_path = save_results(results)

    print(f"\n{'='*60}")
    print(f"  COMPLETE in {total_time:.1f}s")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'='*60}")
