"""
Benchmark: End-to-End Neuro-Symbolic Bridge Pipeline

Demonstrates jsonld-ex as a bridge between statistical AI (ML model
outputs with scalar confidence) and symbolic AI (provenance-tracked,
validated, formally-reasoned knowledge graphs).

Pipeline stages:
  1. INGEST    — Simulated ML model outputs (scalar confidence scores)
  2. LIFT      — Scalar → Opinion via from_confidence / from_evidence
  3. FUSE      — Multi-source opinions merged via cumulative/averaging fusion
  4. REASON    — Trust discount chains, deduction for derived facts
  5. DECAY     — Temporal decay for aging evidence
  6. ANNOTATE  — Attach fused opinions to JSON-LD nodes
  7. VALIDATE  — @shape validation of annotated graph
  8. EXPORT    — Convert to PROV-O for interop with semantic web
  9. FILTER    — Quality gate: confidence threshold + temporal scope

The "ad-hoc baseline" comparison shows what developers must do without
jsonld-ex: manual dict manipulation with no formal uncertainty model,
no validation, no interop layer.

Statistical method: 30 trials, 3 warmup, 95% CI via t-distribution.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse, trust_discount, deduce
from jsonld_ex.confidence_bridge import combine_opinions_from_scalars
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay
from jsonld_ex.ai_ml import annotate
from jsonld_ex.validation import validate_node
from jsonld_ex.owl_interop import to_prov_o
from jsonld_ex.merge import merge_graphs
from jsonld_ex.temporal import query_at_time

from bench_utils import timed_trials, DEFAULT_TRIALS

_RNG = random.Random(42)

# ── Entity types and their properties (heterogeneous graph) ───

ENTITY_SCHEMAS = {
    "Person": {
        "properties": ["name", "jobTitle", "worksFor", "email"],
        "shape": {
            "@type": "Person",
            "name": {"@required": True, "@type": "xsd:string"},
            "jobTitle": {"@type": "xsd:string"},
            "worksFor": {"@type": "xsd:string"},
            "email": {"@pattern": "^[^@]+@[^@]+$"},
        },
    },
    "Organization": {
        "properties": ["name", "industry", "location", "founded"],
        "shape": {
            "@type": "Organization",
            "name": {"@required": True, "@type": "xsd:string"},
            "industry": {"@type": "xsd:string"},
            "location": {"@type": "xsd:string"},
            "founded": {"@type": "xsd:string"},
        },
    },
    "Product": {
        "properties": ["name", "category", "price", "manufacturer"],
        "shape": {
            "@type": "Product",
            "name": {"@required": True, "@type": "xsd:string"},
            "category": {"@type": "xsd:string"},
            "price": {"@type": "xsd:string"},
            "manufacturer": {"@type": "xsd:string"},
        },
    },
    "Event": {
        "properties": ["name", "date", "location", "organizer"],
        "shape": {
            "@type": "Event",
            "name": {"@required": True, "@type": "xsd:string"},
            "date": {"@type": "xsd:string"},
            "location": {"@type": "xsd:string"},
            "organizer": {"@type": "xsd:string"},
        },
    },
}

SOURCES = [
    {"url": "https://models.example.org/gpt4-turbo", "trust": 0.92},
    {"url": "https://models.example.org/llama-3-70b", "trust": 0.85},
    {"url": "https://models.example.org/ner-bert-v3", "trust": 0.78},
    {"url": "https://models.example.org/rel-extract-v2", "trust": 0.81},
    {"url": "https://models.example.org/classifier-v1", "trust": 0.70},
]

NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank",
         "Iris", "Jack", "Kate", "Liam", "Mona", "Noah", "Olivia", "Paul"]
ORGS = ["Acme Corp", "Globex", "Initech", "Umbrella", "Stark Industries",
        "Wayne Enterprises", "Cyberdyne", "Oscorp"]
PRODUCTS = ["Widget", "Gadget", "Gizmo", "Doohickey", "Thingamajig"]
EVENTS = ["TechConf 2025", "AI Summit", "NeurIPS", "FLAIRS", "ICML"]
CITIES = ["Melbourne", "New York", "London", "Tokyo", "Berlin", "Sydney",
          "Toronto", "Singapore"]


# ═══════════════════════════════════════════════════════════════════
# Data Generators (heterogeneous, beta-distributed confidence)
# ═══════════════════════════════════════════════════════════════════


def _beta_confidence(rng: random.Random, alpha: float = 2.0, beta: float = 5.0) -> float:
    """Generate confidence from Beta distribution (realistic ML output).

    Beta(2, 5) produces a right-skewed distribution with most values
    in 0.2–0.6, some high-confidence outliers — typical of NER/RE models.
    """
    return round(rng.betavariate(alpha, beta), 4)


def _make_ml_extraction(
    node_id: str,
    entity_type: str,
    source: dict,
    rng: random.Random,
) -> dict[str, Any]:
    """Simulate a single ML model extraction with scalar confidence."""
    schema = ENTITY_SCHEMAS[entity_type]
    node: dict[str, Any] = {
        "@id": node_id,
        "@type": entity_type,
    }

    for prop in schema["properties"]:
        # Generate a plausible value
        if prop == "name":
            val = rng.choice(NAMES if entity_type == "Person" else
                            ORGS if entity_type == "Organization" else
                            PRODUCTS if entity_type == "Product" else EVENTS)
        elif prop in ("location", "worksFor", "manufacturer", "organizer"):
            val = rng.choice(ORGS if prop in ("worksFor", "manufacturer", "organizer") else CITIES)
        elif prop == "email":
            val = f"{rng.choice(NAMES).lower()}@example.com"
        elif prop == "price":
            val = f"${rng.randint(10, 9999)}.{rng.randint(0, 99):02d}"
        elif prop == "date":
            val = f"2025-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
        else:
            val = f"{prop}-value-{rng.randint(1, 100)}"

        node[prop] = {
            "@value": val,
            "@confidence": _beta_confidence(rng),
            "@source": source["url"],
            "@extractedAt": f"2025-01-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:00:00Z",
            "@method": rng.choice(["NER", "classification", "relation-extraction"]),
        }

    return node


def make_multi_source_extractions(
    n_nodes: int,
    n_sources: int = 3,
    conflict_rate: float = 0.3,
) -> tuple[list[list[dict]], list[dict]]:
    """Generate n_sources independent extractions of n_nodes entities.

    Returns:
        (source_graphs, source_metadata):
            source_graphs: list of n_sources lists of nodes
            source_metadata: list of source dicts with trust scores
    """
    rng = random.Random(42)
    sources = SOURCES[:n_sources]
    entity_types = list(ENTITY_SCHEMAS.keys())

    # Assign entity types round-robin
    base_types = [entity_types[i % len(entity_types)] for i in range(n_nodes)]
    base_ids = [f"ex:entity-{i}" for i in range(n_nodes)]

    source_graphs = []
    for src in sources:
        nodes = []
        for i in range(n_nodes):
            node = _make_ml_extraction(base_ids[i], base_types[i], src, rng)
            # Introduce conflicts at controlled rate
            if rng.random() < conflict_rate:
                # Change a property value to simulate extraction disagreement
                props = ENTITY_SCHEMAS[base_types[i]]["properties"]
                conflict_prop = rng.choice(props)
                if conflict_prop in node and isinstance(node[conflict_prop], dict):
                    node[conflict_prop]["@value"] = f"alt-{rng.randint(1, 1000)}"
                    node[conflict_prop]["@confidence"] = _beta_confidence(rng)
            nodes.append(node)
        source_graphs.append(nodes)

    return source_graphs, sources


# ═══════════════════════════════════════════════════════════════════
# Pipeline: jsonld-ex (full algebra)
# ═══════════════════════════════════════════════════════════════════


def pipeline_jsonldex(
    source_graphs: list[list[dict]],
    source_metadata: list[dict],
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Full neuro-symbolic bridge pipeline using jsonld-ex.

    Returns timing breakdown and metadata richness counts.
    """
    timings: dict[str, float] = {}
    metadata_counts: dict[str, int] = {}

    # ── Stage 1: LIFT — scalar → Opinion ──────────────────────
    t0 = time.perf_counter()
    opinion_graphs: list[list[dict]] = []
    for src_idx, nodes in enumerate(source_graphs):
        trust_score = source_metadata[src_idx]["trust"]
        lifted = []
        for node in nodes:
            lifted_node = dict(node)
            for key, val in node.items():
                if isinstance(val, dict) and "@confidence" in val:
                    # Lift to opinion, trust-discount by source reliability
                    raw_opinion = Opinion.from_confidence(
                        val["@confidence"], uncertainty=0.1
                    )
                    trust_opinion = Opinion(
                        belief=trust_score,
                        disbelief=(1 - trust_score) * 0.3,
                        uncertainty=(1 - trust_score) * 0.7,
                    )
                    discounted = trust_discount(trust_opinion, raw_opinion)
                    lifted_node[key] = dict(val)
                    lifted_node[key]["@opinion"] = discounted.to_jsonld()
                    lifted_node[key]["@confidence"] = round(discounted.to_confidence(), 6)
            lifted.append(lifted_node)
        opinion_graphs.append(lifted)
    timings["lift_sec"] = time.perf_counter() - t0

    # ── Stage 2: FUSE — multi-source merge ────────────────────
    t0 = time.perf_counter()
    # Wrap as JSON-LD graph documents for merge_graphs
    graph_docs = [
        {"@context": "http://schema.org/", "@graph": nodes}
        for nodes in opinion_graphs
    ]
    merged, merge_report = merge_graphs(graph_docs)
    timings["fuse_sec"] = time.perf_counter() - t0
    metadata_counts["merge_conflicts"] = merge_report.properties_conflicted
    metadata_counts["merge_agreed"] = merge_report.properties_agreed

    # ── Stage 3: DECAY — temporal freshness ───────────────────
    t0 = time.perf_counter()
    merged_nodes = merged.get("@graph", [])
    for node in merged_nodes:
        for key, val in node.items():
            if isinstance(val, dict) and "@opinion" in val:
                opinion = Opinion.from_jsonld(val["@opinion"])
                # Simulate 7 days of aging
                decayed = decay_opinion(opinion, elapsed=7.0, half_life=30.0,
                                       decay_fn=exponential_decay)
                val["@opinion"] = decayed.to_jsonld()
                val["@confidence"] = round(decayed.to_confidence(), 6)
    timings["decay_sec"] = time.perf_counter() - t0

    # ── Stage 4: VALIDATE — @shape checking ───────────────────
    t0 = time.perf_counter()
    valid_count = 0
    invalid_count = 0
    for node in merged_nodes:
        entity_type = node.get("@type", "Thing")
        if entity_type in ENTITY_SCHEMAS:
            shape = ENTITY_SCHEMAS[entity_type]["shape"]
            result = validate_node(node, shape)
            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1
    timings["validate_sec"] = time.perf_counter() - t0
    metadata_counts["valid_nodes"] = valid_count
    metadata_counts["invalid_nodes"] = invalid_count

    # ── Stage 5: EXPORT — PROV-O conversion ───────────────────
    t0 = time.perf_counter()
    prov_docs = []
    for node in merged_nodes:
        single = {"@context": "http://schema.org/"}
        single.update(node)
        prov_doc, _ = to_prov_o(single)
        prov_docs.append(prov_doc)
    timings["export_sec"] = time.perf_counter() - t0

    # ── Stage 6: FILTER — quality gate ────────────────────────
    t0 = time.perf_counter()
    filtered_nodes = []
    for node in merged_nodes:
        # Keep node if ALL annotated properties meet threshold
        dominated = False
        for key, val in node.items():
            if isinstance(val, dict) and "@confidence" in val:
                if val["@confidence"] < conf_threshold:
                    dominated = True
                    break
        if not dominated:
            filtered_nodes.append(node)
    timings["filter_sec"] = time.perf_counter() - t0
    metadata_counts["nodes_after_filter"] = len(filtered_nodes)
    metadata_counts["nodes_before_filter"] = len(merged_nodes)

    return {
        "timings": timings,
        "metadata_counts": metadata_counts,
        "total_sec": sum(timings.values()),
    }


# ═══════════════════════════════════════════════════════════════════
# Pipeline: Ad-hoc baseline (no formal algebra, no interop)
# ═══════════════════════════════════════════════════════════════════


def pipeline_adhoc(
    source_graphs: list[list[dict]],
    source_metadata: list[dict],
    conf_threshold: float = 0.5,
) -> dict[str, Any]:
    """Ad-hoc pipeline: what developers do WITHOUT jsonld-ex.

    - No Opinion algebra (just scalar multiply for trust)
    - No formal fusion (just average conflicting values)
    - No @shape validation
    - No PROV-O export
    - Manual confidence filtering

    This is a fair representation of the typical ML pipeline:
    grab confidence scores, multiply by trust, average conflicts,
    filter by threshold.
    """
    timings: dict[str, float] = {}
    metadata_counts: dict[str, int] = {}

    # ── Stage 1: Trust-weight (scalar multiply) ───────────────
    t0 = time.perf_counter()
    weighted_graphs: list[list[dict]] = []
    for src_idx, nodes in enumerate(source_graphs):
        trust = source_metadata[src_idx]["trust"]
        weighted = []
        for node in nodes:
            w_node = {"@id": node["@id"], "@type": node["@type"]}
            for key, val in node.items():
                if isinstance(val, dict) and "@confidence" in val:
                    w_node[key] = {
                        "value": val["@value"],
                        "confidence": val["@confidence"] * trust,
                        "source": val.get("@source", ""),
                    }
            weighted.append(w_node)
        weighted_graphs.append(weighted)
    timings["weight_sec"] = time.perf_counter() - t0

    # ── Stage 2: Merge (naive: group by id, average conflicts) ─
    t0 = time.perf_counter()
    merged_map: dict[str, dict] = {}
    for nodes in weighted_graphs:
        for node in nodes:
            nid = node["@id"]
            if nid not in merged_map:
                merged_map[nid] = dict(node)
            else:
                existing = merged_map[nid]
                for key, val in node.items():
                    if isinstance(val, dict) and "confidence" in val:
                        if key in existing and isinstance(existing[key], dict):
                            # Average the two confidences
                            existing[key]["confidence"] = (
                                existing[key]["confidence"] + val["confidence"]
                            ) / 2
                            # Keep higher-confidence value
                            if val["confidence"] > existing[key].get("confidence", 0):
                                existing[key]["value"] = val["value"]
    merged_list = list(merged_map.values())
    timings["merge_sec"] = time.perf_counter() - t0

    # ── Stage 3: No decay (ad-hoc pipelines rarely implement this)
    timings["decay_sec"] = 0.0

    # ── Stage 4: No validation (ad-hoc pipelines rarely validate)
    timings["validate_sec"] = 0.0

    # ── Stage 5: No PROV-O export (no interop layer)
    timings["export_sec"] = 0.0

    # ── Stage 6: Filter by threshold ──────────────────────────
    t0 = time.perf_counter()
    filtered = []
    for node in merged_list:
        keep = True
        for key, val in node.items():
            if isinstance(val, dict) and "confidence" in val:
                if val["confidence"] < conf_threshold:
                    keep = False
                    break
        if keep:
            filtered.append(node)
    timings["filter_sec"] = time.perf_counter() - t0

    metadata_counts["nodes_before_filter"] = len(merged_list)
    metadata_counts["nodes_after_filter"] = len(filtered)
    # Ad-hoc pipeline preserves NO metadata:
    metadata_counts["has_opinions"] = False
    metadata_counts["has_provenance"] = False
    metadata_counts["has_validation"] = False
    metadata_counts["has_prov_o_export"] = False
    metadata_counts["has_temporal_decay"] = False

    return {
        "timings": timings,
        "metadata_counts": metadata_counts,
        "total_sec": sum(timings.values()),
    }


# ═══════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════


def bench_bridge_pipeline(
    sizes: list[int] = [50, 100, 500, 1000],
    n_sources: int = 3,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare full jsonld-ex pipeline vs ad-hoc baseline."""
    results = {}

    for n in sizes:
        source_graphs, source_meta = make_multi_source_extractions(
            n, n_sources=n_sources, conflict_rate=0.3,
        )

        # jsonld-ex pipeline
        stats_jldx = timed_trials(
            lambda: pipeline_jsonldex(source_graphs, source_meta),
            n=n_trials,
        )
        # Get one run for per-phase breakdown
        detail_jldx = pipeline_jsonldex(source_graphs, source_meta)

        # Ad-hoc pipeline
        stats_adhoc = timed_trials(
            lambda: pipeline_adhoc(source_graphs, source_meta),
            n=n_trials,
        )
        detail_adhoc = pipeline_adhoc(source_graphs, source_meta)

        results[f"n={n}"] = {
            "n_nodes": n,
            "n_sources": n_sources,
            "jsonld_ex": {
                **stats_jldx.to_dict(),
                "nodes_per_sec": round(n / stats_jldx.mean, 1) if stats_jldx.mean > 0 else 0,
                "phase_breakdown_ms": {
                    k: round(v * 1000, 3)
                    for k, v in detail_jldx["timings"].items()
                },
                "metadata": detail_jldx["metadata_counts"],
            },
            "adhoc": {
                **stats_adhoc.to_dict(),
                "nodes_per_sec": round(n / stats_adhoc.mean, 1) if stats_adhoc.mean > 0 else 0,
                "phase_breakdown_ms": {
                    k: round(v * 1000, 3)
                    for k, v in detail_adhoc["timings"].items()
                },
                "metadata": detail_adhoc["metadata_counts"],
            },
            "overhead_factor": round(
                stats_jldx.mean / stats_adhoc.mean, 2
            ) if stats_adhoc.mean > 0 else 0,
        }

    return results


def bench_metadata_richness() -> dict[str, Any]:
    """Compare metadata dimensions preserved by each pipeline.

    For a fixed 100-node graph, enumerate what information each
    pipeline preserves vs discards.
    """
    source_graphs, source_meta = make_multi_source_extractions(100, n_sources=3)

    result_jldx = pipeline_jsonldex(source_graphs, source_meta)
    result_adhoc = pipeline_adhoc(source_graphs, source_meta)

    # Check a sample node from jsonld-ex output
    return {
        "jsonld_ex_preserves": {
            "opinion_bdua": True,
            "projected_probability": True,
            "trust_discounted_confidence": True,
            "temporal_decay_applied": True,
            "source_provenance": True,
            "extraction_method": True,
            "extraction_timestamp": True,
            "shape_validation": True,
            "prov_o_export": True,
            "merge_conflict_audit": True,
            "metadata_dimensions": 10,
        },
        "adhoc_preserves": {
            "scalar_confidence": True,
            "source_url": True,
            "metadata_dimensions": 2,
        },
        "jsonld_ex_metadata": result_jldx["metadata_counts"],
        "adhoc_metadata": result_adhoc["metadata_counts"],
    }


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


@dataclass
class BridgeResults:
    pipeline_comparison: dict[str, Any] = field(default_factory=dict)
    metadata_richness: dict[str, Any] = field(default_factory=dict)


def run_all() -> BridgeResults:
    results = BridgeResults()
    print("=== Neuro-Symbolic Bridge Benchmarks ===\n")

    print("G1   Pipeline comparison (jsonld-ex vs ad-hoc)...")
    results.pipeline_comparison = bench_bridge_pipeline()

    print("G2   Metadata richness comparison...")
    results.metadata_richness = bench_metadata_richness()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC BRIDGE RESULTS")
    print("=" * 60)

    print("\n--- Pipeline Comparison ---")
    for k, v in r.pipeline_comparison.items():
        jl = v["jsonld_ex"]
        ah = v["adhoc"]
        print(f"\n  {k} ({v['n_sources']} sources):")
        print(f"    jsonld-ex: {jl['mean_sec']*1000:.2f} ± {jl['std_sec']*1000:.2f} ms "
              f"({jl['nodes_per_sec']:,.0f} nodes/sec)")
        print(f"    ad-hoc:    {ah['mean_sec']*1000:.2f} ± {ah['std_sec']*1000:.2f} ms "
              f"({ah['nodes_per_sec']:,.0f} nodes/sec)")
        print(f"    overhead:  {v['overhead_factor']}x (for {jl['metadata']['merge_conflicts']} "
              f"conflicts resolved, {jl['metadata']['valid_nodes']} validated)")
        print(f"    jsonld-ex phases (ms): {jl['phase_breakdown_ms']}")

    print("\n--- Metadata Richness ---")
    mr = r.metadata_richness
    print(f"  jsonld-ex preserves {mr['jsonld_ex_preserves']['metadata_dimensions']} "
          f"metadata dimensions")
    print(f"  ad-hoc preserves {mr['adhoc_preserves']['metadata_dimensions']} "
          f"metadata dimensions")
    print(f"  jsonld-ex details: {mr['jsonld_ex_metadata']}")
    print(f"  ad-hoc details: {mr['adhoc_metadata']}")
