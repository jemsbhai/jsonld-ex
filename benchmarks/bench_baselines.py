"""
Benchmark: Baseline Comparisons Against Existing Tools

Compares jsonld-ex against established alternatives for the same tasks:
  1. PROV-O provenance construction: rdflib manual triples vs jsonld-ex annotate
  2. SHACL validation: pyshacl vs jsonld-ex validate_node
  3. Graph merge: rdflib Graph union vs jsonld-ex merge_graphs
  4. Temporal query: SPARQL via rdflib vs jsonld-ex query_at_time

All measurements use n=30 trials with warmup, reporting mean ± stddev and 95% CI.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

# ── jsonld-ex imports ────────────────────────────────────────

from jsonld_ex import (
    annotate,
    to_prov_o,
    from_prov_o,
    validate_node,
    merge_graphs,
    query_at_time,
    shape_to_shacl,
)

from data_generators import (
    make_annotated_graph,
    make_annotated_person,
    make_conflicting_graphs,
    make_temporal_graph,
    make_sensor_batch,
)
from bench_utils import timed_trials, DEFAULT_TRIALS

# ── rdflib / pyshacl imports ─────────────────────────────────

import rdflib
from rdflib import Graph, Literal, URIRef, Namespace, BNode, RDF, XSD
from rdflib.namespace import PROV, RDFS

try:
    from pyshacl import validate as pyshacl_validate
    HAS_PYSHACL = True
except ImportError:
    HAS_PYSHACL = False
    print("WARNING: pyshacl not installed — SHACL baselines will be skipped")


# ── Namespaces ───────────────────────────────────────────────

SCHEMA = Namespace("http://schema.org/")
EX = Namespace("http://example.org/")
JLDX = Namespace("http://jsonld-ex.org/")


# ═══════════════════════════════════════════════════════════════
# Comparison 1: PROV-O Provenance Construction
# ═══════════════════════════════════════════════════════════════


def _build_prov_o_rdflib(nodes: list[dict]) -> Graph:
    """Build PROV-O provenance graph using rdflib triples (the manual way).

    For each annotated node, creates the full PROV-O structure:
      - prov:Entity for the node
      - prov:Activity for the extraction
      - prov:SoftwareAgent for the model
      - prov:wasGeneratedBy, prov:wasAssociatedWith, etc.
      - prov:value for the confidence score
    This is what developers must do WITHOUT jsonld-ex.
    """
    g = Graph()
    g.bind("prov", PROV)
    g.bind("schema", SCHEMA)
    g.bind("ex", EX)

    for node in nodes:
        node_uri = URIRef(f"http://example.org/{node['@id']}")
        g.add((node_uri, RDF.type, SCHEMA.Person))

        for prop_name in ("name", "worksFor", "location"):
            if prop_name not in node or not isinstance(node[prop_name], dict):
                continue
            prop_data = node[prop_name]

            # The value itself
            g.add((node_uri, SCHEMA[prop_name],
                   Literal(prop_data["@value"])))

            # PROV-O provenance structure
            entity_bn = BNode()
            g.add((entity_bn, RDF.type, PROV.Entity))
            g.add((entity_bn, PROV.value,
                   Literal(prop_data["@value"])))

            if "@confidence" in prop_data:
                g.add((entity_bn, JLDX.confidence,
                       Literal(prop_data["@confidence"], datatype=XSD.double)))

            if "@source" in prop_data:
                agent_uri = URIRef(prop_data["@source"])
                g.add((agent_uri, RDF.type, PROV.SoftwareAgent))

                activity_bn = BNode()
                g.add((activity_bn, RDF.type, PROV.Activity))
                g.add((activity_bn, PROV.wasAssociatedWith, agent_uri))
                g.add((entity_bn, PROV.wasGeneratedBy, activity_bn))

                if "@extractedAt" in prop_data:
                    g.add((activity_bn, PROV.endedAtTime,
                           Literal(prop_data["@extractedAt"],
                                   datatype=XSD.dateTime)))

            if "@method" in prop_data:
                g.add((entity_bn, JLDX.method,
                       Literal(prop_data["@method"])))

            # Link entity to node
            g.add((node_uri, PROV.qualifiedGeneration, entity_bn))

    return g


def _build_prov_o_jsonldex(nodes: list[dict]) -> list:
    """Build PROV-O using jsonld-ex: annotate values, then convert."""
    results = []
    for node in nodes:
        single = {"@context": "http://schema.org/"}
        single.update(node)
        prov_doc, _ = to_prov_o(single)
        results.append(prov_doc)
    return results


def bench_prov_o_construction(
    sizes: list[int] = [10, 100, 500],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare PROV-O construction: rdflib manual vs jsonld-ex."""
    results = {}
    for n in sizes:
        doc = make_annotated_graph(n)
        nodes = doc["@graph"]

        # rdflib baseline
        stats_rdflib = timed_trials(
            lambda: _build_prov_o_rdflib(nodes),
            n=n_trials,
        )

        # jsonld-ex
        stats_jldx = timed_trials(
            lambda: _build_prov_o_jsonldex(nodes),
            n=n_trials,
        )

        # Lines of code comparison (approximate)
        # rdflib: ~30 lines per property × 3 props = ~90 lines of manual triple code
        # jsonld-ex: annotate() + to_prov_o() = ~3 lines

        speedup = stats_rdflib.mean / stats_jldx.mean if stats_jldx.mean > 0 else 0

        results[f"n={n}"] = {
            "rdflib": {
                **stats_rdflib.to_dict(),
                "nodes_per_sec": round(n / stats_rdflib.mean, 1) if stats_rdflib.mean > 0 else 0,
            },
            "jsonld_ex": {
                **stats_jldx.to_dict(),
                "nodes_per_sec": round(n / stats_jldx.mean, 1) if stats_jldx.mean > 0 else 0,
            },
            "speedup": round(speedup, 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Comparison 2: SHACL Validation
# ═══════════════════════════════════════════════════════════════


SENSOR_SHAPE_JSONLDEX = {
    "@type": "SensorReading",
    "value": {"@required": True, "@type": "xsd:double"},
    "unit": {"@required": True, "@type": "xsd:string"},
    "axis": {"@type": "xsd:string"},
}

# Equivalent SHACL shape graph for pyshacl
SHACL_SHAPE_TTL = """
@prefix sh: <http://www.w3.org/ns/shacl#> .
@prefix schema: <http://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

schema:SensorReadingShape
    a sh:NodeShape ;
    sh:targetClass schema:SensorReading ;
    sh:property [
        sh:path schema:value ;
        sh:minCount 1 ;
        sh:datatype xsd:double ;
    ] ;
    sh:property [
        sh:path schema:unit ;
        sh:minCount 1 ;
        sh:datatype xsd:string ;
    ] ;
    sh:property [
        sh:path schema:axis ;
        sh:datatype xsd:string ;
    ] .
"""


def _sensor_to_rdf_graph(readings: list[dict], n: int) -> Graph:
    """Convert sensor readings to an rdflib Graph for pyshacl validation."""
    g = Graph()
    g.bind("schema", SCHEMA)
    for i, r in enumerate(readings[:n]):
        node = URIRef(r.get("@id", f"urn:sensor:s{i}"))
        g.add((node, RDF.type, SCHEMA.SensorReading))
        val = r.get("value", {})
        raw_val = val.get("@value", val) if isinstance(val, dict) else val
        g.add((node, SCHEMA.value, Literal(raw_val, datatype=XSD.double)))
        g.add((node, SCHEMA.unit, Literal(r.get("unit", ""))))
        if "axis" in r:
            g.add((node, SCHEMA.axis, Literal(r["axis"])))
    return g


def bench_shacl_validation(
    sizes: list[int] = [10, 50, 100],
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare validation: pyshacl vs jsonld-ex validate_node."""
    if not HAS_PYSHACL:
        return {"skipped": "pyshacl not installed"}

    # Parse SHACL shape once
    shacl_graph = Graph()
    shacl_graph.parse(data=SHACL_SHAPE_TTL, format="turtle")

    results = {}
    for n in sizes:
        readings = make_sensor_batch(n)

        # pyshacl baseline
        data_graph = _sensor_to_rdf_graph(readings, n)
        stats_pyshacl = timed_trials(
            lambda: pyshacl_validate(data_graph, shacl_graph=shacl_graph,
                                      abort_on_first=False),
            n=n_trials,
        )

        # jsonld-ex
        def do_jldx_validate():
            for r in readings[:n]:
                validate_node(r, SENSOR_SHAPE_JSONLDEX)

        stats_jldx = timed_trials(do_jldx_validate, n=n_trials)

        speedup = stats_pyshacl.mean / stats_jldx.mean if stats_jldx.mean > 0 else 0

        results[f"n={n}"] = {
            "pyshacl": {
                **stats_pyshacl.to_dict(),
                "nodes_per_sec": round(n / stats_pyshacl.mean, 1) if stats_pyshacl.mean > 0 else 0,
            },
            "jsonld_ex": {
                **stats_jldx.to_dict(),
                "nodes_per_sec": round(n / stats_jldx.mean, 1) if stats_jldx.mean > 0 else 0,
            },
            "speedup": round(speedup, 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Comparison 3: Graph Merge
# ═══════════════════════════════════════════════════════════════


def _graphs_to_rdflib(graphs: list[dict]) -> list[Graph]:
    """Convert jsonld-ex style graphs to rdflib Graphs."""
    rdf_graphs = []
    for doc in graphs:
        g = Graph()
        g.bind("schema", SCHEMA)
        for node in doc.get("@graph", []):
            node_uri = URIRef(f"http://example.org/{node['@id']}")
            g.add((node_uri, RDF.type, SCHEMA[node.get("@type", "Thing")]))
            for prop in ("name", "worksFor", "location"):
                if prop in node:
                    val = node[prop]
                    raw = val.get("@value", val) if isinstance(val, dict) else val
                    g.add((node_uri, SCHEMA[prop], Literal(raw)))
                    # Note: rdflib union LOSES confidence, source, method metadata
        rdf_graphs.append(g)
    return rdf_graphs


def _rdflib_merge_with_conflict_resolution(rdf_graphs: list[Graph]) -> tuple[Graph, dict]:
    """Merge graphs using rdflib with conflict detection and resolution.

    This is the fair comparison: what a developer would actually build
    with rdflib to get equivalent functionality to jsonld-ex merge_graphs.

    Steps:
      1. Union all triples
      2. Detect conflicts (same subject+predicate, different objects)
      3. Resolve conflicts by picking highest confidence
      4. Build clean merged graph
    """
    # Step 1: Union
    merged = Graph()
    merged.bind("schema", SCHEMA)
    merged.bind("jldx", JLDX)
    for g in rdf_graphs:
        merged += g

    # Step 2: Detect conflicts via SPARQL
    # Find subject-predicate pairs with multiple distinct object values
    conflict_query = """
    SELECT ?s ?p (COUNT(DISTINCT ?o) AS ?cnt)
    WHERE {
        ?s ?p ?o .
        FILTER(?p != rdf:type)
    }
    GROUP BY ?s ?p
    HAVING (COUNT(DISTINCT ?o) > 1)
    """
    conflicts = list(merged.query(conflict_query))

    # Step 3: Resolve each conflict (pick highest confidence if available,
    # otherwise keep first value)
    report = {"conflicts_detected": len(conflicts), "conflicts_resolved": 0}
    for row in conflicts:
        s, p = row[0], row[1]
        # Get all objects for this s-p pair
        objects = list(merged.objects(s, p))
        if len(objects) <= 1:
            continue

        # Try to find confidence annotations (would need reification in practice)
        # In a real system this requires even more SPARQL. Here we simulate
        # the minimum: keep first value, remove extras.
        keeper = objects[0]
        for obj in objects[1:]:
            merged.remove((s, p, obj))
        report["conflicts_resolved"] += 1

    return merged, report


def bench_graph_merge(
    sizes: list[int] = [10, 100, 500],
    n_sources: int = 3,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare merge: rdflib (union + conflict resolution) vs jsonld-ex merge_graphs."""
    results = {}
    for n in sizes:
        graphs = make_conflicting_graphs(n, n_sources, conflict_rate=0.3)

        # rdflib baseline: union + SPARQL conflict detection + resolution
        rdf_graphs = _graphs_to_rdflib(graphs)
        stats_rdflib = timed_trials(
            lambda: _rdflib_merge_with_conflict_resolution(rdf_graphs),
            n=n_trials,
        )

        # jsonld-ex (confidence-aware conflict resolution, provenance tracking)
        stats_jldx = timed_trials(
            lambda: merge_graphs(graphs),
            n=n_trials,
        )

        speedup = stats_rdflib.mean / stats_jldx.mean if stats_jldx.mean > 0 else 0

        results[f"n={n}"] = {
            "rdflib_merge": {
                **stats_rdflib.to_dict(),
                "nodes_per_sec": round(n / stats_rdflib.mean, 1) if stats_rdflib.mean > 0 else 0,
            },
            "jsonld_ex_merge": {
                **stats_jldx.to_dict(),
                "nodes_per_sec": round(n / stats_jldx.mean, 1) if stats_jldx.mean > 0 else 0,
            },
            "speedup": round(speedup, 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Comparison 4: Temporal Query
# ═══════════════════════════════════════════════════════════════


def _temporal_to_rdflib(nodes: list[dict]) -> Graph:
    """Convert temporal jsonld-ex nodes to an rdflib Graph with reification."""
    g = Graph()
    g.bind("schema", SCHEMA)
    g.bind("jldx", JLDX)

    for node in nodes:
        node_uri = URIRef(f"http://example.org/{node['@id']}")
        g.add((node_uri, RDF.type, SCHEMA[node.get("@type", "Thing")]))

        if "name" in node:
            g.add((node_uri, SCHEMA.name, Literal(node["name"])))

        titles = node.get("jobTitle", [])
        if not isinstance(titles, list):
            titles = [titles]

        for i, title in enumerate(titles):
            if isinstance(title, dict):
                stmt = BNode()
                g.add((stmt, RDF.type, RDF.Statement))
                g.add((stmt, RDF.subject, node_uri))
                g.add((stmt, RDF.predicate, SCHEMA.jobTitle))
                g.add((stmt, RDF.object, Literal(title.get("@value", ""))))
                if "@validFrom" in title:
                    g.add((stmt, JLDX.validFrom,
                           Literal(title["@validFrom"], datatype=XSD.dateTime)))
                if "@validUntil" in title:
                    g.add((stmt, JLDX.validUntil,
                           Literal(title["@validUntil"], datatype=XSD.dateTime)))
                if "@confidence" in title:
                    g.add((stmt, JLDX.confidence,
                           Literal(title["@confidence"], datatype=XSD.double)))

    return g


TEMPORAL_SPARQL = """
PREFIX jldx: <http://jsonld-ex.org/>
PREFIX schema: <http://schema.org/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?subject ?title ?validFrom ?validUntil
WHERE {
    ?stmt rdf:type rdf:Statement ;
          rdf:subject ?subject ;
          rdf:predicate schema:jobTitle ;
          rdf:object ?title ;
          jldx:validFrom ?validFrom ;
          jldx:validUntil ?validUntil .
    FILTER(?validFrom <= "2022-06-15T00:00:00Z"^^xsd:dateTime &&
           ?validUntil >= "2022-06-15T00:00:00Z"^^xsd:dateTime)
}
"""


def bench_temporal_query(
    sizes: list[int] = [100, 500, 1000],
    versions_per_node: int = 4,
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Compare temporal query: SPARQL via rdflib vs jsonld-ex query_at_time."""
    results = {}
    for n in sizes:
        nodes = make_temporal_graph(n, versions_per_node)

        # rdflib + SPARQL baseline (convert once, time the query)
        rdf_graph = _temporal_to_rdflib(nodes)
        stats_sparql = timed_trials(
            lambda: list(rdf_graph.query(TEMPORAL_SPARQL)),
            n=n_trials,
        )

        # jsonld-ex
        stats_jldx = timed_trials(
            lambda: query_at_time(nodes, "2022-06-15T00:00:00Z"),
            n=n_trials,
        )

        speedup = stats_sparql.mean / stats_jldx.mean if stats_jldx.mean > 0 else 0

        results[f"n={n}"] = {
            "rdflib_sparql": {
                **stats_sparql.to_dict(),
                "nodes_per_sec": round(n / stats_sparql.mean, 1) if stats_sparql.mean > 0 else 0,
            },
            "jsonld_ex": {
                **stats_jldx.to_dict(),
                "nodes_per_sec": round(n / stats_jldx.mean, 1) if stats_jldx.mean > 0 else 0,
            },
            "speedup": round(speedup, 2),
        }
    return results


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════


@dataclass
class BaselineResults:
    prov_o_construction: dict[str, Any] = field(default_factory=dict)
    shacl_validation: dict[str, Any] = field(default_factory=dict)
    graph_merge: dict[str, Any] = field(default_factory=dict)
    temporal_query: dict[str, Any] = field(default_factory=dict)


def run_all() -> BaselineResults:
    results = BaselineResults()
    print("=== Baseline Comparisons ===\n")

    print("B.1  PROV-O construction: rdflib vs jsonld-ex...")
    results.prov_o_construction = bench_prov_o_construction()

    print("B.2  SHACL validation: pyshacl vs jsonld-ex...")
    results.shacl_validation = bench_shacl_validation()

    print("B.3  Graph merge: rdflib union vs jsonld-ex...")
    results.graph_merge = bench_graph_merge()

    print("B.4  Temporal query: SPARQL vs jsonld-ex...")
    results.temporal_query = bench_temporal_query()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)

    print("\n--- PROV-O Construction ---")
    for k, v in r.prov_o_construction.items():
        rl = v['rdflib']
        jl = v['jsonld_ex']
        print(f"  {k}: rdflib {rl['mean_sec']*1000:.2f}ms vs jsonld-ex {jl['mean_sec']*1000:.2f}ms "
              f"(speedup: {v['speedup']}x)")

    print("\n--- SHACL Validation ---")
    if "skipped" in r.shacl_validation:
        print(f"  {r.shacl_validation['skipped']}")
    else:
        for k, v in r.shacl_validation.items():
            ps = v['pyshacl']
            jl = v['jsonld_ex']
            print(f"  {k}: pyshacl {ps['mean_sec']*1000:.2f}ms vs jsonld-ex {jl['mean_sec']*1000:.2f}ms "
                  f"(speedup: {v['speedup']}x)")

    print("\n--- Graph Merge ---")
    for k, v in r.graph_merge.items():
        rl = v['rdflib_merge']
        jl = v['jsonld_ex_merge']
        print(f"  {k}: rdflib {rl['mean_sec']*1000:.2f}ms vs jsonld-ex {jl['mean_sec']*1000:.2f}ms "
              f"(speedup: {v['speedup']}x)")

    print("\n--- Temporal Query ---")
    for k, v in r.temporal_query.items():
        sp = v['rdflib_sparql']
        jl = v['jsonld_ex']
        print(f"  {k}: SPARQL {sp['mean_sec']*1000:.2f}ms vs jsonld-ex {jl['mean_sec']*1000:.2f}ms "
              f"(speedup: {v['speedup']}x)")
