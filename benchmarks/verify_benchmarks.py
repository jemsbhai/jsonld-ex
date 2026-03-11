"""
Benchmark Verification Script

Checks that all benchmark files can import their dependencies
and run a minimal smoke test against the current jsonld-ex version.

Usage:
    cd benchmarks
    python verify_benchmarks.py

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

# Add parent to path so we can import jsonld_ex from the source tree
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "python", "src"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

failures: list[str] = []


def check(name: str, fn):
    """Run a check function, report pass/fail."""
    global failures
    try:
        fn()
        print(f"  [{PASS}] {name}")
    except Exception as e:
        print(f"  [{FAIL}] {name}: {e}")
        failures.append(name)


def skip(name: str, reason: str):
    print(f"  [{SKIP}] {name}: {reason}")


def main() -> int:
    global failures
    failures = []

    # ── 0. Version check ──
    print("\n=== jsonld-ex Version ===")
    try:
        import jsonld_ex
        ver = getattr(jsonld_ex, "__version__", "unknown")
        print(f"  Installed version: {ver}")
    except ImportError:
        print(f"  [{FAIL}] jsonld-ex not importable!")
        print("  Run: pip install -e ../packages/python")
        return 1

    # ── 1. Core imports ──
    print("\n=== Core Library Imports ===")

    check("confidence_algebra", lambda: importlib.import_module("jsonld_ex.confidence_algebra"))
    check("confidence_bridge", lambda: importlib.import_module("jsonld_ex.confidence_bridge"))
    check("confidence_decay", lambda: importlib.import_module("jsonld_ex.confidence_decay"))
    check("confidence_byzantine", lambda: importlib.import_module("jsonld_ex.confidence_byzantine"))
    check("inference", lambda: importlib.import_module("jsonld_ex.inference"))
    check("ai_ml", lambda: importlib.import_module("jsonld_ex.ai_ml"))
    check("validation", lambda: importlib.import_module("jsonld_ex.validation"))
    check("owl_interop", lambda: importlib.import_module("jsonld_ex.owl_interop"))
    check("merge", lambda: importlib.import_module("jsonld_ex.merge"))
    check("temporal", lambda: importlib.import_module("jsonld_ex.temporal"))
    check("multinomial_algebra", lambda: importlib.import_module("jsonld_ex.multinomial_algebra"))
    check("sl_network", lambda: importlib.import_module("jsonld_ex.sl_network"))

    # ── 2. Optional imports ──
    print("\n=== Optional Dependencies ===")

    has_cbor = False
    try:
        import cbor2
        has_cbor = True
        check("cbor_ld", lambda: importlib.import_module("jsonld_ex.cbor_ld"))
    except ImportError:
        skip("cbor_ld", "cbor2 not installed (pip install cbor2)")

    has_mqtt = False
    try:
        import paho.mqtt
        has_mqtt = True
        check("mqtt", lambda: importlib.import_module("jsonld_ex.mqtt"))
    except ImportError:
        skip("mqtt", "paho-mqtt not installed (pip install paho-mqtt)")

    has_rdflib = False
    try:
        import rdflib
        has_rdflib = True
        check("rdflib", lambda: None)
    except ImportError:
        skip("rdflib", "not installed (pip install rdflib>=7.0)")

    has_pyshacl = False
    try:
        import pyshacl
        has_pyshacl = True
        check("pyshacl", lambda: None)
    except ImportError:
        skip("pyshacl", "not installed (pip install pyshacl>=0.26)")

    # ── 3. Benchmark file imports ──
    print("\n=== Benchmark File Imports ===")

    check("bench_utils", lambda: importlib.import_module("bench_utils"))
    check("data_generators", lambda: importlib.import_module("data_generators"))
    check("bench_algebra", lambda: importlib.import_module("bench_algebra"))
    check("bench_owl_rdf", lambda: importlib.import_module("bench_owl_rdf"))
    check("bench_multi_agent", lambda: importlib.import_module("bench_multi_agent"))
    check("bench_rag", lambda: importlib.import_module("bench_rag"))

    if has_cbor and has_mqtt:
        check("bench_iot", lambda: importlib.import_module("bench_iot"))
    else:
        skip("bench_iot", "requires cbor2 + paho-mqtt")

    if has_rdflib:
        check("bench_baselines", lambda: importlib.import_module("bench_baselines"))
    else:
        skip("bench_baselines", "requires rdflib")

    check("bench_bridge", lambda: importlib.import_module("bench_bridge"))

    # ── 4. Minimal smoke tests ──
    print("\n=== Smoke Tests (minimal execution) ===")

    # Algebra smoke: create opinion, fuse two
    def smoke_algebra():
        from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse
        a = Opinion(0.6, 0.2, 0.2)
        b = Opinion(0.3, 0.4, 0.3)
        c = cumulative_fuse(a, b)
        assert abs(c.belief + c.disbelief + c.uncertainty - 1.0) < 1e-9
    check("algebra: fuse two opinions", smoke_algebra)

    # Multinomial smoke
    def smoke_multinomial():
        from jsonld_ex.multinomial_algebra import (
            MultinomialOpinion, multinomial_cumulative_fuse,
        )
        a = MultinomialOpinion(
            beliefs={"x": 0.3, "y": 0.3}, uncertainty=0.4,
            base_rates={"x": 0.5, "y": 0.5},
        )
        d = a.to_dict()
        b = MultinomialOpinion.from_dict(d)
        assert a == b
    check("multinomial: round-trip serialization", smoke_multinomial)

    # SLNetwork smoke
    def smoke_slnetwork():
        from jsonld_ex.sl_network import SLNetwork, SLNode, SLEdge, infer_node
        from jsonld_ex.confidence_algebra import Opinion
        net = SLNetwork()
        net.add_node(SLNode("A", Opinion(0.8, 0.1, 0.1)))
        net.add_node(SLNode("B", Opinion(0.0, 0.0, 1.0)))
        net.add_edge(SLEdge("A", "B", conditional=Opinion(0.9, 0.05, 0.05)))
        result = infer_node(net, "B")
        assert result.opinion.belief > 0
    check("sl_network: basic inference", smoke_slnetwork)

    # Data generators smoke
    def smoke_generators():
        from data_generators import make_annotated_graph, make_sensor_batch
        doc = make_annotated_graph(10)
        assert len(doc["@graph"]) == 10
        batch = make_sensor_batch(5)
        assert len(batch) == 5
    check("data_generators: make graphs", smoke_generators)

    # bench_utils smoke
    def smoke_bench_utils():
        from bench_utils import timed_trials
        import time
        stats = timed_trials(lambda: time.sleep(0.001), n=3, warmup=1)
        assert stats.mean > 0
        assert stats.n == 3
    check("bench_utils: timed_trials", smoke_bench_utils)

    # OWL interop smoke
    def smoke_owl():
        from jsonld_ex import annotate, to_prov_o
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "name": annotate("Alice", confidence=0.9, source="test"),
        }
        prov, report = to_prov_o(doc)
        assert "@graph" in prov or "@type" in prov
    check("owl_interop: annotate + to_prov_o", smoke_owl)

    # Merge smoke
    def smoke_merge():
        from data_generators import make_conflicting_graphs
        from jsonld_ex import merge_graphs
        graphs = make_conflicting_graphs(5, 2, conflict_rate=0.3)
        merged, report = merge_graphs(graphs)
        assert "@graph" in merged
    check("merge: conflicting graphs", smoke_merge)

    # Temporal smoke
    def smoke_temporal():
        from data_generators import make_temporal_graph
        from jsonld_ex import query_at_time
        nodes = make_temporal_graph(5, versions_per_node=3)
        result = query_at_time(nodes, "2022-06-15T00:00:00Z")
        assert isinstance(result, list)
    check("temporal: query_at_time", smoke_temporal)

    # Trust propagation smoke
    def smoke_trust():
        from jsonld_ex.sl_network import (
            SLNetwork, SLNode, propagate_trust,
        )
        from jsonld_ex.sl_network.types import TrustEdge
        from jsonld_ex.confidence_algebra import Opinion
        net = SLNetwork()
        net.add_node(SLNode("agent_A", Opinion(0.0, 0.0, 1.0), node_type="agent"))
        net.add_node(SLNode("agent_B", Opinion(0.0, 0.0, 1.0), node_type="agent"))
        net.add_trust_edge(TrustEdge(
            source_id="agent_A", target_id="agent_B",
            trust_opinion=Opinion(0.8, 0.1, 0.1),
        ))
        result = propagate_trust(net, "agent_A")
        assert "agent_B" in result.derived_trusts
    check("sl_network: trust propagation", smoke_trust)

    # BN interop smoke (optional)
    try:
        import pgmpy
        def smoke_bn():
            from pgmpy.utils import get_example_model
            from jsonld_ex.sl_network import from_bayesian_network
            bn = get_example_model("asia")
            net = from_bayesian_network(bn, default_sample_count=100)
            assert len(net) > 0
        check("bn_interop: asia network", smoke_bn)
    except (ImportError, TypeError):
        skip("bn_interop", "pgmpy not available")

    # ── Summary ──
    print("\n" + "=" * 50)
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
