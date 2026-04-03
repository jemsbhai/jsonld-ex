"""Test that to_prov_o handles @graph documents natively.

RED phase: these tests should FAIL on current code because to_prov_o
does not recurse into @graph member nodes.

Run:
    cd E:\\data\\code\\claudecode\\jsonld\\jsonld-ex
    python -m pytest tests/test_prov_o_graph.py -v
"""
import pytest
from jsonld_ex import to_prov_o, from_prov_o


def _make_graph_doc(n=3):
    """Create a small @graph document with annotated nodes."""
    nodes = []
    for i in range(n):
        nodes.append({
            "@id": f"ex:person-{i}",
            "@type": "Person",
            "name": {
                "@value": f"Person-{i}",
                "@confidence": 0.9,
                "@source": "https://model.example.org/ner-v1",
                "@extractedAt": "2025-06-01T12:00:00Z",
                "@method": "NER",
            },
            "age": {
                "@value": 30 + i,
                "@confidence": 0.7,
            },
        })
    return {
        "@context": "http://schema.org/",
        "@graph": nodes,
    }


class TestToProvOGraphSupport:
    """to_prov_o should process all nodes in a @graph document."""

    def test_graph_doc_converts_all_nodes(self):
        """Each annotated property in each @graph node should produce PROV-O entities."""
        doc = _make_graph_doc(3)
        prov_doc, report = to_prov_o(doc)

        # 3 nodes x 2 annotated properties = 6 converted
        assert report.nodes_converted == 6, (
            f"Expected 6 converted nodes (3 nodes x 2 props), got {report.nodes_converted}"
        )

    def test_graph_doc_produces_prov_entities(self):
        """PROV-O output should contain Entity nodes for annotated values."""
        doc = _make_graph_doc(2)
        prov_doc, report = to_prov_o(doc)

        graph = prov_doc.get("@graph", [])
        # Should have entities, agents, activities in the graph
        prov_ns = "http://www.w3.org/ns/prov#"
        entity_types = [
            n for n in graph
            if n.get("@type") == f"{prov_ns}Entity"
        ]
        # 2 nodes x 2 annotated props = 4 entities
        assert len(entity_types) == 4, (
            f"Expected 4 PROV-O Entity nodes, got {len(entity_types)}"
        )

    def test_graph_doc_preserves_context(self):
        """Output should have PROV-O context."""
        doc = _make_graph_doc(1)
        prov_doc, report = to_prov_o(doc)

        ctx = prov_doc.get("@context", {})
        assert "prov" in ctx

    def test_single_node_still_works(self):
        """Existing single-node behavior must not break."""
        single = {
            "@context": "http://schema.org/",
            "@id": "ex:person-0",
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
                "@source": "https://model.example.org/ner-v1",
            },
        }
        prov_doc, report = to_prov_o(single)
        assert report.nodes_converted == 1

    def test_graph_round_trip(self):
        """to_prov_o -> from_prov_o should preserve confidence for @graph docs."""
        doc = _make_graph_doc(2)
        prov_doc, _ = to_prov_o(doc)
        restored, _ = from_prov_o(prov_doc)

        # The restored doc should have confidence values
        graph = restored.get("@graph", [restored])
        # At least check that some confidence was preserved
        found_confidence = False
        for node in graph:
            for key, val in node.items():
                if isinstance(val, dict) and "@confidence" in val:
                    found_confidence = True
                    break
        assert found_confidence, "Round-trip lost all @confidence annotations"
