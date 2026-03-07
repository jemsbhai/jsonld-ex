"""Tests for JSON-LD bridge (Tier 3, Step 8).

The bridge converts between application-level JSON-LD documents
(annotated with @confidence, @asOf, @validFrom, @validUntil) and
SLNetwork graphs for structured inference.

This is distinct from to_jsonld()/from_jsonld() on SLNetwork, which
serialize the internal SLNetwork representation. The bridge works
with standard jsonld-ex annotated documents.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import SLNode, SLEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.jsonld_bridge import (
    network_from_jsonld_graph,
    network_to_jsonld_graph,
)


# -- Fixtures --

NOW_STR = "2025-06-01T12:00:00+00:00"
NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
YESTERDAY_STR = "2025-05-31T12:00:00+00:00"
YESTERDAY = datetime(2025, 5, 31, 12, 0, 0, tzinfo=timezone.utc)


# ===============================================================
# network_from_jsonld_graph: basic node extraction
# ===============================================================


class TestFromGraphNodes:
    """Nodes are extracted from @graph items."""

    def test_nodes_from_ids(self):
        graph = [
            {"@id": "rain", "@confidence": 0.7},
            {"@id": "wet_grass", "@confidence": 0.5},
        ]
        net = network_from_jsonld_graph(graph)
        assert net.has_node("rain")
        assert net.has_node("wet_grass")
        assert net.node_count() == 2

    def test_scalar_confidence_to_opinion(self):
        """Scalar @confidence is converted to an Opinion."""
        graph = [{"@id": "x", "@confidence": 0.8}]
        net = network_from_jsonld_graph(graph, default_uncertainty=0.1)
        op = net.get_node("x").opinion
        assert op.belief == pytest.approx(0.8)
        # disbelief = 1 - belief - uncertainty = 0.1
        assert op.disbelief == pytest.approx(0.1)
        assert op.uncertainty == pytest.approx(0.1)

    def test_scalar_confidence_zero_uncertainty(self):
        """With default_uncertainty=0, scalar becomes dogmatic."""
        graph = [{"@id": "x", "@confidence": 0.8}]
        net = network_from_jsonld_graph(graph, default_uncertainty=0.0)
        op = net.get_node("x").opinion
        assert op.belief == pytest.approx(0.8)
        assert op.disbelief == pytest.approx(0.2)
        assert op.uncertainty == pytest.approx(0.0)

    def test_full_opinion_dict(self):
        """A full opinion dict with belief/disbelief/uncertainty is used directly."""
        graph = [
            {
                "@id": "x",
                "@confidence": {
                    "@belief": 0.7,
                    "@disbelief": 0.1,
                    "@uncertainty": 0.2,
                },
            },
        ]
        net = network_from_jsonld_graph(graph)
        op = net.get_node("x").opinion
        assert op.belief == pytest.approx(0.7)
        assert op.disbelief == pytest.approx(0.1)
        assert op.uncertainty == pytest.approx(0.2)

    def test_no_confidence_gets_vacuous(self):
        """Nodes without @confidence get a vacuous opinion."""
        graph = [{"@id": "x"}]
        net = network_from_jsonld_graph(graph)
        op = net.get_node("x").opinion
        assert op.uncertainty == pytest.approx(1.0)

    def test_timestamp_from_as_of(self):
        """@asOf is mapped to node timestamp."""
        graph = [{"@id": "x", "@confidence": 0.7, "@asOf": NOW_STR}]
        net = network_from_jsonld_graph(graph)
        assert net.get_node("x").timestamp == NOW

    def test_no_timestamp_is_none(self):
        graph = [{"@id": "x", "@confidence": 0.7}]
        net = network_from_jsonld_graph(graph)
        assert net.get_node("x").timestamp is None

    def test_label_from_type(self):
        """@type is stored as the node label."""
        graph = [{"@id": "x", "@type": "Observation", "@confidence": 0.7}]
        net = network_from_jsonld_graph(graph)
        assert net.get_node("x").label == "Observation"


# ===============================================================
# network_from_jsonld_graph: edge extraction
# ===============================================================


class TestFromGraphEdges:
    """Edges are extracted from named relationship properties."""

    def test_edge_from_property(self):
        graph = [
            {"@id": "rain", "@confidence": 0.7},
            {
                "@id": "wet_grass", "@confidence": 0.5,
                "causedBy": {
                    "@id": "rain",
                    "@confidence": 0.9,
                },
            },
        ]
        net = network_from_jsonld_graph(graph, edge_properties=["causedBy"])
        assert net.has_edge("rain", "wet_grass")
        edge = net.get_edge("rain", "wet_grass")
        assert edge.conditional.belief == pytest.approx(0.9)

    def test_edge_with_validity_window(self):
        graph = [
            {"@id": "a", "@confidence": 0.7},
            {
                "@id": "b", "@confidence": 0.5,
                "dependsOn": {
                    "@id": "a",
                    "@confidence": 0.9,
                    "@validFrom": YESTERDAY_STR,
                    "@validUntil": NOW_STR,
                },
            },
        ]
        net = network_from_jsonld_graph(graph, edge_properties=["dependsOn"])
        edge = net.get_edge("a", "b")
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until == NOW

    def test_edge_with_timestamp(self):
        graph = [
            {"@id": "a", "@confidence": 0.7},
            {
                "@id": "b", "@confidence": 0.5,
                "dependsOn": {
                    "@id": "a",
                    "@confidence": 0.9,
                    "@asOf": YESTERDAY_STR,
                },
            },
        ]
        net = network_from_jsonld_graph(graph, edge_properties=["dependsOn"])
        edge = net.get_edge("a", "b")
        assert edge.timestamp == YESTERDAY

    def test_edge_without_confidence_gets_vacuous_conditional(self):
        """Edge property with @id but no @confidence gets vacuous conditional."""
        graph = [
            {"@id": "a", "@confidence": 0.7},
            {
                "@id": "b", "@confidence": 0.5,
                "dependsOn": {"@id": "a"},
            },
        ]
        net = network_from_jsonld_graph(graph, edge_properties=["dependsOn"])
        edge = net.get_edge("a", "b")
        assert edge.conditional.uncertainty == pytest.approx(1.0)

    def test_non_edge_properties_ignored(self):
        """Properties not in edge_properties are not treated as edges."""
        graph = [
            {"@id": "a", "@confidence": 0.7, "label": "node A"},
            {
                "@id": "b", "@confidence": 0.5,
                "causedBy": {"@id": "a", "@confidence": 0.9},
            },
        ]
        # Only "causedBy" is an edge property
        net = network_from_jsonld_graph(graph, edge_properties=["causedBy"])
        assert net.has_edge("a", "b")
        assert net.edge_count() == 1


# ===============================================================
# network_to_jsonld_graph: export
# ===============================================================


class TestToGraph:
    """Export SLNetwork back to JSON-LD graph."""

    def test_nodes_exported(self):
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="rain", opinion=Opinion(0.7, 0.1, 0.2),
            timestamp=NOW,
        ))
        graph = network_to_jsonld_graph(net)
        assert len(graph) >= 1
        rain = next(n for n in graph if n["@id"] == "rain")
        assert "@confidence" in rain

    def test_confidence_is_projected_probability(self):
        """Exported @confidence is the projected probability P = b + a*u."""
        net = SLNetwork()
        op = Opinion(0.7, 0.1, 0.2, base_rate=0.5)
        net.add_node(SLNode(node_id="x", opinion=op))
        graph = network_to_jsonld_graph(net)
        x = next(n for n in graph if n["@id"] == "x")
        expected_p = op.belief + op.base_rate * op.uncertainty  # 0.7 + 0.5*0.2 = 0.8
        assert x["@confidence"] == pytest.approx(expected_p)

    def test_full_opinion_in_metadata(self):
        """The full opinion tuple is available in @opinion."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="x", opinion=Opinion(0.7, 0.1, 0.2)))
        graph = network_to_jsonld_graph(net)
        x = next(n for n in graph if n["@id"] == "x")
        assert "@opinion" in x
        assert x["@opinion"]["@belief"] == pytest.approx(0.7)
        assert x["@opinion"]["@disbelief"] == pytest.approx(0.1)
        assert x["@opinion"]["@uncertainty"] == pytest.approx(0.2)

    def test_timestamp_exported_as_as_of(self):
        net = SLNetwork()
        net.add_node(SLNode(node_id="x", opinion=Opinion(0.7, 0.1, 0.2), timestamp=NOW))
        graph = network_to_jsonld_graph(net)
        x = next(n for n in graph if n["@id"] == "x")
        assert x.get("@asOf") == NOW.isoformat()

    def test_no_timestamp_no_as_of(self):
        net = SLNetwork()
        net.add_node(SLNode(node_id="x", opinion=Opinion(0.7, 0.1, 0.2)))
        graph = network_to_jsonld_graph(net)
        x = next(n for n in graph if n["@id"] == "x")
        assert "@asOf" not in x or x["@asOf"] is None

    def test_edges_exported(self):
        net = SLNetwork()
        net.add_node(SLNode(node_id="a", opinion=Opinion(0.7, 0.1, 0.2)))
        net.add_node(SLNode(node_id="b", opinion=Opinion(0.5, 0.3, 0.2)))
        net.add_edge(SLEdge(
            source_id="a", target_id="b",
            conditional=Opinion(0.9, 0.05, 0.05),
            valid_from=YESTERDAY, valid_until=NOW,
        ))
        graph = network_to_jsonld_graph(net)
        b = next(n for n in graph if n["@id"] == "b")
        assert "slnet:parentEdges" in b
        edge_data = b["slnet:parentEdges"][0]
        assert edge_data["@id"] == "a"
        assert "@confidence" in edge_data


# ===============================================================
# Round-trip
# ===============================================================


class TestRoundTrip:
    """network_from_jsonld_graph -> network_to_jsonld_graph preserves data."""

    def test_node_opinions_survive_round_trip(self):
        graph = [
            {"@id": "a", "@confidence": 0.8, "@asOf": NOW_STR},
            {"@id": "b", "@confidence": 0.6},
        ]
        net = network_from_jsonld_graph(graph, default_uncertainty=0.1)
        exported = network_to_jsonld_graph(net)

        # Re-import
        net2 = network_from_jsonld_graph(exported, default_uncertainty=0.1)
        # Opinions should be close (projected probability round-trips)
        op_a = net.get_node("a").opinion
        op_a2 = net2.get_node("a").opinion
        assert op_a.belief == pytest.approx(op_a2.belief, abs=1e-6)
        assert op_a.uncertainty == pytest.approx(op_a2.uncertainty, abs=1e-6)

    def test_timestamp_survives_round_trip(self):
        graph = [{"@id": "x", "@confidence": 0.7, "@asOf": NOW_STR}]
        net = network_from_jsonld_graph(graph)
        exported = network_to_jsonld_graph(net)
        net2 = network_from_jsonld_graph(exported)
        assert net2.get_node("x").timestamp == NOW
