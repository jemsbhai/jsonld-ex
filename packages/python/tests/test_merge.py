"""Tests for graph merging with confidence-aware conflict resolution."""

import pytest

from jsonld_ex.merge import (
    merge_graphs,
    diff_graphs,
    MergeReport,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _person_graph(node_id, name, name_conf, source=None, extracted_at=None, extra=None):
    """Build a simple Person graph for testing."""
    name_val = {"@value": name, "@confidence": name_conf}
    if source:
        name_val["@source"] = source
    if extracted_at:
        name_val["@extractedAt"] = extracted_at
    node = {"@id": node_id, "@type": "Person", "name": name_val}
    if extra:
        node.update(extra)
    return {"@context": "http://schema.org/", "@graph": [node]}


# ═══════════════════════════════════════════════════════════════════
# merge_graphs — basic
# ═══════════════════════════════════════════════════════════════════


class TestMergeBasic:
    def test_two_graphs_same_value_combines_confidence(self):
        g1 = _person_graph("ex:alice", "Alice", 0.8, source="A")
        g2 = _person_graph("ex:alice", "Alice", 0.7, source="B")
        merged, report = merge_graphs([g1, g2])

        nodes = merged["@graph"]
        assert len(nodes) == 1
        alice = nodes[0]
        assert alice["@id"] == "ex:alice"
        # noisy-OR of 0.8, 0.7 = 1 - 0.2*0.3 = 0.94
        assert alice["name"]["@confidence"] == pytest.approx(0.94)
        assert report.properties_agreed == 1
        assert report.properties_conflicted == 0
        assert report.nodes_merged == 1

    def test_two_graphs_different_values_conflict(self):
        g1 = _person_graph("ex:alice", "Alice Smith", 0.9)
        g2 = _person_graph("ex:alice", "A. Smith", 0.7)
        merged, report = merge_graphs([g1, g2], conflict_strategy="highest")

        alice = merged["@graph"][0]
        assert alice["name"]["@value"] == "Alice Smith"
        assert report.properties_conflicted == 1
        assert len(report.conflicts) == 1
        assert report.conflicts[0].winner_value == "Alice Smith"

    def test_disjoint_nodes_all_kept(self):
        g1 = {"@context": "http://schema.org/", "@graph": [
            {"@id": "ex:alice", "@type": "Person", "name": {"@value": "Alice", "@confidence": 0.9}}
        ]}
        g2 = {"@context": "http://schema.org/", "@graph": [
            {"@id": "ex:bob", "@type": "Person", "name": {"@value": "Bob", "@confidence": 0.8}}
        ]}
        merged, report = merge_graphs([g1, g2])
        assert len(merged["@graph"]) == 2
        assert report.nodes_merged == 2

    def test_three_graphs(self):
        g1 = _person_graph("ex:alice", "Alice", 0.7, source="A")
        g2 = _person_graph("ex:alice", "Alice", 0.6, source="B")
        g3 = _person_graph("ex:alice", "Alice", 0.5, source="C")
        merged, report = merge_graphs([g1, g2, g3])

        alice = merged["@graph"][0]
        # noisy-OR of 0.7, 0.6, 0.5 = 1 - 0.3*0.4*0.5 = 0.94
        assert alice["name"]["@confidence"] == pytest.approx(0.94)
        assert report.source_count == 3

    def test_preserves_context(self):
        g1 = _person_graph("ex:alice", "Alice", 0.9)
        g2 = _person_graph("ex:alice", "Alice", 0.8)
        merged, _ = merge_graphs([g1, g2])
        assert merged["@context"] == "http://schema.org/"


class TestMergeConflictStrategies:
    def test_highest_picks_max_confidence(self):
        g1 = _person_graph("ex:alice", "Alice", 0.9)
        g2 = _person_graph("ex:alice", "ALICE", 0.6)
        merged, _ = merge_graphs([g1, g2], conflict_strategy="highest")
        assert merged["@graph"][0]["name"]["@value"] == "Alice"

    def test_weighted_vote_two_beat_one(self):
        g1 = {"@context": "http://schema.org/", "@graph": [
            {"@id": "ex:x", "title": {"@value": "Engineer", "@confidence": 0.7}}
        ]}
        g2 = {"@context": "http://schema.org/", "@graph": [
            {"@id": "ex:x", "title": {"@value": "Engineer", "@confidence": 0.6}}
        ]}
        g3 = {"@context": "http://schema.org/", "@graph": [
            {"@id": "ex:x", "title": {"@value": "Manager", "@confidence": 0.8}}
        ]}
        merged, report = merge_graphs(
            [g1, g2, g3], conflict_strategy="weighted_vote"
        )
        # noisy-OR(0.7, 0.6) = 0.88 > 0.8
        assert merged["@graph"][0]["title"]["@value"] == "Engineer"
        assert report.properties_conflicted == 1

    def test_recency_prefers_newer(self):
        g1 = _person_graph("ex:alice", "Old Name", 0.9, extracted_at="2024-01-01T00:00:00Z")
        g2 = _person_graph("ex:alice", "New Name", 0.7, extracted_at="2025-06-01T00:00:00Z")
        merged, _ = merge_graphs([g1, g2], conflict_strategy="recency")
        assert merged["@graph"][0]["name"]["@value"] == "New Name"

    def test_union_keeps_all(self):
        g1 = _person_graph("ex:alice", "Alice", 0.9)
        g2 = _person_graph("ex:alice", "ALICE", 0.7)
        merged, report = merge_graphs([g1, g2], conflict_strategy="union")
        names = merged["@graph"][0]["name"]
        assert isinstance(names, list)
        assert len(names) == 2
        assert report.properties_union == 1


class TestMergeCombinationMethods:
    def test_average_combination(self):
        g1 = _person_graph("ex:alice", "Alice", 0.8)
        g2 = _person_graph("ex:alice", "Alice", 0.6)
        merged, _ = merge_graphs(
            [g1, g2], confidence_combination="average"
        )
        assert merged["@graph"][0]["name"]["@confidence"] == pytest.approx(0.7)

    def test_max_combination(self):
        g1 = _person_graph("ex:alice", "Alice", 0.8)
        g2 = _person_graph("ex:alice", "Alice", 0.6)
        merged, _ = merge_graphs(
            [g1, g2], confidence_combination="max"
        )
        assert merged["@graph"][0]["name"]["@confidence"] == pytest.approx(0.8)


class TestMergeEdgeCases:
    def test_fewer_than_two_graphs_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            merge_graphs([{"@graph": []}])

    def test_empty_graphs(self):
        g1 = {"@graph": []}
        g2 = {"@graph": []}
        merged, report = merge_graphs([g1, g2])
        assert merged["@graph"] == []
        assert report.nodes_merged == 0

    def test_top_level_node_without_graph(self):
        """Documents that are a single node without @graph."""
        g1 = {"@context": "http://schema.org/", "@id": "ex:a", "@type": "Person",
               "name": {"@value": "Alice", "@confidence": 0.9}}
        g2 = {"@context": "http://schema.org/", "@id": "ex:a", "@type": "Person",
               "name": {"@value": "Alice", "@confidence": 0.8}}
        merged, report = merge_graphs([g1, g2])
        assert report.nodes_merged == 1

    def test_no_confidence_on_values(self):
        """Plain values without @confidence should still merge."""
        g1 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        merged, report = merge_graphs([g1, g2])
        # No confidence to combine — just keeps one
        assert merged["@graph"][0]["name"] == "Alice"
        assert report.properties_agreed == 1

    def test_type_union(self):
        """Different @type values should be unioned."""
        g1 = {"@graph": [{"@id": "ex:a", "@type": "Person", "name": {"@value": "A", "@confidence": 0.9}}]}
        g2 = {"@graph": [{"@id": "ex:a", "@type": "Employee", "name": {"@value": "A", "@confidence": 0.8}}]}
        merged, _ = merge_graphs([g1, g2])
        node = merged["@graph"][0]
        assert set(node["@type"]) == {"Person", "Employee"}

    def test_anonymous_nodes_passed_through(self):
        g1 = {"@graph": [{"@type": "Thing", "label": "anon1"}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": {"@value": "A", "@confidence": 0.9}}]}
        merged, report = merge_graphs([g1, g2])
        # 1 identified node + 1 anonymous
        assert len(merged["@graph"]) == 2

    def test_extra_properties_merged(self):
        """Properties that only exist in one source should be kept."""
        g1 = _person_graph("ex:alice", "Alice", 0.9, extra={"age": {"@value": 30, "@confidence": 0.8}})
        g2 = _person_graph("ex:alice", "Alice", 0.8, extra={"email": {"@value": "a@x.com", "@confidence": 0.7}})
        merged, report = merge_graphs([g1, g2])
        alice = merged["@graph"][0]
        assert "age" in alice
        assert "email" in alice


# ═══════════════════════════════════════════════════════════════════
# diff_graphs
# ═══════════════════════════════════════════════════════════════════


class TestDiffGraphs:
    def test_identical_graphs(self):
        g = {"@graph": [{"@id": "ex:a", "name": {"@value": "Alice", "@confidence": 0.9}}]}
        d = diff_graphs(g, g)
        assert len(d["added"]) == 0
        assert len(d["removed"]) == 0
        assert len(d["modified"]) == 0
        assert len(d["unchanged"]) == 1

    def test_added_node(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": "Alice"}, {"@id": "ex:b", "name": "Bob"}]}
        d = diff_graphs(g1, g2)
        added_ids = [e["@id"] for e in d["added"] if "node" in e]
        assert "ex:b" in added_ids

    def test_removed_node(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": "Alice"}, {"@id": "ex:b", "name": "Bob"}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        d = diff_graphs(g1, g2)
        removed_ids = [e["@id"] for e in d["removed"] if "node" in e]
        assert "ex:b" in removed_ids

    def test_modified_property(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": {"@value": "Alice", "@confidence": 0.9}}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": {"@value": "ALICE", "@confidence": 0.8}}]}
        d = diff_graphs(g1, g2)
        assert len(d["modified"]) == 1
        assert d["modified"][0]["property"] == "name"

    def test_added_property(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": "Alice", "age": 30}]}
        d = diff_graphs(g1, g2)
        added_props = [e for e in d["added"] if "property" in e]
        assert len(added_props) == 1
        assert added_props[0]["property"] == "age"

    def test_removed_property(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": "Alice", "age": 30}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": "Alice"}]}
        d = diff_graphs(g1, g2)
        removed_props = [e for e in d["removed"] if "property" in e]
        assert len(removed_props) == 1
        assert removed_props[0]["property"] == "age"

    def test_unchanged_tracks_confidence_delta(self):
        g1 = {"@graph": [{"@id": "ex:a", "name": {"@value": "Alice", "@confidence": 0.9}}]}
        g2 = {"@graph": [{"@id": "ex:a", "name": {"@value": "Alice", "@confidence": 0.95}}]}
        d = diff_graphs(g1, g2)
        assert len(d["unchanged"]) == 1
        u = d["unchanged"][0]
        assert u["confidence_a"] == 0.9
        assert u["confidence_b"] == 0.95

    def test_empty_graphs(self):
        d = diff_graphs({"@graph": []}, {"@graph": []})
        assert d == {"added": [], "removed": [], "modified": [], "unchanged": []}


# ═══════════════════════════════════════════════════════════════════
# MergeReport
# ═══════════════════════════════════════════════════════════════════


class TestMergeReport:
    def test_report_fields(self):
        g1 = _person_graph("ex:alice", "Alice", 0.9)
        g2 = _person_graph("ex:alice", "Bob", 0.7)
        _, report = merge_graphs([g1, g2])
        assert report.source_count == 2
        assert report.nodes_merged == 1
        assert report.properties_conflicted == 1
        assert isinstance(report.conflicts, list)

    def test_conflict_details(self):
        g1 = _person_graph("ex:alice", "Alice", 0.9)
        g2 = _person_graph("ex:alice", "Bob", 0.7)
        _, report = merge_graphs([g1, g2], conflict_strategy="highest")
        c = report.conflicts[0]
        assert c.node_id == "ex:alice"
        assert c.property_name == "name"
        assert c.resolution == "highest"
        assert c.winner_value == "Alice"
