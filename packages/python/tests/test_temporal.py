"""Tests for temporal extensions."""

import pytest

from jsonld_ex.temporal import (
    add_temporal,
    query_at_time,
    temporal_diff,
    TemporalDiffResult,
)


# ═══════════════════════════════════════════════════════════════════
# add_temporal
# ═══════════════════════════════════════════════════════════════════


class TestAddTemporal:
    def test_valid_from(self):
        r = add_temporal("Engineer", valid_from="2020-01-01")
        assert r == {"@value": "Engineer", "@validFrom": "2020-01-01"}

    def test_valid_until(self):
        r = add_temporal("Engineer", valid_until="2024-12-31")
        assert r == {"@value": "Engineer", "@validUntil": "2024-12-31"}

    def test_as_of(self):
        r = add_temporal("Engineer", as_of="2023-06-15T10:00:00Z")
        assert r == {"@value": "Engineer", "@asOf": "2023-06-15T10:00:00Z"}

    def test_all_three(self):
        r = add_temporal(
            "Engineer",
            valid_from="2020-01-01",
            valid_until="2024-12-31",
            as_of="2023-06-15",
        )
        assert r["@validFrom"] == "2020-01-01"
        assert r["@validUntil"] == "2024-12-31"
        assert r["@asOf"] == "2023-06-15"

    def test_preserves_existing_annotations(self):
        """Should compose with @confidence etc."""
        annotated = {"@value": "Engineer", "@confidence": 0.9, "@source": "model-v1"}
        r = add_temporal(annotated, valid_from="2020-01-01")
        assert r["@confidence"] == 0.9
        assert r["@source"] == "model-v1"
        assert r["@validFrom"] == "2020-01-01"

    def test_wraps_plain_value(self):
        r = add_temporal(42, valid_from="2020-01-01")
        assert r["@value"] == 42

    def test_no_qualifiers_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            add_temporal("X")

    def test_invalid_timestamp_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            add_temporal("X", valid_from="not-a-date")

    def test_from_after_until_raises(self):
        with pytest.raises(ValueError, match="must not be after"):
            add_temporal("X", valid_from="2025-01-01", valid_until="2020-01-01")

    def test_from_equals_until_ok(self):
        """Same start and end (point-in-time) is valid."""
        r = add_temporal("X", valid_from="2025-01-01", valid_until="2025-01-01")
        assert r["@validFrom"] == "2025-01-01"

    def test_non_string_timestamp_raises(self):
        with pytest.raises(TypeError, match="must be a string"):
            add_temporal("X", valid_from=20200101)  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# query_at_time
# ═══════════════════════════════════════════════════════════════════


_GRAPH = [
    {
        "@id": "ex:alice",
        "@type": "Person",
        "jobTitle": {
            "@value": "Engineer",
            "@confidence": 0.9,
            "@validFrom": "2020-01-01T00:00:00Z",
            "@validUntil": "2023-12-31T23:59:59Z",
        },
    },
    {
        "@id": "ex:alice",
        "@type": "Person",
        "jobTitle": {
            "@value": "Manager",
            "@confidence": 0.85,
            "@validFrom": "2024-01-01T00:00:00Z",
        },
    },
    {
        "@id": "ex:bob",
        "@type": "Person",
        "name": {"@value": "Bob", "@confidence": 0.95},  # no temporal → always valid
    },
]


class TestQueryAtTime:
    def test_during_engineer_period(self):
        results = query_at_time(_GRAPH, "2022-06-15T00:00:00Z")
        titles = [
            n["jobTitle"]["@value"]
            for n in results if "jobTitle" in n
        ]
        assert "Engineer" in titles
        assert "Manager" not in titles

    def test_during_manager_period(self):
        results = query_at_time(_GRAPH, "2025-01-15T00:00:00Z")
        titles = [
            n["jobTitle"]["@value"]
            for n in results if "jobTitle" in n
        ]
        assert "Manager" in titles
        assert "Engineer" not in titles

    def test_always_valid_nodes_included(self):
        results = query_at_time(_GRAPH, "2022-06-15T00:00:00Z")
        bobs = [n for n in results if n.get("@id") == "ex:bob"]
        assert len(bobs) == 1

    def test_before_all_periods(self):
        results = query_at_time(_GRAPH, "2019-01-01T00:00:00Z")
        titles = [
            n["jobTitle"]["@value"]
            for n in results if "jobTitle" in n
        ]
        assert titles == []

    def test_at_boundary_from(self):
        """Exact validFrom timestamp should be included."""
        results = query_at_time(_GRAPH, "2020-01-01T00:00:00Z")
        titles = [n["jobTitle"]["@value"] for n in results if "jobTitle" in n]
        assert "Engineer" in titles

    def test_at_boundary_until(self):
        """Exact validUntil timestamp should be included."""
        results = query_at_time(_GRAPH, "2023-12-31T23:59:59Z")
        titles = [n["jobTitle"]["@value"] for n in results if "jobTitle" in n]
        assert "Engineer" in titles

    def test_filter_specific_property(self):
        """When property_name is given, only filter that property."""
        graph = [
            {
                "@id": "ex:alice",
                "jobTitle": {"@value": "Eng", "@validFrom": "2020-01-01", "@validUntil": "2023-12-31"},
                "name": {"@value": "Alice", "@confidence": 0.9},
            },
        ]
        results = query_at_time(graph, "2019-06-01", property_name="jobTitle")
        alice = results[0] if results else None
        # jobTitle should be excluded (before validFrom), but name stays
        assert alice is not None
        assert "jobTitle" not in alice
        assert alice["name"]["@value"] == "Alice"

    def test_empty_graph(self):
        assert query_at_time([], "2025-01-01") == []


# Multi-valued property (list)
class TestQueryAtTimeMultiValue:
    def test_list_property_filtered(self):
        graph = [
            {
                "@id": "ex:x",
                "tags": [
                    {"@value": "old-tag", "@validFrom": "2020-01-01", "@validUntil": "2022-12-31"},
                    {"@value": "new-tag", "@validFrom": "2023-01-01"},
                    {"@value": "always-tag"},
                ],
            },
        ]
        results = query_at_time(graph, "2024-06-01")
        tags = results[0]["tags"]
        if isinstance(tags, list):
            vals = [t["@value"] if isinstance(t, dict) else t for t in tags]
        else:
            vals = [tags["@value"] if isinstance(tags, dict) else tags]
        assert "new-tag" in vals
        assert "always-tag" in vals
        assert "old-tag" not in vals


# ═══════════════════════════════════════════════════════════════════
# temporal_diff
# ═══════════════════════════════════════════════════════════════════


class TestTemporalDiff:
    def test_value_change_detected(self):
        graph = [
            {
                "@id": "ex:alice",
                "@type": "Person",
                "jobTitle": {
                    "@value": "Engineer",
                    "@validFrom": "2020-01-01",
                    "@validUntil": "2023-12-31",
                },
            },
            {
                "@id": "ex:alice",
                "@type": "Person",
                "jobTitle": {
                    "@value": "Manager",
                    "@validFrom": "2024-01-01",
                },
            },
        ]
        # We need to query the graph flat — both nodes represent alice at different times.
        # For temporal_diff to work, we build a single-node graph with list values:
        graph_combined = [
            {
                "@id": "ex:alice",
                "@type": "Person",
                "jobTitle": [
                    {"@value": "Engineer", "@validFrom": "2020-01-01", "@validUntil": "2023-12-31"},
                    {"@value": "Manager", "@validFrom": "2024-01-01"},
                ],
            },
        ]
        d = temporal_diff(graph_combined, "2022-06-01", "2025-01-01")
        # Engineer → Manager is a modification
        assert len(d.modified) == 1
        assert d.modified[0]["value_at_t1"]["@value"] == "Engineer"
        assert d.modified[0]["value_at_t2"]["@value"] == "Manager"

    def test_property_added_over_time(self):
        graph = [
            {
                "@id": "ex:alice",
                "name": "Alice",
                "title": {"@value": "Director", "@validFrom": "2025-01-01"},
            },
        ]
        d = temporal_diff(graph, "2024-06-01", "2025-06-01")
        added_props = [e for e in d.added if "property" in e]
        assert any(e["property"] == "title" for e in added_props)

    def test_property_removed_over_time(self):
        graph = [
            {
                "@id": "ex:alice",
                "name": "Alice",
                "temp_badge": {"@value": "V-123", "@validFrom": "2024-01-01", "@validUntil": "2024-06-30"},
            },
        ]
        d = temporal_diff(graph, "2024-03-01", "2025-01-01")
        removed_props = [e for e in d.removed if "property" in e]
        assert any(e["property"] == "temp_badge" for e in removed_props)

    def test_no_changes(self):
        graph = [
            {"@id": "ex:alice", "name": "Alice"},
        ]
        d = temporal_diff(graph, "2024-01-01", "2025-01-01")
        assert len(d.added) == 0
        assert len(d.removed) == 0
        assert len(d.modified) == 0
        assert len(d.unchanged) == 1

    def test_empty_graph(self):
        d = temporal_diff([], "2024-01-01", "2025-01-01")
        assert d.added == []
        assert d.removed == []

    def test_node_appears_at_t2(self):
        graph = [
            {
                "@id": "ex:new",
                "name": {"@value": "New Entity", "@validFrom": "2025-06-01"},
            },
        ]
        d = temporal_diff(graph, "2024-01-01", "2026-01-01")
        added_nodes = [e for e in d.added if "state" in e]
        assert any(e["@id"] == "ex:new" for e in added_nodes)


# ═══════════════════════════════════════════════════════════════════
# Timestamp parsing edge cases
# ═══════════════════════════════════════════════════════════════════


class TestTimestampParsing:
    def test_date_only(self):
        r = add_temporal("X", valid_from="2025-01-15")
        assert r["@validFrom"] == "2025-01-15"

    def test_datetime_with_z(self):
        r = add_temporal("X", valid_from="2025-01-15T10:30:00Z")
        assert r["@validFrom"] == "2025-01-15T10:30:00Z"

    def test_datetime_with_offset(self):
        r = add_temporal("X", valid_from="2025-01-15T10:30:00+05:30")
        assert r["@validFrom"] == "2025-01-15T10:30:00+05:30"

    def test_datetime_with_milliseconds(self):
        r = add_temporal("X", valid_from="2025-01-15T10:30:00.123Z")
        assert r["@validFrom"] == "2025-01-15T10:30:00.123Z"
