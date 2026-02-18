"""Regression tests for data_rights.py bugs found in external review.

Bug 1: export_portable() merges distinct nodes when @id is missing.
Bug 2: right_of_access_report() has the same node-collision bug.
Bug 3: request_restriction() stores caller's list by reference (aliasing).
Bug 4: export_portable() docstring says "personal data" but exports all
        subject-tagged properties regardless of is_personal_data().

RED PHASE: Tests written before fixes are applied.

References:
    - GDPR Art. 15 (Right of Access)
    - GDPR Art. 18 (Right to Restriction of Processing)
    - GDPR Art. 20 (Right to Data Portability)
"""

import pytest

from jsonld_ex.data_protection import annotate_protection
from jsonld_ex.data_rights import (
    export_portable,
    right_of_access_report,
    request_restriction,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _graph_with_id_less_nodes():
    """Two distinct nodes belonging to the same subject, neither has @id.

    Node A holds a name property; node B holds a diagnosis property.
    They are separate graph entries and must remain separate records
    in any export or report.
    """
    return [
        {
            # No @id
            "@type": "ContactInfo",
            "name": annotate_protection(
                "Alice Smith",
                personal_data_category="regular",
                legal_basis="consent",
                data_subject="ex:subjects/alice",
            ),
        },
        {
            # No @id
            "@type": "MedicalRecord",
            "diagnosis": annotate_protection(
                "common cold",
                personal_data_category="sensitive",
                legal_basis="vital_interest",
                data_subject="ex:subjects/alice",
            ),
        },
    ]


def _graph_with_mixed_ids():
    """Three nodes: two without @id (distinct types), one with @id.

    This tests that the fix handles the mix correctly — nodes WITH @id
    still group by their @id, and nodes WITHOUT @id are kept separate.
    """
    return [
        {
            "@id": "ex:record/1",
            "@type": "Person",
            "email": annotate_protection(
                "alice@example.com",
                personal_data_category="regular",
                legal_basis="consent",
                data_subject="ex:subjects/alice",
            ),
        },
        {
            # No @id
            "@type": "LabResult",
            "result": annotate_protection(
                "positive",
                personal_data_category="sensitive",
                legal_basis="vital_interest",
                data_subject="ex:subjects/alice",
            ),
        },
        {
            # No @id
            "@type": "Prescription",
            "medication": annotate_protection(
                "amoxicillin",
                personal_data_category="regular",
                legal_basis="contract",
                data_subject="ex:subjects/alice",
            ),
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# Bug 1: export_portable() node collision on missing @id
# ═══════════════════════════════════════════════════════════════════


class TestExportPortableNodeCollision:
    """export_portable() must not merge distinct nodes that lack @id."""

    def test_two_id_less_nodes_produce_two_records(self):
        """Each node without @id must appear as a separate record."""
        graph = _graph_with_id_less_nodes()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        assert len(export.records) == 2, (
            f"Expected 2 records for 2 distinct nodes, got {len(export.records)}"
        )

    def test_id_less_records_have_distinct_types(self):
        """The two separate records should preserve their respective @type."""
        graph = _graph_with_id_less_nodes()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        types = {r["type"] for r in export.records}
        assert "ContactInfo" in types
        assert "MedicalRecord" in types

    def test_id_less_records_have_distinct_properties(self):
        """Each record should have only its own properties, not merged ones."""
        graph = _graph_with_id_less_nodes()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        for rec in export.records:
            props = rec["properties"]
            # Each record should have exactly 1 property, not 2
            assert len(props) == 1, (
                f"Record {rec['type']} has {len(props)} properties "
                f"(expected 1): {list(props.keys())}"
            )

    def test_mixed_id_and_id_less_nodes(self):
        """Nodes with @id group normally; nodes without @id stay separate."""
        graph = _graph_with_mixed_ids()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        # 1 node with @id + 2 nodes without @id = 3 records
        assert len(export.records) == 3

    def test_values_not_overwritten_by_collision(self):
        """Verify no data loss: each value appears in exactly one record."""
        graph = _graph_with_id_less_nodes()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        all_values = []
        for rec in export.records:
            for v in rec["properties"].values():
                all_values.append(v)
        assert "Alice Smith" in all_values
        assert "common cold" in all_values


# ═══════════════════════════════════════════════════════════════════
# Bug 2: right_of_access_report() same node-collision bug
# ═══════════════════════════════════════════════════════════════════


class TestAccessReportNodeCollision:
    """right_of_access_report() must not merge distinct nodes that lack @id."""

    def test_two_id_less_nodes_produce_two_records(self):
        graph = _graph_with_id_less_nodes()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert len(report.records) == 2, (
            f"Expected 2 records for 2 distinct nodes, got {len(report.records)}"
        )

    def test_id_less_records_have_distinct_types(self):
        graph = _graph_with_id_less_nodes()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        types = {r["type"] for r in report.records}
        assert "ContactInfo" in types
        assert "MedicalRecord" in types

    def test_id_less_records_have_distinct_properties(self):
        graph = _graph_with_id_less_nodes()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        for rec in report.records:
            props = rec["properties"]
            assert len(props) == 1, (
                f"Record {rec['type']} has {len(props)} properties "
                f"(expected 1): {list(props.keys())}"
            )

    def test_mixed_id_and_id_less_nodes(self):
        graph = _graph_with_mixed_ids()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert len(report.records) == 3

    def test_property_count_accurate(self):
        """Total property count must reflect all properties, not collapsed ones."""
        graph = _graph_with_id_less_nodes()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert report.total_property_count == 2

    def test_categories_accurate_across_id_less_nodes(self):
        """Categories should reflect both nodes even when @id is absent."""
        graph = _graph_with_id_less_nodes()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert "regular" in report.categories
        assert "sensitive" in report.categories


# ═══════════════════════════════════════════════════════════════════
# Bug 3: request_restriction() aliasing bug
# ═══════════════════════════════════════════════════════════════════


class TestRestrictionAliasing:
    """request_restriction() must not store the caller's list by reference."""

    def test_mutation_after_call_does_not_affect_graph(self):
        """Mutating the input list after the call must not change the graph."""
        graph = _graph_with_id_less_nodes()
        restrictions = ["profiling", "automated_decision"]
        request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Accuracy contested",
            processing_restrictions=restrictions,
        )

        # Mutate the caller's list AFTER the call
        restrictions.append("marketing")
        restrictions[0] = "TAMPERED"

        # Graph entries should still have the original values
        for node in graph:
            for key, prop in node.items():
                if key.startswith("@"):
                    continue
                values = prop if isinstance(prop, list) else [prop]
                for v in values:
                    if isinstance(v, dict) and "@processingRestrictions" in v:
                        stored = v["@processingRestrictions"]
                        assert "marketing" not in stored, (
                            "Caller mutation leaked into graph"
                        )
                        assert "TAMPERED" not in stored, (
                            "Caller mutation leaked into graph"
                        )
                        assert stored == ["profiling", "automated_decision"]

    def test_each_property_gets_independent_copy(self):
        """Each property should get its own independent copy of the list."""
        graph = _graph_with_id_less_nodes()
        restrictions = ["profiling"]
        request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Test",
            processing_restrictions=restrictions,
        )

        # Collect all stored restriction lists
        stored_lists = []
        for node in graph:
            for key, prop in node.items():
                if key.startswith("@"):
                    continue
                values = prop if isinstance(prop, list) else [prop]
                for v in values:
                    if isinstance(v, dict) and "@processingRestrictions" in v:
                        stored_lists.append(v["@processingRestrictions"])

        assert len(stored_lists) == 2  # Two properties for alice

        # Mutating one stored list must not affect the other
        stored_lists[0].append("INJECTED")
        assert "INJECTED" not in stored_lists[1], (
            "Restriction lists are aliased between properties"
        )
