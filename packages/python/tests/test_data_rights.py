"""Tests for data_rights module — Phase 2: Data Subject Rights & Compliance Operations.

TDD: Write tests FIRST (red phase), then implement (green phase).

These operations transform graphs and produce reports. They complement
the annotation fields in data_protection.py.
"""

import pytest
from datetime import datetime, timezone

from jsonld_ex.data_protection import annotate_protection, create_consent_record


# Import the module under test — these don't exist yet (red phase).
from jsonld_ex.data_rights import (
    request_erasure,
    execute_erasure,
    request_restriction,
    export_portable,
    rectify_data,
    right_of_access_report,
    validate_retention,
    audit_trail,
    # Data structures
    ErasurePlan,
    ErasureAudit,
    RestrictionResult,
    PortableExport,
    AccessReport,
    RetentionViolation,
    AuditEntry,
)


# ── Helpers ───────────────────────────────────────────────────────

def _make_graph():
    """Create a realistic test graph with multiple data subjects."""
    return [
        {
            "@id": "ex:record/1",
            "@type": "Person",
            "name": annotate_protection(
                "Alice Smith",
                personal_data_category="regular",
                legal_basis="consent",
                data_subject="ex:subjects/alice",
                data_controller="ex:corp/acme",
                jurisdiction="EU",
                retention_until="2025-01-01T00:00:00Z",
            ),
            "email": annotate_protection(
                "alice@example.com",
                personal_data_category="regular",
                legal_basis="consent",
                data_subject="ex:subjects/alice",
                data_controller="ex:corp/acme",
            ),
        },
        {
            "@id": "ex:record/2",
            "@type": "Person",
            "name": annotate_protection(
                "Bob Jones",
                personal_data_category="regular",
                legal_basis="contract",
                data_subject="ex:subjects/bob",
                data_controller="ex:corp/acme",
            ),
        },
        {
            "@id": "ex:record/3",
            "@type": "MedicalRecord",
            "diagnosis": annotate_protection(
                "common cold",
                personal_data_category="sensitive",
                legal_basis="vital_interest",
                data_subject="ex:subjects/alice",
                data_controller="ex:corp/hospital",
                retention_until="2030-06-15T00:00:00Z",
            ),
        },
        {
            "@id": "ex:record/4",
            "@type": "SensorReading",
            "temperature": annotate_protection(
                22.5,
                personal_data_category="non_personal",
            ),
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# request_erasure()
# ═══════════════════════════════════════════════════════════════════


class TestRequestErasure:
    """Tests for request_erasure() — GDPR Art. 17 Right to Erasure."""

    def test_marks_all_subject_nodes(self):
        """All nodes belonging to the data subject are marked."""
        graph = _make_graph()
        plan = request_erasure(graph, data_subject="ex:subjects/alice")
        assert isinstance(plan, ErasurePlan)
        # Alice has data in record/1 (name, email) and record/3 (diagnosis)
        assert len(plan.affected_node_ids) == 2
        assert "ex:record/1" in plan.affected_node_ids
        assert "ex:record/3" in plan.affected_node_ids

    def test_plan_includes_property_count(self):
        graph = _make_graph()
        plan = request_erasure(graph, data_subject="ex:subjects/alice")
        # record/1 has 2 properties (name, email), record/3 has 1 (diagnosis)
        assert plan.affected_property_count == 3

    def test_marks_erasure_requested_on_properties(self):
        """After request, affected property values have @erasureRequested=True."""
        graph = _make_graph()
        plan = request_erasure(graph, data_subject="ex:subjects/alice")
        # Check that the graph was mutated in place
        rec1 = graph[0]
        assert rec1["name"]["@erasureRequested"] is True
        assert rec1["email"]["@erasureRequested"] is True

    def test_sets_erasure_requested_at(self):
        graph = _make_graph()
        plan = request_erasure(
            graph,
            data_subject="ex:subjects/alice",
            requested_at="2026-02-13T10:00:00Z",
        )
        rec1 = graph[0]
        assert rec1["name"]["@erasureRequestedAt"] == "2026-02-13T10:00:00Z"

    def test_no_matching_subject(self):
        graph = _make_graph()
        plan = request_erasure(graph, data_subject="ex:subjects/unknown")
        assert len(plan.affected_node_ids) == 0
        assert plan.affected_property_count == 0

    def test_does_not_affect_other_subjects(self):
        """Bob's data should be untouched."""
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        bob_rec = graph[1]
        assert "@erasureRequested" not in bob_rec["name"]

    def test_does_not_affect_non_personal_data(self):
        """Non-personal data nodes are untouched."""
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        sensor = graph[3]
        assert "@erasureRequested" not in sensor["temperature"]


# ═══════════════════════════════════════════════════════════════════
# execute_erasure()
# ═══════════════════════════════════════════════════════════════════


class TestExecuteErasure:
    """Tests for execute_erasure() — actually removes or anonymizes data."""

    def test_removes_marked_properties(self):
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        audit = execute_erasure(graph, data_subject="ex:subjects/alice")
        assert isinstance(audit, ErasureAudit)
        # record/1: name and email should be erased
        rec1 = graph[0]
        assert rec1["name"]["@value"] is None
        assert rec1["email"]["@value"] is None

    def test_sets_erasure_completed_at(self):
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        audit = execute_erasure(
            graph,
            data_subject="ex:subjects/alice",
            completed_at="2026-02-20T10:00:00Z",
        )
        rec1 = graph[0]
        assert rec1["name"]["@erasureCompletedAt"] == "2026-02-20T10:00:00Z"

    def test_audit_contains_erased_count(self):
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        audit = execute_erasure(graph, data_subject="ex:subjects/alice")
        assert audit.erased_property_count == 3
        assert len(audit.erased_node_ids) == 2

    def test_skips_nodes_not_marked_for_erasure(self):
        """Only nodes with @erasureRequested=True are erased."""
        graph = _make_graph()
        # Don't call request_erasure first — nothing is marked
        audit = execute_erasure(graph, data_subject="ex:subjects/alice")
        assert audit.erased_property_count == 0

    def test_bob_untouched_after_alice_erasure(self):
        graph = _make_graph()
        request_erasure(graph, data_subject="ex:subjects/alice")
        execute_erasure(graph, data_subject="ex:subjects/alice")
        bob = graph[1]
        assert bob["name"]["@value"] == "Bob Jones"


# ═══════════════════════════════════════════════════════════════════
# request_restriction()
# ═══════════════════════════════════════════════════════════════════


class TestRequestRestriction:
    """Tests for request_restriction() — GDPR Art. 18."""

    def test_marks_restriction(self):
        graph = _make_graph()
        result = request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Accuracy contested",
        )
        assert isinstance(result, RestrictionResult)
        rec1 = graph[0]
        assert rec1["name"]["@restrictProcessing"] is True
        assert rec1["name"]["@restrictionReason"] == "Accuracy contested"

    def test_restriction_count(self):
        graph = _make_graph()
        result = request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Legal dispute",
        )
        # Alice: record/1 (name, email) + record/3 (diagnosis) = 3 props
        assert result.restricted_property_count == 3

    def test_no_matching_subject(self):
        graph = _make_graph()
        result = request_restriction(
            graph,
            data_subject="ex:subjects/unknown",
            reason="test",
        )
        assert result.restricted_property_count == 0

    def test_with_processing_restrictions_list(self):
        graph = _make_graph()
        result = request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Accuracy contested",
            processing_restrictions=["profiling", "automated_decision"],
        )
        rec1 = graph[0]
        assert rec1["name"]["@processingRestrictions"] == [
            "profiling",
            "automated_decision",
        ]


# ═══════════════════════════════════════════════════════════════════
# export_portable()
# ═══════════════════════════════════════════════════════════════════


class TestExportPortable:
    """Tests for export_portable() — GDPR Art. 20 Right to Data Portability."""

    def test_exports_subject_data_as_json(self):
        graph = _make_graph()
        export = export_portable(
            graph,
            data_subject="ex:subjects/alice",
            format="json",
        )
        assert isinstance(export, PortableExport)
        assert export.format == "json"
        assert export.data_subject == "ex:subjects/alice"
        # Should contain Alice's data from both records
        assert len(export.records) == 2

    def test_export_includes_values(self):
        graph = _make_graph()
        export = export_portable(
            graph,
            data_subject="ex:subjects/alice",
            format="json",
        )
        # Records should contain the actual data values
        values = []
        for rec in export.records:
            for prop_name, prop_val in rec.get("properties", {}).items():
                values.append(prop_val)
        assert "Alice Smith" in values
        assert "alice@example.com" in values

    def test_export_no_matching_subject(self):
        graph = _make_graph()
        export = export_portable(
            graph,
            data_subject="ex:subjects/unknown",
            format="json",
        )
        assert len(export.records) == 0

    def test_export_excludes_non_personal(self):
        """Non-personal data is not included in portability exports."""
        graph = _make_graph()
        export = export_portable(
            graph,
            data_subject="ex:subjects/alice",
            format="json",
        )
        # Sensor reading (non-personal) should not appear
        node_ids = [r["node_id"] for r in export.records]
        assert "ex:record/4" not in node_ids


# ═══════════════════════════════════════════════════════════════════
# rectify_data()
# ═══════════════════════════════════════════════════════════════════


class TestRectifyData:
    """Tests for rectify_data() — GDPR Art. 16 Right to Rectification."""

    def test_updates_value(self):
        node = annotate_protection(
            "John Doe",
            personal_data_category="regular",
            data_subject="ex:subjects/john",
        )
        result = rectify_data(
            node,
            new_value="Jane Doe",
            note="Name corrected after marriage",
        )
        assert result["@value"] == "Jane Doe"

    def test_sets_rectification_metadata(self):
        node = annotate_protection("Old Name", personal_data_category="regular")
        result = rectify_data(
            node,
            new_value="New Name",
            note="Typo corrected",
            rectified_at="2026-02-13T12:00:00Z",
        )
        assert result["@rectifiedAt"] == "2026-02-13T12:00:00Z"
        assert result["@rectificationNote"] == "Typo corrected"

    def test_preserves_existing_annotations(self):
        """Rectification preserves all existing protection annotations."""
        node = annotate_protection(
            "Old Value",
            personal_data_category="sensitive",
            legal_basis="consent",
            jurisdiction="EU",
            data_subject="ex:subjects/alice",
        )
        result = rectify_data(node, new_value="New Value", note="corrected")
        assert result["@personalDataCategory"] == "sensitive"
        assert result["@legalBasis"] == "consent"
        assert result["@jurisdiction"] == "EU"
        assert result["@dataSubject"] == "ex:subjects/alice"

    def test_returns_new_dict(self):
        """rectify_data returns a new dict, does not mutate the original."""
        node = annotate_protection("Old", personal_data_category="regular")
        result = rectify_data(node, new_value="New", note="fix")
        assert node["@value"] == "Old"  # Original unchanged
        assert result["@value"] == "New"


# ═══════════════════════════════════════════════════════════════════
# right_of_access_report()
# ═══════════════════════════════════════════════════════════════════


class TestRightOfAccessReport:
    """Tests for right_of_access_report() — GDPR Art. 15."""

    def test_report_structure(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert isinstance(report, AccessReport)
        assert report.data_subject == "ex:subjects/alice"
        assert len(report.records) == 2  # record/1 and record/3

    def test_report_includes_categories(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        # Should include both regular and sensitive categories
        assert "regular" in report.categories
        assert "sensitive" in report.categories

    def test_report_includes_controllers(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert "ex:corp/acme" in report.controllers
        assert "ex:corp/hospital" in report.controllers

    def test_report_includes_legal_bases(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert "consent" in report.legal_bases
        assert "vital_interest" in report.legal_bases

    def test_report_includes_jurisdictions(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert "EU" in report.jurisdictions

    def test_report_no_matching_subject(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/unknown")
        assert len(report.records) == 0
        assert len(report.categories) == 0

    def test_report_property_count(self):
        graph = _make_graph()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert report.total_property_count == 3  # name, email, diagnosis


# ═══════════════════════════════════════════════════════════════════
# validate_retention()
# ═══════════════════════════════════════════════════════════════════


class TestValidateRetention:
    """Tests for validate_retention() — retention deadline compliance."""

    def test_finds_expired_retention(self):
        graph = _make_graph()
        # record/1 has retention_until=2025-01-01 — already expired
        violations = validate_retention(
            graph,
            as_of="2026-02-13T00:00:00Z",
        )
        assert len(violations) >= 1
        expired_ids = [v.node_id for v in violations]
        assert "ex:record/1" in expired_ids

    def test_violation_includes_details(self):
        graph = _make_graph()
        violations = validate_retention(graph, as_of="2026-02-13T00:00:00Z")
        v = next(v for v in violations if v.node_id == "ex:record/1")
        assert isinstance(v, RetentionViolation)
        assert v.retention_until == "2025-01-01T00:00:00Z"
        assert v.property_name == "name"  # first expired property found

    def test_no_violations_before_deadline(self):
        graph = _make_graph()
        # Check at a time before any retention expires
        violations = validate_retention(graph, as_of="2024-06-01T00:00:00Z")
        assert len(violations) == 0

    def test_nodes_without_retention_are_fine(self):
        """Nodes without @retentionUntil don't produce violations."""
        graph = _make_graph()
        violations = validate_retention(graph, as_of="2026-02-13T00:00:00Z")
        violated_ids = [v.node_id for v in violations]
        # Bob (record/2) has no retention — should not appear
        assert "ex:record/2" not in violated_ids

    def test_future_retention_not_violated(self):
        graph = _make_graph()
        # record/3 has retention_until=2030-06-15 — still valid in 2026
        violations = validate_retention(graph, as_of="2026-02-13T00:00:00Z")
        violated_ids = [v.node_id for v in violations]
        assert "ex:record/3" not in violated_ids


# ═══════════════════════════════════════════════════════════════════
# audit_trail()
# ═══════════════════════════════════════════════════════════════════


class TestAuditTrail:
    """Tests for audit_trail() — complete processing history for a data subject."""

    def test_basic_audit_trail(self):
        graph = _make_graph()
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        assert isinstance(entries, list)
        assert all(isinstance(e, AuditEntry) for e in entries)
        # Should have entries for all Alice's properties
        assert len(entries) >= 2  # At least record/1 and record/3

    def test_audit_entry_fields(self):
        graph = _make_graph()
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        entry = entries[0]
        assert entry.node_id is not None
        assert entry.property_name is not None
        assert entry.data_subject == "ex:subjects/alice"

    def test_audit_reflects_erasure_request(self):
        """Audit trail captures erasure request events."""
        graph = _make_graph()
        request_erasure(
            graph,
            data_subject="ex:subjects/alice",
            requested_at="2026-02-13T10:00:00Z",
        )
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        # At least some entries should show erasure_requested
        erasure_entries = [e for e in entries if e.erasure_requested is True]
        assert len(erasure_entries) >= 1

    def test_audit_reflects_restriction(self):
        graph = _make_graph()
        request_restriction(
            graph,
            data_subject="ex:subjects/alice",
            reason="Legal dispute",
        )
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        restricted = [e for e in entries if e.restrict_processing is True]
        assert len(restricted) >= 1

    def test_audit_no_matching_subject(self):
        graph = _make_graph()
        entries = audit_trail(graph, data_subject="ex:subjects/unknown")
        assert entries == []

    def test_audit_includes_legal_basis(self):
        graph = _make_graph()
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        bases = {e.legal_basis for e in entries if e.legal_basis}
        assert "consent" in bases
