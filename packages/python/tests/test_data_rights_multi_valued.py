"""Regression tests for multi-valued property collapse in data_rights.py.

Bug: export_portable() and right_of_access_report() store extracted
values into a dict keyed by property name:

    node_props[key]["properties"][prop_name] = prop_val.get("@value")

If a node contains a list for the same property (e.g. two ``email``
entries both tagged to the same data subject), earlier values are
silently overwritten by the last one.

Impact:
  - Data loss in portability output (GDPR Art. 20 violation —
    incomplete data export).
  - AccessReport.total_property_count can exceed the number of
    actual property entries in records[*].properties, producing
    internal inconsistency (GDPR Art. 15 — inaccurate report).

Why earlier tests missed it: All existing tests use only single-value-
per-property cases.  No test has a list of annotated values under the
same property name within a single node.

RED PHASE: Tests written before fix.

References:
    - GDPR Art. 15 (Right of Access)
    - GDPR Art. 20 (Right to Data Portability)
"""

import pytest

from jsonld_ex.data_protection import annotate_protection
from jsonld_ex.data_rights import (
    export_portable,
    right_of_access_report,
    audit_trail,
)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _graph_with_multi_valued_property():
    """A single node where 'email' is a list of two annotated values.

    Both emails belong to the same data subject.  FHIR-style clinical
    data and JSON-LD graphs commonly use lists for repeating elements
    (e.g. multiple telecom entries, multiple addresses).
    """
    return [
        {
            "@id": "ex:record/1",
            "@type": "Person",
            "email": [
                annotate_protection(
                    "alice@work.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                    data_controller="ex:corp/acme",
                ),
                annotate_protection(
                    "alice@personal.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                    data_controller="ex:corp/acme",
                ),
            ],
            "name": annotate_protection(
                "Alice Smith",
                personal_data_category="regular",
                legal_basis="consent",
                data_subject="ex:subjects/alice",
                data_controller="ex:corp/acme",
            ),
        },
    ]


def _graph_with_multi_valued_mixed_subjects():
    """A node where 'email' has three entries: two for alice, one for bob.

    Only alice's entries should appear in alice's export/report.
    This also verifies that subject filtering still works correctly
    when properties are lists with mixed subjects.
    """
    return [
        {
            "@id": "ex:record/shared",
            "@type": "ContactList",
            "email": [
                annotate_protection(
                    "alice@work.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
                annotate_protection(
                    "bob@work.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/bob",
                ),
                annotate_protection(
                    "alice@personal.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
            ],
        },
    ]


def _graph_with_multiple_multi_valued_properties():
    """A node with two multi-valued properties: 'email' (2 entries)
    and 'phone' (2 entries), all for the same subject.

    Verifies that the fix handles multiple repeated properties
    within the same node, not just one.
    """
    return [
        {
            "@id": "ex:record/1",
            "@type": "Person",
            "email": [
                annotate_protection(
                    "alice@work.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
                annotate_protection(
                    "alice@personal.com",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
            ],
            "phone": [
                annotate_protection(
                    "+1-555-0100",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
                annotate_protection(
                    "+1-555-0200",
                    personal_data_category="regular",
                    legal_basis="consent",
                    data_subject="ex:subjects/alice",
                ),
            ],
        },
    ]


# ═══════════════════════════════════════════════════════════════════
# Bug 5: export_portable() multi-valued property collapse
# ═══════════════════════════════════════════════════════════════════


class TestExportPortableMultiValuedCollapse:
    """export_portable() must not discard earlier values when a property
    appears multiple times within a single node."""

    def test_both_emails_present_in_export(self):
        """Two emails on the same node must both appear in the export."""
        graph = _graph_with_multi_valued_property()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        assert len(export.records) == 1  # one node → one record

        record = export.records[0]
        email_values = record["properties"]["email"]
        assert isinstance(email_values, list), (
            f"Multi-valued property should be a list, got {type(email_values).__name__}"
        )
        assert len(email_values) == 2, (
            f"Expected 2 email values, got {len(email_values)}"
        )
        assert "alice@work.com" in email_values
        assert "alice@personal.com" in email_values

    def test_single_valued_property_unchanged(self):
        """A single-valued property should still work as before."""
        graph = _graph_with_multi_valued_property()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        record = export.records[0]
        # 'name' is single-valued — should be a plain value, not a list
        name_value = record["properties"]["name"]
        assert name_value == "Alice Smith"

    def test_mixed_subjects_only_includes_target(self):
        """When a list has entries for different subjects, only the target's
        values should appear in the export."""
        graph = _graph_with_multi_valued_mixed_subjects()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        assert len(export.records) == 1

        record = export.records[0]
        email_values = record["properties"]["email"]
        assert isinstance(email_values, list)
        assert len(email_values) == 2
        assert "alice@work.com" in email_values
        assert "alice@personal.com" in email_values
        assert "bob@work.com" not in email_values

    def test_multiple_multi_valued_properties(self):
        """Multiple repeated properties on the same node must all be preserved."""
        graph = _graph_with_multiple_multi_valued_properties()
        export = export_portable(graph, data_subject="ex:subjects/alice")
        assert len(export.records) == 1

        record = export.records[0]
        email_values = record["properties"]["email"]
        phone_values = record["properties"]["phone"]
        assert isinstance(email_values, list)
        assert isinstance(phone_values, list)
        assert len(email_values) == 2
        assert len(phone_values) == 2
        assert "+1-555-0100" in phone_values
        assert "+1-555-0200" in phone_values


# ═══════════════════════════════════════════════════════════════════
# Bug 5: right_of_access_report() multi-valued property collapse
# ═══════════════════════════════════════════════════════════════════


class TestAccessReportMultiValuedCollapse:
    """right_of_access_report() must not discard earlier values when a
    property appears multiple times within a single node."""

    def test_both_emails_present_in_report(self):
        """Two emails on the same node must both appear in the report."""
        graph = _graph_with_multi_valued_property()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert len(report.records) == 1

        record = report.records[0]
        email_values = record["properties"]["email"]
        assert isinstance(email_values, list)
        assert len(email_values) == 2
        assert "alice@work.com" in email_values
        assert "alice@personal.com" in email_values

    def test_total_property_count_matches_actual_entries(self):
        """total_property_count must equal the actual number of property
        values represented in the records — no internal inconsistency.

        Before the fix, total_property_count was 3 (name + 2 emails)
        but the properties dict only had 2 keys (name, email) with
        email silently collapsed to one value.
        """
        graph = _graph_with_multi_valued_property()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")

        # Count actual property values in records
        actual_count = 0
        for rec in report.records:
            for value in rec["properties"].values():
                if isinstance(value, list):
                    actual_count += len(value)
                else:
                    actual_count += 1

        assert report.total_property_count == actual_count, (
            f"total_property_count={report.total_property_count} but "
            f"actual entries in records={actual_count}"
        )

    def test_total_property_count_is_three(self):
        """Explicitly: 1 name + 2 emails = 3 properties total."""
        graph = _graph_with_multi_valued_property()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert report.total_property_count == 3

    def test_mixed_subjects_count_correct(self):
        """Only alice's 2 emails should count, not bob's."""
        graph = _graph_with_multi_valued_mixed_subjects()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert report.total_property_count == 2

        record = report.records[0]
        email_values = record["properties"]["email"]
        assert isinstance(email_values, list)
        assert len(email_values) == 2

    def test_single_valued_property_unchanged(self):
        """Single-valued properties must still work as before."""
        graph = _graph_with_multi_valued_property()
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        record = report.records[0]
        name_value = record["properties"]["name"]
        assert name_value == "Alice Smith"

    def test_categories_collected_from_all_values(self):
        """Metadata should be collected from every value in a list,
        not just the last one."""
        graph = [
            {
                "@id": "ex:record/1",
                "@type": "Person",
                "contact": [
                    annotate_protection(
                        "alice@work.com",
                        personal_data_category="regular",
                        legal_basis="consent",
                        data_subject="ex:subjects/alice",
                    ),
                    annotate_protection(
                        "alice-medical@hospital.com",
                        personal_data_category="sensitive",
                        legal_basis="vital_interest",
                        data_subject="ex:subjects/alice",
                    ),
                ],
            },
        ]
        report = right_of_access_report(graph, data_subject="ex:subjects/alice")
        assert "regular" in report.categories
        assert "sensitive" in report.categories
        assert "consent" in report.legal_bases
        assert "vital_interest" in report.legal_bases


# ═══════════════════════════════════════════════════════════════════
# Audit trail: verify it already handles multi-valued correctly
# ═══════════════════════════════════════════════════════════════════


class TestAuditTrailMultiValued:
    """audit_trail() iterates via _iter_subject_properties which correctly
    yields each list entry.  Verify this is not affected by the bug."""

    def test_audit_trail_returns_all_entries(self):
        """audit_trail must return one AuditEntry per property value."""
        graph = _graph_with_multi_valued_property()
        entries = audit_trail(graph, data_subject="ex:subjects/alice")
        # 2 emails + 1 name = 3 entries
        assert len(entries) == 3
        values = [e.value for e in entries]
        assert "alice@work.com" in values
        assert "alice@personal.com" in values
        assert "Alice Smith" in values
