"""Tests for context module — versioning and diff (GAP-CTX1)."""

import pytest

from jsonld_ex.context import (
    context_diff,
    check_compatibility,
    ContextDiff,
    TermChange,
)


# ═══════════════════════════════════════════════════════════════════
# context_diff
# ═══════════════════════════════════════════════════════════════════


class TestContextDiff:
    """context_diff() compares two context dicts."""

    def test_identical_contexts(self):
        """No changes → empty diff."""
        ctx = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        diff = context_diff(ctx, ctx)
        assert diff.added == {}
        assert diff.removed == {}
        assert diff.changed == {}

    def test_added_terms(self):
        """New terms in new context."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        diff = context_diff(old, new)
        assert "age" in diff.added
        assert diff.added["age"] == "http://schema.org/age"
        assert diff.removed == {}
        assert diff.changed == {}

    def test_removed_terms(self):
        """Terms missing in new context."""
        old = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        new = {"name": "http://schema.org/name"}
        diff = context_diff(old, new)
        assert "age" in diff.removed
        assert diff.removed["age"] == "http://schema.org/age"
        assert diff.added == {}
        assert diff.changed == {}

    def test_changed_iri_mapping(self):
        """Term maps to different IRI."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://example.org/name"}
        diff = context_diff(old, new)
        assert "name" in diff.changed
        assert diff.changed["name"].old_value == "http://schema.org/name"
        assert diff.changed["name"].new_value == "http://example.org/name"

    def test_changed_expanded_definition(self):
        """Term changes from string IRI to expanded definition."""
        old = {"name": "http://schema.org/name"}
        new = {"name": {"@id": "http://schema.org/name", "@type": "xsd:string"}}
        diff = context_diff(old, new)
        assert "name" in diff.changed

    def test_changed_type_coercion(self):
        """Same IRI but type coercion changed."""
        old = {"age": {"@id": "http://schema.org/age", "@type": "xsd:string"}}
        new = {"age": {"@id": "http://schema.org/age", "@type": "xsd:integer"}}
        diff = context_diff(old, new)
        assert "age" in diff.changed

    def test_unchanged_expanded_definitions(self):
        """Equivalent expanded definitions → no change."""
        old = {"name": {"@id": "http://schema.org/name", "@type": "xsd:string"}}
        new = {"name": {"@id": "http://schema.org/name", "@type": "xsd:string"}}
        diff = context_diff(old, new)
        assert diff.changed == {}

    def test_ignores_at_keywords(self):
        """@-prefixed context keys (@vocab, @base, @version) tracked separately."""
        old = {"@vocab": "http://schema.org/", "name": "name"}
        new = {"@vocab": "http://example.org/", "name": "name"}
        diff = context_diff(old, new)
        # @vocab change should be in changed, not ignored
        assert "@vocab" in diff.changed

    def test_mixed_changes(self):
        """Combination of added, removed, and changed."""
        old = {
            "name": "http://schema.org/name",
            "age": "http://schema.org/age",
            "email": "http://schema.org/email",
        }
        new = {
            "name": "http://example.org/name",  # changed
            "email": "http://schema.org/email",  # unchanged
            "phone": "http://schema.org/telephone",  # added
        }
        diff = context_diff(old, new)
        assert "phone" in diff.added
        assert "age" in diff.removed
        assert "name" in diff.changed
        assert "email" not in diff.changed
        assert "email" not in diff.added
        assert "email" not in diff.removed

    def test_version_extraction(self):
        """@contextVersion extracted from both contexts."""
        old = {"@contextVersion": "1.0.0", "name": "http://schema.org/name"}
        new = {"@contextVersion": "2.0.0", "name": "http://schema.org/name"}
        diff = context_diff(old, new)
        assert diff.old_version == "1.0.0"
        assert diff.new_version == "2.0.0"

    def test_no_version(self):
        """Missing @contextVersion → None."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://schema.org/name"}
        diff = context_diff(old, new)
        assert diff.old_version is None
        assert diff.new_version is None


# ═══════════════════════════════════════════════════════════════════
# check_compatibility
# ═══════════════════════════════════════════════════════════════════


class TestCheckCompatibility:
    """check_compatibility() classifies changes as breaking/non-breaking."""

    def test_additions_only_compatible(self):
        """Only added terms → backward compatible."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        result = check_compatibility(old, new)
        assert result.compatible is True
        assert result.breaking == []

    def test_removal_is_breaking(self):
        """Removed term → breaking change."""
        old = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        new = {"name": "http://schema.org/name"}
        result = check_compatibility(old, new)
        assert result.compatible is False
        assert any(b.change_type == "removed" and b.term == "age" for b in result.breaking)

    def test_changed_iri_is_breaking(self):
        """Changed IRI mapping → breaking."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://example.org/name"}
        result = check_compatibility(old, new)
        assert result.compatible is False
        assert any(b.change_type == "changed-mapping" for b in result.breaking)

    def test_changed_type_coercion_is_breaking(self):
        """Changed @type on a term → breaking."""
        old = {"age": {"@id": "http://schema.org/age", "@type": "xsd:string"}}
        new = {"age": {"@id": "http://schema.org/age", "@type": "xsd:integer"}}
        result = check_compatibility(old, new)
        assert result.compatible is False
        assert any(b.change_type == "changed-type" for b in result.breaking)

    def test_added_type_coercion_is_breaking(self):
        """Adding @type where none existed → breaking (changes interpretation)."""
        old = {"age": "http://schema.org/age"}
        new = {"age": {"@id": "http://schema.org/age", "@type": "xsd:integer"}}
        result = check_compatibility(old, new)
        assert result.compatible is False

    def test_identical_compatible(self):
        """No changes → compatible."""
        ctx = {"name": "http://schema.org/name"}
        result = check_compatibility(ctx, ctx)
        assert result.compatible is True
        assert result.breaking == []
        assert result.non_breaking == []

    def test_vocab_change_is_breaking(self):
        """@vocab change → breaking (affects all unqualified terms)."""
        old = {"@vocab": "http://schema.org/"}
        new = {"@vocab": "http://example.org/"}
        result = check_compatibility(old, new)
        assert result.compatible is False

    def test_non_breaking_additions_listed(self):
        """Added terms appear in non_breaking list."""
        old = {"name": "http://schema.org/name"}
        new = {"name": "http://schema.org/name", "age": "http://schema.org/age"}
        result = check_compatibility(old, new)
        assert any(nb.change_type == "added" and nb.term == "age" for nb in result.non_breaking)
