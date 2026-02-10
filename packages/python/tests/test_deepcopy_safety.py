"""Tests proving input documents are NOT mutated by owl_interop functions.

TDD contract: these tests MUST pass both BEFORE and AFTER removing
copy.deepcopy from to_prov_o.  If any test fails after the change,
the optimization is unsafe and must be reverted.

Scientific rigor: we verify byte-for-byte JSON equality of the input
document before and after each function call, guaranteeing zero
side-effects for existing users.
"""

import copy
import json
import pytest

from jsonld_ex.ai_ml import annotate
from jsonld_ex.owl_interop import (
    to_prov_o,
    from_prov_o,
    compare_with_prov_o,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def simple_annotated():
    """Single annotated property — minimal reproduction case."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/alice",
        "name": annotate(
            "Alice Smith",
            confidence=0.95,
            source="https://models.example.org/gpt4",
            extracted_at="2025-01-15T10:30:00Z",
            method="NER",
        ),
    }


@pytest.fixture
def multi_annotated():
    """Multiple annotated properties — stress nested dict traversal."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/bob",
        "name": annotate("Bob Jones", confidence=0.92, source="https://m.example.org/v1"),
        "email": annotate(
            "bob@example.com",
            confidence=0.88,
            source="https://m.example.org/v1",
            method="regex-extraction",
            human_verified=True,
        ),
        "jobTitle": annotate(
            "Engineer",
            confidence=0.75,
            extracted_at="2025-01-10T08:00:00Z",
        ),
    }


@pytest.fixture
def nested_doc():
    """Nested nodes — tests that inner dicts aren't mutated."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/charlie",
        "name": annotate("Charlie", confidence=0.9),
        "address": {
            "@type": "PostalAddress",
            "streetAddress": annotate("123 Main St", confidence=0.8),
        },
    }


@pytest.fixture
def list_doc():
    """Annotated values inside lists — edge case for mutation."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/diana",
        "name": annotate("Diana", confidence=0.95),
        "knows": [
            annotate("http://example.org/friend1", confidence=0.7),
            annotate("http://example.org/friend2", confidence=0.6),
        ],
    }


@pytest.fixture
def dict_context_doc():
    """Document with dict @context — tests context merging path."""
    return {
        "@context": {
            "schema": "http://schema.org/",
            "name": "schema:name",
        },
        "@type": "Person",
        "@id": "http://example.org/eve",
        "name": annotate("Eve", confidence=0.85, source="https://m.example.org/v2"),
    }


@pytest.fixture
def list_context_doc():
    """Document with list @context — tests array context merging path."""
    return {
        "@context": [
            "http://schema.org/",
            {"custom": "http://custom.example.org/"},
        ],
        "@type": "Person",
        "@id": "http://example.org/frank",
        "name": annotate("Frank", confidence=0.77),
    }


@pytest.fixture
def unannotated_doc():
    """No annotations — should pass through unchanged."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "name": "Plain Jane",
        "email": "jane@example.com",
    }


# ── Helpers ─────────────────────────────────────────────────────


def _snapshot(doc: dict) -> str:
    """Deterministic JSON serialization for comparison."""
    return json.dumps(doc, sort_keys=True, ensure_ascii=True)


# ═══════════════════════════════════════════════════════════════════
# to_prov_o: INPUT MUST NOT BE MUTATED
# ═══════════════════════════════════════════════════════════════════


class TestToProvONoMutation:
    """Guarantee: to_prov_o(doc) never modifies doc."""

    def test_simple_annotated_not_mutated(self, simple_annotated):
        before = _snapshot(simple_annotated)
        original_ref = copy.deepcopy(simple_annotated)

        prov_doc, report = to_prov_o(simple_annotated)

        after = _snapshot(simple_annotated)
        assert before == after, (
            f"to_prov_o mutated the input document!\n"
            f"Before: {before}\n"
            f"After:  {after}"
        )
        assert report.success is True

    def test_multi_annotated_not_mutated(self, multi_annotated):
        before = _snapshot(multi_annotated)

        to_prov_o(multi_annotated)

        assert _snapshot(multi_annotated) == before

    def test_nested_doc_not_mutated(self, nested_doc):
        before = _snapshot(nested_doc)

        to_prov_o(nested_doc)

        assert _snapshot(nested_doc) == before

    def test_list_doc_not_mutated(self, list_doc):
        before = _snapshot(list_doc)

        to_prov_o(list_doc)

        assert _snapshot(list_doc) == before

    def test_dict_context_not_mutated(self, dict_context_doc):
        before = _snapshot(dict_context_doc)

        to_prov_o(dict_context_doc)

        assert _snapshot(dict_context_doc) == before

    def test_list_context_not_mutated(self, list_context_doc):
        before = _snapshot(list_context_doc)

        to_prov_o(list_context_doc)

        assert _snapshot(list_context_doc) == before

    def test_unannotated_doc_not_mutated(self, unannotated_doc):
        before = _snapshot(unannotated_doc)

        to_prov_o(unannotated_doc)

        assert _snapshot(unannotated_doc) == before

    def test_repeated_calls_same_input(self, simple_annotated):
        """Calling to_prov_o multiple times on same doc must be safe."""
        before = _snapshot(simple_annotated)

        for _ in range(10):
            to_prov_o(simple_annotated)

        assert _snapshot(simple_annotated) == before

    def test_output_independent_of_input(self, simple_annotated):
        """Modifying the output must not affect the input."""
        before = _snapshot(simple_annotated)

        prov_doc, _ = to_prov_o(simple_annotated)

        # Mutate the output aggressively
        prov_doc["@context"]["INJECTED"] = "evil"
        prov_doc["@graph"].clear()

        assert _snapshot(simple_annotated) == before


# ═══════════════════════════════════════════════════════════════════
# from_prov_o: INPUT MUST NOT BE MUTATED
# (from_prov_o uses deepcopy internally — this test ensures it stays)
# ═══════════════════════════════════════════════════════════════════


class TestFromProvONoMutation:
    """Guarantee: from_prov_o(prov_doc) never modifies prov_doc."""

    def test_prov_doc_not_mutated(self, simple_annotated):
        prov_doc, _ = to_prov_o(simple_annotated)
        before = _snapshot(prov_doc)

        from_prov_o(prov_doc)

        assert _snapshot(prov_doc) == before


# ═══════════════════════════════════════════════════════════════════
# compare_with_prov_o: INPUT MUST NOT BE MUTATED
# (calls to_prov_o internally)
# ═══════════════════════════════════════════════════════════════════


class TestCompareNoMutation:
    """Guarantee: compare_with_prov_o(doc) never modifies doc."""

    def test_compare_does_not_mutate(self, simple_annotated):
        before = _snapshot(simple_annotated)

        compare_with_prov_o(simple_annotated)

        assert _snapshot(simple_annotated) == before
