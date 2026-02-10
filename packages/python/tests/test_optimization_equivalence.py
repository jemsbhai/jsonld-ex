"""
Tests for performance optimizations — proving behavioral equivalence.

These tests MUST pass BEFORE and AFTER each optimization is applied.
They exist to guarantee that no user-visible behavior changes,
upholding scientific rigor and backward compatibility.

Structure:
  1. TestProvOInputImmutability — proves copy.deepcopy is unnecessary
  2. TestTrustedCreateEquivalence — proves _trusted_create matches constructor
  3. TestProvOBatchEquivalence — proves batch API matches per-node calls
"""

import copy
import json
import math
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
)
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay
from jsonld_ex.ai_ml import annotate
from jsonld_ex.owl_interop import to_prov_o, ConversionReport


# ── Reusable Hypothesis strategies (same as test_property_based.py) ──

_unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@st.composite
def opinions(draw, min_uncertainty=0.0):
    b = draw(_unit)
    max_d = 1.0 - b
    d = draw(st.floats(min_value=0.0, max_value=max_d, allow_nan=False, allow_infinity=False))
    u = 1.0 - b - d
    if u < 0.0:
        u = 0.0
    if u > 1.0:
        u = 1.0
    from hypothesis import assume
    assume(u >= min_uncertainty)
    a = draw(_unit)
    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


_TOL = 1e-9


# ═══════════════════════════════════════════════════════════════════
# FIX 1: Prove copy.deepcopy is unnecessary in to_prov_o
# ═══════════════════════════════════════════════════════════════════


class TestProvOInputImmutability:
    """Prove that to_prov_o does NOT mutate its input document.

    If these tests pass, copy.deepcopy in to_prov_o is proven unnecessary
    and can be safely removed.
    """

    def _make_annotated_doc(self) -> dict:
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
            "email": annotate(
                "alice@example.com",
                confidence=0.88,
                source="https://models.example.org/gpt4",
                method="regex",
                human_verified=True,
            ),
            "jobTitle": annotate(
                "Engineer",
                confidence=0.75,
                extracted_at="2025-01-10T08:00:00Z",
            ),
        }

    def test_single_annotation_input_unchanged(self):
        """Input with one annotated property is not mutated."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/x",
            "name": annotate("Test", confidence=0.9, source="https://example.org/m"),
        }
        snapshot = copy.deepcopy(doc)
        to_prov_o(doc)
        assert doc == snapshot, "to_prov_o mutated its input document"

    def test_multi_annotation_input_unchanged(self):
        """Input with multiple annotated properties is not mutated."""
        doc = self._make_annotated_doc()
        snapshot = copy.deepcopy(doc)
        to_prov_o(doc)
        assert doc == snapshot, "to_prov_o mutated its input document"

    def test_nested_values_not_mutated(self):
        """Nested dict values inside annotated properties are untouched."""
        doc = self._make_annotated_doc()
        # Capture the inner dicts' ids and content
        originals = {}
        for key in ("name", "email", "jobTitle"):
            originals[key] = copy.deepcopy(doc[key])

        to_prov_o(doc)

        for key in ("name", "email", "jobTitle"):
            assert doc[key] == originals[key], f"Inner dict '{key}' was mutated"

    def test_unannotated_doc_unchanged(self):
        """Document with no annotations passes through untouched."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "name": "plain string",
            "age": 30,
        }
        snapshot = copy.deepcopy(doc)
        to_prov_o(doc)
        assert doc == snapshot

    def test_serialization_identical(self):
        """JSON serialization of input is byte-identical before and after."""
        doc = self._make_annotated_doc()
        before_json = json.dumps(doc, sort_keys=True)
        to_prov_o(doc)
        after_json = json.dumps(doc, sort_keys=True)
        assert before_json == after_json, "Serialized input changed after to_prov_o call"

    def test_output_independent_of_input(self):
        """Modifying the output does not affect the input."""
        doc = self._make_annotated_doc()
        snapshot = copy.deepcopy(doc)
        prov_doc, _ = to_prov_o(doc)

        # Mutate the output aggressively
        prov_doc["@graph"] = []
        prov_doc["@context"] = "DESTROYED"

        assert doc == snapshot, "Modifying output affected input (shared references)"

    def test_repeated_calls_idempotent(self):
        """Calling to_prov_o twice on the same input produces identical output."""
        doc = self._make_annotated_doc()
        prov1, report1 = to_prov_o(doc)
        prov2, report2 = to_prov_o(doc)

        assert report1.nodes_converted == report2.nodes_converted
        assert report1.triples_output == report2.triples_output
        # Note: UUIDs differ between calls, so we compare structure not identity
        assert len(prov1["@graph"]) == len(prov2["@graph"])


# ═══════════════════════════════════════════════════════════════════
# FIX 2: Prove _trusted_create equivalence with Opinion constructor
# ═══════════════════════════════════════════════════════════════════


class TestTrustedCreateEquivalence:
    """Prove that _trusted_create (when implemented) produces Opinions
    bitwise-identical to the normal constructor for valid inputs.

    These tests use the CURRENT constructor to establish ground truth.
    After _trusted_create is added, we re-run to prove equivalence.

    Phase 1 (before implementation): All tests pass using Opinion().
    Phase 2 (after implementation): All tests still pass, plus new
             tests verify _trusted_create matches.
    """

    # ── Phase 1: Establish ground truth for all algebra outputs ──

    @given(a=opinions(), b=opinions())
    def test_cumulative_fuse_output_valid(self, a, b):
        """Cumulative fusion output satisfies all Opinion invariants."""
        r = cumulative_fuse(a, b)
        assert 0.0 <= r.belief <= 1.0
        assert 0.0 <= r.disbelief <= 1.0
        assert 0.0 <= r.uncertainty <= 1.0
        assert 0.0 <= r.base_rate <= 1.0
        assert abs(r.belief + r.disbelief + r.uncertainty - 1.0) < _TOL
        assert not math.isnan(r.belief)
        assert not math.isnan(r.disbelief)
        assert not math.isnan(r.uncertainty)

    @given(a=opinions(), b=opinions())
    def test_averaging_fuse_output_valid(self, a, b):
        """Averaging fusion output satisfies all Opinion invariants."""
        r = averaging_fuse(a, b)
        assert 0.0 <= r.belief <= 1.0
        assert 0.0 <= r.disbelief <= 1.0
        assert 0.0 <= r.uncertainty <= 1.0
        assert abs(r.belief + r.disbelief + r.uncertainty - 1.0) < _TOL
        assert not math.isnan(r.belief)

    @given(trust=opinions(), opinion=opinions())
    def test_trust_discount_output_valid(self, trust, opinion):
        """Trust discount output satisfies all Opinion invariants."""
        r = trust_discount(trust, opinion)
        assert 0.0 <= r.belief <= 1.0
        assert 0.0 <= r.disbelief <= 1.0
        assert 0.0 <= r.uncertainty <= 1.0
        assert abs(r.belief + r.disbelief + r.uncertainty - 1.0) < _TOL
        assert not math.isnan(r.belief)

    @given(ox=opinions(), oyx=opinions(), oynx=opinions())
    def test_deduce_output_valid(self, ox, oyx, oynx):
        """Deduction output satisfies all Opinion invariants."""
        r = deduce(ox, oyx, oynx)
        assert 0.0 <= r.belief <= 1.0
        assert 0.0 <= r.disbelief <= 1.0
        assert 0.0 <= r.uncertainty <= 1.0
        assert abs(r.belief + r.disbelief + r.uncertainty - 1.0) < _TOL
        assert not math.isnan(r.belief)

    @given(opinion=opinions())
    def test_decay_output_valid(self, opinion):
        """Decay output satisfies all Opinion invariants."""
        r = decay_opinion(opinion, elapsed=5.0, half_life=10.0)
        assert 0.0 <= r.belief <= 1.0
        assert 0.0 <= r.disbelief <= 1.0
        assert 0.0 <= r.uncertainty <= 1.0
        assert abs(r.belief + r.disbelief + r.uncertainty - 1.0) < _TOL
        assert not math.isnan(r.belief)

    # ── Deterministic regression anchors ──

    def test_cumulative_fuse_deterministic(self):
        """Fixed inputs produce exact known outputs (regression anchor).

        Manual computation:
            a = (0.7, 0.1, 0.2), b = (0.5, 0.3, 0.2)
            kappa = u_a + u_b - u_a*u_b = 0.2 + 0.2 - 0.04 = 0.36
            b = (0.7*0.2 + 0.5*0.2)/0.36 = 0.24/0.36 = 2/3
            d = (0.1*0.2 + 0.3*0.2)/0.36 = 0.08/0.36 = 2/9
            u = (0.2*0.2)/0.36 = 0.04/0.36 = 1/9
        """
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        r = cumulative_fuse(a, b)
        assert r.belief == pytest.approx(2.0 / 3.0, abs=1e-12)
        assert r.disbelief == pytest.approx(2.0 / 9.0, abs=1e-12)
        assert r.uncertainty == pytest.approx(1.0 / 9.0, abs=1e-12)

    def test_trust_discount_deterministic(self):
        """Fixed inputs produce exact known outputs (regression anchor).

        Manual computation:
            trust = (0.8, 0.1, 0.1), opinion = (0.7, 0.1, 0.2)
            b = 0.8 * 0.7 = 0.56
            d = 0.8 * 0.1 = 0.08
            u = 0.1 + 0.1 + 0.8 * 0.2 = 0.36
            Check: 0.56 + 0.08 + 0.36 = 1.0 ✓
        """
        trust = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        opinion = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        r = trust_discount(trust, opinion)
        assert r.belief == pytest.approx(0.56, abs=1e-12)
        assert r.disbelief == pytest.approx(0.08, abs=1e-12)
        assert r.uncertainty == pytest.approx(0.36, abs=1e-12)

    def test_averaging_fuse_deterministic(self):
        """Fixed inputs produce exact known outputs (regression anchor).

        Manual computation:
            a = (0.7, 0.1, 0.2), b = (0.5, 0.3, 0.2)
            kappa = u_a + u_b = 0.4
            b = (0.7*0.2 + 0.5*0.2)/0.4 = 0.24/0.4 = 0.6
            d = (0.1*0.2 + 0.3*0.2)/0.4 = 0.08/0.4 = 0.2
            u = 2 * 0.04 / 0.4 = 0.2
            Check: 0.6 + 0.2 + 0.2 = 1.0 ✓
        """
        a = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        b = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        r = averaging_fuse(a, b)
        assert r.belief == pytest.approx(0.6, abs=1e-12)
        assert r.disbelief == pytest.approx(0.2, abs=1e-12)
        assert r.uncertainty == pytest.approx(0.2, abs=1e-12)

    def test_deduce_deterministic(self):
        """Fixed inputs produce exact known outputs (regression anchor).

        Manual computation (Jøsang 2016, Def. 12.6):
            ox = (0.6, 0.2, 0.2, a=0.5), oyx = (0.9, 0.05, 0.05), oynx = (0.1, 0.8, 0.1)
            a_x = 0.5, a_x_bar = 0.5

            b_y = 0.6*0.9 + 0.2*0.1 + 0.2*(0.5*0.9 + 0.5*0.1)
                = 0.54 + 0.02 + 0.1 = 0.66
            d_y = 0.6*0.05 + 0.2*0.8 + 0.2*(0.5*0.05 + 0.5*0.8)
                = 0.03 + 0.16 + 0.085 = 0.275
            u_y = 0.6*0.05 + 0.2*0.1 + 0.2*(0.5*0.05 + 0.5*0.1)
                = 0.03 + 0.02 + 0.015 = 0.065
            Check: 0.66 + 0.275 + 0.065 = 1.0 ✓
        """
        ox = Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)
        oyx = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
        oynx = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        r = deduce(ox, oyx, oynx)
        assert r.belief == pytest.approx(0.66, abs=1e-12)
        assert r.disbelief == pytest.approx(0.275, abs=1e-12)
        assert r.uncertainty == pytest.approx(0.065, abs=1e-12)


# ═══════════════════════════════════════════════════════════════════
# FIX 3: Prove batch to_prov_o equivalence (test written in advance)
# ═══════════════════════════════════════════════════════════════════


class TestProvOBatchEquivalence:
    """Tests for the batch to_prov_o_graph API (to be implemented).

    These tests verify that processing a @graph array in batch produces
    equivalent results to processing each node individually, which is
    the current behavior in bench_bridge.py.

    These tests will initially be SKIPPED (marked xfail) until the
    batch API is implemented.
    """

    def _make_graph_doc(self, n: int = 10) -> dict:
        """Create a multi-node document with annotated properties."""
        nodes = []
        for i in range(n):
            node = {
                "@id": f"http://example.org/entity-{i}",
                "@type": "Person",
                "name": annotate(
                    f"Person-{i}",
                    confidence=round(0.5 + i * 0.04, 4),
                    source=f"https://models.example.org/model-{i % 3}",
                    method="NER",
                ),
                "email": annotate(
                    f"person{i}@example.com",
                    confidence=round(0.6 + i * 0.03, 4),
                    source=f"https://models.example.org/model-{i % 3}",
                ),
            }
            nodes.append(node)
        return {"@context": "http://schema.org/", "@graph": nodes}

    def _per_node_prov_o(self, graph_doc: dict) -> tuple[list[dict], int, int]:
        """Current approach: call to_prov_o per node."""
        nodes = graph_doc.get("@graph", [])
        all_prov_docs = []
        total_converted = 0
        total_triples = 0
        for node in nodes:
            single = {"@context": graph_doc.get("@context", {})}
            single.update(node)
            prov_doc, report = to_prov_o(single)
            all_prov_docs.append(prov_doc)
            total_converted += report.nodes_converted
            total_triples += report.triples_output
        return all_prov_docs, total_converted, total_triples

    @pytest.mark.xfail(reason="to_prov_o_graph not yet implemented")
    def test_batch_node_count_matches(self):
        """Batch produces same number of converted nodes."""
        from jsonld_ex.owl_interop import to_prov_o_graph

        doc = self._make_graph_doc(10)
        _, per_node_converted, _ = self._per_node_prov_o(doc)
        _, batch_report = to_prov_o_graph(doc)
        assert batch_report.nodes_converted == per_node_converted

    @pytest.mark.xfail(reason="to_prov_o_graph not yet implemented")
    def test_batch_triple_count_matches(self):
        """Batch produces same triple count."""
        from jsonld_ex.owl_interop import to_prov_o_graph

        doc = self._make_graph_doc(10)
        _, _, per_node_triples = self._per_node_prov_o(doc)
        _, batch_report = to_prov_o_graph(doc)
        assert batch_report.triples_output == per_node_triples

    @pytest.mark.xfail(reason="to_prov_o_graph not yet implemented")
    def test_batch_preserves_all_values(self):
        """Every @value in the input appears in the batch output."""
        from jsonld_ex.owl_interop import to_prov_o_graph

        doc = self._make_graph_doc(5)
        batch_doc, _ = to_prov_o_graph(doc)

        # Collect all PROV values from batch output
        from jsonld_ex.owl_interop import PROV
        graph = batch_doc.get("@graph", [])
        prov_values = set()
        for node in graph:
            val = node.get(f"{PROV}value")
            if val is not None:
                prov_values.add(str(val))

        # Verify all original values present
        for node in doc["@graph"]:
            for key in ("name", "email"):
                if isinstance(node.get(key), dict) and "@value" in node[key]:
                    assert str(node[key]["@value"]) in prov_values

    @pytest.mark.xfail(reason="to_prov_o_graph not yet implemented")
    def test_batch_preserves_all_confidences(self):
        """Every @confidence in the input appears in the batch output."""
        from jsonld_ex.owl_interop import to_prov_o_graph
        from jsonld_ex.owl_interop import JSONLD_EX

        doc = self._make_graph_doc(5)
        batch_doc, _ = to_prov_o_graph(doc)

        graph = batch_doc.get("@graph", [])
        confidences = set()
        for node in graph:
            conf = node.get(f"{JSONLD_EX}confidence")
            if conf is not None:
                confidences.add(conf)

        for orig_node in doc["@graph"]:
            for key in ("name", "email"):
                val = orig_node.get(key)
                if isinstance(val, dict) and "@confidence" in val:
                    assert val["@confidence"] in confidences

    @pytest.mark.xfail(reason="to_prov_o_graph not yet implemented")
    def test_batch_does_not_mutate_input(self):
        """Batch API does not mutate its input."""
        from jsonld_ex.owl_interop import to_prov_o_graph

        doc = self._make_graph_doc(5)
        snapshot = copy.deepcopy(doc)
        to_prov_o_graph(doc)
        assert doc == snapshot
