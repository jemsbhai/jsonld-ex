"""Tests for the similarity() dispatcher function (Step 5).

The dispatcher resolves which metric to use via three mechanisms
(in priority order):
  1. Explicit ``metric`` keyword  (highest priority)
  2. ``@similarity`` key inside a ``term_definition`` dict
  3. Default: ``"cosine"``

It then delegates to the corresponding registered function.
"""

import pytest
from jsonld_ex.similarity import (
    similarity,
    register_similarity_metric,
    reset_similarity_registry,
)
from jsonld_ex.vector import vector_term_definition, cosine_similarity


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_similarity_registry()
    yield
    reset_similarity_registry()


# ── Default behaviour ───────────────────────────────────────────────────

class TestDefaultMetric:
    """When no metric or term_definition is given, use cosine."""

    def test_default_is_cosine(self):
        a, b = [1.0, 0.0], [1.0, 0.0]
        assert similarity(a, b) == pytest.approx(1.0)

    def test_default_matches_cosine_similarity(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert similarity(a, b) == pytest.approx(cosine_similarity(a, b))


# ── Explicit metric keyword ─────────────────────────────────────────────

class TestExplicitMetric:
    def test_cosine(self):
        a, b = [1.0, 0.0], [0.0, 1.0]
        assert similarity(a, b, metric="cosine") == pytest.approx(0.0)

    def test_euclidean(self):
        a, b = [0.0, 0.0], [3.0, 4.0]
        assert similarity(a, b, metric="euclidean") == pytest.approx(5.0)

    def test_dot_product(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert similarity(a, b, metric="dot_product") == pytest.approx(32.0)

    def test_manhattan(self):
        a, b = [0.0, 0.0], [3.0, 4.0]
        assert similarity(a, b, metric="manhattan") == pytest.approx(7.0)

    def test_unknown_metric_raises(self):
        with pytest.raises(KeyError, match="no_such"):
            similarity([1.0], [2.0], metric="no_such")

    def test_custom_metric(self):
        register_similarity_metric("always_half", lambda a, b: 0.5)
        assert similarity([1.0], [2.0], metric="always_half") == 0.5


# ── Term definition resolution ──────────────────────────────────────────

class TestTermDefinitionResolution:
    def test_reads_similarity_from_term_def(self):
        td = vector_term_definition(
            "emb", "http://ex.org/emb", 2, similarity="euclidean"
        )
        a, b = [0.0, 0.0], [3.0, 4.0]
        assert similarity(a, b, term_definition=td["emb"]) == pytest.approx(5.0)

    def test_term_def_manhattan(self):
        td = vector_term_definition(
            "emb", "http://ex.org/emb", similarity="manhattan"
        )
        a, b = [0.0, 0.0], [3.0, 4.0]
        assert similarity(a, b, term_definition=td["emb"]) == pytest.approx(7.0)

    def test_term_def_without_similarity_falls_back_to_cosine(self):
        td = vector_term_definition("emb", "http://ex.org/emb", 2)
        a, b = [1.0, 0.0], [1.0, 0.0]
        assert similarity(a, b, term_definition=td["emb"]) == pytest.approx(1.0)

    def test_term_def_with_custom_metric(self):
        register_similarity_metric("neg_dot", lambda a, b: -sum(x * y for x, y in zip(a, b)))
        td = vector_term_definition(
            "emb", "http://ex.org/emb", similarity="neg_dot"
        )
        assert similarity([1.0, 2.0], [3.0, 4.0], term_definition=td["emb"]) == pytest.approx(-11.0)

    def test_term_def_with_unregistered_metric_raises(self):
        td = vector_term_definition(
            "emb", "http://ex.org/emb", similarity="not_registered_yet"
        )
        with pytest.raises(KeyError, match="not_registered_yet"):
            similarity([1.0], [2.0], term_definition=td["emb"])


# ── Priority: explicit metric overrides term_definition ─────────────────

class TestPriorityOverride:
    def test_metric_overrides_term_def(self):
        td = vector_term_definition(
            "emb", "http://ex.org/emb", similarity="manhattan"
        )
        a, b = [0.0, 0.0], [3.0, 4.0]
        # term_def says manhattan (would give 7), but metric says euclidean
        result = similarity(a, b, metric="euclidean", term_definition=td["emb"])
        assert result == pytest.approx(5.0)

    def test_metric_overrides_term_def_to_cosine(self):
        td = vector_term_definition(
            "emb", "http://ex.org/emb", similarity="euclidean"
        )
        a, b = [1.0, 0.0], [1.0, 0.0]
        result = similarity(a, b, metric="cosine", term_definition=td["emb"])
        assert result == pytest.approx(1.0)


# ── Validation passthrough ──────────────────────────────────────────────

class TestValidationPassthrough:
    """The dispatcher must not swallow validation errors from the
    underlying metric — they must propagate unchanged."""

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            similarity([1.0, 2.0], [1.0], metric="euclidean")

    def test_empty_vectors(self):
        with pytest.raises(ValueError, match="empty"):
            similarity([], [], metric="dot_product")

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            similarity([float("nan")], [1.0], metric="manhattan")

    def test_zero_vector_cosine(self):
        with pytest.raises(ValueError, match="zero-magnitude"):
            similarity([0.0, 0.0], [1.0, 2.0], metric="cosine")
