"""Tests for vector extensions — edge cases."""

import math
import pytest
from jsonld_ex.vector import (
    vector_term_definition, validate_vector, cosine_similarity,
    extract_vectors, strip_vectors_for_rdf,
)


class TestVectorTermDefinition:
    def test_basic(self):
        defn = vector_term_definition("embedding", "http://ex.org/embedding", 768)
        assert defn["embedding"]["@container"] == "@vector"
        assert defn["embedding"]["@dimensions"] == 768

    def test_no_dimensions(self):
        defn = vector_term_definition("emb", "http://ex.org/emb")
        assert "@dimensions" not in defn["emb"]

    def test_zero_dimensions(self):
        with pytest.raises(ValueError):
            vector_term_definition("emb", "http://ex.org/emb", 0)

    def test_negative_dimensions(self):
        with pytest.raises(ValueError):
            vector_term_definition("emb", "http://ex.org/emb", -5)

    def test_bool_dimensions_rejected(self):
        with pytest.raises(ValueError):
            vector_term_definition("emb", "http://ex.org/emb", True)

    def test_float_dimensions_rejected(self):
        with pytest.raises(ValueError):
            vector_term_definition("emb", "http://ex.org/emb", 3.5)


class TestValidateVector:
    def test_valid(self):
        ok, errors = validate_vector([1.0, 2.0, 3.0])
        assert ok
        assert errors == []

    def test_with_dimensions(self):
        ok, errors = validate_vector([1.0, 2.0], expected_dimensions=2)
        assert ok

    def test_dimension_mismatch(self):
        ok, errors = validate_vector([1.0, 2.0], expected_dimensions=3)
        assert not ok
        assert "mismatch" in errors[0]

    def test_empty_vector(self):
        ok, errors = validate_vector([])
        assert not ok
        assert "empty" in errors[0].lower()

    def test_not_a_list(self):
        ok, errors = validate_vector("not a vector")
        assert not ok

    def test_nan_element(self):
        ok, errors = validate_vector([1.0, float("nan"), 3.0])
        assert not ok
        assert "[1]" in errors[0]

    def test_inf_element(self):
        ok, errors = validate_vector([float("inf"), 2.0])
        assert not ok

    def test_bool_element_rejected(self):
        ok, errors = validate_vector([True, False, 1.0])
        assert not ok
        assert "[0]" in errors[0]

    def test_string_element(self):
        ok, errors = validate_vector([1.0, "bad", 3.0])
        assert not ok

    def test_tuple_input(self):
        ok, errors = validate_vector((1.0, 2.0, 3.0))
        assert ok

    def test_integers_valid(self):
        ok, errors = validate_vector([1, 2, 3])
        assert ok


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_empty_vectors(self):
        with pytest.raises(ValueError, match="empty"):
            cosine_similarity([], [])

    def test_zero_vector_raises(self):
        with pytest.raises(ValueError, match="zero-magnitude"):
            cosine_similarity([0.0, 0.0], [1.0, 2.0])

    def test_nan_raises(self):
        with pytest.raises(ValueError, match="finite"):
            cosine_similarity([float("nan"), 1.0], [1.0, 1.0])

    def test_inf_raises(self):
        with pytest.raises(ValueError, match="finite"):
            cosine_similarity([1.0, 1.0], [float("inf"), 1.0])

    def test_bool_raises(self):
        with pytest.raises(TypeError, match="number"):
            cosine_similarity([True, 1.0], [1.0, 1.0])


class TestExtractVectors:
    def test_basic(self):
        node = {"name": "test", "embedding": [1.0, 2.0, 3.0]}
        result = extract_vectors(node, ["embedding"])
        assert result == {"embedding": [1.0, 2.0, 3.0]}

    def test_missing_property(self):
        result = extract_vectors({"name": "test"}, ["embedding"])
        assert result == {}

    def test_non_dict(self):
        assert extract_vectors("not a dict", ["embedding"]) == {}

    def test_non_numeric_list(self):
        node = {"tags": ["a", "b", "c"]}
        result = extract_vectors(node, ["tags"])
        assert result == {}


class TestStripVectorsForRdf:
    def test_removes_vectors(self):
        doc = {"@type": "Product", "name": "Widget", "embedding": [1.0, 2.0]}
        result = strip_vectors_for_rdf(doc, ["embedding"])
        assert "embedding" not in result
        assert result["name"] == "Widget"

    def test_nested_graph(self):
        doc = {"@graph": [{"embedding": [1.0], "name": "A"}]}
        result = strip_vectors_for_rdf(doc, ["embedding"])
        assert "embedding" not in result["@graph"][0]

    def test_preserves_non_vector(self):
        doc = {"name": "test", "age": 30}
        result = strip_vectors_for_rdf(doc, ["embedding"])
        assert result == doc

    def test_list_input(self):
        docs = [{"embedding": [1.0]}, {"embedding": [2.0]}]
        result = strip_vectors_for_rdf(docs, ["embedding"])
        assert all("embedding" not in d for d in result)

    def test_scalar_passthrough(self):
        assert strip_vectors_for_rdf("hello", ["embedding"]) == "hello"
