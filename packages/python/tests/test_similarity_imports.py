"""Smoke tests: every new public symbol is importable from jsonld_ex."""

import pytest


class TestTopLevelImports:
    """Verify that all new similarity symbols are accessible from the
    top-level package — the way most users will import them."""

    def test_similarity_dispatcher(self):
        from jsonld_ex import similarity
        assert callable(similarity)

    def test_euclidean_distance(self):
        from jsonld_ex import euclidean_distance
        assert callable(euclidean_distance)

    def test_dot_product(self):
        from jsonld_ex import dot_product
        assert callable(dot_product)

    def test_manhattan_distance(self):
        from jsonld_ex import manhattan_distance
        assert callable(manhattan_distance)

    def test_chebyshev_distance(self):
        from jsonld_ex import chebyshev_distance
        assert callable(chebyshev_distance)

    def test_hamming_distance(self):
        from jsonld_ex import hamming_distance
        assert callable(hamming_distance)

    def test_jaccard_similarity(self):
        from jsonld_ex import jaccard_similarity
        assert callable(jaccard_similarity)

    def test_register_similarity_metric(self):
        from jsonld_ex import register_similarity_metric
        assert callable(register_similarity_metric)

    def test_get_similarity_metric(self):
        from jsonld_ex import get_similarity_metric
        assert callable(get_similarity_metric)

    def test_list_similarity_metrics(self):
        from jsonld_ex import list_similarity_metrics
        assert callable(list_similarity_metrics)

    def test_unregister_similarity_metric(self):
        from jsonld_ex import unregister_similarity_metric
        assert callable(unregister_similarity_metric)

    def test_builtin_metric_names(self):
        from jsonld_ex import BUILTIN_METRIC_NAMES
        assert isinstance(BUILTIN_METRIC_NAMES, frozenset)
        assert len(BUILTIN_METRIC_NAMES) == 7


class TestBackwardsCompat:
    """Existing import paths must keep working."""

    def test_cosine_from_vector(self):
        from jsonld_ex.vector import cosine_similarity
        assert callable(cosine_similarity)

    def test_cosine_from_top_level(self):
        from jsonld_ex import cosine_similarity
        assert callable(cosine_similarity)

    def test_cosine_identity_matches(self):
        """The cosine in the registry is the same object as vector.py's."""
        from jsonld_ex.vector import cosine_similarity as from_vector
        from jsonld_ex.similarity import get_similarity_metric
        from_registry = get_similarity_metric("cosine")
        assert from_registry is from_vector

    def test_vector_term_definition_from_top_level(self):
        from jsonld_ex import vector_term_definition
        assert callable(vector_term_definition)

    def test_validate_vector_from_top_level(self):
        from jsonld_ex import validate_vector
        assert callable(validate_vector)
