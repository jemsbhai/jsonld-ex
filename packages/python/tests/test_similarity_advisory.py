"""Tests for MetricProperties — structured metadata for similarity metrics.

Phase 1 of the Metric Selection Advisory features.  Every property claim
is a mathematical fact that must be defensible.  These tests define the
contract before any implementation exists.

Test organisation
-----------------
- TestMetricPropertiesDataclass: structural tests (fields, defaults, frozen)
- TestBuiltinMetricProperties: mathematical facts for each of the 7 built-ins
- TestExampleMetricProperties: properties for the 10 example metrics
- TestPropertiesRegistry: get/list/register-with-properties API
- TestPropertiesEdgeCases: unknown metrics, custom without properties, etc.
"""

from __future__ import annotations

import pytest

from jsonld_ex.similarity import (
    MetricProperties,
    VectorProperties,
    analyze_vectors,
    compare_metrics,
    evaluate_metrics,
    recommend_metric,
    HeuristicRecommender,
    get_metric_properties,
    get_all_metric_properties,
    list_similarity_metrics,
    register_similarity_metric,
    unregister_similarity_metric,
    reset_similarity_registry,
    BUILTIN_METRIC_NAMES,
)


# ═══════════════════════════════════════════════════════════════════
# Helper: check numpy availability (for skip markers)
# ═══════════════════════════════════════════════════════════════════


def _has_numpy() -> bool:
    try:
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════
# Fixture: isolate registry state
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the similarity registry before and after every test."""
    reset_similarity_registry()
    yield
    reset_similarity_registry()


# ═══════════════════════════════════════════════════════════════════
# TestMetricPropertiesDataclass
# ═══════════════════════════════════════════════════════════════════


class TestMetricPropertiesDataclass:
    """Verify the MetricProperties dataclass structure."""

    def test_required_fields_exist(self):
        """MetricProperties must have all documented fields."""
        props = MetricProperties(
            name="test",
            kind="distance",
            range_min=0.0,
            range_max=None,
            bounded=False,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=True,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=("test data",),
        )
        assert props.name == "test"
        assert props.kind == "distance"
        assert props.range_min == 0.0
        assert props.range_max is None
        assert props.bounded is False
        assert props.metric_space is True
        assert props.symmetric is True
        assert props.normalization_sensitive is True
        assert props.zero_vector_behavior == "accepts"
        assert props.computational_complexity == "O(n)"
        assert props.best_for == ("test data",)

    def test_frozen_immutability(self):
        """MetricProperties instances must be immutable (frozen dataclass)."""
        props = MetricProperties(
            name="test",
            kind="distance",
            range_min=0.0,
            range_max=None,
            bounded=False,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=True,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=("test data",),
        )
        with pytest.raises(AttributeError):
            props.name = "changed"  # type: ignore[misc]

    def test_kind_values(self):
        """kind must be one of the allowed values."""
        valid_kinds = {"distance", "similarity", "inner_product", "divergence", "correlation"}
        for kind in valid_kinds:
            props = MetricProperties(
                name="test", kind=kind, range_min=0.0, range_max=1.0,
                bounded=True, metric_space=False, symmetric=True,
                normalization_sensitive=False, zero_vector_behavior="accepts",
                computational_complexity="O(n)", best_for=(),
            )
            assert props.kind == kind

    def test_zero_vector_behavior_values(self):
        """zero_vector_behavior must be one of the allowed values."""
        for behavior in ("accepts", "rejects", "convention"):
            props = MetricProperties(
                name="test", kind="distance", range_min=0.0, range_max=1.0,
                bounded=True, metric_space=False, symmetric=True,
                normalization_sensitive=False, zero_vector_behavior=behavior,
                computational_complexity="O(n)", best_for=(),
            )
            assert props.zero_vector_behavior == behavior

    def test_range_none_means_unbounded(self):
        """None for range_min means -inf; None for range_max means +inf."""
        props = MetricProperties(
            name="dot", kind="inner_product", range_min=None, range_max=None,
            bounded=False, metric_space=False, symmetric=True,
            normalization_sensitive=True, zero_vector_behavior="accepts",
            computational_complexity="O(n)", best_for=(),
        )
        assert props.range_min is None
        assert props.range_max is None

    def test_best_for_is_tuple(self):
        """best_for must be a tuple of strings (immutable)."""
        props = MetricProperties(
            name="test", kind="distance", range_min=0.0, range_max=None,
            bounded=False, metric_space=True, symmetric=True,
            normalization_sensitive=True, zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=("text embeddings", "unit-normalized vectors"),
        )
        assert isinstance(props.best_for, tuple)
        assert all(isinstance(s, str) for s in props.best_for)


# ═══════════════════════════════════════════════════════════════════
# TestBuiltinMetricProperties — mathematical facts
# ═══════════════════════════════════════════════════════════════════


class TestBuiltinMetricProperties:
    """Each built-in metric must have correct, pre-defined properties.

    These are mathematical facts, not opinions.  Each assertion is
    accompanied by a brief justification comment.
    """

    def test_all_builtins_have_properties(self):
        """Every built-in metric must have MetricProperties available."""
        for name in BUILTIN_METRIC_NAMES:
            props = get_metric_properties(name)
            assert isinstance(props, MetricProperties)
            assert props.name == name

    # --- cosine ---

    def test_cosine_properties(self):
        p = get_metric_properties("cosine")
        # Cosine similarity: cos(theta) ∈ [-1, 1]
        assert p.kind == "similarity"
        assert p.range_min == -1.0
        assert p.range_max == 1.0
        # Bounded: [-1, 1] is finite and dimension-independent
        assert p.bounded is True
        # Not a metric space: fails triangle inequality
        # (cosine distance = 1 - cosine is a semimetric but fails triangle ineq.)
        assert p.metric_space is False
        assert p.symmetric is True
        # Cosine is magnitude-invariant by definition (normalises internally)
        assert p.normalization_sensitive is False
        # Our implementation raises on zero vectors (0/0 undefined)
        assert p.zero_vector_behavior == "rejects"
        assert p.computational_complexity == "O(n)"

    # --- euclidean ---

    def test_euclidean_properties(self):
        p = get_metric_properties("euclidean")
        # L2 norm: always ≥ 0, no finite upper bound
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None  # unbounded above
        assert p.bounded is False
        # Euclidean is the canonical metric space
        assert p.metric_space is True
        assert p.symmetric is True
        # Scaling vectors changes distances: d(2a, 2b) = 2*d(a, b)
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n)"

    # --- dot_product ---

    def test_dot_product_properties(self):
        p = get_metric_properties("dot_product")
        # Inner product: can be any real number
        assert p.kind == "inner_product"
        assert p.range_min is None  # -inf
        assert p.range_max is None  # +inf
        assert p.bounded is False
        # Not a metric: no triangle inequality, can be negative
        assert p.metric_space is False
        assert p.symmetric is True
        # dot(2a, b) = 2 * dot(a, b)
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n)"

    # --- manhattan ---

    def test_manhattan_properties(self):
        p = get_metric_properties("manhattan")
        # L1 norm: always ≥ 0, no finite upper bound
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None
        assert p.bounded is False
        # Manhattan is a proper metric (satisfies all four axioms)
        assert p.metric_space is True
        assert p.symmetric is True
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n)"

    # --- chebyshev ---

    def test_chebyshev_properties(self):
        p = get_metric_properties("chebyshev")
        # L-inf norm: max(|a_i - b_i|) ≥ 0, no finite upper bound
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None
        assert p.bounded is False
        # L-inf is a proper metric (limit of Lp norms)
        assert p.metric_space is True
        assert p.symmetric is True
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n)"

    # --- hamming ---

    def test_hamming_properties(self):
        p = get_metric_properties("hamming")
        # Count of mismatches: ∈ [0, n] where n = dimension
        assert p.kind == "distance"
        assert p.range_min == 0.0
        # Upper bound is dimension-dependent → None
        assert p.range_max is None
        assert p.bounded is False
        # Hamming is a proper metric on fixed-length strings/vectors
        assert p.metric_space is True
        assert p.symmetric is True
        # Hamming counts positions, not magnitudes: d([0,0],[1,1]) = d([0,0],[5,5]) = 2
        assert p.normalization_sensitive is False
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n)"

    # --- jaccard ---

    def test_jaccard_properties(self):
        p = get_metric_properties("jaccard")
        # |A∩B|/|A∪B| ∈ [0, 1]
        assert p.kind == "similarity"
        assert p.range_min == 0.0
        assert p.range_max == 1.0
        assert p.bounded is True
        # Jaccard similarity is not a metric (1-J is a metric on sets)
        assert p.metric_space is False
        assert p.symmetric is True
        # Jaccard treats nonzero as "present" — magnitude doesn't matter
        assert p.normalization_sensitive is False
        # Our implementation returns 1.0 for (∅, ∅) by convention
        assert p.zero_vector_behavior == "convention"
        assert p.computational_complexity == "O(n)"


# ═══════════════════════════════════════════════════════════════════
# TestExampleMetricProperties — properties for example metrics
# ═══════════════════════════════════════════════════════════════════


class TestExampleMetricProperties:
    """Properties for the 10 example metrics from similarity_examples.py.

    These are defined as companion constants in similarity_examples.py
    and can be passed to register_similarity_metric(properties=...).
    """

    def test_canberra_properties(self):
        from jsonld_ex.similarity_examples import (
            canberra_distance,
            CANBERRA_PROPERTIES,
        )
        p = CANBERRA_PROPERTIES
        assert isinstance(p, MetricProperties)
        assert p.name == "canberra"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        # Upper bound = n (dimension), so dimension-dependent → None
        assert p.range_max is None
        assert p.bounded is False
        # Canberra is a proper metric
        assert p.metric_space is True
        assert p.symmetric is True
        # Sensitive to scaling: d(2a, 2b) != d(a, b) in general
        assert p.normalization_sensitive is True
        # 0/0 terms contribute 0.0 by convention
        assert p.zero_vector_behavior == "convention"
        assert p.computational_complexity == "O(n)"

    def test_bray_curtis_properties(self):
        from jsonld_ex.similarity_examples import (
            bray_curtis_dissimilarity,
            BRAY_CURTIS_PROPERTIES,
        )
        p = BRAY_CURTIS_PROPERTIES
        assert p.name == "bray_curtis"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max == 1.0  # for non-negative inputs
        assert p.bounded is True
        # Bray-Curtis fails triangle inequality in general
        assert p.metric_space is False
        assert p.symmetric is True
        # It's a ratio: numerator and denominator scale equally
        assert p.normalization_sensitive is False
        assert p.zero_vector_behavior == "convention"
        assert p.computational_complexity == "O(n)"

    def test_haversine_properties(self):
        from jsonld_ex.similarity_examples import (
            haversine_distance,
            HAVERSINE_PROPERTIES,
        )
        p = HAVERSINE_PROPERTIES
        assert p.name == "haversine"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        # Half Earth circumference ≈ 20015.09 km
        assert p.range_max is not None
        assert 20015.0 < p.range_max < 20016.0
        assert p.bounded is True
        # Great-circle distance is a proper metric on the sphere
        assert p.metric_space is True
        assert p.symmetric is True
        # Fixed 2-element [lat, lon] — "normalization" not applicable
        # but scaling coordinates would change the result, so True
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        # Always 2 elements, so effectively O(1)
        assert p.computational_complexity == "O(1)"

    def test_dtw_properties(self):
        from jsonld_ex.similarity_examples import (
            dtw_distance,
            DTW_PROPERTIES,
        )
        p = DTW_PROPERTIES
        assert p.name == "dtw"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None
        assert p.bounded is False
        # DTW without constraints does NOT satisfy triangle inequality
        assert p.metric_space is False
        assert p.symmetric is True
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        # Full DP matrix: O(n^2) for equal-length vectors
        assert p.computational_complexity == "O(n^2)"

    def test_kl_divergence_properties(self):
        from jsonld_ex.similarity_examples import (
            kl_divergence,
            KL_DIVERGENCE_PROPERTIES,
        )
        p = KL_DIVERGENCE_PROPERTIES
        assert p.name == "kl_divergence"
        assert p.kind == "divergence"
        assert p.range_min == 0.0
        assert p.range_max is None  # can be +inf
        assert p.bounded is False
        assert p.metric_space is False
        # KL(P||Q) != KL(Q||P) — this is the key fact
        assert p.symmetric is False
        # Inputs must be probability distributions (sum to 1)
        assert p.normalization_sensitive is True
        # Zero in Q with nonzero in P → undefined (implementation raises)
        assert p.zero_vector_behavior == "rejects"
        assert p.computational_complexity == "O(n)"

    def test_wasserstein_properties(self):
        from jsonld_ex.similarity_examples import (
            wasserstein_distance,
            WASSERSTEIN_PROPERTIES,
        )
        p = WASSERSTEIN_PROPERTIES
        assert p.name == "wasserstein"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None
        assert p.bounded is False
        # Wasserstein-1 is a proper metric on probability distributions
        assert p.metric_space is True
        assert p.symmetric is True
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        assert p.computational_complexity == "O(n log n)"

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy not installed"
    )
    def test_mahalanobis_properties(self):
        from jsonld_ex.similarity_examples import MAHALANOBIS_PROPERTIES
        p = MAHALANOBIS_PROPERTIES
        assert p.name == "mahalanobis"
        assert p.kind == "distance"
        assert p.range_min == 0.0
        assert p.range_max is None
        assert p.bounded is False
        # Mahalanobis with valid positive-definite inverse covariance is a metric
        assert p.metric_space is True
        assert p.symmetric is True
        # Depends on covariance — not purely normalization, but scaling matters
        assert p.normalization_sensitive is True
        assert p.zero_vector_behavior == "accepts"
        # Matrix-vector multiply: O(n^2)
        assert p.computational_complexity == "O(n^2)"

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy not installed"
    )
    def test_soft_cosine_properties(self):
        from jsonld_ex.similarity_examples import SOFT_COSINE_PROPERTIES
        p = SOFT_COSINE_PROPERTIES
        assert p.name == "soft_cosine"
        assert p.kind == "similarity"
        assert p.range_min == -1.0
        assert p.range_max == 1.0
        assert p.bounded is True
        assert p.metric_space is False
        assert p.symmetric is True
        # Like cosine, normalises internally
        assert p.normalization_sensitive is False
        assert p.zero_vector_behavior == "rejects"
        assert p.computational_complexity == "O(n^2)"

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy not installed"
    )
    def test_spearman_properties(self):
        from jsonld_ex.similarity_examples import SPEARMAN_PROPERTIES
        p = SPEARMAN_PROPERTIES
        assert p.name == "spearman"
        assert p.kind == "correlation"
        assert p.range_min == -1.0
        assert p.range_max == 1.0
        assert p.bounded is True
        assert p.metric_space is False
        assert p.symmetric is True
        # Rank-based: only order matters, not magnitude
        assert p.normalization_sensitive is False
        assert p.zero_vector_behavior == "convention"
        # Ranking step is O(n log n), Pearson on ranks is O(n)
        assert p.computational_complexity == "O(n log n)"

    @pytest.mark.skipif(
        not _has_numpy(), reason="numpy not installed"
    )
    def test_kendall_tau_properties(self):
        from jsonld_ex.similarity_examples import KENDALL_TAU_PROPERTIES
        p = KENDALL_TAU_PROPERTIES
        assert p.name == "kendall_tau"
        assert p.kind == "correlation"
        assert p.range_min == -1.0
        assert p.range_max == 1.0
        assert p.bounded is True
        assert p.metric_space is False
        assert p.symmetric is True
        assert p.normalization_sensitive is False
        assert p.zero_vector_behavior == "convention"
        # All-pairs comparison: O(n^2)
        assert p.computational_complexity == "O(n^2)"


# ═══════════════════════════════════════════════════════════════════
# TestPropertiesRegistry — API for querying and registering
# ═══════════════════════════════════════════════════════════════════


class TestPropertiesRegistry:
    """Test the get/list/register-with-properties API."""

    def test_get_metric_properties_returns_correct_type(self):
        props = get_metric_properties("cosine")
        assert isinstance(props, MetricProperties)

    def test_get_all_metric_properties_returns_all_builtins(self):
        all_props = get_all_metric_properties()
        assert isinstance(all_props, dict)
        for name in BUILTIN_METRIC_NAMES:
            assert name in all_props
            assert isinstance(all_props[name], MetricProperties)

    def test_get_all_metric_properties_keys_match_names(self):
        """Dict keys must match the .name field of each MetricProperties."""
        all_props = get_all_metric_properties()
        for key, props in all_props.items():
            assert key == props.name

    def test_register_custom_with_properties(self):
        """Custom metrics can optionally provide MetricProperties."""
        custom_props = MetricProperties(
            name="my_metric",
            kind="distance",
            range_min=0.0,
            range_max=1.0,
            bounded=True,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=False,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=("custom use case",),
        )
        register_similarity_metric(
            "my_metric",
            lambda a, b: 0.5,
            properties=custom_props,
        )
        retrieved = get_metric_properties("my_metric")
        assert retrieved == custom_props

    def test_register_custom_without_properties(self):
        """Custom metrics without properties → get_metric_properties returns None."""
        register_similarity_metric("bare_metric", lambda a, b: 0.0)
        result = get_metric_properties("bare_metric")
        assert result is None

    def test_custom_properties_appear_in_get_all(self):
        """Custom metrics with properties appear in get_all_metric_properties."""
        custom_props = MetricProperties(
            name="my_metric",
            kind="similarity",
            range_min=0.0,
            range_max=1.0,
            bounded=True,
            metric_space=False,
            symmetric=True,
            normalization_sensitive=False,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=(),
        )
        register_similarity_metric("my_metric", lambda a, b: 0.5, properties=custom_props)
        all_props = get_all_metric_properties()
        assert "my_metric" in all_props
        assert all_props["my_metric"] == custom_props

    def test_custom_without_properties_excluded_from_get_all(self):
        """Custom metrics without properties do NOT appear in get_all."""
        register_similarity_metric("bare_metric", lambda a, b: 0.0)
        all_props = get_all_metric_properties()
        assert "bare_metric" not in all_props

    def test_unregister_removes_properties(self):
        """Unregistering a custom metric also removes its properties."""
        custom_props = MetricProperties(
            name="temp_metric",
            kind="distance",
            range_min=0.0,
            range_max=None,
            bounded=False,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=True,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=(),
        )
        register_similarity_metric("temp_metric", lambda a, b: 0.0, properties=custom_props)
        assert get_metric_properties("temp_metric") is not None
        unregister_similarity_metric("temp_metric")
        assert get_metric_properties("temp_metric") is None

    def test_reset_restores_only_builtin_properties(self):
        """After reset, only built-in properties remain."""
        custom_props = MetricProperties(
            name="custom",
            kind="distance",
            range_min=0.0,
            range_max=None,
            bounded=False,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=True,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=(),
        )
        register_similarity_metric("custom", lambda a, b: 0.0, properties=custom_props)
        reset_similarity_registry()
        assert get_metric_properties("custom") is None
        # Built-ins still present
        for name in BUILTIN_METRIC_NAMES:
            assert get_metric_properties(name) is not None

    def test_force_override_preserves_new_properties(self):
        """force=True with new properties replaces the old properties."""
        new_props = MetricProperties(
            name="cosine",
            kind="similarity",
            range_min=0.0,
            range_max=1.0,
            bounded=True,
            metric_space=False,
            symmetric=True,
            normalization_sensitive=False,
            zero_vector_behavior="rejects",
            computational_complexity="O(n)",
            best_for=("custom override",),
        )
        register_similarity_metric(
            "cosine",
            lambda a, b: 0.42,
            force=True,
            properties=new_props,
        )
        retrieved = get_metric_properties("cosine")
        assert retrieved == new_props

    def test_properties_name_must_match_registration_name(self):
        """If properties.name != registration name, raise ValueError."""
        mismatched = MetricProperties(
            name="wrong_name",
            kind="distance",
            range_min=0.0,
            range_max=None,
            bounded=False,
            metric_space=True,
            symmetric=True,
            normalization_sensitive=True,
            zero_vector_behavior="accepts",
            computational_complexity="O(n)",
            best_for=(),
        )
        with pytest.raises(ValueError, match="name.*mismatch|must match"):
            register_similarity_metric(
                "correct_name",
                lambda a, b: 0.0,
                properties=mismatched,
            )

    def test_register_with_properties_example_workflow(self):
        """Demonstrate the intended workflow for example metrics."""
        from jsonld_ex.similarity_examples import (
            canberra_distance,
            CANBERRA_PROPERTIES,
        )
        register_similarity_metric(
            "canberra", canberra_distance, properties=CANBERRA_PROPERTIES
        )
        props = get_metric_properties("canberra")
        assert props is not None
        assert props.name == "canberra"
        assert props.kind == "distance"


# ═══════════════════════════════════════════════════════════════════
# TestPropertiesEdgeCases
# ═══════════════════════════════════════════════════════════════════


class TestPropertiesEdgeCases:
    """Edge cases and error handling for metric properties."""

    def test_get_properties_unknown_metric(self):
        """Querying properties for an unregistered metric returns None."""
        assert get_metric_properties("nonexistent") is None

    def test_get_all_returns_snapshot(self):
        """get_all_metric_properties returns a copy, not a live reference."""
        all1 = get_all_metric_properties()
        all2 = get_all_metric_properties()
        assert all1 is not all2
        assert all1 == all2

    def test_builtin_properties_survive_registry_reset(self):
        """Built-in properties are restored after reset_similarity_registry."""
        original = get_metric_properties("euclidean")
        reset_similarity_registry()
        restored = get_metric_properties("euclidean")
        assert original == restored


# ═══════════════════════════════════════════════════════════════════
# Phase 2: compare_metrics
# ═══════════════════════════════════════════════════════════════════


class TestCompareMetricsBasic:
    """Basic functionality: all 7 built-ins on a normal vector pair."""

    def test_returns_dict_with_expected_keys(self):
        result = compare_metrics([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert "results" in result
        assert "vectors_dimension" in result
        assert "metrics_computed" in result
        assert "metrics_failed" in result

    def test_dimension_reported(self):
        result = compare_metrics([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        assert result["vectors_dimension"] == 3

    def test_all_builtins_computed_by_default(self):
        result = compare_metrics([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        for name in BUILTIN_METRIC_NAMES:
            assert name in result["results"], f"Missing metric: {name}"
        assert result["metrics_computed"] == len(BUILTIN_METRIC_NAMES)
        assert result["metrics_failed"] == 0

    def test_each_result_has_score_kind_error(self):
        result = compare_metrics([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        for name, entry in result["results"].items():
            assert "score" in entry, f"{name}: missing 'score'"
            assert "kind" in entry, f"{name}: missing 'kind'"
            assert "error" in entry, f"{name}: missing 'error'"

    def test_scores_are_floats(self):
        result = compare_metrics([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
        for name, entry in result["results"].items():
            assert isinstance(entry["score"], float), f"{name}: score not float"
            assert entry["error"] is None

    def test_kind_matches_properties(self):
        """The 'kind' field must agree with MetricProperties for built-ins."""
        result = compare_metrics([1.0, 0.0], [0.0, 1.0])
        for name, entry in result["results"].items():
            props = get_metric_properties(name)
            if props is not None:
                assert entry["kind"] == props.kind, (
                    f"{name}: kind={entry['kind']} but properties say {props.kind}"
                )

    def test_cosine_score_correct(self):
        """Spot-check: cosine([1,0],[0,1]) == 0.0 (orthogonal)."""
        result = compare_metrics([1.0, 0.0], [0.0, 1.0])
        assert result["results"]["cosine"]["score"] == pytest.approx(0.0)

    def test_euclidean_score_correct(self):
        """Spot-check: euclidean([0,0],[3,4]) == 5.0."""
        result = compare_metrics([0.0, 0.0], [3.0, 4.0])
        assert result["results"]["euclidean"]["score"] == pytest.approx(5.0)

    def test_identical_vectors(self):
        """Identical non-zero vectors: distances=0, cosine=1, jaccard=1."""
        result = compare_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        r = result["results"]
        assert r["cosine"]["score"] == pytest.approx(1.0)
        assert r["euclidean"]["score"] == pytest.approx(0.0)
        assert r["manhattan"]["score"] == pytest.approx(0.0)
        assert r["chebyshev"]["score"] == pytest.approx(0.0)
        assert r["hamming"]["score"] == pytest.approx(0.0)
        assert r["dot_product"]["score"] == pytest.approx(14.0)
        assert r["jaccard"]["score"] == pytest.approx(1.0)


class TestCompareMetricsFiltered:
    """Compute only a subset of metrics via the metrics kwarg."""

    def test_explicit_metric_list(self):
        result = compare_metrics(
            [1.0, 2.0], [3.0, 4.0], metrics=["cosine", "euclidean"]
        )
        assert set(result["results"].keys()) == {"cosine", "euclidean"}
        assert result["metrics_computed"] == 2

    def test_single_metric(self):
        result = compare_metrics([1.0, 2.0], [3.0, 4.0], metrics=["manhattan"])
        assert set(result["results"].keys()) == {"manhattan"}
        assert result["metrics_computed"] == 1

    def test_unknown_metric_raises(self):
        """Requesting an unregistered metric name raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            compare_metrics([1.0], [2.0], metrics=["nonexistent"])

    def test_custom_metric_included(self):
        """Custom-registered metrics can be selected by name."""
        register_similarity_metric("always_one", lambda a, b: 1.0)
        result = compare_metrics(
            [1.0, 2.0], [3.0, 4.0], metrics=["always_one"]
        )
        assert result["results"]["always_one"]["score"] == 1.0

    def test_custom_without_properties_kind_is_none(self):
        """Custom metric without properties → kind=None."""
        register_similarity_metric("bare", lambda a, b: 0.0)
        result = compare_metrics([1.0], [2.0], metrics=["bare"])
        assert result["results"]["bare"]["kind"] is None

    def test_custom_with_properties_kind_reported(self):
        """Custom metric with properties → kind from properties."""
        props = MetricProperties(
            name="my_dist", kind="distance", range_min=0.0, range_max=None,
            bounded=False, metric_space=True, symmetric=True,
            normalization_sensitive=True, zero_vector_behavior="accepts",
            computational_complexity="O(n)", best_for=(),
        )
        register_similarity_metric("my_dist", lambda a, b: 0.5, properties=props)
        result = compare_metrics([1.0], [2.0], metrics=["my_dist"])
        assert result["results"]["my_dist"]["kind"] == "distance"


class TestCompareMetricsErrorHandling:
    """Graceful handling when a metric fails on the given pair."""

    def test_cosine_zero_vector_reports_error(self):
        """Cosine rejects zero vectors — error is captured, not raised."""
        result = compare_metrics([0.0, 0.0], [1.0, 2.0])
        entry = result["results"]["cosine"]
        assert entry["score"] is None
        assert entry["error"] is not None
        assert isinstance(entry["error"], str)
        # Other metrics should still succeed
        assert result["results"]["euclidean"]["error"] is None
        assert result["metrics_failed"] >= 1

    def test_metrics_computed_excludes_failures(self):
        """metrics_computed counts only successful computations."""
        result = compare_metrics([0.0, 0.0], [1.0, 2.0])
        total = result["metrics_computed"] + result["metrics_failed"]
        assert total == len(result["results"])

    def test_custom_metric_exception_captured(self):
        """A custom metric that raises is captured as an error."""
        def bad_metric(a, b):
            raise RuntimeError("intentional failure")

        register_similarity_metric("bad", bad_metric)
        result = compare_metrics([1.0], [2.0], metrics=["bad"])
        entry = result["results"]["bad"]
        assert entry["score"] is None
        assert "intentional failure" in entry["error"]
        assert result["metrics_failed"] == 1
        assert result["metrics_computed"] == 0

    def test_include_errors_false_skips_failures(self):
        """include_errors=False omits metrics that failed."""
        result = compare_metrics(
            [0.0, 0.0], [1.0, 2.0], include_errors=False
        )
        # Cosine should not appear (it errors on zero vectors)
        assert "cosine" not in result["results"]
        # Other metrics should still be present
        assert "euclidean" in result["results"]

    def test_include_errors_false_counts(self):
        """With include_errors=False, metrics_failed still reported correctly."""
        result = compare_metrics(
            [0.0, 0.0], [1.0, 2.0], include_errors=False
        )
        assert result["metrics_failed"] >= 1
        assert result["metrics_computed"] == len(result["results"])


class TestCompareMetricsEdgeCases:
    """Edge cases: binary vectors, high-dim, etc."""

    def test_binary_vectors(self):
        """Binary vectors should work for all metrics."""
        result = compare_metrics([1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 1.0, 0.0])
        assert result["metrics_failed"] == 0
        # Hamming should be 2 (positions 0 and 1 differ)
        assert result["results"]["hamming"]["score"] == pytest.approx(2.0)
        # Jaccard: intersection={2}, union={0,1,2} → 1/3
        assert result["results"]["jaccard"]["score"] == pytest.approx(1.0 / 3.0)

    def test_high_dimensional(self):
        """100-dimensional vectors should work without issue."""
        a = [float(i) for i in range(100)]
        b = [float(i + 1) for i in range(100)]
        result = compare_metrics(a, b)
        assert result["vectors_dimension"] == 100
        assert result["metrics_computed"] == len(BUILTIN_METRIC_NAMES)

    def test_single_dimension(self):
        """1-D vectors are valid."""
        result = compare_metrics([3.0], [7.0])
        assert result["vectors_dimension"] == 1
        assert result["results"]["euclidean"]["score"] == pytest.approx(4.0)
        assert result["results"]["manhattan"]["score"] == pytest.approx(4.0)
        assert result["results"]["chebyshev"]["score"] == pytest.approx(4.0)

    def test_negative_values(self):
        """Negative values are valid for all metrics."""
        result = compare_metrics([-1.0, -2.0], [1.0, 2.0])
        assert result["metrics_failed"] == 0

    def test_dimension_mismatch_raises(self):
        """Dimension mismatch should raise before any computation."""
        with pytest.raises(ValueError, match="[Dd]imension"):
            compare_metrics([1.0, 2.0], [1.0])

    def test_empty_vectors_raises(self):
        """Empty vectors should raise before any computation."""
        with pytest.raises(ValueError, match="[Ee]mpty"):
            compare_metrics([], [])


# ═══════════════════════════════════════════════════════════════════
# Phase 3: VectorProperties, analyze_vectors, recommend_metric
# ═══════════════════════════════════════════════════════════════════


class TestVectorPropertiesDataclass:
    """Structural tests for VectorProperties."""

    def test_required_fields(self):
        vp = VectorProperties(
            n_vectors=10,
            dimensionality=3,
            is_binary=False,
            sparsity=0.0,
            is_unit_normalized=False,
            all_non_negative=True,
            magnitude_cv=0.1,
        )
        assert vp.n_vectors == 10
        assert vp.dimensionality == 3
        assert vp.is_binary is False
        assert vp.sparsity == 0.0
        assert vp.is_unit_normalized is False
        assert vp.all_non_negative is True
        assert vp.magnitude_cv == 0.1

    def test_frozen(self):
        vp = VectorProperties(
            n_vectors=2, dimensionality=3, is_binary=False,
            sparsity=0.0, is_unit_normalized=False,
            all_non_negative=True, magnitude_cv=0.0,
        )
        with pytest.raises(AttributeError):
            vp.n_vectors = 5  # type: ignore[misc]


class TestAnalyzeVectors:
    """Tests for analyze_vectors — deterministic data property detection."""

    # --- binary detection ---

    def test_binary_vectors_detected(self):
        vectors = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
        vp = analyze_vectors(vectors)
        assert vp.is_binary is True

    def test_non_binary_vectors(self):
        vectors = [
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.2],
        ]
        vp = analyze_vectors(vectors)
        assert vp.is_binary is False

    def test_all_zeros_is_binary(self):
        """All-zero vectors are technically binary ({0, 1} values only)."""
        vectors = [[0.0, 0.0], [0.0, 0.0]]
        vp = analyze_vectors(vectors)
        assert vp.is_binary is True

    # --- sparsity ---

    def test_sparsity_all_nonzero(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        vp = analyze_vectors(vectors)
        assert vp.sparsity == pytest.approx(0.0)

    def test_sparsity_all_zero(self):
        vectors = [[0.0, 0.0], [0.0, 0.0]]
        vp = analyze_vectors(vectors)
        assert vp.sparsity == pytest.approx(1.0)

    def test_sparsity_half(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        vp = analyze_vectors(vectors)
        # 2 zeros out of 4 total elements
        assert vp.sparsity == pytest.approx(0.5)

    # --- unit normalization ---

    def test_unit_normalized_detected(self):
        import math
        # Manually construct unit vectors
        s = math.sqrt(2.0)
        vectors = [
            [1.0 / s, 1.0 / s],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        vp = analyze_vectors(vectors)
        assert vp.is_unit_normalized is True

    def test_non_unit_normalized(self):
        vectors = [[3.0, 4.0], [1.0, 0.0]]  # first has norm 5
        vp = analyze_vectors(vectors)
        assert vp.is_unit_normalized is False

    def test_zero_vector_not_unit_normalized(self):
        """A sample containing a zero vector cannot be unit-normalized."""
        vectors = [[1.0, 0.0], [0.0, 0.0]]
        vp = analyze_vectors(vectors)
        assert vp.is_unit_normalized is False

    # --- non-negative ---

    def test_all_non_negative(self):
        vectors = [[0.0, 1.0, 2.0], [3.0, 0.0, 4.0]]
        vp = analyze_vectors(vectors)
        assert vp.all_non_negative is True

    def test_has_negatives(self):
        vectors = [[1.0, -1.0], [2.0, 3.0]]
        vp = analyze_vectors(vectors)
        assert vp.all_non_negative is False

    # --- magnitude coefficient of variation ---

    def test_identical_magnitudes_cv_zero(self):
        """Vectors with identical norms have CV = 0."""
        # Both have norm 5
        vectors = [[3.0, 4.0], [4.0, 3.0]]
        vp = analyze_vectors(vectors)
        assert vp.magnitude_cv == pytest.approx(0.0, abs=1e-9)

    def test_varied_magnitudes_cv_positive(self):
        """Vectors with different norms have CV > 0."""
        vectors = [[1.0, 0.0], [10.0, 0.0]]
        vp = analyze_vectors(vectors)
        assert vp.magnitude_cv > 0.0

    def test_zero_vectors_cv_zero(self):
        """All-zero vectors: mean norm = 0, CV = 0 by convention."""
        vectors = [[0.0, 0.0], [0.0, 0.0]]
        vp = analyze_vectors(vectors)
        assert vp.magnitude_cv == pytest.approx(0.0)

    # --- dimensionality and count ---

    def test_dimensionality(self):
        vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        vp = analyze_vectors(vectors)
        assert vp.dimensionality == 4

    def test_n_vectors(self):
        vectors = [[1.0], [2.0], [3.0], [4.0], [5.0]]
        vp = analyze_vectors(vectors)
        assert vp.n_vectors == 5

    # --- input validation ---

    def test_fewer_than_two_vectors_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least 2"):
            analyze_vectors([[1.0, 2.0]])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least 2"):
            analyze_vectors([])

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="[Dd]imension"):
            analyze_vectors([[1.0, 2.0], [1.0, 2.0, 3.0]])

    def test_empty_vectors_raises(self):
        with pytest.raises(ValueError, match="[Ee]mpty"):
            analyze_vectors([[], []])


class TestHeuristicRecommender:
    """Tests for the built-in rule-based recommendation engine.

    Each test constructs vectors with known properties and verifies
    that the recommendations cite the correct rationale.
    """

    def _top_metrics(self, vectors, n=3):
        """Helper: return the top-n recommended metric names."""
        result = recommend_metric(vectors)
        return [r["metric"] for r in result["recommendations"][:n]]

    def _find_rec(self, result, metric_name):
        """Helper: find a recommendation by metric name."""
        for r in result["recommendations"]:
            if r["metric"] == metric_name:
                return r
        return None

    # --- binary data ---

    def test_binary_recommends_hamming_and_jaccard(self):
        """Pure binary vectors → hamming and jaccard should rank highly."""
        vectors = [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
        top = self._top_metrics(vectors)
        assert "hamming" in top
        assert "jaccard" in top

    def test_binary_rationale_cites_binary(self):
        vectors = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
        result = recommend_metric(vectors)
        hamming_rec = self._find_rec(result, "hamming")
        assert hamming_rec is not None
        assert "binary" in hamming_rec["rationale"].lower()

    # --- unit-normalized ---

    def test_unit_normalized_recommends_cosine(self):
        """Unit-normalized vectors → cosine should rank #1."""
        import math
        s = math.sqrt(2.0)
        vectors = [
            [1.0 / s, 1.0 / s],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        top = self._top_metrics(vectors)
        assert top[0] == "cosine"

    def test_unit_normalized_rationale_cites_property(self):
        import math
        s = math.sqrt(2.0)
        vectors = [[1.0 / s, 1.0 / s], [1.0, 0.0], [0.0, 1.0]]
        result = recommend_metric(vectors)
        cosine_rec = self._find_rec(result, "cosine")
        assert cosine_rec is not None
        assert "unit" in cosine_rec["rationale"].lower() or \
               "normaliz" in cosine_rec["rationale"].lower()

    # --- high sparsity ---

    def test_sparse_vectors_recommend_cosine_or_jaccard(self):
        """Highly sparse vectors → cosine and jaccard rank highly."""
        vectors = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7],
        ]
        top = self._top_metrics(vectors)
        # At least one of cosine/jaccard should be in top 3
        assert "cosine" in top or "jaccard" in top

    # --- high magnitude variation ---

    def test_high_magnitude_variation_recommends_cosine(self):
        """When magnitudes vary widely, cosine is preferred (scale-invariant)."""
        vectors = [
            [1.0, 1.0, 1.0],
            [100.0, 100.0, 100.0],
            [0.01, 0.01, 0.01],
        ]
        top = self._top_metrics(vectors)
        assert "cosine" in top[:2]

    def test_high_magnitude_rationale_cites_magnitude(self):
        vectors = [
            [1.0, 1.0],
            [1000.0, 1000.0],
        ]
        result = recommend_metric(vectors)
        cosine_rec = self._find_rec(result, "cosine")
        assert cosine_rec is not None
        assert "magnitud" in cosine_rec["rationale"].lower()

    # --- recommendations structure ---

    def test_result_has_expected_top_level_keys(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        result = recommend_metric(vectors)
        assert "data_properties" in result
        assert "recommendations" in result
        assert "inconclusive" in result

    def test_data_properties_is_vector_properties(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        result = recommend_metric(vectors)
        vp = result["data_properties"]
        assert isinstance(vp, VectorProperties)

    def test_recommendations_are_ranked(self):
        vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        result = recommend_metric(vectors)
        recs = result["recommendations"]
        assert len(recs) >= 1
        ranks = [r["rank"] for r in recs]
        assert ranks == sorted(ranks)
        assert ranks[0] == 1

    def test_each_recommendation_has_required_fields(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        result = recommend_metric(vectors)
        for rec in result["recommendations"]:
            assert "metric" in rec
            assert "rank" in rec
            assert "rationale" in rec
            assert isinstance(rec["metric"], str)
            assert isinstance(rec["rank"], int)
            assert isinstance(rec["rationale"], str)
            assert len(rec["rationale"]) > 0

    def test_only_registered_metrics_recommended(self):
        """Recommendations must only include currently registered metrics."""
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        result = recommend_metric(vectors)
        registered = set(list_similarity_metrics())
        for rec in result["recommendations"]:
            assert rec["metric"] in registered, (
                f"Recommended unregistered metric: {rec['metric']}"
            )

    # --- custom engine ---

    def test_custom_engine_used(self):
        """A custom recommendation engine replaces the heuristic."""

        class AlwaysCosineEngine:
            def recommend(self, properties, available_metrics):
                return [
                    {"metric": "cosine", "rank": 1, "rationale": "Custom engine says so."},
                ]

        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = recommend_metric(vectors, engine=AlwaysCosineEngine())
        assert len(result["recommendations"]) == 1
        assert result["recommendations"][0]["metric"] == "cosine"
        assert "Custom engine" in result["recommendations"][0]["rationale"]

    def test_engine_receives_vector_properties_and_available_metrics(self):
        """The engine's recommend() gets VectorProperties and a dict of MetricProperties."""
        captured = {}

        class CapturingEngine:
            def recommend(self, properties, available_metrics):
                captured["properties"] = properties
                captured["available"] = available_metrics
                return []

        vectors = [[1.0, 2.0], [3.0, 4.0]]
        recommend_metric(vectors, engine=CapturingEngine())
        assert isinstance(captured["properties"], VectorProperties)
        assert isinstance(captured["available"], dict)
        # Should include at least the built-in properties
        for name in BUILTIN_METRIC_NAMES:
            assert name in captured["available"]

    # --- inconclusive ---

    def test_inconclusive_flag_is_bool(self):
        vectors = [[1.0, 2.0], [3.0, 4.0]]
        result = recommend_metric(vectors)
        assert isinstance(result["inconclusive"], bool)

    # --- input validation delegated to analyze_vectors ---

    def test_fewer_than_two_vectors_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least 2"):
            recommend_metric([[1.0, 2.0]])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least 2"):
            recommend_metric([])


class TestHeuristicRecommenderDirectly:
    """Test HeuristicRecommender as a standalone object."""

    def test_instantiable(self):
        engine = HeuristicRecommender()
        assert hasattr(engine, "recommend")

    def test_recommend_returns_list(self):
        engine = HeuristicRecommender()
        vp = VectorProperties(
            n_vectors=3, dimensionality=4, is_binary=True,
            sparsity=0.5, is_unit_normalized=False,
            all_non_negative=True, magnitude_cv=0.0,
        )
        available = get_all_metric_properties()
        recs = engine.recommend(vp, available)
        assert isinstance(recs, list)
        assert len(recs) >= 1

    def test_recommend_with_empty_available_returns_empty(self):
        """If no metrics have properties, no recommendations possible."""
        engine = HeuristicRecommender()
        vp = VectorProperties(
            n_vectors=2, dimensionality=3, is_binary=False,
            sparsity=0.0, is_unit_normalized=False,
            all_non_negative=True, magnitude_cv=0.0,
        )
        recs = engine.recommend(vp, {})
        assert recs == []


# ═══════════════════════════════════════════════════════════════════
# Phase 4: evaluate_metrics
# ═══════════════════════════════════════════════════════════════════


class TestEvaluateMetricsBasic:
    """Basic functionality and return structure."""

    @pytest.fixture()
    def separable_pairs(self):
        """Pairs where cosine perfectly separates similar from dissimilar.

        Similar pairs: parallel vectors (cosine=1).
        Dissimilar pairs: orthogonal vectors (cosine=0).
        """
        return [
            ([1.0, 0.0], [2.0, 0.0], 1.0),   # parallel
            ([0.0, 1.0], [0.0, 3.0], 1.0),   # parallel
            ([1.0, 0.0], [0.0, 1.0], 0.0),   # orthogonal
            ([0.0, 1.0], [1.0, 0.0], 0.0),   # orthogonal
        ]

    def test_returns_expected_top_level_keys(self, separable_pairs):
        result = evaluate_metrics(separable_pairs)
        assert "results" in result
        assert "ranking" in result
        assert "pairs_evaluated" in result
        assert "metrics_evaluated" in result
        assert "metrics_failed" in result

    def test_pairs_evaluated_count(self, separable_pairs):
        result = evaluate_metrics(separable_pairs)
        assert result["pairs_evaluated"] == 4

    def test_all_builtins_evaluated_by_default(self, separable_pairs):
        result = evaluate_metrics(separable_pairs)
        for name in BUILTIN_METRIC_NAMES:
            assert name in result["results"], f"Missing metric: {name}"

    def test_each_result_has_required_fields(self, separable_pairs):
        result = evaluate_metrics(separable_pairs)
        required = {
            "mean_separation", "rank_correlation", "auc",
            "higher_is_better", "error",
        }
        for name, entry in result["results"].items():
            for field in required:
                assert field in entry, f"{name}: missing '{field}'"

    def test_ranking_is_list_of_metric_names(self, separable_pairs):
        result = evaluate_metrics(separable_pairs)
        assert isinstance(result["ranking"], list)
        for name in result["ranking"]:
            assert isinstance(name, str)
            assert name in result["results"]


class TestEvaluateMetricsPerfectSeparation:
    """With perfectly separable data, the best metric should get AUC = 1.0."""

    def test_cosine_auc_perfect_on_parallel_vs_orthogonal(self):
        """Cosine = 1.0 for parallel, 0.0 for orthogonal → AUC = 1.0."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 3.0], 1.0),
            ([3.0, 0.0], [6.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
            ([0.0, 1.0], [1.0, 0.0], 0.0),
            ([0.0, 5.0], [5.0, 0.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        entry = result["results"]["cosine"]
        assert entry["auc"] == pytest.approx(1.0)
        assert entry["error"] is None

    def test_euclidean_separates_close_vs_far(self):
        """Close pairs (small distance) are similar; far apart are dissimilar."""
        pairs = [
            ([0.0, 0.0], [0.1, 0.0], 1.0),   # close
            ([0.0, 0.0], [0.0, 0.1], 1.0),   # close
            ([0.0, 0.0], [10.0, 0.0], 0.0),  # far
            ([0.0, 0.0], [0.0, 10.0], 0.0),  # far
        ]
        result = evaluate_metrics(pairs, metrics=["euclidean"])
        entry = result["results"]["euclidean"]
        # Euclidean is a distance: lower = more similar
        assert entry["higher_is_better"] is False
        assert entry["auc"] == pytest.approx(1.0)

    def test_perfect_separation_mean_separation_positive(self):
        """With perfect separation, mean_separation should be positive."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        # For cosine (similarity): mean(similar) > mean(dissimilar)
        # mean_separation should be positive after direction correction
        assert result["results"]["cosine"]["mean_separation"] > 0

    def test_perfect_rank_correlation(self):
        """With perfectly separable data, rank correlation should be high."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 3.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
            ([0.0, 1.0], [1.0, 0.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        rc = result["results"]["cosine"]["rank_correlation"]
        # Should be strongly positive (perfect monotonic relationship)
        assert rc is not None
        assert rc > 0.8


class TestEvaluateMetricsDirection:
    """Correct handling of higher_is_better for different metric kinds."""

    def test_similarity_higher_is_better_true(self):
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        assert result["results"]["cosine"]["higher_is_better"] is True

    def test_distance_higher_is_better_false(self):
        pairs = [
            ([0.0, 0.0], [0.1, 0.0], 1.0),
            ([0.0, 0.0], [10.0, 0.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["euclidean"])
        assert result["results"]["euclidean"]["higher_is_better"] is False

    def test_higher_is_better_override(self):
        """Explicit override takes precedence over properties."""
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(
            pairs,
            metrics=["cosine"],
            higher_is_better={"cosine": False},
        )
        assert result["results"]["cosine"]["higher_is_better"] is False

    def test_custom_metric_no_properties_default_true(self):
        """Custom metric without properties defaults to higher_is_better=True."""
        register_similarity_metric("custom_sim", lambda a, b: sum(x * y for x, y in zip(a, b)))
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["custom_sim"])
        entry = result["results"]["custom_sim"]
        assert entry["higher_is_better"] is True
        # Must warn that direction was assumed
        assert entry.get("direction_inferred") is False
        assert "warning" in entry
        assert entry["warning"] is not None

    def test_custom_metric_with_override_no_warning(self):
        """Custom metric with explicit override should not warn."""
        register_similarity_metric("custom_dist", lambda a, b: abs(a[0] - b[0]))
        pairs = [
            ([0.0], [0.1], 1.0),
            ([0.0], [10.0], 0.0),
        ]
        result = evaluate_metrics(
            pairs,
            metrics=["custom_dist"],
            higher_is_better={"custom_dist": False},
        )
        entry = result["results"]["custom_dist"]
        assert entry["higher_is_better"] is False
        assert entry["direction_inferred"] is True
        assert entry.get("warning") is None

    def test_custom_metric_with_properties_infers_direction(self):
        """Custom metric with properties infers direction correctly."""
        props = MetricProperties(
            name="my_dist", kind="distance", range_min=0.0, range_max=None,
            bounded=False, metric_space=True, symmetric=True,
            normalization_sensitive=True, zero_vector_behavior="accepts",
            computational_complexity="O(n)", best_for=(),
        )
        register_similarity_metric(
            "my_dist",
            lambda a, b: abs(a[0] - b[0]),
            properties=props,
        )
        pairs = [
            ([0.0], [0.1], 1.0),
            ([0.0], [10.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["my_dist"])
        entry = result["results"]["my_dist"]
        assert entry["higher_is_better"] is False
        assert entry["direction_inferred"] is True
        assert entry.get("warning") is None


class TestEvaluateMetricsEdgeCases:
    """Edge cases: all-same labels, single pair, metric errors."""

    def test_all_same_labels_auc_none(self):
        """AUC is undefined when all labels are the same class."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 3.0], 1.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        assert result["results"]["cosine"]["auc"] is None

    def test_all_same_labels_rank_correlation_none(self):
        """Rank correlation is undefined when all labels are the same."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 3.0], 1.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        assert result["results"]["cosine"]["rank_correlation"] is None

    def test_all_same_labels_mean_separation_none(self):
        """Mean separation is undefined with only one class."""
        pairs = [
            ([1.0, 0.0], [2.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 3.0], 1.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        assert result["results"]["cosine"]["mean_separation"] is None

    def test_single_pair_computes(self):
        """A single pair should not crash."""
        pairs = [([1.0, 0.0], [0.0, 1.0], 0.0)]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        assert result["pairs_evaluated"] == 1
        # Single pair with one class → AUC/correlation undefined
        assert result["results"]["cosine"]["auc"] is None

    def test_metric_error_on_some_pairs_captured(self):
        """If a metric errors on some pairs, report the error."""
        def picky_metric(a, b):
            if a[0] == 0.0:
                raise ValueError("Cannot handle zero")
            return sum(x * y for x, y in zip(a, b))

        register_similarity_metric("picky", picky_metric)
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 1.0], 1.0),  # will fail
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["picky"])
        entry = result["results"]["picky"]
        assert entry["error"] is not None
        assert result["metrics_failed"] == 1

    def test_all_metrics_error_gracefully(self):
        """If every metric errors on every pair, no crash."""
        def always_fail(a, b):
            raise RuntimeError("always fails")

        register_similarity_metric("fail", always_fail)
        pairs = [([1.0], [2.0], 1.0), ([3.0], [4.0], 0.0)]
        result = evaluate_metrics(pairs, metrics=["fail"])
        assert result["metrics_failed"] == 1
        assert result["metrics_evaluated"] == 0

    def test_empty_pairs_raises(self):
        with pytest.raises(ValueError, match="[Aa]t least 1|[Nn]o.*pairs|[Ee]mpty"):
            evaluate_metrics([])

    def test_unknown_metric_raises(self):
        pairs = [([1.0], [2.0], 1.0)]
        with pytest.raises(KeyError, match="nonexistent"):
            evaluate_metrics(pairs, metrics=["nonexistent"])


class TestEvaluateMetricsFiltered:
    """Compute only a subset of metrics."""

    def test_explicit_metric_list(self):
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine", "euclidean"])
        assert set(result["results"].keys()) == {"cosine", "euclidean"}

    def test_custom_metric_evaluatable(self):
        register_similarity_metric(
            "simple_dot", lambda a, b: sum(x * y for x, y in zip(a, b))
        )
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["simple_dot"])
        assert "simple_dot" in result["results"]
        entry = result["results"]["simple_dot"]
        assert entry["auc"] is not None  # binary labels, should compute


class TestEvaluateMetricsRanking:
    """The ranking field should order metrics by quality."""

    def test_ranking_puts_perfect_metric_first(self):
        """Cosine perfectly separates, euclidean doesn't (magnitude confounds)."""
        pairs = [
            # Similar: parallel but different magnitudes
            ([1.0, 0.0], [100.0, 0.0], 1.0),
            ([0.0, 1.0], [0.0, 50.0], 1.0),
            # Dissimilar: orthogonal but close in magnitude
            ([1.0, 0.0], [0.0, 1.0], 0.0),
            ([0.0, 2.0], [2.0, 0.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine", "euclidean"])
        # Cosine should rank higher (AUC = 1.0)
        assert result["ranking"][0] == "cosine"

    def test_ranking_contains_all_evaluated_metrics(self):
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine", "euclidean", "manhattan"])
        assert set(result["ranking"]) == {"cosine", "euclidean", "manhattan"}

    def test_failed_metrics_at_end_of_ranking(self):
        """Metrics that errored should appear at the end."""
        def bad(a, b):
            raise RuntimeError("fail")

        register_similarity_metric("bad", bad)
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),
            ([1.0, 0.0], [0.0, 1.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["cosine", "bad"])
        assert result["ranking"][-1] == "bad"


class TestEvaluateMetricsContinuousLabels:
    """Continuous relevance scores in [0, 1]."""

    def test_continuous_labels_rank_correlation(self):
        """Rank correlation should work with continuous labels."""
        # cosine increases with similarity, labels increase with similarity
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 1.0),     # identical → cosine=1.0
            ([1.0, 0.0], [1.0, 0.1], 0.8),     # very similar
            ([1.0, 0.0], [1.0, 1.0], 0.4),     # less similar
            ([1.0, 0.0], [0.0, 1.0], 0.0),     # orthogonal
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        rc = result["results"]["cosine"]["rank_correlation"]
        assert rc is not None
        assert rc > 0.5  # strong positive correlation

    def test_continuous_labels_auc_thresholds_at_half(self):
        """AUC should threshold labels at 0.5 for binary classification."""
        pairs = [
            ([1.0, 0.0], [1.0, 0.0], 0.9),   # similar (>= 0.5)
            ([1.0, 0.0], [1.0, 0.1], 0.7),   # similar (>= 0.5)
            ([1.0, 0.0], [0.0, 1.0], 0.2),   # dissimilar (< 0.5)
            ([1.0, 0.0], [-1.0, 0.0], 0.1),  # dissimilar (< 0.5)
        ]
        result = evaluate_metrics(pairs, metrics=["cosine"])
        auc = result["results"]["cosine"]["auc"]
        assert auc is not None
        assert auc > 0.5  # should separate reasonably


class TestEvaluateMetricsAUCMathematical:
    """Verify AUC computation is mathematically correct."""

    def test_auc_perfect_similarity(self):
        """All similar scores > all dissimilar scores → AUC = 1.0."""
        # Use a trivial metric where we control exact scores
        register_similarity_metric("always_high_sim", lambda a, b: a[0])
        pairs = [
            ([10.0], [0.0], 1.0),   # score = 10.0 (similar)
            ([9.0], [0.0], 1.0),    # score = 9.0  (similar)
            ([1.0], [0.0], 0.0),    # score = 1.0  (dissimilar)
            ([0.0], [0.0], 0.0),    # score = 0.0  (dissimilar)
        ]
        result = evaluate_metrics(pairs, metrics=["always_high_sim"])
        assert result["results"]["always_high_sim"]["auc"] == pytest.approx(1.0)

    def test_auc_perfect_anti_correlation(self):
        """All similar scores < all dissimilar scores → AUC = 0.0."""
        register_similarity_metric("inverted", lambda a, b: a[0])
        pairs = [
            ([0.0], [0.0], 1.0),    # score = 0.0 (similar)
            ([1.0], [0.0], 1.0),    # score = 1.0 (similar)
            ([9.0], [0.0], 0.0),    # score = 9.0 (dissimilar)
            ([10.0], [0.0], 0.0),   # score = 10.0 (dissimilar)
        ]
        result = evaluate_metrics(pairs, metrics=["inverted"])
        assert result["results"]["inverted"]["auc"] == pytest.approx(0.0)

    def test_auc_random_half(self):
        """Symmetrically interleaved scores → AUC = 0.5."""
        register_similarity_metric("alternating", lambda a, b: a[0])
        # pos_scores = [3.0, 2.0], neg_scores = [4.0, 1.0]
        # Pairwise: 3>4? N, 3>1? Y, 2>4? N, 2>1? Y → concordant=2, discordant=2
        # AUC = 2/4 = 0.5
        pairs = [
            ([3.0], [0.0], 1.0),   # similar, score=3.0
            ([4.0], [0.0], 0.0),   # dissimilar, score=4.0
            ([2.0], [0.0], 1.0),   # similar, score=2.0
            ([1.0], [0.0], 0.0),   # dissimilar, score=1.0
        ]
        result = evaluate_metrics(pairs, metrics=["alternating"])
        auc = result["results"]["alternating"]["auc"]
        assert auc == pytest.approx(0.5)

    def test_auc_with_tied_scores(self):
        """Tied scores between classes contribute 0.5 per pair."""
        register_similarity_metric("constant", lambda a, b: 5.0)
        pairs = [
            ([1.0], [0.0], 1.0),
            ([1.0], [0.0], 0.0),
        ]
        result = evaluate_metrics(pairs, metrics=["constant"])
        # All scores tied → AUC = 0.5
        assert result["results"]["constant"]["auc"] == pytest.approx(0.5)

