"""Tests for built-in similarity metric implementations (Step 2).

Each metric is tested for:
  - correctness on known inputs
  - mathematical properties (symmetry, identity, non-negativity, etc.)
  - input validation (dimension mismatch, empty, NaN, Inf, bool)
  - edge cases specific to that metric

We also test the shared validation helper to avoid regression if it is
refactored later.
"""

import math
import pytest
from jsonld_ex.similarity import (
    euclidean_distance,
    dot_product,
    manhattan_distance,
    chebyshev_distance,
    hamming_distance,
    jaccard_similarity,
)
from jsonld_ex.vector import cosine_similarity


# ── Shared validation behaviour ─────────────────────────────────────────
# All seven metrics must reject the same bad inputs.  We parameterise to
# avoid duplicating near-identical tests seven times.

ALL_METRICS = [
    ("cosine", cosine_similarity),
    ("euclidean", euclidean_distance),
    ("dot_product", dot_product),
    ("manhattan", manhattan_distance),
    ("chebyshev", chebyshev_distance),
    ("hamming", hamming_distance),
    ("jaccard", jaccard_similarity),
]


class TestSharedValidation:
    """Every metric must reject these inputs identically."""

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_dimension_mismatch(self, name, fn):
        with pytest.raises(ValueError, match="mismatch"):
            fn([1.0, 2.0], [1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_empty_vectors(self, name, fn):
        with pytest.raises(ValueError, match="empty"):
            fn([], [])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_nan_rejected(self, name, fn):
        with pytest.raises(ValueError, match="finite"):
            fn([float("nan"), 1.0], [1.0, 1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_inf_rejected(self, name, fn):
        with pytest.raises(ValueError, match="finite"):
            fn([1.0, 1.0], [float("inf"), 1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_neg_inf_rejected(self, name, fn):
        with pytest.raises(ValueError, match="finite"):
            fn([float("-inf"), 1.0], [1.0, 1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_bool_rejected(self, name, fn):
        with pytest.raises(TypeError, match="number"):
            fn([True, 1.0], [1.0, 1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_string_element_rejected(self, name, fn):
        with pytest.raises(TypeError, match="number"):
            fn([1.0, "bad"], [1.0, 1.0])

    @pytest.mark.parametrize("name,fn", ALL_METRICS, ids=[m[0] for m in ALL_METRICS])
    def test_integer_elements_accepted(self, name, fn):
        # Must not raise — ints are valid numeric types
        fn([1, 2, 3], [4, 5, 6])


# ── Euclidean distance ──────────────────────────────────────────────────

class TestEuclideanDistance:
    """L2 distance: d(a,b) = sqrt(sum((a_i - b_i)^2))."""

    def test_identical_vectors(self):
        assert euclidean_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value_2d(self):
        # (0,0) to (3,4) = 5.0
        assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)

    def test_known_value_1d(self):
        assert euclidean_distance([0.0], [7.0]) == pytest.approx(7.0)

    def test_negative_coordinates(self):
        # (-1, -1) to (2, 3) = sqrt(9 + 16) = 5.0
        assert euclidean_distance([-1.0, -1.0], [2.0, 3.0]) == pytest.approx(5.0)

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert euclidean_distance(a, b) == pytest.approx(euclidean_distance(b, a))

    def test_non_negativity(self):
        assert euclidean_distance([1.0, -2.0], [-3.0, 4.0]) >= 0.0

    def test_triangle_inequality(self):
        a, b, c = [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        d_ab = euclidean_distance(a, b)
        d_bc = euclidean_distance(b, c)
        d_ac = euclidean_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-12

    def test_zero_vectors(self):
        """Euclidean distance between zero vectors is 0 — well-defined."""
        assert euclidean_distance([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_one_zero_vector(self):
        """Distance from origin equals the vector's norm."""
        assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(5.0)

    def test_high_dimensional(self):
        n = 768
        a = [1.0] * n
        b = [0.0] * n
        # sqrt(768 * 1^2) = sqrt(768)
        assert euclidean_distance(a, b) == pytest.approx(math.sqrt(n))


# ── Dot product ─────────────────────────────────────────────────────────

class TestDotProduct:
    """Inner product: dot(a,b) = sum(a_i * b_i)."""

    def test_known_value(self):
        # [1,2,3] . [4,5,6] = 4 + 10 + 18 = 32
        assert dot_product([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) == pytest.approx(32.0)

    def test_orthogonal(self):
        assert dot_product([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_parallel_same_direction(self):
        assert dot_product([2.0, 0.0], [3.0, 0.0]) == pytest.approx(6.0)

    def test_parallel_opposite_direction(self):
        assert dot_product([2.0, 0.0], [-3.0, 0.0]) == pytest.approx(-6.0)

    def test_symmetry(self):
        a, b = [1.0, -2.0, 3.0], [4.0, 5.0, -6.0]
        assert dot_product(a, b) == pytest.approx(dot_product(b, a))

    def test_zero_vector(self):
        """dot(a, 0) = 0 for any a."""
        assert dot_product([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_linearity_scalar_factor(self):
        """dot(k*a, b) = k * dot(a, b)."""
        a, b = [1.0, 2.0], [3.0, 4.0]
        k = 2.5
        ka = [k * x for x in a]
        assert dot_product(ka, b) == pytest.approx(k * dot_product(a, b))

    def test_self_dot_is_squared_norm(self):
        """dot(a, a) = ||a||^2."""
        a = [3.0, 4.0]
        assert dot_product(a, a) == pytest.approx(25.0)

    def test_can_be_negative(self):
        """Dot product is NOT non-negative in general."""
        result = dot_product([1.0, 0.0], [-1.0, 0.0])
        assert result < 0

    def test_1d(self):
        assert dot_product([5.0], [3.0]) == pytest.approx(15.0)

    def test_high_dimensional(self):
        n = 768
        a = [1.0] * n
        b = [2.0] * n
        assert dot_product(a, b) == pytest.approx(2.0 * n)


# ── Manhattan distance ──────────────────────────────────────────────────

class TestManhattanDistance:
    """L1 / city-block distance: d(a,b) = sum(|a_i - b_i|)."""

    def test_identical_vectors(self):
        assert manhattan_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value_2d(self):
        # |0-3| + |0-4| = 7
        assert manhattan_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(7.0)

    def test_known_value_1d(self):
        assert manhattan_distance([0.0], [7.0]) == pytest.approx(7.0)

    def test_negative_coordinates(self):
        # |-1-2| + |-1-3| = 3 + 4 = 7
        assert manhattan_distance([-1.0, -1.0], [2.0, 3.0]) == pytest.approx(7.0)

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert manhattan_distance(a, b) == pytest.approx(manhattan_distance(b, a))

    def test_non_negativity(self):
        assert manhattan_distance([1.0, -2.0], [-3.0, 4.0]) >= 0.0

    def test_triangle_inequality(self):
        a, b, c = [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        d_ab = manhattan_distance(a, b)
        d_bc = manhattan_distance(b, c)
        d_ac = manhattan_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-12

    def test_zero_vectors(self):
        assert manhattan_distance([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_one_zero_vector(self):
        # L1 norm of [3, 4] = 7
        assert manhattan_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(7.0)

    def test_dominates_euclidean(self):
        """L1 >= L2 in all cases (Cauchy-Schwarz consequence)."""
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert manhattan_distance(a, b) >= euclidean_distance(a, b) - 1e-12

    def test_high_dimensional(self):
        n = 768
        a = [1.0] * n
        b = [0.0] * n
        # sum of |1 - 0| = 768
        assert manhattan_distance(a, b) == pytest.approx(float(n))


# ── Chebyshev distance ──────────────────────────────────────────────────

class TestChebyshevDistance:
    """L-infinity distance: d(a,b) = max(|a_i - b_i|)."""

    def test_identical_vectors(self):
        assert chebyshev_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value_2d(self):
        # max(|0-3|, |0-4|) = 4
        assert chebyshev_distance([0.0, 0.0], [3.0, 4.0]) == pytest.approx(4.0)

    def test_known_value_1d(self):
        assert chebyshev_distance([0.0], [7.0]) == pytest.approx(7.0)

    def test_negative_coordinates(self):
        # max(|-1-2|, |-1-3|) = max(3, 4) = 4
        assert chebyshev_distance([-1.0, -1.0], [2.0, 3.0]) == pytest.approx(4.0)

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert chebyshev_distance(a, b) == pytest.approx(chebyshev_distance(b, a))

    def test_non_negativity(self):
        assert chebyshev_distance([1.0, -2.0], [-3.0, 4.0]) >= 0.0

    def test_triangle_inequality(self):
        a, b, c = [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]
        d_ab = chebyshev_distance(a, b)
        d_bc = chebyshev_distance(b, c)
        d_ac = chebyshev_distance(a, c)
        assert d_ac <= d_ab + d_bc + 1e-12

    def test_zero_vectors(self):
        assert chebyshev_distance([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_single_large_difference_dominates(self):
        """Only the max component matters."""
        # Differences: |1-1|=0, |2-2|=0, |3-100|=97
        assert chebyshev_distance([1.0, 2.0, 3.0], [1.0, 2.0, 100.0]) == pytest.approx(97.0)

    def test_high_dimensional(self):
        n = 768
        a = [1.0] * n
        b = [0.0] * n
        # max(|1-0|) = 1.0 regardless of dimension
        assert chebyshev_distance(a, b) == pytest.approx(1.0)


# ── Hamming distance ────────────────────────────────────────────────────

class TestHammingDistance:
    """Hamming distance: count of positions where a_i != b_i."""

    def test_identical_vectors(self):
        assert hamming_distance([1, 0, 1, 1], [1, 0, 1, 1]) == 0

    def test_completely_different(self):
        assert hamming_distance([1, 0, 1, 0], [0, 1, 0, 1]) == 4

    def test_one_difference(self):
        assert hamming_distance([1, 0, 0], [1, 0, 1]) == 1

    def test_known_binary(self):
        assert hamming_distance([1, 1, 0, 0], [1, 0, 0, 1]) == 2

    def test_symmetry(self):
        a, b = [1, 0, 1, 0], [0, 0, 1, 1]
        assert hamming_distance(a, b) == hamming_distance(b, a)

    def test_non_negativity(self):
        assert hamming_distance([1, 0], [0, 1]) >= 0

    def test_triangle_inequality(self):
        a = [1, 0, 0, 0]
        b = [1, 1, 0, 0]
        c = [1, 1, 1, 0]
        d_ab = hamming_distance(a, b)
        d_bc = hamming_distance(b, c)
        d_ac = hamming_distance(a, c)
        assert d_ac <= d_ab + d_bc

    def test_returns_integer_value(self):
        """Hamming distance is always a non-negative integer."""
        result = hamming_distance([1, 0, 1], [0, 0, 1])
        assert result == int(result)

    def test_float_exact_equality(self):
        """Works on floats via exact equality — values that differ are counted."""
        assert hamming_distance([1.0, 2.0, 3.0], [1.0, 2.5, 3.0]) == 1

    def test_zero_vectors(self):
        assert hamming_distance([0, 0, 0], [0, 0, 0]) == 0

    def test_high_dimensional_binary(self):
        n = 768
        a = [1] * n
        b = [0] * n
        assert hamming_distance(a, b) == n


# ── Jaccard similarity ──────────────────────────────────────────────────

class TestJaccardSimilarity:
    """Binary set Jaccard: |A ∩ B| / |A ∪ B| where A = {i : a_i ≠ 0}."""

    def test_identical_binary(self):
        assert jaccard_similarity([1, 0, 1, 1], [1, 0, 1, 1]) == pytest.approx(1.0)

    def test_disjoint(self):
        # A = {0, 2}, B = {1, 3} → intersection empty → 0.0
        assert jaccard_similarity([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # A = {0, 1, 2}, B = {1, 2, 3} → |∩| = 2, |∪| = 4 → 0.5
        assert jaccard_similarity([1, 1, 1, 0], [0, 1, 1, 1]) == pytest.approx(0.5)

    def test_subset(self):
        # A = {0, 1}, B = {0, 1, 2} → |∩| = 2, |∪| = 3 → 2/3
        assert jaccard_similarity([1, 1, 0], [1, 1, 1]) == pytest.approx(2.0 / 3.0)

    def test_both_zero_vectors(self):
        """Convention: J(∅, ∅) = 1.0 — identical empty sets."""
        assert jaccard_similarity([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)

    def test_one_zero_vector(self):
        """One empty, one non-empty → 0.0."""
        assert jaccard_similarity([0, 0, 0], [1, 0, 1]) == pytest.approx(0.0)

    def test_symmetry(self):
        a, b = [1, 0, 1, 0], [0, 1, 1, 0]
        assert jaccard_similarity(a, b) == pytest.approx(jaccard_similarity(b, a))

    def test_bounded_zero_to_one(self):
        """Jaccard similarity is always in [0, 1]."""
        a, b = [1, 0, 1, 0, 1], [0, 1, 0, 1, 1]
        result = jaccard_similarity(a, b)
        assert 0.0 <= result <= 1.0

    def test_nonzero_values_treated_as_set_membership(self):
        """Any nonzero value (not just 1) counts as set membership."""
        # A = {0, 1, 2} (all nonzero), B = {0, 1, 2} → 1.0
        assert jaccard_similarity([5.0, 3.0, 7.0], [1.0, 2.0, 9.0]) == pytest.approx(1.0)

    def test_negative_values_are_nonzero(self):
        """Negative values count as set membership."""
        # A = {0, 1} (nonzero), B = {0, 1} → 1.0
        assert jaccard_similarity([-1.0, 2.0], [3.0, -4.0]) == pytest.approx(1.0)

    def test_mixed_zero_nonzero(self):
        # A = {0, 2} (indices with nonzero), B = {1, 2}
        # |∩| = 1 ({2}), |∪| = 3 ({0,1,2}) → 1/3
        assert jaccard_similarity([1.0, 0.0, 1.0], [0.0, 1.0, 1.0]) == pytest.approx(1.0 / 3.0)

    def test_high_dimensional(self):
        n = 768
        a = [1] * n
        b = [1] * n
        assert jaccard_similarity(a, b) == pytest.approx(1.0)


# ── Cross-metric consistency checks ─────────────────────────────────────

class TestCrossMetricConsistency:
    """Verify mathematical relationships that must hold across metrics."""

    def test_unit_vectors_cosine_vs_dot(self):
        """For unit vectors, cosine similarity equals dot product."""
        # Normalize [3, 4] -> [0.6, 0.8]
        a = [0.6, 0.8]
        b = [1.0, 0.0]
        cos = cosine_similarity(a, b)
        dot = dot_product(a, b)
        assert cos == pytest.approx(dot, abs=1e-9)

    def test_euclidean_from_dot_product(self):
        """||a-b||^2 = dot(a,a) - 2*dot(a,b) + dot(b,b)."""
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        d_sq = euclidean_distance(a, b) ** 2
        expansion = dot_product(a, a) - 2 * dot_product(a, b) + dot_product(b, b)
        assert d_sq == pytest.approx(expansion, abs=1e-9)

    def test_l1_geq_l2(self):
        """Manhattan distance >= Euclidean distance (always)."""
        a, b = [1.0, -3.0, 5.0, -7.0], [2.0, 4.0, -6.0, 8.0]
        assert manhattan_distance(a, b) >= euclidean_distance(a, b) - 1e-12

    def test_l2_geq_l1_over_sqrt_n(self):
        """Euclidean >= Manhattan / sqrt(n)  (norm equivalence bound)."""
        a, b = [1.0, -3.0, 5.0, -7.0], [2.0, 4.0, -6.0, 8.0]
        n = len(a)
        assert euclidean_distance(a, b) >= manhattan_distance(a, b) / math.sqrt(n) - 1e-12

    def test_chebyshev_leq_euclidean(self):
        """Chebyshev (L∞) <= Euclidean (L2) — max component <= root of sum of squares."""
        a, b = [1.0, -3.0, 5.0, -7.0], [2.0, 4.0, -6.0, 8.0]
        assert chebyshev_distance(a, b) <= euclidean_distance(a, b) + 1e-12

    def test_chebyshev_leq_manhattan(self):
        """Chebyshev (L∞) <= Manhattan (L1) — max <= sum."""
        a, b = [1.0, -3.0, 5.0, -7.0], [2.0, 4.0, -6.0, 8.0]
        assert chebyshev_distance(a, b) <= manhattan_distance(a, b) + 1e-12

    def test_lp_norm_ordering(self):
        """L∞ <= L2 <= L1 — fundamental norm ordering."""
        a, b = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]
        l_inf = chebyshev_distance(a, b)
        l2 = euclidean_distance(a, b)
        l1 = manhattan_distance(a, b)
        assert l_inf <= l2 + 1e-12
        assert l2 <= l1 + 1e-12

    def test_hamming_leq_dimension(self):
        """Hamming distance is bounded by vector dimension."""
        a, b = [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]
        assert hamming_distance(a, b) <= len(a)

    def test_jaccard_complement_of_distance(self):
        """For binary vectors: Jaccard distance = 1 - Jaccard similarity."""
        a, b = [1, 0, 1, 1, 0], [1, 1, 0, 1, 0]
        j = jaccard_similarity(a, b)
        assert 0.0 <= j <= 1.0
        assert (1.0 - j) >= 0.0  # distance is non-negative
