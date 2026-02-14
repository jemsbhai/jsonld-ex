"""Tests for the similarity_examples registry recipes.

Pure-Python metrics are tested unconditionally.  NumPy / SciPy metrics
are skipped when the dependency is absent.
"""

import math
import pytest

from jsonld_ex.similarity import register_similarity_metric, reset_similarity_registry
from jsonld_ex.similarity_examples import (
    canberra_distance,
    bray_curtis_dissimilarity,
    haversine_distance,
    dtw_distance,
)

# Optional imports — guarded
np = pytest.importorskip("numpy", reason="numpy required") if False else None
try:
    import numpy as np  # type: ignore[no-redef]

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import scipy  # type: ignore[import-untyped]

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_similarity_registry()
    yield
    reset_similarity_registry()


# ── Canberra distance ───────────────────────────────────────────────────

class TestCanberraDistance:
    def test_identical(self):
        assert canberra_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self):
        # |1-2|/(1+2) + |3-4|/(3+4) = 1/3 + 1/7
        expected = 1.0 / 3.0 + 1.0 / 7.0
        assert canberra_distance([1.0, 3.0], [2.0, 4.0]) == pytest.approx(expected)

    def test_both_zero_element(self):
        """0/0 convention: contributes 0."""
        assert canberra_distance([0.0, 1.0], [0.0, 2.0]) == pytest.approx(1.0 / 3.0)

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert canberra_distance(a, b) == pytest.approx(canberra_distance(b, a))

    def test_non_negativity(self):
        assert canberra_distance([1.0, -2.0], [-3.0, 4.0]) >= 0.0

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError, match="mismatch"):
            canberra_distance([1.0], [1.0, 2.0])

    def test_registrable(self):
        register_similarity_metric("canberra", canberra_distance)
        from jsonld_ex.similarity import get_similarity_metric

        assert get_similarity_metric("canberra") is canberra_distance


# ── Bray-Curtis dissimilarity ───────────────────────────────────────────

class TestBrayCurtis:
    def test_identical(self):
        assert bray_curtis_dissimilarity([1.0, 2.0], [1.0, 2.0]) == 0.0

    def test_known_value(self):
        # |1-3| + |2-4| = 2+2 = 4,  (1+3) + (2+4) = 10 → 4/10 = 0.4
        assert bray_curtis_dissimilarity([1.0, 2.0], [3.0, 4.0]) == pytest.approx(0.4)

    def test_both_zero(self):
        assert bray_curtis_dissimilarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_bounded_for_non_negative(self):
        result = bray_curtis_dissimilarity([1.0, 5.0, 3.0], [2.0, 1.0, 4.0])
        assert 0.0 <= result <= 1.0

    def test_symmetry(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert bray_curtis_dissimilarity(a, b) == pytest.approx(
            bray_curtis_dissimilarity(b, a)
        )


# ── Haversine distance ──────────────────────────────────────────────────

class TestHaversineDistance:
    def test_same_point(self):
        assert haversine_distance([40.0, -74.0], [40.0, -74.0]) == pytest.approx(0.0)

    def test_known_distance_nyc_london(self):
        # NYC (40.7128, -74.0060) to London (51.5074, -0.1278)
        # Expected ~5570 km
        d = haversine_distance([40.7128, -74.0060], [51.5074, -0.1278])
        assert 5500 < d < 5650

    def test_antipodal(self):
        """Opposite sides of Earth ≈ half circumference ≈ 20015 km."""
        d = haversine_distance([0.0, 0.0], [0.0, 180.0])
        assert 20000 < d < 20100

    def test_symmetry(self):
        a, b = [40.7128, -74.006], [51.5074, -0.1278]
        assert haversine_distance(a, b) == pytest.approx(haversine_distance(b, a))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-element"):
            haversine_distance([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])


# ── DTW distance ────────────────────────────────────────────────────────

class TestDTWDistance:
    def test_identical(self):
        assert dtw_distance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_shifted_sequence(self):
        """DTW should handle shifted patterns gracefully."""
        # [1, 2, 3] vs [2, 3, 4]: cost = |1-2| + |2-3| + |3-4| = 3
        # But DTW can align better: 1→2(1), 2→3(1), 3→4(1) = 3
        result = dtw_distance([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert result >= 0.0

    def test_constant_vs_ramp(self):
        a = [0.0, 0.0, 0.0]
        b = [0.0, 1.0, 2.0]
        result = dtw_distance(a, b)
        assert result > 0.0

    def test_symmetry(self):
        a, b = [1.0, 3.0, 5.0], [2.0, 4.0, 6.0]
        assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))

    def test_non_negativity(self):
        assert dtw_distance([1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]) >= 0.0


# ── Mahalanobis (factory, requires numpy) ───────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestMahalanobis:
    def test_identity_covariance_equals_euclidean(self):
        """With identity covariance, Mahalanobis == Euclidean."""
        from jsonld_ex.similarity_examples import make_mahalanobis
        from jsonld_ex.similarity import euclidean_distance

        # 3x3 identity
        inv_cov = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        maha = make_mahalanobis(inv_cov)
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        assert maha(a, b) == pytest.approx(euclidean_distance(a, b))

    def test_registrable(self):
        from jsonld_ex.similarity_examples import make_mahalanobis

        inv_cov = [[1.0, 0.0], [0.0, 1.0]]
        maha = make_mahalanobis(inv_cov)
        register_similarity_metric("mahalanobis", maha)
        from jsonld_ex.similarity import get_similarity_metric

        assert get_similarity_metric("mahalanobis") is maha

    def test_shape_mismatch_raises(self):
        from jsonld_ex.similarity_examples import make_mahalanobis

        inv_cov = [[1.0, 0.0], [0.0, 1.0]]  # 2x2
        maha = make_mahalanobis(inv_cov)
        with pytest.raises(ValueError, match="shape"):
            maha([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])  # 3-dim vectors


# ── Soft Cosine (factory, requires numpy) ───────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestSoftCosine:
    def test_identity_matrix_equals_cosine(self):
        """With identity similarity matrix, soft cosine == cosine."""
        from jsonld_ex.similarity_examples import make_soft_cosine
        from jsonld_ex.vector import cosine_similarity

        sim = [[1.0, 0.0], [0.0, 1.0]]
        sc = make_soft_cosine(sim)
        a, b = [3.0, 4.0], [1.0, 0.0]
        assert sc(a, b) == pytest.approx(cosine_similarity(a, b), abs=1e-9)

    def test_correlated_features(self):
        """Features that are similar should boost the similarity score."""
        from jsonld_ex.similarity_examples import make_soft_cosine

        # Features 0 and 1 are 0.8 similar
        sim = [[1.0, 0.8], [0.8, 1.0]]
        sc = make_soft_cosine(sim)
        # a only has feature 0, b only has feature 1
        result = sc([1.0, 0.0], [0.0, 1.0])
        # With identity matrix this would be 0.0; with correlation it should be > 0
        assert result > 0.0

    def test_shape_mismatch_raises(self):
        from jsonld_ex.similarity_examples import make_soft_cosine

        sim = [[1.0, 0.0], [0.0, 1.0]]
        sc = make_soft_cosine(sim)
        with pytest.raises(ValueError, match="shape"):
            sc([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])


# ── KL divergence (requires scipy) ─────────────────────────────────────

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestKLDivergence:
    def test_identical_distributions(self):
        from jsonld_ex.similarity_examples import kl_divergence

        p = [0.25, 0.25, 0.25, 0.25]
        assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_asymmetric(self):
        from jsonld_ex.similarity_examples import kl_divergence

        p = [0.5, 0.5]
        q = [0.9, 0.1]
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp)

    def test_non_negative(self):
        from jsonld_ex.similarity_examples import kl_divergence

        p = [0.3, 0.7]
        q = [0.6, 0.4]
        assert kl_divergence(p, q) >= 0.0

    def test_non_distribution_raises(self):
        from jsonld_ex.similarity_examples import kl_divergence

        with pytest.raises(ValueError, match="sum to"):
            kl_divergence([0.5, 0.3], [0.5, 0.5])  # a sums to 0.8


# ── Wasserstein distance (requires scipy) ───────────────────────────────

@pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")
class TestWassersteinDistance:
    def test_identical(self):
        from jsonld_ex.similarity_examples import wasserstein_distance

        p = [0.25, 0.25, 0.25, 0.25]
        assert wasserstein_distance(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_symmetry(self):
        from jsonld_ex.similarity_examples import wasserstein_distance

        p = [0.5, 0.3, 0.2]
        q = [0.1, 0.6, 0.3]
        assert wasserstein_distance(p, q) == pytest.approx(
            wasserstein_distance(q, p)
        )


# ── Spearman rank correlation (requires numpy) ─────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestSpearmanRank:
    def test_perfect_positive(self):
        from jsonld_ex.similarity_examples import spearman_rank_correlation

        assert spearman_rank_correlation(
            [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
        ) == pytest.approx(1.0)

    def test_perfect_negative(self):
        from jsonld_ex.similarity_examples import spearman_rank_correlation

        assert spearman_rank_correlation(
            [1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]
        ) == pytest.approx(-1.0)

    def test_no_correlation(self):
        from jsonld_ex.similarity_examples import spearman_rank_correlation

        # Not perfectly 0, but close for a known uncorrelated example
        result = spearman_rank_correlation(
            [1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 1.0, 5.0, 3.0]
        )
        assert -1.0 <= result <= 1.0


# ── Kendall's Tau (requires numpy) ──────────────────────────────────────

@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
class TestKendallTau:
    def test_perfect_concordance(self):
        from jsonld_ex.similarity_examples import kendall_tau

        assert kendall_tau(
            [1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]
        ) == pytest.approx(1.0)

    def test_perfect_discordance(self):
        from jsonld_ex.similarity_examples import kendall_tau

        assert kendall_tau(
            [1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]
        ) == pytest.approx(-1.0)

    def test_bounded(self):
        from jsonld_ex.similarity_examples import kendall_tau

        result = kendall_tau(
            [1.0, 5.0, 2.0, 4.0, 3.0], [3.0, 1.0, 5.0, 2.0, 4.0]
        )
        assert -1.0 <= result <= 1.0
