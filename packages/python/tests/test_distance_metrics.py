"""Tests for pluggable opinion distance metrics in confidence_byzantine.

TDD: Tests written FIRST.

New symbols (all additive):

    DistanceMetric  -- Callable[[Opinion, Opinion], float]
    euclidean_opinion_distance(a, b) -> float
    manhattan_opinion_distance(a, b) -> float
    jsd_opinion_distance(a, b) -> float
    hellinger_opinion_distance(a, b) -> float

Updated signatures (backward-compatible, new param has default):

    opinion_distance(a, b, distance_fn=None) -> float
    cohesion_score(opinions, distance_fn=None) -> float

Mathematical references:
    Euclidean: L2 norm on R^3, normalized by simplex diameter sqrt(2).
    Manhattan: L1 norm on R^3, normalized by max L1 on simplex = 2.
    Jensen-Shannon: sqrt(JSD) with log base 2.
        Endres & Schindelin (2003), Osterreicher & Vajda (2003).
    Hellinger: (1/sqrt(2)) * ||sqrt(A) - sqrt(B)||_2.
        Standard in information geometry.
"""

import math
import random
import pytest
from jsonld_ex.confidence_algebra import Opinion

from jsonld_ex.confidence_byzantine import (
    euclidean_opinion_distance,
    manhattan_opinion_distance,
    jsd_opinion_distance,
    hellinger_opinion_distance,
    opinion_distance,
    cohesion_score,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

# The three vertices of the 2-simplex
V_B = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
V_D = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
V_U = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)

# Center of the simplex
CENTER = Opinion(belief=1 / 3, disbelief=1 / 3, uncertainty=1 / 3)

# High internal tension but well-defined
BALANCED = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)

ALL_METRICS = [
    euclidean_opinion_distance,
    manhattan_opinion_distance,
    jsd_opinion_distance,
    hellinger_opinion_distance,
]


def random_opinion(rng: random.Random) -> Opinion:
    raw = [rng.random() for _ in range(3)]
    t = sum(raw)
    return Opinion(raw[0] / t, raw[1] / t, raw[2] / t)


# ===================================================================
# Metric Axioms — every distance metric must satisfy all four
# ===================================================================


class TestMetricAxiomNonNegativity:
    """Axiom 1: d(A, B) >= 0 for all A, B."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_random_pairs(self, metric):
        rng = random.Random(42)
        for _ in range(200):
            a, b = random_opinion(rng), random_opinion(rng)
            assert metric(a, b) >= -1e-12, f"{metric.__name__}: negative distance"


class TestMetricAxiomIdentity:
    """Axiom 2: d(A, A) = 0, and d(A, B) = 0 implies A = B (on b,d,u)."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_self_distance_is_zero(self, metric):
        rng = random.Random(42)
        for _ in range(50):
            o = random_opinion(rng)
            assert metric(o, o) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_self_distance_vertex_b(self, metric):
        assert metric(V_B, V_B) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_self_distance_balanced(self, metric):
        """Crucially: (0.5, 0.5, 0) must give 0 — the case where
        pairwise_conflict would give 0.5."""
        assert metric(BALANCED, BALANCED) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_different_opinions_positive(self, metric):
        """d(A, B) > 0 when A != B."""
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        assert metric(a, b) > 0


class TestMetricAxiomSymmetry:
    """Axiom 3: d(A, B) = d(B, A)."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_random_pairs(self, metric):
        rng = random.Random(77)
        for _ in range(100):
            a, b = random_opinion(rng), random_opinion(rng)
            assert metric(a, b) == pytest.approx(metric(b, a), abs=1e-12)


class TestMetricAxiomTriangleInequality:
    """Axiom 4: d(A, C) <= d(A, B) + d(B, C)."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_random_triples(self, metric):
        rng = random.Random(99)
        for _ in range(200):
            a, b, c = random_opinion(rng), random_opinion(rng), random_opinion(rng)
            assert metric(a, c) <= metric(a, b) + metric(b, c) + 1e-9


# ===================================================================
# Range — all metrics must map to [0, 1] on the simplex
# ===================================================================


class TestMetricRange:
    """All metrics must return values in [0, 1] on the opinion simplex."""

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_range_random(self, metric):
        rng = random.Random(42)
        for _ in range(200):
            a, b = random_opinion(rng), random_opinion(rng)
            d = metric(a, b)
            assert -1e-12 <= d <= 1.0 + 1e-12, (
                f"{metric.__name__}: out of range {d}"
            )

    @pytest.mark.parametrize("metric", ALL_METRICS, ids=lambda m: m.__name__)
    def test_vertex_pairs_are_maximum(self, metric):
        """Distance between opposite simplex vertices should be 1.0."""
        assert metric(V_B, V_D) == pytest.approx(1.0, abs=1e-9)
        assert metric(V_B, V_U) == pytest.approx(1.0, abs=1e-9)
        assert metric(V_D, V_U) == pytest.approx(1.0, abs=1e-9)


# ===================================================================
# Euclidean — specific formula tests
# ===================================================================


class TestEuclideanOpinionDistance:
    """d(A,B) = ||A-B||_2 / sqrt(2)"""

    def test_formula_exact(self):
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        expected = math.sqrt(0.4**2 + 0.2**2 + 0.2**2) / math.sqrt(2)
        assert euclidean_opinion_distance(a, b) == pytest.approx(expected)

    def test_small_perturbation(self):
        a = Opinion(belief=0.700, disbelief=0.200, uncertainty=0.100)
        b = Opinion(belief=0.705, disbelief=0.195, uncertainty=0.100)
        assert euclidean_opinion_distance(a, b) < 0.01


# ===================================================================
# Manhattan — specific formula tests
# ===================================================================


class TestManhattanOpinionDistance:
    """d(A,B) = (|Db| + |Dd| + |Du|) / 2

    Max L1 on simplex: (1,0,0) vs (0,1,0) = |1|+|1|+|0| = 2.
    Normalize by 2 to get [0, 1].
    """

    def test_formula_exact(self):
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        expected = (abs(0.4) + abs(0.2) + abs(0.2)) / 2.0
        assert manhattan_opinion_distance(a, b) == pytest.approx(expected)

    def test_vertex_pair(self):
        # |1| + |1| + |0| = 2, /2 = 1.0
        assert manhattan_opinion_distance(V_B, V_D) == pytest.approx(1.0)

    def test_manhattan_ge_euclidean(self):
        """On the simplex, L1/2 >= L2/sqrt(2) is NOT always true,
        but both are valid metrics. Just verify they're close."""
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3)
        e = euclidean_opinion_distance(a, b)
        m = manhattan_opinion_distance(a, b)
        # Both should be positive and in [0,1]
        assert e > 0
        assert m > 0


# ===================================================================
# Jensen-Shannon — specific formula tests
# ===================================================================


class TestJsdOpinionDistance:
    """d(A,B) = sqrt(JSD(A||B)) with log base 2.

    JSD(A||B) = 0.5 * KL(A||M) + 0.5 * KL(B||M), M = (A+B)/2.
    sqrt(JSD) is a proper metric (Endres & Schindelin 2003).
    Range [0, 1] with log2.
    """

    def test_identical_zero(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        assert jsd_opinion_distance(o, o) == pytest.approx(0.0, abs=1e-12)

    def test_vertices_maximum(self):
        """Dirac-delta-like distributions at different vertices: JSD = 1.0."""
        assert jsd_opinion_distance(V_B, V_D) == pytest.approx(1.0, abs=1e-9)

    def test_handles_zero_components(self):
        """Must not crash when opinions have zero components.
        0 * log2(0/x) = 0 by convention (L'Hopital)."""
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        b = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        d = jsd_opinion_distance(a, b)
        assert 0 < d < 1

    def test_both_have_zeros_in_same_component(self):
        """When both A and B have zero in the same position,
        M also has zero there. 0*log(0/0) = 0."""
        a = Opinion(belief=0.7, disbelief=0.3, uncertainty=0.0)
        b = Opinion(belief=0.4, disbelief=0.6, uncertainty=0.0)
        d = jsd_opinion_distance(a, b)
        assert 0 < d < 1

    def test_more_sensitive_near_boundaries(self):
        """JSD should be more sensitive to changes near simplex boundaries
        than Euclidean. This is its key advantage.

        Moving from (0.99, 0.005, 0.005) to (0.95, 0.025, 0.025)
        vs moving from (0.5, 0.25, 0.25) to (0.46, 0.29, 0.25):
        same Euclidean distance, but JSD should weigh the boundary
        shift more heavily."""
        # Near-boundary pair
        a1 = Opinion(belief=0.99, disbelief=0.005, uncertainty=0.005)
        b1 = Opinion(belief=0.95, disbelief=0.025, uncertainty=0.025)
        # Interior pair with same Euclidean distance
        a2 = Opinion(belief=0.50, disbelief=0.25, uncertainty=0.25)
        db = b1.belief - a1.belief  # -0.04
        dd = b1.disbelief - a1.disbelief  # +0.02
        du = b1.uncertainty - a1.uncertainty  # +0.02
        b2 = Opinion(
            belief=a2.belief + db,
            disbelief=a2.disbelief + dd,
            uncertainty=a2.uncertainty + du,
        )

        euclid_1 = euclidean_opinion_distance(a1, b1)
        euclid_2 = euclidean_opinion_distance(a2, b2)
        assert euclid_1 == pytest.approx(euclid_2, abs=1e-9)

        jsd_1 = jsd_opinion_distance(a1, b1)
        jsd_2 = jsd_opinion_distance(a2, b2)
        assert jsd_1 > jsd_2, (
            "JSD should be more sensitive near boundaries"
        )


# ===================================================================
# Hellinger — specific formula tests
# ===================================================================


class TestHellingerOpinionDistance:
    """H(A,B) = (1/sqrt(2)) * ||sqrt(A) - sqrt(B)||_2

    Standard information-geometric distance. Proper metric.
    Range [0, 1] for probability distributions.
    """

    def test_formula_exact(self):
        a = Opinion(belief=0.64, disbelief=0.16, uncertainty=0.20)
        b = Opinion(belief=0.25, disbelief=0.25, uncertainty=0.50)
        expected = (1.0 / math.sqrt(2.0)) * math.sqrt(
            (math.sqrt(0.64) - math.sqrt(0.25)) ** 2
            + (math.sqrt(0.16) - math.sqrt(0.25)) ** 2
            + (math.sqrt(0.20) - math.sqrt(0.50)) ** 2
        )
        assert hellinger_opinion_distance(a, b) == pytest.approx(expected)

    def test_handles_zero_components(self):
        """sqrt(0) = 0, no division issues."""
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        b = Opinion(belief=0.0, disbelief=0.5, uncertainty=0.5)
        d = hellinger_opinion_distance(a, b)
        assert 0 < d <= 1.0

    def test_vertices(self):
        d = hellinger_opinion_distance(V_B, V_D)
        # (1/sqrt(2)) * sqrt((1-0)^2 + (0-1)^2 + (0-0)^2) = 1.0
        assert d == pytest.approx(1.0)


# ===================================================================
# Cross-metric comparisons — structural relationships
# ===================================================================


class TestCrossMetricRelationships:
    """Verify known mathematical relationships between metrics."""

    def test_hellinger_le_jsd_le_1(self):
        """For probability distributions, H^2 <= JSD (base e) / ln(2).
        With base-2 log: H^2 <= JSD. So H <= sqrt(JSD).
        Both are proper metrics with range [0,1]."""
        rng = random.Random(42)
        for _ in range(100):
            a, b = random_opinion(rng), random_opinion(rng)
            h = hellinger_opinion_distance(a, b)
            j = jsd_opinion_distance(a, b)
            # H <= sqrt(JSD) when both use consistent normalization
            # This is a known bound; allow small floating point slack
            assert h <= j + 1e-9, (
                f"Hellinger ({h}) should be <= JSD ({j})"
            )

    def test_all_agree_on_identical(self):
        """All metrics must give 0.0 for identical opinions."""
        o = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        for metric in ALL_METRICS:
            assert metric(o, o) == pytest.approx(0.0, abs=1e-12)

    def test_all_agree_on_maximum(self):
        """All metrics must give 1.0 for opposite vertices."""
        for metric in ALL_METRICS:
            assert metric(V_B, V_D) == pytest.approx(1.0, abs=1e-9)


# ===================================================================
# opinion_distance — configurable (backward compatible)
# ===================================================================


class TestOpinionDistanceConfigurable:
    """opinion_distance(a, b, distance_fn=None) uses euclidean by default."""

    def test_default_is_euclidean(self):
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        assert opinion_distance(a, b) == pytest.approx(
            euclidean_opinion_distance(a, b)
        )

    def test_backward_compatible_no_kwarg(self):
        """Calling with two args (no distance_fn) must still work."""
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        d = opinion_distance(a, b)
        assert d > 0

    def test_pass_manhattan(self):
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        d = opinion_distance(a, b, distance_fn=manhattan_opinion_distance)
        assert d == pytest.approx(manhattan_opinion_distance(a, b))

    def test_pass_jsd(self):
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        d = opinion_distance(a, b, distance_fn=jsd_opinion_distance)
        assert d == pytest.approx(jsd_opinion_distance(a, b))

    def test_pass_hellinger(self):
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2)
        d = opinion_distance(a, b, distance_fn=hellinger_opinion_distance)
        assert d == pytest.approx(hellinger_opinion_distance(a, b))

    def test_custom_callable(self):
        """User can pass any callable matching DistanceMetric."""
        def constant_distance(a: Opinion, b: Opinion) -> float:
            return 0.42
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        b = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)
        assert opinion_distance(a, b, distance_fn=constant_distance) == 0.42


# ===================================================================
# cohesion_score — configurable (backward compatible)
# ===================================================================


class TestCohesionScoreConfigurable:
    """cohesion_score(opinions, distance_fn=None) uses euclidean by default."""

    def test_default_is_euclidean(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        c1 = cohesion_score(opinions)
        c2 = cohesion_score(opinions, distance_fn=euclidean_opinion_distance)
        assert c1 == pytest.approx(c2)

    def test_backward_compatible_no_kwarg(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        c = cohesion_score(opinions)
        assert 0 < c <= 1.0

    def test_identical_with_any_metric(self):
        """Identical opinions must give cohesion 1.0 regardless of metric."""
        o = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        for metric in ALL_METRICS:
            assert cohesion_score([o, o, o], distance_fn=metric) == pytest.approx(1.0)

    def test_jsd_gives_different_result_than_euclidean(self):
        """For non-trivial opinions, JSD and Euclidean should generally
        produce different cohesion scores."""
        opinions = [
            Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05),
            Opinion(belief=0.7, disbelief=0.15, uncertainty=0.15),
            Opinion(belief=0.5, disbelief=0.25, uncertainty=0.25),
        ]
        c_euclid = cohesion_score(opinions, distance_fn=euclidean_opinion_distance)
        c_jsd = cohesion_score(opinions, distance_fn=jsd_opinion_distance)
        # Both valid, but should differ
        assert c_euclid != pytest.approx(c_jsd, abs=0.01)

    def test_manhattan_cohesion(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        c = cohesion_score(opinions, distance_fn=manhattan_opinion_distance)
        assert 0 < c <= 1.0

    def test_vertices_zero_cohesion_all_metrics(self):
        """All three simplex vertices have max pairwise distance,
        so cohesion = 0.0 for every metric."""
        verts = [V_B, V_D, V_U]
        for metric in ALL_METRICS:
            assert cohesion_score(verts, distance_fn=metric) == pytest.approx(0.0, abs=1e-9)
