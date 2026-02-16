"""Example custom similarity metrics for the jsonld-ex registry.

This module provides **ready-to-use recipes** for domain-specific
similarity and distance functions.  None of these are registered
automatically — users opt in by calling
:func:`~jsonld_ex.similarity.register_similarity_metric`.

Each function follows the standard ``(list[float], list[float]) -> float``
signature.  Functions that require additional parameters (covariance
matrices, feature similarity matrices, etc.) are implemented as
**factory functions** that return a closure with the correct signature.

Dependency notes
----------------
- **Pure-Python metrics** (canberra, bray_curtis): no external deps.
- **NumPy metrics** (mahalanobis, soft_cosine, spearman, kendall_tau):
  require ``numpy``.  Import guarded.
- **SciPy metrics** (wasserstein, kl_divergence): require ``scipy``.
  Import guarded.
- **Geo metrics** (haversine): pure-Python, planned for future
  ``jsonld-ex[geo]`` extra.
- **DTW**: pure-Python reference impl; production use should prefer
  ``dtw-python`` or ``tslearn``.

Usage example::

    from jsonld_ex.similarity import register_similarity_metric
    from jsonld_ex.similarity_examples import canberra_distance

    register_similarity_metric("canberra", canberra_distance)
"""

from __future__ import annotations

import math
from typing import Callable, List

from jsonld_ex.similarity import _validate_vector_pair, MetricProperties, SimilarityFunction


# =====================================================================
# Pure-Python metrics (no external dependencies)
# =====================================================================


def canberra_distance(a: list[float], b: list[float]) -> float:
    """Compute Canberra distance between two vectors.

    Returns ``sum(|a_i - b_i| / (|a_i| + |b_i|))``.  Dimensions where
    both values are zero contribute 0.0 (0/0 convention).

    Sensitive to values near zero — useful in ecology, composition data,
    and contexts where small differences at small magnitudes matter.
    """
    _validate_vector_pair(a, b)
    total = 0.0
    for x, y in zip(a, b):
        denom = abs(x) + abs(y)
        if denom != 0:
            total += abs(x - y) / denom
    return total


def bray_curtis_dissimilarity(a: list[float], b: list[float]) -> float:
    """Compute Bray-Curtis dissimilarity between two vectors.

    Returns ``sum(|a_i - b_i|) / sum(|a_i| + |b_i|)``.  Bounded to
    [0, 1] for non-negative inputs.  Returns 0.0 when both vectors are
    all-zero (convention: identical empty samples).

    Common in ecology and microbiome analysis for comparing species
    abundance profiles.
    """
    _validate_vector_pair(a, b)
    numerator = sum(abs(x - y) for x, y in zip(a, b))
    denominator = sum(abs(x) + abs(y) for x, y in zip(a, b))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def haversine_distance(a: list[float], b: list[float]) -> float:
    """Compute great-circle distance between two geographic points.

    Expects **exactly 2-element vectors**: ``[latitude, longitude]`` in
    **degrees**.  Returns distance in **kilometres**.

    Uses the Haversine formula with Earth's mean radius (6371.0088 km).

    .. note::

       Planned for the future ``jsonld-ex[geo]`` extra.
    """
    _validate_vector_pair(a, b)
    if len(a) != 2:
        raise ValueError(
            f"Haversine requires 2-element vectors [lat, lon], got {len(a)} elements"
        )
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def dtw_distance(a: list[float], b: list[float]) -> float:
    """Compute Dynamic Time Warping distance (1-D reference implementation).

    Uses a full-matrix DP approach with Euclidean element-wise cost.
    This is an O(n*m) reference implementation — for production use on
    long sequences, prefer ``dtw-python`` or ``tslearn``.

    .. note::

       Unlike other metrics, DTW is meaningful on **sequences** and does
       not require equal lengths.  However, the registry signature
       requires matched dimensions, so both vectors must have the same
       length in this implementation.
    """
    _validate_vector_pair(a, b)
    n, m = len(a), len(b)
    # Full cost matrix
    dtw_matrix = [[math.inf] * (m + 1) for _ in range(n + 1)]
    dtw_matrix[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i - 1][j],      # insertion
                dtw_matrix[i][j - 1],      # deletion
                dtw_matrix[i - 1][j - 1],  # match
            )
    return dtw_matrix[n][m]


# =====================================================================
# Factory functions (metrics requiring extra parameters)
# =====================================================================


def make_mahalanobis(
    cov_inv: list[list[float]],
) -> SimilarityFunction:
    """Create a Mahalanobis distance function with a fixed inverse covariance.

    Parameters
    ----------
    cov_inv:
        The **inverse** of the covariance matrix, as a list of lists.
        Must be square with side length equal to the vector dimension.

    Returns
    -------
    SimilarityFunction
        A closure ``(a, b) -> float`` suitable for
        :func:`register_similarity_metric`.

    Requires ``numpy`` at **call** time (not at factory time).

    Example::

        import numpy as np
        from jsonld_ex.similarity import register_similarity_metric
        from jsonld_ex.similarity_examples import make_mahalanobis

        cov = np.cov(data.T)
        metric = make_mahalanobis(np.linalg.inv(cov).tolist())
        register_similarity_metric("mahalanobis", metric)
    """

    def _mahalanobis(a: list[float], b: list[float]) -> float:
        _validate_vector_pair(a, b)
        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Mahalanobis distance requires numpy. "
                "Install with: pip install numpy"
            ) from exc
        diff = np.array(a) - np.array(b)
        vi = np.array(cov_inv)
        if vi.shape != (len(a), len(a)):
            raise ValueError(
                f"Inverse covariance shape {vi.shape} does not match "
                f"vector dimension {len(a)}"
            )
        return float(np.sqrt(diff @ vi @ diff))

    _mahalanobis.__doc__ = "Mahalanobis distance (fixed inverse covariance)."
    return _mahalanobis


def make_soft_cosine(
    similarity_matrix: list[list[float]],
) -> SimilarityFunction:
    """Create a Soft Cosine similarity function with a fixed feature similarity matrix.

    Parameters
    ----------
    similarity_matrix:
        A symmetric matrix where entry (i, j) represents the similarity
        between features i and j.  Diagonal should be 1.0.

    Returns
    -------
    SimilarityFunction
        A closure ``(a, b) -> float``.

    Requires ``numpy``.

    Example::

        from jsonld_ex.similarity_examples import make_soft_cosine

        # Feature similarity: features 0 and 1 are 0.5 similar
        sim = [[1.0, 0.5], [0.5, 1.0]]
        metric = make_soft_cosine(sim)
    """

    def _soft_cosine(a: list[float], b: list[float]) -> float:
        _validate_vector_pair(a, b)
        try:
            import numpy as np  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Soft Cosine similarity requires numpy. "
                "Install with: pip install numpy"
            ) from exc
        va, vb = np.array(a), np.array(b)
        s = np.array(similarity_matrix)
        if s.shape != (len(a), len(a)):
            raise ValueError(
                f"Similarity matrix shape {s.shape} does not match "
                f"vector dimension {len(a)}"
            )
        num = float(va @ s @ vb)
        da = float(np.sqrt(va @ s @ va))
        db = float(np.sqrt(vb @ s @ vb))
        if da == 0 or db == 0:
            raise ValueError(
                "Cannot compute soft cosine with zero-magnitude vector"
            )
        return num / (da * db)

    _soft_cosine.__doc__ = "Soft Cosine similarity (fixed feature similarity matrix)."
    return _soft_cosine


# =====================================================================
# Metrics requiring scipy
# =====================================================================


def kl_divergence(a: list[float], b: list[float]) -> float:
    """Compute Kullback-Leibler divergence KL(a || b).

    Both vectors are treated as **discrete probability distributions**
    and must sum to approximately 1.0.  Entries must be non-negative.

    .. warning::

       KL divergence is **asymmetric**: ``KL(a || b) != KL(b || a)``.
       It is undefined when ``b_i = 0`` and ``a_i > 0``.

    Requires ``scipy``.
    """
    _validate_vector_pair(a, b)
    try:
        from scipy.special import rel_entr  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "KL divergence requires scipy. "
            "Install with: pip install scipy"
        ) from exc
    import numpy as np  # type: ignore[import-untyped]
    pa, pb = np.array(a, dtype=float), np.array(b, dtype=float)
    if np.any(pa < 0) or np.any(pb < 0):
        raise ValueError("KL divergence requires non-negative inputs")
    if not (0.99 <= pa.sum() <= 1.01):
        raise ValueError(
            f"Vector a must sum to ~1.0 (probability distribution), got {pa.sum():.4f}"
        )
    if not (0.99 <= pb.sum() <= 1.01):
        raise ValueError(
            f"Vector b must sum to ~1.0 (probability distribution), got {pb.sum():.4f}"
        )
    return float(np.sum(rel_entr(pa, pb)))


def wasserstein_distance(a: list[float], b: list[float]) -> float:
    """Compute 1-D Wasserstein distance (Earth Mover's Distance).

    Both vectors are treated as **discrete probability distributions**
    over the same support points ``{0, 1, ..., n-1}``.

    Requires ``scipy``.
    """
    _validate_vector_pair(a, b)
    try:
        from scipy.stats import wasserstein_distance as _wd  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Wasserstein distance requires scipy. "
            "Install with: pip install scipy"
        ) from exc
    import numpy as np  # type: ignore[import-untyped]
    support = np.arange(len(a))
    return float(_wd(support, support, a, b))


# =====================================================================
# Rank-correlation metrics (require numpy)
# =====================================================================


def spearman_rank_correlation(a: list[float], b: list[float]) -> float:
    """Compute Spearman's rank correlation coefficient.

    Returns a value in [-1, 1].  Measures monotonic relationship
    between two ranked variables.

    Requires ``numpy``.
    """
    _validate_vector_pair(a, b)
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Spearman correlation requires numpy. "
            "Install with: pip install numpy"
        ) from exc

    def _rankdata(x: list[float]) -> list[float]:
        """Simple ranking (average method for ties)."""
        arr = sorted(enumerate(x), key=lambda t: t[1])
        ranks = [0.0] * len(x)
        i = 0
        while i < len(arr):
            j = i
            while j < len(arr) - 1 and arr[j + 1][1] == arr[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[arr[k][0]] = avg_rank
            i = j + 1
        return ranks

    ra = np.array(_rankdata(a))
    rb = np.array(_rankdata(b))
    # Pearson correlation of ranks
    ra_m = ra - ra.mean()
    rb_m = rb - rb.mean()
    num = float(np.sum(ra_m * rb_m))
    denom = float(np.sqrt(np.sum(ra_m**2) * np.sum(rb_m**2)))
    if denom == 0:
        return 0.0
    return num / denom


def kendall_tau(a: list[float], b: list[float]) -> float:
    """Compute Kendall's Tau-b rank correlation coefficient.

    Returns a value in [-1, 1].  Measures ordinal association based
    on concordant and discordant pairs.

    Requires ``numpy``.
    """
    _validate_vector_pair(a, b)
    try:
        import numpy as np  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "Kendall's Tau requires numpy. "
            "Install with: pip install numpy"
        ) from exc
    n = len(a)
    concordant = 0
    discordant = 0
    tied_a = 0
    tied_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 and db == 0:
                continue
            elif da == 0:
                tied_a += 1
            elif db == 0:
                tied_b += 1
            elif (da > 0 and db > 0) or (da < 0 and db < 0):
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt(
        (concordant + discordant + tied_a)
        * (concordant + discordant + tied_b)
    )
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


# =====================================================================
# Companion MetricProperties for each example metric
# =====================================================================

CANBERRA_PROPERTIES = MetricProperties(
    name="canberra",
    kind="distance",
    range_min=0.0,
    range_max=None,  # upper bound = dimension
    bounded=False,
    metric_space=True,  # Lance & Williams (1967)
    symmetric=True,
    normalization_sensitive=True,
    zero_vector_behavior="convention",  # 0/0 terms contribute 0.0
    computational_complexity="O(n)",
    best_for=("composition data", "ecology", "values near zero"),
)

BRAY_CURTIS_PROPERTIES = MetricProperties(
    name="bray_curtis",
    kind="distance",
    range_min=0.0,
    range_max=1.0,  # |a-b|/(|a|+|b|) ≤ 1 by triangle inequality
    bounded=True,
    metric_space=False,  # fails triangle inequality
    symmetric=True,
    normalization_sensitive=False,  # ratio cancels uniform scaling
    zero_vector_behavior="convention",  # returns 0.0 for (0, 0)
    computational_complexity="O(n)",
    best_for=("ecology", "microbiome analysis", "species abundance"),
)

HAVERSINE_PROPERTIES = MetricProperties(
    name="haversine",
    kind="distance",
    range_min=0.0,
    range_max=math.pi * 6371.0088,  # half circumference ≈ 20015.087 km
    bounded=True,
    metric_space=True,  # great-circle distance is a proper metric
    symmetric=True,
    normalization_sensitive=True,
    zero_vector_behavior="accepts",
    computational_complexity="O(1)",  # fixed 2-element vectors
    best_for=("geographic coordinates", "geospatial queries"),
)

DTW_PROPERTIES = MetricProperties(
    name="dtw",
    kind="distance",
    range_min=0.0,
    range_max=None,
    bounded=False,
    metric_space=False,  # unconstrained DTW fails triangle inequality
    symmetric=True,
    normalization_sensitive=True,
    zero_vector_behavior="accepts",
    computational_complexity="O(n^2)",  # full DP matrix
    best_for=("time series", "sequence alignment", "temporal patterns"),
)

MAHALANOBIS_PROPERTIES = MetricProperties(
    name="mahalanobis",
    kind="distance",
    range_min=0.0,
    range_max=None,
    bounded=False,
    metric_space=True,  # with valid positive-definite inverse covariance
    symmetric=True,
    normalization_sensitive=True,
    zero_vector_behavior="accepts",
    computational_complexity="O(n^2)",  # matrix-vector multiply
    best_for=("correlated features", "outlier detection", "multivariate data"),
)

SOFT_COSINE_PROPERTIES = MetricProperties(
    name="soft_cosine",
    kind="similarity",
    range_min=-1.0,
    range_max=1.0,
    bounded=True,
    metric_space=False,
    symmetric=True,
    normalization_sensitive=False,  # normalises internally
    zero_vector_behavior="rejects",  # division by zero
    computational_complexity="O(n^2)",  # matrix-vector multiply
    best_for=("semantic text similarity", "feature-correlated embeddings"),
)

KL_DIVERGENCE_PROPERTIES = MetricProperties(
    name="kl_divergence",
    kind="divergence",
    range_min=0.0,
    range_max=None,  # can be +inf
    bounded=False,
    metric_space=False,  # asymmetric, fails triangle inequality
    symmetric=False,  # KL(P||Q) != KL(Q||P)
    normalization_sensitive=True,  # inputs must be distributions
    zero_vector_behavior="rejects",  # zero in Q with nonzero P is undefined
    computational_complexity="O(n)",
    best_for=("probability distributions", "information theory", "model comparison"),
)

WASSERSTEIN_PROPERTIES = MetricProperties(
    name="wasserstein",
    kind="distance",
    range_min=0.0,
    range_max=None,
    bounded=False,
    metric_space=True,  # Wasserstein-1 is a proper metric on distributions
    symmetric=True,
    normalization_sensitive=True,
    zero_vector_behavior="accepts",
    computational_complexity="O(n log n)",  # CDF sort + linear pass
    best_for=("probability distributions", "generative model evaluation"),
)

SPEARMAN_PROPERTIES = MetricProperties(
    name="spearman",
    kind="correlation",
    range_min=-1.0,
    range_max=1.0,
    bounded=True,
    metric_space=False,
    symmetric=True,
    normalization_sensitive=False,  # rank-based, only order matters
    zero_vector_behavior="convention",  # all-tied ranks → 0.0
    computational_complexity="O(n log n)",  # ranking step dominates
    best_for=("ranked data", "monotonic relationships", "ordinal scales"),
)

KENDALL_TAU_PROPERTIES = MetricProperties(
    name="kendall_tau",
    kind="correlation",
    range_min=-1.0,
    range_max=1.0,
    bounded=True,
    metric_space=False,
    symmetric=True,
    normalization_sensitive=False,  # ordinal, magnitude-invariant
    zero_vector_behavior="convention",  # all-tied → 0.0
    computational_complexity="O(n^2)",  # all-pairs comparison
    best_for=("ranked data", "ordinal association", "concordance analysis"),
)
