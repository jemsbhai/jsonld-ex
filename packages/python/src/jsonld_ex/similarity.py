"""Similarity metric registry for vector operations.

Provides a registry of named similarity / distance functions that can be
used with jsonld-ex vector embeddings.  Ships with four built-in metrics
(cosine, euclidean, dot_product, manhattan) and allows users to register
custom metrics at runtime.

Built-in metrics are protected from accidental removal but can be
intentionally overridden via ``force=True``.

Design principles
-----------------
* **Backwards compatible** — ``cosine_similarity`` remains in ``vector.py``
  and is re-used here as the ``"cosine"`` built-in.
* **Modular** — this module owns *only* the registry and new metric
  implementations.  ``vector.py`` is not modified.
* **Fail-loud** — invalid names, non-callables, and unknown lookups raise
  immediately rather than returning sentinel values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional

from jsonld_ex.vector import cosine_similarity as _builtin_cosine

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SimilarityFunction = Callable[[List[float], List[float]], float]


# ---------------------------------------------------------------------------
# MetricProperties — structured metadata for each metric
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricProperties:
    """Structured metadata describing a similarity or distance metric.

    Every field represents a mathematically defensible property.
    Instances are immutable (frozen) to prevent accidental mutation.

    Parameters
    ----------
    name:
        Metric identifier (must match the registry key).
    kind:
        One of ``"distance"``, ``"similarity"``, ``"inner_product"``,
        ``"divergence"``, or ``"correlation"``.
    range_min:
        Lower bound of the metric's output, or ``None`` for :math:`-\\infty`.
    range_max:
        Upper bound of the metric's output, or ``None`` for :math:`+\\infty`
        (also used when the bound is dimension-dependent).
    bounded:
        ``True`` only when both bounds are finite constants independent
        of input dimensionality.
    metric_space:
        ``True`` if the function satisfies all four metric-space axioms
        (non-negativity, identity of indiscernibles, symmetry, triangle
        inequality).  Only applies to distance-type functions.
    symmetric:
        ``True`` if ``f(a, b) == f(b, a)`` for all valid inputs.
    normalization_sensitive:
        ``True`` if uniformly scaling both vectors changes the result.
        ``False`` for scale-invariant functions (e.g. cosine, Jaccard).
    zero_vector_behavior:
        One of ``"accepts"`` (well-defined on zero vectors),
        ``"rejects"`` (raises an error), or ``"convention"`` (returns a
        defined value by convention, e.g. Jaccard(\u2205, \u2205) = 1.0).
    computational_complexity:
        Asymptotic time complexity as a string, e.g. ``"O(n)"``,
        ``"O(n^2)"``, ``"O(n log n)"``.
    best_for:
        Tuple of short, factual descriptions of data types or use cases
        where this metric is most appropriate.
    """

    name: str
    kind: str
    range_min: Optional[float]
    range_max: Optional[float]
    bounded: bool
    metric_space: bool
    symmetric: bool
    normalization_sensitive: bool
    zero_vector_behavior: str
    computational_complexity: str
    best_for: tuple[str, ...] = ()

# ---------------------------------------------------------------------------
# Shared input validation
# ---------------------------------------------------------------------------


def _validate_vector_pair(a: list[float], b: list[float]) -> None:
    """Validate two vectors for use in a similarity / distance function.

    Checks performed (matching ``cosine_similarity`` in vector.py):
      - Both vectors are non-empty
      - Dimensions match
      - Every element is a finite, non-boolean number

    Raises
    ------
    ValueError
        On dimension mismatch, empty vectors, or non-finite elements.
    TypeError
        On non-numeric or boolean elements.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be empty")
    for i, (x, y) in enumerate(zip(a, b)):
        for label, v in (("a", x), ("b", y)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(
                    f"Vector {label}[{i}] must be a number, "
                    f"got: {type(v).__name__}"
                )
            if math.isnan(v) or math.isinf(v):
                raise ValueError(
                    f"Vector {label}[{i}] must be finite, got: {v}"
                )


# ---------------------------------------------------------------------------
# Built-in metric implementations
# ---------------------------------------------------------------------------


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Compute Euclidean (L2) distance between two vectors.

    Returns ``sqrt(sum((a_i - b_i)^2))``.  Always non-negative;
    returns 0.0 for identical vectors (including zero vectors).
    """
    _validate_vector_pair(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def dot_product(a: list[float], b: list[float]) -> float:
    """Compute the inner (dot) product of two vectors.

    Returns ``sum(a_i * b_i)``.  Can be negative.  For unit vectors
    the result equals cosine similarity.
    """
    _validate_vector_pair(a, b)
    return sum(x * y for x, y in zip(a, b))


def manhattan_distance(a: list[float], b: list[float]) -> float:
    """Compute Manhattan (L1 / city-block) distance between two vectors.

    Returns ``sum(|a_i - b_i|)``.  Always non-negative;
    returns 0.0 for identical vectors.  Always >= Euclidean distance.
    """
    _validate_vector_pair(a, b)
    return sum(abs(x - y) for x, y in zip(a, b))


def chebyshev_distance(a: list[float], b: list[float]) -> float:
    """Compute Chebyshev (L-infinity) distance between two vectors.

    Returns ``max(|a_i - b_i|)``.  Always non-negative;
    returns 0.0 for identical vectors.  Completes the Lp norm family
    alongside Manhattan (L1) and Euclidean (L2).
    """
    _validate_vector_pair(a, b)
    return max(abs(x - y) for x, y in zip(a, b))


def hamming_distance(a: list[float], b: list[float]) -> float:
    """Compute Hamming distance between two vectors.

    Returns the number of positions where ``a_i != b_i`` (using exact
    equality).  Most meaningful on binary or integer vectors, but works
    on any numeric vectors.  The result is always a non-negative integer
    (returned as ``float`` for signature consistency).
    """
    _validate_vector_pair(a, b)
    return float(sum(1 for x, y in zip(a, b) if x != y))


def jaccard_similarity(a: list[float], b: list[float]) -> float:
    """Compute binary-set Jaccard similarity (IoU) between two vectors.

    Treats each vector as a set of *active* indices: ``A = {i : a_i != 0}``.
    Returns ``|A \u2229 B| / |A \u222a B|``.

    Convention: when both vectors are all-zero (both sets are empty),
    returns 1.0 — identical empty sets.

    Most meaningful on binary or indicator vectors.  Continuous nonzero
    values are treated as set membership (any nonzero = present).
    """
    _validate_vector_pair(a, b)
    intersection = 0
    union = 0
    for x, y in zip(a, b):
        a_nz = x != 0
        b_nz = y != 0
        if a_nz or b_nz:
            union += 1
            if a_nz and b_nz:
                intersection += 1
    if union == 0:
        return 1.0  # J(∅, ∅) = 1.0 by convention
    return intersection / union


# ---------------------------------------------------------------------------
# Immutable reference table of built-in metrics
# ---------------------------------------------------------------------------

_BUILTIN_METRICS: dict[str, SimilarityFunction] = {
    "cosine": _builtin_cosine,
    "euclidean": euclidean_distance,
    "dot_product": dot_product,
    "manhattan": manhattan_distance,
    "chebyshev": chebyshev_distance,
    "hamming": hamming_distance,
    "jaccard": jaccard_similarity,
}

BUILTIN_METRIC_NAMES: frozenset[str] = frozenset(_BUILTIN_METRICS)

# ---------------------------------------------------------------------------
# Built-in metric properties (mathematical facts)
# ---------------------------------------------------------------------------

_BUILTIN_PROPERTIES: dict[str, MetricProperties] = {
    "cosine": MetricProperties(
        name="cosine",
        kind="similarity",
        range_min=-1.0,
        range_max=1.0,
        bounded=True,
        metric_space=False,   # fails triangle inequality
        symmetric=True,
        normalization_sensitive=False,  # divides by norms
        zero_vector_behavior="rejects",  # 0/0 undefined
        computational_complexity="O(n)",
        best_for=("text embeddings", "unit-normalized vectors", "direction comparison"),
    ),
    "euclidean": MetricProperties(
        name="euclidean",
        kind="distance",
        range_min=0.0,
        range_max=None,  # unbounded above
        bounded=False,
        metric_space=True,  # canonical Lp metric
        symmetric=True,
        normalization_sensitive=True,  # d(ca, cb) = |c| * d(a, b)
        zero_vector_behavior="accepts",
        computational_complexity="O(n)",
        best_for=("spatial data", "low-dimensional vectors", "absolute distance"),
    ),
    "dot_product": MetricProperties(
        name="dot_product",
        kind="inner_product",
        range_min=None,  # -inf
        range_max=None,  # +inf
        bounded=False,
        metric_space=False,  # can be negative, no triangle inequality
        symmetric=True,
        normalization_sensitive=True,  # dot(ca, b) = c * dot(a, b)
        zero_vector_behavior="accepts",
        computational_complexity="O(n)",
        best_for=("unnormalized embeddings", "relevance scoring", "inner product spaces"),
    ),
    "manhattan": MetricProperties(
        name="manhattan",
        kind="distance",
        range_min=0.0,
        range_max=None,
        bounded=False,
        metric_space=True,  # L1 metric
        symmetric=True,
        normalization_sensitive=True,
        zero_vector_behavior="accepts",
        computational_complexity="O(n)",
        best_for=("high-dimensional sparse data", "grid-based distances", "feature importance"),
    ),
    "chebyshev": MetricProperties(
        name="chebyshev",
        kind="distance",
        range_min=0.0,
        range_max=None,
        bounded=False,
        metric_space=True,  # L-inf metric (limit of Lp)
        symmetric=True,
        normalization_sensitive=True,
        zero_vector_behavior="accepts",
        computational_complexity="O(n)",
        best_for=("worst-case deviation", "tolerance checking", "game board distances"),
    ),
    "hamming": MetricProperties(
        name="hamming",
        kind="distance",
        range_min=0.0,
        range_max=None,  # upper bound = dimension (not constant)
        bounded=False,
        metric_space=True,  # proper metric on fixed-length sequences
        symmetric=True,
        normalization_sensitive=False,  # counts positions, not magnitudes
        zero_vector_behavior="accepts",
        computational_complexity="O(n)",
        best_for=("binary vectors", "categorical features", "error detection"),
    ),
    "jaccard": MetricProperties(
        name="jaccard",
        kind="similarity",
        range_min=0.0,
        range_max=1.0,
        bounded=True,
        metric_space=False,  # similarity, not distance (1-J is a metric)
        symmetric=True,
        normalization_sensitive=False,  # treats nonzero as set membership
        zero_vector_behavior="convention",  # J(∅, ∅) = 1.0
        computational_complexity="O(n)",
        best_for=("binary/indicator vectors", "set overlap", "presence/absence data"),
    ),
}

# ---------------------------------------------------------------------------
# Mutable working registries (shallow-copied from built-in tables)
# ---------------------------------------------------------------------------

_registry: dict[str, SimilarityFunction] = dict(_BUILTIN_METRICS)
_properties_registry: dict[str, MetricProperties] = dict(_BUILTIN_PROPERTIES)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_similarity_metric(
    name: str,
    fn: SimilarityFunction,
    *,
    force: bool = False,
    properties: MetricProperties | None = None,
) -> None:
    """Register a similarity metric under *name*.

    Parameters
    ----------
    name:
        Non-empty identifier for the metric.
    fn:
        A callable accepting two ``list[float]`` vectors and returning a
        ``float`` score.
    force:
        If *False* (default), raises ``ValueError`` when *name* is already
        registered.  Pass ``True`` to intentionally override an existing
        metric (including built-ins).
    properties:
        Optional :class:`MetricProperties` describing the metric.  If
        provided, ``properties.name`` must match *name*.  When ``None``,
        no properties are stored (the metric can still be used, but
        :func:`get_metric_properties` will return ``None`` for it).

    Raises
    ------
    ValueError
        If *name* is empty/whitespace, already registered without
        *force*, or *properties.name* does not match *name*.
    TypeError
        If *fn* is not callable.
    """
    # -- name validation --
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "Metric name must be a non-empty string, got: "
            f"{name!r}"
        )

    # -- callable validation --
    if not callable(fn):
        raise TypeError(
            f"Metric must be callable, got: {type(fn).__name__}"
        )

    # -- duplicate check --
    if not force and name in _registry:
        raise ValueError(
            f"Metric '{name}' is already registered. "
            "Pass force=True to override."
        )

    # -- properties name must match --
    if properties is not None and properties.name != name:
        raise ValueError(
            f"Properties name mismatch: properties.name={properties.name!r} "
            f"but registering as {name!r}. These must match."
        )

    _registry[name] = fn
    if properties is not None:
        _properties_registry[name] = properties
    elif name in _properties_registry and name not in _BUILTIN_PROPERTIES:
        # If re-registering a custom metric without properties, clear old ones
        del _properties_registry[name]


def get_similarity_metric(name: str) -> SimilarityFunction:
    """Return the metric registered under *name*.

    Raises
    ------
    KeyError
        If no metric is registered with that name.
    """
    try:
        return _registry[name]
    except KeyError:
        raise KeyError(
            f"No similarity metric registered as '{name}'. "
            f"Available: {sorted(_registry)}"
        ) from None


def list_similarity_metrics() -> list[str]:
    """Return a sorted list of all registered metric names.

    The returned list is a snapshot — mutating it does not affect the
    registry.
    """
    return sorted(_registry)


def get_metric_properties(name: str) -> MetricProperties | None:
    """Return the :class:`MetricProperties` for a registered metric.

    Returns ``None`` if the metric is not registered or was registered
    without properties.
    """
    return _properties_registry.get(name)


def get_all_metric_properties() -> dict[str, MetricProperties]:
    """Return a snapshot of all metrics that have properties.

    The returned dict maps metric names to their :class:`MetricProperties`.
    Custom metrics registered without properties are excluded.  Mutating
    the returned dict does not affect the registry.
    """
    return dict(_properties_registry)


def unregister_similarity_metric(name: str) -> None:
    """Remove a **custom** metric from the registry.

    Built-in metrics cannot be unregistered — use
    ``register_similarity_metric(name, fn, force=True)`` to override
    them instead.

    Raises
    ------
    KeyError
        If *name* is not currently registered.
    ValueError
        If *name* is a built-in metric.
    """
    if name in BUILTIN_METRIC_NAMES:
        raise ValueError(
            f"Cannot unregister built-in metric '{name}'. "
            "Use register_similarity_metric(..., force=True) to override."
        )
    if name not in _registry:
        raise KeyError(f"Metric '{name}' is not registered")
    del _registry[name]
    _properties_registry.pop(name, None)


def reset_similarity_registry() -> None:
    """Restore the registry to its initial state (built-ins only).

    Primarily useful in tests to guarantee isolation.  Also restores any
    built-in that was force-overridden back to its original
    implementation and properties.
    """
    _registry.clear()
    _registry.update(_BUILTIN_METRICS)
    _properties_registry.clear()
    _properties_registry.update(_BUILTIN_PROPERTIES)


def compare_metrics(
    a: list[float],
    b: list[float],
    *,
    metrics: list[str] | None = None,
    include_errors: bool = True,
) -> dict:
    """Compute multiple metrics on the same vector pair.

    Returns a structured comparison with each metric's score, kind, and
    any error that occurred.  Useful for exploring which metric is most
    appropriate for a given data type.

    Parameters
    ----------
    a, b:
        The two vectors to compare.
    metrics:
        List of metric names to compute.  ``None`` (default) computes
        **all** currently registered metrics.
    include_errors:
        If ``True`` (default), metrics that raise on this pair are
        included in the results with ``score=None`` and the error
        message.  If ``False``, failed metrics are omitted from
        ``results`` (but still counted in ``metrics_failed``).

    Returns
    -------
    dict
        ``{"results": {name: {"score", "kind", "error"}, ...},
        "vectors_dimension": int,
        "metrics_computed": int, "metrics_failed": int}``

    Raises
    ------
    ValueError
        If vectors are empty or have mismatched dimensions (checked
        once, before any metric is called).
    KeyError
        If *metrics* contains an unregistered name.
    """
    # Validate vectors once up-front so bad input raises immediately
    _validate_vector_pair(a, b)

    # Resolve which metrics to run
    if metrics is not None:
        names = list(metrics)
        # Validate all names exist before computing anything
        for name in names:
            get_similarity_metric(name)  # raises KeyError if missing
    else:
        names = sorted(_registry)

    results: dict[str, dict] = {}
    computed = 0
    failed = 0

    for name in names:
        fn = _registry[name]
        props = _properties_registry.get(name)
        kind = props.kind if props is not None else None
        try:
            score = fn(a, b)
            results[name] = {"score": float(score), "kind": kind, "error": None}
            computed += 1
        except Exception as exc:
            failed += 1
            if include_errors:
                results[name] = {"score": None, "kind": kind, "error": str(exc)}

    return {
        "results": results,
        "vectors_dimension": len(a),
        "metrics_computed": computed,
        "metrics_failed": failed,
    }


def similarity(
    a: list[float],
    b: list[float],
    *,
    metric: str | None = None,
    term_definition: dict | None = None,
) -> float:
    """Compute similarity (or distance) between two vectors.

    Metric resolution order:

    1. **metric** — if given, used directly (highest priority).
    2. **term_definition** — if given, reads the ``@similarity`` key.
    3. Falls back to ``"cosine"``.

    Parameters
    ----------
    a, b:
        The two vectors to compare.
    metric:
        Explicit metric name (must be registered).  Overrides any
        ``@similarity`` declared in *term_definition*.
    term_definition:
        A vector term definition dict (the *inner* dict, e.g.
        ``{"@id": ..., "@container": "@vector", "@similarity": ...}``).
        If present and *metric* is ``None``, the ``@similarity`` value
        is used as the metric name.

    Returns
    -------
    float
        The similarity score or distance value, depending on the metric.

    Raises
    ------
    KeyError
        If the resolved metric name is not registered.
    ValueError / TypeError
        Propagated from the underlying metric function on invalid input.
    """
    if metric is None:
        if term_definition is not None:
            metric = term_definition.get("@similarity", "cosine")
        else:
            metric = "cosine"
    fn = get_similarity_metric(metric)
    return fn(a, b)


# ---------------------------------------------------------------------------
# VectorProperties — deterministic data analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VectorProperties:
    """Statistical properties of a sample of vectors.

    Produced by :func:`analyze_vectors`.  Every field is a deterministic,
    mathematically well-defined property of the input data.

    Parameters
    ----------
    n_vectors:
        Number of vectors in the sample.
    dimensionality:
        Length of each vector.
    is_binary:
        ``True`` if every element across all vectors is 0.0 or 1.0.
    sparsity:
        Fraction of zero-valued elements across all vectors.
        0.0 = fully dense, 1.0 = all zeros.
    is_unit_normalized:
        ``True`` if every vector has L2 norm within 1e-6 of 1.0.
    all_non_negative:
        ``True`` if no element is negative.
    magnitude_cv:
        Coefficient of variation (std / mean) of the L2 norms.
        0.0 when all norms are identical or all norms are zero.
    """

    n_vectors: int
    dimensionality: int
    is_binary: bool
    sparsity: float
    is_unit_normalized: bool
    all_non_negative: bool
    magnitude_cv: float


def analyze_vectors(vectors: list[list[float]]) -> VectorProperties:
    """Compute deterministic statistical properties of a vector sample.

    Parameters
    ----------
    vectors:
        A list of at least 2 equal-length, non-empty numeric vectors.

    Returns
    -------
    VectorProperties
        Frozen dataclass with all detected properties.

    Raises
    ------
    ValueError
        If fewer than 2 vectors, empty vectors, or mismatched dimensions.
    """
    if len(vectors) < 2:
        raise ValueError(
            f"At least 2 vectors required for analysis, got {len(vectors)}"
        )
    dim = len(vectors[0])
    if dim == 0:
        raise ValueError("Vectors must not be empty")
    for i, v in enumerate(vectors):
        if len(v) != dim:
            raise ValueError(
                f"Dimension mismatch: vector 0 has {dim} elements, "
                f"vector {i} has {len(v)}"
            )

    n = len(vectors)
    total_elements = n * dim

    # Single pass over all elements
    zero_count = 0
    is_binary = True
    all_non_negative = True

    norms: list[float] = []
    for v in vectors:
        sq_sum = 0.0
        for val in v:
            if val == 0.0:
                zero_count += 1
            else:
                if val != 1.0:
                    is_binary = False
                if val < 0.0:
                    all_non_negative = False
            sq_sum += val * val
        norms.append(math.sqrt(sq_sum))

    sparsity = zero_count / total_elements if total_elements > 0 else 0.0

    # Unit normalization: all norms ≈ 1.0 (tolerance 1e-6)
    _UNIT_NORM_TOL = 1e-6
    is_unit_normalized = all(
        abs(norm - 1.0) <= _UNIT_NORM_TOL for norm in norms
    )

    # Magnitude coefficient of variation: std(norms) / mean(norms)
    mean_norm = sum(norms) / n
    if mean_norm == 0.0:
        magnitude_cv = 0.0
    else:
        variance = sum((nm - mean_norm) ** 2 for nm in norms) / n
        magnitude_cv = math.sqrt(variance) / mean_norm

    return VectorProperties(
        n_vectors=n,
        dimensionality=dim,
        is_binary=is_binary,
        sparsity=sparsity,
        is_unit_normalized=is_unit_normalized,
        all_non_negative=all_non_negative,
        magnitude_cv=magnitude_cv,
    )


# ---------------------------------------------------------------------------
# Recommendation engine — pluggable interface + built-in heuristic
# ---------------------------------------------------------------------------


class HeuristicRecommender:
    """Rule-based metric recommendation engine.

    Maps detected :class:`VectorProperties` to a ranked list of metric
    recommendations.  Each rule is grounded in a mathematical property
    of the metric and the data.

    Heuristic thresholds
    --------------------
    * **Sparsity > 0.5** — more than half of elements are zero.  Cosine
      and Jaccard are effective because they focus on the non-zero
      overlap and are not distorted by the large number of shared zeros
      (Aggarwal et al., 2001 *On the Surprising Behavior of Distance
      Metrics in High Dimensional Space*).
    * **Magnitude CV > 0.5** — the coefficient of variation of L2 norms
      exceeds 0.5, indicating substantial scale differences between
      vectors.  Magnitude-invariant metrics (cosine, Jaccard, Hamming)
      are preferred because distances like Euclidean would be dominated
      by magnitude rather than direction.
    * **Dimensionality > 100** — in high-dimensional spaces, Lp distance
      metrics concentrate (all pairwise distances converge), reducing
      discriminative power.  Cosine similarity, which operates on
      direction rather than absolute distance, is more robust to this
      concentration effect (Aggarwal et al., 2001).
    """

    # Thresholds (documented heuristics)
    SPARSITY_THRESHOLD = 0.5
    MAGNITUDE_CV_THRESHOLD = 0.5
    HIGH_DIM_THRESHOLD = 100

    def recommend(
        self,
        properties: VectorProperties,
        available_metrics: dict[str, MetricProperties],
    ) -> list[dict]:
        """Produce ranked metric recommendations from data properties.

        Parameters
        ----------
        properties:
            Output of :func:`analyze_vectors`.
        available_metrics:
            Dict of metric name → :class:`MetricProperties` for all
            metrics the user has registered (with properties).

        Returns
        -------
        list[dict]
            Each dict has ``metric``, ``rank``, ``rationale``.
            Empty list if *available_metrics* is empty.
        """
        if not available_metrics:
            return []

        # Accumulate (metric_name, score, rationale_parts)
        # Higher score = better recommendation.
        scores: dict[str, tuple[float, list[str]]] = {
            name: (0.0, []) for name in available_metrics
        }

        def _boost(name: str, amount: float, reason: str) -> None:
            if name in scores:
                current_score, reasons = scores[name]
                scores[name] = (current_score + amount, reasons + [reason])

        # --- Rule 1: Binary data ---
        if properties.is_binary:
            _boost("hamming", 10.0, "Binary data ({0,1} values): Hamming distance counts bit differences directly.")
            _boost("jaccard", 10.0, "Binary data ({0,1} values): Jaccard measures set overlap (IoU).")
            _boost("cosine", 3.0, "Binary data: cosine similarity is well-defined on binary vectors.")

        # --- Rule 2: High sparsity ---
        if properties.sparsity > self.SPARSITY_THRESHOLD and not properties.is_binary:
            _boost("cosine", 6.0,
                   f"High sparsity ({properties.sparsity:.0%}): cosine focuses on "
                   f"non-zero overlap, unaffected by shared zeros.")
            _boost("jaccard", 5.0,
                   f"High sparsity ({properties.sparsity:.0%}): Jaccard measures "
                   f"set overlap, ignoring shared-zero dimensions.")

        # --- Rule 3: Unit-normalized vectors ---
        if properties.is_unit_normalized:
            _boost("cosine", 8.0,
                   "Vectors are unit-normalized: cosine similarity is the "
                   "natural choice and equals dot product for unit vectors.")
            _boost("dot_product", 6.0,
                   "Vectors are unit-normalized: dot product equals cosine "
                   "similarity and avoids the normalization computation.")
            _boost("euclidean", 4.0,
                   "Vectors are unit-normalized: Euclidean distance is "
                   "monotonically related to cosine (d² = 2 - 2·cos).")

        # --- Rule 4: High magnitude variation ---
        if properties.magnitude_cv > self.MAGNITUDE_CV_THRESHOLD:
            _boost("cosine", 7.0,
                   f"High magnitude variation (CV={properties.magnitude_cv:.2f}): "
                   f"cosine is magnitude-invariant, comparing direction only.")
            # Penalise magnitude-sensitive distances
            for name, mp in available_metrics.items():
                if mp.normalization_sensitive and mp.kind == "distance":
                    _boost(name, -3.0,
                           f"High magnitude variation (CV={properties.magnitude_cv:.2f}): "
                           f"{name} is magnitude-sensitive and may be dominated by scale differences.")

        # --- Rule 5: High dimensionality ---
        if properties.dimensionality > self.HIGH_DIM_THRESHOLD:
            _boost("cosine", 4.0,
                   f"High dimensionality ({properties.dimensionality}D): cosine is "
                   f"robust to distance concentration in high-dimensional spaces.")
            _boost("manhattan", 2.0,
                   f"High dimensionality ({properties.dimensionality}D): Manhattan (L1) "
                   f"is more discriminative than Euclidean (L2) in high dimensions.")

        # --- Rule 6: General-purpose baseline (low boost) ---
        # Give cosine and euclidean small base scores so they always
        # appear if nothing else is strongly indicated.
        if not properties.is_binary:
            _boost("cosine", 1.0, "General-purpose: cosine is widely applicable for embeddings.")
            _boost("euclidean", 1.0, "General-purpose: Euclidean is the standard distance metric.")

        # Sort by score descending, then alphabetically for ties
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1][0], item[0]),
        )

        # Build output, filtering out metrics with score <= 0
        # (unless we'd return nothing — always return at least the best)
        results: list[dict] = []
        for rank_idx, (name, (score, reasons)) in enumerate(ranked):
            if score <= 0 and results:
                break
            # Combine rationale parts into a single string
            rationale = " ".join(reasons) if reasons else (
                f"{name} is available but no strong signal was detected "
                f"for this data profile."
            )
            results.append({
                "metric": name,
                "rank": rank_idx + 1,
                "rationale": rationale,
            })

        return results


def recommend_metric(
    vectors: list[list[float]],
    *,
    engine: object | None = None,
) -> dict:
    """Analyze vectors and recommend the best similarity metric(s).

    This is the main entry point for metric selection.  It performs
    two separable stages:

    1. **Analysis** — :func:`analyze_vectors` computes deterministic
       data properties.
    2. **Recommendation** — an engine maps properties to a ranked list
       of metrics with rationale.

    Parameters
    ----------
    vectors:
        A list of at least 2 equal-length numeric vectors.
    engine:
        A recommendation engine with a ``recommend(properties,
        available_metrics)`` method.  Defaults to
        :class:`HeuristicRecommender`.

    Returns
    -------
    dict
        ``{"data_properties": VectorProperties,
        "recommendations": [...], "inconclusive": bool}``
    """
    vp = analyze_vectors(vectors)
    available = get_all_metric_properties()

    if engine is None:
        engine = HeuristicRecommender()

    recs = engine.recommend(vp, available)  # type: ignore[union-attr]

    # Determine if the analysis was inconclusive: no strong recommendations
    inconclusive = len(recs) == 0 or (
        len(recs) >= 1
        and all(
            "no strong signal" in r.get("rationale", "").lower()
            for r in recs
        )
    )

    return {
        "data_properties": vp,
        "recommendations": recs,
        "inconclusive": inconclusive,
    }


# ---------------------------------------------------------------------------
# evaluate_metrics — empirical metric evaluation on labeled pairs
# ---------------------------------------------------------------------------

# Direction-mapping: kinds where higher score means more similar.
_HIGHER_IS_BETTER_KINDS: frozenset[str] = frozenset(
    {"similarity", "correlation", "inner_product"}
)


def _resolve_higher_is_better(
    name: str,
    override: dict[str, bool] | None,
) -> tuple[bool, bool, str | None]:
    """Determine whether higher metric values mean more similar.

    Returns ``(higher_is_better, direction_inferred, warning_or_none)``.

    Resolution order:
    1. Explicit override dict
    2. MetricProperties.kind (if registered with properties)
    3. Default ``True`` with a warning
    """
    if override and name in override:
        return override[name], True, None

    props = _properties_registry.get(name)
    if props is not None:
        hib = props.kind in _HIGHER_IS_BETTER_KINDS
        return hib, True, None

    # Unknown direction — default to True with warning
    return True, False, (
        f"No properties registered for metric '{name}'. "
        f"Assuming higher scores indicate greater similarity "
        f"(higher_is_better=True). Provide properties or use the "
        f"higher_is_better parameter to specify the correct direction."
    )


def _spearman_rank_correlation(
    x: list[float], y: list[float]
) -> float | None:
    """Compute Spearman's rank correlation between *x* and *y*.

    Returns ``None`` if either variable is constant (zero variance
    in ranks), which makes the correlation mathematically undefined.

    Uses average-rank for ties.
    """
    n = len(x)
    if n < 2:
        return None

    def _rank(vals: list[float]) -> list[float]:
        """Assign average ranks (1-based)."""
        indexed = sorted(enumerate(vals), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)

    # Pearson correlation of ranks
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var_x = sum((a - mx) ** 2 for a in rx)
    var_y = sum((b - my) ** 2 for b in ry)

    if var_x == 0.0 or var_y == 0.0:
        return None  # constant variable → undefined

    return cov / math.sqrt(var_x * var_y)


def _mann_whitney_auc(
    pos_scores: list[float], neg_scores: list[float]
) -> float | None:
    """Compute exact AUC via the Mann–Whitney U statistic.

    ``AUC = P(score_pos > score_neg) + 0.5 \u00b7 P(score_pos = score_neg)``

    Returns ``None`` if either class is empty (AUC is undefined).
    """
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return None

    concordant = 0
    tied = 0
    for p in pos_scores:
        for q in neg_scores:
            if p > q:
                concordant += 1
            elif p == q:
                tied += 1

    return (concordant + 0.5 * tied) / (n_pos * n_neg)


def evaluate_metrics(
    labeled_pairs: list[tuple[list[float], list[float], float]],
    *,
    metrics: list[str] | None = None,
    higher_is_better: dict[str, bool] | None = None,
) -> dict:
    """Evaluate registered metrics on labeled vector pairs.

    For each metric, computes three separation quality measures:

    * **mean_separation** — difference of mean scores between the
      similar and dissimilar groups, direction-corrected so that
      positive always indicates good separation.
    * **rank_correlation** — Spearman correlation between metric
      scores and labels, direction-corrected so that positive
      indicates agreement.
    * **auc** — exact ROC-AUC via the Mann–Whitney U statistic.
      1.0 = perfect separation, 0.5 = random, 0.0 = perfectly
      inverted.  Labels are binarised at 0.5 for AUC computation.

    All three measures are **direction-corrected**: higher values
    always mean better separation, regardless of whether the metric
    is a distance or a similarity.

    Parameters
    ----------
    labeled_pairs:
        Each element is ``(vector_a, vector_b, label)`` where *label*
        is a float in [0, 1].  For AUC and mean_separation, labels
        are binarised at 0.5 (>= 0.5 = similar).
    metrics:
        Metric names to evaluate.  ``None`` = all registered.
    higher_is_better:
        Per-metric override for score direction.  ``True`` means
        higher scores indicate greater similarity.

    Returns
    -------
    dict
        ``{"results": {...}, "ranking": [...],
        "pairs_evaluated": int, "metrics_evaluated": int,
        "metrics_failed": int}``
    """
    if not labeled_pairs:
        raise ValueError("At least 1 labeled pair is required")

    # Resolve metric names
    if metrics is not None:
        names = list(metrics)
        for name in names:
            get_similarity_metric(name)  # KeyError if missing
    else:
        names = sorted(_registry)

    n_pairs = len(labeled_pairs)
    labels = [lp[2] for lp in labeled_pairs]

    # Binary labels for AUC and mean_separation (threshold at 0.5)
    binary_labels = [1 if lab >= 0.5 else 0 for lab in labels]
    has_pos = any(b == 1 for b in binary_labels)
    has_neg = any(b == 0 for b in binary_labels)
    has_both_classes = has_pos and has_neg

    # Check if labels are constant (for rank correlation)
    labels_constant = all(lab == labels[0] for lab in labels)

    results: dict[str, dict] = {}
    evaluated = 0
    failed = 0

    for name in names:
        fn = _registry[name]
        hib, inferred, warning = _resolve_higher_is_better(
            name, higher_is_better
        )

        # Compute scores on all pairs
        scores: list[float] = []
        pair_error: str | None = None
        for vec_a, vec_b, _label in labeled_pairs:
            try:
                scores.append(float(fn(vec_a, vec_b)))
            except Exception as exc:
                pair_error = (
                    f"Metric '{name}' failed on a pair: {exc}"
                )
                break

        if pair_error is not None or len(scores) != n_pairs:
            failed += 1
            results[name] = {
                "mean_separation": None,
                "rank_correlation": None,
                "auc": None,
                "higher_is_better": hib,
                "direction_inferred": inferred,
                "warning": warning,
                "error": pair_error or "Incomplete score computation",
            }
            continue

        # --- Mean separation (direction-corrected) ---
        mean_sep: float | None = None
        if has_both_classes:
            sim_scores = [
                s for s, b in zip(scores, binary_labels) if b == 1
            ]
            dis_scores = [
                s for s, b in zip(scores, binary_labels) if b == 0
            ]
            mean_sim = sum(sim_scores) / len(sim_scores)
            mean_dis = sum(dis_scores) / len(dis_scores)
            # Direction-correct: positive = good separation
            if hib:
                mean_sep = mean_sim - mean_dis
            else:
                mean_sep = mean_dis - mean_sim

        # --- Rank correlation (direction-corrected) ---
        rank_corr: float | None = None
        if not labels_constant:
            raw_corr = _spearman_rank_correlation(scores, labels)
            if raw_corr is not None:
                rank_corr = raw_corr if hib else -raw_corr

        # --- AUC (direction-corrected via Mann–Whitney U) ---
        auc: float | None = None
        if has_both_classes:
            pos_scores = [
                s for s, b in zip(scores, binary_labels) if b == 1
            ]
            neg_scores = [
                s for s, b in zip(scores, binary_labels) if b == 0
            ]
            raw_auc = _mann_whitney_auc(pos_scores, neg_scores)
            if raw_auc is not None:
                auc = raw_auc if hib else (1.0 - raw_auc)

        evaluated += 1
        results[name] = {
            "mean_separation": mean_sep,
            "rank_correlation": rank_corr,
            "auc": auc,
            "higher_is_better": hib,
            "direction_inferred": inferred,
            "warning": warning,
            "error": None,
        }

    # --- Ranking: sort by AUC descending, failed metrics last ---
    def _sort_key(name: str) -> tuple[int, float, str]:
        entry = results[name]
        if entry["error"] is not None:
            return (1, 0.0, name)  # failed → last
        auc_val = entry["auc"]
        if auc_val is None:
            return (0, -0.5, name)  # undefined AUC → middle
        return (0, -auc_val, name)  # higher AUC → first

    ranking = sorted(results.keys(), key=_sort_key)

    return {
        "results": results,
        "ranking": ranking,
        "pairs_evaluated": n_pairs,
        "metrics_evaluated": evaluated,
        "metrics_failed": failed,
    }
