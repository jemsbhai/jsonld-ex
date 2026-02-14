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
from typing import Callable, List

from jsonld_ex.vector import cosine_similarity as _builtin_cosine

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

SimilarityFunction = Callable[[List[float], List[float]], float]

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
# Mutable working registry (shallow-copied from _BUILTIN_METRICS)
# ---------------------------------------------------------------------------

_registry: dict[str, SimilarityFunction] = dict(_BUILTIN_METRICS)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_similarity_metric(
    name: str,
    fn: SimilarityFunction,
    *,
    force: bool = False,
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

    Raises
    ------
    ValueError
        If *name* is empty/whitespace, or already registered without
        *force*.
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

    _registry[name] = fn


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


def reset_similarity_registry() -> None:
    """Restore the registry to its initial state (built-ins only).

    Primarily useful in tests to guarantee isolation.  Also restores any
    built-in that was force-overridden back to its original
    implementation.
    """
    _registry.clear()
    _registry.update(_BUILTIN_METRICS)


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
