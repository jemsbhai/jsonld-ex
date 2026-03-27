"""
Vector Search over JSON-LD Knowledge Graphs.

Brute-force vector similarity search over ``@vector`` container
fields in JSON-LD documents, with hybrid symbolic+vector queries
and SL-uncertainty-aware ranking.

The key insight: **the knowledge graph IS the vector index.**
No separate vector DB needed.  Zero indexing time (data-oblivious).

Three search modes:

1. **vector_search** — Pure cosine similarity over ``@vector`` fields.
   Scans all nodes in ``@graph``, extracts the named vector property,
   and returns top-k by similarity.

2. **hybrid_search** — Symbolic property filters (exact match on
   ``@type``, ``category``, etc.) applied *before* vector similarity.
   Reduces the search space without a separate metadata index.

3. **uncertainty_aware_search** — Uses ``quantization_to_opinion()``
   from the quantization-SL bridge to rank by *projected probability*
   rather than raw similarity.  Quantization distortion becomes
   epistemic uncertainty in the ranking, automatically downweighting
   results from aggressively quantized vectors.

Performance note:
    This is O(n) brute-force search, suitable for knowledge graphs
    up to ~100K nodes.  For larger collections, this module still
    serves as a correctness reference — the same JSON-LD documents
    can be loaded into a dedicated vector DB for approximate search,
    with ``@quantization`` metadata informing the index configuration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from jsonld_ex.quantization_bridge import quantization_to_opinion
from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# RESULT TYPE
# ═══════════════════════════════════════════════════════════════════


@dataclass
class SearchResult:
    """A single search result from vector search.

    Attributes:
        node: The full JSON-LD node dict.
        node_id: The ``@id`` of the node (or ``""`` if absent).
        score: Raw cosine similarity score in [-1, 1].
        opinion: SL opinion (only for uncertainty-aware search).
    """
    node: Dict[str, Any]
    node_id: str
    score: float
    opinion: Optional[Opinion] = None


# ═══════════════════════════════════════════════════════════════════
# VECTOR EXTRACTION
# ═══════════════════════════════════════════════════════════════════


def _extract_vector(node: Dict[str, Any], vector_property: str) -> Optional[List[float]]:
    """Extract a vector from a node's named property.

    Handles three storage patterns:
    1. Plain list: ``{"embedding": [0.1, 0.2, ...]}``
    2. @value wrapper: ``{"embedding": {"@value": [0.1, 0.2, ...], ...}}``
    3. @vector container: ``{"embedding": {"@container": "@vector", "@value": [...]}}``
    """
    val = node.get(vector_property)
    if val is None:
        return None

    # Pattern 1: plain list
    if isinstance(val, list):
        if len(val) > 0 and isinstance(val[0], (int, float)):
            return val
        return None

    # Pattern 2 & 3: dict with @value
    if isinstance(val, dict):
        inner = val.get("@value")
        if isinstance(inner, list) and len(inner) > 0 and isinstance(inner[0], (int, float)):
            return inner
        return None

    return None


def _extract_quantization(node: Dict[str, Any], vector_property: str) -> tuple:
    """Extract quantization metadata (method, bit_width) from a node.

    Returns (method, bit_width) or ("scalar", 32) as default.
    """
    val = node.get(vector_property)
    if not isinstance(val, dict):
        return ("scalar", 32)

    q = val.get("@quantization")
    if not isinstance(q, dict):
        return ("scalar", 32)

    method = q.get("method", "scalar")
    bit_width = q.get("bitWidth", 32)
    if not isinstance(bit_width, int) or bit_width < 1:
        bit_width = 32

    return (method, bit_width)


# ═══════════════════════════════════════════════════════════════════
# COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


# ═══════════════════════════════════════════════════════════════════
# GRAPH NODE EXTRACTION
# ═══════════════════════════════════════════════════════════════════


def _get_nodes(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract nodes from a JSON-LD document.

    Handles both ``@graph`` arrays and single-node documents.
    """
    graph = doc.get("@graph")
    if isinstance(graph, list):
        return [n for n in graph if isinstance(n, dict)]

    # Single-node document (the doc itself is the node)
    if "@type" in doc or "@id" in doc:
        return [doc]

    return []


def _matches_filters(node: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    """Check if a node matches all filter conditions (AND logic)."""
    for key, expected in filters.items():
        actual = node.get(key)
        if actual != expected:
            return False
    return True


# ═══════════════════════════════════════════════════════════════════
# VECTOR SEARCH
# ═══════════════════════════════════════════════════════════════════


def vector_search(
    doc: Dict[str, Any],
    query: List[float],
    vector_property: str,
    *,
    k: int = 10,
) -> List[SearchResult]:
    """Brute-force cosine similarity search over @vector fields.

    Scans all nodes in the document, extracts the named vector
    property, computes cosine similarity with the query, and
    returns the top-k results sorted by descending similarity.

    Args:
        doc: JSON-LD document (with ``@graph`` or single node).
        query: Query vector (same dimensionality as stored vectors).
        vector_property: Name of the property containing vectors.
        k: Maximum number of results to return.

    Returns:
        List of SearchResult, sorted by descending similarity.
    """
    nodes = _get_nodes(doc)
    results: List[SearchResult] = []

    for node in nodes:
        vec = _extract_vector(node, vector_property)
        if vec is None:
            continue

        sim = _cosine_sim(query, vec)
        results.append(SearchResult(
            node=node,
            node_id=node.get("@id", ""),
            score=sim,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


# ═══════════════════════════════════════════════════════════════════
# HYBRID SEARCH
# ═══════════════════════════════════════════════════════════════════


def hybrid_search(
    doc: Dict[str, Any],
    query: List[float],
    vector_property: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 10,
) -> List[SearchResult]:
    """Symbolic property filters + vector similarity search.

    Applies exact-match filters on node properties *before* computing
    vector similarity, reducing the search space.

    Args:
        doc: JSON-LD document.
        query: Query vector.
        vector_property: Vector property name.
        filters: Dict of property_name → expected_value for exact
            match filtering.  All conditions must match (AND logic).
        k: Maximum results.

    Returns:
        Filtered results sorted by descending similarity.
    """
    if filters is None:
        filters = {}

    nodes = _get_nodes(doc)
    results: List[SearchResult] = []

    for node in nodes:
        # Apply symbolic filters first
        if filters and not _matches_filters(node, filters):
            continue

        vec = _extract_vector(node, vector_property)
        if vec is None:
            continue

        sim = _cosine_sim(query, vec)
        results.append(SearchResult(
            node=node,
            node_id=node.get("@id", ""),
            score=sim,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]


# ═══════════════════════════════════════════════════════════════════
# UNCERTAINTY-AWARE SEARCH
# ═══════════════════════════════════════════════════════════════════


def uncertainty_aware_search(
    doc: Dict[str, Any],
    query: List[float],
    vector_property: str,
    *,
    filters: Optional[Dict[str, Any]] = None,
    k: int = 10,
    similarity_range: tuple = (-1.0, 1.0),
    base_rate: float = 0.5,
) -> List[SearchResult]:
    """SL-uncertainty-aware vector search.

    Uses ``quantization_to_opinion()`` to create an SL opinion for
    each similarity score, incorporating quantization distortion as
    epistemic uncertainty.  Results are ranked by *projected
    probability* rather than raw similarity.

    This means:
    - A high similarity from a heavily quantized vector (high
      uncertainty) ranks *lower* than the same similarity from
      a lightly quantized vector (low uncertainty).
    - The ranking automatically adapts to mixed-precision graphs
      where some nodes have full-precision vectors and others
      have quantized ones.

    Args:
        doc: JSON-LD document.
        query: Query vector.
        vector_property: Vector property name.
        filters: Optional symbolic filters (AND logic).
        k: Maximum results.
        similarity_range: Range of the similarity metric.
        base_rate: SL base rate for opinions.

    Returns:
        Results sorted by descending projected probability, each
        with an ``opinion`` field.
    """
    if filters is None:
        filters = {}

    nodes = _get_nodes(doc)
    results: List[SearchResult] = []

    for node in nodes:
        if filters and not _matches_filters(node, filters):
            continue

        vec = _extract_vector(node, vector_property)
        if vec is None:
            continue

        sim = _cosine_sim(query, vec)
        method, bit_width = _extract_quantization(node, vector_property)

        opinion = quantization_to_opinion(
            similarity=sim,
            bit_width=bit_width,
            method=method,
            similarity_range=similarity_range,
            base_rate=base_rate,
        )

        results.append(SearchResult(
            node=node,
            node_id=node.get("@id", ""),
            score=sim,
            opinion=opinion,
        ))

    # Sort by projected probability (not raw similarity)
    results.sort(key=lambda r: r.opinion.projected_probability(), reverse=True)
    return results[:k]
