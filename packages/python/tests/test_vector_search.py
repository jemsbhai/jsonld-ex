"""Tests for vector search module.

Verifies brute-force vector search, hybrid symbolic+vector queries,
and SL-uncertainty-aware ranking over JSON-LD knowledge graphs with
@vector container fields.

The key insight: the knowledge graph IS the vector index. No separate
vector DB needed. Zero indexing time (data-oblivious).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import pytest

from jsonld_ex.vector_search import (
    # Core search
    vector_search,
    # Hybrid search
    hybrid_search,
    # Uncertainty-aware search
    uncertainty_aware_search,
    # Result type
    SearchResult,
)
from jsonld_ex.vector import quantization_descriptor


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _unit_vec(dims: int, seed: int) -> List[float]:
    rng = random.Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(dims)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _make_product(idx: int, category: str, price: float, dims: int = 8) -> Dict[str, Any]:
    """Build a product node with @vector embedding."""
    return {
        "@id": f"urn:product:{idx}",
        "@type": "Product",
        "name": f"Product {idx}",
        "category": category,
        "price": price,
        "embedding": {
            "@container": "@vector",
            "@dimensions": dims,
            "@similarity": "cosine",
            "@value": _unit_vec(dims, seed=idx),
        },
    }


def _make_quantized_product(idx: int, category: str, price: float,
                             dims: int = 8, bit_width: int = 4) -> Dict[str, Any]:
    """Build a product node with @vector + @quantization."""
    return {
        "@id": f"urn:product:{idx}",
        "@type": "Product",
        "name": f"Product {idx}",
        "category": category,
        "price": price,
        "embedding": {
            "@container": "@vector",
            "@dimensions": dims,
            "@similarity": "cosine",
            "@quantization": quantization_descriptor(
                method="turboquant", bit_width=bit_width,
                rotation_seed=42, has_residual_qjl=True,
            ),
            "@value": _unit_vec(dims, seed=idx),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def product_graph():
    """A @graph with 10 products across 3 categories."""
    categories = ["electronics", "clothing", "food"]
    prices = [29.99, 49.99, 9.99, 199.99, 15.99,
              79.99, 5.99, 399.99, 24.99, 12.99]
    return {
        "@context": "https://schema.org/",
        "@graph": [
            _make_product(i, categories[i % 3], prices[i])
            for i in range(10)
        ],
    }


@pytest.fixture
def quantized_graph():
    """A @graph with quantized vector embeddings."""
    categories = ["electronics", "clothing", "food"]
    prices = [29.99, 49.99, 9.99, 199.99, 15.99,
              79.99, 5.99, 399.99, 24.99, 12.99]
    return {
        "@context": "https://schema.org/",
        "@graph": [
            _make_quantized_product(i, categories[i % 3], prices[i])
            for i in range(10)
        ],
    }


@pytest.fixture
def query_vec():
    """A query vector (same dims as product embeddings)."""
    return _unit_vec(8, seed=0)  # Same as product 0


# ═══════════════════════════════════════════════════════════════════
# SearchResult
# ═══════════════════════════════════════════════════════════════════


class TestSearchResult:
    """SearchResult dataclass."""

    def test_has_required_fields(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=1)
        r = results[0]
        assert hasattr(r, "node")
        assert hasattr(r, "score")
        assert hasattr(r, "node_id")

    def test_score_is_float(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=1)
        assert isinstance(results[0].score, float)

    def test_node_is_dict(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=1)
        assert isinstance(results[0].node, dict)


# ═══════════════════════════════════════════════════════════════════
# Brute-force vector search
# ═══════════════════════════════════════════════════════════════════


class TestVectorSearch:
    """Brute-force cosine similarity search over @vector fields."""

    def test_returns_k_results(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=5)
        assert len(results) == 5

    def test_results_sorted_descending(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_result_is_exact_match(self, product_graph, query_vec):
        """Query vec == product 0's embedding → top result is product 0."""
        results = vector_search(product_graph, query_vec, "embedding", k=1)
        assert results[0].node_id == "urn:product:0"
        assert abs(results[0].score - 1.0) < 1e-6  # cosine = 1.0

    def test_k_larger_than_graph(self, product_graph, query_vec):
        """k > graph size → return all nodes."""
        results = vector_search(product_graph, query_vec, "embedding", k=100)
        assert len(results) == 10

    def test_scores_in_valid_range(self, product_graph, query_vec):
        """Cosine similarity ∈ [-1, 1]."""
        results = vector_search(product_graph, query_vec, "embedding", k=10)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-9

    def test_node_id_extracted(self, product_graph, query_vec):
        results = vector_search(product_graph, query_vec, "embedding", k=3)
        for r in results:
            assert r.node_id.startswith("urn:product:")

    def test_annotated_value_extracted(self, product_graph, query_vec):
        """Vectors inside @value wrappers are correctly extracted."""
        results = vector_search(product_graph, query_vec, "embedding", k=1)
        # The node should contain the full embedding dict
        assert "embedding" in results[0].node

    def test_empty_graph(self, query_vec):
        graph = {"@graph": []}
        results = vector_search(graph, query_vec, "embedding", k=5)
        assert len(results) == 0

    def test_nodes_without_vector_skipped(self, query_vec):
        """Nodes missing the vector property are skipped."""
        graph = {
            "@graph": [
                {"@id": "urn:a", "@type": "Thing", "name": "no vector"},
                {
                    "@id": "urn:b", "@type": "Product",
                    "embedding": {
                        "@container": "@vector",
                        "@value": _unit_vec(8, seed=99),
                    },
                },
            ],
        }
        results = vector_search(graph, _unit_vec(8, seed=99), "embedding", k=5)
        assert len(results) == 1
        assert results[0].node_id == "urn:b"

    def test_flat_vector_list(self, query_vec):
        """Support vectors stored as plain lists (no @value wrapper)."""
        vec = _unit_vec(8, seed=42)
        graph = {
            "@graph": [
                {"@id": "urn:x", "embedding": vec},
            ],
        }
        results = vector_search(graph, query_vec, "embedding", k=1)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════
# Hybrid search (symbolic + vector)
# ═══════════════════════════════════════════════════════════════════


class TestHybridSearch:
    """Symbolic property filters + vector similarity."""

    def test_filter_by_type(self, product_graph, query_vec):
        """Filter by @type before vector search."""
        results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"@type": "Product"}, k=5,
        )
        assert len(results) == 5
        for r in results:
            assert r.node["@type"] == "Product"

    def test_filter_by_category(self, product_graph, query_vec):
        """Filter by category property."""
        results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"category": "electronics"}, k=10,
        )
        # Products 0, 3, 6, 9 are electronics (i % 3 == 0)
        assert len(results) == 4
        for r in results:
            assert r.node["category"] == "electronics"

    def test_filter_reduces_results(self, product_graph, query_vec):
        """Filtering should reduce result count."""
        all_results = vector_search(product_graph, query_vec, "embedding", k=10)
        filtered = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"category": "food"}, k=10,
        )
        assert len(filtered) < len(all_results)

    def test_filter_no_match(self, product_graph, query_vec):
        """Filter that matches nothing → empty results."""
        results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"category": "nonexistent"}, k=10,
        )
        assert len(results) == 0

    def test_results_still_sorted(self, product_graph, query_vec):
        """Filtered results are still sorted by similarity."""
        results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"category": "clothing"}, k=10,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_multiple_filters(self, product_graph, query_vec):
        """Multiple filter conditions (AND logic)."""
        results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={"category": "electronics", "@type": "Product"}, k=10,
        )
        for r in results:
            assert r.node["category"] == "electronics"
            assert r.node["@type"] == "Product"

    def test_empty_filters_same_as_vector_search(self, product_graph, query_vec):
        """Empty filters dict → same as plain vector search."""
        hybrid_results = hybrid_search(
            product_graph, query_vec, "embedding",
            filters={}, k=5,
        )
        vector_results = vector_search(
            product_graph, query_vec, "embedding", k=5,
        )
        assert len(hybrid_results) == len(vector_results)
        for h, v in zip(hybrid_results, vector_results):
            assert h.node_id == v.node_id


# ═══════════════════════════════════════════════════════════════════
# Uncertainty-aware search
# ═══════════════════════════════════════════════════════════════════


class TestUncertaintyAwareSearch:
    """SL-uncertainty-aware ranking using quantization_to_opinion()."""

    def test_returns_results(self, quantized_graph, query_vec):
        results = uncertainty_aware_search(
            quantized_graph, query_vec, "embedding", k=5,
        )
        assert len(results) == 5

    def test_results_have_opinion(self, quantized_graph, query_vec):
        """Each result should have an SL opinion."""
        results = uncertainty_aware_search(
            quantized_graph, query_vec, "embedding", k=1,
        )
        r = results[0]
        assert hasattr(r, "opinion")
        assert r.opinion is not None

    def test_opinion_has_uncertainty(self, quantized_graph, query_vec):
        """Opinion uncertainty reflects quantization distortion."""
        results = uncertainty_aware_search(
            quantized_graph, query_vec, "embedding", k=1,
        )
        op = results[0].opinion
        # 4-bit turboquant → small but nonzero uncertainty
        assert op.uncertainty > 0.0
        assert op.uncertainty < 0.5  # shouldn't be huge at 4 bits

    def test_results_sorted_by_projected_probability(self, quantized_graph, query_vec):
        """Results ranked by projected probability, not raw similarity."""
        results = uncertainty_aware_search(
            quantized_graph, query_vec, "embedding", k=10,
        )
        probs = [r.opinion.projected_probability() for r in results]
        assert probs == sorted(probs, reverse=True)

    def test_higher_bits_lower_uncertainty(self, query_vec):
        """More bits → less quantization uncertainty."""
        dims = 8
        graph_4bit = {
            "@graph": [_make_quantized_product(0, "x", 1.0, dims, bit_width=4)],
        }
        graph_8bit = {
            "@graph": [_make_quantized_product(0, "x", 1.0, dims, bit_width=8)],
        }
        r4 = uncertainty_aware_search(graph_4bit, query_vec, "embedding", k=1)
        r8 = uncertainty_aware_search(graph_8bit, query_vec, "embedding", k=1)
        assert r8[0].opinion.uncertainty < r4[0].opinion.uncertainty

    def test_no_quantization_metadata_defaults(self, product_graph, query_vec):
        """Nodes without @quantization → default uncertainty (scalar, 32-bit)."""
        results = uncertainty_aware_search(
            product_graph, query_vec, "embedding", k=1,
        )
        # No @quantization → uses scalar/32-bit → very low uncertainty
        assert results[0].opinion is not None
        assert results[0].opinion.uncertainty < 0.01  # near-zero distortion

    def test_score_and_opinion_both_present(self, quantized_graph, query_vec):
        """Results have both raw score and SL opinion."""
        results = uncertainty_aware_search(
            quantized_graph, query_vec, "embedding", k=1,
        )
        r = results[0]
        assert isinstance(r.score, float)
        assert r.opinion is not None
        assert abs(r.score) <= 1.0 + 1e-9
