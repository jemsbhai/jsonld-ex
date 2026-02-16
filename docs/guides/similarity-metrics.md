# Similarity Metrics Guide

**Modules:** `jsonld_ex.similarity`, `jsonld_ex.similarity_examples`, `jsonld_ex.metric_advisory`

## Overview

jsonld-ex provides an extensible similarity metric system with three layers:

1. **Metric Registry** — 7 built-in metrics + extensible registration for custom metrics
2. **Example Metrics** — 10 additional metrics demonstrating extensibility
3. **Advisory System** — Analyze vectors, compare metrics, recommend the best metric, evaluate empirically

## Built-in Metrics

All built-in metrics are available via `similarity(a, b, metric="name")`:

| Metric | Range | Higher = | Use Case |
|--------|-------|----------|----------|
| `cosine` | [-1, 1] | More similar | Text/embedding similarity, direction-focused |
| `euclidean` | [0, ∞) | Less similar | Spatial distance, clustering |
| `dot_product` | (-∞, ∞) | More similar | Unit vectors (equals cosine), MIPS |
| `manhattan` | [0, ∞) | Less similar | High-dimensional spaces, L1 regularized |
| `chebyshev` | [0, ∞) | Less similar | Worst-case dimension difference |
| `hamming` | [0, n] | Less similar | Binary/integer vectors, error detection |
| `jaccard` | [0, 1] | More similar | Set membership, binary features |

**Norm ordering guarantee:** chebyshev ≤ euclidean ≤ manhattan (for all vectors).

```python
from jsonld_ex import similarity

a = [1.0, 0.0, 0.5]
b = [0.8, 0.2, 0.6]

similarity(a, b, metric="cosine")     # Direction similarity
similarity(a, b, metric="euclidean")  # L2 distance
similarity(a, b, metric="manhattan")  # L1 distance
```

## The `@similarity` Keyword

Specify the intended metric in vector term definitions for self-describing data:

```python
from jsonld_ex import vector_term_definition

term_def = vector_term_definition(
    "embedding",
    "http://example.org/embedding",
    dimensions=768,
    similarity="cosine",  # Documents intended comparison method
)
```

## Custom Metrics

Register domain-specific metrics at runtime:

```python
from jsonld_ex import register_similarity_metric, similarity

def weighted_cosine(a, b):
    """Weight dimensions by importance."""
    weights = [1.0, 2.0, 0.5]  # domain-specific weights
    wa = [x * w for x, w in zip(a, weights)]
    wb = [x * w for x, w in zip(b, weights)]
    dot = sum(x * y for x, y in zip(wa, wb))
    norm_a = sum(x**2 for x in wa) ** 0.5
    norm_b = sum(x**2 for x in wb) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

register_similarity_metric("weighted_cosine", weighted_cosine)
similarity(a, b, metric="weighted_cosine")  # Uses your function
```

Built-in metrics are protected from accidental overwrite (use `force=True` to override).

## Example Metrics

10 additional metrics in `jsonld_ex.similarity_examples` demonstrate the registry's extensibility:

| Metric | Domain |
|--------|--------|
| Canberra distance | Ecology, environmental science |
| Bray-Curtis | Community composition |
| Haversine | Geospatial coordinates |
| DTW (Dynamic Time Warping) | Time series |
| Mahalanobis | Correlated features |
| Soft Cosine | NLP with word similarity matrices |
| KL Divergence | Probability distributions |
| Wasserstein (Earth Mover's) | Distribution comparison |
| Spearman rank correlation | Ranked data |
| Kendall tau | Ordinal agreement |

## Advisory System

The advisory system helps choose the right metric for your data. It consists of four tools:

### 1. Analyze Vectors — Detect Properties

```python
from jsonld_ex import analyze_vectors

props = analyze_vectors([0.1, 0.0, 0.0, 0.8, 0.0, 0.0, 0.1])
# VectorProperties:
#   is_binary=False, sparsity=0.571, is_unit_normalized=False,
#   is_non_negative=True, magnitude_cv=... 
```

Detects: binary vectors, sparsity ratio, unit normalization, non-negativity, magnitude variation.

### 2. Compare Metrics — Side-by-Side

```python
from jsonld_ex import compare_metrics

results = compare_metrics(
    [1.0, 0.0, 0.5],
    [0.8, 0.2, 0.6],
    metrics=["cosine", "euclidean", "manhattan"],
)
# {"cosine": 0.943, "euclidean": 0.316, "manhattan": 0.5}
```

### 3. Recommend Metric — Heuristic Rules

```python
from jsonld_ex import recommend_metric, analyze_vectors

props = analyze_vectors([1, 0, 0, 1, 0, 1])
rec = recommend_metric(props)
# Recommendation(metric="jaccard", reason="Binary vectors → Jaccard (Levandowsky & Winter, 1971)")
```

The `HeuristicRecommender` applies 6 rules with academic citations:
- Binary vectors → Jaccard
- Very sparse vectors → Cosine (robust to missing dimensions)
- Unit-normalized → Dot product (equals cosine, faster)
- High magnitude variation → Cosine (magnitude-invariant)
- Non-negative features → Cosine
- General case → Euclidean

### 4. Evaluate Metrics — Empirical Testing

```python
from jsonld_ex import evaluate_metrics

pairs = [([1, 0], [0.9, 0.1]), ([1, 0], [0, 1]), ([0.5, 0.5], [0.5, 0.5])]
labels = [1.0, 0.0, 1.0]  # 1 = similar, 0 = dissimilar

results = evaluate_metrics(pairs, labels, metrics=["cosine", "euclidean"])
# {
#   "cosine":    {"auc": 1.0, "spearman": 0.866, ...},
#   "euclidean": {"auc": 1.0, "spearman": -0.866, ...},  # negative = distance metric
# }
```

Uses exact Mann-Whitney AUC (no ties approximation), Spearman with average-rank ties, and direction-corrected measures (higher-is-better vs lower-is-better).

## Metric Properties

Every metric has machine-readable properties for programmatic selection:

```python
from jsonld_ex import get_metric_properties

props = get_metric_properties("cosine")
# MetricProperties:
#   kind="similarity", range=(-1.0, 1.0), boundedness="bounded",
#   metric_space=False, symmetry="symmetric",
#   normalization_sensitive=False, zero_vector_behavior="undefined",
#   computational_complexity="O(n)", best_for="direction comparison"
```
