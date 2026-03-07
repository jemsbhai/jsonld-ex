# Formal Error Analysis: Approximate DAG Inference in SLNetwork

**Author:** Muntaser Syed (with Claude)  
**Date:** March 2026  
**Status:** Pre-implementation analysis — must be completed before coding

---

## 1. Problem Statement

In a tree-structured SL network, inference is exact: each node has at most
one parent, so `deduce(ω_parent, ω_{child|parent}, ω_{child|¬parent})` is
applied once per edge in topological order.

In a general DAG, a node may have multiple parents. The question is:
**how do we compute the deduced opinion at a multi-parent node?**

Two approaches exist, each with distinct error characteristics.

---

## 2. Approach A: Fuse-Parents Approximation

### Algorithm (Deduce-per-parent, then fuse)

For a node Y with parents X₁, X₂, ..., Xₖ connected by individual SLEdges:

1. Compute each parent's inferred opinion: ω_{X₁}, ω_{X₂}, ..., ω_{Xₖ}
2. For each parent Xᵢ, deduce Y's opinion via that parent alone:
   ω_{Y,via_i} = deduce(ω_{Xᵢ}, ω_{Y|Xᵢ=T}, ω_{Y|Xᵢ=F})
3. Fuse all per-parent deduction results:
   ω_Y = cumulative_fuse(ω_{Y,via_1}, ω_{Y,via_2}, ..., ω_{Y,via_k})

### Why Deduce-Then-Fuse, Not Fuse-Then-Deduce

An alternative would be to fuse parent opinions first and then deduce
once. This requires choosing a single conditional for the compound
antecedent — but each edge Xᵢ → Y has its own distinct conditional
ω_{Y|Xᵢ}. There is no principled way to derive ω_{Y|compound} from
individual conditionals without additional assumptions.

Deduce-then-fuse avoids this problem: each edge's conditional is used
exactly once, and the results are fused as independent evidence about
the same proposition Y. The independence assumption is explicit and
well-characterized (see Error Characterization below).

### Error Characterization

**Source of error:** `cumulative_fuse()` assumes the parent opinions are
*independent evidence about the same proposition*. But in a DAG, parents
may share ancestors (the d-separation problem from Bayesian networks).
When parents X₁ and X₂ share a common ancestor Z, the evidence from Z
is double-counted in the fused opinion.

**Formal statement:** Let X₁, X₂ have a common ancestor Z. Then:

    cumulative_fuse(ω_{X₁}, ω_{X₂}) ≠ ω_{X₁ ∧ X₂}  (in general)

because cumulative fusion's evidence accumulation model treats each input
as contributing independent observations, but X₁ and X₂ carry correlated
evidence derived from the same ancestral source Z.

**Direction of bias:** Evidence double-counting causes the fused opinion
to be **overconfident** (lower uncertainty than warranted). The belief
and disbelief components are inflated relative to the true joint opinion.

**Magnitude:** The error is proportional to:
- The *dependency strength* between parents (more shared ancestors → more error)
- The *informativeness* of shared ancestors (dogmatic ancestors cause more
  double-counting than vacuous ones)
- The *depth* of shared ancestry (deeper sharing → more accumulated error)

**When the error is zero:**
- Parents share NO common ancestors (fully independent subtrees)
- All shared ancestors are vacuous (ω = (0,0,1,a)) — vacuous opinions
  contribute no evidence to fuse, so double-counting has no effect
- The graph is a tree (trivially — only one parent per node)

### Complexity

O(n · k) where n = node count, k = max parent count.

---

## 3. Approach B: Full Enumeration

### Algorithm

For a node Y with parents X₁, X₂, ..., Xₖ and a MultiParentEdge
providing a full conditional table with 2ᵏ entries:

1. Compute each parent's inferred opinion: ω_{X₁}, ..., ω_{Xₖ}
2. Project each parent opinion to a probability: p_{Xᵢ} = P(ω_{Xᵢ})
3. For each of the 2ᵏ parent-state configurations (s₁, s₂, ..., sₖ):
   a. Compute the configuration probability (assuming independence):
      w(s₁,...,sₖ) = ∏ᵢ [sᵢ · p_{Xᵢ} + (1-sᵢ) · (1-p_{Xᵢ})]
   b. Look up the conditional opinion for this configuration:
      ω_{Y|(s₁,...,sₖ)}
4. Compute a weighted combination:
   For each component c ∈ {b, d, u}:
      c_Y = Σ_{configs} w(config) · c_{Y|config}
5. Compute base rate:
      a_Y = Σ_{configs} w(config) · P(ω_{Y|config})

### Error Characterization

**Source of error:** Step 3a assumes parent states are independent when
computing configuration probabilities via the product formula. When parents
share common ancestors, their states are correlated, and the product formula
is incorrect.

**This is the same fundamental problem as Approach A**, manifesting in a
different place. Both approaches break the correlation structure by treating
dependent variables as independent.

**Additional information loss:** Step 2 projects each parent opinion to a
scalar probability, discarding the (b, d, u) decomposition. This means
the enumeration cannot distinguish between a parent whose P=0.7 comes from
high belief vs. moderate belief with base-rate-distributed uncertainty.

**Direction of bias:** Same as Approach A — overconfidence from
independence assumption. Additionally, the projection in Step 2 introduces
a systematic loss of epistemic information.

**When the error is zero:**
- Parents are truly independent (no shared ancestors)
- All parent opinions are dogmatic (u=0), so projection is lossless

### Complexity

O(n · 2ᵏ) — exponential in parent count. We warn when k > 10
(2¹⁰ = 1024 configurations) and refuse when k > 20.

---

## 4. Approach A vs. B: When to Use Each

| Criterion | Approach A (fuse) | Approach B (enumerate) |
|-----------|-------------------|------------------------|
| Requires MultiParentEdge table | No | Yes |
| Handles single-parent edges | Yes | N/A (uses SLEdge) |
| Computational complexity | O(n·k) | O(n·2ᵏ) |
| Information preserved | Full opinion tuple | Projected probability only |
| Error source | Independence in fusion | Independence in enumeration |
| Better when parents independent | Equivalent | Equivalent |
| Better when parents dependent | Neither — both biased | Neither — both biased |
| Better with high uncertainty parents | Approach A (vacuity mitigates) | Approach B (more configs) |

**Recommendation:** Default to Approach A for nodes with single-parent
SLEdges. Use Approach B only when a MultiParentEdge with a full
conditional table is available.

---

## 5. What We Document Honestly

For the implementation and any publication, we state:

1. **DAG inference in SLNetwork is approximate.** Exact inference in SL
   over general DAGs is an open problem (Jøsang 2016 does not provide
   a closed-form solution for multi-parent deduction).

2. **The approximation assumes independence of parent evidence.** This is
   the same assumption made by naive Bayes classifiers and many practical
   probabilistic systems, but it IS an assumption with known failure modes.

3. **The error direction is overconfidence** — the fused opinion has lower
   uncertainty than a hypothetical exact solution would produce.

4. **The error vanishes for tree-structured subgraphs** and for nodes
   whose parents have fully independent ancestry.

5. **We characterize the error empirically** in tests by comparing
   approximate results against exact enumeration on small graphs where
   both are computable.

---

## 6. Implementation Plan

Given this analysis:

1. Implement Approach A (fuse-parents) for nodes with single-parent SLEdges
   connecting multiple parents through the adjacency structure.

2. Implement Approach B (full enumeration) for nodes with a MultiParentEdge.

3. Add a `method` parameter:
   - `"approximate"` → Approach A (default for DAGs)
   - `"enumerate"` → Approach B (requires MultiParentEdge)
   - `"auto"` → exact for trees, approximate for DAGs

4. Warn when enumeration has k > 10 parents.

5. Tests verify:
   - DAG inference matches exact tree inference when the DAG is actually a tree
   - Diamond graph (shared ancestor) produces a valid opinion (b+d+u=1)
   - Diamond result differs from independent-tree result (confirming the
     approximation has an effect)
   - Full enumeration matches Approach A when conditional table entries are
     consistent with the individual SLEdge conditionals
   - Overconfidence direction: DAG approximation produces lower uncertainty
     than would be obtained by treating parents as independent subtrees
     (when parents DO share ancestors)
