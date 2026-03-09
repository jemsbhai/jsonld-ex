"""
Inference algorithms for SLNetwork.

Implements forward inference over a DAG of SL opinion nodes using
the existing ``deduce()`` and ``cumulative_fuse()`` operators from
``confidence_algebra``.

Algorithm 1 — Exact tree inference (Step 4):
    When every node has at most one parent, inference is a single
    forward pass in topological order.  At each non-root node:
        ω_child = deduce(ω_parent, ω_{child|parent}, ω_{child|¬parent})
    This is exact — no approximation.

Algorithm 2 — Approximate DAG inference (Step 5):
    When a node has multiple parents connected by individual SLEdges,
    we deduce per-parent and then fuse the results:
        For each parent Xᵢ:
            ω_{Y,via_i} = deduce(ω_{Xᵢ}, ω_{Y|Xᵢ}, cf_i)
        ω_Y = cumulative_fuse(ω_{Y,via_1}, ..., ω_{Y,via_k})

    **Approximation:** cumulative_fuse assumes the per-parent deduction
    results are independent evidence about Y.  When parents share common
    ancestors, this assumption is violated and the result is overconfident
    (lower uncertainty than warranted).  See DAG_ERROR_ANALYSIS.md.

Algorithm 3 — Full enumeration (Step 5):
    When a node has a MultiParentEdge with a full 2^k conditional table,
    we enumerate all parent-state configurations:
        ω_Y = Σ w(config) · ω_{Y|config}
    where w(config) is computed from projected parent probabilities
    assuming independence.

    **Approximation:** The independence assumption in computing
    configuration weights has the same error characteristics as
    Algorithm 2.  Additionally, parent opinions are projected to
    scalars, losing the (b, d, u) decomposition.

References:
    Jøsang, A. (2016). Subjective Logic, §12.6 (deduction operator).
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Literal, Optional

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse, deduce
from jsonld_ex.multinomial_algebra import (
    MultinomialOpinion,
    multinomial_deduce,
)
from jsonld_ex.sl_network.counterfactuals import (
    CounterfactualFn,
    get_counterfactual_fn,
    vacuous_counterfactual,
)
from jsonld_ex.sl_network.types import (
    InferenceResult,
    InferenceStep,
    MultinomialEdge,
    MultiParentEdge,
    SLEdge,
)


# Forward reference to avoid circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jsonld_ex.sl_network.network import SLNetwork


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

_ENUMERATE_WARN_THRESHOLD = 10
"""Warn when enumerating configurations for nodes with more parents."""

_ENUMERATE_MAX_PARENTS = 20
"""Refuse enumeration for nodes with more than this many parents."""


# ═══════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════


def infer_node(
    network: SLNetwork,
    node_id: str,
    counterfactual_fn: str | CounterfactualFn = "vacuous",
    method: Literal["auto", "exact", "approximate", "enumerate"] = "auto",
) -> InferenceResult:
    """Run inference to compute the opinion at a given node.

    Propagates opinions from root nodes through the DAG using
    ``deduce()`` at each edge, collecting a full inference trace.

    Args:
        network:           The SLNetwork to infer over.
        node_id:           The node to query.
        counterfactual_fn: Strategy for computing counterfactuals
                           when ``SLEdge.counterfactual`` is None.
                           Either a string name (``"vacuous"``,
                           ``"adversarial"``, ``"prior"``) or a
                           callable ``(Opinion) -> Opinion``.
        method:            Inference algorithm selection:
                           - ``"auto"``: exact for trees, approximate
                             for general DAGs.
                           - ``"exact"``: require tree structure.
                           - ``"approximate"``: deduce-per-parent then
                             fuse.  Works for any DAG.
                           - ``"enumerate"``: full 2^k enumeration
                             for nodes with MultiParentEdge.  Falls
                             back to approximate for nodes without
                             a conditional table.

    Returns:
        InferenceResult with the final opinion, trace, and
        all intermediate opinions.

    Raises:
        NodeNotFoundError: If ``node_id`` is not in the network.
        ValueError: If ``method="exact"`` but the graph is not a tree,
                    or ``method`` is unrecognized.
    """
    from jsonld_ex.sl_network.network import NodeNotFoundError

    if not network.has_node(node_id):
        raise NodeNotFoundError(node_id)

    cf_fn = get_counterfactual_fn(counterfactual_fn)

    # Determine effective method
    effective_method = method
    if method == "auto":
        effective_method = "exact" if network.is_tree() else "approximate"

    if effective_method == "exact":
        if not network.is_tree():
            raise ValueError(
                "method='exact' requires a tree-structured network "
                "(every node has at most one parent), but this network "
                "has multi-parent nodes."
            )
        return _infer_tree(network, node_id, cf_fn)

    if effective_method == "approximate":
        return _infer_dag_approximate(network, node_id, cf_fn)

    if effective_method == "enumerate":
        return _infer_dag_enumerate(network, node_id, cf_fn)

    raise ValueError(
        f"Unknown inference method {method!r}. "
        f"Available: 'auto', 'exact', 'approximate', 'enumerate'."
    )


def infer_all(
    network: SLNetwork,
    counterfactual_fn: str | CounterfactualFn = "vacuous",
    method: Literal["auto", "exact", "approximate", "enumerate"] = "auto",
) -> dict[str, InferenceResult]:
    """Run inference for every node in the network.

    Performs a single forward pass and wraps each node's result.

    Args:
        network:           The SLNetwork to infer over.
        counterfactual_fn: Counterfactual strategy (see ``infer_node``).
        method:            Inference algorithm (see ``infer_node``).

    Returns:
        Dict mapping node_id → InferenceResult for every node.
    """
    if network.node_count() == 0:
        return {}

    cf_fn = get_counterfactual_fn(counterfactual_fn)

    # Pick any leaf to trigger full graph processing
    leaves = network.get_leaves()
    if not leaves:
        leaves = sorted(network.get_roots())

    first_leaf = leaves[0]
    full_result = infer_node(network, first_leaf, cf_fn, method)

    # Build per-node results from the intermediate opinions
    results: dict[str, InferenceResult] = {}
    for nid in full_result.topological_order:
        opinion = full_result.intermediate_opinions.get(nid)
        if opinion is None:
            opinion = network.get_node(nid).opinion
        node_steps = [s for s in full_result.steps if s.node_id == nid]
        results[nid] = InferenceResult(
            query_node=nid,
            opinion=opinion,
            steps=node_steps,
            intermediate_opinions=full_result.intermediate_opinions,
            topological_order=full_result.topological_order,
            multinomial_intermediate_opinions=(
                full_result.multinomial_intermediate_opinions
            ),
        )

    return results


# ═══════════════════════════════════════════════════════════════════
# EXACT TREE INFERENCE (Algorithm 1)
# ═══════════════════════════════════════════════════════════════════


def _infer_tree(
    network: SLNetwork,
    query_node: str,
    cf_fn: CounterfactualFn,
) -> InferenceResult:
    """Exact inference over a tree-structured SL network.

    Single forward pass in topological order.  At each non-root node,
    computes:
        ω_child = deduce(ω_parent, conditional, counterfactual)

    Multinomial extension: when a parent has a multinomial opinion and
    the edge is a MultinomialEdge, dispatches to multinomial_deduce().

    Complexity: O(n) where n = number of nodes.
    """
    topo_order = network.topological_sort()
    intermediate: dict[str, Opinion] = {}
    multinomial_intermediate: dict[str, MultinomialOpinion] = {}
    steps: list[InferenceStep] = []

    for nid in topo_order:
        node = network.get_node(nid)
        parents = network.get_parents(nid)

        if len(parents) == 0:
            # Root node
            intermediate[nid] = node.opinion
            if node.is_multinomial:
                multinomial_intermediate[nid] = node.multinomial_opinion
            steps.append(InferenceStep(
                node_id=nid,
                operation="passthrough",
                inputs={},
                result=node.opinion,
            ))

        elif len(parents) == 1:
            parent_id = parents[0]
            if network.has_multinomial_edge(parent_id, nid):
                # Multinomial path
                _deduce_single_parent_multinomial(
                    network, nid, parent_id,
                    intermediate, multinomial_intermediate, steps,
                )
            else:
                # Standard binary path
                _deduce_single_parent(
                    network, nid, parent_id, intermediate, steps, cf_fn
                )

        else:
            raise ValueError(
                f"Node {nid!r} has {len(parents)} parents — "
                f"exact tree inference requires at most 1 parent per node."
            )

    query_opinion = intermediate.get(query_node)
    if query_opinion is None:
        query_opinion = network.get_node(query_node).opinion

    return InferenceResult(
        query_node=query_node,
        opinion=query_opinion,
        steps=steps,
        intermediate_opinions=intermediate,
        topological_order=topo_order,
        multinomial_intermediate_opinions=multinomial_intermediate,
    )


# ═══════════════════════════════════════════════════════════════════
# APPROXIMATE DAG INFERENCE (Algorithm 2)
# ═══════════════════════════════════════════════════════════════════


def _infer_dag_approximate(
    network: SLNetwork,
    query_node: str,
    cf_fn: CounterfactualFn,
) -> InferenceResult:
    """Approximate inference over a general DAG.

    For each node in topological order:
      - Root (0 parents): passthrough.
      - Single parent: exact deduce (same as tree inference).
      - Multiple parents with individual SLEdges: deduce per-parent,
        then cumulative_fuse the results.

    **Approximation:** The fuse step assumes the per-parent deduction
    results are independent evidence about the child proposition.
    When parents share common ancestors, this assumption is violated.
    See DAG_ERROR_ANALYSIS.md for full error characterization.

    Complexity: O(n · k) where k = max parent count.
    """
    topo_order = network.topological_sort()
    intermediate: dict[str, Opinion] = {}
    multinomial_intermediate: dict[str, MultinomialOpinion] = {}
    steps: list[InferenceStep] = []

    for nid in topo_order:
        node = network.get_node(nid)
        parents = network.get_parents(nid)

        if len(parents) == 0:
            # Root node
            intermediate[nid] = node.opinion
            if node.is_multinomial:
                multinomial_intermediate[nid] = node.multinomial_opinion
            steps.append(InferenceStep(
                node_id=nid,
                operation="passthrough",
                inputs={},
                result=node.opinion,
            ))

        elif len(parents) == 1:
            parent_id = parents[0]
            if network.has_multinomial_edge(parent_id, nid):
                # Multinomial path
                _deduce_single_parent_multinomial(
                    network, nid, parent_id,
                    intermediate, multinomial_intermediate, steps,
                )
            else:
                # Standard binary path
                _deduce_single_parent(
                    network, nid, parent_id, intermediate, steps, cf_fn
                )

        else:
            # Multiple parents: deduce per-parent, then fuse
            _deduce_multi_parent_approximate(
                network, nid, parents, intermediate, steps, cf_fn
            )

    query_opinion = intermediate.get(query_node)
    if query_opinion is None:
        query_opinion = network.get_node(query_node).opinion

    return InferenceResult(
        query_node=query_node,
        opinion=query_opinion,
        steps=steps,
        intermediate_opinions=intermediate,
        topological_order=topo_order,
        multinomial_intermediate_opinions=multinomial_intermediate,
    )


def _deduce_single_parent(
    network: SLNetwork,
    nid: str,
    parent_id: str,
    intermediate: dict[str, Opinion],
    steps: list[InferenceStep],
    cf_fn: CounterfactualFn,
) -> None:
    """Deduce a node's opinion from its single parent.  Exact."""
    parent_opinion = intermediate[parent_id]
    edge = network.get_edge(parent_id, nid)

    counterfactual = (
        edge.counterfactual
        if edge.counterfactual is not None
        else cf_fn(edge.conditional)
    )

    deduced = deduce(parent_opinion, edge.conditional, counterfactual)
    intermediate[nid] = deduced
    steps.append(InferenceStep(
        node_id=nid,
        operation="deduce",
        inputs={
            "parent": parent_opinion,
            "conditional": edge.conditional,
            "counterfactual": counterfactual,
        },
        result=deduced,
    ))


def _deduce_single_parent_multinomial(
    network: SLNetwork,
    nid: str,
    parent_id: str,
    intermediate: dict[str, Opinion],
    multinomial_intermediate: dict[str, MultinomialOpinion],
    steps: list[InferenceStep],
) -> None:
    """Multinomial deduction from a single parent via MultinomialEdge.

    When the parent has a multinomial opinion and the edge is a
    MultinomialEdge, applies ``multinomial_deduce()`` to produce
    a MultinomialOpinion for the child node.

    The child's binomial ``opinion`` field is stored in ``intermediate``
    for backward compatibility.  The multinomial result is stored in
    ``multinomial_intermediate``.
    """
    edge = network.get_multinomial_edge(parent_id, nid)
    node = network.get_node(nid)

    if parent_id in multinomial_intermediate:
        # Parent has a multinomial opinion — use multinomial_deduce
        parent_multi = multinomial_intermediate[parent_id]
        deduced_multi = multinomial_deduce(parent_multi, edge.conditionals)
        multinomial_intermediate[nid] = deduced_multi
    # else: parent has no multinomial opinion (shouldn't normally
    # happen in a well-constructed network, but we handle it
    # gracefully by skipping multinomial computation)

    # For binomial intermediate: use the node's prior opinion
    intermediate[nid] = node.opinion
    steps.append(InferenceStep(
        node_id=nid,
        operation="multinomial_deduce",
        inputs={},
        result=node.opinion,
    ))


def _deduce_multi_parent_approximate(
    network: SLNetwork,
    nid: str,
    parents: list[str],
    intermediate: dict[str, Opinion],
    steps: list[InferenceStep],
    cf_fn: CounterfactualFn,
) -> None:
    """Deduce per-parent, then fuse.

    For each parent Xᵢ:
        ω_{Y,via_i} = deduce(ω_{Xᵢ}, ω_{Y|Xᵢ}, cf_i)
    ω_Y = cumulative_fuse(ω_{Y,via_1}, ..., ω_{Y,via_k})

    Approximation: assumes per-parent deductions are independent.
    """
    per_parent_deductions: list[Opinion] = []
    per_parent_inputs: dict[str, Opinion] = {}

    for parent_id in parents:
        parent_opinion = intermediate[parent_id]

        # Try to get individual SLEdge for this parent
        if network.has_edge(parent_id, nid):
            edge = network.get_edge(parent_id, nid)
            counterfactual = (
                edge.counterfactual
                if edge.counterfactual is not None
                else cf_fn(edge.conditional)
            )
            deduced_via_parent = deduce(
                parent_opinion, edge.conditional, counterfactual
            )
        else:
            # Parent connected via MultiParentEdge — no individual
            # conditional available.  Fall back to passthrough from
            # parent (the enumerate method handles this case properly).
            deduced_via_parent = parent_opinion

        per_parent_deductions.append(deduced_via_parent)
        per_parent_inputs[f"via_{parent_id}"] = deduced_via_parent

    # Fuse all per-parent deduction results
    fused = cumulative_fuse(*per_parent_deductions)
    intermediate[nid] = fused

    steps.append(InferenceStep(
        node_id=nid,
        operation="fuse_parents",
        inputs=per_parent_inputs,
        result=fused,
    ))


# ═══════════════════════════════════════════════════════════════════
# FULL ENUMERATION (Algorithm 3)
# ═══════════════════════════════════════════════════════════════════


def _infer_dag_enumerate(
    network: SLNetwork,
    query_node: str,
    cf_fn: CounterfactualFn,
) -> InferenceResult:
    """Inference with full enumeration for MultiParentEdge nodes.

    For nodes with a MultiParentEdge (full 2^k conditional table),
    enumerates all parent-state configurations weighted by projected
    parent probabilities.

    For nodes without a MultiParentEdge, falls back to the
    approximate deduce-per-parent-then-fuse approach.

    **Approximation:** Configuration weights assume parent states are
    independent.  See DAG_ERROR_ANALYSIS.md for error characterization.

    Complexity: O(n · 2^k) where k = max parent count.
    """
    topo_order = network.topological_sort()
    intermediate: dict[str, Opinion] = {}
    multinomial_intermediate: dict[str, MultinomialOpinion] = {}
    steps: list[InferenceStep] = []

    for nid in topo_order:
        node = network.get_node(nid)
        parents = network.get_parents(nid)

        if len(parents) == 0:
            intermediate[nid] = node.opinion
            if node.is_multinomial:
                multinomial_intermediate[nid] = node.multinomial_opinion
            steps.append(InferenceStep(
                node_id=nid,
                operation="passthrough",
                inputs={},
                result=node.opinion,
            ))

        elif len(parents) == 1:
            parent_id = parents[0]
            if network.has_multinomial_edge(parent_id, nid):
                _deduce_single_parent_multinomial(
                    network, nid, parent_id,
                    intermediate, multinomial_intermediate, steps,
                )
            else:
                _deduce_single_parent(
                    network, nid, parent_id, intermediate, steps, cf_fn
                )

        else:
            # Check if a MultiParentEdge exists for this node
            try:
                mpe = network.get_multi_parent_edge(nid)
                _deduce_enumerate(
                    network, nid, mpe, intermediate, steps
                )
            except ValueError:
                # No MultiParentEdge — fall back to approximate
                _deduce_multi_parent_approximate(
                    network, nid, parents, intermediate, steps, cf_fn
                )

    query_opinion = intermediate.get(query_node)
    if query_opinion is None:
        query_opinion = network.get_node(query_node).opinion

    return InferenceResult(
        query_node=query_node,
        opinion=query_opinion,
        steps=steps,
        intermediate_opinions=intermediate,
        topological_order=topo_order,
        multinomial_intermediate_opinions=multinomial_intermediate,
    )


def _deduce_enumerate(
    network: SLNetwork,
    nid: str,
    mpe: MultiParentEdge,
    intermediate: dict[str, Opinion],
    steps: list[InferenceStep],
) -> None:
    """Full enumeration over a MultiParentEdge conditional table.

    For each of the 2^k parent-state configurations:
        1. Compute configuration weight from projected parent probabilities
           (assuming independence):
              w(s₁,...,sₖ) = ∏ᵢ [sᵢ · pᵢ + (1-sᵢ) · (1-pᵢ)]
        2. Look up the conditional opinion for this configuration
        3. Accumulate: c_Y += w · c_{Y|config}  for c ∈ {b, d, u}

    Base rate:
        a_Y = Σ w(config) · P(ω_{Y|config})
    """
    k = len(mpe.parent_ids)

    # Safety check
    if k > _ENUMERATE_MAX_PARENTS:
        raise ValueError(
            f"Enumeration for node {nid!r} requires 2^{k} = {2**k} "
            f"configurations — exceeds maximum of 2^{_ENUMERATE_MAX_PARENTS}. "
            f"Use method='approximate' instead."
        )
    if k > _ENUMERATE_WARN_THRESHOLD:
        warnings.warn(
            f"Enumerating 2^{k} = {2**k} parent configurations for "
            f"node {nid!r}. This may be slow.",
            RuntimeWarning,
            stacklevel=4,
        )

    # Project each parent's inferred opinion to a probability
    parent_probs: list[float] = []
    parent_input_opinions: dict[str, Opinion] = {}
    for pid in mpe.parent_ids:
        parent_op = intermediate[pid]
        parent_probs.append(parent_op.projected_probability())
        parent_input_opinions[pid] = parent_op

    # Enumerate all 2^k configurations
    b_y = 0.0
    d_y = 0.0
    u_y = 0.0
    a_y = 0.0

    for config, cond_opinion in mpe.conditionals.items():
        # Compute configuration weight (product of marginals)
        weight = 1.0
        for i, state in enumerate(config):
            p_i = parent_probs[i]
            if state:
                weight *= p_i
            else:
                weight *= (1.0 - p_i)

        b_y += weight * cond_opinion.belief
        d_y += weight * cond_opinion.disbelief
        u_y += weight * cond_opinion.uncertainty
        a_y += weight * cond_opinion.projected_probability()

    # Construct the result opinion
    # The weighted sum should satisfy b+d+u=1 because:
    #   Σ weights = 1 (they form a probability distribution)
    #   and each cond_opinion has b+d+u=1
    result = Opinion(
        belief=b_y,
        disbelief=d_y,
        uncertainty=u_y,
        base_rate=max(0.0, min(1.0, a_y)),  # Clamp for numerical safety
    )

    intermediate[nid] = result
    steps.append(InferenceStep(
        node_id=nid,
        operation="enumerate",
        inputs=parent_input_opinions,
        result=result,
    ))
