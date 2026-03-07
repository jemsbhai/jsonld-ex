"""
Trust propagation algorithms for SLNetwork (Tier 2).

Implements transitive trust computation per Jøsang (2016, §14.3–§14.5).

Single-path transitive trust (§14.3):
    Q trusts A (ω_QA), A trusts B (ω_AB).
    Q's derived trust in B: trust_discount(ω_QA, ω_AB).
    For longer chains: left-fold trust_discount().

Multi-path trust fusion (§14.5):
    When Q can reach B through multiple independent paths, compute
    transitive trust along each path, then fuse the derived trust
    opinions via cumulative_fuse (or averaging_fuse for correlated paths).

This module composes ``trust_discount`` and ``cumulative_fuse`` from
``confidence_algebra.py`` — it never reimplements their logic.

References:
    Jøsang, A. (2016). Subjective Logic, §14.3 (transitive trust),
    §14.5 (multi-path trust fusion).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Literal

from jsonld_ex.confidence_algebra import (
    Opinion,
    averaging_fuse,
    cumulative_fuse,
    trust_discount,
)
from jsonld_ex.sl_network.types import (
    InferenceResult,
    InferenceStep,
    SLEdge,
    SLNode,
    TrustPropagationResult,
)

if TYPE_CHECKING:
    from jsonld_ex.sl_network.network import SLNetwork


def propagate_trust(
    network: SLNetwork,
    querying_agent: str,
    fusion_method: Literal["cumulative", "averaging"] = "cumulative",
) -> TrustPropagationResult:
    """Compute transitive trust from a querying agent to all reachable agents.

    Enumerates all simple (cycle-free) paths through the trust subgraph
    from ``querying_agent``, computing per-path transitive trust via
    left-fold of ``trust_discount()``.  When multiple paths reach the
    same agent, their derived trust opinions are fused.

    Args:
        network:         The SLNetwork containing agent nodes and trust edges.
        querying_agent:  The agent whose perspective is being computed.
        fusion_method:   ``"cumulative"`` (default, for independent paths)
                         or ``"averaging"`` (for correlated paths).

    Returns:
        A ``TrustPropagationResult`` with derived trust opinions,
        paths, and an inference trace.

    Raises:
        NodeNotFoundError: If ``querying_agent`` is not in the network.

    References:
        Jøsang, A. (2016). Subjective Logic, §14.3, §14.5.
    """
    from jsonld_ex.sl_network.network import NodeNotFoundError

    # ── Validate querying agent exists ──
    if not network.has_node(querying_agent):
        raise NodeNotFoundError(querying_agent)

    # ── Phase 1: Enumerate all simple paths, collect per-path opinions ──
    #
    # For each reachable agent, collect all per-path derived trust
    # opinions.  A "simple path" visits each agent at most once
    # (cycle prevention).
    #
    # per_path_opinions[agent_id] = [(opinion, path), ...]
    per_path_opinions: dict[str, list[tuple[Opinion, list[str]]]] = {}
    steps: list[InferenceStep] = []

    # Queue entries: (current_agent, derived_trust_to_current, path_from_Q)
    # path_from_Q includes querying_agent as first element and current_agent as last
    queue: deque[tuple[str, Opinion, list[str]]] = deque()

    # Seed: querying agent's direct trust edges
    for te in network.get_trust_edges_from(querying_agent):
        target = te.target_id

        # Direct (single-hop) trust — the edge opinion itself
        path = [querying_agent, target]
        per_path_opinions.setdefault(target, []).append(
            (te.trust_opinion, path)
        )

        steps.append(InferenceStep(
            node_id=target,
            operation="direct_trust",
            inputs={"trust_edge": te.trust_opinion},
            result=te.trust_opinion,
        ))

        queue.append((target, te.trust_opinion, path))

    # BFS: propagate through transitive trust edges
    while queue:
        current_agent, current_derived_trust, current_path = queue.popleft()

        for te in network.get_trust_edges_from(current_agent):
            target = te.target_id

            # Cycle prevention: skip if target is already on this path
            if target in current_path:
                continue

            # Transitive trust along this specific path
            transitive_trust = trust_discount(
                current_derived_trust, te.trust_opinion
            )

            path = current_path + [target]
            per_path_opinions.setdefault(target, []).append(
                (transitive_trust, path)
            )

            steps.append(InferenceStep(
                node_id=target,
                operation="trust_discount",
                inputs={
                    "derived_trust": current_derived_trust,
                    "edge_trust": te.trust_opinion,
                },
                result=transitive_trust,
            ))

            queue.append((target, transitive_trust, path))

    # ── Phase 2: Fuse multi-path opinions ──
    fuse_fn = cumulative_fuse if fusion_method == "cumulative" else averaging_fuse

    derived_trusts: dict[str, Opinion] = {}
    trust_paths: dict[str, list[str]] = {}

    for agent_id, path_opinions in per_path_opinions.items():
        opinions = [op for op, _ in path_opinions]
        paths = [p for _, p in path_opinions]

        if len(opinions) == 1:
            # Single path — no fusion needed
            derived_trusts[agent_id] = opinions[0]
            trust_paths[agent_id] = paths[0]
        else:
            # Multi-path — fuse all per-path opinions
            fused = fuse_fn(*opinions)
            derived_trusts[agent_id] = fused
            # Store the shortest path as representative
            trust_paths[agent_id] = min(paths, key=len)

            steps.append(InferenceStep(
                node_id=agent_id,
                operation=f"{fusion_method}_fuse",
                inputs={
                    f"path_{i}": op for i, op in enumerate(opinions)
                },
                result=fused,
            ))

    return TrustPropagationResult(
        querying_agent=querying_agent,
        derived_trusts=derived_trusts,
        trust_paths=trust_paths,
        steps=steps,
    )


def infer_with_trust(
    network: SLNetwork,
    query_node: str,
    querying_agent: str,
    trust_fusion: str = "cumulative",
    content_fusion: str = "cumulative",
    counterfactual_fn: str = "vacuous",
) -> InferenceResult:
    """Combined trust + content inference.

    Full pipeline:
        1. Propagate transitive trust from ``querying_agent``.
        2. For each content node with attestations:
           a. Trust-discount each attestation by derived trust.
           b. Fuse discounted attestations at the content node.
        3. Build a temporary content network with effective opinions.
        4. Run deduction inference on the content network.
        5. Return the opinion at ``query_node``.

    Content nodes without attestations keep their original marginal
    opinions (priors).

    Args:
        network:           The full SLNetwork (agents + content).
        query_node:        Content node to query.
        querying_agent:    Agent whose perspective is being computed.
        trust_fusion:      Fusion method for multi-path trust.
        content_fusion:    Fusion method for multi-agent attestation fusion.
        counterfactual_fn: Strategy for deduction counterfactuals.

    Returns:
        An ``InferenceResult`` for ``query_node``.

    Raises:
        NodeNotFoundError: If ``query_node`` or ``querying_agent`` is missing.

    References:
        Jøsang, A. (2016). Subjective Logic, §14.3, §12.6.
    """
    from jsonld_ex.sl_network.inference import infer_node
    from jsonld_ex.sl_network.network import NodeNotFoundError, SLNetwork as _SLNetwork

    # ── Validate inputs ──
    if not network.has_node(querying_agent):
        raise NodeNotFoundError(querying_agent)
    if not network.has_node(query_node):
        raise NodeNotFoundError(query_node)

    # ── Step 1: Propagate trust ──
    trust_result = propagate_trust(
        network, querying_agent, fusion_method=trust_fusion,
    )

    # ── Step 2: Compute effective opinions for attested content nodes ──
    content_fuse_fn = cumulative_fuse if content_fusion == "cumulative" else averaging_fuse
    effective_opinions: dict[str, Opinion] = {}

    for content_id in network.get_content_nodes():
        attestations = network.get_attestations_for(content_id)
        if not attestations:
            # No attestations — keep original marginal opinion
            continue

        discounted: list[Opinion] = []
        for att in attestations:
            agent_id = att.agent_id
            if agent_id in trust_result.derived_trusts:
                # Trust-discount the attestation
                derived_trust = trust_result.derived_trusts[agent_id]
                disc = trust_discount(derived_trust, att.opinion)
            elif agent_id == querying_agent:
                # Querying agent's own attestation — no discounting needed
                disc = att.opinion
            else:
                # Agent not reachable via trust — treat as vacuous trust
                disc = trust_discount(
                    Opinion(0.0, 0.0, 1.0, base_rate=0.5),
                    att.opinion,
                )
            discounted.append(disc)

        if len(discounted) == 1:
            effective_opinions[content_id] = discounted[0]
        else:
            effective_opinions[content_id] = content_fuse_fn(*discounted)

    # ── Step 3: Build temporary content network with effective opinions ──
    temp = _SLNetwork(name="_trust_inference_temp")

    for content_id in network.get_content_nodes():
        original_node = network.get_node(content_id)
        if content_id in effective_opinions:
            # Replace opinion with trust-discounted effective opinion
            node = SLNode(
                node_id=content_id,
                opinion=effective_opinions[content_id],
                node_type="content",
                label=original_node.label,
                metadata=original_node.metadata,
            )
        else:
            node = original_node
        temp.add_node(node)

    # Copy deduction edges
    content_ids = set(network.get_content_nodes())
    for (src, tgt) in list(network._edges.keys()):
        if src in content_ids and tgt in content_ids:
            temp.add_edge(network._edges[(src, tgt)])

    # Copy multi-parent edges
    for tgt, mpe in network._multi_parent_edges.items():
        if tgt in content_ids and all(
            pid in content_ids for pid in mpe.parent_ids
        ):
            temp.add_edge(mpe)

    # ── Step 4: Run content inference ──
    return infer_node(temp, query_node, counterfactual_fn=counterfactual_fn)
