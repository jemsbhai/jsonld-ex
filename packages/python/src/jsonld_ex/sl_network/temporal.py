"""
Temporal decay for SLNetwork graphs (Tier 3).

Provides three temporal models:
    Model A -- Node-level decay (decay_network_nodes):
        Decay each node's opinion by its age before inference.
    Model B -- Edge-level decay (decay_network_edges):
        Decay edge conditional/counterfactual opinions by edge age.
    Model C -- Point-in-time snapshots (network_at_time):
        Filter to edges/nodes valid at a given timestamp.
        (Implemented in a later step.)

All functions return a NEW SLNetwork instance, leaving the original
unchanged (immutability).  Decay is delegated to the existing
``decay_opinion()`` from ``confidence_decay.py`` -- this module
composes, never reimplements.

References:
    Josang, A. (2016). Subjective Logic, Ch. 5.3 (opinion aging).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from dataclasses import dataclass

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import SLEdge, SLNode


# Type alias matching confidence_decay's decay function signature.
DecayFunction = Callable[[float, float], float]


@dataclass(frozen=True)
class TemporalDiffResult:
    """Result of comparing inferred opinions at two points in time.

    Produced by ``SLNetwork.infer_temporal_diff()``.

    Attributes:
        node_id:             The queried node.
        t1:                  The earlier time point.
        t2:                  The later time point.
        opinion_at_t1:       Inferred opinion at t1.
        opinion_at_t2:       Inferred opinion at t2.
        delta_belief:        opinion_at_t2.belief - opinion_at_t1.belief.
        delta_disbelief:     opinion_at_t2.disbelief - opinion_at_t1.disbelief.
        delta_uncertainty:   opinion_at_t2.uncertainty - opinion_at_t1.uncertainty.
    """

    node_id: str
    t1: datetime
    t2: datetime
    opinion_at_t1: Opinion
    opinion_at_t2: Opinion
    delta_belief: float
    delta_disbelief: float
    delta_uncertainty: float


def _elapsed_seconds(timestamp: datetime, reference_time: datetime) -> float:
    """Compute non-negative elapsed seconds between timestamp and reference_time.

    Returns 0.0 if timestamp is after reference_time (future observations
    have zero elapsed time).
    """
    delta = (reference_time - timestamp).total_seconds()
    return max(0.0, delta)


def decay_network_nodes(
    network: SLNetwork,
    reference_time: datetime,
    default_half_life: float,
    decay_fn: DecayFunction = exponential_decay,
) -> SLNetwork:
    """Return a new network with all node opinions decayed by their age.

    Model A: each node's opinion is decayed according to the time
    elapsed since its timestamp.  Edges are copied unchanged.

    Args:
        network:           The source network (not modified).
        reference_time:    The "now" against which node ages are computed.
        default_half_life: Half-life in seconds, used when a node has
                           no per-node half_life set.
        decay_fn:          Decay function (default: exponential_decay).

    Returns:
        A new SLNetwork with decayed node opinions.
    """
    new_net = SLNetwork(name=network.name)

    # Decay and add nodes
    for node_id in network.topological_sort():
        node = network.get_node(node_id)
        if node.timestamp is not None:
            elapsed = _elapsed_seconds(node.timestamp, reference_time)
            if elapsed > 0.0:
                half_life = node.half_life if node.half_life is not None else default_half_life
                decayed_op = decay_opinion(
                    node.opinion,
                    elapsed=elapsed,
                    half_life=half_life,
                    decay_fn=decay_fn,
                )
            else:
                decayed_op = node.opinion
        else:
            # No timestamp: no age to compute, leave unchanged
            decayed_op = node.opinion

        new_node = SLNode(
            node_id=node.node_id,
            opinion=decayed_op,
            node_type=node.node_type,
            label=node.label,
            metadata=node.metadata,
            timestamp=node.timestamp,
            half_life=node.half_life,
            multinomial_opinion=node.multinomial_opinion,
        )
        new_net.add_node(new_node)

    # Copy all edges unchanged
    _copy_edges(network, new_net)

    return new_net


def decay_network_edges(
    network: SLNetwork,
    reference_time: datetime,
    default_half_life: float,
    decay_fn: DecayFunction = exponential_decay,
) -> SLNetwork:
    """Return a new network with all edge conditional opinions decayed by their age.

    Model B: each edge's conditional (and counterfactual, if present)
    opinion is decayed according to the time elapsed since the edge's
    timestamp.  Node opinions are copied unchanged.

    This models the decay of relationship knowledge independently from
    the decay of proposition knowledge.  For example, "fever implies
    infection" (established medical fact, slow decay) vs "patient has
    fever" (recent observation, fast decay).

    Args:
        network:           The source network (not modified).
        reference_time:    The "now" against which edge ages are computed.
        default_half_life: Half-life in seconds, used when an edge has
                           no per-edge half_life set.
        decay_fn:          Decay function (default: exponential_decay).

    Returns:
        A new SLNetwork with decayed edge opinions.
    """
    new_net = SLNetwork(name=network.name)

    # Copy all nodes unchanged
    for node_id in network.topological_sort():
        new_net.add_node(network.get_node(node_id))

    # Decay and add simple edges
    for node_id in network.topological_sort():
        for parent_id in network.get_parents(node_id):
            if not network.has_edge(parent_id, node_id):
                continue  # Multi-parent edge parent, handled below
            edge = network.get_edge(parent_id, node_id)
            if edge.timestamp is not None:
                elapsed = _elapsed_seconds(edge.timestamp, reference_time)
                if elapsed > 0.0:
                    half_life = edge.half_life if edge.half_life is not None else default_half_life
                    decayed_cond = decay_opinion(
                        edge.conditional,
                        elapsed=elapsed,
                        half_life=half_life,
                        decay_fn=decay_fn,
                    )
                    decayed_cf = None
                    if edge.counterfactual is not None:
                        decayed_cf = decay_opinion(
                            edge.counterfactual,
                            elapsed=elapsed,
                            half_life=half_life,
                            decay_fn=decay_fn,
                        )
                else:
                    decayed_cond = edge.conditional
                    decayed_cf = edge.counterfactual
            else:
                decayed_cond = edge.conditional
                decayed_cf = edge.counterfactual

            new_edge = SLEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                conditional=decayed_cond,
                counterfactual=decayed_cf,
                edge_type=edge.edge_type,
                metadata=edge.metadata,
                timestamp=edge.timestamp,
                half_life=edge.half_life,
                valid_from=edge.valid_from,
                valid_until=edge.valid_until,
            )
            new_net.add_edge(new_edge)

    # Copy multinomial edges unchanged (temporal decay on MultinomialOpinion is future work)
    for node_id in network.topological_sort():
        for parent_id in network.get_parents(node_id):
            if network.has_multinomial_edge(parent_id, node_id):
                new_net.add_edge(
                    network.get_multinomial_edge(parent_id, node_id)
                )

    # Copy multi-parent edges unchanged (temporal decay on MPEs is future work)
    for node_id in network.topological_sort():
        try:
            mpe = network.get_multi_parent_edge(node_id)
            new_net.add_edge(mpe)
        except ValueError:
            pass  # No binary multi-parent edge for this node

    # Copy multi-parent multinomial edges unchanged
    for node_id in network.topological_sort():
        if network.has_multi_parent_multinomial_edge(node_id):
            new_net.add_edge(
                network.get_multi_parent_multinomial_edge(node_id)
            )

    # Copy trust and attestation edges unchanged
    _copy_trust_and_attestation_edges(network, new_net)

    return new_net


def network_at_time(
    network: SLNetwork,
    timestamp: datetime,
) -> SLNetwork:
    """Return a subnetwork containing only edges valid at the given time.

    Model C: point-in-time snapshot. An edge is valid at time t iff:
        (valid_from is None or valid_from <= t)
        and (valid_until is None or t <= valid_until)

    All nodes are preserved regardless of edge filtering. Edges with
    no validity bounds (both None) are always included.

    Node opinions are copied exactly (no decay applied). Use
    ``decay_network_nodes`` or ``infer_at`` for decay.

    Args:
        network:   The source network (not modified).
        timestamp: The point in time to query.

    Returns:
        A new SLNetwork with only temporally valid edges.
    """
    new_net = SLNetwork(name=network.name)

    # Copy all nodes
    for node_id in network.topological_sort():
        new_net.add_node(network.get_node(node_id))

    # Copy only valid edges
    for node_id in network.topological_sort():
        for parent_id in network.get_parents(node_id):
            if not network.has_edge(parent_id, node_id):
                continue  # Multi-parent edge parent, handled below
            edge = network.get_edge(parent_id, node_id)
            if _edge_valid_at(edge, timestamp):
                new_net.add_edge(edge)

    # Copy multinomial edges (no validity filtering yet — always included)
    for node_id in network.topological_sort():
        for parent_id in network.get_parents(node_id):
            if network.has_multinomial_edge(parent_id, node_id):
                new_net.add_edge(
                    network.get_multinomial_edge(parent_id, node_id)
                )

    # Multi-parent edges: include if all validity bounds pass
    # (MultiParentEdge does not currently have temporal fields,
    # so they are always included.)
    for node_id in network.topological_sort():
        try:
            mpe = network.get_multi_parent_edge(node_id)
            new_net.add_edge(mpe)
        except ValueError:
            pass

    # Multi-parent multinomial edges: always included
    for node_id in network.topological_sort():
        if network.has_multi_parent_multinomial_edge(node_id):
            new_net.add_edge(
                network.get_multi_parent_multinomial_edge(node_id)
            )

    # Copy trust and attestation edges
    _copy_trust_and_attestation_edges(network, new_net)

    return new_net


def _edge_valid_at(edge: SLEdge, timestamp: datetime) -> bool:
    """Check whether an edge's validity window contains the timestamp.

    Rules:
        - valid_from is None:  valid from the beginning of time.
        - valid_until is None: valid until the end of time.
        - Both None:           always valid.
    """
    if edge.valid_from is not None and timestamp < edge.valid_from:
        return False
    if edge.valid_until is not None and timestamp > edge.valid_until:
        return False
    return True


# -- Internal helpers --------------------------------------------------


def _copy_edges(source: SLNetwork, dest: SLNetwork) -> None:
    """Copy all deduction edges (simple, multi-parent, multinomial) from source to dest."""
    for node_id in source.topological_sort():
        for parent_id in source.get_parents(node_id):
            if source.has_edge(parent_id, node_id):
                dest.add_edge(source.get_edge(parent_id, node_id))
            elif source.has_multinomial_edge(parent_id, node_id):
                dest.add_edge(source.get_multinomial_edge(parent_id, node_id))
    # Multi-parent edges (binary): check each node
    for node_id in source.topological_sort():
        try:
            mpe = source.get_multi_parent_edge(node_id)
            dest.add_edge(mpe)
        except ValueError:
            pass  # No binary multi-parent edge for this node
    # Multi-parent multinomial edges: check each node
    for node_id in source.topological_sort():
        if source.has_multi_parent_multinomial_edge(node_id):
            dest.add_edge(source.get_multi_parent_multinomial_edge(node_id))

    _copy_trust_and_attestation_edges(source, dest)


def _copy_trust_and_attestation_edges(
    source: SLNetwork, dest: SLNetwork,
) -> None:
    """Copy trust and attestation edges from source to dest."""
    for agent_id in source.get_agents():
        for te in source.get_trust_edges_from(agent_id):
            if not dest.has_node(te.source_id) or not dest.has_node(te.target_id):
                continue
            try:
                dest.add_trust_edge(te)
            except ValueError:
                pass  # Already added

    for content_id in source.get_content_nodes():
        for ae in source.get_attestations_for(content_id):
            if not dest.has_node(ae.agent_id) or not dest.has_node(ae.content_id):
                continue
            try:
                dest.add_attestation(ae)
            except ValueError:
                pass  # Already added
