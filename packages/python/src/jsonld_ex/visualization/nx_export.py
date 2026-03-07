"""
NetworkX DiGraph export for jsonld-ex graph structures.

Requires the ``networkx`` package (optional dependency).
Install with: ``pip install jsonld-ex[viz]``

Converts an SLNetwork into a ``networkx.DiGraph`` with rich node
and edge attributes suitable for graph analysis (centrality, paths,
clustering) and rendering via NetworkX's drawing backends.

Node attributes:
    - belief, disbelief, uncertainty, base_rate (opinion components)
    - projected_probability (P(ω) = b + a·u)
    - node_type ("content" or "agent")
    - label (human-readable name, falls back to node_id)

Edge attributes (vary by edge_type):
    - edge_type: "deduction", "trust", or "attestation"
    - Deduction: cond_belief, cond_disbelief, cond_uncertainty
    - Trust: trust_belief, trust_disbelief, trust_uncertainty
    - Attestation: attest_belief, attest_disbelief, attest_uncertainty
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jsonld_ex.sl_network.network import SLNetwork


def to_networkx(network: SLNetwork) -> "nx.DiGraph":
    """Convert an SLNetwork to a NetworkX DiGraph.

    Every node and edge carries rich attributes extracted from the
    SL opinion structures, enabling downstream graph analysis.

    Args:
        network: The SLNetwork to convert.

    Returns:
        A ``networkx.DiGraph`` with node and edge attributes.

    Raises:
        ImportError: If ``networkx`` is not installed.  Install
            with ``pip install jsonld-ex[viz]``.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "networkx is required for to_networkx(). "
            "Install it with: pip install jsonld-ex[viz]"
        ) from None

    G = nx.DiGraph(name=network.name)

    # ── Nodes ──
    for node_id in sorted(network._nodes.keys()):
        node = network._nodes[node_id]
        op = node.opinion
        G.add_node(
            node_id,
            belief=op.belief,
            disbelief=op.disbelief,
            uncertainty=op.uncertainty,
            base_rate=op.base_rate,
            projected_probability=op.belief + op.base_rate * op.uncertainty,
            node_type=node.node_type,
            label=node.label or node.node_id,
        )

    # ── Deduction edges ──
    for (src, tgt), edge in network._edges.items():
        c = edge.conditional
        G.add_edge(
            src,
            tgt,
            edge_type="deduction",
            cond_belief=c.belief,
            cond_disbelief=c.disbelief,
            cond_uncertainty=c.uncertainty,
        )

    # ── Trust edges ──
    for (src, tgt), te in network._trust_edges.items():
        t = te.trust_opinion
        G.add_edge(
            src,
            tgt,
            edge_type="trust",
            trust_belief=t.belief,
            trust_disbelief=t.disbelief,
            trust_uncertainty=t.uncertainty,
        )

    # ── Attestation edges ──
    for (aid, cid), ae in network._attestation_edges.items():
        a = ae.opinion
        G.add_edge(
            aid,
            cid,
            edge_type="attestation",
            attest_belief=a.belief,
            attest_disbelief=a.disbelief,
            attest_uncertainty=a.uncertainty,
        )

    return G
