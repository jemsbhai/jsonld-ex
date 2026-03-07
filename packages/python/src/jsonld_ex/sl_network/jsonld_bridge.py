"""
JSON-LD Bridge for SLNetwork (Tier 3, Step 8).

Converts between application-level JSON-LD documents (annotated with
@confidence, @asOf, @validFrom, @validUntil) and SLNetwork graphs
for structured inference.

This is distinct from ``SLNetwork.to_jsonld()`` / ``from_jsonld()``,
which serialize the internal SLNetwork representation.  The bridge
works with standard jsonld-ex annotated documents that a user or
application would produce.

Mapping conventions:
    - Each ``@id`` in the graph becomes an SLNode.
    - Scalar ``@confidence`` is converted to an Opinion using
      ``default_uncertainty`` to split the non-belief mass.
    - A dict ``@confidence`` with ``@belief``, ``@disbelief``,
      ``@uncertainty`` keys is used as a full Opinion.
    - ``@asOf`` maps to node/edge timestamp.
    - ``@validFrom`` / ``@validUntil`` map to edge temporal bounds.
    - Properties listed in ``edge_properties`` are treated as
      deduction relationships; all others are ignored.

References:
    Josang, A. (2016). Subjective Logic, Ch. 7, Ch. 12.6.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import SLEdge, SLNode


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO 8601 string to datetime, or return None."""
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _parse_opinion(
    conf_value: Any,
    default_uncertainty: float,
    opinion_value: Any = None,
) -> Opinion:
    """Convert confidence/opinion values to an Opinion.

    If ``opinion_value`` is a dict with @belief/@disbelief/@uncertainty,
    it is used directly (lossless).  Otherwise ``conf_value`` is treated
    as a scalar projected probability and converted using
    ``default_uncertainty`` (lossy: the projected probability P = b + a*u
    does not uniquely determine the underlying opinion).

    Args:
        conf_value:          Scalar float or None.
        default_uncertainty:  Used when conf_value is a scalar.
        opinion_value:        Full opinion dict, or None.

    Returns:
        An Opinion instance.
    """
    # Prefer full opinion dict (lossless)
    if opinion_value is not None and isinstance(opinion_value, dict):
        return Opinion(
            belief=float(opinion_value["@belief"]),
            disbelief=float(opinion_value["@disbelief"]),
            uncertainty=float(opinion_value["@uncertainty"]),
            base_rate=float(opinion_value.get("@baseRate", 0.5)),
        )

    if conf_value is None:
        return Opinion(0.0, 0.0, 1.0, base_rate=0.5)

    # @confidence can itself be a dict (inline opinion)
    if isinstance(conf_value, dict):
        return Opinion(
            belief=float(conf_value["@belief"]),
            disbelief=float(conf_value["@disbelief"]),
            uncertainty=float(conf_value["@uncertainty"]),
            base_rate=float(conf_value.get("@baseRate", 0.5)),
        )

    # Scalar confidence: lossy conversion
    c = float(conf_value)
    u = min(default_uncertainty, 1.0 - c)
    d = 1.0 - c - u
    return Opinion(belief=c, disbelief=d, uncertainty=u, base_rate=0.5)


def network_from_jsonld_graph(
    graph: list[dict[str, Any]],
    opinion_key: str = "@confidence",
    timestamp_key: str = "@asOf",
    edge_properties: Sequence[str] = (),
    default_uncertainty: float = 0.0,
) -> SLNetwork:
    """Build an SLNetwork from a JSON-LD graph with annotations.

    Each item in ``graph`` with an ``@id`` becomes an SLNode.
    Properties listed in ``edge_properties`` that contain a nested
    dict with ``@id`` are treated as deduction edges from the
    referenced node (parent) to the containing node (child).

    Args:
        graph:               A list of JSON-LD node dicts.
        opinion_key:         The key holding confidence data
                             (default: ``"@confidence"``).
        timestamp_key:       The key holding observation time
                             (default: ``"@asOf"``).
        edge_properties:     Property names that represent deduction
                             relationships.
        default_uncertainty: Uncertainty assigned when converting a
                             scalar confidence to an Opinion.

    Returns:
        An SLNetwork populated with nodes and edges.
    """
    net = SLNetwork()
    edge_props = set(edge_properties)

    # -- Phase 1: Create all nodes --
    for item in graph:
        node_id = item.get("@id")
        if node_id is None:
            continue

        opinion = _parse_opinion(
            item.get(opinion_key), default_uncertainty,
            opinion_value=item.get("@opinion"),
        )
        timestamp = _parse_datetime(item.get(timestamp_key))
        label = item.get("@type")
        if isinstance(label, list):
            label = label[0] if label else None

        node = SLNode(
            node_id=node_id,
            opinion=opinion,
            timestamp=timestamp,
            label=label,
        )
        net.add_node(node)

    # -- Phase 2: Extract edges from edge_properties --
    for item in graph:
        child_id = item.get("@id")
        if child_id is None:
            continue

        for prop in edge_props:
            prop_val = item.get(prop)
            if prop_val is None:
                continue

            # Normalize to list for uniform handling
            targets = prop_val if isinstance(prop_val, list) else [prop_val]

            for target in targets:
                if not isinstance(target, dict) or "@id" not in target:
                    continue

                parent_id = target["@id"]
                if not net.has_node(parent_id):
                    continue

                conditional = _parse_opinion(
                    target.get(opinion_key), default_uncertainty,
                    opinion_value=target.get("@opinion"),
                )
                timestamp = _parse_datetime(target.get(timestamp_key))
                valid_from = _parse_datetime(target.get("@validFrom"))
                valid_until = _parse_datetime(target.get("@validUntil"))

                edge = SLEdge(
                    source_id=parent_id,
                    target_id=child_id,
                    conditional=conditional,
                    timestamp=timestamp,
                    valid_from=valid_from,
                    valid_until=valid_until,
                )
                net.add_edge(edge)

    return net


def network_to_jsonld_graph(
    network: SLNetwork,
    include_inferred: bool = True,
) -> list[dict[str, Any]]:
    """Export an SLNetwork as a JSON-LD graph with annotations.

    Each node becomes a dict with ``@id``, ``@confidence`` (the
    projected probability), ``@opinion`` (the full opinion tuple),
    and ``@asOf`` (if the node has a timestamp).

    Edges are nested in the child node under ``slnet:parentEdges``.

    Args:
        network:         The SLNetwork to export.
        include_inferred: Reserved for future use (include inferred
                          opinions in the output).

    Returns:
        A list of JSON-LD node dicts.
    """
    # Collect edges grouped by child node
    child_edges: dict[str, list[SLEdge]] = {}
    for node_id in network.topological_sort():
        for parent_id in network.get_parents(node_id):
            if network.has_edge(parent_id, node_id):
                child_edges.setdefault(node_id, []).append(
                    network.get_edge(parent_id, node_id)
                )

    result: list[dict[str, Any]] = []

    for node_id in network.topological_sort():
        node = network.get_node(node_id)
        op = node.opinion
        projected = op.belief + op.base_rate * op.uncertainty

        entry: dict[str, Any] = {
            "@id": node_id,
            "@confidence": projected,
            "@opinion": {
                "@belief": op.belief,
                "@disbelief": op.disbelief,
                "@uncertainty": op.uncertainty,
                "@baseRate": op.base_rate,
            },
        }

        if node.label is not None:
            entry["@type"] = node.label

        if node.timestamp is not None:
            entry["@asOf"] = node.timestamp.isoformat()

        # Add parent edges
        edges = child_edges.get(node_id, [])
        if edges:
            edge_list = []
            for edge in edges:
                edge_entry: dict[str, Any] = {
                    "@id": edge.source_id,
                    "@confidence": (
                        edge.conditional.belief
                        + edge.conditional.base_rate
                        * edge.conditional.uncertainty
                    ),
                    "@opinion": {
                        "@belief": edge.conditional.belief,
                        "@disbelief": edge.conditional.disbelief,
                        "@uncertainty": edge.conditional.uncertainty,
                        "@baseRate": edge.conditional.base_rate,
                    },
                }
                if edge.timestamp is not None:
                    edge_entry["@asOf"] = edge.timestamp.isoformat()
                if edge.valid_from is not None:
                    edge_entry["@validFrom"] = edge.valid_from.isoformat()
                if edge.valid_until is not None:
                    edge_entry["@validUntil"] = edge.valid_until.isoformat()
                edge_list.append(edge_entry)
            entry["slnet:parentEdges"] = edge_list

        result.append(entry)

    return result
