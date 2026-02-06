"""
Graph Merging with Confidence-Aware Conflict Resolution for JSON-LD-Ex.

Merges multiple JSON-LD graphs from different sources, using @confidence
to resolve conflicts and combining provenance chains.  This module
builds on :mod:`inference` for principled multi-source combination and
conflict resolution.

Typical ML pipeline scenario: three knowledge extraction models each
produce a JSON-LD graph about the same entities.  ``merge_graphs``
aligns them by ``@id``, boosts confidence where they agree (noisy-OR),
and picks winners where they disagree — with a full audit trail.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from jsonld_ex.ai_ml import get_confidence
from jsonld_ex.inference import combine_sources, resolve_conflict

# ── Data Structures ────────────────────────────────────────────────


@dataclass
class MergeConflict:
    """Record of a single conflict encountered during merge."""

    node_id: str
    property_name: str
    values: list[dict[str, Any]]
    resolution: str
    winner_value: Any


@dataclass
class MergeReport:
    """Audit trail produced by :func:`merge_graphs`."""

    nodes_merged: int = 0
    properties_agreed: int = 0
    properties_conflicted: int = 0
    properties_union: int = 0
    conflicts: list[MergeConflict] = field(default_factory=list)
    source_count: int = 0


# ── Internal: annotation keys we treat as metadata, not data ──────

_ANNOTATION_KEYS = frozenset({
    "@confidence", "@source", "@extractedAt", "@method",
    "@humanVerified", "@derivedFrom",
})

_NODE_IDENTITY_KEYS = frozenset({"@id", "@type", "@context"})


# ═══════════════════════════════════════════════════════════════════
# GRAPH MERGING
# ═══════════════════════════════════════════════════════════════════


def merge_graphs(
    graphs: Sequence[dict[str, Any]],
    conflict_strategy: Literal[
        "highest", "weighted_vote", "union", "recency"
    ] = "highest",
    confidence_combination: Literal[
        "noisy_or", "average", "max"
    ] = "noisy_or",
) -> tuple[dict[str, Any], MergeReport]:
    """Merge multiple JSON-LD graphs with confidence-aware conflict resolution.

    Algorithm:
        1. Extract nodes from each graph (flatten ``@graph`` arrays and
           top-level node objects).
        2. Index nodes by ``@id``.  Nodes without ``@id`` are collected
           into an anonymous pool and passed through unchanged.
        3. For each shared ``@id``, merge properties:
           - If sources **agree** on a value → combine confidence scores.
           - If sources **conflict** → apply *conflict_strategy*.
           - If *conflict_strategy* is ``"union"`` → keep all values.
        4. Produce a merged ``@graph`` document and an audit report.

    Args:
        graphs: Two or more JSON-LD documents.  Each may contain a
            top-level node or a ``@graph`` array.
        conflict_strategy: How to resolve disagreements.  See
            :func:`~jsonld_ex.inference.resolve_conflict` for
            ``"highest"``, ``"weighted_vote"``, ``"recency"``.
            ``"union"`` keeps all conflicting values as an array.
        confidence_combination: Method for boosting confidence when
            sources agree.  Passed to
            :func:`~jsonld_ex.inference.combine_sources`.

    Returns:
        A tuple of (merged_document, MergeReport).

    Raises:
        ValueError: If fewer than 2 graphs are provided.
    """
    if len(graphs) < 2:
        raise ValueError("merge_graphs requires at least 2 graphs")

    report = MergeReport(source_count=len(graphs))

    # Step 1 — collect all nodes, indexed by @id
    id_buckets: dict[str, list[dict[str, Any]]] = {}
    anonymous_nodes: list[dict[str, Any]] = []
    merged_context = None

    for graph in graphs:
        # Capture the first non-None context we see
        if merged_context is None and "@context" in graph:
            merged_context = copy.deepcopy(graph["@context"])

        for node in _extract_nodes(graph):
            node_id = node.get("@id")
            if node_id is None:
                anonymous_nodes.append(copy.deepcopy(node))
            else:
                id_buckets.setdefault(node_id, []).append(node)

    # Step 2 — merge each bucket
    merged_nodes: list[dict[str, Any]] = []

    for node_id, nodes in id_buckets.items():
        merged = _merge_node_group(
            node_id,
            nodes,
            conflict_strategy=conflict_strategy,
            confidence_combination=confidence_combination,
            report=report,
        )
        merged_nodes.append(merged)
        report.nodes_merged += 1

    # Pass through anonymous nodes
    merged_nodes.extend(anonymous_nodes)

    # Step 3 — assemble output document
    result: dict[str, Any] = {}
    if merged_context is not None:
        result["@context"] = merged_context
    result["@graph"] = merged_nodes

    return result, report


# ═══════════════════════════════════════════════════════════════════
# GRAPH DIFF
# ═══════════════════════════════════════════════════════════════════


def diff_graphs(
    a: dict[str, Any],
    b: dict[str, Any],
) -> dict[str, Any]:
    """Compute a semantic diff between two JSON-LD graphs.

    Compares nodes by ``@id`` and properties by value (ignoring
    annotation metadata like ``@confidence``).

    Returns:
        A dict with keys:
            ``added``     — nodes/properties in *b* but not *a*.
            ``removed``   — nodes/properties in *a* but not *b*.
            ``modified``  — properties with different values.
            ``unchanged`` — properties that match.
    """
    nodes_a = _index_by_id(_extract_nodes(a))
    nodes_b = _index_by_id(_extract_nodes(b))

    ids_a = set(nodes_a.keys())
    ids_b = set(nodes_b.keys())

    added: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    modified: list[dict[str, Any]] = []
    unchanged: list[dict[str, Any]] = []

    # Nodes only in b
    for nid in ids_b - ids_a:
        added.append({"@id": nid, "node": nodes_b[nid]})

    # Nodes only in a
    for nid in ids_a - ids_b:
        removed.append({"@id": nid, "node": nodes_a[nid]})

    # Nodes in both — compare properties
    for nid in ids_a & ids_b:
        node_a = nodes_a[nid]
        node_b = nodes_b[nid]

        props_a = _data_properties(node_a)
        props_b = _data_properties(node_b)

        all_props = set(props_a.keys()) | set(props_b.keys())

        for prop in all_props:
            val_a = props_a.get(prop)
            val_b = props_b.get(prop)

            if val_a is None:
                added.append({"@id": nid, "property": prop, "value": val_b})
            elif val_b is None:
                removed.append({"@id": nid, "property": prop, "value": val_a})
            elif _bare_value(val_a) == _bare_value(val_b):
                # Same value, possibly different annotations
                conf_a = get_confidence(val_a) if isinstance(val_a, dict) else None
                conf_b = get_confidence(val_b) if isinstance(val_b, dict) else None
                entry: dict[str, Any] = {
                    "@id": nid,
                    "property": prop,
                    "value": _bare_value(val_a),
                }
                if conf_a is not None or conf_b is not None:
                    entry["confidence_a"] = conf_a
                    entry["confidence_b"] = conf_b
                unchanged.append(entry)
            else:
                modified.append({
                    "@id": nid,
                    "property": prop,
                    "value_a": val_a,
                    "value_b": val_b,
                })

    return {
        "added": added,
        "removed": removed,
        "modified": modified,
        "unchanged": unchanged,
    }


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _extract_nodes(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull all nodes from a JSON-LD document."""
    if "@graph" in doc:
        graph = doc["@graph"]
        if isinstance(graph, list):
            return list(graph)
        if isinstance(graph, dict):
            return [graph]
        return []
    # Top-level node (has @id or @type but no @graph)
    if "@id" in doc or "@type" in doc:
        # Strip @context — it's document-level, not node-level
        node = {k: v for k, v in doc.items() if k != "@context"}
        return [node]
    return []


def _index_by_id(nodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index a node list by @id.  Last-write-wins for duplicates."""
    index: dict[str, dict[str, Any]] = {}
    for node in nodes:
        nid = node.get("@id")
        if nid is not None:
            index[nid] = node
    return index


def _data_properties(node: dict[str, Any]) -> dict[str, Any]:
    """Extract non-identity properties from a node."""
    return {
        k: v for k, v in node.items()
        if k not in _NODE_IDENTITY_KEYS
    }


def _bare_value(val: Any) -> Any:
    """Strip annotation metadata to get the raw value for comparison."""
    if isinstance(val, dict):
        if "@value" in val:
            return val["@value"]
        # It might be a nested node — return @id if present
        if "@id" in val:
            return val["@id"]
    return val


def _merge_node_group(
    node_id: str,
    nodes: list[dict[str, Any]],
    conflict_strategy: str,
    confidence_combination: str,
    report: MergeReport,
) -> dict[str, Any]:
    """Merge a group of nodes that share the same @id."""
    merged: dict[str, Any] = {"@id": node_id}

    # Collect @type (union of all types)
    types: set[str] = set()
    for node in nodes:
        t = node.get("@type")
        if isinstance(t, str):
            types.add(t)
        elif isinstance(t, list):
            types.update(t)
    if len(types) == 1:
        merged["@type"] = types.pop()
    elif len(types) > 1:
        merged["@type"] = sorted(types)

    # Collect all data properties across all nodes
    all_props: set[str] = set()
    for node in nodes:
        for k in node:
            if k not in _NODE_IDENTITY_KEYS:
                all_props.add(k)

    for prop in sorted(all_props):
        values = []
        for node in nodes:
            if prop in node:
                values.append(node[prop])

        if len(values) == 0:
            continue
        elif len(values) == 1:
            merged[prop] = copy.deepcopy(values[0])
            report.properties_agreed += 1
        else:
            # Multiple sources provided this property — agree or conflict?
            bare_vals = [_bare_value(v) for v in values]
            if _all_equal(bare_vals):
                # Agreement — combine confidence
                merged[prop] = _combine_agreed(
                    values, confidence_combination
                )
                report.properties_agreed += 1
            else:
                # Conflict
                if conflict_strategy == "union":
                    merged[prop] = [copy.deepcopy(v) for v in values]
                    report.properties_union += 1
                    report.conflicts.append(MergeConflict(
                        node_id=node_id,
                        property_name=prop,
                        values=values,
                        resolution="union (all kept)",
                        winner_value=[_bare_value(v) for v in values],
                    ))
                else:
                    winner = _resolve_property_conflict(
                        values, conflict_strategy
                    )
                    merged[prop] = copy.deepcopy(winner)
                    report.conflicts.append(MergeConflict(
                        node_id=node_id,
                        property_name=prop,
                        values=values,
                        resolution=conflict_strategy,
                        winner_value=_bare_value(winner),
                    ))
                report.properties_conflicted += 1

    return merged


def _all_equal(values: list[Any]) -> bool:
    """Check if all values are equal (for conflict detection)."""
    if len(values) <= 1:
        return True
    first = values[0]
    return all(v == first for v in values[1:])


def _combine_agreed(
    values: list[Any],
    method: str,
) -> Any:
    """Combine agreed-upon values by boosting confidence."""
    # Extract confidence scores from all sources
    conf_scores = []
    best_value = values[0]
    best_conf = -1.0

    for v in values:
        c = get_confidence(v) if isinstance(v, dict) else None
        if c is not None:
            conf_scores.append(c)
            if c > best_conf:
                best_conf = c
                best_value = v

    if len(conf_scores) < 2:
        # Not enough confidence data to combine — return richest value
        return copy.deepcopy(best_value)

    combined = combine_sources(conf_scores, method=method)  # type: ignore[arg-type]
    result = copy.deepcopy(best_value)
    if isinstance(result, dict):
        result["@confidence"] = round(combined.score, 10)
    return result


def _resolve_property_conflict(
    values: list[Any],
    strategy: str,
) -> Any:
    """Resolve conflicting property values using inference module."""
    # Build assertion dicts expected by resolve_conflict
    assertions = []
    for v in values:
        if isinstance(v, dict) and "@value" in v:
            a = copy.deepcopy(v)
            # Ensure @confidence exists for resolution
            if "@confidence" not in a:
                a["@confidence"] = 0.5  # default uncertainty
            assertions.append(a)
        else:
            # Wrap plain values
            c = get_confidence(v) if isinstance(v, dict) else None
            assertions.append({
                "@value": _bare_value(v),
                "@confidence": c if c is not None else 0.5,
                **({"@extractedAt": v.get("@extractedAt")}
                   if isinstance(v, dict) and "@extractedAt" in v else {}),
                **({"@source": v.get("@source")}
                   if isinstance(v, dict) and "@source" in v else {}),
            })

    result = resolve_conflict(assertions, strategy=strategy)  # type: ignore[arg-type]
    return result.winner
