"""
SLNetwork — Core graph container for Subjective Logic networks.

A directed acyclic graph (DAG) of SL opinion nodes connected by
conditional (deduction) edges.  Provides graph construction, cycle
detection, topological sorting, and graph queries.

Inference is delegated to ``inference.py`` (Tier 1, Step 4).

Design Decisions:
    - Internal storage uses ``dict[str, SLNode]`` for O(1) node lookup
      and adjacency lists (``dict[str, list[str]]``) for edges.
    - Cycle detection on ``add_edge()`` via DFS — edges that would
      create cycles are rejected immediately.
    - ``topological_sort()`` uses Kahn's algorithm with deterministic
      tie-breaking by node_id (lexicographic) for reproducibility.
    - Thread-safe reads but NOT concurrent writes (documented).
    - MultiParentEdge is stored as a separate mapping and contributes
      to the adjacency structure like normal edges.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 7, §12.6, §14.3.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.types import (
    AttestationEdge,
    InferenceResult,
    MultinomialEdge,
    MultiParentEdge,
    MultiParentMultinomialEdge,
    SLEdge,
    SLNode,
    TrustEdge,
    TrustPropagationResult,
)


# ═══════════════════════════════════════════════════════════════════
# CUSTOM EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════


class CycleError(ValueError):
    """Raised when adding an edge would create a cycle in the DAG.

    Attributes:
        path: The cycle path as a list of node IDs.
    """

    def __init__(self, path: list[str]) -> None:
        self.path = path
        cycle_str = " → ".join(path)
        super().__init__(f"Adding edge would create cycle: {cycle_str}")


class NodeNotFoundError(KeyError):
    """Raised when referencing a node that does not exist in the network.

    Attributes:
        node_id: The missing node's ID.
    """

    def __init__(self, node_id: str) -> None:
        self.node_id = node_id
        super().__init__(f"Node {node_id!r} not found in network")


# ═══════════════════════════════════════════════════════════════════
# SERIALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════


def _dt_to_str(dt: datetime | None) -> str | None:
    """Serialize a datetime to ISO 8601 string, or None."""
    if dt is None:
        return None
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    """Deserialize an ISO 8601 string to datetime, or None."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _parse_bool_tuple(key_str: str, expected_length: int) -> tuple[bool, ...]:
    """Parse a stringified boolean tuple back to a tuple of bools.

    Handles the format produced by ``str((True, False))`` which gives
    ``"(True, False)"``.  Also handles compact forms like
    ``"True,False"``.

    Args:
        key_str: String representation of a bool tuple.
        expected_length: Expected number of elements.

    Returns:
        Tuple of bools.

    Raises:
        ValueError: If parsing fails or length doesn't match.
    """
    # Strip outer parens and whitespace
    cleaned = key_str.strip().strip("()")
    parts = [p.strip() for p in cleaned.split(",")]

    if len(parts) != expected_length:
        raise ValueError(
            f"Expected {expected_length} elements in bool tuple, "
            f"got {len(parts)} from {key_str!r}"
        )

    result: list[bool] = []
    for p in parts:
        if p == "True":
            result.append(True)
        elif p == "False":
            result.append(False)
        else:
            raise ValueError(
                f"Expected 'True' or 'False', got {p!r} in {key_str!r}"
            )
    return tuple(result)


def _parse_str_tuple(key_str: str, expected_length: int) -> tuple[str, ...]:
    """Parse a stringified string-tuple back to a tuple of strings.

    Handles the format produced by ``str(("a", "b"))`` which gives
    ``"('a', 'b')"``.

    Args:
        key_str: String representation of a string tuple.
        expected_length: Expected number of elements.

    Returns:
        Tuple of strings.

    Raises:
        ValueError: If parsing fails or length doesn't match.
    """
    # Strip outer parens and whitespace
    cleaned = key_str.strip().strip("()")
    parts = [p.strip().strip("'\"") for p in cleaned.split(",")]
    # Handle trailing comma for single-element tuples: "('a',)" -> ["a", ""]
    parts = [p for p in parts if p]

    if len(parts) != expected_length:
        raise ValueError(
            f"Expected {expected_length} elements in str tuple, "
            f"got {len(parts)} from {key_str!r}"
        )

    return tuple(parts)


# ═══════════════════════════════════════════════════════════════════
# SLNetwork CLASS
# ═══════════════════════════════════════════════════════════════════


class SLNetwork:
    """A directed acyclic graph of Subjective Logic opinion nodes.

    Nodes represent propositions with marginal opinions.  Edges
    represent conditional deduction relationships (Tier 1) or trust
    and attestation relationships (Tier 2, future).

    Graph construction enforces the DAG invariant: every ``add_edge()``
    call checks for cycles and rejects edges that would violate it.

    Thread Safety:
        Read operations (get_node, get_parents, topological_sort, etc.)
        are safe for concurrent access.  Write operations (add_node,
        add_edge, remove_node, remove_edge) are NOT thread-safe and
        must be externally synchronized if used concurrently.

    Example::

        from jsonld_ex.confidence_algebra import Opinion
        from jsonld_ex.sl_network import SLNetwork, SLNode, SLEdge

        net = SLNetwork(name="medical_diagnosis")
        net.add_node(SLNode("fever", Opinion(0.8, 0.1, 0.1)))
        net.add_node(SLNode("infection", Opinion(0.5, 0.2, 0.3)))
        net.add_edge(SLEdge(
            source_id="fever",
            target_id="infection",
            conditional=Opinion(0.9, 0.05, 0.05),
        ))
        print(net.topological_sort())  # ['fever', 'infection']
    """

    def __init__(self, name: str | None = None) -> None:
        """Create an empty SL network.

        Args:
            name: Optional human-readable name for the network.
        """
        self._name: str | None = name

        # Node storage: node_id → SLNode
        self._nodes: dict[str, SLNode] = {}

        # Adjacency lists (forward and reverse)
        # _children[parent] = [child1, child2, ...]
        # _parents[child] = [parent1, parent2, ...]
        self._children: dict[str, list[str]] = {}
        self._parents: dict[str, list[str]] = {}

        # Edge storage: (source_id, target_id) → SLEdge
        self._edges: dict[tuple[str, str], SLEdge] = {}

        # Multi-parent edge storage: target_id → MultiParentEdge
        self._multi_parent_edges: dict[str, MultiParentEdge] = {}

        # Multinomial edge storage: (source_id, target_id) → MultinomialEdge
        self._multinomial_edges: dict[tuple[str, str], MultinomialEdge] = {}

        # Multi-parent multinomial edge storage: target_id → MultiParentMultinomialEdge
        self._multi_parent_multinomial_edges: dict[str, MultiParentMultinomialEdge] = {}

        # Trust edge storage (Tier 2): (source_id, target_id) → TrustEdge
        self._trust_edges: dict[tuple[str, str], TrustEdge] = {}

        # Attestation edge storage (Tier 2): (agent_id, content_id) → AttestationEdge
        self._attestation_edges: dict[tuple[str, str], AttestationEdge] = {}

    # ── Properties ─────────────────────────────────────────────────

    @property
    def name(self) -> str | None:
        """The network's human-readable name."""
        return self._name

    # ── Graph Construction ─────────────────────────────────────────

    def add_node(self, node: SLNode) -> None:
        """Add a node to the network.

        Args:
            node: The SLNode to add.

        Raises:
            TypeError: If ``node`` is not an SLNode.
            ValueError: If a node with the same ID already exists.
        """
        if not isinstance(node, SLNode):
            raise TypeError(
                f"Expected SLNode, got {type(node).__name__}"
            )
        if node.node_id in self._nodes:
            raise ValueError(
                f"Node {node.node_id!r} already exists in network"
            )
        self._nodes[node.node_id] = node
        self._children[node.node_id] = []
        self._parents[node.node_id] = []

    def add_edge(self, edge: SLEdge | MultiParentEdge) -> None:
        """Add a directed edge to the network.

        For ``SLEdge``: connects source → target.
        For ``MultiParentEdge``: connects each parent → target.

        Validates that all referenced nodes exist and that adding the
        edge does not create a cycle.

        Args:
            edge: The edge to add.

        Raises:
            TypeError: If ``edge`` is not SLEdge or MultiParentEdge.
            NodeNotFoundError: If any referenced node is not in the network.
            ValueError: If the edge already exists.
            CycleError: If adding the edge would create a cycle.
        """
        if isinstance(edge, SLEdge):
            self._add_simple_edge(edge)
        elif isinstance(edge, MultiParentEdge):
            self._add_multi_parent_edge(edge)
        elif isinstance(edge, MultinomialEdge):
            self._add_multinomial_edge(edge)
        elif isinstance(edge, MultiParentMultinomialEdge):
            self._add_multi_parent_multinomial_edge(edge)
        else:
            raise TypeError(
                f"Expected SLEdge, MultiParentEdge, MultinomialEdge, "
                f"or MultiParentMultinomialEdge, "
                f"got {type(edge).__name__}"
            )

    def _add_simple_edge(self, edge: SLEdge) -> None:
        """Add a single-parent deduction edge."""
        src, tgt = edge.source_id, edge.target_id

        # Validate nodes exist
        if src not in self._nodes:
            raise NodeNotFoundError(src)
        if tgt not in self._nodes:
            raise NodeNotFoundError(tgt)

        # Check for duplicate edge (including cross-type conflicts)
        if (src, tgt) in self._edges:
            raise ValueError(
                f"Edge {src!r} → {tgt!r} already exists"
            )
        if (src, tgt) in self._multinomial_edges:
            raise ValueError(
                f"Edge {src!r} → {tgt!r} already exists as MultinomialEdge"
            )

        # Cycle detection: would adding src → tgt create a cycle?
        # A cycle exists iff there is already a path from tgt to src.
        if self._has_path(tgt, src):
            # Reconstruct the cycle path for the error message
            path = self._find_path(tgt, src)
            raise CycleError([src] + path + [src])

        # Commit the edge
        self._edges[(src, tgt)] = edge
        self._children[src].append(tgt)
        self._parents[tgt].append(src)

    def _add_multi_parent_edge(self, edge: MultiParentEdge) -> None:
        """Add a multi-parent conditional table edge."""
        tgt = edge.target_id

        # Validate target node exists
        if tgt not in self._nodes:
            raise NodeNotFoundError(tgt)

        # Validate all parent nodes exist
        for pid in edge.parent_ids:
            if pid not in self._nodes:
                raise NodeNotFoundError(pid)

        # Check for existing multi-parent edge on this target
        if tgt in self._multi_parent_edges:
            raise ValueError(
                f"MultiParentEdge for target {tgt!r} already exists"
            )

        # Cycle detection: for each parent, check that adding
        # parent → target wouldn't create a cycle
        for pid in edge.parent_ids:
            if (pid, tgt) in self._edges:
                raise ValueError(
                    f"Edge {pid!r} → {tgt!r} already exists as SLEdge"
                )
            if self._has_path(tgt, pid):
                path = self._find_path(tgt, pid)
                raise CycleError([pid] + path + [pid])

        # Commit: add adjacency entries for all parents
        for pid in edge.parent_ids:
            self._children[pid].append(tgt)
            self._parents[tgt].append(pid)

        self._multi_parent_edges[tgt] = edge

    def _add_multinomial_edge(self, edge: MultinomialEdge) -> None:
        """Add a multinomial conditional edge."""
        src, tgt = edge.source_id, edge.target_id

        # Validate nodes exist
        if src not in self._nodes:
            raise NodeNotFoundError(src)
        if tgt not in self._nodes:
            raise NodeNotFoundError(tgt)

        # Check for duplicate edge (including cross-type conflicts)
        if (src, tgt) in self._multinomial_edges:
            raise ValueError(
                f"MultinomialEdge {src!r} → {tgt!r} already exists"
            )
        if (src, tgt) in self._edges:
            raise ValueError(
                f"Edge {src!r} → {tgt!r} already exists as SLEdge"
            )

        # Cycle detection: would adding src → tgt create a cycle?
        if self._has_path(tgt, src):
            path = self._find_path(tgt, src)
            raise CycleError([src] + path + [src])

        # Commit the edge
        self._multinomial_edges[(src, tgt)] = edge
        self._children[src].append(tgt)
        self._parents[tgt].append(src)

    def _add_multi_parent_multinomial_edge(
        self, edge: MultiParentMultinomialEdge
    ) -> None:
        """Add a multi-parent multinomial conditional table edge."""
        tgt = edge.target_id

        # Validate target node exists
        if tgt not in self._nodes:
            raise NodeNotFoundError(tgt)

        # Validate all parent nodes exist
        for pid in edge.parent_ids:
            if pid not in self._nodes:
                raise NodeNotFoundError(pid)

        # Check for existing multi-parent multinomial edge on this target
        if tgt in self._multi_parent_multinomial_edges:
            raise ValueError(
                f"MultiParentMultinomialEdge for target {tgt!r} already exists"
            )

        # Cycle detection and cross-type conflict check for each parent
        for pid in edge.parent_ids:
            if (pid, tgt) in self._edges:
                raise ValueError(
                    f"Edge {pid!r} → {tgt!r} already exists as SLEdge"
                )
            if self._has_path(tgt, pid):
                path = self._find_path(tgt, pid)
                raise CycleError([pid] + path + [pid])

        # Commit: add adjacency entries for all parents
        for pid in edge.parent_ids:
            self._children[pid].append(tgt)
            self._parents[tgt].append(pid)

        self._multi_parent_multinomial_edges[tgt] = edge

    # ── Agent Nodes (Tier 2) ───────────────────────────────────────────

    def add_agent(self, node: SLNode) -> None:
        """Add an agent node to the network.

        Convenience method that validates ``node.node_type == "agent"``
        before delegating to ``add_node()``.

        Args:
            node: The SLNode to add.  Must have ``node_type="agent"``.

        Raises:
            TypeError:  If ``node`` is not an SLNode.
            ValueError: If ``node.node_type`` is not ``"agent"``.
            ValueError: If a node with the same ID already exists.
        """
        if not isinstance(node, SLNode):
            raise TypeError(
                f"Expected SLNode, got {type(node).__name__}"
            )
        if node.node_type != "agent":
            raise ValueError(
                f"add_agent() requires node_type='agent', "
                f"got {node.node_type!r}"
            )
        self.add_node(node)

    # ── Trust & Attestation (Tier 2) ────────────────────────────────────

    def add_trust_edge(self, edge: TrustEdge) -> None:
        """Add a trust relationship between two agent nodes.

        Trust edges live in a separate store from deduction edges
        and do NOT participate in cycle detection (trust subgraph
        is independent from the content deduction DAG).

        Args:
            edge: The TrustEdge to add.

        Raises:
            TypeError:  If ``edge`` is not a TrustEdge.
            NodeNotFoundError: If either agent node is not in the network.
            ValueError: If the trust edge already exists.
        """
        if not isinstance(edge, TrustEdge):
            raise TypeError(
                f"Expected TrustEdge, got {type(edge).__name__}"
            )
        src, tgt = edge.source_id, edge.target_id

        if src not in self._nodes:
            raise NodeNotFoundError(src)
        if tgt not in self._nodes:
            raise NodeNotFoundError(tgt)

        key = (src, tgt)
        if key in self._trust_edges:
            raise ValueError(
                f"Trust edge {src!r} → {tgt!r} already exists"
            )

        self._trust_edges[key] = edge

    def add_attestation(self, edge: AttestationEdge) -> None:
        """Add an attestation: an agent's opinion about a content node.

        Args:
            edge: The AttestationEdge to add.

        Raises:
            TypeError:  If ``edge`` is not an AttestationEdge.
            NodeNotFoundError: If agent or content node is missing.
            ValueError: If the attestation already exists.
        """
        if not isinstance(edge, AttestationEdge):
            raise TypeError(
                f"Expected AttestationEdge, got {type(edge).__name__}"
            )
        aid, cid = edge.agent_id, edge.content_id

        if aid not in self._nodes:
            raise NodeNotFoundError(aid)
        if cid not in self._nodes:
            raise NodeNotFoundError(cid)

        key = (aid, cid)
        if key in self._attestation_edges:
            raise ValueError(
                f"Attestation {aid!r} → {cid!r} already exists"
            )

        self._attestation_edges[key] = edge

    def get_trust_edges_from(self, agent_id: str) -> list[TrustEdge]:
        """Return all outgoing trust edges from the given agent.

        Args:
            agent_id: The trusting agent's ID.

        Returns:
            List of TrustEdge objects, sorted by target_id.

        Raises:
            NodeNotFoundError: If the agent node does not exist.
        """
        if agent_id not in self._nodes:
            raise NodeNotFoundError(agent_id)
        return sorted(
            (
                te for (src, _), te in self._trust_edges.items()
                if src == agent_id
            ),
            key=lambda te: te.target_id,
        )

    def get_attestations_for(self, content_id: str) -> list[AttestationEdge]:
        """Return all attestation edges targeting the given content node.

        Args:
            content_id: The content node ID.

        Returns:
            List of AttestationEdge objects, sorted by agent_id.

        Raises:
            NodeNotFoundError: If the content node does not exist.
        """
        if content_id not in self._nodes:
            raise NodeNotFoundError(content_id)
        return sorted(
            (
                ae for (_, cid), ae in self._attestation_edges.items()
                if cid == content_id
            ),
            key=lambda ae: ae.agent_id,
        )

    # ── Node Filtering (Tier 2) ────────────────────────────────────────

    def get_agents(self) -> list[str]:
        """Return sorted list of agent node IDs."""
        return sorted(
            nid for nid, n in self._nodes.items()
            if n.node_type == "agent"
        )

    def get_content_nodes(self) -> list[str]:
        """Return sorted list of content node IDs."""
        return sorted(
            nid for nid, n in self._nodes.items()
            if n.node_type == "content"
        )

    # ── Subgraph Extraction (Tier 2) ───────────────────────────────────

    def get_trust_subgraph(self) -> SLNetwork:
        """Return a new SLNetwork containing only agent nodes and trust edges.

        The returned network has no content nodes, no deduction edges,
        and no attestation edges.
        """
        sub = SLNetwork(name=f"{self._name}_trust" if self._name else None)
        for nid in self.get_agents():
            sub.add_node(self._nodes[nid])
        for key, te in self._trust_edges.items():
            sub.add_trust_edge(te)
        return sub

    def get_content_subgraph(self) -> SLNetwork:
        """Return a new SLNetwork containing only content nodes and deduction edges.

        The returned network has no agent nodes, no trust edges,
        and no attestation edges.
        """
        sub = SLNetwork(name=f"{self._name}_content" if self._name else None)
        content_ids = set(self.get_content_nodes())
        for nid in sorted(content_ids):
            sub.add_node(self._nodes[nid])
        # Copy deduction edges where both endpoints are content nodes
        for (src, tgt), edge in self._edges.items():
            if src in content_ids and tgt in content_ids:
                sub.add_edge(edge)
        # Copy multi-parent edges where target and all parents are content
        for tgt, mpe in self._multi_parent_edges.items():
            if tgt in content_ids and all(
                pid in content_ids for pid in mpe.parent_ids
            ):
                sub.add_edge(mpe)
        return sub

    # ── Trust Propagation (Tier 2) ─────────────────────────────────────

    def propagate_trust(
        self,
        querying_agent: str,
        fusion_method: str = "cumulative",
    ) -> TrustPropagationResult:
        """Compute transitive trust from a querying agent.

        Delegates to ``trust.propagate_trust()``.

        Args:
            querying_agent: The agent whose perspective is being computed.
            fusion_method:  ``"cumulative"`` or ``"averaging"``.

        Returns:
            A ``TrustPropagationResult``.
        """
        from jsonld_ex.sl_network.trust import propagate_trust as _propagate
        return _propagate(
            self, querying_agent, fusion_method=fusion_method,
        )

    def infer_at(
        self,
        node_id: str,
        reference_time: datetime,
        decay_model: Literal["nodes", "edges", "both"] = "both",
        default_half_life: float = 86400.0,
        decay_fn: Callable[[float, float], float] | None = None,
        **kwargs,
    ) -> InferenceResult:
        """Infer with temporal decay applied.

        1. Apply decay to node opinions (Model A) and/or edge
           conditionals (Model B) based on ``decay_model``.
        2. Run standard inference on the decayed network.

        The original network is not modified.

        Args:
            node_id:           The node to query.
            reference_time:    The "now" for computing elapsed time.
            decay_model:       ``"nodes"``, ``"edges"``, or ``"both"``
                               (default).
            default_half_life: Half-life in seconds when a node/edge
                               has no per-element half_life set.
            decay_fn:          Custom decay function, or None for
                               exponential decay.
            **kwargs:          Forwarded to ``infer_node()``.

        Returns:
            An ``InferenceResult`` for the queried node.
        """
        from jsonld_ex.sl_network.temporal import (
            decay_network_nodes,
            decay_network_edges,
        )
        from jsonld_ex.confidence_decay import exponential_decay
        from jsonld_ex.sl_network.inference import infer_node as _infer_node

        fn = decay_fn if decay_fn is not None else exponential_decay
        net = self

        if decay_model in ("nodes", "both"):
            net = decay_network_nodes(
                net, reference_time=reference_time,
                default_half_life=default_half_life, decay_fn=fn,
            )
        if decay_model in ("edges", "both"):
            net = decay_network_edges(
                net, reference_time=reference_time,
                default_half_life=default_half_life, decay_fn=fn,
            )

        return _infer_node(net, node_id, **kwargs)

    def infer_temporal_diff(
        self,
        node_id: str,
        t1: datetime,
        t2: datetime,
        **kwargs,
    ):
        """Compare inferred opinions at two points in time.

        Runs ``infer_at()`` at both ``t1`` and ``t2``, then computes
        component-wise deltas (t2 - t1).

        Args:
            node_id: The node to query.
            t1:      The first (typically earlier) time point.
            t2:      The second (typically later) time point.
            **kwargs: Forwarded to ``infer_at()`` (decay_model,
                      default_half_life, decay_fn, etc.).

        Returns:
            A ``TemporalDiffResult`` with opinions at both times
            and their deltas.
        """
        from jsonld_ex.sl_network.temporal import TemporalDiffResult

        result_t1 = self.infer_at(node_id, reference_time=t1, **kwargs)
        result_t2 = self.infer_at(node_id, reference_time=t2, **kwargs)

        op1 = result_t1.opinion
        op2 = result_t2.opinion

        return TemporalDiffResult(
            node_id=node_id,
            t1=t1,
            t2=t2,
            opinion_at_t1=op1,
            opinion_at_t2=op2,
            delta_belief=op2.belief - op1.belief,
            delta_disbelief=op2.disbelief - op1.disbelief,
            delta_uncertainty=op2.uncertainty - op1.uncertainty,
        )

    def infer_with_trust(
        self,
        query_node: str,
        querying_agent: str,
        trust_fusion: str = "cumulative",
        content_fusion: str = "cumulative",
        counterfactual_fn: str = "vacuous",
    ) -> InferenceResult:
        """Run combined trust + content inference.

        Delegates to ``trust.infer_with_trust()``.

        Args:
            query_node:       Content node to query.
            querying_agent:   Agent whose perspective is being computed.
            trust_fusion:     Fusion method for multi-path trust.
            content_fusion:   Fusion method for multi-agent attestations.
            counterfactual_fn: Strategy for deduction counterfactuals.

        Returns:
            An ``InferenceResult`` for the query node.
        """
        from jsonld_ex.sl_network.trust import (
            infer_with_trust as _infer_with_trust,
        )
        return _infer_with_trust(
            self,
            query_node=query_node,
            querying_agent=querying_agent,
            trust_fusion=trust_fusion,
            content_fusion=content_fusion,
            counterfactual_fn=counterfactual_fn,
        )

    # ── Node Removal ───────────────────────────────────────────────

    def remove_node(self, node_id: str) -> None:
        """Remove a node and all its incident edges.

        Args:
            node_id: ID of the node to remove.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)

        # Remove all outgoing edges (this node as source)
        for child in list(self._children[node_id]):
            self._remove_edge_internal(node_id, child)

        # Remove all incoming edges (this node as target)
        for parent in list(self._parents[node_id]):
            self._remove_edge_internal(parent, node_id)

        # Remove multi-parent edge if this node is a target
        if node_id in self._multi_parent_edges:
            del self._multi_parent_edges[node_id]

        # Remove multi-parent edges where this node is a parent
        for tgt, mpe in list(self._multi_parent_edges.items()):
            if node_id in mpe.parent_ids:
                # Remove the entire multi-parent edge since it's
                # now structurally invalid
                for pid in mpe.parent_ids:
                    if pid in self._children and tgt in self._children[pid]:
                        self._children[pid].remove(tgt)
                    if tgt in self._parents and pid in self._parents[tgt]:
                        self._parents[tgt].remove(pid)
                del self._multi_parent_edges[tgt]

        # Remove multinomial edges where this node is source or target
        for (src, tgt) in list(self._multinomial_edges.keys()):
            if src == node_id or tgt == node_id:
                del self._multinomial_edges[(src, tgt)]
                # Adjacency cleanup is already handled above via
                # _remove_edge_internal for outgoing/incoming edges

        # Remove multi-parent multinomial edge if this node is a target
        if node_id in self._multi_parent_multinomial_edges:
            del self._multi_parent_multinomial_edges[node_id]

        # Remove multi-parent multinomial edges where this node is a parent
        for tgt, mpe in list(self._multi_parent_multinomial_edges.items()):
            if node_id in mpe.parent_ids:
                # Remove the entire edge since it's now structurally invalid
                for pid in mpe.parent_ids:
                    if pid in self._children and tgt in self._children[pid]:
                        self._children[pid].remove(tgt)
                    if tgt in self._parents and pid in self._parents[tgt]:
                        self._parents[tgt].remove(pid)
                del self._multi_parent_multinomial_edges[tgt]

        # Remove the node itself
        del self._nodes[node_id]
        del self._children[node_id]
        del self._parents[node_id]

    def remove_edge(self, source_id: str, target_id: str) -> None:
        """Remove a directed edge.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Raises:
            NodeNotFoundError: If either node does not exist.
            ValueError: If the edge does not exist.
        """
        if source_id not in self._nodes:
            raise NodeNotFoundError(source_id)
        if target_id not in self._nodes:
            raise NodeNotFoundError(target_id)

        if (source_id, target_id) not in self._edges:
            raise ValueError(
                f"Edge {source_id!r} → {target_id!r} does not exist"
            )

        self._remove_edge_internal(source_id, target_id)

    def _remove_edge_internal(self, source_id: str, target_id: str) -> None:
        """Remove an edge from internal storage (no validation)."""
        key = (source_id, target_id)
        if key in self._edges:
            del self._edges[key]
        if key in self._multinomial_edges:
            del self._multinomial_edges[key]
        if target_id in self._children.get(source_id, []):
            self._children[source_id].remove(target_id)
        if source_id in self._parents.get(target_id, []):
            self._parents[target_id].remove(source_id)

    # ── Graph Queries ──────────────────────────────────────────────

    def get_node(self, node_id: str) -> SLNode:
        """Retrieve a node by ID.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return self._nodes[node_id]

    def get_edge(self, source_id: str, target_id: str) -> SLEdge:
        """Retrieve an edge by source and target.

        Raises:
            NodeNotFoundError: If either node does not exist.
            ValueError: If the edge does not exist.
        """
        if source_id not in self._nodes:
            raise NodeNotFoundError(source_id)
        if target_id not in self._nodes:
            raise NodeNotFoundError(target_id)
        key = (source_id, target_id)
        if key not in self._edges:
            raise ValueError(
                f"Edge {source_id!r} → {target_id!r} does not exist"
            )
        return self._edges[key]

    def get_multi_parent_edge(self, target_id: str) -> MultiParentEdge:
        """Retrieve the multi-parent edge for a target node.

        Raises:
            NodeNotFoundError: If the target node does not exist.
            ValueError: If no multi-parent edge exists for this target.
        """
        if target_id not in self._nodes:
            raise NodeNotFoundError(target_id)
        if target_id not in self._multi_parent_edges:
            raise ValueError(
                f"No MultiParentEdge for target {target_id!r}"
            )
        return self._multi_parent_edges[target_id]

    def get_parents(self, node_id: str) -> list[str]:
        """Return the parent node IDs for a given node.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return list(self._parents[node_id])

    def get_children(self, node_id: str) -> list[str]:
        """Return the child node IDs for a given node.

        Raises:
            NodeNotFoundError: If the node does not exist.
        """
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)
        return list(self._children[node_id])

    def get_roots(self) -> list[str]:
        """Return node IDs with no parents (in-degree 0).

        Returns a sorted list for deterministic output.
        """
        return sorted(
            nid for nid, parents in self._parents.items()
            if len(parents) == 0
        )

    def get_leaves(self) -> list[str]:
        """Return node IDs with no children (out-degree 0).

        Returns a sorted list for deterministic output.
        """
        return sorted(
            nid for nid, children in self._children.items()
            if len(children) == 0
        )

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the network."""
        return node_id in self._nodes

    def has_edge(self, source_id: str, target_id: str) -> bool:
        """Check if an edge exists in the network."""
        return (source_id, target_id) in self._edges

    def get_multinomial_edge(
        self, source_id: str, target_id: str
    ) -> MultinomialEdge:
        """Return the MultinomialEdge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            The MultinomialEdge.

        Raises:
            ValueError: If no MultinomialEdge exists between the nodes.
        """
        key = (source_id, target_id)
        if key not in self._multinomial_edges:
            raise ValueError(
                f"MultinomialEdge {source_id!r} → {target_id!r} not found"
            )
        return self._multinomial_edges[key]

    def has_multinomial_edge(self, source_id: str, target_id: str) -> bool:
        """Check if a MultinomialEdge exists between two nodes."""
        return (source_id, target_id) in self._multinomial_edges

    def get_multi_parent_multinomial_edge(
        self, target_id: str
    ) -> MultiParentMultinomialEdge:
        """Return the MultiParentMultinomialEdge for a target node.

        Args:
            target_id: Target node ID.

        Returns:
            The MultiParentMultinomialEdge.

        Raises:
            ValueError: If no MultiParentMultinomialEdge exists for this target.
        """
        if target_id not in self._multi_parent_multinomial_edges:
            raise ValueError(
                f"MultiParentMultinomialEdge for target {target_id!r} not found"
            )
        return self._multi_parent_multinomial_edges[target_id]

    def has_multi_parent_multinomial_edge(self, target_id: str) -> bool:
        """Check if a MultiParentMultinomialEdge exists for a target node."""
        return target_id in self._multi_parent_multinomial_edges

    def node_count(self) -> int:
        """Return the number of nodes in the network."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the network.

        Counts simple edges plus the individual parent→target edges
        contributed by multi-parent edges (both binary and multinomial).
        """
        multi_edges = sum(
            len(mpe.parent_ids) for mpe in self._multi_parent_edges.values()
        )
        multi_multinomial_edges = sum(
            len(mpe.parent_ids)
            for mpe in self._multi_parent_multinomial_edges.values()
        )
        return (
            len(self._edges)
            + multi_edges
            + len(self._multinomial_edges)
            + multi_multinomial_edges
        )

    # ── Structural Analysis ────────────────────────────────────────

    def is_tree(self) -> bool:
        """Check if the network is a tree (every node has at most 1 parent).

        An empty network or a single-node network is considered a tree.
        A forest (multiple disconnected trees) returns True.
        """
        return all(
            len(parents) <= 1
            for parents in self._parents.values()
        )

    def is_dag(self) -> bool:
        """Check if the network is a valid DAG (no cycles).

        This should always return True since cycle detection is
        enforced on ``add_edge()``.  This method provides an
        explicit verification.
        """
        # Use Kahn's algorithm: if topological sort processes all
        # nodes, the graph is a DAG.
        try:
            topo = self.topological_sort()
            return len(topo) == len(self._nodes)
        except ValueError:
            return False  # pragma: no cover — shouldn't happen

    def topological_sort(self) -> list[str]:
        """Return a topological ordering of all nodes.

        Uses Kahn's algorithm with lexicographic tie-breaking
        for deterministic output across runs.

        Returns:
            List of node IDs in topological order (roots first).

        Raises:
            ValueError: If the graph contains a cycle (should not
                happen if edges are added through ``add_edge()``).
        """
        # Compute in-degrees
        in_degree: dict[str, int] = {nid: 0 for nid in self._nodes}
        for nid in self._nodes:
            in_degree[nid] = len(self._parents[nid])

        # Initialize queue with zero-in-degree nodes (sorted for determinism)
        queue: list[str] = sorted(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        # Use a heap-like sorted insertion for deterministic ordering
        result: list[str] = []

        while queue:
            # Pop the lexicographically smallest node
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for each child
            for child in sorted(self._children[node]):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    # Insert into sorted position
                    self._sorted_insert(queue, child)

        if len(result) != len(self._nodes):
            raise ValueError(
                "Graph contains a cycle — topological sort is impossible"
            )

        return result

    @staticmethod
    def _sorted_insert(queue: list[str], item: str) -> None:
        """Insert an item into a sorted list maintaining order."""
        # Simple insertion sort — efficient for small queues
        for i, existing in enumerate(queue):
            if item < existing:
                queue.insert(i, item)
                return
        queue.append(item)

    # ── Cycle Detection Helpers ────────────────────────────────────

    def _has_path(self, source: str, target: str) -> bool:
        """Check if there is a directed path from source to target.

        Uses BFS for simplicity and O(V+E) worst case.
        """
        if source == target:
            return True
        visited: set[str] = set()
        queue: deque[str] = deque([source])
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            for child in self._children.get(current, []):
                if child not in visited:
                    queue.append(child)
        return False

    def _find_path(self, source: str, target: str) -> list[str]:
        """Find a directed path from source to target (BFS).

        Returns the path as a list of node IDs (inclusive of both
        source and target).  Returns empty list if no path exists.
        """
        if source == target:
            return [source]
        visited: set[str] = set()
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == target:
                return path
            if current in visited:
                continue
            visited.add(current)
            for child in self._children.get(current, []):
                if child not in visited:
                    queue.append(path + [child])
        return []

    # ── Serialization (dict-based, JSON-LD deferred to Step 8) ─────

    def to_dict(self) -> dict[str, Any]:
        """Serialize the network to a plain dictionary.

        Returns:
            A dictionary with 'name', 'nodes', 'edges', and
            'multi_parent_edges' keys.
        """
        return {
            "name": self._name,
            "nodes": {
                nid: {
                    "node_id": n.node_id,
                    "opinion": n.opinion.to_jsonld(),
                    "node_type": n.node_type,
                    "label": n.label,
                    "metadata": n.metadata,
                    "timestamp": _dt_to_str(n.timestamp),
                    "half_life": n.half_life,
                    **({
                        "multinomial_opinion": n.multinomial_opinion.to_dict()
                    } if n.multinomial_opinion is not None else {}),
                }
                for nid, n in self._nodes.items()
            },
            "edges": {
                f"{src}->{tgt}": {
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "conditional": e.conditional.to_jsonld(),
                    "counterfactual": (
                        e.counterfactual.to_jsonld()
                        if e.counterfactual is not None
                        else None
                    ),
                    "edge_type": e.edge_type,
                    "metadata": e.metadata,
                    "timestamp": _dt_to_str(e.timestamp),
                    "half_life": e.half_life,
                    "valid_from": _dt_to_str(e.valid_from),
                    "valid_until": _dt_to_str(e.valid_until),
                }
                for (src, tgt), e in self._edges.items()
            },
            "multi_parent_edges": {
                tgt: {
                    "target_id": mpe.target_id,
                    "parent_ids": list(mpe.parent_ids),
                    "conditionals": {
                        str(k): v.to_jsonld()
                        for k, v in mpe.conditionals.items()
                    },
                    "edge_type": mpe.edge_type,
                }
                for tgt, mpe in self._multi_parent_edges.items()
            },
            "trust_edges": {
                f"{src}->{tgt}": {
                    "source_id": te.source_id,
                    "target_id": te.target_id,
                    "trust_opinion": te.trust_opinion.to_jsonld(),
                    "edge_type": te.edge_type,
                    "metadata": te.metadata,
                    "timestamp": _dt_to_str(te.timestamp),
                    "half_life": te.half_life,
                }
                for (src, tgt), te in self._trust_edges.items()
            },
            "attestation_edges": {
                f"{aid}->{cid}": {
                    "agent_id": ae.agent_id,
                    "content_id": ae.content_id,
                    "opinion": ae.opinion.to_jsonld(),
                    "edge_type": ae.edge_type,
                    "metadata": ae.metadata,
                    "timestamp": _dt_to_str(ae.timestamp),
                    "half_life": ae.half_life,
                }
                for (aid, cid), ae in self._attestation_edges.items()
            },
            "multinomial_edges": {
                f"{src}->{tgt}": {
                    "source_id": me.source_id,
                    "target_id": me.target_id,
                    "conditionals": {
                        state: mop.to_dict()
                        for state, mop in me.conditionals.items()
                    },
                    "edge_type": me.edge_type,
                    "metadata": me.metadata,
                    "timestamp": _dt_to_str(me.timestamp),
                    "half_life": me.half_life,
                    "valid_from": _dt_to_str(me.valid_from),
                    "valid_until": _dt_to_str(me.valid_until),
                }
                for (src, tgt), me in self._multinomial_edges.items()
            },
            "multi_parent_multinomial_edges": {
                tgt: {
                    "target_id": mpe.target_id,
                    "parent_ids": list(mpe.parent_ids),
                    "conditionals": {
                        str(k): v.to_dict()
                        for k, v in mpe.conditionals.items()
                    },
                    "edge_type": mpe.edge_type,
                    "metadata": mpe.metadata,
                    "timestamp": _dt_to_str(mpe.timestamp),
                    "half_life": mpe.half_life,
                    "valid_from": _dt_to_str(mpe.valid_from),
                    "valid_until": _dt_to_str(mpe.valid_until),
                }
                for tgt, mpe in self._multi_parent_multinomial_edges.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLNetwork:
        """Deserialize a network from a plain dictionary.

        Inverse of ``to_dict()``.  Nodes are added first (in sorted
        order for determinism), then edges.

        Args:
            data: Dictionary with 'name', 'nodes', 'edges', and
                  optionally 'multi_parent_edges' keys.

        Returns:
            A reconstructed SLNetwork.

        Raises:
            KeyError: If required keys are missing.
            ValueError: If the data is structurally invalid.
        """
        net = cls(name=data.get("name"))

        # ── Reconstruct nodes ──
        for nid, ndata in sorted(data.get("nodes", {}).items()):
            op_data = ndata["opinion"]
            opinion = Opinion.from_jsonld(op_data)
            mop_data = ndata.get("multinomial_opinion")
            mop = (
                MultinomialOpinion.from_dict(mop_data)
                if mop_data is not None
                else None
            )
            node = SLNode(
                node_id=ndata["node_id"],
                opinion=opinion,
                node_type=ndata.get("node_type", "content"),
                label=ndata.get("label"),
                metadata=ndata.get("metadata", {}),
                timestamp=_str_to_dt(ndata.get("timestamp")),
                half_life=ndata.get("half_life"),
                multinomial_opinion=mop,
            )
            net.add_node(node)

        # ── Reconstruct simple edges ──
        for _key, edata in data.get("edges", {}).items():
            cond = Opinion.from_jsonld(edata["conditional"])
            cf = (
                Opinion.from_jsonld(edata["counterfactual"])
                if edata.get("counterfactual") is not None
                else None
            )
            edge = SLEdge(
                source_id=edata["source_id"],
                target_id=edata["target_id"],
                conditional=cond,
                counterfactual=cf,
                edge_type=edata.get("edge_type", "deduction"),
                metadata=edata.get("metadata", {}),
                timestamp=_str_to_dt(edata.get("timestamp")),
                half_life=edata.get("half_life"),
                valid_from=_str_to_dt(edata.get("valid_from")),
                valid_until=_str_to_dt(edata.get("valid_until")),
            )
            net.add_edge(edge)

        # ── Reconstruct multi-parent edges ──
        for _tgt, mpe_data in data.get("multi_parent_edges", {}).items():
            parent_ids = tuple(mpe_data["parent_ids"])
            k = len(parent_ids)
            conditionals: dict[tuple[bool, ...], Opinion] = {}
            for key_str, op_data in mpe_data["conditionals"].items():
                # Parse stringified tuple key: "(True, False)" → (True, False)
                bool_key = _parse_bool_tuple(key_str, k)
                conditionals[bool_key] = Opinion.from_jsonld(op_data)
            mpe = MultiParentEdge(
                target_id=mpe_data["target_id"],
                parent_ids=parent_ids,
                conditionals=conditionals,
                edge_type=mpe_data.get("edge_type", "deduction"),
            )
            net.add_edge(mpe)

        # ── Reconstruct trust edges (Tier 2) ──
        for _key, tedata in data.get("trust_edges", {}).items():
            trust_op = Opinion.from_jsonld(tedata["trust_opinion"])
            te = TrustEdge(
                source_id=tedata["source_id"],
                target_id=tedata["target_id"],
                trust_opinion=trust_op,
                edge_type=tedata.get("edge_type", "trust"),
                metadata=tedata.get("metadata", {}),
                timestamp=_str_to_dt(tedata.get("timestamp")),
                half_life=tedata.get("half_life"),
            )
            net.add_trust_edge(te)

        # ── Reconstruct attestation edges (Tier 2) ──
        for _key, aedata in data.get("attestation_edges", {}).items():
            att_op = Opinion.from_jsonld(aedata["opinion"])
            ae = AttestationEdge(
                agent_id=aedata["agent_id"],
                content_id=aedata["content_id"],
                opinion=att_op,
                edge_type=aedata.get("edge_type", "attestation"),
                metadata=aedata.get("metadata", {}),
                timestamp=_str_to_dt(aedata.get("timestamp")),
                half_life=aedata.get("half_life"),
            )
            net.add_attestation(ae)

        # ── Reconstruct multinomial edges ──
        for _key, medata in data.get("multinomial_edges", {}).items():
            conditionals: dict[str, MultinomialOpinion] = {}
            for state, mop_data in medata["conditionals"].items():
                conditionals[state] = MultinomialOpinion.from_dict(mop_data)
            me = MultinomialEdge(
                source_id=medata["source_id"],
                target_id=medata["target_id"],
                conditionals=conditionals,
                edge_type=medata.get("edge_type", "deduction"),
                metadata=medata.get("metadata", {}),
                timestamp=_str_to_dt(medata.get("timestamp")),
                half_life=medata.get("half_life"),
                valid_from=_str_to_dt(medata.get("valid_from")),
                valid_until=_str_to_dt(medata.get("valid_until")),
            )
            net.add_edge(me)

        # ── Reconstruct multi-parent multinomial edges ──
        for _tgt, mpe_data in data.get(
            "multi_parent_multinomial_edges", {}
        ).items():
            parent_ids = tuple(mpe_data["parent_ids"])
            conditionals_m: dict[tuple[str, ...], MultinomialOpinion] = {}
            for key_str, mop_data in mpe_data["conditionals"].items():
                # Parse stringified tuple key: "('a', 'b')" → ("a", "b")
                str_key = _parse_str_tuple(key_str, len(parent_ids))
                conditionals_m[str_key] = MultinomialOpinion.from_dict(
                    mop_data
                )
            mpe = MultiParentMultinomialEdge(
                parent_ids=parent_ids,
                target_id=mpe_data["target_id"],
                conditionals=conditionals_m,
                edge_type=mpe_data.get("edge_type", "deduction"),
                metadata=mpe_data.get("metadata", {}),
                timestamp=_str_to_dt(mpe_data.get("timestamp")),
                half_life=mpe_data.get("half_life"),
                valid_from=_str_to_dt(mpe_data.get("valid_from")),
                valid_until=_str_to_dt(mpe_data.get("valid_until")),
            )
            net.add_edge(mpe)

        return net

    def to_jsonld(self) -> dict[str, Any]:
        """Serialize the network to a JSON-LD compatible document.

        Produces a JSON-LD document with a ``@context`` defining the
        SLNetwork vocabulary, and ``@graph`` containing typed nodes
        and edges.

        Returns:
            A JSON-LD document as a dictionary.
        """
        context = {
            "slnet": "https://jsonld-ex.github.io/ns/sl-network#",
            "@vocab": "https://jsonld-ex.github.io/ns/sl-network#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "nodes": {"@id": "slnet:nodes", "@container": "@index"},
            "edges": {"@id": "slnet:edges", "@container": "@set"},
            "opinion": {"@id": "slnet:opinion"},
            "conditional": {"@id": "slnet:conditional"},
            "counterfactual": {"@id": "slnet:counterfactual"},
        }

        graph_nodes = []
        for nid, n in self._nodes.items():
            node_dict: dict[str, Any] = {
                "@id": f"slnet:node/{nid}",
                "@type": "SLNode",
                "nodeId": n.node_id,
                "opinion": n.opinion.to_jsonld(),
                "nodeType": n.node_type,
                "label": n.label,
                "metadata": n.metadata if n.metadata else None,
                "timestamp": _dt_to_str(n.timestamp),
                "halfLife": n.half_life,
            }
            if n.multinomial_opinion is not None:
                node_dict["multinomialOpinion"] = (
                    n.multinomial_opinion.to_jsonld()
                )
            graph_nodes.append(node_dict)

        graph_edges = []
        for (src, tgt), e in self._edges.items():
            graph_edges.append({
                "@type": "SLEdge",
                "sourceId": e.source_id,
                "targetId": e.target_id,
                "conditional": e.conditional.to_jsonld(),
                "counterfactual": (
                    e.counterfactual.to_jsonld()
                    if e.counterfactual is not None
                    else None
                ),
                "edgeType": e.edge_type,
                "metadata": e.metadata if e.metadata else None,
                "timestamp": _dt_to_str(e.timestamp),
                "halfLife": e.half_life,
                "validFrom": _dt_to_str(e.valid_from),
                "validUntil": _dt_to_str(e.valid_until),
            })

        graph_mpes = []
        for tgt, mpe in self._multi_parent_edges.items():
            graph_mpes.append({
                "@type": "MultiParentEdge",
                "targetId": mpe.target_id,
                "parentIds": list(mpe.parent_ids),
                "conditionals": {
                    str(k): v.to_jsonld()
                    for k, v in mpe.conditionals.items()
                },
                "edgeType": mpe.edge_type,
            })

        graph_trust_edges = []
        for (src, tgt), te in self._trust_edges.items():
            graph_trust_edges.append({
                "@type": "TrustEdge",
                "sourceId": te.source_id,
                "targetId": te.target_id,
                "trustOpinion": te.trust_opinion.to_jsonld(),
                "edgeType": te.edge_type,
                "metadata": te.metadata if te.metadata else None,
                "timestamp": _dt_to_str(te.timestamp),
                "halfLife": te.half_life,
            })

        graph_attestations = []
        for (aid, cid), ae in self._attestation_edges.items():
            graph_attestations.append({
                "@type": "AttestationEdge",
                "agentId": ae.agent_id,
                "contentId": ae.content_id,
                "opinion": ae.opinion.to_jsonld(),
                "edgeType": ae.edge_type,
                "metadata": ae.metadata if ae.metadata else None,
                "timestamp": _dt_to_str(ae.timestamp),
                "halfLife": ae.half_life,
            })

        graph_multinomial_edges = []
        for (src, tgt), me in self._multinomial_edges.items():
            graph_multinomial_edges.append({
                "@type": "MultinomialEdge",
                "sourceId": me.source_id,
                "targetId": me.target_id,
                "conditionals": {
                    state: mop.to_jsonld()
                    for state, mop in me.conditionals.items()
                },
                "edgeType": me.edge_type,
                "metadata": me.metadata if me.metadata else None,
                "timestamp": _dt_to_str(me.timestamp),
                "halfLife": me.half_life,
                "validFrom": _dt_to_str(me.valid_from),
                "validUntil": _dt_to_str(me.valid_until),
            })

        graph_multi_parent_multinomial_edges = []
        for tgt, mpe in self._multi_parent_multinomial_edges.items():
            graph_multi_parent_multinomial_edges.append({
                "@type": "MultiParentMultinomialEdge",
                "targetId": mpe.target_id,
                "parentIds": list(mpe.parent_ids),
                "conditionals": {
                    str(k): v.to_jsonld()
                    for k, v in mpe.conditionals.items()
                },
                "edgeType": mpe.edge_type,
                "metadata": mpe.metadata if mpe.metadata else None,
                "timestamp": _dt_to_str(mpe.timestamp),
                "halfLife": mpe.half_life,
                "validFrom": _dt_to_str(mpe.valid_from),
                "validUntil": _dt_to_str(mpe.valid_until),
            })

        return {
            "@context": context,
            "@type": "SLNetwork",
            "name": self._name,
            "nodes": graph_nodes,
            "edges": graph_edges,
            "multiParentEdges": graph_mpes if graph_mpes else None,
            "trustEdges": graph_trust_edges if graph_trust_edges else None,
            "attestations": graph_attestations if graph_attestations else None,
            "multinomialEdges": (
                graph_multinomial_edges
                if graph_multinomial_edges else None
            ),
            "multiParentMultinomialEdges": (
                graph_multi_parent_multinomial_edges
                if graph_multi_parent_multinomial_edges else None
            ),
        }

    @classmethod
    def from_jsonld(cls, data: dict[str, Any]) -> SLNetwork:
        """Deserialize a network from a JSON-LD document.

        Inverse of ``to_jsonld()``.  Ignores ``@context`` — only
        reads the structural content.

        Args:
            data: A JSON-LD document as produced by ``to_jsonld()``.

        Returns:
            A reconstructed SLNetwork.
        """
        net = cls(name=data.get("name"))

        # ── Reconstruct nodes ──
        for ndata in data.get("nodes", []):
            opinion = Opinion.from_jsonld(ndata["opinion"])
            mop_data = ndata.get("multinomialOpinion")
            mop = (
                MultinomialOpinion.from_jsonld(mop_data)
                if mop_data is not None
                else None
            )
            node = SLNode(
                node_id=ndata["nodeId"],
                opinion=opinion,
                node_type=ndata.get("nodeType", "content"),
                label=ndata.get("label"),
                metadata=ndata.get("metadata") or {},
                timestamp=_str_to_dt(ndata.get("timestamp")),
                half_life=ndata.get("halfLife"),
                multinomial_opinion=mop,
            )
            net.add_node(node)

        # ── Reconstruct edges ──
        for edata in data.get("edges", []):
            cond = Opinion.from_jsonld(edata["conditional"])
            cf = (
                Opinion.from_jsonld(edata["counterfactual"])
                if edata.get("counterfactual") is not None
                else None
            )
            edge = SLEdge(
                source_id=edata["sourceId"],
                target_id=edata["targetId"],
                conditional=cond,
                counterfactual=cf,
                edge_type=edata.get("edgeType", "deduction"),
                metadata=edata.get("metadata") or {},
                timestamp=_str_to_dt(edata.get("timestamp")),
                half_life=edata.get("halfLife"),
                valid_from=_str_to_dt(edata.get("validFrom")),
                valid_until=_str_to_dt(edata.get("validUntil")),
            )
            net.add_edge(edge)

        # ── Reconstruct multi-parent edges ──
        for mpe_data in data.get("multiParentEdges", []) or []:
            parent_ids = tuple(mpe_data["parentIds"])
            k = len(parent_ids)
            conditionals: dict[tuple[bool, ...], Opinion] = {}
            for key_str, op_data in mpe_data["conditionals"].items():
                bool_key = _parse_bool_tuple(key_str, k)
                conditionals[bool_key] = Opinion.from_jsonld(op_data)
            mpe = MultiParentEdge(
                target_id=mpe_data["targetId"],
                parent_ids=parent_ids,
                conditionals=conditionals,
                edge_type=mpe_data.get("edgeType", "deduction"),
            )
            net.add_edge(mpe)

        # ── Reconstruct trust edges (Tier 2) ──
        for tedata in data.get("trustEdges", []) or []:
            trust_op = Opinion.from_jsonld(tedata["trustOpinion"])
            te = TrustEdge(
                source_id=tedata["sourceId"],
                target_id=tedata["targetId"],
                trust_opinion=trust_op,
                edge_type=tedata.get("edgeType", "trust"),
                metadata=tedata.get("metadata") or {},
                timestamp=_str_to_dt(tedata.get("timestamp")),
                half_life=tedata.get("halfLife"),
            )
            net.add_trust_edge(te)

        # ── Reconstruct attestation edges (Tier 2) ──
        for aedata in data.get("attestations", []) or []:
            att_op = Opinion.from_jsonld(aedata["opinion"])
            ae = AttestationEdge(
                agent_id=aedata["agentId"],
                content_id=aedata["contentId"],
                opinion=att_op,
                edge_type=aedata.get("edgeType", "attestation"),
                metadata=aedata.get("metadata") or {},
                timestamp=_str_to_dt(aedata.get("timestamp")),
                half_life=aedata.get("halfLife"),
            )
            net.add_attestation(ae)

        # ── Reconstruct multinomial edges ──
        for medata in data.get("multinomialEdges", []) or []:
            conditionals_me: dict[str, MultinomialOpinion] = {}
            for state, mop_data in medata["conditionals"].items():
                conditionals_me[state] = MultinomialOpinion.from_jsonld(
                    mop_data
                )
            me = MultinomialEdge(
                source_id=medata["sourceId"],
                target_id=medata["targetId"],
                conditionals=conditionals_me,
                edge_type=medata.get("edgeType", "deduction"),
                metadata=medata.get("metadata") or {},
                timestamp=_str_to_dt(medata.get("timestamp")),
                half_life=medata.get("halfLife"),
                valid_from=_str_to_dt(medata.get("validFrom")),
                valid_until=_str_to_dt(medata.get("validUntil")),
            )
            net.add_edge(me)

        # ── Reconstruct multi-parent multinomial edges ──
        for mpe_data in data.get(
            "multiParentMultinomialEdges", []
        ) or []:
            parent_ids = tuple(mpe_data["parentIds"])
            conditionals_mpm: dict[tuple[str, ...], MultinomialOpinion] = {}
            for key_str, mop_data in mpe_data["conditionals"].items():
                str_key = _parse_str_tuple(key_str, len(parent_ids))
                conditionals_mpm[str_key] = MultinomialOpinion.from_jsonld(
                    mop_data
                )
            mpe = MultiParentMultinomialEdge(
                parent_ids=parent_ids,
                target_id=mpe_data["targetId"],
                conditionals=conditionals_mpm,
                edge_type=mpe_data.get("edgeType", "deduction"),
                metadata=mpe_data.get("metadata") or {},
                timestamp=_str_to_dt(mpe_data.get("timestamp")),
                half_life=mpe_data.get("halfLife"),
                valid_from=_str_to_dt(mpe_data.get("validFrom")),
                valid_until=_str_to_dt(mpe_data.get("validUntil")),
            )
            net.add_edge(mpe)

        return net

    # ── Representation ─────────────────────────────────────────────

    def __repr__(self) -> str:
        name_str = f" {self._name!r}" if self._name else ""
        return (
            f"SLNetwork{name_str}("
            f"nodes={len(self._nodes)}, "
            f"edges={self.edge_count()})"
        )

    def __len__(self) -> int:
        """Return the number of nodes in the network."""
        return len(self._nodes)
