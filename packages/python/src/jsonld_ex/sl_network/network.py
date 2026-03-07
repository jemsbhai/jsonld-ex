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
from typing import Any, Dict, List, Optional, Union

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    InferenceResult,
    MultiParentEdge,
    SLEdge,
    SLNode,
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
        else:
            raise TypeError(
                f"Expected SLEdge or MultiParentEdge, "
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

        # Check for duplicate edge
        if (src, tgt) in self._edges:
            raise ValueError(
                f"Edge {src!r} → {tgt!r} already exists"
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

    def node_count(self) -> int:
        """Return the number of nodes in the network."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Return the number of edges in the network.

        Counts simple edges plus the individual parent→target edges
        contributed by multi-parent edges.
        """
        multi_edges = sum(
            len(mpe.parent_ids) for mpe in self._multi_parent_edges.values()
        )
        return len(self._edges) + multi_edges

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
        }

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
