"""
Data structures for the SLNetwork inference engine.

Defines the core types used throughout the SLNetwork module:
nodes, edges, multi-parent edges, and inference results.

All types are frozen (immutable) dataclasses.  SLNetwork composes
existing ``Opinion`` objects from ``jsonld_ex.confidence_algebra``
— it never reimplements opinion algebra.

Design Decisions:
    - Frozen dataclasses enforce immutability: graph mutations happen
      by adding/removing nodes and edges, not by mutating them in place.
    - Custom ``__hash__`` based on identity fields (node_id for nodes,
      (source_id, target_id) for edges) so that types remain hashable
      despite containing mutable ``dict`` metadata fields.
    - Validation in ``__post_init__`` catches construction errors early.
    - ``NodeType`` and ``EdgeType`` are Literal types, extensible across
      tiers: Tier 1 uses "content" nodes and "deduction" edges; Tier 2
      adds "agent" nodes, "trust" and "attestation" edges.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 7 (deduction),
    §10.2 (trust discount), §12.6 (deduction operator),
    §14.3 (transitive trust).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product as itertools_product
from typing import Any, Dict, List, Literal, Optional, Tuple

from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# TYPE ALIASES
# ═══════════════════════════════════════════════════════════════════

NodeType = Literal["content", "agent"]
"""Node classification.

- ``"content"``: A proposition whose truth value is uncertain (Tier 1).
- ``"agent"``: An entity that holds and reports opinions (Tier 2).
"""

EdgeType = Literal["deduction", "trust", "attestation"]
"""Edge classification.

- ``"deduction"``: Parent opinion conditionally implies child (Tier 1).
- ``"trust"``: Agent-to-agent trust relationship (Tier 2).
- ``"attestation"``: Agent attests to a content proposition (Tier 2).
"""


# ═══════════════════════════════════════════════════════════════════
# NODES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SLNode:
    """A node in an SL network — a proposition with a marginal opinion.

    For root nodes, ``opinion`` is the directly assigned marginal opinion.
    For non-root nodes, ``opinion`` serves as a prior that may be
    overridden by inference; the inferred opinion is computed from
    parent opinions via deduction.

    Attributes:
        node_id:   Unique identifier within the network.  Must be a
                   non-empty string.
        opinion:   Marginal SL opinion for this proposition.
        node_type: ``"content"`` (Tier 1) or ``"agent"`` (Tier 2).
        label:     Optional human-readable description.
        metadata:  Arbitrary key–value pairs for application-specific
                   data (provenance URIs, timestamps, etc.).

    Invariant:
        ``opinion.belief + opinion.disbelief + opinion.uncertainty == 1``
        (enforced by the ``Opinion`` constructor).
    """

    node_id: str
    opinion: Opinion
    node_type: NodeType = "content"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ── Validate node_id ──
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError(
                f"node_id must be a non-empty string, got {self.node_id!r}"
            )
        # ── Validate opinion ──
        if not isinstance(self.opinion, Opinion):
            raise TypeError(
                f"opinion must be an Opinion instance, "
                f"got {type(self.opinion).__name__}"
            )
        # ── Validate node_type ──
        if self.node_type not in ("content", "agent"):
            raise ValueError(
                f"node_type must be 'content' or 'agent', "
                f"got {self.node_type!r}"
            )

    def __hash__(self) -> int:
        """Hash by node_id — the unique identity of a node."""
        return hash(self.node_id)

    def __repr__(self) -> str:
        op = self.opinion
        return (
            f"SLNode(id={self.node_id!r}, "
            f"opinion=({op.belief:.3f},{op.disbelief:.3f},{op.uncertainty:.3f}), "
            f"type={self.node_type!r})"
        )


# ═══════════════════════════════════════════════════════════════════
# EDGES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SLEdge:
    """A directed deduction edge in an SL network.

    Represents a conditional relationship: "if the source proposition
    is true, the target proposition has opinion ``conditional``."

    The ``counterfactual`` is the opinion about the target given the
    source is *false*.  If ``None``, inference will compute it using
    a counterfactual strategy (vacuous, adversarial, or prior-based).

    Attributes:
        source_id:      Parent node (antecedent).
        target_id:      Child node (consequent).
        conditional:    ω_{target|source=True}.
        counterfactual: ω_{target|source=False}, or ``None`` to compute.
        edge_type:      ``"deduction"`` for content inference (Tier 1).
        metadata:       Arbitrary key–value pairs.

    Constraints:
        - ``source_id != target_id`` (no self-loops).
        - Both IDs must be non-empty strings.
    """

    source_id: str
    target_id: str
    conditional: Opinion
    counterfactual: Opinion | None = None
    edge_type: EdgeType = "deduction"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ── Validate IDs ──
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError(
                f"source_id must be a non-empty string, got {self.source_id!r}"
            )
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError(
                f"target_id must be a non-empty string, got {self.target_id!r}"
            )
        if self.source_id == self.target_id:
            raise ValueError(
                f"Self-loops are not allowed: "
                f"source_id == target_id == {self.source_id!r}"
            )
        # ── Validate conditional opinion ──
        if not isinstance(self.conditional, Opinion):
            raise TypeError(
                f"conditional must be an Opinion instance, "
                f"got {type(self.conditional).__name__}"
            )
        # ── Validate counterfactual if provided ──
        if self.counterfactual is not None and not isinstance(
            self.counterfactual, Opinion
        ):
            raise TypeError(
                f"counterfactual must be an Opinion or None, "
                f"got {type(self.counterfactual).__name__}"
            )
        # ── Validate edge_type ──
        if self.edge_type not in ("deduction", "trust", "attestation"):
            raise ValueError(
                f"edge_type must be 'deduction', 'trust', or 'attestation', "
                f"got {self.edge_type!r}"
            )

    def __hash__(self) -> int:
        """Hash by (source_id, target_id) — the identity of an edge."""
        return hash((self.source_id, self.target_id))

    def __repr__(self) -> str:
        c = self.conditional
        cf_str = "None" if self.counterfactual is None else "set"
        return (
            f"SLEdge({self.source_id!r}→{self.target_id!r}, "
            f"cond=({c.belief:.3f},{c.disbelief:.3f},{c.uncertainty:.3f}), "
            f"cf={cf_str})"
        )


# ═══════════════════════════════════════════════════════════════════
# MULTI-PARENT EDGES
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MultiParentEdge:
    """A conditional opinion table for nodes with two or more parents.

    For a node with *k* binary parents, the table maps each of the
    2^k parent-state configurations to a conditional opinion about
    the target node.

    Example (k=2 parents A, B):
        conditionals = {
            (True,  True):  Opinion(0.9, 0.05, 0.05),  # ω_{Y|A=T,B=T}
            (True,  False): Opinion(0.6, 0.2,  0.2),   # ω_{Y|A=T,B=F}
            (False, True):  Opinion(0.5, 0.3,  0.2),   # ω_{Y|A=F,B=T}
            (False, False): Opinion(0.1, 0.7,  0.2),   # ω_{Y|A=F,B=F}
        }

    Attributes:
        target_id:    The child node receiving conditional influence.
        parent_ids:   Ordered tuple of parent node IDs.  Order matches
                      the boolean tuple keys in ``conditionals``.
        conditionals: Mapping from parent-state configuration to the
                      conditional opinion about the target.  Must have
                      exactly 2^k entries covering all boolean combinations.
        edge_type:    Always ``"deduction"`` for content inference.

    Validation:
        - At least 2 parents (single-parent edges use ``SLEdge``).
        - Exactly 2^k entries in ``conditionals``.
        - Each key is a tuple of *k* booleans.
        - Each value is an ``Opinion`` instance.
        - All parent IDs are non-empty, distinct strings.
        - ``target_id`` is not in ``parent_ids`` (no self-loops).
    """

    target_id: str
    parent_ids: tuple[str, ...]
    conditionals: dict[tuple[bool, ...], Opinion]
    edge_type: EdgeType = "deduction"

    def __post_init__(self) -> None:
        # ── Validate target_id ──
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError(
                f"target_id must be a non-empty string, got {self.target_id!r}"
            )

        # ── Validate parent_ids ──
        if not isinstance(self.parent_ids, tuple):
            raise TypeError(
                f"parent_ids must be a tuple, "
                f"got {type(self.parent_ids).__name__}"
            )
        if len(self.parent_ids) < 2:
            raise ValueError(
                f"MultiParentEdge requires at least 2 parents, "
                f"got {len(self.parent_ids)}. "
                f"Use SLEdge for single-parent edges."
            )
        for pid in self.parent_ids:
            if not isinstance(pid, str) or not pid.strip():
                raise ValueError(
                    f"All parent_ids must be non-empty strings, "
                    f"got {pid!r}"
                )

        # ── Check for duplicate parent IDs ──
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError(
                f"parent_ids must be distinct, got {self.parent_ids!r}"
            )

        # ── No self-loop: target not in parents ──
        if self.target_id in self.parent_ids:
            raise ValueError(
                f"target_id {self.target_id!r} cannot be in parent_ids "
                f"(self-loop)"
            )

        # ── Validate conditionals completeness ──
        k = len(self.parent_ids)
        expected_count = 2 ** k
        if len(self.conditionals) != expected_count:
            raise ValueError(
                f"conditionals must have 2^{k} = {expected_count} entries "
                f"for {k} parents, got {len(self.conditionals)}"
            )

        # ── Validate each entry ──
        expected_keys = set(itertools_product([True, False], repeat=k))
        actual_keys = set(self.conditionals.keys())

        missing = expected_keys - actual_keys
        if missing:
            raise ValueError(
                f"conditionals is missing entries for: "
                f"{sorted(missing, key=str)}"
            )

        extra = actual_keys - expected_keys
        if extra:
            raise ValueError(
                f"conditionals has unexpected keys: "
                f"{sorted(extra, key=str)}"
            )

        for key, opinion in self.conditionals.items():
            if not isinstance(key, tuple) or len(key) != k:
                raise ValueError(
                    f"Each conditional key must be a {k}-tuple of bools, "
                    f"got {key!r}"
                )
            if not all(isinstance(v, bool) for v in key):
                raise TypeError(
                    f"Conditional key values must be bool, got {key!r}"
                )
            if not isinstance(opinion, Opinion):
                raise TypeError(
                    f"Conditional value for {key} must be an Opinion, "
                    f"got {type(opinion).__name__}"
                )

        # ── Validate edge_type ──
        if self.edge_type not in ("deduction", "trust", "attestation"):
            raise ValueError(
                f"edge_type must be 'deduction', 'trust', or 'attestation', "
                f"got {self.edge_type!r}"
            )

    def __hash__(self) -> int:
        """Hash by (target_id, parent_ids)."""
        return hash((self.target_id, self.parent_ids))

    def __repr__(self) -> str:
        k = len(self.parent_ids)
        parents = ", ".join(self.parent_ids)
        return (
            f"MultiParentEdge(target={self.target_id!r}, "
            f"parents=({parents}), entries=2^{k}={2**k})"
        )


# ═══════════════════════════════════════════════════════════════════
# INFERENCE RESULTS
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class InferenceStep:
    """A single step in an inference trace.

    Records one operation (deduce, fuse_parents, passthrough) performed
    during graph inference, with all inputs and the resulting opinion.
    This provides full auditability for every intermediate computation.

    Attributes:
        node_id:   The node this step computes an opinion for.
        operation: The operation name: ``"deduce"``, ``"fuse_parents"``,
                   ``"passthrough"``, ``"enumerate"``.
        inputs:    Named inputs to the operation (e.g., ``{"parent": ω,
                   "conditional": ω, "counterfactual": ω}``).
        result:    The resulting opinion after the operation.
    """

    node_id: str
    operation: str
    inputs: dict[str, Opinion]
    result: Opinion

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError(
                f"node_id must be a non-empty string, got {self.node_id!r}"
            )
        if not isinstance(self.operation, str) or not self.operation.strip():
            raise ValueError(
                f"operation must be a non-empty string, "
                f"got {self.operation!r}"
            )
        if not isinstance(self.result, Opinion):
            raise TypeError(
                f"result must be an Opinion, "
                f"got {type(self.result).__name__}"
            )
        # Validate all input values are Opinions
        for name, op in self.inputs.items():
            if not isinstance(op, Opinion):
                raise TypeError(
                    f"inputs[{name!r}] must be an Opinion, "
                    f"got {type(op).__name__}"
                )

    def __hash__(self) -> int:
        return hash((self.node_id, self.operation))

    def __repr__(self) -> str:
        r = self.result
        return (
            f"InferenceStep({self.node_id!r}, op={self.operation!r}, "
            f"result=({r.belief:.3f},{r.disbelief:.3f},{r.uncertainty:.3f}))"
        )


@dataclass(frozen=True)
class InferenceResult:
    """Complete result of running inference on a node.

    Contains the final opinion, the full inference trace, all
    intermediate per-node opinions, and the topological processing
    order used during inference.

    Attributes:
        query_node:            The node that was queried.
        opinion:               The final inferred opinion at query_node.
        steps:                 Ordered list of every inference step
                               (deduce/fuse/passthrough) performed.
        intermediate_opinions: Per-node inferred opinions for all nodes
                               in the connected component.
        topological_order:     The order in which nodes were processed
                               during inference.
    """

    query_node: str
    opinion: Opinion
    steps: list[InferenceStep]
    intermediate_opinions: dict[str, Opinion]
    topological_order: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.query_node, str) or not self.query_node.strip():
            raise ValueError(
                f"query_node must be a non-empty string, "
                f"got {self.query_node!r}"
            )
        if not isinstance(self.opinion, Opinion):
            raise TypeError(
                f"opinion must be an Opinion, "
                f"got {type(self.opinion).__name__}"
            )

    def __hash__(self) -> int:
        return hash(self.query_node)

    def __repr__(self) -> str:
        op = self.opinion
        return (
            f"InferenceResult(node={self.query_node!r}, "
            f"opinion=({op.belief:.3f},{op.disbelief:.3f},{op.uncertainty:.3f}), "
            f"steps={len(self.steps)})"
        )


# ═══════════════════════════════════════════════════════════════════
# TRUST EDGES (Tier 2)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrustEdge:
    """Trust relationship between two agents.

    Agent ``source_id`` trusts agent ``target_id`` with opinion
    ``trust_opinion``.  Used for transitive trust propagation
    (Jøsang 2016, §14.3).

    Attributes:
        source_id:      The trusting agent.
        target_id:      The trusted agent.
        trust_opinion:  ω_{source→target} — how much source trusts target.
        edge_type:      Always ``"trust"``.
        metadata:       Arbitrary key–value pairs.

    Constraints:
        - ``source_id != target_id`` (no self-trust loops).
        - Both IDs must be non-empty strings.
        - ``trust_opinion`` must be an ``Opinion`` instance.
    """

    source_id: str
    target_id: str
    trust_opinion: Opinion
    edge_type: EdgeType = "trust"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ── Validate source_id ──
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError(
                f"source_id must be a non-empty string, got {self.source_id!r}"
            )
        # ── Validate target_id ──
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError(
                f"target_id must be a non-empty string, got {self.target_id!r}"
            )
        # ── No self-loops ──
        if self.source_id == self.target_id:
            raise ValueError(
                f"Self-trust loops are not allowed: "
                f"source_id == target_id == {self.source_id!r}"
            )
        # ── Validate trust_opinion ──
        if not isinstance(self.trust_opinion, Opinion):
            raise TypeError(
                f"trust_opinion must be an Opinion instance, "
                f"got {type(self.trust_opinion).__name__}"
            )

    def __hash__(self) -> int:
        """Hash by (source_id, target_id) — the identity of a trust edge."""
        return hash((self.source_id, self.target_id))

    def __repr__(self) -> str:
        t = self.trust_opinion
        return (
            f"TrustEdge({self.source_id!r}→{self.target_id!r}, "
            f"trust=({t.belief:.3f},{t.disbelief:.3f},{t.uncertainty:.3f}))"
        )


# ═══════════════════════════════════════════════════════════════════
# ATTESTATION EDGES (Tier 2)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class AttestationEdge:
    """An agent attests to a content proposition.

    Agent ``agent_id`` holds opinion ``opinion`` about content node
    ``content_id``.  During trust-aware inference, the attestation
    is trust-discounted by the querying agent's derived trust in
    the attesting agent.

    Attributes:
        agent_id:   The attesting agent.
        content_id: The content node being attested.
        opinion:    Agent's opinion about the content proposition.
        edge_type:  Always ``"attestation"``.
        metadata:   Arbitrary key–value pairs.

    Constraints:
        - ``agent_id != content_id`` (no self-reference).
        - Both IDs must be non-empty strings.
        - ``opinion`` must be an ``Opinion`` instance.
    """

    agent_id: str
    content_id: str
    opinion: Opinion
    edge_type: EdgeType = "attestation"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ── Validate agent_id ──
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise ValueError(
                f"agent_id must be a non-empty string, got {self.agent_id!r}"
            )
        # ── Validate content_id ──
        if not isinstance(self.content_id, str) or not self.content_id.strip():
            raise ValueError(
                f"content_id must be a non-empty string, got {self.content_id!r}"
            )
        # ── No self-reference ──
        if self.agent_id == self.content_id:
            raise ValueError(
                f"Self-reference not allowed: "
                f"agent_id == content_id == {self.agent_id!r}"
            )
        # ── Validate opinion ──
        if not isinstance(self.opinion, Opinion):
            raise TypeError(
                f"opinion must be an Opinion instance, "
                f"got {type(self.opinion).__name__}"
            )

    def __hash__(self) -> int:
        """Hash by (agent_id, content_id) — the identity of an attestation."""
        return hash((self.agent_id, self.content_id))

    def __repr__(self) -> str:
        o = self.opinion
        return (
            f"AttestationEdge({self.agent_id!r}→{self.content_id!r}, "
            f"opinion=({o.belief:.3f},{o.disbelief:.3f},{o.uncertainty:.3f}))"
        )


# ═══════════════════════════════════════════════════════════════════
# TRUST PROPAGATION RESULTS (Tier 2)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TrustPropagationResult:
    """Result of transitive trust computation.

    For a querying agent *Q*, computes Q's derived trust in every
    reachable agent in the trust subgraph.  Produced by
    ``propagate_trust()``.

    Attributes:
        querying_agent: The agent whose perspective is being computed.
        derived_trusts: Mapping from agent_id to the transitive trust
                        opinion that the querying agent has in that agent.
        trust_paths:    Mapping from agent_id to the path (list of node
                        IDs) from the querying agent to that agent.
        steps:          Ordered list of every ``trust_discount()`` or
                        ``cumulative_fuse()`` call performed during
                        propagation.

    References:
        Jøsang, A. (2016). Subjective Logic, §14.3 (transitive trust),
        §14.5 (multi-path trust fusion).
    """

    querying_agent: str
    derived_trusts: dict[str, Opinion]
    trust_paths: dict[str, list[str]]
    steps: list[InferenceStep]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.querying_agent, str)
            or not self.querying_agent.strip()
        ):
            raise ValueError(
                f"querying_agent must be a non-empty string, "
                f"got {self.querying_agent!r}"
            )

    def __hash__(self) -> int:
        """Hash by querying_agent."""
        return hash(self.querying_agent)

    def __repr__(self) -> str:
        n = len(self.derived_trusts)
        return (
            f"TrustPropagationResult("
            f"agent={self.querying_agent!r}, "
            f"derived_trusts={n})"
        )
