"""
Compliance Protocol Adapters — bridging algebra to data infrastructure.

The compliance algebra (§5–§9) defines operators for erasure propagation
and review-due triggers that require data from organizational systems
(lineage graphs, review schedules) rather than from any single standard
like FHIR. This module provides:

1. **Typed protocols** — minimal interfaces that organizations implement
   to connect their infrastructure (Apache Atlas, OpenLineage, custom DBs).

2. **Composition functions** — gather data from protocol implementations
   and delegate to the pure algebra operators.

3. **Reference implementations** — zero-dependency in-memory implementations
   for testing, experiments, and quick starts.

Architecture:
    ┌──────────────────────────────────────────────┐
    │        Compliance Algebra (pure math)          │
    │  erasure_scope_opinion · residual_contamination│
    │  review_due_trigger                            │
    └───────────────┬──────────────┬────────────────┘
                    │              │
         ┌──────────▼──────┐  ┌───▼──────────────────┐
         │  Composition     │  │  Composition          │
         │  Functions       │  │  Functions             │
         │  erasure_scope_  │  │  review_due_           │
         │  assessment()    │  │  assessment()          │
         │  contamination_  │  │                        │
         │  risk()          │  │                        │
         └──────────┬───────┘  └───┬────────────────────┘
                    │              │
              ┌─────▼─────┐  ┌────▼──────────┐
              │ Lineage    │  │ ReviewSchedule│
              │ Provider   │  │ Provider      │
              │ (Protocol) │  │ (Protocol)    │
              └─────┬──────┘  └────┬──────────┘
                    │              │
              ┌─────▼──────┐ ┌────▼──────────┐
              │ Simple     │ │ Simple        │
              │ LineageGraph│ │ ReviewSchedule│
              │ (reference)│ │ (reference)   │
              └────────────┘ └───────────────┘

Mathematical source of truth: compliance_algebra.md (§8.2, §9)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Protocol, Set, runtime_checkable

from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    erasure_scope_opinion,
    residual_contamination,
    review_due_trigger,
)


# ═══════════════════════════════════════════════════════════════════
# PROTOCOLS
# ═══════════════════════════════════════════════════════════════════


@runtime_checkable
class LineageProvider(Protocol):
    """Protocol for data lineage graph access.

    Per §9.1 (Definition 15): A directed acyclic graph G = (V, E)
    where V = {D₁, …, Dₙ} are datasets and (Dᵢ, Dⱼ) ∈ E means
    Dⱼ was derived from Dᵢ.

    Organizations implement this to connect their lineage infrastructure
    (Apache Atlas, Microsoft Purview, OpenLineage, custom systems).
    """

    def get_descendants(self, node_id: str) -> Set[str]:
        """Return all transitive descendants of node_id.

        Per Definition 15: descendants(Dₛ) in the lineage DAG.
        """
        ...

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Return all transitive ancestors of node_id.

        Used by contamination_risk() to compute A_j⁺.
        """
        ...

    def get_erasure_opinion(self, node_id: str) -> ComplianceOpinion:
        """Return the erasure completeness opinion for a node.

        Per Definition 16 (§9.2): ω_e^i = (eᵢ, ēᵢ, u_e^i, a_e^i)
        where eᵢ is belief that personal data has been completely
        erased from Dᵢ.

        Should return a vacuous opinion for nodes with no erasure
        evidence (epistemic default: no evidence ≠ compliance).
        """
        ...

    def get_exempt_nodes(self) -> Set[str]:
        """Return node IDs exempt from erasure under Art. 17(3).

        Art. 17(3) exceptions:
            (a) freedom of expression
            (b) legal obligation
            (c) public interest in health
            (d) archiving in public interest
            (e) legal claims
        """
        ...


@runtime_checkable
class ReviewScheduleProvider(Protocol):
    """Protocol for compliance review schedule access.

    Per §8.2 (Definition 13): mandatory reviews (Art. 45(3) adequacy,
    Art. 35(11) DPIA) that, when missed, accelerate uncertainty growth.

    Organizations implement this to connect their GRC (Governance,
    Risk, Compliance) systems.
    """

    def get_review_due(self, assessment_id: str) -> Optional[float]:
        """Return the review-due time for an assessment.

        Returns None if no review is scheduled.
        Time is in the same units as the algebra (typically days
        since epoch or ordinal).
        """
        ...

    def get_half_life(self, assessment_id: str) -> float:
        """Return the normal decay half-life for an assessment.

        Illustrative values from §8:
            Adequacy decision: ~730 days (2 years)
            DPIA validity: ~365 days (1 year)
            Consent freshness: ~365–730 days
            Technical measures: ~182 days (6 months)
        """
        ...

    def get_accelerated_half_life(self, assessment_id: str) -> float:
        """Return the accelerated half-life for post-trigger decay.

        Should be shorter than get_half_life() to model the
        faster uncertainty growth after a missed review.
        """
        ...


# ═══════════════════════════════════════════════════════════════════
# REFERENCE IMPLEMENTATION: SimpleLineageGraph
# ═══════════════════════════════════════════════════════════════════


class SimpleLineageGraph:
    """In-memory DAG for data lineage — reference implementation.

    Zero external dependencies. Suitable for testing, experiments,
    and quick prototyping. For production use, implement LineageProvider
    over your actual lineage infrastructure.

    The graph is stored as adjacency lists (children and parents).
    Transitive closures are computed on demand via BFS.
    """

    def __init__(self) -> None:
        self._children: Dict[str, Set[str]] = {}
        self._parents: Dict[str, Set[str]] = {}
        self._erasure_opinions: Dict[str, ComplianceOpinion] = {}
        self._exempt: Dict[str, str] = {}  # node_id → reason

    def add_edge(self, parent: str, child: str) -> None:
        """Add a derivation edge: child was derived from parent."""
        self._children.setdefault(parent, set()).add(child)
        self._parents.setdefault(child, set()).add(parent)
        # Ensure both nodes exist in both dicts
        self._children.setdefault(child, set())
        self._parents.setdefault(parent, set())

    def get_descendants(self, node_id: str) -> Set[str]:
        """Return all transitive descendants via BFS."""
        visited: Set[str] = set()
        queue = deque(self._children.get(node_id, set()))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(
                    c for c in self._children.get(node, set())
                    if c not in visited
                )
        return visited

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Return all transitive ancestors via BFS."""
        visited: Set[str] = set()
        queue = deque(self._parents.get(node_id, set()))
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(
                    p for p in self._parents.get(node, set())
                    if p not in visited
                )
        return visited

    def set_erasure_opinion(
        self, node_id: str, opinion: ComplianceOpinion,
    ) -> None:
        """Set the erasure completeness opinion for a node."""
        self._erasure_opinions[node_id] = opinion

    def get_erasure_opinion(self, node_id: str) -> ComplianceOpinion:
        """Return erasure opinion, defaulting to vacuous if unknown.

        Vacuous = (0, 0, 1, 0.5): no evidence of erasure or persistence.
        This is the epistemically correct default.
        """
        return self._erasure_opinions.get(
            node_id,
            ComplianceOpinion.create(0.0, 0.0, 1.0, 0.5),
        )

    def add_exempt(self, node_id: str, reason: str) -> None:
        """Mark a node as exempt from erasure under Art. 17(3)."""
        self._exempt[node_id] = reason

    def get_exempt_nodes(self) -> Set[str]:
        """Return all exempt node IDs."""
        return set(self._exempt.keys())

    def get_scope(self, source_id: str) -> Set[str]:
        """Compute erasure scope: {source} ∪ descendants \\ exempt.

        Per Definition 15 (§9.1):
            Scope(R) = {Dₛ} ∪ descendants(Dₛ)
            S = Scope(R) \\ Exempt(R)
        """
        full_scope = {source_id} | self.get_descendants(source_id)
        return full_scope - self.get_exempt_nodes()


# ═══════════════════════════════════════════════════════════════════
# REFERENCE IMPLEMENTATION: SimpleReviewSchedule
# ═══════════════════════════════════════════════════════════════════


class SimpleReviewSchedule:
    """In-memory review schedule — reference implementation.

    Zero external dependencies. Stores review-due dates and
    half-life parameters per assessment ID.

    Default accelerated_half_life = half_life / 4 (aggressive
    post-trigger decay). Override per assessment as needed.
    """

    def __init__(
        self,
        default_half_life: float = 365.0,
        default_accel_factor: float = 4.0,
    ) -> None:
        self._default_half_life = default_half_life
        self._default_accel_factor = default_accel_factor
        self._review_due: Dict[str, float] = {}
        self._half_lives: Dict[str, float] = {}
        self._accel_half_lives: Dict[str, float] = {}

    def set_review_due(
        self, assessment_id: str, due_time: float,
    ) -> None:
        """Set the review-due time for an assessment."""
        self._review_due[assessment_id] = due_time

    def get_review_due(self, assessment_id: str) -> Optional[float]:
        """Return review-due time, or None if not scheduled."""
        return self._review_due.get(assessment_id)

    def set_half_life(
        self, assessment_id: str, half_life: float,
    ) -> None:
        """Set a per-assessment normal half-life."""
        self._half_lives[assessment_id] = half_life

    def get_half_life(self, assessment_id: str) -> float:
        """Return half-life (per-assessment or default)."""
        return self._half_lives.get(
            assessment_id, self._default_half_life,
        )

    def set_accelerated_half_life(
        self, assessment_id: str, half_life: float,
    ) -> None:
        """Set a per-assessment accelerated half-life."""
        self._accel_half_lives[assessment_id] = half_life

    def get_accelerated_half_life(self, assessment_id: str) -> float:
        """Return accelerated half-life (per-assessment or derived).

        Default: half_life / accel_factor (typically /4).
        """
        if assessment_id in self._accel_half_lives:
            return self._accel_half_lives[assessment_id]
        return self.get_half_life(assessment_id) / self._default_accel_factor


# ═══════════════════════════════════════════════════════════════════
# COMPOSITION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def erasure_scope_assessment(
    source_id: str,
    lineage: LineageProvider,
) -> ComplianceOpinion:
    """Composite erasure completeness across lineage scope.

    Per §9.2 (Theorem 5):
        ω_E^R = J⊓(ω_e^i : D_i ∈ S)
    where S = Scope(R) \\ Exempt(R).

    Gathers erasure opinions from the lineage provider for all
    nodes in scope, then delegates to erasure_scope_opinion().

    Args:
        source_id: The source dataset targeted for erasure.
        lineage:   A LineageProvider implementation.

    Returns:
        ComplianceOpinion for composite erasure completeness.

    Raises:
        ValueError: If scope is empty (all nodes exempt).
    """
    # Compute scope: {source} ∪ descendants \\ exempt
    descendants = lineage.get_descendants(source_id)
    full_scope = {source_id} | descendants
    exempt = lineage.get_exempt_nodes()
    scope = full_scope - exempt

    if not scope:
        raise ValueError(
            f"Empty scope for erasure of '{source_id}': "
            f"all nodes are exempt. No erasure assessment possible."
        )

    # Gather per-node erasure opinions
    opinions = [lineage.get_erasure_opinion(node) for node in sorted(scope)]

    return erasure_scope_opinion(*opinions)


def contamination_risk(
    node_id: str,
    lineage: LineageProvider,
) -> ComplianceOpinion:
    """Residual contamination risk at a specific node.

    Per Definition 17 (§9.3):
        A_j⁺ = ancestors(D_j) ∩ S ∪ {D_j}

    Gathers erasure opinions for the node and all its ancestors,
    then delegates to residual_contamination().

    Note: ancestor filtering by scope/exemption is not applied here
    because contamination risk considers all actual data paths,
    regardless of erasure obligations. An exempt ancestor that
    retains data still contributes contamination risk.

    Args:
        node_id: The node to assess.
        lineage: A LineageProvider implementation.

    Returns:
        ComplianceOpinion where:
            lawfulness  = clean probability (r̄)
            violation   = contamination risk (r)
            uncertainty = residual uncertainty (u_r)
    """
    ancestors = lineage.get_ancestors(node_id)
    a_j_plus = sorted(ancestors | {node_id})

    opinions = [lineage.get_erasure_opinion(n) for n in a_j_plus]

    return residual_contamination(*opinions)


def review_due_assessment(
    opinion: ComplianceOpinion,
    assessment_id: str,
    assessment_time: float,
    schedule: ReviewScheduleProvider,
) -> ComplianceOpinion:
    """Bridge review schedule to review_due_trigger.

    Per Definition 13 (§8.2). Checks if a review is due and,
    if the assessment time is past the review-due date, applies
    accelerated decay toward vacuity.

    If no review is scheduled (get_review_due returns None),
    the opinion is returned unchanged.

    Args:
        opinion:         The compliance opinion to evaluate.
        assessment_id:   Identifier for the assessment/review item.
        assessment_time: Current assessment time.
        schedule:        A ReviewScheduleProvider implementation.

    Returns:
        ComplianceOpinion after applying review-due trigger (if applicable).
    """
    due_time = schedule.get_review_due(assessment_id)
    if due_time is None:
        return opinion if isinstance(opinion, ComplianceOpinion) else ComplianceOpinion.from_opinion(opinion)

    accel_hl = schedule.get_accelerated_half_life(assessment_id)

    return review_due_trigger(
        opinion=opinion,
        assessment_time=assessment_time,
        trigger_time=due_time,
        accelerated_half_life=accel_hl,
    )
