"""
Enhanced Byzantine-Resistant Fusion for Subjective Logic Opinions.

Extends the basic ``robust_fuse`` in ``confidence_algebra`` with:

  - **Three removal strategies**: ``most_conflicting`` (pure discord),
    ``least_trusted`` (trust-weighted), ``combined`` (discord x distrust).
  - **Rich reporting**: every removal records the agent's original index,
    opinion, discord score, and a human-readable reason.
  - **Group cohesion metric**: a scalar summary of overall agreement.
  - **Full conflict matrix**: the nxn pairwise conflict matrix for
    downstream analysis or visualization.
  - **Pluggable distance metrics**: Euclidean (default), Manhattan,
    Jensen-Shannon divergence, and Hellinger distance on the opinion
    simplex, with support for user-defined metrics.

The original ``robust_fuse`` is NOT modified -- this module is purely
additive and can be used as a drop-in upgrade.

Usage::

    from jsonld_ex.confidence_byzantine import byzantine_fuse, ByzantineConfig

    # Default (most_conflicting, threshold=0.15):
    report = byzantine_fuse(opinions)

    # Trust-weighted:
    cfg = ByzantineConfig(
        strategy="least_trusted",
        trust_weights=[0.9, 0.9, 0.8, 0.1],
    )
    report = byzantine_fuse(opinions, config=cfg)

    # Cohesion with information-theoretic distance:
    from jsonld_ex.confidence_byzantine import cohesion_score, jsd_opinion_distance
    c = cohesion_score(opinions, distance_fn=jsd_opinion_distance)

References:
    Josang, A. (2016). Subjective Logic, S12.3.4 (Conflict),
    S14.5 (Trust networks with unreliable agents).
    Endres, D. M. & Schindelin, J. E. (2003). A new metric for
    probability distributions. IEEE Trans. Info. Theory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
)


# -------------------------------------------------------------------
# Distance metric type and constants
# -------------------------------------------------------------------

# Any callable (Opinion, Opinion) -> float in [0, 1] qualifies.
# Same pattern as DecayFunction in confidence_decay.
DistanceMetric = Callable[[Opinion, Opinion], float]

# Maximum Euclidean distance between two points on the 2-simplex
# {(b,d,u) : b+d+u=1, b,d,u >= 0}.  Achieved between vertices,
# e.g. (1,0,0) and (0,1,0): sqrt(1^2 + 1^2 + 0^2) = sqrt(2).
_MAX_SIMPLEX_DISTANCE = math.sqrt(2.0)

_LOG2 = math.log(2.0)


# -------------------------------------------------------------------
# Built-in distance metrics
# -------------------------------------------------------------------


def euclidean_opinion_distance(a: Opinion, b: Opinion) -> float:
    """Euclidean (L2) distance on the opinion simplex, normalized to [0, 1].

    Formula::

        d(A, B) = ||A - B||_2 / sqrt(2)

    Properties:
        - Uniform sensitivity across the simplex (a 0.1 shift in
          belief contributes the same distance whether belief is
          near 0 or near 1).
        - Simple, interpretable, well-understood.
        - Proper metric (non-negativity, identity, symmetry, triangle).

    Best for: general-purpose use where uniform sensitivity is acceptable.

    Args:
        a: First opinion.
        b: Second opinion.

    Returns:
        Distance in [0, 1].
    """
    raw = math.sqrt(
        (a.belief - b.belief) ** 2
        + (a.disbelief - b.disbelief) ** 2
        + (a.uncertainty - b.uncertainty) ** 2
    )
    return raw / _MAX_SIMPLEX_DISTANCE


def manhattan_opinion_distance(a: Opinion, b: Opinion) -> float:
    """Manhattan (L1) distance on the opinion simplex, normalized to [0, 1].

    Formula::

        d(A, B) = (|b_A - b_B| + |d_A - d_B| + |u_A - u_B|) / 2

    The maximum L1 distance on the simplex is 2 (between any two
    vertices, e.g. |1-0| + |0-1| + |0-0| = 2), hence the /2
    normalization.

    Properties:
        - More robust to single-dimension outliers than Euclidean.
        - Proper metric.

    Best for: robustness when one component dominates the difference.

    Args:
        a: First opinion.
        b: Second opinion.

    Returns:
        Distance in [0, 1].
    """
    raw = (
        abs(a.belief - b.belief)
        + abs(a.disbelief - b.disbelief)
        + abs(a.uncertainty - b.uncertainty)
    )
    return raw / 2.0


def _safe_xlogx(x: float) -> float:
    """Compute x * log2(x) with the convention 0 * log(0) = 0.

    This is the limiting value by L'Hopital's rule and is standard
    in information theory.
    """
    if x <= 0.0:
        return 0.0
    return x * math.log2(x)


def _kl_divergence_base2(p: tuple[float, ...], q: tuple[float, ...]) -> float:
    """KL(P || Q) in bits (log base 2).

    Assumes P and Q are valid distributions (sum to 1, non-negative).
    When p_i > 0 and q_i = 0, KL is infinite; on the simplex with
    M = (P+Q)/2, this cannot happen (M_i >= P_i/2 > 0 when P_i > 0).
    """
    total = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            if qi <= 0.0:
                return math.inf
            total += pi * math.log2(pi / qi)
    return total


def jsd_opinion_distance(a: Opinion, b: Opinion) -> float:
    """Jensen-Shannon divergence distance on the opinion simplex.

    Formula::

        M = (A + B) / 2
        JSD(A, B) = 0.5 * KL(A || M) + 0.5 * KL(B || M)
        d(A, B) = sqrt(JSD(A, B))

    Uses log base 2, so JSD is in [0, 1] and sqrt(JSD) is in [0, 1].

    sqrt(JSD) is a proper metric (Endres & Schindelin 2003,
    Osterreicher & Vajda 2003).

    Properties:
        - More sensitive near simplex boundaries than Euclidean.
          A shift from u=0.01 to u=0.1 is weighted more heavily
          than u=0.4 to u=0.49, because the information-theoretic
          cost of the shift is higher near certainty.
        - Well-defined when components are zero (0*log(0) = 0 by
          convention; M always has M_i >= max(A_i, B_i)/2 > 0
          when either A_i or B_i is positive).
        - Proper metric.

    Best for: information-theoretic rigor; emphasis on changes near
    certainty/ignorance boundaries.

    Args:
        a: First opinion.
        b: Second opinion.

    Returns:
        Distance in [0, 1].
    """
    pa = (a.belief, a.disbelief, a.uncertainty)
    pb = (b.belief, b.disbelief, b.uncertainty)
    m = tuple((ai + bi) / 2.0 for ai, bi in zip(pa, pb))

    jsd = 0.5 * _kl_divergence_base2(pa, m) + 0.5 * _kl_divergence_base2(pb, m)

    # Clamp for floating-point artifacts (JSD can be -1e-17 due to
    # cancellation when A == B)
    if jsd < 0.0:
        jsd = 0.0

    return math.sqrt(jsd)


def hellinger_opinion_distance(a: Opinion, b: Opinion) -> float:
    """Hellinger distance on the opinion simplex.

    Formula::

        H(A, B) = (1 / sqrt(2)) * ||sqrt(A) - sqrt(B)||_2

    Properties:
        - More sensitive near boundaries than Euclidean (because
          sqrt compresses differences near 1 and amplifies them
          near 0).
        - Numerically stable (no logarithms, no division-by-zero).
        - Known bound: H^2 <= JSD (base-2 log), so H <= sqrt(JSD).
        - Proper metric.

    Best for: boundary sensitivity similar to JSD but with simpler
    computation and better numerical stability.

    Args:
        a: First opinion.
        b: Second opinion.

    Returns:
        Distance in [0, 1].
    """
    raw = math.sqrt(
        (math.sqrt(a.belief) - math.sqrt(b.belief)) ** 2
        + (math.sqrt(a.disbelief) - math.sqrt(b.disbelief)) ** 2
        + (math.sqrt(a.uncertainty) - math.sqrt(b.uncertainty)) ** 2
    )
    return raw / _MAX_SIMPLEX_DISTANCE


# -------------------------------------------------------------------
# Types
# -------------------------------------------------------------------

ByzantineStrategy = Literal["most_conflicting", "least_trusted", "combined"]


@dataclass(frozen=True)
class ByzantineConfig:
    """Configuration for :func:`byzantine_fuse`.

    Attributes:
        threshold:     Discord score above which an agent may be removed.
                       Default 0.15.
        max_removals:  Maximum agents to remove.  ``None`` = ``floor(n/2)``
                       (never remove a majority).
        strategy:      Removal priority strategy.
        trust_weights: Per-agent trust scores in [0, 1].  Required for
                       ``"least_trusted"`` and ``"combined"`` strategies.
                       Length must match the number of opinions.
        min_agents:    Never reduce below this many agents.  Default 2.
    """

    threshold: float = 0.15
    max_removals: Optional[int] = None
    strategy: ByzantineStrategy = "most_conflicting"
    trust_weights: Optional[Sequence[float]] = None
    min_agents: int = 2


@dataclass(frozen=True)
class AgentRemoval:
    """Record of a single agent removed during Byzantine fusion.

    Attributes:
        index:         Position in the **original** opinion list.
        opinion:       The removed agent's opinion.
        discord_score: The agent's score at time of removal (meaning
                       depends on strategy).
        reason:        Human-readable explanation of why removed.
    """

    index: int
    opinion: Opinion
    discord_score: float
    reason: str


@dataclass(frozen=True)
class ByzantineFusionReport:
    """Complete report from :func:`byzantine_fuse`.

    Attributes:
        fused:             The final fused opinion after removals.
        removed:           Ordered list of agent removals.
        conflict_matrix:   nxn pairwise conflict matrix of the
                           **original** (pre-removal) opinion set.
        cohesion_score:    Group cohesion of the **surviving** agents.
                           1.0 = perfect agreement, 0.0 = total conflict.
        surviving_indices: Indices (into original list) of agents that
                           survived filtering.
    """

    fused: Opinion
    removed: list[AgentRemoval]
    conflict_matrix: list[list[float]]
    cohesion_score: float
    surviving_indices: list[int]


# -------------------------------------------------------------------
# High-level utility functions
# -------------------------------------------------------------------


def opinion_distance(
    a: Opinion,
    b: Opinion,
    distance_fn: Optional[DistanceMetric] = None,
) -> float:
    """Distance between two opinions on the simplex.

    Dispatches to the given distance function, defaulting to
    :func:`euclidean_opinion_distance`.

    This is fundamentally different from :func:`pairwise_conflict`
    (Josang 2016, S12.3.4), which measures evidential tension and
    can be non-zero for identical opinions.

    Note:
        Base rate is intentionally excluded from all distance metrics.
        Distance measures how the *evidence* differs (belief, disbelief,
        uncertainty), not the prior assumption.

    Args:
        a:           First opinion.
        b:           Second opinion.
        distance_fn: Distance metric to use.  ``None`` = Euclidean.
                     Must be a callable ``(Opinion, Opinion) -> float``
                     returning a value in [0, 1].

    Returns:
        Distance in [0, 1].
    """
    fn = distance_fn if distance_fn is not None else euclidean_opinion_distance
    return fn(a, b)


def build_conflict_matrix(opinions: Sequence[Opinion]) -> list[list[float]]:
    """Build the symmetric nxn pairwise conflict matrix.

    ``matrix[i][j]`` = :func:`pairwise_conflict(opinions[i], opinions[j])`.
    Diagonal is always 0.0.

    This uses Josang's pairwise_conflict (b_A * d_B + d_A * b_B),
    which measures *evidential tension* -- the degree to which one
    agent's belief overlaps with the other's disbelief.  This is the
    correct metric for Byzantine agent detection (identifying agents
    whose evidence opposes the group).

    Args:
        opinions: One or more opinions.

    Returns:
        nxn list-of-lists with float values in [0, 1].

    Raises:
        ValueError: If opinions is empty.
    """
    n = len(opinions)
    if n == 0:
        raise ValueError("build_conflict_matrix requires at least one opinion")

    mat: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_conflict(opinions[i], opinions[j])
            mat[i][j] = c
            mat[j][i] = c
    return mat


def cohesion_score(
    opinions: Sequence[Opinion],
    distance_fn: Optional[DistanceMetric] = None,
) -> float:
    """Group cohesion based on opinion distance in the simplex.

    Measures how much the agents *agree* -- i.e., how close their
    opinions are in the (b, d, u) simplex.

    Formula::

        cohesion = 1 - mean(distance(i, j))  for all pairs i < j

    The distance metric is pluggable.  By default, Euclidean distance
    is used.  Available built-in metrics:

    - :func:`euclidean_opinion_distance` -- uniform sensitivity,
      simple, interpretable.  Good general default.
    - :func:`manhattan_opinion_distance` -- more robust to single-
      dimension outliers.
    - :func:`jsd_opinion_distance` -- information-theoretic; more
      sensitive near simplex boundaries (near certainty/ignorance).
    - :func:`hellinger_opinion_distance` -- boundary-sensitive like
      JSD but simpler and numerically stabler.

    Any callable matching ``DistanceMetric`` can also be used.

    Note on metric choice:
        This function uses *opinion distance* (how much agents differ),
        NOT Josang's ``pairwise_conflict`` (evidential tension).
        The distinction is critical: ``pairwise_conflict`` is non-zero
        for identical opinions when both b > 0 and d > 0, making it
        unsuitable for cohesion measurement.

    Args:
        opinions:    One or more opinions.
        distance_fn: Distance metric.  ``None`` = Euclidean.

    Returns:
        Cohesion score in [0, 1].  1.0 = perfect agreement.

    Raises:
        ValueError: If opinions is empty.
    """
    n = len(opinions)
    if n == 0:
        raise ValueError("cohesion_score requires at least one opinion")
    if n == 1:
        return 1.0

    fn = distance_fn if distance_fn is not None else euclidean_opinion_distance

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += fn(opinions[i], opinions[j])
            pairs += 1

    mean_distance = total / pairs if pairs > 0 else 0.0
    return 1.0 - mean_distance


# -------------------------------------------------------------------
# Core: byzantine_fuse
# -------------------------------------------------------------------


def _compute_discord_scores(
    indexed: list[tuple[int, Opinion]],
) -> list[float]:
    """Mean pairwise conflict for each agent in the current group.

    Uses Josang's pairwise_conflict (evidential tension), which is the
    correct metric for identifying adversarial agents: an agent whose
    belief strongly overlaps with the group's disbelief has high discord.
    """
    n = len(indexed)
    discord = [0.0] * n
    for i in range(n):
        for j in range(i + 1, n):
            c = pairwise_conflict(indexed[i][1], indexed[j][1])
            discord[i] += c
            discord[j] += c
    for i in range(n):
        discord[i] /= (n - 1) if n > 1 else 1.0
    return discord


def _pick_most_conflicting(
    indexed: list[tuple[int, Opinion]],
    discord: list[float],
    _trust_weights: Optional[Sequence[float]],
) -> tuple[int, float, str]:
    """Select the agent with highest mean discord."""
    worst = max(range(len(indexed)), key=lambda k: discord[k])
    return worst, discord[worst], "highest discord in group"


def _pick_least_trusted(
    indexed: list[tuple[int, Opinion]],
    discord: list[float],
    trust_weights: Optional[Sequence[float]],
) -> tuple[int, float, str]:
    """Select the agent with lowest trust among those above threshold.

    Uses discord as tiebreaker -- among agents with similar trust,
    the most discordant is removed.
    """
    assert trust_weights is not None
    # Score: lower trust -> higher removal priority; discord as tiebreaker
    scores = [
        (-trust_weights[idx], discord[k])
        for k, (idx, _) in enumerate(indexed)
    ]
    worst = max(range(len(indexed)), key=lambda k: scores[k])
    orig_idx = indexed[worst][0]
    return (
        worst,
        discord[worst],
        f"lowest trust ({trust_weights[orig_idx]:.3f}) with discord {discord[worst]:.3f}",
    )


def _pick_combined(
    indexed: list[tuple[int, Opinion]],
    discord: list[float],
    trust_weights: Optional[Sequence[float]],
) -> tuple[int, float, str]:
    """Select agent with highest discord * (1 - trust).

    Combines both signals: high conflict + low trust = highest
    removal priority.
    """
    assert trust_weights is not None
    scores = [
        discord[k] * (1.0 - trust_weights[indexed[k][0]])
        for k in range(len(indexed))
    ]
    worst = max(range(len(indexed)), key=lambda k: scores[k])
    orig_idx = indexed[worst][0]
    return (
        worst,
        discord[worst],
        f"combined score {scores[worst]:.3f} "
        f"(discord={discord[worst]:.3f}, trust={trust_weights[orig_idx]:.3f})",
    )


_STRATEGY_PICKERS = {
    "most_conflicting": _pick_most_conflicting,
    "least_trusted": _pick_least_trusted,
    "combined": _pick_combined,
}


def byzantine_fuse(
    opinions: Sequence[Opinion],
    config: Optional[ByzantineConfig] = None,
) -> ByzantineFusionReport:
    """Enhanced Byzantine-resistant fusion with configurable strategies.

    Algorithm:
        1. Build the full pairwise conflict matrix (preserved in report).
        2. Iteratively remove agents according to the chosen strategy
           until the group is cohesive or limits are reached.
        3. Fuse surviving agents via :func:`cumulative_fuse`.
        4. Compute cohesion of the surviving group.

    Note on metrics:
        The *discord scores* used for agent removal are based on
        Josang's ``pairwise_conflict`` (evidential tension), which
        correctly identifies agents whose evidence opposes the group.

        The *cohesion score* in the report uses ``opinion_distance``
        (Euclidean distance on the simplex by default), which correctly
        measures how much the surviving agents agree (identical
        opinions = 1.0).

        These are deliberately different metrics for different purposes.
        See :func:`cohesion_score` docstring for the full rationale.

    Args:
        opinions: List of opinions from independent agents.
        config:   Fusion configuration.  ``None`` uses defaults.

    Returns:
        :class:`ByzantineFusionReport` with fused opinion, removal
        details, conflict matrix, cohesion score, and surviving indices.

    Raises:
        ValueError: If opinions is empty, or if trust_weights are
                    required but missing/wrong length.
    """
    if len(opinions) == 0:
        raise ValueError("byzantine_fuse requires at least one opinion")

    cfg = config or ByzantineConfig()

    # Validate trust_weights for strategies that need them
    if cfg.strategy in ("least_trusted", "combined"):
        if cfg.trust_weights is None:
            raise ValueError(
                f"trust_weights are required for strategy '{cfg.strategy}'"
            )
        if len(cfg.trust_weights) != len(opinions):
            raise ValueError(
                f"trust_weights length ({len(cfg.trust_weights)}) must match "
                f"opinions length ({len(opinions)})"
            )

    # Build original conflict matrix (always for the report)
    original_matrix = build_conflict_matrix(opinions)

    # Single opinion: short-circuit
    if len(opinions) == 1:
        return ByzantineFusionReport(
            fused=opinions[0],
            removed=[],
            conflict_matrix=original_matrix,
            cohesion_score=1.0,
            surviving_indices=[0],
        )

    # Resolve max_removals
    max_removals = cfg.max_removals
    if max_removals is None:
        max_removals = len(opinions) // 2

    picker = _STRATEGY_PICKERS[cfg.strategy]

    # Track: (original_index, opinion)
    indexed: list[tuple[int, Opinion]] = list(enumerate(opinions))
    removed: list[AgentRemoval] = []

    for _ in range(max_removals):
        n = len(indexed)
        if n <= cfg.min_agents:
            break

        discord = _compute_discord_scores(indexed)

        # Pick victim
        local_idx, score, reason = picker(indexed, discord, cfg.trust_weights)

        if discord[local_idx] < cfg.threshold:
            break  # Group is cohesive enough

        orig_idx, orig_opinion = indexed.pop(local_idx)
        removed.append(AgentRemoval(
            index=orig_idx,
            opinion=orig_opinion,
            discord_score=score,
            reason=reason,
        ))

    # Fuse survivors
    surviving = [op for _, op in indexed]
    fused = cumulative_fuse(*surviving)

    surviving_indices = [idx for idx, _ in indexed]
    final_cohesion = cohesion_score(surviving) if len(surviving) > 1 else 1.0

    return ByzantineFusionReport(
        fused=fused,
        removed=removed,
        conflict_matrix=original_matrix,
        cohesion_score=final_cohesion,
        surviving_indices=surviving_indices,
    )
