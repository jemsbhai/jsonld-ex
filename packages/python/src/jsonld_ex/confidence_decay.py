"""
Temporal Decay for Subjective Logic Opinions.

Models the natural degradation of confidence over time: as evidence
ages, it becomes less reliable.  The decay process migrates mass from
belief and disbelief into uncertainty, reflecting the epistemic reality
that old evidence is less trustworthy than fresh evidence.

Mathematical model:
    Given a decay factor λ ∈ [0, 1] (computed from elapsed time):
        b′ = λ · b
        d′ = λ · d
        u′ = 1 − b′ − d′

    Key properties preserved:
        - b′/d′ = b/d  (evidence direction unchanged)
        - b′ + d′ + u′ = 1  (valid opinion)
        - P(ω′) → a as λ → 0  (reverts to prior)

Extensibility:
    Users can provide any decay function matching the protocol::

        def my_decay(elapsed: float, half_life: float) -> float:
            '''Return a factor in [0, 1].'''
            ...

    Built-in functions:
        - exponential_decay: λ = 2^(−t/τ)  [default, smooth, standard]
        - linear_decay:      λ = max(0, 1 − t/(2τ))  [simple, bounded]
        - step_decay:        λ = 1 if t < τ else 0  [binary freshness]

Usage::

    from jsonld_ex.confidence_decay import decay_opinion, linear_decay

    # Default (exponential):
    fresh = decay_opinion(opinion, elapsed=3600, half_life=86400)

    # Custom:
    stale = decay_opinion(opinion, elapsed=3600, half_life=86400,
                          decay_fn=linear_decay)

References:
    Jøsang, A. (2016). Subjective Logic, §10.4 (Opinion Aging).
    The decay model here generalizes Jøsang's aging operator by
    accepting arbitrary decay functions while preserving the core
    invariants (additivity, ratio preservation, monotonicity).
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

from jsonld_ex.confidence_algebra import Opinion


# Type alias for decay functions.
# Protocol: (elapsed, half_life) -> factor in [0, 1]
DecayFunction = Callable[[float, float], float]


# ═══════════════════════════════════════════════════════════════════
# Built-in decay functions
# ═══════════════════════════════════════════════════════════════════


def exponential_decay(elapsed: float, half_life: float) -> float:
    """Exponential decay: λ = 2^(−t/τ).

    The standard radioactive-decay model.  Smooth, never reaches
    zero, asymptotically approaches it.

    Args:
        elapsed:   Time since the opinion was formed (≥ 0).
        half_life: Time for the factor to halve (> 0).

    Returns:
        Decay factor in (0, 1].
    """
    return 2.0 ** (-elapsed / half_life)


def linear_decay(elapsed: float, half_life: float) -> float:
    """Linear decay: λ = max(0, 1 − t/(2τ)).

    Reaches zero at t = 2·half_life, then stays at zero.
    Simpler to reason about than exponential; useful when
    a hard expiry is acceptable.

    Args:
        elapsed:   Time since the opinion was formed (≥ 0).
        half_life: Time for the factor to halve (> 0).
                   Full decay occurs at 2·half_life.

    Returns:
        Decay factor in [0, 1].
    """
    return max(0.0, 1.0 - elapsed / (2.0 * half_life))


def step_decay(elapsed: float, half_life: float) -> float:
    """Step decay: λ = 1 if t < τ, else 0.

    Binary freshness model: evidence is either fully fresh or
    completely stale.  Useful for hard TTL-style expiration.

    Args:
        elapsed:   Time since the opinion was formed (≥ 0).
        half_life: Threshold time (> 0).  At exactly this time,
                   evidence expires.

    Returns:
        1.0 if elapsed < half_life, else 0.0.
    """
    return 1.0 if elapsed < half_life else 0.0


# ═══════════════════════════════════════════════════════════════════
# Core decay operator
# ═══════════════════════════════════════════════════════════════════


def decay_opinion(
    opinion: Opinion,
    elapsed: float,
    half_life: float,
    decay_fn: DecayFunction | None = None,
) -> Opinion:
    """Apply temporal decay to an opinion.

    Migrates mass from belief and disbelief into uncertainty,
    modeling the loss of evidential relevance over time.

    The belief/disbelief ratio is preserved: we forget *how much*
    evidence we had, not *which direction* it pointed.

    Args:
        opinion:   The opinion to decay.
        elapsed:   Time elapsed since the opinion was formed.
                   Units are arbitrary but must be consistent
                   with half_life (e.g., both in seconds, hours,
                   or days).
        half_life: Time for belief and disbelief to halve.
                   Must be positive.
        decay_fn:  Custom decay function.  Signature:
                   ``(elapsed: float, half_life: float) -> float``
                   Must return a value in [0, 1].
                   Default: :func:`exponential_decay`.

    Returns:
        New Opinion with decayed belief/disbelief and increased
        uncertainty.  Base rate is preserved.

    Raises:
        ValueError: If elapsed < 0, half_life ≤ 0, or the decay
                    function returns a value outside [0, 1].

    Example::

        >>> o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        >>> decay_opinion(o, elapsed=10.0, half_life=10.0)
        Opinion(b=0.4000, d=0.0500, u=0.5500, a=0.5000)
    """
    # ── Input validation ───────────────────────────────────────
    if elapsed < 0:
        raise ValueError(
            f"elapsed must be non-negative, got: {elapsed}"
        )
    if half_life <= 0:
        raise ValueError(
            f"half_life must be positive, got: {half_life}"
        )

    # ── Compute decay factor ───────────────────────────────────
    fn = decay_fn if decay_fn is not None else exponential_decay
    factor = fn(elapsed, half_life)

    if not (0.0 <= factor <= 1.0):
        raise ValueError(
            f"decay factor must be in [0, 1], got: {factor} "
            f"from decay_fn({elapsed}, {half_life})"
        )

    # ── Apply decay ────────────────────────────────────────────
    new_b = factor * opinion.belief
    new_d = factor * opinion.disbelief
    new_u = 1.0 - new_b - new_d

    # Clamp to handle IEEE 754 floating-point artifacts
    # (e.g., 1.0 - 0.8 - 0.2 = -5.5e-17)
    if new_u < 0.0:
        new_u = 0.0

    return Opinion(
        belief=new_b,
        disbelief=new_d,
        uncertainty=new_u,
        base_rate=opinion.base_rate,
    )
