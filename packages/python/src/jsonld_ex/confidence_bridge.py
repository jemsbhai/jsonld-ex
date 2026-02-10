"""
Bridge between the scalar inference API and the formal confidence algebra.

This module demonstrates the relationship between existing methods in
``inference.py`` and the Subjective Logic operators, and provides
upgrade paths for users who want richer uncertainty semantics.

Proven exact equivalences:
    1. Multiply chain ≡ iterated trust discount with base_rate=0.
       Proof (induction on chain length n):
         Base case (n=1): trust = (c, 1−c, 0), assertion = (1, 0, 0, a=0).
           b = c·1 = c, d = c·0 = 0, u = (1−c) + 0 + c·0 = 1−c.
           P = c + 0·(1−c) = c.  ✓
         Inductive step: assume after k discounts b_k = ∏_{i=1}^{k} c_i.
           Discount by c_{k+1}: b_{k+1} = c_{k+1} · b_k = ∏_{i=1}^{k+1} c_i.
           P = b_{k+1} + 0·u_{k+1} = ∏c_i.  ✓
    2. Scalar average (n=2) ≡ averaging fusion of dogmatic opinions.
       For dogmatic opinions (u=0), the fallback formula reduces to
       (b_A + b_B) / 2, which is the arithmetic mean.  This holds
       only for n=2; pairwise left-fold for n>2 is NOT equivalent
       to the n-ary arithmetic mean.

Non-equivalences (important for scientific accuracy):
    3. Noisy-OR ≠ cumulative fusion.  Both combine independent
       sources, but noisy-OR computes 1 − ∏(1−p_i) while cumulative
       fusion operates on the (b, d, u) triple.  They produce
       different scalar values.  Neither subsumes the other.
    4. Dempster-Shafer combination ≠ any single Subjective Logic
       operator.  DS theory uses a different evidence model.

Key insight:
    The algebra is strictly more expressive than scalar confidence:
    it distinguishes epistemic states that identical scalars conflate
    (e.g., "strong evidence for 0.75" vs "no evidence, base rate 0.75").
    Scalar methods operate in a 1D subspace; the algebra adds the
    uncertainty dimension, enabling downstream agents to make
    calibrated accept/reject decisions.

Usage patterns::

    # Existing code (unchanged, still works):
    combine_sources([0.9, 0.7], "noisy_or")

    # New: get full Opinion with uncertainty metadata:
    combine_opinions_from_scalars([0.9, 0.7])

    # New: drop-in replacement for multiply chains:
    scalar_propagate_via_algebra([0.9, 0.8], "multiply")
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
)
from jsonld_ex.inference import (
    PropagationResult,
    ConflictReport,
    _validate_confidence,
    resolve_conflict,
)


# ═══════════════════════════════════════════════════════════════════
# Opinion-returning convenience functions
# ═══════════════════════════════════════════════════════════════════


def combine_opinions_from_scalars(
    scores: Sequence[float],
    uncertainty: float = 0.0,
    fusion: Literal["cumulative", "averaging"] = "cumulative",
    base_rate: float = 0.5,
) -> Opinion:
    """Combine scalar confidence scores via the formal algebra.

    Lifts scalar scores to Opinions, fuses them, and returns the
    full Opinion (preserving uncertainty metadata).

    Default mapping (uncertainty=0, fusion="cumulative"):
        Each score p is mapped to Opinion(b=p, d=0, u=1-p),
        treating "not confident" as "uncertain" rather than
        "disbelieving."  This is the natural interpretation for
        ML model outputs.

    Args:
        scores:      Confidence scores, each in [0, 1].
        uncertainty: Uncertainty to assign to each source.
                     When 0.0 (default) and fusion="cumulative",
                     scores are mapped as (b=p, d=0, u=1-p).
                     When > 0, mass is redistributed from
                     belief/disbelief to uncertainty.
        fusion:      "cumulative" (independent sources) or
                     "averaging" (correlated sources).
        base_rate:   Prior probability for each opinion.

    Returns:
        Fused Opinion with full uncertainty metadata.
    """
    if len(scores) == 0:
        raise ValueError("Scores must contain at least one value")
    for s in scores:
        _validate_confidence(s)

    if fusion == "cumulative" and uncertainty == 0.0:
        # Natural mapping: p → (b=p, d=0, u=1-p)
        opinions = [
            Opinion(belief=p, disbelief=0.0, uncertainty=1.0 - p, base_rate=base_rate)
            for p in scores
        ]
        return cumulative_fuse(*opinions)
    else:
        opinions = [
            Opinion.from_confidence(p, uncertainty=uncertainty, base_rate=base_rate)
            for p in scores
        ]
        if fusion == "cumulative":
            return cumulative_fuse(*opinions)
        else:
            return averaging_fuse(*opinions)


def propagate_opinions_from_scalars(
    chain: Sequence[float],
    trust_uncertainty: float = 0.0,
    base_rate: float = 0.0,
) -> Opinion:
    """Propagate confidence through a chain via trust discount.

    Lifts scalar chain scores to trust Opinions and applies
    iterated trust discount.

    With default parameters (trust_uncertainty=0, base_rate=0),
    this produces the same scalar as propagate_confidence('multiply')
    via to_confidence(), since P = b + 0·u = b = ∏c_i.

    Args:
        chain:             Confidence scores along the inference path.
        trust_uncertainty: Uncertainty in each trust link.
                           When 0.0 (default), produces dogmatic
                           trust opinions matching scalar multiply.
                           When > 0, accounts for epistemic
                           uncertainty in the trust relationship.
        base_rate:         Prior probability for the assertion.
                           Default 0.0 ensures P = b (exact multiply).

    Returns:
        Opinion representing the propagated confidence.
    """
    if len(chain) == 0:
        raise ValueError("Chain must contain at least one score")
    for s in chain:
        _validate_confidence(s)

    # The assertion being propagated: absolute belief
    current = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=base_rate)

    # Apply trust discounts in reverse order (innermost first)
    for c in reversed(chain):
        if trust_uncertainty == 0.0:
            trust_opinion = Opinion(
                belief=c, disbelief=1.0 - c, uncertainty=0.0
            )
        else:
            trust_opinion = Opinion.from_confidence(c, uncertainty=trust_uncertainty)
        current = trust_discount(trust_opinion, current)

    return current


def resolve_conflict_with_opinions(
    assertions: Sequence[dict[str, Any]],
    strategy: Literal["highest", "weighted_vote", "recency"] = "highest",
) -> ConflictReport:
    """Resolve conflicts and enrich the winner with Opinion metadata.

    Delegates to the existing resolve_conflict, then attaches a
    full Opinion to the winner for downstream uncertainty-aware
    processing.

    Args:
        assertions: Candidate assertions with @confidence.
        strategy:   Resolution strategy (same as resolve_conflict).

    Returns:
        ConflictReport with winner enriched by @opinion field.
    """
    report = resolve_conflict(assertions, strategy=strategy)

    # Enrich winner with Opinion metadata
    confidence = report.winner.get("@confidence", 0.5)
    opinion = Opinion.from_confidence(confidence)
    report.winner["@opinion"] = opinion.to_jsonld()

    return report


# ═══════════════════════════════════════════════════════════════════
# Drop-in replacement for propagate_confidence('multiply')
# ═══════════════════════════════════════════════════════════════════


def scalar_propagate_via_algebra(
    chain: Sequence[float],
    method: Literal["multiply"],
) -> PropagationResult:
    """Drop-in replacement for propagate_confidence('multiply').

    Internally uses trust discount with base_rate=0, producing
    the exact same scalar result as the multiply method.

    This proves the equivalence: multiply ≡ iterated trust discount
    when trust opinions are dogmatic and base_rate=0.

    Args:
        chain:  Confidence scores along the inference path.
        method: Must be "multiply".

    Returns:
        PropagationResult (same type as propagate_confidence).
    """
    if method != "multiply":
        raise ValueError(
            f"Algebra bridge only supports 'multiply', got: {method!r}"
        )
    if len(chain) == 0:
        raise ValueError("Chain must contain at least one score")

    chain_list = list(chain)
    opinion = propagate_opinions_from_scalars(chain_list, trust_uncertainty=0.0)

    return PropagationResult(
        score=opinion.to_confidence(),
        method=method,
        input_scores=chain_list,
    )
