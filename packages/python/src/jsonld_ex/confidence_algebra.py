"""
Formal Confidence Algebra for JSON-LD-Ex.

Grounds uncertainty representation and propagation in Jøsang's Subjective
Logic framework (Jøsang 2016), providing a rigorous mathematical foundation
for confidence scores in AI/ML data exchange.

Core Concept — Opinion:
    An opinion ω = (b, d, u, a) represents a subjective belief about a
    binary proposition, where:
        b ∈ [0,1]  — belief (evidence FOR)
        d ∈ [0,1]  — disbelief (evidence AGAINST)
        u ∈ [0,1]  — uncertainty (absence of evidence)
        a ∈ [0,1]  — base rate (prior probability)
    Constraint: b + d + u = 1

    The projected probability P(ω) = b + a·u maps an opinion back to a
    scalar probability, collapsing the three-dimensional opinion into a
    single number by distributing uncertainty according to the base rate.

Why not just use scalar confidence?
    A scalar c ∈ [0,1] conflates two distinct phenomena:
      1. How much the evidence supports the claim (belief vs disbelief)
      2. How much evidence exists at all (uncertainty)

    Example: c = 0.5 could mean:
      - "We have strong evidence that the probability is 50%" (u ≈ 0)
      - "We have no evidence at all" (u ≈ 1, a = 0.5)

    These cases require very different downstream behavior in agentic
    systems.  The Opinion type distinguishes them.

References:
    Jøsang, A. (2016). Subjective Logic: A Formalism for Reasoning Under
    Uncertainty. Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# Tolerance for floating-point comparison in the b + d + u = 1 constraint
_ADDITIVITY_TOL = 1e-9


def _validate_component(value: Any, name: str) -> float:
    """Validate a single opinion component."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number, got: bool")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got: {type(value).__name__}")
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got: {value}")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got: {value}")
    return float(value)


@dataclass(frozen=True, eq=True)
class Opinion:
    """A subjective opinion ω = (b, d, u, a) per Subjective Logic.

    Represents a nuanced belief state that distinguishes between
    evidence for, evidence against, and absence of evidence — unlike
    a bare scalar confidence score.

    Attributes:
        belief:      Evidence FOR the proposition.       b ∈ [0, 1]
        disbelief:   Evidence AGAINST the proposition.   d ∈ [0, 1]
        uncertainty: Lack of evidence.                   u ∈ [0, 1]
        base_rate:   Prior probability (default 0.5).    a ∈ [0, 1]

    Invariant:
        b + d + u = 1  (within floating-point tolerance)
    """

    belief: float
    disbelief: float
    uncertainty: float
    base_rate: float = 0.5

    def __post_init__(self) -> None:
        # Validate individual components
        b = _validate_component(self.belief, "belief")
        d = _validate_component(self.disbelief, "disbelief")
        u = _validate_component(self.uncertainty, "uncertainty")
        a = _validate_component(self.base_rate, "base_rate")

        # Since frozen=True, use object.__setattr__ for any normalization
        object.__setattr__(self, "belief", b)
        object.__setattr__(self, "disbelief", d)
        object.__setattr__(self, "uncertainty", u)
        object.__setattr__(self, "base_rate", a)

        # Additivity constraint: b + d + u = 1
        total = b + d + u
        if abs(total - 1.0) > _ADDITIVITY_TOL:
            raise ValueError(
                f"belief + disbelief + uncertainty must sum to 1, "
                f"got {b} + {d} + {u} = {total}"
            )

    # ── Projections ────────────────────────────────────────────────

    def projected_probability(self) -> float:
        """Compute P(ω) = b + a·u.

        Maps the three-dimensional opinion to a scalar probability by
        distributing uncertainty mass according to the base rate.
        """
        return self.belief + self.base_rate * self.uncertainty

    def to_confidence(self) -> float:
        """Alias for projected_probability().

        Provides the scalar confidence score that this opinion
        corresponds to, enabling interoperability with systems that
        only support scalar @confidence values.
        """
        return self.projected_probability()

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_confidence(
        cls,
        confidence: float,
        uncertainty: float = 0.0,
        base_rate: float = 0.5,
    ) -> Opinion:
        """Create an Opinion from a scalar confidence score.

        Maps a scalar c ∈ [0,1] to an opinion by distributing
        (1 - uncertainty) between belief and disbelief proportionally
        to c and (1 - c).

        Args:
            confidence:  Scalar confidence in [0, 1].
            uncertainty: Fraction of mass assigned to uncertainty [0, 1].
                         Default 0.0 yields a dogmatic opinion.
            base_rate:   Prior probability, default 0.5.

        When uncertainty = 0 (default), this produces a *dogmatic* opinion
        where P(ω) = b = confidence.

        When uncertainty > 0, the remaining mass (1 - uncertainty) is split
        proportionally:
            b = confidence × (1 - uncertainty)
            d = (1 - confidence) × (1 - uncertainty)

        This ensures P(ω) = b + a·u preserves the original confidence
        when a matches the implied base rate.
        """
        # Validate inputs
        c = _validate_component(confidence, "@confidence")
        u = _validate_component(uncertainty, "uncertainty")
        _validate_component(base_rate, "base_rate")

        remaining = 1.0 - u
        b = c * remaining
        d = (1.0 - c) * remaining

        return cls(belief=b, disbelief=d, uncertainty=u, base_rate=base_rate)

    @classmethod
    def from_evidence(
        cls,
        positive: int | float,
        negative: int | float,
        prior_weight: float = 2.0,
        base_rate: float = 0.5,
    ) -> Opinion:
        """Create an Opinion from evidence counts (Jøsang 2016, §3.2).

        Uses the standard evidence-to-opinion mapping:
            b = r / (r + s + W)
            d = s / (r + s + W)
            u = W / (r + s + W)

        where r = positive evidence, s = negative evidence,
        W = non-informative prior weight.

        Args:
            positive:     Count of positive observations (≥ 0).
            negative:     Count of negative observations (≥ 0).
            prior_weight: Weight of non-informative prior (> 0, default 2).
            base_rate:    Prior probability, default 0.5.

        This mapping has the property that as evidence grows, uncertainty
        shrinks — reflecting increased epistemic confidence.
        """
        if positive < 0 or negative < 0:
            raise ValueError("Evidence counts must be non-negative")
        if prior_weight <= 0:
            raise ValueError("prior_weight must be positive")

        r = float(positive)
        s = float(negative)
        W = float(prior_weight)
        total = r + s + W

        return cls(
            belief=r / total,
            disbelief=s / total,
            uncertainty=W / total,
            base_rate=base_rate,
        )

    # ── Serialization ──────────────────────────────────────────────

    def to_jsonld(self) -> dict[str, Any]:
        """Serialize to a JSON-LD compatible dict.

        Returns a dictionary suitable for embedding in a JSON-LD
        document as an annotation value.
        """
        return {
            "@type": "Opinion",
            "belief": self.belief,
            "disbelief": self.disbelief,
            "uncertainty": self.uncertainty,
            "baseRate": self.base_rate,
        }

    @classmethod
    def from_jsonld(cls, data: dict[str, Any]) -> Opinion:
        """Deserialize from a JSON-LD compatible dict."""
        return cls(
            belief=data["belief"],
            disbelief=data["disbelief"],
            uncertainty=data["uncertainty"],
            base_rate=data.get("baseRate", 0.5),
        )

    # ── Representation ─────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Opinion(b={self.belief:.4f}, d={self.disbelief:.4f}, "
            f"u={self.uncertainty:.4f}, a={self.base_rate:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════
# UTILITY
# ═══════════════════════════════════════════════════════════════════


def _require_opinion(value: object, name: str) -> None:
    """Raise TypeError if *value* is not an Opinion."""
    if not isinstance(value, Opinion):
        raise TypeError(
            f"{name} must be an Opinion, got: {type(value).__name__}"
        )


# ═══════════════════════════════════════════════════════════════════
# OPERATORS
# ═══════════════════════════════════════════════════════════════════


def cumulative_fuse(*opinions: Opinion) -> Opinion:
    """Cumulative fusion (⊕) — combine independent evidence sources.

    Per Jøsang 2016 §12.3.  When two or more independent sources
    observe the same proposition, cumulative fusion combines their
    evidence additively, reducing uncertainty.

    For two opinions ω_A = (b_A, d_A, u_A) and ω_B = (b_B, d_B, u_B):

    If at least one is non-dogmatic (u > 0):
        κ = u_A + u_B − u_A · u_B        (normalization denominator)
        b = (b_A · u_B + b_B · u_A) / κ
        d = (d_A · u_B + d_B · u_A) / κ
        u = (u_A · u_B) / κ

    If both are dogmatic (u_A = u_B = 0):
        The standard formula has a 0/0 indeterminate form.
        Per Josang (2016, §12.3, Eq. 12.4), the limit with
        equal relative dogmatism γ_A = γ_B = 0.5 yields:
        b = 0.5 · b_A + 0.5 · b_B
        d = 0.5 · d_A + 0.5 · d_B
        u = 0
        (i.e., simple average of the two dogmatic opinions).

    Properties:
        - Commutativity:  A ⊕ B = B ⊕ A
        - Associativity:  (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)
        - Identity:       A ⊕ vacuous = A
        - Uncertainty reduction: u_{A⊕B} ≤ min(u_A, u_B)

    Args:
        *opinions: Two or more opinions to fuse.  A single opinion
                   is returned unchanged.

    Returns:
        Fused Opinion.

    Raises:
        ValueError: If fewer than one opinion is provided.
    """
    if len(opinions) == 0:
        raise ValueError("cumulative_fuse requires at least one opinion")
    if len(opinions) == 1:
        return opinions[0]

    result = opinions[0]
    for i in range(1, len(opinions)):
        result = _cumulative_fuse_pair(result, opinions[i])
    return result


def _cumulative_fuse_pair(a: Opinion, b: Opinion) -> Opinion:
    """Cumulative fusion of exactly two opinions."""
    u_a, u_b = a.uncertainty, b.uncertainty

    # Dogmatic case: both have u = 0
    if u_a == 0.0 and u_b == 0.0:
        # Limit form with equal weight γ_A = γ_B = 0.5
        gamma_a = 0.5
        gamma_b = 0.5
        fused_b = gamma_a * a.belief + gamma_b * b.belief
        fused_d = gamma_a * a.disbelief + gamma_b * b.disbelief
        fused_u = 0.0
    else:
        # Standard cumulative fusion formula
        kappa = u_a + u_b - u_a * u_b  # always > 0 if at least one u > 0
        fused_b = (a.belief * u_b + b.belief * u_a) / kappa
        fused_d = (a.disbelief * u_b + b.disbelief * u_a) / kappa
        fused_u = (u_a * u_b) / kappa

    # Use average base rate for fused opinions
    fused_a = (a.base_rate + b.base_rate) / 2.0

    return Opinion(
        belief=fused_b,
        disbelief=fused_d,
        uncertainty=fused_u,
        base_rate=fused_a,
    )


def averaging_fuse(*opinions: Opinion) -> Opinion:
    """Averaging fusion (⊘) — combine dependent/correlated sources.

    Per Jøsang (2016, §12.5).  When sources may be correlated (e.g.,
    reading the same underlying data), averaging fusion avoids
    double-counting evidence.

    For n opinions ω_i = (b_i, d_i, u_i) with equal weight:

        U_i = ∏_{j≠i} u_j   (product of all OTHER uncertainties)
        κ   = Σ_i U_i

        b = Σ_i (b_i · U_i) / κ
        d = Σ_i (d_i · U_i) / κ
        u = n · ∏_i u_i / κ

    When all U_i = 0 (κ = 0), the formula has an indeterminate form.
    The limit with equal relative weight yields the simple average:
        b = Σ b_i / n,  d = Σ d_i / n,  u = 0.

    Key properties:
        - Commutativity:  order of opinions does not matter.
        - Idempotence:    A ⊘ A ⊘ ... ⊘ A = A.
        - NOT associative: (A ⊘ B) ⊘ C ≠ A ⊘ (B ⊘ C) in general.
          The simultaneous n-ary formula must be used for n > 2.

    For n = 2 this is mathematically identical to the pairwise formula.

    Args:
        *opinions: One or more opinions to fuse.

    Returns:
        Fused Opinion.

    Raises:
        ValueError: If no opinions are provided.
    """
    if len(opinions) == 0:
        raise ValueError("averaging_fuse requires at least one opinion")
    if len(opinions) == 1:
        return opinions[0]
    if len(opinions) == 2:
        # 2-source: delegate to the pairwise formula (identical result,
        # avoids unnecessary product computation).
        return _averaging_fuse_pair(opinions[0], opinions[1])

    return _averaging_fuse_nary(opinions)


def _averaging_fuse_pair(a: Opinion, b: Opinion) -> Opinion:
    """Averaging fusion of exactly two opinions (equal weight).

    Correct denominator per Jøsang 2016 §12.5 is (u_A + u_B),
    which guarantees b + d + u = 1.
    """
    u_a, u_b = a.uncertainty, b.uncertainty

    # Dogmatic case: both have u = 0
    if u_a == 0.0 and u_b == 0.0:
        fused_b = (a.belief + b.belief) / 2.0
        fused_d = (a.disbelief + b.disbelief) / 2.0
        fused_u = 0.0
    else:
        kappa = u_a + u_b
        if kappa == 0.0:
            # Degenerate case — fall back to simple average
            fused_b = (a.belief + b.belief) / 2.0
            fused_d = (a.disbelief + b.disbelief) / 2.0
            fused_u = 0.0
        else:
            fused_b = (a.belief * u_b + b.belief * u_a) / kappa
            fused_d = (a.disbelief * u_b + b.disbelief * u_a) / kappa
            fused_u = 2.0 * u_a * u_b / kappa

    fused_a = (a.base_rate + b.base_rate) / 2.0

    return Opinion(
        belief=fused_b,
        disbelief=fused_d,
        uncertainty=fused_u,
        base_rate=fused_a,
    )


def _averaging_fuse_nary(opinions: tuple[Opinion, ...] | list[Opinion]) -> Opinion:
    """Simultaneous n-ary averaging fusion (Jøsang 2016, §12.5).

    Uses the proper simultaneous formula for n ≥ 3 opinions with
    equal weight.  NOT a pairwise fold — averaging fusion is not
    associative, so pairwise application gives incorrect results.

    Formula:
        U_i = ∏_{j≠i} u_j
        κ   = Σ_i U_i
        b   = Σ_i (b_i · U_i) / κ
        d   = Σ_i (d_i · U_i) / κ
        u   = n · ∏_i u_i / κ

    When κ = 0 (all U_i are zero — occurs when enough opinions
    are dogmatic), the limit with equal weight reduces to a
    simple average:  b = Σb_i/n, d = Σd_i/n, u = 0.
    """
    n = len(opinions)

    # ── Compute the full product of all uncertainties ──
    # We use math.prod for clarity and precision.
    uncertainties = [o.uncertainty for o in opinions]
    full_product = math.prod(uncertainties)

    # ── Compute U_i = full_product / u_i for each i ──
    # When u_i = 0, U_i = ∏_{j≠i} u_j.  If ANY other u_j = 0
    # then U_i = 0 as well.  We handle this carefully to avoid
    # division by zero.
    capital_u = []
    for i, u_i in enumerate(uncertainties):
        if u_i != 0.0:
            capital_u.append(full_product / u_i)
        else:
            # u_i = 0: compute ∏_{j≠i} u_j directly
            product = 1.0
            for j, u_j in enumerate(uncertainties):
                if j != i:
                    product *= u_j
            capital_u.append(product)

    kappa = sum(capital_u)

    # ── Dogmatic fallback: κ = 0 ──
    # κ = 0 requires ≥ 2 dogmatic opinions (proof: if only one u_i = 0,
    # then U_i = ∏_{j≠i} u_j > 0 for the dogmatic i, so κ > 0).
    #
    # In the limit, dogmatic opinions have infinite relative evidence
    # weight.  Non-dogmatic opinions' U_i terms vanish at higher order
    # (ε² vs ε), contributing nothing.  The correct limit with equal
    # relative dogmatism among the dogmatic subset is their simple
    # average.
    if kappa == 0.0:
        dogmatic = [o for o in opinions if o.uncertainty == 0.0]
        z = len(dogmatic) if dogmatic else n  # fallback to all if none found
        pool = dogmatic if dogmatic else list(opinions)
        fused_b = sum(o.belief for o in pool) / z
        fused_d = sum(o.disbelief for o in pool) / z
        fused_u = 0.0
    else:
        fused_b = sum(o.belief * u_i for o, u_i in zip(opinions, capital_u)) / kappa
        fused_d = sum(o.disbelief * u_i for o, u_i in zip(opinions, capital_u)) / kappa
        fused_u = n * full_product / kappa

    fused_a = sum(o.base_rate for o in opinions) / n

    return Opinion(
        belief=fused_b,
        disbelief=fused_d,
        uncertainty=fused_u,
        base_rate=fused_a,
    )


def trust_discount(trust: Opinion, opinion: Opinion) -> Opinion:
    """Trust discount (⊗) — propagate opinion through a trust chain.

    Per Jøsang 2016 §14.3.  If agent A trusts agent B with opinion
    ω_AB, and B holds opinion ω_Bx about proposition x, then A's
    derived opinion about x is:

        b_Ax = b_AB · b_Bx
        d_Ax = b_AB · d_Bx
        u_Ax = d_AB + u_AB + b_AB · u_Bx

    Intuition:
        - Full trust (b_AB = 1):  A adopts B's opinion unchanged.
        - Zero trust (b_AB = 0):  Result is vacuous (total uncertainty).
        - Partial trust: opinion is "diluted" toward uncertainty.

    Args:
        trust:   A's trust opinion about B.
        opinion: B's opinion about proposition x.

    Returns:
        A's derived opinion about x after discounting by trust.
    """
    b_trust = trust.belief

    fused_b = b_trust * opinion.belief
    fused_d = b_trust * opinion.disbelief
    fused_u = trust.disbelief + trust.uncertainty + b_trust * opinion.uncertainty

    return Opinion(
        belief=fused_b,
        disbelief=fused_d,
        uncertainty=fused_u,
        base_rate=opinion.base_rate,
    )


def deduce(
    opinion_x: Opinion,
    opinion_y_given_x: Opinion,
    opinion_y_given_not_x: Opinion,
) -> Opinion:
    """Deduction operator — conditional reasoning under uncertainty.

    Per Jøsang (2016, §12.6).  Given an opinion about an antecedent x
    and two conditional opinions about y (one conditioned on x being
    true, one on x being false), derive an opinion about y.

    This is the subjective-logic generalisation of the **law of total
    probability**:
        P(y) = P(x)·P(y|x) + P(¬x)·P(y|¬x)

    Formula (Def. 12.6):
        Let ā_x = 1 − a_x.  For each component c ∈ {b, d, u}:

        c_y = b_x · c_{y|x}
            + d_x · c_{y|¬x}
            + u_x · (a_x · c_{y|x} + ā_x · c_{y|¬x})

    Base rate:
        a_y = a_x · P(y|x) + ā_x · P(y|¬x)

    where P(y|x) = b_{y|x} + a_{y|x} · u_{y|x}.

    Key properties:
        - **Additivity**:  b_y + d_y + u_y = 1 always.
        - **Classical limit**: When all opinions are dogmatic (u=0),
          reduces to the law of total probability.
        - **Projected probability**: P(ω_y) = P(x)·P(y|x) + (1-P(x))·P(y|¬x).

    Args:
        opinion_x:            Opinion about antecedent x (ω_x).
        opinion_y_given_x:    Conditional opinion about y given x (ω_{y|x}).
        opinion_y_given_not_x: Conditional opinion about y given ¬x (ω_{y|¬x}).

    Returns:
        Deduced opinion about y (ω_y).

    Raises:
        TypeError: If any argument is not an Opinion.
    """
    _require_opinion(opinion_x, "opinion_x")
    _require_opinion(opinion_y_given_x, "opinion_y_given_x")
    _require_opinion(opinion_y_given_not_x, "opinion_y_given_not_x")

    b_x = opinion_x.belief
    d_x = opinion_x.disbelief
    u_x = opinion_x.uncertainty
    a_x = opinion_x.base_rate
    a_x_bar = 1.0 - a_x

    # Shorthand for the two conditional opinions
    yx = opinion_y_given_x
    ynx = opinion_y_given_not_x

    # ── Deduction formula (Def. 12.6) for each component ──
    b_y = b_x * yx.belief + d_x * ynx.belief + u_x * (a_x * yx.belief + a_x_bar * ynx.belief)
    d_y = b_x * yx.disbelief + d_x * ynx.disbelief + u_x * (a_x * yx.disbelief + a_x_bar * ynx.disbelief)
    u_y = b_x * yx.uncertainty + d_x * ynx.uncertainty + u_x * (a_x * yx.uncertainty + a_x_bar * ynx.uncertainty)

    # ── Base rate: a_y = a_x · P(y|x) + ā_x · P(y|¬x) ──
    p_y_given_x = yx.projected_probability()
    p_y_given_not_x = ynx.projected_probability()
    a_y = a_x * p_y_given_x + a_x_bar * p_y_given_not_x

    return Opinion(
        belief=b_y,
        disbelief=d_y,
        uncertainty=u_y,
        base_rate=a_y,
    )
