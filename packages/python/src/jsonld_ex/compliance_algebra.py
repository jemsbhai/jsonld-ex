"""
Compliance Algebra — Modeling Regulatory Uncertainty with Subjective Logic.

Implements the compliance algebra formalized in compliance_algebra.md
(Syed, Silaghi, Abujar, Alssadi 2026). All operators are grounded in
specific GDPR provisions with formally proven properties.

The algebra models compliance status as uncertain epistemic states using
Jøsang's Subjective Logic, distinguishing between well-evidenced compliance,
absence of evidence, and conflicting evidence — states that binary pass/fail
systems collapse into a single classification.

Operators:
    §5  jurisdictional_meet      — conjunction across jurisdictions
    §6  compliance_propagation   — propagation through derivation chains
    §7  consent_validity         — six-condition GDPR Art. 7 composition
    §7  withdrawal_override      — proposition replacement at withdrawal
    §8  expiry_trigger           — asymmetric l → v transition
    §8  review_due_trigger       — accelerated decay toward vacuity
    §8  regulatory_change_trigger — proposition replacement at reg change
    §9  erasure_scope_opinion    — composite erasure completeness
    §9  residual_contamination   — disjunctive contamination risk

Dependencies:
    - jsonld_ex.confidence_algebra (Opinion)
    - jsonld_ex.confidence_decay (decay_opinion)
    - stdlib only (dataclasses, math, typing)

References:
    Jøsang, A. (2016). Subjective Logic. Springer.
    GDPR: https://gdpr-info.eu/ (Articles 4, 5, 6, 7, 17, 25, 35, 44–49)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple, Union

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import decay_opinion


# ═══════════════════════════════════════════════════════════════════
# §4 — COMPLIANCE OPINION (Definition 2)
# ═══════════════════════════════════════════════════════════════════


@dataclass(frozen=True, eq=False)
class ComplianceOpinion(Opinion):
    """A compliance opinion ω = (l, v, u, a) per Definition 2.

    Extends Opinion with domain-specific semantics:
        l (lawfulness)  = belief      — evidence of compliance
        v (violation)   = disbelief   — evidence of violation
        u (uncertainty) = uncertainty — absence of evidence
        a (base_rate)   = base_rate   — prior compliance probability

    Invariant: l + v + u = 1, all components in [0, 1].
    Projected probability: P(ω) = l + a·u.

    The class adds no new dataclass fields. It provides property
    aliases, domain-specific factories, and compliance notation repr.
    All existing SL operators accept ComplianceOpinion via isinstance.

    Independence assumptions and bias directions are documented per
    operator. See compliance_algebra.md §12.1.
    """

    # ── Domain-specific property aliases ───────────────────────

    @property
    def lawfulness(self) -> float:
        """Evidence of compliance (alias for belief)."""
        return self.belief

    @property
    def violation(self) -> float:
        """Evidence of violation (alias for disbelief)."""
        return self.disbelief

    # .uncertainty and .base_rate are inherited from Opinion.

    # ── Factory methods ────────────────────────────────────────

    @classmethod
    def create(
        cls,
        lawfulness: float,
        violation: float,
        uncertainty: float,
        base_rate: float = 0.5,
    ) -> ComplianceOpinion:
        """Create from compliance-domain parameters (l, v, u, a).

        Args:
            lawfulness:  Evidence of compliance.    l ∈ [0, 1]
            violation:   Evidence of violation.     v ∈ [0, 1]
            uncertainty: Absence of evidence.       u ∈ [0, 1]
            base_rate:   Prior compliance rate.     a ∈ [0, 1]

        Constraint: l + v + u = 1.

        Raises:
            ValueError: If constraint violated or components out of range.
        """
        return cls(
            belief=lawfulness,
            disbelief=violation,
            uncertainty=uncertainty,
            base_rate=base_rate,
        )

    @classmethod
    def from_opinion(cls, opinion: Opinion) -> ComplianceOpinion:
        """Wrap an existing Opinion as a ComplianceOpinion.

        Used to convert results from standard SL operators (which
        return plain Opinion) into the compliance domain.
        """
        return cls(
            belief=opinion.belief,
            disbelief=opinion.disbelief,
            uncertainty=opinion.uncertainty,
            base_rate=opinion.base_rate,
        )

    # ── Equality (interoperable with Opinion) ──────────────────

    def __eq__(self, other: object) -> bool:
        """Compare by field values, not by type.

        Allows ComplianceOpinion == Opinion when all four
        components match, enabling natural interoperability.
        """
        if isinstance(other, Opinion):
            return (
                self.belief == other.belief
                and self.disbelief == other.disbelief
                and self.uncertainty == other.uncertainty
                and self.base_rate == other.base_rate
            )
        return NotImplemented

    def __hash__(self) -> int:
        """Consistent with __eq__: hash on field values only."""
        return hash((self.belief, self.disbelief, self.uncertainty, self.base_rate))

    # ── Representation ─────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ComplianceOpinion(l={self.lawfulness:.4f}, "
            f"v={self.violation:.4f}, "
            f"u={self.uncertainty:.4f}, "
            f"a={self.base_rate:.4f})"
        )


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════


def _as_compliance(op: Opinion) -> ComplianceOpinion:
    """Normalize any Opinion to ComplianceOpinion."""
    if isinstance(op, ComplianceOpinion):
        return op
    return ComplianceOpinion.from_opinion(op)


# ═══════════════════════════════════════════════════════════════════
# §5 — JURISDICTIONAL MEET (Definition 3, Theorem 1)
# ═══════════════════════════════════════════════════════════════════


def _jurisdictional_meet_pair(
    w1: Opinion, w2: Opinion,
) -> ComplianceOpinion:
    """Binary jurisdictional meet per Definition 3.

    Given ω₁ = (l₁, v₁, u₁, a₁) and ω₂ = (l₂, v₂, u₂, a₂):
        l⊓ = l₁ · l₂
        v⊓ = v₁ + v₂ − v₁ · v₂
        u⊓ = (1 − v₁)(1 − v₂) − l₁ · l₂
        a⊓ = a₁ · a₂

    Lawfulness is a conjunction: both must be satisfied.
    Violation is a disjunction: either constitutes violation.
    """
    l1, v1 = w1.belief, w1.disbelief
    l2, v2 = w2.belief, w2.disbelief

    l_meet = l1 * l2
    v_meet = v1 + v2 - v1 * v2
    u_meet = (1.0 - v1) * (1.0 - v2) - l1 * l2
    a_meet = w1.base_rate * w2.base_rate

    # Clamp for floating-point safety
    if u_meet < 0.0:
        u_meet = 0.0

    return ComplianceOpinion(
        belief=l_meet,
        disbelief=v_meet,
        uncertainty=u_meet,
        base_rate=a_meet,
    )


def jurisdictional_meet(*opinions: Opinion) -> ComplianceOpinion:
    """Jurisdictional meet — conjunction of compliance requirements.

    Per Definition 3 (§5). Models composite compliance across multiple
    regulatory jurisdictions: satisfaction of ALL requirements.

    N-ary via left-fold; associativity proven in Theorem 1(f).

    Algebraic structure: bounded commutative monoid.
        Identity:     ω∅ = (1, 0, 0, 1)
        Annihilator:  ω⊥ = (0, 1, 0, 0)

    Independence assumption: jurisdictional opinions are assessed
    independently. Bias direction: non-conservative (underestimates
    violation under positive correlation). See §12.1.

    Args:
        *opinions: One or more opinions to combine.

    Returns:
        ComplianceOpinion representing composite compliance.

    Raises:
        ValueError: If no opinions provided.
    """
    if len(opinions) == 0:
        raise ValueError("jurisdictional_meet requires at least one opinion")
    if len(opinions) == 1:
        return _as_compliance(opinions[0])

    result = opinions[0]
    for i in range(1, len(opinions)):
        result = _jurisdictional_meet_pair(result, opinions[i])
    # _jurisdictional_meet_pair always returns ComplianceOpinion
    return result  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════
# §6 — COMPLIANCE PROPAGATION (Definitions 4–7, Theorem 2)
# ═══════════════════════════════════════════════════════════════════


def compliance_propagation(
    source: Opinion,
    derivation_trust: Opinion,
    purpose_compat: Opinion,
) -> ComplianceOpinion:
    """Compliance propagation through data derivation chains.

    Per Definition 6 (§6). The derived dataset's compliance is the
    three-way jurisdictional meet of source compliance, derivation
    process trust, and purpose compatibility:

        Prop(τ, π, ω_S) = J⊓(ω_S, τ, π)

    Explicit components:
        l_D = t · p · l_S
        v_D = 1 − (1−f)(1−q)(1−v_S)
        u_D = (1−f)(1−q)(1−v_S) − t·p·l_S
        a_D = a_τ · a_π · a_S

    The separation of τ (derivation trust) and π (purpose compatibility)
    reflects distinct GDPR provisions: Art. 5(1)(a)/25 govern process
    lawfulness; Art. 5(1)(b)/6(4) govern purpose limitation.

    Independence assumption: τ, π, ω_S are independent. Under positive
    correlation, the result biases toward optimism (non-conservative).
    See §12.1.

    Args:
        source:           Source dataset compliance opinion (ω_S).
        derivation_trust: Derivation process trust opinion (τ).
        purpose_compat:   Purpose compatibility opinion (π).

    Returns:
        ComplianceOpinion for the derived dataset.
    """
    return jurisdictional_meet(source, derivation_trust, purpose_compat)


@dataclass
class ProvenanceChain:
    """Ordered provenance chain recording each derivation step.

    Per Definition 7 (§6). Records [(ω_S, t₀), (τ₁, π₁, t₁), ...]
    as the legally required audit artifact under Art. 30 and Art. 5(2).
    The algebraic result is a computed summary verifiable from the chain.
    """

    source: Opinion
    """Source dataset compliance opinion."""

    source_timestamp: float
    """Timestamp of the source assessment."""

    steps: List[Tuple[Opinion, Opinion, float]] = field(default_factory=list)
    """List of (derivation_trust, purpose_compat, timestamp) tuples."""

    def add_step(
        self,
        trust: Opinion,
        purpose: Opinion,
        timestamp: float,
    ) -> None:
        """Append a derivation step to the chain.

        Args:
            trust:     Derivation process trust opinion (τ_i).
            purpose:   Purpose compatibility opinion (π_i).
            timestamp: Timestamp of this derivation step.
        """
        self.steps.append((trust, purpose, timestamp))

    def compute(self) -> ComplianceOpinion:
        """Compute the derived compliance by iterative propagation.

        Applies compliance_propagation for each step in order,
        matching the chain associativity of Theorem 2(e).

        Returns:
            ComplianceOpinion for the final derived dataset.
        """
        current = _as_compliance(self.source)
        for trust, purpose, _ts in self.steps:
            current = compliance_propagation(current, trust, purpose)
        return current


# ═══════════════════════════════════════════════════════════════════
# §7 — CONSENT ASSESSMENT (Definitions 8–11, Theorem 3)
# ═══════════════════════════════════════════════════════════════════


def consent_validity(
    *conditions: Opinion,
    freely_given: Union[Opinion, None] = None,
    specific: Union[Opinion, None] = None,
    informed: Union[Opinion, None] = None,
    unambiguous: Union[Opinion, None] = None,
    demonstrable: Union[Opinion, None] = None,
    distinguishable: Union[Opinion, None] = None,
) -> ComplianceOpinion:
    """Consent validity via six-condition jurisdictional meet.

    Per Definition 8 (§7.1). GDPR Art. 4(11) and Art. 7 define six
    conditions for valid consent. The consent validity opinion is:

        ω_c^P = J⊓(ω_free, ω_spec, ω_inf, ω_unamb, ω_demo, ω_dist)

    Accepts either six keyword arguments (domain-explicit) or six
    positional arguments (programmatic use). Keyword arguments take
    precedence if both are provided.

    Independence assumption: the six conditions may be positively
    correlated in practice, biasing toward optimism. See §12.1.

    Args:
        *conditions:     Six positional opinions (alternative to kwargs).
        freely_given:    Art. 7(4) — consent freely given.
        specific:        Art. 4(11) — specific to purpose.
        informed:        Art. 4(11) — data subject informed.
        unambiguous:     Art. 4(11) — unambiguous indication.
        demonstrable:    Art. 7(1) — controller can demonstrate.
        distinguishable: Art. 7(2) — distinguishable from other matters.

    Returns:
        ComplianceOpinion for composite consent validity.

    Raises:
        ValueError: If neither six positional nor six keyword args provided.
    """
    # Collect opinions from kwargs if provided
    kwargs_list = [
        freely_given, specific, informed,
        unambiguous, demonstrable, distinguishable,
    ]
    kwargs_provided = [op for op in kwargs_list if op is not None]

    if kwargs_provided:
        if len(kwargs_provided) != 6:
            raise ValueError(
                "consent_validity requires all six keyword arguments "
                "or six positional arguments"
            )
        return jurisdictional_meet(*kwargs_provided)

    if len(conditions) == 6:
        return jurisdictional_meet(*conditions)

    raise ValueError(
        "consent_validity requires exactly six consent condition opinions, "
        f"got {len(conditions)} positional and "
        f"{len(kwargs_provided)} keyword arguments"
    )


def withdrawal_override(
    consent_opinion: Opinion,
    withdrawal_opinion: Opinion,
    assessment_time: float,
    withdrawal_time: float,
) -> ComplianceOpinion:
    """Withdrawal override — proposition replacement at withdrawal.

    Per Definition 11 (§7.2). Post-withdrawal, the compliance-relevant
    question changes from "was valid consent given?" to "has the
    controller ceased processing?" — structurally different propositions.

    No existing SL operator models this proposition replacement: fusion
    combines evidence about a single proposition; trust discount relates
    opinions about the same proposition. The contribution is identifying
    that replacement is the correct semantic operation (Art. 7(3)).

    Sharp temporal boundary at t_w (modeling simplification). Practical
    delay in stopping processing is captured by uncertainty in ω_w.

    Theorem 3 properties:
        (a) Withdrawal dominance: t ≥ t_w → depends only on ω_w.
        (b) Pre-withdrawal preservation: t < t_w → assessed on ω_c.
        (c) Non-interference: per-purpose indexing (caller's responsibility).

    Args:
        consent_opinion:    Pre-withdrawal consent opinion (ω_c^P).
        withdrawal_opinion: Withdrawal implementation opinion (ω_w^P).
        assessment_time:    Time of assessment (t).
        withdrawal_time:    Time of withdrawal (t_w).

    Returns:
        ComplianceOpinion — consent if t < t_w, withdrawal if t ≥ t_w.
    """
    if assessment_time < withdrawal_time:
        return _as_compliance(consent_opinion)
    return _as_compliance(withdrawal_opinion)


# ═══════════════════════════════════════════════════════════════════
# §8 — TEMPORAL DECAY TRIGGERS (Definitions 12–14, Theorem 4)
# ═══════════════════════════════════════════════════════════════════


def expiry_trigger(
    opinion: Opinion,
    assessment_time: float,
    trigger_time: float,
    residual_factor: float = 0.0,
) -> ComplianceOpinion:
    """Expiry trigger — asymmetric l → v transition.

    Per Definition 12 (§8.2). At trigger time t_T, lawfulness transfers
    to violation (not uncertainty). An expired deadline is a known fact,
    not a source of epistemic uncertainty.

    Pre-trigger (t < t_T): return opinion unchanged.
    Post-trigger (t ≥ t_T):
        l' = γ · l
        v' = v + (1 − γ) · l
        u' = u  (unchanged)

    The parameter γ (residual_factor):
        γ = 0: hard expiry — all lawfulness converts to violation
        γ = 1: no immediate effect

    Theorem 4 properties:
        (b) Constraint preservation: l' + v' + u' = 1.
        (c) Monotonicity: l' ≤ l, v' ≥ v, u' = u.
        (d) Hard expiry: γ=0 → l'=0, v'=v+l.

    Args:
        opinion:         The compliance opinion to evaluate.
        assessment_time: Time of assessment (t).
        trigger_time:    Expiry trigger time (t_T).
        residual_factor: γ ∈ [0, 1], fraction of lawfulness retained.

    Returns:
        ComplianceOpinion after applying expiry (if triggered).
    """
    if assessment_time < trigger_time:
        return _as_compliance(opinion)

    gamma = residual_factor
    l = opinion.belief
    v = opinion.disbelief
    u = opinion.uncertainty
    a = opinion.base_rate

    l_prime = gamma * l
    v_prime = v + (1.0 - gamma) * l
    u_prime = u  # unchanged — expiry is a known fact

    return ComplianceOpinion(
        belief=l_prime,
        disbelief=v_prime,
        uncertainty=u_prime,
        base_rate=a,
    )


def review_due_trigger(
    opinion: Opinion,
    assessment_time: float,
    trigger_time: float,
    accelerated_half_life: float,
) -> ComplianceOpinion:
    """Review-due trigger — accelerated decay toward vacuity.

    Per Definition 13 (§8.2). A missed mandatory review (e.g., Art. 45(3)
    adequacy review, Art. 35(11) DPIA review) accelerates uncertainty
    growth. Unlike expiry, review-due moves toward vacuity (increased u),
    not violation: a missed review means we lack current evidence, not
    that we know the situation is non-compliant.

    Pre-trigger (t < t_T): return opinion unchanged.
    Post-trigger (t ≥ t_T): apply accelerated decay using existing
    decay_opinion with elapsed = assessment_time − trigger_time.

    Args:
        opinion:                The compliance opinion to evaluate.
        assessment_time:        Time of assessment (t).
        trigger_time:           Review-due trigger time (t_T).
        accelerated_half_life:  Shorter half-life for post-trigger decay.

    Returns:
        ComplianceOpinion after applying accelerated decay (if triggered).
    """
    if assessment_time < trigger_time:
        return _as_compliance(opinion)

    elapsed = assessment_time - trigger_time
    decayed = decay_opinion(
        opinion, elapsed=elapsed, half_life=accelerated_half_life,
    )
    return ComplianceOpinion.from_opinion(decayed)


def regulatory_change_trigger(
    opinion: Opinion,
    assessment_time: float,
    trigger_time: float,
    new_opinion: Opinion,
) -> ComplianceOpinion:
    """Regulatory change trigger — proposition replacement.

    Per Definition 14 (§8.2). At trigger time, the compliance opinion
    is replaced by a new assessment reflecting the changed legal
    framework. Models discrete legal events such as adequacy decision
    revocation or new regulation taking effect.

    Same proposition-replacement semantics as withdrawal_override.

    Theorem 4(e): trigger ordering is non-commutative by design —
    the order of regulatory events matters.

    Args:
        opinion:         The current compliance opinion.
        assessment_time: Time of assessment (t).
        trigger_time:    Regulatory change time (t_T).
        new_opinion:     New assessment under changed framework (ω_new).

    Returns:
        ComplianceOpinion — original if t < t_T, new if t ≥ t_T.
    """
    if assessment_time < trigger_time:
        return _as_compliance(opinion)
    return _as_compliance(new_opinion)


# ═══════════════════════════════════════════════════════════════════
# §9 — ERASURE PROPAGATION (Definitions 15–17, Theorem 5, Prop. 1)
# ═══════════════════════════════════════════════════════════════════


def erasure_scope_opinion(*per_node_opinions: Opinion) -> ComplianceOpinion:
    """Composite erasure completeness — n-ary jurisdictional meet.

    Per §9.2. Complete erasure requires ALL nodes to be erased — a
    conjunction:

        ω_E^R = J⊓(ω_e^i : D_i ∈ S)

    Theorem 5 properties:
        (a) Exponential degradation: e_R = ∏ e_i.
        (b) Scope monotonicity: adding a node decreases e_R.
        (c) Exception filtering: removing a node increases e_R.
        (d) Perfect source erasure: e=1 contributes no degradation.

    Independence bias direction: conservative (overestimates risk).
    Opposite to compliance operators. See §12.1.

    Args:
        *per_node_opinions: Erasure completeness opinions for each node.

    Returns:
        ComplianceOpinion for composite erasure completeness.

    Raises:
        ValueError: If no opinions provided.
    """
    return jurisdictional_meet(*per_node_opinions)


def residual_contamination(*ancestor_opinions: Opinion) -> ComplianceOpinion:
    """Residual contamination risk at a node given its ancestors.

    Per Definition 17 (§9.3). Node D_j is contaminated if personal data
    persists in D_j or any of its ancestors — a disjunction of per-node
    persistence.

    Given A_j⁺ = ancestors(D_j) ∩ S ∪ {D_j} with erasure opinions
    ω_e^i = (e_i, ē_i, u_e^i, a_e^i):

        r_j   = 1 − ∏(1 − ē_i)           contamination risk
        r̄_j   = ∏ e_i                      clean probability
        u_r^j = ∏(1 − ē_i) − ∏ e_i        uncertainty

    Where ē_i = disbelief (evidence of data persistence),
          e_i = belief (evidence of erasure completeness).

    Proposition 1: r + r̄ + u_r = 1, all non-negative. Risk is
    monotonically non-decreasing in |A_j⁺|.

    Equal-reach assumption: contamination from any ancestor is treated
    equally regardless of derivation distance. See §9.3.

    Independence bias direction: conservative (overestimates risk).

    Args:
        *ancestor_opinions: Erasure opinions for the node and its ancestors.

    Returns:
        ComplianceOpinion where:
            lawfulness  = r̄ (clean probability)
            violation   = r (contamination risk)
            uncertainty = u_r

    Raises:
        ValueError: If no opinions provided.
    """
    if len(ancestor_opinions) == 0:
        raise ValueError(
            "residual_contamination requires at least one opinion"
        )

    # ∏(1 − ē_i) where ē_i = disbelief (persistence evidence)
    prod_one_minus_ebar = math.prod(
        1.0 - op.disbelief for op in ancestor_opinions
    )

    # ∏ e_i where e_i = belief (erasure evidence)
    prod_e = math.prod(op.belief for op in ancestor_opinions)

    r = 1.0 - prod_one_minus_ebar        # contamination risk
    r_bar = prod_e                        # clean probability
    u_r = prod_one_minus_ebar - prod_e    # uncertainty

    # Clamp for floating-point safety
    if u_r < 0.0:
        u_r = 0.0
    if r < 0.0:
        r = 0.0

    # Average base rate across ancestors
    a = sum(op.base_rate for op in ancestor_opinions) / len(ancestor_opinions)

    return ComplianceOpinion(
        belief=r_bar,       # clean = lawfulness
        disbelief=r,        # contamination = violation
        uncertainty=u_r,
        base_rate=a,
    )
