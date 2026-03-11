"""
Multinomial Confidence Algebra for JSON-LD-Ex.

Extends the binomial confidence algebra to k-ary domains using
Jøsang's multinomial opinions (Subjective Logic, Ch. 3.5).

A multinomial opinion ω_X = (b_X, u_X, a_X) represents a subjective
belief about a variable X that can take one of k mutually exclusive
values, where:

    b_X : dict[str, float]  — belief mass distribution over k states
    u_X : float             — epistemic uncertainty (lack of evidence)
    a_X : dict[str, float]  — base rate (prior) distribution over k states

Constraints:
    Σ b_X(x) + u_X = 1     (additivity)
    Σ a_X(x) = 1            (base rates form a distribution)
    b_X(x) ≥ 0, u_X ≥ 0, a_X(x) ≥ 0

Projected probability:
    P_X(x) = b_X(x) + a_X(x) · u_X

This generalizes the binomial Opinion (k=2) in confidence_algebra.py.
When k=2, a MultinomialOpinion with states {T, F} is equivalent to
Opinion(belief=b(T), disbelief=b(F), uncertainty=u, base_rate=a(T)).

The evidence-to-opinion mapping uses the Dirichlet multinomial model
with a *dynamic* non-informative prior weight W that starts at the
domain cardinality k (when no evidence exists) and converges to 2
as evidence accumulates (matching the binomial case).

    W = (k + 2·k·Σr) / (1 + k·Σr)

    b(x) = r(x) / (Σr + W)
    u    = W / (Σr + W)

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 3.5 (Multinomial Opinions),
    §3.5.2 (Dirichlet Multinomial Model), §3.5.5 (Mapping).
    Jøsang, A. (2022). FUSION 2022 Tutorial, slides 20–30.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional


# Tolerance for floating-point comparison in sum constraints
_ADDITIVITY_TOL = 1e-9

# Tolerance for clamping machine-epsilon boundary overshoots
_BOUNDARY_TOL = 1e-12


def _validate_non_negative(value: float, name: str) -> float:
    """Validate and clamp a single non-negative component."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError(f"{name} must be finite, got: {value}")
    # Clamp machine-epsilon undershoots
    if -_BOUNDARY_TOL <= value < 0.0:
        value = 0.0
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got: {value}")
    return value


class MultinomialOpinion:
    """A multinomial subjective opinion ω_X = (b_X, u_X, a_X).

    Represents a subjective belief about a variable that can take one
    of k mutually exclusive values, generalizing the binomial Opinion
    to arbitrary discrete domains.

    Attributes:
        beliefs:    Immutable mapping from state name to belief mass.
        uncertainty: Epistemic uncertainty (lack of evidence).
        base_rates: Immutable mapping from state name to base rate.
        domain:     Tuple of sorted state names.
        cardinality: Number of states (k).

    Invariants:
        Σ beliefs(x) + uncertainty = 1
        Σ base_rates(x) = 1
        All values ≥ 0

    References:
        Jøsang (2016), Ch. 3.5; Jøsang FUSION 2022 Tutorial, slides 20–30.
    """

    __slots__ = ("_beliefs", "_uncertainty", "_base_rates", "_domain")

    def __init__(
        self,
        beliefs: Mapping[str, float],
        uncertainty: float,
        base_rates: Mapping[str, float],
    ) -> None:
        # ── Validate domain ──
        if len(beliefs) < 2:
            raise ValueError(
                f"Domain must have at least 2 states, got {len(beliefs)}"
            )
        if set(beliefs.keys()) != set(base_rates.keys()):
            raise ValueError(
                f"beliefs and base_rates must have the same domain. "
                f"beliefs keys: {sorted(beliefs.keys())}, "
                f"base_rates keys: {sorted(base_rates.keys())}"
            )

        # ── Validate and clamp individual components ──
        validated_beliefs: dict[str, float] = {}
        for key, val in beliefs.items():
            validated_beliefs[key] = _validate_non_negative(
                float(val), f"beliefs[{key!r}]"
            )

        validated_uncertainty = _validate_non_negative(
            float(uncertainty), "uncertainty"
        )

        validated_base_rates: dict[str, float] = {}
        for key, val in base_rates.items():
            validated_base_rates[key] = _validate_non_negative(
                float(val), f"base_rates[{key!r}]"
            )

        # ── Check additivity: Σb + u = 1 ──
        belief_sum = sum(validated_beliefs.values())
        total = belief_sum + validated_uncertainty
        if abs(total - 1.0) > _ADDITIVITY_TOL:
            raise ValueError(
                f"sum(beliefs) + uncertainty must sum to 1, "
                f"got {belief_sum} + {validated_uncertainty} = {total}"
            )

        # ── Check base rate distribution: Σa = 1 ──
        base_rate_sum = sum(validated_base_rates.values())
        if abs(base_rate_sum - 1.0) > _ADDITIVITY_TOL:
            raise ValueError(
                f"base_rates must sum to 1, got {base_rate_sum}"
            )

        # ── Store as immutable mappings ──
        domain = tuple(sorted(validated_beliefs.keys()))
        object.__setattr__(self, "_beliefs", MappingProxyType(validated_beliefs))
        object.__setattr__(self, "_uncertainty", validated_uncertainty)
        object.__setattr__(self, "_base_rates", MappingProxyType(validated_base_rates))
        object.__setattr__(self, "_domain", domain)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"MultinomialOpinion is immutable: cannot set {name!r}"
        )

    # ── Properties ─────────────────────────────────────────────────

    @property
    def beliefs(self) -> Mapping[str, float]:
        """Immutable belief mass distribution."""
        return self._beliefs

    @property
    def uncertainty(self) -> float:
        """Epistemic uncertainty mass."""
        return self._uncertainty

    @property
    def base_rates(self) -> Mapping[str, float]:
        """Immutable base rate distribution."""
        return self._base_rates

    @property
    def domain(self) -> tuple[str, ...]:
        """Sorted tuple of state names."""
        return self._domain

    @property
    def cardinality(self) -> int:
        """Number of states in the domain (k)."""
        return len(self._domain)

    # ── Projections ────────────────────────────────────────────────

    def projected_probability(self) -> dict[str, float]:
        """Compute P_X(x) = b_X(x) + a_X(x) · u_X for each state.

        Returns a dictionary mapping state names to projected
        probabilities.  The values sum to 1.
        """
        return {
            x: self._beliefs[x] + self._base_rates[x] * self._uncertainty
            for x in self._domain
        }

    # ── Factory methods ────────────────────────────────────────────

    @classmethod
    def from_evidence(
        cls,
        evidence: Mapping[str, int | float],
        base_rates: Mapping[str, float] | None = None,
    ) -> MultinomialOpinion:
        """Create a MultinomialOpinion from evidence counts.

        Uses the Dirichlet multinomial model with dynamic prior weight:

            W = (k + 2·k·Σr) / (1 + k·Σr)
            b(x) = r(x) / (Σr + W)
            u    = W / (Σr + W)

        When Σr = 0 (no evidence), W = k and u = 1 (vacuous).
        As Σr → ∞, W → 2 and the opinion becomes dogmatic.

        Args:
            evidence: Mapping from state name to observation count (≥ 0).
            base_rates: Optional prior distribution.  If None, defaults
                to uniform (1/k for each state).

        Returns:
            A MultinomialOpinion reflecting the evidence.

        Raises:
            ValueError: If evidence counts are negative or domain is
                too small.

        References:
            Jøsang (2016), §3.5.2, §3.5.5.
            Jøsang FUSION 2022, slide 25 (dynamic W formula).
        """
        if len(evidence) < 2:
            raise ValueError(
                f"Domain must have at least 2 states, got {len(evidence)}"
            )

        # Validate evidence counts
        for key, count in evidence.items():
            if count < 0:
                raise ValueError(
                    f"Evidence counts must be non-negative, "
                    f"got evidence[{key!r}] = {count}"
                )

        k = len(evidence)
        r_total = sum(float(v) for v in evidence.values())

        # Dynamic non-informative prior weight (Jøsang FUSION 2022)
        # W = (k + 2*k*Σr) / (1 + k*Σr)
        # When Σr=0: W=k.  As Σr→∞: W→2.
        denominator = 1.0 + k * r_total
        W = (k + 2.0 * k * r_total) / denominator

        total = r_total + W

        beliefs = {
            key: float(count) / total
            for key, count in evidence.items()
        }
        uncertainty = W / total

        if base_rates is None:
            base_rates = {key: 1.0 / k for key in evidence}

        return cls(
            beliefs=beliefs,
            uncertainty=uncertainty,
            base_rates=base_rates,
        )

    # ── Representation ─────────────────────────────────────────────

    def __repr__(self) -> str:
        k = self.cardinality
        b_str = ", ".join(
            f"{x}:{self._beliefs[x]:.3f}" for x in self._domain
        )
        return (
            f"MultinomialOpinion(k={k}, beliefs={{{b_str}}}, "
            f"u={self._uncertainty:.3f})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultinomialOpinion):
            return NotImplemented
        if self._domain != other._domain:
            return False
        if abs(self._uncertainty - other._uncertainty) > _ADDITIVITY_TOL:
            return False
        for x in self._domain:
            if abs(self._beliefs[x] - other._beliefs[x]) > _ADDITIVITY_TOL:
                return False
            if abs(self._base_rates[x] - other._base_rates[x]) > _ADDITIVITY_TOL:
                return False
        return True

    def __hash__(self) -> int:
        return hash((
            self._domain,
            tuple(self._beliefs[x] for x in self._domain),
            self._uncertainty,
            tuple(self._base_rates[x] for x in self._domain),
        ))

    # ── Serialization ──────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary.

        Returns:
            A dict with keys ``beliefs``, ``uncertainty``, ``base_rates``.
            Inner mappings are plain ``dict`` objects (not MappingProxyType).
        """
        return {
            "beliefs": dict(self._beliefs),
            "uncertainty": self._uncertainty,
            "base_rates": dict(self._base_rates),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MultinomialOpinion:
        """Reconstruct a MultinomialOpinion from a plain dictionary.

        Inverse of :meth:`to_dict`.

        Args:
            data: Dict with keys ``beliefs``, ``uncertainty``, ``base_rates``.

        Returns:
            A MultinomialOpinion matching the serialized values.
        """
        return cls(
            beliefs=data["beliefs"],
            uncertainty=data["uncertainty"],
            base_rates=data["base_rates"],
        )

    def to_jsonld(self) -> dict[str, Any]:
        """Serialize to a JSON-LD compatible dictionary.

        Follows the same pattern as ``Opinion.to_jsonld()`` from
        ``confidence_algebra``.  Uses camelCase for JSON-LD keys
        (``baseRates`` instead of ``base_rates``).

        Returns:
            A dict with ``@type``, ``beliefs``, ``uncertainty``,
            ``baseRates``.
        """
        return {
            "@type": "MultinomialOpinion",
            "beliefs": dict(self._beliefs),
            "uncertainty": self._uncertainty,
            "baseRates": dict(self._base_rates),
        }

    @classmethod
    def from_jsonld(cls, data: dict[str, Any]) -> MultinomialOpinion:
        """Reconstruct a MultinomialOpinion from a JSON-LD dictionary.

        Inverse of :meth:`to_jsonld`.  Accepts both ``baseRates``
        (preferred, camelCase) and ``base_rates`` (fallback, snake_case)
        for robustness.

        Args:
            data: Dict with ``beliefs``, ``uncertainty``, and either
                ``baseRates`` or ``base_rates``.

        Returns:
            A MultinomialOpinion matching the serialized values.
        """
        base_rates = data.get("baseRates", data.get("base_rates"))
        return cls(
            beliefs=data["beliefs"],
            uncertainty=data["uncertainty"],
            base_rates=base_rates,
        )


# ═══════════════════════════════════════════════════════════════════
# COARSENING AND PROMOTION
# ═══════════════════════════════════════════════════════════════════


def coarsen(
    opinion: MultinomialOpinion,
    focus_state: str,
) -> "Opinion":
    """Coarsen a multinomial opinion to a binomial opinion.

    Selects one state as the "focus" (positive) state and collapses
    all other states into the complementary (negative) class.

    Mapping:
        belief    = b(focus_state)
        disbelief = Σ b(x) for x ≠ focus_state
        uncertainty = u  (unchanged)
        base_rate = a(focus_state)

    The projected probability is preserved:
        P_binomial = b + a·u = b(focus) + a(focus)·u = P_multinomial(focus)

    Args:
        opinion: A MultinomialOpinion to coarsen.
        focus_state: The state to treat as "positive" (True).

    Returns:
        A binomial ``Opinion`` from ``confidence_algebra``.

    Raises:
        ValueError: If focus_state is not in the opinion's domain.

    References:
        Jøsang (2016), §3.5.4 (Coarsening Example: From Ternary to Binary).
    """
    from jsonld_ex.confidence_algebra import Opinion

    if focus_state not in opinion.domain:
        raise ValueError(
            f"focus_state {focus_state!r} not in domain {opinion.domain}"
        )

    belief = opinion.beliefs[focus_state]
    disbelief = sum(
        b for x, b in opinion.beliefs.items() if x != focus_state
    )
    uncertainty = opinion.uncertainty
    base_rate = opinion.base_rates[focus_state]

    return Opinion(
        belief=belief,
        disbelief=disbelief,
        uncertainty=uncertainty,
        base_rate=base_rate,
    )


def promote(
    opinion: "Opinion",
    true_state: str = "T",
    false_state: str = "F",
) -> MultinomialOpinion:
    """Promote a binomial opinion to a MultinomialOpinion with k=2.

    Mapping:
        beliefs = {true_state: b, false_state: d}
        uncertainty = u
        base_rates = {true_state: a, false_state: 1-a}

    The projected probability is preserved:
        P_multinomial(true_state) = b + a·u = P_binomial

    Args:
        opinion: A binomial ``Opinion`` from ``confidence_algebra``.
        true_state: Name for the positive state (default "T").
        false_state: Name for the negative state (default "F").

    Returns:
        A MultinomialOpinion with two states.

    Raises:
        ValueError: If true_state == false_state.
    """
    if true_state == false_state:
        raise ValueError(
            f"true_state and false_state must be different, "
            f"got {true_state!r} for both"
        )

    return MultinomialOpinion(
        beliefs={true_state: opinion.belief, false_state: opinion.disbelief},
        uncertainty=opinion.uncertainty,
        base_rates={
            true_state: opinion.base_rate,
            false_state: 1.0 - opinion.base_rate,
        },
    )


# ═══════════════════════════════════════════════════════════════════
# MULTINOMIAL FUSION OPERATORS
# ═══════════════════════════════════════════════════════════════════


def multinomial_cumulative_fuse(
    *opinions: MultinomialOpinion,
) -> MultinomialOpinion:
    """Cumulative fusion (⊕) for multinomial opinions.

    Combines independent evidence sources additively, reducing
    uncertainty.  Generalizes binomial cumulative_fuse() to k-ary
    domains.

    For two opinions with at least one non-dogmatic (u > 0):
        κ = u_A + u_B − u_A · u_B
        b_fused(x) = (b_A(x) · u_B + b_B(x) · u_A) / κ
        u_fused = (u_A · u_B) / κ

    When both are dogmatic (u_A = u_B = 0), the limit with equal
    relative dogmatism yields the simple average of beliefs.

    Properties:
        - Commutativity:  A ⊕ B = B ⊕ A
        - Associativity:  (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)  for beliefs
          and uncertainty.  Base rate averaging is associative only
          when all input base rates are identical.  When base rates
          differ, the fused base rate depends on fusion order.
          This is consistent with binomial cumulative_fuse().
        - Identity:       A ⊕ vacuous = A
        - Uncertainty reduction: u_{A⊕B} ≤ min(u_A, u_B)

    Args:
        *opinions: Two or more MultinomialOpinions to fuse.
            All must share the same domain.  A single opinion
            is returned unchanged.

    Returns:
        Fused MultinomialOpinion.

    Raises:
        ValueError: If fewer than one opinion, or domains differ.

    References:
        Jøsang (2016), §12.3 (Cumulative Fusion).
    """
    if len(opinions) == 0:
        raise ValueError(
            "multinomial_cumulative_fuse requires at least one opinion"
        )
    if len(opinions) == 1:
        return opinions[0]

    result = opinions[0]
    for i in range(1, len(opinions)):
        result = _multinomial_cumulative_fuse_pair(result, opinions[i])
    return result


def _multinomial_cumulative_fuse_pair(
    a: MultinomialOpinion,
    b: MultinomialOpinion,
) -> MultinomialOpinion:
    """Cumulative fusion of exactly two multinomial opinions."""
    # Validate same domain
    if a.domain != b.domain:
        raise ValueError(
            f"Cannot fuse opinions with different domains: "
            f"{a.domain} vs {b.domain}"
        )

    domain = a.domain
    u_a, u_b = a.uncertainty, b.uncertainty

    if u_a == 0.0 and u_b == 0.0:
        # Dogmatic case: limit with equal weight γ_A = γ_B = 0.5
        fused_beliefs = {
            x: 0.5 * a.beliefs[x] + 0.5 * b.beliefs[x]
            for x in domain
        }
        fused_u = 0.0
    else:
        # Standard cumulative fusion
        kappa = u_a + u_b - u_a * u_b  # > 0 if at least one u > 0
        fused_beliefs = {
            x: (a.beliefs[x] * u_b + b.beliefs[x] * u_a) / kappa
            for x in domain
        }
        fused_u = (u_a * u_b) / kappa

    # Average base rates
    fused_base_rates = {
        x: (a.base_rates[x] + b.base_rates[x]) / 2.0
        for x in domain
    }

    return MultinomialOpinion(
        beliefs=fused_beliefs,
        uncertainty=fused_u,
        base_rates=fused_base_rates,
    )


# ═══════════════════════════════════════════════════════════════════
# MULTINOMIAL DEDUCTION
# ═══════════════════════════════════════════════════════════════════


def multinomial_deduce(
    parent: MultinomialOpinion,
    conditionals: Mapping[str, MultinomialOpinion],
) -> MultinomialOpinion:
    """Multinomial deduction — conditional reasoning under uncertainty.

    Generalizes binomial ``deduce()`` to k-ary parent and child domains.

    Given a parent opinion ω_X over domain X = {x_1, ..., x_k} and
    conditional opinions ω_{Y|x_i} for each parent state x_i, each
    over the same child domain Y = {y_1, ..., y_m}, compute the
    deduced opinion ω_Y about Y.

    For each child state y_j:

        b_Y(y_j) = Σ_i b_X(x_i) · b_{Y|x_i}(y_j)
                 + u_X · Σ_i a_X(x_i) · b_{Y|x_i}(y_j)

        u_Y = Σ_i b_X(x_i) · u_{Y|x_i}
            + u_X · Σ_i a_X(x_i) · u_{Y|x_i}

        a_Y(y_j) = Σ_i a_X(x_i) · P_{Y|x_i}(y_j)

    Properties:
        - **Additivity**: Σ b_Y(y_j) + u_Y = 1  (always).
        - **Classical limit**: When all opinions are dogmatic (u=0),
          reduces to the law of total probability.
        - **Component-wise LTP**: For each component c ∈ {b, u}:
              c_Y(y) = Σ_i P_X(x_i) · c_{Y|x_i}(y)
          This is exact.

    Note on projected probability:
        P_Y(y) = b_Y(y) + a_Y(y)·u_Y does NOT in general equal
        Σ_i P_X(x_i) · P_{Y|x_i}(y).  The discrepancy arises because
        a_Y is weighted by parent base rates a_X(x_i), while u_Y is
        weighted by projected probabilities P_X(x_i).  The product
        a_Y·u_Y mixes these two weighting schemes.  The projected
        probability LTP does hold in the dogmatic limit (all u=0).
        This is consistent with the binomial deduce() operator.

    Args:
        parent: Opinion about the parent variable X.
        conditionals: Mapping from each parent state x_i to the
            conditional opinion ω_{Y|x_i} about Y.  Must cover
            all states in parent.domain, and all conditional
            opinions must share the same child domain.

    Returns:
        Deduced MultinomialOpinion about Y.

    Raises:
        ValueError: If conditionals don't cover all parent states,
            or child domains are inconsistent.

    References:
        Jøsang (2016), Ch. 9 (Multinomial Deduction), §12.6 (Binomial).
    """
    # ── Validate conditionals cover all parent states ──
    missing = set(parent.domain) - set(conditionals.keys())
    if missing:
        raise ValueError(
            f"Missing conditional opinions for parent states: "
            f"{sorted(missing)}"
        )

    # ── Validate all conditionals share the same child domain ──
    child_domains = [
        conditionals[x].domain for x in parent.domain
    ]
    if len(set(child_domains)) != 1:
        raise ValueError(
            f"All conditional opinions must share the same child domain. "
            f"Got: {dict(zip(parent.domain, child_domains))}"
        )
    child_domain = child_domains[0]

    # ── Compute deduced beliefs ──
    # b_Y(y_j) = Σ_i b_X(x_i) · b_{Y|x_i}(y_j)
    #          + u_X · Σ_i a_X(x_i) · b_{Y|x_i}(y_j)
    deduced_beliefs: dict[str, float] = {}
    for y in child_domain:
        evidence_sum = sum(
            parent.beliefs[x] * conditionals[x].beliefs[y]
            for x in parent.domain
        )
        prior_sum = sum(
            parent.base_rates[x] * conditionals[x].beliefs[y]
            for x in parent.domain
        )
        deduced_beliefs[y] = evidence_sum + parent.uncertainty * prior_sum

    # ── Compute deduced uncertainty ──
    # u_Y = Σ_i b_X(x_i) · u_{Y|x_i}
    #     + u_X · Σ_i a_X(x_i) · u_{Y|x_i}
    evidence_u = sum(
        parent.beliefs[x] * conditionals[x].uncertainty
        for x in parent.domain
    )
    prior_u = sum(
        parent.base_rates[x] * conditionals[x].uncertainty
        for x in parent.domain
    )
    deduced_u = evidence_u + parent.uncertainty * prior_u

    # ── Compute deduced base rates ──
    # a_Y(y_j) = Σ_i a_X(x_i) · P_{Y|x_i}(y_j)
    deduced_base_rates: dict[str, float] = {}
    for y in child_domain:
        deduced_base_rates[y] = sum(
            parent.base_rates[x] * conditionals[x].projected_probability()[y]
            for x in parent.domain
        )

    return MultinomialOpinion(
        beliefs=deduced_beliefs,
        uncertainty=deduced_u,
        base_rates=deduced_base_rates,
    )
