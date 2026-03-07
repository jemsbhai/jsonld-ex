"""
Counterfactual strategies for SL deduction.

When an ``SLEdge`` has ``counterfactual=None``, inference must compute
ω_{y|¬x} (the opinion about y given x is false) from the conditional
ω_{y|x} (the opinion about y given x is true).

This module provides three canonical strategies and a registry for
user-defined strategies.

Strategies:
    **vacuous** (default):
        ω_{y|¬x} = (0, 0, 1, a_{y|x}).
        When x is false, we have *no information* about y.
        This is the most conservative strategy — it maximizes
        uncertainty about the consequent when the antecedent fails.
        Mathematically, this means the deduced opinion's uncertainty
        reflects both the antecedent's uncertainty AND the lack of
        any counterfactual relationship.

    **adversarial**:
        ω_{y|¬x} = (d_{y|x}, b_{y|x}, u_{y|x}, 1 - a_{y|x}).
        When x is false, the evidence inverts: what supported y now
        opposes it, and vice versa.  This models strict causal
        opposition — the antecedent's absence has the *opposite*
        effect of its presence.  Base rate also inverts.

    **prior**:
        ω_{y|¬x} = (0, 0, 1, a_{y|x}).
        Identical to vacuous in the binomial case.  Semantically
        distinct: "revert to the prior" vs "complete ignorance."
        Included for API clarity and future extension to multinomial
        opinions where the distinction matters.

Mathematical Note:
    The choice of counterfactual strategy affects the deduced opinion
    ω_y through the law of total probability generalization:

        c_y = b_x · c_{y|x} + d_x · c_{y|¬x}
            + u_x · (a_x · c_{y|x} + ā_x · c_{y|¬x})

    where c ∈ {b, d, u}.  The counterfactual only contributes when
    d_x > 0 or u_x > 0 (i.e., when the antecedent is not certain).

References:
    Jøsang, A. (2016). Subjective Logic, §12.6 (deduction operator).
"""

from __future__ import annotations

from typing import Callable, Dict

from jsonld_ex.confidence_algebra import Opinion


# ═══════════════════════════════════════════════════════════════════
# TYPE ALIAS
# ═══════════════════════════════════════════════════════════════════

CounterfactualFn = Callable[[Opinion], Opinion]
"""A function that computes a counterfactual opinion from a conditional.

Takes ω_{y|x} and returns ω_{y|¬x}.  Must return a valid Opinion
(b + d + u = 1, all components in [0, 1]).
"""


# ═══════════════════════════════════════════════════════════════════
# STRATEGIES
# ═══════════════════════════════════════════════════════════════════


def vacuous_counterfactual(conditional: Opinion) -> Opinion:
    """Return a vacuous opinion — complete ignorance about y given ¬x.

    ω_{y|¬x} = (0, 0, 1, a_{y|x})

    This is the most conservative counterfactual: when the antecedent
    is false, we know nothing about the consequent.  The base rate is
    preserved from the conditional to maintain consistency in the
    base rate computation of the deduced opinion.

    Args:
        conditional: The conditional opinion ω_{y|x}.

    Returns:
        Vacuous opinion with the same base rate.
    """
    return Opinion(
        belief=0.0,
        disbelief=0.0,
        uncertainty=1.0,
        base_rate=conditional.base_rate,
    )


def adversarial_counterfactual(conditional: Opinion) -> Opinion:
    """Return an adversarial opinion — evidence inverts given ¬x.

    ω_{y|¬x} = (d_{y|x}, b_{y|x}, u_{y|x}, 1 - a_{y|x})

    Models strict causal opposition: if the antecedent being true
    supports the consequent, then the antecedent being false opposes
    it with equal strength.  Uncertainty is preserved (our confidence
    in the *relationship* doesn't change, only its direction).

    The base rate inverts because the prior expectation of y also
    reverses when the causal relationship reverses.

    Args:
        conditional: The conditional opinion ω_{y|x}.

    Returns:
        Opinion with belief and disbelief swapped, inverted base rate.
    """
    return Opinion(
        belief=conditional.disbelief,
        disbelief=conditional.belief,
        uncertainty=conditional.uncertainty,
        base_rate=1.0 - conditional.base_rate,
    )


def prior_counterfactual(conditional: Opinion) -> Opinion:
    """Return a prior-based opinion — revert to the base rate given ¬x.

    ω_{y|¬x} = (0, 0, 1, a_{y|x})

    Semantically: when the antecedent is false, our belief about
    the consequent reverts to its prior (base rate).  In the binomial
    case this is identical to vacuous_counterfactual.  The distinction
    matters for:
        - API clarity (the *intent* differs)
        - Future multinomial extension where prior ≠ vacuous

    Args:
        conditional: The conditional opinion ω_{y|x}.

    Returns:
        Vacuous opinion with the same base rate.
    """
    return Opinion(
        belief=0.0,
        disbelief=0.0,
        uncertainty=1.0,
        base_rate=conditional.base_rate,
    )


# ═══════════════════════════════════════════════════════════════════
# STRATEGY REGISTRY
# ═══════════════════════════════════════════════════════════════════

COUNTERFACTUAL_STRATEGIES: dict[str, CounterfactualFn] = {
    "vacuous": vacuous_counterfactual,
    "adversarial": adversarial_counterfactual,
    "prior": prior_counterfactual,
}
"""Registry of named counterfactual strategies.

Users can look up strategies by name or pass callables directly.
"""


def get_counterfactual_fn(
    strategy: str | CounterfactualFn,
) -> CounterfactualFn:
    """Resolve a counterfactual strategy by name or return it directly.

    Args:
        strategy: Either a string naming a built-in strategy
                  (``"vacuous"``, ``"adversarial"``, ``"prior"``)
                  or a callable ``(Opinion) -> Opinion``.

    Returns:
        The counterfactual function.

    Raises:
        ValueError: If the string name is not recognized.
        TypeError: If the argument is neither a string nor callable.
    """
    if isinstance(strategy, str):
        if strategy not in COUNTERFACTUAL_STRATEGIES:
            raise ValueError(
                f"Unknown counterfactual strategy {strategy!r}. "
                f"Available: {sorted(COUNTERFACTUAL_STRATEGIES.keys())}"
            )
        return COUNTERFACTUAL_STRATEGIES[strategy]
    if callable(strategy):
        return strategy
    raise TypeError(
        f"strategy must be a string or callable, "
        f"got {type(strategy).__name__}"
    )
