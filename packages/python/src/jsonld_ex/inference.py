"""
Confidence Propagation and Multi-Source Combination for JSON-LD-Ex.

Novel ML contribution: no existing JSON-LD/RDF tool provides confidence
propagation through inference chains or principled multi-source
combination.  PROV-O tracks provenance but does not propagate or
combine confidence scores.

Supported operations:

  1. **Chain propagation** — When assertion Y is inferred from X via
     rule R, what is the combined confidence of Y?

  2. **Multi-source combination** — When sources A and B independently
     assert the same fact, what is the combined confidence?

  3. **Conflict resolution** — When sources disagree, which assertion
     should be preferred?

Methods are grounded in the uncertainty literature:
  - Noisy-OR gate (Pearl 1988)
  - Dempster–Shafer theory of evidence (Shafer 1976)
  - Bayesian posterior update
  - Dampened multiplication for long chains
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence

from jsonld_ex.ai_ml import _validate_confidence, get_confidence

# ── Data Structures ────────────────────────────────────────────────


@dataclass
class PropagationResult:
    """Result of a confidence propagation or combination operation."""

    score: float
    method: str
    input_scores: list[float]
    provenance_trail: list[str] = field(default_factory=list)

    def to_annotation(self) -> dict[str, Any]:
        """Emit a jsonld-ex annotation fragment."""
        return {
            "@confidence": round(self.score, 10),
            "@method": f"confidence-{self.method}",
            "@derivedFrom": self.provenance_trail or self.input_scores,
        }


@dataclass
class ConflictReport:
    """Detailed report from conflict resolution."""

    winner: dict[str, Any]
    strategy: str
    candidates: list[dict[str, Any]]
    confidence_scores: list[float]
    reason: str


# ═══════════════════════════════════════════════════════════════════
# CHAIN PROPAGATION
# ═══════════════════════════════════════════════════════════════════


def propagate_confidence(
    chain: Sequence[float],
    method: Literal["multiply", "bayesian", "min", "dampened"] = "multiply",
) -> PropagationResult:
    """Propagate confidence through an inference chain.

    Given a chain of confidence scores [c₁, c₂, …, cₙ] representing
    the confidence at each step of a derivation, compute the combined
    confidence of the final conclusion.

    Args:
        chain: Ordered confidence scores along the inference path.
               Each value must be in [0, 1].
        method:
            ``"multiply"``  — Product of all scores.  Conservative but
                can underestimate for long chains.
            ``"bayesian"``  — Sequential Bayesian update treating each
                score as the likelihood ratio P(E|H)/P(E|¬H) with a
                uniform prior.
            ``"min"``       — Weakest-link: returns min(chain).
            ``"dampened"``  — product^(1/√n).  Attenuates the
                over-penalisation of plain multiplication in long chains.

    Returns:
        PropagationResult with the combined score and metadata.

    Raises:
        ValueError: If chain is empty or scores are outside [0, 1].
        TypeError:  If scores are not numeric.

    Example::

        >>> propagate_confidence([0.9, 0.8]).score
        0.72
    """
    if len(chain) == 0:
        raise ValueError("Chain must contain at least one confidence score")

    scores = list(chain)
    for s in scores:
        _validate_confidence(s)

    if method == "multiply":
        result = _chain_multiply(scores)
    elif method == "bayesian":
        result = _chain_bayesian(scores)
    elif method == "min":
        result = min(scores)
    elif method == "dampened":
        result = _chain_dampened(scores)
    else:
        raise ValueError(
            f"Unknown propagation method: {method!r}. "
            f"Expected one of: multiply, bayesian, min, dampened"
        )

    return PropagationResult(
        score=result,
        method=method,
        input_scores=scores,
    )


# ── Chain helpers ──────────────────────────────────────────────────


def _chain_multiply(scores: list[float]) -> float:
    """Straight product: ∏ cᵢ."""
    result = 1.0
    for s in scores:
        result *= s
    return result


def _chain_bayesian(scores: list[float]) -> float:
    """Sequential Bayesian update with a uniform (0.5) prior.

    Each score cᵢ is treated as a likelihood ratio:
        odds_posterior = odds_prior × (cᵢ / (1 − cᵢ))

    The final posterior probability is returned.

    Edge cases:
      - cᵢ = 1.0 → treated as 0.9999 to avoid division by zero
      - cᵢ = 0.0 → treated as 0.0001 to avoid log(0)
    """
    log_odds = 0.0  # log-odds of uniform prior = log(0.5/0.5) = 0
    for c in scores:
        c_safe = max(1e-4, min(c, 1.0 - 1e-4))
        log_odds += math.log(c_safe / (1.0 - c_safe))
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


def _chain_dampened(scores: list[float]) -> float:
    """Dampened product: (∏ cᵢ)^(1/√n).

    For n = 1 this equals the score itself.
    As the chain grows, the exponent approaches 0, preventing the
    combined confidence from collapsing to near-zero.
    """
    n = len(scores)
    product = _chain_multiply(scores)
    if product == 0.0:
        return 0.0
    exponent = 1.0 / math.sqrt(n)
    return product ** exponent


# ═══════════════════════════════════════════════════════════════════
# MULTI-SOURCE COMBINATION
# ═══════════════════════════════════════════════════════════════════


def combine_sources(
    scores: Sequence[float],
    method: Literal["average", "max", "noisy_or", "dempster_shafer"] = "noisy_or",
) -> PropagationResult:
    """Combine confidence from multiple independent sources.

    When two or more sources independently assert the same fact, the
    combined confidence should generally be *higher* than any single
    source (unless sources are very unreliable).

    Args:
        scores: Confidence scores from each source, each in [0, 1].
        method:
            ``"average"``          — Arithmetic mean.
            ``"max"``              — Optimistic: highest score.
            ``"noisy_or"``         — 1 − ∏(1 − pᵢ).
                Probability that *at least one* source is correct,
                assuming independence (Pearl 1988).
            ``"dempster_shafer"``  — Dempster's rule of combination
                (Shafer 1976).  Models belief, disbelief, and
                uncertainty explicitly.

    Returns:
        PropagationResult with the combined score and metadata.

    Raises:
        ValueError: If scores is empty or values are outside [0, 1].

    Example::

        >>> combine_sources([0.9, 0.7]).score   # noisy_or
        0.97
    """
    if len(scores) == 0:
        raise ValueError("Scores must contain at least one value")

    scores_list = list(scores)
    for s in scores_list:
        _validate_confidence(s)

    if method == "average":
        result = sum(scores_list) / len(scores_list)
    elif method == "max":
        result = max(scores_list)
    elif method == "noisy_or":
        result = _noisy_or(scores_list)
    elif method == "dempster_shafer":
        result = _dempster_shafer(scores_list)
    else:
        raise ValueError(
            f"Unknown combination method: {method!r}. "
            f"Expected one of: average, max, noisy_or, dempster_shafer"
        )

    return PropagationResult(
        score=result,
        method=method,
        input_scores=scores_list,
    )


# ── Combination helpers ───────────────────────────────────────────


def _noisy_or(scores: list[float]) -> float:
    """Noisy-OR: 1 − ∏(1 − pᵢ).

    Models the probability that at least one of n independent sources
    is correct.  Standard in Bayesian network causal gates.
    """
    product_complement = 1.0
    for p in scores:
        product_complement *= (1.0 - p)
    return 1.0 - product_complement


def _dempster_shafer(scores: list[float]) -> float:
    """Dempster's rule of combination (pairwise).

    Each score pᵢ induces a basic probability assignment (BPA):
        m({True})  = pᵢ          (belief in the assertion)
        m(Θ)       = 1 − pᵢ      (uncertainty / uncommitted mass)

    Dempster's rule combines two BPAs:
        K   = m₁({T})·m₂({F}) + m₁({F})·m₂({T})   (conflict mass)
        Bel = (m₁({T})·m₂({T}) + m₁({T})·m₂(Θ) + m₁(Θ)·m₂({T})) / (1−K)

    Since our BPAs have no explicit disbelief (m({False}) = 0),
    conflict K = 0 and the formula simplifies.

    For n > 2 sources the combination is applied pairwise left to right
    (Dempster's rule is associative and commutative).
    """
    # BPA: belief (b), uncertainty (u).  No disbelief in this model.
    belief = scores[0]
    uncertainty = 1.0 - scores[0]

    for i in range(1, len(scores)):
        b2 = scores[i]
        u2 = 1.0 - scores[i]

        # Combined belief (no conflict mass because m({F}) = 0 for both)
        new_belief = belief * b2 + belief * u2 + uncertainty * b2
        new_uncertainty = uncertainty * u2

        total = new_belief + new_uncertainty
        if total == 0:
            belief = 0.0
            uncertainty = 1.0
        else:
            belief = new_belief / total
            uncertainty = new_uncertainty / total

    return belief


# ═══════════════════════════════════════════════════════════════════
# CONFLICT RESOLUTION
# ═══════════════════════════════════════════════════════════════════


def resolve_conflict(
    assertions: Sequence[dict[str, Any]],
    strategy: Literal["highest", "weighted_vote", "recency"] = "highest",
) -> ConflictReport:
    """Resolve conflicting assertions using confidence and metadata.

    Given multiple assertions for the same property from different
    sources, select a winner and produce an auditable report.

    Each assertion is a dict that **must** contain ``"@value"`` and
    ``"@confidence"``.  For the ``"recency"`` strategy, assertions
    should also contain ``"@extractedAt"`` (ISO 8601 timestamp).

    Args:
        assertions: Candidate assertions with confidence metadata.
        strategy:
            ``"highest"``       — Pick the assertion with the highest
                confidence score.  Ties broken by list order.
            ``"weighted_vote"`` — Group assertions by ``@value``.
                Within each group, combine confidence via noisy-OR.
                The group with the highest combined score wins.
            ``"recency"``       — Prefer the most recently extracted
                assertion, using confidence as a tiebreaker.

    Returns:
        ConflictReport with the winner, strategy used, and reason.

    Raises:
        ValueError: If assertions is empty.
        KeyError:   If a required field is missing from an assertion.

    Example::

        >>> a = [{"@value": "Eng", "@confidence": 0.9},
        ...      {"@value": "Mgr", "@confidence": 0.85}]
        >>> resolve_conflict(a).winner["@value"]
        'Eng'
    """
    if len(assertions) == 0:
        raise ValueError("Assertions must be non-empty")

    candidates = list(assertions)
    conf_scores = []
    for a in candidates:
        c = a.get("@confidence")
        if c is None:
            raise KeyError(
                f"Assertion missing '@confidence': {a!r}"
            )
        _validate_confidence(c)
        conf_scores.append(c)

    if strategy == "highest":
        return _resolve_highest(candidates, conf_scores)
    elif strategy == "weighted_vote":
        return _resolve_weighted_vote(candidates, conf_scores)
    elif strategy == "recency":
        return _resolve_recency(candidates, conf_scores)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. "
            f"Expected one of: highest, weighted_vote, recency"
        )


# ── Resolution helpers ─────────────────────────────────────────────


def _resolve_highest(
    candidates: list[dict], conf_scores: list[float]
) -> ConflictReport:
    best_idx = 0
    for i in range(1, len(candidates)):
        if conf_scores[i] > conf_scores[best_idx]:
            best_idx = i
    return ConflictReport(
        winner=candidates[best_idx],
        strategy="highest",
        candidates=candidates,
        confidence_scores=conf_scores,
        reason=f"Highest confidence: {conf_scores[best_idx]:.4f}",
    )


def _resolve_weighted_vote(
    candidates: list[dict], conf_scores: list[float]
) -> ConflictReport:
    """Group by @value, combine via noisy-OR, pick best group."""
    groups: dict[Any, list[float]] = {}
    group_assertions: dict[Any, list[dict]] = {}

    for assertion, score in zip(candidates, conf_scores):
        val = assertion.get("@value")
        # Use a hashable key: convert unhashable types to str
        try:
            key = val
            hash(key)
        except TypeError:
            key = str(val)
        groups.setdefault(key, []).append(score)
        group_assertions.setdefault(key, []).append(assertion)

    best_key = None
    best_combined = -1.0
    for key, group_scores in groups.items():
        combined = _noisy_or(group_scores) if len(group_scores) > 1 else group_scores[0]
        if combined > best_combined:
            best_combined = combined
            best_key = key

    # Pick the highest-confidence individual assertion from the winning group
    winning_group = group_assertions[best_key]
    winning_scores = groups[best_key]
    best_in_group = max(range(len(winning_group)), key=lambda i: winning_scores[i])

    winner = dict(winning_group[best_in_group])
    winner["@confidence"] = best_combined  # reflect combined score

    n_supporters = len(winning_group)
    return ConflictReport(
        winner=winner,
        strategy="weighted_vote",
        candidates=candidates,
        confidence_scores=conf_scores,
        reason=(
            f"Value {best_key!r} supported by {n_supporters} source(s) "
            f"with combined noisy-OR confidence: {best_combined:.4f}"
        ),
    )


def _resolve_recency(
    candidates: list[dict], conf_scores: list[float]
) -> ConflictReport:
    """Prefer the most recently extracted assertion."""
    # Sort by extractedAt descending, then confidence descending as tiebreaker
    indexed = list(enumerate(candidates))

    def sort_key(item: tuple[int, dict]) -> tuple[str, float]:
        idx, a = item
        timestamp = a.get("@extractedAt", "")
        if not isinstance(timestamp, str):
            timestamp = str(timestamp)
        return (timestamp, conf_scores[idx])

    indexed.sort(key=sort_key, reverse=True)
    best_idx, best = indexed[0]

    return ConflictReport(
        winner=best,
        strategy="recency",
        candidates=candidates,
        confidence_scores=conf_scores,
        reason=(
            f"Most recent assertion (extractedAt={best.get('@extractedAt', 'N/A')}) "
            f"with confidence {conf_scores[best_idx]:.4f}"
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# GRAPH-LEVEL HELPERS
# ═══════════════════════════════════════════════════════════════════


def propagate_graph_confidence(
    doc: dict[str, Any],
    property_chain: Sequence[str],
    method: Literal["multiply", "bayesian", "min", "dampened"] = "multiply",
) -> PropagationResult:
    """Propagate confidence along a property chain in a JSON-LD graph.

    Given a document and a chain of property names, extract the
    confidence at each step and propagate.

    Args:
        doc: JSON-LD document (compact form).
        property_chain: Ordered property names forming the inference path.
        method: Propagation method (see :func:`propagate_confidence`).

    Returns:
        PropagationResult with combined confidence and property trail.

    Raises:
        KeyError: If a property in the chain is not found in the document.

    Example::

        >>> doc = {
        ...     "source_fact": {"@value": "X", "@confidence": 0.9},
        ...     "inferred": {"@value": "Y", "@confidence": 0.8},
        ... }
        >>> propagate_graph_confidence(doc, ["source_fact", "inferred"]).score
        0.72
    """
    scores: list[float] = []
    trail: list[str] = []

    current = doc
    for prop in property_chain:
        if not isinstance(current, dict):
            raise KeyError(
                f"Cannot traverse property {prop!r}: current node is not a dict"
            )
        value = current.get(prop)
        if value is None:
            raise KeyError(f"Property {prop!r} not found in document")

        # Extract confidence from this step
        c = get_confidence(value) if isinstance(value, dict) else None
        if c is None:
            # If there's a nested document, traverse into it but mark
            # the step as fully confident (no uncertainty at this link)
            c = 1.0

        scores.append(c)
        trail.append(prop)

        # Move into nested node for next step
        if isinstance(value, dict):
            # If it's an annotated value, we stay at current level for the
            # next property.  If it has sub-properties (like a node), use it.
            if "@value" in value and len(value) <= 5:
                # Annotated leaf — next property should come from parent
                pass
            else:
                current = value

    result = propagate_confidence(scores, method=method)
    result.provenance_trail = trail
    return result
