# Confidence Algebra Guide

**Module:** `jsonld_ex.confidence_algebra`  
**Foundation:** Jøsang's Subjective Logic (2016)

## Why Not Just Scalar Confidence?

A scalar confidence score (e.g., 0.85) conflates two very different situations:

- **High evidence, some disagreement:** 85% of evidence supports the claim, 10% contradicts it, 5% is missing.
- **Low evidence, no disagreement:** 5% of evidence supports the claim, 0% contradicts it, 95% is missing.

Both produce a projected probability near 0.85, but they have radically different implications for decision-making. The Subjective Logic opinion ω = (b, d, u, a) separates these dimensions.

## Opinions

An opinion has four components:

| Component | Symbol | Meaning |
|-----------|--------|---------|
| Belief | b | Evidence FOR the proposition |
| Disbelief | d | Evidence AGAINST the proposition |
| Uncertainty | u | Absence of evidence |
| Base rate | a | Prior probability (default 0.5) |

**Constraint:** b + d + u = 1.0  
**Projected probability:** P(ω) = b + a·u

```python
from jsonld_ex import Opinion

# High evidence, some disagreement
sensor = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
sensor.projected_probability()  # 0.85

# Low evidence, no disagreement  
rumor = Opinion(belief=0.05, disbelief=0.0, uncertainty=0.95)
rumor.projected_probability()  # 0.525

# Same projected probability, very different uncertainty
```

## Fusion: Combining Evidence from Multiple Sources

### Cumulative Fusion — Independent Sources

When sources are independent (no shared evidence), cumulative fusion additively reduces uncertainty. This is the most common operator.

**Algebraic properties:** Commutative, associative, identity element (vacuous opinion).

```python
from jsonld_ex import Opinion, cumulative_fuse

sensor_a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
sensor_b = Opinion(belief=0.7, disbelief=0.05, uncertainty=0.25)

fused = cumulative_fuse(sensor_a, sensor_b)
# Uncertainty decreases: fused.uncertainty < min(0.1, 0.25)
# Order doesn't matter: cumulative_fuse(b, a) == cumulative_fuse(a, b)
```

### Averaging Fusion — Correlated Sources

When sources share evidence (e.g., two models trained on overlapping data), averaging fusion prevents double-counting.

**Algebraic properties:** Commutative, but NOT associative.

```python
from jsonld_ex import averaging_fuse

model_a = Opinion(belief=0.75, disbelief=0.05, uncertainty=0.2)
model_b = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)

avg = averaging_fuse(model_a, model_b)
# More conservative than cumulative — doesn't reduce uncertainty as aggressively
```

### Robust Fusion — Adversarial Environments

When some sources may be compromised, robust fusion detects and removes outliers before cumulative fusion.

```python
from jsonld_ex import Opinion, robust_fuse

agents = [
    Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),   # honest
    Opinion(belief=0.75, disbelief=0.05, uncertainty=0.2),  # honest
    Opinion(belief=0.01, disbelief=0.98, uncertainty=0.01), # adversarial
]

fused, removed = robust_fuse(agents)
# removed contains the adversarial agent's index
# fused reflects only the honest agents
```

## Trust Discounting — Propagating Through Trust Chains

If you don't fully trust a source, their opinion should be discounted proportionally. Full trust preserves the opinion; zero trust yields total uncertainty.

```python
from jsonld_ex import Opinion, trust_discount

# Bob's opinion about a claim
bobs_opinion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

# Alice's trust in Bob
alice_trusts_bob = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)

# Alice's derived opinion about the claim (via Bob)
alices_view = trust_discount(alice_trusts_bob, bobs_opinion)
# Higher uncertainty than Bob's direct opinion
```

## Deduction — Subjective Logic Modus Ponens

Given an opinion about X and conditional opinions about Y|X and Y|¬X, derive an opinion about Y (Jøsang Def. 12.6).

```python
from jsonld_ex import Opinion, deduce

calibrated = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
accurate_if_calibrated = Opinion(belief=0.95, disbelief=0.02, uncertainty=0.03)
accurate_if_not = Opinion(belief=0.3, disbelief=0.4, uncertainty=0.3)

reading_accurate = deduce(calibrated, accurate_if_calibrated, accurate_if_not)
```

## Temporal Decay — Evidence Aging

Old evidence is less reliable. Decay migrates belief and disbelief toward uncertainty over time, preserving the b/d ratio.

```python
from jsonld_ex import Opinion, decay_opinion, exponential_decay, linear_decay, step_decay

opinion = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)

# Three decay functions available
decayed = decay_opinion(opinion, elapsed_seconds=3600, half_life_seconds=7200)
decayed = decay_opinion(opinion, 3600, 7200, decay_fn=linear_decay)
decayed = decay_opinion(opinion, 3600, 7200, decay_fn=step_decay)
```

## Conflict Detection

```python
from jsonld_ex import Opinion, pairwise_conflict, conflict_metric

optimist = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
pessimist = Opinion(belief=0.05, disbelief=0.9, uncertainty=0.05)

pairwise_conflict(optimist, pessimist)  # High: con(A,B) = b_A·d_B + d_A·b_B

confused = Opinion(belief=0.45, disbelief=0.45, uncertainty=0.1)
conflict_metric(confused)  # High: 1 − |b−d| − u
```

## Scalar Bridge

Bridge legacy scalar scores to opinions, proving exact equivalence under default parameters.

```python
from jsonld_ex import combine_opinions_from_scalars, propagate_opinions_from_scalars

fused = combine_opinions_from_scalars([0.9, 0.85, 0.7], fusion="cumulative")

# With defaults, propagation equals scalar multiplication
propagated = propagate_opinions_from_scalars([0.9, 0.8, 0.95])
# propagated.projected_probability() ≈ 0.684
```

## Operators Reference

| Operator | Use Case | Assoc. | Commut. |
|----------|----------|:------:|:-------:|
| `cumulative_fuse` | Independent sources | ✓ | ✓ |
| `averaging_fuse` | Correlated sources | ✗ | ✓ |
| `robust_fuse` | Adversarial environments | — | ✓ |
| `trust_discount` | Trust chain propagation | ✓ | ✗ |
| `deduce` | Conditional reasoning | — | — |
| `pairwise_conflict` | Source disagreement | — | ✓ |
| `conflict_metric` | Internal conflict | — | — |
| `decay_opinion` | Evidence aging | — | — |
| `combine_opinions_from_scalars` | Scalar → opinion fusion | — | — |
| `propagate_opinions_from_scalars` | Scalar → trust chain | — | — |

## Further Reading

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer.
