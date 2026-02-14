# jsonld-ex Confidence Algebra Specification

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/confidence-algebra](https://jsonld-ex.github.io/ns/confidence-algebra)

---

## 1. Introduction

This document specifies the formal confidence algebra underlying the jsonld-ex extension. The algebra provides a mathematically rigorous framework for representing, combining, propagating, and decaying uncertainty in AI/ML-generated linked data.

The algebra is grounded in **Jøsang's Subjective Logic** (Jøsang, 2016), a framework for probabilistic reasoning under epistemic uncertainty. Subjective Logic extends classical probability theory by explicitly modeling the *absence of evidence* as a first-class quantity, enabling downstream systems to distinguish between confident assertions and uninformed defaults — a distinction that scalar confidence scores conflate.

### 1.1 Motivation

A scalar confidence score c ∈ [0,1] conflates two distinct epistemic states:

1. **Informed moderate confidence** — "We have substantial evidence, and it points to a probability near c."
2. **Uninformed prior** — "We have no evidence; c reflects our default assumption."

These states demand different downstream behavior. An agentic system encountering c = 0.5 should act differently depending on whether that 0.5 reflects a balanced assessment of extensive evidence or complete ignorance. The Opinion model distinguishes these cases.

**Concrete example:** Consider two medical diagnostic models that both output confidence 0.7 for "patient has condition X":

- **Model A** was trained on 100,000 similar cases and has high evidential support: ω_A = (0.60, 0.20, 0.20, 0.50). P(ω_A) = 0.60 + 0.50 × 0.20 = 0.70.
- **Model B** was applied outside its training distribution and has weak evidential support: ω_B = (0.10, 0.05, 0.85, 0.70). P(ω_B) = 0.10 + 0.70 × 0.85 = 0.695.

Both project to approximately 0.7 probability, but the downstream decision should differ: Model A's assertion warrants action; Model B's warrants further testing. The uncertainty component (u = 0.20 vs u = 0.85) captures this distinction.

### 1.2 Design Rationale: Why Subjective Logic?

Several frameworks model uncertainty beyond scalar probability. We selected Subjective Logic over alternatives for the following reasons:

| Framework | Strengths | Limitations for JSON-LD |
|-----------|-----------|------------------------|
| **Dempster-Shafer Theory** (Shafer, 1976) | Rich evidence model, power-set belief functions | Computationally expensive for large hypothesis spaces (2ⁿ); not naturally scoped to binary propositions |
| **Fuzzy Logic** (Zadeh, 1965) | Handles vagueness and gradual membership | Models linguistic imprecision, not epistemic uncertainty; no evidence accumulation semantics |
| **Bayesian Networks** (Pearl, 1988) | Principled probabilistic reasoning | Requires full joint distribution; no compact single-assertion representation |
| **Imprecise Probabilities** (Walley, 1991) | Interval-valued; robust | No standard composition operators analogous to fusion/discount |
| **Subjective Logic** (Jøsang, 2016) | Compact binary opinion; closed-form operators; maps to/from scalar probability | Limited to binary/multinomial propositions (sufficient for assertion-level confidence) |

Subjective Logic was chosen because it provides:

1. **Compact representation** — A single tuple (b, d, u, a) per assertion, adding minimal overhead to JSON-LD documents.
2. **Closed-form operators** — Fusion, discount, and deduction have exact algebraic formulas, enabling efficient computation without sampling or iteration (except robust fusion).
3. **Backward compatibility** — Scalar confidence is a degenerate case (u = 0). Non-extended processors can use the projected probability P(ω) = b + au as a drop-in scalar.
4. **Semantic richness** — Distinguishes "informed 0.5" from "ignorant 0.5," enabling calibrated downstream decisions.
5. **Formal properties** — Operators preserve validity (additivity), have proven commutativity/associativity properties, and degrade gracefully.
6. **Evidence grounding** — Direct mapping from observation counts to opinions via the Beta distribution, connecting to Bayesian statistics.

### 1.3 Scope

This specification defines:

- The **Opinion** data type and its invariants (§2)
- **Projection** to scalar probability (§3)
- **Evidence mapping** from observation counts (§4)
- Six **algebraic operators**: cumulative fusion, averaging fusion, trust discount, deduction, pairwise conflict, and internal conflict (§5–§10)
- **Robust fusion** with Byzantine resistance (§11)
- **Temporal decay** modeling evidence aging (§12)
- **Bridge theorems** relating the algebra to classical scalar methods (§13)
- **JSON-LD serialization** format (§14)
- **Implementation notes** on floating-point handling and tolerances (§15)

### 1.4 Conformance

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.5 Notation

Throughout this document:

- Greek letter ω denotes an opinion.
- Subscripts identify the agent or source (e.g., ω_A is agent A's opinion).
- The components of ω are written as (b, d, u, a) for belief, disbelief, uncertainty, and base rate.
- The notation P(ω) denotes the projected probability of opinion ω.
- The symbol ⊕ denotes cumulative fusion, ⊘ denotes averaging fusion, and ⊗ denotes trust discount.
- The notation ā_x = 1 − a_x denotes the complement of a base rate.

---

## 2. Opinion Space

### 2.1 Definition

**Definition 2.1 (Opinion).** A *subjective opinion* about a binary proposition x is a tuple:

$$\omega_x = (b, d, u, a)$$

where:

| Component | Symbol | Range | Interpretation |
|-----------|--------|-------|----------------|
| Belief | b | [0, 1] | Evidence FOR the proposition |
| Disbelief | d | [0, 1] | Evidence AGAINST the proposition |
| Uncertainty | u | [0, 1] | Absence of evidence (ignorance) |
| Base rate | a | [0, 1] | Prior probability of the proposition |

subject to the **additivity constraint**:

$$b + d + u = 1$$

The opinion space Ω is the set of all tuples (b, d, u, a) satisfying the above constraints. Geometrically, the (b, d, u) components form a point in a 2-simplex (a triangle in 3D whose vertices represent full belief, full disbelief, and full uncertainty).

### 2.2 Geometric Interpretation

The opinion triangle is a barycentric coordinate system:

```
              u = 1
             (uncertainty)
              /\
             /  \
            /    \
           / ω_v  \      ω_v = vacuous opinion (0, 0, 1)
          /   ·    \
         /     \    \
        /       ·ω   \   ω = interior point (b, d, u)
       /         \    \
      /           \    \
     /             ·    \
    /_______________\____\
  b = 1            d = 1
 (belief)       (disbelief)

Bottom edge (u = 0): dogmatic opinions (classical probability)
Top vertex (u = 1): vacuous opinion (total ignorance)
```

- The **bottom edge** (u = 0) is the dogmatic subspace, equivalent to classical probability theory. Points on this edge are fully committed: b represents P(x) and d = 1 − P(x).
- The **top vertex** (u = 1) is the vacuous opinion — complete ignorance.
- **Interior points** have partial evidence and partial ignorance. The closer to the bottom edge, the more evidence has been observed.

### 2.3 Validation Rules

Conforming processors MUST enforce:

1. Each component b, d, u, a MUST be a finite number (not NaN, not ±Infinity).
2. Each component MUST be in the closed interval [0, 1].
3. The sum b + d + u MUST equal 1 within floating-point tolerance (|b + d + u − 1| ≤ ε, where ε = 10⁻⁹).
4. Boolean values MUST NOT be accepted as numeric components (Python's `True`/`False` are int subclasses and must be explicitly rejected).

### 2.4 Special Cases

**Definition 2.2 (Vacuous Opinion).** An opinion ω = (0, 0, 1, a) is *vacuous*. It represents total ignorance — no evidence exists for or against the proposition. The projected probability equals the base rate: P(ω) = a.

**Definition 2.3 (Dogmatic Opinion).** An opinion ω = (b, d, 0, a) with u = 0 is *dogmatic*. All probability mass is committed to belief or disbelief; no uncertainty remains. The projected probability equals the belief: P(ω) = b.

**Definition 2.4 (Absolute Opinion).** An opinion ω = (1, 0, 0, a) is *absolutely believing*; ω = (0, 1, 0, a) is *absolutely disbelieving*. These represent maximal epistemic commitment.

### 2.5 Relationship to Classical Probability

Classical probability theory operates in the dogmatic subspace (u = 0). The Opinion model is a strict generalization: every classical probability p corresponds to the dogmatic opinion (p, 1−p, 0, a) for any base rate a. The algebra is strictly more expressive than scalar confidence because it adds the uncertainty dimension.

### 2.6 Relationship to the Beta Distribution

The Opinion model has a direct connection to Bayesian statistics through the Beta distribution (Jøsang, 2016, §3.2).

Given r positive observations and s negative observations with prior weight W, the evidence mapping (see §4) produces an opinion (b, d, u, a) where:

$$b = \frac{r}{r + s + W}, \quad d = \frac{s}{r + s + W}, \quad u = \frac{W}{r + s + W}$$

The corresponding Beta distribution is Beta(α, β) where:

$$\alpha = r + W \cdot a, \quad \beta = s + W \cdot (1 - a)$$

The projected probability P(ω) = b + a·u equals the mean of this Beta distribution:

$$P(\omega) = \frac{r + W \cdot a}{r + s + W} = \frac{\alpha}{\alpha + \beta} = E[\text{Beta}(\alpha, \beta)]$$

This connection ensures that the Opinion model is statistically principled: opinions derived from evidence counts correspond exactly to Bayesian posterior estimates under a Beta-Binomial model.

---

## 3. Projected Probability

### 3.1 Definition

**Definition 3.1 (Projected Probability).** The *projected probability* of an opinion ω = (b, d, u, a) is:

$$P(\omega) = b + a \cdot u$$

This maps the three-dimensional opinion to a scalar probability by distributing the uncertainty mass according to the base rate. Intuitively: "what would the probability be if we resolved our uncertainty according to our prior?"

### 3.2 Properties

**Property 3.1 (Dogmatic reduction).** For dogmatic opinions (u = 0): P(ω) = b.

**Property 3.2 (Vacuous default).** For vacuous opinions (b = 0, d = 0, u = 1): P(ω) = a. Total ignorance defaults to the prior.

**Property 3.3 (Bounded).** P(ω) ∈ [0, 1] for all valid opinions.

*Proof:* Since b ≥ 0, a ≥ 0, u ≥ 0: P(ω) = b + au ≥ 0. Since b + u ≤ 1 (because d ≥ 0) and a ≤ 1: P(ω) = b + au ≤ b + u ≤ 1. ∎

**Property 3.4 (Monotonicity in belief).** For fixed d, u, a: P(ω) is strictly increasing in b. *Proof:* ∂P/∂b = 1 > 0. (Note: increasing b while fixing d and a requires decreasing u to maintain b + d + u = 1, so P changes by Δb − aΔb = (1−a)Δb ≥ 0.) ∎

### 3.3 Conversion from Scalar Confidence

**Definition 3.2 (from_confidence).** A scalar confidence value c ∈ [0, 1] is mapped to an opinion given a specified uncertainty u₀ ∈ [0, 1] and base rate a:

$$b = c \cdot (1 - u_0)$$
$$d = (1 - c) \cdot (1 - u_0)$$
$$u = u_0$$

This distributes the non-uncertain mass proportionally to c and (1 − c).

**Default behavior:** When u₀ = 0, this produces the dogmatic opinion (c, 1−c, 0, a), where P(ω) = c exactly.

**Verification of additivity:** b + d + u = c(1 − u₀) + (1 − c)(1 − u₀) + u₀ = (1 − u₀)(c + 1 − c) + u₀ = (1 − u₀) + u₀ = 1. ✓

### 3.4 Round-trip Properties

**Property 3.5 (Dogmatic round-trip).** For any scalar c ∈ [0, 1]:

$$\text{from\_confidence}(c, u_0 = 0).to\_confidence() = c$$

*Proof:* from_confidence(c, 0) = (c, 1−c, 0, a). to_confidence() = P(ω) = c + a · 0 = c. ∎

**Property 3.6 (Non-dogmatic round-trip).** For u₀ > 0, the round-trip is exact only when c = a. In general:

$$P(\text{from\_confidence}(c, u_0, a)) = c(1 - u_0) + a \cdot u_0$$

This equals c only when u₀ = 0 or when c = a. The difference |P − c| = u₀|a − c| quantifies the "projection error" introduced by distributing uncertainty according to the base rate.

*Practical implication:* When lifting scalar scores to opinions with uncertainty > 0, the projected probability shifts toward the base rate. This is the mathematically correct behavior — adding uncertainty should make the result less extreme — but users should be aware of it.

### 3.5 Worked Example

Given ω = (0.6, 0.1, 0.3, 0.5):

$$P(\omega) = 0.6 + 0.5 \times 0.3 = 0.6 + 0.15 = 0.75$$

Interpretation: "We have moderate belief (0.6), low disbelief (0.1), and some uncertainty (0.3). Given our neutral prior (0.5), the effective probability is 0.75."

Compare with ω' = (0.75, 0.25, 0.0, 0.5) — a dogmatic opinion with P(ω') = 0.75. Both project to 0.75, but ω has 30% uncertainty while ω' has none. Fusing ω with additional evidence will change the result more than fusing ω'.

---

## 4. Evidence Mapping

### 4.1 Definition

**Definition 4.1 (Evidence-to-Opinion Mapping).** Given r positive observations, s negative observations, and a non-informative prior weight W > 0 (Jøsang, 2016, §3.2):

$$b = \frac{r}{r + s + W}$$

$$d = \frac{s}{r + s + W}$$

$$u = \frac{W}{r + s + W}$$

The base rate a is specified independently (default: 0.5).

### 4.2 Properties

**Property 4.1 (Uncertainty reduction).** As evidence accumulates (r + s → ∞), uncertainty approaches zero: u → 0. The opinion becomes increasingly dogmatic.

**Property 4.2 (Vacuous limit).** With no evidence (r = s = 0): b = 0, d = 0, u = 1. The opinion is vacuous.

**Property 4.3 (Prior weight interpretation).** The parameter W controls how much evidence is needed to overcome the prior. W = 2 (default) means that 2 observations are equivalent in weight to the prior. Higher W makes the opinion more conservative (slower to commit).

**Property 4.4 (Additivity).** b + d + u = r/(r+s+W) + s/(r+s+W) + W/(r+s+W) = (r+s+W)/(r+s+W) = 1. ✓

### 4.3 Worked Example

A binary classifier evaluates 80 test samples: 72 positive, 8 negative. With W = 2:

$$b = \frac{72}{72 + 8 + 2} = \frac{72}{82} \approx 0.878$$

$$d = \frac{8}{82} \approx 0.098$$

$$u = \frac{2}{82} \approx 0.024$$

$$P(\omega) = 0.878 + 0.5 \times 0.024 = 0.890$$

Uncertainty is low (0.024) because we have substantial evidence. If instead we had only 7 positive and 1 negative (W = 2):

$$b = 7/10 = 0.700, \quad d = 1/10 = 0.100, \quad u = 2/10 = 0.200$$

$$P(\omega) = 0.700 + 0.5 \times 0.200 = 0.800$$

The uncertainty is much higher (0.200), correctly reflecting the smaller sample size.

### 4.4 Validation Rules

- r MUST be non-negative (r ≥ 0). Non-integer values are permitted (for weighted evidence).
- s MUST be non-negative (s ≥ 0).
- W MUST be strictly positive (W > 0).

---

## 5. Cumulative Fusion (⊕)

### 5.1 Purpose

Cumulative fusion combines **independent evidence** from multiple sources about the same proposition. When two independent observers each provide evidence, cumulative fusion aggregates their evidence additively, reducing overall uncertainty.

**When to use:** Multiple independent sensors measuring the same quantity; multiple ML models trained on non-overlapping data; independent annotators labeling the same example.

### 5.2 Binary Formula

**Definition 5.1 (Cumulative Fusion of Two Opinions).** Given ω_A = (b_A, d_A, u_A, a_A) and ω_B = (b_B, d_B, u_B, a_B):

**Case 1: At least one non-dogmatic opinion (u_A + u_B − u_A · u_B > 0):**

$$\kappa = u_A + u_B - u_A \cdot u_B$$

$$b_{A \oplus B} = \frac{b_A \cdot u_B + b_B \cdot u_A}{\kappa}$$

$$d_{A \oplus B} = \frac{d_A \cdot u_B + d_B \cdot u_A}{\kappa}$$

$$u_{A \oplus B} = \frac{u_A \cdot u_B}{\kappa}$$

**Case 2: Both dogmatic (u_A = u_B = 0):**

The standard formula has a 0/0 indeterminate form. Per Jøsang (2016, §12.3, Eq. 12.4), the limit with equal relative dogmatism (γ_A = γ_B = 0.5) yields:

$$b_{A \oplus B} = \gamma_A \cdot b_A + \gamma_B \cdot b_B = 0.5 \cdot b_A + 0.5 \cdot b_B$$

$$d_{A \oplus B} = 0.5 \cdot d_A + 0.5 \cdot d_B$$

$$u_{A \oplus B} = 0$$

**Derivation of dogmatic limit:** Consider the limit as u_A → 0⁺ and u_B → 0⁺ with u_A/u_B → γ_B/γ_A (the relative dogmatism ratio). In the standard formula, κ → u_A + u_B (since u_A·u_B is second-order). Then b → (b_A·u_B + b_B·u_A)/(u_A + u_B). With u_A/u_B = γ_B/γ_A, this gives b → γ_A·b_A + γ_B·b_B. The equal-weight case (γ_A = γ_B = 0.5) produces the simple average.

**Base rate:** The fused base rate is the arithmetic mean:

$$a_{A \oplus B} = \frac{a_A + a_B}{2}$$

### 5.3 N-ary Extension

For n > 2 opinions, cumulative fusion is applied by iterated pairwise fusion:

$$\omega_1 \oplus \omega_2 \oplus \cdots \oplus \omega_n = ((\omega_1 \oplus \omega_2) \oplus \omega_3) \oplus \cdots \oplus \omega_n$$

This is well-defined because cumulative fusion is commutative (Property 5.1) and associative (Property 5.3). The order of fusion does not affect the result.

### 5.4 Formal Properties

**Property 5.1 (Commutativity).**

$$\omega_A \oplus \omega_B = \omega_B \oplus \omega_A$$

*Proof:* The formula is symmetric in A and B. In the non-dogmatic case: κ is symmetric (u_A + u_B − u_A·u_B = u_B + u_A − u_B·u_A), and the numerators b_A·u_B + b_B·u_A and d_A·u_B + d_B·u_A are symmetric by commutativity of addition and multiplication. The dogmatic case (simple average) is also symmetric. ∎

**Property 5.2 (Vacuous Identity).**

$$\omega_A \oplus \omega_{vacuous} = \omega_A$$

where ω_vacuous = (0, 0, 1, a_v).

*Proof:* κ = u_A + 1 − u_A·1 = 1. So b = (b_A·1 + 0·u_A)/1 = b_A, d = (d_A·1 + 0·u_A)/1 = d_A, u = u_A·1/1 = u_A. The opinion is unchanged (the base rate averages with a_v, which is the only effect). ∎

**Property 5.3 (Associativity).**

$$(\omega_A \oplus \omega_B) \oplus \omega_C = \omega_A \oplus (\omega_B \oplus \omega_C)$$

This property is proven in Jøsang (2016, §12.3) by algebraic expansion of both sides and showing equality of all three components. It is also verified empirically in the reference implementation's property-based test suite using Hypothesis.

**Property 5.4 (Uncertainty Reduction).**

$$u_{A \oplus B} \leq \min(u_A, u_B)$$

*Proof (non-dogmatic case):* We have u_{A⊕B} = u_A·u_B / κ where κ = u_A + u_B − u_A·u_B. To show u_{A⊕B} ≤ u_A:

$$\frac{u_A \cdot u_B}{u_A + u_B - u_A \cdot u_B} \leq u_A$$

$$\Leftrightarrow u_A \cdot u_B \leq u_A(u_A + u_B - u_A \cdot u_B)$$

$$\Leftrightarrow u_B \leq u_A + u_B - u_A \cdot u_B$$

$$\Leftrightarrow 0 \leq u_A - u_A \cdot u_B = u_A(1 - u_B) \geq 0 \quad \checkmark$$

By symmetry, u_{A⊕B} ≤ u_B. Hence u_{A⊕B} ≤ min(u_A, u_B). ∎

**Remark.** Uncertainty strictly decreases when both opinions have u > 0. Combining independent evidence always makes us more certain (less uncertain).

**Property 5.5 (Additivity preservation).**

$$b_{A \oplus B} + d_{A \oplus B} + u_{A \oplus B} = 1$$

*Proof (non-dogmatic case):*

$$\frac{b_A u_B + b_B u_A + d_A u_B + d_B u_A + u_A u_B}{\kappa}$$

$$= \frac{u_B(b_A + d_A) + u_A(b_B + d_B) + u_A u_B}{\kappa}$$

$$= \frac{u_B(1 - u_A) + u_A(1 - u_B) + u_A u_B}{\kappa}$$

$$= \frac{u_B - u_A u_B + u_A - u_A u_B + u_A u_B}{\kappa} = \frac{u_A + u_B - u_A u_B}{\kappa} = \frac{\kappa}{\kappa} = 1 \quad \blacksquare$$

### 5.5 Worked Example

**Scenario:** Two independent NER models extract "Jane Doe" from a document.

- Model A: ω_A = (0.8, 0.1, 0.1, 0.5) — high confidence, low uncertainty.
- Model B: ω_B = (0.6, 0.1, 0.3, 0.5) — moderate confidence, more uncertain.

**Computation:**

$$\kappa = 0.1 + 0.3 - 0.1 \times 0.3 = 0.37$$

$$b = \frac{0.8 \times 0.3 + 0.6 \times 0.1}{0.37} = \frac{0.24 + 0.06}{0.37} = \frac{0.30}{0.37} \approx 0.811$$

$$d = \frac{0.1 \times 0.3 + 0.1 \times 0.1}{0.37} = \frac{0.03 + 0.01}{0.37} = \frac{0.04}{0.37} \approx 0.108$$

$$u = \frac{0.1 \times 0.3}{0.37} = \frac{0.03}{0.37} \approx 0.081$$

**Result:** ω_{A⊕B} ≈ (0.811, 0.108, 0.081, 0.50).

**Interpretation:**
- Belief increased from max(0.8, 0.6) = 0.8 to 0.811 — combining evidence strengthened the belief.
- Uncertainty decreased from min(0.1, 0.3) = 0.1 to 0.081 — we are more certain than either individual source.
- P(ω_{A⊕B}) = 0.811 + 0.5 × 0.081 ≈ 0.851 > max(P(ω_A), P(ω_B)) = max(0.85, 0.75) = 0.85.

---

## 6. Averaging Fusion (⊘)

### 6.1 Purpose

Averaging fusion combines **dependent or correlated** sources. Unlike cumulative fusion, it does not double-count evidence. When sources may have observed the same underlying data, averaging fusion avoids artificially inflating confidence.

**When to use:** Multiple models trained on overlapping data; redundant sensors reading from the same physical process; aggregating expert opinions that share a common information base.

### 6.2 Binary Formula

**Definition 6.1 (Averaging Fusion of Two Opinions).** Given ω_A = (b_A, d_A, u_A, a_A) and ω_B = (b_B, d_B, u_B, a_B) with equal weight:

**Case 1: At least one non-dogmatic (u_A + u_B > 0):**

$$\kappa = u_A + u_B$$

$$b_{A \oslash B} = \frac{b_A \cdot u_B + b_B \cdot u_A}{\kappa}$$

$$d_{A \oslash B} = \frac{d_A \cdot u_B + d_B \cdot u_A}{\kappa}$$

$$u_{A \oslash B} = \frac{2 \cdot u_A \cdot u_B}{\kappa}$$

**Case 2: Both dogmatic (u_A = u_B = 0):**

$$b_{A \oslash B} = \frac{b_A + b_B}{2}, \quad d_{A \oslash B} = \frac{d_A + d_B}{2}, \quad u_{A \oslash B} = 0$$

**Additivity verification (Case 1):**

$$\frac{b_A u_B + b_B u_A + d_A u_B + d_B u_A + 2 u_A u_B}{u_A + u_B}$$

$$= \frac{u_B(b_A + d_A) + u_A(b_B + d_B) + 2 u_A u_B}{u_A + u_B}$$

$$= \frac{u_B(1 - u_A) + u_A(1 - u_B) + 2 u_A u_B}{u_A + u_B} = \frac{u_A + u_B}{u_A + u_B} = 1 \quad \checkmark$$

**Base rate:**

$$a_{A \oslash B} = \frac{a_A + a_B}{2}$$

### 6.3 N-ary Simultaneous Formula

**Definition 6.2 (N-ary Averaging Fusion).** For n ≥ 3 opinions ω_1, ..., ω_n with equal weight (Jøsang, 2016, §12.5):

$$U_i = \prod_{j \neq i} u_j \quad \text{(product of all OTHER uncertainties)}$$

$$\kappa = \sum_{i=1}^{n} U_i$$

$$b = \frac{\sum_{i=1}^{n} b_i \cdot U_i}{\kappa}$$

$$d = \frac{\sum_{i=1}^{n} d_i \cdot U_i}{\kappa}$$

$$u = \frac{n \cdot \prod_{i=1}^{n} u_i}{\kappa}$$

**Base rate:** a = (1/n) Σ a_i.

**Efficient computation of U_i:** When all u_i > 0, compute the full product P = ∏u_i, then U_i = P / u_i. When u_i = 0 for some i, compute U_i = ∏_{j≠i} u_j directly, noting that U_i = 0 whenever any other u_j = 0.

**Additivity verification:**

$$\frac{\sum_i b_i U_i + \sum_i d_i U_i + n \prod_i u_i}{\kappa} = \frac{\sum_i (b_i + d_i) U_i + n \prod_i u_i}{\kappa}$$

$$= \frac{\sum_i (1 - u_i) U_i + n \prod_i u_i}{\kappa} = \frac{\sum_i U_i - \sum_i u_i U_i + n \prod_i u_i}{\kappa}$$

Note that u_i U_i = u_i ∏_{j≠i} u_j = ∏_j u_j for each i. So Σ u_i U_i = n ∏_j u_j. Therefore:

$$= \frac{\kappa - n \prod u_i + n \prod u_i}{\kappa} = \frac{\kappa}{\kappa} = 1 \quad \checkmark$$

### 6.4 Dogmatic Fallback (κ = 0)

When κ = 0, the n-ary formula has an indeterminate form 0/0. This occurs when at least two opinions are dogmatic (u_i = 0).

**Derivation of the limit:** Let D = {i : u_i = 0} be the set of dogmatic opinions, with |D| ≥ 2. Let N = {i : u_i > 0} be the non-dogmatic opinions. Consider the limit as the dogmatic opinions' uncertainties approach zero: u_i → 0⁺ for i ∈ D.

For a dogmatic opinion i ∈ D:

$$U_i = \prod_{j \neq i} u_j = \prod_{j \in D \setminus \{i\}} u_j \cdot \prod_{j \in N} u_j$$

The first product has |D| − 1 terms approaching zero, so U_i is of order ε^{|D|−1} · C where C = ∏_{j∈N} u_j is a positive constant.

For a non-dogmatic opinion k ∈ N:

$$U_k = \prod_{j \neq k} u_j = \prod_{j \in D} u_j \cdot \prod_{j \in N \setminus \{k\}} u_j$$

This has |D| terms approaching zero, so U_k is of order ε^{|D|} · C', which is a higher power of ε than U_i. Therefore U_k/U_i → 0 as ε → 0: the non-dogmatic opinions' weights vanish relative to the dogmatic ones.

In the limit, only the dogmatic opinions contribute with equal weight among themselves:

$$b = \frac{1}{|D|} \sum_{i \in D} b_i, \quad d = \frac{1}{|D|} \sum_{i \in D} d_i, \quad u = 0$$

### 6.5 Formal Properties

**Property 6.1 (Commutativity).**

$$\omega_A \oslash \omega_B = \omega_B \oslash \omega_A$$

*Proof:* Both the binary and n-ary formulas are symmetric in their inputs. The U_i and κ computations involve only products and sums, which are commutative. ∎

**Property 6.2 (Idempotence).**

$$\underbrace{\omega \oslash \omega \oslash \cdots \oslash \omega}_{n \text{ copies}} = \omega$$

*Proof:* For n copies of ω = (b, d, u, a): U_i = u^{n−1} for all i. κ = n · u^{n−1}. Then:

$$b_{\text{fused}} = \frac{\sum_{i=1}^{n} b \cdot u^{n-1}}{n \cdot u^{n-1}} = \frac{n \cdot b \cdot u^{n-1}}{n \cdot u^{n-1}} = b$$

Similarly d_fused = d. And:

$$u_{\text{fused}} = \frac{n \cdot u^n}{n \cdot u^{n-1}} = u$$

All components are preserved. ∎

*Significance:* Idempotence means that averaging the same source with itself does not change the result — exactly the right behavior for correlated sources that share the same underlying evidence.

**Property 6.3 (Non-associativity). ⚠ CRITICAL WARNING.**

$$(\omega_A \oslash \omega_B) \oslash \omega_C \neq \omega_A \oslash \omega_B \oslash \omega_C \quad \text{(in general)}$$

Averaging fusion is NOT associative for n > 2. The simultaneous n-ary formula (Definition 6.2) MUST be used for three or more opinions. Using pairwise iteration is a mathematical error that produces quantitatively different results.

*Counterexample:* Let ω_A = (0.8, 0.1, 0.1, 0.5), ω_B = (0.3, 0.5, 0.2, 0.5), ω_C = (0.6, 0.2, 0.2, 0.5).

**Pairwise left-fold:**

Step 1: ω_A ⊘ ω_B. κ = 0.1 + 0.2 = 0.3. b = (0.8×0.2 + 0.3×0.1)/0.3 = 0.19/0.3 ≈ 0.633. d = (0.1×0.2 + 0.5×0.1)/0.3 = 0.07/0.3 ≈ 0.233. u = 2×0.1×0.2/0.3 ≈ 0.133.

Step 2: ω' ⊘ ω_C. κ = 0.133 + 0.2 = 0.333. b = (0.633×0.2 + 0.6×0.133)/0.333 = (0.127 + 0.080)/0.333 ≈ 0.621. u = 2×0.133×0.2/0.333 ≈ 0.160.

**Simultaneous 3-ary formula:**

U_A = u_B · u_C = 0.2 × 0.2 = 0.04. U_B = u_A · u_C = 0.1 × 0.2 = 0.02. U_C = u_A · u_B = 0.1 × 0.2 = 0.02. κ = 0.08.

b = (0.8×0.04 + 0.3×0.02 + 0.6×0.02)/0.08 = (0.032 + 0.006 + 0.012)/0.08 = 0.050/0.08 = 0.625.

u = 3 × 0.1 × 0.2 × 0.2 / 0.08 = 3 × 0.004 / 0.08 = 0.012/0.08 = 0.150.

**Result:** Pairwise gives b ≈ 0.621, u ≈ 0.160; simultaneous gives b = 0.625, u = 0.150. The values differ.

### 6.6 Comparison with Cumulative Fusion

| Property | Cumulative (⊕) | Averaging (⊘) |
|----------|----------------|----------------|
| Evidence model | Independent sources | Correlated/dependent sources |
| Associativity | Yes | **No** (n-ary formula required) |
| Idempotence | No (A ⊕ A ≠ A) | Yes (A ⊘ A = A) |
| Uncertainty effect | Always reduces | May increase, decrease, or preserve |
| N-ary computation | Iterated pairwise (safe) | Simultaneous formula (required) |
| Use case | Multiple independent sensors | Redundant/overlapping models |
| Evidence accumulation | Yes (more sources = more evidence) | No (correlated evidence not double-counted) |

### 6.7 Worked Example

**Scenario:** Three pathologists examine the same biopsy slide (correlated evidence).

- Dr. A: ω_A = (0.7, 0.1, 0.2, 0.5)
- Dr. B: ω_B = (0.8, 0.1, 0.1, 0.5)
- Dr. C: ω_C = (0.6, 0.2, 0.2, 0.5)

**Simultaneous 3-ary formula:**

U_A = 0.1 × 0.2 = 0.02, U_B = 0.2 × 0.2 = 0.04, U_C = 0.2 × 0.1 = 0.02. κ = 0.08.

b = (0.7×0.02 + 0.8×0.04 + 0.6×0.02)/0.08 = (0.014 + 0.032 + 0.012)/0.08 = 0.058/0.08 = 0.725.

d = (0.1×0.02 + 0.1×0.04 + 0.2×0.02)/0.08 = (0.002 + 0.004 + 0.004)/0.08 = 0.010/0.08 = 0.125.

u = 3 × 0.2 × 0.1 × 0.2 / 0.08 = 3 × 0.004 / 0.08 = 0.150.

**Result:** ω ≈ (0.725, 0.125, 0.150, 0.50). P(ω) = 0.725 + 0.5 × 0.15 = 0.800.

**Interpretation:** The averaged opinion is close to the mean of the individual opinions, weighted by their respective uncertainties. Dr. B's lower uncertainty (0.1) gives it more weight than the others. Uncertainty is preserved, not reduced — correctly reflecting that the three pathologists share the same evidence base.

---

## 7. Trust Discount (⊗)

### 7.1 Purpose

Trust discount propagates an opinion through a trust chain. If agent A trusts agent B to some degree, and B holds an opinion about proposition x, then A can derive a discounted opinion about x that reflects both B's assessment and A's trust in B.

**When to use:** Propagating confidence through inference chains; weighting assertions from sources of varying reliability; multi-hop knowledge graphs where trust degrades along edges.

### 7.2 Formula

**Definition 7.1 (Trust Discount).** Given trust opinion ω_{A→B} = (b_T, d_T, u_T, a_T) representing A's trust in B, and B's opinion ω_{B:x} = (b_x, d_x, u_x, a_x) about proposition x:

$$b_{A:x} = b_T \cdot b_x$$

$$d_{A:x} = b_T \cdot d_x$$

$$u_{A:x} = d_T + u_T + b_T \cdot u_x$$

The base rate is preserved from the source opinion: a_{A:x} = a_x.

### 7.3 Additivity Verification

$$b_{A:x} + d_{A:x} + u_{A:x} = b_T b_x + b_T d_x + d_T + u_T + b_T u_x$$

$$= b_T(b_x + d_x + u_x) + d_T + u_T = b_T \cdot 1 + d_T + u_T = b_T + d_T + u_T = 1 \quad \checkmark$$

### 7.4 Intuition

- **Full trust** (b_T = 1, d_T = 0, u_T = 0): A adopts B's opinion unchanged, since b_{A:x} = b_x, d_{A:x} = d_x, u_{A:x} = u_x.
- **Zero trust** (b_T = 0): b_{A:x} = 0, d_{A:x} = 0, u_{A:x} = d_T + u_T = 1. The result is vacuous — A learns nothing from an untrusted source.
- **Partial trust**: B's opinion is "diluted" toward uncertainty. The higher the trust, the more of B's belief/disbelief is preserved.

### 7.5 Formal Properties

**Property 7.1 (Non-commutativity).**

$$\omega_{A \to B} \otimes \omega_{B:x} \neq \omega_{B:x} \otimes \omega_{A \to B} \quad \text{(in general)}$$

Trust discount is inherently directional: "A trusts B about x" is different from "x trusts B about A." The first operand is always the trust opinion and the second is the source opinion.

**Property 7.2 (Chain associativity).** For a trust chain A → B → C → x:

$$\omega_{A \to B} \otimes (\omega_{B \to C} \otimes \omega_{C:x}) = (\omega_{A \to B} \otimes \omega_{B \to C}) \otimes \omega_{C:x}$$

*Proof sketch:* Both sides yield b = b_{A→B} · b_{B→C} · b_{C:x} (the belief product), d = b_{A→B} · b_{B→C} · d_{C:x}, and the same uncertainty term (by algebraic expansion). See Jøsang (2016, §14.3) for the full proof. ∎

This means trust discount can be safely applied iteratively along a chain of arbitrary length.

**Property 7.3 (Vacuous trust annihilation).** If ω_{A→B} is vacuous (b_T = 0, d_T = 0, u_T = 1), then the result is vacuous regardless of B's opinion: ω_{A:x} = (0, 0, 1, a_x).

**Property 7.4 (Distrust yields vacuity).** If b_T = 0 (A has no belief in B, regardless of how disbelief and uncertainty split), then b_{A:x} = d_{A:x} = 0 and u_{A:x} = 1. An untrusted source contributes no information.

### 7.6 Worked Example

**Scenario:** A news aggregator (A) has moderate trust in a source (B), and B claims a stock will rise.

- Trust: ω_{A→B} = (0.7, 0.1, 0.2, 0.5) — mostly trusted but with some uncertainty.
- B's claim: ω_{B:x} = (0.9, 0.05, 0.05, 0.5) — B is very confident.

**Computation:**

$$b_{A:x} = 0.7 \times 0.9 = 0.63$$

$$d_{A:x} = 0.7 \times 0.05 = 0.035$$

$$u_{A:x} = 0.1 + 0.2 + 0.7 \times 0.05 = 0.335$$

**Result:** ω_{A:x} = (0.63, 0.035, 0.335, 0.50). P(ω) = 0.63 + 0.5 × 0.335 = 0.798.

**Interpretation:** B was very confident (P = 0.925), but after discounting by A's trust (belief 0.7), the derived probability drops to 0.798 and the uncertainty increases from 0.05 to 0.335. The trust gap introduces significant epistemic uncertainty — correctly reflecting that A shouldn't be as certain as B about B's own claim.

---

## 8. Deduction

### 8.1 Purpose

Deduction performs conditional reasoning under uncertainty. Given an opinion about an antecedent x and conditional opinions about a consequent y (conditioned on x being true or false), it derives an opinion about y. This is the subjective-logic generalization of the **law of total probability**.

**When to use:** If-then reasoning under uncertainty; computing P(disease | symptom) when both the symptom's presence and the conditional probability are uncertain; cascaded classifier outputs.

### 8.2 Formula

**Definition 8.1 (Deduction Operator).** Given:

- ω_x = (b_x, d_x, u_x, a_x) — opinion about antecedent x
- ω_{y|x} = (b_{y|x}, d_{y|x}, u_{y|x}, a_{y|x}) — conditional opinion about y given x is true
- ω_{y|¬x} = (b_{y|¬x}, d_{y|¬x}, u_{y|¬x}, a_{y|¬x}) — conditional opinion about y given x is false

Let ā_x = 1 − a_x. For each component c ∈ {b, d, u}:

$$c_y = b_x \cdot c_{y|x} + d_x \cdot c_{y|\neg x} + u_x \cdot (a_x \cdot c_{y|x} + \bar{a}_x \cdot c_{y|\neg x})$$

Explicitly:

$$b_y = b_x \cdot b_{y|x} + d_x \cdot b_{y|\neg x} + u_x \cdot (a_x \cdot b_{y|x} + \bar{a}_x \cdot b_{y|\neg x})$$

$$d_y = b_x \cdot d_{y|x} + d_x \cdot d_{y|\neg x} + u_x \cdot (a_x \cdot d_{y|x} + \bar{a}_x \cdot d_{y|\neg x})$$

$$u_y = b_x \cdot u_{y|x} + d_x \cdot u_{y|\neg x} + u_x \cdot (a_x \cdot u_{y|x} + \bar{a}_x \cdot u_{y|\neg x})$$

**Interpretation:** The formula partitions the probability space into three cases: x is true (weight b_x), x is false (weight d_x), and x is unknown (weight u_x). In the unknown case, x is assumed true with probability a_x and false with probability 1 − a_x.

**Base rate:**

$$a_y = a_x \cdot P(\omega_{y|x}) + \bar{a}_x \cdot P(\omega_{y|\neg x})$$

where P(ω) denotes the projected probability.

### 8.3 Formal Properties

**Property 8.1 (Additivity preservation).**

$$b_y + d_y + u_y = 1$$

*Proof:* Sum the three component formulas:

$$b_y + d_y + u_y = b_x \underbrace{(b_{y|x} + d_{y|x} + u_{y|x})}_{= 1} + d_x \underbrace{(b_{y|\neg x} + d_{y|\neg x} + u_{y|\neg x})}_{= 1} + u_x \cdot (a_x \cdot 1 + \bar{a}_x \cdot 1)$$

$$= b_x + d_x + u_x \cdot 1 = b_x + d_x + u_x = 1 \quad \blacksquare$$

**Property 8.2 (Classical limit).** When all three opinions are dogmatic (u_x = u_{y|x} = u_{y|¬x} = 0):

$$P(\omega_y) = b_y = b_x \cdot b_{y|x} + d_x \cdot b_{y|\neg x} = P(x) \cdot P(y|x) + (1 - P(x)) \cdot P(y|\neg x)$$

This is exactly the **law of total probability**.

*Proof:* With u_x = 0: the u_x·(...) term vanishes. With u_{y|x} = 0: b_{y|x} = P(y|x) and d_{y|x} = 1 − P(y|x). The result follows by substitution. ∎

**Property 8.3 (Projected probability consistency).**

$$P(\omega_y) = P(\omega_x) \cdot P(\omega_{y|x}) + (1 - P(\omega_x)) \cdot P(\omega_{y|\neg x})$$

This ensures that deduction is consistent with classical probability at the projection level, regardless of the uncertainty values. See Jøsang (2016, §12.6) for the full proof.

### 8.4 Worked Example

**Scenario:** We want to deduce whether a patient has a disease (y) given uncertain evidence of a symptom (x).

- Symptom presence: ω_x = (0.7, 0.1, 0.2, 0.3). "We're fairly sure the symptom is present."
- P(disease | symptom): ω_{y|x} = (0.8, 0.1, 0.1, 0.5). "If the symptom is present, disease is likely."
- P(disease | no symptom): ω_{y|¬x} = (0.1, 0.7, 0.2, 0.5). "If no symptom, disease is unlikely."

**Computation:**

$$b_y = 0.7 \times 0.8 + 0.1 \times 0.1 + 0.2 \times (0.3 \times 0.8 + 0.7 \times 0.1)$$

$$= 0.56 + 0.01 + 0.2 \times (0.24 + 0.07) = 0.56 + 0.01 + 0.062 = 0.632$$

$$d_y = 0.7 \times 0.1 + 0.1 \times 0.7 + 0.2 \times (0.3 \times 0.1 + 0.7 \times 0.7)$$

$$= 0.07 + 0.07 + 0.2 \times (0.03 + 0.49) = 0.07 + 0.07 + 0.104 = 0.244$$

$$u_y = 0.7 \times 0.1 + 0.1 \times 0.2 + 0.2 \times (0.3 \times 0.1 + 0.7 \times 0.2)$$

$$= 0.07 + 0.02 + 0.2 \times (0.03 + 0.14) = 0.07 + 0.02 + 0.034 = 0.124$$

**Check:** 0.632 + 0.244 + 0.124 = 1.000 ✓

**Base rate:**

$P(\omega_{y|x}) = 0.8 + 0.5 \times 0.1 = 0.85$

$P(\omega_{y|\neg x}) = 0.1 + 0.5 \times 0.2 = 0.20$

$a_y = 0.3 \times 0.85 + 0.7 \times 0.20 = 0.255 + 0.140 = 0.395$

**Result:** ω_y ≈ (0.632, 0.244, 0.124, 0.395). P(ω_y) = 0.632 + 0.395 × 0.124 ≈ 0.681.

---

## 9. Pairwise Conflict

### 9.1 Purpose

Pairwise conflict measures the degree to which two opinions *disagree* — specifically, how much one's belief overlaps with the other's disbelief. This metric is used by robust fusion (§11) to identify adversarial or outlier agents.

### 9.2 Formula

**Definition 9.1 (Pairwise Conflict).** For opinions ω_A = (b_A, d_A, u_A, a_A) and ω_B = (b_B, d_B, u_B, a_B):

$$\text{con}(\omega_A, \omega_B) = b_A \cdot d_B + d_A \cdot b_B$$

Per Jøsang (2016, §12.3.4).

### 9.3 Formal Properties

**Property 9.1 (Range).** con(ω_A, ω_B) ∈ [0, 1].

*Proof:* Since b, d ∈ [0, 1] and b + d ≤ 1 for each opinion:

$$b_A d_B + d_A b_B \leq b_A + d_A \leq 1$$

(because d_B ≤ 1 and b_B ≤ 1, so b_A d_B ≤ b_A and d_A b_B ≤ d_A). And b_A d_B + d_A b_B ≥ 0 since all components are non-negative. ∎

**Property 9.2 (Symmetry).** con(ω_A, ω_B) = con(ω_B, ω_A). Immediate from b_A d_B + d_A b_B = b_B d_A + d_B b_A.

**Property 9.3 (Zero conflict — full agreement).** If both opinions fully believe (b_A = b_B = 1, d_A = d_B = 0), then con = 1·0 + 0·1 = 0. Similarly if both fully disbelieve.

**Property 9.4 (Maximum conflict).** con = 1 when ω_A = (1, 0, 0, a) and ω_B = (0, 1, 0, a): one fully believes, the other fully disbelieves. con = 1·1 + 0·0 = 1.

**Property 9.5 (Vacuous ⇒ zero).** If either opinion is vacuous (b = d = 0, u = 1), then all products involving b or d of that opinion are zero, so con = 0. Ignorance cannot conflict with anything.

**Property 9.6 (Uncertainty reduces conflict).** For fixed belief directions (b/d ratios), increasing uncertainty in either opinion reduces conflict, because it reduces the magnitudes of b and d.

### 9.4 Worked Example

**Scenario:** Two annotators label a text as positive or negative sentiment.

- Annotator A: ω_A = (0.8, 0.1, 0.1, 0.5) — strongly believes positive.
- Annotator B: ω_B = (0.2, 0.6, 0.2, 0.5) — leans negative.

con(A, B) = 0.8 × 0.6 + 0.1 × 0.2 = 0.48 + 0.02 = 0.50. Significant conflict.

Compare with a more uncertain annotator C: ω_C = (0.1, 0.3, 0.6, 0.5).

con(A, C) = 0.8 × 0.3 + 0.1 × 0.1 = 0.24 + 0.01 = 0.25. Less conflict — C's high uncertainty moderates the disagreement.

---

## 10. Internal Conflict (Balance Metric)

### 10.1 Purpose

Internal conflict measures how much the evidence *within a single opinion* is self-contradictory. This typically arises after fusing opinions from agents that disagree — the fused result has high belief AND high disbelief simultaneously, reflecting the unresolved disagreement.

### 10.2 Formula

**Definition 10.1 (Internal Conflict).** For opinion ω = (b, d, u, a):

$$\text{conflict}(\omega) = \max(0, \; 1 - |b - d| - u)$$

The `max(0, ...)` clamp handles IEEE 754 floating-point arithmetic artifacts.

### 10.3 Derivation

The internal conflict captures the "balanced evidence mass" — the portion of committed mass (b + d = 1 − u) that is evenly split between belief and disbelief.

Without loss of generality, assume b ≥ d. Then |b − d| = b − d represents the "net directional evidence." The committed mass is (1 − u). The balanced portion is:

$$\text{balanced} = (1 - u) - |b - d| = (b + d) - (b - d) = 2d$$

Alternatively: conflict = 1 − |b − d| − u = 1 − (b − d) − u = (1 − u) − (b − d) = (b + d) − (b − d) = 2 · min(b, d).

This is the amount of mass that "cancels out" — evidence fighting against itself.

### 10.4 Interpretation

| Condition | Conflict | Interpretation |
|-----------|----------|----------------|
| b ≈ d, u small | High | Evidence cancels out — genuine disagreement |
| b ≫ d or d ≫ b | Low | Clear evidential direction |
| u ≈ 1 | Low | Ignorance, not disagreement |

**Key insight:** This metric distinguishes *conflict* (high b and high d) from *ignorance* (high u). A scalar confidence of 0.5 could be either — this metric separates the two.

### 10.5 Formal Properties

**Property 10.1 (Range).** conflict(ω) ∈ [0, 1].

*Proof:* Maximum occurs when b = d = 0.5, u = 0: conflict = 1 − 0 − 0 = 1. The min clamp ensures 0 is the lower bound. ∎

**Property 10.2 (Maximum conflict).** conflict = 1 when b = d = 0.5, u = 0. Evidence is perfectly balanced and fully committed — maximal internal disagreement.

**Property 10.3 (Zero conflict for dogmatic agreement).** When b = 1, d = 0: conflict = 1 − 1 − 0 = 0. Or when d = 1, b = 0: conflict = 1 − 1 − 0 = 0.

**Property 10.4 (Zero conflict for ignorance).** When u = 1: conflict = max(0, 1 − 0 − 1) = 0.

**Property 10.5 (Relationship to min).** conflict(ω) = 2 · min(b, d). *Proof:* If b ≥ d: conflict = 1 − (b − d) − u = (b + d + u) − (b − d) − u = 2d = 2·min(b,d). If d ≥ b: similarly 2b = 2·min(b,d). ∎

### 10.6 Worked Example

**Scenario:** After fusing two conflicting agent opinions:

ω_fused = (0.45, 0.40, 0.15, 0.5).

conflict = max(0, 1 − |0.45 − 0.40| − 0.15) = max(0, 1 − 0.05 − 0.15) = 0.80.

High internal conflict (0.80) signals that the fused result reflects genuine disagreement among the original agents. A downstream system should treat this assertion with caution — perhaps requesting additional evidence or human review.

Compare with ω' = (0.80, 0.05, 0.15, 0.5): conflict = max(0, 1 − 0.75 − 0.15) = 0.10. Low conflict — clear evidential direction.

---

## 11. Robust Fusion

### 11.1 Purpose

Robust fusion provides **Byzantine-resistant** opinion combination. In multi-agent systems, some agents may be compromised, adversarial, or simply miscalibrated. Robust fusion identifies and removes outlier opinions before fusing the cohesive majority.

### 11.2 Algorithm

**Algorithm 11.1 (Robust Fusion with Iterative Conflict Filtering).**

**Input:**
- opinions: List of n opinions ω_1, ..., ω_n (n ≥ 1)
- threshold: Discord score threshold θ ∈ [0, 1] (default: 0.15)
- max_removals: Maximum agents to remove M (default: ⌊n/2⌋)

**Output:**
- fused: Fused opinion from the cohesive subset
- removed: Indices of removed opinions in the original list

```
PROCEDURE robust_fuse(opinions, θ, M):
    indexed ← [(0, ω_0), (1, ω_1), ..., (n-1, ω_{n-1})]
    removed ← []
    
    FOR iteration = 1 TO M:
        n_current ← |indexed|
        IF n_current ≤ 2:
            BREAK  // Cannot go below 2 agents
        
        // Step 1: Compute discord scores
        FOR i = 0 TO n_current - 1:
            discord[i] ← 0
            FOR j = 0 TO n_current - 1, j ≠ i:
                discord[i] ← discord[i] + con(indexed[i].ω, indexed[j].ω)
            discord[i] ← discord[i] / (n_current - 1)
        
        // Step 2: Find worst agent
        worst ← argmax_i(discord[i])
        
        // Step 3: Check convergence
        IF discord[worst] < θ:
            BREAK  // Group is cohesive
        
        // Step 4: Remove worst agent
        removed ← removed ∪ {indexed[worst].original_index}
        indexed ← indexed \ {indexed[worst]}
    
    // Step 5: Fuse remaining opinions via cumulative fusion
    fused ← cumulative_fuse(indexed[0].ω, ..., indexed[k].ω)
    RETURN (fused, removed)
```

**Complexity:** O(M · n²) for pairwise conflict computation across iterations.

### 11.3 Discord Score

**Definition 11.1 (Discord Score).** The discord score of agent i among a set of n agents is the mean pairwise conflict with all other agents:

$$\text{discord}(i) = \frac{1}{n - 1} \sum_{j \neq i} \text{con}(\omega_i, \omega_j)$$

A high discord score indicates that agent i disagrees with the majority.

### 11.4 Convergence

The algorithm terminates when one of these conditions is met:

1. The worst discord score falls below the threshold θ (cohesive group achieved).
2. The maximum number of removals M is reached.
3. Only 2 agents remain (minimum viable group for fusion).

### 11.5 Threshold Selection

The default threshold θ = 0.15 is calibrated to detect strong disagreements while allowing moderate diversity:

| Scenario | Pairwise conflict | Discord exceeds θ? |
|----------|------------------|-------------------|
| (0.8, 0.1, 0.1) vs (0.2, 0.7, 0.1) | 0.8×0.7 + 0.1×0.2 = 0.58 | Yes (0.58 ≫ 0.15) |
| (0.7, 0.1, 0.2) vs (0.6, 0.2, 0.2) | 0.7×0.2 + 0.1×0.6 = 0.20 | Yes (marginally) |
| (0.7, 0.1, 0.2) vs (0.7, 0.2, 0.1) | 0.7×0.2 + 0.1×0.7 = 0.21 | Yes (marginally) |
| (0.6, 0.1, 0.3) vs (0.5, 0.2, 0.3) | 0.6×0.2 + 0.1×0.5 = 0.17 | Yes (barely) |
| (0.7, 0.1, 0.2) vs (0.6, 0.1, 0.3) | 0.7×0.1 + 0.1×0.6 = 0.13 | No (cohesive) |

### 11.6 Safety Guarantee

**Theorem 11.1 (Majority preservation).** The constraint max_removals ≤ ⌊n/2⌋ ensures that a majority of agents is never removed.

*Proof:* At most ⌊n/2⌋ agents are removed, leaving at least n − ⌊n/2⌋ = ⌈n/2⌉ agents. For n ≥ 2, ⌈n/2⌉ ≥ n/2, so a strict majority remains. ∎

This prevents a scenario where a minority of adversarial agents could cause the majority to be discarded.

### 11.7 Worked Example

**Scenario:** Five sensor agents report readings; one is faulty.

- ω_1 = (0.8, 0.1, 0.1, 0.5) — honest
- ω_2 = (0.7, 0.1, 0.2, 0.5) — honest
- ω_3 = (0.75, 0.05, 0.2, 0.5) — honest
- ω_4 = (0.7, 0.15, 0.15, 0.5) — honest
- ω_5 = (0.05, 0.85, 0.1, 0.5) — faulty (reports opposite)

Discord scores for ω_5 against the honest agents will be high (e.g., con(ω_1, ω_5) = 0.8×0.85 + 0.1×0.05 = 0.685). The mean discord for ω_5 will far exceed θ = 0.15, so it will be removed in the first iteration.

After removal, the remaining four honest agents have low mutual discord (all believe ≈ 0.7-0.8 with low disbelief), so the algorithm converges. The final result is the cumulative fusion of four cohesive opinions.

---

## 12. Temporal Decay

### 12.1 Purpose

Temporal decay models the natural degradation of confidence over time. As evidence ages, it becomes less reliable. The decay process migrates mass from belief and disbelief into uncertainty, reflecting the epistemic reality that old evidence is less trustworthy than fresh evidence.

This generalizes Jøsang's opinion aging operator (Jøsang, 2016, §10.4) by accepting arbitrary decay functions while preserving the core invariants.

### 12.2 Decay Operator

**Definition 12.1 (Decay Operator).** Given an opinion ω = (b, d, u, a), an elapsed time t ≥ 0, a half-life τ > 0, and a decay function f: ℝ⁺ × ℝ⁺ → [0, 1]:

$$\lambda = f(t, \tau)$$

$$b' = \lambda \cdot b$$

$$d' = \lambda \cdot d$$

$$u' = 1 - b' - d'$$

The base rate is preserved: a' = a.

### 12.3 Built-in Decay Functions

**Definition 12.2 (Exponential Decay).**

$$f_{\text{exp}}(t, \tau) = 2^{-t/\tau}$$

The standard smooth decay model. At t = τ, the factor is exactly 0.5 (half-life semantics). Never reaches zero; asymptotically approaches it. This is the default.

| Elapsed (t/τ) | Factor (λ) | Interpretation |
|---------------|-----------|----------------|
| 0 | 1.000 | Fresh evidence |
| 0.5 | 0.707 | Slightly aged |
| 1.0 | 0.500 | Half-life reached |
| 2.0 | 0.250 | Significantly aged |
| 5.0 | 0.031 | Very stale |
| 10.0 | 0.001 | Nearly forgotten |

**Definition 12.3 (Linear Decay).**

$$f_{\text{lin}}(t, \tau) = \max\left(0, \; 1 - \frac{t}{2\tau}\right)$$

Reaches zero at t = 2τ, then remains at zero. At t = τ, the factor is 0.5 (consistent with half-life semantics). Simple and predictable.

**Definition 12.4 (Step Decay).**

$$f_{\text{step}}(t, \tau) = \begin{cases} 1 & \text{if } t < \tau \\ 0 & \text{if } t \geq \tau \end{cases}$$

Binary freshness: evidence is either fully fresh (λ = 1) or completely stale (λ = 0). Useful for hard TTL-style expiration in IoT systems.

### 12.4 Extensibility

The decay operator accepts any function f(t, τ) → [0, 1] as a decay function. Users MAY define custom decay functions for domain-specific aging models (e.g., seasonal decay, stepwise regulatory expiry). Conforming processors MUST validate that the decay function returns a value in [0, 1].

### 12.5 Formal Properties

**Theorem 12.1 (Ratio Preservation).**

$$\frac{b'}{d'} = \frac{b}{d} \quad \text{(when } d > 0\text{)}$$

*Proof:* b'/d' = (λb)/(λd) = b/d. ∎

*Interpretation:* Decay forgets *how much* evidence we had, not *which direction* it pointed. The belief-to-disbelief ratio is invariant under decay. If we started leaning 4:1 toward belief, after decay we still lean 4:1 — but with less conviction and more uncertainty.

**Theorem 12.2 (Additivity Preservation).**

$$b' + d' + u' = 1$$

*Proof:* b' + d' + u' = λb + λd + (1 − λb − λd) = 1. ∎

**Theorem 12.3 (Monotonic Uncertainty Increase).** For λ ∈ [0, 1) and non-vacuous opinions (b + d > 0):

$$u' > u$$

*Proof:* u' = 1 − λ(b + d) = 1 − λ(1 − u). Since λ < 1:

$$u' = 1 - \lambda + \lambda u > 1 - 1 + u = u$$

(The inequality holds because 1 − λ > 0 and we add λu ≥ 0.) ∎

**Theorem 12.4 (Reversion to Prior).** As λ → 0:

$$b' \to 0, \quad d' \to 0, \quad u' \to 1$$

$$P(\omega') = b' + a \cdot u' \to 0 + a \cdot 1 = a$$

*Interpretation:* Fully decayed evidence reverts to the prior. The projected probability returns to the base rate — we are back to total ignorance. This is the epistemically correct behavior: evidence that has completely expired should not influence decisions.

**Theorem 12.5 (Identity at Zero Elapsed Time).** When t = 0, all built-in decay functions return λ = 1, and ω' = ω. No decay occurs.

*Proof:* f_exp(0, τ) = 2⁰ = 1. f_lin(0, τ) = max(0, 1) = 1. f_step(0, τ) = 1 (since 0 < τ). ∎

**Theorem 12.6 (Composability).** Applying decay with factor λ₁ followed by decay with factor λ₂ is equivalent to a single decay with factor λ₁·λ₂.

*Proof:* After first decay: b' = λ₁b, d' = λ₁d. After second decay: b'' = λ₂·b' = λ₁λ₂·b, d'' = λ₁λ₂·d. This equals a single decay with factor λ₁λ₂. ∎

*Practical significance:* For exponential decay, f(t₁, τ) · f(t₂, τ) = 2^{−t₁/τ} · 2^{−t₂/τ} = 2^{−(t₁+t₂)/τ} = f(t₁ + t₂, τ). So decaying by t₁ then t₂ is identical to decaying by t₁ + t₂ — the operator is consistent with elapsed time being additive.

### 12.6 Validation Rules

- Elapsed time t MUST be non-negative (t ≥ 0).
- Half-life τ MUST be strictly positive (τ > 0).
- The decay function MUST return a value in [0, 1].
- The resulting u' is clamped to max(0, u') to handle IEEE 754 floating-point artifacts (e.g., 1.0 − 0.8 − 0.2 may produce −5.5 × 10⁻¹⁷ rather than 0.0).

### 12.7 Worked Example

**Scenario:** A sensor reading is 1 day old with a 24-hour half-life.

ω = (0.8, 0.1, 0.1, 0.5). Elapsed = 24h. Half-life = 24h. Exponential decay.

$$\lambda = 2^{-24/24} = 2^{-1} = 0.5$$

$$b' = 0.5 \times 0.8 = 0.40$$

$$d' = 0.5 \times 0.1 = 0.05$$

$$u' = 1 - 0.40 - 0.05 = 0.55$$

**Result:** ω' = (0.40, 0.05, 0.55, 0.50). P(ω') = 0.40 + 0.5 × 0.55 = 0.675.

The evidence direction is preserved (b'/d' = 8 = b/d), but the opinion has shifted significantly toward uncertainty. The projected probability dropped from 0.85 to 0.675 — still positive, but with much less conviction.

After 3 half-lives (72h): λ = 0.125. b' = 0.1, d' = 0.0125, u' = 0.8875. P(ω') = 0.1 + 0.5 × 0.8875 = 0.544. Nearly at the prior (0.5).

---

## 13. Bridge Theorems

This section formally establishes the relationship between the Subjective Logic algebra and classical scalar confidence methods. These theorems justify the algebra as a strict generalization — scalar methods are special cases, not alternatives.

### 13.1 Theorem: Multiply Chain ≡ Iterated Trust Discount

**Theorem 13.1.** The scalar multiply propagation method (∏ cᵢ) is exactly equivalent to iterated trust discount with dogmatic trust opinions and base_rate = 0.

**Proof by induction on chain length n:**

**Base case (n = 1):** Given a single confidence c₁, construct the trust opinion ω_T = (c₁, 1 − c₁, 0) and the initial assertion ω₀ = (1, 0, 0, a = 0). Apply trust discount:

$$b = c_1 \cdot 1 = c_1$$
$$d = c_1 \cdot 0 = 0$$
$$u = (1 - c_1) + 0 + c_1 \cdot 0 = 1 - c_1$$

The projected probability: P = c₁ + 0 · (1 − c₁) = c₁ = ∏ c_i. ✓

**Inductive step:** Assume that after k trust discounts with scores c₁, ..., c_k, the resulting opinion has:

$$b_k = \prod_{i=1}^{k} c_i, \quad d_k = 0, \quad u_k = 1 - \prod_{i=1}^{k} c_i$$

(This can be verified: b_k + d_k + u_k = ∏c_i + 0 + 1 − ∏c_i = 1 ✓)

Discount by c_{k+1} with trust opinion ω_T = (c_{k+1}, 1 − c_{k+1}, 0):

$$b_{k+1} = c_{k+1} \cdot b_k = c_{k+1} \cdot \prod_{i=1}^{k} c_i = \prod_{i=1}^{k+1} c_i$$

$$d_{k+1} = c_{k+1} \cdot d_k = c_{k+1} \cdot 0 = 0$$

$$u_{k+1} = (1 - c_{k+1}) + 0 + c_{k+1} \cdot u_k = 1 - c_{k+1} + c_{k+1}(1 - \prod_{i=1}^{k} c_i) = 1 - \prod_{i=1}^{k+1} c_i$$

$$P = b_{k+1} + 0 \cdot u_{k+1} = \prod_{i=1}^{k+1} c_i \quad \blacksquare$$

**Significance:** This proves that trust discount is a strict generalization of scalar multiplication. When trust is uncertain (u_T > 0), the algebra provides richer semantics while degrading gracefully to the classical case when trust is dogmatic.

### 13.2 Theorem: Scalar Average (n=2) ≡ Averaging Fusion of Dogmatic Opinions

**Theorem 13.2.** For two dogmatic opinions created from scalars c_A and c_B (with u = 0), averaging fusion produces the arithmetic mean:

$$P(\omega_A \oslash \omega_B) = \frac{c_A + c_B}{2}$$

*Proof:* Map c_A → ω_A = (c_A, 1−c_A, 0, a) and c_B → ω_B = (c_B, 1−c_B, 0, a). With u_A = u_B = 0, the dogmatic fallback gives:

$$b = \frac{b_A + b_B}{2} = \frac{c_A + c_B}{2}$$

Since u = 0: P = b = (c_A + c_B)/2. ∎

**⚠ Caveat (n > 2):** This equivalence holds ONLY for n = 2. For n > 2, pairwise left-folding of averaging fusion does NOT produce the arithmetic mean of the n scalars, because averaging fusion is not associative (Property 6.3). The simultaneous n-ary formula must be used, and for dogmatic opinions with equal weight, the n-ary formula does produce the simple average (all U_i are equal when all u_i = 0, via the dogmatic fallback). However, for non-dogmatic opinions, the behavior differs from scalar averaging.

### 13.3 Non-equivalence: Noisy-OR ≠ Cumulative Fusion

**Theorem 13.3.** The noisy-OR combination (1 − ∏(1 − pᵢ)) is NOT equivalent to cumulative fusion for any consistent mapping from scalars to opinions.

*Proof by counterexample:* Let c_A = 0.9, c_B = 0.7.

**Noisy-OR:** 1 − (1 − 0.9)(1 − 0.7) = 1 − 0.1 × 0.3 = 1 − 0.03 = 0.97.

**Cumulative fusion with natural mapping** (b = p, d = 0, u = 1 − p): ω_A = (0.9, 0, 0.1, 0.5), ω_B = (0.7, 0, 0.3, 0.5).

κ = 0.1 + 0.3 − 0.03 = 0.37.

b = (0.9 × 0.3 + 0.7 × 0.1) / 0.37 = 0.34/0.37 ≈ 0.919.

u = 0.03/0.37 ≈ 0.081.

P = 0.919 + 0.5 × 0.081 ≈ 0.960 ≠ 0.97.

**With dogmatic mapping** (b = p, d = 1−p, u = 0): Both opinions are dogmatic. Dogmatic fallback gives b = (0.9 + 0.7)/2 = 0.8. P = 0.8 ≠ 0.97.

No consistent mapping produces the noisy-OR result. The two methods model different independence assumptions. ∎

### 13.4 Non-equivalence: Dempster-Shafer ≠ Any Single SL Operator

**Theorem 13.4.** Dempster-Shafer combination is not equivalent to any single Subjective Logic operator.

*Justification:* Dempster-Shafer theory uses a different evidence model (basic probability assignments over a power set of hypotheses) with a different combination rule (Dempster's rule with conflict normalization). While Subjective Logic and Dempster-Shafer share conceptual heritage, their mathematical frameworks are distinct. The key difference is that Dempster's rule normalizes by (1 − K) where K is the conflict mass, while cumulative fusion normalizes by κ = u_A + u_B − u_A u_B. No consistent mapping exists that makes one a special case of the other for all inputs.

### 13.5 Summary of Expressiveness

$$\text{Scalar confidence} \subset \text{Subjective Logic opinions}$$

The algebra is strictly more expressive: it distinguishes epistemic states that identical scalar values conflate. Scalar methods operate in the 1D dogmatic subspace (u = 0); the algebra adds the uncertainty dimension, enabling calibrated accept/reject decisions that consider *how much we know* alongside *what we believe*.

The bridge theorems show that scalar methods are recoverable as special cases: multiply is trust discount with dogmatic trust, average is averaging fusion of dogmatic opinions. This ensures backward compatibility — existing scalar-based systems produce identical results when opinions have u = 0.

---

## 14. JSON-LD Serialization

### 14.1 Opinion Serialization Format

An Opinion is serialized as a JSON-LD node with type `jex:Opinion`:

```json
{
  "@context": {
    "jex": "https://w3id.org/jsonld-ex/",
    "Opinion": "jex:Opinion",
    "belief": "jex:belief",
    "disbelief": "jex:disbelief",
    "uncertainty": "jex:uncertainty",
    "baseRate": "jex:baseRate"
  },
  "@type": "Opinion",
  "belief": 0.70,
  "disbelief": 0.10,
  "uncertainty": 0.20,
  "baseRate": 0.50
}
```

### 14.2 Field Mapping

| JSON-LD field | Opinion component | IRI |
|---------------|-------------------|-----|
| `belief` | b | `https://w3id.org/jsonld-ex/belief` |
| `disbelief` | d | `https://w3id.org/jsonld-ex/disbelief` |
| `uncertainty` | u | `https://w3id.org/jsonld-ex/uncertainty` |
| `baseRate` | a | `https://w3id.org/jsonld-ex/baseRate` |

### 14.3 Deserialization Rules

- All four fields MUST be present (except `baseRate`, which defaults to 0.5 if absent).
- Each field MUST be a finite number in [0, 1].
- The additivity constraint b + d + u = 1 MUST be validated upon deserialization within tolerance ε = 10⁻⁹.

### 14.4 Inline Usage

An Opinion MAY appear as the value of `@confidence` in place of a scalar, enabling gradual migration:

```json
{
  "name": {
    "@value": "Jane Doe",
    "@confidence": {
      "@type": "Opinion",
      "belief": 0.85,
      "disbelief": 0.05,
      "uncertainty": 0.10,
      "baseRate": 0.50
    },
    "@source": "https://model.example.org/ner-v4"
  }
}
```

A non-extended processor treats the `@confidence` value as a regular JSON object — it will be preserved but not interpreted. An extended processor interprets it as a full Opinion, gaining access to the belief/disbelief/uncertainty decomposition for downstream reasoning.

### 14.5 Backward Compatibility

When `@confidence` is a scalar number, it is treated as the projected probability P(ω) of an implicit dogmatic opinion (c, 1−c, 0, 0.5). This ensures full backward compatibility: all existing scalar-confidence documents are valid under the extended specification.

---

## 15. Implementation Notes

### 15.1 Floating-Point Tolerance

The additivity constraint b + d + u = 1 is checked with tolerance ε = 10⁻⁹. This accounts for IEEE 754 double-precision arithmetic artifacts. For example:

- 0.1 + 0.2 + 0.7 evaluates to 0.9999999999999999 (not 1.0) in many languages.
- The decay operator may produce u' = 1 − 0.8 − 0.2 = −5.5 × 10⁻¹⁷ rather than 0.0. This is clamped to 0.0.

### 15.2 Division by Zero Prevention

Several formulas have denominators that can reach zero:

- **Cumulative fusion:** κ = u_A + u_B − u_A u_B = 0 iff u_A = u_B = 0 (both dogmatic). → Use the dogmatic fallback formula.
- **Averaging fusion (binary):** κ = u_A + u_B = 0 iff both are dogmatic. → Use the dogmatic fallback.
- **Averaging fusion (n-ary):** κ = Σ U_i = 0 iff at least two opinions are dogmatic. → Use the dogmatic fallback that averages only the dogmatic subset.
- **U_i computation:** U_i = full_product / u_i when u_i = 0. → Compute ∏_{j≠i} u_j directly by iterating over all j ≠ i.

### 15.3 Numerical Stability

For n-ary averaging fusion with many opinions, the product ∏ u_i can underflow to zero even when all u_i > 0 (if n is large and u_i values are small). Implementations SHOULD detect this case and fall back to computing U_i values in log-space or via compensated products.

### 15.4 Boolean Rejection

In Python, `bool` is a subclass of `int` (`True == 1`, `False == 0`). Processors MUST explicitly check `isinstance(value, bool)` before `isinstance(value, (int, float))` to reject booleans from all opinion components. This prevents subtle bugs where `True` is silently accepted as belief = 1.

### 15.5 Frozen Immutability

The Opinion type is implemented as a frozen dataclass. Opinion objects are immutable after construction. All operators return new Opinion objects rather than modifying existing ones. This prevents shared-state bugs in concurrent systems and makes the algebra referentially transparent.

---

## 16. Classical Scalar Methods (Informative)

This section documents the scalar confidence methods provided by the reference implementation for backward compatibility and comparison. These methods are NOT part of the formal algebra but are included for completeness, bridge theorem support, and migration guidance.

### 16.1 Chain Propagation Methods

| Method | Formula | Properties | Use Case |
|--------|---------|------------|----------|
| `multiply` | ∏ cᵢ | Conservative; exponential decay with chain length | Short inference chains |
| `bayesian` | Sequential Bayesian update with uniform prior | Probabilistically principled; handles extreme values gracefully | Calibrated probabilistic systems |
| `min` | min(cᵢ) | Weakest-link; ignores all but the least confident step | Safety-critical applications |
| `dampened` | (∏ cᵢ)^{1/√n} | Attenuates over-penalization in long chains | Long inference chains |

**Bayesian update detail:** Each score cᵢ is treated as a likelihood ratio. Starting from a uniform prior (odds = 1), log-odds are accumulated: log_odds += log(cᵢ/(1−cᵢ)). The final posterior is odds/(1+odds). Scores of 0 and 1 are clamped to [10⁻⁴, 1−10⁻⁴] to avoid infinities.

**Dampened multiplication detail:** The exponent 1/√n attenuates the product's decay rate. For n = 1, the result equals the input. For n = 4, the effective power is 1/2 (square root of the product). For n = 100, the power is 1/10 (tenth root), preventing collapse to near-zero.

### 16.2 Multi-Source Combination Methods

| Method | Formula | Properties | Use Case |
|--------|---------|------------|----------|
| `average` | (1/n) Σ cᵢ | Simple; may underestimate when sources agree | Quick aggregation |
| `max` | max(cᵢ) | Optimistic; ignores disagreeing sources | Best-case estimation |
| `noisy_or` | 1 − ∏(1 − cᵢ) | Models P(at least one correct); assumes independence | Independent binary sources |
| `dempster_shafer` | Dempster's rule | Evidence-theoretic; handles uncertainty | Formal evidence combination |

### 16.3 Conflict Resolution Strategies

| Strategy | Selection Criterion | Tiebreaker |
|----------|-------------------|------------|
| `highest` | Highest `@confidence` value | List order |
| `weighted_vote` | Group by `@value`, combine groups via noisy-OR | Highest individual within winning group |
| `recency` | Most recent `@extractedAt` timestamp | Highest `@confidence` |

---

## 17. Operator Summary

| Operator | Symbol | Arity | Commutative | Associative | Key Property |
|----------|--------|-------|-------------|-------------|--------------|
| Cumulative Fusion | ⊕ | n ≥ 2 | Yes | Yes | Reduces uncertainty (independent evidence) |
| Averaging Fusion | ⊘ | n ≥ 2 | Yes | **No** | Idempotent (correlated evidence) |
| Trust Discount | ⊗ | Binary | **No** | Yes (chains) | Dilutes toward uncertainty |
| Deduction | — | Ternary | N/A | N/A | Generalizes law of total probability |
| Pairwise Conflict | con | Binary | Yes | N/A | Measures inter-agent disagreement |
| Internal Conflict | conflict | Unary | N/A | N/A | Distinguishes conflict from ignorance |
| Robust Fusion | — | n ≥ 2 | Yes | N/A | Byzantine-resistant (removes outliers) |
| Temporal Decay | — | Unary + params | N/A | Yes (composable) | Preserves evidence direction ratio |

---

## 18. References

- Jøsang, A. (2016). *Subjective Logic: A Formalism for Reasoning Under Uncertainty.* Springer. ISBN 978-3-319-42335-7.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems.* Morgan Kaufmann.
- Shafer, G. (1976). *A Mathematical Theory of Evidence.* Princeton University Press.
- Walley, P. (1991). *Statistical Reasoning with Imprecise Probabilities.* Chapman and Hall.
- Zadeh, L. A. (1965). Fuzzy sets. *Information and Control*, 8(3), 338–353.
- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- IETF. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. https://www.rfc-editor.org/rfc/rfc2119
