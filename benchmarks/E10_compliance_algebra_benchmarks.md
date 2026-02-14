# Experiment Suite E10: Compliance Algebra Benchmarks

**Module under test:** `jsonld_ex.compliance_algebra`  
**Mathematical spec:** `compliance_algebra.md` (Syed et al. 2026)  
**Priority:** HIGH — this is a novel contribution unique to jsonld-ex  
**Track:** A (Algebra Core), local CPU only  
**Estimated total time:** 10–14 hours across all experiments

---

## Motivation

The compliance algebra is, to our knowledge, the first application of opinion
algebras to model regulatory compliance as uncertain epistemic states.
The experiments must:

1. **Verify** proven algebraic properties at scale (not just unit tests)
2. **Demonstrate** measurable superiority over binary compliance models
3. **Quantify** how uncertainty compounds across realistic GDPR scenarios
4. **Characterize** limitations honestly (independence bias, calibration gap)

This maps to the NeurIPS D&B track requirements for rigorous empirical
evaluation of tools and benchmarks.

---

## E10.1 — Property-Based Verification of Algebraic Theorems

**Goal:** Verify all 5 theorems + Proposition 1 using randomized
property-based testing (Hypothesis library), going far beyond the
~103 hand-crafted unit tests.

**Hypothesis:** All proven properties hold across 10,000+ random
opinion inputs, with zero violations.

**Protocol:**
1. Use the Hypothesis library to generate random ComplianceOpinions
   (l, v, u ≥ 0, l+v+u = 1, a ∈ [0,1])
2. For each operator, test every property from its theorem:

   **Theorem 1 (Jurisdictional Meet) — 9 properties × 10,000 inputs:**
   (a) constraint: l+v+u = 1
   (b) non-negativity: l,v,u ≥ 0
   (c) monotonic restriction: l⊓ ≤ min(l₁, l₂)
   (d) monotonic violation: v⊓ ≥ max(v₁, v₂)
   (e) commutativity: J⊓(ω₁,ω₂) = J⊓(ω₂,ω₁)
   (f) associativity: J⊓(ω₁,J⊓(ω₂,ω₃)) = J⊓(J⊓(ω₁,ω₂),ω₃)
   (g) identity: J⊓(ω, (1,0,0,1)) = ω
   (h) annihilator: J⊓(ω, (0,1,0,0)) = (0,1,0,0)
   (i) non-idempotency: J⊓(ω,ω) ≠ ω for l ∉ {0,1}

   **Theorem 2 (Compliance Propagation) — 6 properties:**
   (a) constraint inherited
   (b) degradation monotonicity: l_D ≤ l_S, v_D ≥ v_S
   (c) identity derivation
   (d) violation annihilation (any of 3 components = ω⊥)
   (e) chain associativity: 2-step = 5-way meet
   (f) multiplicative decay: l_{D_n} = l_S · ∏(t_i · p_i)

   **Theorem 3 (Withdrawal Override) — 3 properties:**
   (a) withdrawal dominance
   (b) pre-withdrawal preservation
   (c) non-interference (purpose-indexed)

   **Theorem 4 (Temporal Triggers) — 5 properties:**
   (a) continuous monotonicity (via decay_opinion)
   (b) expiry constraint preservation
   (c) expiry monotonicity: l' ≤ l, v' ≥ v, u' = u
   (d) hard expiry: γ=0 → l'=0, v'=v+l
   (e) non-commutativity of distinct trigger types

   **Theorem 5 (Erasure) — 4 properties:**
   (a) exponential degradation: e_R = ∏ e_i
   (b) scope monotonicity
   (c) exception filtering
   (d) perfect source identity

   **Proposition 1 (Residual Contamination) — 3 properties:**
   (a) constraint: r+r̄+u_r = 1
   (b) non-negativity
   (c) monotonic depth increase

3. Report: total tests executed, violations found (expect 0),
   property coverage percentage, execution time.

**Metrics:** Pass rate (target: 100%), total property-test executions
(target: ≥200,000 across all properties), wall-clock time.

**Features exercised:** All 11 compliance algebra symbols.

**Deliverable:** Notebook + table of property × pass_count × violations.

**Estimated time:** 1–2 hours.

---

## E10.2 — Binary Compliance vs. Compliance Algebra (Table 2 Brought to Life)

**Goal:** Demonstrate concretely that the compliance algebra captures
information that binary compliance checking loses, using realistic
GDPR scenarios.

**Hypothesis:** The algebra produces strictly more informative
assessments than binary, enabling different (better) decisions across
all 8 dimensions from Table 2 of the formalization.

**Protocol:**
1. Construct 6 GDPR scenarios with varying epistemic states:

   **Scenario A — Strong compliance, low uncertainty:**
   ω = (0.85, 0.05, 0.10, 0.5)
   Binary: COMPLIANT. Algebra: high confidence, low residual risk.

   **Scenario B — Absence of evidence:**
   ω = (0.1, 0.05, 0.85, 0.5)
   Binary: COMPLIANT (just barely, or UNKNOWN?). Algebra: mostly
   uncertain — "we don't know" is distinct from "we know it's ok."

   **Scenario C — Conflicting evidence:**
   ω = (0.4, 0.45, 0.15, 0.5)
   Binary: NON-COMPLIANT (marginally). Algebra: high conflict,
   investigation needed — qualitatively different from Scenario D.

   **Scenario D — Clear violation:**
   ω = (0.05, 0.9, 0.05, 0.5)
   Binary: NON-COMPLIANT. Algebra: near-certain violation.

   **Scenario E — Moderate uncertainty:**
   ω = (0.5, 0.15, 0.35, 0.5)
   Binary: COMPLIANT. Algebra: moderate belief with substantial
   residual uncertainty — "probably ok but gather more evidence."

   **Scenario F — Dogmatic coin flip:**
   ω = (0.45, 0.45, 0.1, 0.5)
   Binary: UNDECIDED. Algebra: strong balanced evidence of both
   compliance AND violation — fundamentally different from B.

2. For each scenario, evaluate under 4 decision rules:
   - D1: Accept/reject at P(ω) = 0.5 threshold
   - D2: Request more evidence if u > 0.3
   - D3: Flag for legal review if conflict > 0.2
   - D4: Escalate if violation evidence v > 0.3 regardless of l

3. Show that binary systems cannot distinguish B, E, F (all map to
   the same P ≈ 0.5 region) while the algebra assigns distinct
   decisions to each.

4. Compose scenarios: apply jurisdictional_meet across pairs of
   scenarios representing EU + US jurisdictions. Show how composite
   uncertainty is qualitatively richer than AND/OR.

**Metrics:**
- Decision diversity: number of distinct decisions across scenarios
  (binary: ≤3; algebra: ≤6)
- Information capacity: bits of decision entropy
- Composite accuracy: does the composite assessment better reflect
  the true combined risk?

**Features exercised:** `ComplianceOpinion`, `jurisdictional_meet`,
`projected_probability`.

**Deliverable:** Notebook + Table 2 recreation with concrete numbers +
decision-comparison figure.

**Estimated time:** 1–2 hours.

---

## E10.3 — Multi-Jurisdictional Composition Scaling

**Goal:** Quantify how lawfulness degrades and violation accumulates
as organizations operate across increasing numbers of jurisdictions.

**Hypothesis:** Composite lawfulness degrades exponentially with
jurisdiction count (Theorem 5(a) pattern), making multi-jurisdictional
compliance fundamentally harder to demonstrate with certainty —
a result binary models hide.

**Protocol:**
1. Define 8 representative jurisdictional profiles:

   | Jurisdiction  | l    | v    | u    | a   | Character              |
   |---------------|------|------|------|-----|------------------------|
   | EU-GDPR       | 0.80 | 0.05 | 0.15 | 0.3 | Strict, well-evidenced |
   | US-CCPA       | 0.70 | 0.10 | 0.20 | 0.6 | Moderate               |
   | UK-DPA        | 0.75 | 0.08 | 0.17 | 0.4 | Post-Brexit uncertain  |
   | Brazil-LGPD   | 0.50 | 0.15 | 0.35 | 0.5 | New, uncertain         |
   | India-DPDP    | 0.40 | 0.10 | 0.50 | 0.5 | Very new, high u       |
   | China-PIPL    | 0.60 | 0.20 | 0.20 | 0.4 | Moderate, higher v     |
   | Japan-APPI    | 0.80 | 0.03 | 0.17 | 0.5 | Strict, well-evidenced |
   | Canada-PIPEDA | 0.72 | 0.08 | 0.20 | 0.5 | Moderate               |

2. Compute jurisdictional_meet for all subsets of size 1 through 8.
3. For each subset size n, compute:
   - Mean composite lawfulness across all C(8,n) combinations
   - Min/max composite lawfulness
   - Mean composite violation
   - Mean composite uncertainty
4. Plot: l, v, u vs. jurisdiction count (expect l exponential decay,
   v asymptotic to 1)
5. Compare: binary AND gives 0 if ANY fails; algebra shows the
   gradient from "1 jurisdiction is questionable" to "3 are."

**Metrics:**
- l(n) curve fit to l₁·l₂·...·lₙ (verify matches theory)
- v(n) curve fit to 1 − ∏(1−vᵢ) (verify matches theory)
- Number of jurisdictions at which P(ω) < 0.5 (practical threshold)

**Features exercised:** `jurisdictional_meet` (n-ary), `ComplianceOpinion`.

**Deliverable:** Notebook + scaling curves + table of composite values.

**Estimated time:** 1 hour.

---

## E10.4 — Consent Condition Sensitivity Analysis

**Goal:** Quantify how weakness in individual GDPR Art. 7 consent
conditions propagates to composite consent validity, and identify
which conditions have the most leverage.

**Hypothesis:** The six-way meet creates a "weakest link" dynamic
where a single poorly-evidenced condition dominates the composite
result, making consent assessment highly sensitive to the least
certain condition.

**Protocol:**
1. Baseline: all 6 conditions at ω = (0.85, 0.05, 0.10, 0.5).
2. Systematically weaken ONE condition from l=0.85 down to l=0.0
   (in steps of 0.05), redistributing to u, then to v.
3. For each weakened level, compute:
   - Composite consent lawfulness
   - Composite projected probability
   - How many "strong" conditions it takes to compensate
4. Repeat with TWO conditions weakened simultaneously (6C2 = 15 pairs).
5. Sensitivity matrix: ∂P(ω_c)/∂l_i for each condition i.

**Metrics:**
- Sensitivity coefficient per condition
- "Collapse threshold": at what l_i does P(ω_c) < 0.5?
- Two-condition interaction effects
- Comparison: binary (all-or-nothing) vs. algebra (gradient visibility)

**Features exercised:** `consent_validity`, `ComplianceOpinion`.

**Deliverable:** Notebook + sensitivity heatmap + collapse threshold table.

**Estimated time:** 1 hour.

---

## E10.5 — Compliance Propagation Through Derivation Chains

**Goal:** Quantify multiplicative lawfulness decay (Theorem 2(f))
through realistic multi-step data derivation chains and show the
practical implications for data governance.

**Hypothesis:** Lawfulness degrades multiplicatively through derivation
chains, and the compliance algebra quantifies this degradation in a way
that binary "pass each step" models cannot.

**Protocol:**
1. Define 3 derivation chain profiles:

   **Chain A — High-quality pipeline (ML production):**
   Each step: τ = (0.95, 0.01, 0.04), π = (0.90, 0.02, 0.08)
   Source: ω_S = (0.90, 0.03, 0.07)

   **Chain B — Mixed-quality pipeline (research prototype):**
   Each step: τ = (0.80, 0.05, 0.15), π = (0.70, 0.10, 0.20)
   Source: ω_S = (0.85, 0.05, 0.10)

   **Chain C — Low-quality pipeline (web scraping):**
   Each step: τ = (0.60, 0.15, 0.25), π = (0.50, 0.20, 0.30)
   Source: ω_S = (0.70, 0.10, 0.20)

2. Compute compliance_propagation for chain lengths 1 through 10.
3. Track: l_D(n), v_D(n), u_D(n) at each depth.
4. Verify Theorem 2(f): l_{D_n} = l_S · ∏(t_i · p_i) holds exactly.
5. Plot all three chain profiles on the same figure.
6. Compute ProvenanceChain equivalence for each.
7. Identify the "compliance horizon" — chain length at which P(ω) < 0.5
   for each profile.

**Metrics:**
- l(n) curve + theoretical prediction (overlay, should match)
- Compliance horizon per profile (expect: A ≈ 8, B ≈ 3, C ≈ 1)
- ProvenanceChain vs. iterative propagation agreement (should be exact)
- Binary comparison: at what depth would binary pass/fail first diverge
  from the algebra's assessment?

**Features exercised:** `compliance_propagation`, `ProvenanceChain`,
`jurisdictional_meet` (via chain associativity).

**Deliverable:** Notebook + chain degradation figure + horizon table.

**Estimated time:** 1 hour.

---

## E10.6 — Erasure Propagation: Table 1 Verification and Scaling

**Goal:** Verify Table 1 of the formalization (erasure degradation)
experimentally and extend to larger lineage graphs.

**Hypothesis:** Composite erasure confidence degrades exponentially
with scope size, making complete erasure assurance practically
impossible for large data lineage graphs — a result with direct
implications for GDPR Art. 17 compliance.

**Protocol:**
1. Verify Table 1 exactly:
   m ∈ {5, 10, 20, 50}, p ∈ {0.99, 0.95, 0.90}
   Compute erasure_scope_opinion for uniform per-node confidence p.
   Verify e_R = p^m within floating-point tolerance.

2. Extend to m ∈ {100, 200, 500, 1000} — scales relevant to
   enterprise data lineage graphs.

3. Non-uniform case: generate realistic per-node erasure confidence
   from Beta(α,β) distributions with varying parameters:
   - Well-governed: Beta(20, 1) → mean p ≈ 0.95
   - Average: Beta(10, 2) → mean p ≈ 0.83
   - Poorly-governed: Beta(5, 3) → mean p ≈ 0.63
   Compute erasure_scope_opinion, compare to p^m with mean p.

4. Residual contamination at varying depths (1 through 20).
   Show monotonic risk increase per Proposition 1.

5. Performance: wall-clock time for erasure_scope_opinion at
   m = 10, 100, 1000, 10000.

**Metrics:**
- Exact Table 1 match (pass/fail)
- Extended table for m up to 1000
- Non-uniform vs. uniform approximation error
- Contamination risk curve
- Throughput (opinions/sec) at scale

**Features exercised:** `erasure_scope_opinion`, `residual_contamination`.

**Deliverable:** Notebook + extended Table 1 + non-uniform comparison +
contamination depth figure + throughput table.

**Estimated time:** 1–2 hours.

---

## E10.7 — Temporal Trigger Interactions and Lifecycle Simulation

**Goal:** Demonstrate that the three trigger types (expiry, review-due,
regulatory change) interact non-commutatively (Theorem 4(e)) and that
realistic compliance lifecycles produce qualitatively different
trajectories under different event orderings.

**Hypothesis:** The same set of regulatory events applied in different
temporal orders produces measurably different final compliance states,
validating the formalization's non-commutativity result.

**Protocol:**
1. Define a baseline compliance opinion and a set of 4 events:
   - t=100: Review due (DPIA, Art. 35(11))
   - t=200: Partial expiry (data retention, γ=0.5, Art. 5(1)(e))
   - t=300: Regulatory change (new adequacy decision)
   - t=400: Hard expiry (full retention period, γ=0.0)

2. Apply events in 3 orderings:
   - Chronological: review → partial → regchange → hard
   - Reversed triggers: hard → regchange → partial → review
   - Interleaved: review → regchange → partial → hard

3. With continuous decay (half_life=365 days) between events.

4. Track ω at each event boundary. Plot l(t), v(t), u(t) trajectories.

5. Measure: final state divergence across orderings.

6. Full lifecycle simulation (1000 days):
   - Day 0: consent assessment (6 conditions)
   - Day 100: consent withdrawal for purpose P₁ (not P₂)
   - Day 200: propagation through 3-step derivation chain
   - Day 300: DPIA review due trigger
   - Day 400: data retention expiry
   - Day 500: erasure request + propagation
   - Day 600: regulatory change (new law)
   - Track compliance opinion throughout, including continuous decay.

**Metrics:**
- State divergence between orderings (L2 norm on [l,v,u] vectors)
- Lifecycle trajectory figures
- Events at which P(ω) crosses decision thresholds

**Features exercised:** `expiry_trigger`, `review_due_trigger`,
`regulatory_change_trigger`, `withdrawal_override`, `consent_validity`,
`compliance_propagation`, `erasure_scope_opinion`, `decay_opinion`.

**Deliverable:** Notebook + trajectory figures + divergence table +
full lifecycle figure.

**Estimated time:** 2 hours.

---

## E10.8 — Independence Assumption Bias Characterization

**Goal:** Honestly quantify the bias introduced by the independence
assumption in the jurisdictional meet, as documented in §12.1.

**Hypothesis:** Under positive correlation (realistic for compliance),
the independence assumption produces measurable and characterizable
bias: non-conservative for compliance operators (underestimates
violation) and conservative for erasure operators (overestimates risk).

**Protocol:**
1. Ground truth construction:
   - Generate 10,000 pairs of correlated compliance opinions using a
     copula model (Gaussian copula with ρ ∈ {0, 0.3, 0.5, 0.7, 0.9}).
   - For each pair, compute "true" composite compliance via Monte Carlo
     simulation (sample 100,000 joint Bernoulli outcomes).

2. Compare independent assumption result (jurisdictional_meet) against
   Monte Carlo ground truth at each correlation level.

3. Measure:
   - Bias in l: E[l_meet − l_true] (expect ≥ 0, non-conservative)
   - Bias in v: E[v_meet − v_true] (expect ≤ 0, non-conservative)
   - RMSE of l_meet vs. l_true
   - Bias magnitude vs. correlation strength (expect monotonic)

4. Repeat for erasure_scope_opinion:
   - Bias in erasure: E[e_meet − e_true] (expect ≤ 0, conservative)
   - Confirm opposite bias direction from compliance operators.

5. Characterize the "safety margin" needed: at what correlation level
   does the bias exceed 5%, 10%, 20%?

**Metrics:**
- Bias vs. correlation curve (both directions)
- RMSE vs. correlation curve
- Safety margin thresholds
- Confirmation of §12.1 bias direction claims

**Features exercised:** `jurisdictional_meet`, `erasure_scope_opinion`,
`ComplianceOpinion`.

**Deliverable:** Notebook + bias curves + safety margin table +
honest assessment paragraph for paper.

**Estimated time:** 2–3 hours (Monte Carlo simulation).

---

## Summary Table

| Exp  | Name                                    | Priority | Est. Time | Key Deliverable                   |
|------|-----------------------------------------|----------|-----------|-----------------------------------|
| E10.1 | Property-based theorem verification     | CRITICAL | 1–2 hr    | 200K+ tests, 0 violations         |
| E10.2 | Binary vs. algebra (Table 2)            | CRITICAL | 1–2 hr    | Decision diversity 6 vs. 3        |
| E10.3 | Multi-jurisdictional scaling            | HIGH     | 1 hr      | l(n) exponential decay curve      |
| E10.4 | Consent condition sensitivity           | HIGH     | 1 hr      | Sensitivity matrix + collapse pts |
| E10.5 | Derivation chain degradation            | HIGH     | 1 hr      | Compliance horizon per profile    |
| E10.6 | Erasure Table 1 + scaling               | HIGH     | 1–2 hr    | Exact table match + extended      |
| E10.7 | Temporal trigger lifecycle              | HIGH     | 2 hr      | Trajectory figures + divergence   |
| E10.8 | Independence bias characterization      | HIGH     | 2–3 hr    | Bias curves + safety margins      |
|      | **Total**                               |          | **10–14 hr** |                                |

---

## Integration with Existing Roadmap

**Track assignment:** Track A (Algebra Core) — extends weeks 3–8.

**Recommended schedule:**
- Week 3–4: E10.1 (property verification) + E10.2 (binary comparison)
- Week 5: E10.3 (multi-jurisdiction) + E10.4 (consent sensitivity)
- Week 6: E10.5 (chain degradation) + E10.6 (erasure scaling)
- Week 7: E10.7 (temporal lifecycle)
- Week 8: E10.8 (independence bias) — the "honest limitations"
  experiment that strengthens the paper's credibility

**Paper section mapping:**
- E10.1 → Supplementary (verification artifact, strengthens rigor)
- E10.2 → Main paper §4 (core result, ~0.5 page)
- E10.3 + E10.5 → Main paper §4 (scaling curves, ~0.5 page)
- E10.4 → Supplementary or main §4 (sensitivity analysis)
- E10.6 → Main paper §4 (Table 1 verification + extension)
- E10.7 → Main paper §4 (lifecycle figure, ~0.25 page)
- E10.8 → Main paper §5 Discussion/Limitations (~0.25 page)

**Critical path:** E10.1 + E10.2 are prerequisites for the paper.
E10.8 is essential for credibility (honest limitation characterization).

---

## Updated Experiment Count

| Suite | # Experiments | Priority | Track |
|-------|:---:|----------|-------|
| E1: Knowledge Fusion | 5 | CRITICAL | A |
| E7: Algebra Superiority | 7 | CRITICAL | A |
| E2: Format Expressiveness | 3 | HIGH | B |
| E3: ML Pipeline Integration | 3 | HIGH | C |
| E8: Feature Showcase | 9 | HIGH | B/C/D |
| E4: Scalability & Perf | 4 | MEDIUM | E |
| E5: Security | 3 | MEDIUM | E |
| E6: Usability | 2 | LOW | E |
| **E10: Compliance Algebra** | **8** | **HIGH** | **A** |
| **Total** | **44** | | |
