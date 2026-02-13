# NeurIPS 2026 Datasets & Benchmarks Track — Experiment Roadmap

**Project:** jsonld-ex — JSON-LD 1.2 Extensions for AI/ML Data Exchange  
**Target Venue:** NeurIPS 2026 Datasets & Benchmarks Track  
**Estimated Deadline:** ~May 2026 (based on NeurIPS 2025 pattern: May 15, 2025)  
**Authors:** Muntaser Syed, Marius Silaghi, Sheikh Abujar, Rwaida Alssadi  
**Date:** February 13, 2026

---

## 1. Paper Narrative & Positioning

### 1.1 Core Thesis

jsonld-ex extends JSON-LD 1.1 with **assertion-level confidence algebra** grounded in Jøsang's Subjective Logic, enabling ML practitioners to attach, propagate, fuse, and filter uncertainty metadata throughout data exchange pipelines — something no existing format supports in a single coherent package.

### 1.2 D&B Track Fit

The NeurIPS D&B track explicitly welcomes:
- **Open-source libraries and tools** that enable or accelerate ML research ✓
- **Data-centric AI methods and tools** to measure and improve data quality ✓
- **Benchmarks** for evaluating data-centric approaches ✓

jsonld-ex fits as a **tool + benchmark** submission:
- The tool: a Python library extending JSON-LD with confidence algebra, validation, and interop
- The benchmark: a comprehensive evaluation suite demonstrating where confidence-aware data exchange improves ML pipeline outcomes

### 1.3 Key Differentiators (What Reviewers Must Walk Away Knowing)

1. **No existing format** provides assertion-level uncertainty that is both mathematically grounded AND interoperable with W3C standards
2. **Subjective Logic fusion** outperforms scalar averaging in multi-source scenarios (conflict detection, regime changes, Byzantine agents)
3. **Backward compatible** with JSON-LD 1.1 — zero migration cost for existing data
4. **Interoperable** — bidirectional round-trip with PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant

### 1.4 Comparison Targets

| Tool | What it does well | What it lacks |
|------|------------------|---------------|
| Croissant (MLCommons) | Dataset cards, ML metadata | No assertion-level confidence |
| PROV-O (W3C) | Provenance graphs | Verbose (7× more triples), no uncertainty algebra |
| SHACL (W3C) | RDF validation | Complex, no inline validation syntax |
| PyLD / rdflib | JSON-LD / RDF processing | No confidence, no validation, no fusion |
| Hugging Face datasets | Dataset loading | Format-specific, no semantic interop |

---

## 2. Experiment Suites Overview

Eight experiment suites, ordered by importance to the paper narrative:

| Suite | Name | Novel Contribution | Priority |
|-------|------|-------------------|----------|
| **E1** | Confidence-Aware Knowledge Fusion | Core novelty — SL vs scalar in realistic scenarios | **CRITICAL** |
| **E7** | Confidence Algebra Superiority | Head-to-head SL vs traditional uncertainty methods | **CRITICAL** |
| **E2** | Format Expressiveness Comparison | Quantitative superiority claims | **HIGH** |
| **E3** | End-to-End ML Pipeline Integration | Practical impact demonstration | **HIGH** |
| **E8** | Feature Showcase | Validation, IoT, temporal, vectors, transport | **HIGH** |
| **E4** | Scalability & Performance | Reviewer due-diligence | **MEDIUM** |
| **E5** | Security & Integrity Validation | Differentiation claim | **MEDIUM** |
| **E6** | Developer Usability Study | Adoption argument | **LOW** |

---

## 3. Experiment Suite E1: Confidence-Aware Knowledge Fusion

**Goal:** Demonstrate that Subjective Logic fusion outperforms scalar confidence approaches in realistic multi-source ML data exchange scenarios.

This is the paper's **most novel contribution** and must be the strongest section.

### E1.1 — Multi-Source NER Fusion (Synthetic + Real)

**Hypothesis:** When multiple NER models disagree on entity labels, SL cumulative fusion with conflict detection produces better F1 than scalar averaging or majority voting.

**Protocol:**
1. Select 3-4 pretrained NER models (e.g., spaCy, Flair, Stanza, an LLM-based NER)
2. Run all models on CoNLL-2003 test set
3. For each token, collect per-model confidence scores
4. Compare fusion strategies:
   - Baseline A: Scalar majority voting (highest average confidence wins)
   - Baseline B: Weighted average by model accuracy on dev set
   - jsonld-ex C: `Opinion.from_confidence()` → `cumulative_fuse()` → select by projected probability
   - jsonld-ex D: Same as C but with `conflict_metric()` → abstain when conflict > threshold
5. Measure: Token-level F1, Entity-level F1, Abstention rate, Precision on non-abstained

**Why this matters:** Multi-source NER fusion is a real production scenario (combining cheap models). The abstention behavior under conflict is unique to SL — scalar approaches can't distinguish "models agree it's 0.5" from "models violently disagree."

**Data:** CoNLL-2003 (public, well-known), optionally OntoNotes 5.0

**jsonld-ex features exercised:** `Opinion.from_confidence`, `cumulative_fuse`, `conflict_metric`, `annotate` (to attach per-model opinions), `merge_graphs`, `filter_by_confidence`

### E1.2 — Temporal Regime Adaptation

**Hypothesis:** SL opinions with temporal decay adapt to distributional shift faster than recursive Bayesian updating.

**Protocol:**
1. Generate synthetic streaming data: 100 timesteps of `P(positive)=0.8`, then sudden shift to `P(positive)=0.2`
2. At each timestep, generate evidence batch (e.g., 5 observations)
3. Compare adaptation speed:
   - Baseline: Bayesian sequential update (Beta prior → posterior)
   - jsonld-ex: `Opinion.from_evidence()` → `cumulative_fuse()` with `decay_opinion()` (λ=0.8, 0.9, 0.95)
4. Measure: KL divergence from true distribution at each timestep, time-to-recovery after shift

**Why this matters:** Streaming ML systems need to detect and adapt to concept drift. The SL decay mechanism provides explicit uncertainty growth over time, while Bayesian priors accumulate inertia.

**Data:** Synthetic (reproducible with fixed seeds), optionally real sensor drift dataset

**jsonld-ex features exercised:** `Opinion.from_evidence`, `cumulative_fuse`, `decay_opinion`, `add_temporal`, `query_at_time`

### E1.3 — Byzantine-Robust Multi-Agent Fusion

**Hypothesis:** `robust_fuse()` with conflict-based agent filtering survives adversarial/faulty agents that scalar fusion cannot detect.

**Protocol:**
1. Simulate 7 agents observing a binary proposition
2. 5 honest agents with `b ∈ [0.65, 0.85]`, 2 rogue agents with `b ∈ [0.0, 0.15]`
3. Compare:
   - Baseline A: Scalar average of all 7 → pulled toward rogues
   - Baseline B: Trimmed mean (remove min/max) → loses honest information
   - jsonld-ex C: `robust_fuse(opinions, threshold=0.15)` → identifies and removes rogues
4. Vary: number of rogues (0, 1, 2, 3), rogue strategy (random, coordinated, mimicking)
5. Measure: Distance from honest-only ground truth, false removal rate, true detection rate

**Why this matters:** Multi-agent knowledge graph construction (e.g., federated annotation) must handle unreliable contributors. This is directly relevant to the D&B track's emphasis on data quality.

**Data:** Synthetic (controlled adversarial scenarios)

**jsonld-ex features exercised:** `robust_fuse`, `pairwise_conflict`, `conflict_metric`, `cumulative_fuse`

### E1.4 — Trust Discount Chain Degradation

**Hypothesis:** In multi-hop trust chains, scalar multiplication decays to false disbelief while SL trust discount correctly converges to the base rate (uncertainty).

**Protocol:**
1. Chain lengths 1 through 20
2. Each link has trust = 0.85
3. Original opinion: b=0.9, d=0.05, u=0.05
4. Compare:
   - Scalar: c_{n+1} = trust × c_n → exponential decay to 0
   - SL: `trust_discount(trust_opinion, current_opinion)` → converges to a=0.5
5. Plot both curves; measure divergence from rational behavior

**Why this matters:** Provenance chains in knowledge graphs involve multiple hops. Scalar trust decays catastrophically; SL preserves rational uncertainty.

**Data:** Synthetic (mathematical comparison)

**jsonld-ex features exercised:** `trust_discount`, `Opinion`, `projected_probability`

### E1.5 — Deduction Under Uncertainty

**Hypothesis:** SL deduction preserves calibration when reasoning through uncertain conditionals, while naive probability multiplication does not.

**Protocol:**
1. Ground truth: P(disease | symptom) from medical knowledge base
2. Create uncertain opinions about symptoms with varying evidence levels
3. Apply deduction to derive disease probabilities
4. Compare:
   - Baseline: P(disease) = P(symptom) × P(disease|symptom) (scalar)
   - jsonld-ex: `deduce(opinion_symptom, opinion_disease_given_symptom, opinion_disease_given_not_symptom)`
5. Measure: ECE (Expected Calibration Error), Brier score

**Data:** Synthetic medical scenario with known ground truth probabilities

**jsonld-ex features exercised:** `deduce`, `Opinion.from_evidence`, `projected_probability`

---

## 4. Experiment Suite E2: Format Expressiveness Comparison

**Goal:** Quantitatively demonstrate that jsonld-ex is more expressive and concise than alternatives for common ML data exchange tasks.

### E2.1 — Verbosity Comparison (Byte Ratio)

**Protocol:**
1. Express the same 10 ML-relevant scenarios in each format:
   - Dataset card with splits and field definitions
   - NER annotations with per-token confidence
   - Sensor reading with uncertainty and calibration
   - Knowledge graph triple with provenance chain
   - Model prediction with confidence, source, timestamp
   - Multi-language content with translation provenance
   - Temporal validity window
   - Validation shape for a Person entity
   - Multi-source fused assertion
   - Invalidated/retracted claim
2. Measure bytes in each format:
   - jsonld-ex (JSON-LD + extensions)
   - PROV-O (RDF/Turtle)
   - SHACL (for validation scenarios)
   - Croissant (for dataset scenarios)
   - Plain JSON (no semantics)
   - Raw JSON-LD 1.1 (manual provenance)
3. Compute byte ratios, node expansion factors

**Already partially measured:** Existing benchmarks show PROV-O is ~4.75× more verbose. Need to extend to all 10 scenarios and all comparison targets.

**jsonld-ex features exercised:** All annotation fields, `to_prov_o`, `shape_to_shacl`, `to_croissant`, `to_rdf_star_ntriples`

### E2.2 — Feature Coverage Matrix

**Protocol:**
1. Define 30+ ML-relevant features (confidence, provenance, temporal validity, validation, vector embeddings, calibration, delegation, invalidation, etc.)
2. For each feature, assess whether each format supports it natively:
   - Natively supported (built-in syntax)
   - Achievable with extension/workaround
   - Not possible
3. Present as heatmap table

**This is a qualitative comparison** but backed by concrete code examples showing how (or whether) each feature can be expressed.

### E2.3 — Round-Trip Fidelity

**Protocol:**
1. Create 100 richly-annotated documents using jsonld-ex
2. Convert to each interop format and back:
   - jsonld-ex → PROV-O → jsonld-ex
   - jsonld-ex → SHACL → jsonld-ex
   - jsonld-ex → OWL → jsonld-ex
   - jsonld-ex → RDF-Star → jsonld-ex
   - jsonld-ex → SSN/SOSA → jsonld-ex
   - jsonld-ex → Croissant → jsonld-ex
3. Measure: Fields preserved, fields lost, fields transformed
4. Report fidelity percentage per format

**Already partially measured:** Existing benchmarks show 100% confidence fidelity for PROV-O round-trip. Need to extend to all formats and all fields.

---

## 5. Experiment Suite E3: End-to-End ML Pipeline Integration

**Goal:** Show that jsonld-ex provides practical value in real ML workflows, not just theoretical advantages.

### E3.1 — RAG Pipeline with Provenance-Aware Retrieval

**Hypothesis:** Filtering RAG context by confidence and detecting conflict between retrieved passages reduces hallucination.

**Protocol:**
1. Use a QA dataset (e.g., Natural Questions, TrivialQA)
2. Create a retrieval corpus with both correct and "poisoned" passages (paraphrased wrong answers)
3. Pipeline:
   a. Retrieve top-k passages per query using sentence-transformers
   b. Annotate each passage with retrieval confidence: `annotate(passage, confidence=cosine_sim, source=doc_id)`
   c. Fuse passage opinions: `cumulative_fuse(*passage_opinions)`
   d. If `conflict_metric(fused) > threshold`: abstain
   e. Otherwise: feed top passages to LLM for answer generation
4. Compare:
   - Baseline: Standard top-k RAG (no filtering)
   - jsonld-ex: Confidence-filtered, conflict-aware RAG
5. Measure: Answer accuracy, hallucination rate, abstention rate

**GPU required:** Yes (sentence-transformers, possibly LLM). Suggest Colab notebook.

**jsonld-ex features exercised:** `annotate`, `Opinion.from_confidence`, `cumulative_fuse`, `conflict_metric`, `filter_by_confidence`, `merge_graphs`

### E3.2 — TruthfulQA Evaluation

**Hypothesis:** On the TruthfulQA benchmark, conflict-aware source fusion improves truthfulness scores compared to naive retrieval.

**Protocol:**
1. Load TruthfulQA dataset (818 questions designed to elicit false answers)
2. For each question, retrieve context from multiple sources
3. Apply jsonld-ex confidence fusion and conflict detection
4. Compare truthfulness scores (MC1, MC2 metrics) with and without conflict filtering

**GPU required:** Yes. Colab notebook.

**Data:** TruthfulQA (public, established benchmark)

### E3.3 — Croissant Metadata Enrichment Pipeline

**Hypothesis:** jsonld-ex's Croissant interop enables enriching standard ML dataset cards with confidence and provenance metadata that Croissant alone cannot express.

**Protocol:**
1. Take 5 real Croissant dataset cards from MLCommons
2. Import via `from_croissant()`
3. Add jsonld-ex annotations (confidence on field completeness, provenance on metadata sources, temporal validity)
4. Export back via `to_croissant()` — verify base Croissant fields preserved
5. Show the enriched metadata enables queries Croissant alone cannot answer:
   - "Which fields have confidence > 0.9?"
   - "When was this metadata last verified?"
   - "What is the provenance chain for the license field?"

**No GPU required.**

**jsonld-ex features exercised:** `from_croissant`, `to_croissant`, `annotate`, `validate_document`, `create_dataset_metadata`

---

## 6. Experiment Suite E4: Scalability & Performance

**Goal:** Demonstrate that jsonld-ex's extensions add acceptable overhead and scale to production workloads.

### E4.1 — Throughput Benchmarks (Existing, Refresh)

**Protocol:** Re-run existing benchmark suite with v0.3.5:
- Fusion throughput: cumulative, averaging, robust (varying n=2..100)
- Trust discount chains: 1..20 hops
- Annotation throughput: annotate() at scale (100, 1K, 10K, 100K nodes)
- Validation throughput: validate_node() at scale
- Interop conversion throughput: to/from PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA

**Existing benchmarks:** 8 scripts in `benchmarks/` directory. Results may be stale (recorded at v0.2.0).

### E4.2 — Comparison Against Baselines (Existing, Refresh)

**Protocol:** Re-run bench_baselines.py comparisons:
- PROV-O construction: rdflib manual triples vs jsonld-ex `annotate()`
- SHACL validation: pyshacl vs jsonld-ex `validate_node()`
- Graph merge: rdflib Graph union vs jsonld-ex `merge_graphs()`
- Temporal query: SPARQL via rdflib vs jsonld-ex `query_at_time()`

### E4.3 — Scaling Study

**Protocol:**
1. Generate graphs at sizes: 1K, 10K, 100K, 1M nodes
2. Measure wall-clock time for:
   - Full annotation pass
   - Confidence filtering
   - Graph merge (two graphs of equal size)
   - Full validation pass
3. Plot scaling curves (should be linear or near-linear)
4. Report peak memory usage

### E4.4 — Batch API Overhead

**Protocol:**
1. Compare per-item vs batch API for 10K items:
   - `annotate()` in loop vs `annotate_batch()`
   - `validate_node()` in loop vs `validate_batch()`
   - `filter_by_confidence()` in loop vs `filter_by_confidence_batch()`
2. Measure: Wall-clock speedup, memory delta

---

## 7. Experiment Suite E5: Security & Integrity Validation

**Goal:** Demonstrate that jsonld-ex's security features address real attack vectors in ML data exchange.

### E5.1 — Context Integrity Verification

**Protocol:**
1. Create valid JSON-LD documents with integrity hashes
2. Tamper with contexts (simulate DNS poisoning, MITM)
3. Verify that `verify_integrity()` detects all tampering
4. Measure: False positive rate (should be 0), detection latency

### E5.2 — Context Allowlist Enforcement

**Protocol:**
1. Define allowlist of trusted contexts
2. Attempt to load documents with unauthorized contexts
3. Verify rejection behavior of `is_context_allowed()`
4. Test edge cases: wildcards, subdomain matching, scheme enforcement

### E5.3 — Validation as Security Layer

**Protocol:**
1. Create malformed/adversarial JSON-LD documents:
   - Type confusion attacks
   - Cardinality violations
   - Cross-property constraint violations
   - Deeply nested structures (DoS attempt)
2. Run through `validate_document()`
3. Measure: Detection rate, processing time, memory safety

---

## 8. Experiment Suite E6: Developer Usability Study

**Goal:** Provide evidence that jsonld-ex is practical for ML engineers unfamiliar with semantic web technologies.

### E6.1 — Lines-of-Code Comparison

**Protocol:**
1. Define 5 common ML data exchange tasks
2. Implement each in:
   - jsonld-ex
   - rdflib + pyshacl (manual RDF approach)
   - Plain JSON (no semantics, manual provenance)
3. Count: Lines of code, number of imports, setup complexity

### E6.2 — API Learnability Assessment (Optional)

**Protocol:**
1. Write 5 tutorial tasks of increasing complexity
2. Recruit 5-10 ML graduate students (if feasible)
3. Measure: Time to complete, errors made, satisfaction rating
4. Compare against equivalent rdflib tasks

**Note:** Human studies require IRB approval. If infeasible by deadline, replace with API surface comparison (number of functions, parameter complexity, documentation coverage).

---

## 9. Experiment Suite E7: Confidence Algebra Superiority

**Goal:** Provide rigorous, head-to-head evidence that Subjective Logic confidence algebra is superior to traditional scalar and probabilistic uncertainty methods across multiple dimensions: information preservation, calibration, composability, and decision quality.

This suite directly addresses the anticipated reviewer question: *"Why not just use scalar confidence / Bayesian / ensemble methods?"*

### E7.1 — Information Preservation: Scalar Collapse Demonstration

**Hypothesis:** A scalar confidence value conflates distinct epistemic states that the SL opinion tuple preserves, leading to information loss that compounds in downstream decisions.

**Protocol:**
1. Construct 4 canonical epistemic states, all with projected probability P(ω) = 0.5:
   - State A: Strong balanced evidence — b=0.45, d=0.45, u=0.1 (high conflict)
   - State B: Total ignorance — b=0.0, d=0.0, u=1.0, a=0.5 (no evidence)
   - State C: Moderate belief with uncertainty — b=0.3, d=0.1, u=0.6 (leaning positive, unsure)
   - State D: Dogmatic coin flip — b=0.5, d=0.5, u=0.0 (certain it's 50/50)
2. Feed each into 3 downstream decision tasks:
   - Accept/reject at threshold 0.5 (all identical under scalar)
   - Request more data if uncertainty > 0.3 (only SL can decide)
   - Flag for human review if conflict > 0.2 (only SL can decide)
3. Show that scalar systems make identical decisions for all 4 states while SL makes appropriate, distinct decisions for each
4. Quantify information loss via Shannon entropy of the decision distribution

**Metrics:** Decision accuracy across states, information-theoretic capacity (bits preserved)

**jsonld-ex features:** `Opinion`, `conflict_metric`, `projected_probability`

### E7.2 — Calibration Comparison: SL Fusion vs Scalar Ensembles

**Hypothesis:** Opinions fused via `cumulative_fuse` produce better-calibrated probability estimates than scalar ensemble methods (mean, median, max, product) when source reliability varies.

**Protocol:**
1. Generate 1000 binary classification scenarios with known ground truth
2. For each, simulate 5 sources with varying reliability (accuracy ∈ [0.55, 0.95])
3. Apply fusion methods:
   - Scalar mean, median, max, geometric mean
   - Bayesian model averaging
   - jsonld-ex `cumulative_fuse` (opinions from `from_confidence` with estimated uncertainty)
   - jsonld-ex `averaging_fuse` (for correlated sources)
4. Measure calibration:
   - Expected Calibration Error (ECE, 15 bins)
   - Brier score
   - Reliability diagram (visual)
   - Log loss
5. Vary source correlation (0%, 50%, 100%) — `averaging_fuse` should outperform `cumulative_fuse` under correlation

**Key insight:** SL fusion should excel when sources have unknown/varying reliability, because the uncertainty dimension captures "how much evidence" not just "what the evidence says."

**jsonld-ex features:** `cumulative_fuse`, `averaging_fuse`, `Opinion.from_confidence`, `Opinion.from_evidence`

### E7.3 — Composability: Associativity and Order-Independence

**Hypothesis:** SL cumulative fusion is associative and commutative — the order of fusion doesn't affect the result — while many scalar methods are order-dependent.

**Protocol:**
1. Generate 100 sets of 5 opinions each
2. For each set, fuse in every permutation order:
   - `cumulative_fuse(a, b, c, d, e)` vs `cumulative_fuse(e, d, c, b, a)` vs all 120 permutations
   - Scalar running average: result depends on order
   - Bayesian sequential update: result depends on order (prior sensitivity)
3. Measure: Maximum deviation across all orderings (should be 0 for SL cumulative, >0 for baselines)
4. Also verify:
   - `cumulative_fuse(a, vacuous) = a` (identity element)
   - Uncertainty monotonically decreases under cumulative fusion

**Why this matters:** In distributed systems (multi-agent KG construction), messages arrive in arbitrary order. Order-dependent fusion produces inconsistent global state. This is a formal mathematical property, not a heuristic argument.

**jsonld-ex features:** `cumulative_fuse`, `averaging_fuse` (note: NOT associative — verify this too, as it demonstrates awareness of limitations)

### E7.4 — Evidence Accumulation Dynamics

**Hypothesis:** `Opinion.from_evidence()` correctly captures the epistemic trajectory from ignorance to certainty as evidence accumulates, while Bayesian and scalar approaches show pathological behavior.

**Protocol:**
1. Ground truth: P(positive) = 0.7
2. Generate evidence stream: 500 observations drawn from Bernoulli(0.7)
3. At each step, compare three belief representations:
   - Bayesian: Beta(α + r, β + s) posterior, MAP estimate
   - Scalar: Running mean of observations
   - SL: `Opinion.from_evidence(r, s, prior_weight=2)`
4. Track at each step:
   - Uncertainty estimate (Beta variance vs SL uncertainty)
   - Distance from true probability
   - Posterior predictive accuracy
5. Introduce an evidence anomaly at step 250 (20 consecutive negatives from a different process) — measure recovery

**Key insight:** SL uncertainty has a clear semantics (W/(r+s+W) → 0 as evidence grows) while Beta variance has a different shape. Neither is "wrong" but SL's direct interpretability is the advantage.

**jsonld-ex features:** `Opinion.from_evidence`, `cumulative_fuse`, `decay_opinion`

### E7.5 — Cumulative vs Averaging Fusion: When Correlation Matters

**Hypothesis:** `cumulative_fuse` gives correct results for independent sources, `averaging_fuse` for correlated sources, and using the wrong one produces measurable error.

**Protocol:**
1. Generate ground truth binary variable
2. Create 5 sources:
   - Independent: each independently observes with noise
   - Correlated: sources share a common noise component
3. Fuse using:
   - `cumulative_fuse` on independent sources (correct) vs correlated sources (incorrect — overcounts evidence)
   - `averaging_fuse` on correlated sources (correct) vs independent sources (incorrect — undercounts evidence)
4. Measure: ECE, Brier, uncertainty level
5. Show that misapplying the fusion type produces systematic bias:
   - Cumulative on correlated → overconfident (uncertainty too low)
   - Averaging on independent → underconfident (uncertainty too high)

**Why this matters:** No scalar method even has this distinction. The ability to choose between fusion operators based on source independence is a unique SL capability.

**jsonld-ex features:** `cumulative_fuse`, `averaging_fuse`, `_averaging_fuse_nary`

### E7.6 — Decay Function Comparison Under Concept Drift

**Hypothesis:** Different decay functions (exponential, linear, step) produce measurably different adaptation profiles, and the optimal choice depends on drift characteristics.

**Protocol:**
1. Generate 3 types of concept drift:
   - Sudden drift (P jumps from 0.8 to 0.2 at t=500)
   - Gradual drift (P linearly changes from 0.8 to 0.2 over t=200..700)
   - Recurring drift (P oscillates between 0.8 and 0.2 with period 200)
2. For each drift type, apply:
   - `exponential_decay` with half_life ∈ {50, 100, 200}
   - `linear_decay` with same half_lives
   - `step_decay` with same half_lives
   - Bayesian forgetting (exponential downweighting of old observations)
   - ADWIN (Adaptive Windowing — standard concept drift detector)
3. Measure: Mean absolute error from true P at each timestep, time-to-detection of drift

**jsonld-ex features:** `decay_opinion`, `exponential_decay`, `linear_decay`, `step_decay`, `Opinion.from_evidence`, `cumulative_fuse`

### E7.7 — Sensitivity Analysis: Parameter Robustness

**Hypothesis:** SL operator results are robust to reasonable parameter choices (base_rate, prior_weight) — the advantage over scalar methods persists across a wide range of settings.

**Protocol:**
1. Take the best-performing E1.1 (NER fusion) and E1.3 (Byzantine) configurations
2. Sweep parameters:
   - `base_rate` ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
   - `prior_weight` (for from_evidence) ∈ {1, 2, 5, 10}
   - `threshold` (for robust_fuse) ∈ {0.05, 0.10, 0.15, 0.20, 0.30}
   - `uncertainty` (for from_confidence) ∈ {0.0, 0.1, 0.2, 0.3, 0.5}
3. For each parameter combination, report:
   - Key metric (F1 for NER, ground truth distance for Byzantine)
   - Is SL still better than scalar baseline?
4. Report: percentage of parameter space where SL outperforms scalar

**Why this matters:** Reviewers will ask "how sensitive are these results to hyperparameters?" This provides a definitive answer.

**jsonld-ex features:** All algebra operators with parameter sweeps

---

## 10. Experiment Suite E8: Feature Showcase

**Goal:** Demonstrate the breadth of jsonld-ex beyond confidence algebra — showcasing validation, IoT/sensor pipelines, temporal modeling, vector embeddings, compact transport, and graph operations. These features justify the "comprehensive toolkit" claim.

### E8.1 — Validation Framework: SHACL Replacement Study

**Hypothesis:** jsonld-ex's native validation (`@shape`) achieves equivalent constraint coverage to SHACL/pyshacl with significantly less code and no RDF knowledge required.

**Protocol:**
1. Define 10 validation scenarios of increasing complexity:
   - Required fields (V1: `@minCount`)
   - Datatype + pattern constraints (V1-V2)
   - Enumeration constraints (V2: `@in`)
   - Cardinality (V1: `@minCount`, `@maxCount`)
   - Logical combinators (V3: `@or`, `@and`, `@not`)
   - Cross-property constraints (V4: `@lessThan`, `@equals`)
   - Nested shape validation (V5)
   - Conditional validation (V7: `@if`/`@then`/`@else`)
   - Shape inheritance (OWL1: `@extends`)
   - Severity levels (V6: `@severity`)
2. Implement each in:
   - jsonld-ex: `validate_node()` with `@shape`
   - pyshacl: SHACL shapes graph + `pyshacl.validate()`
   - JSON Schema: equivalent constraints where possible
3. Compare:
   - Lines of code (shape definition + validation call)
   - Bytes of constraint definition
   - Validation throughput (documents/sec)
   - Coverage: which constraints each tool can express
4. Measure round-trip: `shape_to_shacl()` → pyshacl validates → `shacl_to_shape()` → jsonld-ex validates → same results?

**jsonld-ex features:** Full validation module (V1-V7, OWL1), `shape_to_shacl`, `shacl_to_shape`, `shape_to_owl_restrictions`, `owl_to_shape`

### E8.2 — IoT Sensor Pipeline: SSN/SOSA Integration

**Hypothesis:** jsonld-ex provides a practical, lightweight alternative to full SSN/SOSA for IoT sensor metadata with measurement uncertainty, calibration tracking, and aggregation.

**Protocol:**
1. Simulate a smart building with 50 sensors (temperature, humidity, CO2, occupancy)
2. Generate 24 hours of readings at 5-minute intervals (288 readings × 50 sensors = 14,400 data points)
3. Pipeline:
   a. Create sensor readings with `annotate()` including `@measurementUncertainty`, `@unit`, `@calibratedAt`, `@calibrationMethod`
   b. Aggregate readings hourly with `@aggregationMethod='mean'`, `@aggregationWindow='PT1H'`, `@aggregationCount=12`
   c. Apply temporal decay: readings older than 6 hours get decayed confidence
   d. Flag sensors needing recalibration: `@calibratedAt` older than 30 days
   e. Export to SSN/SOSA with `to_ssn()` for standards-compliant downstream systems
   f. Import from SSN/SOSA with `from_ssn()` — verify round-trip fidelity
4. Compare:
   - jsonld-ex annotation size vs full SSN/SOSA Turtle representation
   - Code complexity: jsonld-ex vs manual rdflib SSN construction
   - Query expressiveness: "find all sensors with uncertainty > 0.5 AND calibration older than 7 days"

**jsonld-ex features:** `annotate` (IoT fields), `to_ssn`, `from_ssn`, `decay_opinion`, `filter_by_confidence`, `annotate_batch`, `add_temporal`, `query_at_time`

### E8.3 — Temporal Modeling: Snapshot Queries and Diff

**Hypothesis:** jsonld-ex's temporal extensions enable time-aware knowledge graph queries that are impractical with standard JSON-LD.

**Protocol:**
1. Create a knowledge graph of an organization over 3 years:
   - Employee joins/leaves (temporal validity windows)
   - Role changes (temporal annotations on relationships)
   - Department restructurings
2. Execute temporal queries:
   - "Who worked in Engineering on 2024-06-15?" → `query_at_time()`
   - "What changed between Q1 and Q2 2024?" → `temporal_diff()`
   - "Show the history of Alice's role" → temporal range query
3. Compare:
   - jsonld-ex: `add_temporal()` + `query_at_time()` + `temporal_diff()`
   - rdflib + SPARQL: manual temporal triples + FILTER clauses
   - Plain JSON: custom date filtering code
4. Measure: Query accuracy, code complexity, response time

**jsonld-ex features:** `add_temporal`, `query_at_time`, `temporal_diff`, `TemporalDiffResult`

### E8.4 — Vector Embeddings: Coexistence with Symbolic Data

**Hypothesis:** jsonld-ex's `@vector` container type enables vector embeddings to coexist with symbolic knowledge graph data in a single document, enabling hybrid symbolic-neural retrieval.

**Protocol:**
1. Create a product catalog (500 items) with:
   - Symbolic data: name, price, category, manufacturer (JSON-LD properties)
   - Vector embeddings: product descriptions encoded via sentence-transformers (768-dim)
2. Define vector term with `vector_term_definition('embedding', 'ex:embedding', dimensions=768)`
3. Demonstrate hybrid queries:
   - "Find products similar to X" → `cosine_similarity()` on embeddings
   - "Find products similar to X in category Y" → vector + SPARQL-like filter
   - "Find products with confidence > 0.8 similar to X" → vector + confidence filter
4. Validate: `validate_vector()` catches dimension mismatches, non-numeric values
5. Compare: jsonld-ex hybrid doc vs separate JSON + numpy array storage

**jsonld-ex features:** `vector_term_definition`, `validate_vector`, `cosine_similarity`, `annotate`, `filter_by_confidence`

### E8.5 — CBOR-LD Compact Transport: IoT Bandwidth Savings

**Hypothesis:** CBOR-LD serialization reduces payload size by 50-80% compared to JSON-LD, enabling jsonld-ex annotations on bandwidth-constrained IoT networks.

**Protocol:**
1. Create annotated sensor readings at various complexity levels:
   - Minimal: value + confidence
   - Medium: value + confidence + source + timestamp + unit
   - Full: all IoT annotation fields (uncertainty, calibration, aggregation)
2. Serialize each as:
   - JSON-LD (standard JSON)
   - CBOR-LD (`to_cbor()`)
   - MQTT payload (`to_mqtt_payload()`)
   - MessagePack (baseline binary)
   - Protocol Buffers (pre-compiled schema)
3. Measure: payload bytes, compression ratio, serialization throughput, deserialization throughput
4. Demonstrate MQTT integration:
   - `derive_mqtt_topic()` from sensor metadata
   - `derive_mqtt_qos()` from confidence level (high confidence → QoS 0, low → QoS 1)
5. Round-trip: `to_cbor()` → `from_cbor()` → verify all fields preserved

**jsonld-ex features:** `to_cbor`, `from_cbor`, `payload_stats`, `to_mqtt_payload`, `from_mqtt_payload`, `derive_mqtt_topic`, `derive_mqtt_qos`

### E8.6 — Graph Merge and Diff Operations

**Hypothesis:** jsonld-ex's confidence-aware graph merge produces higher quality fused knowledge graphs than naive union operations.

**Protocol:**
1. Create 3 partially overlapping knowledge graphs (100 nodes each, ~30% overlap):
   - Graph A: high-confidence source (academic database)
   - Graph B: medium-confidence source (web scraping)
   - Graph C: low-confidence source (crowdsourcing)
2. Introduce deliberate conflicts in overlapping regions (10% contradicting facts)
3. Merge using:
   - rdflib `Graph.__iadd__()` (naive union — last wins)
   - jsonld-ex `merge_graphs()` (confidence-aware — highest confidence wins)
4. Evaluate merged graph against ground truth
5. Use `diff_graphs()` to identify exactly what changed between original and merged graphs
6. Generate `MergeReport` showing conflict resolution decisions

**jsonld-ex features:** `merge_graphs`, `diff_graphs`, `MergeReport`, `MergeConflict`, `resolve_conflict`

### E8.7 — Context Versioning and Migration

**Hypothesis:** jsonld-ex's context versioning tools detect breaking changes and enable safe schema evolution.

**Protocol:**
1. Simulate a schema evolution scenario:
   - v1.0: base context with 20 terms
   - v1.1: adds 5 terms (non-breaking)
   - v2.0: renames 2 terms, changes type of 1 term, removes 1 term (breaking)
2. Run `context_diff()` between each version pair
3. Run `check_compatibility()` to classify changes as breaking/non-breaking
4. Demonstrate migration use case: document written against v1.0 processed with v2.0 context → detect incompatibility → guide user on fixes
5. Compare: jsonld-ex `check_compatibility()` vs manual diffing of JSON contexts

**jsonld-ex features:** `context_diff`, `check_compatibility`, `ContextDiff`, `CompatibilityResult`

### E8.8 — Provenance Chain Tracking: Full Lifecycle Demo

**Hypothesis:** jsonld-ex's annotation system supports complete data lifecycle provenance — from creation through delegation, derivation, and eventual invalidation.

**Protocol:**
1. Simulate a data lifecycle:
   a. Original extraction: `annotate(data, source='model-v1', confidence=0.8, method='NER')`
   b. Human review: `annotate(data, source='reviewer-A', confidence=0.95, humanVerified=True)`
   c. Delegation: `annotate(data, delegatedBy='team-lead')`
   d. Derivation: `annotate(derived_data, derivedFrom='original-id')`
   e. Translation: `annotate(translated, translatedFrom='en', translationModel='opus-mt')`
   f. Invalidation: `annotate(old_data, invalidatedAt='2025-12-01', invalidationReason='superseded')`
2. Extract full provenance chain: `get_provenance()` at each stage
3. Filter active (non-invalidated) data: `filter_by_confidence()` with invalidation exclusion
4. Convert to PROV-O: `to_prov_o()` → verify complete Activity/Entity/Agent graph
5. Convert to RDF-Star: `to_rdf_star_ntriples()` → verify all annotation fields preserved

**jsonld-ex features:** `annotate` (all 22 fields), `get_provenance`, `filter_by_confidence`, `to_prov_o`, `from_prov_o`, `to_rdf_star_ntriples`, `from_rdf_star_ntriples`

### E8.9 — MCP Server: LLM Agent Integration

**Hypothesis:** The 41-tool MCP server enables LLM agents to perform confidence-aware knowledge graph operations without manual code, demonstrating the tool's applicability in the agentic AI era.

**Protocol:**
1. Document the 41 MCP tools and their input/output schemas
2. Demonstrate 5 agentic workflows:
   - Agent annotates extracted entities with confidence
   - Agent fuses opinions from multiple extraction passes
   - Agent validates a document against a shape
   - Agent queries temporal state of a knowledge graph
   - Agent detects and resolves conflicts in merged data
3. Compare: MCP tool calls vs equivalent Python API calls (should be 1:1)
4. Measure: Tool call latency overhead vs direct API

**Note:** This is primarily a qualitative demonstration for the paper's "broader impact" section. Quantitative results focus on latency overhead.

---

## 11. NeurIPS D&B Track Requirements Checklist

| Requirement | Status | Action Needed |
|------------|--------|---------------|
| Open-source code | ✅ GitHub (MIT/Apache?) | Verify license |
| Code documented | ⚠️ Partial | Complete API docs before submission |
| Publicly available by camera-ready | ✅ PyPI | Ensure stable release |
| Croissant metadata | ⚠️ Can generate | Create Croissant card for jsonld-ex itself |
| Dataset documentation | N/A (tool, not dataset) | Document benchmark suite |
| Author statement (rights) | ❌ Not yet | Draft for supplementary |
| Explicit license | ⚠️ Check | Confirm license in repo |
| Long-term preservation | ✅ PyPI + GitHub | Consider Zenodo archive |
| Reproducibility | ⚠️ Partial | Pin seeds, log env, `requirements-bench.txt` |
| Supplementary materials | ❌ Not yet | Prepare datasheet, extended results |

---

## 12. 13-Week Execution Timeline (Parallel)

**Week 0 = ~February 17, 2026.  Target submission: ~May 15, 2026.**

Multiple team members execute experiments in parallel across platforms. Work is divided into **tracks** that can proceed independently after shared infrastructure is in place.

### Team Allocation

| Track | Owner(s) | Platform | Focus |
|-------|----------|----------|-------|
| **Track A: Algebra Core** | TBD | Local (CPU) | E1, E7 — confidence algebra experiments |
| **Track B: Interop & Format** | TBD | Local (CPU) | E2, E8.1, E8.6, E8.7, E8.8 — expressiveness, validation, interop |
| **Track C: ML Pipeline** | TBD | Colab (GPU) | E3, E8.4, E8.9 — RAG, TruthfulQA, vectors, MCP |
| **Track D: IoT & Transport** | TBD | Local (CPU) | E8.2, E8.3, E8.5 — SSN/SOSA, temporal, CBOR/MQTT |
| **Track E: Perf & Security** | TBD | Local (CPU) | E4, E5, E6 — scalability, security, usability |
| **Track W: Paper Writing** | Muntaser + all | — | LaTeX drafting, figures, supplementary |

### Phase 1: Infrastructure (Weeks 1-2) — ALL TRACKS

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 1 | Re-run all existing benchmarks on v0.3.5 | Track E | Updated `benchmark_results_latest.json` |
| 1 | Implement remaining medium-priority gaps if needed | Track A | Code + tests |
| 1 | Set up reproducibility infra: pinned deps, random seeds, env logging script | All | `requirements-bench.txt`, seed config |
| 1 | Create shared experiment template (notebook + results format) | Muntaser | `notebooks/template.ipynb` |
| 2 | Create Colab notebook templates for GPU experiments | Track C | `notebooks/` directory |
| 2 | Pin dependency versions, create requirements-bench.txt | All | Lock file |
| 2 | Set up shared results repository/folder structure | All | `experiments/` directory |

### Phase 2: Core Experiments (Weeks 3-8) — PARALLEL TRACKS

**Track A: Algebra Core (E1 + E7)**

| Week | Experiment | Deliverable |
|------|-----------|-------------|
| 3 | E7.1 Scalar collapse demo | Notebook + figure |
| 3 | E7.3 Composability / associativity | Notebook + table |
| 3 | E1.4 Trust discount chain viz | Figure + data |
| 4 | E7.2 Calibration comparison (SL vs scalar ensembles) | Notebook + reliability diagrams |
| 4 | E7.4 Evidence accumulation dynamics | Notebook + convergence plots |
| 5 | E7.5 Cumulative vs averaging under correlation | Notebook + results |
| 5 | E7.6 Decay function comparison under drift | Notebook + adaptation curves |
| 6 | E1.1 NER fusion — synthetic first, then CoNLL-2003 | Notebook + F1 tables |
| 7 | E1.2 Temporal regime adaptation | Notebook + KL plots |
| 7 | E1.3 Byzantine-robust fusion | Notebook + detection curves |
| 8 | E1.5 Deduction under uncertainty | Notebook + ECE/Brier |
| 8 | E7.7 Sensitivity analysis (sweeps on E1.1 + E1.3) | Heatmaps + robustness table |

**Track B: Interop & Format (E2 + E8 subset)**

| Week | Experiment | Deliverable |
|------|-----------|-------------|
| 3 | E2.2 Feature coverage matrix (all formats) | Heatmap table |
| 3 | E8.7 Context versioning and migration | Notebook + results |
| 4 | E2.1 Verbosity comparison (10 scenarios, 6 formats) | Table + byte ratios |
| 4 | E8.8 Provenance chain full lifecycle | Notebook + PROV-O/RDF-Star output |
| 5 | E2.3 Round-trip fidelity (all 6 interop formats) | Table + fidelity % |
| 5 | E8.1 Validation SHACL replacement study (10 scenarios) | Notebook + LoC/throughput tables |
| 6 | E8.6 Graph merge and diff operations | Notebook + accuracy tables |
| 7 | Integrate E2 results into paper tables | Publication-ready tables |

**Track C: ML Pipeline (E3 + E8 subset) — GPU**

| Week | Experiment | Deliverable |
|------|-----------|-------------|
| 3 | E8.4 Vector embeddings — product catalog + hybrid query | Colab notebook |
| 4 | E3.1 RAG pipeline — synthetic poisoned corpus | Colab notebook |
| 5 | E3.1 RAG pipeline — Natural Questions / TrivialQA | Colab notebook + accuracy tables |
| 6 | E3.2 TruthfulQA evaluation | Colab notebook + MC1/MC2 scores |
| 7 | E3.3 Croissant enrichment pipeline (5 real cards) | Notebook + before/after comparison |
| 8 | E8.9 MCP server agent demo (5 workflows) | Notebook + latency measurements |

**Track D: IoT & Transport (E8 subset)**

| Week | Experiment | Deliverable |
|------|-----------|-------------|
| 3 | E8.5 CBOR-LD compact transport (3 complexity levels) | Notebook + compression tables |
| 4 | E8.2 IoT sensor pipeline (50 sensors, 24h sim) | Notebook + SSN/SOSA round-trip |
| 5 | E8.3 Temporal modeling — 3-year org KG | Notebook + query results |
| 6 | E8.5 MQTT integration demo | Notebook + QoS mapping results |
| 7 | Integrate IoT results into paper | Publication-ready tables |

**Track E: Performance & Security**

| Week | Experiment | Deliverable |
|------|-----------|-------------|
| 3 | E4.1 Throughput benchmarks refresh (v0.3.5) | Updated JSON results |
| 3 | E4.2 Baseline comparisons refresh | Updated JSON results |
| 4 | E4.3 Scaling study (1K → 1M nodes) | Scaling curves + memory profiles |
| 4 | E4.4 Batch API overhead | Speedup table |
| 5 | E5.1 Context integrity verification | Results + detection rates |
| 5 | E5.2 Context allowlist enforcement | Results + edge cases |
| 6 | E5.3 Adversarial validation | Results + detection rates |
| 6 | E6.1 Lines-of-code comparison (5 tasks × 3 approaches) | Comparison table |

### Phase 3: Paper Writing (Weeks 9-11) — TRACK W + ALL

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 9 | Paper skeleton: all sections outlined | Muntaser | LaTeX structure |
| 9 | Draft: Introduction + Related Work | Track W | LaTeX sections |
| 9 | Collect all experiment results into shared folder | All tracks | `experiments/results/` |
| 10 | Draft: System Design (2 pages) | Track W | LaTeX section |
| 10 | Draft: Experiments (3-4 pages, integrate all results) | Track W | LaTeX section |
| 10 | Generate all figures and tables | All tracks | Publication-ready figures |
| 11 | Draft: Discussion + Conclusion | Track W | LaTeX sections |
| 11 | Complete first full draft | Track W | Full paper PDF |

### Phase 4: Polish & Submit (Weeks 12-13)

| Week | Task | Owner | Deliverable |
|------|------|-------|-------------|
| 12 | Co-author review and feedback round | All | Annotated draft |
| 12 | Prepare supplementary materials | All tracks | Supplementary PDF |
| 12 | Create Croissant metadata card for jsonld-ex | Track B | Croissant JSON |
| 12 | Prepare author statement, license verification | Muntaser | Supplementary PDF |
| 13 | Final revisions from co-author feedback | Track W | Revised paper |
| 13 | Format check, proofread, figure quality review | All | Camera-ready PDF |
| 13 | Submit to OpenReview | Muntaser | Submission confirmation |

### Critical Path

The minimum viable paper requires results from: **E1.1, E7.1, E7.2, E2.1, E2.2, E4.1**. Everything else strengthens the submission but is not blocking. Paper writing (Phase 3) can begin as soon as Track A delivers E7.1-E7.3 results (week 3-4).

---

## 13. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| NER models don't produce meaningful confidence spread | Medium | High (E1.1 fails) | Use softmax calibration; fall back to synthetic if needed |
| TruthfulQA experiment shows no improvement | Medium | Medium | Analyze failure modes honestly; this is science, not marketing |
| Scalability bottleneck at 1M nodes | Low | Medium | Profile early (week 1), optimize if needed |
| Co-author review delays | Medium | High | Share early drafts weekly from week 10 |
| NeurIPS 2026 deadline earlier than expected | Low | Critical | Track announcements; have paper skeleton ready by week 8 |
| GPU compute insufficient for RAG experiments | Low | Medium | Use Google Colab Pro; scope down LLM to small model |
| Reviewer asks for comparison we didn't include | High | Medium | Prepare extended comparison table in supplementary |
| Track coordination overhead / merge conflicts | Medium | Medium | Weekly sync meetings, shared results format, clear ownership |
| Team member availability gaps | Medium | High | Critical path has single-track fallback; MVP needs only Track A + E |
| Inconsistent experiment methodology across tracks | Low | High | Shared notebook template, reproducibility checklist, peer review of notebooks |

---

## 14. Paper Outline (Preliminary)

**Title:** jsonld-ex: Confidence-Aware JSON-LD Extensions for ML Data Exchange

### Abstract (~250 words)
- Problem: ML pipelines exchange data without formal uncertainty representation
- Solution: jsonld-ex extends JSON-LD with Subjective Logic confidence algebra
- Results: Improved fusion accuracy, conflict detection, provenance tracking
- Impact: Open-source, backward-compatible, W3C-interoperable

### 1. Introduction (1 page)
- Data exchange in ML is ad-hoc (CSV, JSON, custom formats)
- Structured data formats exist but lack confidence metadata
- jsonld-ex bridges the gap with assertion-level uncertainty

### 2. Related Work (1 page)
- JSON-LD 1.1 and the W3C ecosystem
- Croissant and ML dataset documentation
- Subjective Logic (Jøsang 2016)
- Uncertainty in ML: calibration, conformal prediction, Bayesian approaches

### 3. System Design (2 pages)
- Opinion model and confidence algebra
- Annotation API and processing pipeline
- Validation framework (SHACL-inspired shapes)
- Interoperability layer (PROV-O, SHACL, OWL, RDF-Star, SSN/SOSA, Croissant)

### 4. Experiments (4-5 pages)
- E7 + E1: Confidence algebra superiority + knowledge fusion (core results, ~2 pages)
- E2: Expressiveness comparison (tables, ~0.5 page)
- E3: End-to-end ML pipeline integration (~0.5 page)
- E8: Feature showcase highlights — validation, IoT, temporal, vectors, transport (~0.5 page)
- E4: Scalability (condensed table, ~0.25 page)
- E5: Security (brief, ~0.25 page)

### 5. Discussion (0.5 page)
- Limitations
- When NOT to use jsonld-ex
- Future work (W3C standardization path)

### 6. Conclusion (0.5 page)

### Supplementary Materials
- Full benchmark tables
- All experimental code and notebooks
- Croissant metadata for jsonld-ex
- API reference
- Extended feature comparison matrix

**Total: ~9 pages** (NeurIPS format allows 9 pages main + unlimited appendix)

**Supplementary (unlimited):** Full E7 parameter sweeps, all E8 experiment details, complete benchmark tables, all notebooks, Croissant card, API reference

**Relationship to FLAIRS-39:** FLAIRS paper (submitted) focuses on formal foundations of the confidence algebra and W3C specification design. NeurIPS paper focuses on the tool as a whole, empirical evaluation across 36 experiments, and practical ML pipeline integration. Minimal overlap in experimental content.

---

## 15. Immediate Next Steps

1. **Assign team members to tracks** — Who owns A/B/C/D/E?
2. **Re-run existing benchmarks** on v0.3.5 to establish current baseline (Track E, Week 1)
3. **Begin E7.1 + E7.3** (scalar collapse + composability) as the fastest-to-results algebra experiments (Track A, Week 3)
4. **Set up reproducibility infrastructure** — pinned deps, seeds, env logging, shared results format (All, Week 1)
5. **Create Colab templates** for GPU experiments (Track C, Week 2)
6. **FLAIRS paper is submitted** — NeurIPS submission should expand scope significantly beyond FLAIRS (more experiments, broader evaluation, tool + benchmark framing vs. FLAIRS’ theoretical focus)

---

## Appendix A: Experiment Count Summary

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
| **Total** | **36** | | |

## Appendix B: Hardware & Compute Requirements

| Experiment | GPU? | Est. Time | Platform | Track |
|-----------|------|-----------|----------|-------|
| E1.1 NER fusion | Optional | 2-4 hrs | Local | A |
| E1.2 Temporal regime | No | < 1 hr | Local | A |
| E1.3 Byzantine | No | < 1 hr | Local | A |
| E1.4 Trust chain | No | < 10 min | Local | A |
| E1.5 Deduction | No | < 30 min | Local | A |
| E7.1 Scalar collapse | No | < 30 min | Local | A |
| E7.2 Calibration | No | 1-2 hrs | Local | A |
| E7.3 Composability | No | < 1 hr | Local | A |
| E7.4 Evidence accumulation | No | < 1 hr | Local | A |
| E7.5 Cumul vs avg fusion | No | < 1 hr | Local | A |
| E7.6 Decay comparison | No | 1-2 hrs | Local | A |
| E7.7 Sensitivity sweeps | No | 2-4 hrs | Local | A |
| E2.1 Verbosity | No | < 2 hrs | Local | B |
| E2.2 Feature matrix | No | Manual | Local | B |
| E2.3 Round-trip fidelity | No | < 2 hrs | Local | B |
| E3.1 RAG pipeline | Yes | 4-8 hrs | Colab | C |
| E3.2 TruthfulQA | Yes | 4-8 hrs | Colab | C |
| E3.3 Croissant | No | < 1 hr | Local | C |
| E8.1 Validation/SHACL | No | 2-3 hrs | Local | B |
| E8.2 IoT sensor pipeline | No | 1-2 hrs | Local | D |
| E8.3 Temporal modeling | No | 1-2 hrs | Local | D |
| E8.4 Vector embeddings | Yes | 2-4 hrs | Colab | C |
| E8.5 CBOR-LD transport | No | 1-2 hrs | Local | D |
| E8.6 Graph merge/diff | No | 1-2 hrs | Local | B |
| E8.7 Context versioning | No | < 1 hr | Local | B |
| E8.8 Provenance lifecycle | No | 1-2 hrs | Local | B |
| E8.9 MCP server demo | No | 1-2 hrs | Local | C |
| E4.1-4.4 Scalability | No | 2-4 hrs | Local | E |
| E5.1-5.3 Security | No | < 1 hr | Local | E |
| E6.1-6.2 Usability | No | Manual | Local | E |
