# Experiment Collation: Notebook vs Existing jsonld-ex Work

**Date:** February 10, 2026
**Source:** jsonldexexperiments.ipynb (13 cells)

---

## Summary

The notebook contains **8 distinct experiments** (some with multiple iterations/fixes).
Each experiment uses a standalone re-implementation of Subjective Logic operators
rather than the jsonld-ex library. Below is the mapping to our existing work,
assessment of novelty, and recommendations.

---

## Experiment Inventory

### Exp 1: Conflict Detection — "The False Agreement"
- **Cell:** 1
- **Thesis:** Scalar averaging conflates ignorance with conflict; SL separates them
- **Scenario:** 5 agents (2 strong YES, 2 strong NO, 1 unsure) → scalar avg = 0.50 (ambiguous), SL shows high belief AND high disbelief (explicit conflict)
- **Operators used:** cumulative_fuse (standalone reimplementation)
- **Proposes:** Conflict metric = 1 - |b-d| - u

**jsonld-ex coverage:** ✅ cumulative_fuse exists. ❌ No explicit conflict metric API.
**Benchmark coverage:** Partially — bench_merge.py tracks agreed/conflicted counts but doesn't compute a formal conflict metric.
**Novelty for paper:** MEDIUM — the "same scalar, different epistemic state" argument is already in bench_algebra.py's Information Richness section. But the conflict metric formulation is new and publishable.

---

### Exp 2: Temporal Decay vs Bayesian Inertia
- **Cell:** 2
- **Thesis:** SL + temporal decay adapts to regime changes faster than recursive Bayesian updates
- **Scenario:** 100 timesteps of positive evidence, then 20 timesteps of negative. Bayesian prior gets stuck near 1.0; SL with decay factor 0.8 reverts quickly.
- **Operators used:** cumulative_fuse_pair, decay_opinion (standalone)

**jsonld-ex coverage:** ✅ confidence_decay.py (exponential, linear, step). ✅ cumulative_fuse.
**Benchmark coverage:** ❌ No temporal adaptation benchmark exists. bench_algebra.py only measures throughput.
**Novelty for paper:** HIGH — demonstrates a practical advantage of decay that our throughput benchmarks don't capture. This is a strong candidate for a paper figure.

---

### Exp 3: Broken Telephone Trust Chain
- **Cell:** 0
- **Thesis:** Scalar multiply decays to 0 in long trust chains (false disbelief); SL trust discount reverts to base rate (correct "unknown")
- **Scenario:** Chain lengths 1–20, trust=0.85, base_rate=0.5. Scalar → 0.0, SL → 0.5.
- **Operators used:** trust_discount (standalone)

**jsonld-ex coverage:** ✅ trust_discount in confidence_algebra.py. ✅ bench_algebra.py has trust chain throughput + scalar equivalence proof.
**Benchmark coverage:** ✅ Already covered — bench_algebra.py's "Trust Discount vs Scalar Multiply Equivalence" table and the Information Richness experiment both address this.
**Novelty for paper:** LOW — already well-covered. The visualization (chart) could be useful for the paper but the underlying argument is in our benchmarks.

---

### Exp 4a: Epistemic Risk Controller (Synthetic Market)
- **Cell:** 3
- **Thesis:** SL detects conflict between trend-follower and mean-reverter signals, exits to cash during confusion → protects capital during crashes
- **Scenario:** Synthetic bull→chop→crash market, 2 expert signals, conflict-aware position sizing
- **Operators used:** cumulative_fuse, from_signal (standalone)

**jsonld-ex coverage:** ❌ No financial application module.
**Benchmark coverage:** ❌ Not covered.
**Novelty for paper:** MEDIUM — compelling demonstration but uses standalone algebra, not jsonld-ex. Would need adaptation to use our library.

### Exp 4b: Real Data COVID Crash
- **Cell:** 4
- **Thesis:** Same as 4a but on real S&P 500 data (Feb-Apr 2020)
- **Operators used:** Same standalone algebra

**jsonld-ex coverage:** ❌
**Novelty for paper:** MEDIUM-HIGH for credibility (real data), but tangential to our core JSON-LD story.

### Exp 4c: Reusable Experiment Engine
- **Cell:** 5
- **Thesis:** Packaged version of Exp 4 as EpistemicRiskExperiment class
- **Status:** Utility code, not a separate experiment

---

### Exp 5: Semantic Drift in RAG — "The Logic Gate"
- **Cell:** 6
- **Thesis:** SL conflict detection filters out high-similarity distractors (hallucinations) in RAG pipelines
- **Scenario:** Simulated retrieval with 30-40% "poison" documents. Baseline accepts poisoned context; SL detects conflict and abstains.
- **Operators used:** cumulative_fuse, from_retrieval (standalone)

**jsonld-ex coverage:** ✅ merge.py + confidence filtering in bench_rag.py
**Benchmark coverage:** Partially — bench_rag.py measures merge+filter throughput but doesn't simulate poisoned retrieval or measure hallucination rates.
**Novelty for paper:** HIGH — this directly demonstrates a safety-critical application that our throughput benchmarks miss. The "abstention under conflict" behavior is a strong paper contribution.

---

### Exp 6: TruthfulQA Real-World Semantic Drift
- **Cell:** 7
- **Thesis:** Same as Exp 5 but on real TruthfulQA dataset with SentenceTransformers
- **Dependencies:** sentence-transformers, datasets, scikit-learn
- **Operators used:** Same standalone algebra

**jsonld-ex coverage:** ❌ No real-dataset evaluation.
**Benchmark coverage:** ❌ Not covered.
**Novelty for paper:** VERY HIGH — real-world benchmark on an established dataset. This is exactly what NeurIPS D&B reviewers want. Would need careful adaptation to use jsonld-ex operators.

---

### Exp 7: OOD Robustness (MNIST vs FashionMNIST)
- **Cells:** 8, 10 (fixed version uses "variance fusion")
- **Thesis:** SL ensemble fusion distinguishes in-distribution from out-of-distribution better than scalar ensembling
- **Metric:** AUROC for OOD detection
- **Note:** The "fixed" version (Cell 10) uses variance→uncertainty mapping, not standard cumulative fusion

**jsonld-ex coverage:** ❌ No deep learning integration. ❌ No variance_fuse operator.
**Benchmark coverage:** ❌
**Novelty for paper:** MEDIUM — interesting but the "variance fusion" is a heuristic not in Jøsang (2016), which could weaken scientific rigor claims. Would need careful framing.

---

### Exp 8: CIFAR-10 Corruption Robustness
- **Cell:** 9
- **Thesis:** SL maintains calibration (low ECE) as data corruption increases, while scalar confidence becomes overconfident
- **Metric:** Expected Calibration Error (ECE) at severity levels 0-4

**jsonld-ex coverage:** ❌ No deep learning integration.
**Benchmark coverage:** ❌ bench_algebra.py has ECE/Brier but only for the algebra itself, not for DL predictions.
**Novelty for paper:** MEDIUM — similar concern as Exp 7 re: variance fusion heuristic.

---

### Exp 9: Byzantine Robustness (Rogue Agent Filtering)
- **Cells:** 11, 12 (optimized with median conflict)
- **Thesis:** Robust fusion using conflict matrix + iterative removal survives 2/5 rogue agents
- **Operators used:** cumulative_fuse, pairwise_conflict, robust_fuse (standalone)
- **Key innovation:** Median conflict aggregation prevents colluding rogues

**jsonld-ex coverage:** ❌ No robust_fuse operator. ❌ No pairwise conflict metric. merge.py has conflict resolution but not Byzantine robustness.
**Benchmark coverage:** ❌
**Novelty for paper:** HIGH — Byzantine robustness is a strong selling point for multi-agent KG construction. However, the robust_fuse algorithm is not in our library.

---

## Collation Matrix

| Experiment | Core Thesis | In jsonld-ex? | In Benchmarks? | Paper Priority |
|-----------|------------|---------------|---------------|---------------|
| 1. Conflict Detection | Scalar conflates ignorance/conflict | Partial | Partial | MEDIUM |
| 2. Temporal Decay vs Bayesian | Decay enables regime adaptation | ✅ | ❌ | HIGH |
| 3. Broken Telephone | Trust discount preserves uncertainty | ✅ | ✅ | LOW (covered) |
| 4. Epistemic Risk (Finance) | Conflict → de-risk | ❌ | ❌ | MEDIUM |
| 5. RAG Semantic Drift | Conflict → abstain on poison | Partial | Partial | HIGH |
| 6. TruthfulQA | Real-world RAG safety | ❌ | ❌ | VERY HIGH |
| 7. OOD Detection | Variance → uncertainty | ❌ | ❌ | MEDIUM |
| 8. CIFAR Corruption | Calibration under shift | ❌ | ❌ | MEDIUM |
| 9. Byzantine Robustness | Median conflict filtering | ❌ | ❌ | HIGH |

---

## Recommendations for NeurIPS Paper

### Tier 1 — Integrate into paper (high impact, feasible)
1. **Exp 2 (Temporal Decay):** Adapt to use jsonld-ex operators. Shows practical advantage beyond throughput. Could be a paper figure.
2. **Exp 5 (RAG Semantic Drift):** Adapt simulation to use jsonld-ex merge + confidence filter. Shows safety-critical application.
3. **Exp 6 (TruthfulQA):** Real-world dataset evaluation. Highest impact for NeurIPS reviewers. Requires integration work.

### Tier 2 — Consider for paper (high impact, more work needed)
4. **Exp 9 (Byzantine Robustness):** Would need to implement robust_fuse and pairwise conflict in jsonld-ex first. Strong multi-agent story.
5. **Exp 1 (Conflict Detection):** Add formal conflict metric to jsonld-ex. Quick to implement.

### Tier 3 — Defer or appendix
6. **Exp 4 (Finance):** Compelling demo but tangential to JSON-LD/ML story.
7. **Exp 7/8 (DL experiments):** Variance fusion is a heuristic, not formal SL. Needs careful scientific framing.
8. **Exp 3 (Trust Chain):** Already covered by existing benchmarks.

---

## Key Gaps Identified

These experiments reveal features that exist as notebook prototypes but NOT in jsonld-ex:

1. **Conflict metric API** — quantify conflict from fused opinion (quick to add)
2. **Robust fusion** — iterative conflict-based agent removal (medium effort)
3. **Pairwise conflict** — Jøsang's conflict measure between two opinions (quick to add)
4. **RAG integration** — semantic drift detection via conflict (medium effort)
5. **Regime change adaptation demo** — temporal decay in streaming context (low effort, just a benchmark)

### Implementation effort estimates:
- Conflict metric + pairwise conflict: ~1 day (TDD)
- Robust fusion operator: ~2-3 days (TDD, needs careful design)
- RAG semantic drift benchmark: ~2 days
- TruthfulQA integration: ~3-4 days (external dependencies)
- Temporal adaptation benchmark: ~1 day

---

## Scientific Rigor Concerns

Several experiments use **non-standard operators** that need scrutiny:

1. **Variance fusion** (Exp 7, 8, 10): Maps ensemble variance to uncertainty. This is a reasonable heuristic but is NOT a Jøsang (2016) operator. If included in the paper, it must be clearly framed as a "bridge heuristic" rather than formal SL, or we need to cite prior work that justifies it.

2. **Conflict metric = 1 - |b-d| - u** (Exp 1): This is a novel formulation. Needs mathematical justification — does it correspond to any established measure in Jøsang's framework?

3. **from_signal / from_retrieval mappings** (Exp 4, 5): These are application-specific mappings from scalar scores to opinions. The mapping choice affects results. Must be justified and sensitivity-analyzed.

4. **Standalone reimplementations**: All experiments reimplement the algebra from scratch rather than using jsonld-ex. Before any go into the paper, they MUST be re-run using jsonld-ex operators to ensure numerical equivalence and to validate our library.
