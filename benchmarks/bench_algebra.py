"""
Benchmark: Confidence Algebra (Subjective Logic)

Measures throughput, scaling, and correctness properties of the formal
confidence algebra — the paper's most novel contribution.

Sections:
  A1: Fusion throughput (cumulative and averaging, varying n)
  A2: Trust discount chain throughput
  A3: Deduction throughput (single and chained)
  A4: Temporal decay throughput
  A5: Opinion formation and serialization throughput
  A6: Scalar ↔ algebra equivalence verification at scale
  A7: Calibration analysis — fused opinions vs ground truth

Statistical method: 30 trials, 3 warmup, 95% CI via t-distribution.
Micro-benchmarks use inner iteration batching for stable μs-level timing.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce,
)
from jsonld_ex.confidence_bridge import (
    combine_opinions_from_scalars,
    propagate_opinions_from_scalars,
)
from jsonld_ex.confidence_decay import (
    decay_opinion,
    exponential_decay,
    linear_decay,
    step_decay,
)
from jsonld_ex.inference import propagate_confidence, combine_sources

from bench_utils import timed_trials, timed_trials_us, DEFAULT_TRIALS


_RNG = random.Random(42)


# ═══════════════════════════════════════════════════════════════════
# Helpers — opinion generators
# ═══════════════════════════════════════════════════════════════════


def _random_opinion(rng: random.Random = _RNG) -> Opinion:
    """Generate a random valid opinion (b + d + u = 1)."""
    # Dirichlet-like: generate 3 uniform, normalize
    raw = [rng.random() for _ in range(3)]
    total = sum(raw)
    b, d, u = raw[0] / total, raw[1] / total, raw[2] / total
    a = rng.uniform(0.1, 0.9)
    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


def _random_confidence(rng: random.Random = _RNG) -> float:
    return round(rng.uniform(0.1, 0.99), 4)


def _make_opinion_batch(n: int) -> list[Opinion]:
    """Generate n random opinions with fixed seed for reproducibility."""
    rng = random.Random(42)
    return [_random_opinion(rng) for _ in range(n)]


def _make_dogmatic_batch(n: int) -> list[Opinion]:
    """Generate n dogmatic opinions (u=0) from scalar confidence."""
    rng = random.Random(42)
    return [
        Opinion.from_confidence(rng.uniform(0.1, 0.99))
        for _ in range(n)
    ]


# ═══════════════════════════════════════════════════════════════════
# A1: Fusion Throughput
# ═══════════════════════════════════════════════════════════════════


def bench_cumulative_fusion(
    sizes: list[int] = [2, 3, 5, 10, 20, 50, 100],
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure cumulative_fuse throughput at varying opinion counts."""
    results = {}
    for n in sizes:
        opinions = _make_opinion_batch(n)

        stats = timed_trials_us(
            lambda: cumulative_fuse(*opinions),
            inner_iterations=inner,
            n=n_trials,
        )

        results[f"n={n}"] = {
            "n_opinions": n,
            "mean_us": round(stats.mean * 1e6, 3),
            "std_us": round(stats.std * 1e6, 3),
            "ops_per_sec": round(1.0 / stats.mean, 0) if stats.mean > 0 else 0,
            **stats.to_dict(),
        }
    return results


def bench_averaging_fusion(
    sizes: list[int] = [2, 3, 5, 10, 20, 50, 100],
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure averaging_fuse throughput (n-ary simultaneous formula)."""
    results = {}
    for n in sizes:
        opinions = _make_opinion_batch(n)

        stats = timed_trials_us(
            lambda: averaging_fuse(*opinions),
            inner_iterations=inner,
            n=n_trials,
        )

        results[f"n={n}"] = {
            "n_opinions": n,
            "mean_us": round(stats.mean * 1e6, 3),
            "std_us": round(stats.std * 1e6, 3),
            "ops_per_sec": round(1.0 / stats.mean, 0) if stats.mean > 0 else 0,
            **stats.to_dict(),
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# A2: Trust Discount Chain Throughput
# ═══════════════════════════════════════════════════════════════════


def bench_trust_discount_chain(
    chain_lengths: list[int] = [2, 5, 10, 20, 50],
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure iterated trust discount through chains of varying length."""
    results = {}
    for length in chain_lengths:
        rng = random.Random(42)
        trust_opinions = [_random_opinion(rng) for _ in range(length)]
        # The final assertion (fully believed)
        assertion = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0, base_rate=0.5)

        def do_chain():
            current = assertion
            for t in reversed(trust_opinions):
                current = trust_discount(t, current)
            return current

        stats = timed_trials_us(do_chain, inner_iterations=inner, n=n_trials)

        results[f"len={length}"] = {
            "chain_length": length,
            "mean_us": round(stats.mean * 1e6, 3),
            "std_us": round(stats.std * 1e6, 3),
            "us_per_hop": round(stats.mean * 1e6 / length, 3),
            **stats.to_dict(),
        }
    return results


def bench_trust_vs_scalar_equivalence(
    chain_lengths: list[int] = [2, 5, 10, 20, 50, 100],
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Verify scalar multiply ≡ trust discount at scale.

    Both produce the same scalar result; we measure:
    1. That results are numerically identical (within float tolerance)
    2. Throughput difference (algebra overhead for richer metadata)
    """
    results = {}
    for length in chain_lengths:
        rng = random.Random(42)
        scores = [round(rng.uniform(0.3, 0.99), 4) for _ in range(length)]

        # Scalar multiply
        stats_scalar = timed_trials_us(
            lambda: propagate_confidence(scores, method="multiply"),
            inner_iterations=inner,
            n=n_trials,
        )

        # Algebraic trust discount (with base_rate=0 for exact equivalence)
        stats_algebra = timed_trials_us(
            lambda: propagate_opinions_from_scalars(scores, base_rate=0.0),
            inner_iterations=inner,
            n=n_trials,
        )

        # Verify equivalence
        scalar_result = propagate_confidence(scores, method="multiply").score
        algebra_result = propagate_opinions_from_scalars(scores, base_rate=0.0).to_confidence()
        equivalent = abs(scalar_result - algebra_result) < 1e-12

        overhead = (stats_algebra.mean / stats_scalar.mean - 1.0) * 100 if stats_scalar.mean > 0 else 0

        results[f"len={length}"] = {
            "chain_length": length,
            "scalar_us": round(stats_scalar.mean * 1e6, 3),
            "algebra_us": round(stats_algebra.mean * 1e6, 3),
            "overhead_pct": round(overhead, 1),
            "scalar_result": round(scalar_result, 12),
            "algebra_result": round(algebra_result, 12),
            "numerically_equivalent": equivalent,
        }
    return results


# ═══════════════════════════════════════════════════════════════════
# A3: Deduction Throughput
# ═══════════════════════════════════════════════════════════════════


def bench_deduction(
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure single deduction and chained deduction throughput."""
    rng = random.Random(42)
    results = {}

    # Single deduction
    omega_x = _random_opinion(rng)
    omega_y_given_x = _random_opinion(rng)
    omega_y_given_not_x = _random_opinion(rng)

    stats_single = timed_trials_us(
        lambda: deduce(omega_x, omega_y_given_x, omega_y_given_not_x),
        inner_iterations=inner,
        n=n_trials,
    )

    results["single"] = {
        "mean_us": round(stats_single.mean * 1e6, 3),
        "std_us": round(stats_single.std * 1e6, 3),
        "ops_per_sec": round(1.0 / stats_single.mean, 0) if stats_single.mean > 0 else 0,
        **stats_single.to_dict(),
    }

    # Chained deduction: output of one feeds as input to next
    for chain_len in [2, 5, 10]:
        # Pre-generate conditional opinions
        conditionals = [
            (_random_opinion(rng), _random_opinion(rng))
            for _ in range(chain_len)
        ]
        start_opinion = _random_opinion(rng)

        def do_chain_deduce():
            current = start_opinion
            for y_given_x, y_given_not_x in conditionals:
                current = deduce(current, y_given_x, y_given_not_x)
            return current

        stats_chain = timed_trials_us(do_chain_deduce, inner_iterations=inner, n=n_trials)

        results[f"chain={chain_len}"] = {
            "chain_length": chain_len,
            "mean_us": round(stats_chain.mean * 1e6, 3),
            "std_us": round(stats_chain.std * 1e6, 3),
            "us_per_stage": round(stats_chain.mean * 1e6 / chain_len, 3),
            **stats_chain.to_dict(),
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# A4: Temporal Decay Throughput
# ═══════════════════════════════════════════════════════════════════


def bench_temporal_decay(
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure decay function throughput for all three methods."""
    opinion = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
    results = {}

    for decay_fn, name in [
        (exponential_decay, "exponential"),
        (linear_decay, "linear"),
        (step_decay, "step"),
    ]:
        for elapsed in [0.5, 1.0, 5.0, 10.0]:
            # Use default args to capture loop variables by value
            if name == "step":
                def fn(_op=opinion, _t=elapsed, _dfn=decay_fn):
                    return decay_opinion(_op, _t, half_life=2.0, decay_fn=_dfn)
            else:
                def fn(_op=opinion, _t=elapsed, _dfn=decay_fn):
                    return decay_opinion(_op, _t, half_life=2.0, decay_fn=_dfn)

            stats = timed_trials_us(fn, inner_iterations=inner, n=n_trials)

            results[f"{name}_t={elapsed}"] = {
                "method": name,
                "elapsed": elapsed,
                "mean_us": round(stats.mean * 1e6, 3),
                "std_us": round(stats.std * 1e6, 3),
                **stats.to_dict(),
            }

    return results


# ═══════════════════════════════════════════════════════════════════
# A5: Opinion Formation & Serialization Throughput
# ═══════════════════════════════════════════════════════════════════


def bench_opinion_formation(
    n_trials: int = DEFAULT_TRIALS,
    inner: int = 1000,
) -> dict[str, Any]:
    """Measure Opinion construction, conversion, and serialization."""
    rng = random.Random(42)
    results = {}

    # from_confidence
    conf = rng.uniform(0.1, 0.99)
    stats = timed_trials_us(
        lambda: Opinion.from_confidence(conf, uncertainty=0.1),
        inner_iterations=inner,
        n=n_trials,
    )
    results["from_confidence"] = {
        "mean_us": round(stats.mean * 1e6, 3),
        "std_us": round(stats.std * 1e6, 3),
    }

    # from_evidence
    stats = timed_trials_us(
        lambda: Opinion.from_evidence(positive=15, negative=3),
        inner_iterations=inner,
        n=n_trials,
    )
    results["from_evidence"] = {
        "mean_us": round(stats.mean * 1e6, 3),
        "std_us": round(stats.std * 1e6, 3),
    }

    # to_jsonld
    opinion = _random_opinion(rng)
    stats = timed_trials_us(
        lambda: opinion.to_jsonld(),
        inner_iterations=inner,
        n=n_trials,
    )
    results["to_jsonld"] = {
        "mean_us": round(stats.mean * 1e6, 3),
        "std_us": round(stats.std * 1e6, 3),
    }

    # from_jsonld
    jld = opinion.to_jsonld()
    stats = timed_trials_us(
        lambda: Opinion.from_jsonld(jld),
        inner_iterations=inner,
        n=n_trials,
    )
    results["from_jsonld"] = {
        "mean_us": round(stats.mean * 1e6, 3),
        "std_us": round(stats.std * 1e6, 3),
    }

    # projected_probability
    stats = timed_trials_us(
        lambda: opinion.projected_probability(),
        inner_iterations=inner,
        n=n_trials,
    )
    results["projected_probability"] = {
        "mean_us": round(stats.mean * 1e6, 3),
        "std_us": round(stats.std * 1e6, 3),
    }

    return results


# ═══════════════════════════════════════════════════════════════════
# A6: Scalar ↔ Algebra Information Comparison
# ═══════════════════════════════════════════════════════════════════


def bench_information_richness(
    n_trials: int = DEFAULT_TRIALS,
) -> dict[str, Any]:
    """Demonstrate cases where scalar conflates but algebra distinguishes.

    Three pairs of opinions that map to the same projected probability
    but carry different uncertainty information:
      - "Strong evidence for 0.75" vs "No evidence, base rate 0.75"
      - "Highly certain 0.50" vs "Total ignorance about 0.50"
      - "Mixed evidence for 0.80" vs "Mild evidence for 0.80"
    """
    cases = [
        {
            "label": "strong_evidence_vs_no_evidence",
            "description": "Both P=0.75 but very different certainty",
            "opinion_a": Opinion(belief=0.75, disbelief=0.25, uncertainty=0.0, base_rate=0.5),
            "opinion_b": Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.75),
        },
        {
            "label": "certain_vs_ignorant_at_050",
            "description": "Both P=0.50 but maximal vs minimal certainty",
            "opinion_a": Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0, base_rate=0.5),
            "opinion_b": Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5),
        },
        {
            "label": "mixed_vs_mild_evidence_at_080",
            "description": "Both P=0.80 but different evidence profiles",
            "opinion_a": Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5),
            "opinion_b": Opinion(belief=0.3, disbelief=0.0, uncertainty=0.7, base_rate=5.0 / 7.0),
        },
    ]

    results = {}
    for case in cases:
        a = case["opinion_a"]
        b = case["opinion_b"]
        results[case["label"]] = {
            "description": case["description"],
            "opinion_a": {"b": a.belief, "d": a.disbelief, "u": a.uncertainty, "a": a.base_rate},
            "opinion_b": {"b": b.belief, "d": b.disbelief, "u": b.uncertainty, "a": b.base_rate},
            "P_a": round(a.projected_probability(), 6),
            "P_b": round(b.projected_probability(), 6),
            "same_scalar": abs(a.projected_probability() - b.projected_probability()) < 1e-9,
            "uncertainty_a": a.uncertainty,
            "uncertainty_b": b.uncertainty,
        }

    return results


# ═══════════════════════════════════════════════════════════════════
# A7: Calibration Analysis
# ═══════════════════════════════════════════════════════════════════


def bench_calibration(
    n_propositions: int = 1000,
    n_sources_per: int = 5,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Measure calibration of fused opinions vs simulated ground truth.

    For each proposition:
    1. Assign a ground truth probability p_true ~ Uniform(0.05, 0.95)
    2. Generate n_sources independent opinions:
       - Each source observes evidence: positive ~ Binomial(20, p_true)
       - Convert to Opinion via from_evidence
    3. Fuse via cumulative_fuse (independent sources)
    4. Check if fused projected_probability is calibrated:
       - Bin by predicted probability
       - Compare mean prediction in each bin to actual fraction of true propositions

    This is a reliability diagram in tabular form.
    """
    rng = random.Random(42)

    predictions = []  # (fused_P, ground_truth_outcome)

    for _ in range(n_propositions):
        p_true = rng.uniform(0.05, 0.95)
        # Ground truth: is the proposition true?
        outcome = 1 if rng.random() < p_true else 0

        # Generate source opinions from evidence
        opinions = []
        for _ in range(n_sources_per):
            n_obs = 20
            pos = sum(1 for _ in range(n_obs) if rng.random() < p_true)
            neg = n_obs - pos
            opinions.append(Opinion.from_evidence(positive=pos, negative=neg))

        # Fuse
        fused = cumulative_fuse(*opinions)
        predictions.append((fused.projected_probability(), outcome, fused.uncertainty))

    # Bin into reliability diagram
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [(p, o, u) for p, o, u in predictions if lo <= p < hi]
        if i == n_bins - 1:
            # Include right edge for last bin
            in_bin = [(p, o, u) for p, o, u in predictions if lo <= p <= hi]

        if len(in_bin) > 0:
            mean_pred = sum(p for p, _, _ in in_bin) / len(in_bin)
            mean_true = sum(o for _, o, _ in in_bin) / len(in_bin)
            mean_uncertainty = sum(u for _, _, u in in_bin) / len(in_bin)
        else:
            mean_pred = (lo + hi) / 2
            mean_true = 0.0
            mean_uncertainty = 0.0

        bins.append({
            "bin": f"[{lo:.1f}, {hi:.1f})",
            "count": len(in_bin),
            "mean_predicted": round(mean_pred, 4),
            "mean_actual": round(mean_true, 4),
            "calibration_error": round(abs(mean_pred - mean_true), 4),
            "mean_uncertainty": round(mean_uncertainty, 4),
        })

    # Overall metrics
    all_pred = [p for p, _, _ in predictions]
    all_true = [o for _, o, _ in predictions]
    all_unc = [u for _, _, u in predictions]

    # Expected Calibration Error (ECE)
    ece = sum(
        b["count"] / n_propositions * b["calibration_error"]
        for b in bins if b["count"] > 0
    )

    # Brier score
    brier = sum((p - o) ** 2 for p, o in zip(all_pred, all_true)) / n_propositions

    return {
        "n_propositions": n_propositions,
        "n_sources_per_proposition": n_sources_per,
        "fusion_method": "cumulative",
        "bins": bins,
        "expected_calibration_error": round(ece, 4),
        "brier_score": round(brier, 4),
        "mean_uncertainty": round(sum(all_unc) / len(all_unc), 4),
    }


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AlgebraResults:
    cumulative_fusion: dict[str, Any] = field(default_factory=dict)
    averaging_fusion: dict[str, Any] = field(default_factory=dict)
    trust_discount_chain: dict[str, Any] = field(default_factory=dict)
    trust_vs_scalar: dict[str, Any] = field(default_factory=dict)
    deduction: dict[str, Any] = field(default_factory=dict)
    temporal_decay: dict[str, Any] = field(default_factory=dict)
    opinion_formation: dict[str, Any] = field(default_factory=dict)
    information_richness: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)


def run_all() -> AlgebraResults:
    results = AlgebraResults()
    print("=== Confidence Algebra Benchmarks ===\n")

    print("A1a  Cumulative fusion throughput...")
    results.cumulative_fusion = bench_cumulative_fusion()

    print("A1b  Averaging fusion throughput (n-ary)...")
    results.averaging_fusion = bench_averaging_fusion()

    print("A2a  Trust discount chain throughput...")
    results.trust_discount_chain = bench_trust_discount_chain()

    print("A2b  Trust discount vs scalar multiply equivalence...")
    results.trust_vs_scalar = bench_trust_vs_scalar_equivalence()

    print("A3   Deduction throughput (single + chained)...")
    results.deduction = bench_deduction()

    print("A4   Temporal decay throughput...")
    results.temporal_decay = bench_temporal_decay()

    print("A5   Opinion formation & serialization...")
    results.opinion_formation = bench_opinion_formation()

    print("A6   Information richness (scalar vs algebra)...")
    results.information_richness = bench_information_richness()

    print("A7   Calibration analysis...")
    results.calibration = bench_calibration()

    return results


if __name__ == "__main__":
    r = run_all()

    print("\n" + "=" * 60)
    print("CONFIDENCE ALGEBRA RESULTS")
    print("=" * 60)

    print("\n--- Cumulative Fusion (μs per fusion) ---")
    for k, v in r.cumulative_fusion.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs "
              f"({v['ops_per_sec']:,.0f} ops/sec)")

    print("\n--- Averaging Fusion (μs per fusion) ---")
    for k, v in r.averaging_fusion.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs "
              f"({v['ops_per_sec']:,.0f} ops/sec)")

    print("\n--- Trust Discount Chain (μs per chain) ---")
    for k, v in r.trust_discount_chain.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs "
              f"({v['us_per_hop']:.2f} μs/hop)")

    print("\n--- Trust vs Scalar Equivalence ---")
    for k, v in r.trust_vs_scalar.items():
        eq = "✓" if v['numerically_equivalent'] else "✗"
        print(f"  {k}: scalar {v['scalar_us']:.2f}μs vs algebra {v['algebra_us']:.2f}μs "
              f"(+{v['overhead_pct']:.1f}% overhead) equivalent={eq}")

    print("\n--- Deduction ---")
    for k, v in r.deduction.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs")

    print("\n--- Temporal Decay ---")
    for k, v in r.temporal_decay.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs")

    print("\n--- Opinion Formation ---")
    for k, v in r.opinion_formation.items():
        print(f"  {k}: {v['mean_us']:.2f} ± {v['std_us']:.2f} μs")

    print("\n--- Information Richness ---")
    for k, v in r.information_richness.items():
        print(f"  {k}: P_a={v['P_a']}, P_b={v['P_b']}, "
              f"same_scalar={v['same_scalar']}, "
              f"u_a={v['uncertainty_a']:.2f}, u_b={v['uncertainty_b']:.2f}")

    print("\n--- Calibration ---")
    cal = r.calibration
    print(f"  ECE: {cal['expected_calibration_error']:.4f}")
    print(f"  Brier score: {cal['brier_score']:.4f}")
    print(f"  Mean uncertainty: {cal['mean_uncertainty']:.4f}")
    print("  Reliability bins:")
    for b in cal['bins']:
        if b['count'] > 0:
            print(f"    {b['bin']}: n={b['count']}, "
                  f"pred={b['mean_predicted']:.3f}, "
                  f"actual={b['mean_actual']:.3f}, "
                  f"err={b['calibration_error']:.3f}, "
                  f"unc={b['mean_uncertainty']:.3f}")
