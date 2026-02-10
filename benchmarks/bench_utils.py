"""
Shared benchmark utilities: timing, statistics, and reporting helpers.

Provides timed_trials() for running benchmarks with proper statistical
rigor (mean, stddev, 95% CI, min, max, n_trials).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable


# Default trial count — enough for stable stddev estimates
DEFAULT_TRIALS = 30

# Warmup iterations before timing (JIT / cache warming)
DEFAULT_WARMUP = 3


@dataclass
class TrialStats:
    """Statistical summary of timed trials."""

    mean: float  # seconds
    std: float  # seconds
    ci95_low: float  # seconds
    ci95_high: float  # seconds
    min: float  # seconds
    max: float  # seconds
    n: int  # trial count

    def mean_ms(self) -> float:
        return round(self.mean * 1000, 3)

    def std_ms(self) -> float:
        return round(self.std * 1000, 3)

    def ci95_ms(self) -> tuple[float, float]:
        return (round(self.ci95_low * 1000, 3), round(self.ci95_high * 1000, 3))

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_sec": round(self.mean, 8),
            "std_sec": round(self.std, 8),
            "ci95_low_sec": round(self.ci95_low, 8),
            "ci95_high_sec": round(self.ci95_high, 8),
            "min_sec": round(self.min, 8),
            "max_sec": round(self.max, 8),
            "n_trials": self.n,
        }


def timed_trials(
    fn: Callable[[], Any],
    n: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
) -> TrialStats:
    """Run *fn* n times after warmup, returning statistical summary.

    Uses t-distribution critical value for 95% CI (approximated for n>=30
    as 1.96; for smaller n uses a lookup table).
    """
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed trials
    times: list[float] = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)

    mean = sum(times) / n
    variance = sum((t - mean) ** 2 for t in times) / (n - 1) if n > 1 else 0.0
    std = math.sqrt(variance)

    # 95% CI using t-distribution critical value
    t_crit = _t_critical(n - 1)
    margin = t_crit * std / math.sqrt(n) if n > 1 else 0.0

    return TrialStats(
        mean=mean,
        std=std,
        ci95_low=mean - margin,
        ci95_high=mean + margin,
        min=min(times),
        max=max(times),
        n=n,
    )


def timed_trials_us(
    fn: Callable[[], Any],
    inner_iterations: int = 1000,
    n: int = DEFAULT_TRIALS,
    warmup: int = DEFAULT_WARMUP,
) -> TrialStats:
    """Like timed_trials but for micro-benchmarks.

    Runs *fn* inner_iterations times per trial, reports per-call stats.
    Returns stats in seconds-per-call (multiply by 1e6 for μs).
    """
    def batch():
        for _ in range(inner_iterations):
            fn()

    stats = timed_trials(batch, n=n, warmup=warmup)

    # Convert from total-batch to per-call
    return TrialStats(
        mean=stats.mean / inner_iterations,
        std=stats.std / inner_iterations,
        ci95_low=stats.ci95_low / inner_iterations,
        ci95_high=stats.ci95_high / inner_iterations,
        min=stats.min / inner_iterations,
        max=stats.max / inner_iterations,
        n=stats.n,
    )


def format_ci(mean: float, std: float, unit: str = "ms") -> str:
    """Format as 'mean ± std unit' for markdown tables."""
    if unit == "ms":
        return f"{mean * 1000:.2f} ± {std * 1000:.2f}"
    elif unit == "μs":
        return f"{mean * 1e6:.2f} ± {std * 1e6:.2f}"
    elif unit == "s":
        return f"{mean:.4f} ± {std:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


# ── t-distribution critical values (two-tailed, 95%) ──


_T_TABLE = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000,
    120: 1.980,
}


def _t_critical(df: int) -> float:
    """Approximate t-critical value for given degrees of freedom."""
    if df >= 120:
        return 1.96
    if df in _T_TABLE:
        return _T_TABLE[df]
    # Linear interpolation between nearest keys
    keys = sorted(_T_TABLE.keys())
    for i, k in enumerate(keys):
        if k > df:
            lo, hi = keys[i - 1], k
            frac = (df - lo) / (hi - lo)
            return _T_TABLE[lo] + frac * (_T_TABLE[hi] - _T_TABLE[lo])
    return 1.96
