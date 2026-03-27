"""Quantization-to-Subjective-Logic uncertainty bridge.

Maps vector quantization distortion to epistemic uncertainty in
Jøsang's Subjective Logic framework, enabling principled reasoning
about the fidelity of quantized similarity scores.

Mathematical foundation
-----------------------
For *b*-bit quantization of unit vectors, the normalized inner-product
distortion rate follows the standard rate-distortion scaling:

    D(b) = k_method · 4^{-b}

where ``k_method`` is a method-dependent constant characterizing the
quantizer's efficiency.  The 4^{-b} = 2^{-2b} scaling is a classical
result from Shannon's source coding theory (Shannon, 1959).

The SL uncertainty is defined (as a modeling choice) as:

    u = min(1, √D(b))

This maps root-mean-square distortion (same scale as inner products
in [-1, 1]) to uncertainty mass.  The motivation is that from the
perspective of a *consumer* who receives quantized vectors without
access to the originals, the distortion constitutes epistemic
uncertainty — they do not know the true inner product.

Important caveats
-----------------
- The distortion rate D(b) has rigorous information-theoretic bounds
  (TurboQuant proves near-optimality within ~2.7× of the Shannon
  lower bound).  However, the mapping u = √D is a *modeling choice*,
  not a uniquely determined result.  Alternative mappings (u = D,
  u = 1 − exp(−D), etc.) are equally defensible mathematically.

- The ``DISTORTION_CONSTANTS`` are *illustrative defaults* reflecting
  the relative efficiency ordering from the literature, not exact
  values from any specific paper.  Actual distortion constants
  depend on vector dimensionality, distribution, and implementation.
  Implementers SHOULD empirically calibrate k for their use case.

- Quantization error is aleatory (systematic, bounded noise) at the
  *quantizer*, but epistemic (unknown) at the *consumer*.  This
  bridge adopts the consumer's epistemic perspective.

References
----------
- Shannon, C.E. (1959). Coding theorems for a discrete source with
  a fidelity criterion. IRE Nat. Conv. Rec., 4, 142-163.
- Zandieh et al. (2025). TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate. ICLR 2026. arXiv:2504.19874.
- Zandieh et al. (2025). PolarQuant. AISTATS 2026. arXiv:2502.02617.
- Zandieh et al. (2024). QJL: 1-Bit Quantized JL Transform.
  AAAI 2025. arXiv:2406.03482.
- Jøsang, A. (2016). Subjective Logic. Springer.
"""

from __future__ import annotations

import math
from typing import Optional

from jsonld_ex.confidence_algebra import Opinion


# ── Distortion constants ───────────────────────────────────────────────
#
# Each constant k represents the method's distortion coefficient in
#     D(b) = k · 4^{-b}
#
# Lower k → less distortion → better quantizer.
#
# Values are set to reflect the relative quality ordering established
# by the TurboQuant paper.  The absolute values are normalised so
# that scalar = 1.0 (baseline) and other methods are expressed
# relative to it.

DISTORTION_CONSTANTS: dict[str, float] = {
    "scalar": 1.0,
    "turboquant": 0.4,
    "polarquant": 0.5,
    "qjl": 1.0,
    "product_quantization": 0.7,
}
"""Illustrative distortion coefficients for known quantization methods.

``scalar`` is the baseline (k = 1.0); other methods are expressed
relative to it.  The relative ordering (turboquant < polarquant <
scalar) reflects the efficiency hierarchy established in the
literature, but the absolute values are *illustrative defaults*,
not exact empirical measurements.

Implementers SHOULD replace these with empirically calibrated
constants for their specific vector distribution and dimensionality.
The default values are suitable for order-of-magnitude reasoning
and for demonstrating the modeling framework.
"""


def quantization_distortion(bit_width: int, method: str) -> float:
    """Compute the normalized inner-product distortion rate.

    D(b) = k_method · 4^{-b}

    Parameters
    ----------
    bit_width:
        Bits per coordinate (positive integer).
    method:
        Quantization method name.  If not found in
        :data:`DISTORTION_CONSTANTS`, falls back to the ``"scalar"``
        constant.

    Returns
    -------
    float
        The expected normalized distortion (non-negative).
    """
    if not isinstance(bit_width, int) or isinstance(bit_width, bool):
        raise TypeError(f"bit_width must be an integer, got: {type(bit_width).__name__}")
    if bit_width < 1:
        raise ValueError(f"bit_width must be >= 1, got: {bit_width}")

    k = DISTORTION_CONSTANTS.get(method, DISTORTION_CONSTANTS["scalar"])
    return k * 4.0 ** (-bit_width)


def distortion_to_uncertainty(distortion: float) -> float:
    """Map quantization distortion to SL uncertainty mass.

    u = min(1.0, √max(0, D))

    The square root maps mean-squared error to root-mean-square error,
    which is on the same scale as inner products in [-1, 1] and thus
    interpretable as uncertainty mass.

    Parameters
    ----------
    distortion:
        The normalized distortion value D ≥ 0.

    Returns
    -------
    float
        Uncertainty mass u ∈ [0, 1].
    """
    if distortion <= 0.0:
        return 0.0
    return min(1.0, math.sqrt(distortion))


def similarity_to_confidence(
    similarity: float,
    range_min: float = -1.0,
    range_max: float = 1.0,
) -> float:
    """Linearly map a similarity score to confidence in [0, 1].

    Parameters
    ----------
    similarity:
        Raw similarity value (e.g. cosine similarity in [-1, 1]).
    range_min:
        Minimum of the similarity scale (maps to 0.0).
    range_max:
        Maximum of the similarity scale (maps to 1.0).

    Returns
    -------
    float
        Confidence ∈ [0, 1], clamped if *similarity* is outside the
        given range.
    """
    if range_max <= range_min:
        raise ValueError(
            f"range_max must be greater than range_min, "
            f"got [{range_min}, {range_max}]"
        )
    normalised = (similarity - range_min) / (range_max - range_min)
    return max(0.0, min(1.0, normalised))


def quantization_to_opinion(
    similarity: float,
    bit_width: int,
    method: str,
    *,
    similarity_range: tuple[float, float] = (-1.0, 1.0),
    base_rate: float = 0.5,
) -> Opinion:
    """Create an SL Opinion from a quantized similarity score.

    Combines three steps:

    1. Compute quantization distortion D(b) for the given method.
    2. Map distortion to uncertainty mass u = min(1, √D).
    3. Map the similarity score to confidence and construct an Opinion
       with the quantization-derived uncertainty.

    Parameters
    ----------
    similarity:
        Raw similarity score (e.g. cosine similarity).
    bit_width:
        Bits per coordinate used in the quantized representation.
    method:
        Quantization method name (e.g. ``"turboquant"``).
    similarity_range:
        ``(min, max)`` of the similarity metric.  Default ``(-1, 1)``
        for cosine similarity.
    base_rate:
        Prior probability for the SL opinion.  Default 0.5.

    Returns
    -------
    Opinion
        An opinion whose uncertainty is grounded in information-
        theoretic distortion bounds rather than heuristic estimation.
    """
    # Step 1: quantization distortion
    d = quantization_distortion(bit_width, method)

    # Step 2: distortion → uncertainty
    u = distortion_to_uncertainty(d)

    # Step 3: similarity → confidence → opinion
    c = similarity_to_confidence(
        similarity,
        range_min=similarity_range[0],
        range_max=similarity_range[1],
    )

    return Opinion.from_confidence(confidence=c, uncertainty=u, base_rate=base_rate)
