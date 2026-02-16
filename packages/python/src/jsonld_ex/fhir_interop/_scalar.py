"""
Scalar-to-Opinion reconstruction and FHIR extension serialization.

Provides the core bridge between FHIR's scalar probability model and
jsonld-ex's Subjective Logic opinion model.
"""

from __future__ import annotations

from typing import Any, Optional

from jsonld_ex.confidence_algebra import Opinion

from jsonld_ex.fhir_interop._constants import (
    FHIR_EXTENSION_URL,
    STATUS_UNCERTAINTY_MULTIPLIER,
)


def scalar_to_opinion(
    probability: float,
    *,
    status: Optional[str] = None,
    basis_count: Optional[int] = None,
    method: Optional[str] = None,
    default_uncertainty: float = 0.15,
    base_rate: Optional[float] = None,
) -> Opinion:
    """Reconstruct an SL opinion from a FHIR scalar probability.

    FHIR represents risk/likelihood as a bare decimal (e.g.
    ``RiskAssessment.prediction.probabilityDecimal = 0.87``).  This
    discards epistemic state: *how much* evidence supports that number?
    Do sources agree?  Is this preliminary or final?

    This function recovers that lost epistemic state by analysing
    metadata signals that FHIR already provides but does not
    structurally exploit:

    - **status** — resource lifecycle stage (final, preliminary, etc.)
    - **basis_count** — number of evidence references cited
    - **method** — whether an evaluation method is documented

    Each signal adjusts the uncertainty budget multiplicatively.
    The remaining mass ``(1 - u)`` is distributed between belief and
    disbelief proportionally to ``probability`` and ``1 - probability``.

    When ``base_rate`` is not explicitly provided, it defaults to the
    input ``probability``.  This ensures that the projected probability
    ``P(ω) = b + a·u`` exactly equals the original scalar, preserving
    semantic fidelity while adding uncertainty information.

    Args:
        probability: Scalar probability in [0, 1].
        status: FHIR resource status code (e.g. "final", "preliminary").
            ``None`` means no status signal — no adjustment applied.
        basis_count: Number of evidence references (``RiskAssessment.basis``,
            etc.).  ``None`` means unknown — no adjustment.  ``0`` means
            explicitly no evidence cited — increases uncertainty.
        method: Evaluation method name.  If truthy, indicates a documented
            methodology — reduces uncertainty.  ``None`` means absent.
        default_uncertainty: Base uncertainty budget before adjustments.
            Must be in [0, 1).  Default 0.15.
        base_rate: Prior probability for the Opinion.  Default ``None``
            means use ``probability`` as the base rate (preserves
            projected probability exactly).

    Returns:
        An ``Opinion`` with the reconstructed epistemic state.

    Raises:
        ValueError: If ``probability`` is outside [0, 1], or
            ``default_uncertainty`` is outside [0, 1), or
            ``basis_count`` is negative.
        TypeError: If ``probability`` is not numeric.
    """
    # ── Input validation ──────────────────────────────────────────
    if not isinstance(probability, (int, float)):
        raise TypeError(
            f"probability must be a number, got: {type(probability).__name__}"
        )
    if probability < 0.0 or probability > 1.0:
        raise ValueError(f"probability must be in [0, 1], got: {probability}")

    if not isinstance(default_uncertainty, (int, float)):
        raise TypeError(
            f"default_uncertainty must be a number, got: "
            f"{type(default_uncertainty).__name__}"
        )
    if default_uncertainty < 0.0:
        raise ValueError(
            f"default_uncertainty must be >= 0, got: {default_uncertainty}"
        )
    if default_uncertainty >= 1.0:
        raise ValueError(
            f"default_uncertainty must be < 1.0, got: {default_uncertainty}"
        )

    if basis_count is not None and basis_count < 0:
        raise ValueError(f"basis_count must be >= 0, got: {basis_count}")

    # ── Compute adjusted uncertainty ──────────────────────────────
    u = float(default_uncertainty)

    # Status signal
    if status is not None:
        multiplier = STATUS_UNCERTAINTY_MULTIPLIER.get(status, 1.0)
        u *= multiplier

    # Basis count signal
    if basis_count is not None:
        if basis_count == 0:
            u *= 1.5
        elif basis_count >= 3:
            u *= 0.7
        else:
            # 1–2 evidence items: modest reduction
            u *= 0.9

    # Method signal
    if method:
        u *= 0.8

    # Clamp uncertainty to valid range [0, 1)
    # Must be strictly < 1.0 to leave mass for b and d.
    u = max(0.0, min(u, 0.99))

    # ── Distribute remaining mass ─────────────────────────────────
    p = float(probability)
    remaining = 1.0 - u
    b = p * remaining
    d = (1.0 - p) * remaining

    # ── Determine base rate ───────────────────────────────────────
    a = base_rate if base_rate is not None else p

    return Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)


def opinion_to_fhir_extension(opinion: Opinion) -> dict[str, Any]:
    """Serialize an SL opinion to a FHIR R4 complex extension.

    Produces a structure conforming to FHIR R4 §2.4 (Extensions).
    This extension can be attached to any FHIR element using the
    ``_<element>`` convention.  Systems that do not recognise the
    extension will silently ignore it.

    Args:
        opinion: The SL opinion to serialize.

    Returns:
        A dict representing a valid FHIR R4 complex extension.
    """
    return {
        "url": FHIR_EXTENSION_URL,
        "extension": [
            {"url": "belief", "valueDecimal": opinion.belief},
            {"url": "disbelief", "valueDecimal": opinion.disbelief},
            {"url": "uncertainty", "valueDecimal": opinion.uncertainty},
            {"url": "baseRate", "valueDecimal": opinion.base_rate},
        ],
    }


def fhir_extension_to_opinion(extension: dict[str, Any]) -> Opinion:
    """Deserialize a FHIR R4 complex extension to an SL opinion.

    Reverses ``opinion_to_fhir_extension()``.  The extension must have
    the correct URL and contain sub-extensions for at least ``belief``,
    ``disbelief``, and ``uncertainty``.  ``baseRate`` defaults to 0.5
    if absent.

    Args:
        extension: A dict representing a FHIR R4 complex extension.

    Returns:
        The deserialized ``Opinion``.

    Raises:
        ValueError: If the extension URL does not match, or required
            sub-extensions (belief, disbelief, uncertainty) are missing.
    """
    url = extension.get("url")
    if url != FHIR_EXTENSION_URL:
        raise ValueError(
            f"Expected extension URL '{FHIR_EXTENSION_URL}', "
            f"got '{url}'"
        )

    # Parse sub-extensions into a lookup dict
    values: dict[str, float] = {}
    for sub in extension.get("extension", []):
        sub_url = sub.get("url")
        if sub_url and "valueDecimal" in sub:
            values[sub_url] = float(sub["valueDecimal"])

    # Validate required fields
    required = {"belief", "disbelief", "uncertainty"}
    missing = required - values.keys()
    if missing:
        raise ValueError(
            f"Missing required sub-extension(s): {sorted(missing)}"
        )

    return Opinion(
        belief=values["belief"],
        disbelief=values["disbelief"],
        uncertainty=values["uncertainty"],
        base_rate=values.get("baseRate", 0.5),
    )
