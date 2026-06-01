"""
HIPAA Compliance Algebra — Healthcare Privacy Operators.

Extends the GDPR compliance algebra (compliance_algebra.py) with
four HIPAA-specific operators grounded in specific regulatory provisions.
All operators reuse existing Subjective Logic primitives from jsonld-ex.

Operators:
    H1  phi_classification              — PHI presence assessment (18 identifiers)
    H2  safe_harbor_assessment          — De-identification via Safe Harbor
    H2  expert_determination_assessment — De-identification via Expert
    H3  minimum_necessary               — Minimum Necessary Rule
    H4  baa_trust_chain                 — Business Associate Agreement chain
    H5  dual_regime_composition         — GDPR x HIPAA cross-regime meet

Mathematical design: EH_DESIGN.md (Theorems H1-H5)

Regulatory references:
    HIPAA Privacy Rule: 45 CFR Part 164, Subparts A and E
    - PHI definition: section 160.103
    - De-identification: section 164.514(a)-(b)
    - Safe Harbor method: section 164.514(b)(2)
    - Expert Determination: section 164.514(b)(1)
    - Minimum Necessary: section 164.502(b)
    - Business Associates: section 164.502(e), 164.504(e)

Scope caveat: We are computer scientists proposing a formal model,
not legal scholars providing authoritative HIPAA interpretation. All
mappings from HIPAA provisions to algebraic operations involve interpretive
choices that legal or compliance experts might dispute.

Dependencies:
    - jsonld_ex.compliance_algebra (ComplianceOpinion, jurisdictional_meet,
      compliance_propagation)
    - jsonld_ex.confidence_algebra (Opinion, trust_discount)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from jsonld_ex.confidence_algebra import Opinion, trust_discount
from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    jurisdictional_meet,
    compliance_propagation,
)


# ═══════════════════════════════════════════════════════════════════
# HIPAA Safe Harbor identifier categories (§164.514(b)(2))
# ═══════════════════════════════════════════════════════════════════

SAFE_HARBOR_IDENTIFIERS: list[str] = [
    "names",
    "geographic_subdivisions",       # smaller than state
    "dates",                         # except year
    "phone_numbers",
    "fax_numbers",
    "email_addresses",
    "social_security_numbers",
    "medical_record_numbers",
    "health_plan_beneficiary_numbers",
    "account_numbers",
    "certificate_license_numbers",
    "vehicle_identifiers",
    "device_identifiers",
    "web_urls",
    "ip_addresses",
    "biometric_identifiers",
    "full_face_photographs",
    "other_unique_identifiers",
]

assert len(SAFE_HARBOR_IDENTIFIERS) == 18, "HIPAA Safe Harbor requires exactly 18 categories"


# ═══════════════════════════════════════════════════════════════════
# H1 — PHI CLASSIFICATION CONFIDENCE (§164.514(b)(2))
# ═══════════════════════════════════════════════════════════════════


def phi_classification(
    identifier_opinions: Sequence[ComplianceOpinion],
) -> ComplianceOpinion:
    """PHI Classification — assess whether data contains PHI.

    Per Definition 3 in EH_DESIGN.md. Data is PHI-free only if ALL
    identifiers are absent — a conjunction modeled via Jurisdictional
    Meet (J_⊓).

    Each input opinion ω_i = (l_i, v_i, u_i, a_i) represents confidence
    that identifier category i is absent or properly handled:
        l_i: belief that identifier i is absent (safe)
        v_i: belief that identifier i is present (PHI risk)
        u_i: uncertainty about identifier i's status

    Result: PHI(ω_1, ..., ω_n) = J_⊓(ω_1, ..., ω_n) with:
        l_PHI = ∏ l_i     (all identifiers must be absent)
        v_PHI = 1 - ∏(1-v_i) (any identifier present → PHI)

    Theorem H1 properties: constraint (a), monotonic restriction (b),
    monotonic violation (c), exponential degradation (d), commutativity (e),
    single-identifier dominance (f), partial coverage (g).

    Independence assumption: identifier presences are assessed
    independently. Positive correlation (identifiers co-occur) means
    true PHI risk EXCEEDS our estimate. Bias: non-conservative.

    Args:
        identifier_opinions: Sequence of compliance opinions, one per
            identifier category assessed. Up to 18 for full Safe Harbor.

    Returns:
        ComplianceOpinion for aggregate PHI classification.

    Raises:
        ValueError: If no opinions provided.
    """
    if len(identifier_opinions) == 0:
        raise ValueError("phi_classification requires at least one identifier opinion")
    return jurisdictional_meet(*identifier_opinions)


# ═══════════════════════════════════════════════════════════════════
# H2 — DE-IDENTIFICATION ASSESSMENT (§164.514(a)-(b))
# ═══════════════════════════════════════════════════════════════════


def safe_harbor_assessment(
    removal_opinions: Sequence[ComplianceOpinion],
) -> ComplianceOpinion:
    """Safe Harbor de-identification assessment (§164.514(b)(2)).

    Per Definition 4 in EH_DESIGN.md. Assesses confidence that ALL 18
    identifier categories have been successfully removed.

    Mathematically identical to phi_classification (both use J_⊓) but
    with different semantic content:
        - PHI Classification asks: "is PHI present?"
        - Safe Harbor asks: "has PHI been successfully removed?"

    Each input ω_ri represents confidence that identifier i has been
    successfully removed:
        l_ri: belief that removal of identifier i is complete
        v_ri: belief that identifier i persists after removal attempt
        u_ri: uncertainty about removal completeness

    Result: l_SH = ∏ l_ri (exponential degradation with identifiers).

    Theorem H2(a-b): constraint, exponential degradation.

    Independence assumption: removal success is independent across
    identifiers. Generally more defensible than H1 independence
    (removing one identifier doesn't affect removing another).

    Args:
        removal_opinions: Sequence of compliance opinions, one per
            identifier removal assessment.

    Returns:
        ComplianceOpinion for aggregate de-identification confidence.

    Raises:
        ValueError: If no opinions provided.
    """
    if len(removal_opinions) == 0:
        raise ValueError("safe_harbor_assessment requires at least one removal opinion")
    return jurisdictional_meet(*removal_opinions)


def expert_determination_assessment(
    expert_opinion: ComplianceOpinion,
    expert_trust: ComplianceOpinion,
) -> ComplianceOpinion:
    """Expert Determination de-identification assessment (§164.514(b)(1)).

    Per Definition 5 in EH_DESIGN.md. A qualified statistical or
    scientific expert determines that the risk of identifying an
    individual is "very small." We model this as trust-discounted
    expert opinion.

    Uses Jøsang's trust discount (SL §10.2):
        l_ED = t · l_expert     (linear in trust level)
        v_ED = t · v_expert
        u_ED = (1-t) + t · u_expert  (distrust → uncertainty, not disbelief)

    Theorem H2(c): l_ED = t · l_expert.
    Theorem H2(e): Expert gives higher confidence than Safe Harbor
    when t > p^18 / l_expert (for uniform per-identifier confidence p).

    The key semantic insight: distrust of the expert produces
    UNCERTAINTY about de-identification, not DISBELIEF. We don't
    know the data is identifiable just because we don't trust the expert.

    Args:
        expert_opinion: The expert's assessment of de-identification
            completeness (ω_expert).
        expert_trust: Trust in the expert's competence and methodology
            (ω_trust). Higher belief → expert opinion more influential.

    Returns:
        ComplianceOpinion for expert-assessed de-identification.
    """
    # trust_discount returns a plain Opinion; convert to ComplianceOpinion
    discounted = trust_discount(expert_trust, expert_opinion)
    return ComplianceOpinion.from_opinion(discounted)


# ═══════════════════════════════════════════════════════════════════
# H3 — MINIMUM NECESSARY RULE (§164.502(b))
# ═══════════════════════════════════════════════════════════════════


def minimum_necessary(
    role_opinion: ComplianceOpinion,
    purpose_opinion: ComplianceOpinion,
    scope_opinion: ComplianceOpinion,
) -> ComplianceOpinion:
    """Minimum Necessary Rule assessment (§164.502(b)).

    Per Definition 6 in EH_DESIGN.md. Three independent conditions
    must hold for a PHI disclosure to satisfy Minimum Necessary:

    1. The requestor has a legitimate role (role_opinion)
    2. The purpose is valid — TPO or authorized (purpose_opinion)
    3. The data scope is limited to what is needed (scope_opinion)

    Models as: MinNec(ω_role, ω_purpose, ω_scope) = J_⊓(ω_role, ω_purpose, ω_scope)

    Theorem H3 properties: constraint (a), monotonic restriction (b),
    weakest-link (c), scope relaxation (d).

    Independence assumption: role, purpose, and scope are assessed
    independently. In practice, certain roles imply purpose (e.g.,
    treating physician → treatment). Independence may overstate
    uncertainty. Bias: conservative (overestimates risk).

    Args:
        role_opinion:    Legitimacy of the requestor's role.
        purpose_opinion: Validity of the stated purpose (TPO, research, etc.).
        scope_opinion:   Adequacy of data scope limitation.

    Returns:
        ComplianceOpinion for Minimum Necessary compliance.
    """
    return jurisdictional_meet(role_opinion, purpose_opinion, scope_opinion)


# ═══════════════════════════════════════════════════════════════════
# H4 — BAA TRUST CHAIN (§164.502(e), §164.504(e))
# ═══════════════════════════════════════════════════════════════════


def baa_trust_chain(
    source: ComplianceOpinion,
    chain: Sequence[Tuple[ComplianceOpinion, ComplianceOpinion]],
) -> ComplianceOpinion:
    """Business Associate Agreement trust chain (§164.502(e)).

    Per Definition 8 in EH_DESIGN.md. Models compliance propagation
    through a chain of business associate relationships:

        Covered Entity → BA₁ → BA₂ → ... → BAₙ

    Each link adds compliance uncertainty via Compliance Propagation:
        BAAChain(ω_S, [(τ₁,π₁), ..., (τₙ,πₙ)]) = Propₙ ∘ ... ∘ Prop₁(ω_S)

    where Propᵢ(ω) = J_⊓(ω, τᵢ, πᵢ) and:
        τᵢ = trust that BA's safeguards are adequate ("satisfactory assurances")
        πᵢ = trust that BA's use is purpose-compatible ("permitted uses")

    Result: l_Dₙ = l_S · ∏(tᵢ · pᵢ) — multiplicative degradation.

    Theorem H4 properties: constraint (a), degradation monotonicity (b),
    multiplicative chain decay (c), identity link (d), violation
    annihilation (e), chain associativity (f).

    The provenance chain Π records each link's opinions and timestamp,
    satisfying §164.530(j) documentation requirements (6-year retention).

    Independence assumption: chain links are assessed independently.
    Poor governance may correlate across links. Bias: non-conservative
    in poorly governed organizations.

    Args:
        source: Covered entity's compliance opinion (ω_S).
        chain:  Sequence of (τ, π) tuples, one per BA link.
            τ = safeguard adequacy opinion
            π = purpose compatibility opinion

    Returns:
        ComplianceOpinion for the end-of-chain compliance.

    Raises:
        ValueError: If chain is empty.
    """
    if len(chain) == 0:
        raise ValueError("baa_trust_chain requires at least one link")

    result = source
    for tau, pi in chain:
        result = compliance_propagation(
            source=result,
            derivation_trust=tau,
            purpose_compat=pi,
        )
    return result


# ═══════════════════════════════════════════════════════════════════
# H5 — DUAL-REGIME COMPOSITION (Cross-regulation)
# ═══════════════════════════════════════════════════════════════════


def dual_regime_composition(
    opinion_a: ComplianceOpinion,
    opinion_b: ComplianceOpinion,
) -> ComplianceOpinion:
    """Dual-regime compliance composition.

    Per Definition 9 in EH_DESIGN.md. Computes composite compliance
    for data processing subject to TWO regulatory regimes simultaneously
    (e.g., HIPAA + GDPR for multinational health systems).

    Uses Jurisdictional Meet — the operator designed for exactly this
    purpose (compliance_algebra.md §5):

        DualRegime(ω_A, ω_B) = J_⊓(ω_A, ω_B)

    Theorem H5 properties: constraint (a), composite ≤ individual (b),
    strictest regime dominates (c), regulatory vacuum (d),
    commutativity (e).

    This is, to our knowledge, the first formal framework for computing
    dual-regime HIPAA × GDPR compliance with uncertainty quantification.

    Independence assumption: regime assessments are independent. This
    is defensible — they are structurally different regulations assessed
    by different standards. Positive correlation (organizations that
    comply with one tend to comply with both) means our estimate is
    non-conservative.

    Args:
        opinion_a: Compliance opinion under first regime (e.g., HIPAA).
        opinion_b: Compliance opinion under second regime (e.g., GDPR).

    Returns:
        ComplianceOpinion for composite dual-regime compliance.
    """
    return jurisdictional_meet(opinion_a, opinion_b)
