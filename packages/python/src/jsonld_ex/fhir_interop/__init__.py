"""
FHIR R4 Interoperability for JSON-LD-Ex.

Bidirectional mapping between FHIR R4 clinical resources and jsonld-ex's
Subjective Logic opinion model.  Provides mathematically grounded
uncertainty that composes, fuses, and propagates correctly — capabilities
that FHIR's scalar probability and categorical code model lack.

Supported FHIR R4 resources (30 types, 4 epistemic tiers):

  Tier 1 — Clinical assertions (rich signal, domain-specific mappings):
  - RiskAssessment      — prediction.probabilityDecimal → Opinion
  - Observation          — interpretation or status fallback → Opinion
  - DiagnosticReport     — conclusion or status fallback → Opinion
  - Condition            — verificationStatus → continuous Opinion
  - AllergyIntolerance   — verificationStatus + criticality → dual Opinion
  - MedicationStatement  — status → adherence confidence Opinion
  - ClinicalImpression   — status + findings → assessment Opinion
  - DetectedIssue        — severity → alert confidence Opinion
  - Immunization         — status → seroconversion base confidence Opinion
  - FamilyMemberHistory  — status → reported-vs-confirmed evidence Opinion
  - Procedure            — status + outcome/complication/followUp → Opinion
  - Consent              — status → consent lawfulness Opinion
  - Provenance           — recorded + agent/entity → chain reliability Opinion
  - QuestionnaireResponse — status × source × completeness → response reliability Opinion
  - Specimen             — status × condition[] × processing × collector × quantity → specimen suitability Opinion

  Tier 2 — Clinical workflow (status-based, "event occurred as documented"):
  - ServiceRequest       — status × intent × priority × evidence → order validity Opinion
  - Encounter            — status → visit validity Opinion
  - MedicationRequest    — status → prescription validity Opinion
  - MedicationAdministration — status → administration validity Opinion
  - CarePlan             — status → plan adherence Opinion
  - Goal                 — lifecycleStatus + achievementStatus → dual Opinion
  - CareTeam             — status → assignment validity Opinion
  - ImagingStudy         — status → study validity Opinion
  - DocumentReference    — status × docStatus × authenticator × author × content → document reliability Opinion

  Tier 3 — Administrative identity ("this record is accurate and current"):
  - Patient              — data completeness → record accuracy Opinion
  - Organization         — boolean active → record validity Opinion
  - Practitioner         — boolean active → record validity Opinion
  - Device               — status → device record validity Opinion

  Tier 4 — Financial ("this claim/adjudication is valid"):
  - Claim                — status → claim validity Opinion
  - ExplanationOfBenefit — status + outcome → dual Opinion

Architecture notes:
  - All public functions accept a ``fhir_version`` parameter (default "R4")
    so that R5/R6 support can be added without breaking existing callers.
  - Version-specific logic is isolated in private ``_*_r4()`` functions.
  - The SL opinion is embedded in FHIR resources via the standard extension
    mechanism, ensuring zero breaking changes to existing FHIR infrastructure.
  - Generic ``_make_status_handler`` factory eliminates boilerplate for
    simple status-based converters.

Round-trip fidelity scope:
  The ``from_fhir() → to_fhir()`` round-trip preserves:
    - Resource id, status, and type
    - Opinion tuples (exact when via extension, reconstructed otherwise)
    - Key metadata: vaccineCode, occurrenceDateTime, relationship,
      outcome text, statusReason, implicated references
  It does NOT preserve raw FHIR arrays that were used to *infer* the
  opinion (evidence[], protocolApplied[], condition[], complication[],
  followUp[], derivedFrom[], etc.).  This is by design: these arrays
  inform the uncertainty budget during ``from_fhir()`` and their
  evidential weight is encoded in the resulting opinion.  Systems
  needing the original FHIR arrays should retain the source resource.

References:
  - HL7 FHIR R4: https://hl7.org/fhir/R4/
  - Jøsang, A. (2016). Subjective Logic. Springer.
"""

from jsonld_ex.fhir_interop._constants import (
    FHIR_EXTENSION_URL,
    SUPPORTED_FHIR_VERSIONS,
    SUPPORTED_RESOURCE_TYPES,
    FAMILY_HISTORY_DEFAULT_UNCERTAINTY,
    CONSENT_STATUS_PROBABILITY,
    CONSENT_STATUS_UNCERTAINTY,
    # Phase 7A
    SERVICE_REQUEST_STATUS_PROBABILITY,
    SERVICE_REQUEST_STATUS_UNCERTAINTY,
    SERVICE_REQUEST_INTENT_MULTIPLIER,
    SERVICE_REQUEST_PRIORITY_MULTIPLIER,
    # Phase 7A — QuestionnaireResponse
    QR_STATUS_PROBABILITY,
    QR_STATUS_UNCERTAINTY,
    QR_SOURCE_RELIABILITY_MULTIPLIER,
    QR_COMPLETENESS_THRESHOLDS,
    # Phase 7A — Specimen
    SPECIMEN_STATUS_PROBABILITY,
    SPECIMEN_STATUS_UNCERTAINTY,
    SPECIMEN_CONDITION_MULTIPLIER,
    # Phase 7B — DocumentReference
    DOC_REF_STATUS_PROBABILITY,
    DOC_REF_STATUS_UNCERTAINTY,
    DOC_REF_DOC_STATUS_MULTIPLIER,
)
from jsonld_ex.fhir_interop._scalar import (
    scalar_to_opinion,
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
)
from jsonld_ex.fhir_interop._converters import (
    from_fhir,
    to_fhir,
)
from jsonld_ex.fhir_interop._fusion import (
    FusionReport,
    TrustChainReport,
    fhir_clinical_fuse,
    fhir_trust_chain,
)
from jsonld_ex.fhir_interop._temporal import (
    fhir_temporal_decay,
)
from jsonld_ex.fhir_interop._escalation import (
    fhir_escalation_policy,
)
from jsonld_ex.fhir_interop._bundle import (
    BundleReport,
    fhir_bundle_annotate,
    fhir_bundle_fuse,
)
from jsonld_ex.fhir_interop._compliance import (
    fhir_consent_to_opinion,
    opinion_to_fhir_consent,
    fhir_consent_validity,
    fhir_consent_withdrawal,
    fhir_multi_site_meet,
    fhir_consent_expiry,
    fhir_consent_regulatory_change,
)
from jsonld_ex.fhir_interop._provenance import (
    fhir_provenance_to_prov_o,
)
from jsonld_ex.fhir_interop._reconciliation import (
    ReconciliationReport,
    fhir_allergy_reconcile,
)
from jsonld_ex.fhir_interop._alert_filter import (
    AlertFilterReport,
    fhir_filter_alerts,
)

__all__ = [
    # Core conversion
    "scalar_to_opinion",
    "opinion_to_fhir_extension",
    "fhir_extension_to_opinion",
    "from_fhir",
    "to_fhir",
    # Constants
    "FHIR_EXTENSION_URL",
    "SUPPORTED_FHIR_VERSIONS",
    "SUPPORTED_RESOURCE_TYPES",
    "FAMILY_HISTORY_DEFAULT_UNCERTAINTY",
    "CONSENT_STATUS_PROBABILITY",
    "CONSENT_STATUS_UNCERTAINTY",
    "SERVICE_REQUEST_STATUS_PROBABILITY",
    "SERVICE_REQUEST_STATUS_UNCERTAINTY",
    "SERVICE_REQUEST_INTENT_MULTIPLIER",
    "SERVICE_REQUEST_PRIORITY_MULTIPLIER",
    "QR_STATUS_PROBABILITY",
    "QR_STATUS_UNCERTAINTY",
    "QR_SOURCE_RELIABILITY_MULTIPLIER",
    "QR_COMPLETENESS_THRESHOLDS",
    "SPECIMEN_STATUS_PROBABILITY",
    "SPECIMEN_STATUS_UNCERTAINTY",
    "SPECIMEN_CONDITION_MULTIPLIER",
    "DOC_REF_STATUS_PROBABILITY",
    "DOC_REF_STATUS_UNCERTAINTY",
    "DOC_REF_DOC_STATUS_MULTIPLIER",
    # Fusion & trust
    "fhir_clinical_fuse",
    "fhir_trust_chain",
    "FusionReport",
    "TrustChainReport",
    # Temporal & escalation
    "fhir_temporal_decay",
    "fhir_escalation_policy",
    # Compliance algebra bridge
    "fhir_consent_to_opinion",
    "opinion_to_fhir_consent",
    "fhir_consent_validity",
    "fhir_consent_withdrawal",
    "fhir_multi_site_meet",
    "fhir_consent_expiry",
    "fhir_consent_regulatory_change",
    # Bundle processing
    "BundleReport",
    "fhir_bundle_annotate",
    "fhir_bundle_fuse",
    # Provenance bridge
    "fhir_provenance_to_prov_o",
    # Allergy reconciliation
    "ReconciliationReport",
    "fhir_allergy_reconcile",
    # Alert filtering
    "AlertFilterReport",
    "fhir_filter_alerts",
]
