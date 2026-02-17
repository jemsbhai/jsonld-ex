"""
FHIR R4 Interoperability for JSON-LD-Ex.

Bidirectional mapping between FHIR R4 clinical resources and jsonld-ex's
Subjective Logic opinion model.  Provides mathematically grounded
uncertainty that composes, fuses, and propagates correctly — capabilities
that FHIR's scalar probability and categorical code model lack.

Supported FHIR R4 resources:
  Phase 1:
  - RiskAssessment      — prediction.probabilityDecimal → Opinion
  - Observation          — interpretation / valueQuantity → Opinion
  - DiagnosticReport     — aggregates Observations via fusion
  - Condition            — verificationStatus → continuous Opinion
  Phase 2:
  - AllergyIntolerance   — verificationStatus + criticality → dual Opinion
  - MedicationStatement  — status → adherence confidence Opinion
  - ClinicalImpression   — status + findings → assessment Opinion
  Phase 3:
  - DetectedIssue        — severity → alert confidence Opinion
  - Immunization         — status → seroconversion base confidence Opinion
  - FamilyMemberHistory  — status → reported-vs-confirmed evidence Opinion
  - Procedure            — status + outcome/complication/followUp → Opinion
  Phase 4:
  - Consent              — status → consent lawfulness Opinion
                           (also available via compliance algebra functions)
  - Provenance           — recorded + agent/entity signals → chain
                           reliability Opinion; bridges to W3C PROV-O
                           via fhir_provenance_to_prov_o()

Architecture notes:
  - All public functions accept a ``fhir_version`` parameter (default "R4")
    so that R5/R6 support can be added without breaking existing callers.
  - Version-specific logic is isolated in private ``_*_r4()`` functions.
  - The SL opinion is embedded in FHIR resources via the standard extension
    mechanism, ensuring zero breaking changes to existing FHIR infrastructure.

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
    FAMILY_HISTORY_DEFAULT_UNCERTAINTY,
    CONSENT_STATUS_PROBABILITY,
    CONSENT_STATUS_UNCERTAINTY,
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
    fhir_clinical_fuse,
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

__all__ = [
    "scalar_to_opinion",
    "opinion_to_fhir_extension",
    "fhir_extension_to_opinion",
    "from_fhir",
    "to_fhir",
    "fhir_clinical_fuse",
    "FusionReport",
    "FHIR_EXTENSION_URL",
    "SUPPORTED_FHIR_VERSIONS",
    "FAMILY_HISTORY_DEFAULT_UNCERTAINTY",
    "fhir_temporal_decay",
    "fhir_escalation_policy",
    # Phase 4: Compliance algebra bridge
    "fhir_consent_to_opinion",
    "opinion_to_fhir_consent",
    "fhir_consent_validity",
    "fhir_consent_withdrawal",
    "fhir_multi_site_meet",
    "fhir_consent_expiry",
    "fhir_consent_regulatory_change",
    "CONSENT_STATUS_PROBABILITY",
    "CONSENT_STATUS_UNCERTAINTY",
    # Phase 4: Bundle processing
    "BundleReport",
    "fhir_bundle_annotate",
    "fhir_bundle_fuse",
    # Phase 4: Provenance bridge
    "fhir_provenance_to_prov_o",
]
