"""Tests for FHIR R4 interoperability module.

Step 1: scalar_to_opinion(), opinion_to_fhir_extension(),
        fhir_extension_to_opinion(), and round-trip preservation.
Step 2: from_fhir() for RiskAssessment, Observation, DiagnosticReport,
        Condition.
Step 3: to_fhir() for all 4 resources + full round-trip.
Step 4: fhir_clinical_fuse()
Step 5: Edge cases, stress tests, and multi-opinion documents.
"""

import math
import pytest
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    pairwise_conflict,
)
from jsonld_ex.owl_interop import ConversionReport


# ── Import targets (will fail until implementation exists) ────────

from jsonld_ex.fhir_interop import (
    FHIR_EXTENSION_URL,
    SUPPORTED_FHIR_VERSIONS,
    scalar_to_opinion,
    opinion_to_fhir_extension,
    fhir_extension_to_opinion,
    from_fhir,
    to_fhir,
    fhir_clinical_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# scalar_to_opinion() tests
# ═══════════════════════════════════════════════════════════════════


class TestScalarToOpinion:
    """Test conversion of FHIR scalar probabilities to SL opinions."""

    # ── Basic conversion ──────────────────────────────────────────

    def test_basic_probability_returns_opinion(self):
        """A bare probability with no metadata produces a valid Opinion."""
        op = scalar_to_opinion(0.87)
        assert isinstance(op, Opinion)
        # b + d + u must equal 1
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_default_uncertainty_budget(self):
        """Default uncertainty budget is 0.15."""
        op = scalar_to_opinion(0.80)
        # With no metadata signals, the default uncertainty budget
        # should be applied: u = 0.15
        assert abs(op.uncertainty - 0.15) < 1e-9

    def test_custom_uncertainty_budget(self):
        """Custom default_uncertainty is respected."""
        op = scalar_to_opinion(0.80, default_uncertainty=0.30)
        assert abs(op.uncertainty - 0.30) < 1e-9

    def test_projected_probability_preserves_input(self):
        """P(ω) = b + a·u should approximate the original probability.

        The scalar-to-opinion mapping distributes mass from b/d into u
        proportionally, so the projected probability should closely
        approximate the original scalar when base_rate = 0.5.
        """
        for p in [0.0, 0.1, 0.25, 0.5, 0.75, 0.87, 0.95, 1.0]:
            op = scalar_to_opinion(p)
            # Allow small floating-point divergence
            assert abs(op.projected_probability() - p) < 0.02, (
                f"P(ω) = {op.projected_probability():.4f} diverges from "
                f"input probability {p}"
            )

    def test_custom_base_rate(self):
        """base_rate parameter is passed through to the Opinion."""
        op = scalar_to_opinion(0.70, base_rate=0.30)
        assert abs(op.base_rate - 0.30) < 1e-9

    # ── Status signal adjustments ─────────────────────────────────

    def test_status_final_reduces_uncertainty(self):
        """status='final' should reduce uncertainty vs default."""
        op_default = scalar_to_opinion(0.80)
        op_final = scalar_to_opinion(0.80, status="final")
        assert op_final.uncertainty < op_default.uncertainty

    def test_status_preliminary_increases_uncertainty(self):
        """status='preliminary' should increase uncertainty vs default."""
        op_default = scalar_to_opinion(0.80)
        op_prelim = scalar_to_opinion(0.80, status="preliminary")
        assert op_prelim.uncertainty > op_default.uncertainty

    def test_status_amended_reduces_uncertainty(self):
        """status='amended' implies review — should reduce uncertainty."""
        op_default = scalar_to_opinion(0.80)
        op_amended = scalar_to_opinion(0.80, status="amended")
        assert op_amended.uncertainty < op_default.uncertainty

    def test_status_entered_in_error_maximizes_uncertainty(self):
        """status='entered-in-error' should yield very high uncertainty."""
        op = scalar_to_opinion(0.80, status="entered-in-error")
        # Should be near-vacuous — uncertainty should be very high
        assert op.uncertainty > 0.5

    def test_unknown_status_treated_as_no_signal(self):
        """An unrecognized status value should not crash; treat as default."""
        op_default = scalar_to_opinion(0.80)
        op_unknown = scalar_to_opinion(0.80, status="some-future-status")
        assert abs(op_unknown.uncertainty - op_default.uncertainty) < 1e-9

    # ── Basis count adjustments ───────────────────────────────────

    def test_high_basis_count_reduces_uncertainty(self):
        """More evidence (basis_count >= 3) should reduce uncertainty."""
        op_none = scalar_to_opinion(0.80, basis_count=0)
        op_many = scalar_to_opinion(0.80, basis_count=5)
        assert op_many.uncertainty < op_none.uncertainty

    def test_zero_basis_count_increases_uncertainty(self):
        """No cited evidence (basis_count=0) should increase uncertainty."""
        op_zero = scalar_to_opinion(0.80, basis_count=0)
        op_some = scalar_to_opinion(0.80, basis_count=2)
        assert op_zero.uncertainty > op_some.uncertainty

    # ── Method signal ─────────────────────────────────────────────

    def test_method_present_reduces_uncertainty(self):
        """A documented method reduces uncertainty."""
        op_no_method = scalar_to_opinion(0.80)
        op_method = scalar_to_opinion(0.80, method="Framingham")
        assert op_method.uncertainty < op_no_method.uncertainty

    # ── Combined signals ──────────────────────────────────────────

    def test_combined_signals_accumulate(self):
        """Multiple positive signals should compound to lower uncertainty."""
        op_bare = scalar_to_opinion(0.80)
        op_rich = scalar_to_opinion(
            0.80, status="final", basis_count=5, method="SCORE2"
        )
        assert op_rich.uncertainty < op_bare.uncertainty

    # ── Invariant preservation ────────────────────────────────────

    def test_additivity_constraint_always_holds(self):
        """b + d + u = 1 must hold for all parameter combinations."""
        cases = [
            dict(probability=0.0),
            dict(probability=1.0),
            dict(probability=0.5, status="final", basis_count=10),
            dict(probability=0.99, status="preliminary", basis_count=0),
            dict(probability=0.01, status="entered-in-error"),
        ]
        for kwargs in cases:
            op = scalar_to_opinion(**kwargs)
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"Additivity violated for {kwargs}: "
                f"{op.belief} + {op.disbelief} + {op.uncertainty} = {total}"
            )

    def test_belief_disbelief_non_negative(self):
        """b and d must remain >= 0 even with aggressive adjustments."""
        # Extreme case: high uncertainty from entered-in-error
        op = scalar_to_opinion(0.01, status="entered-in-error")
        assert op.belief >= 0.0
        assert op.disbelief >= 0.0
        assert op.uncertainty >= 0.0

    # ── Edge cases ────────────────────────────────────────────────

    def test_probability_zero(self):
        """probability=0.0 should yield belief near 0."""
        op = scalar_to_opinion(0.0)
        assert op.belief < 0.01

    def test_probability_one(self):
        """probability=1.0 should yield disbelief near 0."""
        op = scalar_to_opinion(1.0)
        assert op.disbelief < 0.01

    # ── Input validation ──────────────────────────────────────────

    def test_probability_below_zero_raises(self):
        """Probability < 0 is invalid."""
        with pytest.raises((ValueError, TypeError)):
            scalar_to_opinion(-0.1)

    def test_probability_above_one_raises(self):
        """Probability > 1 is invalid."""
        with pytest.raises((ValueError, TypeError)):
            scalar_to_opinion(1.1)

    def test_negative_basis_count_raises(self):
        """basis_count < 0 is invalid."""
        with pytest.raises(ValueError):
            scalar_to_opinion(0.5, basis_count=-1)

    def test_invalid_uncertainty_budget_raises(self):
        """default_uncertainty outside [0, 1) is invalid."""
        with pytest.raises(ValueError):
            scalar_to_opinion(0.5, default_uncertainty=1.5)

    def test_uncertainty_budget_one_raises(self):
        """default_uncertainty=1.0 would leave no mass for b/d."""
        with pytest.raises(ValueError):
            scalar_to_opinion(0.5, default_uncertainty=1.0)

    # ── Property-based tests ──────────────────────────────────────

    @given(
        p=st.floats(min_value=0.0, max_value=1.0),
        u=st.floats(min_value=0.0, max_value=0.99),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_valid_opinion(self, p, u):
        """Any valid probability + uncertainty budget yields a valid Opinion."""
        assume(not math.isnan(p) and not math.isnan(u))
        op = scalar_to_opinion(p, default_uncertainty=u)
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        assert op.belief >= 0.0
        assert op.disbelief >= 0.0
        assert op.uncertainty >= 0.0


# ═══════════════════════════════════════════════════════════════════
# opinion_to_fhir_extension() / fhir_extension_to_opinion() tests
# ═══════════════════════════════════════════════════════════════════


class TestFhirExtensionRoundTrip:
    """Test FHIR extension serialization and round-trip preservation."""

    # ── Extension structure ───────────────────────────────────────

    def test_extension_has_correct_url(self):
        """Extension must use the registered jsonld-ex FHIR extension URL."""
        op = Opinion(belief=0.65, disbelief=0.15, uncertainty=0.20)
        ext = opinion_to_fhir_extension(op)
        assert ext["url"] == FHIR_EXTENSION_URL

    def test_extension_structure_is_valid_fhir(self):
        """Extension must follow FHIR R4 complex extension structure.

        FHIR complex extensions use nested extension arrays, not
        direct value fields at the top level.
        """
        op = Opinion(belief=0.65, disbelief=0.15, uncertainty=0.20)
        ext = opinion_to_fhir_extension(op)

        assert "url" in ext
        assert "extension" in ext
        # Must have sub-extensions for belief, disbelief, uncertainty, baseRate
        sub_urls = {sub["url"] for sub in ext["extension"]}
        assert "belief" in sub_urls
        assert "disbelief" in sub_urls
        assert "uncertainty" in sub_urls
        assert "baseRate" in sub_urls

    def test_extension_values_are_decimals(self):
        """Sub-extension values must use valueDecimal (FHIR decimal type)."""
        op = Opinion(belief=0.65, disbelief=0.15, uncertainty=0.20)
        ext = opinion_to_fhir_extension(op)
        for sub in ext["extension"]:
            assert "valueDecimal" in sub, (
                f"Sub-extension '{sub['url']}' missing valueDecimal"
            )
            assert isinstance(sub["valueDecimal"], (int, float))

    def test_extension_values_match_opinion(self):
        """Extension values must exactly match the source Opinion."""
        op = Opinion(
            belief=0.65, disbelief=0.15, uncertainty=0.20, base_rate=0.40
        )
        ext = opinion_to_fhir_extension(op)
        values = {sub["url"]: sub["valueDecimal"] for sub in ext["extension"]}
        assert abs(values["belief"] - 0.65) < 1e-9
        assert abs(values["disbelief"] - 0.15) < 1e-9
        assert abs(values["uncertainty"] - 0.20) < 1e-9
        assert abs(values["baseRate"] - 0.40) < 1e-9

    # ── Deserialization ───────────────────────────────────────────

    def test_extension_to_opinion_basic(self):
        """Deserialize a well-formed FHIR extension to an Opinion."""
        ext = {
            "url": FHIR_EXTENSION_URL,
            "extension": [
                {"url": "belief", "valueDecimal": 0.70},
                {"url": "disbelief", "valueDecimal": 0.10},
                {"url": "uncertainty", "valueDecimal": 0.20},
                {"url": "baseRate", "valueDecimal": 0.50},
            ],
        }
        op = fhir_extension_to_opinion(ext)
        assert isinstance(op, Opinion)
        assert abs(op.belief - 0.70) < 1e-9
        assert abs(op.disbelief - 0.10) < 1e-9
        assert abs(op.uncertainty - 0.20) < 1e-9
        assert abs(op.base_rate - 0.50) < 1e-9

    def test_extension_to_opinion_default_base_rate(self):
        """Missing baseRate sub-extension defaults to 0.5."""
        ext = {
            "url": FHIR_EXTENSION_URL,
            "extension": [
                {"url": "belief", "valueDecimal": 0.60},
                {"url": "disbelief", "valueDecimal": 0.10},
                {"url": "uncertainty", "valueDecimal": 0.30},
            ],
        }
        op = fhir_extension_to_opinion(ext)
        assert abs(op.base_rate - 0.50) < 1e-9

    def test_extension_to_opinion_missing_required_field_raises(self):
        """Missing belief/disbelief/uncertainty should raise ValueError."""
        ext = {
            "url": FHIR_EXTENSION_URL,
            "extension": [
                {"url": "belief", "valueDecimal": 0.70},
                # missing disbelief and uncertainty
            ],
        }
        with pytest.raises(ValueError):
            fhir_extension_to_opinion(ext)

    def test_extension_to_opinion_wrong_url_raises(self):
        """Extension with wrong URL should raise ValueError."""
        ext = {
            "url": "http://example.org/wrong-extension",
            "extension": [
                {"url": "belief", "valueDecimal": 0.70},
                {"url": "disbelief", "valueDecimal": 0.10},
                {"url": "uncertainty", "valueDecimal": 0.20},
            ],
        }
        with pytest.raises(ValueError):
            fhir_extension_to_opinion(ext)

    # ── Round-trip preservation ───────────────────────────────────

    def test_round_trip_exact(self):
        """opinion → extension → opinion must preserve all values."""
        original = Opinion(
            belief=0.65, disbelief=0.15, uncertainty=0.20, base_rate=0.40
        )
        ext = opinion_to_fhir_extension(original)
        recovered = fhir_extension_to_opinion(ext)
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9
        assert abs(recovered.base_rate - original.base_rate) < 1e-9

    def test_round_trip_default_base_rate(self):
        """Round-trip with default base_rate=0.5 preserves correctly."""
        original = Opinion(belief=0.80, disbelief=0.10, uncertainty=0.10)
        ext = opinion_to_fhir_extension(original)
        recovered = fhir_extension_to_opinion(ext)
        assert abs(recovered.base_rate - 0.50) < 1e-9
        assert abs(recovered.belief - original.belief) < 1e-9

    @given(
        b=st.floats(min_value=0.0, max_value=1.0),
        d=st.floats(min_value=0.0, max_value=1.0),
        a=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_hypothesis_round_trip(self, b, d, a):
        """Property-based: any valid Opinion survives a round-trip."""
        assume(not math.isnan(b) and not math.isnan(d) and not math.isnan(a))
        u = 1.0 - b - d
        assume(u >= 0.0 and u <= 1.0)
        try:
            original = Opinion(belief=b, disbelief=d, uncertainty=u, base_rate=a)
        except (ValueError, TypeError):
            assume(False)  # skip invalid opinions
        ext = opinion_to_fhir_extension(original)
        recovered = fhir_extension_to_opinion(ext)
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9
        assert abs(recovered.base_rate - original.base_rate) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# FHIR_EXTENSION_URL and module constants
# ═══════════════════════════════════════════════════════════════════


class TestModuleConstants:
    """Test that module-level constants are correctly defined."""

    def test_extension_url_is_resolvable_pattern(self):
        """Extension URL must follow FHIR convention: a resolvable HTTPS URL."""
        assert FHIR_EXTENSION_URL.startswith("https://")
        assert "fhir" in FHIR_EXTENSION_URL.lower()

    def test_supported_versions_includes_r4(self):
        """R4 must be in the supported versions tuple."""
        assert "R4" in SUPPORTED_FHIR_VERSIONS


# ═══════════════════════════════════════════════════════════════════
# Step 2: from_fhir() — FHIR R4 resource → jsonld-ex document
# ═══════════════════════════════════════════════════════════════════
#
# from_fhir() accepts a FHIR R4 resource dict and returns:
#   (jsonld_ex_doc, ConversionReport)
#
# The output document contains:
#   "@type"         : "fhir:<ResourceType>"
#   "id"            : FHIR resource id (if present)
#   "status"        : FHIR resource status
#   "opinions"      : list of dicts, each with:
#       "field"     : dotted path to the source FHIR field
#       "value"     : the original scalar/categorical value
#       "opinion"   : Opinion object (reconstructed or recovered)
#       "source"    : "extension" | "reconstructed"
#   Plus resource-specific metadata.
#
# ═══════════════════════════════════════════════════════════════════


# ── FHIR R4 fixture helpers ───────────────────────────────────────


def _risk_assessment(
    *,
    probability=0.87,
    outcome="Cardiovascular event",
    status="final",
    basis=None,
    method=None,
    extensions=None,
):
    """Build a minimal FHIR R4 RiskAssessment resource."""
    prediction = {
        "outcome": {"text": outcome},
        "probabilityDecimal": probability,
    }
    if extensions:
        prediction["_probabilityDecimal"] = {"extension": extensions}

    resource = {
        "resourceType": "RiskAssessment",
        "id": "risk-example-1",
        "status": status,
        "prediction": [prediction],
    }
    if basis is not None:
        resource["basis"] = [{"reference": ref} for ref in basis]
    if method is not None:
        resource["method"] = {"text": method}
    return resource


def _observation(
    *,
    status="final",
    code_text="Blood Pressure",
    code_system="http://loinc.org",
    code_code="85354-9",
    value_quantity=None,
    interpretation=None,
    extensions=None,
):
    """Build a minimal FHIR R4 Observation resource."""
    resource = {
        "resourceType": "Observation",
        "id": "obs-example-1",
        "status": status,
        "code": {
            "coding": [{"system": code_system, "code": code_code}],
            "text": code_text,
        },
    }
    if value_quantity is not None:
        resource["valueQuantity"] = value_quantity
    if interpretation is not None:
        resource["interpretation"] = [{"coding": [{"code": interpretation}]}]
    if extensions:
        resource["_interpretation"] = {"extension": extensions}
    return resource


def _diagnostic_report(
    *,
    status="final",
    code_text="Lipid Panel",
    conclusion=None,
    results=None,
    extensions=None,
):
    """Build a minimal FHIR R4 DiagnosticReport resource."""
    resource = {
        "resourceType": "DiagnosticReport",
        "id": "report-example-1",
        "status": status,
        "code": {"text": code_text},
    }
    if conclusion:
        resource["conclusion"] = conclusion
    if results:
        resource["result"] = [{"reference": f"Observation/{r}"} for r in results]
    if extensions:
        resource["_conclusion"] = {"extension": extensions}
    return resource


def _condition(
    *,
    clinical_status="active",
    verification_status="confirmed",
    code_text="Type 2 Diabetes",
    evidence=None,
    extensions=None,
):
    """Build a minimal FHIR R4 Condition resource."""
    resource = {
        "resourceType": "Condition",
        "id": "condition-example-1",
        "clinicalStatus": {
            "coding": [{"code": clinical_status}],
        },
        "verificationStatus": {
            "coding": [{"code": verification_status}],
        },
        "code": {"text": code_text},
    }
    if evidence:
        resource["evidence"] = evidence
    if extensions:
        resource["_verificationStatus"] = {"extension": extensions}
    return resource


# ── RiskAssessment tests ──────────────────────────────────────────


class TestFromFhirRiskAssessment:
    """from_fhir() for FHIR R4 RiskAssessment resources."""

    def test_basic_conversion(self):
        """RiskAssessment with probabilityDecimal → Opinion."""
        resource = _risk_assessment(probability=0.87, status="final")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:RiskAssessment"
        assert len(doc["opinions"]) == 1

        entry = doc["opinions"][0]
        assert entry["value"] == 0.87
        assert isinstance(entry["opinion"], Opinion)
        # Additivity
        op = entry["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_status_signal_extracted(self):
        """Status is used as a signal — 'final' vs 'preliminary' differ."""
        res_final = _risk_assessment(probability=0.70, status="final")
        res_prelim = _risk_assessment(probability=0.70, status="preliminary")

        doc_final, _ = from_fhir(res_final)
        doc_prelim, _ = from_fhir(res_prelim)

        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_final < u_prelim

    def test_basis_count_extracted(self):
        """basis array length is used as evidence count signal."""
        res_none = _risk_assessment(probability=0.70, basis=[])
        res_many = _risk_assessment(
            probability=0.70,
            basis=["Lab/1", "Lab/2", "Lab/3", "Imaging/1"],
        )

        doc_none, _ = from_fhir(res_none)
        doc_many, _ = from_fhir(res_many)

        u_none = doc_none["opinions"][0]["opinion"].uncertainty
        u_many = doc_many["opinions"][0]["opinion"].uncertainty
        assert u_many < u_none

    def test_method_signal_extracted(self):
        """method field presence reduces uncertainty."""
        res_no_method = _risk_assessment(probability=0.70)
        res_method = _risk_assessment(probability=0.70, method="Framingham")

        doc_no, _ = from_fhir(res_no_method)
        doc_yes, _ = from_fhir(res_method)

        u_no = doc_no["opinions"][0]["opinion"].uncertainty
        u_yes = doc_yes["opinions"][0]["opinion"].uncertainty
        assert u_yes < u_no

    def test_extension_recovery_overrides_reconstruction(self):
        """If opinion extension is present, use exact values instead of heuristic."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.60, disbelief=0.10, uncertainty=0.30, base_rate=0.45)
        )
        resource = _risk_assessment(
            probability=0.72,
            status="preliminary",
            extensions=[ext],
        )
        doc, report = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        op = entry["opinion"]
        assert abs(op.belief - 0.60) < 1e-9
        assert abs(op.disbelief - 0.10) < 1e-9
        assert abs(op.uncertainty - 0.30) < 1e-9
        assert abs(op.base_rate - 0.45) < 1e-9

    def test_reconstructed_source_flag(self):
        """Without extension, source should be 'reconstructed'."""
        resource = _risk_assessment(probability=0.87)
        doc, _ = from_fhir(resource)
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_report_nodes_converted(self):
        """ConversionReport tracks how many predictions were converted."""
        resource = _risk_assessment()
        _, report = from_fhir(resource)
        assert report.nodes_converted >= 1

    def test_missing_prediction_produces_empty_opinions(self):
        """RiskAssessment without prediction array → empty opinions, still succeeds."""
        resource = {
            "resourceType": "RiskAssessment",
            "id": "risk-empty",
            "status": "final",
        }
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc["opinions"] == []


# ── Observation tests ─────────────────────────────────────────────


class TestFromFhirObservation:
    """from_fhir() for FHIR R4 Observation resources."""

    def test_basic_with_interpretation(self):
        """Observation with interpretation code → Opinion."""
        resource = _observation(
            status="final",
            interpretation="H",  # High
        )
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:Observation"
        assert len(doc["opinions"]) >= 1

        entry = doc["opinions"][0]
        assert isinstance(entry["opinion"], Opinion)
        op = entry["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_interpretation_codes_produce_different_opinions(self):
        """Different interpretation codes (H, L, N) yield different opinions."""
        opinions = {}
        for code in ["H", "L", "N"]:
            resource = _observation(status="final", interpretation=code)
            doc, _ = from_fhir(resource)
            opinions[code] = doc["opinions"][0]["opinion"]

        # H (High) and L (Low) are abnormal — should have different
        # belief profiles than N (Normal)
        assert opinions["H"].belief != opinions["N"].belief
        assert opinions["L"].belief != opinions["N"].belief

    def test_status_signal_affects_uncertainty(self):
        """Observation status modulates uncertainty."""
        res_final = _observation(status="final", interpretation="H")
        res_prelim = _observation(status="preliminary", interpretation="H")

        doc_final, _ = from_fhir(res_final)
        doc_prelim, _ = from_fhir(res_prelim)

        u_final = doc_final["opinions"][0]["opinion"].uncertainty
        u_prelim = doc_prelim["opinions"][0]["opinion"].uncertainty
        assert u_final < u_prelim

    def test_value_quantity_captured(self):
        """valueQuantity is preserved in the output for reference."""
        resource = _observation(
            value_quantity={"value": 140, "unit": "mmHg"},
            interpretation="H",
        )
        doc, _ = from_fhir(resource)
        assert doc.get("value") == {"value": 140, "unit": "mmHg"}

    def test_extension_recovery(self):
        """Opinion extension on interpretation overrides heuristic."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15, base_rate=0.50)
        )
        resource = _observation(interpretation="H", extensions=[ext])
        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.80) < 1e-9

    def test_no_interpretation_no_opinion(self):
        """Observation without interpretation → empty opinions list."""
        resource = _observation()  # no interpretation
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc["opinions"] == []


# ── DiagnosticReport tests ────────────────────────────────────────


class TestFromFhirDiagnosticReport:
    """from_fhir() for FHIR R4 DiagnosticReport resources."""

    def test_basic_conversion(self):
        """DiagnosticReport with conclusion → output document."""
        resource = _diagnostic_report(
            status="final",
            conclusion="Lipid levels within normal range",
        )
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:DiagnosticReport"
        assert doc["conclusion"] == "Lipid levels within normal range"

    def test_status_captured(self):
        """Status is captured in the output document."""
        resource = _diagnostic_report(status="preliminary")
        doc, _ = from_fhir(resource)
        assert doc["status"] == "preliminary"

    def test_result_references_captured(self):
        """Result references are preserved for downstream fusion."""
        resource = _diagnostic_report(
            results=["obs-ldl", "obs-hdl", "obs-trig"],
        )
        doc, _ = from_fhir(resource)
        assert len(doc["result_references"]) == 3
        assert "Observation/obs-ldl" in doc["result_references"]

    def test_extension_recovery_on_conclusion(self):
        """Opinion extension on conclusion overrides reconstruction."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.75, disbelief=0.10, uncertainty=0.15, base_rate=0.50)
        )
        resource = _diagnostic_report(
            conclusion="Normal findings",
            extensions=[ext],
        )
        doc, _ = from_fhir(resource)

        # Should have an opinion from the extension
        assert len(doc["opinions"]) >= 1
        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.75) < 1e-9

    def test_no_conclusion_no_opinion(self):
        """DiagnosticReport without conclusion → no opinions, still succeeds."""
        resource = _diagnostic_report()
        doc, report = from_fhir(resource)
        assert report.success is True
        assert doc["opinions"] == []


# ── Condition tests ───────────────────────────────────────────────


class TestFromFhirCondition:
    """from_fhir() for FHIR R4 Condition resources."""

    def test_basic_confirmed(self):
        """Condition with verificationStatus='confirmed' → high-belief Opinion."""
        resource = _condition(verification_status="confirmed")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:Condition"
        assert len(doc["opinions"]) == 1

        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        # Confirmed → strong belief
        assert op.belief > 0.7

    def test_provisional_has_moderate_uncertainty(self):
        """verificationStatus='provisional' → moderate uncertainty."""
        resource = _condition(verification_status="provisional")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        # Provisional → meaningful uncertainty
        assert op.uncertainty > 0.2

    def test_differential_has_high_uncertainty(self):
        """verificationStatus='differential' → high uncertainty."""
        resource = _condition(verification_status="differential")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.3

    def test_refuted_has_high_disbelief(self):
        """verificationStatus='refuted' → high disbelief."""
        resource = _condition(verification_status="refuted")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.disbelief > 0.6

    def test_unconfirmed_intermediate(self):
        """verificationStatus='unconfirmed' → between provisional and confirmed."""
        res_unconf = _condition(verification_status="unconfirmed")
        res_conf = _condition(verification_status="confirmed")

        doc_unconf, _ = from_fhir(res_unconf)
        doc_conf, _ = from_fhir(res_conf)

        u_unconf = doc_unconf["opinions"][0]["opinion"].uncertainty
        u_conf = doc_conf["opinions"][0]["opinion"].uncertainty
        assert u_unconf > u_conf

    def test_clinical_status_captured(self):
        """clinicalStatus is preserved in the output document."""
        resource = _condition(clinical_status="active")
        doc, _ = from_fhir(resource)
        assert doc["clinicalStatus"] == "active"

    def test_evidence_count_affects_opinion(self):
        """More evidence items should reduce uncertainty."""
        res_no_ev = _condition(evidence=[])
        res_ev = _condition(evidence=[
            {"code": [{"text": "Lab test"}], "detail": [{"reference": "Obs/1"}]},
            {"code": [{"text": "Imaging"}], "detail": [{"reference": "Obs/2"}]},
            {"code": [{"text": "History"}], "detail": [{"reference": "Obs/3"}]},
        ])

        doc_no, _ = from_fhir(res_no_ev)
        doc_ev, _ = from_fhir(res_ev)

        u_no = doc_no["opinions"][0]["opinion"].uncertainty
        u_ev = doc_ev["opinions"][0]["opinion"].uncertainty
        assert u_ev < u_no

    def test_extension_recovery(self):
        """Opinion extension on verificationStatus overrides heuristic."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20, base_rate=0.50)
        )
        resource = _condition(extensions=[ext])
        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.55) < 1e-9


# ── Cross-resource tests ─────────────────────────────────────────


class TestFromFhirGeneral:
    """Cross-cutting from_fhir() behavior tests."""

    def test_unsupported_resource_type_warning(self):
        """Unsupported FHIR resource type → success with warning."""
        resource = {"resourceType": "Patient", "id": "p-1"}
        doc, report = from_fhir(resource)
        assert report.success is True
        assert len(report.warnings) > 0
        assert any("Patient" in w for w in report.warnings)
        assert doc["opinions"] == []

    def test_missing_resource_type_raises(self):
        """Resource dict without resourceType → ValueError."""
        with pytest.raises(ValueError, match="resourceType"):
            from_fhir({"id": "missing-type"})

    def test_unsupported_fhir_version_raises(self):
        """Explicit unsupported fhir_version → ValueError."""
        resource = _risk_assessment()
        with pytest.raises(ValueError, match="R5"):
            from_fhir(resource, fhir_version="R5")

    def test_report_is_conversion_report(self):
        """Return type is (dict, ConversionReport)."""
        resource = _risk_assessment()
        doc, report = from_fhir(resource)
        assert isinstance(doc, dict)
        assert isinstance(report, ConversionReport)

    def test_additivity_holds_for_all_resources(self):
        """b + d + u = 1 invariant holds across all resource types."""
        resources = [
            _risk_assessment(probability=0.50),
            _observation(interpretation="H"),
            _condition(verification_status="provisional"),
        ]
        for resource in resources:
            doc, _ = from_fhir(resource)
            for entry in doc["opinions"]:
                op = entry["opinion"]
                total = op.belief + op.disbelief + op.uncertainty
                assert abs(total - 1.0) < 1e-9, (
                    f"Additivity violated for {resource['resourceType']}: "
                    f"{op.belief} + {op.disbelief} + {op.uncertainty} = {total}"
                )


# ═══════════════════════════════════════════════════════════════════
# Step 3: to_fhir() — jsonld-ex document → FHIR R4 resource
# ═══════════════════════════════════════════════════════════════════
#
# to_fhir() accepts a jsonld-ex document (as produced by from_fhir())
# and returns:
#   (fhir_resource, ConversionReport)
#
# The output resource:
#   - Has the correct resourceType
#   - Embeds opinions as FHIR extensions on the appropriate elements
#   - Uses projected probability for scalar fields (RiskAssessment)
#   - Preserves all non-opinion metadata from the jsonld-ex doc
#
# ═══════════════════════════════════════════════════════════════════


class TestToFhirRiskAssessment:
    """to_fhir() for RiskAssessment jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex doc → valid RiskAssessment with extension."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.50)
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "risk-1",
            "status": "final",
            "opinions": [{
                "field": "prediction[0].probabilityDecimal",
                "value": 0.87,
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "RiskAssessment"
        assert resource["id"] == "risk-1"
        assert resource["status"] == "final"

    def test_probability_from_projected(self):
        """probabilityDecimal uses opinion's projected probability."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.50)
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "risk-1",
            "status": "final",
            "opinions": [{
                "field": "prediction[0].probabilityDecimal",
                "value": 0.87,
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        pred = resource["prediction"][0]
        expected_p = op.projected_probability()
        assert abs(pred["probabilityDecimal"] - expected_p) < 1e-9

    def test_extension_embedded(self):
        """Opinion is embedded as a FHIR extension on _probabilityDecimal."""
        op = Opinion(belief=0.65, disbelief=0.15, uncertainty=0.20, base_rate=0.50)
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "risk-1",
            "status": "final",
            "opinions": [{
                "field": "prediction[0].probabilityDecimal",
                "value": 0.80,
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        pred = resource["prediction"][0]
        assert "_probabilityDecimal" in pred
        exts = pred["_probabilityDecimal"]["extension"]
        # Find our extension
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1

        # Verify values
        sub_vals = {s["url"]: s["valueDecimal"] for s in our_ext[0]["extension"]}
        assert abs(sub_vals["belief"] - 0.65) < 1e-9
        assert abs(sub_vals["disbelief"] - 0.15) < 1e-9
        assert abs(sub_vals["uncertainty"] - 0.20) < 1e-9

    def test_empty_opinions_produces_minimal_resource(self):
        """No opinions → resource without prediction array."""
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "risk-empty",
            "status": "final",
            "opinions": [],
        }
        resource, report = to_fhir(doc)
        assert report.success is True
        assert resource["resourceType"] == "RiskAssessment"
        assert resource.get("prediction", []) == []


class TestToFhirObservation:
    """to_fhir() for Observation jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex Observation doc → valid FHIR Observation."""
        op = Opinion(belief=0.75, disbelief=0.10, uncertainty=0.15)
        doc = {
            "@type": "fhir:Observation",
            "id": "obs-1",
            "status": "final",
            "opinions": [{
                "field": "interpretation",
                "value": "H",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "Observation"
        assert resource["status"] == "final"

    def test_interpretation_code_preserved(self):
        """Original interpretation code is preserved in output."""
        op = Opinion(belief=0.75, disbelief=0.10, uncertainty=0.15)
        doc = {
            "@type": "fhir:Observation",
            "id": "obs-1",
            "status": "final",
            "opinions": [{
                "field": "interpretation",
                "value": "H",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        interps = resource.get("interpretation", [])
        assert len(interps) >= 1
        codes = [c["code"] for i in interps for c in i.get("coding", [])]
        assert "H" in codes

    def test_extension_on_interpretation(self):
        """Opinion extension is attached to _interpretation."""
        op = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        doc = {
            "@type": "fhir:Observation",
            "id": "obs-1",
            "status": "final",
            "opinions": [{
                "field": "interpretation",
                "value": "H",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_interpretation" in resource
        exts = resource["_interpretation"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1

    def test_value_quantity_preserved(self):
        """valueQuantity from the doc is carried through."""
        op = Opinion(belief=0.75, disbelief=0.10, uncertainty=0.15)
        doc = {
            "@type": "fhir:Observation",
            "id": "obs-1",
            "status": "final",
            "value": {"value": 140, "unit": "mmHg"},
            "opinions": [{
                "field": "interpretation",
                "value": "H",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)
        assert resource.get("valueQuantity") == {"value": 140, "unit": "mmHg"}


class TestToFhirDiagnosticReport:
    """to_fhir() for DiagnosticReport jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex DiagnosticReport → valid FHIR DiagnosticReport."""
        doc = {
            "@type": "fhir:DiagnosticReport",
            "id": "report-1",
            "status": "final",
            "conclusion": "Normal lipid levels",
            "result_references": ["Observation/obs-1"],
            "opinions": [],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "DiagnosticReport"
        assert resource["conclusion"] == "Normal lipid levels"
        assert resource["status"] == "final"

    def test_result_references_exported(self):
        """result_references are converted to FHIR result array."""
        doc = {
            "@type": "fhir:DiagnosticReport",
            "id": "report-1",
            "status": "final",
            "result_references": ["Observation/obs-ldl", "Observation/obs-hdl"],
            "opinions": [],
        }
        resource, _ = to_fhir(doc)

        results = resource.get("result", [])
        assert len(results) == 2
        refs = [r["reference"] for r in results]
        assert "Observation/obs-ldl" in refs
        assert "Observation/obs-hdl" in refs

    def test_extension_on_conclusion(self):
        """Opinion on conclusion is attached as _conclusion extension."""
        op = Opinion(belief=0.75, disbelief=0.10, uncertainty=0.15)
        doc = {
            "@type": "fhir:DiagnosticReport",
            "id": "report-1",
            "status": "final",
            "conclusion": "Normal findings",
            "result_references": [],
            "opinions": [{
                "field": "conclusion",
                "value": "Normal findings",
                "opinion": op,
                "source": "extension",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_conclusion" in resource
        exts = resource["_conclusion"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1


class TestToFhirCondition:
    """to_fhir() for Condition jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex Condition → valid FHIR Condition."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:Condition",
            "id": "cond-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "Condition"

    def test_verification_status_preserved(self):
        """verificationStatus code is preserved from the opinion value."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:Condition",
            "id": "cond-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        vs = resource.get("verificationStatus", {})
        codes = [c["code"] for c in vs.get("coding", [])]
        assert "confirmed" in codes

    def test_clinical_status_preserved(self):
        """clinicalStatus from doc is preserved in output resource."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:Condition",
            "id": "cond-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        cs = resource.get("clinicalStatus", {})
        codes = [c["code"] for c in cs.get("coding", [])]
        assert "active" in codes

    def test_extension_on_verification_status(self):
        """Opinion extension is attached to _verificationStatus."""
        op = Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20)
        doc = {
            "@type": "fhir:Condition",
            "id": "cond-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "provisional",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_verificationStatus" in resource
        exts = resource["_verificationStatus"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1


# ── to_fhir() general & validation tests ─────────────────────────


class TestToFhirGeneral:
    """Cross-cutting to_fhir() behavior tests."""

    def test_missing_type_raises(self):
        """Document without @type → ValueError."""
        with pytest.raises(ValueError, match="@type"):
            to_fhir({"opinions": []})

    def test_unrecognized_type_raises(self):
        """Document with unsupported @type → ValueError."""
        with pytest.raises(ValueError, match="Patient"):
            to_fhir({"@type": "fhir:Patient", "opinions": []})

    def test_report_is_conversion_report(self):
        """Return type is (dict, ConversionReport)."""
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "r-1",
            "status": "final",
            "opinions": [],
        }
        resource, report = to_fhir(doc)
        assert isinstance(resource, dict)
        assert isinstance(report, ConversionReport)

    def test_nodes_converted_count(self):
        """nodes_converted reflects opinions exported."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "r-1",
            "status": "final",
            "opinions": [{
                "field": "prediction[0].probabilityDecimal",
                "value": 0.80,
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        _, report = to_fhir(doc)
        assert report.nodes_converted >= 1


# ── Full round-trip: FHIR → jsonld-ex → FHIR ─────────────────────


class TestFhirFullRoundTrip:
    """FHIR resource → from_fhir() → to_fhir() round-trip fidelity."""

    def test_risk_assessment_round_trip_probability(self):
        """RiskAssessment: FHIR → doc → FHIR preserves projected probability."""
        original = _risk_assessment(probability=0.87, status="final")
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        orig_prob = original["prediction"][0]["probabilityDecimal"]
        recov_prob = recovered["prediction"][0]["probabilityDecimal"]
        # Projected probability should approximate the original
        assert abs(recov_prob - orig_prob) < 0.02

    def test_risk_assessment_round_trip_extension_exact(self):
        """RiskAssessment with extension: exact opinion preserved through round-trip."""
        op = Opinion(belief=0.60, disbelief=0.10, uncertainty=0.30, base_rate=0.45)
        ext = opinion_to_fhir_extension(op)
        original = _risk_assessment(probability=0.72, extensions=[ext])

        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        # Recover the extension from the output
        pred_ext = recovered["prediction"][0]["_probabilityDecimal"]["extension"]
        our_ext = [e for e in pred_ext if e.get("url") == FHIR_EXTENSION_URL][0]
        recovered_op = fhir_extension_to_opinion(our_ext)

        assert abs(recovered_op.belief - op.belief) < 1e-9
        assert abs(recovered_op.disbelief - op.disbelief) < 1e-9
        assert abs(recovered_op.uncertainty - op.uncertainty) < 1e-9
        assert abs(recovered_op.base_rate - op.base_rate) < 1e-9

    def test_condition_round_trip_status(self):
        """Condition: verification status survives round-trip."""
        original = _condition(
            verification_status="provisional",
            clinical_status="active",
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        vs_codes = [
            c["code"]
            for c in recovered.get("verificationStatus", {}).get("coding", [])
        ]
        assert "provisional" in vs_codes

        cs_codes = [
            c["code"]
            for c in recovered.get("clinicalStatus", {}).get("coding", [])
        ]
        assert "active" in cs_codes

    def test_observation_round_trip_interpretation(self):
        """Observation: interpretation code survives round-trip."""
        original = _observation(status="final", interpretation="H")
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        interps = recovered.get("interpretation", [])
        codes = [c["code"] for i in interps for c in i.get("coding", [])]
        assert "H" in codes

    def test_diagnostic_report_round_trip_conclusion(self):
        """DiagnosticReport: conclusion + references survive round-trip."""
        original = _diagnostic_report(
            status="final",
            conclusion="All normal",
            results=["obs-1", "obs-2"],
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        assert recovered["conclusion"] == "All normal"
        refs = [r["reference"] for r in recovered.get("result", [])]
        assert "Observation/obs-1" in refs
        assert "Observation/obs-2" in refs


# ═══════════════════════════════════════════════════════════════════
# Step 4: fhir_clinical_fuse() — fuse opinions across FHIR resources
# ═══════════════════════════════════════════════════════════════════
#
# fhir_clinical_fuse() accepts a list of jsonld-ex documents (as
# produced by from_fhir()) and fuses their opinions using Subjective
# Logic operators.  This is the KEY capability FHIR lacks: combining
# uncertain evidence from multiple clinical sources into a single
# mathematically grounded opinion.
#
# Signature:
#   fhir_clinical_fuse(
#       docs: Sequence[dict],
#       *,
#       method: str = "cumulative",  # or "averaging", "robust"
#   ) -> tuple[Opinion, FusionReport]
#
# FusionReport contains:
#   - input_count: int — number of input documents
#   - opinions_fused: int — number of opinions actually fused
#   - method: str — fusion method used
#   - conflict_scores: list[float] — pairwise conflict scores
#   - warnings: list[str] — any issues encountered
#
# ═══════════════════════════════════════════════════════════════════


# ── Helper to build jsonld-ex docs for fusion ─────────────────────

def _make_doc(resource_type, opinion, *, field="test_field", value=None, doc_id=None):
    """Build a minimal jsonld-ex document with one opinion."""
    return {
        "@type": f"fhir:{resource_type}",
        "id": doc_id or f"{resource_type.lower()}-1",
        "status": "final",
        "opinions": [{
            "field": field,
            "value": value,
            "opinion": opinion,
            "source": "reconstructed",
        }],
    }


class TestFhirClinicalFuseBasic:
    """Basic fhir_clinical_fuse() behavior."""

    def test_two_agreeing_observations(self):
        """Fusing two agreeing Observations -> lower uncertainty than either."""
        op1 = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        op2 = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)

        doc1 = _make_doc("Observation", op1, field="interpretation", value="H")
        doc2 = _make_doc("Observation", op2, field="interpretation", value="H")

        fused, report = fhir_clinical_fuse([doc1, doc2])

        assert isinstance(fused, Opinion)
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9
        # Cumulative fusion of agreeing sources should reduce uncertainty
        assert fused.uncertainty < min(op1.uncertainty, op2.uncertainty)

    def test_three_sources_stronger_than_two(self):
        """Three concordant sources -> less uncertainty than any pair."""
        ops = [
            Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
            Opinion(belief=0.72, disbelief=0.08, uncertainty=0.20),
            Opinion(belief=0.68, disbelief=0.12, uncertainty=0.20),
        ]
        docs = [_make_doc("Observation", op, field="interpretation", value="H") for op in ops]

        fused_3, _ = fhir_clinical_fuse(docs)
        fused_2, _ = fhir_clinical_fuse(docs[:2])

        assert fused_3.uncertainty < fused_2.uncertainty

    def test_single_document_returns_its_opinion(self):
        """Single input -> opinion returned unchanged."""
        op = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        doc = _make_doc("RiskAssessment", op, field="prediction[0].probabilityDecimal")

        fused, report = fhir_clinical_fuse([doc])

        assert abs(fused.belief - op.belief) < 1e-9
        assert abs(fused.disbelief - op.disbelief) < 1e-9
        assert abs(fused.uncertainty - op.uncertainty) < 1e-9
        assert report.opinions_fused == 1

    def test_empty_list_raises(self):
        """Empty document list -> ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty|[Nn]o.*doc|at least"):
            fhir_clinical_fuse([])

    def test_additivity_invariant(self):
        """Fused opinion must satisfy b + d + u = 1."""
        ops = [
            Opinion(belief=0.60, disbelief=0.20, uncertainty=0.20),
            Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15),
            Opinion(belief=0.50, disbelief=0.10, uncertainty=0.40),
        ]
        docs = [_make_doc("Observation", op) for op in ops]

        fused, _ = fhir_clinical_fuse(docs)
        total = fused.belief + fused.disbelief + fused.uncertainty
        assert abs(total - 1.0) < 1e-9


class TestFhirClinicalFuseMethods:
    """Fusion method selection and behavior."""

    def test_cumulative_is_default(self):
        """Default method is cumulative."""
        op1 = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        op2 = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)
        docs = [_make_doc("Observation", op1), _make_doc("Observation", op2)]

        _, report = fhir_clinical_fuse(docs)
        assert report.method == "cumulative"

    def test_averaging_method(self):
        """method='averaging' uses averaging fusion."""
        op1 = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        op2 = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)
        docs = [_make_doc("Observation", op1), _make_doc("Observation", op2)]

        fused, report = fhir_clinical_fuse(docs, method="averaging")
        assert report.method == "averaging"
        assert isinstance(fused, Opinion)
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_cumulative_vs_averaging_differ(self):
        """Cumulative and averaging fusion produce different results."""
        op1 = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        op2 = Opinion(belief=0.60, disbelief=0.15, uncertainty=0.25)
        docs = [_make_doc("Observation", op1), _make_doc("Observation", op2)]

        fused_cum, _ = fhir_clinical_fuse(docs, method="cumulative")
        fused_avg, _ = fhir_clinical_fuse(docs, method="averaging")

        # Cumulative should reduce uncertainty more aggressively
        assert fused_cum.uncertainty < fused_avg.uncertainty

    def test_invalid_method_raises(self):
        """Unrecognized method -> ValueError."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        docs = [_make_doc("Observation", op)]

        with pytest.raises(ValueError, match="[Mm]ethod|[Uu]nsupported"):
            fhir_clinical_fuse(docs, method="nonexistent")


class TestFhirClinicalFuseConflict:
    """Conflict detection and reporting."""

    def test_conflict_scores_reported(self):
        """FusionReport includes pairwise conflict scores."""
        op1 = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        op2 = Opinion(belief=0.10, disbelief=0.70, uncertainty=0.20)
        docs = [_make_doc("Observation", op1), _make_doc("Observation", op2)]

        _, report = fhir_clinical_fuse(docs)
        assert hasattr(report, "conflict_scores")
        assert len(report.conflict_scores) >= 1
        # These two strongly disagree -- conflict should be high
        assert report.conflict_scores[0] > 0.3

    def test_agreeing_sources_low_conflict(self):
        """Sources that agree should have low conflict."""
        op1 = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        op2 = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)
        docs = [_make_doc("Observation", op1), _make_doc("Observation", op2)]

        _, report = fhir_clinical_fuse(docs)
        for score in report.conflict_scores:
            assert score < 0.2

    def test_robust_method_filters_outlier(self):
        """method='robust' removes conflicting outlier before fusion."""
        honest_1 = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        honest_2 = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)
        rogue = Opinion(belief=0.05, disbelief=0.85, uncertainty=0.10)

        docs = [
            _make_doc("Observation", honest_1, doc_id="honest-1"),
            _make_doc("Observation", honest_2, doc_id="honest-2"),
            _make_doc("Observation", rogue, doc_id="rogue"),
        ]

        fused, report = fhir_clinical_fuse(docs, method="robust")
        assert report.method == "robust"
        # Fused result should align with honest sources, not rogue
        assert fused.belief > 0.5
        assert fused.disbelief < 0.3


class TestFhirClinicalFuseMixedResources:
    """Fusion across different resource types."""

    def test_mixed_resource_types(self):
        """Opinions from different FHIR resource types can be fused."""
        obs_op = Opinion(belief=0.75, disbelief=0.08, uncertainty=0.17)
        risk_op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)

        doc_obs = _make_doc("Observation", obs_op, field="interpretation", value="H")
        doc_risk = _make_doc("RiskAssessment", risk_op, field="prediction[0].probabilityDecimal")

        fused, report = fhir_clinical_fuse([doc_obs, doc_risk])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2

    def test_documents_with_no_opinions_skipped(self):
        """Documents with empty opinions list -> skipped with warning."""
        op = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        doc_with = _make_doc("Observation", op)
        doc_empty = {
            "@type": "fhir:DiagnosticReport",
            "id": "report-empty",
            "status": "final",
            "opinions": [],
        }

        fused, report = fhir_clinical_fuse([doc_with, doc_empty])
        assert report.opinions_fused == 1
        assert len(report.warnings) >= 1
        assert any("skip" in w.lower() or "empty" in w.lower() or "no opinion" in w.lower()
                    for w in report.warnings)


class TestFhirClinicalFuseEndToEnd:
    """End-to-end: raw FHIR resources -> from_fhir -> fhir_clinical_fuse."""

    def test_e2e_observations_fusion(self):
        """Full pipeline: FHIR Observations -> from_fhir -> fuse."""
        obs1 = _observation(status="final", interpretation="H")
        obs2 = _observation(status="final", interpretation="H")

        doc1, _ = from_fhir(obs1)
        doc2, _ = from_fhir(obs2)

        fused, report = fhir_clinical_fuse([doc1, doc2])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_e2e_risk_and_condition_fusion(self):
        """Full pipeline: RiskAssessment + Condition -> fuse."""
        risk = _risk_assessment(probability=0.80, status="final")
        cond = _condition(verification_status="confirmed")

        doc_risk, _ = from_fhir(risk)
        doc_cond, _ = from_fhir(cond)

        fused, report = fhir_clinical_fuse([doc_risk, doc_cond])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2
        # Both sources agree (high belief) -- fused should have high belief
        assert fused.belief > 0.5



# ═══════════════════════════════════════════════════════════════════
# Step 5: Edge cases, stress tests, and multi-opinion documents
# ═══════════════════════════════════════════════════════════════════


class TestFhirEdgeCases:
    """Edge cases and boundary conditions across the FHIR pipeline."""

    def test_from_fhir_multiple_predictions(self):
        """RiskAssessment with multiple predictions → multiple opinions."""
        resource = {
            "resourceType": "RiskAssessment",
            "id": "risk-multi",
            "status": "final",
            "prediction": [
                {"probabilityDecimal": 0.30, "outcome": {"text": "Stroke"}},
                {"probabilityDecimal": 0.70, "outcome": {"text": "MI"}},
            ],
        }
        doc, report = from_fhir(resource)
        assert len(doc["opinions"]) == 2
        assert report.nodes_converted == 2
        # Each opinion should be valid
        for entry in doc["opinions"]:
            op = entry["opinion"]
            assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_to_fhir_multiple_predictions_round_trip(self):
        """Multiple predictions survive from_fhir → to_fhir round-trip."""
        resource = {
            "resourceType": "RiskAssessment",
            "id": "risk-multi",
            "status": "final",
            "prediction": [
                {"probabilityDecimal": 0.30, "outcome": {"text": "Stroke"}},
                {"probabilityDecimal": 0.70, "outcome": {"text": "MI"}},
            ],
        }
        doc, _ = from_fhir(resource)
        recovered, report = to_fhir(doc)
        assert len(recovered["prediction"]) == 2
        assert report.nodes_converted == 2

    def test_extreme_probability_zero(self):
        """probability=0.0 through full pipeline."""
        resource = _risk_assessment(probability=0.0, status="final")
        doc, _ = from_fhir(resource)
        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.01
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

        recovered, _ = to_fhir(doc)
        assert recovered["prediction"][0]["probabilityDecimal"] < 0.02

    def test_extreme_probability_one(self):
        """probability=1.0 through full pipeline."""
        resource = _risk_assessment(probability=1.0, status="final")
        doc, _ = from_fhir(resource)
        op = doc["opinions"][0]["opinion"]
        assert op.disbelief < 0.01
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

        recovered, _ = to_fhir(doc)
        assert recovered["prediction"][0]["probabilityDecimal"] > 0.98

    def test_fuse_many_sources(self):
        """Fusing 10 agreeing sources → very low uncertainty."""
        ops = [
            Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
            for _ in range(10)
        ]
        docs = [_make_doc("Observation", op) for op in ops]

        fused, report = fhir_clinical_fuse(docs)
        assert report.opinions_fused == 10
        # 10 agreeing sources should drive uncertainty very low
        assert fused.uncertainty < 0.05
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

    def test_fuse_all_empty_docs_raises(self):
        """All documents with empty opinions → ValueError."""
        docs = [
            {"@type": "fhir:Observation", "id": "e1", "opinions": []},
            {"@type": "fhir:Observation", "id": "e2", "opinions": []},
        ]
        with pytest.raises(ValueError, match="[Nn]o.*opinion|empty"):
            fhir_clinical_fuse(docs)

    def test_from_fhir_preserves_resource_id(self):
        """Resource id is preserved through from_fhir for all types."""
        resources = [
            _risk_assessment(),
            _observation(interpretation="H"),
            _diagnostic_report(conclusion="Normal"),
            _condition(),
        ]
        expected_ids = [
            "risk-example-1", "obs-example-1",
            "report-example-1", "condition-example-1",
        ]
        for resource, expected_id in zip(resources, expected_ids):
            doc, _ = from_fhir(resource)
            assert doc["id"] == expected_id

    def test_fusion_report_conflict_count(self):
        """n opinions produce n*(n-1)/2 pairwise conflict scores."""
        ops = [
            Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20),
            Opinion(belief=0.60, disbelief=0.15, uncertainty=0.25),
            Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15),
        ]
        docs = [_make_doc("Observation", op) for op in ops]

        _, report = fhir_clinical_fuse(docs)
        # 3 opinions → 3 pairs: (0,1), (0,2), (1,2)
        assert len(report.conflict_scores) == 3

    def test_condition_entered_in_error_high_uncertainty(self):
        """Condition with entered-in-error → very high uncertainty opinion."""
        resource = _condition(verification_status="entered-in-error")
        doc, _ = from_fhir(resource)
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.5
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_fuse_single_opinion_multi_field_doc(self):
        """Document with multiple opinions: fuse extracts first from each doc."""
        op1 = Opinion(belief=0.70, disbelief=0.10, uncertainty=0.20)
        op2 = Opinion(belief=0.60, disbelief=0.15, uncertainty=0.25)
        doc = {
            "@type": "fhir:RiskAssessment",
            "id": "multi-op",
            "status": "final",
            "opinions": [
                {"field": "prediction[0]", "value": 0.80, "opinion": op1, "source": "reconstructed"},
                {"field": "prediction[1]", "value": 0.60, "opinion": op2, "source": "reconstructed"},
            ],
        }
        # fhir_clinical_fuse takes first opinion per doc
        fused, report = fhir_clinical_fuse([doc])
        assert report.opinions_fused == 1
        assert abs(fused.belief - op1.belief) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Phase 2, Step 1: from_fhir() for AllergyIntolerance,
#                  MedicationStatement, ClinicalImpression
# ═══════════════════════════════════════════════════════════════════
#
# Phase 2 adds 3 Tier-1/Tier-2 resources that deepen the clinical
# fusion story.  Each has distinct uncertainty semantics:
#
#   AllergyIntolerance  — verificationStatus + criticality
#                         (multi-dimensional opinion)
#   MedicationStatement — status maps to adherence confidence
#                         (was the medication actually taken?)
#   ClinicalImpression  — status + finding[] references
#                         (practitioner's overall assessment)
#
# ═══════════════════════════════════════════════════════════════════


# ── Phase 2 fixture helpers ───────────────────────────────────────


def _allergy_intolerance(
    *,
    clinical_status="active",
    verification_status="confirmed",
    criticality=None,
    code_text="Penicillin",
    category=None,
    reaction=None,
    extensions=None,
):
    """Build a minimal FHIR R4 AllergyIntolerance resource."""
    resource = {
        "resourceType": "AllergyIntolerance",
        "id": "allergy-example-1",
        "clinicalStatus": {
            "coding": [{"code": clinical_status}],
        },
        "verificationStatus": {
            "coding": [{"code": verification_status}],
        },
        "code": {"text": code_text},
    }
    if criticality is not None:
        resource["criticality"] = criticality
    if category is not None:
        resource["category"] = category if isinstance(category, list) else [category]
    if reaction is not None:
        resource["reaction"] = reaction
    if extensions is not None:
        resource["_verificationStatus"] = {"extension": extensions}
    return resource


def _medication_statement(
    *,
    status="active",
    medication_text="Atorvastatin 20mg",
    information_source=None,
    derived_from=None,
    extensions=None,
):
    """Build a minimal FHIR R4 MedicationStatement resource."""
    resource = {
        "resourceType": "MedicationStatement",
        "id": "medstmt-example-1",
        "status": status,
        "medicationCodeableConcept": {"text": medication_text},
        "subject": {"reference": "Patient/example"},
    }
    if information_source is not None:
        resource["informationSource"] = information_source
    if derived_from is not None:
        resource["derivedFrom"] = [
            {"reference": ref} for ref in derived_from
        ]
    if extensions is not None:
        resource["_status"] = {"extension": extensions}
    return resource


def _clinical_impression(
    *,
    status="completed",
    summary=None,
    finding=None,
    extensions=None,
):
    """Build a minimal FHIR R4 ClinicalImpression resource."""
    resource = {
        "resourceType": "ClinicalImpression",
        "id": "impression-example-1",
        "status": status,
        "subject": {"reference": "Patient/example"},
    }
    if summary is not None:
        resource["summary"] = summary
    if finding is not None:
        resource["finding"] = finding
    if extensions is not None:
        resource["_summary"] = {"extension": extensions}
    return resource


# ── AllergyIntolerance tests ──────────────────────────────────────


class TestFromFhirAllergyIntolerance:
    """from_fhir() for FHIR R4 AllergyIntolerance resources."""

    def test_basic_confirmed_allergy(self):
        """AllergyIntolerance with confirmed verificationStatus → Opinion."""
        resource = _allergy_intolerance(verification_status="confirmed")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:AllergyIntolerance"
        assert len(doc["opinions"]) >= 1

        # verificationStatus opinion should exist
        vs_opinions = [o for o in doc["opinions"] if o["field"] == "verificationStatus"]
        assert len(vs_opinions) == 1
        op = vs_opinions[0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        # Confirmed → strong belief
        assert op.belief > 0.7

    def test_unconfirmed_has_high_uncertainty(self):
        """verificationStatus='unconfirmed' → higher uncertainty than confirmed."""
        res_conf = _allergy_intolerance(verification_status="confirmed")
        res_unconf = _allergy_intolerance(verification_status="unconfirmed")

        doc_conf, _ = from_fhir(res_conf)
        doc_unconf, _ = from_fhir(res_unconf)

        vs_conf = [o for o in doc_conf["opinions"] if o["field"] == "verificationStatus"][0]
        vs_unconf = [o for o in doc_unconf["opinions"] if o["field"] == "verificationStatus"][0]

        assert vs_unconf["opinion"].uncertainty > vs_conf["opinion"].uncertainty

    def test_refuted_has_high_disbelief(self):
        """verificationStatus='refuted' → high disbelief."""
        resource = _allergy_intolerance(verification_status="refuted")
        doc, _ = from_fhir(resource)

        vs_op = [o for o in doc["opinions"] if o["field"] == "verificationStatus"][0]
        assert vs_op["opinion"].disbelief > 0.6

    def test_criticality_high_produces_opinion(self):
        """criticality='high' → additional opinion with high belief."""
        resource = _allergy_intolerance(criticality="high")
        doc, _ = from_fhir(resource)

        crit_opinions = [o for o in doc["opinions"] if o["field"] == "criticality"]
        assert len(crit_opinions) == 1
        op = crit_opinions[0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        # High criticality → high belief in severity
        assert op.belief > 0.6

    def test_criticality_low_produces_lower_belief(self):
        """criticality='low' → belief lower than criticality='high'."""
        res_high = _allergy_intolerance(criticality="high")
        res_low = _allergy_intolerance(criticality="low")

        doc_high, _ = from_fhir(res_high)
        doc_low, _ = from_fhir(res_low)

        crit_high = [o for o in doc_high["opinions"] if o["field"] == "criticality"][0]
        crit_low = [o for o in doc_low["opinions"] if o["field"] == "criticality"][0]

        assert crit_high["opinion"].belief > crit_low["opinion"].belief

    def test_criticality_unable_to_assess_high_uncertainty(self):
        """criticality='unable-to-assess' → high uncertainty."""
        resource = _allergy_intolerance(criticality="unable-to-assess")
        doc, _ = from_fhir(resource)

        crit_op = [o for o in doc["opinions"] if o["field"] == "criticality"][0]
        assert crit_op["opinion"].uncertainty > 0.3

    def test_no_criticality_produces_only_verification_opinion(self):
        """No criticality field → only verificationStatus opinion."""
        resource = _allergy_intolerance()  # no criticality
        doc, _ = from_fhir(resource)

        crit_opinions = [o for o in doc["opinions"] if o["field"] == "criticality"]
        assert len(crit_opinions) == 0
        # But verificationStatus should still be present
        vs_opinions = [o for o in doc["opinions"] if o["field"] == "verificationStatus"]
        assert len(vs_opinions) == 1

    def test_clinical_status_captured(self):
        """clinicalStatus is preserved in the output document."""
        resource = _allergy_intolerance(clinical_status="active")
        doc, _ = from_fhir(resource)
        assert doc["clinicalStatus"] == "active"

    def test_extension_recovery_on_verification_status(self):
        """Opinion extension on verificationStatus overrides heuristic."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20, base_rate=0.50)
        )
        resource = _allergy_intolerance(extensions=[ext])
        doc, _ = from_fhir(resource)

        vs_op = [o for o in doc["opinions"] if o["field"] == "verificationStatus"][0]
        assert vs_op["source"] == "extension"
        assert abs(vs_op["opinion"].belief - 0.55) < 1e-9

    def test_report_nodes_converted(self):
        """ConversionReport counts converted nodes."""
        resource = _allergy_intolerance(criticality="high")
        _, report = from_fhir(resource)
        # At least 2: verificationStatus + criticality
        assert report.nodes_converted >= 2


# ── MedicationStatement tests ─────────────────────────────────────


class TestFromFhirMedicationStatement:
    """from_fhir() for FHIR R4 MedicationStatement resources."""

    def test_basic_active_statement(self):
        """MedicationStatement with status='active' → Opinion."""
        resource = _medication_statement(status="active")
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:MedicationStatement"
        assert len(doc["opinions"]) == 1

        entry = doc["opinions"][0]
        assert entry["field"] == "status"
        assert entry["value"] == "active"
        op = entry["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        # Active → high adherence confidence
        assert op.belief > 0.5

    def test_completed_has_high_belief(self):
        """status='completed' → high belief (medication course finished)."""
        resource = _medication_statement(status="completed")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief > 0.7

    def test_not_taken_has_low_belief(self):
        """status='not-taken' → very low belief (explicitly not taking)."""
        resource = _medication_statement(status="not-taken")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.15

    def test_stopped_has_low_belief(self):
        """status='stopped' → low belief (discontinued)."""
        resource = _medication_statement(status="stopped")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.belief < 0.25

    def test_intended_has_moderate_belief(self):
        """status='intended' → moderate belief with higher uncertainty."""
        resource = _medication_statement(status="intended")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        # Intended but not yet started — moderate
        assert 0.3 < op.belief < 0.8

    def test_on_hold_has_moderate_belief(self):
        """status='on-hold' → lower belief than active."""
        res_active = _medication_statement(status="active")
        res_hold = _medication_statement(status="on-hold")

        doc_active, _ = from_fhir(res_active)
        doc_hold, _ = from_fhir(res_hold)

        assert doc_hold["opinions"][0]["opinion"].belief < doc_active["opinions"][0]["opinion"].belief

    def test_unknown_has_high_uncertainty(self):
        """status='unknown' → high uncertainty (maximally uncertain adherence)."""
        resource = _medication_statement(status="unknown")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.3

    def test_entered_in_error_has_very_high_uncertainty(self):
        """status='entered-in-error' → very high uncertainty."""
        resource = _medication_statement(status="entered-in-error")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.5

    def test_information_source_affects_uncertainty(self):
        """informationSource from Practitioner → less uncertainty than Patient.

        Practitioner-reported medication use is generally more reliable
        than patient self-report (recall bias, health literacy, etc.).
        """
        res_patient = _medication_statement(
            information_source={"reference": "Patient/p1"}
        )
        res_practitioner = _medication_statement(
            information_source={"reference": "Practitioner/dr1"}
        )

        doc_patient, _ = from_fhir(res_patient)
        doc_practitioner, _ = from_fhir(res_practitioner)

        u_patient = doc_patient["opinions"][0]["opinion"].uncertainty
        u_practitioner = doc_practitioner["opinions"][0]["opinion"].uncertainty
        assert u_practitioner < u_patient

    def test_derived_from_reduces_uncertainty(self):
        """derivedFrom references (supporting evidence) reduce uncertainty."""
        res_no_ev = _medication_statement()
        res_ev = _medication_statement(
            derived_from=["MedicationRequest/rx1", "MedicationDispense/disp1",
                          "Observation/lab1"]
        )

        doc_no, _ = from_fhir(res_no_ev)
        doc_ev, _ = from_fhir(res_ev)

        u_no = doc_no["opinions"][0]["opinion"].uncertainty
        u_ev = doc_ev["opinions"][0]["opinion"].uncertainty
        assert u_ev < u_no

    def test_extension_recovery(self):
        """Opinion extension on status overrides heuristic."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.90, disbelief=0.02, uncertainty=0.08, base_rate=0.50)
        )
        resource = _medication_statement(extensions=[ext])
        doc, _ = from_fhir(resource)

        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.90) < 1e-9

    def test_medication_text_captured(self):
        """Medication text is preserved in the output document."""
        resource = _medication_statement(medication_text="Metformin 500mg")
        doc, _ = from_fhir(resource)
        assert doc.get("medication") == "Metformin 500mg"


# ── ClinicalImpression tests ──────────────────────────────────────


class TestFromFhirClinicalImpression:
    """from_fhir() for FHIR R4 ClinicalImpression resources."""

    def test_basic_completed_impression(self):
        """ClinicalImpression with status='completed' → Opinion."""
        resource = _clinical_impression(
            status="completed",
            summary="Patient presents with mild hypertension",
        )
        doc, report = from_fhir(resource)

        assert report.success is True
        assert doc["@type"] == "fhir:ClinicalImpression"
        assert len(doc["opinions"]) >= 1

        entry = doc["opinions"][0]
        op = entry["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_completed_lower_uncertainty_than_in_progress(self):
        """status='completed' → less uncertainty than 'in-progress'."""
        res_completed = _clinical_impression(status="completed", summary="Diagnosis confirmed")
        res_progress = _clinical_impression(status="in-progress", summary="Evaluating")

        doc_completed, _ = from_fhir(res_completed)
        doc_progress, _ = from_fhir(res_progress)

        u_completed = doc_completed["opinions"][0]["opinion"].uncertainty
        u_progress = doc_progress["opinions"][0]["opinion"].uncertainty
        assert u_completed < u_progress

    def test_entered_in_error_very_high_uncertainty(self):
        """status='entered-in-error' → very high uncertainty."""
        resource = _clinical_impression(status="entered-in-error", summary="Void")
        doc, _ = from_fhir(resource)

        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty > 0.5

    def test_findings_reduce_uncertainty(self):
        """More findings → lower uncertainty (more evidence)."""
        res_no_findings = _clinical_impression(
            summary="Assessment",
            finding=[],
        )
        res_findings = _clinical_impression(
            summary="Assessment",
            finding=[
                {"itemCodeableConcept": {"text": "Elevated BP"}},
                {"itemCodeableConcept": {"text": "Elevated cholesterol"}},
                {"itemReference": {"reference": "Observation/obs-1"}},
            ],
        )

        doc_no, _ = from_fhir(res_no_findings)
        doc_yes, _ = from_fhir(res_findings)

        u_no = doc_no["opinions"][0]["opinion"].uncertainty
        u_yes = doc_yes["opinions"][0]["opinion"].uncertainty
        assert u_yes < u_no

    def test_summary_preserved(self):
        """Summary text is preserved in the output document."""
        resource = _clinical_impression(
            summary="Likely Type 2 DM based on labs and history"
        )
        doc, _ = from_fhir(resource)
        assert doc.get("summary") == "Likely Type 2 DM based on labs and history"

    def test_no_summary_still_succeeds(self):
        """ClinicalImpression without summary → still produces output."""
        resource = _clinical_impression(status="completed")
        doc, report = from_fhir(resource)
        assert report.success is True
        # Status alone should produce an opinion about the assessment
        assert len(doc["opinions"]) >= 1

    def test_finding_references_captured(self):
        """Finding references are preserved for downstream fusion."""
        resource = _clinical_impression(
            finding=[
                {"itemReference": {"reference": "Condition/cond-1"}},
                {"itemReference": {"reference": "Observation/obs-1"}},
            ],
        )
        doc, _ = from_fhir(resource)
        refs = doc.get("finding_references", [])
        assert "Condition/cond-1" in refs
        assert "Observation/obs-1" in refs

    def test_extension_recovery_on_summary(self):
        """Opinion extension on summary overrides reconstruction."""
        ext = opinion_to_fhir_extension(
            Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15, base_rate=0.50)
        )
        resource = _clinical_impression(
            summary="Confirmed diagnosis",
            extensions=[ext],
        )
        doc, _ = from_fhir(resource)

        # Find the summary-sourced opinion
        summary_ops = [o for o in doc["opinions"] if o["field"] == "summary"]
        assert len(summary_ops) == 1
        assert summary_ops[0]["source"] == "extension"
        assert abs(summary_ops[0]["opinion"].belief - 0.80) < 1e-9


# ── Phase 2 cross-resource tests ──────────────────────────────────


class TestFromFhirPhase2General:
    """Cross-cutting from_fhir() tests for Phase 2 resources."""

    def test_phase2_types_in_supported_set(self):
        """All Phase 2 resource types are handled by from_fhir()."""
        for rt in ["AllergyIntolerance", "MedicationStatement", "ClinicalImpression"]:
            resource = {"resourceType": rt, "id": f"{rt.lower()}-1"}
            # Add required fields based on type
            if rt == "AllergyIntolerance":
                resource["verificationStatus"] = {"coding": [{"code": "confirmed"}]}
                resource["clinicalStatus"] = {"coding": [{"code": "active"}]}
            elif rt == "MedicationStatement":
                resource["status"] = "active"
                resource["medicationCodeableConcept"] = {"text": "Test"}
                resource["subject"] = {"reference": "Patient/p1"}
            elif rt == "ClinicalImpression":
                resource["status"] = "completed"
                resource["subject"] = {"reference": "Patient/p1"}
                resource["summary"] = "Test"

            doc, report = from_fhir(resource)
            assert report.success is True
            assert doc["@type"] == f"fhir:{rt}"

    def test_additivity_holds_for_phase2_resources(self):
        """b + d + u = 1 invariant holds across all Phase 2 resource types."""
        resources = [
            _allergy_intolerance(criticality="high"),
            _medication_statement(status="active"),
            _clinical_impression(status="completed", summary="Test"),
        ]
        for resource in resources:
            doc, _ = from_fhir(resource)
            for entry in doc["opinions"]:
                op = entry["opinion"]
                total = op.belief + op.disbelief + op.uncertainty
                assert abs(total - 1.0) < 1e-9, (
                    f"Additivity violated for {resource['resourceType']}: "
                    f"{op.belief} + {op.disbelief} + {op.uncertainty} = {total}"
                )

    def test_phase2_resources_fusable_with_phase1(self):
        """Phase 2 documents can be fused with Phase 1 documents."""
        # Phase 1 resource
        risk = _risk_assessment(probability=0.80, status="final")
        doc_risk, _ = from_fhir(risk)

        # Phase 2 resource
        impression = _clinical_impression(
            status="completed",
            summary="Elevated cardiovascular risk",
        )
        doc_impression, _ = from_fhir(impression)

        # Should fuse without error
        fused, report = fhir_clinical_fuse([doc_risk, doc_impression])
        assert isinstance(fused, Opinion)
        assert report.opinions_fused == 2
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# Phase 2, Step 2: to_fhir() for AllergyIntolerance,
#                  MedicationStatement, ClinicalImpression
#                  + full round-trip tests
# ═══════════════════════════════════════════════════════════════════


class TestToFhirAllergyIntolerance:
    """to_fhir() for AllergyIntolerance jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex AllergyIntolerance → valid FHIR AllergyIntolerance."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:AllergyIntolerance",
            "id": "allergy-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "AllergyIntolerance"
        assert resource["id"] == "allergy-1"

    def test_verification_status_preserved(self):
        """verificationStatus code is preserved from the opinion value."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:AllergyIntolerance",
            "id": "allergy-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        vs = resource.get("verificationStatus", {})
        codes = [c["code"] for c in vs.get("coding", [])]
        assert "confirmed" in codes

    def test_clinical_status_preserved(self):
        """clinicalStatus from doc is preserved in output resource."""
        op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        doc = {
            "@type": "fhir:AllergyIntolerance",
            "id": "allergy-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        cs = resource.get("clinicalStatus", {})
        codes = [c["code"] for c in cs.get("coding", [])]
        assert "active" in codes

    def test_extension_on_verification_status(self):
        """Opinion extension is attached to _verificationStatus."""
        op = Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20)
        doc = {
            "@type": "fhir:AllergyIntolerance",
            "id": "allergy-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "unconfirmed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_verificationStatus" in resource
        exts = resource["_verificationStatus"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1

    def test_criticality_opinion_exported(self):
        """Criticality opinion is exported with extension."""
        vs_op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10)
        crit_op = Opinion(belief=0.76, disbelief=0.09, uncertainty=0.15)
        doc = {
            "@type": "fhir:AllergyIntolerance",
            "id": "allergy-1",
            "clinicalStatus": "active",
            "opinions": [
                {
                    "field": "verificationStatus",
                    "value": "confirmed",
                    "opinion": vs_op,
                    "source": "reconstructed",
                },
                {
                    "field": "criticality",
                    "value": "high",
                    "opinion": crit_op,
                    "source": "reconstructed",
                },
            ],
        }
        resource, report = to_fhir(doc)

        assert resource.get("criticality") == "high"
        assert "_criticality" in resource
        exts = resource["_criticality"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1
        assert report.nodes_converted >= 2


class TestToFhirMedicationStatement:
    """to_fhir() for MedicationStatement jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex MedicationStatement → valid FHIR MedicationStatement."""
        op = Opinion(belief=0.72, disbelief=0.13, uncertainty=0.15)
        doc = {
            "@type": "fhir:MedicationStatement",
            "id": "medstmt-1",
            "status": "active",
            "medication": "Atorvastatin 20mg",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "MedicationStatement"
        assert resource["status"] == "active"

    def test_medication_text_preserved(self):
        """Medication text from doc is preserved in output."""
        op = Opinion(belief=0.72, disbelief=0.13, uncertainty=0.15)
        doc = {
            "@type": "fhir:MedicationStatement",
            "id": "medstmt-1",
            "status": "active",
            "medication": "Metformin 500mg",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        med_cc = resource.get("medicationCodeableConcept", {})
        assert med_cc.get("text") == "Metformin 500mg"

    def test_extension_on_status(self):
        """Opinion extension is attached to _status."""
        op = Opinion(belief=0.72, disbelief=0.13, uncertainty=0.15)
        doc = {
            "@type": "fhir:MedicationStatement",
            "id": "medstmt-1",
            "status": "active",
            "opinions": [{
                "field": "status",
                "value": "active",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_status" in resource
        exts = resource["_status"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1


class TestToFhirClinicalImpression:
    """to_fhir() for ClinicalImpression jsonld-ex documents."""

    def test_basic_export(self):
        """jsonld-ex ClinicalImpression → valid FHIR ClinicalImpression."""
        op = Opinion(belief=0.65, disbelief=0.10, uncertainty=0.25)
        doc = {
            "@type": "fhir:ClinicalImpression",
            "id": "impression-1",
            "status": "completed",
            "summary": "Mild hypertension",
            "finding_references": ["Observation/obs-1"],
            "opinions": [{
                "field": "summary",
                "value": "Mild hypertension",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, report = to_fhir(doc)

        assert report.success is True
        assert resource["resourceType"] == "ClinicalImpression"
        assert resource["status"] == "completed"
        assert resource["summary"] == "Mild hypertension"

    def test_finding_references_exported(self):
        """finding_references are converted to FHIR finding array."""
        op = Opinion(belief=0.65, disbelief=0.10, uncertainty=0.25)
        doc = {
            "@type": "fhir:ClinicalImpression",
            "id": "impression-1",
            "status": "completed",
            "finding_references": ["Condition/cond-1", "Observation/obs-1"],
            "opinions": [{
                "field": "status",
                "value": "completed",
                "opinion": op,
                "source": "reconstructed",
            }],
        }
        resource, _ = to_fhir(doc)

        findings = resource.get("finding", [])
        assert len(findings) == 2
        refs = []
        for f in findings:
            ref_obj = f.get("itemReference", {})
            refs.append(ref_obj.get("reference"))
        assert "Condition/cond-1" in refs
        assert "Observation/obs-1" in refs

    def test_extension_on_summary(self):
        """Opinion on summary is attached as _summary extension."""
        op = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15)
        doc = {
            "@type": "fhir:ClinicalImpression",
            "id": "impression-1",
            "status": "completed",
            "summary": "Confirmed diagnosis",
            "opinions": [{
                "field": "summary",
                "value": "Confirmed diagnosis",
                "opinion": op,
                "source": "extension",
            }],
        }
        resource, _ = to_fhir(doc)

        assert "_summary" in resource
        exts = resource["_summary"]["extension"]
        our_ext = [e for e in exts if e.get("url") == FHIR_EXTENSION_URL]
        assert len(our_ext) == 1


# ── Phase 2 full round-trip: FHIR → jsonld-ex → FHIR ─────────────


class TestFhirPhase2RoundTrip:
    """FHIR resource → from_fhir() → to_fhir() round-trip for Phase 2."""

    def test_allergy_round_trip_verification_status(self):
        """AllergyIntolerance: verification status survives round-trip."""
        original = _allergy_intolerance(
            verification_status="unconfirmed",
            clinical_status="active",
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        vs_codes = [
            c["code"]
            for c in recovered.get("verificationStatus", {}).get("coding", [])
        ]
        assert "unconfirmed" in vs_codes

        cs_codes = [
            c["code"]
            for c in recovered.get("clinicalStatus", {}).get("coding", [])
        ]
        assert "active" in cs_codes

    def test_allergy_round_trip_extension_exact(self):
        """AllergyIntolerance with extension: exact opinion preserved."""
        op = Opinion(belief=0.55, disbelief=0.25, uncertainty=0.20, base_rate=0.45)
        ext = opinion_to_fhir_extension(op)
        original = _allergy_intolerance(extensions=[ext])

        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        # Recover the extension from output
        vs_ext = recovered["_verificationStatus"]["extension"]
        our_ext = [e for e in vs_ext if e.get("url") == FHIR_EXTENSION_URL][0]
        recovered_op = fhir_extension_to_opinion(our_ext)

        assert abs(recovered_op.belief - op.belief) < 1e-9
        assert abs(recovered_op.disbelief - op.disbelief) < 1e-9
        assert abs(recovered_op.uncertainty - op.uncertainty) < 1e-9
        assert abs(recovered_op.base_rate - op.base_rate) < 1e-9

    def test_allergy_round_trip_criticality(self):
        """AllergyIntolerance: criticality survives round-trip."""
        original = _allergy_intolerance(
            verification_status="confirmed",
            criticality="high",
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        assert recovered.get("criticality") == "high"

    def test_medication_statement_round_trip_status(self):
        """MedicationStatement: status survives round-trip."""
        original = _medication_statement(status="on-hold")
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        assert recovered["status"] == "on-hold"

    def test_medication_statement_round_trip_medication(self):
        """MedicationStatement: medication text survives round-trip."""
        original = _medication_statement(
            status="active",
            medication_text="Lisinopril 10mg",
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        med_cc = recovered.get("medicationCodeableConcept", {})
        assert med_cc.get("text") == "Lisinopril 10mg"

    def test_clinical_impression_round_trip_summary(self):
        """ClinicalImpression: summary + references survive round-trip."""
        original = _clinical_impression(
            status="completed",
            summary="Elevated cardiovascular risk",
            finding=[
                {"itemReference": {"reference": "Observation/obs-1"}},
                {"itemReference": {"reference": "Condition/cond-1"}},
            ],
        )
        doc, _ = from_fhir(original)
        recovered, _ = to_fhir(doc)

        assert recovered["summary"] == "Elevated cardiovascular risk"
        findings = recovered.get("finding", [])
        refs = []
        for f in findings:
            ref_obj = f.get("itemReference", {})
            refs.append(ref_obj.get("reference"))
        assert "Observation/obs-1" in refs
        assert "Condition/cond-1" in refs
