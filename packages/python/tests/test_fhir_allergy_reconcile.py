"""
Tests for fhir_allergy_reconcile() — multi-EHR allergy list reconciliation.

TDD Red Phase: Defines expected behavior for reconciling allergy lists
from multiple EHR systems using Subjective Logic fusion.

Clinical scenario: A patient transfers between hospitals.  Each hospital
has its own AllergyIntolerance list with potentially overlapping,
conflicting, or unique entries.  Reconciliation matches allergies by
substance code, fuses opinions for matched entries, and preserves
unique entries — producing a single reconciled allergy list.

Matching key hierarchy:
  1. code.coding: same system + code (e.g., SNOMED 91936005)
  2. code.text: case-insensitive string match (fallback)
  3. No match → preserved as unique entry
"""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    pairwise_conflict,
)
from jsonld_ex.fhir_interop._reconciliation import (
    fhir_allergy_reconcile,
    ReconciliationReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allergy_doc(
    doc_id: str,
    *,
    code_text: str = "Penicillin",
    code_coding: list[dict] | None = None,
    vs_code: str = "confirmed",
    vs_opinion: Opinion | None = None,
    criticality_code: str | None = None,
    criticality_opinion: Opinion | None = None,
    clinical_status: str = "active",
    patient: str | None = "Patient/pat-1",
    category: list[str] | None = None,
    allergy_type: str | None = None,
) -> dict:
    """Build a minimal jsonld-ex AllergyIntolerance doc (as from from_fhir)."""
    if vs_opinion is None:
        vs_opinion = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)

    opinions = [
        {
            "field": "verificationStatus",
            "value": vs_code,
            "opinion": vs_opinion,
            "source": "reconstructed",
        }
    ]

    if criticality_code is not None:
        if criticality_opinion is None:
            criticality_opinion = Opinion(
                belief=0.70, disbelief=0.10, uncertainty=0.20, base_rate=0.5
            )
        opinions.append({
            "field": "criticality",
            "value": criticality_code,
            "opinion": criticality_opinion,
            "source": "reconstructed",
        })

    doc: dict = {
        "@type": "fhir:AllergyIntolerance",
        "id": doc_id,
        "clinicalStatus": clinical_status,
        "opinions": opinions,
    }

    code: dict = {}
    if code_text is not None:
        code["text"] = code_text
    if code_coding is not None:
        code["coding"] = code_coding
    if code:
        doc["code"] = code

    if patient is not None:
        doc["patient"] = patient
    if category is not None:
        doc["category"] = category
    if allergy_type is not None:
        doc["type"] = allergy_type

    return doc


# Fixture opinions with distinct values for testing
HIGH_CONF = Opinion(belief=0.90, disbelief=0.03, uncertainty=0.07, base_rate=0.5)
MOD_CONF = Opinion(belief=0.65, disbelief=0.10, uncertainty=0.25, base_rate=0.5)
LOW_CONF = Opinion(belief=0.30, disbelief=0.40, uncertainty=0.30, base_rate=0.5)
REFUTED_OP = Opinion(belief=0.05, disbelief=0.80, uncertainty=0.15, base_rate=0.5)

SNOMED_PENICILLIN = [
    {"system": "http://snomed.info/sct", "code": "91936005", "display": "Allergy to penicillin"}
]
SNOMED_LATEX = [
    {"system": "http://snomed.info/sct", "code": "300916003", "display": "Latex allergy"}
]
SNOMED_PEANUT = [
    {"system": "http://snomed.info/sct", "code": "91935009", "display": "Allergy to peanut"}
]


# ===================================================================
# 1. BASIC FUNCTIONALITY
# ===================================================================


class TestBasicReconciliation:
    """Core reconciliation behavior with simple inputs."""

    def test_returns_list_and_report(self):
        """Reconciliation returns a list of docs and a report."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin")]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert isinstance(result, list)
        assert isinstance(report, ReconciliationReport)

    def test_single_list_passthrough(self):
        """A single list with no other lists returns entries unchanged."""
        list_a = [
            _allergy_doc("a-1", code_text="Penicillin"),
            _allergy_doc("a-2", code_text="Latex"),
        ]

        result, report = fhir_allergy_reconcile([list_a])

        assert len(result) == 2
        assert report.input_list_count == 1
        assert report.matched_groups == 0
        assert report.unique_entries == 2

    def test_two_lists_same_substance_fuses(self):
        """Same substance in two lists → single fused entry."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=MOD_CONF)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 1
        assert report.matched_groups == 1
        assert report.unique_entries == 0
        assert report.output_count == 1

    def test_two_lists_different_substances_both_preserved(self):
        """Different substances → both preserved as unique entries."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Latex")]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 2
        assert report.matched_groups == 0
        assert report.unique_entries == 2

    def test_empty_lists_raises(self):
        """No lists at all → ValueError."""
        with pytest.raises(ValueError, match="[Ee]mpty|at least one"):
            fhir_allergy_reconcile([])

    def test_all_empty_sublists(self):
        """Multiple empty lists → empty result with clean report."""
        result, report = fhir_allergy_reconcile([[], []])

        assert result == []
        assert report.total_input_entries == 0
        assert report.output_count == 0


# ===================================================================
# 2. MATCHING LOGIC
# ===================================================================


class TestMatchingLogic:
    """Verify the matching key hierarchy: coding > text."""

    def test_match_by_coding_system_and_code(self):
        """Same system + code in coding → match, even if text differs."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin allergy",
                               code_coding=SNOMED_PENICILLIN, vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Allergy to penicillin",
                               code_coding=SNOMED_PENICILLIN, vs_opinion=MOD_CONF)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 1
        assert report.matched_groups == 1

    def test_different_coding_no_match(self):
        """Different SNOMED codes → no match, both preserved."""
        list_a = [_allergy_doc("a-1", code_coding=SNOMED_PENICILLIN)]
        list_b = [_allergy_doc("b-1", code_coding=SNOMED_LATEX)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 2
        assert report.matched_groups == 0

    def test_match_by_text_case_insensitive(self):
        """Text-only match is case-insensitive."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="penicillin", vs_opinion=MOD_CONF)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 1
        assert report.matched_groups == 1

    def test_text_match_fallback_when_no_coding(self):
        """When neither has coding, fall back to text match."""
        list_a = [_allergy_doc("a-1", code_text="Peanuts", code_coding=None)]
        list_b = [_allergy_doc("b-1", code_text="Peanuts", code_coding=None)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 1
        assert report.matched_groups == 1

    def test_coding_match_takes_precedence_over_text_mismatch(self):
        """If codings match, text difference doesn't prevent matching."""
        list_a = [_allergy_doc("a-1", code_text="Pen-V allergy",
                               code_coding=SNOMED_PENICILLIN)]
        list_b = [_allergy_doc("b-1", code_text="PCN hypersensitivity",
                               code_coding=SNOMED_PENICILLIN)]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 1
        assert report.matched_groups == 1

    def test_no_code_field_treated_as_unique(self):
        """Entries with no code field cannot be matched — always unique."""
        doc_no_code: dict = {
            "@type": "fhir:AllergyIntolerance",
            "id": "no-code-1",
            "clinicalStatus": "active",
            "opinions": [{
                "field": "verificationStatus",
                "value": "confirmed",
                "opinion": HIGH_CONF,
                "source": "reconstructed",
            }],
        }
        list_a = [doc_no_code]
        list_b = [_allergy_doc("b-1", code_text="Penicillin")]

        result, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(result) == 2
        assert report.unique_entries == 2


# ===================================================================
# 3. FUSION CORRECTNESS
# ===================================================================


class TestFusionCorrectness:
    """Verify fused opinions match manual Subjective Logic operations."""

    def test_fused_verification_status_matches_manual_cumulative(self):
        """Fused verificationStatus opinion == manual cumulative_fuse."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=MOD_CONF)]

        result, _ = fhir_allergy_reconcile([list_a, list_b], method="cumulative")

        vs_ops = [o for o in result[0]["opinions"] if o["field"] == "verificationStatus"]
        assert len(vs_ops) == 1
        fused = vs_ops[0]["opinion"]

        expected = cumulative_fuse(HIGH_CONF, MOD_CONF)
        assert fused.belief == pytest.approx(expected.belief, abs=1e-12)
        assert fused.disbelief == pytest.approx(expected.disbelief, abs=1e-12)
        assert fused.uncertainty == pytest.approx(expected.uncertainty, abs=1e-12)

    def test_fused_verification_status_averaging(self):
        """Averaging fusion produces correct result."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=MOD_CONF)]

        result, _ = fhir_allergy_reconcile([list_a, list_b], method="averaging")

        vs_ops = [o for o in result[0]["opinions"] if o["field"] == "verificationStatus"]
        fused = vs_ops[0]["opinion"]

        expected = averaging_fuse(HIGH_CONF, MOD_CONF)
        assert fused.belief == pytest.approx(expected.belief, abs=1e-12)

    def test_fused_criticality_independent_from_verification(self):
        """Both opinions fuse independently when matched."""
        crit_a = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15, base_rate=0.5)
        crit_b = Opinion(belief=0.60, disbelief=0.10, uncertainty=0.30, base_rate=0.5)

        list_a = [_allergy_doc("a-1", code_text="Penicillin",
                               vs_opinion=HIGH_CONF,
                               criticality_code="high",
                               criticality_opinion=crit_a)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin",
                               vs_opinion=MOD_CONF,
                               criticality_code="high",
                               criticality_opinion=crit_b)]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        crit_ops = [o for o in result[0]["opinions"] if o["field"] == "criticality"]
        assert len(crit_ops) == 1
        fused_crit = crit_ops[0]["opinion"]

        expected_crit = cumulative_fuse(crit_a, crit_b)
        assert fused_crit.belief == pytest.approx(expected_crit.belief, abs=1e-12)

    def test_fused_source_is_fused(self):
        """Fused opinions have source='fused'."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin")]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        vs_ops = [o for o in result[0]["opinions"] if o["field"] == "verificationStatus"]
        assert vs_ops[0]["source"] == "fused"

    def test_three_lists_triple_fusion(self):
        """Three lists with same substance → triple fusion."""
        op_c = Opinion(belief=0.50, disbelief=0.20, uncertainty=0.30, base_rate=0.5)
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=MOD_CONF)]
        list_c = [_allergy_doc("c-1", code_text="Penicillin", vs_opinion=op_c)]

        result, report = fhir_allergy_reconcile([list_a, list_b, list_c])

        assert len(result) == 1
        assert report.matched_groups == 1
        assert report.input_list_count == 3

    def test_method_parameter_respected(self):
        """Different methods produce different results."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=LOW_CONF)]

        result_cum, _ = fhir_allergy_reconcile([list_a, list_b], method="cumulative")
        result_avg, _ = fhir_allergy_reconcile([list_a, list_b], method="averaging")

        vs_cum = [o for o in result_cum[0]["opinions"] if o["field"] == "verificationStatus"][0]
        vs_avg = [o for o in result_avg[0]["opinions"] if o["field"] == "verificationStatus"][0]

        # Cumulative and averaging produce different results
        assert vs_cum["opinion"].belief != pytest.approx(vs_avg["opinion"].belief, abs=1e-6)

    def test_invalid_method_raises(self):
        """Unsupported method → ValueError."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]

        with pytest.raises(ValueError, match="[Uu]nsupported.*method"):
            fhir_allergy_reconcile([list_a], method="invalid")


# ===================================================================
# 4. CONFLICT DETECTION
# ===================================================================


class TestConflictDetection:
    """Report includes pairwise conflict scores for all matched groups."""

    def test_agreeing_sources_low_conflict(self):
        """Two sources that agree → low conflict score."""
        op_a = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)
        op_b = Opinion(belief=0.80, disbelief=0.07, uncertainty=0.13, base_rate=0.5)

        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=op_a)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=op_b)]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(report.conflicts) == 1
        conflict_entry = report.conflicts[0]
        assert "scores" in conflict_entry
        # Agreeing sources → low score
        assert all(s < 0.5 for s in conflict_entry["scores"])

    def test_disagreeing_sources_high_conflict(self):
        """Confirmed vs refuted → high conflict score."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=REFUTED_OP)]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(report.conflicts) == 1
        # Disagreeing sources → higher conflict than agreeing
        disagree_max = max(report.conflicts[0]["scores"])

        # Compare against agreeing case
        op_agree = Opinion(belief=0.80, disbelief=0.07, uncertainty=0.13, base_rate=0.5)
        list_c = [_allergy_doc("c-1", code_text="Penicillin", vs_opinion=op_agree)]
        _, report_agree = fhir_allergy_reconcile([list_a, list_c])
        agree_max = max(report_agree.conflicts[0]["scores"])

        assert disagree_max > agree_max

    def test_conflict_entry_identifies_substance(self):
        """Each conflict entry identifies which substance group it belongs to."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin", vs_opinion=HIGH_CONF)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", vs_opinion=REFUTED_OP)]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert report.conflicts[0].get("match_key") is not None

    def test_no_conflict_for_unique_entries(self):
        """Unique entries (no match) produce no conflict entries."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Latex")]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert len(report.conflicts) == 0


# ===================================================================
# 5. OUTPUT FORMAT
# ===================================================================


class TestOutputFormat:
    """Reconciled entries are valid jsonld-ex docs."""

    def test_fused_entry_has_type(self):
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin")]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        assert result[0]["@type"] == "fhir:AllergyIntolerance"

    def test_fused_entry_preserves_code(self):
        list_a = [_allergy_doc("a-1", code_text="Penicillin",
                               code_coding=SNOMED_PENICILLIN)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin",
                               code_coding=SNOMED_PENICILLIN)]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        assert result[0]["code"]["text"] == "Penicillin"
        assert result[0]["code"]["coding"][0]["code"] == "91936005"

    def test_unique_entry_preserved_verbatim(self):
        """Unique entries pass through without modification."""
        original = _allergy_doc("a-1", code_text="Penicillin",
                                category=["medication"], allergy_type="allergy")
        list_a = [original]

        result, _ = fhir_allergy_reconcile([list_a])

        assert result[0]["id"] == "a-1"
        assert result[0].get("category") == ["medication"]
        assert result[0].get("type") == "allergy"

    def test_fused_entry_preserves_clinical_status(self):
        list_a = [_allergy_doc("a-1", code_text="Penicillin", clinical_status="active")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", clinical_status="active")]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        assert result[0].get("clinicalStatus") is not None

    def test_fused_entry_preserves_patient(self):
        list_a = [_allergy_doc("a-1", code_text="Penicillin", patient="Patient/pat-1")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin", patient="Patient/pat-1")]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        assert result[0].get("patient") == "Patient/pat-1"


# ===================================================================
# 6. REPORT COMPLETENESS
# ===================================================================


class TestReportCompleteness:
    """ReconciliationReport captures full audit trail."""

    def test_report_counts_accurate(self):
        """All report counts add up correctly."""
        list_a = [
            _allergy_doc("a-1", code_text="Penicillin"),
            _allergy_doc("a-2", code_text="Latex"),
        ]
        list_b = [
            _allergy_doc("b-1", code_text="Penicillin"),
            _allergy_doc("b-2", code_text="Peanuts"),
        ]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert report.input_list_count == 2
        assert report.total_input_entries == 4
        assert report.matched_groups == 1       # Penicillin matched
        assert report.unique_entries == 2       # Latex + Peanuts
        assert report.output_count == 3         # 1 fused + 2 unique

    def test_report_warnings_empty_for_clean_input(self):
        list_a = [_allergy_doc("a-1", code_text="Penicillin")]
        list_b = [_allergy_doc("b-1", code_text="Penicillin")]

        _, report = fhir_allergy_reconcile([list_a, list_b])

        assert report.warnings == []


# ===================================================================
# 7. MULTI-OPINION FUSION (verificationStatus + criticality)
# ===================================================================


class TestMultiOpinionFusion:
    """Both opinion fields fuse independently when matched."""

    def test_only_verification_status_when_one_has_criticality(self):
        """If only one entry has criticality, it's preserved (not fused)."""
        list_a = [_allergy_doc("a-1", code_text="Penicillin",
                               vs_opinion=HIGH_CONF,
                               criticality_code="high",
                               criticality_opinion=Opinion(
                                   belief=0.80, disbelief=0.05,
                                   uncertainty=0.15, base_rate=0.5))]
        list_b = [_allergy_doc("b-1", code_text="Penicillin",
                               vs_opinion=MOD_CONF)]  # no criticality

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        # verificationStatus should be fused (both have it)
        vs_ops = [o for o in result[0]["opinions"] if o["field"] == "verificationStatus"]
        assert len(vs_ops) == 1
        assert vs_ops[0]["source"] == "fused"

        # criticality should be preserved from the one that has it
        crit_ops = [o for o in result[0]["opinions"] if o["field"] == "criticality"]
        assert len(crit_ops) == 1

    def test_both_opinions_fused_when_both_present(self):
        """Both verificationStatus and criticality fuse when all entries have them."""
        crit_a = Opinion(belief=0.80, disbelief=0.05, uncertainty=0.15, base_rate=0.5)
        crit_b = Opinion(belief=0.55, disbelief=0.15, uncertainty=0.30, base_rate=0.5)

        list_a = [_allergy_doc("a-1", code_text="Penicillin",
                               vs_opinion=HIGH_CONF,
                               criticality_code="high",
                               criticality_opinion=crit_a)]
        list_b = [_allergy_doc("b-1", code_text="Penicillin",
                               vs_opinion=MOD_CONF,
                               criticality_code="high",
                               criticality_opinion=crit_b)]

        result, _ = fhir_allergy_reconcile([list_a, list_b])

        vs_ops = [o for o in result[0]["opinions"] if o["field"] == "verificationStatus"]
        crit_ops = [o for o in result[0]["opinions"] if o["field"] == "criticality"]

        assert len(vs_ops) == 1
        assert len(crit_ops) == 1
        assert vs_ops[0]["source"] == "fused"
        assert crit_ops[0]["source"] == "fused"


# ===================================================================
# 8. COMPLEX SCENARIO — REALISTIC HOSPITAL TRANSFER
# ===================================================================


class TestRealisticScenario:
    """End-to-end test mimicking a real hospital transfer."""

    def test_three_hospital_transfer(self):
        """Patient with records at 3 hospitals.

        Hospital A: Penicillin (confirmed, high), Latex (unconfirmed)
        Hospital B: Penicillin (confirmed, high), Peanuts (confirmed)
        Hospital C: Penicillin (unconfirmed, low), Latex (confirmed)

        Expected:
          - Penicillin: 3-way fusion (matched across all 3)
          - Latex: 2-way fusion (A + C)
          - Peanuts: unique (only B)
          Total output: 3 entries
        """
        pen_a = Opinion(belief=0.90, disbelief=0.03, uncertainty=0.07, base_rate=0.5)
        pen_b = Opinion(belief=0.88, disbelief=0.04, uncertainty=0.08, base_rate=0.5)
        pen_c = Opinion(belief=0.50, disbelief=0.15, uncertainty=0.35, base_rate=0.5)

        latex_a = Opinion(belief=0.45, disbelief=0.15, uncertainty=0.40, base_rate=0.5)
        latex_c = Opinion(belief=0.82, disbelief=0.06, uncertainty=0.12, base_rate=0.5)

        peanut_b = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)

        list_a = [
            _allergy_doc("a-pen", code_text="Penicillin",
                         code_coding=SNOMED_PENICILLIN, vs_opinion=pen_a,
                         criticality_code="high"),
            _allergy_doc("a-lat", code_text="Latex",
                         code_coding=SNOMED_LATEX, vs_opinion=latex_a),
        ]
        list_b = [
            _allergy_doc("b-pen", code_text="Penicillin",
                         code_coding=SNOMED_PENICILLIN, vs_opinion=pen_b,
                         criticality_code="high"),
            _allergy_doc("b-pea", code_text="Peanuts",
                         code_coding=SNOMED_PEANUT, vs_opinion=peanut_b),
        ]
        list_c = [
            _allergy_doc("c-pen", code_text="Penicillin",
                         code_coding=SNOMED_PENICILLIN, vs_opinion=pen_c,
                         criticality_code="low"),
            _allergy_doc("c-lat", code_text="Latex",
                         code_coding=SNOMED_LATEX, vs_opinion=latex_c),
        ]

        result, report = fhir_allergy_reconcile([list_a, list_b, list_c])

        assert report.input_list_count == 3
        assert report.total_input_entries == 6
        assert report.output_count == 3
        assert report.matched_groups == 2  # Penicillin (3-way) + Latex (2-way)
        assert report.unique_entries == 1  # Peanuts

        # Verify all three substances present
        substances = set()
        for entry in result:
            code = entry.get("code", {})
            text = code.get("text", "")
            substances.add(text.lower())
        assert "penicillin" in substances
        assert "latex" in substances
        assert "peanuts" in substances
