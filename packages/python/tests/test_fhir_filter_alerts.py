"""
Tests for fhir_filter_alerts() — confidence-based alert fatigue reduction.

TDD Red Phase: Defines expected behavior for filtering DetectedIssue
alerts using Subjective Logic projected probability and uncertainty.

Clinical motivation: Drug-drug interaction alerts fire constantly with
>90% override rates.  SL opinions let us suppress low-confidence alerts
while preserving high-confidence ones — but filtering on projected
probability alone collapses the opinion to a scalar, undermining SL's
multi-dimensional advantage.

The ``uncertainty_ceiling`` parameter restores that advantage: alerts
with uncertainty above the ceiling are always kept regardless of
projected probability, because the evidence base is too thin to
justify suppression.  This is both epistemically sound and clinically
safe (fail-safe: never suppress what you cannot assess).

Design principles:
  - No data is lost: suppressed alerts are returned separately, not deleted
  - Fail-safe: docs with no opinions are KEPT, not suppressed
  - ``threshold`` alone is a pragmatic simplification
  - ``threshold`` + ``uncertainty_ceiling`` is the recommended configuration
"""

from __future__ import annotations

import pytest
from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop._alert_filter import (
    fhir_filter_alerts,
    AlertFilterReport,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _alert_doc(
    doc_id: str,
    opinion: Opinion,
    *,
    field: str = "severity",
    value: str = "moderate",
    status: str = "final",
    implicated: list[str] | None = None,
) -> dict:
    """Build a minimal jsonld-ex DetectedIssue doc."""
    doc: dict = {
        "@type": "fhir:DetectedIssue",
        "id": doc_id,
        "status": status,
        "opinions": [
            {
                "field": field,
                "value": value,
                "opinion": opinion,
                "source": "reconstructed",
            }
        ],
    }
    if implicated:
        doc["implicated_references"] = implicated
    return doc


# ---------------------------------------------------------------------------
# Fixture opinions — designed to test specific filtering scenarios
# ---------------------------------------------------------------------------

# High confidence, well-evidenced alert → should be kept
HIGH_CONF = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)
# P = 0.85 + 0.5*0.10 = 0.90

# Moderate confidence → kept at default threshold 0.5
MOD_CONF = Opinion(belief=0.55, disbelief=0.15, uncertainty=0.30, base_rate=0.5)
# P = 0.55 + 0.5*0.30 = 0.70

# Low belief, low uncertainty — genuine low risk → suppress
LOW_RISK = Opinion(belief=0.15, disbelief=0.70, uncertainty=0.15, base_rate=0.5)
# P = 0.15 + 0.5*0.15 = 0.225

# High uncertainty, low belief — barely any evidence → base rate carries it
HIGH_UNC = Opinion(belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.5)
# P = 0.10 + 0.5*0.85 = 0.525  (above 0.5 threshold due to base rate!)

# Borderline: projected probability exactly near threshold
BORDERLINE_ABOVE = Opinion(belief=0.40, disbelief=0.10, uncertainty=0.50, base_rate=0.5)
# P = 0.40 + 0.5*0.50 = 0.65

BORDERLINE_BELOW = Opinion(belief=0.20, disbelief=0.30, uncertainty=0.50, base_rate=0.5)
# P = 0.20 + 0.5*0.50 = 0.45


# ===================================================================
# 1. BASIC FUNCTIONALITY
# ===================================================================


class TestBasicFiltering:
    """Core filter behavior with default parameters."""

    def test_returns_kept_suppressed_and_report(self):
        """Function returns three values: kept, suppressed, report."""
        docs = [_alert_doc("d-1", HIGH_CONF)]
        kept, suppressed, report = fhir_filter_alerts(docs)

        assert isinstance(kept, list)
        assert isinstance(suppressed, list)
        assert isinstance(report, AlertFilterReport)

    def test_high_confidence_alert_kept(self):
        """Alert with P=0.90 is kept at default threshold 0.5."""
        docs = [_alert_doc("d-1", HIGH_CONF)]
        kept, suppressed, _ = fhir_filter_alerts(docs)

        assert len(kept) == 1
        assert len(suppressed) == 0
        assert kept[0]["id"] == "d-1"

    def test_low_risk_alert_suppressed(self):
        """Alert with P=0.225 is suppressed at default threshold 0.5."""
        docs = [_alert_doc("d-1", LOW_RISK)]
        kept, suppressed, _ = fhir_filter_alerts(docs)

        assert len(kept) == 0
        assert len(suppressed) == 1
        assert suppressed[0]["id"] == "d-1"

    def test_mixed_alerts_partitioned(self):
        """Multiple alerts are correctly partitioned."""
        docs = [
            _alert_doc("high", HIGH_CONF),
            _alert_doc("low", LOW_RISK),
            _alert_doc("mod", MOD_CONF),
        ]
        kept, suppressed, report = fhir_filter_alerts(docs)

        assert report.total_input == 3
        assert report.kept_count == 2      # HIGH_CONF + MOD_CONF
        assert report.suppressed_count == 1  # LOW_RISK
        kept_ids = {d["id"] for d in kept}
        assert "high" in kept_ids
        assert "mod" in kept_ids
        assert suppressed[0]["id"] == "low"

    def test_empty_input(self):
        """Empty input → empty results, clean report."""
        kept, suppressed, report = fhir_filter_alerts([])

        assert kept == []
        assert suppressed == []
        assert report.total_input == 0
        assert report.kept_count == 0
        assert report.suppressed_count == 0


# ===================================================================
# 2. THRESHOLD PARAMETER
# ===================================================================


class TestThresholdParameter:
    """Verify threshold controls the cutoff correctly."""

    def test_threshold_zero_keeps_everything(self):
        """threshold=0.0 → all alerts kept (projected prob is always >= 0)."""
        docs = [
            _alert_doc("high", HIGH_CONF),
            _alert_doc("low", LOW_RISK),
        ]
        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=0.0)

        assert len(kept) == 2
        assert len(suppressed) == 0

    def test_threshold_one_suppresses_almost_everything(self):
        """threshold=1.0 → only alerts with P >= 1.0 kept."""
        docs = [
            _alert_doc("high", HIGH_CONF),  # P = 0.90 < 1.0
            _alert_doc("low", LOW_RISK),
        ]
        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=1.0)

        assert len(kept) == 0
        assert len(suppressed) == 2

    def test_custom_threshold(self):
        """Custom threshold partitions correctly."""
        docs = [
            _alert_doc("high", HIGH_CONF),      # P = 0.90
            _alert_doc("mod", MOD_CONF),         # P = 0.70
            _alert_doc("border", BORDERLINE_ABOVE),  # P = 0.65
            _alert_doc("low", LOW_RISK),         # P = 0.225
        ]
        # threshold = 0.7 → only HIGH_CONF and MOD_CONF kept
        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=0.7)

        kept_ids = {d["id"] for d in kept}
        assert "high" in kept_ids
        assert "mod" in kept_ids
        assert len(suppressed) == 2

    def test_threshold_boundary_gte(self):
        """Alerts with projected prob exactly at threshold are KEPT (>=)."""
        # Create an opinion where P = exactly 0.5
        exact_half = Opinion(belief=0.25, disbelief=0.25, uncertainty=0.50, base_rate=0.5)
        # P = 0.25 + 0.5*0.50 = 0.50
        docs = [_alert_doc("exact", exact_half)]

        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=0.5)

        assert len(kept) == 1
        assert len(suppressed) == 0


# ===================================================================
# 3. UNCERTAINTY CEILING — THE SL ADVANTAGE
# ===================================================================


class TestUncertaintyCeiling:
    """The uncertainty_ceiling restores SL's multi-dimensional advantage.

    Without it, filtering collapses the opinion to a scalar (projected
    probability), which is exactly the "scalar confidence trap" that
    jsonld-ex's core thesis criticizes.  The uncertainty_ceiling ensures
    that alerts with thin evidence bases are never suppressed, regardless
    of where the base rate pushes the projected probability.
    """

    def test_high_uncertainty_kept_despite_low_projected(self):
        """Alert with u=0.85 is kept even though threshold alone would
        pass it — but more importantly, this tests the case where
        threshold alone would SUPPRESS a high-uncertainty alert.

        Use a base_rate that makes P < threshold for high uncertainty.
        """
        # b=0.05, d=0.05, u=0.90, a=0.3 → P = 0.05 + 0.3*0.90 = 0.32
        thin_evidence = Opinion(belief=0.05, disbelief=0.05, uncertainty=0.90, base_rate=0.3)
        docs = [_alert_doc("thin", thin_evidence)]

        # Without ceiling: P=0.32 < 0.5 → suppressed
        kept_no_ceil, supp_no_ceil, _ = fhir_filter_alerts(docs, threshold=0.5)
        assert len(supp_no_ceil) == 1

        # With ceiling: u=0.90 >= 0.7 → kept regardless
        kept_ceil, supp_ceil, _ = fhir_filter_alerts(
            docs, threshold=0.5, uncertainty_ceiling=0.7
        )
        assert len(kept_ceil) == 1
        assert len(supp_ceil) == 0

    def test_low_uncertainty_not_affected_by_ceiling(self):
        """Well-evidenced low-risk alert (u=0.15) is still suppressed
        even with uncertainty_ceiling set.
        """
        docs = [_alert_doc("low", LOW_RISK)]  # u=0.15, P=0.225

        kept, suppressed, _ = fhir_filter_alerts(
            docs, threshold=0.5, uncertainty_ceiling=0.7
        )

        assert len(kept) == 0
        assert len(suppressed) == 1

    def test_ceiling_none_disables_feature(self):
        """uncertainty_ceiling=None (default) → no ceiling applied."""
        # HIGH_UNC: u=0.85, P=0.525 — above threshold 0.5, so kept
        docs = [_alert_doc("unc", HIGH_UNC)]

        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=0.5)

        assert len(kept) == 1  # P=0.525 >= 0.5

    def test_ceiling_zero_keeps_everything(self):
        """uncertainty_ceiling=0.0 → all alerts have u >= 0, so all kept."""
        docs = [
            _alert_doc("high", HIGH_CONF),
            _alert_doc("low", LOW_RISK),
        ]
        kept, suppressed, _ = fhir_filter_alerts(
            docs, threshold=0.5, uncertainty_ceiling=0.0
        )

        assert len(kept) == 2
        assert len(suppressed) == 0

    def test_scalar_trap_demonstration(self):
        """Demonstrate the scalar confidence trap that uncertainty_ceiling solves.

        Two alerts with identical projected probability but very different
        evidence quality:
          - Well-evidenced: b=0.40, d=0.35, u=0.25, a=0.5 → P=0.525
          - Thin evidence:  b=0.10, d=0.05, u=0.85, a=0.5 → P=0.525

        Pure threshold filtering treats them identically.
        With uncertainty_ceiling, the thin-evidence alert is flagged.
        """
        well_evidenced = Opinion(belief=0.40, disbelief=0.35, uncertainty=0.25, base_rate=0.5)
        thin_evidence = Opinion(belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.5)

        assert well_evidenced.projected_probability() == pytest.approx(
            thin_evidence.projected_probability(), abs=1e-9
        ), "Sanity check: both must have same P for this test"

        docs = [
            _alert_doc("well", well_evidenced),
            _alert_doc("thin", thin_evidence),
        ]

        # Pure threshold: both kept (P=0.525 >= 0.5)
        kept_scalar, _, _ = fhir_filter_alerts(docs, threshold=0.5)
        assert len(kept_scalar) == 2

        # With ceiling: thin kept because u >= 0.7, well is evaluated
        # normally by threshold (P=0.525 >= 0.5 → kept).
        # Both are kept, but for DIFFERENT REASONS — the report should
        # reflect this.
        kept_sl, _, report = fhir_filter_alerts(
            docs, threshold=0.5, uncertainty_ceiling=0.7
        )
        assert len(kept_sl) == 2

        # Now raise threshold above P=0.525: without ceiling the
        # well-evidenced alert is suppressed — correctly, because
        # we genuinely have mixed evidence.  The thin-evidence one
        # is ALSO suppressed — incorrectly, because we barely know
        # anything.  Ceiling rescues it.
        kept_high, supp_high, _ = fhir_filter_alerts(
            docs, threshold=0.6
        )
        assert len(supp_high) == 2  # Both suppressed

        kept_high_ceil, supp_high_ceil, _ = fhir_filter_alerts(
            docs, threshold=0.6, uncertainty_ceiling=0.7
        )
        kept_ids = {d["id"] for d in kept_high_ceil}
        supp_ids = {d["id"] for d in supp_high_ceil}
        assert "thin" in kept_ids    # rescued by ceiling
        assert "well" in supp_ids    # correctly suppressed


# ===================================================================
# 4. FAIL-SAFE BEHAVIOR
# ===================================================================


class TestFailSafe:
    """Docs without opinions are KEPT, never suppressed.

    Suppressing an alert that couldn't be assessed is clinically unsafe.
    """

    def test_no_opinions_kept_with_warning(self):
        doc_no_op = {
            "@type": "fhir:DetectedIssue",
            "id": "no-op",
            "status": "final",
            "opinions": [],
        }
        kept, suppressed, report = fhir_filter_alerts([doc_no_op])

        assert len(kept) == 1
        assert len(suppressed) == 0
        assert any("no-op" in w for w in report.warnings)

    def test_missing_opinions_key_kept_with_warning(self):
        doc_broken = {
            "@type": "fhir:DetectedIssue",
            "id": "broken",
            "status": "final",
        }
        kept, suppressed, report = fhir_filter_alerts([doc_broken])

        assert len(kept) == 1
        assert len(suppressed) == 0
        assert len(report.warnings) > 0


# ===================================================================
# 5. REPORT COMPLETENESS
# ===================================================================


class TestReportCompleteness:
    """AlertFilterReport captures full audit trail."""

    def test_report_counts(self):
        docs = [
            _alert_doc("high", HIGH_CONF),
            _alert_doc("low", LOW_RISK),
            _alert_doc("mod", MOD_CONF),
        ]
        _, _, report = fhir_filter_alerts(docs, threshold=0.5)

        assert report.total_input == 3
        assert report.kept_count + report.suppressed_count == report.total_input

    def test_report_projected_probabilities(self):
        """Report includes projected probabilities for both lists."""
        docs = [
            _alert_doc("high", HIGH_CONF),   # P = 0.90
            _alert_doc("low", LOW_RISK),     # P = 0.225
        ]
        _, _, report = fhir_filter_alerts(docs, threshold=0.5)

        assert len(report.kept_projected) == 1
        assert report.kept_projected[0] == pytest.approx(0.90, abs=1e-9)
        assert len(report.suppressed_projected) == 1
        assert report.suppressed_projected[0] == pytest.approx(0.225, abs=1e-9)

    def test_report_warnings_empty_for_clean_input(self):
        docs = [_alert_doc("d-1", HIGH_CONF)]
        _, _, report = fhir_filter_alerts(docs)

        assert report.warnings == []


# ===================================================================
# 6. PARAMETER VALIDATION
# ===================================================================


class TestParameterValidation:
    """Invalid parameters raise clear errors."""

    def test_threshold_below_zero_raises(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            fhir_filter_alerts([], threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="[Tt]hreshold"):
            fhir_filter_alerts([], threshold=1.1)

    def test_ceiling_below_zero_raises(self):
        with pytest.raises(ValueError, match="[Uu]ncertainty.*ceiling"):
            fhir_filter_alerts([], uncertainty_ceiling=-0.1)

    def test_ceiling_above_one_raises(self):
        with pytest.raises(ValueError, match="[Uu]ncertainty.*ceiling"):
            fhir_filter_alerts([], uncertainty_ceiling=1.1)


# ===================================================================
# 7. BASE RATE SENSITIVITY
# ===================================================================


class TestBaseRateSensitivity:
    """Verify filtering works correctly with non-default base rates.

    The projected probability P = b + a·u depends on the base rate a.
    Real clinical alerts have varying base rates (drug interaction
    prevalence ranges from <1% to >20%).  All previous tests used
    a=0.5; these tests verify the formula is applied correctly.
    """

    def test_low_base_rate_reduces_projected_prob(self):
        """Low base rate (rare interaction) → lower projected probability.

        b=0.10, d=0.05, u=0.85, a=0.02 → P = 0.10 + 0.02·0.85 = 0.117
        Same opinion with a=0.5 → P = 0.10 + 0.5·0.85 = 0.525

        The rare interaction should be suppressed; the common-base-rate
        version should not.
        """
        rare = Opinion(belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.02)
        common = Opinion(belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.50)

        docs = [
            _alert_doc("rare", rare),
            _alert_doc("common", common),
        ]

        kept, suppressed, report = fhir_filter_alerts(docs, threshold=0.5)

        kept_ids = {d["id"] for d in kept}
        supp_ids = {d["id"] for d in suppressed}
        assert "common" in kept_ids   # P=0.525 >= 0.5
        assert "rare" in supp_ids     # P=0.117 < 0.5

    def test_high_base_rate_increases_projected_prob(self):
        """High base rate (common interaction) → higher projected prob.

        b=0.10, d=0.05, u=0.85, a=0.90 → P = 0.10 + 0.90·0.85 = 0.865
        """
        common_interaction = Opinion(
            belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.90
        )
        docs = [_alert_doc("common", common_interaction)]

        kept, suppressed, _ = fhir_filter_alerts(docs, threshold=0.5)
        assert len(kept) == 1  # P=0.865 >= 0.5

    def test_base_rate_zero_uncertainty_irrelevant(self):
        """With a=0, uncertainty contributes nothing to P.

        b=0.10, d=0.05, u=0.85, a=0.0 → P = 0.10 + 0.0·0.85 = 0.10
        All projected probability comes from belief alone.
        """
        pessimistic_prior = Opinion(
            belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.0
        )
        docs = [_alert_doc("pess", pessimistic_prior)]

        kept, suppressed, report = fhir_filter_alerts(docs, threshold=0.5)
        assert len(suppressed) == 1  # P=0.10 < 0.5
        assert report.suppressed_projected[0] == pytest.approx(0.10, abs=1e-9)

    def test_ceiling_rescues_regardless_of_base_rate(self):
        """Uncertainty ceiling overrides base rate effect.

        Even with a=0.02 making P=0.117 (well below threshold),
        ceiling rescues because u=0.85 >= 0.7.
        """
        rare = Opinion(belief=0.10, disbelief=0.05, uncertainty=0.85, base_rate=0.02)
        docs = [_alert_doc("rare", rare)]

        # Without ceiling: suppressed
        _, supp, _ = fhir_filter_alerts(docs, threshold=0.5)
        assert len(supp) == 1

        # With ceiling: kept
        kept, _, _ = fhir_filter_alerts(
            docs, threshold=0.5, uncertainty_ceiling=0.7
        )
        assert len(kept) == 1


# ===================================================================
# 8. FIRST-OPINION SELECTION
# ===================================================================


class TestFirstOpinionSelection:
    """When a doc has multiple opinions, the first controls filtering.

    This is explicitly documented behavior: DetectedIssue typically has
    one opinion (severity or status), but if multiple exist, the first
    is used as the primary filter criterion.
    """

    def test_first_opinion_controls_filter(self):
        """First opinion determines keep/suppress, second is ignored."""
        high = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.10, base_rate=0.5)
        low = Opinion(belief=0.15, disbelief=0.70, uncertainty=0.15, base_rate=0.5)

        # First opinion is high → kept, even though second is low
        doc_high_first = {
            "@type": "fhir:DetectedIssue",
            "id": "high-first",
            "status": "final",
            "opinions": [
                {"field": "severity", "value": "high", "opinion": high, "source": "reconstructed"},
                {"field": "status", "value": "final", "opinion": low, "source": "reconstructed"},
            ],
        }
        # First opinion is low → suppressed, even though second is high
        doc_low_first = {
            "@type": "fhir:DetectedIssue",
            "id": "low-first",
            "status": "final",
            "opinions": [
                {"field": "severity", "value": "low", "opinion": low, "source": "reconstructed"},
                {"field": "status", "value": "final", "opinion": high, "source": "reconstructed"},
            ],
        }

        kept, suppressed, _ = fhir_filter_alerts(
            [doc_high_first, doc_low_first], threshold=0.5
        )

        kept_ids = {d["id"] for d in kept}
        supp_ids = {d["id"] for d in suppressed}
        assert "high-first" in kept_ids
        assert "low-first" in supp_ids
