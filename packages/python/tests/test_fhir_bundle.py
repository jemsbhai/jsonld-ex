"""Tests for FHIR R4 Bundle processing functions.

TDD Red Phase — tests for:
  fhir_bundle_annotate()  — annotate all supported entries in a Bundle
  fhir_bundle_fuse()      — fuse multiple Bundles from different sources

These tests import from the public API (fhir_interop) and will fail
until _bundle.py is implemented.
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    opinion_to_fhir_extension,
    FHIR_EXTENSION_URL,
)
from jsonld_ex.fhir_interop._bundle import (
    fhir_bundle_annotate,
    fhir_bundle_fuse,
    BundleReport,
)


# ── Test helpers ──────────────────────────────────────────────────


def _bundle(entries, *, bundle_type="collection", bundle_id="test-bundle"):
    """Build a minimal FHIR R4 Bundle."""
    return {
        "resourceType": "Bundle",
        "id": bundle_id,
        "type": bundle_type,
        "entry": entries,
    }


def _entry(resource, *, fullUrl=None):
    """Build a Bundle entry wrapping a resource."""
    e = {"resource": resource}
    if fullUrl is not None:
        e["fullUrl"] = fullUrl
    return e


def _observation(obs_id, *, status="final", code="H"):
    """Build a minimal Observation resource."""
    return {
        "resourceType": "Observation",
        "id": obs_id,
        "status": status,
        "interpretation": [{"coding": [{"code": code}]}],
    }


def _condition(cond_id, *, vs_code="confirmed"):
    """Build a minimal Condition resource."""
    return {
        "resourceType": "Condition",
        "id": cond_id,
        "verificationStatus": {"coding": [{"code": vs_code}]},
    }


def _risk_assessment(ra_id, *, probability=0.85, status="final"):
    """Build a minimal RiskAssessment resource."""
    return {
        "resourceType": "RiskAssessment",
        "id": ra_id,
        "status": status,
        "prediction": [{"probabilityDecimal": probability}],
    }


def _consent(consent_id, *, status="active"):
    """Build a minimal Consent resource."""
    return {
        "resourceType": "Consent",
        "id": consent_id,
        "status": status,
    }


def _unsupported_resource(res_id):
    """Build a resource type not supported by from_fhir."""
    return {
        "resourceType": "Patient",
        "id": res_id,
        "name": [{"family": "Smith", "given": ["Jane"]}],
    }


# ═══════════════════════════════════════════════════════════════════
# fhir_bundle_annotate() tests
# ═══════════════════════════════════════════════════════════════════


class TestBundleAnnotateBasic:
    """Core behavior for single-bundle annotation."""

    def test_returns_bundle_report(self):
        """fhir_bundle_annotate returns (docs, BundleReport)."""
        bundle = _bundle([_entry(_observation("obs-1"))])
        docs, report = fhir_bundle_annotate(bundle)
        assert isinstance(report, BundleReport)
        assert isinstance(docs, list)

    def test_single_observation_annotated(self):
        """Single Observation entry produces one annotated doc."""
        bundle = _bundle([_entry(_observation("obs-1"))])
        docs, report = fhir_bundle_annotate(bundle)
        assert len(docs) == 1
        assert docs[0]["@type"] == "fhir:Observation"
        assert len(docs[0]["opinions"]) >= 1
        assert report.total_entries == 1
        assert report.annotated == 1

    def test_multiple_entries_annotated(self):
        """Multiple supported entries all get annotated."""
        bundle = _bundle([
            _entry(_observation("obs-1")),
            _entry(_condition("cond-1")),
            _entry(_risk_assessment("ra-1")),
        ])
        docs, report = fhir_bundle_annotate(bundle)
        assert len(docs) == 3
        assert report.total_entries == 3
        assert report.annotated == 3
        types = {d["@type"] for d in docs}
        assert types == {"fhir:Observation", "fhir:Condition", "fhir:RiskAssessment"}

    def test_empty_bundle(self):
        """Empty Bundle produces empty list and zero counts."""
        bundle = _bundle([])
        docs, report = fhir_bundle_annotate(bundle)
        assert docs == []
        assert report.total_entries == 0
        assert report.annotated == 0

    def test_preserves_entry_order(self):
        """Annotated docs are returned in the same order as entries."""
        bundle = _bundle([
            _entry(_condition("cond-1")),
            _entry(_observation("obs-1")),
            _entry(_consent("consent-1")),
        ])
        docs, _ = fhir_bundle_annotate(bundle)
        assert docs[0]["@type"] == "fhir:Condition"
        assert docs[1]["@type"] == "fhir:Observation"
        assert docs[2]["@type"] == "fhir:Consent"

    def test_preserves_resource_ids(self):
        """Resource ids pass through annotation."""
        bundle = _bundle([
            _entry(_observation("obs-42")),
            _entry(_condition("cond-99")),
        ])
        docs, _ = fhir_bundle_annotate(bundle)
        assert docs[0]["id"] == "obs-42"
        assert docs[1]["id"] == "cond-99"


class TestBundleAnnotateUnsupported:
    """Handling of unsupported resource types and malformed entries."""

    def test_unsupported_resource_skipped(self):
        """Unsupported resource types are skipped with a warning."""
        bundle = _bundle([
            _entry(_unsupported_resource("patient-1")),
            _entry(_observation("obs-1")),
        ])
        docs, report = fhir_bundle_annotate(bundle)
        # Unsupported still produces a doc (with empty opinions per from_fhir)
        assert report.total_entries == 2
        assert report.annotated == 1
        assert report.skipped == 1
        assert len(report.warnings) >= 1

    def test_entry_without_resource_skipped(self):
        """Entry without a 'resource' key is skipped."""
        bundle = _bundle([
            {"request": {"method": "GET", "url": "Patient/1"}},
            _entry(_observation("obs-1")),
        ])
        docs, report = fhir_bundle_annotate(bundle)
        assert report.total_entries == 2
        assert report.skipped >= 1

    def test_mixed_supported_unsupported(self):
        """Mixed bundle: supported are annotated, unsupported are skipped."""
        bundle = _bundle([
            _entry(_observation("obs-1")),
            _entry(_unsupported_resource("patient-1")),
            _entry(_condition("cond-1")),
            _entry(_unsupported_resource("patient-2")),
        ])
        docs, report = fhir_bundle_annotate(bundle)
        assert report.annotated == 2
        assert report.skipped == 2


class TestBundleAnnotateBundleTypes:
    """All valid FHIR Bundle types are accepted."""

    @pytest.mark.parametrize("bundle_type", [
        "document", "message", "transaction", "searchset",
        "collection", "batch", "history",
    ])
    def test_bundle_type_accepted(self, bundle_type):
        """fhir_bundle_annotate accepts all standard Bundle types."""
        bundle = _bundle(
            [_entry(_observation("obs-1"))],
            bundle_type=bundle_type,
        )
        docs, report = fhir_bundle_annotate(bundle)
        assert report.annotated == 1


class TestBundleAnnotateValidation:
    """Input validation and error handling."""

    def test_not_a_bundle_raises(self):
        """Non-Bundle resource raises ValueError."""
        with pytest.raises(ValueError, match="Bundle"):
            fhir_bundle_annotate({"resourceType": "Patient", "id": "p1"})

    def test_missing_resourceType_raises(self):
        """Missing resourceType raises ValueError."""
        with pytest.raises(ValueError):
            fhir_bundle_annotate({"type": "collection"})

    def test_bundle_without_entry_key(self):
        """Bundle with no 'entry' key returns empty results."""
        bundle = {"resourceType": "Bundle", "type": "collection"}
        docs, report = fhir_bundle_annotate(bundle)
        assert docs == []
        assert report.total_entries == 0


class TestBundleAnnotateMetadata:
    """Bundle-level metadata preservation."""

    def test_report_contains_bundle_id(self):
        """Report includes the source Bundle id."""
        bundle = _bundle(
            [_entry(_observation("obs-1"))],
            bundle_id="my-bundle-42",
        )
        _, report = fhir_bundle_annotate(bundle)
        assert report.bundle_id == "my-bundle-42"

    def test_report_contains_bundle_type(self):
        """Report includes the Bundle type."""
        bundle = _bundle(
            [_entry(_observation("obs-1"))],
            bundle_type="searchset",
        )
        _, report = fhir_bundle_annotate(bundle)
        assert report.bundle_type == "searchset"


class TestBundleAnnotateExtensionRecovery:
    """Extension-based opinion recovery works through bundle annotation."""

    def test_extension_opinions_recovered(self):
        """Opinions embedded as FHIR extensions are recovered."""
        injected = Opinion(belief=0.90, disbelief=0.02, uncertainty=0.08)
        ext = opinion_to_fhir_extension(injected)
        obs = _observation("obs-ext-1")
        obs["_interpretation"] = {"extension": [ext]}

        bundle = _bundle([_entry(obs)])
        docs, _ = fhir_bundle_annotate(bundle)
        op = docs[0]["opinions"][0]["opinion"]
        assert abs(op.belief - 0.90) < 1e-9
        assert docs[0]["opinions"][0]["source"] == "extension"


# ═══════════════════════════════════════════════════════════════════
# fhir_bundle_fuse() tests
# ═══════════════════════════════════════════════════════════════════


class TestBundleFuseBasic:
    """Core behavior for multi-bundle fusion."""

    def test_returns_fused_docs_and_report(self):
        """fhir_bundle_fuse returns (fused_docs, BundleReport)."""
        b1 = _bundle([_entry(_observation("obs-1"))])
        b2 = _bundle([_entry(_observation("obs-1"))])
        fused, report = fhir_bundle_fuse([b1, b2])
        assert isinstance(fused, list)
        assert isinstance(report, BundleReport)

    def test_same_resource_fused(self):
        """Same resource (type+id) from two bundles is fused into one doc."""
        b1 = _bundle([_entry(_observation("obs-1", code="H"))])
        b2 = _bundle([_entry(_observation("obs-1", code="L"))])
        fused, report = fhir_bundle_fuse([b1, b2])
        # Should produce one fused document for obs-1
        assert len(fused) == 1
        assert fused[0]["id"] == "obs-1"
        assert report.groups_fused >= 1

    def test_different_resources_not_fused(self):
        """Different resource ids stay separate."""
        b1 = _bundle([_entry(_observation("obs-1"))])
        b2 = _bundle([_entry(_observation("obs-2"))])
        fused, report = fhir_bundle_fuse([b1, b2])
        assert len(fused) == 2
        ids = {d["id"] for d in fused}
        assert ids == {"obs-1", "obs-2"}

    def test_different_resource_types_not_fused(self):
        """Same id but different types are NOT fused."""
        b1 = _bundle([_entry(_observation("shared-id"))])
        b2 = _bundle([_entry(_condition("shared-id"))])
        fused, report = fhir_bundle_fuse([b1, b2])
        assert len(fused) == 2

    def test_single_bundle_passthrough(self):
        """Single bundle input: no fusion, just annotation."""
        b1 = _bundle([
            _entry(_observation("obs-1")),
            _entry(_condition("cond-1")),
        ])
        fused, report = fhir_bundle_fuse([b1])
        assert len(fused) == 2
        assert report.groups_fused == 0

    def test_three_bundles_fused(self):
        """Three bundles with overlapping resources are fused correctly."""
        b1 = _bundle([_entry(_observation("obs-1", code="H"))])
        b2 = _bundle([_entry(_observation("obs-1", code="N"))])
        b3 = _bundle([_entry(_observation("obs-1", code="L"))])
        fused, report = fhir_bundle_fuse([b1, b2, b3])
        assert len(fused) == 1
        assert report.groups_fused == 1

    def test_empty_bundles(self):
        """Empty bundles produce empty results."""
        fused, report = fhir_bundle_fuse([_bundle([]), _bundle([])])
        assert fused == []
        assert report.groups_fused == 0


class TestBundleFuseOpinionIntegrity:
    """Verify the fused opinion is mathematically sound."""

    def test_fused_opinion_valid(self):
        """Fused opinion sums to 1.0."""
        b1 = _bundle([_entry(_observation("obs-1", code="H"))])
        b2 = _bundle([_entry(_observation("obs-1", code="L"))])
        fused, _ = fhir_bundle_fuse([b1, b2])
        op = fused[0]["opinions"][0]["opinion"]
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_fused_opinion_source_marked(self):
        """Fused opinion is marked as 'fused', not 'reconstructed'."""
        b1 = _bundle([_entry(_observation("obs-1", code="H"))])
        b2 = _bundle([_entry(_observation("obs-1", code="N"))])
        fused, _ = fhir_bundle_fuse([b1, b2])
        assert fused[0]["opinions"][0]["source"] == "fused"


class TestBundleFuseConflictDetection:
    """Conflict detection between sources during fusion."""

    def test_conflict_scores_reported(self):
        """Conflicting sources produce non-zero conflict scores."""
        # H (high) vs L (low) should conflict
        b1 = _bundle([_entry(_observation("obs-1", code="H"))])
        b2 = _bundle([_entry(_observation("obs-1", code="L"))])
        _, report = fhir_bundle_fuse([b1, b2])
        assert len(report.conflict_scores) >= 1
        assert any(s > 0.0 for s in report.conflict_scores)

    def test_agreeing_sources_low_conflict(self):
        """Agreeing sources produce lower conflict than disagreeing."""
        # Agreeing: both H
        b1_agree = _bundle([_entry(_observation("obs-1", code="H"))])
        b2_agree = _bundle([_entry(_observation("obs-1", code="H"))])
        _, report_agree = fhir_bundle_fuse([b1_agree, b2_agree])

        # Disagreeing: H vs L
        b1_disagree = _bundle([_entry(_observation("obs-2", code="H"))])
        b2_disagree = _bundle([_entry(_observation("obs-2", code="L"))])
        _, report_disagree = fhir_bundle_fuse([b1_disagree, b2_disagree])

        max_agree = max(report_agree.conflict_scores) if report_agree.conflict_scores else 0.0
        max_disagree = max(report_disagree.conflict_scores) if report_disagree.conflict_scores else 0.0
        assert max_agree < max_disagree, (
            f"Agreeing conflict {max_agree} should be < disagreeing {max_disagree}"
        )


class TestBundleFuseMethod:
    """Fusion method selection."""

    def test_default_method_cumulative(self):
        """Default fusion method is cumulative."""
        b1 = _bundle([_entry(_observation("obs-1"))])
        b2 = _bundle([_entry(_observation("obs-1"))])
        _, report = fhir_bundle_fuse([b1, b2])
        assert report.fusion_method == "cumulative"

    def test_averaging_method(self):
        """Averaging fusion method can be selected."""
        b1 = _bundle([_entry(_observation("obs-1"))])
        b2 = _bundle([_entry(_observation("obs-1"))])
        _, report = fhir_bundle_fuse([b1, b2], method="averaging")
        assert report.fusion_method == "averaging"

    def test_invalid_method_raises(self):
        """Invalid fusion method raises ValueError."""
        b1 = _bundle([_entry(_observation("obs-1"))])
        with pytest.raises(ValueError, match="method"):
            fhir_bundle_fuse([b1], method="invalid")


class TestBundleFuseValidation:
    """Input validation."""

    def test_empty_list_raises(self):
        """Empty bundle list raises ValueError."""
        with pytest.raises(ValueError):
            fhir_bundle_fuse([])

    def test_non_bundle_in_list_raises(self):
        """Non-Bundle resource in list raises ValueError."""
        with pytest.raises(ValueError, match="Bundle"):
            fhir_bundle_fuse([{"resourceType": "Patient", "id": "p1"}])


class TestBundleFuseMixedResources:
    """Fusion with heterogeneous resource types."""

    def test_mixed_types_fused_independently(self):
        """Different resource types are grouped and fused independently."""
        b1 = _bundle([
            _entry(_observation("obs-1", code="H")),
            _entry(_condition("cond-1", vs_code="confirmed")),
        ])
        b2 = _bundle([
            _entry(_observation("obs-1", code="N")),
            _entry(_condition("cond-1", vs_code="provisional")),
        ])
        fused, report = fhir_bundle_fuse([b1, b2])
        assert len(fused) == 2
        assert report.groups_fused == 2
        types = {d["@type"] for d in fused}
        assert types == {"fhir:Observation", "fhir:Condition"}

    def test_unique_plus_overlapping(self):
        """Unique resources stay separate, overlapping are fused."""
        b1 = _bundle([
            _entry(_observation("obs-1")),
            _entry(_condition("cond-unique-1")),
        ])
        b2 = _bundle([
            _entry(_observation("obs-1")),
            _entry(_risk_assessment("ra-unique-1")),
        ])
        fused, report = fhir_bundle_fuse([b1, b2])
        # obs-1 fused → 1, cond-unique-1 → 1, ra-unique-1 → 1 = 3 total
        assert len(fused) == 3
        assert report.groups_fused == 1


class TestBundleFuseUnsupported:
    """Unsupported resources in bundles to be fused."""

    def test_unsupported_resources_skipped(self):
        """Unsupported resources are skipped during fusion."""
        b1 = _bundle([
            _entry(_observation("obs-1")),
            _entry(_unsupported_resource("patient-1")),
        ])
        b2 = _bundle([
            _entry(_observation("obs-1")),
        ])
        fused, report = fhir_bundle_fuse([b1, b2])
        # obs-1 fused, patient-1 skipped
        fused_types = {d["@type"] for d in fused}
        assert "fhir:Observation" in fused_types
        assert report.skipped >= 1
