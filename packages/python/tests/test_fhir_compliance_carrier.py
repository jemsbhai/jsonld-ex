"""Tests for the FHIR R4 resource-level compliance opinion carrier.

TDD RED PHASE: tests written first; the implementation does not exist yet.

Motivation (IEEE HealthCom 2026 camera-ready, reviewer 2): the paper
must state how a compliance opinion (l, v, u, a) about a FHIR R4 resource
or Bundle is carried inside FHIR R4. The clinical-assertion carrier
already exists (opinion_to_fhir_extension on the qualified element via
the ``_element`` convention). This module adds the resource-level carrier:
the same four-component complex extension placed in ``Resource.meta.extension``,
under a distinct canonical URL so that a compliance opinion (lawfulness)
is never confused with a clinical opinion (truth of an assertion).

FHIR source of truth: HL7 FHIR R4 v4.0.1
  - Extensibility: https://hl7.org/fhir/R4/extensibility.html
  - Meta datatype (Element, so it may carry extensions):
    https://hl7.org/fhir/R4/resource.html#Meta
Mathematical source of truth: compliance_algebra.md (Definition 2) and
experiments/EH_DESIGN.md of the HealthCom paper.

Public API under test (jsonld_ex.fhir_interop):
  FHIR_COMPLIANCE_EXTENSION_URL
  ComplianceOpinionCarrier            dataclass(opinion, regime, assessed_at)
  fhir_attach_compliance_opinion(resource, opinion, *, regime=None,
                                 assessed_at=None, fhir_version="R4") -> dict
  fhir_read_compliance_opinion(resource, *, fhir_version="R4")
                                 -> ComplianceOpinionCarrier | None
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from jsonld_ex.compliance_algebra import ComplianceOpinion
from jsonld_ex.fhir_interop._constants import FHIR_EXTENSION_URL
from jsonld_ex.fhir_interop._scalar import opinion_to_fhir_extension

# ── Import targets (fail until the implementation exists) ─────────
from jsonld_ex.fhir_interop import (
    FHIR_COMPLIANCE_EXTENSION_URL,
    ComplianceOpinionCarrier,
    fhir_attach_compliance_opinion,
    fhir_read_compliance_opinion,
)

TOL = 1e-12


def _patient() -> dict:
    return {
        "resourceType": "Patient",
        "id": "p1",
        "meta": {"versionId": "3", "lastUpdated": "2026-06-01T00:00:00Z"},
        "name": [{"family": "Doe", "given": ["Jane"]}],
        "birthDate": "1970-01-01",
    }


def _bundle() -> dict:
    return {
        "resourceType": "Bundle",
        "id": "b1",
        "type": "collection",
        "entry": [{"resource": _patient()}],
    }


def _phi_like_opinion() -> ComplianceOpinion:
    # Values shaped like the EH2 Phase A result (raw Synthea patient).
    return ComplianceOpinion.create(
        lawfulness=8.443126369728699e-09,
        violation=0.9999992152832762,
        uncertainty=1.0 - 8.443126369728699e-09 - 0.9999992152832762,
        base_rate=3.814697265625e-06,
    )


def _find_compliance_extensions(resource: dict) -> list[dict]:
    return [
        e for e in resource.get("meta", {}).get("extension", [])
        if e.get("url") == FHIR_COMPLIANCE_EXTENSION_URL
    ]


# ── URL and namespace ─────────────────────────────────────────────

def test_compliance_extension_url_is_library_scoped_and_distinct():
    assert FHIR_COMPLIANCE_EXTENSION_URL.startswith("https://jsonld-ex.github.io/ns/fhir/")
    assert FHIR_COMPLIANCE_EXTENSION_URL != FHIR_EXTENSION_URL


# ── Attach: structure ─────────────────────────────────────────────

def test_attach_places_complex_extension_in_meta():
    op = ComplianceOpinion.create(0.652, 0.30, 0.048, 0.5)
    out = fhir_attach_compliance_opinion(_patient(), op)
    exts = _find_compliance_extensions(out)
    assert len(exts) == 1
    ext = exts[0]
    subs = {s["url"]: s for s in ext["extension"]}
    assert set(subs) == {"belief", "disbelief", "uncertainty", "baseRate"}
    assert subs["belief"]["valueDecimal"] == 0.652
    assert subs["disbelief"]["valueDecimal"] == 0.30
    assert subs["uncertainty"]["valueDecimal"] == 0.048
    assert subs["baseRate"]["valueDecimal"] == 0.5


def test_attach_preserves_existing_meta_fields_and_other_extensions():
    res = _patient()
    res["meta"]["extension"] = [{"url": "http://example.org/other", "valueString": "x"}]
    op = ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5)
    out = fhir_attach_compliance_opinion(res, op)
    assert out["meta"]["versionId"] == "3"
    assert out["meta"]["lastUpdated"] == "2026-06-01T00:00:00Z"
    assert {"url": "http://example.org/other", "valueString": "x"} in out["meta"]["extension"]
    assert len(out["meta"]["extension"]) == 2


def test_attach_creates_meta_when_absent():
    res = _patient()
    del res["meta"]
    out = fhir_attach_compliance_opinion(res, ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))
    assert len(_find_compliance_extensions(out)) == 1


def test_attach_does_not_mutate_input():
    res = _patient()
    snapshot = copy.deepcopy(res)
    fhir_attach_compliance_opinion(res, ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))
    assert res == snapshot


def test_attach_twice_replaces_rather_than_duplicates():
    res = _patient()
    first = fhir_attach_compliance_opinion(res, ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))
    second = fhir_attach_compliance_opinion(first, ComplianceOpinion.create(0.2, 0.7, 0.1, 0.5))
    exts = _find_compliance_extensions(second)
    assert len(exts) == 1
    assert fhir_read_compliance_opinion(second).opinion.belief == 0.2


def test_attach_works_on_bundle_resource():
    out = fhir_attach_compliance_opinion(_bundle(), _phi_like_opinion())
    assert out["resourceType"] == "Bundle"
    assert len(_find_compliance_extensions(out)) == 1
    # entries untouched
    assert out["entry"] == _bundle()["entry"]


def test_attach_leaves_clinical_element_extensions_untouched():
    res = _patient()
    res["active"] = True
    clinical = opinion_to_fhir_extension(ComplianceOpinion.create(0.8, 0.1, 0.1, 0.5))
    res["_active"] = {"extension": [clinical]}
    out = fhir_attach_compliance_opinion(res, ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))
    assert out["_active"] == {"extension": [clinical]}
    assert out["_active"]["extension"][0]["url"] == FHIR_EXTENSION_URL


def test_attach_optional_regime_and_assessed_at_sub_extensions():
    out = fhir_attach_compliance_opinion(
        _patient(),
        ComplianceOpinion.create(0.652, 0.30, 0.048, 0.5),
        regime="HIPAA",
        assessed_at="2026-06-01T15:11:42Z",
    )
    subs = {s["url"]: s for s in _find_compliance_extensions(out)[0]["extension"]}
    assert subs["regime"]["valueCode"] == "HIPAA"
    assert subs["assessedAt"]["valueDateTime"] == "2026-06-01T15:11:42Z"


# ── Attach: validation ────────────────────────────────────────────

@pytest.mark.parametrize("bad", [None, "Patient", 42, [], {}, {"id": "x"}, {"resourceType": 7}])
def test_attach_rejects_non_resource_input(bad):
    with pytest.raises(ValueError):
        fhir_attach_compliance_opinion(bad, ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))


def test_attach_rejects_non_compliance_opinion():
    with pytest.raises(TypeError):
        fhir_attach_compliance_opinion(_patient(), (0.9, 0.05, 0.05, 0.5))


def test_attach_rejects_unsupported_fhir_version():
    with pytest.raises(ValueError):
        fhir_attach_compliance_opinion(
            _patient(), ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5), fhir_version="R5"
        )


@pytest.mark.parametrize("regime", ["", 12])
def test_attach_rejects_bad_regime(regime):
    with pytest.raises(ValueError):
        fhir_attach_compliance_opinion(
            _patient(), ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5), regime=regime
        )


# ── Read ──────────────────────────────────────────────────────────

def test_read_returns_none_when_absent():
    assert fhir_read_compliance_opinion(_patient()) is None
    res = _patient()
    del res["meta"]
    assert fhir_read_compliance_opinion(res) is None


def test_read_returns_carrier_with_compliance_opinion_type():
    op = ComplianceOpinion.create(0.652, 0.30, 0.048, 0.5)
    carrier = fhir_read_compliance_opinion(
        fhir_attach_compliance_opinion(_patient(), op, regime="HIPAA", assessed_at="2026-06-01T15:11:42Z")
    )
    assert isinstance(carrier, ComplianceOpinionCarrier)
    assert isinstance(carrier.opinion, ComplianceOpinion)
    assert carrier.regime == "HIPAA"
    assert carrier.assessed_at == "2026-06-01T15:11:42Z"


def test_read_without_optional_fields_gives_none_metadata():
    carrier = fhir_read_compliance_opinion(
        fhir_attach_compliance_opinion(_patient(), ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5))
    )
    assert carrier.regime is None
    assert carrier.assessed_at is None


def test_round_trip_is_exact_in_memory_and_through_json():
    op = _phi_like_opinion()
    attached = fhir_attach_compliance_opinion(_bundle(), op, regime="HIPAA")
    reloaded = json.loads(json.dumps(attached))
    for res in (attached, reloaded):
        got = fhir_read_compliance_opinion(res).opinion
        assert got.belief == op.belief
        assert got.disbelief == op.disbelief
        assert got.uncertainty == op.uncertainty
        assert got.base_rate == op.base_rate
        assert math.isclose(got.belief + got.disbelief + got.uncertainty, 1.0, abs_tol=1e-9)


def test_read_ignores_clinical_extension_with_other_url_in_meta():
    res = _patient()
    res["meta"]["extension"] = [opinion_to_fhir_extension(ComplianceOpinion.create(0.8, 0.1, 0.1, 0.5))]
    assert fhir_read_compliance_opinion(res) is None


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda subs: subs.pop("uncertainty"),                       # missing component
        lambda subs: subs["belief"].__setitem__("valueDecimal", 1.5),  # out of range
        lambda subs: subs["belief"].__setitem__("valueDecimal", 0.9),  # constraint broken
    ],
)
def test_read_raises_on_malformed_extension(corrupt):
    out = fhir_attach_compliance_opinion(_patient(), ComplianceOpinion.create(0.652, 0.30, 0.048, 0.5))
    ext = _find_compliance_extensions(out)[0]
    subs = {s["url"]: s for s in ext["extension"]}
    corrupt(subs)
    ext["extension"] = list(subs.values())
    with pytest.raises(ValueError):
        fhir_read_compliance_opinion(out)


def test_read_rejects_unsupported_fhir_version():
    with pytest.raises(ValueError):
        fhir_read_compliance_opinion(_patient(), fhir_version="R5")
