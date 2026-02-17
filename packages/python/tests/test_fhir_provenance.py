"""Tests for FHIR R4 Provenance ↔ jsonld-ex bidirectional conversion.

TDD Red Phase — tests for:
  1. _from_provenance_r4()       — FHIR Provenance → jsonld-ex doc
  2. _to_provenance_r4()         — jsonld-ex doc → FHIR Provenance
  3. fhir_provenance_to_prov_o() — jsonld-ex Provenance doc → W3C PROV-O

Design decisions:
  - Proposition under assessment: "this provenance record is reliable
    and complete."
  - Base probability: 0.75 (moderate positive prior — a provenance
    record exists because *something* was documented).
  - Default uncertainty: 0.20 (higher than clinical resources because
    provenance metadata is often incomplete).
  - Opinion field: "recorded" when timestamp present; "provenance"
    as synthetic field when timestamp absent.
  - Option A architecture: from_fhir/to_fhir produce standard
    jsonld-ex doc; fhir_provenance_to_prov_o() is a separate bridge.
  - All FHIR Provenance data (targets, agents, entities, activity,
    policy, reason, location, period) is preserved in the jsonld-ex
    doc — no data is excluded.

All tests are expected to FAIL until the implementation is written.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    FHIR_EXTENSION_URL,
    opinion_to_fhir_extension,
)
from jsonld_ex.fhir_interop._constants import SUPPORTED_RESOURCE_TYPES
from jsonld_ex.owl_interop import ConversionReport


# ═══════════════════════════════════════════════════════════════════
# Test Fixtures — FHIR R4 Provenance Resources
# ═══════════════════════════════════════════════════════════════════


def _minimal_provenance() -> dict:
    """Minimal valid FHIR R4 Provenance: 1 target, 1 agent, recorded."""
    return {
        "resourceType": "Provenance",
        "id": "prov-001",
        "target": [{"reference": "Observation/obs-1"}],
        "recorded": "2024-06-15T10:30:00Z",
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/"
                            "provenance-participant-type",
                            "code": "author",
                        }
                    ]
                },
                "who": {"reference": "Practitioner/dr-smith"},
            }
        ],
    }


def _rich_provenance() -> dict:
    """Feature-rich Provenance: multiple agents, entities, activity, etc."""
    return {
        "resourceType": "Provenance",
        "id": "prov-002",
        "target": [
            {"reference": "Observation/obs-1"},
            {"reference": "Observation/obs-2"},
        ],
        "recorded": "2024-06-15T14:00:00Z",
        "agent": [
            {
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/"
                            "provenance-participant-type",
                            "code": "author",
                        }
                    ]
                },
                "who": {"reference": "Practitioner/dr-smith"},
            },
            {
                "type": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/"
                            "provenance-participant-type",
                            "code": "verifier",
                        }
                    ]
                },
                "who": {"reference": "Practitioner/dr-jones"},
            },
        ],
        "entity": [
            {
                "role": "source",
                "what": {"reference": "DocumentReference/lab-report-1"},
            },
            {
                "role": "derivation",
                "what": {"reference": "Observation/original-obs"},
            },
        ],
        "activity": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/"
                    "v3-DataOperation",
                    "code": "UPDATE",
                }
            ]
        },
        "reason": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/"
                        "v3-ActReason",
                        "code": "TREAT",
                    }
                ]
            }
        ],
        "policy": ["http://example.org/policy/hipaa"],
        "location": {"reference": "Location/hospital-main"},
    }


def _no_agent_provenance() -> dict:
    """Provenance with empty agent list — higher uncertainty."""
    return {
        "resourceType": "Provenance",
        "id": "prov-003",
        "target": [{"reference": "Observation/obs-1"}],
        "recorded": "2024-06-15T10:30:00Z",
        "agent": [],
    }


def _no_recorded_provenance() -> dict:
    """Provenance without a recorded timestamp — higher uncertainty."""
    return {
        "resourceType": "Provenance",
        "id": "prov-004",
        "target": [{"reference": "Observation/obs-1"}],
        "agent": [
            {
                "type": {"coding": [{"code": "author"}]},
                "who": {"reference": "Practitioner/dr-smith"},
            }
        ],
    }


def _agent_without_who() -> dict:
    """Agent has type but no who reference — less identifiable."""
    return {
        "resourceType": "Provenance",
        "id": "prov-005",
        "target": [{"reference": "Observation/obs-1"}],
        "recorded": "2024-06-15T10:30:00Z",
        "agent": [
            {
                "type": {"coding": [{"code": "author"}]},
                # No 'who' — anonymous agent
            }
        ],
    }


def _delegation_provenance() -> dict:
    """Agent with onBehalfOf — delegation chain."""
    return {
        "resourceType": "Provenance",
        "id": "prov-006",
        "target": [{"reference": "Observation/obs-1"}],
        "recorded": "2024-06-15T10:30:00Z",
        "agent": [
            {
                "type": {"coding": [{"code": "author"}]},
                "who": {"reference": "Practitioner/dr-smith"},
                "onBehalfOf": {"reference": "Organization/hospital"},
            }
        ],
    }


def _make_doc(
    *,
    prov_id: str = "prov-001",
    recorded: str = "2024-06-15T10:30:00Z",
    targets: list | None = None,
    agents: list | None = None,
    entities: list | None = None,
    opinions: list | None = None,
    **extra,
) -> dict:
    """Build a jsonld-ex Provenance doc for to_fhir tests."""
    doc = {
        "@type": "fhir:Provenance",
        "id": prov_id,
        "recorded": recorded,
        "targets": targets if targets is not None else [{"reference": "Observation/obs-1"}],
        "agents": agents if agents is not None else [],
        "entities": entities if entities is not None else [],
        "opinions": opinions if opinions is not None else [
            {
                "field": "recorded",
                "value": recorded,
                "opinion": Opinion(
                    belief=0.60, disbelief=0.15,
                    uncertainty=0.25, base_rate=0.75,
                ),
                "source": "reconstructed",
            }
        ],
    }
    doc.update(extra)
    return doc


# ═══════════════════════════════════════════════════════════════════
# Section 1: Registration & Dispatch
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRegistration:
    """Provenance must be wired into both dispatch tables."""

    def test_provenance_in_supported_resource_types(self):
        assert "Provenance" in SUPPORTED_RESOURCE_TYPES

    def test_from_fhir_dispatches_provenance(self):
        doc, report = from_fhir(_minimal_provenance())
        assert report.success
        assert doc["@type"] == "fhir:Provenance"
        # Must NOT produce the "Unsupported" fallback
        for w in report.warnings:
            assert "Unsupported" not in w

    def test_to_fhir_dispatches_provenance(self):
        resource, report = to_fhir(_make_doc())
        assert report.success
        assert resource["resourceType"] == "Provenance"

    def test_from_fhir_unsupported_before_registration_would_warn(self):
        """Sanity: an unknown type still produces the warning (regression guard)."""
        doc, report = from_fhir({"resourceType": "MadeUpResource"})
        assert any("Unsupported" in w for w in report.warnings)


# ═══════════════════════════════════════════════════════════════════
# Section 2: from_fhir — Basic Conversion
# ═══════════════════════════════════════════════════════════════════


class TestFromProvenanceBasic:
    """Core structural output of _from_provenance_r4."""

    def test_doc_type_and_id(self):
        doc, _ = from_fhir(_minimal_provenance())
        assert doc["@type"] == "fhir:Provenance"
        assert doc["id"] == "prov-001"

    def test_has_opinions(self):
        doc, report = from_fhir(_minimal_provenance())
        assert len(doc["opinions"]) >= 1
        assert report.nodes_converted >= 1

    def test_opinion_field_is_recorded(self):
        """When recorded is present, opinion field should be 'recorded'."""
        doc, _ = from_fhir(_minimal_provenance())
        entry = doc["opinions"][0]
        assert entry["field"] == "recorded"
        assert entry["value"] == "2024-06-15T10:30:00Z"
        assert entry["source"] == "reconstructed"

    def test_opinion_field_when_no_recorded(self):
        """When recorded is absent, opinion field should be 'provenance'."""
        doc, _ = from_fhir(_no_recorded_provenance())
        entry = doc["opinions"][0]
        assert entry["field"] == "provenance"

    def test_opinion_is_valid_sl(self):
        """Reconstructed opinion must satisfy b + d + u = 1, all >= 0."""
        doc, _ = from_fhir(_minimal_provenance())
        op = doc["opinions"][0]["opinion"]
        assert isinstance(op, Opinion)
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        assert op.belief >= 0
        assert op.disbelief >= 0
        assert op.uncertainty >= 0
        assert 0 <= op.base_rate <= 1

    def test_recorded_preserved(self):
        doc, _ = from_fhir(_minimal_provenance())
        assert doc["recorded"] == "2024-06-15T10:30:00Z"

    def test_targets_preserved(self):
        doc, _ = from_fhir(_minimal_provenance())
        assert "targets" in doc
        assert len(doc["targets"]) == 1
        assert doc["targets"][0]["reference"] == "Observation/obs-1"

    def test_agents_preserved_with_type_and_who(self):
        doc, _ = from_fhir(_minimal_provenance())
        assert "agents" in doc
        assert len(doc["agents"]) == 1
        agent = doc["agents"][0]
        assert agent["who"] == "Practitioner/dr-smith"
        assert agent["type"] == "author"


# ═══════════════════════════════════════════════════════════════════
# Section 3: from_fhir — Rich Resource (No Data Excluded)
# ═══════════════════════════════════════════════════════════════════


class TestFromProvenanceRich:
    """All FHIR Provenance fields must be preserved — nothing excluded."""

    def test_multiple_targets(self):
        doc, _ = from_fhir(_rich_provenance())
        refs = [t["reference"] for t in doc["targets"]]
        assert len(refs) == 2
        assert "Observation/obs-1" in refs
        assert "Observation/obs-2" in refs

    def test_multiple_agents_with_types(self):
        doc, _ = from_fhir(_rich_provenance())
        assert len(doc["agents"]) == 2
        types = {a["type"] for a in doc["agents"]}
        assert "author" in types
        assert "verifier" in types

    def test_entities_with_role_and_what(self):
        doc, _ = from_fhir(_rich_provenance())
        assert "entities" in doc
        assert len(doc["entities"]) == 2
        roles = {e["role"] for e in doc["entities"]}
        assert "source" in roles
        assert "derivation" in roles
        whats = {e["what"] for e in doc["entities"]}
        assert "DocumentReference/lab-report-1" in whats
        assert "Observation/original-obs" in whats

    def test_activity_preserved(self):
        doc, _ = from_fhir(_rich_provenance())
        assert "activity" in doc
        assert doc["activity"] == "UPDATE"

    def test_policy_preserved(self):
        doc, _ = from_fhir(_rich_provenance())
        assert "policy" in doc
        assert "http://example.org/policy/hipaa" in doc["policy"]

    def test_reason_preserved(self):
        doc, _ = from_fhir(_rich_provenance())
        assert "reason" in doc
        assert "TREAT" in doc["reason"]

    def test_location_preserved(self):
        doc, _ = from_fhir(_rich_provenance())
        assert doc["location"] == "Location/hospital-main"

    def test_delegation_on_behalf_of_preserved(self):
        doc, _ = from_fhir(_delegation_provenance())
        agent = doc["agents"][0]
        assert agent.get("onBehalfOf") == "Organization/hospital"

    def test_period_preserved(self):
        resource = _minimal_provenance()
        resource["period"] = {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-12-31T23:59:59Z",
        }
        doc, _ = from_fhir(resource)
        assert "period" in doc
        assert doc["period"]["start"] == "2024-01-01T00:00:00Z"
        assert doc["period"]["end"] == "2024-12-31T23:59:59Z"


# ═══════════════════════════════════════════════════════════════════
# Section 4: Uncertainty Budget Adjustments
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceUncertaintySignals:
    """Metadata signals must adjust the uncertainty budget correctly.

    Each test uses *comparative* assertions (signal-present vs
    signal-absent) rather than absolute thresholds, following the
    same philosophy used in test_fhir_bundle.py.
    """

    def test_recorded_present_lowers_uncertainty(self):
        """Recorded timestamp → lower u (documented when)."""
        u_with = from_fhir(_minimal_provenance())[0]["opinions"][0]["opinion"].uncertainty
        u_without = from_fhir(_no_recorded_provenance())[0]["opinions"][0]["opinion"].uncertainty
        assert u_with < u_without

    def test_multiple_agents_lower_uncertainty(self):
        """≥2 agents (corroboration) → lower u than single agent."""
        u_single = from_fhir(_minimal_provenance())[0]["opinions"][0]["opinion"].uncertainty
        u_multi = from_fhir(_rich_provenance())[0]["opinions"][0]["opinion"].uncertainty
        assert u_multi < u_single

    def test_no_agents_raises_uncertainty(self):
        """Empty agent list → higher u than one agent."""
        u_one = from_fhir(_minimal_provenance())[0]["opinions"][0]["opinion"].uncertainty
        u_none = from_fhir(_no_agent_provenance())[0]["opinions"][0]["opinion"].uncertainty
        assert u_none > u_one

    def test_agent_without_who_higher_uncertainty(self):
        """Anonymous agent (no who ref) → higher u than identified agent."""
        u_identified = from_fhir(_minimal_provenance())[0]["opinions"][0]["opinion"].uncertainty
        u_anon = from_fhir(_agent_without_who())[0]["opinions"][0]["opinion"].uncertainty
        assert u_anon > u_identified

    def test_entities_with_roles_lower_uncertainty(self):
        """Entities with explicit roles → more complete chain → lower u."""
        u_no_ent = from_fhir(_minimal_provenance())[0]["opinions"][0]["opinion"].uncertainty
        u_with_ent = from_fhir(_rich_provenance())[0]["opinions"][0]["opinion"].uncertainty
        assert u_with_ent < u_no_ent

    def test_uncertainty_always_in_valid_range(self):
        """u must be in [0, 1) for every signal combination."""
        fixtures = [
            _minimal_provenance(),
            _rich_provenance(),
            _no_agent_provenance(),
            _no_recorded_provenance(),
            _agent_without_who(),
            _delegation_provenance(),
        ]
        for resource in fixtures:
            doc, _ = from_fhir(resource)
            u = doc["opinions"][0]["opinion"].uncertainty
            assert 0.0 <= u < 1.0, (
                f"u={u} out of range for {resource['id']}"
            )

    def test_belief_always_positive(self):
        """b must be > 0 (base probability 0.75 ensures positive belief)."""
        fixtures = [
            _minimal_provenance(),
            _rich_provenance(),
            _no_agent_provenance(),
            _no_recorded_provenance(),
        ]
        for resource in fixtures:
            doc, _ = from_fhir(resource)
            b = doc["opinions"][0]["opinion"].belief
            assert b > 0.0, f"b={b} not positive for {resource['id']}"


# ═══════════════════════════════════════════════════════════════════
# Section 5: Extension Recovery
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceExtensionRecovery:
    """Exact opinion recovery from FHIR extensions on _recorded."""

    def test_extension_recovers_exact_opinion(self):
        exact_op = Opinion(
            belief=0.80, disbelief=0.05,
            uncertainty=0.15, base_rate=0.70,
        )
        resource = _minimal_provenance()
        resource["_recorded"] = {
            "extension": [opinion_to_fhir_extension(exact_op)]
        }

        doc, _ = from_fhir(resource)
        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        assert abs(entry["opinion"].belief - 0.80) < 1e-9
        assert abs(entry["opinion"].disbelief - 0.05) < 1e-9
        assert abs(entry["opinion"].uncertainty - 0.15) < 1e-9
        assert abs(entry["opinion"].base_rate - 0.70) < 1e-9

    def test_extension_takes_precedence_over_reconstruction(self):
        """Extension opinion overrides all reconstruction signals."""
        low_op = Opinion(
            belief=0.10, disbelief=0.80,
            uncertainty=0.10, base_rate=0.30,
        )
        resource = _rich_provenance()  # would normally yield high confidence
        resource["_recorded"] = {
            "extension": [opinion_to_fhir_extension(low_op)]
        }

        doc, _ = from_fhir(resource)
        op = doc["opinions"][0]["opinion"]
        assert abs(op.belief - 0.10) < 1e-9
        assert abs(op.disbelief - 0.80) < 1e-9

    def test_no_extension_uses_reconstruction(self):
        doc, _ = from_fhir(_minimal_provenance())
        assert doc["opinions"][0]["source"] == "reconstructed"


# ═══════════════════════════════════════════════════════════════════
# Section 6: to_fhir — Reverse Conversion
# ═══════════════════════════════════════════════════════════════════


class TestToProvenanceBasic:
    """to_fhir must reconstruct a valid FHIR R4 Provenance resource."""

    def test_resource_type_and_id(self):
        resource, report = to_fhir(_make_doc())
        assert resource["resourceType"] == "Provenance"
        assert resource["id"] == "prov-001"
        assert report.success

    def test_recorded_reconstructed(self):
        resource, _ = to_fhir(_make_doc())
        assert resource["recorded"] == "2024-06-15T10:30:00Z"

    def test_targets_reconstructed(self):
        doc = _make_doc(targets=[
            {"reference": "Observation/obs-1"},
            {"reference": "Observation/obs-2"},
        ])
        resource, _ = to_fhir(doc)
        assert "target" in resource
        assert len(resource["target"]) == 2
        refs = [t["reference"] for t in resource["target"]]
        assert "Observation/obs-1" in refs

    def test_agents_reconstructed_with_fhir_structure(self):
        doc = _make_doc(agents=[
            {"type": "author", "who": "Practitioner/dr-smith"},
            {"type": "verifier", "who": "Practitioner/dr-jones"},
        ])
        resource, _ = to_fhir(doc)
        assert "agent" in resource
        assert len(resource["agent"]) == 2
        for agent in resource["agent"]:
            assert "type" in agent
            assert "who" in agent
            # Type should be FHIR CodeableConcept format
            assert "coding" in agent["type"]

    def test_entities_reconstructed_with_fhir_structure(self):
        doc = _make_doc(entities=[
            {"role": "source", "what": "DocumentReference/doc-1"},
            {"role": "derivation", "what": "Observation/obs-orig"},
        ])
        resource, _ = to_fhir(doc)
        assert "entity" in resource
        assert len(resource["entity"]) == 2
        for entity in resource["entity"]:
            assert "role" in entity
            assert "what" in entity
            # what should be FHIR Reference format
            assert "reference" in entity["what"]

    def test_extension_embedded_on_recorded(self):
        resource, report = to_fhir(_make_doc())
        assert "_recorded" in resource
        ext_list = resource["_recorded"]["extension"]
        assert len(ext_list) == 1
        assert ext_list[0]["url"] == FHIR_EXTENSION_URL
        assert report.nodes_converted >= 1

    def test_activity_reconstructed(self):
        doc = _make_doc(activity="UPDATE")
        resource, _ = to_fhir(doc)
        assert "activity" in resource
        # Should be FHIR CodeableConcept format
        assert "coding" in resource["activity"]

    def test_policy_reconstructed(self):
        doc = _make_doc(policy=["http://example.org/policy/hipaa"])
        resource, _ = to_fhir(doc)
        assert resource["policy"] == ["http://example.org/policy/hipaa"]

    def test_reason_reconstructed(self):
        doc = _make_doc(reason=["TREAT"])
        resource, _ = to_fhir(doc)
        assert "reason" in resource

    def test_location_reconstructed(self):
        doc = _make_doc(location="Location/hospital-main")
        resource, _ = to_fhir(doc)
        assert "location" in resource
        assert resource["location"]["reference"] == "Location/hospital-main"

    def test_delegation_reconstructed(self):
        doc = _make_doc(agents=[
            {
                "type": "author",
                "who": "Practitioner/dr-smith",
                "onBehalfOf": "Organization/hospital",
            },
        ])
        resource, _ = to_fhir(doc)
        agent = resource["agent"][0]
        assert "onBehalfOf" in agent
        assert agent["onBehalfOf"]["reference"] == "Organization/hospital"

    def test_period_reconstructed(self):
        doc = _make_doc(period={
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-12-31T23:59:59Z",
        })
        resource, _ = to_fhir(doc)
        assert resource["period"]["start"] == "2024-01-01T00:00:00Z"
        assert resource["period"]["end"] == "2024-12-31T23:59:59Z"


# ═══════════════════════════════════════════════════════════════════
# Section 7: Round-Trip Fidelity
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTrip:
    """from_fhir → to_fhir must preserve key data."""

    def test_round_trip_minimal(self):
        original = _minimal_provenance()
        doc, _ = from_fhir(original)
        restored, _ = to_fhir(doc)

        assert restored["resourceType"] == "Provenance"
        assert restored["id"] == "prov-001"
        assert restored["recorded"] == "2024-06-15T10:30:00Z"
        assert len(restored.get("target", [])) == 1
        assert len(restored.get("agent", [])) == 1
        assert "_recorded" in restored

    def test_round_trip_extension_preserves_exact_opinion(self):
        exact_op = Opinion(
            belief=0.70, disbelief=0.10,
            uncertainty=0.20, base_rate=0.65,
        )
        original = _minimal_provenance()
        original["_recorded"] = {
            "extension": [opinion_to_fhir_extension(exact_op)]
        }

        doc, _ = from_fhir(original)
        assert doc["opinions"][0]["source"] == "extension"

        restored, _ = to_fhir(doc)
        doc2, _ = from_fhir(restored)
        op2 = doc2["opinions"][0]["opinion"]

        assert abs(op2.belief - 0.70) < 1e-9
        assert abs(op2.disbelief - 0.10) < 1e-9
        assert abs(op2.uncertainty - 0.20) < 1e-9
        assert abs(op2.base_rate - 0.65) < 1e-9

    def test_round_trip_rich_preserves_all_metadata(self):
        original = _rich_provenance()
        doc, _ = from_fhir(original)
        restored, _ = to_fhir(doc)

        assert len(restored.get("target", [])) == 2
        assert len(restored.get("agent", [])) == 2
        assert len(restored.get("entity", [])) == 2
        assert "activity" in restored
        assert "policy" in restored

    def test_round_trip_delegation(self):
        original = _delegation_provenance()
        doc, _ = from_fhir(original)
        restored, _ = to_fhir(doc)

        assert len(restored.get("agent", [])) == 1
        agent = restored["agent"][0]
        assert "onBehalfOf" in agent


# ═══════════════════════════════════════════════════════════════════
# Section 8: fhir_provenance_to_prov_o Bridge
# ═══════════════════════════════════════════════════════════════════

# The PROV namespace used by owl_interop.py
_PROV = "http://www.w3.org/ns/prov#"
_JSONLD_EX = "http://www.w3.org/ns/jsonld-ex/"
_RDFS = "http://www.w3.org/2000/01/rdf-schema#"
_XSD = "http://www.w3.org/2001/XMLSchema#"


def _find_nodes_by_type(graph: list, type_iri: str) -> list[dict]:
    """Find all nodes in a PROV-O graph with the given @type."""
    result = []
    for n in graph:
        if not isinstance(n, dict):
            continue
        t = n.get("@type", [])
        if isinstance(t, str):
            t = [t]
        if type_iri in t:
            result.append(n)
    return result


def _graph_has_property(graph: list, prop_iri: str) -> bool:
    """Check whether any node in the graph has the given property."""
    return any(prop_iri in n for n in graph if isinstance(n, dict))


class TestFhirProvenanceToPROVO:
    """fhir_provenance_to_prov_o bridges the jsonld-ex doc to W3C PROV-O."""

    def test_importable(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o  # noqa: F811
        assert callable(fhir_provenance_to_prov_o)

    def test_returns_doc_and_report(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, report = fhir_provenance_to_prov_o(doc)

        assert isinstance(prov_o, dict)
        assert isinstance(report, ConversionReport)
        assert report.success

    def test_prov_context_present(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        ctx = prov_o.get("@context", {})
        assert "prov" in ctx or _PROV in str(ctx)

    def test_graph_present(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        assert "@graph" in prov_o
        assert len(prov_o["@graph"]) >= 1

    def test_target_maps_to_prov_entity(self):
        """FHIR target references → prov:Entity nodes."""
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        entities = _find_nodes_by_type(prov_o["@graph"], f"{_PROV}Entity")
        assert len(entities) >= 1

    def test_practitioner_agent_maps_to_prov_person(self):
        """Agent with Practitioner/… who → prov:Person (or prov:Agent)."""
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)
        graph = prov_o["@graph"]

        # Accept either prov:Person or prov:Agent for Practitioner
        agent_nodes = (
            _find_nodes_by_type(graph, f"{_PROV}Person")
            + _find_nodes_by_type(graph, f"{_PROV}Agent")
        )
        assert len(agent_nodes) >= 1

    def test_device_agent_maps_to_prov_software_agent(self):
        """Agent with Device/… who → prov:SoftwareAgent."""
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        resource = {
            "resourceType": "Provenance",
            "id": "prov-device",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [
                {
                    "type": {"coding": [{"code": "assembler"}]},
                    "who": {"reference": "Device/lab-analyzer"},
                }
            ],
        }
        doc, _ = from_fhir(resource)
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        sw_agents = _find_nodes_by_type(
            prov_o["@graph"], f"{_PROV}SoftwareAgent"
        )
        assert len(sw_agents) >= 1

    def test_recorded_maps_to_generated_at_time(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        assert _graph_has_property(prov_o["@graph"], f"{_PROV}generatedAtTime")

    def test_activity_maps_to_prov_activity(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_rich_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        activities = _find_nodes_by_type(
            prov_o["@graph"], f"{_PROV}Activity"
        )
        assert len(activities) >= 1

    def test_delegation_maps_to_acted_on_behalf_of(self):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_delegation_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        assert _graph_has_property(
            prov_o["@graph"], f"{_PROV}actedOnBehalfOf"
        )

    def test_sl_opinion_preserved_as_confidence(self):
        """SL opinion from the jsonld-ex doc should appear as metadata."""
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        doc, _ = from_fhir(_minimal_provenance())
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        has_confidence = _graph_has_property(
            prov_o["@graph"], f"{_JSONLD_EX}confidence"
        ) or _graph_has_property(
            prov_o["@graph"], f"{_JSONLD_EX}opinion"
        )
        assert has_confidence


# ═══════════════════════════════════════════════════════════════════
# Section 9: Entity Role → PROV-O Mapping Completeness
# ═══════════════════════════════════════════════════════════════════


class TestEntityRoleMappingCompleteness:
    """All five FHIR R4 Provenance entity roles must have PROV-O mappings.

    FHIR R4 Provenance.entity.role value-set (required):
      derivation → prov:wasDerivedFrom
      revision   → prov:wasRevisionOf
      quotation  → prov:wasQuotedFrom
      source     → prov:hadPrimarySource
      removal    → prov:wasInvalidatedBy
    """

    ROLE_TO_PROV = {
        "derivation": "wasDerivedFrom",
        "revision": "wasRevisionOf",
        "quotation": "wasQuotedFrom",
        "source": "hadPrimarySource",
        "removal": "wasInvalidatedBy",
    }

    @pytest.mark.parametrize("role,prov_prop", ROLE_TO_PROV.items())
    def test_entity_role_maps_to_prov_property(self, role, prov_prop):
        from jsonld_ex.fhir_interop import fhir_provenance_to_prov_o

        resource = {
            "resourceType": "Provenance",
            "id": f"prov-role-{role}",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-smith"}}],
            "entity": [
                {
                    "role": role,
                    "what": {"reference": f"DocumentReference/doc-{role}"},
                }
            ],
        }
        doc, _ = from_fhir(resource)
        prov_o, _ = fhir_provenance_to_prov_o(doc)

        assert _graph_has_property(prov_o["@graph"], f"{_PROV}{prov_prop}"), (
            f"FHIR entity role '{role}' should map to prov:{prov_prop}"
        )


# ═══════════════════════════════════════════════════════════════════
# Section 10: Edge Cases
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceEdgeCases:

    def test_empty_agent_list(self):
        doc, report = from_fhir(_no_agent_provenance())
        assert report.success
        assert len(doc["opinions"]) >= 1
        assert doc.get("agents", []) == [] or doc.get("agents") is not None

    def test_agent_with_no_type_coding(self):
        """Agent without type coding should still be preserved."""
        resource = {
            "resourceType": "Provenance",
            "id": "prov-no-type",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [
                {"who": {"reference": "Practitioner/dr-smith"}},
            ],
        }
        doc, report = from_fhir(resource)
        assert report.success
        assert len(doc["agents"]) == 1
        assert doc["agents"][0]["who"] == "Practitioner/dr-smith"

    def test_entity_with_no_role(self):
        """Entity without role should be preserved (role = None)."""
        resource = {
            "resourceType": "Provenance",
            "id": "prov-no-role",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-smith"}}],
            "entity": [
                {"what": {"reference": "DocumentReference/doc-1"}},
            ],
        }
        doc, report = from_fhir(resource)
        assert report.success
        assert len(doc["entities"]) == 1

    def test_no_target(self):
        """Provenance without targets should not crash."""
        resource = {
            "resourceType": "Provenance",
            "id": "prov-no-target",
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-smith"}}],
        }
        doc, report = from_fhir(resource)
        assert report.success
        assert doc.get("targets", []) == [] or doc.get("targets") is not None

    def test_multiple_entities_same_role(self):
        resource = {
            "resourceType": "Provenance",
            "id": "prov-multi-source",
            "target": [{"reference": "DiagnosticReport/dr-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-smith"}}],
            "entity": [
                {"role": "source", "what": {"reference": "Observation/obs-1"}},
                {"role": "source", "what": {"reference": "Observation/obs-2"}},
                {"role": "source", "what": {"reference": "Observation/obs-3"}},
            ],
        }
        doc, report = from_fhir(resource)
        assert report.success
        assert len(doc["entities"]) == 3

    def test_organization_agent(self):
        resource = {
            "resourceType": "Provenance",
            "id": "prov-org",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-06-15T10:30:00Z",
            "agent": [
                {
                    "type": {"coding": [{"code": "custodian"}]},
                    "who": {"reference": "Organization/hospital-main"},
                }
            ],
        }
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["agents"][0]["who"] == "Organization/hospital-main"

    def test_no_opinions_doc_still_valid_for_to_fhir(self):
        """Doc with empty opinions list should still produce a valid resource."""
        doc = _make_doc(opinions=[])
        resource, report = to_fhir(doc)
        assert report.success
        assert resource["resourceType"] == "Provenance"
        # No extension when no opinions
        assert "_recorded" not in resource or report.nodes_converted == 0


# ═══════════════════════════════════════════════════════════════════
# Section 11: Existing-Handler Regression Guard
# ═══════════════════════════════════════════════════════════════════


class TestExistingHandlersUnaffected:
    """Adding Provenance must not break any existing handler."""

    @pytest.mark.parametrize(
        "resource_type",
        [
            "RiskAssessment",
            "Observation",
            "Condition",
            "Consent",
            "Immunization",
            "Procedure",
        ],
    )
    def test_existing_type_still_dispatches(self, resource_type):
        """Each existing type must still be in SUPPORTED_RESOURCE_TYPES."""
        assert resource_type in SUPPORTED_RESOURCE_TYPES
