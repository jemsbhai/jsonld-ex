"""Tests for Provenance round-trip CONTENT fidelity.

RED PHASE: These tests strengthen the existing Provenance round-trip
coverage by verifying that array field *content* (reference values,
role codes, type codes) survives from_fhir → to_fhir, not just counts.

Existing test_fhir_provenance.py verifies:
  ✅ Array lengths (target=2, agent=2, entity=2)
  ✅ recorded value
  ✅ activity/policy presence
  ✅ onBehalfOf presence

This file adds:
  ❌→✅ target reference values survive round-trip
  ❌→✅ agent type codes survive round-trip
  ❌→✅ agent who references survive round-trip
  ❌→✅ entity roles survive round-trip
  ❌→✅ entity what references survive round-trip
  ❌→✅ onBehalfOf reference value survives round-trip
  ❌→✅ activity code value survives round-trip
  ❌→✅ policy URI values survive round-trip
  ❌→✅ reason code values survive round-trip
  ❌→✅ location reference value survives round-trip
  ❌→✅ period start/end values survive round-trip
"""

from jsonld_ex.fhir_interop import from_fhir, to_fhir


# ═══════════════════════════════════════════════════════════════════
# Fixture
# ═══════════════════════════════════════════════════════════════════


def _rich_provenance():
    """Full-featured FHIR R4 Provenance with every optional field."""
    return {
        "resourceType": "Provenance",
        "id": "prov-rt-rich",
        "target": [
            {"reference": "Observation/obs-alpha"},
            {"reference": "Observation/obs-beta"},
        ],
        "recorded": "2024-07-01T09:00:00Z",
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
                "who": {"reference": "Practitioner/dr-alice"},
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
                "who": {"reference": "Practitioner/dr-bob"},
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
        "period": {
            "start": "2024-01-01T00:00:00Z",
            "end": "2024-12-31T23:59:59Z",
        },
    }


def _delegation_provenance():
    """Provenance with agent delegation (onBehalfOf)."""
    return {
        "resourceType": "Provenance",
        "id": "prov-rt-deleg",
        "target": [{"reference": "Observation/obs-1"}],
        "recorded": "2024-07-01T09:00:00Z",
        "agent": [
            {
                "type": {
                    "coding": [{"code": "author"}]
                },
                "who": {"reference": "Practitioner/dr-alice"},
                "onBehalfOf": {"reference": "Organization/hospital-main"},
            }
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Round-trip helpers
# ═══════════════════════════════════════════════════════════════════


def _round_trip(resource):
    """from_fhir → to_fhir round-trip, returning the restored FHIR resource."""
    doc, report_from = from_fhir(resource)
    assert report_from.success, f"from_fhir failed: {report_from.warnings}"
    restored, report_to = to_fhir(doc)
    assert report_to.success, f"to_fhir failed: {report_to.warnings}"
    return restored


# ═══════════════════════════════════════════════════════════════════
# Target reference values
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTripTargetContent:
    """Target reference strings must survive round-trip, not just count."""

    def test_single_target_reference_survives(self):
        resource = {
            "resourceType": "Provenance",
            "id": "prov-tgt-1",
            "target": [{"reference": "Observation/obs-123"}],
            "recorded": "2024-07-01T09:00:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-x"}}],
        }
        restored = _round_trip(resource)

        targets = restored.get("target", [])
        assert len(targets) == 1
        assert targets[0]["reference"] == "Observation/obs-123"

    def test_multiple_target_references_survive(self):
        restored = _round_trip(_rich_provenance())

        refs = {t["reference"] for t in restored.get("target", [])}
        assert "Observation/obs-alpha" in refs
        assert "Observation/obs-beta" in refs


# ═══════════════════════════════════════════════════════════════════
# Agent type codes and who references
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTripAgentContent:
    """Agent type codes and who references must survive round-trip."""

    def test_single_agent_who_survives(self):
        resource = {
            "resourceType": "Provenance",
            "id": "prov-agt-1",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-07-01T09:00:00Z",
            "agent": [
                {
                    "type": {"coding": [{"code": "author"}]},
                    "who": {"reference": "Practitioner/dr-alice"},
                }
            ],
        }
        restored = _round_trip(resource)

        agents = restored.get("agent", [])
        assert len(agents) == 1
        assert agents[0]["who"]["reference"] == "Practitioner/dr-alice"

    def test_agent_type_code_survives(self):
        restored = _round_trip(_rich_provenance())

        agents = restored.get("agent", [])
        type_codes = set()
        for agent in agents:
            for coding in agent.get("type", {}).get("coding", []):
                type_codes.add(coding.get("code"))
        assert "author" in type_codes
        assert "verifier" in type_codes

    def test_agent_who_references_survive(self):
        restored = _round_trip(_rich_provenance())

        who_refs = {a["who"]["reference"] for a in restored.get("agent", [])}
        assert "Practitioner/dr-alice" in who_refs
        assert "Practitioner/dr-bob" in who_refs

    def test_on_behalf_of_reference_value_survives(self):
        """onBehalfOf reference VALUE must survive, not just presence."""
        restored = _round_trip(_delegation_provenance())

        agent = restored["agent"][0]
        assert agent["onBehalfOf"]["reference"] == "Organization/hospital-main"


# ═══════════════════════════════════════════════════════════════════
# Entity roles and what references
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTripEntityContent:
    """Entity roles and what references must survive round-trip."""

    def test_entity_roles_survive(self):
        restored = _round_trip(_rich_provenance())

        entities = restored.get("entity", [])
        roles = {e["role"] for e in entities}
        assert "source" in roles
        assert "derivation" in roles

    def test_entity_what_references_survive(self):
        restored = _round_trip(_rich_provenance())

        entities = restored.get("entity", [])
        what_refs = {e["what"]["reference"] for e in entities}
        assert "DocumentReference/lab-report-1" in what_refs
        assert "Observation/original-obs" in what_refs

    def test_single_entity_content_survives(self):
        resource = {
            "resourceType": "Provenance",
            "id": "prov-ent-1",
            "target": [{"reference": "Observation/obs-1"}],
            "recorded": "2024-07-01T09:00:00Z",
            "agent": [{"who": {"reference": "Practitioner/dr-x"}}],
            "entity": [
                {
                    "role": "quotation",
                    "what": {"reference": "DocumentReference/doc-42"},
                }
            ],
        }
        restored = _round_trip(resource)

        entities = restored.get("entity", [])
        assert len(entities) == 1
        assert entities[0]["role"] == "quotation"
        assert entities[0]["what"]["reference"] == "DocumentReference/doc-42"


# ═══════════════════════════════════════════════════════════════════
# Scalar metadata fields — content values
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTripMetadataContent:
    """Activity, policy, reason, location, period values must survive."""

    def test_activity_code_value_survives(self):
        restored = _round_trip(_rich_provenance())

        activity = restored.get("activity", {})
        codes = [c.get("code") for c in activity.get("coding", [])]
        assert "UPDATE" in codes

    def test_policy_uri_value_survives(self):
        restored = _round_trip(_rich_provenance())

        policies = restored.get("policy", [])
        assert "http://example.org/policy/hipaa" in policies

    def test_reason_code_value_survives(self):
        restored = _round_trip(_rich_provenance())

        reason = restored.get("reason", [])
        # reason could be reconstructed as CodeableConcept array
        found_treat = False
        for r in reason:
            if isinstance(r, dict):
                for coding in r.get("coding", []):
                    if coding.get("code") == "TREAT":
                        found_treat = True
            elif isinstance(r, str) and r == "TREAT":
                found_treat = True
        assert found_treat, f"TREAT not found in reason: {reason}"

    def test_location_reference_value_survives(self):
        restored = _round_trip(_rich_provenance())

        location = restored.get("location", {})
        assert location.get("reference") == "Location/hospital-main"

    def test_period_start_end_values_survive(self):
        restored = _round_trip(_rich_provenance())

        period = restored.get("period", {})
        assert period.get("start") == "2024-01-01T00:00:00Z"
        assert period.get("end") == "2024-12-31T23:59:59Z"

    def test_recorded_value_survives(self):
        restored = _round_trip(_rich_provenance())
        assert restored.get("recorded") == "2024-07-01T09:00:00Z"


# ═══════════════════════════════════════════════════════════════════
# Full fidelity: every field in one combined assertion
# ═══════════════════════════════════════════════════════════════════


class TestProvenanceRoundTripFullFidelity:
    """Single test verifying ALL content survives the full round-trip."""

    def test_all_rich_provenance_content_survives(self):
        """Every field in the rich fixture must survive with correct values."""
        original = _rich_provenance()
        restored = _round_trip(original)

        # Identity
        assert restored["resourceType"] == "Provenance"
        assert restored["id"] == "prov-rt-rich"
        assert restored["recorded"] == "2024-07-01T09:00:00Z"

        # Targets — values, not just count
        target_refs = {t["reference"] for t in restored["target"]}
        assert target_refs == {"Observation/obs-alpha", "Observation/obs-beta"}

        # Agents — type codes AND who references
        assert len(restored["agent"]) == 2
        agent_map = {}
        for a in restored["agent"]:
            code = None
            for c in a.get("type", {}).get("coding", []):
                code = c.get("code")
            who = a["who"]["reference"]
            agent_map[code] = who
        assert agent_map.get("author") == "Practitioner/dr-alice"
        assert agent_map.get("verifier") == "Practitioner/dr-bob"

        # Entities — roles AND what references
        assert len(restored["entity"]) == 2
        entity_map = {}
        for e in restored["entity"]:
            entity_map[e["role"]] = e["what"]["reference"]
        assert entity_map.get("source") == "DocumentReference/lab-report-1"
        assert entity_map.get("derivation") == "Observation/original-obs"

        # Activity
        activity_codes = [
            c["code"] for c in restored["activity"].get("coding", [])
        ]
        assert "UPDATE" in activity_codes

        # Policy
        assert "http://example.org/policy/hipaa" in restored["policy"]

        # Location
        assert restored["location"]["reference"] == "Location/hospital-main"

        # Period
        assert restored["period"]["start"] == "2024-01-01T00:00:00Z"
        assert restored["period"]["end"] == "2024-12-31T23:59:59Z"
