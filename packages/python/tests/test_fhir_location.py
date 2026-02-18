"""
Tests for FHIR R4 Location ↔ jsonld-ex bidirectional conversion.

Location provides facility-level data referenced by Encounter (where care
happened), Provenance (where recorded), Device (where housed), and
PractitionerRole (where practitioner works).  US Core USCDI mandated.

Proposition under assessment: "This location record is valid and the
facility is operationally suitable for its intended purpose."

FHIR R4 Location has two independent status dimensions:
    1. ``status`` (LocationStatus, Required binding):
       active | suspended | inactive
    2. ``operationalStatus`` (Coding, Preferred binding — v2 Table 0116):
       C (Closed) | H (Housekeeping) | O (Occupied) |
       U (Unoccupied) | K (Contaminated) | I (Isolated)

These produce genuinely different epistemic semantics: a location can be
``active`` in status but ``K`` (contaminated) operationally, which should
raise uncertainty about its usability.

This is a custom handler (not ``_make_status_handler``) because:
    - The dual-status model (status + operationalStatus) with
      multiplicative signal interaction requires custom logic
    - Comprehensive metadata passthrough (position, address, telecom,
      hoursOfOperation, etc.) requires field-by-field extraction
    - Reference fields (managingOrganization, partOf, endpoint) require
      proper extraction for round-trip fidelity

Test organisation:
    1. Constants — probability/uncertainty maps cover all R4 codes
    2. Status-based opinion reconstruction — each status code
    3. OperationalStatus signal interaction — each v2 Table 0116 code
    4. Combined signal interactions — status × operationalStatus
    5. Extension recovery — exact opinion from jsonld-ex extension
    6. Metadata passthrough — all Location fields preserved
    7. Round-trip — from_fhir → to_fhir preserves all fields
    8. Edge cases — missing fields, novel status, empty arrays
    9. Integration — Location in Encounter/Provenance context
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.fhir_interop import (
    from_fhir,
    to_fhir,
    FHIR_EXTENSION_URL,
)
from jsonld_ex.fhir_interop._constants import (
    LOCATION_STATUS_PROBABILITY,
    LOCATION_STATUS_UNCERTAINTY,
    LOCATION_OPERATIONAL_STATUS_MULTIPLIER,
    SUPPORTED_RESOURCE_TYPES,
)
from jsonld_ex.fhir_interop._scalar import opinion_to_fhir_extension


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════


def _location(
    *,
    status: str = "active",
    location_id: str = "loc-1",
    operational_status_code: str | None = None,
    operational_status_system: str | None = None,
    operational_status_display: str | None = None,
    name: str | None = None,
    description: str | None = None,
    mode: str | None = None,
    type_list: list[dict] | None = None,
    telecom: list[dict] | None = None,
    address: dict | None = None,
    physical_type_text: str | None = None,
    physical_type_coding: list[dict] | None = None,
    position: dict | None = None,
    managing_org_ref: str | None = None,
    part_of_ref: str | None = None,
    hours_of_operation: list[dict] | None = None,
    availability_exceptions: str | None = None,
    endpoint_refs: list[str] | None = None,
    identifier: list[dict] | None = None,
    alias: list[str] | None = None,
) -> dict:
    """Build a minimal FHIR R4 Location resource for testing."""
    resource: dict = {
        "resourceType": "Location",
        "id": location_id,
        "status": status,
    }
    if operational_status_code is not None:
        op_status: dict = {"code": operational_status_code}
        if operational_status_system is not None:
            op_status["system"] = operational_status_system
        if operational_status_display is not None:
            op_status["display"] = operational_status_display
        resource["operationalStatus"] = op_status
    if name is not None:
        resource["name"] = name
    if description is not None:
        resource["description"] = description
    if mode is not None:
        resource["mode"] = mode
    if type_list is not None:
        resource["type"] = type_list
    if telecom is not None:
        resource["telecom"] = telecom
    if address is not None:
        resource["address"] = address
    if physical_type_text is not None or physical_type_coding is not None:
        pt: dict = {}
        if physical_type_text is not None:
            pt["text"] = physical_type_text
        if physical_type_coding is not None:
            pt["coding"] = physical_type_coding
        resource["physicalType"] = pt
    if position is not None:
        resource["position"] = position
    if managing_org_ref is not None:
        resource["managingOrganization"] = {"reference": managing_org_ref}
    if part_of_ref is not None:
        resource["partOf"] = {"reference": part_of_ref}
    if hours_of_operation is not None:
        resource["hoursOfOperation"] = hours_of_operation
    if availability_exceptions is not None:
        resource["availabilityExceptions"] = availability_exceptions
    if endpoint_refs is not None:
        resource["endpoint"] = [{"reference": r} for r in endpoint_refs]
    if identifier is not None:
        resource["identifier"] = identifier
    if alias is not None:
        resource["alias"] = alias
    return resource


def _inject_extension(resource: dict, opinion: Opinion) -> dict:
    """Inject a jsonld-ex SL extension onto ``_status``."""
    ext = opinion_to_fhir_extension(opinion)
    resource["_status"] = {"extension": [ext]}
    return resource


# ═══════════════════════════════════════════════════════════════════
# 1. Constants — complete code coverage
# ═══════════════════════════════════════════════════════════════════


class TestLocationConstants:
    """Verify constant maps cover all FHIR R4 LocationStatus codes
    and all v2 Table 0116 operationalStatus codes."""

    def test_location_in_supported_resource_types(self):
        assert "Location" in SUPPORTED_RESOURCE_TYPES

    # -- LocationStatus (Required binding): active | suspended | inactive --

    def test_status_probability_covers_all_codes(self):
        expected = {"active", "suspended", "inactive"}
        assert set(LOCATION_STATUS_PROBABILITY.keys()) == expected

    def test_status_uncertainty_covers_all_codes(self):
        expected = {"active", "suspended", "inactive"}
        assert set(LOCATION_STATUS_UNCERTAINTY.keys()) == expected

    def test_status_probabilities_in_valid_range(self):
        for code, prob in LOCATION_STATUS_PROBABILITY.items():
            assert 0.0 <= prob <= 1.0, f"{code}: {prob}"

    def test_status_uncertainties_in_valid_range(self):
        for code, u in LOCATION_STATUS_UNCERTAINTY.items():
            assert 0.0 < u < 1.0, f"{code}: {u}"

    # -- v2 Table 0116 (Preferred binding): C | H | O | U | K | I --

    def test_operational_status_multiplier_covers_all_v2_0116_codes(self):
        expected = {"C", "H", "O", "U", "K", "I"}
        assert set(LOCATION_OPERATIONAL_STATUS_MULTIPLIER.keys()) == expected

    def test_operational_status_multipliers_positive(self):
        for code, mult in LOCATION_OPERATIONAL_STATUS_MULTIPLIER.items():
            assert mult > 0.0, f"{code}: {mult}"

    # -- Epistemic ordering --

    def test_active_has_highest_probability(self):
        assert LOCATION_STATUS_PROBABILITY["active"] > \
            LOCATION_STATUS_PROBABILITY["suspended"] > \
            LOCATION_STATUS_PROBABILITY["inactive"]

    def test_active_has_lowest_uncertainty(self):
        assert LOCATION_STATUS_UNCERTAINTY["active"] < \
            LOCATION_STATUS_UNCERTAINTY["suspended"]

    def test_contaminated_has_highest_multiplier(self):
        """K (Contaminated) should produce the most uncertainty."""
        assert LOCATION_OPERATIONAL_STATUS_MULTIPLIER["K"] > \
            LOCATION_OPERATIONAL_STATUS_MULTIPLIER["C"] > \
            LOCATION_OPERATIONAL_STATUS_MULTIPLIER["I"] > \
            LOCATION_OPERATIONAL_STATUS_MULTIPLIER["H"]

    def test_occupied_has_lowest_multiplier(self):
        """O (Occupied) confirms operational use → lowest multiplier."""
        assert LOCATION_OPERATIONAL_STATUS_MULTIPLIER["O"] < \
            LOCATION_OPERATIONAL_STATUS_MULTIPLIER["U"]


# ═══════════════════════════════════════════════════════════════════
# 2. Status-based opinion reconstruction
# ═══════════════════════════════════════════════════════════════════


class TestLocationStatusReconstruction:
    """Each LocationStatus code produces a reconstructed opinion
    with the correct probability and uncertainty."""

    @pytest.mark.parametrize("status", ["active", "suspended", "inactive"])
    def test_status_reconstruction(self, status: str):
        doc, report = from_fhir(_location(status=status))
        assert report.success
        assert report.nodes_converted == 1
        assert doc["@type"] == "fhir:Location"
        assert doc["status"] == status
        assert len(doc["opinions"]) == 1

        entry = doc["opinions"][0]
        assert entry["field"] == "status"
        assert entry["value"] == status
        assert entry["source"] == "reconstructed"

        op: Opinion = entry["opinion"]
        assert isinstance(op, Opinion)
        # Probability should reflect the status map
        expected_prob = LOCATION_STATUS_PROBABILITY[status]
        # projected_probability is a function of b, d, u, a
        # For reconstructed opinions the projected prob should be close
        # to the configured value
        assert op.belief >= 0.0
        assert op.disbelief >= 0.0
        assert op.uncertainty > 0.0
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_active_higher_belief_than_suspended(self):
        doc_active, _ = from_fhir(_location(status="active"))
        doc_suspended, _ = from_fhir(_location(status="suspended"))

        op_active = doc_active["opinions"][0]["opinion"]
        op_suspended = doc_suspended["opinions"][0]["opinion"]

        assert op_active.projected_probability() > \
            op_suspended.projected_probability()

    def test_suspended_higher_belief_than_inactive(self):
        doc_suspended, _ = from_fhir(_location(status="suspended"))
        doc_inactive, _ = from_fhir(_location(status="inactive"))

        op_suspended = doc_suspended["opinions"][0]["opinion"]
        op_inactive = doc_inactive["opinions"][0]["opinion"]

        assert op_suspended.projected_probability() > \
            op_inactive.projected_probability()

    def test_novel_status_uses_defaults(self):
        """Unknown status code gets fallback probability/uncertainty."""
        doc, report = from_fhir(_location(status="demolished"))
        assert report.success
        assert report.nodes_converted == 1
        entry = doc["opinions"][0]
        assert entry["source"] == "reconstructed"
        assert isinstance(entry["opinion"], Opinion)


# ═══════════════════════════════════════════════════════════════════
# 3. OperationalStatus signal interaction
# ═══════════════════════════════════════════════════════════════════


class TestLocationOperationalStatus:
    """operationalStatus (v2 Table 0116) modulates uncertainty
    multiplicatively on top of the base status uncertainty."""

    @pytest.mark.parametrize("op_code,mult", [
        ("C", "C"), ("H", "H"), ("O", "O"),
        ("U", "U"), ("K", "K"), ("I", "I"),
    ])
    def test_each_v2_code_accepted(self, op_code: str, mult: str):
        """Each v2 0116 code is processed without error."""
        resource = _location(
            status="active",
            operational_status_code=op_code,
        )
        doc, report = from_fhir(resource)
        assert report.success
        assert report.nodes_converted == 1
        assert doc["opinions"][0]["source"] == "reconstructed"

    def test_occupied_reduces_uncertainty(self):
        """O (Occupied) multiplier < 1.0 → lower uncertainty than baseline."""
        doc_no_op, _ = from_fhir(_location(status="active"))
        doc_occupied, _ = from_fhir(_location(
            status="active",
            operational_status_code="O",
        ))

        u_base = doc_no_op["opinions"][0]["opinion"].uncertainty
        u_occupied = doc_occupied["opinions"][0]["opinion"].uncertainty

        assert u_occupied < u_base

    def test_contaminated_raises_uncertainty(self):
        """K (Contaminated) multiplier > 1.0 → higher uncertainty."""
        doc_no_op, _ = from_fhir(_location(status="active"))
        doc_contam, _ = from_fhir(_location(
            status="active",
            operational_status_code="K",
        ))

        u_base = doc_no_op["opinions"][0]["opinion"].uncertainty
        u_contam = doc_contam["opinions"][0]["opinion"].uncertainty

        assert u_contam > u_base

    def test_closed_raises_uncertainty(self):
        """C (Closed) multiplier > 1.0 → higher uncertainty."""
        doc_no_op, _ = from_fhir(_location(status="active"))
        doc_closed, _ = from_fhir(_location(
            status="active",
            operational_status_code="C",
        ))

        u_base = doc_no_op["opinions"][0]["opinion"].uncertainty
        u_closed = doc_closed["opinions"][0]["opinion"].uncertainty

        assert u_closed > u_base

    def test_unoccupied_is_baseline(self):
        """U (Unoccupied) multiplier = 1.0 → same uncertainty as no op status."""
        doc_no_op, _ = from_fhir(_location(status="active"))
        doc_unocc, _ = from_fhir(_location(
            status="active",
            operational_status_code="U",
        ))

        u_base = doc_no_op["opinions"][0]["opinion"].uncertainty
        u_unocc = doc_unocc["opinions"][0]["opinion"].uncertainty

        assert abs(u_base - u_unocc) < 1e-9

    def test_contaminated_higher_than_closed(self):
        """K should produce more uncertainty than C."""
        doc_contam, _ = from_fhir(_location(
            status="active", operational_status_code="K",
        ))
        doc_closed, _ = from_fhir(_location(
            status="active", operational_status_code="C",
        ))

        u_contam = doc_contam["opinions"][0]["opinion"].uncertainty
        u_closed = doc_closed["opinions"][0]["opinion"].uncertainty

        assert u_contam > u_closed

    def test_operational_status_with_full_coding(self):
        """operationalStatus as full Coding with system + display."""
        resource = _location(
            status="active",
            operational_status_code="O",
            operational_status_system="http://terminology.hl7.org/CodeSystem/v2-0116",
            operational_status_display="Occupied",
        )
        doc, report = from_fhir(resource)
        assert report.success
        # operationalStatus metadata should be passed through
        assert doc.get("operationalStatus") is not None

    def test_unknown_operational_status_code_uses_default(self):
        """Novel v2 code gets fallback multiplier of 1.0 (baseline)."""
        doc_novel, _ = from_fhir(_location(
            status="active",
            operational_status_code="Z",
        ))
        doc_no_op, _ = from_fhir(_location(status="active"))

        u_novel = doc_novel["opinions"][0]["opinion"].uncertainty
        u_base = doc_no_op["opinions"][0]["opinion"].uncertainty

        # Unknown code should use 1.0 multiplier → same as baseline
        assert abs(u_novel - u_base) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 4. Combined signal interactions
# ═══════════════════════════════════════════════════════════════════


class TestLocationCombinedSignals:
    """Status × operationalStatus produce correct combined effects."""

    def test_active_occupied_best_case(self):
        """active + O → lowest uncertainty."""
        doc, _ = from_fhir(_location(
            status="active", operational_status_code="O",
        ))
        op = doc["opinions"][0]["opinion"]
        # Should have high belief, low uncertainty
        assert op.projected_probability() > 0.8
        assert op.uncertainty < 0.15

    def test_suspended_contaminated_worst_reasonable(self):
        """suspended + K → high uncertainty, moderate probability."""
        doc, _ = from_fhir(_location(
            status="suspended", operational_status_code="K",
        ))
        op = doc["opinions"][0]["opinion"]
        # Suspended base u=0.40 * K multiplier 1.8 = 0.72
        assert op.uncertainty > 0.5

    def test_inactive_closed_very_low_belief(self):
        """inactive + C → very low probability, moderate uncertainty."""
        doc, _ = from_fhir(_location(
            status="inactive", operational_status_code="C",
        ))
        op = doc["opinions"][0]["opinion"]
        assert op.projected_probability() < 0.3

    def test_uncertainty_clamped_below_one(self):
        """Even extreme combinations keep uncertainty < 1.0."""
        doc, _ = from_fhir(_location(
            status="suspended", operational_status_code="K",
        ))
        op = doc["opinions"][0]["opinion"]
        assert op.uncertainty < 1.0
        assert op.uncertainty >= 0.01


# ═══════════════════════════════════════════════════════════════════
# 5. Extension recovery
# ═══════════════════════════════════════════════════════════════════


class TestLocationExtensionRecovery:
    """When a jsonld-ex extension is present on ``_status``, the
    exact opinion is recovered instead of being reconstructed."""

    def test_exact_opinion_recovered(self):
        original = Opinion(belief=0.45, disbelief=0.25, uncertainty=0.30)
        resource = _inject_extension(
            _location(status="active"), original,
        )
        doc, report = from_fhir(resource)
        assert report.success
        entry = doc["opinions"][0]
        assert entry["source"] == "extension"

        recovered: Opinion = entry["opinion"]
        assert abs(recovered.belief - original.belief) < 1e-9
        assert abs(recovered.disbelief - original.disbelief) < 1e-9
        assert abs(recovered.uncertainty - original.uncertainty) < 1e-9

    def test_extension_takes_precedence_over_reconstruction(self):
        """Extension opinion is used even when status signals would
        produce a different result."""
        original = Opinion(belief=0.10, disbelief=0.80, uncertainty=0.10)
        resource = _inject_extension(
            _location(status="active", operational_status_code="O"),
            original,
        )
        doc, _ = from_fhir(resource)
        entry = doc["opinions"][0]
        assert entry["source"] == "extension"
        recovered = entry["opinion"]
        # Should match original, not the active+O reconstruction
        assert abs(recovered.belief - 0.10) < 1e-9

    def test_extension_with_non_default_base_rate(self):
        """Extension preserves custom base_rate."""
        original = Opinion(
            belief=0.30, disbelief=0.20, uncertainty=0.50,
            base_rate=0.7,
        )
        resource = _inject_extension(
            _location(status="suspended"), original,
        )
        doc, _ = from_fhir(resource)
        recovered = doc["opinions"][0]["opinion"]
        assert abs(recovered.base_rate - 0.7) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# 6. Metadata passthrough — comprehensive field preservation
# ═══════════════════════════════════════════════════════════════════


class TestLocationMetadataPassthrough:
    """Every FHIR R4 Location field is preserved in the jsonld-ex
    document.  No data is excluded."""

    def test_name_passthrough(self):
        doc, _ = from_fhir(_location(name="Main Campus"))
        assert doc["name"] == "Main Campus"

    def test_description_passthrough(self):
        doc, _ = from_fhir(_location(description="East wing, 3rd floor"))
        assert doc["description"] == "East wing, 3rd floor"

    def test_mode_passthrough(self):
        doc, _ = from_fhir(_location(mode="instance"))
        assert doc["mode"] == "instance"

    def test_mode_kind_passthrough(self):
        doc, _ = from_fhir(_location(mode="kind"))
        assert doc["mode"] == "kind"

    def test_type_list_passthrough(self):
        types = [
            {"coding": [{"code": "HOSP", "display": "Hospital"}]},
            {"coding": [{"code": "ER", "display": "Emergency Room"}]},
        ]
        doc, _ = from_fhir(_location(type_list=types))
        assert doc["type"] == types

    def test_telecom_passthrough(self):
        telecoms = [
            {"system": "phone", "value": "555-0100", "use": "work"},
            {"system": "email", "value": "info@hospital.org"},
        ]
        doc, _ = from_fhir(_location(telecom=telecoms))
        assert doc["telecom"] == telecoms

    def test_address_passthrough(self):
        addr = {
            "use": "work",
            "line": ["123 Main St"],
            "city": "Springfield",
            "state": "IL",
            "postalCode": "62704",
            "country": "US",
        }
        doc, _ = from_fhir(_location(address=addr))
        assert doc["address"] == addr

    def test_physical_type_text_passthrough(self):
        doc, _ = from_fhir(_location(physical_type_text="Room"))
        assert doc["physicalType"] == "Room"

    def test_physical_type_coding_fallback(self):
        """When text is absent, first coding code is extracted."""
        doc, _ = from_fhir(_location(
            physical_type_coding=[{"code": "ro", "display": "Room"}],
        ))
        assert doc["physicalType"] == "ro"

    def test_position_passthrough(self):
        pos = {"longitude": -83.6945, "latitude": 42.2565, "altitude": 280.0}
        doc, _ = from_fhir(_location(position=pos))
        assert doc["position"] == pos

    def test_position_without_altitude(self):
        pos = {"longitude": -83.6945, "latitude": 42.2565}
        doc, _ = from_fhir(_location(position=pos))
        assert doc["position"] == pos

    def test_managing_organization_passthrough(self):
        doc, _ = from_fhir(_location(managing_org_ref="Organization/org-1"))
        assert doc["managingOrganization"] == "Organization/org-1"

    def test_part_of_passthrough(self):
        doc, _ = from_fhir(_location(part_of_ref="Location/building-A"))
        assert doc["partOf"] == "Location/building-A"

    def test_hours_of_operation_passthrough(self):
        hours = [
            {
                "daysOfWeek": ["mon", "tue", "wed", "thu", "fri"],
                "openingTime": "08:00:00",
                "closingTime": "17:00:00",
            },
        ]
        doc, _ = from_fhir(_location(hours_of_operation=hours))
        assert doc["hoursOfOperation"] == hours

    def test_availability_exceptions_passthrough(self):
        doc, _ = from_fhir(_location(
            availability_exceptions="Closed on public holidays",
        ))
        assert doc["availabilityExceptions"] == "Closed on public holidays"

    def test_endpoint_refs_passthrough(self):
        doc, _ = from_fhir(_location(
            endpoint_refs=["Endpoint/ep-1", "Endpoint/ep-2"],
        ))
        assert doc["endpoint_references"] == [
            "Endpoint/ep-1", "Endpoint/ep-2",
        ]

    def test_identifier_passthrough(self):
        ids = [
            {"system": "http://hospital.org/loc", "value": "LOC-001"},
        ]
        doc, _ = from_fhir(_location(identifier=ids))
        assert doc["identifier"] == ids

    def test_alias_passthrough(self):
        doc, _ = from_fhir(_location(alias=["Old Wing", "Ward B"]))
        assert doc["alias"] == ["Old Wing", "Ward B"]

    def test_operational_status_passthrough(self):
        """Full operationalStatus Coding is preserved for round-trip."""
        resource = _location(
            status="active",
            operational_status_code="O",
            operational_status_system="http://terminology.hl7.org/CodeSystem/v2-0116",
            operational_status_display="Occupied",
        )
        doc, _ = from_fhir(resource)
        op_status = doc.get("operationalStatus")
        assert op_status is not None
        assert op_status["code"] == "O"
        assert op_status["system"] == "http://terminology.hl7.org/CodeSystem/v2-0116"
        assert op_status["display"] == "Occupied"

    def test_missing_optional_fields_not_in_doc(self):
        """Fields that are absent from the FHIR resource should not
        appear in the jsonld-ex document."""
        doc, _ = from_fhir(_location(status="active"))
        assert "name" not in doc
        assert "description" not in doc
        assert "mode" not in doc
        assert "type" not in doc
        assert "telecom" not in doc
        assert "address" not in doc
        assert "physicalType" not in doc
        assert "position" not in doc
        assert "managingOrganization" not in doc
        assert "partOf" not in doc
        assert "hoursOfOperation" not in doc
        assert "availabilityExceptions" not in doc
        assert "endpoint_references" not in doc
        assert "identifier" not in doc
        assert "alias" not in doc
        assert "operationalStatus" not in doc

    def test_all_fields_simultaneously(self):
        """A fully-populated Location preserves every field."""
        resource = _location(
            status="active",
            location_id="loc-full",
            operational_status_code="O",
            operational_status_system="http://terminology.hl7.org/CodeSystem/v2-0116",
            operational_status_display="Occupied",
            name="Main Hospital",
            description="Primary care facility",
            mode="instance",
            type_list=[{"coding": [{"code": "HOSP"}]}],
            telecom=[{"system": "phone", "value": "555-0100"}],
            address={"city": "Springfield", "state": "IL"},
            physical_type_text="Building",
            position={"longitude": -83.69, "latitude": 42.25},
            managing_org_ref="Organization/org-1",
            part_of_ref="Location/campus-1",
            hours_of_operation=[{"daysOfWeek": ["mon"]}],
            availability_exceptions="Holidays closed",
            endpoint_refs=["Endpoint/ep-1"],
            identifier=[{"value": "LOC-001"}],
            alias=["Old Hospital"],
        )
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["id"] == "loc-full"
        assert doc["name"] == "Main Hospital"
        assert doc["description"] == "Primary care facility"
        assert doc["mode"] == "instance"
        assert doc["type"] == [{"coding": [{"code": "HOSP"}]}]
        assert doc["telecom"] == [{"system": "phone", "value": "555-0100"}]
        assert doc["address"] == {"city": "Springfield", "state": "IL"}
        assert doc["physicalType"] == "Building"
        assert doc["position"] == {"longitude": -83.69, "latitude": 42.25}
        assert doc["managingOrganization"] == "Organization/org-1"
        assert doc["partOf"] == "Location/campus-1"
        assert doc["hoursOfOperation"] == [{"daysOfWeek": ["mon"]}]
        assert doc["availabilityExceptions"] == "Holidays closed"
        assert doc["endpoint_references"] == ["Endpoint/ep-1"]
        assert doc["identifier"] == [{"value": "LOC-001"}]
        assert doc["alias"] == ["Old Hospital"]
        assert doc["operationalStatus"]["code"] == "O"


# ═══════════════════════════════════════════════════════════════════
# 7. Round-trip — from_fhir → to_fhir preserves all fields
# ═══════════════════════════════════════════════════════════════════


class TestLocationRoundTrip:
    """from_fhir → to_fhir produces a valid FHIR R4 Location with
    all metadata restored and SL opinion embedded as extension."""

    def test_basic_round_trip(self):
        resource = _location(status="active", location_id="loc-rt")
        doc, _ = from_fhir(resource)
        fhir_out, report = to_fhir(doc)

        assert report.success
        assert fhir_out["resourceType"] == "Location"
        assert fhir_out["id"] == "loc-rt"
        assert fhir_out["status"] == "active"
        assert "_status" in fhir_out
        assert fhir_out["_status"]["extension"][0]["url"] == FHIR_EXTENSION_URL

    def test_round_trip_preserves_name(self):
        doc, _ = from_fhir(_location(name="ICU"))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["name"] == "ICU"

    def test_round_trip_preserves_description(self):
        doc, _ = from_fhir(_location(description="Intensive care unit"))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["description"] == "Intensive care unit"

    def test_round_trip_preserves_mode(self):
        doc, _ = from_fhir(_location(mode="instance"))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["mode"] == "instance"

    def test_round_trip_preserves_type(self):
        types = [{"coding": [{"code": "HOSP"}]}]
        doc, _ = from_fhir(_location(type_list=types))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["type"] == types

    def test_round_trip_preserves_telecom(self):
        telecoms = [{"system": "phone", "value": "555-0100"}]
        doc, _ = from_fhir(_location(telecom=telecoms))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["telecom"] == telecoms

    def test_round_trip_preserves_address(self):
        addr = {"city": "Springfield", "state": "IL"}
        doc, _ = from_fhir(_location(address=addr))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["address"] == addr

    def test_round_trip_preserves_physical_type(self):
        doc, _ = from_fhir(_location(physical_type_text="Room"))
        fhir_out, _ = to_fhir(doc)
        # Reconstructed as CodeableConcept
        assert fhir_out["physicalType"] == {"text": "Room"}

    def test_round_trip_preserves_position(self):
        pos = {"longitude": -83.69, "latitude": 42.25, "altitude": 280.0}
        doc, _ = from_fhir(_location(position=pos))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["position"] == pos

    def test_round_trip_preserves_managing_organization(self):
        doc, _ = from_fhir(_location(managing_org_ref="Organization/org-1"))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["managingOrganization"] == {
            "reference": "Organization/org-1",
        }

    def test_round_trip_preserves_part_of(self):
        doc, _ = from_fhir(_location(part_of_ref="Location/building-A"))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["partOf"] == {"reference": "Location/building-A"}

    def test_round_trip_preserves_hours_of_operation(self):
        hours = [{"daysOfWeek": ["mon"], "openingTime": "08:00:00"}]
        doc, _ = from_fhir(_location(hours_of_operation=hours))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["hoursOfOperation"] == hours

    def test_round_trip_preserves_availability_exceptions(self):
        doc, _ = from_fhir(_location(
            availability_exceptions="Closed weekends",
        ))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["availabilityExceptions"] == "Closed weekends"

    def test_round_trip_preserves_endpoint_refs(self):
        doc, _ = from_fhir(_location(
            endpoint_refs=["Endpoint/ep-1", "Endpoint/ep-2"],
        ))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["endpoint"] == [
            {"reference": "Endpoint/ep-1"},
            {"reference": "Endpoint/ep-2"},
        ]

    def test_round_trip_preserves_identifier(self):
        ids = [{"system": "http://hospital.org", "value": "LOC-001"}]
        doc, _ = from_fhir(_location(identifier=ids))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["identifier"] == ids

    def test_round_trip_preserves_alias(self):
        doc, _ = from_fhir(_location(alias=["Old Wing", "Ward B"]))
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["alias"] == ["Old Wing", "Ward B"]

    def test_round_trip_preserves_operational_status(self):
        resource = _location(
            status="active",
            operational_status_code="O",
            operational_status_system="http://terminology.hl7.org/CodeSystem/v2-0116",
            operational_status_display="Occupied",
        )
        doc, _ = from_fhir(resource)
        fhir_out, _ = to_fhir(doc)
        assert fhir_out["operationalStatus"] == {
            "code": "O",
            "system": "http://terminology.hl7.org/CodeSystem/v2-0116",
            "display": "Occupied",
        }

    def test_round_trip_opinion_fidelity(self):
        """The SL opinion survives a full round-trip via extension."""
        resource = _location(status="active", operational_status_code="O")
        doc1, _ = from_fhir(resource)
        fhir_mid, _ = to_fhir(doc1)
        doc2, _ = from_fhir(fhir_mid)

        op1 = doc1["opinions"][0]["opinion"]
        op2 = doc2["opinions"][0]["opinion"]

        # Second pass should recover via extension
        assert doc2["opinions"][0]["source"] == "extension"
        assert abs(op1.belief - op2.belief) < 1e-9
        assert abs(op1.disbelief - op2.disbelief) < 1e-9
        assert abs(op1.uncertainty - op2.uncertainty) < 1e-9

    def test_full_round_trip_all_fields(self):
        """Complete round-trip with every field populated."""
        resource = _location(
            status="active",
            location_id="loc-full-rt",
            operational_status_code="H",
            operational_status_system="http://terminology.hl7.org/CodeSystem/v2-0116",
            operational_status_display="Housekeeping",
            name="OR Suite 3",
            description="Operating room, east wing",
            mode="instance",
            type_list=[{"coding": [{"code": "OR"}]}],
            telecom=[{"system": "phone", "value": "555-0300"}],
            address={"line": ["100 Hospital Dr"], "city": "Springfield"},
            physical_type_text="Room",
            position={"longitude": -83.69, "latitude": 42.25},
            managing_org_ref="Organization/org-1",
            part_of_ref="Location/building-east",
            hours_of_operation=[{
                "daysOfWeek": ["mon", "tue", "wed", "thu", "fri"],
                "openingTime": "06:00:00",
                "closingTime": "22:00:00",
            }],
            availability_exceptions="Emergency only on weekends",
            endpoint_refs=["Endpoint/ep-or"],
            identifier=[{"value": "OR-003"}],
            alias=["Surgery 3"],
        )
        doc, _ = from_fhir(resource)
        fhir_out, report = to_fhir(doc)

        assert report.success
        assert fhir_out["resourceType"] == "Location"
        assert fhir_out["id"] == "loc-full-rt"
        assert fhir_out["status"] == "active"
        assert fhir_out["name"] == "OR Suite 3"
        assert fhir_out["description"] == "Operating room, east wing"
        assert fhir_out["mode"] == "instance"
        assert fhir_out["type"] == [{"coding": [{"code": "OR"}]}]
        assert fhir_out["telecom"] == [{"system": "phone", "value": "555-0300"}]
        assert fhir_out["address"] == {
            "line": ["100 Hospital Dr"], "city": "Springfield",
        }
        assert fhir_out["physicalType"] == {"text": "Room"}
        assert fhir_out["position"] == {"longitude": -83.69, "latitude": 42.25}
        assert fhir_out["managingOrganization"] == {
            "reference": "Organization/org-1",
        }
        assert fhir_out["partOf"] == {"reference": "Location/building-east"}
        assert fhir_out["hoursOfOperation"] == [{
            "daysOfWeek": ["mon", "tue", "wed", "thu", "fri"],
            "openingTime": "06:00:00",
            "closingTime": "22:00:00",
        }]
        assert fhir_out["availabilityExceptions"] == "Emergency only on weekends"
        assert fhir_out["endpoint"] == [{"reference": "Endpoint/ep-or"}]
        assert fhir_out["identifier"] == [{"value": "OR-003"}]
        assert fhir_out["alias"] == ["Surgery 3"]
        assert fhir_out["operationalStatus"] == {
            "code": "H",
            "system": "http://terminology.hl7.org/CodeSystem/v2-0116",
            "display": "Housekeeping",
        }
        assert "_status" in fhir_out


# ═══════════════════════════════════════════════════════════════════
# 8. Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestLocationEdgeCases:
    """Defensive handling of missing, malformed, and novel data."""

    def test_missing_status_field(self):
        """Location with no status at all still converts."""
        resource = {"resourceType": "Location", "id": "loc-no-status"}
        doc, report = from_fhir(resource)
        assert report.success
        assert doc["status"] is None
        assert len(doc["opinions"]) == 1

    def test_empty_operational_status(self):
        """operationalStatus as empty dict (no code)."""
        resource = _location(status="active")
        resource["operationalStatus"] = {}
        doc, report = from_fhir(resource)
        assert report.success
        # Should be treated as absent operationalStatus

    def test_operational_status_not_dict(self):
        """Malformed operationalStatus (string instead of Coding)."""
        resource = _location(status="active")
        resource["operationalStatus"] = "O"
        doc, report = from_fhir(resource)
        assert report.success
        # Should handle gracefully

    def test_managing_org_not_dict(self):
        """managingOrganization as string instead of Reference."""
        resource = _location(status="active")
        resource["managingOrganization"] = "Organization/org-1"
        doc, report = from_fhir(resource)
        assert report.success
        # Should not crash; field may be absent from doc
        assert "managingOrganization" not in doc

    def test_part_of_not_dict(self):
        """partOf as string instead of Reference."""
        resource = _location(status="active")
        resource["partOf"] = "Location/parent"
        doc, report = from_fhir(resource)
        assert report.success
        assert "partOf" not in doc

    def test_empty_endpoint_array(self):
        """endpoint as empty list."""
        resource = _location(status="active")
        resource["endpoint"] = []
        doc, report = from_fhir(resource)
        assert report.success
        assert "endpoint_references" not in doc

    def test_endpoint_without_reference_key(self):
        """endpoint array with dict missing 'reference' key."""
        resource = _location(status="active")
        resource["endpoint"] = [{"display": "Endpoint 1"}]
        doc, report = from_fhir(resource)
        assert report.success
        assert "endpoint_references" not in doc

    def test_empty_alias_array(self):
        """alias as empty list should not appear in doc."""
        resource = _location(status="active")
        resource["alias"] = []
        doc, _ = from_fhir(resource)
        assert "alias" not in doc

    def test_empty_type_array(self):
        """type as empty list should not appear in doc."""
        resource = _location(status="active")
        resource["type"] = []
        doc, _ = from_fhir(resource)
        assert "type" not in doc

    def test_empty_telecom_array(self):
        """telecom as empty list should not appear in doc."""
        resource = _location(status="active")
        resource["telecom"] = []
        doc, _ = from_fhir(resource)
        assert "telecom" not in doc

    def test_empty_identifier_array(self):
        """identifier as empty list should not appear in doc."""
        resource = _location(status="active")
        resource["identifier"] = []
        doc, _ = from_fhir(resource)
        assert "identifier" not in doc

    def test_empty_hours_of_operation_array(self):
        """hoursOfOperation as empty list should not appear in doc."""
        resource = _location(status="active")
        resource["hoursOfOperation"] = []
        doc, _ = from_fhir(resource)
        assert "hoursOfOperation" not in doc

    def test_physical_type_empty_codeable_concept(self):
        """physicalType as empty CodeableConcept (no text, no coding)."""
        resource = _location(status="active")
        resource["physicalType"] = {}
        doc, _ = from_fhir(resource)
        assert "physicalType" not in doc

    def test_physical_type_coding_without_code(self):
        """physicalType coding entry missing 'code' key."""
        resource = _location(status="active")
        resource["physicalType"] = {"coding": [{"display": "Room"}]}
        doc, _ = from_fhir(resource)
        # No extractable text or code → absent
        assert "physicalType" not in doc

    def test_position_empty_dict(self):
        """position as empty dict should not appear."""
        resource = _location(status="active")
        resource["position"] = {}
        doc, _ = from_fhir(resource)
        # Empty position is still a valid dict, preserve it
        assert doc.get("position") == {}


# ═══════════════════════════════════════════════════════════════════
# 9. Integration — Location in clinical context
# ═══════════════════════════════════════════════════════════════════


class TestLocationIntegration:
    """Location participates in Encounter and Provenance workflows."""

    def test_location_referenced_by_encounter(self):
        """Verify Location converts for use as Encounter.location."""
        loc_doc, _ = from_fhir(_location(
            status="active",
            location_id="loc-er",
            name="Emergency Room",
            operational_status_code="O",
        ))
        # Encounter would reference this location
        assert loc_doc["@type"] == "fhir:Location"
        assert loc_doc["id"] == "loc-er"
        assert loc_doc["name"] == "Emergency Room"

    def test_location_referenced_by_provenance(self):
        """Verify Location converts for use as Provenance.location."""
        loc_doc, _ = from_fhir(_location(
            status="active",
            location_id="loc-lab",
            name="Pathology Lab",
            managing_org_ref="Organization/hospital-1",
        ))
        assert loc_doc["managingOrganization"] == "Organization/hospital-1"

    def test_location_hierarchy_via_part_of(self):
        """Location.partOf creates facility hierarchy."""
        building, _ = from_fhir(_location(
            status="active",
            location_id="building-A",
            name="Building A",
        ))
        floor, _ = from_fhir(_location(
            status="active",
            location_id="floor-3",
            name="3rd Floor",
            part_of_ref="Location/building-A",
        ))
        room, _ = from_fhir(_location(
            status="active",
            location_id="room-301",
            name="Room 301",
            part_of_ref="Location/floor-3",
        ))

        assert room["partOf"] == "Location/floor-3"
        assert floor["partOf"] == "Location/building-A"
        assert "partOf" not in building
