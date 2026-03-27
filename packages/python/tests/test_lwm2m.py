"""Tests for LwM2M transport module.

Verifies OMA LwM2M object/resource model mapping from JSON-LD
metadata, registration parameter derivation, and CoRE Link Format
generation for device management on constrained IoT networks.

LwM2M (Lightweight Machine to Machine) is built on CoAP and uses
a numbered object/resource model defined by IPSO Smart Objects.

References:
    OMA LwM2M v1.1: https://openmobilealliance.org/release/LightweightM2M/
    IPSO Smart Objects: https://technical.openmobilealliance.org/OMNA/LwM2M/LwM2MRegistry.html
    RFC 6690: CoRE Link Format
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

from jsonld_ex.lwm2m import (
    # Object/resource mapping
    derive_lwm2m_objects,
    derive_lwm2m_registration,
    derive_lwm2m_links,
    # Known object registry
    IPSO_OBJECT_REGISTRY,
    # Resource value extraction
    extract_lwm2m_resources,
    # Constants
    BINDING_MODE_UDP,
    BINDING_MODE_TCP,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def temperature_doc():
    """Temperature sensor reading matching IPSO Object 3303."""
    return {
        "@context": "https://schema.org/",
        "@type": "TemperatureSensor",
        "@id": "urn:imei:123456789012345",
        "@confidence": 0.95,
        "temperature": {
            "@value": 36.7,
            "@confidence": 0.88,
            "@validUntil": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        },
        "minMeasuredValue": -10.0,
        "maxMeasuredValue": 85.0,
        "sensorUnits": "Cel",
    }


@pytest.fixture
def humidity_doc():
    """Humidity sensor matching IPSO Object 3304."""
    return {
        "@context": "https://schema.org/",
        "@type": "HumiditySensor",
        "@id": "urn:imei:987654321098765",
        "humidity": {"@value": 45.2},
        "sensorUnits": "%RH",
    }


@pytest.fixture
def generic_doc():
    """Document with no known IPSO mapping."""
    return {
        "@context": "https://schema.org/",
        "@type": "CustomDevice",
        "@id": "urn:dev:custom-001",
        "status": "active",
        "batteryLevel": 87,
    }


@pytest.fixture
def multi_sensor_doc():
    """Document with multiple sensor types."""
    return {
        "@context": "https://schema.org/",
        "@type": "MultiSensor",
        "@id": "urn:dev:multi-001",
        "temperature": {"@value": 22.5},
        "humidity": {"@value": 60.0},
        "pressure": {"@value": 1013.25},
    }


# ═══════════════════════════════════════════════════════════════════
# IPSO Object Registry
# ═══════════════════════════════════════════════════════════════════


class TestIPSORegistry:
    """Known IPSO Smart Object mappings."""

    def test_temperature_registered(self):
        assert "TemperatureSensor" in IPSO_OBJECT_REGISTRY
        assert IPSO_OBJECT_REGISTRY["TemperatureSensor"]["object_id"] == 3303

    def test_humidity_registered(self):
        assert "HumiditySensor" in IPSO_OBJECT_REGISTRY
        assert IPSO_OBJECT_REGISTRY["HumiditySensor"]["object_id"] == 3304

    def test_registry_has_resource_ids(self):
        """Each registry entry should have resource mappings."""
        for type_name, entry in IPSO_OBJECT_REGISTRY.items():
            assert "object_id" in entry, f"{type_name} missing object_id"
            assert "resources" in entry, f"{type_name} missing resources"
            assert isinstance(entry["resources"], dict)

    def test_registry_values_are_ints(self):
        """Object and resource IDs must be integers."""
        for type_name, entry in IPSO_OBJECT_REGISTRY.items():
            assert isinstance(entry["object_id"], int)
            for res_name, res_id in entry["resources"].items():
                assert isinstance(res_id, int), (
                    f"{type_name}.{res_name} has non-int resource ID"
                )


# ═══════════════════════════════════════════════════════════════════
# Object/resource derivation
# ═══════════════════════════════════════════════════════════════════


class TestLwM2MObjects:
    """Derive LwM2M object instances from JSON-LD documents."""

    def test_known_type_maps_to_ipso(self, temperature_doc):
        """TemperatureSensor → IPSO Object 3303."""
        objects = derive_lwm2m_objects(temperature_doc)
        assert len(objects) >= 1
        obj = objects[0]
        assert obj["object_id"] == 3303
        assert obj["instance_id"] == 0

    def test_temperature_resources(self, temperature_doc):
        """Temperature value maps to resource 5700."""
        objects = derive_lwm2m_objects(temperature_doc)
        resources = objects[0]["resources"]
        assert 5700 in resources
        assert resources[5700] == 36.7

    def test_min_max_resources(self, temperature_doc):
        """Min/max measured values map to resources 5601/5602."""
        objects = derive_lwm2m_objects(temperature_doc)
        resources = objects[0]["resources"]
        assert 5601 in resources
        assert resources[5601] == -10.0
        assert 5602 in resources
        assert resources[5602] == 85.0

    def test_units_resource(self, temperature_doc):
        """sensorUnits maps to resource 5701."""
        objects = derive_lwm2m_objects(temperature_doc)
        resources = objects[0]["resources"]
        assert 5701 in resources
        assert resources[5701] == "Cel"

    def test_humidity_maps_to_3304(self, humidity_doc):
        objects = derive_lwm2m_objects(humidity_doc)
        assert objects[0]["object_id"] == 3304

    def test_unknown_type_uses_custom_object(self, generic_doc):
        """Unknown @type → custom object ID range (26241+)."""
        objects = derive_lwm2m_objects(generic_doc)
        assert len(objects) >= 1
        assert objects[0]["object_id"] >= 26241  # custom object range

    def test_generic_resources_use_sequential_ids(self, generic_doc):
        """Unmapped properties get sequential resource IDs."""
        objects = derive_lwm2m_objects(generic_doc)
        resources = objects[0]["resources"]
        assert len(resources) >= 1


# ═══════════════════════════════════════════════════════════════════
# Resource value extraction
# ═══════════════════════════════════════════════════════════════════


class TestResourceExtraction:
    """Extract LwM2M resource values from JSON-LD properties."""

    def test_annotated_value_extracts_inner(self):
        """@value wrapper → extract inner value."""
        resources = extract_lwm2m_resources(
            {"temp": {"@value": 36.7, "@confidence": 0.9}},
            resource_map={"temp": 5700},
        )
        assert resources[5700] == 36.7

    def test_plain_value_preserved(self):
        """Plain value (no @value wrapper) → use directly."""
        resources = extract_lwm2m_resources(
            {"temp": 36.7},
            resource_map={"temp": 5700},
        )
        assert resources[5700] == 36.7

    def test_string_value(self):
        resources = extract_lwm2m_resources(
            {"units": "Cel"},
            resource_map={"units": 5701},
        )
        assert resources[5701] == "Cel"

    def test_unmapped_properties_skipped(self):
        """Properties not in resource_map are skipped."""
        resources = extract_lwm2m_resources(
            {"temp": 36.7, "unknown": "value"},
            resource_map={"temp": 5700},
        )
        assert 5700 in resources
        assert len(resources) == 1


# ═══════════════════════════════════════════════════════════════════
# Registration parameters
# ═══════════════════════════════════════════════════════════════════


class TestLwM2MRegistration:
    """LwM2M registration parameters from JSON-LD metadata."""

    def test_endpoint_from_id(self, temperature_doc):
        """@id → endpoint client name."""
        reg = derive_lwm2m_registration(temperature_doc)
        assert reg["endpoint"] == "urn:imei:123456789012345"

    def test_lifetime_from_valid_until(self, temperature_doc):
        """@validUntil → registration lifetime in seconds."""
        reg = derive_lwm2m_registration(temperature_doc)
        assert "lifetime" in reg
        assert 3500 <= reg["lifetime"] <= 3700

    def test_default_lifetime_without_valid_until(self, generic_doc):
        """No @validUntil → default lifetime (86400s = 24h)."""
        reg = derive_lwm2m_registration(generic_doc)
        assert reg["lifetime"] == 86400

    def test_binding_mode_default(self, temperature_doc):
        """Default binding mode is UDP."""
        reg = derive_lwm2m_registration(temperature_doc)
        assert reg["binding"] == BINDING_MODE_UDP

    def test_object_links_present(self, temperature_doc):
        """Registration includes object links."""
        reg = derive_lwm2m_registration(temperature_doc)
        assert "objects" in reg
        assert len(reg["objects"]) >= 1

    def test_missing_id_uses_unknown(self):
        reg = derive_lwm2m_registration({"@type": "Sensor"})
        assert reg["endpoint"] == "unknown"


# ═══════════════════════════════════════════════════════════════════
# CoRE Link Format
# ═══════════════════════════════════════════════════════════════════


class TestCoRELinks:
    """CoRE Link Format (RFC 6690) generation for LwM2M discovery."""

    def test_link_format_string(self, temperature_doc):
        """Generates RFC 6690 link format string."""
        links = derive_lwm2m_links(temperature_doc)
        assert isinstance(links, str)

    def test_contains_object_path(self, temperature_doc):
        """Links contain /object_id/instance_id paths."""
        links = derive_lwm2m_links(temperature_doc)
        assert "/3303/0" in links

    def test_angle_bracket_syntax(self, temperature_doc):
        """CoRE links use <path> syntax."""
        links = derive_lwm2m_links(temperature_doc)
        assert "</3303/0>" in links

    def test_multiple_objects(self, multi_sensor_doc):
        """Multiple sensor properties generate multiple links."""
        links = derive_lwm2m_links(multi_sensor_doc)
        # Should have at least temperature and humidity objects
        assert links.count("<") >= 1

    def test_unknown_type_link(self, generic_doc):
        """Unknown type still generates a link."""
        links = derive_lwm2m_links(generic_doc)
        assert "</" in links


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestLwM2MEdgeCases:

    def test_empty_document(self):
        objects = derive_lwm2m_objects({})
        # Empty doc → at least a custom object with no resources
        assert isinstance(objects, list)

    def test_empty_document_registration(self):
        reg = derive_lwm2m_registration({})
        assert reg["endpoint"] == "unknown"
        assert reg["lifetime"] == 86400

    def test_empty_document_links(self):
        links = derive_lwm2m_links({})
        assert isinstance(links, str)
