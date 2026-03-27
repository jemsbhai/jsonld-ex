"""
LwM2M Transport Optimization for JSON-LD-Ex.

Maps JSON-LD documents to the OMA LwM2M (Lightweight Machine to
Machine) object/resource model, enabling semantic IoT device
management on constrained networks.

LwM2M is built on CoAP and uses a numbered object/resource hierarchy
defined by IPSO Smart Objects.  This module bridges JSON-LD's
property-based model and LwM2M's numbered resource model.

Metadata mappings:

- **Object ID**: From ``@type`` via IPSO Smart Object registry
  (e.g. ``TemperatureSensor`` → Object 3303).  Unknown types
  get a custom object ID in the 26241+ range.
- **Instance ID**: Sequential (0, 1, ...) per object type.
- **Resource IDs**: From property names via per-object resource
  maps (e.g. ``temperature`` → Resource 5700).
- **Registration**: ``@id`` → endpoint name, ``@validUntil`` →
  registration lifetime.
- **CoRE Links**: RFC 6690 link-format for LwM2M discovery.

No LwM2M client library is required — this module returns plain
dicts and strings suitable for any LwM2M implementation
(``aiocoap``, ``leshan``, ``Anjay``).

References:
    OMA LwM2M v1.1: https://openmobilealliance.org/release/LightweightM2M/
    IPSO Smart Objects: https://technical.openmobilealliance.org/OMNA/LwM2M/LwM2MRegistry.html
    RFC 6690: Constrained RESTful Environments (CoRE) Link Format.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from jsonld_ex._transport_common import (
    find_valid_until,
    seconds_remaining,
)


# ── Binding modes (LwM2M v1.1 §5.3.1) ─────────────────────────────

BINDING_MODE_UDP: str = "U"
"""UDP binding (default for constrained devices)."""

BINDING_MODE_TCP: str = "T"
"""TCP binding."""

# ── Custom object ID base ──────────────────────────────────────────
# OMA reserved range for custom objects: 26241–42768
_CUSTOM_OBJECT_BASE = 26241

# ── Default registration lifetime (seconds) ───────────────────────
_DEFAULT_LIFETIME = 86400  # 24 hours


# ═══════════════════════════════════════════════════════════════════
# IPSO SMART OBJECT REGISTRY
# ═══════════════════════════════════════════════════════════════════


IPSO_OBJECT_REGISTRY: Dict[str, Dict[str, Any]] = {
    "TemperatureSensor": {
        "object_id": 3303,
        "resources": {
            "temperature": 5700,      # Sensor Value
            "minMeasuredValue": 5601,  # Min Measured Value
            "maxMeasuredValue": 5602,  # Max Measured Value
            "minRangeValue": 5603,     # Min Range Value
            "maxRangeValue": 5604,     # Max Range Value
            "sensorUnits": 5701,       # Sensor Units
            "resetMinMaxValues": 5605, # Reset Min/Max
        },
    },
    "HumiditySensor": {
        "object_id": 3304,
        "resources": {
            "humidity": 5700,
            "minMeasuredValue": 5601,
            "maxMeasuredValue": 5602,
            "minRangeValue": 5603,
            "maxRangeValue": 5604,
            "sensorUnits": 5701,
        },
    },
    "Barometer": {
        "object_id": 3315,
        "resources": {
            "pressure": 5700,
            "minMeasuredValue": 5601,
            "maxMeasuredValue": 5602,
            "minRangeValue": 5603,
            "maxRangeValue": 5604,
            "sensorUnits": 5701,
        },
    },
    "Accelerometer": {
        "object_id": 3313,
        "resources": {
            "x": 5702,               # X Value
            "y": 5703,               # Y Value
            "z": 5704,               # Z Value
            "sensorUnits": 5701,
            "minRangeValue": 5603,
            "maxRangeValue": 5604,
        },
    },
    "Illuminance": {
        "object_id": 3301,
        "resources": {
            "illuminance": 5700,
            "minMeasuredValue": 5601,
            "maxMeasuredValue": 5602,
            "sensorUnits": 5701,
        },
    },
    "DigitalOutput": {
        "object_id": 3201,
        "resources": {
            "digitalOutputState": 5550,
            "digitalOutputPolarity": 5551,
        },
    },
    "AnalogInput": {
        "object_id": 3202,
        "resources": {
            "analogInputValue": 5600,
            "minMeasuredValue": 5601,
            "maxMeasuredValue": 5602,
            "minRangeValue": 5603,
            "maxRangeValue": 5604,
        },
    },
    "GenericSensor": {
        "object_id": 3300,
        "resources": {
            "sensorValue": 5700,
            "minMeasuredValue": 5601,
            "maxMeasuredValue": 5602,
            "sensorUnits": 5701,
        },
    },
}
"""Registry mapping JSON-LD ``@type`` names to IPSO Smart Object IDs.

Each entry contains:
- ``object_id``: The OMA-registered IPSO object identifier.
- ``resources``: Mapping of property names to IPSO resource IDs.

This registry covers common sensor types.  Implementers can extend
it for domain-specific objects.
"""


# ═══════════════════════════════════════════════════════════════════
# RESOURCE VALUE EXTRACTION
# ═══════════════════════════════════════════════════════════════════


def extract_lwm2m_resources(
    doc: Dict[str, Any],
    resource_map: Dict[str, int],
) -> Dict[int, Any]:
    """Extract LwM2M resource values from JSON-LD properties.

    For each property in *resource_map*, looks up the corresponding
    value in *doc*.  Annotated values (``{"@value": ...}``) are
    unwrapped to their inner value.

    Args:
        doc: JSON-LD document or property subset.
        resource_map: Property name → IPSO resource ID mapping.

    Returns:
        Dict of resource ID → value.
    """
    resources: Dict[int, Any] = {}

    for prop_name, res_id in resource_map.items():
        if prop_name not in doc:
            continue

        val = doc[prop_name]

        # Unwrap annotated values
        if isinstance(val, dict) and "@value" in val:
            val = val["@value"]

        resources[res_id] = val

    return resources


# ═══════════════════════════════════════════════════════════════════
# OBJECT/RESOURCE DERIVATION
# ═══════════════════════════════════════════════════════════════════


def derive_lwm2m_objects(
    doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Derive LwM2M object instances from a JSON-LD document.

    Maps ``@type`` to a known IPSO Smart Object ID, then extracts
    resource values from document properties.  Unknown types get
    a custom object ID and sequential resource numbering.

    Args:
        doc: JSON-LD document.

    Returns:
        List of object instance dicts, each with:
        - ``object_id`` (int): IPSO or custom object ID.
        - ``instance_id`` (int): Instance index (0-based).
        - ``resources`` (dict): Resource ID → value.
    """
    type_val = doc.get("@type")
    if isinstance(type_val, list):
        type_val = type_val[0] if type_val else None
    type_str = str(type_val) if type_val is not None else None

    # Check IPSO registry
    if type_str and type_str in IPSO_OBJECT_REGISTRY:
        entry = IPSO_OBJECT_REGISTRY[type_str]
        resources = extract_lwm2m_resources(doc, entry["resources"])
        return [{
            "object_id": entry["object_id"],
            "instance_id": 0,
            "resources": resources,
        }]

    # Try matching sensor properties to known objects
    matched_objects = _match_properties_to_objects(doc)
    if matched_objects:
        return matched_objects

    # Fallback: custom object with sequential resource IDs
    return [_build_custom_object(doc)]


def _match_properties_to_objects(
    doc: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Try to match document properties to known IPSO objects.

    For multi-sensor documents (e.g. temperature + humidity), this
    creates separate object instances for each recognized sensor
    property.
    """
    objects: List[Dict[str, Any]] = []
    instance_counters: Dict[int, int] = {}

    # Map of property name → (object_id, resource_id)
    property_to_object: Dict[str, tuple] = {}
    for type_name, entry in IPSO_OBJECT_REGISTRY.items():
        for prop, res_id in entry["resources"].items():
            if res_id == 5700:  # Primary sensor value
                property_to_object[prop] = (entry["object_id"], res_id)

    for prop_name, val in doc.items():
        if prop_name.startswith("@"):
            continue
        if prop_name in property_to_object:
            obj_id, res_id = property_to_object[prop_name]

            # Unwrap annotated value
            actual_val = val
            if isinstance(val, dict) and "@value" in val:
                actual_val = val["@value"]

            inst_id = instance_counters.get(obj_id, 0)
            instance_counters[obj_id] = inst_id + 1

            objects.append({
                "object_id": obj_id,
                "instance_id": inst_id,
                "resources": {res_id: actual_val},
            })

    return objects


def _build_custom_object(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Build a custom LwM2M object for unknown @type."""
    # Deterministic custom object ID from type string hash
    type_val = doc.get("@type")
    if type_val is not None:
        if isinstance(type_val, list):
            type_val = type_val[0] if type_val else "unknown"
        obj_id = _CUSTOM_OBJECT_BASE + (hash(str(type_val)) % 1000)
    else:
        obj_id = _CUSTOM_OBJECT_BASE

    # Sequential resource IDs for non-@ properties
    resources: Dict[int, Any] = {}
    res_id = 0
    for key, val in doc.items():
        if key.startswith("@"):
            continue
        actual_val = val
        if isinstance(val, dict) and "@value" in val:
            actual_val = val["@value"]
        resources[res_id] = actual_val
        res_id += 1

    return {
        "object_id": obj_id,
        "instance_id": 0,
        "resources": resources,
    }


# ═══════════════════════════════════════════════════════════════════
# REGISTRATION PARAMETERS
# ═══════════════════════════════════════════════════════════════════


def derive_lwm2m_registration(
    doc: Dict[str, Any],
    binding: str = BINDING_MODE_UDP,
) -> Dict[str, Any]:
    """Derive LwM2M registration parameters from JSON-LD metadata.

    LwM2M clients register with a server providing endpoint name,
    lifetime, binding mode, and supported objects (LwM2M v1.1 §5.3).

    Args:
        doc: JSON-LD document.
        binding: Binding mode (default UDP).

    Returns:
        Dict with:
        - ``endpoint`` (str): From ``@id`` or ``"unknown"``.
        - ``lifetime`` (int): From ``@validUntil`` or default 86400s.
        - ``binding`` (str): ``"U"`` (UDP) or ``"T"`` (TCP).
        - ``objects`` (list): Object instances from
          :func:`derive_lwm2m_objects`.
    """
    reg: Dict[str, Any] = {}

    # -- Endpoint --
    doc_id = doc.get("@id")
    reg["endpoint"] = str(doc_id) if doc_id is not None else "unknown"

    # -- Lifetime from @validUntil --
    valid_until = find_valid_until(doc)
    if valid_until is not None:
        remaining = seconds_remaining(valid_until)
        if remaining is not None and remaining > 0:
            reg["lifetime"] = int(math.ceil(remaining))
        else:
            reg["lifetime"] = _DEFAULT_LIFETIME
    else:
        reg["lifetime"] = _DEFAULT_LIFETIME

    # -- Binding mode --
    reg["binding"] = binding

    # -- Object instances --
    reg["objects"] = derive_lwm2m_objects(doc)

    return reg


# ═══════════════════════════════════════════════════════════════════
# CoRE LINK FORMAT
# ═══════════════════════════════════════════════════════════════════


def derive_lwm2m_links(doc: Dict[str, Any]) -> str:
    """Generate CoRE Link Format (RFC 6690) for LwM2M discovery.

    Produces the link-format string used in LwM2M registration
    and ``.well-known/core`` discovery responses.

    Args:
        doc: JSON-LD document.

    Returns:
        RFC 6690 link-format string.

    Examples::

        >>> derive_lwm2m_links({"@type": "TemperatureSensor", "temperature": {"@value": 36.7}})
        '</3303/0>'
    """
    objects = derive_lwm2m_objects(doc)

    links: List[str] = []
    for obj in objects:
        obj_id = obj["object_id"]
        inst_id = obj["instance_id"]
        links.append(f"</{obj_id}/{inst_id}>")

    return ",".join(links) if links else ""
