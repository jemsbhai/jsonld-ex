"""
FHIR R4 Provenance → W3C PROV-O bridge.

Converts a jsonld-ex Provenance document (produced by ``from_fhir()``)
into a W3C PROV-O JSON-LD graph, bridging the FHIR R4 Provenance
resource model to the existing ``to_prov_o()`` ecosystem in
``owl_interop.py``.

FHIR R4 Provenance is a constrained profile of W3C PROV.  The mapping:
  - Provenance.target[]          → prov:Entity (the things being described)
  - Provenance.agent[].who       → prov:Person | prov:SoftwareAgent | prov:Agent
  - Provenance.recorded          → prov:generatedAtTime
  - Provenance.activity          → prov:Activity
  - Provenance.entity[].role     → prov:wasDerivedFrom / wasRevisionOf /
                                   wasQuotedFrom / hadPrimarySource /
                                   wasInvalidatedBy
  - Provenance.agent[].onBehalfOf → prov:actedOnBehalfOf
  - SL opinion                   → jsonld-ex:confidence / jsonld-ex:opinion
"""

from __future__ import annotations

import uuid
from typing import Any

from jsonld_ex.owl_interop import (
    PROV,
    RDFS,
    XSD,
    JSONLD_EX,
    ConversionReport,
)


# ── Entity role → PROV-O property mapping ─────────────────────────
#
# FHIR R4 Provenance.entity.role ValueSet (required binding):
#   http://hl7.org/fhir/R4/valueset-provenance-entity-role.html
#
# Each role maps to a specific W3C PROV-O property connecting the
# target entity to the source entity.

_ENTITY_ROLE_TO_PROV: dict[str, str] = {
    "derivation": "wasDerivedFrom",
    "revision": "wasRevisionOf",
    "quotation": "wasQuotedFrom",
    "source": "hadPrimarySource",
    "removal": "wasInvalidatedBy",
}


# ── Agent type → PROV-O class mapping ────────────────────────────
#
# Heuristic based on the FHIR Reference target type:
#   Practitioner / PractitionerRole / Patient / RelatedPerson → prov:Person
#   Device                                                    → prov:SoftwareAgent
#   Organization                                              → prov:Organization
#   Anything else                                             → prov:Agent

_AGENT_REF_TYPE_TO_PROV_CLASS: dict[str, str] = {
    "Practitioner": f"{PROV}Person",
    "PractitionerRole": f"{PROV}Person",
    "Patient": f"{PROV}Person",
    "RelatedPerson": f"{PROV}Person",
    "Device": f"{PROV}SoftwareAgent",
    "Organization": f"{PROV}Organization",
}


def _agent_prov_class(who_ref: str | None) -> str:
    """Determine the PROV-O class for an agent based on its who reference."""
    if who_ref is None:
        return f"{PROV}Agent"
    # Extract resource type from "ResourceType/id"
    if "/" in who_ref:
        ref_type = who_ref.split("/")[0]
        return _AGENT_REF_TYPE_TO_PROV_CLASS.get(ref_type, f"{PROV}Agent")
    return f"{PROV}Agent"


def fhir_provenance_to_prov_o(
    doc: dict[str, Any],
) -> tuple[dict[str, Any], ConversionReport]:
    """Convert a jsonld-ex Provenance doc to a W3C PROV-O JSON-LD graph.

    Takes the output of ``from_fhir()`` for a FHIR Provenance resource
    and produces a PROV-O graph compatible with the existing
    ``to_prov_o()`` / ``from_prov_o()`` ecosystem.

    Args:
        doc: A jsonld-ex document with ``@type == "fhir:Provenance"``,
            as produced by ``from_fhir()``.

    Returns:
        Tuple of ``(prov_o_doc, ConversionReport)``.

    Raises:
        ValueError: If ``doc["@type"]`` is not ``"fhir:Provenance"``.
    """
    doc_type = doc.get("@type", "")
    if doc_type != "fhir:Provenance":
        raise ValueError(
            f"Expected @type 'fhir:Provenance', got '{doc_type}'"
        )

    report = ConversionReport(success=True)
    graph_nodes: list[dict[str, Any]] = []
    output_triples = 0

    recorded = doc.get("recorded")
    targets = doc.get("targets", [])
    agents = doc.get("agents", [])
    entities = doc.get("entities", [])
    activity_code = doc.get("activity")
    opinions = doc.get("opinions", [])

    # ── Create prov:Activity node (if activity is present) ────────
    activity_id = None
    if activity_code is not None:
        activity_id = f"_:activity-{uuid.uuid4().hex[:8]}"
        activity_node: dict[str, Any] = {
            "@id": activity_id,
            "@type": f"{PROV}Activity",
            f"{RDFS}label": activity_code,
        }
        output_triples += 2  # type + label
        graph_nodes.append(activity_node)

    # ── Create prov:Entity nodes for each target ──────────────────
    target_entity_ids: list[str] = []
    for t in targets:
        ref = t.get("reference", f"_:target-{uuid.uuid4().hex[:8]}")
        entity_id = ref  # use FHIR reference as the entity @id
        entity_node: dict[str, Any] = {
            "@id": entity_id,
            "@type": f"{PROV}Entity",
        }
        output_triples += 1  # type

        # Link to activity
        if activity_id is not None:
            entity_node[f"{PROV}wasGeneratedBy"] = {"@id": activity_id}
            output_triples += 1

        # recorded → generatedAtTime
        if recorded is not None:
            entity_node[f"{PROV}generatedAtTime"] = {
                "@value": recorded,
                "@type": f"{XSD}dateTime",
            }
            output_triples += 1

        target_entity_ids.append(entity_id)
        graph_nodes.append(entity_node)

    # ── Create agent nodes ────────────────────────────────────────
    for ag in agents:
        who = ag.get("who")
        agent_type_code = ag.get("type")
        on_behalf_of = ag.get("onBehalfOf")

        agent_id = who if who else f"_:agent-{uuid.uuid4().hex[:8]}"
        prov_class = _agent_prov_class(who)

        agent_node: dict[str, Any] = {
            "@id": agent_id,
            "@type": prov_class,
        }
        output_triples += 1  # type

        if agent_type_code is not None:
            agent_node[f"{RDFS}label"] = agent_type_code
            output_triples += 1

        # onBehalfOf → prov:actedOnBehalfOf
        if on_behalf_of is not None:
            agent_node[f"{PROV}actedOnBehalfOf"] = {"@id": on_behalf_of}
            output_triples += 1

        graph_nodes.append(agent_node)

        # Link target entities to this agent via wasAttributedTo
        for eid in target_entity_ids:
            # Find the entity node and add attribution
            for node in graph_nodes:
                if node.get("@id") == eid:
                    existing = node.get(f"{PROV}wasAttributedTo")
                    if existing is None:
                        node[f"{PROV}wasAttributedTo"] = {"@id": agent_id}
                    elif isinstance(existing, dict):
                        node[f"{PROV}wasAttributedTo"] = [
                            existing, {"@id": agent_id},
                        ]
                    elif isinstance(existing, list):
                        existing.append({"@id": agent_id})
                    output_triples += 1
                    break

        # Link activity to agent via wasAssociatedWith
        if activity_id is not None:
            for node in graph_nodes:
                if node.get("@id") == activity_id:
                    existing = node.get(f"{PROV}wasAssociatedWith")
                    if existing is None:
                        node[f"{PROV}wasAssociatedWith"] = {"@id": agent_id}
                    elif isinstance(existing, dict):
                        node[f"{PROV}wasAssociatedWith"] = [
                            existing, {"@id": agent_id},
                        ]
                    elif isinstance(existing, list):
                        existing.append({"@id": agent_id})
                    output_triples += 1
                    break

    # ── Map entity roles to PROV-O properties ─────────────────────
    for ent in entities:
        role = ent.get("role")
        what = ent.get("what")

        if what is None:
            continue

        prov_prop = _ENTITY_ROLE_TO_PROV.get(role)

        # Create a prov:Entity for the source/related entity
        source_entity_id = what
        source_node: dict[str, Any] = {
            "@id": source_entity_id,
            "@type": f"{PROV}Entity",
        }
        output_triples += 1
        graph_nodes.append(source_node)

        # Link each target entity to the source entity via the role property
        if prov_prop is not None:
            for eid in target_entity_ids:
                for node in graph_nodes:
                    if node.get("@id") == eid:
                        prop_iri = f"{PROV}{prov_prop}"
                        existing = node.get(prop_iri)
                        if existing is None:
                            node[prop_iri] = {"@id": source_entity_id}
                        elif isinstance(existing, dict):
                            node[prop_iri] = [
                                existing, {"@id": source_entity_id},
                            ]
                        elif isinstance(existing, list):
                            existing.append({"@id": source_entity_id})
                        output_triples += 1
                        break

    # ── Attach SL opinion metadata ────────────────────────────────
    if opinions:
        op = opinions[0].get("opinion")
        if op is not None:
            pp = op.projected_probability()
            # Attach confidence to the first target entity
            for node in graph_nodes:
                if node.get("@id") in target_entity_ids:
                    node[f"{JSONLD_EX}confidence"] = pp
                    node[f"{JSONLD_EX}opinion"] = {
                        "belief": op.belief,
                        "disbelief": op.disbelief,
                        "uncertainty": op.uncertainty,
                        "baseRate": op.base_rate,
                    }
                    output_triples += 2
                    break

    report.nodes_converted = len(graph_nodes)
    report.triples_output = output_triples

    # ── Build final PROV-O document ───────────────────────────────
    prov_context = {
        "prov": PROV,
        "xsd": XSD,
        "rdfs": RDFS,
        "jsonld-ex": JSONLD_EX,
    }

    prov_o_doc: dict[str, Any] = {
        "@context": prov_context,
        "@graph": graph_nodes,
    }

    return prov_o_doc, report
