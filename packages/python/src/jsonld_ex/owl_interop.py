"""
OWL/RDF Interoperability Extensions for JSON-LD-Ex.

Bidirectional mapping between jsonld-ex extensions and established
semantic web standards:
  - PROV-O: W3C Provenance Ontology
  - SHACL:  Shapes Constraint Language
  - OWL:    Web Ontology Language class restrictions
  - RDF-star: Statements about statements

This module enables measured comparison against SOTA alternatives and
proves that jsonld-ex integrates with the existing ecosystem.
"""

from __future__ import annotations
import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from jsonld_ex.ai_ml import (
    JSONLD_EX_NAMESPACE,
    get_confidence,
    get_provenance,
    ProvenanceMetadata,
)

# ── Namespace Constants ────────────────────────────────────────────

PROV = "http://www.w3.org/ns/prov#"
SHACL = "http://www.w3.org/ns/shacl#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
JSONLD_EX = JSONLD_EX_NAMESPACE  # http://www.w3.org/ns/jsonld-ex/
SOSA = "http://www.w3.org/ns/sosa/"
SSN = "http://www.w3.org/ns/ssn/"
SSN_SYSTEM = "http://www.w3.org/ns/ssn/systems/"
QUDT = "http://qudt.org/schema/qudt/"


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class ConversionReport:
    """Report from a conversion operation."""
    success: bool
    nodes_converted: int = 0
    triples_input: int = 0
    triples_output: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def compression_ratio(self) -> float:
        """Ratio of output triples to input triples (lower = more compact)."""
        if self.triples_input == 0:
            return 0.0
        return self.triples_output / self.triples_input


# ═══════════════════════════════════════════════════════════════════
# PROV-O MAPPING
# ═══════════════════════════════════════════════════════════════════

def to_prov_o(doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex annotations to PROV-O graph.

    Walks the document looking for annotated values (those with
    @confidence, @source, @extractedAt, @method, or @humanVerified).
    Each annotated value is converted into PROV-O entities:

    Mapping:
        @value         → prov:Entity (prov:value = the literal)
        @source        → prov:SoftwareAgent + prov:wasAttributedTo
        @extractedAt   → prov:generatedAtTime on the Entity
        @method        → prov:Activity (rdfs:label = method) + prov:wasGeneratedBy
        @confidence    → jsonld-ex:confidence on the Entity (no PROV equivalent)
        @humanVerified → prov:wasAttributedTo prov:Person (if True)

    Args:
        doc: A JSON-LD document (compact form) with jsonld-ex annotations.

    Returns:
        Tuple of (PROV-O JSON-LD document, ConversionReport).
    """
    report = ConversionReport(success=True)
    graph_nodes: list[dict[str, Any]] = []
    input_triples = 0
    output_triples = 0

    # Context for the output PROV-O document
    prov_context = {
        "prov": PROV,
        "xsd": XSD,
        "rdfs": RDFS,
        "jsonld-ex": JSONLD_EX,
    }

    # Preserve original context entries
    original_ctx = doc.get("@context", {})
    if isinstance(original_ctx, str):
        prov_context["_original"] = original_ctx
    elif isinstance(original_ctx, dict):
        for k, v in original_ctx.items():
            if k not in prov_context:
                prov_context[k] = v
    elif isinstance(original_ctx, list):
        for item in original_ctx:
            if isinstance(item, dict):
                for k, v in item.items():
                    if k not in prov_context:
                        prov_context[k] = v

    def _process_node(node: dict[str, Any], parent_id: Optional[str] = None) -> dict[str, Any]:
        nonlocal input_triples, output_triples
        processed = {}
        node_id = node.get("@id", parent_id or f"_:node-{uuid.uuid4().hex[:8]}")

        for key, value in node.items():
            if key == "@context":
                continue

            if isinstance(value, dict) and "@value" in value:
                prov = get_provenance(value)
                has_annotations = any([
                    prov.confidence is not None,
                    prov.source is not None,
                    prov.extracted_at is not None,
                    prov.method is not None,
                    prov.human_verified is not None,
                    prov.derived_from is not None,
                    prov.delegated_by is not None,
                    prov.invalidated_at is not None,
                    prov.invalidation_reason is not None,
                ])

                if has_annotations:
                    input_triples += 1  # The original annotated triple

                    # Create PROV-O Entity for the extracted value
                    entity_id = f"_:entity-{uuid.uuid4().hex[:8]}"
                    entity: dict[str, Any] = {
                        "@id": entity_id,
                        "@type": f"{PROV}Entity",
                        f"{PROV}value": value["@value"],
                    }
                    output_triples += 2  # type + value

                    # Link back: subject --property--> entity
                    processed[key] = {"@id": entity_id}
                    output_triples += 1

                    if prov.confidence is not None:
                        entity[f"{JSONLD_EX}confidence"] = prov.confidence
                        output_triples += 1

                    if prov.source is not None:
                        agent_id = prov.source
                        agent: dict[str, Any] = {
                            "@id": agent_id,
                            "@type": f"{PROV}SoftwareAgent",
                        }
                        entity[f"{PROV}wasAttributedTo"] = {"@id": agent_id}

                        # Delegation: agent actedOnBehalfOf delegator(s)
                        if prov.delegated_by is not None:
                            delegates = prov.delegated_by
                            if isinstance(delegates, str):
                                agent[f"{PROV}actedOnBehalfOf"] = {"@id": delegates}
                                output_triples += 1
                            elif isinstance(delegates, list):
                                agent[f"{PROV}actedOnBehalfOf"] = [
                                    {"@id": d} for d in delegates
                                ]
                                output_triples += len(delegates)

                        graph_nodes.append(agent)
                        output_triples += 2  # agent type + attribution

                    if prov.extracted_at is not None:
                        entity[f"{PROV}generatedAtTime"] = {
                            "@value": prov.extracted_at,
                            "@type": f"{XSD}dateTime",
                        }
                        output_triples += 1

                    if prov.method is not None:
                        activity_id = f"_:activity-{uuid.uuid4().hex[:8]}"
                        activity: dict[str, Any] = {
                            "@id": activity_id,
                            "@type": f"{PROV}Activity",
                            f"{RDFS}label": prov.method,
                        }
                        entity[f"{PROV}wasGeneratedBy"] = {"@id": activity_id}

                        if prov.source is not None:
                            activity[f"{PROV}wasAssociatedWith"] = {"@id": prov.source}
                            output_triples += 1

                        graph_nodes.append(activity)
                        output_triples += 3  # activity type + label + wasGeneratedBy

                    if prov.derived_from is not None:
                        sources = prov.derived_from
                        if isinstance(sources, str):
                            entity[f"{PROV}wasDerivedFrom"] = {"@id": sources}
                            output_triples += 1
                        elif isinstance(sources, list):
                            entity[f"{PROV}wasDerivedFrom"] = [
                                {"@id": s} for s in sources
                            ]
                            output_triples += len(sources)

                    if prov.human_verified is True:
                        verifier_id = f"_:human-verifier-{uuid.uuid4().hex[:8]}"
                        verifier: dict[str, Any] = {
                            "@id": verifier_id,
                            "@type": f"{PROV}Person",
                            f"{RDFS}label": "Human Verifier",
                        }
                        entity[f"{PROV}wasAttributedTo"] = [
                            entity.get(f"{PROV}wasAttributedTo", {}),
                            {"@id": verifier_id},
                        ] if f"{PROV}wasAttributedTo" in entity else {"@id": verifier_id}
                        graph_nodes.append(verifier)
                        output_triples += 2  # verifier type + attribution

                    # Invalidation → prov:wasInvalidatedBy
                    if prov.invalidated_at is not None or prov.invalidation_reason is not None:
                        inv_activity_id = f"_:invalidation-{uuid.uuid4().hex[:8]}"
                        inv_activity: dict[str, Any] = {
                            "@id": inv_activity_id,
                            "@type": f"{PROV}Activity",
                        }
                        if prov.invalidated_at is not None:
                            inv_activity[f"{PROV}atTime"] = {
                                "@value": prov.invalidated_at,
                                "@type": f"{XSD}dateTime",
                            }
                            output_triples += 1
                        if prov.invalidation_reason is not None:
                            inv_activity[f"{RDFS}label"] = prov.invalidation_reason
                            output_triples += 1
                        entity[f"{PROV}wasInvalidatedBy"] = {"@id": inv_activity_id}
                        graph_nodes.append(inv_activity)
                        output_triples += 2  # activity type + wasInvalidatedBy link

                    graph_nodes.append(entity)
                    report.nodes_converted += 1
                else:
                    processed[key] = value
            elif isinstance(value, dict) and "@value" not in value and key != "@graph":
                # Nested node — recurse
                processed[key] = _process_node(value, None)
            elif isinstance(value, list):
                processed_list = []
                for item in value:
                    if isinstance(item, dict) and "@value" in item:
                        # Recursively handle annotated values in lists
                        sub_node = {key: item}
                        sub_result = _process_node(sub_node, node_id)
                        processed_list.append(sub_result.get(key, item))
                    else:
                        processed_list.append(item)
                processed[key] = processed_list
            else:
                processed[key] = value

        return processed

    # Process the main document (read-only traversal — no deepcopy needed;
    # _process_node builds entirely new dicts and never mutates its input)
    main_node = _process_node(doc)

    # Remove @context from main_node (we'll add our own)
    main_node.pop("@context", None)

    # Build final PROV-O document
    all_nodes = [main_node] + graph_nodes
    prov_doc: dict[str, Any] = {
        "@context": prov_context,
        "@graph": all_nodes,
    }

    report.triples_input = input_triples
    report.triples_output = output_triples
    return prov_doc, report


def to_prov_o_graph(doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Batch-convert a JSON-LD document with @graph to PROV-O.

    Equivalent to calling to_prov_o per-node but processes the entire
    @graph array in a single pass, producing one unified PROV-O graph.

    Args:
        doc: A JSON-LD document with @graph array containing annotated nodes.

    Returns:
        Tuple of (PROV-O JSON-LD document, ConversionReport).
    """
    nodes = doc.get("@graph", [])
    base_context = doc.get("@context", {})

    combined_graph: list[dict[str, Any]] = []
    report = ConversionReport(success=True)

    for node in nodes:
        # Build a standalone single-node document
        single: dict[str, Any] = {"@context": base_context}
        single.update(node)

        prov_doc, node_report = to_prov_o(single)

        # Accumulate graph nodes from each conversion
        combined_graph.extend(prov_doc.get("@graph", []))
        report.nodes_converted += node_report.nodes_converted
        report.triples_input += node_report.triples_input
        report.triples_output += node_report.triples_output
        report.warnings.extend(node_report.warnings)
        report.errors.extend(node_report.errors)

    # Build unified PROV-O context
    prov_context: dict[str, Any] = {
        "prov": PROV,
        "xsd": XSD,
        "rdfs": RDFS,
        "jsonld-ex": JSONLD_EX,
    }
    if isinstance(base_context, str):
        prov_context["_original"] = base_context
    elif isinstance(base_context, dict):
        for k, v in base_context.items():
            if k not in prov_context:
                prov_context[k] = v

    return {"@context": prov_context, "@graph": combined_graph}, report


def from_prov_o(prov_doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert PROV-O provenance graph back to jsonld-ex inline annotations.

    Traverses PROV-O Entity→Activity→Agent chains and collapses them
    into inline @confidence/@source/@method annotations on values.

    Args:
        prov_doc: A JSON-LD document using PROV-O vocabulary.

    Returns:
        Tuple of (jsonld-ex annotated document, ConversionReport).
    """
    report = ConversionReport(success=True)

    graph = prov_doc.get("@graph", [])
    if not isinstance(graph, list):
        graph = [graph]

    # Index all nodes by @id
    nodes_by_id: dict[str, dict] = {}
    for node in graph:
        if isinstance(node, dict) and "@id" in node:
            nodes_by_id[node["@id"]] = node

    # Find prov:Entity nodes — these are the annotated values
    entity_type = f"{PROV}Entity"
    entities: dict[str, dict] = {}
    for nid, node in nodes_by_id.items():
        node_types = node.get("@type", [])
        if isinstance(node_types, str):
            node_types = [node_types]
        if entity_type in node_types or "prov:Entity" in node_types:
            entities[nid] = node

    # Find the "main" node (non-PROV-O typed node, or first non-entity)
    main_node: Optional[dict] = None
    for node in graph:
        if not isinstance(node, dict):
            continue
        node_types = node.get("@type", [])
        if isinstance(node_types, str):
            node_types = [node_types]
        is_prov_type = any(
            t.startswith(PROV) or t.startswith("prov:")
            for t in node_types
        )
        if not is_prov_type and node_types:
            main_node = copy.deepcopy(node)
            break

    if main_node is None:
        report.success = False
        report.errors.append("No main (non-PROV) node found in graph")
        return prov_doc, report

    # For each property on the main node, check if it references an entity
    for key, value in list(main_node.items()):
        if key.startswith("@"):
            continue

        ref_id = None
        if isinstance(value, dict) and "@id" in value:
            ref_id = value["@id"]
        elif isinstance(value, str) and value in entities:
            ref_id = value

        if ref_id and ref_id in entities:
            entity = entities[ref_id]
            annotated = _entity_to_annotation(entity, nodes_by_id)
            if annotated is not None:
                main_node[key] = annotated
                report.nodes_converted += 1

    # Clean up context
    main_node.pop("@context", None)
    result: dict[str, Any] = {"@context": prov_doc.get("@context", {})}
    result.update(main_node)

    return result, report


def _entity_to_annotation(
    entity: dict[str, Any],
    all_nodes: dict[str, dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Convert a PROV-O Entity back to a jsonld-ex annotated value."""
    raw_value = entity.get(f"{PROV}value") or entity.get("prov:value")
    if raw_value is None:
        return None

    # Extract the literal value
    if isinstance(raw_value, dict) and "@value" in raw_value:
        literal = raw_value["@value"]
    else:
        literal = raw_value

    result: dict[str, Any] = {"@value": literal}

    # Confidence
    conf = entity.get(f"{JSONLD_EX}confidence")
    if conf is None:
        conf = entity.get("jsonld-ex:confidence")
    if conf is not None:
        result["@confidence"] = conf if not isinstance(conf, dict) else conf.get("@value", conf)

    # Source (from prov:wasAttributedTo → SoftwareAgent)
    attr = entity.get(f"{PROV}wasAttributedTo") or entity.get("prov:wasAttributedTo")
    if attr is not None:
        attr_list = attr if isinstance(attr, list) else [attr]
        for a in attr_list:
            agent_id = a.get("@id") if isinstance(a, dict) else a
            if agent_id and agent_id in all_nodes:
                agent = all_nodes[agent_id]
                agent_types = agent.get("@type", [])
                if isinstance(agent_types, str):
                    agent_types = [agent_types]
                if f"{PROV}SoftwareAgent" in agent_types or "prov:SoftwareAgent" in agent_types:
                    result["@source"] = agent_id
                elif f"{PROV}Person" in agent_types or "prov:Person" in agent_types:
                    result["@humanVerified"] = True

    # ExtractedAt (from prov:generatedAtTime)
    gen_time = entity.get(f"{PROV}generatedAtTime") or entity.get("prov:generatedAtTime")
    if gen_time is not None:
        if isinstance(gen_time, dict) and "@value" in gen_time:
            result["@extractedAt"] = gen_time["@value"]
        else:
            result["@extractedAt"] = gen_time

    # Method (from prov:wasGeneratedBy → Activity → rdfs:label)
    gen_by = entity.get(f"{PROV}wasGeneratedBy") or entity.get("prov:wasGeneratedBy")
    if gen_by is not None:
        activity_id = gen_by.get("@id") if isinstance(gen_by, dict) else gen_by
        if activity_id and activity_id in all_nodes:
            activity = all_nodes[activity_id]
            label = (
                activity.get(f"{RDFS}label")
                or activity.get("rdfs:label")
            )
            if label is not None:
                result["@method"] = label if not isinstance(label, dict) else label.get("@value", label)

    # DerivedFrom (from prov:wasDerivedFrom)
    derived = entity.get(f"{PROV}wasDerivedFrom") or entity.get("prov:wasDerivedFrom")
    if derived is not None:
        if isinstance(derived, list):
            result["@derivedFrom"] = [
                d.get("@id") if isinstance(d, dict) else d
                for d in derived
            ]
        elif isinstance(derived, dict):
            result["@derivedFrom"] = derived.get("@id", derived)
        else:
            result["@derivedFrom"] = derived

    # Invalidation (from prov:wasInvalidatedBy → Activity)
    inv_by = entity.get(f"{PROV}wasInvalidatedBy") or entity.get("prov:wasInvalidatedBy")
    if inv_by is not None:
        inv_id = inv_by.get("@id") if isinstance(inv_by, dict) else inv_by
        if inv_id and inv_id in all_nodes:
            inv_activity = all_nodes[inv_id]
            at_time = inv_activity.get(f"{PROV}atTime") or inv_activity.get("prov:atTime")
            if at_time is not None:
                if isinstance(at_time, dict) and "@value" in at_time:
                    result["@invalidatedAt"] = at_time["@value"]
                else:
                    result["@invalidatedAt"] = at_time
            inv_label = inv_activity.get(f"{RDFS}label") or inv_activity.get("rdfs:label")
            if inv_label is not None:
                result["@invalidationReason"] = inv_label if not isinstance(inv_label, dict) else inv_label.get("@value", inv_label)

    # DelegatedBy (from prov:actedOnBehalfOf on the SoftwareAgent)
    if attr is not None:
        attr_list_d = attr if isinstance(attr, list) else [attr]
        for a in attr_list_d:
            agent_id_d = a.get("@id") if isinstance(a, dict) else a
            if agent_id_d and agent_id_d in all_nodes:
                agent_d = all_nodes[agent_id_d]
                behalf = (
                    agent_d.get(f"{PROV}actedOnBehalfOf")
                    or agent_d.get("prov:actedOnBehalfOf")
                )
                if behalf is not None:
                    if isinstance(behalf, list):
                        result["@delegatedBy"] = [
                            b.get("@id") if isinstance(b, dict) else b
                            for b in behalf
                        ]
                    elif isinstance(behalf, dict):
                        result["@delegatedBy"] = behalf.get("@id", behalf)
                    else:
                        result["@delegatedBy"] = behalf

    return result


# ═══════════════════════════════════════════════════════════════════
# SHACL MAPPING
# ═══════════════════════════════════════════════════════════════════

def shape_to_shacl(
    shape: dict[str, Any],
    target_class: Optional[str] = None,
    shape_iri: Optional[str] = None,
) -> dict[str, Any]:
    """Convert jsonld-ex @shape to SHACL shape graph (JSON-LD serialized).

    Mapping:
        @required               → sh:minCount 1
        @type (xsd datatype)    → sh:datatype
        @minimum                → sh:minInclusive
        @maximum                → sh:maxInclusive
        @minLength              → sh:minLength
        @maxLength              → sh:maxLength
        @pattern                → sh:pattern

    Args:
        shape: A jsonld-ex @shape definition dict.
        target_class: IRI of the target class (defaults to shape's @type).
        shape_iri: IRI for the generated SHACL shape node.

    Returns:
        A SHACL shape graph as JSON-LD.
    """
    target = target_class or shape.get("@type")
    if target is None:
        raise ValueError("Shape must have @type or target_class must be provided")

    shape_id = shape_iri or f"_:shape-{uuid.uuid4().hex[:8]}"

    shacl_context = {
        "sh": SHACL,
        "xsd": XSD,
        "rdfs": RDFS,
    }

    properties = []
    for prop_name, constraint in shape.items():
        if prop_name.startswith("@") or not isinstance(constraint, dict):
            continue

        sh_property: dict[str, Any] = {
            f"{SHACL}path": {"@id": prop_name},
        }

        # -- Cardinality: @required / @minCount / @maxCount -------------------
        # If @minCount is explicitly set, use it (takes precedence over @required).
        # Otherwise, @required maps to sh:minCount 1.
        if "@minCount" in constraint:
            sh_property[f"{SHACL}minCount"] = constraint["@minCount"]
        elif constraint.get("@required"):
            sh_property[f"{SHACL}minCount"] = 1

        if "@maxCount" in constraint:
            sh_property[f"{SHACL}maxCount"] = constraint["@maxCount"]

        # -- Datatype ---------------------------------------------------------
        xsd_type = constraint.get("@type")
        if xsd_type:
            resolved = xsd_type.replace("xsd:", XSD) if xsd_type.startswith("xsd:") else xsd_type
            sh_property[f"{SHACL}datatype"] = {"@id": resolved}

        # -- Numeric range ----------------------------------------------------
        if "@minimum" in constraint:
            sh_property[f"{SHACL}minInclusive"] = constraint["@minimum"]

        if "@maximum" in constraint:
            sh_property[f"{SHACL}maxInclusive"] = constraint["@maximum"]

        # -- String length ----------------------------------------------------
        if "@minLength" in constraint:
            sh_property[f"{SHACL}minLength"] = constraint["@minLength"]

        if "@maxLength" in constraint:
            sh_property[f"{SHACL}maxLength"] = constraint["@maxLength"]

        # -- Enumeration: @in → sh:in (RDF list) ------------------------------
        if "@in" in constraint:
            sh_property[f"{SHACL}in"] = {"@list": constraint["@in"]}

        # -- Pattern ----------------------------------------------------------
        if "@pattern" in constraint:
            sh_property[f"{SHACL}pattern"] = constraint["@pattern"]

        # -- Logical combinators: @or / @and / @not ---------------------------
        if "@or" in constraint:
            sh_property[f"{SHACL}or"] = {
                "@list": [_constraint_to_shacl(branch) for branch in constraint["@or"]],
            }

        if "@and" in constraint:
            sh_property[f"{SHACL}and"] = {
                "@list": [_constraint_to_shacl(branch) for branch in constraint["@and"]],
            }

        if "@not" in constraint:
            sh_property[f"{SHACL}not"] = _constraint_to_shacl(constraint["@not"])

        # -- Conditional: @if/@then/@else (GAP-V7) ----------------------------
        if "@if" in constraint:
            if_shacl = _constraint_to_shacl(constraint["@if"])
            then_shacl = _constraint_to_shacl(constraint.get("@then", {}))
            has_else = "@else" in constraint

            if has_else:
                else_shacl = _constraint_to_shacl(constraint["@else"])
                # (P ∧ Q) ∨ (¬P ∧ R)
                sh_property[f"{SHACL}or"] = {
                    "@list": [
                        {f"{SHACL}and": {"@list": [if_shacl, then_shacl]}},
                        {f"{SHACL}and": {"@list": [{f"{SHACL}not": if_shacl}, else_shacl]}},
                    ],
                    f"{JSONLD_EX}conditionalType": "if-then-else",
                }
            else:
                # ¬P ∨ Q
                sh_property[f"{SHACL}or"] = {
                    "@list": [
                        {f"{SHACL}not": if_shacl},
                        then_shacl,
                    ],
                    f"{JSONLD_EX}conditionalType": "if-then",
                }

        # -- Cross-property constraints ---------------------------------------
        if "@lessThan" in constraint:
            sh_property[f"{SHACL}lessThan"] = {"@id": constraint["@lessThan"]}

        if "@lessThanOrEquals" in constraint:
            sh_property[f"{SHACL}lessThanOrEquals"] = {"@id": constraint["@lessThanOrEquals"]}

        if "@equals" in constraint:
            sh_property[f"{SHACL}equals"] = {"@id": constraint["@equals"]}

        if "@disjoint" in constraint:
            sh_property[f"{SHACL}disjoint"] = {"@id": constraint["@disjoint"]}

        properties.append(sh_property)

    shacl_shape: dict[str, Any] = {
        "@id": shape_id,
        "@type": f"{SHACL}NodeShape",
        f"{SHACL}targetClass": {"@id": target},
        f"{SHACL}property": properties,
    }

    extra_shapes: list[dict[str, Any]] = []

    # -- @extends → sh:node + parent NodeShape(s) (GAP-OWL1) -----------------
    extends_raw = shape.get("@extends")
    if extends_raw is not None:
        parents = extends_raw if isinstance(extends_raw, list) else [extends_raw]
        parent_ids: list[str] = []
        for parent in parents:
            if isinstance(parent, dict):
                parent_shacl = shape_to_shacl(parent)
                parent_graph = parent_shacl.get("@graph", [])
                if parent_graph:
                    parent_node = parent_graph[0]
                    parent_ids.append(parent_node["@id"])
                    extra_shapes.extend(parent_graph)

        if len(parent_ids) == 1:
            shacl_shape[f"{SHACL}node"] = {"@id": parent_ids[0]}
            shacl_shape[f"{JSONLD_EX}extends"] = {"@id": parent_ids[0]}
        elif parent_ids:
            shacl_shape[f"{SHACL}node"] = [{"@id": pid} for pid in parent_ids]
            shacl_shape[f"{JSONLD_EX}extends"] = [{"@id": pid} for pid in parent_ids]

    return {
        "@context": shacl_context,
        "@graph": [shacl_shape] + extra_shapes,
    }


def shacl_to_shape(shacl_doc: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Convert SHACL shape graph to jsonld-ex @shape.

    Inverse of shape_to_shacl. Handles common SHACL property constraints.
    Returns warnings for SHACL features without @shape equivalent.

    Args:
        shacl_doc: A SHACL shape graph as JSON-LD.

    Returns:
        Tuple of (jsonld-ex @shape dict, list of warning strings).
    """
    warnings: list[str] = []

    graph = shacl_doc.get("@graph", [])
    if not isinstance(graph, list):
        graph = [graph]

    # Find NodeShape(s)
    shape_nodes = []
    for node in graph:
        if not isinstance(node, dict):
            continue
        node_types = node.get("@type", [])
        if isinstance(node_types, str):
            node_types = [node_types]
        if f"{SHACL}NodeShape" in node_types or "sh:NodeShape" in node_types:
            shape_nodes.append(node)

    if not shape_nodes:
        return {}, ["No sh:NodeShape found in document"]

    # Convert first shape (multi-shape support could be added later)
    shacl_shape = shape_nodes[0]
    if len(shape_nodes) > 1:
        warnings.append(f"Multiple NodeShapes found; converting first only ({shacl_shape.get('@id', 'anonymous')})")

    result: dict[str, Any] = {}

    # Target class → @type
    target = (
        shacl_shape.get(f"{SHACL}targetClass")
        or shacl_shape.get("sh:targetClass")
    )
    if target is not None:
        result["@type"] = target.get("@id") if isinstance(target, dict) else target

    # Properties
    props = (
        shacl_shape.get(f"{SHACL}property")
        or shacl_shape.get("sh:property")
        or []
    )
    if not isinstance(props, list):
        props = [props]

    for prop in props:
        if not isinstance(prop, dict):
            continue

        path = prop.get(f"{SHACL}path") or prop.get("sh:path")
        if path is None:
            warnings.append("Property constraint without sh:path — skipped")
            continue
        prop_name = path.get("@id") if isinstance(path, dict) else path

        constraint: dict[str, Any] = {}

        # sh:minCount → @required (=1) or @minCount (>1)
        min_count = prop.get(f"{SHACL}minCount") or prop.get("sh:minCount")
        if min_count is not None:
            mc = int(min_count)
            if mc == 1:
                constraint["@required"] = True
            elif mc > 1:
                constraint["@minCount"] = mc

        # sh:maxCount → @maxCount
        max_count = prop.get(f"{SHACL}maxCount") or prop.get("sh:maxCount")
        if max_count is not None:
            constraint["@maxCount"] = int(max_count)

        # sh:datatype → @type
        datatype = prop.get(f"{SHACL}datatype") or prop.get("sh:datatype")
        if datatype is not None:
            dt_iri = datatype.get("@id") if isinstance(datatype, dict) else datatype
            # Convert back to xsd: prefix form for readability
            if dt_iri.startswith(XSD):
                constraint["@type"] = "xsd:" + dt_iri[len(XSD):]
            else:
                constraint["@type"] = dt_iri

        # Numeric constraints
        min_inc = prop.get(f"{SHACL}minInclusive")
        if min_inc is None:
            min_inc = prop.get("sh:minInclusive")
        if min_inc is not None:
            constraint["@minimum"] = min_inc

        max_inc = prop.get(f"{SHACL}maxInclusive")
        if max_inc is None:
            max_inc = prop.get("sh:maxInclusive")
        if max_inc is not None:
            constraint["@maximum"] = max_inc

        # String length constraints
        min_len = prop.get(f"{SHACL}minLength")
        if min_len is None:
            min_len = prop.get("sh:minLength")
        if min_len is not None:
            constraint["@minLength"] = min_len

        max_len = prop.get(f"{SHACL}maxLength")
        if max_len is None:
            max_len = prop.get("sh:maxLength")
        if max_len is not None:
            constraint["@maxLength"] = max_len

        # Pattern
        pattern = prop.get(f"{SHACL}pattern") or prop.get("sh:pattern")
        if pattern is not None:
            constraint["@pattern"] = pattern

        # sh:in → @in
        sh_in = prop.get(f"{SHACL}in") or prop.get("sh:in")
        if sh_in is not None:
            if isinstance(sh_in, dict) and "@list" in sh_in:
                constraint["@in"] = sh_in["@list"]
            elif isinstance(sh_in, list):
                constraint["@in"] = sh_in

        # -- Conditional (@if/@then/@else) detection (GAP-V7) ----------------
        sh_or = prop.get(f"{SHACL}or") or prop.get("sh:or")
        _conditional_handled = False
        if isinstance(sh_or, dict):
            cond_type = sh_or.get(f"{JSONLD_EX}conditionalType")
            if cond_type == "if-then":
                branches = sh_or.get("@list", [])
                if len(branches) == 2:
                    # Branch 0: sh:not(if_constraints) → extract if
                    not_node = branches[0].get(f"{SHACL}not", {})
                    constraint["@if"] = _shacl_to_constraint(not_node)
                    # Branch 1: then_constraints
                    constraint["@then"] = _shacl_to_constraint(branches[1])
                    _conditional_handled = True
            elif cond_type == "if-then-else":
                branches = sh_or.get("@list", [])
                if len(branches) == 2:
                    # Branch 0: sh:and([if, then])
                    and0 = branches[0].get(f"{SHACL}and", {})
                    and0_items = and0.get("@list", []) if isinstance(and0, dict) else []
                    if len(and0_items) == 2:
                        constraint["@if"] = _shacl_to_constraint(and0_items[0])
                        constraint["@then"] = _shacl_to_constraint(and0_items[1])
                    # Branch 1: sh:and([sh:not(if), else])
                    and1 = branches[1].get(f"{SHACL}and", {})
                    and1_items = and1.get("@list", []) if isinstance(and1, dict) else []
                    if len(and1_items) == 2:
                        constraint["@else"] = _shacl_to_constraint(and1_items[1])
                    _conditional_handled = True

        # sh:or / sh:and / sh:not → @or / @and / @not
        if sh_or is not None and not _conditional_handled:
            branches = sh_or.get("@list", sh_or) if isinstance(sh_or, dict) else sh_or
            constraint["@or"] = [_shacl_to_constraint(b) for b in branches]

        sh_and = prop.get(f"{SHACL}and") or prop.get("sh:and")
        if sh_and is not None:
            branches = sh_and.get("@list", sh_and) if isinstance(sh_and, dict) else sh_and
            constraint["@and"] = [_shacl_to_constraint(b) for b in branches]

        sh_not = prop.get(f"{SHACL}not") or prop.get("sh:not")
        if sh_not is not None:
            constraint["@not"] = _shacl_to_constraint(sh_not)

        # Cross-property constraints
        for shacl_key, shape_key in [
            ("lessThan", "@lessThan"), ("lessThanOrEquals", "@lessThanOrEquals"),
            ("equals", "@equals"), ("disjoint", "@disjoint"),
        ]:
            val = prop.get(f"{SHACL}{shacl_key}") or prop.get(f"sh:{shacl_key}")
            if val is not None:
                constraint[shape_key] = val.get("@id") if isinstance(val, dict) else val

        # Warn on unsupported SHACL features
        unsupported_keys = [
            (f"{SHACL}sparql", "sh:sparql"),
            (f"{SHACL}qualifiedValueShape", "sh:qualifiedValueShape"),
            (f"{SHACL}class", "sh:class"),
            (f"{SHACL}node", "sh:node"),
            (f"{SHACL}hasValue", "sh:hasValue"),
            (f"{SHACL}uniqueLang", "sh:uniqueLang"),
        ]
        for full_key, short_key in unsupported_keys:
            if full_key in prop or short_key in prop:
                warnings.append(
                    f"SHACL constraint {short_key} on '{prop_name}' has no @shape equivalent"
                )

        if constraint:
            result[prop_name] = constraint

    # -- @extends reconstruction (GAP-OWL1) -----------------------------------
    extends_ref = (
        shacl_shape.get(f"{JSONLD_EX}extends")
        or shacl_shape.get("jsonld-ex:extends")
    )
    if extends_ref is not None:
        refs = extends_ref if isinstance(extends_ref, list) else [extends_ref]
        parent_shapes: list[dict[str, Any]] = []
        # Build lookup of all shapes by @id
        shapes_by_id = {s.get("@id"): s for s in shape_nodes if s.get("@id")}
        # Also include shapes beyond shape_nodes (they may be in the full graph)
        for node in graph:
            if isinstance(node, dict) and node.get("@id"):
                shapes_by_id.setdefault(node["@id"], node)

        for ref in refs:
            pid = ref.get("@id") if isinstance(ref, dict) else ref
            parent_node = shapes_by_id.get(pid)
            if parent_node is not None:
                # Convert parent NodeShape to jsonld-ex shape
                parent_doc = {
                    "@context": shacl_doc.get("@context", {}),
                    "@graph": [parent_node],
                }
                parent_shape, _pw = shacl_to_shape(parent_doc)
                parent_shapes.append(parent_shape)

        if len(parent_shapes) == 1:
            result["@extends"] = parent_shapes[0]
        elif parent_shapes:
            result["@extends"] = parent_shapes

    return result, warnings


# ═══════════════════════════════════════════════════════════════════
# OWL MAPPING
# ═══════════════════════════════════════════════════════════════════

def shape_to_owl_restrictions(
    shape: dict[str, Any],
    class_iri: Optional[str] = None,
) -> dict[str, Any]:
    """Convert jsonld-ex @shape to OWL class restrictions.

    Mapping:
        @required               → owl:minCardinality 1
        @minCount               → owl:minCardinality (takes precedence over @required)
        @maxCount               → owl:maxCardinality
        @type (xsd)             → owl:allValuesFrom + xsd:datatype
        @minimum/@maximum       → owl:DatatypeRestriction + xsd:minInclusive/maxInclusive
        @minLength/@maxLength   → owl:DatatypeRestriction + xsd:minLength/maxLength
        @pattern                → owl:DatatypeRestriction + xsd:pattern
        @in                     → owl:allValuesFrom + owl:oneOf

    When @type is combined with any facets above, they merge into a single
    OWL 2 DatatypeRestriction (§7.5 of the OWL 2 Structural Specification).
    When facets appear without @type, the base datatype defaults to
    xsd:string (for string facets) or xsd:decimal (for numeric facets).

    Args:
        shape: A jsonld-ex @shape definition dict.
        class_iri: IRI for the OWL class. Defaults to shape's @type.

    Returns:
        OWL axioms as JSON-LD.
    """
    target = class_iri or shape.get("@type")
    if target is None:
        raise ValueError("Shape must have @type or class_iri must be provided")

    owl_context = {
        "owl": OWL,
        "xsd": XSD,
        "rdfs": RDFS,
        "rdf": RDF,
    }

    restrictions = []
    unmappable_annotations: list[dict[str, Any]] = []

    for prop_name, constraint in shape.items():
        if prop_name.startswith("@") or not isinstance(constraint, dict):
            continue

        # Cardinality: @required / @minCount / @maxCount
        # @minCount takes precedence over @required when both are present.
        min_card = constraint.get("@minCount")
        if min_card is None and constraint.get("@required"):
            min_card = 1

        if min_card is not None:
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}minCardinality": {
                    "@value": min_card,
                    "@type": f"{XSD}nonNegativeInteger",
                },
            })

        if "@maxCount" in constraint:
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}maxCardinality": {
                    "@value": constraint["@maxCount"],
                    "@type": f"{XSD}nonNegativeInteger",
                },
            })

        # @type and/or XSD facets → owl:allValuesFrom
        # When @type is combined with constraining facets, they merge into
        # a single OWL 2 DatatypeRestriction (§7.5 of the structural spec).
        # Supported facets: @minimum/@maximum (numeric), @minLength/@maxLength
        # (string length), @pattern (regex).
        xsd_type = constraint.get("@type")

        # Collect all XSD constraining facets present on this property
        facets: list[dict[str, Any]] = []
        if "@minimum" in constraint:
            facets.append({f"{XSD}minInclusive": constraint["@minimum"]})
        if "@maximum" in constraint:
            facets.append({f"{XSD}maxInclusive": constraint["@maximum"]})
        if "@minLength" in constraint:
            facets.append({f"{XSD}minLength": constraint["@minLength"]})
        if "@maxLength" in constraint:
            facets.append({f"{XSD}maxLength": constraint["@maxLength"]})
        if "@pattern" in constraint:
            facets.append({f"{XSD}pattern": constraint["@pattern"]})

        if xsd_type:
            resolved = xsd_type.replace("xsd:", XSD) if xsd_type.startswith("xsd:") else xsd_type

            if facets:
                # Combined: type + facets → DatatypeRestriction
                restrictions.append({
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": prop_name},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}onDatatype": {"@id": resolved},
                        f"{OWL}withRestrictions": {"@list": facets},
                    },
                })
            else:
                # Simple type restriction (no facets)
                restrictions.append({
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": prop_name},
                    f"{OWL}allValuesFrom": {"@id": resolved},
                })
        elif facets:
            # Facets without explicit @type → default to xsd:string for
            # string facets (minLength/maxLength/pattern), xsd:decimal
            # for numeric facets (minimum/maximum).
            has_string_facets = any(
                f"{XSD}{k}" in f
                for f in facets
                for k in ("minLength", "maxLength", "pattern")
            )
            default_dt = f"{XSD}string" if has_string_facets else f"{XSD}decimal"

            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {
                    "@type": f"{RDFS}Datatype",
                    f"{OWL}onDatatype": {"@id": default_dt},
                    f"{OWL}withRestrictions": {"@list": facets},
                },
            })

        # @in → owl:allValuesFrom + owl:oneOf (enumerated datarange)
        if "@in" in constraint:
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {
                    "@type": f"{RDFS}Datatype",
                    f"{OWL}oneOf": {"@list": constraint["@in"]},
                },
            })

        # @or/@and/@not → OWL 2 DataRange expressions (§7 of OWL 2 spec)
        # @or  → owl:allValuesFrom with owl:unionOf
        # @and → owl:allValuesFrom with owl:intersectionOf
        # @not → owl:allValuesFrom with owl:datatypeComplementOf
        if "@or" in constraint:
            members = [_constraint_to_owl_datarange(b) for b in constraint["@or"]]
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {
                    "@type": f"{RDFS}Datatype",
                    f"{OWL}unionOf": {"@list": members},
                },
            })

        if "@and" in constraint:
            members = [_constraint_to_owl_datarange(b) for b in constraint["@and"]]
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {
                    "@type": f"{RDFS}Datatype",
                    f"{OWL}intersectionOf": {"@list": members},
                },
            })

        if "@not" in constraint:
            complement = _constraint_to_owl_datarange(constraint["@not"])
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {
                    "@type": f"{RDFS}Datatype",
                    f"{OWL}datatypeComplementOf": complement,
                },
            })

        # Task 7: Unmappable constraints → preserve in jex: namespace
        # These constraints have no OWL property-restriction equivalent.
        _UNMAPPABLE_KEYS = {
            "@lessThan": "lessThan",
            "@lessThanOrEquals": "lessThanOrEquals",
            "@equals": "equals",
            "@disjoint": "disjoint",
            "@severity": "severity",
        }
        for shape_key, jex_local in _UNMAPPABLE_KEYS.items():
            if shape_key in constraint:
                unmappable_annotations.append({
                    "property": prop_name,
                    "key": f"{JSONLD_EX}{jex_local}",
                    "value": constraint[shape_key],
                })

        # @if/@then/@else → no OWL equivalent, preserve entire conditional
        if "@if" in constraint:
            cond: dict[str, Any] = {"@if": constraint["@if"]}
            if "@then" in constraint:
                cond["@then"] = constraint["@then"]
            if "@else" in constraint:
                cond["@else"] = constraint["@else"]
            unmappable_annotations.append({
                "property": prop_name,
                "key": f"{JSONLD_EX}conditional",
                "value": cond,
            })

    # Task 6: @extends → rdfs:subClassOf parent class IRI(s)
    extends_raw = shape.get("@extends")
    if extends_raw is not None:
        parents = extends_raw if isinstance(extends_raw, list) else [extends_raw]
        for parent in parents:
            parent_iri = parent if isinstance(parent, str) else parent.get("@type", parent)
            restrictions.append({"@id": parent_iri})

    # Build OWL class with restrictions as superclasses
    owl_class: dict[str, Any] = {
        "@id": target,
        "@type": f"{OWL}Class",
    }
    if restrictions:
        owl_class[f"{RDFS}subClassOf"] = restrictions if len(restrictions) > 1 else restrictions[0]

    # Attach unmappable constraint annotations on the OWL class
    for ann in unmappable_annotations:
        owl_class[ann["key"]] = ann["value"]

    return {
        "@context": owl_context,
        "@graph": [owl_class],
    }


def owl_to_shape(owl_doc: dict[str, Any]) -> dict[str, Any]:
    """Convert OWL class restrictions back to a jsonld-ex @shape definition.

    Inverse of :func:`shape_to_owl_restrictions`.  Parses OWL Restriction
    nodes from ``rdfs:subClassOf`` and reconstructs the equivalent jsonld-ex
    constraint dictionary.

    Mapping (reverse of shape_to_owl_restrictions):
        owl:minCardinality 1          → @required: True
        owl:minCardinality N (N > 1)  → @minCount: N
        owl:maxCardinality            → @maxCount
        owl:allValuesFrom simple IRI  → @type
        owl:allValuesFrom DatatypeRestriction → @type + facets
        owl:allValuesFrom + owl:oneOf → @in
        owl:allValuesFrom + owl:unionOf → @or (recursive)
        owl:allValuesFrom + owl:intersectionOf → @and (recursive)
        owl:allValuesFrom + owl:datatypeComplementOf → @not (recursive)
        rdfs:subClassOf plain IRI     → @extends
        jex: namespace annotations    → unmappable constraints restored

    When the base datatype is a default (xsd:string for string facets,
    xsd:decimal for numeric facets), the @type is omitted to match the
    original jsonld-ex convention.

    Args:
        owl_doc: OWL axioms as JSON-LD (output of shape_to_owl_restrictions).

    Returns:
        A jsonld-ex @shape definition dict.
    """
    graph = owl_doc.get("@graph", [])
    if not isinstance(graph, list):
        graph = [graph]
    if not graph:
        raise ValueError("OWL document must contain at least one @graph node")

    owl_class = graph[0]
    class_iri = owl_class.get("@id")
    if class_iri is None:
        raise ValueError("OWL class node must have @id")

    shape: dict[str, Any] = {"@type": class_iri}

    # Collect rdfs:subClassOf entries
    sub_raw = owl_class.get(f"{RDFS}subClassOf")
    if sub_raw is None:
        # Check for jex: annotations even without restrictions
        _restore_jex_annotations(owl_class, shape)
        return shape

    entries = sub_raw if isinstance(sub_raw, list) else [sub_raw]

    # Separate OWL Restrictions from plain IRI superclasses (@extends)
    extends: list[str] = []
    properties: dict[str, dict[str, Any]] = {}

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        # Plain IRI reference (not a Restriction) → @extends
        if entry.get("@type") != f"{OWL}Restriction":
            iri = entry.get("@id")
            if iri:
                extends.append(iri)
            continue

        # It's an owl:Restriction — extract property and constraint
        prop_node = entry.get(f"{OWL}onProperty", {})
        prop_iri = prop_node.get("@id") if isinstance(prop_node, dict) else prop_node
        if not prop_iri:
            continue

        # Ensure property dict exists
        if prop_iri not in properties:
            properties[prop_iri] = {}
        prop = properties[prop_iri]

        # owl:minCardinality
        min_card_node = entry.get(f"{OWL}minCardinality")
        if min_card_node is not None:
            val = min_card_node.get("@value", min_card_node) if isinstance(min_card_node, dict) else min_card_node
            if val == 1:
                prop["@required"] = True
            else:
                prop["@minCount"] = val

        # owl:maxCardinality
        max_card_node = entry.get(f"{OWL}maxCardinality")
        if max_card_node is not None:
            val = max_card_node.get("@value", max_card_node) if isinstance(max_card_node, dict) else max_card_node
            prop["@maxCount"] = val

        # owl:allValuesFrom
        avf = entry.get(f"{OWL}allValuesFrom")
        if avf is not None:
            _parse_allvaluesfrom(avf, prop)

    # Set @extends
    if len(extends) == 1:
        shape["@extends"] = extends[0]
    elif len(extends) > 1:
        shape["@extends"] = extends

    # Attach property constraints
    for prop_iri, constraint in properties.items():
        shape[prop_iri] = constraint

    # Restore jex: namespace annotations (unmappable constraints)
    _restore_jex_annotations(owl_class, shape)

    return shape


# ── Default datatypes to strip on reverse mapping ──────────────────
_DEFAULT_NUMERIC_DT = f"{XSD}decimal"
_DEFAULT_STRING_DT = f"{XSD}string"

# Facet IRI → (@shape key, is_string_facet)
_FACET_MAP: dict[str, tuple[str, bool]] = {
    f"{XSD}minInclusive": ("@minimum", False),
    f"{XSD}maxInclusive": ("@maximum", False),
    f"{XSD}minLength": ("@minLength", True),
    f"{XSD}maxLength": ("@maxLength", True),
    f"{XSD}pattern": ("@pattern", True),
}

# jex: local name → @shape key
_JEX_ANNOTATION_MAP: dict[str, str] = {
    "lessThan": "@lessThan",
    "lessThanOrEquals": "@lessThanOrEquals",
    "equals": "@equals",
    "disjoint": "@disjoint",
    "severity": "@severity",
}


def _parse_allvaluesfrom(avf: Any, prop: dict[str, Any]) -> None:
    """Parse an owl:allValuesFrom value into jsonld-ex constraint keys."""
    # Simple IRI → @type
    if isinstance(avf, dict) and "@id" in avf and "@type" not in avf:
        prop["@type"] = _iri_to_xsd_prefixed(avf["@id"])
        return

    if not isinstance(avf, dict):
        return

    # owl:oneOf → @in
    one_of = avf.get(f"{OWL}oneOf")
    if one_of is not None:
        items = one_of.get("@list", one_of) if isinstance(one_of, dict) else one_of
        prop["@in"] = items
        return

    # owl:unionOf → @or
    union_of = avf.get(f"{OWL}unionOf")
    if union_of is not None:
        members = union_of.get("@list", union_of) if isinstance(union_of, dict) else union_of
        prop["@or"] = [_owl_datarange_to_constraint(m) for m in members]
        return

    # owl:intersectionOf → @and
    intersection_of = avf.get(f"{OWL}intersectionOf")
    if intersection_of is not None:
        members = intersection_of.get("@list", intersection_of) if isinstance(intersection_of, dict) else intersection_of
        prop["@and"] = [_owl_datarange_to_constraint(m) for m in members]
        return

    # owl:datatypeComplementOf → @not
    complement_of = avf.get(f"{OWL}datatypeComplementOf")
    if complement_of is not None:
        prop["@not"] = _owl_datarange_to_constraint(complement_of)
        return

    # DatatypeRestriction: owl:onDatatype + owl:withRestrictions
    on_datatype = avf.get(f"{OWL}onDatatype")
    with_restrictions = avf.get(f"{OWL}withRestrictions")
    if on_datatype is not None and with_restrictions is not None:
        dt_iri = on_datatype.get("@id") if isinstance(on_datatype, dict) else on_datatype
        facets_list = with_restrictions.get("@list", with_restrictions) if isinstance(with_restrictions, dict) else with_restrictions

        has_string_facet = False
        has_numeric_facet = False
        for facet_dict in facets_list:
            if not isinstance(facet_dict, dict):
                continue
            for facet_iri, (shape_key, is_string) in _FACET_MAP.items():
                if facet_iri in facet_dict:
                    prop[shape_key] = facet_dict[facet_iri]
                    if is_string:
                        has_string_facet = True
                    else:
                        has_numeric_facet = True

        # Determine if the datatype was a default that should be stripped
        is_default = (
            (has_string_facet and dt_iri == _DEFAULT_STRING_DT)
            or (has_numeric_facet and not has_string_facet and dt_iri == _DEFAULT_NUMERIC_DT)
        )
        if not is_default:
            prop["@type"] = _iri_to_xsd_prefixed(dt_iri)


def _owl_datarange_to_constraint(node: Any) -> dict[str, Any]:
    """Convert an OWL 2 DataRange node back to a jsonld-ex constraint dict.

    Inverse of :func:`_constraint_to_owl_datarange`.  Handles simple IRI refs,
    DatatypeRestrictions, and recursive unionOf/intersectionOf/complementOf.
    """
    if not isinstance(node, dict):
        return {}

    # Simple IRI ref → {"@type": "xsd:..."}
    if "@id" in node and "@type" not in node:
        return {"@type": _iri_to_xsd_prefixed(node["@id"])}

    # Recursive combinators
    union_of = node.get(f"{OWL}unionOf")
    if union_of is not None:
        members = union_of.get("@list", union_of) if isinstance(union_of, dict) else union_of
        return {"@or": [_owl_datarange_to_constraint(m) for m in members]}

    intersection_of = node.get(f"{OWL}intersectionOf")
    if intersection_of is not None:
        members = intersection_of.get("@list", intersection_of) if isinstance(intersection_of, dict) else intersection_of
        return {"@and": [_owl_datarange_to_constraint(m) for m in members]}

    complement_of = node.get(f"{OWL}datatypeComplementOf")
    if complement_of is not None:
        return {"@not": _owl_datarange_to_constraint(complement_of)}

    # DatatypeRestriction
    on_datatype = node.get(f"{OWL}onDatatype")
    with_restrictions = node.get(f"{OWL}withRestrictions")
    if on_datatype is not None and with_restrictions is not None:
        result: dict[str, Any] = {}
        dt_iri = on_datatype.get("@id") if isinstance(on_datatype, dict) else on_datatype
        facets_list = with_restrictions.get("@list", with_restrictions) if isinstance(with_restrictions, dict) else with_restrictions

        has_string_facet = False
        has_numeric_facet = False
        for facet_dict in facets_list:
            if not isinstance(facet_dict, dict):
                continue
            for facet_iri, (shape_key, is_string) in _FACET_MAP.items():
                if facet_iri in facet_dict:
                    result[shape_key] = facet_dict[facet_iri]
                    if is_string:
                        has_string_facet = True
                    else:
                        has_numeric_facet = True

        is_default = (
            (has_string_facet and dt_iri == _DEFAULT_STRING_DT)
            or (has_numeric_facet and not has_string_facet and dt_iri == _DEFAULT_NUMERIC_DT)
        )
        if not is_default:
            result["@type"] = _iri_to_xsd_prefixed(dt_iri)
        return result

    return {}


def _iri_to_xsd_prefixed(iri: str) -> str:
    """Convert a full XSD IRI to xsd: prefixed form, or return as-is."""
    if iri.startswith(XSD):
        return "xsd:" + iri[len(XSD):]
    return iri


def _restore_jex_annotations(
    owl_class: dict[str, Any],
    shape: dict[str, Any],
) -> None:
    """Restore jex: namespace annotations from OWL class to shape.

    Unmappable constraints are stored as class-level annotations in the
    jex: namespace by shape_to_owl_restrictions.  This function detects
    them and places them back on the shape.
    """
    for key, value in owl_class.items():
        if not key.startswith(JSONLD_EX):
            continue
        local_name = key[len(JSONLD_EX):]

        # Known scalar annotations → @-prefixed keys
        if local_name in _JEX_ANNOTATION_MAP:
            shape[_JEX_ANNOTATION_MAP[local_name]] = value
        elif local_name == "conditional":
            # Restore @if/@then/@else from the stored conditional dict
            if isinstance(value, dict):
                for cond_key in ("@if", "@then", "@else"):
                    if cond_key in value:
                        shape[cond_key] = value[cond_key]
            else:
                shape[f"{JSONLD_EX}conditional"] = value
        else:
            # Unknown jex: annotation — preserve as-is
            shape[key] = value


# ═══════════════════════════════════════════════════════════════════
# RDF-STAR MAPPING
# ═══════════════════════════════════════════════════════════════════

def to_rdf_star_ntriples(
    doc: dict[str, Any],
    base_subject: Optional[str] = None,
) -> tuple[str, ConversionReport]:
    """Convert jsonld-ex annotated values to RDF-star N-Triples.

    Each annotated value generates:
        <<subject predicate "value">> jsonld-ex:confidence 0.95 .
        <<subject predicate "value">> jsonld-ex:source <uri> .
        etc.

    Args:
        doc: A JSON-LD document with jsonld-ex annotations.
        base_subject: IRI for the document subject.

    Returns:
        Tuple of (N-Triples string, ConversionReport).
    """
    report = ConversionReport(success=True)
    lines: list[str] = []

    subject = base_subject or doc.get("@id", "_:subject")
    if not subject.startswith("_:") and not subject.startswith("<"):
        subject = f"<{subject}>"

    for key, value in doc.items():
        if key.startswith("@"):
            continue

        if isinstance(value, dict) and "@value" in value:
            prov = get_provenance(value)
            has_annotations = any([
                prov.confidence is not None,
                prov.source is not None,
                prov.extracted_at is not None,
                prov.method is not None,
                prov.human_verified is not None,
                prov.derived_from is not None,
                prov.delegated_by is not None,
                prov.invalidated_at is not None,
                prov.invalidation_reason is not None,
            ])

            if not has_annotations:
                # Plain value — emit standard triple
                literal = _format_literal(value["@value"])
                lines.append(f"{subject} <{key}> {literal} .")
                report.triples_output += 1
                continue

            # Build the embedded triple
            literal = _format_literal(value["@value"])
            embedded = f"<< {subject} <{key}> {literal} >>"
            report.triples_input += 1

            # Base triple
            lines.append(f"{subject} <{key}> {literal} .")
            report.triples_output += 1

            # Annotation triples
            if prov.confidence is not None:
                lines.append(f'{embedded} <{JSONLD_EX}confidence> "{prov.confidence}"^^<{XSD}double> .')
                report.triples_output += 1

            if prov.source is not None:
                lines.append(f"{embedded} <{JSONLD_EX}source> <{prov.source}> .")
                report.triples_output += 1

            if prov.extracted_at is not None:
                lines.append(f'{embedded} <{JSONLD_EX}extractedAt> "{prov.extracted_at}"^^<{XSD}dateTime> .')
                report.triples_output += 1

            if prov.method is not None:
                lines.append(f'{embedded} <{JSONLD_EX}method> "{prov.method}" .')
                report.triples_output += 1

            if prov.human_verified is not None:
                val = "true" if prov.human_verified else "false"
                lines.append(f'{embedded} <{JSONLD_EX}humanVerified> "{val}"^^<{XSD}boolean> .')
                report.triples_output += 1

            if prov.derived_from is not None:
                sources = prov.derived_from
                if isinstance(sources, str):
                    sources = [sources]
                for src in sources:
                    lines.append(f'{embedded} <{JSONLD_EX}derivedFrom> <{src}> .')
                    report.triples_output += 1

            if prov.delegated_by is not None:
                delegates = prov.delegated_by
                if isinstance(delegates, str):
                    delegates = [delegates]
                for dlg in delegates:
                    lines.append(f'{embedded} <{JSONLD_EX}delegatedBy> <{dlg}> .')
                    report.triples_output += 1

            if prov.invalidated_at is not None:
                lines.append(f'{embedded} <{JSONLD_EX}invalidatedAt> "{prov.invalidated_at}"^^<{XSD}dateTime> .')
                report.triples_output += 1

            if prov.invalidation_reason is not None:
                lines.append(f'{embedded} <{JSONLD_EX}invalidationReason> "{_escape_ntriples(prov.invalidation_reason)}" .')
                report.triples_output += 1

            report.nodes_converted += 1
        else:
            # Non-annotated value — pass through
            if isinstance(value, str):
                if value.startswith("http://") or value.startswith("https://"):
                    lines.append(f"{subject} <{key}> <{value}> .")
                else:
                    lines.append(f'{subject} <{key}> "{_escape_ntriples(value)}" .')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                xsd_t = f"{XSD}integer" if isinstance(value, int) else f"{XSD}double"
                lines.append(f'{subject} <{key}> "{value}"^^<{xsd_t}> .')
            elif isinstance(value, bool):
                val = "true" if value else "false"
                lines.append(f'{subject} <{key}> "{val}"^^<{XSD}boolean> .')
            report.triples_output += 1

    return "\n".join(lines), report


# ═══════════════════════════════════════════════════════════════════
# SSN/SOSA MAPPING
# ═══════════════════════════════════════════════════════════════════

def to_ssn(doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex annotated document to SSN/SOSA observation graph.

    Maps jsonld-ex IoT annotations to W3C SOSA core classes:
        @value                → sosa:hasSimpleResult (or sosa:hasResult + qudt if @unit)
        @source               → sosa:madeBySensor → sosa:Sensor
        @extractedAt          → sosa:resultTime
        @method               → sosa:usedProcedure → sosa:Procedure
        @confidence           → jsonld-ex:confidence (no SOSA equivalent)
        @unit                 → qudt:unit on sosa:Result
        @measurementUncertainty → ssn-system:Accuracy on Sensor capability
        @calibratedAt/Method/Authority → properties on ssn-system:SystemCapability
        @aggregationMethod/Window/Count → sosa:Procedure parameters
        parent @id/@type      → sosa:hasFeatureOfInterest → sosa:FeatureOfInterest
        property key          → sosa:observedProperty → sosa:ObservableProperty

    Args:
        doc: A JSON-LD document with jsonld-ex annotations.

    Returns:
        Tuple of (SSN/SOSA JSON-LD document, ConversionReport).
    """
    report = ConversionReport(success=True)
    graph_nodes: list[dict[str, Any]] = []
    input_triples = 0
    output_triples = 0

    # Build output context
    ssn_context: dict[str, Any] = {
        "sosa": SOSA,
        "ssn": SSN,
        "ssn-system": SSN_SYSTEM,
        "qudt": QUDT,
        "xsd": XSD,
        "rdfs": RDFS,
        "jsonld-ex": JSONLD_EX,
    }

    # Preserve original context entries
    original_ctx = doc.get("@context", {})
    if isinstance(original_ctx, str):
        ssn_context["_original"] = original_ctx
    elif isinstance(original_ctx, dict):
        for k, v in original_ctx.items():
            if k not in ssn_context:
                ssn_context[k] = v
    elif isinstance(original_ctx, list):
        for item in original_ctx:
            if isinstance(item, dict):
                for k, v in item.items():
                    if k not in ssn_context:
                        ssn_context[k] = v

    parent_id = doc.get("@id", f"_:node-{uuid.uuid4().hex[:8]}")
    parent_type = doc.get("@type")

    # Create FeatureOfInterest from parent node
    foi_id = parent_id
    foi_node: dict[str, Any] = {
        "@id": foi_id,
        "@type": [f"{SOSA}FeatureOfInterest"],
    }
    if parent_type:
        types = parent_type if isinstance(parent_type, list) else [parent_type]
        foi_node["@type"] = [f"{SOSA}FeatureOfInterest"] + [
            t for t in types if t != f"{SOSA}FeatureOfInterest"
        ]
    output_triples += 1  # type triple

    # Track whether we actually emitted any observations
    has_observations = False

    # Build main node for non-annotated properties
    main_node: dict[str, Any] = {}
    if "@id" in doc:
        main_node["@id"] = doc["@id"]
    if "@type" in doc:
        main_node["@type"] = doc["@type"]

    # Track sensors we've already created (dedup by @source IRI)
    sensors_created: dict[str, dict[str, Any]] = {}

    for key, value in doc.items():
        if key.startswith("@"):
            continue

        if not (isinstance(value, dict) and "@value" in value):
            # Non-annotated property — preserve on main node
            main_node[key] = value
            continue

        prov = get_provenance(value)
        has_annotations = any([
            prov.confidence is not None,
            prov.source is not None,
            prov.extracted_at is not None,
            prov.method is not None,
            prov.human_verified is not None,
            prov.measurement_uncertainty is not None,
            prov.unit is not None,
            prov.calibrated_at is not None,
            prov.calibration_method is not None,
            prov.calibration_authority is not None,
            prov.aggregation_method is not None,
            prov.aggregation_window is not None,
            prov.aggregation_count is not None,
        ])

        if not has_annotations:
            main_node[key] = value
            continue

        input_triples += 1
        has_observations = True

        # ── Create sosa:Observation ───────────────────────────────
        obs_id = f"_:obs-{uuid.uuid4().hex[:8]}"
        obs: dict[str, Any] = {
            "@id": obs_id,
            "@type": f"{SOSA}Observation",
        }
        output_triples += 1  # type

        # FeatureOfInterest
        obs[f"{SOSA}hasFeatureOfInterest"] = {"@id": foi_id}
        output_triples += 1

        # ObservableProperty from key
        prop_id = key
        obs[f"{SOSA}observedProperty"] = {"@id": prop_id}
        output_triples += 1

        obs_prop_node: dict[str, Any] = {
            "@id": prop_id,
            "@type": f"{SOSA}ObservableProperty",
        }
        graph_nodes.append(obs_prop_node)
        output_triples += 1  # type

        # ── Result ───────────────────────────────────────────
        if prov.unit is not None:
            # Structured result with QUDT unit
            result_id = f"_:result-{uuid.uuid4().hex[:8]}"
            result_node: dict[str, Any] = {
                "@id": result_id,
                "@type": f"{SOSA}Result",
                f"{QUDT}numericValue": value["@value"],
                f"{QUDT}unit": prov.unit,
            }
            obs[f"{SOSA}hasResult"] = {"@id": result_id}
            graph_nodes.append(result_node)
            output_triples += 4  # type + numericValue + unit + hasResult
        else:
            obs[f"{SOSA}hasSimpleResult"] = value["@value"]
            output_triples += 1

        # ── resultTime ───────────────────────────────────────
        if prov.extracted_at is not None:
            obs[f"{SOSA}resultTime"] = {
                "@value": prov.extracted_at,
                "@type": f"{XSD}dateTime",
            }
            output_triples += 1

        # ── Confidence (jsonld-ex namespace, no SOSA equivalent) ───
        if prov.confidence is not None:
            obs[f"{JSONLD_EX}confidence"] = prov.confidence
            output_triples += 1

        # ── Sensor from @source ──────────────────────────────
        sensor_node: dict[str, Any] | None = None
        if prov.source is not None:
            obs[f"{SOSA}madeBySensor"] = {"@id": prov.source}
            output_triples += 1

            if prov.source not in sensors_created:
                sensor_node = {
                    "@id": prov.source,
                    "@type": f"{SOSA}Sensor",
                }
                sensors_created[prov.source] = sensor_node
                output_triples += 1  # type
            else:
                sensor_node = sensors_created[prov.source]

        # ── Procedure from @method or @aggregationMethod ────────
        proc_label = prov.aggregation_method or prov.method
        if proc_label is not None:
            proc_id = f"_:proc-{uuid.uuid4().hex[:8]}"
            proc_node: dict[str, Any] = {
                "@id": proc_id,
                "@type": f"{SOSA}Procedure",
                f"{RDFS}label": proc_label,
            }
            output_triples += 2  # type + label

            if prov.aggregation_window is not None:
                proc_node[f"{JSONLD_EX}aggregationWindow"] = prov.aggregation_window
                output_triples += 1
            if prov.aggregation_count is not None:
                proc_node[f"{JSONLD_EX}aggregationCount"] = prov.aggregation_count
                output_triples += 1

            obs[f"{SOSA}usedProcedure"] = {"@id": proc_id}
            graph_nodes.append(proc_node)
            output_triples += 1  # usedProcedure link

        # ── SystemCapability on Sensor ────────────────────────
        has_sys_cap = any([
            prov.measurement_uncertainty is not None,
            prov.calibrated_at is not None,
            prov.calibration_method is not None,
            prov.calibration_authority is not None,
        ])
        if has_sys_cap and sensor_node is not None:
            cap_id = f"_:cap-{uuid.uuid4().hex[:8]}"
            cap_node: dict[str, Any] = {
                "@id": cap_id,
                "@type": f"{SSN_SYSTEM}SystemCapability",
            }
            output_triples += 1  # type

            # Accuracy
            if prov.measurement_uncertainty is not None:
                acc_id = f"_:acc-{uuid.uuid4().hex[:8]}"
                acc_node: dict[str, Any] = {
                    "@id": acc_id,
                    "@type": f"{SSN_SYSTEM}Accuracy",
                    f"{JSONLD_EX}value": prov.measurement_uncertainty,
                }
                cap_node[f"{SSN_SYSTEM}hasSystemProperty"] = {"@id": acc_id}
                graph_nodes.append(acc_node)
                output_triples += 3  # type + value + hasSystemProperty link

            # Calibration properties
            if prov.calibrated_at is not None:
                cap_node[f"{JSONLD_EX}calibratedAt"] = prov.calibrated_at
                output_triples += 1
            if prov.calibration_method is not None:
                cap_node[f"{JSONLD_EX}calibrationMethod"] = prov.calibration_method
                output_triples += 1
            if prov.calibration_authority is not None:
                cap_node[f"{JSONLD_EX}calibrationAuthority"] = prov.calibration_authority
                output_triples += 1

            sensor_node[f"{SSN_SYSTEM}hasSystemCapability"] = {"@id": cap_id}
            graph_nodes.append(cap_node)
            output_triples += 1  # hasSystemCapability link

        graph_nodes.append(obs)
        report.nodes_converted += 1

    # Add sensor nodes (deduped)
    for s in sensors_created.values():
        graph_nodes.append(s)

    # Add FOI if we emitted observations
    if has_observations:
        graph_nodes.append(foi_node)

    # Add main node (non-annotated properties)
    if main_node:
        graph_nodes.insert(0, main_node)

    report.triples_input = input_triples
    report.triples_output = output_triples

    return {"@context": ssn_context, "@graph": graph_nodes}, report


def from_ssn(ssn_doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert SSN/SOSA observation graph back to jsonld-ex annotations.

    Parses SOSA Observation nodes and extracts annotation fields that
    map to jsonld-ex keys.  Information not representable in jsonld-ex
    (e.g. sosa:Platform, ssn:Deployment) is logged as warnings.

    Args:
        ssn_doc: A JSON-LD document using SSN/SOSA vocabulary.

    Returns:
        Tuple of (jsonld-ex annotated document, ConversionReport).
    """
    report = ConversionReport(success=True)

    graph = ssn_doc.get("@graph", [])
    if not isinstance(graph, list):
        graph = [graph]

    # Index all nodes by @id
    nodes_by_id: dict[str, dict] = {}
    for node in graph:
        if isinstance(node, dict) and "@id" in node:
            nodes_by_id[node["@id"]] = node

    # Find Observation nodes
    obs_type = f"{SOSA}Observation"
    observations: list[dict] = []
    for node in graph:
        if not isinstance(node, dict):
            continue
        nt = node.get("@type", [])
        if isinstance(nt, str):
            nt = [nt]
        if obs_type in nt:
            observations.append(node)

    if not observations:
        report.success = False
        report.errors.append("No sosa:Observation found in graph")
        return ssn_doc, report

    # Determine parent node from FeatureOfInterest of first observation
    result_doc: dict[str, Any] = {"@context": ssn_doc.get("@context", {})}

    first_obs = observations[0]
    foi_ref = (
        first_obs.get(f"{SOSA}hasFeatureOfInterest")
        or first_obs.get("sosa:hasFeatureOfInterest")
    )
    if foi_ref is not None:
        foi_id = foi_ref.get("@id") if isinstance(foi_ref, dict) else foi_ref
        result_doc["@id"] = foi_id
        # Try to recover original @type from FOI node
        foi_node = nodes_by_id.get(foi_id)
        if foi_node is not None:
            foi_types = foi_node.get("@type", [])
            if isinstance(foi_types, str):
                foi_types = [foi_types]
            non_sosa_types = [
                t for t in foi_types if t != f"{SOSA}FeatureOfInterest"
            ]
            if non_sosa_types:
                result_doc["@type"] = (
                    non_sosa_types[0] if len(non_sosa_types) == 1
                    else non_sosa_types
                )

    # Convert each observation to an annotated property
    for obs in observations:
        annotated = _observation_to_annotation(obs, nodes_by_id, report)
        if annotated is not None:
            prop_key, ann_value = annotated
            result_doc[prop_key] = ann_value
            report.nodes_converted += 1

    # Warn on unsupported SOSA node types present in the graph
    unsupported_sosa_types = [
        (f"{SOSA}Platform", "Platform"),
        (f"{SOSA}Actuator", "Actuator"),
        (f"{SOSA}Actuation", "Actuation"),
        (f"{SOSA}Sample", "Sample"),
        (f"{SOSA}Sampling", "Sampling"),
        (f"{SOSA}Sampler", "Sampler"),
    ]
    for node in graph:
        if not isinstance(node, dict):
            continue
        nt = node.get("@type", [])
        if isinstance(nt, str):
            nt = [nt]
        for type_iri, label in unsupported_sosa_types:
            if type_iri in nt:
                report.warnings.append(
                    f"SOSA {label} node '{node.get('@id', 'anonymous')}' "
                    f"has no jsonld-ex equivalent (dropped)"
                )

    return result_doc, report


def _observation_to_annotation(
    obs: dict[str, Any],
    all_nodes: dict[str, dict[str, Any]],
    report: ConversionReport,
) -> tuple[str, dict[str, Any]] | None:
    """Convert a single SOSA Observation to a (property_key, annotated_value) pair."""

    # Determine the property key from sosa:observedProperty
    obs_prop = (
        obs.get(f"{SOSA}observedProperty")
        or obs.get("sosa:observedProperty")
    )
    if obs_prop is None:
        report.warnings.append(
            f"Observation '{obs.get('@id', '?')}' has no sosa:observedProperty — skipped"
        )
        return None
    prop_key = obs_prop.get("@id") if isinstance(obs_prop, dict) else obs_prop

    # Extract value from hasSimpleResult or hasResult
    result_value: Any = None
    unit: str | None = None

    simple = (
        obs.get(f"{SOSA}hasSimpleResult")
        or obs.get("sosa:hasSimpleResult")
    )
    structured = (
        obs.get(f"{SOSA}hasResult")
        or obs.get("sosa:hasResult")
    )

    if structured is not None:
        result_id = structured.get("@id") if isinstance(structured, dict) else structured
        result_node = all_nodes.get(result_id) if result_id else None
        if result_node is not None:
            result_value = (
                result_node.get(f"{QUDT}numericValue")
                or result_node.get("qudt:numericValue")
            )
            unit = (
                result_node.get(f"{QUDT}unit")
                or result_node.get("qudt:unit")
            )
    elif simple is not None:
        result_value = simple

    if result_value is None:
        report.warnings.append(
            f"Observation '{obs.get('@id', '?')}' has no result value — skipped"
        )
        return None

    # Build annotated value
    annotated: dict[str, Any] = {"@value": result_value}

    if unit is not None:
        annotated["@unit"] = unit

    # Confidence
    conf = (
        obs.get(f"{JSONLD_EX}confidence")
        or obs.get("jsonld-ex:confidence")
    )
    if conf is not None:
        annotated["@confidence"] = conf if not isinstance(conf, dict) else conf.get("@value", conf)

    # Source from madeBySensor
    sensor_ref = (
        obs.get(f"{SOSA}madeBySensor")
        or obs.get("sosa:madeBySensor")
    )
    if sensor_ref is not None:
        sensor_id = sensor_ref.get("@id") if isinstance(sensor_ref, dict) else sensor_ref
        annotated["@source"] = sensor_id

        # Check sensor for SystemCapability → Accuracy
        sensor_node = all_nodes.get(sensor_id)
        if sensor_node is not None:
            cap_ref = (
                sensor_node.get(f"{SSN_SYSTEM}hasSystemCapability")
                or sensor_node.get("ssn-system:hasSystemCapability")
            )
            if cap_ref is not None:
                cap_id = cap_ref.get("@id") if isinstance(cap_ref, dict) else cap_ref
                cap_node = all_nodes.get(cap_id)
                if cap_node is not None:
                    # Extract Accuracy
                    sys_prop_ref = (
                        cap_node.get(f"{SSN_SYSTEM}hasSystemProperty")
                        or cap_node.get("ssn-system:hasSystemProperty")
                    )
                    if sys_prop_ref is not None:
                        sp_id = sys_prop_ref.get("@id") if isinstance(sys_prop_ref, dict) else sys_prop_ref
                        sp_node = all_nodes.get(sp_id)
                        if sp_node is not None:
                            sp_types = sp_node.get("@type", [])
                            if isinstance(sp_types, str):
                                sp_types = [sp_types]
                            if f"{SSN_SYSTEM}Accuracy" in sp_types:
                                acc_val = (
                                    sp_node.get(f"{JSONLD_EX}value")
                                    or sp_node.get("jsonld-ex:value")
                                )
                                if acc_val is not None:
                                    annotated["@measurementUncertainty"] = acc_val

    # ExtractedAt from resultTime
    result_time = (
        obs.get(f"{SOSA}resultTime")
        or obs.get("sosa:resultTime")
    )
    if result_time is not None:
        if isinstance(result_time, dict) and "@value" in result_time:
            annotated["@extractedAt"] = result_time["@value"]
        else:
            annotated["@extractedAt"] = result_time

    # Method from usedProcedure
    proc_ref = (
        obs.get(f"{SOSA}usedProcedure")
        or obs.get("sosa:usedProcedure")
    )
    if proc_ref is not None:
        proc_id = proc_ref.get("@id") if isinstance(proc_ref, dict) else proc_ref
        proc_node = all_nodes.get(proc_id)
        if proc_node is not None:
            label = (
                proc_node.get(f"{RDFS}label")
                or proc_node.get("rdfs:label")
            )
            if label is not None:
                method_str = label if not isinstance(label, dict) else label.get("@value", label)
                annotated["@method"] = method_str

    return prop_key, annotated


# ═══════════════════════════════════════════════════════════════════
# METRICS / COMPARISON UTILITIES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VerbosityComparison:
    """Comparison metrics between jsonld-ex and an alternative representation."""
    jsonld_ex_triples: int
    alternative_triples: int
    jsonld_ex_bytes: int
    alternative_bytes: int
    triple_reduction_pct: float
    byte_reduction_pct: float
    alternative_name: str


def compare_with_prov_o(doc: dict[str, Any]) -> VerbosityComparison:
    """Measure verbosity reduction of jsonld-ex vs equivalent PROV-O.

    Takes a jsonld-ex annotated document, converts to PROV-O, and
    compares triple counts and serialization sizes.
    """
    import json

    prov_doc, report = to_prov_o(doc)

    # Count jsonld-ex triples (approximate: 1 per annotated value + 1 per annotation field)
    ex_triples = _count_jsonld_ex_triples(doc)

    jsonld_ex_bytes = len(json.dumps(doc, indent=2).encode("utf-8"))
    prov_bytes = len(json.dumps(prov_doc, indent=2).encode("utf-8"))

    triple_reduction = (
        (report.triples_output - ex_triples) / report.triples_output * 100
        if report.triples_output > 0 else 0.0
    )
    byte_reduction = (
        (prov_bytes - jsonld_ex_bytes) / prov_bytes * 100
        if prov_bytes > 0 else 0.0
    )

    return VerbosityComparison(
        jsonld_ex_triples=ex_triples,
        alternative_triples=report.triples_output,
        jsonld_ex_bytes=jsonld_ex_bytes,
        alternative_bytes=prov_bytes,
        triple_reduction_pct=triple_reduction,
        byte_reduction_pct=byte_reduction,
        alternative_name="PROV-O",
    )


def compare_with_shacl(shape: dict[str, Any]) -> VerbosityComparison:
    """Measure verbosity reduction of jsonld-ex @shape vs equivalent SHACL."""
    import json

    shacl_doc = shape_to_shacl(shape)

    shape_bytes = len(json.dumps({"@shape": shape}, indent=2).encode("utf-8"))
    shacl_bytes = len(json.dumps(shacl_doc, indent=2).encode("utf-8"))

    # Count constraints in @shape
    shape_constraints = sum(
        len(c) for c in shape.values()
        if isinstance(c, dict) and not c == shape.get("@type")
    )
    # Count triples in SHACL
    shacl_triples = _count_shacl_triples(shacl_doc)

    triple_reduction = (
        (shacl_triples - shape_constraints) / shacl_triples * 100
        if shacl_triples > 0 else 0.0
    )
    byte_reduction = (
        (shacl_bytes - shape_bytes) / shacl_bytes * 100
        if shacl_bytes > 0 else 0.0
    )

    return VerbosityComparison(
        jsonld_ex_triples=shape_constraints,
        alternative_triples=shacl_triples,
        jsonld_ex_bytes=shape_bytes,
        alternative_bytes=shacl_bytes,
        triple_reduction_pct=triple_reduction,
        byte_reduction_pct=byte_reduction,
        alternative_name="SHACL",
    )


# ── Internal Helpers ───────────────────────────────────────────────


def _constraint_to_owl_datarange(constraint: dict[str, Any]) -> dict[str, Any]:
    """Convert a single jsonld-ex constraint branch to an OWL 2 DataRange.

    Used by @or/@and/@not to recursively convert each branch into the
    appropriate OWL 2 data range expression (§7 of the OWL 2 Structural
    Specification).

    Returns:
        - Simple IRI ref ``{"@id": xsd:...}`` when only @type is present.
        - DatatypeRestriction ``{"@type": rdfs:Datatype, owl:onDatatype: ...,
          owl:withRestrictions: ...}`` when @type + facets are present.
        - Facet-only DatatypeRestriction (defaults to xsd:string or xsd:decimal)
          when only facets are present.
        - Nested DataRange for recursive @or/@and/@not.
    """
    xsd_type = constraint.get("@type")

    # Collect XSD constraining facets
    facets: list[dict[str, Any]] = []
    if "@minimum" in constraint:
        facets.append({f"{XSD}minInclusive": constraint["@minimum"]})
    if "@maximum" in constraint:
        facets.append({f"{XSD}maxInclusive": constraint["@maximum"]})
    if "@minLength" in constraint:
        facets.append({f"{XSD}minLength": constraint["@minLength"]})
    if "@maxLength" in constraint:
        facets.append({f"{XSD}maxLength": constraint["@maxLength"]})
    if "@pattern" in constraint:
        facets.append({f"{XSD}pattern": constraint["@pattern"]})

    # Recursive combinators
    if "@or" in constraint:
        members = [_constraint_to_owl_datarange(b) for b in constraint["@or"]]
        return {
            "@type": f"{RDFS}Datatype",
            f"{OWL}unionOf": {"@list": members},
        }
    if "@and" in constraint:
        members = [_constraint_to_owl_datarange(b) for b in constraint["@and"]]
        return {
            "@type": f"{RDFS}Datatype",
            f"{OWL}intersectionOf": {"@list": members},
        }
    if "@not" in constraint:
        complement = _constraint_to_owl_datarange(constraint["@not"])
        return {
            "@type": f"{RDFS}Datatype",
            f"{OWL}datatypeComplementOf": complement,
        }

    if xsd_type:
        resolved = xsd_type.replace("xsd:", XSD) if xsd_type.startswith("xsd:") else xsd_type
        if facets:
            return {
                "@type": f"{RDFS}Datatype",
                f"{OWL}onDatatype": {"@id": resolved},
                f"{OWL}withRestrictions": {"@list": facets},
            }
        return {"@id": resolved}

    if facets:
        has_string_facets = any(
            f"{XSD}{k}" in f
            for f in facets
            for k in ("minLength", "maxLength", "pattern")
        )
        default_dt = f"{XSD}string" if has_string_facets else f"{XSD}decimal"
        return {
            "@type": f"{RDFS}Datatype",
            f"{OWL}onDatatype": {"@id": default_dt},
            f"{OWL}withRestrictions": {"@list": facets},
        }

    # Fallback: return empty datatype node
    return {"@type": f"{RDFS}Datatype"}


def _constraint_to_shacl(constraint: dict[str, Any]) -> dict[str, Any]:
    """Convert a single jsonld-ex constraint dict to a SHACL property shape fragment.

    Used by logical combinators (@or/@and/@not) to convert each branch.
    """
    result: dict[str, Any] = {}

    xsd_type = constraint.get("@type")
    if xsd_type:
        resolved = xsd_type.replace("xsd:", XSD) if xsd_type.startswith("xsd:") else xsd_type
        result[f"{SHACL}datatype"] = {"@id": resolved}

    if "@minimum" in constraint:
        result[f"{SHACL}minInclusive"] = constraint["@minimum"]
    if "@maximum" in constraint:
        result[f"{SHACL}maxInclusive"] = constraint["@maximum"]
    if "@minLength" in constraint:
        result[f"{SHACL}minLength"] = constraint["@minLength"]
    if "@maxLength" in constraint:
        result[f"{SHACL}maxLength"] = constraint["@maxLength"]
    if "@pattern" in constraint:
        result[f"{SHACL}pattern"] = constraint["@pattern"]
    if "@in" in constraint:
        result[f"{SHACL}in"] = {"@list": constraint["@in"]}
    if "@minCount" in constraint:
        result[f"{SHACL}minCount"] = constraint["@minCount"]
    if "@maxCount" in constraint:
        result[f"{SHACL}maxCount"] = constraint["@maxCount"]

    # Cross-property
    for key, shacl_key in [
        ("@lessThan", "lessThan"), ("@lessThanOrEquals", "lessThanOrEquals"),
        ("@equals", "equals"), ("@disjoint", "disjoint"),
    ]:
        if key in constraint:
            result[f"{SHACL}{shacl_key}"] = {"@id": constraint[key]}

    # Recursive logical combinators
    if "@or" in constraint:
        result[f"{SHACL}or"] = {
            "@list": [_constraint_to_shacl(b) for b in constraint["@or"]],
        }
    if "@and" in constraint:
        result[f"{SHACL}and"] = {
            "@list": [_constraint_to_shacl(b) for b in constraint["@and"]],
        }
    if "@not" in constraint:
        result[f"{SHACL}not"] = _constraint_to_shacl(constraint["@not"])

    return result


def _shacl_to_constraint(shacl_node: dict[str, Any]) -> dict[str, Any]:
    """Convert a SHACL property shape fragment back to a jsonld-ex constraint dict.

    Inverse of :func:`_constraint_to_shacl`.  Used by logical combinator
    round-trips (@or/@and/@not).
    """
    result: dict[str, Any] = {}

    datatype = shacl_node.get(f"{SHACL}datatype") or shacl_node.get("sh:datatype")
    if datatype is not None:
        dt_iri = datatype.get("@id") if isinstance(datatype, dict) else datatype
        if dt_iri.startswith(XSD):
            result["@type"] = "xsd:" + dt_iri[len(XSD):]
        else:
            result["@type"] = dt_iri

    for shacl_key, shape_key in [
        ("minInclusive", "@minimum"), ("maxInclusive", "@maximum"),
        ("minLength", "@minLength"), ("maxLength", "@maxLength"),
        ("pattern", "@pattern"),
        ("minCount", "@minCount"), ("maxCount", "@maxCount"),
    ]:
        val = shacl_node.get(f"{SHACL}{shacl_key}")
        if val is None:
            val = shacl_node.get(f"sh:{shacl_key}")
        if val is not None:
            result[shape_key] = val

    # sh:in
    sh_in = shacl_node.get(f"{SHACL}in") or shacl_node.get("sh:in")
    if sh_in is not None:
        if isinstance(sh_in, dict) and "@list" in sh_in:
            result["@in"] = sh_in["@list"]
        elif isinstance(sh_in, list):
            result["@in"] = sh_in

    # Cross-property
    for shacl_key, shape_key in [
        ("lessThan", "@lessThan"), ("lessThanOrEquals", "@lessThanOrEquals"),
        ("equals", "@equals"), ("disjoint", "@disjoint"),
    ]:
        val = shacl_node.get(f"{SHACL}{shacl_key}") or shacl_node.get(f"sh:{shacl_key}")
        if val is not None:
            result[shape_key] = val.get("@id") if isinstance(val, dict) else val

    # Recursive logical combinators
    sh_or = shacl_node.get(f"{SHACL}or") or shacl_node.get("sh:or")
    if sh_or is not None:
        branches = sh_or.get("@list", sh_or) if isinstance(sh_or, dict) else sh_or
        result["@or"] = [_shacl_to_constraint(b) for b in branches]

    sh_and = shacl_node.get(f"{SHACL}and") or shacl_node.get("sh:and")
    if sh_and is not None:
        branches = sh_and.get("@list", sh_and) if isinstance(sh_and, dict) else sh_and
        result["@and"] = [_shacl_to_constraint(b) for b in branches]

    sh_not = shacl_node.get(f"{SHACL}not") or shacl_node.get("sh:not")
    if sh_not is not None:
        result["@not"] = _shacl_to_constraint(sh_not)

    return result


def _format_literal(value: Any) -> str:
    """Format a Python value as an N-Triples literal."""
    if isinstance(value, bool):
        return f'"{str(value).lower()}"^^<{XSD}boolean>'
    if isinstance(value, int):
        return f'"{value}"^^<{XSD}integer>'
    if isinstance(value, float):
        return f'"{value}"^^<{XSD}double>'
    return f'"{_escape_ntriples(str(value))}"'


def _escape_ntriples(s: str) -> str:
    """Escape a string for N-Triples."""
    return (
        s.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def _count_jsonld_ex_triples(doc: dict[str, Any]) -> int:
    """Approximate triple count for a jsonld-ex document."""
    count = 0
    for key, value in doc.items():
        if key.startswith("@") and key != "@type":
            continue
        if key == "@type":
            count += 1
            continue
        if isinstance(value, dict) and "@value" in value:
            count += 1  # The base triple
            # Don't count annotation keywords as separate triples —
            # that's the whole point of inline annotations
        elif isinstance(value, list):
            count += len(value)
        else:
            count += 1
    return count


def _count_shacl_triples(shacl_doc: dict[str, Any]) -> int:
    """Approximate triple count for a SHACL shape graph."""
    count = 0
    graph = shacl_doc.get("@graph", [])
    if not isinstance(graph, list):
        graph = [graph]
    for node in graph:
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if key == "@id":
                continue
            if key == "@type":
                count += 1
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        # Each property constraint is multiple triples
                        count += len([k for k in item if k != "@id"])
                    else:
                        count += 1
            else:
                count += 1
    return count
