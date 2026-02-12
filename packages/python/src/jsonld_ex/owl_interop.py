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
        @required           → owl:minCardinality 1
        @type (xsd)         → owl:allValuesFrom + xsd:datatype
        @minimum/@maximum   → owl:onDataRange with xsd:minInclusive/maxInclusive

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

    for prop_name, constraint in shape.items():
        if prop_name.startswith("@") or not isinstance(constraint, dict):
            continue

        # @required → owl:minCardinality
        if constraint.get("@required"):
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}minCardinality": {
                    "@value": 1,
                    "@type": f"{XSD}nonNegativeInteger",
                },
            })

        # @type → owl:allValuesFrom
        xsd_type = constraint.get("@type")
        if xsd_type:
            resolved = xsd_type.replace("xsd:", XSD) if xsd_type.startswith("xsd:") else xsd_type
            restrictions.append({
                "@type": f"{OWL}Restriction",
                f"{OWL}onProperty": {"@id": prop_name},
                f"{OWL}allValuesFrom": {"@id": resolved},
            })

    # Build OWL class with restrictions as superclasses
    owl_class: dict[str, Any] = {
        "@id": target,
        "@type": f"{OWL}Class",
    }
    if restrictions:
        owl_class[f"{RDFS}subClassOf"] = restrictions if len(restrictions) > 1 else restrictions[0]

    return {
        "@context": owl_context,
        "@graph": [owl_class],
    }


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
