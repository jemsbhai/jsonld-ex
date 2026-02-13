"""
DPV v2.2 Bidirectional Interoperability for JSON-LD-Ex.

Maps jsonld-ex data protection annotations to W3C Data Privacy
Vocabulary (DPV) v2.2 concepts and back.

Follows the same pattern as owl_interop.py (to_prov_o / from_prov_o).

References:
  - DPV v2.2: https://w3id.org/dpv
  - DPV-GDPR: https://w3id.org/dpv/legal/eu/gdpr
  - DPV-PD:   https://w3id.org/dpv/pd
  - DPV-LOC:  https://w3id.org/dpv/loc
"""

from __future__ import annotations
import json
import uuid
from typing import Any, Optional

from jsonld_ex.owl_interop import ConversionReport, VerbosityComparison


# ── Namespace Constants ───────────────────────────────────────────

DPV = "https://w3id.org/dpv#"
EU_GDPR = "https://w3id.org/dpv/legal/eu/gdpr#"
DPV_LOC = "https://w3id.org/dpv/loc#"
DPV_PD = "https://w3id.org/dpv/pd#"


# ── Mapping Tables ────────────────────────────────────────────────

LEGAL_BASIS_TO_DPV: dict[str, str] = {
    "consent": f"{EU_GDPR}A6-1-a",
    "contract": f"{EU_GDPR}A6-1-b",
    "legal_obligation": f"{EU_GDPR}A6-1-c",
    "vital_interest": f"{EU_GDPR}A6-1-d",
    "public_task": f"{EU_GDPR}A6-1-e",
    "legitimate_interest": f"{EU_GDPR}A6-1-f",
}

DPV_TO_LEGAL_BASIS: dict[str, str] = {v: k for k, v in LEGAL_BASIS_TO_DPV.items()}

CATEGORY_TO_DPV: dict[str, str] = {
    "regular": f"{DPV_PD}PersonalData",
    "sensitive": f"{DPV_PD}SensitivePersonalData",
    "special_category": f"{DPV_PD}SpecialCategoryPersonalData",
    "pseudonymized": f"{DPV}PseudonymisedData",
    "anonymized": f"{DPV}AnonymisedData",
    "synthetic": f"{DPV}SyntheticData",
    "non_personal": f"{DPV}NonPersonalData",
}

DPV_TO_CATEGORY: dict[str, str] = {v: k for k, v in CATEGORY_TO_DPV.items()}

ACCESS_LEVEL_TO_DPV: dict[str, str] = {
    "public": f"{DPV}PublicAccess",
    "internal": f"{DPV}InternalAccess",
    "restricted": f"{DPV}RestrictedAccess",
    "confidential": f"{DPV}ConfidentialAccess",
    "secret": f"{DPV}SecretAccess",
}

DPV_TO_ACCESS_LEVEL: dict[str, str] = {v: k for k, v in ACCESS_LEVEL_TO_DPV.items()}


# ── DPV Context ───────────────────────────────────────────────────

_DPV_CONTEXT: dict[str, str] = {
    "dpv": DPV,
    "eu-gdpr": EU_GDPR,
    "dpv-loc": DPV_LOC,
    "dpv-pd": DPV_PD,
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


# ── Forward Conversion: jsonld-ex → DPV ──────────────────────────

def to_dpv(doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert jsonld-ex data protection annotations to a DPV v2.2 graph.

    Walks the document looking for annotated values with data protection
    fields (@personalDataCategory, @legalBasis, @dataController, etc.)
    and produces DPV-compliant nodes.

    Mapping:
        @personalDataCategory → dpv-pd:<Category> type
        @legalBasis           → dpv:hasLegalBasis → eu-gdpr:A6-1-*
        @dataController       → dpv:hasDataController → dpv:DataController
        @dataProcessor        → dpv:hasDataProcessor → dpv:DataProcessor
        @dataSubject          → dpv:hasDataSubject
        @jurisdiction         → dpv:hasJurisdiction
        @retentionUntil       → dpv:hasStorage → dpv:StorageDeletion
        @consent              → dpv:hasConsent → dpv:Consent
        @accessLevel          → dpv:hasOrganisationalMeasure
        @erasureRequested     → dpv:hasRight → eu-gdpr:A17

    Args:
        doc: A JSON-LD document with jsonld-ex data protection annotations.

    Returns:
        Tuple of (DPV JSON-LD document, ConversionReport).
    """
    report = ConversionReport(success=True)
    graph_nodes: list[dict[str, Any]] = []
    output_triples = 0

    # Build context
    ctx = dict(_DPV_CONTEXT)
    original_ctx = doc.get("@context", {})
    if isinstance(original_ctx, dict):
        for k, v in original_ctx.items():
            if k not in ctx:
                ctx[k] = v

    def _process_property(
        node_id: str,
        prop_name: str,
        prop_val: dict[str, Any],
    ) -> None:
        nonlocal output_triples
        report.nodes_converted += 1

        # PersonalDataHandling node — central DPV concept
        handling_id = f"_:handling-{uuid.uuid4().hex[:8]}"
        handling: dict[str, Any] = {
            "@id": handling_id,
            "@type": f"{DPV}PersonalDataHandling",
        }

        # Data category
        cat = prop_val.get("@personalDataCategory")
        if cat and cat in CATEGORY_TO_DPV:
            handling["dpv:hasPersonalData"] = {
                "@id": f"_:pd-{uuid.uuid4().hex[:8]}",
                "@type": CATEGORY_TO_DPV[cat],
            }
            output_triples += 2

        # Legal basis
        basis = prop_val.get("@legalBasis")
        if basis and basis in LEGAL_BASIS_TO_DPV:
            handling["dpv:hasLegalBasis"] = {"@id": LEGAL_BASIS_TO_DPV[basis]}
            output_triples += 1

        # Data controller
        controller = prop_val.get("@dataController")
        if controller:
            handling["dpv:hasDataController"] = {
                "@id": controller,
                "@type": f"{DPV}DataController",
            }
            output_triples += 2

        # Data processor
        processor = prop_val.get("@dataProcessor")
        if processor:
            handling["dpv:hasDataProcessor"] = {
                "@id": processor,
                "@type": f"{DPV}DataProcessor",
            }
            output_triples += 2

        # Data subject
        subject = prop_val.get("@dataSubject")
        if subject:
            handling["dpv:hasDataSubject"] = {"@id": subject}
            output_triples += 1

        # Jurisdiction
        jur = prop_val.get("@jurisdiction")
        if jur:
            handling["dpv:hasJurisdiction"] = {
                "@id": f"{DPV_LOC}{jur}",
            }
            output_triples += 1

        # Processing purpose
        purpose = prop_val.get("@processingPurpose")
        if purpose:
            purposes = purpose if isinstance(purpose, list) else [purpose]
            handling["dpv:hasPurpose"] = [
                {"@id": p} if p.startswith("http") else {"dpv:hasDescription": p}
                for p in purposes
            ]
            output_triples += len(purposes)

        # Retention
        retention = prop_val.get("@retentionUntil")
        if retention:
            handling["dpv:hasStorage"] = {
                "@type": f"{DPV}StorageDeletion",
                "dpv:hasDuration": retention,
            }
            output_triples += 2

        # Consent
        consent = prop_val.get("@consent")
        if consent and isinstance(consent, dict):
            consent_node: dict[str, Any] = {
                "@id": f"_:consent-{uuid.uuid4().hex[:8]}",
                "@type": f"{DPV}Consent",
            }
            given_at = consent.get("@consentGivenAt")
            if given_at:
                consent_node["dpv:hasProvisionTime"] = given_at
                output_triples += 1
            scope = consent.get("@consentScope")
            if scope:
                consent_node["dpv:hasScope"] = scope
                output_triples += 1
            withdrawn = consent.get("@consentWithdrawnAt")
            if withdrawn:
                consent_node["dpv:hasWithdrawalTime"] = withdrawn
                output_triples += 1
            handling["dpv:hasConsent"] = consent_node
            output_triples += 2  # type + link

        # Access level
        access = prop_val.get("@accessLevel")
        if access and access in ACCESS_LEVEL_TO_DPV:
            handling["dpv:hasOrganisationalMeasure"] = {
                "@id": ACCESS_LEVEL_TO_DPV[access],
            }
            output_triples += 1

        # Erasure right (Phase 2)
        if prop_val.get("@erasureRequested") is True:
            right_node: dict[str, Any] = {
                "@id": f"_:right-{uuid.uuid4().hex[:8]}",
                "@type": f"{EU_GDPR}A17",
            }
            requested_at = prop_val.get("@erasureRequestedAt")
            if requested_at:
                right_node["dpv:hasRequestTime"] = requested_at
                output_triples += 1
            handling["dpv:hasRight"] = right_node
            output_triples += 2

        # Restriction (Phase 2)
        if prop_val.get("@restrictProcessing") is True:
            handling["dpv:hasRight"] = handling.get("dpv:hasRight") or []
            restriction_node = {
                "@id": f"_:right-{uuid.uuid4().hex[:8]}",
                "@type": f"{EU_GDPR}A18",
            }
            reason = prop_val.get("@restrictionReason")
            if reason:
                restriction_node["dpv:hasDescription"] = reason
                output_triples += 1
            # Handle single vs list for dpv:hasRight
            existing_right = handling.get("dpv:hasRight")
            if existing_right and not isinstance(existing_right, list):
                handling["dpv:hasRight"] = [existing_right, restriction_node]
            elif isinstance(existing_right, list):
                existing_right.append(restriction_node)
            else:
                handling["dpv:hasRight"] = restriction_node
            output_triples += 2

        # Link handling to source node
        handling["dpv:hasSourceNode"] = {"@id": node_id}
        handling["dpv:hasSourceProperty"] = prop_name
        output_triples += 2

        graph_nodes.append(handling)

    # Walk the document
    _walk_doc(doc, _process_property)

    report.triples_output = output_triples

    dpv_doc: dict[str, Any] = {"@context": ctx}
    if graph_nodes:
        dpv_doc["@graph"] = graph_nodes

    return dpv_doc, report


# ── Reverse Conversion: DPV → jsonld-ex ──────────────────────────

def from_dpv(dpv_doc: dict[str, Any]) -> tuple[dict[str, Any], ConversionReport]:
    """Convert a DPV v2.2 graph back to jsonld-ex data protection annotations.

    Looks for dpv:PersonalDataHandling nodes and extracts the mapped
    fields back to jsonld-ex @-prefixed annotations.

    Args:
        dpv_doc: A DPV JSON-LD document.

    Returns:
        Tuple of (jsonld-ex document, ConversionReport).
    """
    report = ConversionReport(success=True)

    graph = dpv_doc.get("@graph", [])
    if not graph:
        return {}, report

    # Group handling nodes by source
    restored_nodes: dict[str, dict[str, Any]] = {}

    for node in graph:
        node_type = node.get("@type", "")
        if not _is_handling_type(node_type):
            continue

        report.nodes_converted += 1

        source_node_id = _extract_id(node.get("dpv:hasSourceNode"))
        source_prop = node.get("dpv:hasSourceProperty", "")

        if not source_node_id:
            continue

        if source_node_id not in restored_nodes:
            restored_nodes[source_node_id] = {"@id": source_node_id}

        prop_val: dict[str, Any] = {"@value": None}

        # Reverse map category
        pd_node = node.get("dpv:hasPersonalData")
        if pd_node:
            pd_type = pd_node.get("@type", "") if isinstance(pd_node, dict) else ""
            cat = DPV_TO_CATEGORY.get(pd_type)
            if cat:
                prop_val["@personalDataCategory"] = cat

        # Reverse map legal basis
        lb_node = node.get("dpv:hasLegalBasis")
        if lb_node:
            lb_id = _extract_id(lb_node)
            basis = DPV_TO_LEGAL_BASIS.get(lb_id, "")
            if basis:
                prop_val["@legalBasis"] = basis

        # Data controller
        ctrl_node = node.get("dpv:hasDataController")
        if ctrl_node:
            ctrl_id = _extract_id(ctrl_node)
            if ctrl_id:
                prop_val["@dataController"] = ctrl_id

        # Data processor
        proc_node = node.get("dpv:hasDataProcessor")
        if proc_node:
            proc_id = _extract_id(proc_node)
            if proc_id:
                prop_val["@dataProcessor"] = proc_id

        # Data subject
        subj_node = node.get("dpv:hasDataSubject")
        if subj_node:
            subj_id = _extract_id(subj_node)
            if subj_id:
                prop_val["@dataSubject"] = subj_id

        # Jurisdiction
        jur_node = node.get("dpv:hasJurisdiction")
        if jur_node:
            jur_id = _extract_id(jur_node)
            if jur_id and jur_id.startswith(DPV_LOC):
                prop_val["@jurisdiction"] = jur_id[len(DPV_LOC):]

        # Retention
        storage_node = node.get("dpv:hasStorage")
        if storage_node and isinstance(storage_node, dict):
            dur = storage_node.get("dpv:hasDuration")
            if dur:
                prop_val["@retentionUntil"] = dur

        # Consent
        consent_node = node.get("dpv:hasConsent")
        if consent_node and isinstance(consent_node, dict):
            consent_rec: dict[str, Any] = {}
            prov_time = consent_node.get("dpv:hasProvisionTime")
            if prov_time:
                consent_rec["@consentGivenAt"] = prov_time
            scope = consent_node.get("dpv:hasScope")
            if scope:
                consent_rec["@consentScope"] = scope
            withdrawal = consent_node.get("dpv:hasWithdrawalTime")
            if withdrawal:
                consent_rec["@consentWithdrawnAt"] = withdrawal
            if consent_rec:
                prop_val["@consent"] = consent_rec

        # Access level
        measure_node = node.get("dpv:hasOrganisationalMeasure")
        if measure_node:
            measure_id = _extract_id(measure_node)
            level = DPV_TO_ACCESS_LEVEL.get(measure_id, "")
            if level:
                prop_val["@accessLevel"] = level

        # Rights (erasure, restriction)
        rights = node.get("dpv:hasRight")
        if rights:
            if not isinstance(rights, list):
                rights = [rights]
            for right in rights:
                right_type = right.get("@type", "") if isinstance(right, dict) else ""
                if right_type.endswith("A17"):
                    prop_val["@erasureRequested"] = True
                    req_time = right.get("dpv:hasRequestTime") if isinstance(right, dict) else None
                    if req_time:
                        prop_val["@erasureRequestedAt"] = req_time
                elif right_type.endswith("A18"):
                    prop_val["@restrictProcessing"] = True
                    desc = right.get("dpv:hasDescription") if isinstance(right, dict) else None
                    if desc:
                        prop_val["@restrictionReason"] = desc

        if source_prop:
            restored_nodes[source_node_id][source_prop] = prop_val

    result_doc: dict[str, Any] = {}
    nodes = list(restored_nodes.values())
    if len(nodes) == 1:
        result_doc = nodes[0]
    elif len(nodes) > 1:
        result_doc = {"@graph": nodes}

    return result_doc, report


# ── Verbosity Comparison ──────────────────────────────────────────

def compare_with_dpv(doc: dict[str, Any]) -> VerbosityComparison:
    """Measure verbosity reduction of jsonld-ex vs equivalent DPV representation.

    Args:
        doc: A jsonld-ex annotated document with data protection fields.

    Returns:
        A :class:`VerbosityComparison` comparing jsonld-ex and DPV.
    """
    dpv_doc, report = to_dpv(doc)

    ex_triples = _count_protection_triples(doc)

    ex_bytes = len(json.dumps(doc, indent=2).encode("utf-8"))
    dpv_bytes = len(json.dumps(dpv_doc, indent=2).encode("utf-8"))

    triple_reduction = (
        (report.triples_output - ex_triples) / report.triples_output * 100
        if report.triples_output > 0 else 0.0
    )
    byte_reduction = (
        (dpv_bytes - ex_bytes) / dpv_bytes * 100
        if dpv_bytes > 0 else 0.0
    )

    return VerbosityComparison(
        jsonld_ex_triples=ex_triples,
        alternative_triples=report.triples_output,
        jsonld_ex_bytes=ex_bytes,
        alternative_bytes=dpv_bytes,
        triple_reduction_pct=round(triple_reduction, 2),
        byte_reduction_pct=round(byte_reduction, 2),
        alternative_name="DPV",
    )


# ── Internal Helpers ──────────────────────────────────────────────

_PROTECTION_KEYS = frozenset({
    "@personalDataCategory", "@legalBasis", "@processingPurpose",
    "@dataController", "@dataProcessor", "@dataSubject",
    "@retentionUntil", "@jurisdiction", "@accessLevel", "@consent",
    "@erasureRequested", "@erasureRequestedAt", "@erasureCompletedAt",
    "@restrictProcessing", "@restrictionReason", "@processingRestrictions",
    "@portabilityFormat", "@rectifiedAt", "@rectificationNote",
})


def _walk_doc(
    doc: dict[str, Any],
    callback: Any,
) -> None:
    """Walk a JSON-LD document, calling callback for each protected property value."""
    node_id = doc.get("@id", f"_:root-{uuid.uuid4().hex[:8]}")

    for key, val in doc.items():
        if key.startswith("@"):
            continue
        values = val if isinstance(val, list) else [val]
        for v in values:
            if isinstance(v, dict) and any(k in _PROTECTION_KEYS for k in v):
                callback(node_id, key, v)


def _extract_id(node: Any) -> str:
    """Extract @id from a node, handling str or dict."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        return node.get("@id", "")
    return ""


def _is_handling_type(type_val: Any) -> bool:
    """Check if a type value indicates a PersonalDataHandling node."""
    if isinstance(type_val, str):
        return "PersonalDataHandling" in type_val
    if isinstance(type_val, list):
        return any("PersonalDataHandling" in str(t) for t in type_val)
    return False


def _count_protection_triples(doc: dict[str, Any]) -> int:
    """Count approximate triple count for jsonld-ex protection annotations."""
    count = 0

    def _counter(node_id: str, prop_name: str, prop_val: dict[str, Any]) -> None:
        nonlocal count
        # 1 triple for the value itself, 1 per annotation field
        count += 1
        for k in prop_val:
            if k in _PROTECTION_KEYS:
                count += 1

    _walk_doc(doc, _counter)
    return count
