
import { JsonLdNode } from './types.js';
import { ConversionReport } from './owl.js';
import { v4 as uuidv4 } from 'uuid';

// ── Namespace Constants ───────────────────────────────────────────

export const DPV = "https://w3id.org/dpv#";
export const EU_GDPR = "https://w3id.org/dpv/legal/eu/gdpr#";
export const DPV_LOC = "https://w3id.org/dpv/loc#";
export const DPV_PD = "https://w3id.org/dpv/pd#";

// ── Mapping Tables ────────────────────────────────────────────────

export const LEGAL_BASIS_TO_DPV: Record<string, string> = {
    "consent": `${EU_GDPR}A6-1-a`,
    "contract": `${EU_GDPR}A6-1-b`,
    "legal_obligation": `${EU_GDPR}A6-1-c`,
    "vital_interest": `${EU_GDPR}A6-1-d`,
    "public_task": `${EU_GDPR}A6-1-e`,
    "legitimate_interest": `${EU_GDPR}A6-1-f`,
};

export const DPV_TO_LEGAL_BASIS: Record<string, string> = Object.entries(LEGAL_BASIS_TO_DPV).reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {});

export const CATEGORY_TO_DPV: Record<string, string> = {
    "regular": `${DPV_PD}PersonalData`,
    "sensitive": `${DPV_PD}SensitivePersonalData`,
    "special_category": `${DPV_PD}SpecialCategoryPersonalData`,
    "pseudonymized": `${DPV}PseudonymisedData`,
    "anonymized": `${DPV}AnonymisedData`,
    "synthetic": `${DPV}SyntheticData`,
    "non_personal": `${DPV}NonPersonalData`,
};

export const DPV_TO_CATEGORY: Record<string, string> = Object.entries(CATEGORY_TO_DPV).reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {});

export const ACCESS_LEVEL_TO_DPV: Record<string, string> = {
    "public": `${DPV}PublicAccess`,
    "internal": `${DPV}InternalAccess`,
    "restricted": `${DPV}RestrictedAccess`,
    "confidential": `${DPV}ConfidentialAccess`,
    "secret": `${DPV}SecretAccess`,
};

export const DPV_TO_ACCESS_LEVEL: Record<string, string> = Object.entries(ACCESS_LEVEL_TO_DPV).reduce((acc, [k, v]) => ({ ...acc, [v]: k }), {});

// ── DPV Context ───────────────────────────────────────────────────

const _DPV_CONTEXT = {
    "dpv": DPV,
    "eu-gdpr": EU_GDPR,
    "dpv-loc": DPV_LOC,
    "dpv-pd": DPV_PD,
    "xsd": "http://www.w3.org/2001/XMLSchema#",
};

// ── Internal Helpers ──────────────────────────────────────────────

const _PROTECTION_KEYS = new Set([
    "@personalDataCategory", "@legalBasis", "@processingPurpose",
    "@dataController", "@dataProcessor", "@dataSubject",
    "@retentionUntil", "@jurisdiction", "@accessLevel", "@consent",
    "@erasureRequested", "@erasureRequestedAt", "@erasureCompletedAt",
    "@restrictProcessing", "@restrictionReason", "@processingRestrictions",
    "@portabilityFormat", "@rectifiedAt", "@rectificationNote",
]);

function walkDoc(doc: JsonLdNode, callback: (nodeId: string, propName: string, propVal: JsonLdNode) => void) {
    const nodeId = doc['@id'] || `_:root-${uuidv4().substring(0, 8)}`;

    for (const key of Object.keys(doc)) {
        if (key.startsWith('@')) continue;

        const val = doc[key];
        const values = Array.isArray(val) ? val : [val];

        for (const v of values) {
            if (typeof v === 'object' && v !== null) {
                // Check if any protection key exists in v
                if (Object.keys(v).some(k => _PROTECTION_KEYS.has(k))) {
                    callback(nodeId, key, v);
                }
                // Recurse? The Python version recursion logic is:
                // _walk_doc calls callback for protected property values.
                // It doesn't seem to recursively walk the whole tree in the provided snippet?
                // Wait, Python `_walk_doc` iterates items and calls callback.
                // It does NOT recurse into children in the provided snippet unless I missed it.
                // Ah, `_walk_doc` provided in previous turn shows simple iteration over direct properties.
                // But to support nested objects, it should probably recurse.
                // However, for strict parity with the provided Python snippet, I will stick to shallow walk or mimic exactly.
                // Actually, looking at `dpv_interop.py`: `_walk_doc` iterates keys of `doc`. It does NOT recurse.
                // So it only works on the passed document structure (handled as a flat-ish graph or just checking direct children).
                // Since JSON-LD can be flattened, this assumption holds for flattened docs.
                // I will implement it as non-recursive for now to match Python snippet.
            }
        }
    }
}

function extractId(node: any): string {
    if (typeof node === 'string') return node;
    if (typeof node === 'object' && node !== null) return node['@id'] || '';
    return '';
}

// ── Forward Conversion: jsonld-ex → DPV ──────────────────────────

export function toDpv(doc: JsonLdNode): { dpvDoc: JsonLdNode, report: ConversionReport } {
    const report = new ConversionReport(true);
    const graphNodes: JsonLdNode[] = [];
    let outputTriples = 0;

    // Build context
    const ctx = { ..._DPV_CONTEXT };
    const originalCtx = doc['@context'];
    if (typeof originalCtx === 'object' && originalCtx !== null) {
        Object.assign(ctx, originalCtx);
    }

    const processProperty = (nodeId: string, propName: string, propVal: JsonLdNode) => {
        report.nodesConverted++;

        const handlingId = `_:handling-${uuidv4().substring(0, 8)}`;
        const handling: JsonLdNode = {
            "@id": handlingId,
            "@type": `${DPV}PersonalDataHandling`,
        };

        // Data category
        const cat = propVal["@personalDataCategory"];
        if (cat && CATEGORY_TO_DPV[cat]) {
            handling["dpv:hasPersonalData"] = {
                "@id": `_:pd-${uuidv4().substring(0, 8)}`,
                "@type": CATEGORY_TO_DPV[cat],
            };
            outputTriples += 2;
        }

        // Legal basis
        const basis = propVal["@legalBasis"];
        if (basis && LEGAL_BASIS_TO_DPV[basis]) {
            handling["dpv:hasLegalBasis"] = { "@id": LEGAL_BASIS_TO_DPV[basis] };
            outputTriples += 1;
        }

        // Data controller
        const controller = propVal["@dataController"];
        if (controller) {
            handling["dpv:hasDataController"] = {
                "@id": controller,
                "@type": `${DPV}DataController`,
            };
            outputTriples += 2;
        }

        // Data subject
        const subject = propVal["@dataSubject"];
        if (subject) {
            handling["dpv:hasDataSubject"] = { "@id": subject };
            outputTriples += 1;
        }

        // Processing purpose
        const purpose = propVal["@processingPurpose"];
        if (purpose) {
            const purposes = Array.isArray(purpose) ? purpose : [purpose];
            handling["dpv:hasPurpose"] = purposes.map(p =>
                p.startsWith("http") ? { "@id": p } : { "dpv:hasDescription": p }
            );
            outputTriples += purposes.length;
        }

        // Consent
        const consent = propVal["@consent"];
        if (consent && typeof consent === 'object') {
            const consentNode: JsonLdNode = {
                "@id": `_:consent-${uuidv4().substring(0, 8)}`,
                "@type": `${DPV}Consent`,
            };

            if (consent["@consentGivenAt"]) {
                consentNode["dpv:hasProvisionTime"] = consent["@consentGivenAt"];
                outputTriples++;
            }
            if (consent["@consentScope"]) {
                consentNode["dpv:hasScope"] = consent["@consentScope"];
                outputTriples++;
            }
            handling["dpv:hasConsent"] = consentNode;
            outputTriples += 2;
        }

        // Link handling to source node
        handling["dpv:hasSourceNode"] = { "@id": nodeId };
        handling["dpv:hasSourceProperty"] = propName;
        outputTriples += 2;

        graphNodes.push(handling);
    };

    walkDoc(doc, processProperty);

    report.triplesOutput = outputTriples;

    const dpvDoc: JsonLdNode = { "@context": ctx };
    if (graphNodes.length > 0) {
        dpvDoc["@graph"] = graphNodes;
    }

    return { dpvDoc, report };
}

// ── Reverse Conversion: DPV → jsonld-ex ──────────────────────────

export function fromDpv(dpvDoc: JsonLdNode): { doc: JsonLdNode, report: ConversionReport } {
    const report = new ConversionReport(true);
    const graph = dpvDoc["@graph"] || [];

    if (!Array.isArray(graph) || graph.length === 0) {
        return { doc: {}, report };
    }

    // Group handling nodes by source
    const restoredNodes: Record<string, any> = {};

    for (const node of graph) {
        // console.log('Processing node:', node['@id'], node['@type']);
        const type = node["@type"];
        const isHandling = typeof type === 'string' ? type.includes("PersonalDataHandling") :
            Array.isArray(type) ? type.some(t => t.includes("PersonalDataHandling")) : false;

        if (!isHandling) continue;

        report.nodesConverted++;

        const sourceNodeId = extractId(node["dpv:hasSourceNode"]);
        // console.log('Source Node ID:', sourceNodeId);
        const sourceProp = node["dpv:hasSourceProperty"];
        // console.log('Source Prop:', sourceProp);

        if (!sourceNodeId) continue;

        if (!restoredNodes[sourceNodeId]) {
            restoredNodes[sourceNodeId] = { "@id": sourceNodeId };
        }

        const propVal: JsonLdNode = { "@value": null };

        // Reverse map category
        const pdNode = node["dpv:hasPersonalData"];
        if (pdNode) {
            const pdType = pdNode["@type"] || "";
            const cat = DPV_TO_CATEGORY[pdType];
            // console.log('PD Type:', pdType, 'Category:', cat);
            if (cat) propVal["@personalDataCategory"] = cat;
        }

        // Reverse map legal basis
        const lbNode = node["dpv:hasLegalBasis"];
        if (lbNode) {
            const lbId = extractId(lbNode);
            const basis = DPV_TO_LEGAL_BASIS[lbId];
            if (basis) propVal["@legalBasis"] = basis;
        }

        // Consent
        const consentNode = node["dpv:hasConsent"];
        if (consentNode) {
            const consentRec: any = {};
            if (consentNode["dpv:hasProvisionTime"]) consentRec["@consentGivenAt"] = consentNode["dpv:hasProvisionTime"];
            if (consentNode["dpv:hasScope"]) consentRec["@consentScope"] = consentNode["dpv:hasScope"];

            if (Object.keys(consentRec).length > 0) {
                propVal["@consent"] = consentRec;
            }
        }

        if (sourceProp) {
            restoredNodes[sourceNodeId][sourceProp] = propVal;
        }
    }

    const nodes = Object.values(restoredNodes);
    let resultDoc: JsonLdNode = {};

    if (nodes.length === 1) {
        resultDoc = nodes[0];
    } else if (nodes.length > 1) {
        resultDoc = { "@graph": nodes };
    }

    return { doc: resultDoc, report };
}
