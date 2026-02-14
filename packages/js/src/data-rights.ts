
import { JsonLdNode, DataProtectionMetadata } from './types.js';
import { isPersonalData } from './data-protection.js';

// ── Data Structures ───────────────────────────────────────────────

export interface ErasurePlan {
    dataSubject: string;
    affectedNodeIds: string[];
    affectedPropertyCount: number;
}

export interface ErasureAudit {
    dataSubject: string;
    erasedNodeIds: string[];
    erasedPropertyCount: number;
}

export interface RestrictionResult {
    dataSubject: string;
    restrictedPropertyCount: number;
}

export interface PortableExport {
    dataSubject: string;
    format: string;
    records: any[];
}

export interface AccessReport {
    dataSubject: string;
    records: any[];
    totalPropertyCount: number;
    categories: string[]; // Set<string> serialized
    controllers: string[]; // Set<string> serialized
    legalBases: string[]; // Set<string> serialized
    jurisdictions: string[]; // Set<string> serialized
}

export interface RetentionViolation {
    nodeId: string;
    propertyName: string;
    retentionUntil: string;
}

export interface AuditEntry {
    nodeId: string;
    propertyName: string;
    dataSubject: string;
    value?: any;
    personalDataCategory?: string;
    legalBasis?: string;
    dataController?: string;
    erasureRequested?: boolean;
    restrictProcessing?: boolean;
}

// ── Internal Helpers ──────────────────────────────────────────────

/**
 * Yield (node, property_name, property_value) for all properties
 * belonging to *dataSubject*.
 */
function* iterSubjectProperties(
    graph: JsonLdNode[],
    dataSubject: string
): Generator<[JsonLdNode, string, JsonLdNode]> {
    for (const node of graph) {
        for (const key of Object.keys(node)) {
            if (key.startsWith('@')) continue;

            const val = node[key];
            const values = Array.isArray(val) ? val : [val];

            for (const v of values) {
                if (typeof v === 'object' && v !== null && v['@dataSubject'] === dataSubject) {
                    yield [node, key, v];
                }
            }
        }
    }
}

// ── Operations ────────────────────────────────────────────────────

/**
 * Mark all properties belonging to a data subject for erasure (GDPR Art. 17).
 * Mutates the graph in place.
 */
export function requestErasure(
    graph: JsonLdNode[],
    options: { dataSubject: string; requestedAt?: string }
): ErasurePlan {
    const { dataSubject, requestedAt } = options;
    const plan: ErasurePlan = {
        dataSubject,
        affectedNodeIds: [],
        affectedPropertyCount: 0
    };

    const seenNodes = new Set<string>();

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        propVal['@erasureRequested'] = true;
        if (requestedAt) {
            propVal['@erasureRequestedAt'] = requestedAt;
        }

        plan.affectedPropertyCount++;
        const nodeId = node['@id'];
        if (nodeId && !seenNodes.has(nodeId)) {
            seenNodes.add(nodeId);
            plan.affectedNodeIds.push(nodeId);
        }
    }

    return plan;
}

/**
 * Execute erasure on properties previously marked with `@erasureRequested`.
 * Sets `@value` to null and records `@erasureCompletedAt`.
 */
export function executeErasure(
    graph: JsonLdNode[],
    options: { dataSubject: string; completedAt?: string }
): ErasureAudit {
    const { dataSubject, completedAt } = options;
    const audit: ErasureAudit = {
        dataSubject,
        erasedNodeIds: [],
        erasedPropertyCount: 0
    };

    const seenNodes = new Set<string>();

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        if (propVal['@erasureRequested'] !== true) continue;

        propVal['@value'] = null;
        if (completedAt) {
            propVal['@erasureCompletedAt'] = completedAt;
        }

        audit.erasedPropertyCount++;
        const nodeId = node['@id'];
        if (nodeId && !seenNodes.has(nodeId)) {
            seenNodes.add(nodeId);
            audit.erasedNodeIds.push(nodeId);
        }
    }

    return audit;
}

/**
 * Mark all properties of a data subject for restricted processing (GDPR Art. 18).
 */
export function requestRestriction(
    graph: JsonLdNode[],
    options: {
        dataSubject: string;
        reason: string;
        processingRestrictions?: string[]
    }
): RestrictionResult {
    const { dataSubject, reason, processingRestrictions } = options;
    const result: RestrictionResult = {
        dataSubject,
        restrictedPropertyCount: 0
    };

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        propVal['@restrictProcessing'] = true;
        propVal['@restrictionReason'] = reason;
        if (processingRestrictions) {
            propVal['@processingRestrictions'] = processingRestrictions;
        }
        result.restrictedPropertyCount++;
    }

    return result;
}

/**
 * Export all personal data for a data subject in a portable format (GDPR Art. 20).
 */
export function exportPortable(
    graph: JsonLdNode[],
    options: { dataSubject: string; format?: string }
): PortableExport {
    const { dataSubject, format = 'json' } = options;
    const exportData: PortableExport = {
        dataSubject,
        format,
        records: []
    };

    const nodeProps = new Map<string, any>();

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        const nodeId = node['@id'] || '';

        if (!nodeProps.has(nodeId)) {
            nodeProps.set(nodeId, {
                node_id: nodeId,
                type: node['@type'],
                properties: {}
            });
        }

        const record = nodeProps.get(nodeId);
        record.properties[propName] = propVal['@value'];
    }

    exportData.records = Array.from(nodeProps.values());
    return exportData;
}

/**
 * Create a rectified copy of an annotated node (GDPR Art. 16).
 * Returns a new object — does not mutate the original.
 */
export function rectifyData(
    node: JsonLdNode,
    options: { newValue: any; note: string; rectifiedAt?: string }
): JsonLdNode {
    const { newValue, note, rectifiedAt } = options;
    const result = { ...node };

    result['@value'] = newValue;
    result['@rectificationNote'] = note;
    if (rectifiedAt) {
        result['@rectifiedAt'] = rectifiedAt;
    }

    return result;
}

/**
 * Generate a structured report of all data held about a subject (GDPR Art. 15).
 */
export function rightOfAccessReport(
    graph: JsonLdNode[],
    options: { dataSubject: string }
): AccessReport {
    const { dataSubject } = options;
    const report: AccessReport = {
        dataSubject,
        records: [],
        totalPropertyCount: 0,
        categories: [],
        controllers: [],
        legalBases: [],
        jurisdictions: []
    };

    const categories = new Set<string>();
    const controllers = new Set<string>();
    const legalBases = new Set<string>();
    const jurisdictions = new Set<string>();
    const nodeData = new Map<string, any>();

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        const nodeId = node['@id'] || '';

        if (!nodeData.has(nodeId)) {
            nodeData.set(nodeId, {
                node_id: nodeId,
                type: node['@type'],
                properties: {}
            });
        }

        const record = nodeData.get(nodeId);
        record.properties[propName] = propVal['@value'];
        report.totalPropertyCount++;

        if (propVal['@personalDataCategory']) categories.add(propVal['@personalDataCategory']);
        if (propVal['@dataController']) controllers.add(propVal['@dataController']);
        if (propVal['@legalBasis']) legalBases.add(propVal['@legalBasis']);
        if (propVal['@jurisdiction']) jurisdictions.add(propVal['@jurisdiction']);
    }

    report.records = Array.from(nodeData.values());
    report.categories = Array.from(categories);
    report.controllers = Array.from(controllers);
    report.legalBases = Array.from(legalBases);
    report.jurisdictions = Array.from(jurisdictions);

    return report;
}

/**
 * Find all properties whose retention deadline has passed.
 */
export function validateRetention(
    graph: JsonLdNode[],
    options: { asOf: string }
): RetentionViolation[] {
    const { asOf } = options;
    const checkTime = new Date(asOf).getTime();
    const violations: RetentionViolation[] = [];

    for (const node of graph) {
        const nodeId = node['@id'] || '';

        for (const key of Object.keys(node)) {
            if (key.startsWith('@')) continue;

            const val = node[key];
            const values = Array.isArray(val) ? val : [val];

            for (const v of values) {
                if (typeof v === 'object' && v !== null) {
                    const retention = v['@retentionUntil'];
                    if (retention) {
                        const retentionTime = new Date(retention).getTime();
                        if (retentionTime < checkTime) {
                            violations.push({
                                nodeId,
                                propertyName: key,
                                retentionUntil: retention
                            });
                        }
                    }
                }
            }
        }
    }

    return violations;
}

/**
 * Build a complete audit trail for a data subject.
 */
export function auditTrail(
    graph: JsonLdNode[],
    options: { dataSubject: string }
): AuditEntry[] {
    const { dataSubject } = options;
    const entries: AuditEntry[] = [];

    for (const [node, propName, propVal] of iterSubjectProperties(graph, dataSubject)) {
        entries.push({
            nodeId: node['@id'] || '',
            propertyName: propName,
            dataSubject,
            value: propVal['@value'],
            personalDataCategory: propVal['@personalDataCategory'],
            legalBasis: propVal['@legalBasis'],
            dataController: propVal['@dataController'],
            erasureRequested: propVal['@erasureRequested'],
            restrictProcessing: propVal['@restrictProcessing']
        });
    }

    return entries;
}
