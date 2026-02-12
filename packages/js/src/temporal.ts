/**
 * Temporal Extensions for JSON-LD-Ex.
 *
 * Adds time-aware assertions to JSON-LD values, enabling knowledge graph
 * versioning and point-in-time queries.
 */

import {
    KEYWORD_VALID_FROM,
    KEYWORD_VALID_UNTIL,
    KEYWORD_AS_OF,
    JSONLD_EX_NAMESPACE,
} from './keywords.js';
import { TemporalDiffResult, TemporalOptions, TemporalQualifiers } from './types.js';

// ── Context Definition ────────────────────────────────────────────

export const TEMPORAL_CONTEXT = {
    '@vocab': JSONLD_EX_NAMESPACE,
    'validFrom': {
        '@id': `${JSONLD_EX_NAMESPACE}validFrom`,
        '@type': 'http://www.w3.org/2001/XMLSchema#dateTime',
    },
    'validUntil': {
        '@id': `${JSONLD_EX_NAMESPACE}validUntil`,
        '@type': 'http://www.w3.org/2001/XMLSchema#dateTime',
    },
    'asOf': {
        '@id': `${JSONLD_EX_NAMESPACE}asOf`,
        '@type': 'http://www.w3.org/2001/XMLSchema#dateTime',
    },
};

// ── Annotation Helpers ────────────────────────────────────────────

/**
 * Add temporal qualifiers to a value.
 *
 * @param value - The value to annotate (plain or already annotated).
 * @param qualifiers - Temporal qualifiers (validFrom, validUntil, asOf).
 * @returns An annotated-value object with temporal qualifiers.
 */
export function addTemporal(
    value: any,
    qualifiers: TemporalOptions
): any {
    if (!qualifiers.validFrom && !qualifiers.validUntil && !qualifiers.asOf) {
        throw new Error('At least one temporal qualifier must be provided');
    }

    // Validate timestamps
    const fromDate = qualifiers.validFrom ? parseTimestamp(qualifiers.validFrom) : null;
    const untilDate = qualifiers.validUntil ? parseTimestamp(qualifiers.validUntil) : null;
    if (qualifiers.asOf) parseTimestamp(qualifiers.asOf);

    if (fromDate && untilDate && fromDate > untilDate) {
        throw new Error(
            `@validFrom (${qualifiers.validFrom}) must not be after @validUntil (${qualifiers.validUntil})`
        );
    }

    let result: any;
    if (value && typeof value === 'object' && '@value' in value) {
        result = { ...value };
    } else {
        result = { '@value': value };
    }

    if (qualifiers.validFrom) result[KEYWORD_VALID_FROM] = qualifiers.validFrom;
    if (qualifiers.validUntil) result[KEYWORD_VALID_UNTIL] = qualifiers.validUntil;
    if (qualifiers.asOf) result[KEYWORD_AS_OF] = qualifiers.asOf;

    return result;
}

// ── Time-slice Query ──────────────────────────────────────────────

/**
 * Return the graph state as of a given timestamp.
 *
 * For each node, retains only the properties whose temporal bounds
 * include *timestamp* (or that have no temporal bounds — treated as
 * always-valid).
 *
 * @param graph - Array of JSON-LD nodes.
 * @param timestamp - ISO 8601 timestamp to query at.
 * @param propertyName - Optional property to filter by.
 */
export function queryAtTime(
    graph: any[],
    timestamp: string,
    propertyName?: string
): any[] {
    const ts = parseTimestamp(timestamp);
    const result: any[] = [];

    for (const node of graph) {
        const filtered = filterNodeAtTime(node, ts, propertyName);
        if (filtered) {
            result.push(filtered);
        }
    }

    return result;
}

function filterNodeAtTime(
    node: any,
    ts: Date,
    propertyName?: string
): any | null {
    const out: any = {};
    let hasAnyData = false;

    for (const [key, value] of Object.entries(node)) {
        // Identity keys always pass through
        if (key === '@id' || key === '@type' || key === '@context') {
            out[key] = value;
            continue;
        }

        // If filtering by specific property, pass others unchanged
        if (propertyName && key !== propertyName) {
            out[key] = value;
            hasAnyData = true;
            continue;
        }

        // Check temporal validity
        if (Array.isArray(value)) {
            const kept = value.filter((v: any) => isValidAt(v, ts));
            if (kept.length > 0) {
                out[key] = kept.length > 1 ? kept : kept[0];
                hasAnyData = true;
            }
        } else if (isValidAt(value, ts)) {
            out[key] = value;
            hasAnyData = true;
        }
    }

    if (!hasAnyData) return null;
    return out;
}

function isValidAt(value: any, ts: Date): boolean {
    if (!value || typeof value !== 'object') return true; // No metadata -> always valid

    const vf = value[KEYWORD_VALID_FROM] || value['validFrom']; // Handle compacted too? sticking to keywords for now
    const vu = value[KEYWORD_VALID_UNTIL] || value['validUntil'];

    if (!vf && !vu) return true;

    if (vf) {
        const fromDt = parseTimestamp(vf);
        if (ts < fromDt) return false;
    }
    if (vu) {
        const untilDt = parseTimestamp(vu);
        if (ts > untilDt) return false;
    }

    return true;
}

// ── Temporal Diff ─────────────────────────────────────────────────

/**
 * Compute what changed between two points in time.
 */
export function temporalDiff(
    graph: any[],
    t1Str: string,
    t2Str: string
): TemporalDiffResult {
    const snap1Nodes = queryAtTime(graph, t1Str).filter((n) => n['@id']);
    const snap2Nodes = queryAtTime(graph, t2Str).filter((n) => n['@id']);

    const snap1 = new Map(snap1Nodes.map((n) => [n['@id'], n]));
    const snap2 = new Map(snap2Nodes.map((n) => [n['@id'], n]));

    const result: TemporalDiffResult = {
        added: [],
        removed: [],
        modified: [],
    };

    const allIds = new Set([...snap1.keys(), ...snap2.keys()]);
    const sortedIds = Array.from(allIds).sort();

    for (const nid of sortedIds) {
        const n1 = snap1.get(nid);
        const n2 = snap2.get(nid);

        if (!n1 && n2) {
            result.added.push({ '@id': nid, state: n2 });
            continue;
        }
        if (n1 && !n2) {
            result.removed.push({ '@id': nid, state: n1 });
            continue;
        }

        if (n1 && n2) {
            // Compare properties
            const props1 = getDataProps(n1);
            const props2 = getDataProps(n2);
            const allProps = new Set([...Object.keys(props1), ...Object.keys(props2)]);
            const changes: Record<string, { from: any; to: any }> = {};
            let modified = false;

            for (const prop of allProps) {
                const v1 = props1[prop];
                const v2 = props2[prop];
                const bare1 = getBareValue(v1);
                const bare2 = getBareValue(v2);

                if (v1 === undefined && v2 !== undefined) {
                    // Property added
                    // We could represent this in modified or added list...
                    // Python implementation puts property adds/removes into top-level added/removed?
                    // Wait, Python implementation returns a list of *diff events*, not nodes with lists of changes.
                    // Let's re-read Python types.
                    // Python: added/removed/modified/unchanged are lists of dicts.
                    // JS Type: modified: Array<{ nodeId: string; changes: Record<string, { from: any; to: any }> }>;
                    // The JS type I defined in types.ts is slightly different from the Python flat list approach.
                    // I will stick to the JS type definition I created in types.ts.
                    changes[prop] = { from: undefined, to: bare2 };
                    modified = true;
                } else if (v1 !== undefined && v2 === undefined) {
                    changes[prop] = { from: bare1, to: undefined };
                    modified = true;
                } else if (!entryEquals(bare1, bare2)) {
                    changes[prop] = { from: bare1, to: bare2 };
                    modified = true;
                }
            }

            if (modified) {
                result.modified.push({
                    nodeId: nid,
                    changes
                });
            }
        }
    }

    return result;
}

function getDataProps(node: any): Record<string, any> {
    const out: Record<string, any> = {};
    for (const [k, v] of Object.entries(node)) {
        if (k !== '@id' && k !== '@type' && k !== '@context') {
            out[k] = v;
        }
    }
    return out;
}

function getBareValue(val: any): any {
    if (val && typeof val === 'object' && '@value' in val) {
        return val['@value'];
    }
    return val;
}

function entryEquals(a: any, b: any): boolean {
    // Simple equality check
    return JSON.stringify(a) === JSON.stringify(b);
}

// ── Helpers ───────────────────────────────────────────────────────

function parseTimestamp(ts: string): Date {
    const d = new Date(ts);
    if (isNaN(d.getTime())) {
        throw new Error(`Invalid timestamp: ${ts}`);
    }
    return d;
}
