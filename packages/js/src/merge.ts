/**
 * Graph Merging with Confidence-Aware Conflict Resolution.
 */

import { combineSources, resolveConflict } from './inference.js';
import { getConfidence } from './extensions/ai-ml.js';
import { MergeReport, MergeConflict, GraphDiff, ConflictReport } from './types.js';

// ── Constants ─────────────────────────────────────────────────────

const ANNOTATION_KEYS = new Set([
    '@confidence',
    '@source',
    '@extractedAt',
    '@method',
    '@humanVerified',
    '@derivedFrom',
]);

const NODE_IDENTITY_KEYS = new Set(['@id', '@type', '@context']);

// ── Graph Merging ─────────────────────────────────────────────────

/**
 * Merge multiple JSON-LD graphs with confidence-aware conflict resolution.
 */
export function mergeGraphs(
    graphs: any[],
    conflictStrategy: 'highest' | 'weighted_vote' | 'recency' | 'union' = 'highest',
    confidenceCombination: 'noisy_or' | 'average' | 'max' = 'noisy_or'
): { merged: any; report: MergeReport } {
    if (graphs.length < 2) {
        throw new Error('mergeGraphs requires at least 2 graphs');
    }

    const report: MergeReport = {
        nodesMerged: 0,
        propertiesAgreed: 0,
        propertiesConflicted: 0,
        propertiesUnion: 0,
        conflicts: [],
        sourceCount: graphs.length,
    };

    // Step 1: Collect all nodes, indexed by @id
    const idBuckets: Map<string, any[]> = new Map();
    const anonymousNodes: any[] = [];
    let mergedContext: any = null;

    for (const graph of graphs) {
        if (mergedContext === null && graph['@context']) {
            mergedContext = JSON.parse(JSON.stringify(graph['@context']));
        }

        const nodes = extractNodes(graph);
        for (const node of nodes) {
            if (node['@id']) {
                if (!idBuckets.has(node['@id'])) {
                    idBuckets.set(node['@id'], []);
                }
                idBuckets.get(node['@id'])!.push(node);
            } else {
                anonymousNodes.push(JSON.parse(JSON.stringify(node)));
            }
        }
    }

    // Step 2: Merge each bucket
    const mergedNodes: any[] = [];

    for (const [nodeId, nodes] of idBuckets.entries()) {
        const merged = mergeNodeGroup(
            nodeId,
            nodes,
            conflictStrategy,
            confidenceCombination,
            report
        );
        mergedNodes.push(merged);
        report.nodesMerged += 1;
    }

    // Pass through anonymous nodes
    mergedNodes.push(...anonymousNodes);

    // Step 3: Assemble output
    const result: any = {};
    if (mergedContext) {
        result['@context'] = mergedContext;
    }

    // If only one node and it was root (no @graph originally), maybe return object?
    // But strictly, we return { @graph: [...] } or { ... }?
    // Python implementation returns { @graph: [...] } generally unless specific structure.
    // We'll standardise on { @graph: [...] } to handle multiple nodes.
    result['@graph'] = mergedNodes;

    return { merged: result, report };
}

// ── Graph Diff ────────────────────────────────────────────────────

export function diffGraphs(a: any, b: any): GraphDiff {
    const nodesA = indexById(extractNodes(a));
    const nodesB = indexById(extractNodes(b));

    const idsA = new Set(nodesA.keys());
    const idsB = new Set(nodesB.keys());

    const added: any[] = [];
    const removed: any[] = [];
    const modified: any[] = [];
    const unchanged: any[] = [];

    // Nodes only in B
    for (const nid of idsB) {
        if (!idsA.has(nid)) {
            added.push({ '@id': nid, node: nodesB.get(nid) });
        }
    }

    // Nodes only in A
    for (const nid of idsA) {
        if (!idsB.has(nid)) {
            removed.push({ '@id': nid, node: nodesA.get(nid) });
        }
    }

    // Nodes in both
    for (const nid of idsA) {
        if (idsB.has(nid)) {
            const nodeA = nodesA.get(nid);
            const nodeB = nodesB.get(nid);

            const propsA = getDataProperties(nodeA);
            const propsB = getDataProperties(nodeB);

            const allProps = new Set([...Object.keys(propsA), ...Object.keys(propsB)]);

            for (const prop of allProps) {
                const valA = propsA[prop];
                const valB = propsB[prop];

                if (valA === undefined) {
                    added.push({ '@id': nid, property: prop, value: valB });
                } else if (valB === undefined) {
                    removed.push({ '@id': nid, property: prop, value: valA });
                } else if (bareValue(valA) === bareValue(valB)) {
                    // Unchanged logic (ignoring confidence diff logic for brevity, or adding?)
                    unchanged.push({ '@id': nid, property: prop, value: bareValue(valA) });
                } else {
                    modified.push({
                        '@id': nid,
                        property: prop,
                        valueA: valA,
                        valueB: valB
                    });
                }
            }
        }
    }

    return { added, removed, modified, unchanged };
}

// ── Helpers ───────────────────────────────────────────────────────

function extractNodes(doc: any): any[] {
    if (doc['@graph']) {
        const g = doc['@graph'];
        return Array.isArray(g) ? g : [g];
    }
    if (doc['@id'] || doc['@type']) {
        const { ['@context']: _, ...node } = doc;
        return [node];
    }
    return [];
}

function indexById(nodes: any[]): Map<string, any> {
    const index = new Map<string, any>();
    nodes.forEach(n => {
        if (n['@id']) index.set(n['@id'], n);
    });
    return index;
}

function getDataProperties(node: any): Record<string, any> {
    const props: Record<string, any> = {};
    for (const key of Object.keys(node)) {
        if (!NODE_IDENTITY_KEYS.has(key)) {
            props[key] = node[key];
        }
    }
    return props;
}

function bareValue(val: any): any {
    if (val && typeof val === 'object') {
        if ('@value' in val) return val['@value'];
        if ('@id' in val) return val['@id'];
    }
    return val;
}

// ── Merge Logic ───────────────────────────────────────────────────

function mergeNodeGroup(
    nodeId: string,
    nodes: any[],
    conflictStrategy: string,
    confidenceCombination: string,
    report: MergeReport
): any {
    const merged: any = { '@id': nodeId };

    // Merge types
    const types = new Set<string>();
    nodes.forEach(n => {
        const t = n['@type'];
        if (typeof t === 'string') types.add(t);
        else if (Array.isArray(t)) t.forEach((x: string) => types.add(x));
    });

    if (types.size === 1) merged['@type'] = Array.from(types)[0];
    else if (types.size > 1) merged['@type'] = Array.from(types).sort();

    // Collect properties
    const allProps = new Set<string>();
    nodes.forEach(n => {
        Object.keys(n).forEach(k => {
            if (!NODE_IDENTITY_KEYS.has(k)) allProps.add(k);
        });
    });

    for (const prop of Array.from(allProps).sort()) {
        const values: any[] = [];
        nodes.forEach(n => {
            if (n[prop] !== undefined) values.push(n[prop]);
        });

        if (values.length === 0) continue;
        if (values.length === 1) {
            merged[prop] = JSON.parse(JSON.stringify(values[0]));
            report.propertiesAgreed += 1;
        } else {
            const bareVals = values.map(bareValue);
            if (allEqual(bareVals)) {
                // Agreement
                merged[prop] = combineAgreed(values, confidenceCombination);
                report.propertiesAgreed += 1;
            } else {
                // Conflict
                if (conflictStrategy === 'union') {
                    merged[prop] = JSON.parse(JSON.stringify(values));
                    report.propertiesUnion += 1;
                    report.conflicts.push({
                        nodeId,
                        propertyName: prop,
                        values: values,
                        resolution: 'union',
                        winnerValue: bareVals
                    });
                } else {
                    // Resolve
                    // Use resolveConflict from inference.ts
                    // Need to normalize values to assertions expected by resolveConflict
                    const assertions = normalizeAssertions(values);
                    const result = resolveConflict(assertions, conflictStrategy as any);
                    merged[prop] = JSON.parse(JSON.stringify(result.winner)); // Use the winner structure

                    // Restore if we stripped structure? 
                    // resolveConflict returns one of the input assertions.
                    // If inputs were plain values, we wrapped them.
                    // If result.winner is a wrapper around a plain value, we should probably unwrap 
                    // or keep it if standard JSON-LD?
                    // Python logic: `merged[prop] = copy.deepcopy(winner)`
                    // `_resolve_property_conflict` returns `result.winner`.

                    // However, if the winner was wrapped by us, we should unwrap it IF the original was plain?
                    // Complexity: mixed plain/annotated.

                    // We'll simplify: if the strategy returns a winner, use it.

                    report.conflicts.push({
                        nodeId,
                        propertyName: prop,
                        values: values,
                        resolution: conflictStrategy,
                        winnerValue: bareValue(merged[prop])
                    });
                    report.propertiesConflicted += 1;
                }
            }
        }
    }

    return merged;
}

function allEqual(values: any[]): boolean {
    if (values.length <= 1) return true;
    const first = values[0];
    return values.slice(1).every(v => v === first);
}

function combineAgreed(values: any[], method: string): any {
    const scores: number[] = [];
    let bestValue = values[0];
    let bestConf = -1.0;

    values.forEach(v => {
        const c = getConfidence(v);
        if (c !== undefined) {
            scores.push(c);
            if (c > bestConf) {
                bestConf = c;
                bestValue = v;
            }
        }
    });

    if (scores.length < 2) return JSON.parse(JSON.stringify(bestValue)); // Not enough to combine

    const combinedScore = combineSources(scores, method as any);
    const result = JSON.parse(JSON.stringify(bestValue));

    if (typeof result === 'object' && result !== null) {
        result['@confidence'] = parseFloat(combinedScore.toFixed(10));
    }
    // If primitive, can't attach confidence without wrapping.
    // Python code assumes `bestValue` can be modified or copied. 
    // If it's a primitive, Python fails? `if isinstance(result, dict):`
    // So if primitive, we lose the combined confidence?
    // In JSON-LD, primitives should be wrapped in { @value: ... } to have annotation.
    // We won't auto-wrap here to avoid structure change.

    return result;
}

function normalizeAssertions(values: any[]): any[] {
    return values.map(v => {
        if (v && typeof v === 'object' && '@value' in v) {
            const a = JSON.parse(JSON.stringify(v));
            if (!a['@confidence']) a['@confidence'] = 0.5;
            return a;
        } else {
            // Check if it's an object (node ref?)
            const c = getConfidence(v);
            if (c !== undefined && typeof v === 'object') {
                // It's already an object with confidence?
                // But wait, if it's { @id: ... }, resolveConflict expects @value or @id?
                return v;
            }

            // Wrap
            return {
                '@value': bareValue(v),
                '@confidence': c ?? 0.5
            };
        }
    });
}
