
/**
 * Batch API for high-throughput annotation, validation, and filtering (GAP-API1).
 * 
 * All functions accept arrays and process with minimal per-call overhead.
 */

import { annotate, getConfidence } from './extensions/ai-ml.js';
import { validateNode } from './extensions/validation.js';
import { JsonLdNode, AnnotatedValue, ProvenanceMetadata, ValidationResult, ShapeDefinition } from './types.js';

export type BatchItem = any | [any, ProvenanceMetadata];

/**
 * Annotate a list of values with shared provenance metadata.
 * 
 * Each element can be either:
 * - A plain value (applied with the shared keyword arguments)
 * - A [value, overrides] tuple where overrides is a map of options 
 *   that override the shared defaults for that item.
 */
export function annotateBatch(
    items: BatchItem[],
    options: ProvenanceMetadata = {}
): AnnotatedValue[] {
    const results: AnnotatedValue[] = [];

    // We don't need to pre-build a "shared" dict like Python because 
    // we can just spread `options` and `overrides` easily in JS.

    for (const item of items) {
        let value: any;
        let itemOptions: ProvenanceMetadata;

        if (Array.isArray(item) && item.length === 2 && typeof item[1] === 'object') {
            value = item[0];
            itemOptions = { ...options, ...item[1] };
        } else {
            value = item;
            itemOptions = options;
        }

        results.push(annotate(value, itemOptions));
    }

    return results;
}

/**
 * Validate a list of nodes against a single shape.
 * 
 * Returns one ValidationResult per node, in order.
 */
export function validateBatch(
    nodes: JsonLdNode[],
    shape: ShapeDefinition
): ValidationResult[] {
    return nodes.map(node => validateNode(node, shape));
}

export type FilterCriteria = string | [string, number][];

/**
 * Filter nodes by confidence threshold on one or more properties.
 */
export function filterByConfidenceBatch(
    nodes: JsonLdNode[],
    criteria: FilterCriteria,
    minConfidence: number = 0.0
): JsonLdNode[] {
    let pairs: [string, number][];

    if (typeof criteria === 'string') {
        pairs = [[criteria, minConfidence]];
    } else {
        pairs = criteria;
    }

    const results: JsonLdNode[] = [];
    for (const node of nodes) {
        if (passesAll(node, pairs)) {
            results.push(node);
        }
    }
    return results;
}

function passesAll(node: JsonLdNode, pairs: [string, number][]): boolean {
    for (const [prop, threshold] of pairs) {
        const propValue = node[prop];
        if (propValue === undefined || propValue === null) {
            return false;
        }

        const values = Array.isArray(propValue) ? propValue : [propValue];

        // Python logic: if ANY value for the property meets threshold, pass.
        // "if not any( (c := get_confidence(v)) is not None and c >= threshold for v in values )"

        const hasPassingValue = values.some(v => {
            const c = getConfidence(v);
            return c !== undefined && c >= threshold;
        });

        if (!hasPassingValue) {
            return false;
        }
    }
    return true;
}
