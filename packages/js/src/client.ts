/**
 * High-level Client API for jsonld-ex.
 *
 * Provides a unified, validated interface for all jsonld-ex functionality.
 */

import {
    annotate,
    getProvenance,
    filterByConfidence,
} from './extensions/ai-ml.js';

import {
    mergeGraphs,
    diffGraphs
} from './merge.js';

import {
    propagateConfidence,
    combineSources,
    resolveConflict
} from './inference.js';

import {
    addTemporal,
    queryAtTime,
    temporalDiff
} from './temporal.js';

import {
    ProvenanceMetadataSchema,
    TemporalOptionsSchema,
    validate
} from './schemas.js';

import {
    ProvenanceMetadata,
    TemporalOptions,
    AnnotatedValue,
    MergeReport,
    GraphDiff,
    PropagationResult,
    ConflictReport,
    TemporalDiffResult
} from './types.js';

export class JsonLdExClient {
    /**
     * Annotate a value with AI/ML provenance metadata.
     * schema-validated.
     */
    public annotate(
        value: string | number | boolean,
        metadata: ProvenanceMetadata
    ): AnnotatedValue {
        const validMetadata = validate(ProvenanceMetadataSchema, metadata);
        return annotate(value, validMetadata);
    }

    /**
     * Extract provenance metadata from an annotated node.
     */
    public getProvenance(node: any): ProvenanceMetadata {
        return getProvenance(node);
    }

    /**
     * Filter a graph to include only nodes/properties meeting a confidence threshold.
     */
    public filterByConfidence(
        graph: any[],
        property: string,
        minConfidence: number
    ): any[] {
        return filterByConfidence(graph, property, minConfidence);
    }

    /**
     * Merge multiple JSON-LD graphs with conflict resolution.
     */
    public merge(
        graphs: any[],
        options: {
            conflictStrategy?: 'highest' | 'weighted_vote' | 'recency' | 'union';
            confidenceCombination?: 'noisy_or' | 'average' | 'max';
        } = {}
    ): { merged: any; report: MergeReport } {
        return mergeGraphs(
            graphs,
            options.conflictStrategy,
            options.confidenceCombination
        );
    }

    /**
     * Compute semantic diff between two graphs.
     */
    public diff(a: any, b: any): GraphDiff {
        return diffGraphs(a, b);
    }

    /**
     * Propagate confidence scores through a chain.
     */
    public propagate(
        chain: number[],
        method: 'multiply' | 'min' | 'average' = 'multiply'
    ): PropagationResult {
        return propagateConfidence(chain, method);
    }

    /**
     * Add temporal qualification to a value or node.
     */
    public addTemporal(
        input: any,
        options: TemporalOptions
    ): any {
        const validOptions = validate(TemporalOptionsSchema, options);
        return addTemporal(input, validOptions);
    }

    /**
     * Query a graph for its state at a specific time.
     */
    public queryAtTime(graph: any[], timestamp: string): any[] {
        return queryAtTime(graph, timestamp);
    }
}

// Default export for easy usage
export default new JsonLdExClient();
