/**
 * Inference Engine for JSON-LD-Ex.
 *
 * Provides algorithms for:
 * 1. Confidence Propagation (how confidence flows through a graph).
 * 2. Confidence Combination (how multiple sources are combined).
 * 3. Conflict Resolution (how to choose between conflicting values).
 */

import { PropagationResult, ConflictReport } from './types.js';

// ── Validation ────────────────────────────────────────────────────

export function validateConfidence(c: number): void {
    if (c < 0.0 || c > 1.0 || isNaN(c)) {
        throw new Error(`Confidence must be in [0, 1], got: ${c}`);
    }
}

// ── Propagation ───────────────────────────────────────────────────

export function propagateConfidence(
    chain: number[],
    method: 'multiply' | 'min' | 'average' = 'multiply'
): PropagationResult {
    if (chain.length === 0) {
        throw new Error('Chain must contain at least one score');
    }
    chain.forEach(validateConfidence);

    let score: number;
    if (method === 'multiply') {
        score = chain.reduce((acc, val) => acc * val, 1.0);
    } else if (method === 'min') {
        score = Math.min(...chain);
    } else if (method === 'average') {
        score = chain.reduce((acc, val) => acc + val, 0.0) / chain.length;
    } else {
        throw new Error(`Unknown propagation method: ${method}`);
    }

    return {
        score,
        method,
        inputScores: chain,
    };
}

// ── Combination ───────────────────────────────────────────────────

export function combineSources(
    scores: number[],
    method: 'noisy_or' | 'average' | 'max' = 'noisy_or'
): number {
    if (scores.length === 0) {
        throw new Error('Scores must contain at least one value');
    }
    scores.forEach(validateConfidence);

    if (method === 'noisy_or') {
        // 1 - Product(1 - p_i)
        const productOfNegations = scores.reduce(
            (acc, val) => acc * (1.0 - val),
            1.0
        );
        return 1.0 - productOfNegations;
    } else if (method === 'average') {
        return scores.reduce((acc, val) => acc + val, 0.0) / scores.length;
    } else if (method === 'max') {
        return Math.max(...scores);
    } else {
        throw new Error(`Unknown combination method: ${method}`);
    }
}

// ── Conflict Resolution ───────────────────────────────────────────

export function resolveConflict(
    assertions: any[],
    strategy: 'highest' | 'weighted_vote' | 'recency' = 'highest'
): ConflictReport {
    if (assertions.length === 0) {
        throw new Error('No assertions to resolve');
    }

    // Filter out invalid candidates if needed? No, assume upstream filtering.

    if (assertions.length === 1) {
        return {
            winner: assertions[0],
            strategy,
            candidates: assertions,
            confidenceScores: [getConfidence(assertions[0])],
            reason: 'Single candidate',
        };
    }

    let winner: any;
    let reason: string = '';

    if (strategy === 'highest') {
        winner = assertions.reduce((prev, curr) => {
            return getConfidence(curr) > getConfidence(prev) ? curr : prev;
        });
        reason = `Highest confidence: ${getConfidence(winner)}`;
    } else if (strategy === 'recency') {
        // Assuming @extractedAt or similar
        winner = assertions.reduce((prev, curr) => {
            const t1 = getDate(prev);
            const t2 = getDate(curr);
            return t2 > t1 ? curr : prev;
        });
        reason = `Most recent: ${getDate(winner).toISOString()}`;
    } else {
        // weighted_vote? Not implementing complex voting logic here yet without more context.
        // Fallback to highest.
        winner = assertions.reduce((prev, curr) => {
            return getConfidence(curr) > getConfidence(prev) ? curr : prev;
        });
        reason = 'Fallback to highest';
    }

    return {
        winner,
        strategy,
        candidates: assertions,
        confidenceScores: assertions.map(getConfidence),
        reason,
    };
}

function getConfidence(node: any): number {
    return node['@confidence'] ?? node['confidence'] ?? 0.5; // Default 0.5
}

function getDate(node: any): Date {
    const s = node['@extractedAt'] ?? node['extractedAt'];
    if (s) return new Date(s);
    return new Date(0); // Epoch if missing
}
