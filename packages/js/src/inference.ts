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
    method: 'multiply' | 'min' | 'average' | 'bayesian' | 'dampened' = 'multiply'
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
    } else if (method === 'bayesian') {
        score = chainBayesian(chain);
    } else if (method === 'dampened') {
        score = chainDampened(chain);
    } else {
        throw new Error(`Unknown propagation method: ${method}`);
    }

    return {
        score,
        method,
        inputScores: chain,
    };
}

function chainBayesian(scores: number[]): number {
    let logOdds = 0.0; // log(0.5/0.5) = 0
    for (const c of scores) {
        // Clamp to avoid Infinity
        const cSafe = Math.max(1e-4, Math.min(c, 1.0 - 1e-4));
        logOdds += Math.log(cSafe / (1.0 - cSafe));
    }
    const odds = Math.exp(logOdds);
    return odds / (1.0 + odds);
}

function chainDampened(scores: number[]): number {
    const n = scores.length;
    const product = scores.reduce((acc, val) => acc * val, 1.0);
    if (product === 0.0) return 0.0;
    return Math.pow(product, 1.0 / Math.sqrt(n));
}

// ── Combination ───────────────────────────────────────────────────

export function combineSources(
    scores: number[],
    method: 'noisy_or' | 'average' | 'max' | 'dempster_shafer' = 'noisy_or'
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
    } else if (method === 'dempster_shafer') {
        return dempsterShafer(scores);
    } else {
        throw new Error(`Unknown combination method: ${method}`);
    }
}

function dempsterShafer(scores: number[]): number {
    let belief = scores[0];
    let uncertainty = 1.0 - scores[0];

    for (let i = 1; i < scores.length; i++) {
        const b2 = scores[i];
        const u2 = 1.0 - scores[i];

        // Combined belief (assuming no disbelief/conflict mass in this simplified model)
        const newBelief = belief * b2 + belief * u2 + uncertainty * b2;
        const newUncertainty = uncertainty * u2;

        const total = newBelief + newUncertainty;
        if (total === 0) {
            belief = 0.0;
            uncertainty = 1.0;
        } else {
            belief = newBelief / total;
            uncertainty = newUncertainty / total;
        }
    }
    return belief;
}

// ── Conflict Resolution ───────────────────────────────────────────

export function resolveConflict(
    assertions: any[],
    strategy: 'highest' | 'weighted_vote' | 'recency' = 'highest'
): ConflictReport {
    if (assertions.length === 0) {
        throw new Error('No assertions to resolve');
    }

    if (assertions.length === 1) {
        return {
            winner: assertions[0],
            strategy,
            candidates: assertions,
            confidenceScores: [getConfidence(assertions[0])],
            reason: 'Single candidate',
        };
    }

    const confScores = assertions.map(getConfidence);

    if (strategy === 'highest') {
        return resolveHighest(assertions, confScores);
    } else if (strategy === 'recency') {
        return resolveRecency(assertions, confScores);
    } else if (strategy === 'weighted_vote') {
        return resolveWeightedVote(assertions, confScores);
    } else {
        throw new Error(`Unknown strategy: ${strategy}`);
    }
}

function resolveHighest(candidates: any[], scores: number[]): ConflictReport {
    let bestIdx = 0;
    for (let i = 1; i < candidates.length; i++) {
        if (scores[i] > scores[bestIdx]) {
            bestIdx = i;
        }
    }
    return {
        winner: candidates[bestIdx],
        strategy: 'highest',
        candidates,
        confidenceScores: scores,
        reason: `Highest confidence: ${scores[bestIdx].toFixed(4)}`,
    };
}

function resolveRecency(candidates: any[], scores: number[]): ConflictReport {
    // Sort by extractedAt descending, then confidence descending
    const indexed = candidates.map((c, i) => ({ c, score: scores[i], idx: i }));

    indexed.sort((a, b) => {
        const t1 = getDate(a.c).getTime();
        const t2 = getDate(b.c).getTime();
        if (t2 !== t1) return t2 - t1; // Descending time
        return b.score - a.score;      // Descending confidence
    });

    const winner = indexed[0];
    return {
        winner: winner.c,
        strategy: 'recency',
        candidates,
        confidenceScores: scores,
        reason: `Most recent: ${getDate(winner.c).toISOString()}`,
    };
}

function resolveWeightedVote(candidates: any[], scores: number[]): ConflictReport {
    const groups = new Map<string, { scores: number[]; assertions: any[] }>();

    for (let i = 0; i < candidates.length; i++) {
        const val = getBareValue(candidates[i]);
        const key = typeof val === 'object' ? JSON.stringify(val) : String(val); // Simple key generation

        if (!groups.has(key)) {
            groups.set(key, { scores: [], assertions: [] });
        }
        const g = groups.get(key)!;
        g.scores.push(scores[i]);
        g.assertions.push(candidates[i]);
    }

    let bestKey: string | null = null;
    let bestCombined = -1.0;

    for (const [key, g] of groups.entries()) {
        const combined = combineSources(g.scores, 'noisy_or');
        if (combined > bestCombined) {
            bestCombined = combined;
            bestKey = key;
        }
    }

    if (!bestKey) {
        // Should not happen if candidates > 0
        return resolveHighest(candidates, scores);
    }

    const winningGroup = groups.get(bestKey)!;

    // Pick highest individual confidence from winning group as base
    const bestInGroupIdx = winningGroup.scores.indexOf(Math.max(...winningGroup.scores));
    const baseWinner = winningGroup.assertions[bestInGroupIdx];

    // Clone and attach combined confidence
    let winner: any;
    if (typeof baseWinner === 'object' && baseWinner !== null) {
        winner = { ...baseWinner }; // Shallow copy
        winner['@confidence'] = parseFloat(bestCombined.toFixed(10));
    } else {
        // Primitive value? This usually shouldn't happen for assertions passed to resolveConflict
        // But if it does, wrap it
        winner = {
            '@value': baseWinner,
            '@confidence': parseFloat(bestCombined.toFixed(10))
        };
    }

    return {
        winner,
        strategy: 'weighted_vote',
        candidates,
        confidenceScores: scores,
        reason: `Value supported by ${winningGroup.scores.length} source(s) with combined noisy-OR confidence: ${bestCombined.toFixed(4)}`
    };
}

function getBareValue(node: any): any {
    if (node && typeof node === 'object' && '@value' in node) {
        return node['@value'];
    }
    return node;
}

function getConfidence(node: any): number {
    return node['@confidence'] ?? node['confidence'] ?? 0.5; // Default 0.5
}

function getDate(node: any): Date {
    const s = node['@extractedAt'] ?? node['extractedAt'];
    if (s) return new Date(s);
    return new Date(0); // Epoch if missing
}

// ── Graph-Level Helpers ───────────────────────────────────────────

/**
 * Propagate confidence along a property chain in a JSON-LD graph.
 */
export function propagateGraphConfidence(
    doc: any,
    propertyChain: string[],
    method: 'multiply' | 'min' | 'average' | 'bayesian' | 'dampened' = 'multiply'
): PropagationResult {
    const scores: number[] = [];
    const trail: string[] = [];

    let current = doc;
    for (const prop of propertyChain) {
        if (!current || typeof current !== 'object') {
            throw new Error(`Cannot traverse property '${prop}': current node is not an object`);
        }

        const value = current[prop];
        if (value === undefined) {
            throw new Error(`Property '${prop}' not found in document`);
        }

        // Extract confidence using helper (or local if simpler)
        let c = getConfidence(value);

        // If undefined (e.g. simple object link without annotation), assume 1.0 (certainty)
        // Check if value is a nested node vs annotated value
        if (value && typeof value === 'object') {
            if (!('@confidence' in value) && !('confidence' in value)) {
                // Assume 1.0 for structural links unless annotated
                c = 1.0;
            }
        } else {
            // Primitive without annotation wrapper -> 1.0
            c = 1.0;
        }

        scores.push(c);
        trail.push(prop);

        // Traverse
        if (value && typeof value === 'object') {
            // If annotated value leaf, stop? Or if array?
            // Simple traversal: strictly follow property chain
            // If it's an annotated value wrapper, we might need to drill down?
            // Python does: if annotated leaf (@value), stay. else current = value.
            if ('@value' in value) {
                // Leaf
            } else {
                current = value;
            }
        }
    }

    const result = propagateConfidence(scores, method);
    result.provenanceTrail = trail;
    return result;
}
