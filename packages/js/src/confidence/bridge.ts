/**
 * Bridge between scalar confidence and Subjective Logic algebra.
 */

import {
    Opinion,
    cumulativeFuse,
    averagingFuse,
    trustDiscount,
} from './algebra.js';
import {
    resolveConflict,
} from '../inference.js';
import {
    PropagationResult,
    ConflictReport,
} from '../types.js';

// ── Opinion-returning convenience functions ───────────────────────

/**
 * Combine scalar confidence scores via the formal algebra.
 *
 * @param scores Confidence scores [0, 1].
 * @param uncertainty Uncertainty to assign to each source (default 0).
 * @param fusion 'cumulative' or 'averaging'.
 * @param baseRate Prior probability.
 */
export function combineOpinionsFromScalars(
    scores: number[],
    uncertainty: number = 0.0,
    fusion: 'cumulative' | 'averaging' = 'cumulative',
    baseRate: number = 0.5
): Opinion {
    if (scores.length === 0) {
        throw new Error('Scores must contain at least one value');
    }

    // Validate scores
    scores.forEach(s => {
        if (s < 0 || s > 1) throw new Error(`Invalid confidence: ${s}`);
    });

    let opinions: Opinion[];

    if (fusion === 'cumulative' && uncertainty === 0.0) {
        // Natural mapping: p -> (b=p, d=0, u=1-p)
        opinions = scores.map(
            (p) => new Opinion(p, 0.0, 1.0 - p, baseRate)
        );
    } else {
        opinions = scores.map((p) =>
            Opinion.fromConfidence(p, uncertainty, baseRate)
        );
    }

    if (fusion === 'cumulative') {
        return cumulativeFuse(...opinions);
    } else {
        return averagingFuse(...opinions);
    }
}

/**
 * Propagate confidence through a chain via trust discount.
 *
 * @param chain Confidence scores along the inference path.
 * @param trustUncertainty Uncertainty in each trust link.
 * @param baseRate Prior probability.
 */
export function propagateOpinionsFromScalars(
    chain: number[],
    trustUncertainty: number = 0.0,
    baseRate: number = 0.0
): Opinion {
    if (chain.length === 0) {
        throw new Error('Chain must contain at least one score');
    }

    // Validate scores
    chain.forEach(s => {
        if (s < 0 || s > 1) throw new Error(`Invalid confidence: ${s}`);
    });

    // The assertion being propagated: absolute belief (b=1, d=0, u=0)
    // unless baseRate modifies it? No, assertion is typically categorical.
    // Code uses b=1, d=0, u=0.
    let current = new Opinion(1.0, 0.0, 0.0, baseRate);

    // Apply trust discounts in reverse order (innermost first)
    // for c in reversed(chain):
    for (let i = chain.length - 1; i >= 0; i--) {
        const c = chain[i];
        let trustOpinion: Opinion;

        if (trustUncertainty === 0.0) {
            // Dogmatic trust: b=c, d=1-c, u=0
            trustOpinion = new Opinion(c, 1.0 - c, 0.0, 0.5); // baseRate of trust? Doesn't matter for discount
        } else {
            trustOpinion = Opinion.fromConfidence(c, trustUncertainty);
        }

        current = trustDiscount(trustOpinion, current);
    }

    return current;
}

/**
 * Resolve conflicts and enrich the winner with Opinion metadata.
 */
export function resolveConflictWithOpinions(
    assertions: any[],
    strategy: 'highest' | 'weighted_vote' | 'recency' = 'highest'
): ConflictReport {
    // We need inference.ts to implement resolveConflict first?
    // The import above suggests we expect it in ../inference.
    // I haven't implemented src/inference.ts yet.
    // I can implement a stub or better, implement src/inference.ts NEXT.
    // But this file depends on it.
    // I will assume resolveConflict exists and matches signature.

    const report = resolveConflict(assertions, strategy);

    // Enrich winner
    const confidence = report.winner['@confidence'] ?? 0.5;
    const opinion = Opinion.fromConfidence(confidence);

    if (!report.winner['@opinion']) {
        report.winner['@opinion'] = opinion.toJsonLd();
    }

    return report;
}
