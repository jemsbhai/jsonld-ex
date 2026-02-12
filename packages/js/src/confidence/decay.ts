/**
 * Temporal Decay for Subjective Logic Opinions.
 *
 * Models the natural degradation of confidence over time: as evidence
 * ages, it becomes less reliable. The decay process migrates mass from
 * belief and disbelief into uncertainty.
 */

import { Opinion } from './algebra.js';

/**
 * Type definition for decay functions.
 * (elapsed, halfLife) -> factor in [0, 1]
 */
export type DecayFunction = (elapsed: number, halfLife: number) => number;

// ── Built-in decay functions ──────────────────────────────────────

/**
 * Exponential decay: λ = 2^(−t/τ).
 * The standard radioactive-decay model.
 */
export function exponentialDecay(elapsed: number, halfLife: number): number {
    return Math.pow(2.0, -elapsed / halfLife);
}

/**
 * Linear decay: λ = max(0, 1 − t/(2τ)).
 * Reaches zero at t = 2·half_life.
 */
export function linearDecay(elapsed: number, halfLife: number): number {
    return Math.max(0.0, 1.0 - elapsed / (2.0 * halfLife));
}

/**
 * Step decay: λ = 1 if t < τ, else 0.
 * Binary freshness model.
 */
export function stepDecay(elapsed: number, halfLife: number): number {
    return elapsed < halfLife ? 1.0 : 0.0;
}

// ── Core decay operator ───────────────────────────────────────────

/**
 * Apply temporal decay to an opinion.
 *
 * @param opinion The opinion to decay.
 * @param elapsed Time elapsed since the opinion was formed.
 * @param halfLife Time for belief and disbelief to halve.
 * @param decayFn Custom decay function (default: exponential).
 * @returns New Opinion with decayed belief/disbelief and increased uncertainty.
 */
export function decayOpinion(
    opinion: Opinion,
    elapsed: number,
    halfLife: number,
    decayFn: DecayFunction = exponentialDecay
): Opinion {
    if (elapsed < 0) {
        throw new Error(`elapsed must be non-negative, got: ${elapsed}`);
    }
    if (halfLife <= 0) {
        throw new Error(`halfLife must be positive, got: ${halfLife}`);
    }

    const factor = decayFn(elapsed, halfLife);

    if (factor < 0.0 || factor > 1.0) {
        throw new Error(`decay factor must be in [0, 1], got: ${factor}`);
    }

    const newB = factor * opinion.belief;
    const newD = factor * opinion.disbelief;
    let newU = 1.0 - newB - newD;

    // Clamp to handle floating-point artifacts
    if (newU < 0.0) newU = 0.0;
    if (newU > 1.0) newU = 1.0; // Should not happen given factor <= 1, but safe

    return new Opinion(newB, newD, newU, opinion.baseRate);
}
