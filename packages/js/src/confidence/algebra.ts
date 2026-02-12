/**
 * Formal Confidence Algebra for JSON-LD-Ex.
 *
 * Grounds uncertainty representation and propagation in Jøsang's Subjective
 * Logic framework, providing a rigorous mathematical foundation for confidence
 * scores in AI/ML data exchange.
 */

// Tolerance for floating-point comparison
const ADDITIVITY_TOL = 1e-9;

function validateComponent(value: number, name: string): number {
    if (typeof value !== 'number') {
        throw new TypeError(`${name} must be a number`);
    }
    if (!Number.isFinite(value)) {
        throw new Error(`${name} must be finite`);
    }
    if (value < 0.0 || value > 1.0) {
        throw new Error(`${name} must be in [0, 1], got: ${value}`);
    }
    return value;
}

/**
 * A subjective opinion ω = (b, d, u, a) per Subjective Logic.
 *
 * Represents a nuanced belief state that distinguishes between
 * evidence for, evidence against, and absence of evidence.
 */
export class Opinion {
    public readonly belief: number;
    public readonly disbelief: number;
    public readonly uncertainty: number;
    public readonly baseRate: number;

    constructor(
        belief: number,
        disbelief: number,
        uncertainty: number,
        baseRate: number = 0.5
    ) {
        this.belief = validateComponent(belief, 'belief');
        this.disbelief = validateComponent(disbelief, 'disbelief');
        this.uncertainty = validateComponent(uncertainty, 'uncertainty');
        this.baseRate = validateComponent(baseRate, 'baseRate');

        const total = this.belief + this.disbelief + this.uncertainty;
        if (Math.abs(total - 1.0) > ADDITIVITY_TOL) {
            throw new Error(
                `belief + disbelief + uncertainty must sum to 1, got ${total}`
            );
        }
    }

    /**
     * Compute P(ω) = b + a·u.
     * Maps opinion to scalar probability.
     */
    public projectedProbability(): number {
        return this.belief + this.baseRate * this.uncertainty;
    }

    /** Alias for projectedProbability() to interop with scalar systems. */
    public toConfidence(): number {
        return this.projectedProbability();
    }

    /**
     * Create an Opinion from a scalar confidence score.
     *
     * @param confidence Scalar confidence in [0, 1].
     * @param uncertainty Fraction of mass assigned to uncertainty [0, 1].
     * @param baseRate Prior probability, default 0.5.
     */
    static fromConfidence(
        confidence: number,
        uncertainty: number = 0.0,
        baseRate: number = 0.5
    ): Opinion {
        validateComponent(confidence, 'confidence');
        const u = validateComponent(uncertainty, 'uncertainty');
        validateComponent(baseRate, 'baseRate');

        const remaining = 1.0 - u;
        const b = confidence * remaining;
        const d = (1.0 - confidence) * remaining;

        return new Opinion(b, d, u, baseRate);
    }

    /**
     * Create an Opinion from evidence counts.
     *
     * @param positive Count of positive observations.
     * @param negative Count of negative observations.
     * @param priorWeight Weight of non-informative prior (default 2.0).
     * @param baseRate Prior probability (default 0.5).
     */
    static fromEvidence(
        positive: number,
        negative: number,
        priorWeight: number = 2.0,
        baseRate: number = 0.5
    ): Opinion {
        if (positive < 0 || negative < 0) {
            throw new Error('Evidence counts must be non-negative');
        }
        if (priorWeight <= 0) {
            throw new Error('priorWeight must be positive');
        }

        const total = positive + negative + priorWeight;
        return new Opinion(
            positive / total,
            negative / total,
            priorWeight / total,
            baseRate
        );
    }

    static fromJsonLd(data: any): Opinion {
        return new Opinion(
            data.belief,
            data.disbelief,
            data.uncertainty,
            data.baseRate ?? 0.5
        );
    }

    toJsonLd(): any {
        return {
            '@type': 'Opinion',
            belief: this.belief,
            disbelief: this.disbelief,
            uncertainty: this.uncertainty,
            baseRate: this.baseRate
        };
    }

    toString(): string {
        return `Opinion(b=${this.belief.toFixed(4)}, d=${this.disbelief.toFixed(4)}, u=${this.uncertainty.toFixed(4)}, a=${this.baseRate.toFixed(4)})`;
    }
}

// ── Operators ─────────────────────────────────────────────────────

/**
 * Cumulative fusion (oplus) — combine independent evidence sources.
 */
export function cumulativeFuse(...opinions: Opinion[]): Opinion {
    if (opinions.length === 0) {
        throw new Error('cumulativeFuse requires at least one opinion');
    }
    if (opinions.length === 1) return opinions[0];

    let result = opinions[0];
    for (let i = 1; i < opinions.length; i++) {
        result = cumulativeFusePair(result, opinions[i]);
    }
    return result;
}

function cumulativeFusePair(a: Opinion, b: Opinion): Opinion {
    const uA = a.uncertainty;
    const uB = b.uncertainty;

    let fusedB: number, fusedD: number, fusedU: number;

    if (uA === 0.0 && uB === 0.0) {
        // Dogmatic case
        const gammaA = 0.5;
        const gammaB = 0.5;
        fusedB = gammaA * a.belief + gammaB * b.belief;
        fusedD = gammaA * a.disbelief + gammaB * b.disbelief;
        fusedU = 0.0;
    } else {
        // Standard case
        const kappa = uA + uB - uA * uB;
        fusedB = (a.belief * uB + b.belief * uA) / kappa;
        fusedD = (a.disbelief * uB + b.disbelief * uA) / kappa;
        fusedU = (uA * uB) / kappa;
    }

    const fusedA = (a.baseRate + b.baseRate) / 2.0;
    return new Opinion(fusedB, fusedD, fusedU, fusedA);
}

/**
 * Averaging fusion (oslash) — combine dependent/correlated sources.
 */
export function averagingFuse(...opinions: Opinion[]): Opinion {
    if (opinions.length === 0) {
        throw new Error('averagingFuse requires at least one opinion');
    }
    if (opinions.length === 1) return opinions[0];
    if (opinions.length === 2) return averagingFusePair(opinions[0], opinions[1]);

    return averagingFuseNary(opinions);
}

function averagingFusePair(a: Opinion, b: Opinion): Opinion {
    const uA = a.uncertainty;
    const uB = b.uncertainty;

    let fusedB: number, fusedD: number, fusedU: number;

    if (uA === 0.0 && uB === 0.0) {
        fusedB = (a.belief + b.belief) / 2.0;
        fusedD = (a.disbelief + b.disbelief) / 2.0;
        fusedU = 0.0;
    } else {
        const kappa = uA + uB;
        if (kappa === 0.0) {
            fusedB = (a.belief + b.belief) / 2.0;
            fusedD = (a.disbelief + b.disbelief) / 2.0;
            fusedU = 0.0;
        } else {
            fusedB = (a.belief * uB + b.belief * uA) / kappa;
            fusedD = (a.disbelief * uB + b.disbelief * uA) / kappa;
            fusedU = (2.0 * uA * uB) / kappa;
        }
    }

    const fusedA = (a.baseRate + b.baseRate) / 2.0;
    return new Opinion(fusedB, fusedD, fusedU, fusedA);
}

function averagingFuseNary(opinions: Opinion[]): Opinion {
    const n = opinions.length;
    const uncertainties = opinions.map((o) => o.uncertainty);

    // Compute product of all uncertainties
    const fullProduct = uncertainties.reduce((acc, u) => acc * u, 1.0);

    // Compute U_i = product / u_i
    const capitalU = uncertainties.map((u, i) => {
        if (u !== 0.0) {
            return fullProduct / u;
        } else {
            // product excluding i
            return uncertainties.reduce((acc, val, idx) => (idx !== i ? acc * val : acc), 1.0);
        }
    });

    const kappa = capitalU.reduce((acc, val) => acc + val, 0.0);

    let fusedB: number, fusedD: number, fusedU: number;

    if (kappa === 0.0) {
        // Dogmatic fallback
        const dogmatic = opinions.filter((o) => o.uncertainty === 0.0);
        const pool = dogmatic.length > 0 ? dogmatic : opinions;
        const z = pool.length;

        fusedB = pool.reduce((sum, o) => sum + o.belief, 0) / z;
        fusedD = pool.reduce((sum, o) => sum + o.disbelief, 0) / z;
        fusedU = 0.0;
    } else {
        fusedB = opinions.reduce((sum, o, i) => sum + o.belief * capitalU[i], 0) / kappa;
        fusedD = opinions.reduce((sum, o, i) => sum + o.disbelief * capitalU[i], 0) / kappa;
        fusedU = (n * fullProduct) / kappa;
    }

    const fusedA = opinions.reduce((sum, o) => sum + o.baseRate, 0) / n;

    return new Opinion(fusedB, fusedD, fusedU, fusedA);
}

/**
 * Trust discount (otimes) — propagate opinion through a trust chain.
 */
export function trustDiscount(trust: Opinion, opinion: Opinion): Opinion {
    const bTrust = trust.belief;
    const fusedB = bTrust * opinion.belief;
    const fusedD = bTrust * opinion.disbelief;
    const fusedU = trust.disbelief + trust.uncertainty + bTrust * opinion.uncertainty;

    return new Opinion(fusedB, fusedD, fusedU, opinion.baseRate);
}

/**
 * Deduction operator — conditional reasoning.
 */
export function deduce(
    opinionX: Opinion,
    opinionYGivenX: Opinion,
    opinionYGivenNotX: Opinion
): Opinion {
    const bX = opinionX.belief;
    const dX = opinionX.disbelief;
    const uX = opinionX.uncertainty;
    const aX = opinionX.baseRate;
    const aXBar = 1.0 - aX;

    const yx = opinionYGivenX;
    const ynx = opinionYGivenNotX;

    const bY = bX * yx.belief + dX * ynx.belief + uX * (aX * yx.belief + aXBar * ynx.belief);
    const dY = bX * yx.disbelief + dX * ynx.disbelief + uX * (aX * yx.disbelief + aXBar * ynx.disbelief);
    const uY = bX * yx.uncertainty + dX * ynx.uncertainty + uX * (aX * yx.uncertainty + aXBar * ynx.uncertainty);

    const pYGivenX = yx.projectedProbability();
    const pYGivenNotX = ynx.projectedProbability();
    const aY = aX * pYGivenX + aXBar * pYGivenNotX;

    return new Opinion(bY, dY, uY, aY);
}

// ── Conflict Detection ────────────────────────────────────────────

export function pairwiseConflict(opA: Opinion, opB: Opinion): number {
    return opA.belief * opB.disbelief + opA.disbelief * opB.belief;
}

export function conflictMetric(opinion: Opinion): number {
    const res = 1.0 - Math.abs(opinion.belief - opinion.disbelief) - opinion.uncertainty;
    return Math.max(0.0, res);
}

/**
 * Byzantine-resistant fusion via iterative conflict filtering.
 */
export function robustFuse(
    opinions: Opinion[],
    threshold: number = 0.15,
    maxRemovals?: number
): { fused: Opinion; removedIndices: number[] } {
    if (opinions.length === 0) {
        throw new Error('robustFuse requires at least one opinion');
    }
    if (opinions.length === 1) {
        return { fused: opinions[0], removedIndices: [] };
    }

    const maxR = maxRemovals !== undefined ? maxRemovals : Math.floor(opinions.length / 2);

    // Track original indices: [index, opinion]
    const indexed: Array<{ idx: number; op: Opinion }> = opinions.map((op, idx) => ({ idx, op }));
    const removed: number[] = [];

    for (let r = 0; r < maxR; r++) {
        const n = indexed.length;
        if (n <= 2) break;

        const discord = new Array(n).fill(0.0);
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const c = pairwiseConflict(indexed[i].op, indexed[j].op);
                discord[i] += c;
                discord[j] += c;
            }
        }

        // Normalize
        for (let i = 0; i < n; i++) {
            discord[i] /= (n - 1);
        }

        // Find worst
        let worstIdx = -1;
        let worstScore = -1.0;
        for (let i = 0; i < n; i++) {
            if (discord[i] > worstScore) {
                worstScore = discord[i];
                worstIdx = i;
            }
        }

        if (worstScore < threshold) break;

        // Remove
        removed.push(indexed[worstIdx].idx);
        indexed.splice(worstIdx, 1);
    }

    const remainingOpinions = indexed.map((item) => item.op);
    const fused = cumulativeFuse(...remainingOpinions);

    return { fused, removedIndices: removed };
}
