
import {
    Opinion,
    cumulativeFuse,
    averagingFuse,
    trustDiscount,
    pairwiseConflict,
    conflictMetric,
    robustFuse,
} from '../../src/confidence/algebra';

describe('Subjective Logic Algebra', () => {
    describe('Opinion', () => {
        it('creates valid opinions', () => {
            const o = new Opinion(0.8, 0.1, 0.1);
            expect(o.belief).toBe(0.8);
            expect(o.disbelief).toBe(0.1);
            expect(o.uncertainty).toBe(0.1);
            expect(o.baseRate).toBe(0.5);
        });

        it('throws on invalid components', () => {
            expect(() => new Opinion(1.1, 0, 0)).toThrow();
            expect(() => new Opinion(0.8, 0.1, 0.2)).toThrow(); // sum > 1
        });

        it('projects probability correctly', () => {
            const o = new Opinion(0.8, 0.1, 0.1, 0.5);
            expect(o.projectedProbability()).toBeCloseTo(0.85);
        });

        it('creates from confidence (scalar)', () => {
            const dogmatic = Opinion.fromConfidence(0.8);
            expect(dogmatic.belief).toBe(0.8);
            expect(dogmatic.uncertainty).toBe(0.0);

            const scaled = Opinion.fromConfidence(0.8, 0.5);
            expect(scaled.belief).toBeCloseTo(0.4);
            expect(scaled.uncertainty).toBe(0.5);
        });

        it('creates from evidence', () => {
            const o = Opinion.fromEvidence(8, 2);
            expect(o.belief).toBeCloseTo(8 / 12);
            expect(o.disbelief).toBeCloseTo(2 / 12);
        });

        it('serializes to JSON-LD', () => {
            const o = new Opinion(0.5, 0.3, 0.2);
            const json = o.toJsonLd();
            expect(json['@type']).toBe('Opinion');
            expect(json.belief).toBe(0.5);
        });

        it('deserializes from JSON-LD', () => {
            const data = { belief: 0.5, disbelief: 0.3, uncertainty: 0.2, baseRate: 0.6 };
            const o = Opinion.fromJsonLd(data);
            expect(o.belief).toBe(0.5);
            expect(o.baseRate).toBe(0.6);
        });
    });

    describe('Operators', () => {
        it('cumulativeFuse reduces uncertainty', () => {
            const o1 = new Opinion(0.4, 0.1, 0.5);
            const o2 = new Opinion(0.4, 0.1, 0.5);
            const fused = cumulativeFuse(o1, o2);
            // u = 0.25 / 0.75 = 1/3
            expect(fused.uncertainty).toBeCloseTo(1 / 3, 5);
            expect(fused.belief).toBeGreaterThan(0.4);
        });

        it('cumulativeFuse handles dogmatic opinions', () => {
            const o1 = new Opinion(1.0, 0.0, 0.0);
            const o2 = new Opinion(0.0, 1.0, 0.0);
            const fused = cumulativeFuse(o1, o2);
            expect(fused.belief).toBe(0.5);
            expect(fused.disbelief).toBe(0.5);
            expect(fused.uncertainty).toBe(0.0);
        });

        it('averagingFuse idempotence', () => {
            const o1 = new Opinion(0.4, 0.1, 0.5);
            const fused = averagingFuse(o1, o1);
            expect(fused.belief).toBeCloseTo(o1.belief);
        });

        it('trustDiscount propagates uncertainty', () => {
            const trust = new Opinion(0.5, 0.0, 0.5);
            const opinion = new Opinion(0.8, 0.0, 0.2);
            const res = trustDiscount(trust, opinion);
            expect(res.belief).toBeCloseTo(0.4);
            expect(res.uncertainty).toBeCloseTo(0.6);
        });
    });

    describe('Conflict', () => {
        it('pairwise conflict detects disagreement', () => {
            const o1 = new Opinion(0.9, 0.0, 0.1);
            const o2 = new Opinion(0.0, 0.9, 0.1);
            // 0.81
            expect(pairwiseConflict(o1, o2)).toBeCloseTo(0.81);
        });

        it('robustFuse filters outliers', () => {
            const good1 = new Opinion(0.8, 0.1, 0.1);
            const good2 = new Opinion(0.75, 0.15, 0.1);
            const bad = new Opinion(0.0, 0.9, 0.1);

            const { fused, removedIndices } = robustFuse([good1, good2, bad], 0.2);

            expect(removedIndices).toContain(2);
            expect(removedIndices).not.toContain(0);
            expect(removedIndices).not.toContain(1);
            expect(fused.belief).toBeGreaterThan(0.7);
        });
    });
});
