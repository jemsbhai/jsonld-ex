
import {
    propagateConfidence,
    combineSources,
    resolveConflict,
} from '../src/inference';

describe('Inference Engine (Scalar)', () => {
    describe('propagateConfidence', () => {
        it('multiply', () => {
            const res = propagateConfidence([0.9, 0.8], 'multiply');
            expect(res.score).toBeCloseTo(0.72);
        });

        it('min', () => {
            const res = propagateConfidence([0.9, 0.8], 'min');
            expect(res.score).toBe(0.8);
        });

        it('average', () => {
            const res = propagateConfidence([0.9, 0.8], 'average');
            expect(res.score).toBeCloseTo(0.85);
        });
    });

    describe('combineSources', () => {
        it('noisy_or', () => {
            // 1 - (1-0.9)*(1-0.8) = 1 - 0.1*0.2 = 1 - 0.02 = 0.98
            const res = combineSources([0.9, 0.8], 'noisy_or');
            expect(res).toBeCloseTo(0.98);
        });

        it('max', () => {
            const res = combineSources([0.9, 0.8], 'max');
            expect(res).toBe(0.9);
        });
    });

    describe('resolveConflict', () => {
        it('highest strategy', () => {
            const c1 = { '@value': 'A', '@confidence': 0.9 };
            const c2 = { '@value': 'B', '@confidence': 0.8 };
            const report = resolveConflict([c1, c2], 'highest');
            expect(report.winner['@value']).toBe('A');
            expect(report.confidenceScores).toEqual([0.9, 0.8]);
        });

        it('recency strategy', () => {
            const c1 = { '@value': 'Old', '@extractedAt': '2023-01-01' };
            const c2 = { '@value': 'New', '@extractedAt': '2024-01-01' };
            const report = resolveConflict([c1, c2], 'recency');
            expect(report.winner['@value']).toBe('New');
        });
    });
});
