
import {
    combineOpinionsFromScalars,
    propagateOpinionsFromScalars,
} from '../../src/confidence/bridge';
import { Opinion } from '../../src/confidence/algebra';

describe('Confidence Bridge', () => {
    describe('combineOpinionsFromScalars', () => {
        it('cumulative + zero uncertainty -> maps to opinion', () => {
            // p=0.8 -> b=0.8, d=0, u=0.2
            const res = combineOpinionsFromScalars([0.8], 0.0, 'cumulative');
            expect(res.belief).toBe(0.8);
            expect(res.uncertainty).toBeCloseTo(0.2);
        });

        it('fuses multiple scores using cumulative fusion', () => {
            // 0.8 and 0.8
            // o1 = (0.8, 0, 0.2)
            // o2 = (0.8, 0, 0.2)
            // kappa = 0.2+0.2 - 0.04 = 0.36
            // u = 0.04 / 0.36 = 1/9 ≈ 0.111
            const res = combineOpinionsFromScalars([0.8, 0.8], 0.0, 'cumulative');
            expect(res.uncertainty).toBeCloseTo(1 / 9);
            expect(res.belief).toBeGreaterThan(0.8);
        });

        it('averaging fusion (dogmatic)', () => {
            const res = combineOpinionsFromScalars([0.8, 0.8], 0.0, 'averaging');
            expect(res.belief).toBeCloseTo(0.8);
            expect(res.uncertainty).toBeCloseTo(0.0);
        });

        it('averaging fusion (explicit uncertainty)', () => {
            // p=0.8, u=0.2 -> b = 0.8 * (1-0.2) = 0.64
            const res = combineOpinionsFromScalars([0.8, 0.8], 0.2, 'averaging');
            expect(res.belief).toBeCloseTo(0.64);
            expect(res.uncertainty).toBeCloseTo(0.2);
        });
    });

    describe('propagateOpinionsFromScalars', () => {
        it('propagate chain (multiply equivalent when dogmatic)', () => {
            // trust=0.9, assertion=0.8
            // Equivalent to 0.9 * 0.8 = 0.72
            const res = propagateOpinionsFromScalars([0.9, 0.8]);
            expect(res.toConfidence()).toBeCloseTo(0.72);
        });

        it('handles uncertainty in trust', () => {
            // trust u=0.1
            const res = propagateOpinionsFromScalars([0.9, 0.8], 0.1);
            // Should result in higher uncertainty than dogmatic
            const dogmatic = propagateOpinionsFromScalars([0.9, 0.8], 0.0);
            expect(res.uncertainty).toBeGreaterThan(dogmatic.uncertainty);
        });
    });
});
