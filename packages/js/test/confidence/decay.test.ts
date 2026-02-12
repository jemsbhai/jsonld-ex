
import { Opinion } from '../../src/confidence/algebra';
import {
    decayOpinion,
    exponentialDecay,
    linearDecay,
    stepDecay,
} from '../../src/confidence/decay';

describe('Confidence Decay', () => {
    const fresh = new Opinion(0.8, 0.1, 0.1);

    describe('Decay Functions', () => {
        it('exponentialDecay', () => {
            // t = half_life -> factor = 2^-1 = 0.5
            expect(exponentialDecay(10, 10)).toBe(0.5);
            // t = 0 -> factor = 1.0
            expect(exponentialDecay(0, 10)).toBe(1.0);
        });

        it('linearDecay', () => {
            // t = half_life -> factor = 1 - 0.5 = 0.5
            expect(linearDecay(10, 10)).toBe(0.5);
            // t = 2 * half_life -> factor = 0
            expect(linearDecay(20, 10)).toBe(0.0);
        });

        it('stepDecay', () => {
            // t < half_life -> 1.0
            expect(stepDecay(9, 10)).toBe(1.0);
            // t >= half_life -> 0.0
            expect(stepDecay(10, 10)).toBe(0.0);
        });
    });

    describe('decayOpinion', () => {
        it('decays belief and disbelief proportionally', () => {
            // factor 0.5
            const decayed = decayOpinion(fresh, 10, 10, () => 0.5);

            expect(decayed.belief).toBe(0.4); // 0.8 * 0.5
            expect(decayed.disbelief).toBe(0.05); // 0.1 * 0.5
            // uncertainty = 1 - 0.4 - 0.05 = 0.55
            expect(decayed.uncertainty).toBeCloseTo(0.55);
        });

        it('preserves baseRate', () => {
            const decayed = decayOpinion(fresh, 100, 10);
            expect(decayed.baseRate).toBe(fresh.baseRate);
        });

        it('clamps elapsed time', () => {
            expect(() => decayOpinion(fresh, -1, 10)).toThrow();
        });
    });
});
