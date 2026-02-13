
import { propagateConfidence, combineSources, resolveConflict, propagateGraphConfidence, validateConfidence } from '../inference.js';

describe('Inference Parity Tests', () => {
    describe('propagateConfidence', () => {
        const chain = [0.9, 0.8];

        it('should support multiply (default)', () => {
            const res = propagateConfidence(chain, 'multiply');
            expect(res.score).toBeCloseTo(0.72);
        });

        it('should support min', () => {
            const res = propagateConfidence(chain, 'min');
            expect(res.score).toBe(0.8);
        });

        it('should support average', () => {
            const res = propagateConfidence(chain, 'average');
            expect(res.score).toBeCloseTo(0.85);
        });

        it('should support bayesian', () => {
            // log(0.9/0.1) + log(0.8/0.2) = 2.197 + 1.386 = 3.583
            // odds = exp(3.583) = 36
            // prob = 36/37 = 0.9729
            const res = propagateConfidence(chain, 'bayesian');
            expect(res.score).toBeGreaterThan(0.9);
            expect(res.score).toBeCloseTo(0.9729, 3);
        });

        it('should support dampened', () => {
            // (0.72)^(1/sqrt(2)) = 0.7927...
            const res = propagateConfidence(chain, 'dampened');
            expect(res.score).toBeCloseTo(0.7927, 3);
        });
    });

    describe('combineSources', () => {
        const scores = [0.9, 0.8];

        it('should support noisy_or (default)', () => {
            // 1 - (0.1 * 0.2) = 1 - 0.02 = 0.98
            expect(combineSources(scores, 'noisy_or')).toBeCloseTo(0.98);
        });

        it('should support max', () => {
            expect(combineSources(scores, 'max')).toBe(0.9);
        });

        it('should support average', () => {
            expect(combineSources(scores, 'average')).toBeCloseTo(0.85);
        });

        it('should support dempster_shafer', () => {
            // Source 1: b=0.9, u=0.1
            // Source 2: b=0.8, u=0.2
            // b_new = 0.9*0.8 + 0.9*0.2 + 0.1*0.8 = 0.72 + 0.18 + 0.08 = 0.98
            // u_new = 0.1*0.2 = 0.02
            // total = 1.0 (no conflict)
            expect(combineSources(scores, 'dempster_shafer')).toBeCloseTo(0.98);
        });

        it('should handle dempster_shafer conflict normalization', () => {
            // This implementation assumes NO disbelief, so conflict K is always 0 in the provided code.
            // (Python logic: m({F}) = 0 for all inputs)
            // So we just check basic combination.
            const s2 = [0.5, 0.5];
            // b1=0.5, u1=0.5. b2=0.5, u2=0.5
            // b = 0.25 + 0.25 + 0.25 = 0.75
            // u = 0.25
            expect(combineSources(s2, 'dempster_shafer')).toBeCloseTo(0.75);
        });
    });

    describe('resolveConflict', () => {
        const candidates = [
            { '@value': 'A', '@confidence': 0.9 },
            { '@value': 'B', '@confidence': 0.8 },
            { '@value': 'A', '@confidence': 0.7 } // 'A' has support from 0.9 and 0.7
        ];

        it('should support highest', () => {
            const res = resolveConflict(candidates, 'highest');
            expect(res.winner['@value']).toBe('A');
            expect(res.reason).toContain('Highest confidence');
            // Should pick the specific instance with 0.9
            expect(res.winner['@confidence']).toBe(0.9);
        });

        it('should support weighted_vote', () => {
            const res = resolveConflict(candidates, 'weighted_vote');
            // Group A: 0.9, 0.7 -> noisy_or(0.9, 0.7) = 1 - (0.1*0.3) = 0.97
            // Group B: 0.8
            // Winner should be A with confidence 0.97
            expect(res.winner['@value']).toBe('A');
            expect(res.winner['@confidence']).toBeCloseTo(0.97);
            expect(res.reason).toContain('source(s)');
        });

        it('should support recency', () => {
            const recencyCandidates = [
                { '@value': 'Old', '@confidence': 0.9, '@extractedAt': '2020-01-01T00:00:00Z' },
                { '@value': 'New', '@confidence': 0.8, '@extractedAt': '2025-01-01T00:00:00Z' }
            ];
            const res = resolveConflict(recencyCandidates, 'recency');
            expect(res.winner['@value']).toBe('New');
            expect(res.reason).toContain('Most recent');
        });
    });

    describe('propagateGraphConfidence', () => {
        const doc = {
            'step1': {
                '@value': 'v1',
                '@confidence': 0.8 // c1
            },
            'link': {
                'step2': {
                    '@value': 'v2',
                    '@confidence': 0.9 // c2
                }
            }
        };

        it('should traverse and propagate', () => {
            // path: link -> step2 is not direct.
            // doc structure: 
            // root -> link (node) -> step2 (annotated value)

            // Wait, propagateGraphConfidence takes propertyChain from root.
            // If we access doc['link'], we get the node.
            // Then from that node, we access 'step2'.
            // So chain = ['link', 'step2']

            // But wait, my implementation of propagateGraphConfidence:
            // 1. current = doc
            // 2. value = doc['link']. c = getConfidence(value). 
            // doc['link'] is { step2: ... }. It has no @confidence. So c=1.0.
            // current becomes doc['link'].
            // 3. value = current['step2']. It is { @value: 'v2', @confidence: 0.9 }.
            // c = 0.9.
            // Traversal stops (or continues if there were more steps).

            // So scores = [1.0, 0.9]. Product = 0.9.

            const res = propagateGraphConfidence(doc, ['link', 'step2'], 'multiply');
            expect(res.score).toBe(0.9);
            expect(res.inputScores).toEqual([1.0, 0.9]);
        });

        it('should handle explicit confidence on intermediate nodes', () => {
            const doc2 = {
                'intermediate': {
                    '@confidence': 0.8,
                    'leaf': {
                        '@value': 'val',
                        '@confidence': 0.9
                    }
                }
            };
            // Path: ['intermediate', 'leaf']
            // 1. doc['intermediate']. has @confidence 0.8. c=0.8.
            // 2. doc['intermediate']['leaf']. has @confidence 0.9. c=0.9.
            // Result = 0.8 * 0.9 = 0.72

            const res = propagateGraphConfidence(doc2, ['intermediate', 'leaf']);
            expect(res.score).toBeCloseTo(0.72);
        });
    });
});
