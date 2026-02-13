
import { mergeGraphs } from '../merge.js';

describe('Merge Parity Tests', () => {

    describe('Conflict Resolution: weighted_vote', () => {
        // Scenario: 
        // Graph A: { @id: 1, prop: "A", @confidence: 0.9 }
        // Graph B: { @id: 1, prop: "A", @confidence: 0.7 } -> Same value, will combine
        // Graph C: { @id: 1, prop: "B", @confidence: 0.8 } -> Different value
        //
        // "A" group: noisy_or(0.9, 0.7) = 0.97
        // "B" group: 0.8
        // Winner: "A" with confidence 0.97

        it('should select value with highest combined support', () => {
            const g1 = { '@id': 'http://example/1', 'prop': { '@value': 'A', '@confidence': 0.9 } };
            const g2 = { '@id': 'http://example/1', 'prop': { '@value': 'A', '@confidence': 0.7 } };
            const g3 = { '@id': 'http://example/1', 'prop': { '@value': 'B', '@confidence': 0.8 } };

            const result = mergeGraphs([g1, g2, g3], 'weighted_vote');
            const merged = result.merged['@graph'][0];

            expect(merged.prop['@value']).toBe('A');
            expect(merged.prop['@confidence']).toBeCloseTo(0.97);
        });
    });

    describe('Combination: dempster_shafer', () => {
        // Scenario:
        // Graph A: { @id: 1, prop: "val", @confidence: 0.9 }
        // Graph B: { @id: 1, prop: "val", @confidence: 0.8 }
        //
        // DS Combination: 0.98 (calculated in previous test)

        it('should combine agreed values using Dempster-Shafer', () => {
            const g1 = { '@id': 'http://example/1', 'prop': { '@value': 'val', '@confidence': 0.9 } };
            const g2 = { '@id': 'http://example/1', 'prop': { '@value': 'val', '@confidence': 0.8 } };

            const result = mergeGraphs([g1, g2], 'highest', 'dempster_shafer');
            const merged = result.merged['@graph'][0];

            expect(merged.prop['@value']).toBe('val');
            expect(merged.prop['@confidence']).toBeCloseTo(0.98);
        });
    });

});
