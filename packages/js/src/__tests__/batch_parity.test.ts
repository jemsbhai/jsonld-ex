
import { annotateBatch, validateBatch, filterByConfidenceBatch } from '../batch.js';
import { ValidationResult } from '../types.js';

describe('Batch Parity Tests', () => {

    describe('annotateBatch', () => {
        it('should annotate a list of values', () => {
            const values = ['A', 'B'];
            const results = annotateBatch(values, { confidence: 0.9, source: 'model-v1' });

            expect(results).toHaveLength(2);
            expect(results[0]['@value']).toBe('A');
            expect(results[0]['@confidence']).toBe(0.9);
            expect(results[0]['@source']).toBe('model-v1');
            expect(results[1]['@value']).toBe('B');
            expect(results[1]['@confidence']).toBe(0.9);
        });

        it('should handle per-item overrides', () => {
            const items = [
                ['A', { confidence: 0.95 }],
                'B' // uses default
            ];
            const results = annotateBatch(items, { confidence: 0.5 });

            expect(results[0]['@confidence']).toBe(0.95);
            expect(results[1]['@confidence']).toBe(0.5);
        });
    });

    describe('validateBatch', () => {
        it('should validation a list of nodes', () => {
            const shape = {
                '@type': 'Person',
                'name': { '@required': true, '@type': 'xsd:string' }
            };
            const nodes = [
                { '@type': 'Person', 'name': 'Alice' },
                { '@type': 'Person' } // invalid
            ];

            const results = validateBatch(nodes, shape);
            expect(results).toHaveLength(2);
            expect(results[0].valid).toBe(true);
            expect(results[1].valid).toBe(false);
        });
    });

    describe('filterByConfidenceBatch', () => {
        it('should filter by single property', () => {
            const nodes = [
                { '@type': 'Thing', 'name': { '@value': 'A', '@confidence': 0.9 } },
                { '@type': 'Thing', 'name': { '@value': 'B', '@confidence': 0.3 } }
            ];

            const results = filterByConfidenceBatch(nodes, 'name', 0.5);
            expect(results).toHaveLength(1);
            expect(results[0]['name']['@value']).toBe('A');
        });

        it('should filter by multiple properties', () => {
            const nodes = [
                {
                    '@type': 'Thing',
                    'name': { '@value': 'A', '@confidence': 0.9 },
                    'desc': { '@value': 'good', '@confidence': 0.8 }
                },
                {
                    '@type': 'Thing',
                    'name': { '@value': 'B', '@confidence': 0.9 },
                    'desc': { '@value': 'bad', '@confidence': 0.2 }
                }
            ];

            const results = filterByConfidenceBatch(nodes, [['name', 0.5], ['desc', 0.5]]);
            expect(results).toHaveLength(1);
            expect(results[0]['name']['@value']).toBe('A');
        });
    });

});
