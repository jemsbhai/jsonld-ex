
import client from '../src/client';
import { JsonLdExClient } from '../src/client';

describe('JsonLdExClient', () => {
    it('is exported as default and named export', () => {
        expect(client).toBeInstanceOf(JsonLdExClient);
    });

    describe('annotate', () => {
        it('validates metadata', () => {
            expect(() => {
                client.annotate('foo', { confidence: 1.5 });
            }).toThrow(); // Zod error
        });

        it('creates annotated value', () => {
            const res = client.annotate('foo', { confidence: 0.9, source: 'http://example.org/model' });
            expect(res['@value']).toBe('foo');
            expect(res['@confidence']).toBe(0.9);
            expect(res['@source']).toBe('http://example.org/model');
        });
    });

    describe('merge', () => {
        const g1 = { '@id': 'http://a', 'p': 'v1' };
        const g2 = { '@id': 'http://a', 'p': 'v2' };

        it('merges graphs', () => {
            const { merged, report } = client.merge([g1, g2]);
            expect(merged['@graph']).toBeDefined();
            expect(report.sourceCount).toBe(2);
        });
    });

    describe('diff', () => {
        const g1 = { '@id': 'http://a', 'p': 'v1' };
        const g2 = { '@id': 'http://a', 'p': 'v2' };

        it('diffs graphs', () => {
            const diff = client.diff(g1, g2);
            expect(diff.modified).toHaveLength(1);
        });
    });

    describe('propagate', () => {
        it('propagates confidence', () => {
            const res = client.propagate([0.8, 0.5], 'multiply');
            expect(res.score).toBeCloseTo(0.4);
        });
    });

    describe('addTemporal', () => {
        it('validates options', () => {
            expect(() => {
                client.addTemporal('foo', { validFrom: 'invalid-date' });
            }).toThrow();
        });

        it('adds temporal qualifiers', () => {
            const now = new Date().toISOString();
            const res = client.addTemporal('foo', { validFrom: now });
            expect(res['@value']).toBe('foo');
            expect(res['@validFrom']).toBe(now); // Keyword usage check
        });
    });
});
