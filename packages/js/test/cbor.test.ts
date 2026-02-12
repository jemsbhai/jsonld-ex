
import { toCbor, fromCbor, payloadStats, DEFAULT_CONTEXT_REGISTRY } from '../src/cbor';
import { decode } from 'cbor-x';

describe('CBOR-LD Extensions', () => {
    const doc = {
        '@context': 'https://www.w3.org/ns/activitystreams',
        'type': 'Note',
        'content': 'Hello world',
    };

    describe('toCbor()', () => {
        it('serializes to CBOR buffer', () => {
            const buf = toCbor(doc);
            expect(Buffer.isBuffer(buf)).toBe(true);
            expect(buf.length).toBeGreaterThan(0);
        });

        it('compresses known contexts', () => {
            const buf = toCbor(doc);
            const decoded = decode(buf);
            // 'https://www.w3.org/ns/activitystreams' maps to 2 in DEFAULT_CONTEXT_REGISTRY
            expect(decoded['@context']).toBe(2);
        });

        it('handles arrays of contexts', () => {
            const docArr = {
                '@context': ['https://schema.org/', 'http://example.org/ctx'],
                'name': 'Test',
            };
            const buf = toCbor(docArr);
            const decoded = decode(buf);
            expect(decoded['@context']).toEqual([1, 'http://example.org/ctx']);
        });
    });

    describe('fromCbor()', () => {
        it('deserializes and restores contexts', () => {
            const buf = toCbor(doc);
            const restored = fromCbor(buf);
            expect(restored['@context']).toBe('https://www.w3.org/ns/activitystreams');
            expect(restored['type']).toBe('Note');
        });

        it('round-trips complex documents', () => {
            const complex = {
                '@context': 'https://schema.org/',
                '@id': 'http://example.org/1',
                'name': 'Node 1',
                'knows': {
                    '@id': 'http://example.org/2',
                    'name': 'Node 2',
                },
            };
            const buf = toCbor(complex);
            const restored = fromCbor(buf);
            expect(restored).toEqual(complex);
        });
    });

    describe('payloadStats()', () => {
        it('computes stats', () => {
            const stats = payloadStats(doc);
            expect(stats.jsonBytes).toBeGreaterThan(0);
            expect(stats.cborBytes).toBeGreaterThan(0);
            expect(stats.cborRatio).toBeLessThan(1.0); // Should be smaller
        });
    });
});
