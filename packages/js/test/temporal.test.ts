import {
    addTemporal,
    queryAtTime,
    temporalDiff,
    TEMPORAL_CONTEXT,
} from '../src/temporal';

describe('Temporal Extensions', () => {
    const t1 = '2023-01-01T00:00:00Z';
    const t2 = '2023-06-01T00:00:00Z';
    const t3 = '2024-01-01T00:00:00Z';

    describe('addTemporal()', () => {
        it('adds temporal qualifiers to a value', () => {
            const result = addTemporal('test', { validFrom: t1, validUntil: t3 });
            expect(result).toEqual({
                '@value': 'test',
                '@validFrom': t1,
                '@validUntil': t3,
            });
        });

        it('preserves existing object structure', () => {
            const input = { '@value': 'test', '@type': 'xsd:string' };
            const result = addTemporal(input, { asOf: t2 });
            expect(result).toEqual({
                '@value': 'test',
                '@type': 'xsd:string',
                '@asOf': t2,
            });
        });

        it('throws if no qualifiers provided', () => {
            expect(() => addTemporal('test', {})).toThrow();
        });

        it('throws if validFrom > validUntil', () => {
            expect(() =>
                addTemporal('test', { validFrom: t3, validUntil: t1 })
            ).toThrow();
        });
    });

    describe('queryAtTime()', () => {
        const graph = [
            {
                '@id': 'http://example.org/Alice',
                'jobTitle': [
                    {
                        '@value': 'Junior Dev',
                        '@validFrom': t1,
                        '@validUntil': t2,
                    },
                    {
                        '@value': 'Senior Dev',
                        '@validFrom': t2,
                    },
                ],
                'name': 'Alice', // Always valid
            },
        ];

        it('returns state at t1 (Junior Dev)', () => {
            // Query slightly after t1 to be safe, or exactly at t1
            // Implementation uses < and > so bounds are inclusive?
            // ts < fromDt -> false. So if ts == fromDt, it returns true. Inclusive start.
            // ts > untilDt -> false. So if ts == untilDt, it returns true. Inclusive end?
            // Let's check logic:
            // if (ts < fromDt) return false;
            // if (ts > untilDt) return false;
            // So [validFrom, validUntil] inclusive.

            const result = queryAtTime(graph, t1);
            expect(result).toHaveLength(1);
            expect(result[0]['jobTitle']['@value']).toBe('Junior Dev');
            expect(result[0]['name']).toBe('Alice');
        });

        it('returns state at t3 (Senior Dev)', () => {
            const result = queryAtTime(graph, t3);
            expect(result).toHaveLength(1);
            expect(result[0]['jobTitle']['@value']).toBe('Senior Dev');
        });

        it('filters out properties invalid at time', () => {
            // At a time before t1, Alice has no jobTitle
            const tPre = '2022-01-01T00:00:00Z';
            const result = queryAtTime(graph, tPre);
            expect(result).toHaveLength(1);
            expect(result[0]['jobTitle']).toBeUndefined();
            expect(result[0]['name']).toBe('Alice');
        });
    });

    describe('temporalDiff()', () => {
        const graph = [
            {
                '@id': 'http://example.org/Bob',
                'status': [
                    { '@value': 'Active', '@validFrom': t1, '@validUntil': t2 },
                    { '@value': 'Inactive', '@validFrom': t2 },
                ],
            },
        ];

        it('detects modifications', () => {
            // Compare t1 (Active) vs t3 (Inactive)
            const diff = temporalDiff(graph, t1, t3);
            expect(diff.modified).toHaveLength(1);
            expect(diff.modified[0].nodeId).toBe('http://example.org/Bob');
            expect(diff.modified[0].changes['status'].from).toBe('Active');
            expect(diff.modified[0].changes['status'].to).toBe('Inactive');
        });

        it('detects property removal', () => {
            // Compare t1 (Active) vs tBefore (undefined)
            // Wait, if I compare t1 vs tPre, Bob exists at t1 but might not exist at tPre?
            // Bob has no properties at tPre except ID?
            // Let's assume Bob exists always but property status is only validFrom t1.

            const tPre = '2022-01-01T00:00:00Z';

            // properties at t1: status=Active
            // properties at tPre: {}
            // Diff t1 -> tPre: status removed? No, t1 is start, tPre is end.
            // Diff tPre -> t1: status added.

            const diffAdded = temporalDiff(graph, tPre, t1);
            // Bob exists in both snapshots (because queryAtTime returns the node if it has ID, even if props are filtered?
            // queryAtTime implementation:
            // if (!hasAnyData) return null;
            // BUT hasAnyData checks if any property survived.
            // If Bob has only @id, and identity keys pass through... 
            // Logic: if key is @id, out[key] = value. distinct from hasAnyData?
            // "Identity keys always pass through"
            // But `hasAnyData` is initially false.
            // If the loop finishes and only @id is present, hasAnyData is false (unless @id sets it? No).
            // Identity keys do NOT set hasAnyData=true in my implementation?
            // Let's check implementation.
            // if (key === '@id' ...) { out[key] = value; continue; }
            // So they don't set hasAnyData.
            // So if only @id remains, return null.
            // So Bob disappears at tPre!

            expect(diffAdded.added).toHaveLength(1);
            expect(diffAdded.added[0]['@id']).toBe('http://example.org/Bob');
        });
    });
});
