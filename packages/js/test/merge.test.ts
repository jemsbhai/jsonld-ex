
import { mergeGraphs, diffGraphs } from '../src/merge';

describe('Graph Merging', () => {
    const graphA = {
        '@context': { '@vocab': 'http://schema.org/' },
        '@graph': [
            {
                '@id': 'http://example.org/alice',
                '@type': 'Person',
                'name': 'Alice',
                'jobTitle': { '@value': 'Engineer', '@confidence': 0.9 }
            }
        ]
    };

    const graphB = {
        '@context': { '@vocab': 'http://schema.org/' },
        '@graph': [
            {
                '@id': 'http://example.org/alice',
                '@type': 'Person',
                'name': 'Alice W.',
                'jobTitle': { '@value': 'Senior Engineer', '@confidence': 0.8 }
            }
        ]
    };

    describe('mergeGraphs', () => {
        it('merges properties with highest confidence strategy', () => {
            const { merged, report } = mergeGraphs([graphA, graphB], 'highest');

            const alice = merged['@graph'].find((n: any) => n['@id'] === 'http://example.org/alice');
            expect(alice).toBeDefined();

            // jobTitle: 0.9 vs 0.8 -> 0.9 wins (Engineer)
            expect(alice['jobTitle']['@value']).toBe('Engineer');

            // name: no confidence? 
            // If no confidence, default 0.5.
            // Alice vs Alice W. -> simple string comparison if tied?
            // Or 'highest' strategy relies on stability?
            // My implementation of resolveConflict doesn't specify stability for ties.
            // Let's assume it picks the first one or stable sort.
        });

        it('reports conflicts', () => {
            const { report } = mergeGraphs([graphA, graphB], 'highest');
            expect(report.propertiesConflicted).toBeGreaterThan(0);
            // jobTitle conflict
            const conflict = report.conflicts.find(c => c.propertyName === 'jobTitle');
            expect(conflict).toBeDefined();
        });

        it('unions conflicting values', () => {
            const { merged } = mergeGraphs([graphA, graphB], 'union');
            const alice = merged['@graph'][0];
            expect(Array.isArray(alice['jobTitle'])).toBe(true);
            expect(alice['jobTitle']).toHaveLength(2);
        });
    });

    describe('diffGraphs', () => {
        it('detects modifications', () => {
            const diff = diffGraphs(graphA, graphB);
            expect(diff.modified.length).toBeGreaterThan(0);
            // jobTitle modified
            const mod = diff.modified.find(m => m.property === 'jobTitle');
            expect(mod).toBeDefined();
        });
    });
});
