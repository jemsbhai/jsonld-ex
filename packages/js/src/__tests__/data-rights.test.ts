
import { describe, it, expect } from '@jest/globals';
import {
    requestErasure,
    executeErasure,
    requestRestriction,
    exportPortable,
    rectifyData,
    rightOfAccessReport,
    validateRetention,
    auditTrail
} from '../data-rights.js';
import { DataProtectionMetadata } from '../types.js';

describe('Data Rights (GDPR Articles 15-20)', () => {
    const aliceId = 'https://example.com/users/alice';

    const createGraph = () => [
        {
            '@id': aliceId,
            '@type': 'Person',
            'name': {
                '@value': 'Alice',
                '@dataSubject': aliceId,
                '@personalDataCategory': 'regular'
            },
            'email': {
                '@value': 'alice@example.com',
                '@dataSubject': aliceId,
                '@personalDataCategory': 'sensitive'
            }
        },
        {
            '@id': 'https://example.com/posts/1',
            'author': { '@id': aliceId },
            'title': { '@value': 'My Post', '@dataSubject': aliceId } // post title is personal data here
        }
    ];

    describe('Right to Erasure (Art. 17)', () => {
        it('requestErasure should mark properties for erasure', () => {
            const graph = createGraph() as any[];
            const plan = requestErasure(graph, {
                dataSubject: aliceId,
                requestedAt: '2024-01-01T00:00:00Z'
            });

            expect(plan.dataSubject).toBe(aliceId);
            expect(plan.affectedPropertyCount).toBe(3); // name, email, title

            // Check graph mutation
            const nameNode = graph.find(n => n['@id'] === aliceId)!['name'];
            expect(nameNode['@erasureRequested']).toBe(true);
            expect(nameNode['@erasureRequestedAt']).toBe('2024-01-01T00:00:00Z');
        });

        it('executeErasure should nullify values marked for erasure', () => {
            const graph = createGraph() as any[];
            // First request
            requestErasure(graph, { dataSubject: aliceId });

            // Then execute
            const audit = executeErasure(graph, {
                dataSubject: aliceId,
                completedAt: '2024-01-02T00:00:00Z'
            });

            expect(audit.erasedPropertyCount).toBe(3);

            const nameNode = graph.find(n => n['@id'] === aliceId)!['name'];
            expect(nameNode['@value']).toBeNull();
            expect(nameNode['@erasureCompletedAt']).toBe('2024-01-02T00:00:00Z');
        });
    });

    describe('Right to Restriction (Art. 18)', () => {
        it('requestRestriction should add restriction metadata', () => {
            const graph = createGraph() as any[];
            const result = requestRestriction(graph, {
                dataSubject: aliceId,
                reason: 'Disputed accuracy',
                processingRestrictions: ['marketing']
            });

            expect(result.restrictedPropertyCount).toBe(3);

            const emailNode = graph.find(n => n['@id'] === aliceId)!['email'];
            expect(emailNode['@restrictProcessing']).toBe(true);
            expect(emailNode['@restrictionReason']).toBe('Disputed accuracy');
            expect(emailNode['@processingRestrictions']).toEqual(['marketing']);
        });
    });

    describe('Right to Data Portability (Art. 20)', () => {
        it('exportPortable should extract data in requested format', () => {
            const graph = createGraph() as any[];
            const exportData = exportPortable(graph, {
                dataSubject: aliceId,
                format: 'json'
            });

            expect(exportData.format).toBe('json');
            expect(exportData.records).toHaveLength(2); // user node and post node

            const userRecord = exportData.records.find((r: any) => r.node_id === aliceId);
            expect(userRecord).toBeDefined();
            expect(userRecord?.properties['name']).toBe('Alice');
        });
    });

    describe('Right to Rectification (Art. 16)', () => {
        it('rectifyData should return a new corrected node', () => {
            const node = {
                '@value': 'Alic',
                '@dataSubject': aliceId
            } as any;

            const corrected = rectifyData(node, {
                newValue: 'Alice',
                note: 'Typo fix',
                rectifiedAt: '2024-01-01T00:00:00Z'
            });

            expect(corrected['@value']).toBe('Alice');
            expect(corrected['@rectificationNote']).toBe('Typo fix');
            expect(corrected['@rectifiedAt']).toBe('2024-01-01T00:00:00Z');

            // Original should be unchanged
            expect(node['@value']).toBe('Alic');
        });
    });

    describe('Right of Access (Art. 15)', () => {
        it('rightOfAccessReport should summarize held data', () => {
            const graph = createGraph() as any[];
            const report = rightOfAccessReport(graph, { dataSubject: aliceId });

            expect(report.dataSubject).toBe(aliceId);
            expect(report.totalPropertyCount).toBe(3);
            expect(report.categories).toContain('regular');
            expect(report.categories).toContain('sensitive');
            expect(report.records).toHaveLength(2);
        });
    });

    describe('Validation & Audit', () => {
        it('validateRetention should identify expired data', () => {
            const graph = [{
                '@id': 'node1',
                'data': {
                    '@value': 'old',
                    '@retentionUntil': '2023-01-01T00:00:00Z'
                }
            }] as any[];

            // Check after expiration
            const violations = validateRetention(graph, { asOf: '2024-01-01T00:00:00Z' });
            expect(violations).toHaveLength(1);
            expect(violations[0].nodeId).toBe('node1');

            // Check before expiration
            const ok = validateRetention(graph, { asOf: '2022-01-01T00:00:00Z' });
            expect(ok).toHaveLength(0);
        });

        it('auditTrail should capture current state', () => {
            const graph = createGraph() as any[];
            // Mark one item for erasure to see it in audit
            requestErasure(graph, { dataSubject: aliceId });

            const trail = auditTrail(graph, { dataSubject: aliceId });
            expect(trail).toHaveLength(3);

            const nameEntry = trail.find((e: any) => e.propertyName === 'name');
            expect(nameEntry?.erasureRequested).toBe(true);
            expect(nameEntry?.value).toBe('Alice');
        });
    });
});
