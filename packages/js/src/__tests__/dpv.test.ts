
import { describe, it, expect } from '@jest/globals';
import { toDpv, fromDpv, DPV, EU_GDPR, DPV_PD } from '../dpv.js';

describe('DPV Interoperability', () => {
    const aliceId = 'https://example.com/users/alice';

    describe('toDpv', () => {
        it('should convert jsonld-ex annotations to DPV', () => {
            const doc = {
                '@context': { '@vocab': 'https://example.com/' },
                '@id': aliceId,
                'name': {
                    '@value': 'Alice',
                    '@personalDataCategory': 'regular',
                    '@legalBasis': 'consent',
                    '@dataSubject': aliceId
                }
            };

            const { dpvDoc, report } = toDpv(doc);

            expect(report.success).toBe(true);
            expect(report.nodesConverted).toBeGreaterThan(0);

            // Check for DPV nodes in the output graph
            const graph = dpvDoc['@graph'];
            expect(graph).toBeDefined();

            const handlingNode = graph.find((n: any) => n['@type'] === `${DPV}PersonalDataHandling`);
            expect(handlingNode).toBeDefined();

            // Verify mappings
            expect(handlingNode['dpv:hasSourceNode']['@id']).toBe(aliceId);
            expect(handlingNode['dpv:hasSourceProperty']).toBe('name');

            // Personal Data Category
            expect(handlingNode['dpv:hasPersonalData']['@type']).toContain('PersonalData');

            // Legal Basis
            expect(handlingNode['dpv:hasLegalBasis']['@id']).toContain('A6-1-a'); // consent

            // Data Subject
            expect(handlingNode['dpv:hasDataSubject']['@id']).toBe(aliceId);
        });

        it('should handle consent records', () => {
            const doc = {
                '@id': aliceId,
                'email': {
                    '@value': 'alice@example.com',
                    '@consent': {
                        '@consentGivenAt': '2024-01-01T00:00:00Z',
                        '@consentScope': ['marketing']
                    }
                }
            };

            const { dpvDoc } = toDpv(doc);
            const graph = dpvDoc['@graph'];
            const handlingNode = graph.find((n: any) => n['@type'] === `${DPV}PersonalDataHandling`);

            const consentNode = handlingNode['dpv:hasConsent'];
            expect(consentNode).toBeDefined();
            expect(consentNode['@type']).toBe(`${DPV}Consent`);
            expect(consentNode['dpv:hasProvisionTime']).toBe('2024-01-01T00:00:00Z');
            expect(consentNode['dpv:hasScope']).toContain('marketing');
        });
    });

    describe('fromDpv', () => {
        it('should convert DPV graph back to jsonld-ex', () => {
            const dpvDoc = {
                '@graph': [
                    {
                        '@id': '_:handling1',
                        '@type': `${DPV}PersonalDataHandling`,
                        'dpv:hasSourceNode': { '@id': aliceId },
                        'dpv:hasSourceProperty': 'name',
                        'dpv:hasPersonalData': { '@type': `${DPV_PD}PersonalData` },
                        'dpv:hasLegalBasis': { '@id': `${EU_GDPR}A6-1-a` }
                    }
                ]
            };

            const { doc, report } = fromDpv(dpvDoc);

            expect(report.success).toBe(true);

            // If single node, doc is the node
            const node = doc;
            expect(node['@id']).toBe(aliceId);

            const nameProp = node['name'];
            expect(nameProp['@personalDataCategory']).toBe('regular');
            expect(nameProp['@legalBasis']).toBe('consent');
        });
    });
});
