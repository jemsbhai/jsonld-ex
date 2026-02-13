import { describe, it, expect } from '@jest/globals';
import {
    annotateProtection,
    createConsentRecord,
    isConsentActive,
    getProtectionMetadata,
    isPersonalData,
    isSensitiveData,
    filterByJurisdiction,
    filterPersonalData,
    LEGAL_BASES,
    PERSONAL_DATA_CATEGORIES
} from '../data-protection.js';

describe('Data Protection (GDPR) Extensions', () => {

    describe('annotateProtection', () => {
        it('should create an annotated value with valid metadata', () => {
            const result = annotateProtection('John Doe', {
                personalDataCategory: 'regular',
                legalBasis: 'consent',
                processingPurpose: 'user_profile',
                dataSubject: 'user:123',
                jurisdiction: 'EU',
            });

            expect(result).toEqual({
                '@value': 'John Doe',
                '@personalDataCategory': 'regular',
                '@legalBasis': 'consent',
                '@processingPurpose': 'user_profile',
                '@dataSubject': 'user:123',
                '@jurisdiction': 'EU',
            });
        });

        it('should throw error for invalid enum values', () => {
            expect(() => {
                annotateProtection('test', { personalDataCategory: 'invalid_category' });
            }).toThrow(/Invalid personal data category/);

            expect(() => {
                annotateProtection('test', { legalBasis: 'invalid_basis' });
            }).toThrow(/Invalid legal basis/);
        });
    });

    describe('Consent Lifecycle', () => {
        it('should create a valid consent record', () => {
            const record = createConsentRecord(
                '2023-01-01T12:00:00Z',
                ['marketing', 'analytics'],
                'specific'
            );

            expect(record).toEqual({
                '@consentGivenAt': '2023-01-01T12:00:00Z',
                '@consentScope': ['marketing', 'analytics'],
                '@consentGranularity': 'specific',
            });
        });

        it('should check if consent is active', () => {
            const record = createConsentRecord('2023-01-01T00:00:00Z', 'all');

            // Active if not withdrawn
            expect(isConsentActive(record)).toBe(true);

            // Active at a time after given
            expect(isConsentActive(record, '2023-02-01T00:00:00Z')).toBe(true);

            // Not active before given
            expect(isConsentActive(record, '2022-12-31T00:00:00Z')).toBe(false);
        });

        it('should handle withdrawn consent', () => {
            const record = createConsentRecord(
                '2023-01-01T00:00:00Z',
                'all',
                undefined,
                '2023-06-01T00:00:00Z'
            );

            // Not active currently (since it has a withdrawn date)
            expect(isConsentActive(record)).toBe(false);

            // Active between given and withdrawn
            expect(isConsentActive(record, '2023-03-01T00:00:00Z')).toBe(true);

            // Not active after withdrawn
            expect(isConsentActive(record, '2023-07-01T00:00:00Z')).toBe(false);
        });
    });

    describe('Classification', () => {
        it('should correctly classify personal data', () => {
            const regular = { '@value': 'John', '@personalDataCategory': 'regular' };
            const sensitive = { '@value': 'Health', '@personalDataCategory': 'sensitive' };
            const anon = { '@value': 'X', '@personalDataCategory': 'anonymized' };

            expect(isPersonalData(regular)).toBe(true);
            expect(isPersonalData(sensitive)).toBe(true);
            expect(isPersonalData(anon)).toBe(false);
        });

        it('should correctly classify sensitive data', () => {
            const sensitive = { '@value': 'Health', '@personalDataCategory': 'sensitive' };
            const biometrics = { '@value': 'Fingerprint', '@personalDataCategory': 'special_category' };
            const regular = { '@value': 'John', '@personalDataCategory': 'regular' };

            expect(isSensitiveData(sensitive)).toBe(true);
            expect(isSensitiveData(biometrics)).toBe(true);
            expect(isSensitiveData(regular)).toBe(false);
        });
    });

    describe('Graph Filtering', () => {
        const graph = [
            {
                '@id': 'user:1',
                'name': { '@value': 'Alice', '@personalDataCategory': 'regular', '@jurisdiction': 'EU' },
                'age': { '@value': 30, '@personalDataCategory': 'regular', '@jurisdiction': 'US' }
            },
            {
                '@id': 'user:2',
                'name': { '@value': 'Bob', '@personalDataCategory': 'anonymized', '@jurisdiction': 'EU' }
            }
        ];

        it('should filter by jurisdiction', () => {
            const euNodes = filterByJurisdiction(graph, 'name', 'EU');
            expect(euNodes).toHaveLength(2); // Alice and Bob both have EU names

            const usNodes = filterByJurisdiction(graph, 'age', 'US');
            expect(usNodes).toHaveLength(1); // Only Alice has US age
            expect(usNodes[0]['@id']).toBe('user:1');
        });

        it('should filter personal data nodes', () => {
            const personalNodes = filterPersonalData(graph);
            expect(personalNodes).toHaveLength(1); // Only Alice has personal data properties
            expect(personalNodes[0]['@id']).toBe('user:1');

            // Bob's name is 'anonymized', which is excluded from personal data
        });
    });
});
