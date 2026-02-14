
import { describe, it, expect } from '@jest/globals';
import {
    contextDiff,
    checkCompatibility,
    ContextDiff,
    CompatibilityResult
} from '../context.js';

describe('Context Versioning', () => {
    describe('contextDiff', () => {
        it('should detect added terms', () => {
            const oldCtx = { 'name': 'http://schema.org/name' };
            const newCtx = {
                'name': 'http://schema.org/name',
                'email': 'http://schema.org/email'
            };

            const diff = contextDiff(oldCtx, newCtx);
            expect(Object.keys(diff.added)).toContain('email');
            expect(Object.keys(diff.removed)).toHaveLength(0);
            expect(Object.keys(diff.changed)).toHaveLength(0);
        });

        it('should detect removed terms', () => {
            const oldCtx = {
                'name': 'http://schema.org/name',
                'email': 'http://schema.org/email'
            };
            const newCtx = { 'name': 'http://schema.org/name' };

            const diff = contextDiff(oldCtx, newCtx);
            expect(Object.keys(diff.removed)).toContain('email');
            expect(Object.keys(diff.added)).toHaveLength(0);
        });

        it('should detect changed terms', () => {
            const oldCtx = { 'name': 'http://schema.org/name' };
            const newCtx = { 'name': 'http://foaf.org/name' };

            const diff = contextDiff(oldCtx, newCtx);
            expect(Object.keys(diff.changed)).toContain('name');
            expect(diff.changed['name'].oldValue).toBe('http://schema.org/name');
            expect(diff.changed['name'].newValue).toBe('http://foaf.org/name');
        });

        it('should handle context version property', () => {
            const oldCtx = { '@contextVersion': '1.0' };
            const newCtx = { '@contextVersion': '1.1' };

            const diff = contextDiff(oldCtx, newCtx);
            expect(diff.oldVersion).toBe('1.0');
            expect(diff.newVersion).toBe('1.1');
            // Should not be listed in changes/added/removed
            expect(Object.keys(diff.changed)).toHaveLength(0);
        });
    });

    describe('checkCompatibility', () => {
        it('should report compatible for additions only', () => {
            const oldCtx = { 'name': 'http://schema.org/name' };
            const newCtx = {
                'name': 'http://schema.org/name',
                'email': 'http://schema.org/email'
            };

            const result = checkCompatibility(oldCtx, newCtx);
            expect(result.compatible).toBe(true);
            expect(result.breaking).toHaveLength(0);
            expect(result.nonBreaking).toHaveLength(1); // Added term
        });

        it('should report breaking for removals', () => {
            const oldCtx = { 'name': 'http://schema.org/name' };
            const newCtx = {};

            const result = checkCompatibility(oldCtx, newCtx);
            expect(result.compatible).toBe(false);
            expect(result.breaking.some((c: any) => c.changeType === 'removed')).toBe(true);
        });

        it('should report breaking for IRI changes', () => {
            const oldCtx = { 'name': 'http://schema.org/name' };
            const newCtx = { 'name': 'http://foaf.org/name' };

            const result = checkCompatibility(oldCtx, newCtx);
            expect(result.compatible).toBe(false);
            expect(result.breaking.some((c: any) => c.changeType === 'changed-mapping')).toBe(true);
        });

        it('should report breaking for @type coercion changes', () => {
            const oldCtx = {
                'age': { '@id': 'http://schema.org/age', '@type': 'xsd:integer' }
            };
            const newCtx = {
                'age': { '@id': 'http://schema.org/age', '@type': 'xsd:string' }
            };

            const result = checkCompatibility(oldCtx, newCtx);
            expect(result.compatible).toBe(false);
            expect(result.breaking.some((c: any) => c.changeType === 'changed-type')).toBe(true);
        });

        it('should report breaking for keyword changes', () => {
            const oldCtx = { '@vocab': 'http://schema.org/' };
            const newCtx = { '@vocab': 'http://foaf.org/' };

            const result = checkCompatibility(oldCtx, newCtx);
            expect(result.compatible).toBe(false);
            expect(result.breaking[0].term).toBe('@vocab');
        });
    });
});
