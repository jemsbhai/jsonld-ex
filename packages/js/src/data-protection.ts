import { DataProtectionMetadata, ConsentRecord, JsonLdNode } from './types.js';

// ── Constants ──────────────────────────────────────────────────────

export const LEGAL_BASES = [
    'consent',
    'contract',
    'legal_obligation',
    'vital_interest',
    'public_task',
    'legitimate_interest',
] as const;

export const PERSONAL_DATA_CATEGORIES = [
    'regular',
    'sensitive',
    'special_category',
    'anonymized',
    'pseudonymized',
    'synthetic',
    'non_personal',
] as const;

// Categories that count as personal data under GDPR.
// Anonymized, synthetic, and non_personal are excluded.
const _PERSONAL_CATEGORIES = new Set([
    'regular',
    'sensitive',
    'special_category',
    'pseudonymized',
]);

const _SENSITIVE_CATEGORIES = new Set([
    'sensitive',
    'special_category',
]);

export const CONSENT_GRANULARITIES = [
    'broad',
    'specific',
    'granular',
] as const;

export const ACCESS_LEVELS = [
    'public',
    'internal',
    'restricted',
    'confidential',
    'secret',
] as const;

// ── Core Annotation Function ─────────────────────────────────────

/**
 * Create a JSON-LD value annotated with data protection metadata.
 *
 * Produces a dict with `@value` plus any specified protection fields.
 * The output is compatible with `ai_ml.annotate()` — both can be
 * merged to combine provenance and data protection metadata on a single node.
 */
export function annotateProtection(
    value: any,
    options: DataProtectionMetadata
): JsonLdNode {
    const {
        personalDataCategory,
        legalBasis,
        processingPurpose,
        dataController,
        dataProcessor,
        dataSubject,
        retentionUntil,
        jurisdiction,
        accessLevel,
        consent,
    } = options;

    // Validate enum fields
    if (personalDataCategory && !PERSONAL_DATA_CATEGORIES.includes(personalDataCategory as any)) {
        throw new Error(
            `Invalid personal data category: '${personalDataCategory}'. Must be one of: ${PERSONAL_DATA_CATEGORIES.join(', ')}`
        );
    }
    if (legalBasis && !LEGAL_BASES.includes(legalBasis as any)) {
        throw new Error(
            `Invalid legal basis: '${legalBasis}'. Must be one of: ${LEGAL_BASES.join(', ')}`
        );
    }
    if (accessLevel && !ACCESS_LEVELS.includes(accessLevel as any)) {
        throw new Error(
            `Invalid access level: '${accessLevel}'. Must be one of: ${ACCESS_LEVELS.join(', ')}`
        );
    }

    const result: JsonLdNode = { '@value': value };

    if (personalDataCategory) result['@personalDataCategory'] = personalDataCategory;
    if (legalBasis) result['@legalBasis'] = legalBasis;
    if (processingPurpose) result['@processingPurpose'] = processingPurpose;
    if (dataController) result['@dataController'] = dataController;
    if (dataProcessor) result['@dataProcessor'] = dataProcessor;
    if (dataSubject) result['@dataSubject'] = dataSubject;
    if (retentionUntil) result['@retentionUntil'] = retentionUntil;
    if (jurisdiction) result['@jurisdiction'] = jurisdiction;
    if (accessLevel) result['@accessLevel'] = accessLevel;
    if (consent) result['@consent'] = consent;

    return result;
}

// ── Consent Lifecycle ─────────────────────────────────────────────

/**
 * Create a structured consent record.
 */
export function createConsentRecord(
    givenAt: string,
    scope: string | string[],
    granularity?: string,
    withdrawnAt?: string
): ConsentRecord {
    if (granularity && !CONSENT_GRANULARITIES.includes(granularity as any)) {
        throw new Error(
            `Invalid consent granularity: '${granularity}'. Must be one of: ${CONSENT_GRANULARITIES.join(', ')}`
        );
    }

    const scopeList = Array.isArray(scope) ? scope : [scope];
    if (scopeList.length === 0) {
        throw new Error('Consent scope must not be empty');
    }

    const record: ConsentRecord = {
        '@consentGivenAt': givenAt,
        '@consentScope': scopeList,
    };

    if (granularity) record['@consentGranularity'] = granularity;
    if (withdrawnAt) record['@consentWithdrawnAt'] = withdrawnAt;

    return record;
}

/**
 * Check whether a consent record is active.
 *
 * @param atTime Optional ISO 8601 datetime to check status at a specific point.
 *               If undefined, checks current status (active = not withdrawn).
 */
export function isConsentActive(
    record: ConsentRecord | undefined | null,
    atTime?: string
): boolean {
    if (!record || typeof record !== 'object') {
        return false;
    }

    const givenAt = record['@consentGivenAt'];
    if (!givenAt) {
        return false;
    }

    const withdrawnAt = record['@consentWithdrawnAt'];

    if (atTime) {
        const checkTime = new Date(atTime).getTime();
        const givenTime = new Date(givenAt).getTime();

        // Not yet given at the check time
        if (checkTime < givenTime) {
            return false;
        }

        // Withdrawn before the check time
        if (withdrawnAt) {
            const withdrawnTime = new Date(withdrawnAt).getTime();
            if (checkTime >= withdrawnTime) {
                return false;
            }
        }

        return true;
    }

    // No specific time — just check if withdrawn
    return !withdrawnAt;
}

// ── Extraction ────────────────────────────────────────────────────

/**
 * Extract data protection metadata from a JSON-LD node.
 */
export function getProtectionMetadata(node: JsonLdNode): DataProtectionMetadata {
    if (!node || typeof node !== 'object') {
        return {};
    }

    return {
        personalDataCategory: node['@personalDataCategory'],
        legalBasis: node['@legalBasis'],
        processingPurpose: node['@processingPurpose'],
        dataController: node['@dataController'],
        dataProcessor: node['@dataProcessor'],
        dataSubject: node['@dataSubject'],
        retentionUntil: node['@retentionUntil'],
        jurisdiction: node['@jurisdiction'],
        accessLevel: node['@accessLevel'],
        consent: node['@consent'],
    };
}

// ── Classification Helpers ────────────────────────────────────────

/**
 * Check if a node is classified as personal data.
 */
export function isPersonalData(node: JsonLdNode): boolean {
    if (!node || typeof node !== 'object') {
        return false;
    }
    const category = node['@personalDataCategory'];
    return typeof category === 'string' && _PERSONAL_CATEGORIES.has(category);
}

/**
 * Check if a node is classified as sensitive or special category data.
 */
export function isSensitiveData(node: JsonLdNode): boolean {
    if (!node || typeof node !== 'object') {
        return false;
    }
    const category = node['@personalDataCategory'];
    return typeof category === 'string' && _SENSITIVE_CATEGORIES.has(category);
}

// ── Graph Filtering ───────────────────────────────────────────────

/**
 * Filter graph nodes where a property has a specific jurisdiction.
 */
export function filterByJurisdiction(
    graph: JsonLdNode[],
    propertyName: string,
    jurisdiction: string
): JsonLdNode[] {
    const results: JsonLdNode[] = [];

    for (const node of graph) {
        const prop = node[propertyName];
        if (prop === undefined || prop === null) {
            continue;
        }

        const values = Array.isArray(prop) ? prop : [prop];
        for (const v of values) {
            if (typeof v === 'object' && v !== null && v['@jurisdiction'] === jurisdiction) {
                results.push(node);
                break;
            }
        }
    }
    return results;
}

/**
 * Filter graph nodes that contain personal data.
 *
 * @param propertyName If specified, only check this property.
 *                     If undefined, check all properties on each node.
 */
export function filterPersonalData(
    graph: JsonLdNode[],
    propertyName?: string
): JsonLdNode[] {
    const results: JsonLdNode[] = [];

    for (const node of graph) {
        if (propertyName) {
            const prop = node[propertyName];
            if (prop === undefined || prop === null) {
                continue;
            }
            const values = Array.isArray(prop) ? prop : [prop];
            if (values.some((v: any) => isPersonalData(v))) {
                results.push(node);
            }
        } else {
            // Check all properties
            let found = false;
            for (const key of Object.keys(node)) {
                if (key.startsWith('@')) {
                    continue;
                }
                const prop = node[key];
                const values = Array.isArray(prop) ? prop : [prop];
                if (values.some((v: any) => isPersonalData(v))) {
                    found = true;
                    break;
                }
            }
            if (found) {
                results.push(node);
            }
        }
    }
    return results;
}
