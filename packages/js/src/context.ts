
import { JsonLdNode } from './types.js';

// ── Types ─────────────────────────────────────────────────────────

export interface TermChange {
    term: string;
    oldValue: any;
    newValue: any;
}

export interface ContextDiff {
    added: Record<string, any>;
    removed: Record<string, any>;
    changed: Record<string, TermChange>;
    oldVersion?: string;
    newVersion?: string;
}

export interface BreakingChange {
    term: string;
    changeType: string;
    detail: string;
}

export interface CompatibilityResult {
    compatible: boolean;
    breaking: BreakingChange[];
    nonBreaking: BreakingChange[];
}

// ── Helpers ───────────────────────────────────────────────────────

function _extractIri(value: any): string | null {
    if (typeof value === 'string') return value;
    if (typeof value === 'object' && value !== null) return value['@id'] || null;
    return null;
}

function _extractType(value: any): string | null {
    if (typeof value === 'string') return null;
    if (typeof value === 'object' && value !== null) return value['@type'] || null;
    return null;
}

// ── Diffing Logic ─────────────────────────────────────────────────

/**
 * Compare two JSON-LD context definitions and return a structured diff.
 */
export function contextDiff(
    oldCtx: Record<string, any>,
    newCtx: Record<string, any>
): ContextDiff {
    const result: ContextDiff = {
        added: {},
        removed: {},
        changed: {},
        oldVersion: oldCtx['@contextVersion'],
        newVersion: newCtx['@contextVersion']
    };

    const oldKeys = new Set(Object.keys(oldCtx).filter(k => k !== '@contextVersion'));
    const newKeys = new Set(Object.keys(newCtx).filter(k => k !== '@contextVersion'));

    // Added terms
    for (const key of newKeys) {
        if (!oldKeys.has(key)) {
            result.added[key] = newCtx[key];
        }
    }

    // Removed terms
    for (const key of oldKeys) {
        if (!newKeys.has(key)) {
            result.removed[key] = oldCtx[key];
        }
    }

    // Changed terms
    for (const key of oldKeys) {
        if (newKeys.has(key)) {
            const oldVal = oldCtx[key];
            const newVal = newCtx[key];

            // Deep comparison for objects or simple equality for primitives
            const isDifferent = JSON.stringify(oldVal) !== JSON.stringify(newVal);

            if (isDifferent) {
                result.changed[key] = {
                    term: key,
                    oldValue: oldVal,
                    newValue: newVal
                };
            }
        }
    }

    return result;
}

/**
 * Check backward compatibility between two context versions.
 */
export function checkCompatibility(
    oldCtx: Record<string, any>,
    newCtx: Record<string, any>
): CompatibilityResult {
    const diff = contextDiff(oldCtx, newCtx);
    const breaking: BreakingChange[] = [];
    const nonBreaking: BreakingChange[] = [];

    // Removals are always breaking
    for (const term of Object.keys(diff.removed)) {
        breaking.push({
            term,
            changeType: 'removed',
            detail: `Term '${term}' was removed`
        });
    }

    // Additions are non-breaking
    for (const term of Object.keys(diff.added)) {
        nonBreaking.push({
            term,
            changeType: 'added',
            detail: `Term '${term}' was added`
        });
    }

    // Changes require classification
    for (const change of Object.values(diff.changed)) {
        const term = change.term;

        // Special context keywords — changes to these are breaking
        if (term.startsWith('@')) {
            breaking.push({
                term,
                changeType: `changed-${term.substring(1)}`,
                detail: `${term} changed from ${JSON.stringify(change.oldValue)} to ${JSON.stringify(change.newValue)}`
            });
            continue;
        }

        const oldIri = _extractIri(change.oldValue);
        const newIri = _extractIri(change.newValue);

        // IRI mapping changed
        if (oldIri !== newIri) {
            breaking.push({
                term,
                changeType: 'changed-mapping',
                detail: `IRI changed from ${oldIri} to ${newIri}`
            });
            continue;
        }

        // Same IRI but definition structure changed (type coercion, etc.)
        const oldType = _extractType(change.oldValue);
        const newType = _extractType(change.newValue);

        if (oldType !== newType) {
            breaking.push({
                term,
                changeType: 'changed-type',
                detail: `@type coercion changed from ${oldType} to ${newType}`
            });
            continue;
        }

        // Other structural change (e.g. @container) — treat as breaking
        breaking.push({
            term,
            changeType: 'changed-definition',
            detail: `Definition changed for term '${term}'`
        });
    }

    return {
        compatible: breaking.length === 0,
        breaking,
        nonBreaking
    };
}
