/**
 * CBOR-LD Serialization for JSON-LD-Ex.
 *
 * Binary-efficient serialization of JSON-LD documents using CBOR.
 * Provides context compression by mapping well-known contexts to integers.
 */

import { encode, decode } from 'cbor-x';

// ── Default Context Registry ──────────────────────────────────────

export const DEFAULT_CONTEXT_REGISTRY: Record<string, number> = {
    'http://schema.org/': 1,
    'https://schema.org/': 1,
    'https://www.w3.org/ns/activitystreams': 2,
    'https://w3id.org/security/v2': 3,
    'https://www.w3.org/2018/credentials/v1': 4,
    'http://www.w3.org/ns/prov#': 5,
};

// ── Types ─────────────────────────────────────────────────────────

export interface PayloadStats {
    jsonBytes: number;
    cborBytes: number;
    cborRatio: number;
}

// ── Serialization ─────────────────────────────────────────────────

/**
 * Serialize a JSON-LD document to CBOR with context compression.
 *
 * @param doc - The JSON-LD document.
 * @param contextRegistry - Optional registry mapping context URLs to integers.
 * @returns CBOR-encoded buffer.
 */
export function toCbor(
    doc: any,
    contextRegistry?: Record<string, number>
): Buffer {
    const registry = contextRegistry || DEFAULT_CONTEXT_REGISTRY;
    const compressed = compressContexts(doc, registry);
    return Buffer.from(encode(compressed));
}

/**
 * Deserialize CBOR bytes back to a JSON-LD document.
 *
 * @param data - CBOR-encoded buffer.
 * @param contextRegistry - Same registry used during serialization.
 * @returns Restored JSON-LD document.
 */
export function fromCbor(
    data: Buffer | Uint8Array,
    contextRegistry?: Record<string, number>
): any {
    const registry = contextRegistry || DEFAULT_CONTEXT_REGISTRY;
    const reverseRegistry = Object.entries(registry).reduce(
        (acc, [k, v]) => ({ ...acc, [v]: k }),
        {} as Record<number, string>
    );

    const decoded = decode(data);
    return decompressContexts(decoded, reverseRegistry);
}

// ── Statistics ────────────────────────────────────────────────────

/**
 * Compare serialization sizes for a document.
 */
export function payloadStats(
    doc: any,
    contextRegistry?: Record<string, number>
): PayloadStats {
    const jsonBytes = Buffer.byteLength(JSON.stringify(doc));
    const cborBytes = toCbor(doc, contextRegistry).length;

    return {
        jsonBytes,
        cborBytes,
        cborRatio: jsonBytes === 0 ? 0 : cborBytes / jsonBytes,
    };
}

// ── Internal Helpers ──────────────────────────────────────────────

function compressContexts(obj: any, registry: Record<string, number>): any {
    if (Array.isArray(obj)) {
        return obj.map((item) => compressContexts(item, registry));
    }

    if (obj && typeof obj === 'object') {
        const result: any = {};
        for (const [k, v] of Object.entries(obj)) {
            if (k === '@context') {
                result[k] = compressContextValue(v, registry);
            } else {
                result[k] = compressContexts(v, registry);
            }
        }
        return result;
    }

    return obj;
}

function compressContextValue(ctx: any, registry: Record<string, number>): any {
    if (typeof ctx === 'string') {
        return registry[ctx] || ctx;
    }
    if (Array.isArray(ctx)) {
        return ctx.map((item) => compressContextValue(item, registry));
    }
    if (ctx && typeof ctx === 'object') {
        // Inline context definition — recurse only for @import if it exists (rare)
        // or just return as is (Python version recurses for @import but typically inline ctx is preserved)
        // We will shallow copy to be safe
        return { ...ctx };
    }
    return ctx;
}

function decompressContexts(obj: any, reverseRegistry: Record<number, string>): any {
    if (Array.isArray(obj)) {
        return obj.map((item) => decompressContexts(item, reverseRegistry));
    }

    if (obj && typeof obj === 'object') {
        const result: any = {};
        for (const [k, v] of Object.entries(obj)) {
            if (k === '@context') {
                result[k] = decompressContextValue(v, reverseRegistry);
            } else {
                result[k] = decompressContexts(v, reverseRegistry);
            }
        }
        return result;
    }

    return obj;
}

function decompressContextValue(ctx: any, reverseRegistry: Record<number, string>): any {
    if (typeof ctx === 'number') {
        return reverseRegistry[ctx] || ctx;
    }
    if (Array.isArray(ctx)) {
        return ctx.map((item) => decompressContextValue(item, reverseRegistry));
    }
    return ctx;
}
