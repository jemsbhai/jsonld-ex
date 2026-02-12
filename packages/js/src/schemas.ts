/**
 * Runtime validation schemas using Zod.
 */

import { z } from 'zod';

// ── Primitives & Helpers ──────────────────────────────────────────

const ConfidenceSchema = z.number().min(0.0).max(1.0);
const TimestampSchema = z.string().datetime(); // ISO 8601
const IriSchema = z.string(); // Simple string for now, could be regex for IRI

// ── Provenance Metadata ───────────────────────────────────────────

export const ProvenanceMetadataSchema = z.object({
    confidence: ConfidenceSchema.optional(),
    source: IriSchema.optional(),
    extractedAt: TimestampSchema.optional(),
    method: z.string().optional(),
    humanVerified: z.boolean().optional(),
    // Multimodal
    mediaType: z.string().optional(),
    contentUrl: IriSchema.optional(),
    contentHash: z.string().optional(),
    // Translation
    translatedFrom: IriSchema.optional(),
    translationModel: z.string().optional(),
    // Measurement
    measurementUncertainty: z.number().positive().optional(),
    unit: z.string().optional(),
});

export const AnnotatedValueSchema = z.object({
    '@value': z.union([z.string(), z.number(), z.boolean()]),
    '@confidence': ConfidenceSchema.optional(),
    '@source': IriSchema.optional(),
    '@extractedAt': TimestampSchema.optional(),
    '@method': z.string().optional(),
    '@humanVerified': z.boolean().optional(),
    '@type': z.string().optional(),
    '@mediaType': z.string().optional(),
    '@language': z.string().optional(),
    '@contentUrl': IriSchema.optional(),
    '@contentHash': z.string().optional(),
    '@translatedFrom': IriSchema.optional(),
    '@translationModel': z.string().optional(),
    '@measurementUncertainty': z.number().positive().optional(),
    '@unit': z.string().optional(),
});

// ── Vector Embeddings ─────────────────────────────────────────────

export const VectorTermDefinitionSchema = z.object({
    '@id': IriSchema,
    '@container': z.literal('@vector'),
    '@dimensions': z.number().int().positive().optional(),
});

// ── Temporal Extensions ───────────────────────────────────────────

export const TemporalQualifiersSchema = z.object({
    '@validFrom': TimestampSchema.optional(),
    '@validUntil': TimestampSchema.optional(),
    '@asOf': TimestampSchema.optional(),
});

export const TemporalOptionsSchema = z.object({
    validFrom: TimestampSchema.optional(),
    validUntil: TimestampSchema.optional(),
    asOf: TimestampSchema.optional(),
});

// ── Confidence / Subjective Logic ─────────────────────────────────

export const OpinionSchema = z.object({
    belief: z.number().min(0).max(1),
    disbelief: z.number().min(0).max(1),
    uncertainty: z.number().min(0).max(1),
    baseRate: z.number().min(0).max(1).default(0.5),
}).refine((data) => {
    const sum = data.belief + data.disbelief + data.uncertainty;
    return Math.abs(sum - 1.0) < 1e-9;
}, {
    message: "Belief + Disbelief + Uncertainty must sum to 1.0",
});

// ── Reports ───────────────────────────────────────────────────────

export const MergeConflictSchema = z.object({
    nodeId: z.string(),
    property: z.string(),
    values: z.array(z.any()),
    resolution: z.string(),
    winner: z.any(),
});

export const MergeReportSchema = z.object({
    nodesMerged: z.number().int().nonnegative(),
    propertiesAgreed: z.number().int().nonnegative(),
    propertiesConflicted: z.number().int().nonnegative(),
    propertiesUnion: z.number().int().nonnegative(),
    conflicts: z.array(MergeConflictSchema),
    sourceCount: z.number().int().positive(),
});

// ── Validation Helper ─────────────────────────────────────────────

/**
 * Validates data against a Zod schema.
 * Throws a ZodError if validation fails.
 */
export function validate<T>(schema: z.ZodType<T>, data: unknown): T {
    return schema.parse(data);
}

/**
 * Safely validates data against a Zod schema.
 * Returns a SafeParseReturnType.
 */
export function safeValidate<T>(schema: z.ZodType<T>, data: unknown) {
    return schema.safeParse(data);
}
