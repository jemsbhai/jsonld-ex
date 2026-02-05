/**
 * Core type definitions for jsonld-ex
 */

// ── Provenance Metadata ───────────────────────────────────────────

/** AI/ML provenance metadata attached to a value */
export interface ProvenanceMetadata {
  /** Confidence score (0.0 to 1.0) */
  confidence?: number;
  /** IRI of the source model or system */
  source?: string;
  /** ISO 8601 timestamp of extraction */
  extractedAt?: string;
  /** Extraction method description */
  method?: string;
  /** Whether a human verified this value */
  humanVerified?: boolean;
}

/** A JSON-LD value annotated with provenance */
export interface AnnotatedValue {
  '@value': string | number | boolean;
  '@confidence'?: number;
  '@source'?: string;
  '@extractedAt'?: string;
  '@method'?: string;
  '@humanVerified'?: boolean;
  '@type'?: string;
  '@language'?: string;
}

// ── Vector Embeddings ─────────────────────────────────────────────

/** Vector embedding definition in context */
export interface VectorTermDefinition {
  '@id': string;
  '@container': '@vector';
  '@dimensions'?: number;
}

/** A node with an attached vector embedding */
export interface EmbeddedNode {
  '@type'?: string | string[];
  '@id'?: string;
  [property: string]: any;
}

// ── Temporal Extensions ───────────────────────────────────────────

/** Temporal qualifiers for a statement */
export interface TemporalQualifiers {
  '@validFrom'?: string;
  '@validUntil'?: string;
  '@asOf'?: string;
}

// ── Security ──────────────────────────────────────────────────────

/** Context reference with integrity verification */
export interface IntegrityContext {
  '@id': string;
  '@integrity': string;
}

/** Resource limits for processor configuration */
export interface ResourceLimits {
  /** Maximum nested context chain depth (default: 10) */
  maxContextDepth?: number;
  /** Maximum @graph nesting depth (default: 100) */
  maxGraphDepth?: number;
  /** Maximum input document size in bytes (default: 10MB) */
  maxDocumentSize?: number;
  /** Processing timeout in milliseconds (default: 30000) */
  maxExpansionTime?: number;
}

/** Context allowlist configuration */
export interface ContextAllowlist {
  /** Exact context IRIs that are allowed */
  allowed?: string[];
  /** Glob/regex patterns for allowed contexts */
  patterns?: (string | RegExp)[];
  /** Block all remote context loading */
  blockRemoteContexts?: boolean;
}

// ── Validation / Shapes ───────────────────────────────────────────

/** Property constraint in a shape definition */
export interface PropertyShape {
  '@required'?: boolean;
  '@type'?: string;
  '@minimum'?: number;
  '@maximum'?: number;
  '@minLength'?: number;
  '@maxLength'?: number;
  '@pattern'?: string;
}

/** Shape definition for a node type */
export interface ShapeDefinition {
  '@type': string;
  [property: string]: PropertyShape | string;
}

/** Validation result */
export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  path: string;
  constraint: string;
  message: string;
  value?: any;
}

export interface ValidationWarning {
  path: string;
  code: string;
  message: string;
}

// ── Processor Options ─────────────────────────────────────────────

/** Extended options for the jsonld-ex processor */
export interface JsonLdExOptions {
  /** Security: resource limits */
  resourceLimits?: ResourceLimits;
  /** Security: context allowlist */
  contextAllowlist?: ContextAllowlist;
  /** Whether to process extension keywords (default: true) */
  processExtensions?: boolean;
  /** Whether to validate vectors during expansion (default: true) */
  validateVectors?: boolean;
  /** Base options passed through to jsonld.js */
  base?: string;
  /** Document loader override */
  documentLoader?: any;
}
