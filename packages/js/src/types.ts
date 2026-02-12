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
  // Multimodal (GAP-MM1)
  mediaType?: string;
  contentUrl?: string;
  contentHash?: string;
  // Translation provenance (GAP-ML2)
  translatedFrom?: string;
  translationModel?: string;
  // Measurement uncertainty (GAP-IOT1)
  measurementUncertainty?: number;
  unit?: string;
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
  '@mediaType'?: string;
  '@language'?: string;
  // Multimodal
  '@contentUrl'?: string;
  '@contentHash'?: string;
  // Translation
  '@translatedFrom'?: string;
  '@translationModel'?: string;
  // Measurement
  '@measurementUncertainty'?: number;
  '@unit'?: string;
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

/** Input options for temporal annotation */
export interface TemporalOptions {
  validFrom?: string;
  validUntil?: string;
  asOf?: string;
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

// ── Merge & Conflict Types ────────────────────────────────────────

/**
 * Report from a merge operation.
 */
export interface MergeReport {
  mergedNodeCount: number;
  conflicts: MergeConflict[];
  strategy: string;
}

/**
 * Details of a conflict encountered during merge.
 */
export interface MergeConflict {
  nodeId: string;
  property: string;
  values: any[];
  resolution: string;
  winner: any;
}

/**
 * Result of a confidence propagation operation.
 */
export interface PropagationResult {
  score: number;
  method: string;
  inputScores: number[];
  provenanceTrail?: string[];
}

/**
 * Report from a conflict resolution operation.
 */
export interface ConflictReport {
  winner: any;
  strategy: string;
  candidates: any[];
  confidenceScores: number[];
  reason: string;
}

/**
 * Result of a temporal difference calculation.
 */
export interface TemporalDiffResult {
  added: JsonLdNode[];
  removed: JsonLdNode[];
  modified: Array<{
    nodeId: string;
    changes: Record<string, { from: any; to: any }>;
  }>;
}

// ── Croissant / Dataset Types ─────────────────────────────────────

export type JsonLdNode = Record<string, any>;

export interface Distribution {
  contentUrl?: string;
  encodingFormat?: string;
  sha256?: string;
  name?: string;
}

export interface RecordSet {
  name: string;
  description?: string;
  field?: Record<string, any>[];
  key?: string | string[];
}

export interface Dataset {
  name: string;
  description?: string;
  distribution?: Distribution[];
  recordSet?: RecordSet[];
  [key: string]: any;
}
