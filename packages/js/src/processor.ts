/**
 * JsonLdEx — Extended JSON-LD Processor
 *
 * Wraps jsonld.js with backward-compatible extensions for AI/ML,
 * security, and validation. All standard JSON-LD 1.1 operations
 * pass through to jsonld.js; extension keywords are processed
 * as an additional layer.
 */

import * as jsonld from 'jsonld';
import { JsonLdExOptions, ResourceLimits, ProvenanceMetadata, ValidationResult, ShapeDefinition } from './types';
import {
  annotate, getConfidence, getProvenance, filterByConfidence,
  aggregateConfidence, AI_ML_CONTEXT,
} from './extensions/ai-ml';
import {
  vectorTermDefinition, validateVector, cosineSimilarity,
  extractVectors, stripVectorsForRdf,
} from './extensions/vector';
import {
  computeIntegrity, verifyIntegrity, integrityContext,
  isContextAllowed, createSecureDocumentLoader,
  enforceResourceLimits, withTimeout, DEFAULT_RESOURCE_LIMITS,
} from './extensions/security';
import { validateNode, validateDocument } from './extensions/validation';
import { JSONLD_EX_NAMESPACE, EXTENSION_KEYWORDS } from './keywords';

export class JsonLdEx {
  private options: JsonLdExOptions;
  private resourceLimits: Required<ResourceLimits>;

  constructor(options: JsonLdExOptions = {}) {
    this.options = options;
    this.resourceLimits = {
      ...DEFAULT_RESOURCE_LIMITS,
      ...options.resourceLimits,
    };
  }

  // ── Core JSON-LD Operations (delegated to jsonld.js) ──────────

  /**
   * Expand a JSON-LD document with extension support.
   * Enforces resource limits and processes extension keywords.
   */
  async expand(doc: any, opts?: any): Promise<any[]> {
    enforceResourceLimits(doc, this.resourceLimits);

    const expandOpts = this.buildProcessorOptions(opts);

    return withTimeout(
      jsonld.expand(doc, expandOpts),
      this.resourceLimits.maxExpansionTime,
      'expansion'
    );
  }

  /**
   * Compact a JSON-LD document.
   */
  async compact(doc: any, ctx: any, opts?: any): Promise<any> {
    enforceResourceLimits(doc, this.resourceLimits);
    const compactOpts = this.buildProcessorOptions(opts);
    return withTimeout(
      jsonld.compact(doc, ctx, compactOpts),
      this.resourceLimits.maxExpansionTime,
      'compaction'
    );
  }

  /**
   * Flatten a JSON-LD document.
   */
  async flatten(doc: any, ctx?: any, opts?: any): Promise<any> {
    enforceResourceLimits(doc, this.resourceLimits);
    const flattenOpts = this.buildProcessorOptions(opts);
    return withTimeout(
      jsonld.flatten(doc, ctx, flattenOpts),
      this.resourceLimits.maxExpansionTime,
      'flattening'
    );
  }

  /**
   * Convert JSON-LD to N-Quads.
   * Strips vector embeddings before conversion (vectors are annotation-only).
   */
  async toRdf(doc: any, opts?: any): Promise<string> {
    enforceResourceLimits(doc, this.resourceLimits);
    const rdfOpts = this.buildProcessorOptions(opts);
    return withTimeout(
      jsonld.toRDF(doc, { ...rdfOpts, format: 'application/n-quads' }) as Promise<string>,
      this.resourceLimits.maxExpansionTime,
      'RDF conversion'
    );
  }

  /**
   * Convert N-Quads to JSON-LD.
   */
  async fromRdf(nquads: string, opts?: any): Promise<any[]> {
    return jsonld.fromRDF(nquads, opts);
  }

  // ── AI/ML Extension Methods ───────────────────────────────────

  /** Create an annotated value with provenance metadata */
  annotate = annotate;

  /** Get the confidence score from a node/value */
  getConfidence(node: any, property?: string): number | undefined {
    if (property) {
      const propValue = node?.[property];
      if (Array.isArray(propValue)) {
        return getConfidence(propValue[0]);
      }
      return getConfidence(propValue);
    }
    return getConfidence(node);
  }

  /** Extract full provenance metadata from a node/value */
  getProvenance(node: any, property?: string): ProvenanceMetadata {
    if (property) {
      const propValue = node?.[property];
      if (Array.isArray(propValue)) {
        return getProvenance(propValue[0]);
      }
      return getProvenance(propValue);
    }
    return getProvenance(node);
  }

  /** Filter graph nodes by minimum confidence on a property */
  filterByConfidence = filterByConfidence;

  /** Aggregate multiple confidence scores */
  aggregateConfidence = aggregateConfidence;

  /** Get the AI/ML context fragment for use in documents */
  getAiMlContext() {
    return AI_ML_CONTEXT;
  }

  // ── Vector Extension Methods ──────────────────────────────────

  /** Create a vector term definition for a context */
  vectorTermDefinition = vectorTermDefinition;

  /** Validate a vector embedding */
  validateVector = validateVector;

  /** Compute cosine similarity between two vectors */
  cosineSimilarity = cosineSimilarity;

  /** Extract all vectors from an expanded node */
  extractVectors = extractVectors;

  /** Strip vectors before RDF conversion */
  stripVectorsForRdf = stripVectorsForRdf;

  // ── Security Extension Methods ────────────────────────────────

  /** Compute integrity hash for a context */
  computeIntegrity = computeIntegrity;

  /** Verify context integrity */
  verifyIntegrity = verifyIntegrity;

  /** Create a context reference with integrity verification */
  integrityContext = integrityContext;

  /** Check if a context URL is allowed */
  isContextAllowed = isContextAllowed;

  // ── Validation Extension Methods ──────────────────────────────

  /** Validate a single node against a shape */
  validateNode = validateNode;

  /** Validate an entire document against shapes */
  validateDocument = validateDocument;

  // ── Utility Methods ───────────────────────────────────────────

  /**
   * Returns the extension namespace IRI.
   */
  get namespace(): string {
    return JSONLD_EX_NAMESPACE;
  }

  /**
   * Returns all extension keywords.
   */
  get extensionKeywords(): readonly string[] {
    return EXTENSION_KEYWORDS;
  }

  // ── Internal ──────────────────────────────────────────────────

  private buildProcessorOptions(opts?: any): any {
    const base: any = { ...opts };

    // Apply secure document loader if allowlist is configured
    if (this.options.contextAllowlist) {
      const defaultLoader = jsonld.documentLoaders.node();
      base.documentLoader = createSecureDocumentLoader(
        defaultLoader,
        this.options.contextAllowlist
      );
    }

    if (this.options.documentLoader) {
      base.documentLoader = this.options.documentLoader;
    }

    if (this.options.base) {
      base.base = this.options.base;
    }

    return base;
  }
}
