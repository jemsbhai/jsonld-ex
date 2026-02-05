/**
 * Vector Embedding Extensions for JSON-LD
 *
 * Provides the @vector container type, enabling vector embeddings
 * to coexist with symbolic linked data in JSON-LD documents.
 *
 * Key design decisions:
 * - Vectors are preserved during expansion but excluded from RDF conversion
 * - Dimension validation is enforced when @dimensions is specified
 * - Efficient binary serialization is supported via CBOR-LD integration
 */

import { CONTAINER_VECTOR, KEYWORD_DIMENSIONS, JSONLD_EX_NAMESPACE } from '../keywords';
import { VectorTermDefinition } from '../types';

// ── Context Definition ────────────────────────────────────────────

/**
 * Creates a context term definition for a vector embedding property.
 *
 * @example
 * ```ts
 * const ctx = {
 *   "@context": {
 *     ...vectorTermDefinition("embedding", "http://example.org/embedding", 768)
 *   }
 * };
 * ```
 */
export function vectorTermDefinition(
  termName: string,
  iri: string,
  dimensions?: number
): Record<string, VectorTermDefinition> {
  const def: VectorTermDefinition = {
    '@id': iri,
    '@container': CONTAINER_VECTOR,
  };
  if (dimensions !== undefined) {
    if (!Number.isInteger(dimensions) || dimensions < 1) {
      throw new RangeError(`@dimensions must be a positive integer, got: ${dimensions}`);
    }
    def['@dimensions'] = dimensions;
  }
  return { [termName]: def };
}

// ── Validation ────────────────────────────────────────────────────

/**
 * Validates a vector embedding value against its term definition.
 *
 * Checks:
 * - Value is an array of numbers
 * - All elements are finite numbers
 * - Dimension count matches @dimensions if specified
 *
 * @returns Object with valid flag and any error messages
 */
export function validateVector(
  vector: any,
  expectedDimensions?: number
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!Array.isArray(vector)) {
    errors.push(`Vector must be an array, got: ${typeof vector}`);
    return { valid: false, errors };
  }

  if (vector.length === 0) {
    errors.push('Vector must not be empty');
    return { valid: false, errors };
  }

  // Check all elements are finite numbers
  for (let i = 0; i < vector.length; i++) {
    if (typeof vector[i] !== 'number' || !isFinite(vector[i])) {
      errors.push(`Vector element [${i}] must be a finite number, got: ${vector[i]}`);
    }
  }

  // Dimension check
  if (expectedDimensions !== undefined && vector.length !== expectedDimensions) {
    errors.push(
      `Vector dimension mismatch: expected ${expectedDimensions}, got ${vector.length}`
    );
  }

  return { valid: errors.length === 0, errors };
}

/**
 * Computes cosine similarity between two vectors.
 * Useful for confidence-weighted similarity in JSON-LD knowledge graphs.
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(`Vector dimension mismatch: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  if (denominator === 0) return 0;

  return dotProduct / denominator;
}

/**
 * Extracts vector embeddings from an expanded JSON-LD node.
 * Returns a map of property IRI → vector values.
 */
export function extractVectors(
  node: any,
  vectorProperties: string[]
): Map<string, number[]> {
  const vectors = new Map<string, number[]>();

  if (node == null || typeof node !== 'object') return vectors;

  for (const prop of vectorProperties) {
    const value = node[prop];
    if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'number') {
      vectors.set(prop, value);
    }
    // Handle expanded form where value is wrapped
    else if (Array.isArray(value) && value.length > 0 && value[0]?.['@value']) {
      // Reconstruct vector from expanded value nodes
      const vec = value.map((v: any) =>
        typeof v === 'object' ? v['@value'] : v
      ).filter((v: any) => typeof v === 'number');
      if (vec.length > 0) {
        vectors.set(prop, vec);
      }
    }
  }

  return vectors;
}

/**
 * Strips vector embeddings from a document for RDF conversion.
 * Vectors are annotation-only and should not be converted to triples.
 *
 * @param doc - Expanded JSON-LD document
 * @param vectorProperties - IRIs of vector properties to strip
 * @returns Document copy with vectors removed
 */
export function stripVectorsForRdf(
  doc: any,
  vectorProperties: string[]
): any {
  if (Array.isArray(doc)) {
    return doc.map((item) => stripVectorsForRdf(item, vectorProperties));
  }

  if (doc == null || typeof doc !== 'object') return doc;

  const result: any = {};
  for (const [key, value] of Object.entries(doc)) {
    if (vectorProperties.includes(key)) continue;
    result[key] = stripVectorsForRdf(value, vectorProperties);
  }
  return result;
}
