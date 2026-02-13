/**
 * AI/ML Extensions for JSON-LD
 *
 * Provides support for:
 * - @confidence: Assertion confidence scores (0.0 - 1.0)
 * - @source: Provenance tracking (model/system IRI)
 * - @extractedAt: Extraction timestamps
 * - @method: Extraction method descriptions
 * - @humanVerified: Human verification flags
 *
 * These extensions enable JSON-LD to serve as a standard interchange
 * format for AI/ML-generated knowledge graphs with full provenance.
 */

import {
  KEYWORD_CONFIDENCE,
  KEYWORD_SOURCE,
  KEYWORD_EXTRACTED_AT,
  KEYWORD_METHOD,
  KEYWORD_HUMAN_VERIFIED,
  KEYWORD_MEDIA_TYPE,
  KEYWORD_CONTENT_URL,
  KEYWORD_CONTENT_HASH,
  KEYWORD_TRANSLATED_FROM,
  KEYWORD_TRANSLATION_MODEL,
  KEYWORD_MEASUREMENT_UNCERTAINTY,
  KEYWORD_UNIT,
  KEYWORD_ACTED_ON_BEHALF_OF,
  KEYWORD_WAS_DERIVED_FROM,
  KEYWORD_WAS_INVALIDATED_BY,
  KEYWORD_CALIBRATION_STATUS,
  KEYWORD_CALIBRATION_TARGET,
  KEYWORD_WAS_AGGREGATED_BY,
  JSONLD_EX_NAMESPACE,
} from '../keywords.js';
import { AnnotatedValue, ProvenanceMetadata } from '../types.js';

// ── Context Definition ────────────────────────────────────────────

/**
 * JSON-LD context fragment that defines the AI/ML extension terms.
 * Merge this into your document's @context to enable provenance annotations.
 */
export const AI_ML_CONTEXT = {
  '@vocab': JSONLD_EX_NAMESPACE,
  'confidence': {
    '@id': `${JSONLD_EX_NAMESPACE}confidence`,
    '@type': 'http://www.w3.org/2001/XMLSchema#double',
  },
  'source': {
    '@id': `${JSONLD_EX_NAMESPACE}source`,
    '@type': '@id',
  },
  'extractedAt': {
    '@id': `${JSONLD_EX_NAMESPACE}extractedAt`,
    '@type': 'http://www.w3.org/2001/XMLSchema#dateTime',
  },
  'method': {
    '@id': `${JSONLD_EX_NAMESPACE}method`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'humanVerified': {
    '@id': `${JSONLD_EX_NAMESPACE}humanVerified`,
    '@type': 'http://www.w3.org/2001/XMLSchema#boolean',
  },
  'mediaType': {
    '@id': `${JSONLD_EX_NAMESPACE}mediaType`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'contentUrl': {
    '@id': `${JSONLD_EX_NAMESPACE}contentUrl`,
    '@type': '@id',
  },
  'contentHash': {
    '@id': `${JSONLD_EX_NAMESPACE}contentHash`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'translatedFrom': {
    '@id': `${JSONLD_EX_NAMESPACE}translatedFrom`,
    '@type': '@id', // or string depending on usage, usually IRI
  },
  'translationModel': {
    '@id': `${JSONLD_EX_NAMESPACE}translationModel`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'measurementUncertainty': {
    '@id': `${JSONLD_EX_NAMESPACE}measurementUncertainty`,
    '@type': 'http://www.w3.org/2001/XMLSchema#double',
  },
  'unit': {
    '@id': `${JSONLD_EX_NAMESPACE}unit`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'actedOnBehalfOf': {
    '@id': `${JSONLD_EX_NAMESPACE}actedOnBehalfOf`,
    '@type': '@id',
  },
  'wasDerivedFrom': {
    '@id': `${JSONLD_EX_NAMESPACE}wasDerivedFrom`,
    '@type': '@id',
  },
  'wasInvalidatedBy': {
    '@id': `${JSONLD_EX_NAMESPACE}wasInvalidatedBy`,
    '@type': '@id',
  },
  'calibrationStatus': {
    '@id': `${JSONLD_EX_NAMESPACE}calibrationStatus`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'calibrationTarget': {
    '@id': `${JSONLD_EX_NAMESPACE}calibrationTarget`,
    '@type': 'http://www.w3.org/2001/XMLSchema#string',
  },
  'wasAggregatedBy': {
    '@id': `${JSONLD_EX_NAMESPACE}wasAggregatedBy`,
    '@type': '@id',
  },
};

// ── Annotation Builder ────────────────────────────────────────────

/**
 * Creates an annotated value with AI/ML provenance metadata.
 *
 * @example
 * ```ts
 * const name = annotate("John Smith", {
 *   confidence: 0.95,
 *   source: "https://ml-model.example.org/ner-v2",
 *   method: "NER",
 * });
 * // Returns: { "@value": "John Smith", "@confidence": 0.95, ... }
 * ```
 */
export function annotate(
  value: string | number | boolean,
  metadata: ProvenanceMetadata
): AnnotatedValue {
  const result: AnnotatedValue = { '@value': value };

  if (metadata.confidence !== undefined) {
    validateConfidence(metadata.confidence);
    result['@confidence'] = metadata.confidence;
  }
  if (metadata.source !== undefined) {
    result['@source'] = metadata.source;
  }
  if (metadata.extractedAt !== undefined) {
    result['@extractedAt'] = metadata.extractedAt;
  }
  if (metadata.method !== undefined) {
    result['@method'] = metadata.method;
  }
  if (metadata.humanVerified !== undefined) {
    result[KEYWORD_HUMAN_VERIFIED] = metadata.humanVerified;
  }
  if (metadata.mediaType !== undefined) {
    result[KEYWORD_MEDIA_TYPE] = metadata.mediaType;
  }
  if (metadata.contentUrl !== undefined) {
    result[KEYWORD_CONTENT_URL] = metadata.contentUrl;
  }
  if (metadata.contentHash !== undefined) {
    result[KEYWORD_CONTENT_HASH] = metadata.contentHash;
  }
  if (metadata.translatedFrom !== undefined) {
    result[KEYWORD_TRANSLATED_FROM] = metadata.translatedFrom;
  }
  if (metadata.translationModel !== undefined) {
    result[KEYWORD_TRANSLATION_MODEL] = metadata.translationModel;
  }
  if (metadata.measurementUncertainty !== undefined) {
    result[KEYWORD_MEASUREMENT_UNCERTAINTY] = metadata.measurementUncertainty;
  }
  if (metadata.unit !== undefined) {
    result[KEYWORD_UNIT] = metadata.unit;
  }
  if (metadata.actedOnBehalfOf !== undefined) {
    result[KEYWORD_ACTED_ON_BEHALF_OF] = metadata.actedOnBehalfOf;
  }
  if (metadata.wasDerivedFrom !== undefined) {
    result[KEYWORD_WAS_DERIVED_FROM] = metadata.wasDerivedFrom;
  }
  if (metadata.wasInvalidatedBy !== undefined) {
    result[KEYWORD_WAS_INVALIDATED_BY] = metadata.wasInvalidatedBy;
  }
  if (metadata.calibrationStatus !== undefined) {
    result[KEYWORD_CALIBRATION_STATUS] = metadata.calibrationStatus;
  }
  if (metadata.calibrationTarget !== undefined) {
    result[KEYWORD_CALIBRATION_TARGET] = metadata.calibrationTarget;
  }
  if (metadata.wasAggregatedBy !== undefined) {
    result[KEYWORD_WAS_AGGREGATED_BY] = metadata.wasAggregatedBy;
  }

  return result;
}

// ── Extraction Helpers ────────────────────────────────────────────

/**
 * Extracts the confidence score from an annotated value or expanded node.
 * Returns undefined if no confidence is present.
 */
export function getConfidence(node: any): number | undefined {
  if (node == null) return undefined;

  // Direct annotation (compact form)
  if (typeof node === 'object' && KEYWORD_CONFIDENCE in node) {
    return node[KEYWORD_CONFIDENCE];
  }

  // Expanded form — look for namespace-qualified key
  const expandedKey = `${JSONLD_EX_NAMESPACE}confidence`;
  if (typeof node === 'object' && expandedKey in node) {
    const val = node[expandedKey];
    if (Array.isArray(val) && val.length > 0) {
      return val[0]['@value'] ?? val[0];
    }
    return val?.['@value'] ?? val;
  }

  return undefined;
}

/**
 * Extracts all provenance metadata from an annotated value.
 */
export function getProvenance(node: any): ProvenanceMetadata {
  const meta: ProvenanceMetadata = {};

  if (node == null || typeof node !== 'object') return meta;

  // Try compact form first, then expanded
  meta.confidence = extractField(node, 'confidence', KEYWORD_CONFIDENCE);
  meta.source = extractField(node, 'source', KEYWORD_SOURCE);
  meta.extractedAt = extractField(node, 'extractedAt', KEYWORD_EXTRACTED_AT);
  meta.method = extractField(node, 'method', KEYWORD_METHOD);
  meta.humanVerified = extractField(node, 'humanVerified', KEYWORD_HUMAN_VERIFIED);
  meta.mediaType = extractField(node, 'mediaType', KEYWORD_MEDIA_TYPE);
  if (!meta.mediaType && node['@type']) {
    meta.mediaType = node['@type'];
  }
  meta.contentUrl = extractField(node, 'contentUrl', KEYWORD_CONTENT_URL);
  meta.contentHash = extractField(node, 'contentHash', KEYWORD_CONTENT_HASH);
  meta.translatedFrom = extractField(node, 'translatedFrom', KEYWORD_TRANSLATED_FROM);
  meta.translationModel = extractField(node, 'translationModel', KEYWORD_TRANSLATION_MODEL);
  meta.measurementUncertainty = extractField(node, 'measurementUncertainty', KEYWORD_MEASUREMENT_UNCERTAINTY);
  meta.unit = extractField(node, 'unit', KEYWORD_UNIT);
  meta.actedOnBehalfOf = extractField(node, 'actedOnBehalfOf', KEYWORD_ACTED_ON_BEHALF_OF);
  meta.wasDerivedFrom = extractField(node, 'wasDerivedFrom', KEYWORD_WAS_DERIVED_FROM);
  meta.wasInvalidatedBy = extractField(node, 'wasInvalidatedBy', KEYWORD_WAS_INVALIDATED_BY);
  meta.calibrationStatus = extractField(node, 'calibrationStatus', KEYWORD_CALIBRATION_STATUS);
  meta.calibrationTarget = extractField(node, 'calibrationTarget', KEYWORD_CALIBRATION_TARGET);
  meta.wasAggregatedBy = extractField(node, 'wasAggregatedBy', KEYWORD_WAS_AGGREGATED_BY);

  // Remove undefined entries
  return Object.fromEntries(
    Object.entries(meta).filter(([_, v]) => v !== undefined)
  ) as ProvenanceMetadata;
}

/**
 * Filters a graph to only include nodes meeting a minimum confidence threshold.
 *
 * @param graph - Array of expanded JSON-LD nodes
 * @param property - Property name to check confidence on
 * @param minConfidence - Minimum confidence threshold (0.0 - 1.0)
 * @returns Filtered array of nodes
 */
export function filterByConfidence(
  graph: any[],
  property: string,
  minConfidence: number
): any[] {
  validateConfidence(minConfidence);

  return graph.filter((node) => {
    const propValue = node[property];
    if (!propValue) return false;

    const values = Array.isArray(propValue) ? propValue : [propValue];
    return values.some((v: any) => {
      const conf = getConfidence(v);
      return conf !== undefined && conf >= minConfidence;
    });
  });
}

/**
 * Aggregates confidence scores across multiple sources using
 * configurable strategies.
 */
export function aggregateConfidence(
  scores: number[],
  strategy: 'mean' | 'max' | 'min' | 'weighted' = 'mean',
  weights?: number[]
): number {
  if (scores.length === 0) return 0;
  scores.forEach(validateConfidence);

  switch (strategy) {
    case 'max':
      return Math.max(...scores);
    case 'min':
      return Math.min(...scores);
    case 'weighted': {
      if (!weights || weights.length !== scores.length) {
        throw new Error('Weights array must match scores array length');
      }
      const totalWeight = weights.reduce((a, b) => a + b, 0);
      const weighted = scores.reduce((sum, s, i) => sum + s * weights[i], 0);
      return weighted / totalWeight;
    }
    case 'mean':
    default:
      return scores.reduce((a, b) => a + b, 0) / scores.length;
  }
}

// ── Internal Helpers ──────────────────────────────────────────────

function validateConfidence(score: number): void {
  if (typeof score !== 'number' || score < 0 || score > 1) {
    throw new RangeError(
      `@confidence must be a number between 0.0 and 1.0, got: ${score}`
    );
  }
}

function extractField(
  node: any,
  compactName: string,
  keyword: string
): any {
  // Compact form (using @-keyword)
  if (keyword in node) {
    return node[keyword];
  }
  // Compact form (using term name)
  if (compactName in node) {
    return node[compactName];
  }
  // Expanded form (namespace-qualified)
  const expandedKey = `${JSONLD_EX_NAMESPACE}${compactName}`;
  if (expandedKey in node) {
    const val = node[expandedKey];
    if (Array.isArray(val) && val.length > 0) {
      return val[0]['@value'] ?? val[0];
    }
    return val?.['@value'] ?? val;
  }
  return undefined;
}
