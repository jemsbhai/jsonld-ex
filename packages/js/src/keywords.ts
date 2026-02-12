/**
 * JSON-LD Extension Keywords
 *
 * These keywords extend the JSON-LD 1.1 specification with support for
 * AI/ML metadata, security hardening, and validation.
 *
 * All extension keywords use the @-prefix convention and are defined
 * under the jsonld-ex namespace to avoid conflicts with future
 * official JSON-LD keywords.
 */

// ── AI/ML Extension Keywords ──────────────────────────────────────
/** Confidence score for an assertion (0.0 - 1.0) */
export const KEYWORD_CONFIDENCE = '@confidence';
/** IRI of the data source or ML model that produced the value */
export const KEYWORD_SOURCE = '@source';
/** ISO 8601 timestamp when the value was extracted/generated */
export const KEYWORD_EXTRACTED_AT = '@extractedAt';
/** Description of the extraction method (e.g., "NER", "classification") */
export const KEYWORD_METHOD = '@method';
/** Boolean indicating whether a human has verified the value */
export const KEYWORD_HUMAN_VERIFIED = '@humanVerified';
// Multimodal
export const KEYWORD_MEDIA_TYPE = '@mediaType';
export const KEYWORD_CONTENT_URL = '@contentUrl';
export const KEYWORD_CONTENT_HASH = '@contentHash';
// Translation
export const KEYWORD_TRANSLATED_FROM = '@translatedFrom';
export const KEYWORD_TRANSLATION_MODEL = '@translationModel';
// Measurement
export const KEYWORD_MEASUREMENT_UNCERTAINTY = '@measurementUncertainty';
export const KEYWORD_UNIT = '@unit';

// ── Vector Embedding Keywords ─────────────────────────────────────
/** Container type for vector embeddings */
export const CONTAINER_VECTOR = '@vector';
/** Dimensionality constraint for vector properties */
export const KEYWORD_DIMENSIONS = '@dimensions';

// ── Temporal Extension Keywords ───────────────────────────────────
/** Start of validity period (ISO 8601) */
export const KEYWORD_VALID_FROM = '@validFrom';
/** End of validity period (ISO 8601) */
export const KEYWORD_VALID_UNTIL = '@validUntil';
/** Point-in-time snapshot reference (ISO 8601) */
export const KEYWORD_AS_OF = '@asOf';

// ── Security Extension Keywords ───────────────────────────────────
/** Cryptographic hash for context integrity verification */
export const KEYWORD_INTEGRITY = '@integrity';

// ── Validation Extension Keywords ─────────────────────────────────
/** Shape definition for native validation */
export const KEYWORD_SHAPE = '@shape';
/** Required constraint */
export const KEYWORD_REQUIRED = '@required';
/** Minimum value / length constraint */
export const KEYWORD_MINIMUM = '@minimum';
/** Maximum value / length constraint */
export const KEYWORD_MAXIMUM = '@maximum';
/** Min string length constraint */
export const KEYWORD_MIN_LENGTH = '@minLength';
/** Max string length constraint */
export const KEYWORD_MAX_LENGTH = '@maxLength';
/** Regex pattern constraint */
export const KEYWORD_PATTERN = '@pattern';

// ── Namespace ─────────────────────────────────────────────────────
/** JSON-LD Extensions namespace IRI */
export const JSONLD_EX_NAMESPACE = 'http://www.w3.org/ns/jsonld-ex/';

/** Full set of extension keywords */
export const EXTENSION_KEYWORDS = [
  KEYWORD_CONFIDENCE, KEYWORD_SOURCE, KEYWORD_EXTRACTED_AT,
  KEYWORD_METHOD, KEYWORD_HUMAN_VERIFIED,
  CONTAINER_VECTOR, KEYWORD_DIMENSIONS,
  KEYWORD_VALID_FROM, KEYWORD_VALID_UNTIL, KEYWORD_AS_OF,
  KEYWORD_INTEGRITY,
  KEYWORD_SHAPE, KEYWORD_REQUIRED, KEYWORD_MINIMUM, KEYWORD_MAXIMUM,
  KEYWORD_MIN_LENGTH, KEYWORD_MAX_LENGTH, KEYWORD_PATTERN,
  KEYWORD_MEDIA_TYPE, KEYWORD_CONTENT_URL, KEYWORD_CONTENT_HASH,
  KEYWORD_TRANSLATED_FROM, KEYWORD_TRANSLATION_MODEL,
  KEYWORD_MEASUREMENT_UNCERTAINTY, KEYWORD_UNIT,
] as const;
