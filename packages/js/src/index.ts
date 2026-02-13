/**
 * @jsonld-ex/core — JSON-LD 1.2 Extensions
 *
 * Reference implementation of proposed JSON-LD 1.2 extensions
 * for AI/ML data exchange, security hardening, and validation.
 */

// Main processor
export { JsonLdEx } from './processor.js';

// Types
export * from './types.js';

// Keywords and namespace
export * from './keywords.js';

// Extensions (for direct access)
export * from './extensions/ai-ml.js';
export * from './extensions/vector.js';
export * from './extensions/security.js';
export * from './extensions/validation.js';
export * from './temporal.js';
export * from './cbor.js';
export * from './schemas.js';
export * from './client.js';
export * from './merge.js';
export * from './inference.js';
export * from './confidence/algebra.js';
export * from './confidence/decay.js';
export * from './data-protection.js';
