/**
 * @jsonld-ex/core — JSON-LD 1.2 Extensions
 *
 * Reference implementation of proposed JSON-LD 1.2 extensions
 * for AI/ML data exchange, security hardening, and validation.
 */

// Main processor
export { JsonLdEx } from './processor';

// Types
export * from './types';

// Keywords and namespace
export * from './keywords';

// Extensions (for direct access)
export * from './extensions/ai-ml';
export * from './extensions/vector';
export * from './extensions/security';
export * from './extensions/validation';
export * from './temporal';
export * from './cbor';
