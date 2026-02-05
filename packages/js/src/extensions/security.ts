/**
 * Security Extensions for JSON-LD
 *
 * Provides:
 * - @integrity: SHA-256/384/512 hash verification for contexts
 * - Context allowlists: Restrict which remote contexts can be loaded
 * - Resource limits: Prevent DoS via recursion/exhaustion
 */

import * as crypto from 'crypto';
import { KEYWORD_INTEGRITY } from '../keywords';
import { IntegrityContext, ResourceLimits, ContextAllowlist } from '../types';

// ── Default Resource Limits ───────────────────────────────────────

export const DEFAULT_RESOURCE_LIMITS: Required<ResourceLimits> = {
  maxContextDepth: 10,
  maxGraphDepth: 100,
  maxDocumentSize: 10 * 1024 * 1024, // 10 MB
  maxExpansionTime: 30_000, // 30 seconds
};

// ── Context Integrity ─────────────────────────────────────────────

type HashAlgorithm = 'sha256' | 'sha384' | 'sha512';

/**
 * Computes an integrity hash for a context string/object.
 *
 * @param context - The context content (string or object to be serialized)
 * @param algorithm - Hash algorithm (default: sha256)
 * @returns Integrity string in format "algorithm-base64hash"
 */
export function computeIntegrity(
  context: string | object,
  algorithm: HashAlgorithm = 'sha256'
): string {
  const content = typeof context === 'string'
    ? context
    : JSON.stringify(context);

  const nodeAlg = algorithm === 'sha256' ? 'sha256'
    : algorithm === 'sha384' ? 'sha384'
    : 'sha512';

  const hash = crypto.createHash(nodeAlg).update(content, 'utf8').digest('base64');
  return `${algorithm}-${hash}`;
}

/**
 * Verifies a context's content against its declared integrity hash.
 *
 * @param context - The context content
 * @param declaredIntegrity - The expected integrity string
 * @returns true if the hash matches
 */
export function verifyIntegrity(
  context: string | object,
  declaredIntegrity: string
): boolean {
  const [algorithm, _expectedHash] = declaredIntegrity.split('-', 2) as [string, string];

  if (!['sha256', 'sha384', 'sha512'].includes(algorithm)) {
    throw new Error(`Unsupported hash algorithm: ${algorithm}`);
  }

  const computed = computeIntegrity(context, algorithm as HashAlgorithm);
  return computed === declaredIntegrity;
}

/**
 * Creates a context reference with integrity verification.
 */
export function integrityContext(
  contextUrl: string,
  contextContent: string | object,
  algorithm: HashAlgorithm = 'sha256'
): IntegrityContext {
  return {
    '@id': contextUrl,
    '@integrity': computeIntegrity(contextContent, algorithm),
  };
}

// ── Context Allowlist ─────────────────────────────────────────────

/**
 * Checks whether a context URL is permitted by the allowlist configuration.
 */
export function isContextAllowed(
  contextUrl: string,
  config: ContextAllowlist
): boolean {
  if (config.blockRemoteContexts) {
    return false;
  }

  // Check exact matches
  if (config.allowed?.includes(contextUrl)) {
    return true;
  }

  // Check patterns
  if (config.patterns) {
    for (const pattern of config.patterns) {
      if (typeof pattern === 'string') {
        // Simple glob: convert * to regex
        const regex = new RegExp(
          '^' + pattern.replace(/\*/g, '.*').replace(/\?/g, '.') + '$'
        );
        if (regex.test(contextUrl)) return true;
      } else if (pattern instanceof RegExp) {
        if (pattern.test(contextUrl)) return true;
      }
    }
  }

  // If allowlist is configured but URL didn't match, deny
  if ((config.allowed && config.allowed.length > 0) ||
      (config.patterns && config.patterns.length > 0)) {
    return false;
  }

  // No allowlist configured = allow all
  return true;
}

/**
 * Creates a secure document loader that enforces allowlists and integrity.
 *
 * Wraps an existing jsonld.js document loader with security checks.
 *
 * @param baseLoader - The underlying document loader (from jsonld.js)
 * @param config - Allowlist configuration
 * @param integrityMap - Optional map of context URL → expected integrity hash
 */
export function createSecureDocumentLoader(
  baseLoader: (url: string) => Promise<any>,
  config: ContextAllowlist,
  integrityMap?: Map<string, string>
): (url: string) => Promise<any> {
  return async (url: string) => {
    // Enforce allowlist
    if (!isContextAllowed(url, config)) {
      throw new Error(
        `Context URL blocked by allowlist: ${url}. ` +
        `Allowed: ${JSON.stringify(config.allowed ?? [])}`
      );
    }

    // Load the document
    const result = await baseLoader(url);

    // Verify integrity if declared
    if (integrityMap?.has(url)) {
      const expected = integrityMap.get(url)!;
      const content = typeof result.document === 'string'
        ? result.document
        : JSON.stringify(result.document);

      if (!verifyIntegrity(content, expected)) {
        throw new Error(
          `Context integrity verification failed for ${url}. ` +
          `Expected: ${expected}`
        );
      }
    }

    return result;
  };
}

// ── Resource Limits ───────────────────────────────────────────────

/**
 * Validates a document against resource limits before processing.
 *
 * @throws Error if any limit is exceeded
 */
export function enforceResourceLimits(
  document: string | object,
  limits: ResourceLimits = DEFAULT_RESOURCE_LIMITS
): void {
  const resolved = { ...DEFAULT_RESOURCE_LIMITS, ...limits };

  // Check document size
  const content = typeof document === 'string'
    ? document
    : JSON.stringify(document);

  if (content.length > resolved.maxDocumentSize) {
    throw new Error(
      `Document size ${content.length} bytes exceeds limit of ${resolved.maxDocumentSize} bytes`
    );
  }

  // Check graph depth
  const parsed = typeof document === 'string' ? JSON.parse(document) : document;
  const depth = measureDepth(parsed);

  if (depth > resolved.maxGraphDepth) {
    throw new Error(
      `Document nesting depth ${depth} exceeds limit of ${resolved.maxGraphDepth}`
    );
  }
}

/**
 * Creates a processing timeout wrapper.
 */
export function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  operation: string = 'processing'
): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`${operation} timed out after ${timeoutMs}ms`));
    }, timeoutMs);

    promise
      .then((result) => {
        clearTimeout(timer);
        resolve(result);
      })
      .catch((err) => {
        clearTimeout(timer);
        reject(err);
      });
  });
}

// ── Helpers ───────────────────────────────────────────────────────

function measureDepth(obj: any, current: number = 0): number {
  if (obj == null || typeof obj !== 'object') return current;

  let maxDepth = current;

  if (Array.isArray(obj)) {
    for (const item of obj) {
      maxDepth = Math.max(maxDepth, measureDepth(item, current + 1));
    }
  } else {
    for (const value of Object.values(obj)) {
      maxDepth = Math.max(maxDepth, measureDepth(value, current + 1));
    }
  }

  return maxDepth;
}
