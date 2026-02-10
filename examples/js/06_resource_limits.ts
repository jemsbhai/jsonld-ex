/**
 * Example 06: Resource Limits
 * ============================
 *
 * Demonstrates resource limit enforcement to prevent denial-of-service
 * attacks via deeply nested documents or oversized payloads.
 *
 * Use case: API gateway protecting JSON-LD processing endpoints.
 *
 * Run: npx ts-node examples/js/06_resource_limits.ts
 */

import { JsonLdEx, enforceResourceLimits, DEFAULT_RESOURCE_LIMITS } from '../../packages/js/dist';

// ── 1. Default resource limits ───────────────────────────────────

console.log('=== 1. Default Resource Limits ===\n');

for (const [key, value] of Object.entries(DEFAULT_RESOURCE_LIMITS)) {
  const unit = key.includes('Size') ? 'bytes' : key.includes('Depth') ? 'levels' : 'ms';
  console.log(`  ${key}: ${value.toLocaleString()} ${unit}`);
}

// ── 2. Document size limit ───────────────────────────────────────

console.log('\n=== 2. Document Size Limit ===\n');

// Normal document — passes
const smallDoc = { '@type': 'Person', name: 'Alice' };
try {
  enforceResourceLimits(smallDoc, { maxDocumentSize: 1024 });
  console.log(`  Small document (${JSON.stringify(smallDoc).length} bytes): ✓ Passed`);
} catch (e: any) {
  console.log(`  Small document: ✗ ${e.message}`);
}

// Oversized document — blocked
const largeDoc = { '@type': 'Person', name: 'A'.repeat(2000) };
try {
  enforceResourceLimits(largeDoc, { maxDocumentSize: 1024 });
  console.log('  Large document: ✓ Passed');
} catch (e: any) {
  console.log(`  Large document (${JSON.stringify(largeDoc).length} bytes): ✗ Blocked`);
  console.log(`    Reason: ${e.message}`);
}

// ── 3. Nesting depth limit ──────────────────────────────────────

console.log('\n=== 3. Nesting Depth Limit ===\n');

// Build a deeply nested document
function buildNested(depth: number): any {
  let doc: any = { '@type': 'Leaf', value: 'data' };
  for (let i = 0; i < depth; i++) {
    doc = { '@type': `Level${depth - i}`, '@graph': [doc] };
  }
  return doc;
}

// Shallow nesting — passes
const shallow = buildNested(5);
try {
  enforceResourceLimits(shallow, { maxGraphDepth: 20 });
  console.log('  Depth 5: ✓ Passed (limit=20)');
} catch (e: any) {
  console.log(`  Depth 5: ✗ ${e.message}`);
}

// Deep nesting — blocked
const deep = buildNested(50);
try {
  enforceResourceLimits(deep, { maxGraphDepth: 20 });
  console.log('  Depth 50: ✓ Passed');
} catch (e: any) {
  console.log('  Depth 50: ✗ Blocked (limit=20)');
  console.log(`    Reason: ${e.message}`);
}

// ── 4. Custom limits for different environments ──────────────────

console.log('\n=== 4. Environment-Specific Limits ===\n');

const environments: Record<string, Record<string, number>> = {
  'IoT / Edge': {
    maxDocumentSize: 64 * 1024,       // 64 KB
    maxGraphDepth: 10,
    maxContextDepth: 3,
    maxExpansionTime: 5_000,
  },
  'Web API': {
    maxDocumentSize: 1 * 1024 * 1024, // 1 MB
    maxGraphDepth: 50,
    maxContextDepth: 10,
    maxExpansionTime: 15_000,
  },
  'Batch Processing': {
    maxDocumentSize: 50 * 1024 * 1024, // 50 MB
    maxGraphDepth: 200,
    maxContextDepth: 20,
    maxExpansionTime: 120_000,
  },
};

for (const [envName, limits] of Object.entries(environments)) {
  console.log(`  ${envName}:`);
  for (const [key, value] of Object.entries(limits)) {
    const unit = key.includes('Size') ? 'bytes' : key.includes('Depth') ? 'levels' : 'ms';
    console.log(`    ${key}: ${value.toLocaleString()} ${unit}`);
  }
  console.log();
}

// ── 5. Using limits with the processor ───────────────────────────

console.log('=== 5. Processor with Resource Limits ===\n');

// Create a processor with IoT limits
const iotProcessor = new JsonLdEx({
  resourceLimits: {
    maxDocumentSize: 64 * 1024,
    maxGraphDepth: 10,
  },
});

// Normal IoT reading — passes
const sensorReading = {
  '@context': 'http://schema.org/',
  '@type': 'Observation',
  value: 23.5,
  unit: 'celsius',
};

try {
  enforceResourceLimits(sensorReading, { maxDocumentSize: 64 * 1024 });
  console.log('  IoT sensor reading: ✓ Within limits');
} catch (e: any) {
  console.log(`  IoT sensor reading: ✗ ${e.message}`);
}

console.log('\n  Resource limits protect against:');
console.log('    • Billion laughs attack (deep nesting)');
console.log('    • Payload bombs (oversized documents)');
console.log('    • Circular context references (context depth)');
console.log('    • Slow-loris attacks (expansion timeout)');
