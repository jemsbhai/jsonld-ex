/**
 * Example 08: Document Validation
 * =================================
 *
 * Demonstrates validating entire JSON-LD graphs containing multiple
 * node types against multiple shape definitions.
 *
 * Use case: Validating an organization's knowledge graph before import.
 *
 * Run: npx ts-node examples/js/08_document_validation.ts
 */

import { validateNode, validateDocument, ShapeDefinition } from '../../packages/js/dist';

// ── 1. Define multiple shapes ────────────────────────────────────

console.log('=== 1. Multiple Shape Definitions ===\n');

const personShape: ShapeDefinition = {
  '@type': 'Person',
  name: { '@required': true, '@type': 'xsd:string', '@minLength': 1 },
  email: { '@required': true, '@pattern': '^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$' },
};

const organizationShape: ShapeDefinition = {
  '@type': 'Organization',
  name: { '@required': true, '@type': 'xsd:string' },
  url: { '@required': true, '@type': 'xsd:string', '@pattern': '^https?://' },
  employeeCount: { '@type': 'xsd:integer', '@minimum': 1 },
};

const productShape: ShapeDefinition = {
  '@type': 'Product',
  name: { '@required': true, '@type': 'xsd:string' },
  price: { '@type': 'xsd:double', '@minimum': 0 },
  sku: { '@required': true, '@type': 'xsd:string', '@pattern': '^[A-Z]{2}-\\d{4}$' },
};

const shapes = [personShape, organizationShape, productShape];
console.log(`Defined ${shapes.length} shapes: Person, Organization, Product`);

// ── 2. Valid document ────────────────────────────────────────────

console.log('\n=== 2. Valid Document ===\n');

const validDoc = {
  '@graph': [
    {
      '@id': '#alice',
      '@type': 'Person',
      name: 'Alice Johnson',
      email: 'alice@acme.com',
    },
    {
      '@id': '#acme',
      '@type': 'Organization',
      name: 'Acme Corp',
      url: 'https://acme.example.com',
      employeeCount: 500,
    },
    {
      '@id': '#widget',
      '@type': 'Product',
      name: 'Super Widget',
      price: 29.99,
      sku: 'SW-1234',
    },
  ],
};

let result = validateDocument(validDoc, shapes);
console.log(`Valid: ${result.valid}`);
console.log(`Errors: ${result.errors.length}`);

// ── 3. Document with errors across node types ────────────────────

console.log('\n=== 3. Document with Errors ===\n');

const invalidDoc = {
  '@graph': [
    {
      '@id': '#bob',
      '@type': 'Person',
      name: 'Bob',
      // Missing required email
    },
    {
      '@id': '#badcorp',
      '@type': 'Organization',
      name: 'Bad Corp',
      url: 'not-a-url',        // Doesn't match URL pattern
      employeeCount: 0,         // Below minimum
    },
    {
      '@id': '#product1',
      '@type': 'Product',
      name: 'Gadget',
      price: -5.00,             // Negative price
      sku: 'invalid-sku',       // Doesn't match SKU pattern
    },
    {
      '@id': '#validperson',
      '@type': 'Person',
      name: 'Charlie Davis',
      email: 'charlie@example.com',
      // This one is valid
    },
  ],
};

result = validateDocument(invalidDoc, shapes);
console.log(`Valid: ${result.valid}`);
console.log(`Total errors: ${result.errors.length}\n`);

for (const err of result.errors) {
  console.log(`  ✗ ${err.path}`);
  console.log(`    [${err.constraint}] ${err.message}\n`);
}

// ── 4. Selective validation (only some types) ────────────────────

console.log('=== 4. Selective Validation ===\n');

// Only validate Person nodes
result = validateDocument(invalidDoc, [personShape]);
console.log('Person-only validation:');
console.log(`  Errors: ${result.errors.length}`);
for (const err of result.errors) {
  console.log(`  ✗ ${err.path}: ${err.message}`);
}

// ── 5. Mixed valid and invalid with summary ─────────────────────

console.log('\n=== 5. Validation Summary Report ===\n');

const largeDoc = {
  '@graph': [
    ...Array.from({ length: 5 }, (_, i) => ({
      '@type': 'Person',
      '@id': `#p${i}`,
      name: `Person ${i}`,
      email: `p${i}@x.com`,
    })),
    { '@type': 'Person', '@id': '#bad1', name: '' },           // Empty name
    { '@type': 'Person', '@id': '#bad2', email: 'x@y.com' },   // Missing name
    { '@type': 'Person', '@id': '#bad3', name: 'Valid', email: 'invalid' },
  ],
};

result = validateDocument(largeDoc, [personShape]);

const totalNodes = largeDoc['@graph'].length;
const errorNodeIds = new Set(result.errors.map((e) => e.path.split('/')[0]));
const errorNodes = errorNodeIds.size;

console.log(`  Total nodes:   ${totalNodes}`);
console.log(`  Valid nodes:   ${totalNodes - errorNodes}`);
console.log(`  Invalid nodes: ${errorNodes}`);
console.log(`  Total errors:  ${result.errors.length}`);
console.log(`  Pass rate:     ${Math.round(((totalNodes - errorNodes) / totalNodes) * 100)}%`);

if (result.errors.length > 0) {
  console.log('\n  Error breakdown:');
  const constraintCounts: Record<string, number> = {};
  for (const err of result.errors) {
    constraintCounts[err.constraint] = (constraintCounts[err.constraint] || 0) + 1;
  }
  const sorted = Object.entries(constraintCounts).sort((a, b) => b[1] - a[1]);
  for (const [constraint, count] of sorted) {
    console.log(`    ${constraint}: ${count}`);
  }
}
