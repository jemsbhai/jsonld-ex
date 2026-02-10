/**
 * Example 07: Shape Validation
 * ==============================
 *
 * Demonstrates @shape for native JSON-LD validation with all
 * constraint types: required, type, min/max, length, and pattern.
 *
 * Use case: API input validation for a user registration endpoint.
 *
 * Run: npx ts-node examples/js/07_shape_validation.ts
 */

import { validateNode, ShapeDefinition, ValidationResult } from '../../packages/js/dist';

// ── 1. Defining shapes ──────────────────────────────────────────

console.log('=== 1. Shape Definitions ===\n');

const personShape: ShapeDefinition = {
  '@type': 'Person',
  name: {
    '@required': true,
    '@type': 'xsd:string',
    '@minLength': 1,
    '@maxLength': 100,
  },
  email: {
    '@required': true,
    '@type': 'xsd:string',
    '@pattern': '^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$',
  },
  age: {
    '@type': 'xsd:integer',
    '@minimum': 0,
    '@maximum': 150,
  },
  bio: {
    '@type': 'xsd:string',
    '@maxLength': 500,
  },
};

console.log('Person shape constraints:');
for (const [prop, constraint] of Object.entries(personShape)) {
  if (prop === '@type') {
    console.log(`  Type: ${constraint}`);
  } else if (typeof constraint === 'object' && constraint !== null) {
    const parts = Object.entries(constraint).map(([k, v]) => `${k}=${v}`);
    console.log(`  ${prop}: ${parts.join(', ')}`);
  }
}

// ── 2. Valid node ────────────────────────────────────────────────

console.log('\n=== 2. Valid Node ===\n');

const validPerson = {
  '@type': 'Person',
  name: 'Alice Johnson',
  email: 'alice@example.com',
  age: 30,
  bio: 'Software engineer who loves open source.',
};

let result = validateNode(validPerson, personShape);
console.log(`Valid: ${result.valid}`);
console.log(`Errors: ${result.errors.length}`);

// ── 3. Required field missing ────────────────────────────────────

console.log('\n=== 3. Required Field Missing ===\n');

const missingEmail = {
  '@type': 'Person',
  name: 'Bob Smith',
  age: 25,
};

result = validateNode(missingEmail, personShape);
console.log(`Valid: ${result.valid}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 4. Type mismatches ──────────────────────────────────────────

console.log('\n=== 4. Type Mismatches ===\n');

const badTypes = {
  '@type': 'Person',
  name: 12345,          // Should be string
  email: 'a@b.com',
  age: 'thirty',        // Should be integer
};

result = validateNode(badTypes, personShape);
console.log(`Valid: ${result.valid}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 5. Range violations ─────────────────────────────────────────

console.log('\n=== 5. Range Violations ===\n');

const outOfRange = {
  '@type': 'Person',
  name: 'Charlie',
  email: 'c@d.com',
  age: -5, // Below minimum
};

result = validateNode(outOfRange, personShape);
console.log(`Valid: ${result.valid}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// Over maximum
const tooOld = {
  '@type': 'Person',
  name: 'Methuselah',
  email: 'm@bible.org',
  age: 969, // Above maximum
};

result = validateNode(tooOld, personShape);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 6. String length violations ──────────────────────────────────

console.log('\n=== 6. String Length Violations ===\n');

const lengthIssues = {
  '@type': 'Person',
  name: '',               // Below minLength
  email: 'a@b.com',
  bio: 'x'.repeat(600),  // Exceeds maxLength
};

result = validateNode(lengthIssues, personShape);
console.log(`Valid: ${result.valid}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 7. Pattern violations ───────────────────────────────────────

console.log('\n=== 7. Pattern Violations ===\n');

const badPatterns = {
  '@type': 'Person',
  name: 'Diana',
  email: 'not-an-email', // Doesn't match email pattern
};

result = validateNode(badPatterns, personShape);
console.log(`Valid: ${result.valid}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 8. Multiple errors at once ───────────────────────────────────

console.log('\n=== 8. Multiple Errors ===\n');

const everythingWrong = {
  '@type': 'Organization', // Wrong type
  age: -10,                // Below minimum, name missing, email missing
};

result = validateNode(everythingWrong, personShape);
console.log(`Valid: ${result.valid}`);
console.log(`Total errors: ${result.errors.length}`);
for (const err of result.errors) {
  console.log(`  ✗ [${err.constraint}] ${err.path}: ${err.message}`);
}

// ── 9. JSON-LD @value wrapped values ────────────────────────────

console.log('\n=== 9. @value Wrapped Values ===\n');

const wrappedNode = {
  '@type': 'Person',
  name: { '@value': 'Eve Nakamura' },
  email: { '@value': 'eve@example.com' },
  age: { '@value': 28 },
};

result = validateNode(wrappedNode, personShape);
console.log(`Valid: ${result.valid}`);
console.log('  @shape validation correctly unwraps @value objects.');
