/**
 * Example 04: Context Integrity Verification
 * ============================================
 *
 * Demonstrates @integrity for cryptographic hash verification of
 * JSON-LD contexts, preventing context injection attacks.
 *
 * Use case: Financial API ensuring context hasn't been tampered with.
 *
 * Run: npx ts-node examples/js/04_context_integrity.ts
 */

import { computeIntegrity, verifyIntegrity, integrityContext } from '../../packages/js/dist';

// ── 1. Computing integrity hashes ────────────────────────────────

console.log('=== 1. Computing Integrity Hashes ===\n');

// A trusted context definition
const schemaContext = {
  '@vocab': 'http://schema.org/',
  name: 'http://schema.org/name',
  amount: { '@id': 'http://schema.org/amount', '@type': 'xsd:decimal' },
  currency: 'http://schema.org/currency',
};

// Compute hash using different algorithms
const sha256Hash = computeIntegrity(schemaContext, 'sha256');
const sha384Hash = computeIntegrity(schemaContext, 'sha384');
const sha512Hash = computeIntegrity(schemaContext, 'sha512');

console.log(`SHA-256: ${sha256Hash}`);
console.log(`SHA-384: ${sha384Hash}`);
console.log(`SHA-512: ${sha512Hash}`);

// ── 2. Verifying integrity ───────────────────────────────────────

console.log('\n=== 2. Verifying Integrity ===\n');

// Verify the context matches its hash
const isValid = verifyIntegrity(schemaContext, sha256Hash);
console.log(`Original context valid: ${isValid}`); // true

// Simulate a tampered context (attacker swaps field mappings)
const tamperedContext = {
  ...schemaContext,
  source: 'http://schema.org/amount',       // Swapped!
  destination: 'http://schema.org/currency', // Swapped!
};

const isValidTampered = verifyIntegrity(tamperedContext, sha256Hash);
console.log(`Tampered context valid: ${isValidTampered}`); // false — attack detected!

// ── 3. Creating integrity-protected context references ───────────

console.log('\n=== 3. Integrity-Protected Context ===\n');

const protectedRef = integrityContext(
  'https://api.bank.example.org/contexts/payment-v2',
  schemaContext,
  'sha256',
);
console.log(`Protected reference: ${JSON.stringify(protectedRef, null, 2)}`);

// ── 4. Using integrity in a document ─────────────────────────────

console.log('\n=== 4. Document with Integrity ===\n');

const document = {
  '@context': {
    '@id': 'https://api.bank.example.org/contexts/payment-v2',
    '@integrity': sha256Hash,
  },
  '@type': 'MoneyTransfer',
  source: 'https://bank.example.org/account/alice',
  destination: 'https://bank.example.org/account/bob',
  amount: '500.00',
  currency: 'USD',
};

console.log(JSON.stringify(document, null, 2));

// ── 5. Attack scenario demonstration ─────────────────────────────

console.log('\n=== 5. Attack Scenario ===\n');

console.log('Scenario: Attacker performs DNS poisoning to redirect context URL');
console.log('  to a malicious context that swaps \'source\' and \'destination\'.\n');

// The legitimate context
const legitimate = { source: 'http://schema.org/sender', destination: 'http://schema.org/recipient' };
const legitimateHash = computeIntegrity(legitimate);
console.log(`Legitimate hash: ${legitimateHash}`);

// The attacker's context (swapped mappings)
const malicious = { source: 'http://schema.org/recipient', destination: 'http://schema.org/sender' };
const maliciousHash = computeIntegrity(malicious);
console.log(`Malicious hash:  ${maliciousHash}`);

// Verification catches the attack
const attackDetected = !verifyIntegrity(malicious, legitimateHash);
console.log(`\nAttack detected: ${attackDetected}`);
console.log('Result: Processor rejects the tampered context → transaction blocked.');
