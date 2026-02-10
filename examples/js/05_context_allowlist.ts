/**
 * Example 05: Context Allowlist
 * ==============================
 *
 * Demonstrates restricting which remote contexts a processor can load,
 * preventing unauthorized context injection.
 *
 * Use case: Enterprise API that only trusts specific context providers.
 *
 * Run: npx ts-node examples/js/05_context_allowlist.ts
 */

import { isContextAllowed } from '../../packages/js/dist';

// ── 1. Exact URL allowlist ───────────────────────────────────────

console.log('=== 1. Exact URL Allowlist ===\n');

const config = {
  allowed: [
    'https://schema.org/',
    'https://w3id.org/security/v2',
    'https://www.w3.org/ns/credentials/v2',
  ],
};

const testUrls1 = [
  'https://schema.org/',
  'https://w3id.org/security/v2',
  'https://malicious.example.org/fake-context',
  'http://schema.org/', // HTTP, not HTTPS
];

for (const url of testUrls1) {
  const allowed = isContextAllowed(url, config);
  const status = allowed ? '✓ ALLOWED' : '✗ BLOCKED';
  console.log(`  ${status}: ${url}`);
}

// ── 2. Pattern-based allowlist ───────────────────────────────────

console.log('\n=== 2. Pattern-Based Allowlist ===\n');

const configPatterns = {
  allowed: ['https://schema.org/'],
  patterns: [
    'https://api.company.example.org/contexts/*',
    'https://w3id.org/*',
  ],
};

const testUrls2 = [
  'https://schema.org/',
  'https://api.company.example.org/contexts/payment-v2',
  'https://api.company.example.org/contexts/user-profile',
  'https://w3id.org/security/v2',
  'https://w3id.org/vc/status-list/2021',
  'https://evil.example.org/contexts/steal-data',
];

for (const url of testUrls2) {
  const allowed = isContextAllowed(url, configPatterns);
  const status = allowed ? '✓ ALLOWED' : '✗ BLOCKED';
  console.log(`  ${status}: ${url}`);
}

// ── 3. Block all remote contexts ─────────────────────────────────

console.log('\n=== 3. Block All Remote Contexts ===\n');

const configBlocked = { blockRemoteContexts: true };

const testUrls3 = [
  'https://schema.org/',
  'https://trusted.example.org/context',
];

for (const url of testUrls3) {
  const allowed = isContextAllowed(url, configBlocked);
  const status = allowed ? '✓ ALLOWED' : '✗ BLOCKED';
  console.log(`  ${status}: ${url}`);
}

console.log('\n  Use case: Offline processing or air-gapped environments.');
console.log('  All contexts must be provided inline or from local cache.');

// ── 4. No allowlist (permissive mode) ────────────────────────────

console.log('\n=== 4. No Allowlist (Permissive) ===\n');

const configOpen = {};

const testUrls4 = [
  'https://schema.org/',
  'https://anything.example.org/any-context',
];

for (const url of testUrls4) {
  const allowed = isContextAllowed(url, configOpen);
  const status = allowed ? '✓ ALLOWED' : '✗ BLOCKED';
  console.log(`  ${status}: ${url}`);
}

console.log('\n  Warning: Permissive mode trusts all remote contexts.');
console.log('  Recommended only for development, not production.');

// ── 5. Enterprise security profile ──────────────────────────────

console.log('\n=== 5. Enterprise Security Profile ===\n');

const enterpriseConfig = {
  allowed: [
    'https://schema.org/',
    'https://www.w3.org/ns/credentials/v2',
  ],
  patterns: [
    'https://api.acmecorp.com/contexts/*',
    'https://internal.acmecorp.com/ontology/*',
  ],
};

const scenarios: Array<[string, string]> = [
  ['Public schema.org context', 'https://schema.org/'],
  ['Internal payment context', 'https://api.acmecorp.com/contexts/payment'],
  ['Internal ontology', 'https://internal.acmecorp.com/ontology/hr-v3'],
  ['Phishing attempt', 'https://api.acmecorp.com.evil.org/contexts/payment'],
  ["Competitor's context", 'https://api.competitor.com/contexts/product'],
];

for (const [desc, url] of scenarios) {
  const allowed = isContextAllowed(url, enterpriseConfig);
  const status = allowed ? '✓ ALLOWED' : '✗ BLOCKED';
  console.log(`  ${status}: ${desc}`);
  console.log(`           ${url}\n`);
}
