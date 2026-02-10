/**
 * Example 01: Confidence Basics
 * =============================
 *
 * Demonstrates how to annotate JSON-LD values with confidence scores,
 * extract them, filter graphs by confidence, and aggregate scores.
 *
 * Use case: An NER model extracts person names from text with varying certainty.
 *
 * Run: npx ts-node examples/js/01_confidence_basics.ts
 */

import { annotate, getConfidence, filterByConfidence, aggregateConfidence } from '../../packages/js/dist';

// ── 1. Annotating values with confidence ─────────────────────────

console.log('=== 1. Annotating Values ===\n');

// High-confidence extraction
const nameHigh = annotate('John Smith', { confidence: 0.95 });
console.log('High confidence:', JSON.stringify(nameHigh));
// { "@value": "John Smith", "@confidence": 0.95 }

// Low-confidence extraction
const nameLow = annotate('J. Smith', { confidence: 0.45 });
console.log('Low confidence: ', JSON.stringify(nameLow));

// Perfect confidence (human-entered data)
const nameExact = annotate('Jane Doe', { confidence: 1.0 });
console.log('Exact value:    ', JSON.stringify(nameExact));

// ── 2. Extracting confidence from nodes ──────────────────────────

console.log('\n=== 2. Extracting Confidence ===\n');

const node = {
  '@type': 'Person',
  name: { '@value': 'John Smith', '@confidence': 0.95 },
  email: { '@value': 'john@example.com', '@confidence': 0.72 },
  age: 30, // No confidence — human-entered
};

const nameConf = getConfidence(node.name);
const emailConf = getConfidence(node.email);
const ageConf = getConfidence(node.age);

console.log(`Name confidence:  ${nameConf}`);   // 0.95
console.log(`Email confidence: ${emailConf}`);  // 0.72
console.log(`Age confidence:   ${ageConf}`);    // undefined (no annotation)

// ── 3. Filtering a graph by confidence threshold ─────────────────

console.log('\n=== 3. Filtering by Confidence ===\n');

const knowledgeGraph = [
  { '@id': '#person1', name: annotate('Alice Johnson', { confidence: 0.97 }) },
  { '@id': '#person2', name: annotate('Bob Smith', { confidence: 0.62 }) },
  { '@id': '#person3', name: annotate('C. Williams', { confidence: 0.31 }) },
  { '@id': '#person4', name: annotate('Diana Prince', { confidence: 0.88 }) },
  { '@id': '#person5', name: annotate('E. Unknown', { confidence: 0.15 }) },
];

// Keep only high-confidence extractions (>= 0.7)
const highConf = filterByConfidence(knowledgeGraph, 'name', 0.7);
console.log(`Nodes with confidence >= 0.7: ${highConf.length}`);
for (const n of highConf) {
  console.log(`  ${n['@id']}: ${(n.name as any)['@value']}`);
}

// Medium confidence threshold (>= 0.5)
const mediumConf = filterByConfidence(knowledgeGraph, 'name', 0.5);
console.log(`\nNodes with confidence >= 0.5: ${mediumConf.length}`);

// ── 4. Aggregating confidence scores ─────────────────────────────

console.log('\n=== 4. Aggregating Confidence ===\n');

// Multiple models extracted the same entity with different confidences
const modelScores = [0.92, 0.87, 0.78];

const meanConf = aggregateConfidence(modelScores, 'mean');
const maxConf = aggregateConfidence(modelScores, 'max');
const minConf = aggregateConfidence(modelScores, 'min');

console.log(`Mean confidence:  ${meanConf.toFixed(4)}`);
console.log(`Max confidence:   ${maxConf.toFixed(4)}`);
console.log(`Min confidence:   ${minConf.toFixed(4)}`);

// Weighted aggregation (trust some models more)
const weighted = aggregateConfidence(modelScores, 'weighted', [3.0, 2.0, 1.0]);
console.log(`Weighted (3:2:1): ${weighted.toFixed(4)}`);

// ── 5. Confidence in a full JSON-LD document ─────────────────────

console.log('\n=== 5. Full Document Example ===\n');

const document = {
  '@context': 'http://schema.org/',
  '@type': 'Person',
  '@id': 'http://example.org/person/12345',
  name: annotate('John Smith', { confidence: 0.95 }),
  jobTitle: annotate('Software Engineer', { confidence: 0.78 }),
  worksFor: {
    '@type': 'Organization',
    name: annotate('Acme Corp', { confidence: 0.85 }),
  },
};

console.log('Document:');
console.log(JSON.stringify(document, null, 2));
