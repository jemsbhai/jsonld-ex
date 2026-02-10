/**
 * Example 02: Provenance Tracking
 * ================================
 *
 * Demonstrates full provenance metadata: @source, @extractedAt,
 * @method, and @humanVerified for AI-generated knowledge graphs.
 *
 * Use case: Tracking which ML model produced each assertion, when,
 * and whether a human has verified it.
 *
 * Run: npx ts-node examples/js/02_provenance_tracking.ts
 */

import { annotate, getConfidence, getProvenance, aggregateConfidence } from '../../packages/js/dist';

// ── 1. Full provenance annotation ────────────────────────────────

console.log('=== 1. Full Provenance Annotation ===\n');

// NER model extracts a person name
const name = annotate('Dr. Sarah Chen', {
  confidence: 0.93,
  source: 'https://models.example.org/ner-bert-large-v4',
  extractedAt: new Date().toISOString(),
  method: 'NER',
  humanVerified: false,
});
console.log('Annotated name:');
console.log(JSON.stringify(name, null, 2));

// Relation extraction model identifies employer
const employer = annotate('MIT', {
  confidence: 0.76,
  source: 'https://models.example.org/rel-extract-v2',
  extractedAt: new Date().toISOString(),
  method: 'relation-extraction',
  humanVerified: false,
});

// ── 2. Extracting provenance metadata ────────────────────────────

console.log('\n=== 2. Extracting Provenance ===\n');

const prov = getProvenance(name);
console.log(`Confidence:     ${prov.confidence}`);
console.log(`Source model:   ${prov.source}`);
console.log(`Extracted at:   ${prov.extractedAt}`);
console.log(`Method:         ${prov.method}`);
console.log(`Human verified: ${prov.humanVerified}`);

// ── 3. Multi-model provenance pipeline ───────────────────────────

console.log('\n=== 3. Multi-Model Pipeline ===\n');

const entityPipeline = {
  '@context': 'http://schema.org/',
  '@type': 'Person',
  '@id': 'http://example.org/entity/sarah-chen',
  name: annotate('Dr. Sarah Chen', {
    confidence: 0.93,
    source: 'https://models.example.org/ner-v4',
    method: 'NER',
  }),
  sameAs: annotate('https://www.wikidata.org/wiki/Q12345', {
    confidence: 0.81,
    source: 'https://models.example.org/entity-linker-v2',
    method: 'entity-linking',
  }),
  description: annotate('Renowned AI researcher known for work in NLP', {
    confidence: 0.67,
    source: 'https://models.example.org/summarizer-v1',
    method: 'abstractive-summarization',
  }),
};

console.log('Multi-model knowledge graph node:');
console.log(JSON.stringify(entityPipeline, null, 2));

// ── 4. Human verification workflow ───────────────────────────────

console.log('\n=== 4. Human Verification Workflow ===\n');

// Before verification
const unverified = annotate('Machine Learning Engineer', {
  confidence: 0.72,
  source: 'https://models.example.org/role-classifier-v1',
  method: 'classification',
  humanVerified: false,
});
console.log(`Before review: confidence=${unverified['@confidence']}, verified=${unverified['@humanVerified']}`);

// After human verifies and corrects
const verified = annotate('Senior ML Engineer', {
  confidence: 1.0,
  source: 'https://models.example.org/role-classifier-v1',
  method: 'classification',
  humanVerified: true,
});
console.log(`After review:  confidence=${verified['@confidence']}, verified=${verified['@humanVerified']}`);

// ── 5. Comparing provenance across sources ───────────────────────

console.log('\n=== 5. Comparing Sources ===\n');

const extractions = [
  annotate('MIT', { confidence: 0.91, source: 'model-A', method: 'NER' }),
  annotate('MIT', { confidence: 0.85, source: 'model-B', method: 'NER' }),
  annotate('Massachusetts Institute of Technology', {
    confidence: 0.78,
    source: 'model-C',
    method: 'entity-resolution',
  }),
];

for (let i = 0; i < extractions.length; i++) {
  const ext = extractions[i];
  const p = getProvenance(ext);
  console.log(`  Source ${i + 1}: '${ext['@value']}' (conf=${p.confidence}, model=${p.source}, method=${p.method})`);
}

// Aggregate confidence from all models
const scores = extractions.map((e) => getProvenance(e).confidence!);
const combined = aggregateConfidence(scores, 'mean');
console.log(`\n  Combined confidence: ${combined.toFixed(4)}`);
