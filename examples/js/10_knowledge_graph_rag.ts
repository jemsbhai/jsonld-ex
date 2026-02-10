/**
 * Example 10: Knowledge Graph RAG Pipeline
 * ==========================================
 *
 * End-to-end use case demonstrating how jsonld-ex extensions support
 * Retrieval-Augmented Generation (RAG) pipelines that build, validate,
 * and query knowledge graphs extracted by LLMs.
 *
 * Pipeline:  Source documents → LLM extraction → JSON-LD with provenance
 *            → Validation → Embedding → Similarity retrieval → Answer
 *
 * Run: npx ts-node examples/js/10_knowledge_graph_rag.ts
 */

import {
  JsonLdEx, annotate, getConfidence, getProvenance,
  filterByConfidence, aggregateConfidence,
  validateNode, validateDocument,
  validateVector, cosineSimilarity,
  vectorTermDefinition, extractVectors, stripVectorsForRdf,
  computeIntegrity, verifyIntegrity,
  ShapeDefinition,
} from '../../packages/js/dist';

const processor = new JsonLdEx();

// ══════════════════════════════════════════════════════════════════
// Stage 1: LLM-Based Entity Extraction
// ══════════════════════════════════════════════════════════════════

console.log('='.repeat(60));
console.log('STAGE 1: LLM Entity Extraction');
console.log('='.repeat(60) + '\n');

const MODEL_V1 = 'https://models.example.org/gpt-extract-v1';
const MODEL_V2 = 'https://models.example.org/llama-extract-v2';
const NOW = new Date().toISOString();

// Two models extract entities from the same document
const extractionsModel1: any[] = [
  {
    '@type': 'Person',
    '@id': '#geoffrey-hinton',
    name: annotate('Geoffrey Hinton', {
      confidence: 0.98, source: MODEL_V1, method: 'NER', extractedAt: NOW,
    }),
    affiliation: annotate('University of Toronto', {
      confidence: 0.92, source: MODEL_V1, method: 'relation-extraction',
    }),
    field: annotate('Deep Learning', {
      confidence: 0.96, source: MODEL_V1, method: 'classification',
    }),
  },
  {
    '@type': 'ScholarlyArticle',
    '@id': '#attention-paper',
    name: annotate('Attention Is All You Need', {
      confidence: 0.99, source: MODEL_V1, method: 'title-extraction',
    }),
    author: annotate('Vaswani et al.', {
      confidence: 0.95, source: MODEL_V1, method: 'NER',
    }),
    year: annotate('2017', {
      confidence: 0.97, source: MODEL_V1, method: 'date-extraction',
    }),
    topic: annotate('Transformer Architecture', {
      confidence: 0.91, source: MODEL_V1, method: 'topic-classification',
    }),
  },
  {
    '@type': 'Organization',
    '@id': '#google-brain',
    name: annotate('Google Brain', {
      confidence: 0.94, source: MODEL_V1, method: 'NER',
    }),
    parentOrganization: annotate('Google', {
      confidence: 0.88, source: MODEL_V1, method: 'relation-extraction',
    }),
  },
];

const extractionsModel2: any[] = [
  {
    '@type': 'Person',
    '@id': '#geoffrey-hinton',
    name: annotate('Geoffrey E. Hinton', {
      confidence: 0.95, source: MODEL_V2, method: 'NER', extractedAt: NOW,
    }),
    affiliation: annotate('University of Toronto', {
      confidence: 0.89, source: MODEL_V2, method: 'relation-extraction',
    }),
    field: annotate('Neural Networks', {
      confidence: 0.90, source: MODEL_V2, method: 'classification',
    }),
  },
];

console.log(`Model 1 (${MODEL_V1}): ${extractionsModel1.length} entities`);
console.log(`Model 2 (${MODEL_V2}): ${extractionsModel2.length} entities`);

// ══════════════════════════════════════════════════════════════════
// Stage 2: Cross-Model Confidence Aggregation
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 2: Cross-Model Confidence Aggregation');
console.log('='.repeat(60) + '\n');

const m1Name = getConfidence(extractionsModel1[0].name)!;
const m2Name = getConfidence(extractionsModel2[0].name)!;

const combined = aggregateConfidence([m1Name, m2Name], 'mean');
console.log(`Hinton name confidence — Model 1: ${m1Name}, Model 2: ${m2Name}`);
console.log(`Aggregated (mean): ${combined.toFixed(4)}`);

const m1Affil = getConfidence(extractionsModel1[0].affiliation)!;
const m2Affil = getConfidence(extractionsModel2[0].affiliation)!;
const combinedAffil = aggregateConfidence([m1Affil, m2Affil], 'mean');
console.log(`\nAffiliation — Model 1: ${m1Affil}, Model 2: ${m2Affil}`);
console.log(`Aggregated (mean): ${combinedAffil.toFixed(4)}`);

// ══════════════════════════════════════════════════════════════════
// Stage 3: Validation
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 3: Schema Validation');
console.log('='.repeat(60) + '\n');

const personShape: ShapeDefinition = {
  '@type': 'Person',
  name: { '@required': true, '@type': 'xsd:string', '@minLength': 1 },
  affiliation: { '@type': 'xsd:string' },
};

const articleShape: ShapeDefinition = {
  '@type': 'ScholarlyArticle',
  name: { '@required': true, '@type': 'xsd:string' },
  author: { '@required': true, '@type': 'xsd:string' },
  year: { '@required': true, '@type': 'xsd:string', '@pattern': '^\\d{4}$' },
};

const orgShape: ShapeDefinition = {
  '@type': 'Organization',
  name: { '@required': true, '@type': 'xsd:string' },
};

const shapes = [personShape, articleShape, orgShape];

const kgDoc = { '@graph': extractionsModel1 };
let result = validateDocument(kgDoc, shapes);

console.log(`Validation result: ${result.valid ? '✓ PASS' : '✗ FAIL'}`);
console.log(`Errors: ${result.errors.length}`);
for (const err of result.errors) {
  console.log(`  ✗ ${err.path}: ${err.message}`);
}

// ══════════════════════════════════════════════════════════════════
// Stage 4: Add Vector Embeddings
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 4: Vector Embeddings');
console.log('='.repeat(60) + '\n');

// Simulated embeddings (4D for demo; real would be 768+)
const entityEmbeddings: Record<string, number[]> = {
  '#geoffrey-hinton': [0.82, 0.45, -0.31, 0.67],
  '#attention-paper': [0.71, 0.52, -0.18, 0.73],
  '#google-brain':    [0.55, 0.38, 0.12, 0.80],
};

// Attach embeddings to entities
for (const entity of extractionsModel1) {
  const eid = entity['@id'];
  if (eid in entityEmbeddings) {
    const vec = entityEmbeddings[eid];
    const { valid, errors } = validateVector(vec, 4);
    if (valid) {
      entity.embedding = vec;
      console.log(`  ✓ Attached 4D embedding to ${eid}`);
    } else {
      console.log(`  ✗ Invalid embedding for ${eid}: ${errors}`);
    }
  }
}

// ══════════════════════════════════════════════════════════════════
// Stage 5: Similarity-Based Retrieval (RAG Query)
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 5: RAG Query — Similarity Retrieval');
console.log('='.repeat(60) + '\n');

const queryEmbedding = [0.78, 0.50, -0.25, 0.70];
console.log(`Query embedding: ${JSON.stringify(queryEmbedding)}`);
console.log("Query: 'Who works on deep learning architectures?'\n");

interface RetrievalResult {
  id: string;
  type: string;
  name: string;
  similarity: number;
  confidence: number | undefined;
}

const results: RetrievalResult[] = [];
for (const entity of extractionsModel1) {
  if (entity.embedding) {
    const sim = cosineSimilarity(queryEmbedding, entity.embedding);
    const nameVal = typeof entity.name === 'object' ? entity.name['@value'] : entity.name;
    const conf = getConfidence(entity.name);
    results.push({
      id: entity['@id'],
      type: entity['@type'],
      name: nameVal,
      similarity: sim,
      confidence: conf,
    });
  }
}

// Rank by similarity
results.sort((a, b) => b.similarity - a.similarity);

console.log('Retrieval results (ranked by cosine similarity):\n');
for (let i = 0; i < results.length; i++) {
  const r = results[i];
  console.log(`  ${i + 1}. ${r.name} (${r.type})`);
  console.log(`     Similarity: ${r.similarity.toFixed(4)}, Confidence: ${r.confidence}`);
}

// ══════════════════════════════════════════════════════════════════
// Stage 6: Confidence-Filtered Answer
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 6: Confidence-Filtered Answer Generation');
console.log('='.repeat(60) + '\n');

const SIMILARITY_THRESHOLD = 0.95;
const CONFIDENCE_THRESHOLD = 0.85;

console.log(`Thresholds — Similarity >= ${SIMILARITY_THRESHOLD}, Confidence >= ${CONFIDENCE_THRESHOLD}\n`);

const relevant = results.filter(
  (r) => r.similarity >= SIMILARITY_THRESHOLD && (r.confidence ?? 0) >= CONFIDENCE_THRESHOLD,
);

if (relevant.length > 0) {
  console.log(`Found ${relevant.length} high-quality result(s):\n`);
  for (const r of relevant) {
    const entity = extractionsModel1.find((e) => e['@id'] === r.id)!;
    const prov = getProvenance(entity.name);

    console.log(`  Entity: ${r.name}`);
    console.log(`  Type: ${r.type}`);
    console.log(`  Similarity: ${r.similarity.toFixed(4)}`);
    console.log(`  Confidence: ${r.confidence}`);
    console.log(`  Source model: ${prov.source}`);
    console.log(`  Extraction method: ${prov.method}`);
  }
} else {
  console.log('No results meet both thresholds.');
  const lowSim = results.filter((r) => r.similarity < SIMILARITY_THRESHOLD);
  if (lowSim.length > 0) {
    console.log(`  Best match was ${lowSim[0].name} (sim=${lowSim[0].similarity.toFixed(4)})`);
  }
}

// ══════════════════════════════════════════════════════════════════
// Stage 7: Context Integrity for Knowledge Graph Export
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 7: Knowledge Graph Export with Integrity');
console.log('='.repeat(60) + '\n');

const exportContext = {
  '@vocab': 'http://schema.org/',
  ex: 'http://www.w3.org/ns/jsonld-ex/',
  ...vectorTermDefinition('embedding', 'http://example.org/embedding', 4),
};

const exportHash = computeIntegrity(exportContext);

const finalDocument = {
  '@context': {
    ...exportContext,
    '@integrity': exportHash,
  },
  '@graph': extractionsModel1,
};

console.log(`Context hash: ${exportHash.substring(0, 50)}...`);
console.log(`Entities: ${finalDocument['@graph'].length}`);
console.log(`Integrity verified: ${verifyIntegrity(exportContext, exportHash)}`);

// ══════════════════════════════════════════════════════════════════
// Stage 8: RDF-Ready Export (vectors stripped)
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('STAGE 8: RDF-Ready Export');
console.log('='.repeat(60) + '\n');

const rdfEntities = extractionsModel1.map((entity) =>
  stripVectorsForRdf(entity, ['embedding']),
);

console.log(`Original entity keys:  ${JSON.stringify(Object.keys(extractionsModel1[0]))}`);
console.log(`RDF-ready entity keys: ${JSON.stringify(Object.keys(rdfEntities[0]))}`);
console.log(`Embeddings stripped:   ${'embedding' in rdfEntities[0] === false}`);

// ══════════════════════════════════════════════════════════════════
// Summary
// ══════════════════════════════════════════════════════════════════

console.log('\n' + '='.repeat(60));
console.log('PIPELINE SUMMARY');
console.log('='.repeat(60) + '\n');

const allConfs: number[] = [];
for (const entity of extractionsModel1) {
  for (const [key, val] of Object.entries(entity)) {
    if (typeof val === 'object' && val !== null && '@confidence' in val) {
      allConfs.push((val as any)['@confidence']);
    }
  }
}

console.log(`  Total entities extracted:    ${extractionsModel1.length}`);
console.log(`  Extraction models used:      2 (${MODEL_V1.split('/').pop()}, ${MODEL_V2.split('/').pop()})`);
console.log(`  Average confidence:          ${(allConfs.reduce((a, b) => a + b, 0) / allConfs.length).toFixed(4)}`);
console.log(`  Schema validation:           ${result.valid ? 'PASS' : 'FAIL'}`);
console.log(`  Entities with embeddings:    ${extractionsModel1.filter((e) => 'embedding' in e).length}`);
console.log(`  Context integrity verified:  true`);
console.log(`  RAG results above threshold: ${relevant.length}`);
console.log(`\n  Extensions used: @confidence, @source, @method, @extractedAt,`);
console.log(`                   @vector, @integrity, @shape`);
