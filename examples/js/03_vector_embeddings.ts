/**
 * Example 03: Vector Embeddings
 * ==============================
 *
 * Demonstrates the @vector container type for storing dense vector
 * embeddings alongside symbolic linked data.
 *
 * Use case: Product catalog with semantic search embeddings.
 *
 * Run: npx ts-node examples/js/03_vector_embeddings.ts
 */

import {
  vectorTermDefinition, validateVector, cosineSimilarity,
  extractVectors, stripVectorsForRdf,
} from '../../packages/js/dist';

// ── 1. Defining vector properties in context ─────────────────────

console.log('=== 1. Vector Term Definition ===\n');

// Define a 4-dimensional embedding (small for demo; real would be 768/1536)
const embeddingDef = vectorTermDefinition('embedding', 'http://example.org/embedding', 4);
console.log(`Term definition: ${JSON.stringify(embeddingDef, null, 2)}`);

// ── 2. Creating documents with embeddings ────────────────────────

console.log('\n=== 2. Documents with Embeddings ===\n');

const products = [
  {
    '@type': 'Product',
    '@id': '#laptop',
    name: 'UltraBook Pro 15',
    category: 'Electronics',
    embedding: [0.82, -0.15, 0.63, 0.41],
  },
  {
    '@type': 'Product',
    '@id': '#tablet',
    name: 'TabletAir 10',
    category: 'Electronics',
    embedding: [0.79, -0.12, 0.58, 0.45],
  },
  {
    '@type': 'Product',
    '@id': '#chair',
    name: 'Ergonomic Office Chair',
    category: 'Furniture',
    embedding: [-0.31, 0.72, -0.18, 0.55],
  },
  {
    '@type': 'Product',
    '@id': '#desk',
    name: 'Standing Desk Pro',
    category: 'Furniture',
    embedding: [-0.28, 0.68, -0.22, 0.61],
  },
];

for (const p of products) {
  console.log(`  ${p['@id']}: ${p.name} → embedding=${JSON.stringify(p.embedding)}`);
}

// ── 3. Validating vectors ────────────────────────────────────────

console.log('\n=== 3. Vector Validation ===\n');

// Valid vector
let result = validateVector([0.1, -0.2, 0.3, 0.4], 4);
console.log(`Valid 4D vector: valid=${result.valid}`);

// Wrong dimensions
result = validateVector([0.1, -0.2, 0.3], 4);
console.log(`Wrong dimensions: valid=${result.valid}, errors=${JSON.stringify(result.errors)}`);

// Invalid elements
result = validateVector([0.1, 'not_a_number' as any, 0.3, 0.4]);
console.log(`Invalid element: valid=${result.valid}, errors=${JSON.stringify(result.errors)}`);

// Empty vector
result = validateVector([]);
console.log(`Empty vector: valid=${result.valid}, errors=${JSON.stringify(result.errors)}`);

// ── 4. Semantic similarity search ────────────────────────────────

console.log('\n=== 4. Semantic Similarity Search ===\n');

// Query: "portable computer" embedding
const queryEmbedding = [0.80, -0.14, 0.60, 0.43];

console.log(`Query embedding: ${JSON.stringify(queryEmbedding)}`);
console.log('Results ranked by cosine similarity:\n');

const similarities: Array<{ product: typeof products[0]; sim: number }> = [];
for (const product of products) {
  const sim = cosineSimilarity(queryEmbedding, product.embedding);
  similarities.push({ product, sim });
}

// Sort by similarity descending
similarities.sort((a, b) => b.sim - a.sim);

for (const { product, sim } of similarities) {
  console.log(`  ${sim.toFixed(4)}  ${product.name} (${product.category})`);
}

// ── 5. Extracting vectors from nodes ─────────────────────────────

console.log('\n=== 5. Extracting Vectors ===\n');

const vectors = extractVectors(products[0], ['embedding']);
for (const [prop, vec] of vectors.entries()) {
  console.log(`  ${prop}: ${JSON.stringify(vec)} (dimensions=${vec.length})`);
}

// ── 6. Stripping vectors for RDF conversion ──────────────────────

console.log('\n=== 6. Stripping Vectors for RDF ===\n');

// Vectors are annotation-only — they should not become RDF triples
const original = { ...products[0] };
const rdfReady = stripVectorsForRdf(original, ['embedding']);

console.log('Original keys:', Object.keys(original));
console.log('RDF-ready keys:', Object.keys(rdfReady));
console.log('Embedding removed:', !('embedding' in rdfReady));

// ── 7. Full document with context ────────────────────────────────

console.log('\n=== 7. Full Document ===\n');

const document = {
  '@context': {
    '@vocab': 'http://schema.org/',
    ...vectorTermDefinition('embedding', 'http://example.org/embedding', 4),
  },
  '@graph': products,
};

console.log(JSON.stringify(document, null, 2));
