/**
 * Example 09: Healthcare Wearable — Posture Monitoring
 * =====================================================
 *
 * End-to-end use case tying together all jsonld-ex extensions for a
 * healthcare IoT posture monitoring system using 6-axis IMU sensors.
 *
 * This example mirrors the validation scenario in the FLAIRS-39 paper:
 * "Extending JSON-LD for Modern AI: Addressing Security, Data Modeling,
 * and Implementation Gaps."
 *
 * System: QMI8658 6-axis IMU sensor → Arm Cortex-M33 MCU → Edge ML model
 *         → JSON-LD output with confidence, provenance, and embeddings.
 *
 * Run: npx ts-node examples/js/09_healthcare_wearable.ts
 */

import {
  JsonLdEx, annotate, getConfidence, getProvenance,
  validateNode, validateVector, cosineSimilarity,
  vectorTermDefinition, extractVectors,
  computeIntegrity, verifyIntegrity,
  ShapeDefinition,
} from '../../packages/js/dist';

const processor = new JsonLdEx({
  resourceLimits: {
    maxDocumentSize: 64 * 1024, // 64 KB — IoT constraint
    maxGraphDepth: 10,
  },
});

// ── 1. Define the healthcare observation context ─────────────────

console.log('=== 1. Healthcare Context Setup ===\n');

const HEALTH_CONTEXT = {
  '@vocab': 'http://schema.org/',
  health: 'http://hl7.org/fhir/',
  ex: 'http://www.w3.org/ns/jsonld-ex/',
  ...vectorTermDefinition(
    'sensorEmbedding',
    'http://example.org/sensor-embedding',
    6, // 6-axis IMU: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
  ),
};

// Compute integrity hash for the context
const contextHash = computeIntegrity(HEALTH_CONTEXT);
console.log(`Context integrity: ${contextHash.substring(0, 50)}...`);

// Define validation shape for observations
const observationShape: ShapeDefinition = {
  '@type': 'MedicalObservation',
  posture: { '@required': true, '@type': 'xsd:string' },
  riskLevel: {
    '@required': true,
    '@type': 'xsd:string',
    '@pattern': '^(low|medium|high|critical)$',
  },
};

console.log('Observation shape defined with required posture and risk level.\n');

// ── 2. Simulate sensor readings and ML classification ────────────

console.log('=== 2. Sensor Readings → ML Classification ===\n');

// Simulated IMU readings (6-axis: accel XYZ + gyro XYZ)
const sensorReadings = [
  { timestamp: '2026-01-15T14:30:00Z', imu: [0.12, -0.95, 0.08, 0.02, -0.01, 0.03] },
  { timestamp: '2026-01-15T14:30:05Z', imu: [0.45, -0.78, 0.35, 0.15, -0.08, 0.12] },
  { timestamp: '2026-01-15T14:30:10Z', imu: [0.68, -0.52, 0.55, 0.28, -0.15, 0.22] },
  { timestamp: '2026-01-15T14:30:15Z', imu: [0.15, -0.93, 0.10, 0.03, -0.02, 0.04] },
];

// Simulated ML model outputs
const classifications = [
  { posture: 'upright', confidence: 0.94, risk: 'low' },
  { posture: 'slouching', confidence: 0.82, risk: 'medium' },
  { posture: 'forward-head', confidence: 0.87, risk: 'high' },
  { posture: 'upright', confidence: 0.91, risk: 'low' },
];

const MODEL_SOURCE = 'https://device.example.org/imu-posture-classifier-v3';
const MODEL_METHOD = 'IMU-6axis-random-forest';

console.log(`Model: ${MODEL_SOURCE}`);
console.log(`Method: ${MODEL_METHOD}`);
console.log(`Readings: ${sensorReadings.length}\n`);

// ── 3. Build JSON-LD observations with full provenance ───────────

console.log('=== 3. Building JSON-LD Observations ===\n');

const observations: any[] = [];
for (let i = 0; i < sensorReadings.length; i++) {
  const reading = sensorReadings[i];
  const classification = classifications[i];

  // Validate sensor embedding
  const { valid, errors } = validateVector(reading.imu, 6);
  if (!valid) {
    console.log(`  ⚠ Invalid sensor data at ${reading.timestamp}: ${errors}`);
    continue;
  }

  const observation = {
    '@context': HEALTH_CONTEXT,
    '@type': 'MedicalObservation',
    '@id': `http://example.org/obs/${reading.timestamp}`,
    dateRecorded: reading.timestamp,
    posture: annotate(classification.posture, {
      confidence: classification.confidence,
      source: MODEL_SOURCE,
      extractedAt: reading.timestamp,
      method: MODEL_METHOD,
    }),
    riskLevel: annotate(classification.risk, {
      confidence: classification.confidence * 0.9, // Risk is derived
      source: MODEL_SOURCE,
      method: 'risk-assessment',
    }),
    sensorEmbedding: reading.imu,
  };
  observations.push(observation);
}

console.log(`Created ${observations.length} observations.\n`);
console.log('Sample observation:');
console.log(JSON.stringify(observations[0], null, 2));

// ── 4. Validate observations ────────────────────────────────────

console.log('\n=== 4. Validating Observations ===\n');

for (const obs of observations) {
  const result = validateNode(obs, observationShape);
  const posture = (obs.posture as any)['@value'];
  const status = result.valid ? '✓' : '✗';
  console.log(`  ${status} ${obs.dateRecorded}: ${posture} (valid=${result.valid})`);
  for (const err of result.errors) {
    console.log(`      Error: [${err.constraint}] ${err.message}`);
  }
}

// ── 5. Confidence-based filtering for clinical alerts ────────────

console.log('\n=== 5. Confidence-Based Clinical Alerts ===\n');

const ALERT_THRESHOLD = 0.85;
console.log(`Alert threshold: confidence >= ${ALERT_THRESHOLD}\n`);

for (const obs of observations) {
  const postureConf = getConfidence(obs.posture);
  const risk = (obs.riskLevel as any)['@value'];
  const posture = (obs.posture as any)['@value'];

  if (postureConf !== undefined && postureConf >= ALERT_THRESHOLD && ['high', 'critical'].includes(risk)) {
    console.log(`  🚨 ALERT: ${posture} detected at ${obs.dateRecorded}`);
    console.log(`     Risk: ${risk}, Confidence: ${postureConf}`);
    const prov = getProvenance(obs.posture);
    console.log(`     Model: ${prov.source}`);
    console.log(`     Method: ${prov.method}\n`);
  } else if (postureConf !== undefined && postureConf < ALERT_THRESHOLD) {
    console.log(`  ⚠ Low confidence (${postureConf}): ${posture} — needs review`);
  }
}

// ── 6. Sensor embedding similarity analysis ─────────────────────

console.log('\n=== 6. Sensor Embedding Analysis ===\n');

console.log('Pairwise cosine similarity of sensor readings:\n');
process.stdout.write('       ');
for (let i = 0; i < observations.length; i++) {
  process.stdout.write(`  R${i + 1}   `);
}
console.log();

for (let i = 0; i < observations.length; i++) {
  process.stdout.write(`  R${i + 1}  `);
  for (let j = 0; j < observations.length; j++) {
    const sim = cosineSimilarity(observations[i].sensorEmbedding, observations[j].sensorEmbedding);
    process.stdout.write(`  ${sim.toFixed(3)}`);
  }
  const posture = (observations[i].posture as any)['@value'];
  console.log(`  (${posture})`);
}

console.log("\n  Note: R1 & R4 (both 'upright') show high similarity (>0.99)");
console.log("  R2 & R3 ('slouching' & 'forward-head') show moderate similarity");

// ── 7. Context integrity verification ────────────────────────────

console.log('\n=== 7. Context Integrity Check ===\n');

const isValid = verifyIntegrity(HEALTH_CONTEXT, contextHash);
console.log(`Context integrity verified: ${isValid}`);

// Simulate tampered context
const tampered = { ...HEALTH_CONTEXT, posture: 'http://evil.example.org/fake-posture' };
const isTampered = verifyIntegrity(tampered, contextHash);
console.log(`Tampered context detected:  ${!isTampered}`);

// ── 8. Summary statistics ────────────────────────────────────────

console.log('\n=== 8. Session Summary ===\n');

const confidences = observations.map((obs) => getConfidence(obs.posture)).filter((c): c is number => c !== undefined);

const postureCounts: Record<string, number> = {};
const riskCounts: Record<string, number> = {};
for (const obs of observations) {
  const posture = (obs.posture as any)['@value'];
  const risk = (obs.riskLevel as any)['@value'];
  postureCounts[posture] = (postureCounts[posture] || 0) + 1;
  riskCounts[risk] = (riskCounts[risk] || 0) + 1;
}

console.log(`Total observations: ${observations.length}`);
console.log(`Avg confidence:     ${(confidences.reduce((a, b) => a + b, 0) / confidences.length).toFixed(4)}`);
console.log(`Min confidence:     ${Math.min(...confidences).toFixed(4)}`);
console.log(`Max confidence:     ${Math.max(...confidences).toFixed(4)}`);
console.log('\nPosture distribution:');
for (const [posture, count] of Object.entries(postureCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${posture}: ${count}`);
}
console.log('\nRisk distribution:');
for (const [risk, count] of Object.entries(riskCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${risk}: ${count}`);
}
