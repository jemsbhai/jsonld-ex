"""
Example 09: Healthcare Wearable — Posture Monitoring
=====================================================

End-to-end use case tying together all jsonld-ex extensions for a
healthcare IoT posture monitoring system using 6-axis IMU sensors.

This example mirrors the validation scenario in the FLAIRS-39 paper:
"Extending JSON-LD for Modern AI: Addressing Security, Data Modeling,
and Implementation Gaps."

System: QMI8658 6-axis IMU sensor → Arm Cortex-M33 MCU → Edge ML model
        → JSON-LD output with confidence, provenance, and embeddings.
"""

import json
from datetime import datetime, timezone
from jsonld_ex import (
    JsonLdEx, annotate, validate_node, validate_vector, cosine_similarity,
)
from jsonld_ex.ai_ml import get_provenance, get_confidence
from jsonld_ex.vector import vector_term_definition, extract_vectors
from jsonld_ex.security import compute_integrity, verify_integrity

processor = JsonLdEx(
    resource_limits={
        "max_document_size": 64 * 1024,  # 64 KB — IoT constraint
        "max_graph_depth": 10,
    }
)

# ── 1. Define the healthcare observation context ─────────────────

print("=== 1. Healthcare Context Setup ===\n")

HEALTH_CONTEXT = {
    "@vocab": "http://schema.org/",
    "health": "http://hl7.org/fhir/",
    "ex": "http://www.w3.org/ns/jsonld-ex/",
    **vector_term_definition(
        "sensorEmbedding",
        "http://example.org/sensor-embedding",
        dimensions=6,  # 6-axis IMU: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
    ),
}

# Compute integrity hash for the context
context_hash = compute_integrity(HEALTH_CONTEXT)
print(f"Context integrity: {context_hash[:50]}...")

# Define validation shape for observations
observation_shape = {
    "@type": "MedicalObservation",
    "posture": {"@required": True, "@type": "xsd:string"},
    "riskLevel": {
        "@required": True,
        "@type": "xsd:string",
        "@pattern": r"^(low|medium|high|critical)$",
    },
}

print("Observation shape defined with required posture and risk level.\n")

# ── 2. Simulate sensor readings and ML classification ────────────

print("=== 2. Sensor Readings → ML Classification ===\n")

# Simulated IMU readings (6-axis: accel XYZ + gyro XYZ)
sensor_readings = [
    {"timestamp": "2026-01-15T14:30:00Z", "imu": [0.12, -0.95, 0.08, 0.02, -0.01, 0.03]},
    {"timestamp": "2026-01-15T14:30:05Z", "imu": [0.45, -0.78, 0.35, 0.15, -0.08, 0.12]},
    {"timestamp": "2026-01-15T14:30:10Z", "imu": [0.68, -0.52, 0.55, 0.28, -0.15, 0.22]},
    {"timestamp": "2026-01-15T14:30:15Z", "imu": [0.15, -0.93, 0.10, 0.03, -0.02, 0.04]},
]

# Simulated ML model outputs
classifications = [
    {"posture": "upright", "confidence": 0.94, "risk": "low"},
    {"posture": "slouching", "confidence": 0.82, "risk": "medium"},
    {"posture": "forward-head", "confidence": 0.87, "risk": "high"},
    {"posture": "upright", "confidence": 0.91, "risk": "low"},
]

MODEL_SOURCE = "https://device.example.org/imu-posture-classifier-v3"
MODEL_METHOD = "IMU-6axis-random-forest"

print(f"Model: {MODEL_SOURCE}")
print(f"Method: {MODEL_METHOD}")
print(f"Readings: {len(sensor_readings)}\n")

# ── 3. Build JSON-LD observations with full provenance ───────────

print("=== 3. Building JSON-LD Observations ===\n")

observations = []
for reading, classification in zip(sensor_readings, classifications):
    # Validate sensor embedding
    valid, errors = validate_vector(reading["imu"], expected_dimensions=6)
    if not valid:
        print(f"  ⚠ Invalid sensor data at {reading['timestamp']}: {errors}")
        continue

    observation = {
        "@context": HEALTH_CONTEXT,
        "@type": "MedicalObservation",
        "@id": f"http://example.org/obs/{reading['timestamp']}",
        "dateRecorded": reading["timestamp"],
        "posture": annotate(
            classification["posture"],
            confidence=classification["confidence"],
            source=MODEL_SOURCE,
            extracted_at=reading["timestamp"],
            method=MODEL_METHOD,
        ),
        "riskLevel": annotate(
            classification["risk"],
            confidence=classification["confidence"] * 0.9,  # Risk is derived
            source=MODEL_SOURCE,
            method="risk-assessment",
        ),
        "sensorEmbedding": reading["imu"],
    }
    observations.append(observation)

print(f"Created {len(observations)} observations.\n")
print("Sample observation:")
print(json.dumps(observations[0], indent=2, default=str))

# ── 4. Validate observations ────────────────────────────────────

print("\n=== 4. Validating Observations ===\n")

for obs in observations:
    result = validate_node(obs, observation_shape)
    posture = obs["posture"]["@value"]
    status = "✓" if result.valid else "✗"
    print(f"  {status} {obs['dateRecorded']}: {posture} (valid={result.valid})")
    for err in result.errors:
        print(f"      Error: [{err.constraint}] {err.message}")

# ── 5. Confidence-based filtering for clinical alerts ────────────

print("\n=== 5. Confidence-Based Clinical Alerts ===\n")

ALERT_THRESHOLD = 0.85

print(f"Alert threshold: confidence >= {ALERT_THRESHOLD}\n")

for obs in observations:
    posture_conf = get_confidence(obs["posture"])
    risk = obs["riskLevel"]["@value"]
    posture = obs["posture"]["@value"]

    if posture_conf and posture_conf >= ALERT_THRESHOLD and risk in ("high", "critical"):
        print(f"  🚨 ALERT: {posture} detected at {obs['dateRecorded']}")
        print(f"     Risk: {risk}, Confidence: {posture_conf}")
        prov = get_provenance(obs["posture"])
        print(f"     Model: {prov.source}")
        print(f"     Method: {prov.method}\n")
    elif posture_conf and posture_conf < ALERT_THRESHOLD:
        print(f"  ⚠ Low confidence ({posture_conf}): {posture} — needs review")

# ── 6. Sensor embedding similarity analysis ─────────────────────

print("\n=== 6. Sensor Embedding Analysis ===\n")

# Compare readings to find similar posture patterns
print("Pairwise cosine similarity of sensor readings:\n")
print("       ", end="")
for i in range(len(observations)):
    print(f"  R{i+1}   ", end="")
print()

for i, obs_i in enumerate(observations):
    print(f"  R{i+1}  ", end="")
    for j, obs_j in enumerate(observations):
        sim = cosine_similarity(obs_i["sensorEmbedding"], obs_j["sensorEmbedding"])
        print(f"  {sim:.3f}", end="")
    posture = obs_i["posture"]["@value"]
    print(f"  ({posture})")

print("\n  Note: R1 & R4 (both 'upright') show high similarity (>0.99)")
print("  R2 & R3 ('slouching' & 'forward-head') show moderate similarity")

# ── 7. Context integrity verification ────────────────────────────

print("\n=== 7. Context Integrity Check ===\n")

# Verify our observations use untampered context
is_valid = verify_integrity(HEALTH_CONTEXT, context_hash)
print(f"Context integrity verified: {is_valid}")

# Simulate tampered context
tampered = {**HEALTH_CONTEXT, "posture": "http://evil.example.org/fake-posture"}
is_tampered = verify_integrity(tampered, context_hash)
print(f"Tampered context detected:  {not is_tampered}")

# ── 8. Summary statistics ────────────────────────────────────────

print("\n=== 8. Session Summary ===\n")

confidences = [get_confidence(obs["posture"]) for obs in observations]
valid_confs = [c for c in confidences if c is not None]

from collections import Counter
posture_counts = Counter(obs["posture"]["@value"] for obs in observations)
risk_counts = Counter(obs["riskLevel"]["@value"] for obs in observations)

print(f"Total observations: {len(observations)}")
print(f"Avg confidence:     {sum(valid_confs)/len(valid_confs):.4f}")
print(f"Min confidence:     {min(valid_confs):.4f}")
print(f"Max confidence:     {max(valid_confs):.4f}")
print(f"\nPosture distribution:")
for posture, count in posture_counts.most_common():
    print(f"  {posture}: {count}")
print(f"\nRisk distribution:")
for risk, count in risk_counts.most_common():
    print(f"  {risk}: {count}")
