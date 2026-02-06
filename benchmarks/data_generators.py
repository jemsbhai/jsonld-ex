"""
Synthetic data generators for jsonld-ex benchmarks.

Produces JSON-LD documents of varying complexity for reproducible
performance measurement.
"""

from __future__ import annotations

import random
import string
from typing import Any

_RNG = random.Random(42)  # fixed seed for reproducibility

SOURCES = [
    "https://models.example.org/ner-v4",
    "https://models.example.org/rel-extract-v2",
    "https://models.example.org/classifier-v1",
    "https://models.example.org/gpt4-turbo",
    "https://models.example.org/llama-3-70b",
]

METHODS = ["NER", "classification", "relation-extraction", "summarization", "QA"]

TYPES = ["Person", "Organization", "Product", "Event", "Place"]

NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Hank"]
ORGS = ["Acme Corp", "Globex", "Initech", "Umbrella", "Stark Industries"]
CITIES = ["Melbourne", "New York", "London", "Tokyo", "Berlin", "Sydney"]


def _rand_id(prefix: str = "ex") -> str:
    suffix = "".join(_RNG.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}:{suffix}"


def _rand_confidence() -> float:
    return round(_RNG.uniform(0.3, 0.99), 3)


def _rand_timestamp(year: int = 2025) -> str:
    month = _RNG.randint(1, 12)
    day = _RNG.randint(1, 28)
    hour = _RNG.randint(0, 23)
    minute = _RNG.randint(0, 59)
    return f"{year}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00Z"


# ── Simple annotated nodes ─────────────────────────────────────


def make_annotated_person(node_id: str | None = None) -> dict[str, Any]:
    """Single Person node with confidence-annotated properties."""
    return {
        "@id": node_id or _rand_id(),
        "@type": "Person",
        "name": {
            "@value": _RNG.choice(NAMES),
            "@confidence": _rand_confidence(),
            "@source": _RNG.choice(SOURCES),
            "@extractedAt": _rand_timestamp(),
            "@method": _RNG.choice(METHODS),
        },
        "worksFor": {
            "@value": _RNG.choice(ORGS),
            "@confidence": _rand_confidence(),
            "@source": _RNG.choice(SOURCES),
        },
        "location": {
            "@value": _RNG.choice(CITIES),
            "@confidence": _rand_confidence(),
        },
    }


def make_annotated_graph(n: int) -> dict[str, Any]:
    """Graph with n annotated Person nodes."""
    return {
        "@context": "http://schema.org/",
        "@graph": [make_annotated_person(f"ex:person-{i}") for i in range(n)],
    }


# ── Conflicting graphs (for merge benchmarks) ─────────────────


def make_conflicting_graphs(
    n_nodes: int,
    n_sources: int,
    conflict_rate: float = 0.3,
) -> list[dict[str, Any]]:
    """Generate n_sources graphs with shared node IDs and controlled conflict rate.

    Args:
        n_nodes: Number of nodes per graph.
        n_sources: Number of independent source graphs.
        conflict_rate: Fraction of properties that differ across sources.

    Returns:
        List of n_sources JSON-LD graph documents.
    """
    base_nodes = [make_annotated_person(f"ex:person-{i}") for i in range(n_nodes)]

    graphs = []
    for src_idx in range(n_sources):
        source_url = f"https://models.example.org/source-{src_idx}"
        nodes = []
        for base in base_nodes:
            node = _deep_copy_node(base)
            # Override source on all properties
            for key in ("name", "worksFor", "location"):
                if key in node and isinstance(node[key], dict):
                    node[key]["@source"] = source_url
                    node[key]["@confidence"] = _rand_confidence()
                    # Introduce conflicts at the specified rate
                    if _RNG.random() < conflict_rate:
                        if key == "name":
                            node[key]["@value"] = _RNG.choice(NAMES)
                        elif key == "worksFor":
                            node[key]["@value"] = _RNG.choice(ORGS)
                        elif key == "location":
                            node[key]["@value"] = _RNG.choice(CITIES)
            nodes.append(node)

        graphs.append({
            "@context": "http://schema.org/",
            "@graph": nodes,
        })

    return graphs


# ── Temporal graphs ────────────────────────────────────────────


def make_temporal_graph(n_nodes: int, versions_per_node: int = 3) -> list[dict[str, Any]]:
    """Graph nodes with multiple time-sliced property values."""
    nodes = []
    for i in range(n_nodes):
        titles = []
        for v in range(versions_per_node):
            year = 2020 + v * 2
            titles.append({
                "@value": f"Title-{v}",
                "@confidence": _rand_confidence(),
                "@validFrom": f"{year}-01-01T00:00:00Z",
                "@validUntil": f"{year + 1}-12-31T23:59:59Z",
            })
        nodes.append({
            "@id": f"ex:person-{i}",
            "@type": "Person",
            "name": _RNG.choice(NAMES),
            "jobTitle": titles,
        })
    return nodes


# ── IoT sensor readings ───────────────────────────────────────


def make_sensor_reading(sensor_id: str | None = None) -> dict[str, Any]:
    """Single IoT sensor reading with annotations."""
    return {
        "@context": "http://schema.org/",
        "@type": "SensorReading",
        "@id": sensor_id or f"urn:sensor:{_rand_id('imu')}",
        "value": {
            "@value": round(_RNG.uniform(-10.0, 10.0), 4),
            "@confidence": _rand_confidence(),
            "@source": "https://device.example.org/imu-classifier-v3",
            "@extractedAt": _rand_timestamp(),
            "@method": "IMU-6axis-classification",
        },
        "unit": "m/s^2",
        "axis": _RNG.choice(["x", "y", "z"]),
    }


def make_sensor_batch(n: int) -> list[dict[str, Any]]:
    """Batch of n sensor readings."""
    return [make_sensor_reading(f"urn:sensor:imu-{i:04d}") for i in range(n)]


# ── Helpers ────────────────────────────────────────────────────


def _deep_copy_node(node: dict) -> dict:
    """Shallow-deep copy adequate for our flat structures."""
    out = {}
    for k, v in node.items():
        if isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = [dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out
