"""Tests for CBOR-LD round-trip with @vector and @quantization metadata.

Priority #3 — Verify that the CBOR-LD serialization module correctly
handles documents containing vector embeddings and quantization
descriptors.  These tests confirm round-trip fidelity and measure
payload reduction from quantized representations.

Part of EN8.5 enhancement: CBOR-LD + TurboQuant compact transport.
"""

import json
import math
import struct
import pytest

cbor2 = pytest.importorskip("cbor2", reason="cbor2 required for CBOR-LD tests")

from jsonld_ex.cbor_ld import to_cbor, from_cbor, payload_stats
from jsonld_ex.vector import quantization_descriptor


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def small_vector_doc():
    """A document with a small 8-dim float vector and @quantization."""
    return {
        "@context": "https://schema.org/",
        "@type": "Product",
        "name": "Widget",
        "embedding": {
            "@container": "@vector",
            "@dimensions": 8,
            "@similarity": "cosine",
            "@quantization": quantization_descriptor(
                method="turboquant",
                bit_width=4,
                rotation_seed=42,
                has_residual_qjl=True,
            ),
            "@value": [0.123, -0.456, 0.789, -0.012, 0.345, -0.678, 0.901, -0.234],
        },
    }


@pytest.fixture
def large_vector_doc():
    """A document with a 768-dim float32 vector (typical sentence embedding)."""
    import random

    rng = random.Random(42)
    # Simulate a unit-normalised 768-dim embedding
    raw = [rng.gauss(0, 1) for _ in range(768)]
    norm = math.sqrt(sum(x * x for x in raw))
    vec = [x / norm for x in raw]

    return {
        "@context": "https://schema.org/",
        "@type": "Product",
        "name": "Large Widget",
        "description": "A product with a full sentence embedding.",
        "embedding": {
            "@container": "@vector",
            "@dimensions": 768,
            "@similarity": "cosine",
            "@quantization": quantization_descriptor(
                method="turboquant",
                bit_width=4,
                rotation_seed=42,
                has_residual_qjl=True,
            ),
            "@value": vec,
        },
    }


@pytest.fixture
def quantized_bytes_doc():
    """A document with a packed quantized byte-string representation.

    This simulates a real-world scenario where the producer has already
    quantized the vector to 4-bit and packed two values per byte.
    The @quantization descriptor tells the consumer how to unpack.
    """
    # 768-dim at 4-bit = 384 bytes (two 4-bit values per byte)
    packed = bytes(range(256)) + bytes(range(128))  # 384 bytes
    return {
        "@context": "https://schema.org/",
        "@type": "Product",
        "name": "Packed Widget",
        "embedding": {
            "@container": "@vector",
            "@dimensions": 768,
            "@similarity": "cosine",
            "@quantization": quantization_descriptor(
                method="turboquant",
                bit_width=4,
                rotation_seed=42,
                has_residual_qjl=True,
            ),
            # Packed bytes as a list of ints (CBOR can encode this;
            # a future optimisation could use CBOR byte strings)
            "@value": list(packed),
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Round-trip fidelity
# ═══════════════════════════════════════════════════════════════════


class TestVectorRoundTrip:
    """CBOR-LD round-trip must preserve vector data and quantization metadata."""

    def test_small_vector_values_preserved(self, small_vector_doc):
        data = to_cbor(small_vector_doc)
        restored = from_cbor(data)

        original_vec = small_vector_doc["embedding"]["@value"]
        restored_vec = restored["embedding"]["@value"]
        assert len(restored_vec) == len(original_vec)
        for orig, rest in zip(original_vec, restored_vec):
            assert abs(orig - rest) < 1e-10, f"Vector value mismatch: {orig} vs {rest}"

    def test_quantization_metadata_preserved(self, small_vector_doc):
        data = to_cbor(small_vector_doc)
        restored = from_cbor(data)

        q = restored["embedding"]["@quantization"]
        assert q["method"] == "turboquant"
        assert q["bitWidth"] == 4
        assert q["rotationSeed"] == 42
        assert q["hasResidualQJL"] is True

    def test_vector_container_metadata_preserved(self, small_vector_doc):
        data = to_cbor(small_vector_doc)
        restored = from_cbor(data)

        emb = restored["embedding"]
        assert emb["@container"] == "@vector"
        assert emb["@dimensions"] == 8
        assert emb["@similarity"] == "cosine"

    def test_large_vector_round_trip(self, large_vector_doc):
        data = to_cbor(large_vector_doc)
        restored = from_cbor(data)

        original_vec = large_vector_doc["embedding"]["@value"]
        restored_vec = restored["embedding"]["@value"]
        assert len(restored_vec) == 768
        for i, (orig, rest) in enumerate(zip(original_vec, restored_vec)):
            assert abs(orig - rest) < 1e-10, f"Mismatch at dim {i}: {orig} vs {rest}"

    def test_quantized_bytes_round_trip(self, quantized_bytes_doc):
        """Packed quantized representation survives CBOR round-trip."""
        data = to_cbor(quantized_bytes_doc)
        restored = from_cbor(data)

        original = quantized_bytes_doc["embedding"]["@value"]
        restored_val = restored["embedding"]["@value"]
        assert restored_val == original

    def test_context_restored_with_vector(self, small_vector_doc):
        """Context compression still works when vectors are present."""
        data = to_cbor(small_vector_doc)
        restored = from_cbor(data)
        assert "schema.org" in restored["@context"]

    def test_product_quantization_descriptor_round_trip(self):
        """PQ descriptor with codebook_size and subvector_count round-trips."""
        doc = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "embedding": {
                "@container": "@vector",
                "@dimensions": 128,
                "@quantization": quantization_descriptor(
                    method="product_quantization",
                    bit_width=8,
                    codebook_size=256,
                    subvector_count=16,
                ),
                "@value": [0.1] * 128,
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)

        q = restored["embedding"]["@quantization"]
        assert q["method"] == "product_quantization"
        assert q["bitWidth"] == 8
        assert q["codebookSize"] == 256
        assert q["subvectorCount"] == 16


# ═══════════════════════════════════════════════════════════════════
# Payload size analysis
# ═══════════════════════════════════════════════════════════════════


class TestVectorPayloadStats:
    """CBOR should provide meaningful compression for vector-heavy documents."""

    def test_cbor_smaller_for_float_vectors(self, large_vector_doc):
        """CBOR encodes float arrays more compactly than JSON text."""
        stats = payload_stats(large_vector_doc)
        assert stats.cbor_bytes < stats.json_bytes, (
            f"CBOR ({stats.cbor_bytes}B) should be smaller than "
            f"JSON ({stats.json_bytes}B) for float vector documents"
        )

    def test_gzip_cbor_much_smaller(self, large_vector_doc):
        """Gzipped CBOR should be substantially smaller than raw JSON."""
        stats = payload_stats(large_vector_doc)
        assert stats.gzip_cbor_bytes < stats.json_bytes * 0.8, (
            f"Gzipped CBOR ({stats.gzip_cbor_bytes}B) should be < 80% "
            f"of JSON ({stats.json_bytes}B)"
        )

    def test_quantized_bytes_much_smaller_than_float_vector(
        self, large_vector_doc, quantized_bytes_doc
    ):
        """A packed 4-bit representation is dramatically smaller than float32.

        768 floats at float32 = ~18,000 JSON chars.
        768 dims at 4-bit packed = 384 int values in [0, 255].
        The packed representation should be much smaller in both JSON
        and CBOR.
        """
        stats_float = payload_stats(large_vector_doc)
        stats_packed = payload_stats(quantized_bytes_doc)

        assert stats_packed.json_bytes < stats_float.json_bytes, (
            f"Packed quantized ({stats_packed.json_bytes}B JSON) should be "
            f"smaller than float ({stats_float.json_bytes}B JSON)"
        )
        assert stats_packed.cbor_bytes < stats_float.cbor_bytes, (
            f"Packed quantized ({stats_packed.cbor_bytes}B CBOR) should be "
            f"smaller than float ({stats_float.cbor_bytes}B CBOR)"
        )

    def test_quantization_metadata_overhead_small(self, small_vector_doc):
        """The @quantization descriptor adds minimal overhead."""
        # Document with quantization metadata
        stats_with = payload_stats(small_vector_doc)

        # Same document without quantization
        doc_without = dict(small_vector_doc)
        doc_without["embedding"] = dict(small_vector_doc["embedding"])
        del doc_without["embedding"]["@quantization"]
        stats_without = payload_stats(doc_without)

        overhead = stats_with.cbor_bytes - stats_without.cbor_bytes
        # Quantization descriptor is a small dict (~5 fields);
        # should add < 100 bytes in CBOR
        assert overhead < 100, (
            f"Quantization metadata overhead ({overhead}B) should be < 100B"
        )


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestVectorEdgeCases:
    """Edge cases for vector data in CBOR-LD."""

    def test_empty_vector(self):
        """Empty vector should round-trip correctly."""
        doc = {
            "@context": "https://schema.org/",
            "embedding": {
                "@container": "@vector",
                "@dimensions": 0,
                "@value": [],
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["embedding"]["@value"] == []

    def test_multiple_vectors_in_one_document(self):
        """Multiple vector fields in a single document."""
        doc = {
            "@context": "https://schema.org/",
            "@type": "Product",
            "titleEmbedding": {
                "@container": "@vector",
                "@dimensions": 4,
                "@value": [0.1, 0.2, 0.3, 0.4],
            },
            "descriptionEmbedding": {
                "@container": "@vector",
                "@dimensions": 4,
                "@quantization": quantization_descriptor("scalar", bit_width=8),
                "@value": [0.5, 0.6, 0.7, 0.8],
            },
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert restored["titleEmbedding"]["@value"] == [0.1, 0.2, 0.3, 0.4]
        assert restored["descriptionEmbedding"]["@value"] == [0.5, 0.6, 0.7, 0.8]
        assert restored["descriptionEmbedding"]["@quantization"]["method"] == "scalar"

    def test_vector_in_graph_array(self):
        """Vectors inside @graph nodes should round-trip."""
        doc = {
            "@context": "https://schema.org/",
            "@graph": [
                {
                    "@id": "ex:product1",
                    "@type": "Product",
                    "embedding": {
                        "@container": "@vector",
                        "@dimensions": 3,
                        "@value": [0.1, 0.2, 0.3],
                    },
                },
                {
                    "@id": "ex:product2",
                    "@type": "Product",
                    "embedding": {
                        "@container": "@vector",
                        "@dimensions": 3,
                        "@quantization": quantization_descriptor("turboquant", 4),
                        "@value": [0.4, 0.5, 0.6],
                    },
                },
            ],
        }
        data = to_cbor(doc)
        restored = from_cbor(data)
        assert len(restored["@graph"]) == 2
        assert restored["@graph"][0]["embedding"]["@value"] == [0.1, 0.2, 0.3]
        assert restored["@graph"][1]["embedding"]["@quantization"]["method"] == "turboquant"
