"""Tests for vector quantization metadata extensions.

TDD RED phase — all tests should FAIL until implementation is complete.

These tests cover the @quantization metadata extension to @vector containers,
inspired by TurboQuant (ICLR 2026), PolarQuant (AISTATS 2026), and QJL
(AAAI 2025) algorithms for near-optimal vector compression.

The extension adds standardized metadata describing how vector embeddings
are quantized, enabling:
- Downstream consumers to reconstruct/dequantize vectors
- Provenance of compression fidelity
- Uncertainty quantification of quantization distortion
"""

import pytest
from jsonld_ex.vector import (
    vector_term_definition,
    validate_vector,
    quantization_descriptor,
    validate_quantization_descriptor,
)


# ── quantization_descriptor() factory ──────────────────────────────────


class TestQuantizationDescriptor:
    """Factory function creates well-formed quantization metadata dicts."""

    def test_minimal_descriptor(self):
        desc = quantization_descriptor(method="scalar", bit_width=8)
        assert desc["method"] == "scalar"
        assert desc["bitWidth"] == 8

    def test_turboquant_descriptor(self):
        desc = quantization_descriptor(
            method="turboquant",
            bit_width=4,
            rotation_seed=42,
            has_residual_qjl=True,
        )
        assert desc["method"] == "turboquant"
        assert desc["bitWidth"] == 4
        assert desc["rotationSeed"] == 42
        assert desc["hasResidualQJL"] is True

    def test_polarquant_descriptor(self):
        desc = quantization_descriptor(
            method="polarquant",
            bit_width=4,
            rotation_seed=123,
        )
        assert desc["method"] == "polarquant"
        assert desc["rotationSeed"] == 123
        assert desc.get("hasResidualQJL") is None  # not set

    def test_product_quantization_descriptor(self):
        desc = quantization_descriptor(
            method="product_quantization",
            bit_width=8,
            codebook_size=256,
            subvector_count=8,
        )
        assert desc["method"] == "product_quantization"
        assert desc["codebookSize"] == 256
        assert desc["subvectorCount"] == 8

    def test_qjl_1bit_descriptor(self):
        desc = quantization_descriptor(method="qjl", bit_width=1)
        assert desc["method"] == "qjl"
        assert desc["bitWidth"] == 1

    def test_optional_fields_absent_when_not_set(self):
        desc = quantization_descriptor(method="scalar", bit_width=4)
        assert "rotationSeed" not in desc
        assert "hasResidualQJL" not in desc
        assert "codebookSize" not in desc
        assert "subvectorCount" not in desc

    def test_has_residual_qjl_false_is_stored(self):
        """Explicitly setting False should still be stored."""
        desc = quantization_descriptor(
            method="turboquant", bit_width=4, has_residual_qjl=False
        )
        assert desc["hasResidualQJL"] is False

    # ── Validation errors at creation ──

    def test_empty_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            quantization_descriptor(method="", bit_width=8)

    def test_whitespace_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            quantization_descriptor(method="   ", bit_width=8)

    def test_non_string_method_raises(self):
        with pytest.raises(TypeError, match="method"):
            quantization_descriptor(method=42, bit_width=8)

    def test_bit_width_zero_raises(self):
        with pytest.raises(ValueError, match="bitWidth"):
            quantization_descriptor(method="scalar", bit_width=0)

    def test_bit_width_negative_raises(self):
        with pytest.raises(ValueError, match="bitWidth"):
            quantization_descriptor(method="scalar", bit_width=-1)

    def test_bit_width_too_large_raises(self):
        with pytest.raises(ValueError, match="bitWidth"):
            quantization_descriptor(method="scalar", bit_width=33)

    def test_bit_width_non_int_raises(self):
        with pytest.raises(TypeError, match="bitWidth"):
            quantization_descriptor(method="scalar", bit_width=4.5)

    def test_bit_width_bool_raises(self):
        with pytest.raises(TypeError, match="bitWidth"):
            quantization_descriptor(method="scalar", bit_width=True)

    def test_rotation_seed_negative_raises(self):
        with pytest.raises(ValueError, match="rotationSeed"):
            quantization_descriptor(method="polarquant", bit_width=4, rotation_seed=-1)

    def test_codebook_size_zero_raises(self):
        with pytest.raises(ValueError, match="codebookSize"):
            quantization_descriptor(
                method="product_quantization", bit_width=8, codebook_size=0
            )

    def test_subvector_count_zero_raises(self):
        with pytest.raises(ValueError, match="subvectorCount"):
            quantization_descriptor(
                method="product_quantization", bit_width=8, subvector_count=0
            )


# ── validate_quantization_descriptor() ─────────────────────────────────


class TestValidateQuantizationDescriptor:
    """Standalone validation of a descriptor dict (e.g. from deserialization)."""

    def test_valid_minimal(self):
        desc = {"method": "scalar", "bitWidth": 8}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is True
        assert errors == []

    def test_valid_full_turboquant(self):
        desc = {
            "method": "turboquant",
            "bitWidth": 4,
            "rotationSeed": 42,
            "hasResidualQJL": True,
        }
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is True

    def test_missing_method(self):
        desc = {"bitWidth": 8}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("method" in e for e in errors)

    def test_missing_bit_width(self):
        desc = {"method": "scalar"}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("bitWidth" in e for e in errors)

    def test_not_a_dict(self):
        ok, errors = validate_quantization_descriptor("not a dict")
        assert ok is False

    def test_bit_width_out_of_range(self):
        desc = {"method": "scalar", "bitWidth": 64}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("bitWidth" in e for e in errors)

    def test_rotation_seed_wrong_type(self):
        desc = {"method": "polarquant", "bitWidth": 4, "rotationSeed": "bad"}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("rotationSeed" in e for e in errors)

    def test_has_residual_qjl_wrong_type(self):
        desc = {"method": "turboquant", "bitWidth": 4, "hasResidualQJL": "yes"}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("hasResidualQJL" in e for e in errors)

    def test_codebook_size_wrong_type(self):
        desc = {"method": "pq", "bitWidth": 8, "codebookSize": 3.5}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("codebookSize" in e for e in errors)

    def test_unknown_fields_allowed(self):
        """Extensibility: unknown fields are not errors."""
        desc = {"method": "custom", "bitWidth": 4, "customParam": 999}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is True

    def test_subvector_count_wrong_type(self):
        desc = {"method": "pq", "bitWidth": 8, "subvectorCount": "eight"}
        ok, errors = validate_quantization_descriptor(desc)
        assert ok is False
        assert any("subvectorCount" in e for e in errors)


# ── vector_term_definition() with @quantization ───────────────────────


class TestVectorTermDefinitionQuantization:
    """The optional *quantization* parameter stores a ``@quantization``
    key in the term definition.
    """

    def test_quantization_omitted_by_default(self):
        defn = vector_term_definition("emb", "http://ex.org/emb", 768)
        assert "@quantization" not in defn["emb"]

    def test_quantization_none_omits_key(self):
        defn = vector_term_definition("emb", "http://ex.org/emb", quantization=None)
        assert "@quantization" not in defn["emb"]

    def test_quantization_stored(self):
        desc = quantization_descriptor(method="turboquant", bit_width=4)
        defn = vector_term_definition(
            "emb", "http://ex.org/emb", 768, quantization=desc
        )
        assert defn["emb"]["@quantization"] == desc

    def test_quantization_coexists_with_similarity(self):
        desc = quantization_descriptor(method="polarquant", bit_width=4)
        defn = vector_term_definition(
            "emb",
            "http://ex.org/emb",
            768,
            similarity="cosine",
            quantization=desc,
        )
        assert defn["emb"]["@container"] == "@vector"
        assert defn["emb"]["@dimensions"] == 768
        assert defn["emb"]["@similarity"] == "cosine"
        assert defn["emb"]["@quantization"]["method"] == "polarquant"

    def test_quantization_invalid_descriptor_raises(self):
        """Passing a malformed descriptor dict should raise ValueError."""
        with pytest.raises(ValueError, match="quantization"):
            vector_term_definition(
                "emb", "http://ex.org/emb", 768, quantization={"bad": "data"}
            )

    def test_quantization_non_dict_raises(self):
        """Passing a non-dict should raise TypeError."""
        with pytest.raises(TypeError, match="quantization"):
            vector_term_definition(
                "emb", "http://ex.org/emb", 768, quantization="turboquant"
            )

    def test_backwards_compat_no_quantization_arg(self):
        """Existing callers using positional args must not break."""
        defn = vector_term_definition("emb", "http://ex.org/emb", 768)
        assert defn["emb"]["@container"] == "@vector"
        assert defn["emb"]["@dimensions"] == 768
        assert "@quantization" not in defn["emb"]
        assert "@similarity" not in defn["emb"]


# ── Full document-level integration ───────────────────────────────────


class TestQuantizedVectorDocument:
    """End-to-end: a JSON-LD document with a quantized @vector field."""

    def test_full_context_with_quantization(self):
        """Build a complete context term definition with all metadata."""
        desc = quantization_descriptor(
            method="turboquant",
            bit_width=4,
            rotation_seed=42,
            has_residual_qjl=True,
        )
        defn = vector_term_definition(
            "embedding",
            "http://example.org/embedding",
            768,
            similarity="cosine",
            quantization=desc,
        )
        ctx = defn["embedding"]

        # Verify all fields present
        assert ctx["@id"] == "http://example.org/embedding"
        assert ctx["@container"] == "@vector"
        assert ctx["@dimensions"] == 768
        assert ctx["@similarity"] == "cosine"
        assert ctx["@quantization"]["method"] == "turboquant"
        assert ctx["@quantization"]["bitWidth"] == 4
        assert ctx["@quantization"]["rotationSeed"] == 42
        assert ctx["@quantization"]["hasResidualQJL"] is True

    def test_document_structure(self):
        """Verify the complete document shape with quantized vectors."""
        desc = quantization_descriptor(method="turboquant", bit_width=4)
        ctx_terms = vector_term_definition(
            "embedding",
            "http://example.org/embedding",
            768,
            quantization=desc,
        )

        doc = {
            "@context": [
                "https://schema.org/",
                ctx_terms,
            ],
            "@type": "Product",
            "name": "Widget",
            # In a real quantized doc, this would be base64 bytes;
            # for now, the spec allows both float arrays and
            # encoded representations
            "embedding": [0.123, -0.456, 0.789],
        }

        assert doc["@context"][1]["embedding"]["@quantization"]["method"] == "turboquant"
        assert doc["@type"] == "Product"
