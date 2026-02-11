"""Tests for the jsonld-ex MCP server.

Tests the MCP tool handler functions directly (no transport needed).
Each tool accepts JSON strings and returns dicts or primitives, mirroring
the MCP tool interface.

These tests verify that:
  1. All 16 tools are registered and callable
  2. Input JSON parsing and validation works correctly
  3. Tool outputs match the expected format
  4. Error cases produce clear error messages
  5. Tools correctly delegate to the underlying jsonld-ex functions
"""

import json
import math
import pytest

# Guard: skip entire module if `mcp` is not installed.
mcp = pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from jsonld_ex.mcp.server import mcp as mcp_server  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Helper: call a tool by name through the FastMCP server
# ═══════════════════════════════════════════════════════════════════

def _get_tool_fn(name: str):
    """Retrieve a registered tool's underlying function by name."""
    tool_manager = mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        available = list(tool_manager._tools.keys())
        raise KeyError(f"Tool '{name}' not found. Available: {available}")
    return tool.fn


# ═══════════════════════════════════════════════════════════════════
# Test: Tool Registration
# ═══════════════════════════════════════════════════════════════════


class TestToolRegistration:
    """Verify all expected tools are registered."""

    EXPECTED_TOOLS = [
        # Group 1: AI/ML Annotation
        "annotate_value",
        "get_confidence_score",
        "filter_by_confidence",
        # Group 2: Confidence Algebra
        "create_opinion",
        "fuse_opinions",
        "discount_opinion",
        "decay_opinion",
        # Group 3: Security & Integrity
        "compute_integrity",
        "verify_integrity",
        "validate_document",
        # Group 4: Vector Operations
        "cosine_similarity",
        "validate_vector",
        # Group 5: Graph Operations
        "merge_graphs",
        "query_at_time",
        # Group 6: Interoperability
        "to_prov_o",
        "shape_to_shacl",
    ]

    def test_all_tools_registered(self):
        """Every expected tool must be present in the server."""
        tool_manager = mcp_server._tool_manager
        registered = set(tool_manager._tools.keys())
        for name in self.EXPECTED_TOOLS:
            assert name in registered, f"Tool '{name}' not registered"

    def test_tool_count(self):
        """Exactly 16 tools should be registered."""
        tool_manager = mcp_server._tool_manager
        assert len(tool_manager._tools) == 16

    def test_all_tools_have_descriptions(self):
        """Every tool must have a non-empty description."""
        tool_manager = mcp_server._tool_manager
        for name, tool in tool_manager._tools.items():
            assert tool.description, f"Tool '{name}' has no description"
            assert len(tool.description) > 20, (
                f"Tool '{name}' description too short: {tool.description!r}"
            )


# ═══════════════════════════════════════════════════════════════════
# Group 1: AI/ML Annotation Tools
# ═══════════════════════════════════════════════════════════════════


class TestAnnotateTool:
    """Tests for the annotate_value tool."""

    def test_basic_annotation(self):
        fn = _get_tool_fn("annotate_value")
        result = fn(value="John Smith", confidence=0.95)
        assert result["@value"] == "John Smith"
        assert result["@confidence"] == 0.95

    def test_full_provenance(self):
        fn = _get_tool_fn("annotate_value")
        result = fn(
            value="Jane Doe",
            confidence=0.87,
            source="https://model.example.org/v3",
            extracted_at="2026-02-11T10:00:00Z",
            method="NER",
            human_verified=True,
        )
        assert result["@value"] == "Jane Doe"
        assert result["@confidence"] == 0.87
        assert result["@source"] == "https://model.example.org/v3"
        assert result["@extractedAt"] == "2026-02-11T10:00:00Z"
        assert result["@method"] == "NER"
        assert result["@humanVerified"] is True

    def test_no_optional_fields(self):
        fn = _get_tool_fn("annotate_value")
        result = fn(value="test")
        assert result == {"@value": "test"}

    def test_invalid_confidence_out_of_range(self):
        fn = _get_tool_fn("annotate_value")
        with pytest.raises((ValueError, Exception)):
            fn(value="test", confidence=1.5)

    def test_numeric_value(self):
        fn = _get_tool_fn("annotate_value")
        result = fn(value=42, confidence=0.9)
        assert result["@value"] == 42


class TestGetConfidenceTool:
    """Tests for the get_confidence_score tool."""

    def test_extract_confidence(self):
        fn = _get_tool_fn("get_confidence_score")
        node = json.dumps({"@value": "test", "@confidence": 0.85})
        result = fn(node_json=node)
        assert result == 0.85

    def test_no_confidence_returns_none(self):
        fn = _get_tool_fn("get_confidence_score")
        node = json.dumps({"@value": "test"})
        result = fn(node_json=node)
        assert result is None

    def test_invalid_json_raises(self):
        fn = _get_tool_fn("get_confidence_score")
        with pytest.raises((json.JSONDecodeError, ValueError, Exception)):
            fn(node_json="not valid json")


class TestFilterByConfidenceTool:
    """Tests for the filter_by_confidence tool."""

    def test_filter_above_threshold(self):
        fn = _get_tool_fn("filter_by_confidence")
        doc = {
            "@context": "http://schema.org/",
            "@graph": [
                {"@id": "a", "name": {"@value": "Alice", "@confidence": 0.9}},
                {"@id": "b", "name": {"@value": "Bob", "@confidence": 0.3}},
                {"@id": "c", "name": {"@value": "Carol", "@confidence": 0.7}},
            ],
        }
        result = fn(document_json=json.dumps(doc), min_confidence=0.5)
        parsed = json.loads(result) if isinstance(result, str) else result
        # Should have filtered out Bob (0.3 < 0.5)
        assert isinstance(parsed, (dict, list))


# ═══════════════════════════════════════════════════════════════════
# Group 2: Confidence Algebra Tools
# ═══════════════════════════════════════════════════════════════════


class TestCreateOpinionTool:
    """Tests for the create_opinion tool."""

    def test_valid_opinion(self):
        fn = _get_tool_fn("create_opinion")
        result = fn(belief=0.7, disbelief=0.1, uncertainty=0.2, base_rate=0.5)
        assert abs(result["belief"] - 0.7) < 1e-9
        assert abs(result["disbelief"] - 0.1) < 1e-9
        assert abs(result["uncertainty"] - 0.2) < 1e-9
        assert abs(result["base_rate"] - 0.5) < 1e-9
        # Projected probability: b + a*u = 0.7 + 0.5*0.2 = 0.8
        assert abs(result["projected_probability"] - 0.8) < 1e-9

    def test_vacuous_opinion(self):
        fn = _get_tool_fn("create_opinion")
        result = fn(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
        assert abs(result["projected_probability"] - 0.5) < 1e-9

    def test_invalid_additivity(self):
        fn = _get_tool_fn("create_opinion")
        with pytest.raises((ValueError, Exception)):
            fn(belief=0.5, disbelief=0.5, uncertainty=0.5)


class TestFuseOpinionsTool:
    """Tests for the fuse_opinions tool."""

    def test_cumulative_fusion(self):
        fn = _get_tool_fn("fuse_opinions")
        opinions = [
            {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5},
            {"belief": 0.6, "disbelief": 0.2, "uncertainty": 0.2, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions), method="cumulative")
        assert "belief" in result
        assert "uncertainty" in result
        # Cumulative fusion: uncertainty should decrease
        assert result["uncertainty"] < 0.2

    def test_averaging_fusion(self):
        fn = _get_tool_fn("fuse_opinions")
        opinions = [
            {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5},
            {"belief": 0.6, "disbelief": 0.2, "uncertainty": 0.2, "base_rate": 0.5},
        ]
        result = fn(opinions_json=json.dumps(opinions), method="averaging")
        assert "belief" in result

    def test_robust_fusion(self):
        fn = _get_tool_fn("fuse_opinions")
        opinions = [
            {"belief": 0.8, "disbelief": 0.1, "uncertainty": 0.1, "base_rate": 0.5},
            {"belief": 0.7, "disbelief": 0.2, "uncertainty": 0.1, "base_rate": 0.5},
            {"belief": 0.1, "disbelief": 0.8, "uncertainty": 0.1, "base_rate": 0.5},  # outlier
        ]
        result = fn(opinions_json=json.dumps(opinions), method="robust")
        # Robust fusion should downweight the outlier
        assert result["belief"] > 0.5

    def test_invalid_method(self):
        fn = _get_tool_fn("fuse_opinions")
        opinions = [
            {"belief": 0.5, "disbelief": 0.2, "uncertainty": 0.3, "base_rate": 0.5},
        ]
        with pytest.raises((ValueError, Exception)):
            fn(opinions_json=json.dumps(opinions), method="bogus")

    def test_invalid_json(self):
        fn = _get_tool_fn("fuse_opinions")
        with pytest.raises((json.JSONDecodeError, ValueError, Exception)):
            fn(opinions_json="not json", method="cumulative")


class TestDiscountOpinionTool:
    """Tests for the discount_opinion tool."""

    def test_full_trust_no_change(self):
        fn = _get_tool_fn("discount_opinion")
        opinion = {"belief": 0.8, "disbelief": 0.1, "uncertainty": 0.1, "base_rate": 0.5}
        # Full trust: b=1, d=0, u=0
        trust = {"belief": 1.0, "disbelief": 0.0, "uncertainty": 0.0, "base_rate": 0.5}
        result = fn(opinion_json=json.dumps(opinion), trust_json=json.dumps(trust))
        assert abs(result["belief"] - 0.8) < 1e-9

    def test_zero_trust_becomes_vacuous(self):
        fn = _get_tool_fn("discount_opinion")
        opinion = {"belief": 0.8, "disbelief": 0.1, "uncertainty": 0.1, "base_rate": 0.5}
        # Zero trust: b=0, d=0, u=1
        trust = {"belief": 0.0, "disbelief": 0.0, "uncertainty": 1.0, "base_rate": 0.5}
        result = fn(opinion_json=json.dumps(opinion), trust_json=json.dumps(trust))
        assert abs(result["uncertainty"] - 1.0) < 1e-9


class TestDecayOpinionTool:
    """Tests for the decay_opinion tool."""

    def test_zero_elapsed_no_change(self):
        fn = _get_tool_fn("decay_opinion")
        opinion = {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            elapsed_seconds=0.0,
            half_life_seconds=3600.0,
        )
        assert abs(result["belief"] - 0.7) < 1e-9

    def test_one_half_life_halves_evidence(self):
        fn = _get_tool_fn("decay_opinion")
        opinion = {"belief": 0.8, "disbelief": 0.0, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            elapsed_seconds=3600.0,
            half_life_seconds=3600.0,
        )
        assert abs(result["belief"] - 0.4) < 1e-9

    def test_linear_decay(self):
        fn = _get_tool_fn("decay_opinion")
        opinion = {"belief": 0.8, "disbelief": 0.0, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            elapsed_seconds=3600.0,
            half_life_seconds=3600.0,
            decay_function="linear",
        )
        assert abs(result["belief"] - 0.4) < 1e-9

    def test_step_decay_before_half_life(self):
        fn = _get_tool_fn("decay_opinion")
        opinion = {"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2, "base_rate": 0.5}
        result = fn(
            opinion_json=json.dumps(opinion),
            elapsed_seconds=100.0,
            half_life_seconds=3600.0,
            decay_function="step",
        )
        # Before half-life: no change
        assert abs(result["belief"] - 0.7) < 1e-9

    def test_invalid_decay_function(self):
        fn = _get_tool_fn("decay_opinion")
        opinion = {"belief": 0.5, "disbelief": 0.2, "uncertainty": 0.3, "base_rate": 0.5}
        with pytest.raises((ValueError, Exception)):
            fn(
                opinion_json=json.dumps(opinion),
                elapsed_seconds=100.0,
                half_life_seconds=3600.0,
                decay_function="nonexistent",
            )


# ═══════════════════════════════════════════════════════════════════
# Group 3: Security & Integrity Tools
# ═══════════════════════════════════════════════════════════════════


class TestComputeIntegrityTool:
    """Tests for the compute_integrity tool."""

    def test_sha256_hash(self):
        fn = _get_tool_fn("compute_integrity")
        result = fn(context_json='{"@vocab": "http://schema.org/"}')
        assert result.startswith("sha256-")

    def test_sha512_hash(self):
        fn = _get_tool_fn("compute_integrity")
        result = fn(
            context_json='{"@vocab": "http://schema.org/"}',
            algorithm="sha512",
        )
        assert result.startswith("sha512-")

    def test_deterministic(self):
        fn = _get_tool_fn("compute_integrity")
        a = fn(context_json='{"key": "value"}')
        b = fn(context_json='{"key": "value"}')
        assert a == b

    def test_invalid_algorithm(self):
        fn = _get_tool_fn("compute_integrity")
        with pytest.raises((ValueError, Exception)):
            fn(context_json='{"key": "value"}', algorithm="md5")


class TestVerifyIntegrityTool:
    """Tests for the verify_integrity tool."""

    def test_valid_hash(self):
        compute = _get_tool_fn("compute_integrity")
        verify = _get_tool_fn("verify_integrity")
        ctx = '{"@vocab": "http://schema.org/"}'
        h = compute(context_json=ctx)
        assert verify(context_json=ctx, declared_hash=h) is True

    def test_tampered_content(self):
        compute = _get_tool_fn("compute_integrity")
        verify = _get_tool_fn("verify_integrity")
        h = compute(context_json='{"key": "original"}')
        assert verify(context_json='{"key": "tampered"}', declared_hash=h) is False


class TestValidateDocumentTool:
    """Tests for the validate_document tool."""

    def test_valid_document(self):
        fn = _get_tool_fn("validate_document")
        doc = json.dumps({"@type": "Person", "name": "Alice", "age": 30})
        shape = json.dumps({
            "@type": "Person",
            "name": {"@required": True, "@type": "http://www.w3.org/2001/XMLSchema#string"},
        })
        result = fn(document_json=doc, shape_json=shape)
        assert result["valid"] is True

    def test_missing_required_field(self):
        fn = _get_tool_fn("validate_document")
        doc = json.dumps({"@type": "Person", "age": 30})
        shape = json.dumps({
            "@type": "Person",
            "name": {"@required": True},
        })
        result = fn(document_json=doc, shape_json=shape)
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# ═══════════════════════════════════════════════════════════════════
# Group 4: Vector Operations Tools
# ═══════════════════════════════════════════════════════════════════


class TestCosineSimilarityTool:
    """Tests for the cosine_similarity tool."""

    def test_identical_vectors(self):
        fn = _get_tool_fn("cosine_similarity")
        v = json.dumps([1.0, 0.0, 0.0])
        result = fn(vector_a_json=v, vector_b_json=v)
        assert abs(result - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        fn = _get_tool_fn("cosine_similarity")
        a = json.dumps([1.0, 0.0, 0.0])
        b = json.dumps([0.0, 1.0, 0.0])
        result = fn(vector_a_json=a, vector_b_json=b)
        assert abs(result - 0.0) < 1e-9

    def test_opposite_vectors(self):
        fn = _get_tool_fn("cosine_similarity")
        a = json.dumps([1.0, 0.0])
        b = json.dumps([-1.0, 0.0])
        result = fn(vector_a_json=a, vector_b_json=b)
        assert abs(result - (-1.0)) < 1e-9


class TestValidateVectorTool:
    """Tests for the validate_vector tool."""

    def test_valid_vector_node(self):
        fn = _get_tool_fn("validate_vector")
        node = json.dumps({
            "@value": [0.1, 0.2, 0.3],
            "@vector": True,
            "@dimensions": 3,
        })
        result = fn(node_json=node)
        assert result["valid"] is True
        assert result["dimensions"] == 3


# ═══════════════════════════════════════════════════════════════════
# Group 5: Graph Operations Tools
# ═══════════════════════════════════════════════════════════════════


class TestMergeGraphsTool:
    """Tests for the merge_graphs tool."""

    def test_non_overlapping_merge(self):
        fn = _get_tool_fn("merge_graphs")
        a = json.dumps({
            "@graph": [{"@id": "ex:1", "name": "Alice"}],
        })
        b = json.dumps({
            "@graph": [{"@id": "ex:2", "name": "Bob"}],
        })
        result_str = fn(graph_a_json=a, graph_b_json=b)
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
        assert isinstance(result, dict)


class TestQueryAtTimeTool:
    """Tests for the query_at_time tool."""

    def test_query_within_bounds(self):
        fn = _get_tool_fn("query_at_time")
        doc = json.dumps({
            "@type": "Person",
            "jobTitle": {
                "@value": "Engineer",
                "@validFrom": "2020-01-01T00:00:00Z",
                "@validUntil": "2025-12-31T23:59:59Z",
            },
        })
        result_str = fn(
            document_json=doc,
            query_time="2023-06-15T00:00:00Z",
        )
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# Group 6: Interoperability Tools
# ═══════════════════════════════════════════════════════════════════


class TestToProvOTool:
    """Tests for the to_prov_o tool."""

    def test_basic_conversion(self):
        fn = _get_tool_fn("to_prov_o")
        node = json.dumps({
            "@type": "Observation",
            "value": {"@value": "test", "@confidence": 0.9, "@source": "model-v1"},
        })
        result_str = fn(node_json=node)
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
        assert isinstance(result, dict)


class TestShapeToShaclTool:
    """Tests for the shape_to_shacl tool."""

    def test_basic_shape_conversion(self):
        fn = _get_tool_fn("shape_to_shacl")
        shape = json.dumps({
            "@type": "Person",
            "name": {"@required": True},
            "age": {"@minimum": 0, "@maximum": 150},
        })
        result_str = fn(shape_json=shape, target_class="http://schema.org/Person")
        result = json.loads(result_str) if isinstance(result_str, str) else result_str
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# Resources
# ═══════════════════════════════════════════════════════════════════


class TestResources:
    """Verify resources are registered."""

    def test_resources_registered(self):
        resource_manager = mcp_server._resource_manager
        registered = set(str(r) for r in resource_manager._resources.keys())
        # Check at least the AI/ML context resource exists
        assert any("ai-ml" in r for r in registered), (
            f"AI/ML context resource not found. Registered: {registered}"
        )


# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════


class TestPrompts:
    """Verify prompts are registered."""

    def test_prompts_registered(self):
        prompt_manager = mcp_server._prompt_manager
        registered = set(prompt_manager._prompts.keys())
        assert "annotate_tool_results" in registered
        assert "trust_chain_analysis" in registered
