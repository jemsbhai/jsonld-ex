"""Tests for security extensions — edge cases."""

import json
import pytest
from jsonld_ex.security import (
    compute_integrity, verify_integrity, integrity_context,
    is_context_allowed, enforce_resource_limits, DEFAULT_RESOURCE_LIMITS,
)


class TestComputeIntegrity:
    def test_string_context(self):
        h = compute_integrity('{"@context": "http://schema.org/"}')
        assert h.startswith("sha256-")

    def test_dict_context_sort_keys(self):
        """Same dict with different insertion order → same hash."""
        a = compute_integrity({"b": 2, "a": 1})
        b = compute_integrity({"a": 1, "b": 2})
        assert a == b

    def test_sha384(self):
        h = compute_integrity("test", algorithm="sha384")
        assert h.startswith("sha384-")

    def test_sha512(self):
        h = compute_integrity("test", algorithm="sha512")
        assert h.startswith("sha512-")

    def test_unsupported_algorithm(self):
        with pytest.raises(ValueError, match="Unsupported algorithm"):
            compute_integrity("test", algorithm="md5")

    def test_none_context_raises(self):
        with pytest.raises(TypeError, match="must not be None"):
            compute_integrity(None)

    def test_non_serializable_raises(self):
        with pytest.raises(TypeError, match="not JSON-serializable"):
            compute_integrity({"key": object()})

    def test_empty_string(self):
        h = compute_integrity("")
        assert h.startswith("sha256-")


class TestVerifyIntegrity:
    def test_roundtrip(self):
        ctx = '{"@vocab": "http://schema.org/"}'
        h = compute_integrity(ctx)
        assert verify_integrity(ctx, h)

    def test_tampered(self):
        h = compute_integrity("original")
        assert not verify_integrity("tampered", h)

    def test_empty_declared(self):
        with pytest.raises(ValueError, match="Invalid integrity string"):
            verify_integrity("test", "")

    def test_no_hyphen(self):
        with pytest.raises(ValueError, match="Invalid integrity string"):
            verify_integrity("test", "sha256abc")

    def test_bad_algorithm(self):
        with pytest.raises(ValueError, match="Invalid integrity string"):
            verify_integrity("test", "md5-abc123")


class TestIntegrityContext:
    def test_creates_reference(self):
        ctx = integrity_context("https://schema.org/", '{"@vocab":"http://schema.org/"}')
        assert ctx["@id"] == "https://schema.org/"
        assert ctx["@integrity"].startswith("sha256-")


class TestIsContextAllowed:
    def test_exact_match(self):
        cfg = {"allowed": ["https://schema.org/"]}
        assert is_context_allowed("https://schema.org/", cfg)
        assert not is_context_allowed("https://evil.org/", cfg)

    def test_pattern_wildcard(self):
        cfg = {"patterns": ["https://example.org/contexts/*"]}
        assert is_context_allowed("https://example.org/contexts/v1", cfg)
        assert not is_context_allowed("https://other.org/contexts/v1", cfg)

    def test_block_remote(self):
        cfg = {"block_remote_contexts": True, "allowed": ["https://schema.org/"]}
        assert not is_context_allowed("https://schema.org/", cfg)

    def test_empty_config_allows_all(self):
        assert is_context_allowed("https://anything.org/", {})

    def test_allowed_list_blocks_unlisted(self):
        cfg = {"allowed": ["https://only-this.org/"]}
        assert not is_context_allowed("https://anything-else.org/", cfg)


class TestEnforceResourceLimits:
    def test_valid_document(self):
        enforce_resource_limits({"key": "value"})  # Should not raise

    def test_none_raises(self):
        with pytest.raises(TypeError, match="must not be None"):
            enforce_resource_limits(None)

    def test_invalid_json_string(self):
        with pytest.raises(ValueError, match="not valid JSON"):
            enforce_resource_limits("{not json}")

    def test_non_serializable_dict(self):
        with pytest.raises(TypeError, match="not JSON-serializable"):
            enforce_resource_limits({"key": object()})

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="must be a str, dict, or list"):
            enforce_resource_limits(42)

    def test_size_limit_string(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits('{"a": 1}', {"max_document_size": 2})

    def test_size_limit_dict(self):
        with pytest.raises(ValueError, match="exceeds limit"):
            enforce_resource_limits({"a": 1}, {"max_document_size": 2})

    def test_depth_limit(self):
        nested = {"a": {"b": {"c": {"d": "deep"}}}}
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(nested, {"max_graph_depth": 2})

    def test_list_document(self):
        enforce_resource_limits([{"a": 1}, {"b": 2}])  # Should not raise

    def test_deeply_nested_capped(self):
        """Extremely deep structures don't cause stack overflow."""
        doc: dict = {}
        current = doc
        for i in range(600):
            current["child"] = {}
            current = current["child"]
        # Should not crash — depth capped at 500
        with pytest.raises(ValueError, match="depth"):
            enforce_resource_limits(doc, {"max_graph_depth": 50})
