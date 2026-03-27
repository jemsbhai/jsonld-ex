"""Tests for HTTP header derivation module.

Verifies HTTP response/request header derivation from JSON-LD
metadata, enabling semantic-aware caching, content negotiation,
and conditional requests on the web.

References:
    RFC 9110: HTTP Semantics (ETag, Cache-Control, Content-Type)
    RFC 8288: Web Linking (Link header)
    W3C JSON-LD 1.1 §4.1: Context discovery via Link header
"""

from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict

import pytest

from jsonld_ex.http_headers import (
    # Response header derivation
    derive_response_headers,
    # Request header derivation
    derive_request_headers,
    # Individual derivation functions
    derive_etag,
    derive_cache_control,
    derive_link_header,
    derive_content_type,
    # Constants
    MEDIA_TYPE_JSONLD,
    MEDIA_TYPE_CBOR,
    MEDIA_TYPE_JSON,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def annotated_doc():
    """Document with full metadata suite."""
    return {
        "@context": "https://schema.org/",
        "@type": "Person",
        "@id": "https://example.org/people/alice",
        "@confidence": 0.95,
        "@source": "https://model.example.org/ner-v3",
        "@integrity": "sha256-2cf24dba5fb0a30e26e83b2ac5b9e29e",
        "name": {
            "@value": "Alice Smith",
            "@confidence": 0.92,
            "@validFrom": "2024-01-01T00:00:00Z",
            "@validUntil": (
                datetime.now(timezone.utc) + timedelta(hours=2)
            ).isoformat(),
        },
    }


@pytest.fixture
def minimal_doc():
    """Document with no jsonld-ex metadata."""
    return {
        "@context": "https://schema.org/",
        "@type": "Person",
        "name": "Bob",
    }


@pytest.fixture
def multi_context_doc():
    """Document with array @context."""
    return {
        "@context": [
            "https://schema.org/",
            "https://w3id.org/security/v2",
            {"custom": "https://example.org/ns/custom#"},
        ],
        "@type": "VerifiableCredential",
    }


@pytest.fixture
def expired_doc():
    """Document with already-expired @validUntil."""
    return {
        "@context": "https://schema.org/",
        "reading": {
            "@value": 42,
            "@validUntil": "2020-01-01T00:00:00Z",
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Content-Type derivation
# ═══════════════════════════════════════════════════════════════════


class TestContentType:
    """Content-Type header from serialization format."""

    def test_jsonld_content_type(self):
        """JSON-LD format → application/ld+json."""
        ct = derive_content_type(compress=False)
        assert ct == MEDIA_TYPE_JSONLD

    def test_cbor_content_type(self):
        """CBOR format → application/cbor."""
        ct = derive_content_type(compress=True)
        assert ct == MEDIA_TYPE_CBOR

    def test_jsonld_includes_charset(self):
        """JSON-LD Content-Type should not include charset (RFC 6838)."""
        ct = derive_content_type(compress=False)
        # application/ld+json already implies UTF-8
        assert "charset" not in ct.lower()


# ═══════════════════════════════════════════════════════════════════
# ETag derivation
# ═══════════════════════════════════════════════════════════════════


class TestETag:
    """ETag header from @integrity metadata."""

    def test_etag_from_integrity(self, annotated_doc):
        """@integrity → quoted ETag string."""
        etag = derive_etag(annotated_doc)
        assert etag is not None
        # ETag must be a quoted string per RFC 9110 §8.8.3
        assert etag.startswith('"')
        assert etag.endswith('"')

    def test_etag_deterministic(self, annotated_doc):
        """Same @integrity → same ETag."""
        etag1 = derive_etag(annotated_doc)
        etag2 = derive_etag(annotated_doc)
        assert etag1 == etag2

    def test_no_etag_without_integrity(self, minimal_doc):
        """No @integrity → no ETag."""
        etag = derive_etag(minimal_doc)
        assert etag is None

    def test_different_integrity_different_etag(self):
        """Different @integrity values produce different ETags."""
        doc1 = {"@integrity": "sha256-aaa"}
        doc2 = {"@integrity": "sha256-bbb"}
        assert derive_etag(doc1) != derive_etag(doc2)


# ═══════════════════════════════════════════════════════════════════
# Cache-Control derivation
# ═══════════════════════════════════════════════════════════════════


class TestCacheControl:
    """Cache-Control and Expires from @validUntil."""

    def test_max_age_from_valid_until(self, annotated_doc):
        """@validUntil → Cache-Control: max-age=N."""
        cc = derive_cache_control(annotated_doc)
        assert cc is not None
        assert "max-age=" in cc
        # ~2 hours = ~7200 seconds
        import re
        match = re.search(r"max-age=(\d+)", cc)
        assert match is not None
        max_age = int(match.group(1))
        assert 7000 <= max_age <= 7300

    def test_no_cache_control_without_temporal(self, minimal_doc):
        """No @validUntil → no Cache-Control."""
        cc = derive_cache_control(minimal_doc)
        assert cc is None

    def test_no_cache_when_expired(self, expired_doc):
        """Expired @validUntil → no-cache."""
        cc = derive_cache_control(expired_doc)
        assert cc is not None
        assert "no-cache" in cc

    def test_nested_valid_until(self):
        """@validUntil in property value is found."""
        future = (
            datetime.now(timezone.utc) + timedelta(minutes=30)
        ).isoformat()
        doc = {
            "temperature": {
                "@value": 25.0,
                "@validUntil": future,
            },
        }
        cc = derive_cache_control(doc)
        assert cc is not None
        assert "max-age=" in cc


# ═══════════════════════════════════════════════════════════════════
# Link header derivation
# ═══════════════════════════════════════════════════════════════════


class TestLinkHeader:
    """Link header from @context for context discovery."""

    def test_single_context(self, annotated_doc):
        """Single string @context → Link header with JSON-LD relation."""
        link = derive_link_header(annotated_doc)
        assert link is not None
        assert "https://schema.org/" in link
        assert 'rel="http://www.w3.org/ns/json-ld#context"' in link

    def test_no_link_without_context(self):
        """No @context → no Link header."""
        doc = {"@type": "Thing"}
        link = derive_link_header(doc)
        assert link is None

    def test_array_context_multiple_links(self, multi_context_doc):
        """Array @context with URLs → multiple link entries."""
        link = derive_link_header(multi_context_doc)
        assert link is not None
        # Should contain both URL contexts, not the inline object
        assert "schema.org" in link
        assert "w3id.org" in link

    def test_inline_context_excluded(self, multi_context_doc):
        """Inline context objects are not included in Link header."""
        link = derive_link_header(multi_context_doc)
        # The {"custom": "..."} inline context should not appear as a link
        assert "example.org/ns/custom" not in link


# ═══════════════════════════════════════════════════════════════════
# Full response header derivation
# ═══════════════════════════════════════════════════════════════════


class TestResponseHeaders:
    """Full response header dict from document metadata."""

    def test_content_type_always_present(self, annotated_doc):
        """Content-Type is always included."""
        headers = derive_response_headers(annotated_doc)
        assert "Content-Type" in headers

    def test_etag_when_integrity(self, annotated_doc):
        """ETag included when @integrity present."""
        headers = derive_response_headers(annotated_doc)
        assert "ETag" in headers

    def test_cache_control_when_temporal(self, annotated_doc):
        """Cache-Control included when @validUntil present."""
        headers = derive_response_headers(annotated_doc)
        assert "Cache-Control" in headers

    def test_link_when_context(self, annotated_doc):
        """Link header included when @context present."""
        headers = derive_response_headers(annotated_doc)
        assert "Link" in headers

    def test_custom_confidence_header(self, annotated_doc):
        """X-JsonLD-Confidence custom header from @confidence."""
        headers = derive_response_headers(annotated_doc)
        assert "X-JsonLD-Confidence" in headers
        assert headers["X-JsonLD-Confidence"] == "0.95"

    def test_custom_source_header(self, annotated_doc):
        """X-JsonLD-Source custom header from @source."""
        headers = derive_response_headers(annotated_doc)
        assert "X-JsonLD-Source" in headers

    def test_custom_type_header(self, annotated_doc):
        """X-JsonLD-Type custom header from @type."""
        headers = derive_response_headers(annotated_doc)
        assert "X-JsonLD-Type" in headers
        assert headers["X-JsonLD-Type"] == "Person"

    def test_minimal_doc_has_content_type_only(self, minimal_doc):
        """Minimal doc → Content-Type and Link, no ETag or Cache-Control."""
        headers = derive_response_headers(minimal_doc)
        assert "Content-Type" in headers
        assert "ETag" not in headers
        assert "Cache-Control" not in headers

    def test_cbor_content_type(self, annotated_doc):
        """compress=True → application/cbor Content-Type."""
        headers = derive_response_headers(annotated_doc, compress=True)
        assert headers["Content-Type"] == MEDIA_TYPE_CBOR


# ═══════════════════════════════════════════════════════════════════
# Request header derivation
# ═══════════════════════════════════════════════════════════════════


class TestRequestHeaders:
    """Request headers for conditional requests."""

    def test_accept_jsonld(self):
        """Default Accept header prefers JSON-LD."""
        headers = derive_request_headers()
        assert "Accept" in headers
        assert MEDIA_TYPE_JSONLD in headers["Accept"]

    def test_accept_cbor(self):
        """compress=True → Accept prefers CBOR."""
        headers = derive_request_headers(compress=True)
        assert MEDIA_TYPE_CBOR in headers["Accept"]

    def test_if_none_match_from_etag(self):
        """Known ETag → If-None-Match for conditional GET."""
        headers = derive_request_headers(etag='"abc123"')
        assert "If-None-Match" in headers
        assert headers["If-None-Match"] == '"abc123"'

    def test_no_if_none_match_without_etag(self):
        """No ETag → no If-None-Match."""
        headers = derive_request_headers()
        assert "If-None-Match" not in headers


# ═══════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════


class TestHTTPEdgeCases:
    """Edge cases for HTTP header derivation."""

    def test_empty_document(self):
        """Empty document produces Content-Type only."""
        headers = derive_response_headers({})
        assert "Content-Type" in headers
        assert "ETag" not in headers
        assert "Cache-Control" not in headers
        assert "Link" not in headers

    def test_array_type_uses_first(self):
        """Array @type → X-JsonLD-Type uses first element."""
        doc = {"@type": ["Person", "Agent"]}
        headers = derive_response_headers(doc)
        assert headers.get("X-JsonLD-Type") == "Person"

    def test_header_values_are_strings(self, annotated_doc):
        """All header values must be strings (HTTP requirement)."""
        headers = derive_response_headers(annotated_doc)
        for key, val in headers.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(val, str), f"Value for {key!r} is not a string: {val!r}"
