"""
Tests for Tier 2 Trust Network data types (Tier 2, Step 1).

Covers construction, validation, immutability, hashability,
repr, and all error paths for:
    - TrustEdge
    - AttestationEdge
    - TrustPropagationResult

TDD RED PHASE: These tests define the contract that the new trust
types in types.py must satisfy.  They should FAIL until the types
are implemented.

References:
    Jøsang, A. (2016). Subjective Logic, §14.3 (transitive trust),
    §14.5 (multi-path trust).
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    InferenceStep,
    TrustEdge,
    AttestationEdge,
    TrustPropagationResult,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def high_trust() -> Opinion:
    """High trust opinion: b=0.9, d=0.02, u=0.08."""
    return Opinion(belief=0.9, disbelief=0.02, uncertainty=0.08)


@pytest.fixture
def moderate_trust() -> Opinion:
    """Moderate trust: b=0.6, d=0.1, u=0.3."""
    return Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)


@pytest.fixture
def low_trust() -> Opinion:
    """Low trust: b=0.2, d=0.5, u=0.3."""
    return Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)


@pytest.fixture
def vacuous() -> Opinion:
    """Complete ignorance: b=0, d=0, u=1."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def content_opinion() -> Opinion:
    """An agent's opinion about a content proposition."""
    return Opinion(belief=0.75, disbelief=0.1, uncertainty=0.15)


# ═══════════════════════════════════════════════════════════════════
# TrustEdge — CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestTrustEdgeConstruction:
    """TrustEdge represents a trust relationship between two agents."""

    def test_basic_construction(self, high_trust: Opinion) -> None:
        """TrustEdge with required fields constructs successfully."""
        edge = TrustEdge(
            source_id="alice",
            target_id="bob",
            trust_opinion=high_trust,
        )
        assert edge.source_id == "alice"
        assert edge.target_id == "bob"
        assert edge.trust_opinion is high_trust
        assert edge.edge_type == "trust"
        assert edge.metadata == {}

    def test_with_metadata(self, moderate_trust: Opinion) -> None:
        """TrustEdge can carry arbitrary metadata."""
        meta = {"established": "2025-01-01", "context": "peer-review"}
        edge = TrustEdge(
            source_id="alice",
            target_id="bob",
            trust_opinion=moderate_trust,
            metadata=meta,
        )
        assert edge.metadata == meta

    def test_edge_type_defaults_to_trust(self, high_trust: Opinion) -> None:
        """The edge_type field defaults to 'trust'."""
        edge = TrustEdge(
            source_id="a",
            target_id="b",
            trust_opinion=high_trust,
        )
        assert edge.edge_type == "trust"

    def test_frozen_immutability(self, high_trust: Opinion) -> None:
        """TrustEdge is frozen — attributes cannot be reassigned."""
        edge = TrustEdge(
            source_id="a",
            target_id="b",
            trust_opinion=high_trust,
        )
        with pytest.raises(AttributeError):
            edge.source_id = "c"  # type: ignore[misc]

    def test_hashable(self, high_trust: Opinion) -> None:
        """TrustEdge is hashable by (source_id, target_id)."""
        e1 = TrustEdge(source_id="a", target_id="b", trust_opinion=high_trust)
        e2 = TrustEdge(source_id="a", target_id="b", trust_opinion=high_trust)
        assert hash(e1) == hash(e2)

    def test_hash_differs_by_endpoints(self, high_trust: Opinion) -> None:
        """Edges with different endpoints have different hashes."""
        e1 = TrustEdge(source_id="a", target_id="b", trust_opinion=high_trust)
        e2 = TrustEdge(source_id="a", target_id="c", trust_opinion=high_trust)
        # Not guaranteed but very likely for simple string hashing
        assert hash(e1) != hash(e2)

    def test_repr_contains_key_info(self, high_trust: Opinion) -> None:
        """repr includes source, target, and trust opinion summary."""
        edge = TrustEdge(
            source_id="alice",
            target_id="bob",
            trust_opinion=high_trust,
        )
        r = repr(edge)
        assert "alice" in r
        assert "bob" in r
        assert "trust" in r.lower() or "Trust" in r


# ═══════════════════════════════════════════════════════════════════
# TrustEdge — VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestTrustEdgeValidation:
    """TrustEdge validates its inputs in __post_init__."""

    def test_empty_source_id_raises(self, high_trust: Opinion) -> None:
        """Empty source_id is rejected."""
        with pytest.raises(ValueError, match="source_id"):
            TrustEdge(source_id="", target_id="b", trust_opinion=high_trust)

    def test_whitespace_source_id_raises(self, high_trust: Opinion) -> None:
        """Whitespace-only source_id is rejected."""
        with pytest.raises(ValueError, match="source_id"):
            TrustEdge(source_id="  ", target_id="b", trust_opinion=high_trust)

    def test_empty_target_id_raises(self, high_trust: Opinion) -> None:
        """Empty target_id is rejected."""
        with pytest.raises(ValueError, match="target_id"):
            TrustEdge(source_id="a", target_id="", trust_opinion=high_trust)

    def test_self_loop_raises(self, high_trust: Opinion) -> None:
        """Self-trust (source == target) is rejected."""
        with pytest.raises(ValueError, match="[Ss]elf"):
            TrustEdge(source_id="a", target_id="a", trust_opinion=high_trust)

    def test_non_opinion_trust_raises(self) -> None:
        """trust_opinion must be an Opinion instance."""
        with pytest.raises(TypeError, match="trust_opinion"):
            TrustEdge(
                source_id="a",
                target_id="b",
                trust_opinion=0.9,  # type: ignore[arg-type]
            )

    def test_non_string_source_raises(self, high_trust: Opinion) -> None:
        """source_id must be a string."""
        with pytest.raises((ValueError, TypeError)):
            TrustEdge(
                source_id=123,  # type: ignore[arg-type]
                target_id="b",
                trust_opinion=high_trust,
            )

    def test_non_string_target_raises(self, high_trust: Opinion) -> None:
        """target_id must be a string."""
        with pytest.raises((ValueError, TypeError)):
            TrustEdge(
                source_id="a",
                target_id=456,  # type: ignore[arg-type]
                trust_opinion=high_trust,
            )


# ═══════════════════════════════════════════════════════════════════
# AttestationEdge — CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestAttestationEdgeConstruction:
    """AttestationEdge: an agent attests to a content proposition."""

    def test_basic_construction(self, content_opinion: Opinion) -> None:
        """AttestationEdge with required fields constructs successfully."""
        edge = AttestationEdge(
            agent_id="alice",
            content_id="claim_x",
            opinion=content_opinion,
        )
        assert edge.agent_id == "alice"
        assert edge.content_id == "claim_x"
        assert edge.opinion is content_opinion
        assert edge.edge_type == "attestation"
        assert edge.metadata == {}

    def test_with_metadata(self, content_opinion: Opinion) -> None:
        """AttestationEdge can carry arbitrary metadata."""
        meta = {"method": "NER", "model": "spacy-v3"}
        edge = AttestationEdge(
            agent_id="model_a",
            content_id="entity_1",
            opinion=content_opinion,
            metadata=meta,
        )
        assert edge.metadata == meta

    def test_edge_type_defaults_to_attestation(
        self, content_opinion: Opinion
    ) -> None:
        """The edge_type field defaults to 'attestation'."""
        edge = AttestationEdge(
            agent_id="a",
            content_id="c",
            opinion=content_opinion,
        )
        assert edge.edge_type == "attestation"

    def test_frozen_immutability(self, content_opinion: Opinion) -> None:
        """AttestationEdge is frozen — attributes cannot be reassigned."""
        edge = AttestationEdge(
            agent_id="a",
            content_id="c",
            opinion=content_opinion,
        )
        with pytest.raises(AttributeError):
            edge.agent_id = "b"  # type: ignore[misc]

    def test_hashable(self, content_opinion: Opinion) -> None:
        """AttestationEdge is hashable by (agent_id, content_id)."""
        e1 = AttestationEdge(
            agent_id="a", content_id="c", opinion=content_opinion
        )
        e2 = AttestationEdge(
            agent_id="a", content_id="c", opinion=content_opinion
        )
        assert hash(e1) == hash(e2)

    def test_hash_differs_by_endpoints(
        self, content_opinion: Opinion
    ) -> None:
        """Edges with different endpoints have different hashes."""
        e1 = AttestationEdge(
            agent_id="a", content_id="c1", opinion=content_opinion
        )
        e2 = AttestationEdge(
            agent_id="a", content_id="c2", opinion=content_opinion
        )
        assert hash(e1) != hash(e2)

    def test_repr_contains_key_info(self, content_opinion: Opinion) -> None:
        """repr includes agent, content, and opinion summary."""
        edge = AttestationEdge(
            agent_id="alice",
            content_id="claim_x",
            opinion=content_opinion,
        )
        r = repr(edge)
        assert "alice" in r
        assert "claim_x" in r


# ═══════════════════════════════════════════════════════════════════
# AttestationEdge — VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestAttestationEdgeValidation:
    """AttestationEdge validates its inputs in __post_init__."""

    def test_empty_agent_id_raises(self, content_opinion: Opinion) -> None:
        """Empty agent_id is rejected."""
        with pytest.raises(ValueError, match="agent_id"):
            AttestationEdge(
                agent_id="",
                content_id="c",
                opinion=content_opinion,
            )

    def test_whitespace_agent_id_raises(
        self, content_opinion: Opinion
    ) -> None:
        """Whitespace-only agent_id is rejected."""
        with pytest.raises(ValueError, match="agent_id"):
            AttestationEdge(
                agent_id="   ",
                content_id="c",
                opinion=content_opinion,
            )

    def test_empty_content_id_raises(self, content_opinion: Opinion) -> None:
        """Empty content_id is rejected."""
        with pytest.raises(ValueError, match="content_id"):
            AttestationEdge(
                agent_id="a",
                content_id="",
                opinion=content_opinion,
            )

    def test_self_reference_raises(self, content_opinion: Opinion) -> None:
        """Agent attesting to itself (agent_id == content_id) is rejected."""
        with pytest.raises(ValueError, match="[Ss]elf"):
            AttestationEdge(
                agent_id="x",
                content_id="x",
                opinion=content_opinion,
            )

    def test_non_opinion_raises(self) -> None:
        """opinion must be an Opinion instance."""
        with pytest.raises(TypeError, match="opinion"):
            AttestationEdge(
                agent_id="a",
                content_id="c",
                opinion=0.75,  # type: ignore[arg-type]
            )

    def test_non_string_agent_raises(self, content_opinion: Opinion) -> None:
        """agent_id must be a string."""
        with pytest.raises((ValueError, TypeError)):
            AttestationEdge(
                agent_id=123,  # type: ignore[arg-type]
                content_id="c",
                opinion=content_opinion,
            )


# ═══════════════════════════════════════════════════════════════════
# TrustPropagationResult — CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestTrustPropagationResultConstruction:
    """TrustPropagationResult holds transitive trust computation results."""

    def test_basic_construction(self, high_trust: Opinion) -> None:
        """TrustPropagationResult with all fields constructs successfully."""
        step = InferenceStep(
            node_id="bob",
            operation="trust_discount",
            inputs={"trust": high_trust, "opinion": high_trust},
            result=high_trust,
        )
        result = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={"bob": high_trust},
            trust_paths={"bob": ["alice", "bob"]},
            steps=[step],
        )
        assert result.querying_agent == "alice"
        assert "bob" in result.derived_trusts
        assert result.derived_trusts["bob"] is high_trust
        assert result.trust_paths["bob"] == ["alice", "bob"]
        assert len(result.steps) == 1

    def test_empty_derived_trusts(self) -> None:
        """An agent with no reachable agents has empty derived_trusts."""
        result = TrustPropagationResult(
            querying_agent="isolated_agent",
            derived_trusts={},
            trust_paths={},
            steps=[],
        )
        assert result.querying_agent == "isolated_agent"
        assert result.derived_trusts == {}
        assert result.trust_paths == {}
        assert result.steps == []

    def test_multiple_derived_trusts(
        self, high_trust: Opinion, moderate_trust: Opinion
    ) -> None:
        """Multiple agents reachable from the querying agent."""
        result = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={
                "bob": high_trust,
                "charlie": moderate_trust,
            },
            trust_paths={
                "bob": ["alice", "bob"],
                "charlie": ["alice", "bob", "charlie"],
            },
            steps=[],
        )
        assert len(result.derived_trusts) == 2
        assert result.derived_trusts["bob"] is high_trust
        assert result.derived_trusts["charlie"] is moderate_trust

    def test_frozen_immutability(self) -> None:
        """TrustPropagationResult is frozen."""
        result = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={},
            trust_paths={},
            steps=[],
        )
        with pytest.raises(AttributeError):
            result.querying_agent = "bob"  # type: ignore[misc]

    def test_hashable(self) -> None:
        """TrustPropagationResult is hashable by querying_agent."""
        r1 = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={},
            trust_paths={},
            steps=[],
        )
        r2 = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={},
            trust_paths={},
            steps=[],
        )
        assert hash(r1) == hash(r2)

    def test_repr_contains_key_info(self, high_trust: Opinion) -> None:
        """repr includes querying agent and count of derived trusts."""
        result = TrustPropagationResult(
            querying_agent="alice",
            derived_trusts={"bob": high_trust},
            trust_paths={"bob": ["alice", "bob"]},
            steps=[],
        )
        r = repr(result)
        assert "alice" in r


# ═══════════════════════════════════════════════════════════════════
# TrustPropagationResult — VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestTrustPropagationResultValidation:
    """TrustPropagationResult validates its inputs."""

    def test_empty_querying_agent_raises(self) -> None:
        """Empty querying_agent is rejected."""
        with pytest.raises(ValueError, match="querying_agent"):
            TrustPropagationResult(
                querying_agent="",
                derived_trusts={},
                trust_paths={},
                steps=[],
            )

    def test_whitespace_querying_agent_raises(self) -> None:
        """Whitespace-only querying_agent is rejected."""
        with pytest.raises(ValueError, match="querying_agent"):
            TrustPropagationResult(
                querying_agent="   ",
                derived_trusts={},
                trust_paths={},
                steps=[],
            )
