"""Tests for temporal fields on SLNetwork types (Tier 3, Step 1).

Verifies that SLNode, SLEdge, TrustEdge, and AttestationEdge
accept optional temporal fields (timestamp, half_life, valid_from,
valid_until) with sensible defaults and validation.

All temporal fields default to None, preserving backward compatibility
with existing Tier 1 and Tier 2 code.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    SLNode,
    SLEdge,
    TrustEdge,
    AttestationEdge,
)


# ── Fixtures ──────────────────────────────────────────────────────

VACUOUS = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0, base_rate=0.5)
CERTAIN = Opinion(belief=0.9, disbelief=0.05, uncertainty=0.05)
NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
YESTERDAY = NOW - timedelta(days=1)
TOMORROW = NOW + timedelta(days=1)
ONE_HOUR = 3600.0
ONE_DAY = 86400.0


# ═══════════════════════════════════════════════════════════════════
# SLNode temporal fields
# ═══════════════════════════════════════════════════════════════════


class TestSLNodeTemporalFields:
    """SLNode gains optional timestamp and half_life fields."""

    def test_defaults_are_none(self):
        """Existing construction without temporal fields still works."""
        node = SLNode(node_id="x", opinion=CERTAIN)
        assert node.timestamp is None
        assert node.half_life is None

    def test_with_timestamp(self):
        node = SLNode(node_id="x", opinion=CERTAIN, timestamp=NOW)
        assert node.timestamp == NOW

    def test_with_half_life(self):
        node = SLNode(node_id="x", opinion=CERTAIN, half_life=ONE_DAY)
        assert node.half_life == ONE_DAY

    def test_with_both(self):
        node = SLNode(
            node_id="x", opinion=CERTAIN, timestamp=NOW, half_life=ONE_HOUR,
        )
        assert node.timestamp == NOW
        assert node.half_life == ONE_HOUR

    def test_frozen_timestamp(self):
        """Temporal fields are immutable on frozen dataclass."""
        node = SLNode(node_id="x", opinion=CERTAIN, timestamp=NOW)
        with pytest.raises(AttributeError):
            node.timestamp = YESTERDAY  # type: ignore[misc]

    def test_frozen_half_life(self):
        node = SLNode(node_id="x", opinion=CERTAIN, half_life=ONE_DAY)
        with pytest.raises(AttributeError):
            node.half_life = ONE_HOUR  # type: ignore[misc]

    def test_negative_half_life_rejected(self):
        """half_life must be positive if provided."""
        with pytest.raises(ValueError, match="half_life"):
            SLNode(node_id="x", opinion=CERTAIN, half_life=-1.0)

    def test_zero_half_life_rejected(self):
        """half_life of zero is degenerate (instant decay to vacuous)."""
        with pytest.raises(ValueError, match="half_life"):
            SLNode(node_id="x", opinion=CERTAIN, half_life=0.0)

    def test_repr_includes_temporal(self):
        """repr should not break with temporal fields set."""
        node = SLNode(node_id="x", opinion=CERTAIN, timestamp=NOW)
        r = repr(node)
        assert "x" in r

    def test_hash_unchanged_by_temporal(self):
        """Hash is identity-based (node_id), not affected by temporal fields."""
        a = SLNode(node_id="x", opinion=CERTAIN)
        b = SLNode(node_id="x", opinion=CERTAIN, timestamp=NOW, half_life=ONE_DAY)
        assert hash(a) == hash(b)

    def test_existing_construction_unchanged(self):
        """Full backward compatibility: all existing SLNode kwargs still work."""
        node = SLNode(
            node_id="test",
            opinion=CERTAIN,
            node_type="content",
            label="A test node",
            metadata={"key": "value"},
        )
        assert node.node_id == "test"
        assert node.label == "A test node"
        assert node.timestamp is None
        assert node.half_life is None


# ═══════════════════════════════════════════════════════════════════
# SLEdge temporal fields
# ═══════════════════════════════════════════════════════════════════


class TestSLEdgeTemporalFields:
    """SLEdge gains timestamp, half_life, valid_from, valid_until."""

    def test_defaults_are_none(self):
        edge = SLEdge(source_id="a", target_id="b", conditional=CERTAIN)
        assert edge.timestamp is None
        assert edge.half_life is None
        assert edge.valid_from is None
        assert edge.valid_until is None

    def test_with_timestamp(self):
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN, timestamp=NOW,
        )
        assert edge.timestamp == NOW

    def test_with_half_life(self):
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            half_life=ONE_DAY,
        )
        assert edge.half_life == ONE_DAY

    def test_with_validity_window(self):
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            valid_from=YESTERDAY, valid_until=TOMORROW,
        )
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until == TOMORROW

    def test_all_temporal_fields(self):
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            timestamp=NOW, half_life=ONE_HOUR,
            valid_from=YESTERDAY, valid_until=TOMORROW,
        )
        assert edge.timestamp == NOW
        assert edge.half_life == ONE_HOUR
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until == TOMORROW

    def test_frozen_temporal_fields(self):
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            timestamp=NOW, valid_from=YESTERDAY,
        )
        with pytest.raises(AttributeError):
            edge.timestamp = YESTERDAY  # type: ignore[misc]
        with pytest.raises(AttributeError):
            edge.valid_from = NOW  # type: ignore[misc]

    def test_negative_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            SLEdge(
                source_id="a", target_id="b", conditional=CERTAIN,
                half_life=-100.0,
            )

    def test_zero_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            SLEdge(
                source_id="a", target_id="b", conditional=CERTAIN,
                half_life=0.0,
            )

    def test_valid_until_before_valid_from_rejected(self):
        """valid_until must not be before valid_from."""
        with pytest.raises(ValueError, match="valid_until.*valid_from"):
            SLEdge(
                source_id="a", target_id="b", conditional=CERTAIN,
                valid_from=TOMORROW, valid_until=YESTERDAY,
            )

    def test_valid_from_without_valid_until_allowed(self):
        """An open-ended validity window is allowed."""
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            valid_from=YESTERDAY,
        )
        assert edge.valid_from == YESTERDAY
        assert edge.valid_until is None

    def test_valid_until_without_valid_from_allowed(self):
        """An edge that expires but has no start bound is allowed."""
        edge = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            valid_until=TOMORROW,
        )
        assert edge.valid_from is None
        assert edge.valid_until == TOMORROW

    def test_hash_unchanged_by_temporal(self):
        a = SLEdge(source_id="a", target_id="b", conditional=CERTAIN)
        b = SLEdge(
            source_id="a", target_id="b", conditional=CERTAIN,
            timestamp=NOW, half_life=ONE_DAY,
            valid_from=YESTERDAY, valid_until=TOMORROW,
        )
        assert hash(a) == hash(b)

    def test_existing_construction_unchanged(self):
        edge = SLEdge(
            source_id="a", target_id="b",
            conditional=CERTAIN,
            counterfactual=VACUOUS,
            edge_type="deduction",
            metadata={"key": "value"},
        )
        assert edge.source_id == "a"
        assert edge.counterfactual == VACUOUS
        assert edge.timestamp is None


# ═══════════════════════════════════════════════════════════════════
# TrustEdge temporal fields
# ═══════════════════════════════════════════════════════════════════


class TestTrustEdgeTemporalFields:
    """TrustEdge gains timestamp and half_life."""

    def test_defaults_are_none(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
        )
        assert edge.timestamp is None
        assert edge.half_life is None

    def test_with_timestamp(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
            timestamp=NOW,
        )
        assert edge.timestamp == NOW

    def test_with_half_life(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
            half_life=ONE_DAY * 365,  # trust decays over a year
        )
        assert edge.half_life == ONE_DAY * 365

    def test_with_both(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
            timestamp=NOW, half_life=ONE_DAY * 30,
        )
        assert edge.timestamp == NOW
        assert edge.half_life == ONE_DAY * 30

    def test_frozen(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
            timestamp=NOW,
        )
        with pytest.raises(AttributeError):
            edge.timestamp = YESTERDAY  # type: ignore[misc]

    def test_negative_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            TrustEdge(
                source_id="alice", target_id="bob", trust_opinion=CERTAIN,
                half_life=-1.0,
            )

    def test_zero_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            TrustEdge(
                source_id="alice", target_id="bob", trust_opinion=CERTAIN,
                half_life=0.0,
            )

    def test_hash_unchanged_by_temporal(self):
        a = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
        )
        b = TrustEdge(
            source_id="alice", target_id="bob", trust_opinion=CERTAIN,
            timestamp=NOW, half_life=ONE_DAY,
        )
        assert hash(a) == hash(b)

    def test_existing_construction_unchanged(self):
        edge = TrustEdge(
            source_id="alice", target_id="bob",
            trust_opinion=CERTAIN,
            edge_type="trust",
            metadata={"verified": True},
        )
        assert edge.metadata == {"verified": True}
        assert edge.timestamp is None


# ═══════════════════════════════════════════════════════════════════
# AttestationEdge temporal fields
# ═══════════════════════════════════════════════════════════════════


class TestAttestationEdgeTemporalFields:
    """AttestationEdge gains timestamp and half_life."""

    def test_defaults_are_none(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
        )
        assert edge.timestamp is None
        assert edge.half_life is None

    def test_with_timestamp(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
            timestamp=NOW,
        )
        assert edge.timestamp == NOW

    def test_with_half_life(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
            half_life=ONE_HOUR,
        )
        assert edge.half_life == ONE_HOUR

    def test_with_both(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
            timestamp=NOW, half_life=ONE_DAY,
        )
        assert edge.timestamp == NOW
        assert edge.half_life == ONE_DAY

    def test_frozen(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
            timestamp=NOW,
        )
        with pytest.raises(AttributeError):
            edge.timestamp = YESTERDAY  # type: ignore[misc]

    def test_negative_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            AttestationEdge(
                agent_id="alice", content_id="claim_x", opinion=CERTAIN,
                half_life=-10.0,
            )

    def test_zero_half_life_rejected(self):
        with pytest.raises(ValueError, match="half_life"):
            AttestationEdge(
                agent_id="alice", content_id="claim_x", opinion=CERTAIN,
                half_life=0.0,
            )

    def test_hash_unchanged_by_temporal(self):
        a = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
        )
        b = AttestationEdge(
            agent_id="alice", content_id="claim_x", opinion=CERTAIN,
            timestamp=NOW, half_life=ONE_DAY,
        )
        assert hash(a) == hash(b)

    def test_existing_construction_unchanged(self):
        edge = AttestationEdge(
            agent_id="alice", content_id="claim_x",
            opinion=CERTAIN,
            edge_type="attestation",
            metadata={"source": "lab_report"},
        )
        assert edge.metadata == {"source": "lab_report"}
        assert edge.timestamp is None
