"""Tests for trust decay during propagation (Tier 3, Step 6).

When temporal parameters are provided to propagate_trust(), trust edge
opinions are decayed by their age before being used in the discount
chain. This ensures that stale trust intermediaries properly degrade
toward vacuous.

Key mathematical properties tested:
    - Fresh trust edges (age=0) produce the same result as non-temporal propagation.
    - Stale trust edges produce higher uncertainty in derived trust.
    - A stale intermediary in a 3-hop chain degrades the entire chain.
    - b+d+u=1 invariant holds at every derived trust opinion.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion, trust_discount
from jsonld_ex.confidence_decay import decay_opinion, exponential_decay
from jsonld_ex.sl_network.types import SLNode, TrustEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.trust import propagate_trust


# -- Fixtures --

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_DAY = 86400.0
VACUOUS = Opinion(0.0, 0.0, 1.0, base_rate=0.5)

HIGH_TRUST = Opinion(0.9, 0.05, 0.05)
MED_TRUST = Opinion(0.7, 0.1, 0.2)


def _build_trust_chain(
    edge_timestamps: dict[tuple[str, str], datetime | None] | None = None,
    edge_half_lives: dict[tuple[str, str], float | None] | None = None,
) -> SLNetwork:
    """Build alice -> bob -> charlie trust chain."""
    edge_timestamps = edge_timestamps or {}
    edge_half_lives = edge_half_lives or {}

    net = SLNetwork()
    for name in ["alice", "bob", "charlie"]:
        net.add_node(SLNode(node_id=name, opinion=VACUOUS, node_type="agent"))

    net.add_trust_edge(TrustEdge(
        source_id="alice", target_id="bob",
        trust_opinion=HIGH_TRUST,
        timestamp=edge_timestamps.get(("alice", "bob")),
        half_life=edge_half_lives.get(("alice", "bob")),
    ))
    net.add_trust_edge(TrustEdge(
        source_id="bob", target_id="charlie",
        trust_opinion=MED_TRUST,
        timestamp=edge_timestamps.get(("bob", "charlie")),
        half_life=edge_half_lives.get(("bob", "charlie")),
    ))
    return net


class TestTrustDecayFreshEdges:
    """Fresh trust edges (timestamp == reference_time) behave identically to non-temporal."""

    def test_fresh_matches_non_temporal(self):
        """Decay with zero elapsed produces the same result as no decay."""
        net_no_ts = _build_trust_chain()
        result_no_ts = propagate_trust(net_no_ts, "alice")

        net_fresh = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): NOW,
                ("bob", "charlie"): NOW,
            },
        )
        result_fresh = propagate_trust(
            net_fresh, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        assert result_fresh.derived_trusts["charlie"].belief == pytest.approx(
            result_no_ts.derived_trusts["charlie"].belief,
        )
        assert result_fresh.derived_trusts["charlie"].uncertainty == pytest.approx(
            result_no_ts.derived_trusts["charlie"].uncertainty,
        )


class TestTrustDecayStaleEdges:
    """Stale trust edges produce higher uncertainty."""

    def test_stale_single_hop(self):
        """A 1-day-old trust edge produces more uncertainty than a fresh one."""
        one_day_ago = NOW - timedelta(days=1)

        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): one_day_ago,
                ("bob", "charlie"): NOW,
            },
        )
        result_stale = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        net_fresh = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): NOW,
                ("bob", "charlie"): NOW,
            },
        )
        result_fresh = propagate_trust(
            net_fresh, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        # Direct trust in bob: stale version has more uncertainty
        assert result_stale.derived_trusts["bob"].uncertainty > result_fresh.derived_trusts["bob"].uncertainty
        # Transitive trust in charlie also degrades
        assert result_stale.derived_trusts["charlie"].uncertainty > result_fresh.derived_trusts["charlie"].uncertainty

    def test_stale_intermediary_degrades_chain(self):
        """A stale intermediary (bob->charlie) degrades the full chain."""
        one_day_ago = NOW - timedelta(days=1)

        # Only the second hop is stale
        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): NOW,
                ("bob", "charlie"): one_day_ago,
            },
        )
        result = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        # Direct trust in bob is fresh (no decay)
        assert result.derived_trusts["bob"].belief == pytest.approx(HIGH_TRUST.belief)

        # But transitive trust in charlie is degraded
        # Manually compute: alice->bob is fresh HIGH_TRUST,
        # bob->charlie is decayed MED_TRUST
        decayed_bc = decay_opinion(MED_TRUST, elapsed=ONE_DAY, half_life=ONE_DAY)
        expected_charlie = trust_discount(HIGH_TRUST, decayed_bc)
        assert result.derived_trusts["charlie"].belief == pytest.approx(expected_charlie.belief)
        assert result.derived_trusts["charlie"].uncertainty == pytest.approx(expected_charlie.uncertainty)


class TestTrustDecayMatchesManual:
    """Temporal trust propagation matches hand-computed decay + discount."""

    def test_two_hop_manual_match(self):
        """Full 2-hop chain with both edges stale matches manual computation."""
        one_day_ago = NOW - timedelta(days=1)

        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): one_day_ago,
                ("bob", "charlie"): one_day_ago,
            },
        )
        result = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        # Manual: decay each trust opinion, then chain
        decayed_ab = decay_opinion(HIGH_TRUST, elapsed=ONE_DAY, half_life=ONE_DAY)
        decayed_bc = decay_opinion(MED_TRUST, elapsed=ONE_DAY, half_life=ONE_DAY)

        expected_bob = decayed_ab
        expected_charlie = trust_discount(decayed_ab, decayed_bc)

        assert result.derived_trusts["bob"].belief == pytest.approx(expected_bob.belief)
        assert result.derived_trusts["bob"].uncertainty == pytest.approx(expected_bob.uncertainty)
        assert result.derived_trusts["charlie"].belief == pytest.approx(expected_charlie.belief)
        assert result.derived_trusts["charlie"].uncertainty == pytest.approx(expected_charlie.uncertainty)


class TestTrustDecayPerEdgeHalfLife:
    """Per-edge half_life overrides the default."""

    def test_per_edge_half_life(self):
        one_day_ago = NOW - timedelta(days=1)

        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): one_day_ago,
                ("bob", "charlie"): one_day_ago,
            },
            edge_half_lives={
                ("alice", "bob"): ONE_DAY * 30,  # slow decay
                ("bob", "charlie"): ONE_DAY,      # fast decay (default)
            },
        )
        result = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )

        decayed_ab = decay_opinion(HIGH_TRUST, elapsed=ONE_DAY, half_life=ONE_DAY * 30)
        decayed_bc = decay_opinion(MED_TRUST, elapsed=ONE_DAY, half_life=ONE_DAY)
        expected_charlie = trust_discount(decayed_ab, decayed_bc)

        assert result.derived_trusts["charlie"].belief == pytest.approx(expected_charlie.belief)
        assert result.derived_trusts["charlie"].uncertainty == pytest.approx(expected_charlie.uncertainty)


class TestTrustDecayNoTimestampNoDecay:
    """Edges without timestamps are not decayed even when temporal params given."""

    def test_no_timestamp_no_decay(self):
        """An edge with no timestamp is used as-is."""
        net = _build_trust_chain()  # no timestamps
        result_plain = propagate_trust(net, "alice")
        result_temporal = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )
        assert result_temporal.derived_trusts["charlie"].belief == pytest.approx(
            result_plain.derived_trusts["charlie"].belief,
        )


class TestTrustDecayBDUInvariant:
    """b+d+u=1 invariant on all derived trust opinions."""

    def test_invariant(self):
        one_day_ago = NOW - timedelta(days=1)
        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): one_day_ago,
                ("bob", "charlie"): one_day_ago,
            },
        )
        result = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )
        for agent_id, op in result.derived_trusts.items():
            total = op.belief + op.disbelief + op.uncertainty
            assert total == pytest.approx(1.0), (
                f"b+d+u != 1 for {agent_id}: {total}"
            )


class TestTrustDecayFutureTimestamp:
    """Trust edges with future timestamps get no decay."""

    def test_future_no_decay(self):
        future = NOW + timedelta(days=1)
        net = _build_trust_chain(
            edge_timestamps={
                ("alice", "bob"): future,
                ("bob", "charlie"): future,
            },
        )
        result = propagate_trust(
            net, "alice",
            reference_time=NOW, default_half_life=ONE_DAY,
        )
        # No decay applied: same as original trust opinions
        assert result.derived_trusts["bob"].belief == pytest.approx(HIGH_TRUST.belief)
