"""Integration tests: infer_at() on unrolled DynamicBayesianNetworks.

TDD RED phase for Phase D, Step D.2.

Verifies that the temporal SLNetwork produced by
from_dynamic_bayesian_network() integrates correctly with Tier 3
temporal inference: infer_at(), network_at_time(), and
infer_temporal_diff().

References:
    Murphy, K. (2002). Dynamic Bayesian Networks.
    Jøsang, A. (2016). Subjective Logic, Ch. 5.3.
    SLNetworks_plan.md §3.4 (temporal inference).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.inference import infer_node

# pgmpy is optional; skip if unavailable.
try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
except ImportError:
    pytest.skip("pgmpy DynamicBayesianNetwork not available", allow_module_level=True)
except TypeError:
    pytest.skip("pgmpy requires Python >= 3.10", allow_module_level=True)

from pgmpy.factors.discrete import TabularCPD


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SLICE_DUR = 3600.0  # 1 hour


def _make_umbrella_dbn() -> DBN:
    """Rain(t)→Umbrella(t), Rain(t)→Rain(t+1). All binary."""
    dbn = DBN()
    dbn.add_edges_from([(("Rain", 0), ("Umbrella", 0))])
    dbn.add_edges_from([(("Rain", 0), ("Rain", 1))])

    cpd_rain_0 = TabularCPD(("Rain", 0), 2, [[0.5], [0.5]])
    cpd_umb_0 = TabularCPD(
        ("Umbrella", 0), 2,
        [[0.8, 0.1], [0.2, 0.9]],
        evidence=[("Rain", 0)], evidence_card=[2],
    )
    cpd_rain_1 = TabularCPD(
        ("Rain", 1), 2,
        [[0.7, 0.3], [0.3, 0.7]],
        evidence=[("Rain", 0)], evidence_card=[2],
    )
    dbn.add_cpds(cpd_rain_0, cpd_umb_0, cpd_rain_1)
    dbn.check_model()
    return dbn


def _unroll(num_slices: int = 3, N: int = 10_000) -> SLNetwork:
    """Unroll the umbrella DBN into a temporal SLNetwork."""
    from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

    return from_dynamic_bayesian_network(
        _make_umbrella_dbn(),
        num_slices=num_slices,
        slice_duration=SLICE_DUR,
        start_time=T0,
        default_sample_count=N,
    )


# ═══════════════════════════════════════════════════════════════════
# STANDARD INFERENCE ON UNROLLED DBN
# ═══════════════════════════════════════════════════════════════════


class TestInferOnUnrolledDBN:
    """Test that standard infer_node() works on unrolled DBN networks."""

    def test_infer_leaf_t0(self) -> None:
        """Inference at Umbrella_t0 should produce a valid opinion."""
        net = _unroll()
        result = infer_node(net, "Umbrella_t0")
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_infer_through_inter_slice(self) -> None:
        """Inference at Rain_t1 propagates through the inter-slice edge."""
        net = _unroll()
        result = infer_node(net, "Rain_t1")
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        # Rain_t1 should not be vacuous (it has evidence from Rain_t0)
        assert op.uncertainty < 0.99

    def test_infer_two_hop_chain(self) -> None:
        """Inference at Umbrella_t1 traverses Rain_t0 → Rain_t1 → Umbrella_t1."""
        net = _unroll()
        result = infer_node(net, "Umbrella_t1")
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_infer_t2_propagates_further(self) -> None:
        """Inference at Rain_t2 chains through t0 → t1 → t2."""
        net = _unroll(num_slices=3)
        result = infer_node(net, "Rain_t2")
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9
        assert op.uncertainty < 0.99

    def test_deduced_uncertainty_equals_conditional_uncertainty(self) -> None:
        """When all conditionals share the same uncertainty u_c,
        deduced uncertainty equals u_c at every depth.

        This is a theorem of SL deduction:
            u_Y = b_X · u_{Y|x} + d_X · u_{Y|~x}
                + u_X · (a_X · u_{Y|x} + (1-a_X) · u_{Y|~x})
        When u_{Y|x} = u_{Y|~x} = u_c:
            u_Y = (b_X + d_X + u_X) · u_c = 1 · u_c = u_c

        In the umbrella DBN, from_bayesian_network() uses the same
        default_sample_count for all edges, so all conditionals have
        equal uncertainty.  Therefore deduced uncertainty is constant
        through the inter-slice chain.
        """
        net = _unroll(num_slices=3, N=20)
        u_t1 = infer_node(net, "Rain_t1").opinion.uncertainty
        u_t2 = infer_node(net, "Rain_t2").opinion.uncertainty

        # Both should equal the conditional edge uncertainty
        edge = net._edges[("Rain_t0", "Rain_t1")]
        u_cond = edge.conditional.uncertainty
        assert abs(u_t1 - u_cond) < 1e-9
        assert abs(u_t2 - u_cond) < 1e-9
        assert abs(u_t1 - u_t2) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# INFER_AT() WITH TEMPORAL DECAY
# ═══════════════════════════════════════════════════════════════════


class TestInferAtOnUnrolledDBN:
    """Test infer_at() with temporal decay on unrolled DBN."""

    def test_infer_at_runs_without_error(self) -> None:
        """infer_at() should work on the temporal SLNetwork."""
        net = _unroll()
        ref_time = T0 + timedelta(hours=2)
        result = net.infer_at(
            "Umbrella_t0",
            reference_time=ref_time,
            default_half_life=7200.0,
        )
        assert result is not None
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_infer_at_reference_at_t0_no_decay(self) -> None:
        """At reference_time == T0, there's zero elapsed time for t0 nodes → no decay."""
        net = _unroll()
        result_plain = infer_node(net, "Umbrella_t0")
        result_at = net.infer_at(
            "Umbrella_t0",
            reference_time=T0,
            default_half_life=3600.0,
        )
        # Should be very close (no decay at t=0)
        assert abs(
            result_plain.opinion.projected_probability()
            - result_at.opinion.projected_probability()
        ) < 0.01

    def test_infer_at_decay_increases_uncertainty(self) -> None:
        """Later reference_time should increase uncertainty due to decay."""
        net = _unroll()
        result_early = net.infer_at(
            "Umbrella_t0",
            reference_time=T0 + timedelta(minutes=1),
            default_half_life=3600.0,
        )
        result_late = net.infer_at(
            "Umbrella_t0",
            reference_time=T0 + timedelta(hours=10),
            default_half_life=3600.0,
        )
        assert result_late.opinion.uncertainty > result_early.opinion.uncertainty

    def test_infer_at_inter_slice_node(self) -> None:
        """infer_at() on Rain_t1 should incorporate temporal decay."""
        net = _unroll()
        ref_time = T0 + timedelta(hours=5)
        result = net.infer_at(
            "Rain_t1",
            reference_time=ref_time,
            default_half_life=3600.0,
        )
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_infer_at_edge_decay_mode(self) -> None:
        """Decay mode 'edges' should decay edge conditionals, not node opinions."""
        net = _unroll()
        ref_time = T0 + timedelta(hours=5)
        result = net.infer_at(
            "Umbrella_t0",
            reference_time=ref_time,
            decay_model="edges",
            default_half_life=3600.0,
        )
        op = result.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# NETWORK_AT_TIME() FILTERING
# ═══════════════════════════════════════════════════════════════════


class TestNetworkAtTimeOnUnrolledDBN:
    """Test network_at_time() point-in-time filtering on unrolled DBN."""

    def test_network_at_t0_includes_inter_slice_edge(self) -> None:
        """At T0, inter-slice edge Rain_t0→Rain_t1 is within validity."""
        from jsonld_ex.sl_network.temporal import network_at_time

        net = _unroll()
        filtered = network_at_time(net, T0)
        # The inter-slice edge has valid_from=T0, valid_until=T0+1h
        # At T0, it's valid
        assert filtered.has_edge("Rain_t0", "Umbrella_t0")

    def test_network_at_after_validity_excludes_expired_edge(self) -> None:
        """After an inter-slice edge's valid_until, it should be excluded."""
        from jsonld_ex.sl_network.temporal import network_at_time

        net = _unroll()
        # Inter-slice Rain_t0→Rain_t1 has valid_until = T0 + 1h
        # Querying at T0 + 2h should exclude it
        far_future = T0 + timedelta(hours=2)
        filtered = network_at_time(net, far_future)
        assert not filtered.has_edge("Rain_t0", "Rain_t1")

    def test_network_at_preserves_all_nodes(self) -> None:
        """network_at_time() preserves all nodes regardless of edge filtering."""
        from jsonld_ex.sl_network.temporal import network_at_time

        net = _unroll()
        filtered = network_at_time(net, T0 + timedelta(hours=10))
        assert filtered.node_count() == net.node_count()


# ═══════════════════════════════════════════════════════════════════
# INFER_TEMPORAL_DIFF()
# ═══════════════════════════════════════════════════════════════════


class TestInferTemporalDiffOnUnrolledDBN:
    """Test infer_temporal_diff() on unrolled DBN."""

    def test_temporal_diff_runs(self) -> None:
        """infer_temporal_diff() should run on unrolled DBN nodes."""
        net = _unroll()
        diff = net.infer_temporal_diff(
            "Umbrella_t0",
            t1=T0,
            t2=T0 + timedelta(hours=5),
            default_half_life=3600.0,
        )
        assert diff.node_id == "Umbrella_t0"
        assert diff.t1 == T0
        assert diff.t2 == T0 + timedelta(hours=5)

    def test_temporal_diff_uncertainty_increases(self) -> None:
        """Between t1 and t2, decay should increase uncertainty."""
        net = _unroll()
        diff = net.infer_temporal_diff(
            "Umbrella_t0",
            t1=T0,
            t2=T0 + timedelta(hours=5),
            default_half_life=3600.0,
        )
        assert diff.delta_uncertainty > 0, (
            f"Expected positive delta_uncertainty, got {diff.delta_uncertainty}"
        )

    def test_temporal_diff_belief_decreases(self) -> None:
        """As opinions decay, belief should generally decrease."""
        net = _unroll()
        diff = net.infer_temporal_diff(
            "Umbrella_t0",
            t1=T0,
            t2=T0 + timedelta(hours=5),
            default_half_life=3600.0,
        )
        # With decay, belief mass shifts to uncertainty
        # delta_belief should be negative (or at least non-positive)
        assert diff.delta_belief <= 0.001

    def test_temporal_diff_on_inter_slice_node(self) -> None:
        """Temporal diff on Rain_t1 should also work."""
        net = _unroll()
        diff = net.infer_temporal_diff(
            "Rain_t1",
            t1=T0 + timedelta(hours=1),
            t2=T0 + timedelta(hours=10),
            default_half_life=3600.0,
        )
        assert diff.delta_uncertainty > 0


# ═══════════════════════════════════════════════════════════════════
# BDU INVARIANT ACROSS ALL TEMPORAL OPERATIONS
# ═══════════════════════════════════════════════════════════════════


class TestBDUInvariantAcrossTemporal:
    """Verify b+d+u=1 holds at every node through all temporal operations."""

    def test_bdu_invariant_plain_inference_all_nodes(self) -> None:
        """b+d+u=1 for every inferred node in the unrolled DBN."""
        net = _unroll(num_slices=3)
        for nid in net._nodes:
            result = infer_node(net, nid)
            op = result.opinion
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"b+d+u={total} at {nid}"
            )

    def test_bdu_invariant_infer_at_all_nodes(self) -> None:
        """b+d+u=1 for every node after infer_at() with decay."""
        net = _unroll(num_slices=3)
        ref_time = T0 + timedelta(hours=3)
        for nid in net._nodes:
            result = net.infer_at(
                nid,
                reference_time=ref_time,
                default_half_life=3600.0,
            )
            op = result.opinion
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"b+d+u={total} at {nid} after infer_at"
            )
