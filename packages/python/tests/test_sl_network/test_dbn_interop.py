"""Tests for DynamicBayesianNetwork → temporal SLNetwork conversion.

TDD RED phase for Phase D, Step D.1.

Maps pgmpy DynamicBayesianNetwork time slices to a temporal SLNetwork
where inter-slice edges carry temporal metadata (timestamps, validity
windows) compatible with Tier 3 temporal inference (infer_at, etc.).

Node naming convention: "{variable}_t{slice_index}"
    e.g., "Rain_t0", "Rain_t1", "Umbrella_t0", "Umbrella_t1"

References:
    Murphy, K. (2002). Dynamic Bayesian Networks.
    Jøsang, A. (2016). Subjective Logic, Ch. 5.3 (opinion aging).
    SLNetworks_plan.md §1.6 (BN interop), §3.4 (temporal inference).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.types import SLEdge, SLNode

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


def _make_umbrella_dbn() -> DBN:
    """Classic umbrella DBN: Rain(t) → Umbrella(t), Rain(t) → Rain(t+1).

    All variables are binary.

    Time slice 0 (intra):
        Rain_0 → Umbrella_0
    Inter-slice:
        Rain_0 → Rain_1

    CPDs:
        P(Rain_0=1) = 0.5
        P(Umbrella_0=1 | Rain_0=1) = 0.9
        P(Umbrella_0=1 | Rain_0=0) = 0.2
        P(Rain_1=1 | Rain_0=1) = 0.7
        P(Rain_1=1 | Rain_0=0) = 0.3
    """
    dbn = DBN()
    # Intra-slice edges (time slice 0)
    dbn.add_edges_from([(("Rain", 0), ("Umbrella", 0))])
    # Inter-slice edges (slice 0 → slice 1)
    dbn.add_edges_from([(("Rain", 0), ("Rain", 1))])

    # CPD for Rain at t=0 (root)
    cpd_rain_0 = TabularCPD(
        variable=("Rain", 0),
        variable_card=2,
        values=[[0.5], [0.5]],
    )

    # CPD for Umbrella at t=0 (depends on Rain_0)
    cpd_umbrella_0 = TabularCPD(
        variable=("Umbrella", 0),
        variable_card=2,
        values=[
            [0.8, 0.1],   # P(U=0|R=0), P(U=0|R=1)
            [0.2, 0.9],   # P(U=1|R=0), P(U=1|R=1)
        ],
        evidence=[("Rain", 0)],
        evidence_card=[2],
    )

    # CPD for Rain at t=1 (depends on Rain_0, inter-slice)
    cpd_rain_1 = TabularCPD(
        variable=("Rain", 1),
        variable_card=2,
        values=[
            [0.7, 0.3],   # P(R1=0|R0=0), P(R1=0|R0=1)
            [0.3, 0.7],   # P(R1=1|R0=0), P(R1=1|R0=1)
        ],
        evidence=[("Rain", 0)],
        evidence_card=[2],
    )

    dbn.add_cpds(cpd_rain_0, cpd_umbrella_0, cpd_rain_1)
    dbn.check_model()
    return dbn


def _make_ternary_dbn() -> DBN:
    """DBN with a ternary variable: Weather(3 states) → Activity(2 states).

    Weather(t) → Activity(t), Weather(t) → Weather(t+1).
    Weather has 3 states: sunny(0), cloudy(1), rainy(2).
    Activity is binary: outdoor(1), indoor(0).
    """
    dbn = DBN()
    dbn.add_edges_from([(("Weather", 0), ("Activity", 0))])
    dbn.add_edges_from([(("Weather", 0), ("Weather", 1))])

    cpd_weather_0 = TabularCPD(
        variable=("Weather", 0),
        variable_card=3,
        values=[[0.4], [0.35], [0.25]],
    )

    # P(Activity | Weather): 2 rows x 3 columns
    cpd_activity_0 = TabularCPD(
        variable=("Activity", 0),
        variable_card=2,
        values=[
            [0.1, 0.5, 0.8],   # P(indoor | sunny), P(indoor | cloudy), P(indoor | rainy)
            [0.9, 0.5, 0.2],   # P(outdoor | ...)
        ],
        evidence=[("Weather", 0)],
        evidence_card=[3],
    )

    # P(Weather_1 | Weather_0): 3 rows x 3 columns
    cpd_weather_1 = TabularCPD(
        variable=("Weather", 1),
        variable_card=3,
        values=[
            [0.6, 0.3, 0.1],   # P(sunny_1 | sunny_0), ...
            [0.3, 0.4, 0.3],   # P(cloudy_1 | ...)
            [0.1, 0.3, 0.6],   # P(rainy_1 | ...)
        ],
        evidence=[("Weather", 0)],
        evidence_card=[3],
    )

    dbn.add_cpds(cpd_weather_0, cpd_activity_0, cpd_weather_1)
    dbn.check_model()
    return dbn


T0 = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
SLICE_DURATION = 3600.0  # 1 hour between slices


# ═══════════════════════════════════════════════════════════════════
# BASIC CONVERSION (2 slices)
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNBasic:
    """Test from_dynamic_bayesian_network() with 2-slice unrolling."""

    def test_converts_without_error(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net is not None

    def test_node_count_2_slices(self) -> None:
        """2 variables × 2 slices = 4 nodes."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.node_count() == 4  # Rain_t0, Umbrella_t0, Rain_t1, Umbrella_t1

    def test_node_naming_convention(self) -> None:
        """Nodes should be named '{variable}_t{slice}'."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        expected = {"Rain_t0", "Umbrella_t0", "Rain_t1", "Umbrella_t1"}
        actual = {nid for nid in net._nodes}
        assert actual == expected

    def test_intra_slice_edges_exist(self) -> None:
        """Rain_t0 → Umbrella_t0 should exist in each slice."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.has_edge("Rain_t0", "Umbrella_t0")
        assert net.has_edge("Rain_t1", "Umbrella_t1")

    def test_inter_slice_edges_exist(self) -> None:
        """Rain_t0 → Rain_t1 should exist (temporal dependency)."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.has_edge("Rain_t0", "Rain_t1")

    def test_edge_count(self) -> None:
        """2 intra-slice + 1 inter-slice = 3 edges per 2-slice unrolling.

        Slice 0: Rain_t0 → Umbrella_t0 (intra)
        Slice 1: Rain_t1 → Umbrella_t1 (intra)
        Inter:   Rain_t0 → Rain_t1
        """
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.edge_count() == 3


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL METADATA
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNTemporalMetadata:
    """Test that nodes/edges carry correct temporal metadata."""

    def test_node_timestamps(self) -> None:
        """Nodes in slice t should have timestamp = start_time + t * duration."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )

        assert net.get_node("Rain_t0").timestamp == T0
        assert net.get_node("Umbrella_t0").timestamp == T0
        assert net.get_node("Rain_t1").timestamp == T0 + timedelta(seconds=SLICE_DURATION)
        assert net.get_node("Umbrella_t1").timestamp == T0 + timedelta(seconds=SLICE_DURATION)

    def test_inter_slice_edge_has_timestamp(self) -> None:
        """Inter-slice edges should carry a timestamp."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        edge = net._edges[("Rain_t0", "Rain_t1")]
        assert edge.timestamp is not None

    def test_inter_slice_edge_validity_window(self) -> None:
        """Inter-slice edges should have valid_from/valid_until spanning the transition."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        edge = net._edges[("Rain_t0", "Rain_t1")]
        assert edge.valid_from == T0
        t1 = T0 + timedelta(seconds=SLICE_DURATION)
        assert edge.valid_until == t1

    def test_intra_slice_edge_timestamp(self) -> None:
        """Intra-slice edges should have timestamp matching their slice."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        edge_t0 = net._edges[("Rain_t0", "Umbrella_t0")]
        assert edge_t0.timestamp == T0

        t1 = T0 + timedelta(seconds=SLICE_DURATION)
        edge_t1 = net._edges[("Rain_t1", "Umbrella_t1")]
        assert edge_t1.timestamp == t1


# ═══════════════════════════════════════════════════════════════════
# OPINION CORRECTNESS
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNOpinionCorrectness:
    """Test that opinions match CPD probabilities in the dogmatic limit."""

    def test_root_opinion_matches_cpd_large_n(self) -> None:
        """Root node Rain_t0 should match P(Rain_0=1) = 0.5 at large N."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        N = 100_000
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION,
            start_time=T0, default_sample_count=N,
        )
        node = net.get_node("Rain_t0")
        assert abs(node.opinion.projected_probability() - 0.5) < 0.01

    def test_intra_edge_conditional_matches_cpd(self) -> None:
        """Intra-slice conditional should match P(U=1|R=1) = 0.9."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        N = 100_000
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION,
            start_time=T0, default_sample_count=N,
        )
        edge = net._edges[("Rain_t0", "Umbrella_t0")]
        assert abs(edge.conditional.projected_probability() - 0.9) < 0.01

    def test_inter_edge_conditional_matches_cpd(self) -> None:
        """Inter-slice conditional should match P(R1=1|R0=1) = 0.7."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        N = 100_000
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION,
            start_time=T0, default_sample_count=N,
        )
        edge = net._edges[("Rain_t0", "Rain_t1")]
        assert abs(edge.conditional.projected_probability() - 0.7) < 0.01
        assert edge.counterfactual is not None
        assert abs(edge.counterfactual.projected_probability() - 0.3) < 0.01

    def test_non_root_slice1_is_vacuous(self) -> None:
        """Rain_t1 is not a root (has inter-slice parent) → vacuous opinion."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        node = net.get_node("Rain_t1")
        assert abs(node.opinion.uncertainty - 1.0) < 1e-9


# ═══════════════════════════════════════════════════════════════════
# MULTI-SLICE UNROLLING
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNMultiSlice:
    """Test unrolling with more than 2 time slices."""

    def test_3_slices_node_count(self) -> None:
        """2 variables × 3 slices = 6 nodes."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=3, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.node_count() == 6

    def test_3_slices_edge_count(self) -> None:
        """3 intra-slice + 2 inter-slice = 5 edges.

        Intra: Rain_t0→Umbrella_t0, Rain_t1→Umbrella_t1, Rain_t2→Umbrella_t2
        Inter: Rain_t0→Rain_t1, Rain_t1→Rain_t2
        """
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=3, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.edge_count() == 5

    def test_3_slices_inter_edges_chain(self) -> None:
        """Inter-slice edges should form a chain: t0→t1→t2."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=3, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.has_edge("Rain_t0", "Rain_t1")
        assert net.has_edge("Rain_t1", "Rain_t2")

    def test_3_slices_timestamps_monotonic(self) -> None:
        """Node timestamps should increase monotonically with slice index."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=3, slice_duration=SLICE_DURATION, start_time=T0,
        )
        for var in ["Rain", "Umbrella"]:
            ts = [net.get_node(f"{var}_t{i}").timestamp for i in range(3)]
            for i in range(len(ts) - 1):
                assert ts[i] < ts[i + 1]

    def test_5_slices_node_count(self) -> None:
        """2 variables × 5 slices = 10 nodes."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=5, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.node_count() == 10


# ═══════════════════════════════════════════════════════════════════
# K-ARY DBN VARIABLES
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNKary:
    """Test k-ary variables in DBN conversion."""

    def test_ternary_dbn_converts(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_ternary_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net is not None

    def test_ternary_root_has_multinomial_opinion(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_ternary_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        node = net.get_node("Weather_t0")
        assert node.is_multinomial
        assert node.multinomial_opinion.cardinality == 3

    def test_ternary_inter_slice_uses_multinomial_edge(self) -> None:
        """Weather(3) → Weather(3) inter-slice should use MultinomialEdge."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_ternary_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        assert net.has_multinomial_edge("Weather_t0", "Weather_t1")

    def test_mixed_cardinality_edge(self) -> None:
        """Weather(3) → Activity(2) should use MultinomialEdge (parent k>2)."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_ternary_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        # Child is binary but parent is ternary → MultinomialEdge
        assert net.has_multinomial_edge("Weather_t0", "Activity_t0")


# ═══════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNErrorHandling:
    """Test input validation."""

    def test_num_slices_less_than_2_raises(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        with pytest.raises(ValueError, match="num_slices"):
            from_dynamic_bayesian_network(
                dbn, num_slices=1, slice_duration=SLICE_DURATION, start_time=T0,
            )

    def test_negative_slice_duration_raises(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        with pytest.raises(ValueError, match="slice_duration"):
            from_dynamic_bayesian_network(
                dbn, num_slices=2, slice_duration=-1.0, start_time=T0,
            )

    def test_default_start_time_is_utc_now(self) -> None:
        """If start_time is None, should default to a reasonable value."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION,
        )
        node = net.get_node("Rain_t0")
        assert node.timestamp is not None


# ═══════════════════════════════════════════════════════════════════
# NETWORK IS A VALID DAG
# ═══════════════════════════════════════════════════════════════════


class TestFromDBNStructuralInvariants:
    """Verify the unrolled network is a valid DAG."""

    def test_topological_sort_succeeds(self) -> None:
        """Unrolled network should be acyclic."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=3, slice_duration=SLICE_DURATION, start_time=T0,
        )
        topo = net.topological_sort()
        assert len(topo) == net.node_count()

    def test_only_root_at_t0(self) -> None:
        """Only slice-0 root variables (with no intra-slice parents) should be roots."""
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        roots = set(net.get_roots())
        # Rain_t0 is the only root: it has no parents
        # Umbrella_t0 depends on Rain_t0
        # Rain_t1 depends on Rain_t0
        # Umbrella_t1 depends on Rain_t1
        assert "Rain_t0" in roots

    def test_all_nodes_content_type(self) -> None:
        from jsonld_ex.sl_network.bn_interop import from_dynamic_bayesian_network

        dbn = _make_umbrella_dbn()
        net = from_dynamic_bayesian_network(
            dbn, num_slices=2, slice_duration=SLICE_DURATION, start_time=T0,
        )
        for nid in net._nodes:
            assert net.get_node(nid).node_type == "content"
