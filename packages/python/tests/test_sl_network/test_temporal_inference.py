"""Tests for infer_at() -- temporal inference (Tier 3, Step 4).

infer_at() applies temporal decay (nodes, edges, or both) to the
network, then delegates to the existing infer_node(). It lives on
SLNetwork as a convenience method.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.confidence_decay import decay_opinion, linear_decay
from jsonld_ex.sl_network.types import SLNode, SLEdge
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.inference import infer_node


# -- Fixtures --

NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
ONE_DAY = 86400.0


def _build_chain() -> SLNetwork:
    """Build rain -> wet_grass with timestamps 1 day ago on everything."""
    one_day_ago = NOW - timedelta(days=1)

    net = SLNetwork()
    net.add_node(SLNode(
        node_id="rain",
        opinion=Opinion(0.7, 0.1, 0.2),
        timestamp=one_day_ago,
    ))
    net.add_node(SLNode(
        node_id="wet_grass",
        opinion=Opinion(0.5, 0.3, 0.2),
        timestamp=one_day_ago,
    ))
    net.add_edge(SLEdge(
        source_id="rain",
        target_id="wet_grass",
        conditional=Opinion(0.9, 0.05, 0.05),
        counterfactual=Opinion(0.1, 0.7, 0.2),
        timestamp=one_day_ago,
    ))
    return net


class TestInferAtBasic:
    """Basic infer_at() behavior."""

    def test_returns_inference_result(self):
        net = _build_chain()
        result = net.infer_at("wet_grass", reference_time=NOW, default_half_life=ONE_DAY)
        assert result.query_node == "wet_grass"
        assert result.opinion is not None
        assert result.opinion.belief + result.opinion.disbelief + result.opinion.uncertainty == pytest.approx(1.0)

    def test_decay_both_is_default(self):
        """Default decay_model is 'both'."""
        net = _build_chain()
        result_both = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="both",
        )
        result_default = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
        )
        assert result_both.opinion.belief == pytest.approx(result_default.opinion.belief)

    def test_no_decay_at_observation_time(self):
        """If reference_time equals all timestamps, result matches plain infer_node."""
        one_day_ago = NOW - timedelta(days=1)
        net = _build_chain()
        result_at = net.infer_at(
            "wet_grass", reference_time=one_day_ago, default_half_life=ONE_DAY,
        )
        result_plain = infer_node(net, "wet_grass")
        assert result_at.opinion.belief == pytest.approx(result_plain.opinion.belief)
        assert result_at.opinion.uncertainty == pytest.approx(result_plain.opinion.uncertainty)


class TestInferAtDecayModels:
    """Different decay_model settings produce different results."""

    def test_nodes_only_does_not_decay_edges(self):
        """decay_model='nodes' decays node opinions but not edge conditionals."""
        net = _build_chain()
        result_nodes = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="nodes",
        )
        # Compare with both: should differ because edge decay is missing
        result_both = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="both",
        )
        # They should not be equal (edge decay changes the result)
        assert result_nodes.opinion.belief != pytest.approx(
            result_both.opinion.belief, abs=1e-6
        )

    def test_edges_only_does_not_decay_nodes(self):
        """decay_model='edges' decays edge conditionals but not node opinions."""
        net = _build_chain()
        result_edges = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="edges",
        )
        result_both = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="both",
        )
        assert result_edges.opinion.belief != pytest.approx(
            result_both.opinion.belief, abs=1e-6
        )

    def test_both_applies_more_decay_than_either_alone(self):
        """Applying both node and edge decay produces more uncertainty."""
        net = _build_chain()
        result_nodes = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="nodes",
        )
        result_edges = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="edges",
        )
        result_both = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_model="both",
        )
        # Both should have at least as much uncertainty as either alone
        assert result_both.opinion.uncertainty >= result_nodes.opinion.uncertainty - 1e-9
        assert result_both.opinion.uncertainty >= result_edges.opinion.uncertainty - 1e-9


class TestInferAtCustomDecayFn:
    """Custom decay function is forwarded."""

    def test_linear_decay_fn(self):
        # At exactly t == half_life, exponential and linear both yield
        # factor = 0.5, so they are identical.  Use t = half_life / 2
        # where they genuinely diverge:
        #   exponential: 2^(-0.5) ~ 0.7071
        #   linear:      1 - 0.5/2 = 0.75
        half_day_ago = NOW - timedelta(seconds=ONE_DAY / 2)
        net = SLNetwork()
        net.add_node(SLNode(
            node_id="rain",
            opinion=Opinion(0.7, 0.1, 0.2),
            timestamp=half_day_ago,
        ))
        net.add_node(SLNode(
            node_id="wet_grass",
            opinion=Opinion(0.5, 0.3, 0.2),
            timestamp=half_day_ago,
        ))
        net.add_edge(SLEdge(
            source_id="rain",
            target_id="wet_grass",
            conditional=Opinion(0.9, 0.05, 0.05),
            counterfactual=Opinion(0.1, 0.7, 0.2),
            timestamp=half_day_ago,
        ))

        result_exp = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
        )
        result_lin = net.infer_at(
            "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
            decay_fn=linear_decay,
        )
        # At t = half_life/2, exponential and linear yield different factors,
        # so the inferred opinions must differ.
        assert result_exp.opinion.belief != pytest.approx(
            result_lin.opinion.belief, abs=1e-6
        )


class TestInferAtRootNode:
    """Inferring a root node with infer_at."""

    def test_root_node_decayed(self):
        """Root nodes just get their opinion decayed, no deduction."""
        net = _build_chain()
        result = net.infer_at(
            "rain", reference_time=NOW, default_half_life=ONE_DAY,
        )
        expected = decay_opinion(
            Opinion(0.7, 0.1, 0.2),
            elapsed=ONE_DAY,
            half_life=ONE_DAY,
        )
        assert result.opinion.belief == pytest.approx(expected.belief)
        assert result.opinion.uncertainty == pytest.approx(expected.uncertainty)


class TestInferAtBDUInvariant:
    """b+d+u=1 invariant always holds."""

    def test_invariant(self):
        net = _build_chain()
        for model in ["nodes", "edges", "both"]:
            result = net.infer_at(
                "wet_grass", reference_time=NOW, default_half_life=ONE_DAY,
                decay_model=model,
            )
            op = result.opinion
            assert op.belief + op.disbelief + op.uncertainty == pytest.approx(1.0)


class TestInferAtOriginalUnchanged:
    """infer_at does not modify the original network."""

    def test_original_preserved(self):
        net = _build_chain()
        original_rain_b = net.get_node("rain").opinion.belief
        _ = net.infer_at("wet_grass", reference_time=NOW, default_half_life=ONE_DAY)
        assert net.get_node("rain").opinion.belief == original_rain_b
