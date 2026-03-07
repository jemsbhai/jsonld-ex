"""
Tests for trust propagation algorithms (Tier 2, Step 2).

Covers single-path transitive trust propagation:
    - Single-hop trust matches trust_discount() directly
    - Multi-hop chains match manual left-fold of trust_discount()
    - Full trust (b=1) is identity through entire chain
    - Vacuous trust at any hop yields vacuous derived trust
    - b+d+u=1 invariant holds at every derived opinion

Multi-path trust fusion is deferred to Step 3.

TDD RED PHASE: These tests should FAIL until trust.py is implemented.

References:
    Jøsang, A. (2016). Subjective Logic, §14.3 (transitive trust).
"""

from __future__ import annotations

import math

import pytest

from jsonld_ex.confidence_algebra import Opinion, trust_discount
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.trust import propagate_trust
from jsonld_ex.sl_network.types import (
    SLNode,
    TrustEdge,
    TrustPropagationResult,
)


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

TOL = 1e-9  # Floating-point comparison tolerance


def assert_opinion_close(
    actual: Opinion, expected: Opinion, tol: float = TOL
) -> None:
    """Assert two opinions are numerically close."""
    assert abs(actual.belief - expected.belief) < tol, (
        f"belief: {actual.belief} != {expected.belief}"
    )
    assert abs(actual.disbelief - expected.disbelief) < tol, (
        f"disbelief: {actual.disbelief} != {expected.disbelief}"
    )
    assert abs(actual.uncertainty - expected.uncertainty) < tol, (
        f"uncertainty: {actual.uncertainty} != {expected.uncertainty}"
    )
    assert abs(actual.base_rate - expected.base_rate) < tol, (
        f"base_rate: {actual.base_rate} != {expected.base_rate}"
    )


def assert_valid_opinion(opinion: Opinion, tol: float = TOL) -> None:
    """Assert b+d+u=1 and all components non-negative."""
    total = opinion.belief + opinion.disbelief + opinion.uncertainty
    assert abs(total - 1.0) < tol, f"b+d+u={total}, expected 1.0"
    assert opinion.belief >= -tol, f"belief={opinion.belief} < 0"
    assert opinion.disbelief >= -tol, f"disbelief={opinion.disbelief} < 0"
    assert opinion.uncertainty >= -tol, f"uncertainty={opinion.uncertainty} < 0"


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def vacuous() -> Opinion:
    """Complete ignorance: b=0, d=0, u=1."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def full_trust() -> Opinion:
    """Full trust: b=1, d=0, u=0."""
    return Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)


@pytest.fixture
def high_trust() -> Opinion:
    """High trust: b=0.9, d=0.02, u=0.08."""
    return Opinion(belief=0.9, disbelief=0.02, uncertainty=0.08)


@pytest.fixture
def moderate_trust() -> Opinion:
    """Moderate trust: b=0.6, d=0.1, u=0.3."""
    return Opinion(belief=0.6, disbelief=0.1, uncertainty=0.3)


@pytest.fixture
def low_trust() -> Opinion:
    """Low trust: b=0.2, d=0.5, u=0.3."""
    return Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)


def _build_chain_network(
    agent_ids: list[str],
    trust_opinions: list[Opinion],
) -> SLNetwork:
    """Build a linear trust chain: agent_ids[0] → [1] → ... → [n].

    Each agent is an agent node.  trust_opinions[i] is the trust
    from agent_ids[i] to agent_ids[i+1].
    """
    assert len(trust_opinions) == len(agent_ids) - 1
    net = SLNetwork(name="trust_chain")
    # Add agent nodes — agents get vacuous content opinion (they're agents, not propositions)
    for aid in agent_ids:
        net.add_node(SLNode(
            node_id=aid,
            opinion=Opinion(0.0, 0.0, 1.0),
            node_type="agent",
        ))
    # Add trust edges
    for i, top in enumerate(trust_opinions):
        net.add_trust_edge(TrustEdge(
            source_id=agent_ids[i],
            target_id=agent_ids[i + 1],
            trust_opinion=top,
        ))
    return net


# ═══════════════════════════════════════════════════════════════════
# SINGLE-HOP TRUST
# ═══════════════════════════════════════════════════════════════════


class TestSingleHopTrust:
    """Single-hop: Q → A.  Derived trust should match trust_discount directly."""

    def test_single_hop_matches_trust_discount(
        self, high_trust: Opinion
    ) -> None:
        """Q trusts A with ω_QA.  Derived trust Q→A = ω_QA (direct edge)."""
        net = _build_chain_network(["Q", "A"], [high_trust])
        result = propagate_trust(net, querying_agent="Q")

        assert isinstance(result, TrustPropagationResult)
        assert result.querying_agent == "Q"
        assert "A" in result.derived_trusts

        # Single hop: derived trust IS the direct trust opinion
        assert_opinion_close(result.derived_trusts["A"], high_trust)

    def test_single_hop_path(self, high_trust: Opinion) -> None:
        """The trust path for a single-hop is [Q, A]."""
        net = _build_chain_network(["Q", "A"], [high_trust])
        result = propagate_trust(net, querying_agent="Q")
        assert result.trust_paths["A"] == ["Q", "A"]

    def test_single_hop_bdu_invariant(self, moderate_trust: Opinion) -> None:
        """b+d+u=1 for single-hop derived trust."""
        net = _build_chain_network(["Q", "A"], [moderate_trust])
        result = propagate_trust(net, querying_agent="Q")
        assert_valid_opinion(result.derived_trusts["A"])

    def test_querying_agent_not_in_derived_trusts(
        self, high_trust: Opinion
    ) -> None:
        """The querying agent should not appear in its own derived_trusts."""
        net = _build_chain_network(["Q", "A"], [high_trust])
        result = propagate_trust(net, querying_agent="Q")
        assert "Q" not in result.derived_trusts


# ═══════════════════════════════════════════════════════════════════
# MULTI-HOP CHAIN (SINGLE PATH)
# ═══════════════════════════════════════════════════════════════════


class TestMultiHopChain:
    """Multi-hop chains: Q → A → B → C.  Left-fold trust_discount."""

    def test_two_hop_matches_manual(
        self, high_trust: Opinion, moderate_trust: Opinion
    ) -> None:
        """Q → A → B.  Q's trust in B = trust_discount(ω_QA, ω_AB)."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [high_trust, moderate_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        expected_QB = trust_discount(high_trust, moderate_trust)
        assert_opinion_close(result.derived_trusts["B"], expected_QB)

    def test_two_hop_also_has_single_hop(
        self, high_trust: Opinion, moderate_trust: Opinion
    ) -> None:
        """Q → A → B.  Q should also have derived trust for A."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [high_trust, moderate_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        assert "A" in result.derived_trusts
        assert_opinion_close(result.derived_trusts["A"], high_trust)

    def test_three_hop_matches_manual(
        self,
        high_trust: Opinion,
        moderate_trust: Opinion,
        low_trust: Opinion,
    ) -> None:
        """Q → A → B → C.  Left-fold: trust_discount(trust_discount(ω_QA, ω_AB), ω_BC)."""
        net = _build_chain_network(
            ["Q", "A", "B", "C"],
            [high_trust, moderate_trust, low_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        step1 = trust_discount(high_trust, moderate_trust)      # Q's trust in B
        expected_QC = trust_discount(step1, low_trust)           # Q's trust in C
        assert_opinion_close(result.derived_trusts["C"], expected_QC)

    def test_three_hop_paths(
        self,
        high_trust: Opinion,
        moderate_trust: Opinion,
        low_trust: Opinion,
    ) -> None:
        """Paths are recorded correctly for each agent in the chain."""
        net = _build_chain_network(
            ["Q", "A", "B", "C"],
            [high_trust, moderate_trust, low_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        assert result.trust_paths["A"] == ["Q", "A"]
        assert result.trust_paths["B"] == ["Q", "A", "B"]
        assert result.trust_paths["C"] == ["Q", "A", "B", "C"]

    def test_three_hop_all_bdu_valid(
        self,
        high_trust: Opinion,
        moderate_trust: Opinion,
        low_trust: Opinion,
    ) -> None:
        """b+d+u=1 holds for all derived trusts in the chain."""
        net = _build_chain_network(
            ["Q", "A", "B", "C"],
            [high_trust, moderate_trust, low_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        for agent_id, opinion in result.derived_trusts.items():
            assert_valid_opinion(opinion)

    def test_chain_inference_steps_recorded(
        self, high_trust: Opinion, moderate_trust: Opinion
    ) -> None:
        """Each trust_discount application is recorded as an InferenceStep."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [high_trust, moderate_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        # At least one step per derived trust
        assert len(result.steps) >= 1


# ═══════════════════════════════════════════════════════════════════
# FULL TRUST IDENTITY
# ═══════════════════════════════════════════════════════════════════


class TestFullTrustIdentity:
    """Full trust (b=1, d=0, u=0) passes opinions unchanged."""

    def test_full_trust_single_hop(self, full_trust: Opinion) -> None:
        """Full trust Q→A preserves A's trust opinion to downstream."""
        net = _build_chain_network(["Q", "A"], [full_trust])
        result = propagate_trust(net, querying_agent="Q")
        assert_opinion_close(result.derived_trusts["A"], full_trust)

    def test_full_trust_chain_preserves_final_edge(
        self, full_trust: Opinion, moderate_trust: Opinion
    ) -> None:
        """Q→A (full) → B (moderate).  Q's trust in B = moderate."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [full_trust, moderate_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        # trust_discount(full, moderate) should equal moderate
        expected = trust_discount(full_trust, moderate_trust)
        assert_opinion_close(result.derived_trusts["B"], expected)
        # And that should be close to moderate_trust itself
        assert_opinion_close(expected, moderate_trust)

    def test_full_trust_everywhere_preserves_all(
        self, full_trust: Opinion
    ) -> None:
        """All-full-trust chain of length 5 preserves identity."""
        agents = ["Q", "A", "B", "C", "D", "E"]
        trusts = [full_trust] * 5
        net = _build_chain_network(agents, trusts)
        result = propagate_trust(net, querying_agent="Q")

        # Every derived trust should be full trust
        for aid in agents[1:]:
            assert_opinion_close(result.derived_trusts[aid], full_trust)


# ═══════════════════════════════════════════════════════════════════
# VACUOUS TRUST DILUTION
# ═══════════════════════════════════════════════════════════════════


class TestVacuousTrustDilution:
    """Vacuous trust (b=0, d=0, u=1) at any hop yields vacuous downstream."""

    def test_vacuous_first_hop(
        self, vacuous: Opinion, high_trust: Opinion
    ) -> None:
        """Q→A (vacuous) → B.  Q's trust in B should be vacuous."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [vacuous, high_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        # Vacuous trust gives vacuous derived trust
        derived_B = result.derived_trusts["B"]
        assert derived_B.belief < TOL
        assert derived_B.disbelief < TOL
        assert abs(derived_B.uncertainty - 1.0) < TOL

    def test_vacuous_middle_hop(
        self, high_trust: Opinion, vacuous: Opinion
    ) -> None:
        """Q→A (high) → B (vacuous) → C.  Q's trust in C should be vacuous."""
        net = _build_chain_network(
            ["Q", "A", "B", "C"],
            [high_trust, vacuous, high_trust],
        )
        result = propagate_trust(net, querying_agent="Q")

        # B's derived trust from Q is trust_discount(high, vacuous)
        # Since vacuous has b=0, the result's belief = b_QA * 0 = 0
        derived_C = result.derived_trusts["C"]
        assert_valid_opinion(derived_C)
        # C should have very low belief due to vacuous middle hop
        assert derived_C.belief < TOL

    def test_vacuous_trust_bdu_invariant(
        self, vacuous: Opinion, moderate_trust: Opinion
    ) -> None:
        """b+d+u=1 holds even with vacuous trust in the chain."""
        net = _build_chain_network(
            ["Q", "A", "B"],
            [vacuous, moderate_trust],
        )
        result = propagate_trust(net, querying_agent="Q")
        for opinion in result.derived_trusts.values():
            assert_valid_opinion(opinion)


# ═══════════════════════════════════════════════════════════════════
# TRUST DEGRADATION ALONG CHAINS
# ═══════════════════════════════════════════════════════════════════


class TestTrustDegradation:
    """Trust diminishes monotonically along chains (Jøsang §14.3)."""

    def test_belief_decreases_along_chain(self) -> None:
        """Each hop reduces derived trust belief (for partial trust)."""
        trust_op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.1)
        agents = [f"agent_{i}" for i in range(6)]
        trusts = [trust_op] * 5
        net = _build_chain_network(agents, trusts)
        result = propagate_trust(net, querying_agent="agent_0")

        prev_belief = 1.0
        for aid in agents[1:]:
            current_belief = result.derived_trusts[aid].belief
            assert current_belief < prev_belief + TOL, (
                f"Belief did not decrease at {aid}: "
                f"{current_belief} >= {prev_belief}"
            )
            prev_belief = current_belief

    def test_uncertainty_increases_along_chain(self) -> None:
        """Each hop increases derived trust uncertainty (for partial trust)."""
        trust_op = Opinion(belief=0.85, disbelief=0.05, uncertainty=0.1)
        agents = [f"agent_{i}" for i in range(6)]
        trusts = [trust_op] * 5
        net = _build_chain_network(agents, trusts)
        result = propagate_trust(net, querying_agent="agent_0")

        prev_uncertainty = 0.0
        for aid in agents[1:]:
            current_u = result.derived_trusts[aid].uncertainty
            assert current_u > prev_uncertainty - TOL, (
                f"Uncertainty did not increase at {aid}: "
                f"{current_u} <= {prev_uncertainty}"
            )
            prev_uncertainty = current_u


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestPropagationEdgeCases:
    """Edge cases for propagate_trust."""

    def test_isolated_agent_no_derived_trusts(self) -> None:
        """An agent with no outgoing trust edges has empty derived_trusts."""
        net = SLNetwork(name="isolated")
        net.add_node(SLNode(
            node_id="lonely",
            opinion=Opinion(0.0, 0.0, 1.0),
            node_type="agent",
        ))
        result = propagate_trust(net, querying_agent="lonely")
        assert result.derived_trusts == {}
        assert result.trust_paths == {}
        assert result.steps == []

    def test_nonexistent_agent_raises(self) -> None:
        """Querying from a non-existent agent raises an error."""
        net = SLNetwork(name="empty")
        with pytest.raises(Exception):  # NodeNotFoundError or similar
            propagate_trust(net, querying_agent="ghost")

    def test_result_type(self, high_trust: Opinion) -> None:
        """propagate_trust returns a TrustPropagationResult."""
        net = _build_chain_network(["Q", "A"], [high_trust])
        result = propagate_trust(net, querying_agent="Q")
        assert isinstance(result, TrustPropagationResult)


# ═══════════════════════════════════════════════════════════════════
# MULTI-PATH TRUST FUSION (Tier 2, Step 3)
# ═══════════════════════════════════════════════════════════════════


def _build_diamond_network(
    trust_QA1: Opinion,
    trust_QA2: Opinion,
    trust_A1B: Opinion,
    trust_A2B: Opinion,
) -> SLNetwork:
    """Build a diamond trust graph: Q → A1 → B, Q → A2 → B.

    Q has two independent paths to B through A1 and A2.
    """
    net = SLNetwork(name="diamond_trust")
    vac = Opinion(0.0, 0.0, 1.0)
    for aid in ["Q", "A1", "A2", "B"]:
        net.add_node(SLNode(node_id=aid, opinion=vac, node_type="agent"))
    net.add_trust_edge(TrustEdge(source_id="Q", target_id="A1", trust_opinion=trust_QA1))
    net.add_trust_edge(TrustEdge(source_id="Q", target_id="A2", trust_opinion=trust_QA2))
    net.add_trust_edge(TrustEdge(source_id="A1", target_id="B", trust_opinion=trust_A1B))
    net.add_trust_edge(TrustEdge(source_id="A2", target_id="B", trust_opinion=trust_A2B))
    return net


class TestMultiPathTrustFusion:
    """Multi-path trust: Q reaches B via two paths, fused result.

    Per Jøsang (2016, §14.5): compute transitive trust along each
    path, then cumulative_fuse the derived trust opinions.
    """

    def test_diamond_fused_result_differs_from_either_path(
        self,
    ) -> None:
        """Fused trust through two paths differs from either single path."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        trust_QA1 = Opinion(0.9, 0.02, 0.08)
        trust_QA2 = Opinion(0.7, 0.1, 0.2)
        trust_A1B = Opinion(0.8, 0.05, 0.15)
        trust_A2B = Opinion(0.6, 0.15, 0.25)

        net = _build_diamond_network(trust_QA1, trust_QA2, trust_A1B, trust_A2B)
        result = propagate_trust(net, querying_agent="Q")

        assert "B" in result.derived_trusts
        fused_B = result.derived_trusts["B"]

        # Manual: each path independently
        path1 = trust_discount(trust_QA1, trust_A1B)
        path2 = trust_discount(trust_QA2, trust_A2B)
        expected_fused = cumulative_fuse(path1, path2)

        # Fused should match cumulative_fuse of the two paths
        assert_opinion_close(fused_B, expected_fused)

        # And it should differ from either single path
        assert abs(fused_B.belief - path1.belief) > 1e-6 or \
               abs(fused_B.uncertainty - path1.uncertainty) > 1e-6
        assert abs(fused_B.belief - path2.belief) > 1e-6 or \
               abs(fused_B.uncertainty - path2.uncertainty) > 1e-6

    def test_diamond_bdu_invariant(self) -> None:
        """b+d+u=1 for multi-path fused trust."""
        trust_QA1 = Opinion(0.9, 0.02, 0.08)
        trust_QA2 = Opinion(0.7, 0.1, 0.2)
        trust_A1B = Opinion(0.8, 0.05, 0.15)
        trust_A2B = Opinion(0.6, 0.15, 0.25)

        net = _build_diamond_network(trust_QA1, trust_QA2, trust_A1B, trust_A2B)
        result = propagate_trust(net, querying_agent="Q")

        for opinion in result.derived_trusts.values():
            assert_valid_opinion(opinion)

    def test_diamond_intermediate_agents_present(self) -> None:
        """Both A1 and A2 should have derived trusts (single-hop each)."""
        t = Opinion(0.8, 0.05, 0.15)
        net = _build_diamond_network(t, t, t, t)
        result = propagate_trust(net, querying_agent="Q")

        assert "A1" in result.derived_trusts
        assert "A2" in result.derived_trusts
        assert "B" in result.derived_trusts

    def test_diamond_fused_has_lower_uncertainty_than_either_path(
        self,
    ) -> None:
        """Cumulative fusion reduces uncertainty compared to single paths."""
        trust_QA1 = Opinion(0.8, 0.05, 0.15)
        trust_QA2 = Opinion(0.7, 0.1, 0.2)
        trust_A1B = Opinion(0.85, 0.05, 0.1)
        trust_A2B = Opinion(0.75, 0.1, 0.15)

        net = _build_diamond_network(trust_QA1, trust_QA2, trust_A1B, trust_A2B)
        result = propagate_trust(net, querying_agent="Q")

        fused_B = result.derived_trusts["B"]
        path1 = trust_discount(trust_QA1, trust_A1B)
        path2 = trust_discount(trust_QA2, trust_A2B)

        # Cumulative fusion reduces uncertainty
        assert fused_B.uncertainty < path1.uncertainty + TOL
        assert fused_B.uncertainty < path2.uncertainty + TOL

    def test_diamond_symmetric_paths_same_as_double_evidence(
        self,
    ) -> None:
        """Symmetric diamond: both paths identical → same as fusing two copies."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        t_qa = Opinion(0.85, 0.05, 0.1)
        t_ab = Opinion(0.8, 0.05, 0.15)

        net = _build_diamond_network(t_qa, t_qa, t_ab, t_ab)
        result = propagate_trust(net, querying_agent="Q")

        single_path = trust_discount(t_qa, t_ab)
        expected = cumulative_fuse(single_path, single_path)

        assert_opinion_close(result.derived_trusts["B"], expected)

    def test_three_paths_to_target(self) -> None:
        """Q → A1 → B, Q → A2 → B, Q → A3 → B.  Three-path fusion."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        vac = Opinion(0.0, 0.0, 1.0)
        net = SLNetwork(name="three_path")
        for aid in ["Q", "A1", "A2", "A3", "B"]:
            net.add_node(SLNode(node_id=aid, opinion=vac, node_type="agent"))

        t_qa1 = Opinion(0.9, 0.02, 0.08)
        t_qa2 = Opinion(0.7, 0.1, 0.2)
        t_qa3 = Opinion(0.6, 0.15, 0.25)
        t_a1b = Opinion(0.85, 0.05, 0.1)
        t_a2b = Opinion(0.75, 0.1, 0.15)
        t_a3b = Opinion(0.65, 0.15, 0.2)

        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A1", trust_opinion=t_qa1))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A2", trust_opinion=t_qa2))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A3", trust_opinion=t_qa3))
        net.add_trust_edge(TrustEdge(source_id="A1", target_id="B", trust_opinion=t_a1b))
        net.add_trust_edge(TrustEdge(source_id="A2", target_id="B", trust_opinion=t_a2b))
        net.add_trust_edge(TrustEdge(source_id="A3", target_id="B", trust_opinion=t_a3b))

        result = propagate_trust(net, querying_agent="Q")

        p1 = trust_discount(t_qa1, t_a1b)
        p2 = trust_discount(t_qa2, t_a2b)
        p3 = trust_discount(t_qa3, t_a3b)
        expected = cumulative_fuse(p1, p2, p3)

        assert_opinion_close(result.derived_trusts["B"], expected)
        assert_valid_opinion(result.derived_trusts["B"])

    def test_multi_path_with_averaging_fusion(self) -> None:
        """Multi-path trust with averaging fusion instead of cumulative."""
        from jsonld_ex.confidence_algebra import averaging_fuse

        trust_QA1 = Opinion(0.9, 0.02, 0.08)
        trust_QA2 = Opinion(0.7, 0.1, 0.2)
        trust_A1B = Opinion(0.8, 0.05, 0.15)
        trust_A2B = Opinion(0.6, 0.15, 0.25)

        net = _build_diamond_network(trust_QA1, trust_QA2, trust_A1B, trust_A2B)
        result = propagate_trust(
            net, querying_agent="Q", fusion_method="averaging"
        )

        p1 = trust_discount(trust_QA1, trust_A1B)
        p2 = trust_discount(trust_QA2, trust_A2B)
        expected = averaging_fuse(p1, p2)

        assert_opinion_close(result.derived_trusts["B"], expected)

    def test_longer_multi_path_chain(self) -> None:
        """Q → A → C → D and Q → B → D.  Mixed chain lengths."""
        from jsonld_ex.confidence_algebra import cumulative_fuse

        vac = Opinion(0.0, 0.0, 1.0)
        net = SLNetwork(name="mixed_chain")
        for aid in ["Q", "A", "B", "C", "D"]:
            net.add_node(SLNode(node_id=aid, opinion=vac, node_type="agent"))

        t_qa = Opinion(0.9, 0.02, 0.08)
        t_qb = Opinion(0.8, 0.05, 0.15)
        t_ac = Opinion(0.85, 0.05, 0.1)
        t_cd = Opinion(0.7, 0.1, 0.2)
        t_bd = Opinion(0.75, 0.1, 0.15)

        net.add_trust_edge(TrustEdge(source_id="Q", target_id="A", trust_opinion=t_qa))
        net.add_trust_edge(TrustEdge(source_id="Q", target_id="B", trust_opinion=t_qb))
        net.add_trust_edge(TrustEdge(source_id="A", target_id="C", trust_opinion=t_ac))
        net.add_trust_edge(TrustEdge(source_id="C", target_id="D", trust_opinion=t_cd))
        net.add_trust_edge(TrustEdge(source_id="B", target_id="D", trust_opinion=t_bd))

        result = propagate_trust(net, querying_agent="Q")

        # Path 1: Q→A→C→D (3-hop)
        q_a = t_qa
        q_c = trust_discount(q_a, t_ac)
        path1_d = trust_discount(q_c, t_cd)

        # Path 2: Q→B→D (2-hop)
        q_b = t_qb
        path2_d = trust_discount(q_b, t_bd)

        expected_d = cumulative_fuse(path1_d, path2_d)
        assert_opinion_close(result.derived_trusts["D"], expected_d)
        assert_valid_opinion(result.derived_trusts["D"])
