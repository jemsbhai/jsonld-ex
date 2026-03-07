"""
Tests for combined trust + content inference (Tier 2, Step 5).

Covers infer_with_trust():
    - Trust discounting applied before content deduction
    - Multiple attestations fused after trust-discounting
    - Full trust preserves attestation opinions
    - Vacuous trust yields vacuous content input
    - End-to-end: trust propagation → attestation discounting → deduction
    - Network method delegation

TDD RED PHASE: These tests should FAIL until infer_with_trust is implemented.

References:
    Jøsang, A. (2016). Subjective Logic, §14.3 (trust discount),
    §12.6 (deduction).
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    deduce,
    trust_discount,
)
from jsonld_ex.sl_network.counterfactuals import vacuous_counterfactual
from jsonld_ex.sl_network.inference import infer_node
from jsonld_ex.sl_network.network import SLNetwork
from jsonld_ex.sl_network.trust import propagate_trust
from jsonld_ex.sl_network.types import (
    AttestationEdge,
    InferenceResult,
    SLEdge,
    SLNode,
    TrustEdge,
)

# Import the function under test — will fail until implemented
from jsonld_ex.sl_network.trust import infer_with_trust


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

TOL = 1e-9

VAC = Opinion(0.0, 0.0, 1.0)


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


def assert_valid_opinion(opinion: Opinion, tol: float = TOL) -> None:
    """Assert b+d+u=1 and all components non-negative."""
    total = opinion.belief + opinion.disbelief + opinion.uncertainty
    assert abs(total - 1.0) < tol, f"b+d+u={total}, expected 1.0"
    assert opinion.belief >= -tol
    assert opinion.disbelief >= -tol
    assert opinion.uncertainty >= -tol


# ═══════════════════════════════════════════════════════════════════
# SINGLE AGENT, SINGLE CONTENT NODE (simplest case)
# ═══════════════════════════════════════════════════════════════════


class TestSingleAgentSingleContent:
    """Q trusts A, A attests to claim_x.  No deduction, just trust-discount."""

    def _build(
        self, trust_op: Opinion, attest_op: Opinion
    ) -> SLNetwork:
        net = SLNetwork(name="simple")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="claim_x", opinion=VAC))
        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=trust_op,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="A", content_id="claim_x", opinion=attest_op,
        ))
        return net

    def test_result_matches_manual_trust_discount(self) -> None:
        """infer_with_trust for a single attestation matches trust_discount."""
        trust_op = Opinion(0.9, 0.02, 0.08)
        attest_op = Opinion(0.8, 0.1, 0.1)
        net = self._build(trust_op, attest_op)

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )

        expected = trust_discount(trust_op, attest_op)
        assert isinstance(result, InferenceResult)
        assert_opinion_close(result.opinion, expected)

    def test_full_trust_preserves_attestation(self) -> None:
        """Full trust (b=1) passes attestation opinion unchanged."""
        full_trust = Opinion(1.0, 0.0, 0.0)
        attest_op = Opinion(0.75, 0.1, 0.15)
        net = self._build(full_trust, attest_op)

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        assert_opinion_close(result.opinion, attest_op)

    def test_vacuous_trust_yields_vacuous_content(self) -> None:
        """Vacuous trust yields vacuous opinion at content node."""
        vac_trust = Opinion(0.0, 0.0, 1.0)
        attest_op = Opinion(0.8, 0.1, 0.1)
        net = self._build(vac_trust, attest_op)

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        # trust_discount with vacuous trust: b=0, d=0, u=1
        assert result.opinion.belief < TOL
        assert result.opinion.disbelief < TOL
        assert abs(result.opinion.uncertainty - 1.0) < TOL

    def test_bdu_invariant(self) -> None:
        """b+d+u=1 at query node."""
        trust_op = Opinion(0.6, 0.15, 0.25)
        attest_op = Opinion(0.7, 0.2, 0.1)
        net = self._build(trust_op, attest_op)

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        assert_valid_opinion(result.opinion)


# ═══════════════════════════════════════════════════════════════════
# MULTIPLE AGENTS ATTESTING SAME CONTENT NODE
# ═══════════════════════════════════════════════════════════════════


class TestMultipleAttestations:
    """Two agents attest to the same content node.  Discounted then fused."""

    def _build(self) -> tuple[SLNetwork, Opinion, Opinion, Opinion, Opinion]:
        trust_QA = Opinion(0.9, 0.02, 0.08)
        trust_QB = Opinion(0.7, 0.1, 0.2)
        attest_A = Opinion(0.8, 0.05, 0.15)
        attest_B = Opinion(0.6, 0.2, 0.2)

        net = SLNetwork(name="multi_attest")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="B", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="claim_x", opinion=VAC))

        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=trust_QA,
        ))
        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="B", trust_opinion=trust_QB,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="A", content_id="claim_x", opinion=attest_A,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="B", content_id="claim_x", opinion=attest_B,
        ))
        return net, trust_QA, trust_QB, attest_A, attest_B

    def test_fused_matches_manual(self) -> None:
        """Two discounted attestations fused matches manual computation."""
        net, trust_QA, trust_QB, attest_A, attest_B = self._build()

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )

        disc_A = trust_discount(trust_QA, attest_A)
        disc_B = trust_discount(trust_QB, attest_B)
        expected = cumulative_fuse(disc_A, disc_B)
        assert_opinion_close(result.opinion, expected)

    def test_fused_lower_uncertainty_than_either(self) -> None:
        """Fusing two attestations reduces uncertainty."""
        net, trust_QA, trust_QB, attest_A, attest_B = self._build()

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )

        disc_A = trust_discount(trust_QA, attest_A)
        disc_B = trust_discount(trust_QB, attest_B)
        assert result.opinion.uncertainty < disc_A.uncertainty + TOL
        assert result.opinion.uncertainty < disc_B.uncertainty + TOL

    def test_bdu_invariant(self) -> None:
        net, *_ = self._build()
        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        assert_valid_opinion(result.opinion)


# ═══════════════════════════════════════════════════════════════════
# TRUST + DEDUCTION (end-to-end)
# ═══════════════════════════════════════════════════════════════════


class TestTrustPlusDeduction:
    """Agent attests to a root node; deduction propagates to child."""

    def _build(self) -> tuple[SLNetwork, Opinion, Opinion, Opinion]:
        """Q trusts A.  A attests claim_x.  claim_x → claim_y (deduction)."""
        trust_QA = Opinion(0.9, 0.02, 0.08)
        attest_A = Opinion(0.8, 0.05, 0.15)
        conditional = Opinion(0.85, 0.05, 0.1)

        net = SLNetwork(name="trust_deduction")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="claim_x", opinion=VAC))
        net.add_node(SLNode(node_id="claim_y", opinion=VAC))

        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=trust_QA,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="A", content_id="claim_x", opinion=attest_A,
        ))
        net.add_edge(SLEdge(
            source_id="claim_x", target_id="claim_y",
            conditional=conditional,
        ))

        return net, trust_QA, attest_A, conditional

    def test_deduction_uses_trust_discounted_root(self) -> None:
        """claim_y opinion derived by deducing from trust-discounted claim_x."""
        net, trust_QA, attest_A, conditional = self._build()

        result = infer_with_trust(
            net, query_node="claim_y", querying_agent="Q",
        )

        # Step 1: trust-discount the attestation
        disc_x = trust_discount(trust_QA, attest_A)
        # Step 2: deduce claim_y from disc_x
        cf = vacuous_counterfactual(conditional)
        expected_y = deduce(disc_x, conditional, cf)

        assert_opinion_close(result.opinion, expected_y)

    def test_can_also_query_root_node(self) -> None:
        """Querying the attested root gives the trust-discounted opinion."""
        net, trust_QA, attest_A, _ = self._build()

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        expected = trust_discount(trust_QA, attest_A)
        assert_opinion_close(result.opinion, expected)

    def test_all_opinions_valid(self) -> None:
        """b+d+u=1 at every node in the inference."""
        net, *_ = self._build()
        result = infer_with_trust(
            net, query_node="claim_y", querying_agent="Q",
        )
        assert_valid_opinion(result.opinion)
        for op in result.intermediate_opinions.values():
            assert_valid_opinion(op)


# ═══════════════════════════════════════════════════════════════════
# TRANSITIVE TRUST + ATTESTATION + DEDUCTION
# ═══════════════════════════════════════════════════════════════════


class TestTransitiveTrustDeduction:
    """Q → A → B (trust chain).  B attests claim_x.  claim_x → claim_y."""

    def test_two_hop_trust_to_deduction(self) -> None:
        trust_QA = Opinion(0.9, 0.02, 0.08)
        trust_AB = Opinion(0.8, 0.05, 0.15)
        attest_B = Opinion(0.85, 0.05, 0.1)
        conditional = Opinion(0.9, 0.03, 0.07)

        net = SLNetwork(name="transitive")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="B", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="claim_x", opinion=VAC))
        net.add_node(SLNode(node_id="claim_y", opinion=VAC))

        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=trust_QA,
        ))
        net.add_trust_edge(TrustEdge(
            source_id="A", target_id="B", trust_opinion=trust_AB,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="B", content_id="claim_x", opinion=attest_B,
        ))
        net.add_edge(SLEdge(
            source_id="claim_x", target_id="claim_y",
            conditional=conditional,
        ))

        result = infer_with_trust(
            net, query_node="claim_y", querying_agent="Q",
        )

        # Manual: Q's transitive trust in B
        q_trust_B = trust_discount(trust_QA, trust_AB)
        # Trust-discount B's attestation
        disc_x = trust_discount(q_trust_B, attest_B)
        # Deduce claim_y
        cf = vacuous_counterfactual(conditional)
        expected_y = deduce(disc_x, conditional, cf)

        assert_opinion_close(result.opinion, expected_y)
        assert_valid_opinion(result.opinion)


# ═══════════════════════════════════════════════════════════════════
# NETWORK METHOD DELEGATION
# ═══════════════════════════════════════════════════════════════════


class TestNetworkInferWithTrust:
    """SLNetwork.infer_with_trust() delegates correctly."""

    def test_network_method_matches_module_function(self) -> None:
        trust_op = Opinion(0.9, 0.02, 0.08)
        attest_op = Opinion(0.8, 0.1, 0.1)

        net = SLNetwork(name="delegation")
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_agent(SLNode(node_id="A", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="claim_x", opinion=VAC))
        net.add_trust_edge(TrustEdge(
            source_id="Q", target_id="A", trust_opinion=trust_op,
        ))
        net.add_attestation(AttestationEdge(
            agent_id="A", content_id="claim_x", opinion=attest_op,
        ))

        mod_result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        net_result = net.infer_with_trust(
            query_node="claim_x", querying_agent="Q",
        )

        assert_opinion_close(net_result.opinion, mod_result.opinion)

    def test_returns_inference_result(self) -> None:
        net = SLNetwork()
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        net.add_node(SLNode(node_id="c", opinion=Opinion(0.5, 0.2, 0.3)))

        result = net.infer_with_trust(
            query_node="c", querying_agent="Q",
        )
        assert isinstance(result, InferenceResult)


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════


class TestInferWithTrustEdgeCases:
    """Edge cases for infer_with_trust."""

    def test_content_node_with_no_attestations_uses_own_opinion(self) -> None:
        """A content node with no attestations keeps its marginal opinion."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        original_op = Opinion(0.6, 0.2, 0.2)
        net.add_node(SLNode(node_id="claim_x", opinion=original_op))

        result = infer_with_trust(
            net, query_node="claim_x", querying_agent="Q",
        )
        # No attestations → node keeps its own opinion
        assert_opinion_close(result.opinion, original_op)

    def test_nonexistent_query_node_raises(self) -> None:
        """Querying a nonexistent content node raises."""
        net = SLNetwork()
        net.add_agent(SLNode(node_id="Q", opinion=VAC, node_type="agent"))
        with pytest.raises(Exception):
            infer_with_trust(
                net, query_node="ghost", querying_agent="Q",
            )

    def test_nonexistent_querying_agent_raises(self) -> None:
        """Querying from a nonexistent agent raises."""
        net = SLNetwork()
        net.add_node(SLNode(node_id="c", opinion=VAC))
        with pytest.raises(Exception):
            infer_with_trust(
                net, query_node="c", querying_agent="ghost",
            )
