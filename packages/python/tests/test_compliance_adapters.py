"""Tests for compliance protocol adapters.

Protocol adapters bridge the compliance algebra's erasure propagation
(§9) and review-due trigger (§8.2) to arbitrary data infrastructure
via typed protocols. Organizations implement two small protocols
(~4 methods each) and get the full algebra for free.

Architecture:
    LineageProvider        — §9 erasure_scope + residual_contamination
    ReviewScheduleProvider — §8.2 review_due_trigger
    SimpleLineageGraph     — reference implementation (in-memory DAG)
    SimpleReviewSchedule   — reference implementation (in-memory schedule)

Composition functions:
    erasure_scope_assessment(source, lineage, ...)
    contamination_risk(node, lineage, ...)
    review_due_assessment(opinion, assessment_id, time, schedule)

Mathematical source of truth: compliance_algebra.md
"""

import math
import pytest

from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    erasure_scope_opinion,
    jurisdictional_meet,
    residual_contamination,
    review_due_trigger,
)
from jsonld_ex.compliance_adapters import (
    # Protocols
    LineageProvider,
    ReviewScheduleProvider,
    # Reference implementations
    SimpleLineageGraph,
    SimpleReviewSchedule,
    # Composition functions
    erasure_scope_assessment,
    contamination_risk,
    review_due_assessment,
)


# ── Tolerance ─────────────────────────────────────────────────────
TOL = 1e-9


def assert_valid_co(op, msg=""):
    prefix = f"{msg}: " if msg else ""
    assert isinstance(op, ComplianceOpinion), (
        f"{prefix}expected ComplianceOpinion, got {type(op).__name__}"
    )
    assert op.lawfulness >= -TOL, f"{prefix}l < 0: {op.lawfulness}"
    assert op.violation >= -TOL, f"{prefix}v < 0: {op.violation}"
    assert op.uncertainty >= -TOL, f"{prefix}u < 0: {op.uncertainty}"
    total = op.lawfulness + op.violation + op.uncertainty
    assert abs(total - 1.0) < TOL, f"{prefix}l+v+u={total}"


# ═══════════════════════════════════════════════════════════════════
# PROTOCOL CONFORMANCE
# ═══════════════════════════════════════════════════════════════════


class TestProtocolConformance:
    """Verify reference implementations satisfy their protocols."""

    def test_simple_lineage_graph_is_lineage_provider(self):
        """SimpleLineageGraph must be a structural subtype of LineageProvider."""
        graph = SimpleLineageGraph()
        assert isinstance(graph, LineageProvider)

    def test_simple_review_schedule_is_review_schedule_provider(self):
        """SimpleReviewSchedule must be a structural subtype of ReviewScheduleProvider."""
        schedule = SimpleReviewSchedule()
        assert isinstance(schedule, ReviewScheduleProvider)

    def test_lineage_provider_has_required_methods(self):
        """Protocol requires get_descendants, get_ancestors,
        get_erasure_opinion, get_exempt_nodes."""
        graph = SimpleLineageGraph()
        assert callable(getattr(graph, "get_descendants", None))
        assert callable(getattr(graph, "get_ancestors", None))
        assert callable(getattr(graph, "get_erasure_opinion", None))
        assert callable(getattr(graph, "get_exempt_nodes", None))

    def test_review_schedule_has_required_methods(self):
        """Protocol requires get_review_due, get_half_life,
        get_accelerated_half_life."""
        schedule = SimpleReviewSchedule()
        assert callable(getattr(schedule, "get_review_due", None))
        assert callable(getattr(schedule, "get_half_life", None))
        assert callable(getattr(schedule, "get_accelerated_half_life", None))


# ═══════════════════════════════════════════════════════════════════
# SimpleLineageGraph
# ═══════════════════════════════════════════════════════════════════


class TestSimpleLineageGraph:
    """Reference implementation: in-memory DAG for data lineage."""

    def test_empty_graph(self):
        """Empty graph has no descendants or ancestors."""
        g = SimpleLineageGraph()
        assert g.get_descendants("A") == set()
        assert g.get_ancestors("A") == set()

    def test_add_edge_and_descendants(self):
        """A→B means B is a descendant of A."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        assert "B" in g.get_descendants("A")
        assert "A" not in g.get_descendants("A")

    def test_add_edge_and_ancestors(self):
        """A→B means A is an ancestor of B."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        assert "A" in g.get_ancestors("B")
        assert "B" not in g.get_ancestors("B")

    def test_transitive_descendants(self):
        """A→B→C: C is a descendant of A (transitive closure)."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        desc_a = g.get_descendants("A")
        assert "B" in desc_a
        assert "C" in desc_a

    def test_transitive_ancestors(self):
        """A→B→C: A is an ancestor of C (transitive closure)."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        anc_c = g.get_ancestors("C")
        assert "B" in anc_c
        assert "A" in anc_c

    def test_diamond_graph(self):
        """Diamond: A→B, A→C, B→D, C→D. D has ancestors {A,B,C}."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        assert g.get_ancestors("D") == {"A", "B", "C"}
        assert g.get_descendants("A") == {"B", "C", "D"}

    def test_set_erasure_opinion(self):
        """Erasure opinions are stored per-node."""
        g = SimpleLineageGraph()
        op = ComplianceOpinion.create(0.9, 0.05, 0.05, 0.5)
        g.set_erasure_opinion("A", op)
        retrieved = g.get_erasure_opinion("A")
        assert abs(retrieved.lawfulness - 0.9) < TOL

    def test_default_erasure_opinion_is_vacuous(self):
        """Nodes without explicit erasure opinion → vacuous (u=1).

        This is the correct epistemic default: we have no evidence
        about whether erasure was completed at this node.
        """
        g = SimpleLineageGraph()
        op = g.get_erasure_opinion("unknown-node")
        assert op.uncertainty > 0.9

    def test_exempt_nodes(self):
        """Exempt nodes (Art. 17(3) exceptions) are tracked."""
        g = SimpleLineageGraph()
        g.add_exempt("D", reason="Art. 17(3)(b) public interest")
        assert "D" in g.get_exempt_nodes()

    def test_exempt_nodes_empty_default(self):
        """No exemptions by default."""
        g = SimpleLineageGraph()
        assert g.get_exempt_nodes() == set()

    def test_scope_definition(self):
        """Scope(R) = {source} ∪ descendants(source) \\ exempt.

        Per Definition 15 (§9.1).
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        g.add_edge("B", "D")
        g.add_exempt("D", reason="public interest")
        # Scope = {A, B, C, D} \ {D} = {A, B, C}
        scope = g.get_scope("A")
        assert scope == {"A", "B", "C"}

    def test_scope_includes_source(self):
        """Source node is always in scope."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        assert "A" in g.get_scope("A")

    def test_scope_with_no_descendants(self):
        """Leaf node scope = {node} if not exempt."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        assert g.get_scope("B") == {"B"}


# ═══════════════════════════════════════════════════════════════════
# SimpleReviewSchedule
# ═══════════════════════════════════════════════════════════════════


class TestSimpleReviewSchedule:
    """Reference implementation: in-memory review schedule."""

    def test_set_and_get_review_due(self):
        """Set a review-due date and retrieve it."""
        s = SimpleReviewSchedule()
        s.set_review_due("dpia-001", due_time=365.0)
        assert s.get_review_due("dpia-001") == 365.0

    def test_unknown_assessment_returns_none(self):
        """Unknown assessment ID → None (no review scheduled)."""
        s = SimpleReviewSchedule()
        assert s.get_review_due("unknown") is None

    def test_default_half_life(self):
        """Default half-life applies to all assessments."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        assert s.get_half_life("anything") == 730.0

    def test_per_assessment_half_life(self):
        """Per-assessment half-life overrides default."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        s.set_half_life("dpia-001", 365.0)
        assert s.get_half_life("dpia-001") == 365.0
        assert s.get_half_life("other") == 730.0

    def test_default_accelerated_half_life(self):
        """Default accelerated half-life = half_life / 4."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        assert s.get_accelerated_half_life("x") == pytest.approx(
            730.0 / 4.0
        )

    def test_per_assessment_accelerated_half_life(self):
        """Per-assessment accelerated half-life overrides default."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        s.set_accelerated_half_life("dpia-001", 90.0)
        assert s.get_accelerated_half_life("dpia-001") == 90.0
        # Other assessments still use default
        assert s.get_accelerated_half_life("other") == pytest.approx(
            730.0 / 4.0
        )


# ═══════════════════════════════════════════════════════════════════
# erasure_scope_assessment() — §9.2 composition function
# ═══════════════════════════════════════════════════════════════════


class TestErasureScopeAssessment:
    """Composite erasure completeness across lineage scope.

    Per §9.2 (Theorem 5):
        ω_E^R = J⊓(ω_e^i : D_i ∈ S)
    where S = Scope(R) \\ Exempt(R).

    Theorem 5 properties:
        (a) Exponential degradation: e_R = ∏ e_i
        (b) Scope monotonicity: adding a node decreases e_R
        (c) Exception filtering: removing a node increases e_R
        (d) Perfect source erasure: e=1 is identity
    """

    def _build_linear_graph(self, n, erasure_confidence=0.95):
        """Build a linear chain: D0→D1→...→D(n-1) with uniform erasure."""
        g = SimpleLineageGraph()
        for i in range(n - 1):
            g.add_edge(f"D{i}", f"D{i+1}")
        op = ComplianceOpinion.create(
            erasure_confidence, 0.03, 1.0 - erasure_confidence - 0.03, 0.5,
        )
        for i in range(n):
            g.set_erasure_opinion(f"D{i}", op)
        return g

    def test_single_node_erasure(self):
        """Single node scope → erasure opinion unchanged.

        Base case: one-node lineage, scope = {source}.
        """
        g = SimpleLineageGraph()
        op = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        g.set_erasure_opinion("A", op)
        result = erasure_scope_assessment("A", g)
        assert abs(result.lawfulness - 0.90) < TOL

    def test_two_node_multiplicative(self):
        """Two nodes: e_R = e_1 · e_2 (Theorem 5(a)).

        Composite erasure = product of individual erasures.
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        op_a = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        op_b = ComplianceOpinion.create(0.80, 0.10, 0.10, 0.5)
        g.set_erasure_opinion("A", op_a)
        g.set_erasure_opinion("B", op_b)

        result = erasure_scope_assessment("A", g)
        # Verify multiplicative: l_R ≈ 0.90 * 0.80 = 0.72
        assert abs(result.lawfulness - 0.90 * 0.80) < 0.01

    def test_exponential_degradation_with_scope_size(self):
        """Theorem 5(a): e_R = p^|S| with uniform confidence p.

        Larger scope → exponentially lower composite confidence.
        """
        for n in [3, 5, 10]:
            p = 0.95
            g = self._build_linear_graph(n, erasure_confidence=p)
            result = erasure_scope_assessment("D0", g)
            expected = p ** n
            assert abs(result.lawfulness - expected) < 0.05, (
                f"n={n}: expected e_R≈{expected:.4f}, got {result.lawfulness:.4f}"
            )

    def test_scope_monotonicity(self):
        """Theorem 5(b): adding a node can only decrease e_R.

        Longer chain → lower composite erasure confidence.
        """
        g3 = self._build_linear_graph(3)
        g5 = self._build_linear_graph(5)
        r3 = erasure_scope_assessment("D0", g3)
        r5 = erasure_scope_assessment("D0", g5)
        assert r5.lawfulness < r3.lawfulness

    def test_exception_filtering_improves_erasure(self):
        """Theorem 5(c): exempting a node increases e_R.

        Art. 17(3) exception removes a node from scope,
        which can only improve composite erasure.
        """
        g_no_exempt = SimpleLineageGraph()
        g_no_exempt.add_edge("A", "B")
        g_no_exempt.add_edge("B", "C")
        op = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        for node in ["A", "B", "C"]:
            g_no_exempt.set_erasure_opinion(node, op)

        g_exempt = SimpleLineageGraph()
        g_exempt.add_edge("A", "B")
        g_exempt.add_edge("B", "C")
        for node in ["A", "B", "C"]:
            g_exempt.set_erasure_opinion(node, op)
        g_exempt.add_exempt("C", reason="Art. 17(3)(d) archiving")

        r_no = erasure_scope_assessment("A", g_no_exempt)
        r_ex = erasure_scope_assessment("A", g_exempt)
        assert r_ex.lawfulness > r_no.lawfulness

    def test_perfect_source_erasure_is_identity(self):
        """Theorem 5(d): e=1 contributes no degradation.

        If source has perfect erasure, the composite only
        depends on descendants.
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        perfect = ComplianceOpinion.create(1.0, 0.0, 0.0, 1.0)
        partial = ComplianceOpinion.create(0.80, 0.10, 0.10, 0.5)
        g.set_erasure_opinion("A", perfect)
        g.set_erasure_opinion("B", partial)

        result = erasure_scope_assessment("A", g)
        # Perfect source is multiplicative identity → result ≈ partial
        assert abs(result.lawfulness - 0.80) < 0.01

    def test_empty_scope_raises(self):
        """All nodes exempt → empty scope → error.

        Cannot assess erasure completeness of empty set.
        """
        g = SimpleLineageGraph()
        g.add_exempt("A", reason="all exempt")
        with pytest.raises(ValueError, match="[Ee]mpty|[Nn]o nodes"):
            erasure_scope_assessment("A", g)

    def test_constraint_preservation(self):
        """Theorem 5 inherits from Theorem 1: l+v+u = 1."""
        g = self._build_linear_graph(5)
        result = erasure_scope_assessment("D0", g)
        assert_valid_co(result, "erasure scope")

    def test_delegates_to_erasure_scope_opinion(self):
        """Composition function must delegate to the algebra.

        The adapter's job is to gather opinions from the lineage
        provider and feed them to erasure_scope_opinion().
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        op_a = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        op_b = ComplianceOpinion.create(0.80, 0.10, 0.10, 0.5)
        g.set_erasure_opinion("A", op_a)
        g.set_erasure_opinion("B", op_b)

        adapter_result = erasure_scope_assessment("A", g)
        direct_result = erasure_scope_opinion(op_a, op_b)
        assert abs(adapter_result.lawfulness - direct_result.lawfulness) < TOL
        assert abs(adapter_result.violation - direct_result.violation) < TOL


# ═══════════════════════════════════════════════════════════════════
# contamination_risk() — §9.3 composition function
# ═══════════════════════════════════════════════════════════════════


class TestContaminationRisk:
    """Residual contamination risk at a specific node.

    Per Definition 17 (§9.3):
        r_j   = 1 − ∏(1 − ē_i)   contamination risk
        r̄_j   = ∏ e_i             clean probability
        u_r^j = ∏(1−ē_i) − ∏ e_i  uncertainty

    Proposition 1:
        r + r̄ + u_r = 1, all non-negative.
        r is monotonically non-decreasing in |A_j⁺|.
    """

    def test_leaf_node_own_risk(self):
        """Leaf node with no ancestors: risk = own erasure opinion.

        A_j⁺ = {D_j} (just the node itself).
        """
        g = SimpleLineageGraph()
        op = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        g.set_erasure_opinion("A", op)
        result = contamination_risk("A", g)
        assert_valid_co(result, "leaf contamination")
        # contamination (violation) = ē = disbelief = 0.05
        # clean (lawfulness) = e = belief = 0.90
        assert abs(result.violation - 0.05) < TOL
        assert abs(result.lawfulness - 0.90) < TOL

    def test_deep_node_higher_risk(self):
        """Deeper nodes have higher contamination risk.

        Proposition 1: r is monotonically non-decreasing in |A_j⁺|.
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        op = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        for node in ["A", "B", "C"]:
            g.set_erasure_opinion(node, op)

        risk_b = contamination_risk("B", g)  # ancestors: {A}
        risk_c = contamination_risk("C", g)  # ancestors: {A, B}

        assert risk_c.violation >= risk_b.violation - TOL

    def test_disjunctive_contamination(self):
        """Contamination is disjunctive: any ancestor persisting → risk.

        r = 1 − ∏(1 − ē_i). If any ē_i > 0, r > 0.
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        clean = ComplianceOpinion.create(1.0, 0.0, 0.0, 1.0)
        dirty = ComplianceOpinion.create(0.50, 0.40, 0.10, 0.5)
        g.set_erasure_opinion("A", dirty)
        g.set_erasure_opinion("B", clean)

        result = contamination_risk("B", g)
        # B is clean but A is dirty → contamination at B
        assert result.violation > 0.3

    def test_constraint_preservation(self):
        """Proposition 1: r + r̄ + u_r = 1."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("B", "C")
        op = ComplianceOpinion.create(0.85, 0.08, 0.07, 0.5)
        for node in ["A", "B", "C"]:
            g.set_erasure_opinion(node, op)

        for node in ["A", "B", "C"]:
            result = contamination_risk(node, g)
            assert_valid_co(result, f"contamination {node}")

    def test_diamond_contamination(self):
        """Diamond graph: A→B, A→C, B→D, C→D.

        Node D has ancestors {A, B, C}. Contamination from any
        of them contributes to D's risk.
        """
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        g.add_edge("A", "C")
        g.add_edge("B", "D")
        g.add_edge("C", "D")
        op = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        for node in ["A", "B", "C", "D"]:
            g.set_erasure_opinion(node, op)

        result = contamination_risk("D", g)
        assert_valid_co(result, "diamond contamination")
        # D has 4 nodes in A_j⁺ → higher risk than single node
        single = contamination_risk("A", g)
        assert result.violation > single.violation

    def test_delegates_to_residual_contamination(self):
        """Adapter must delegate to algebra's residual_contamination()."""
        g = SimpleLineageGraph()
        g.add_edge("A", "B")
        op_a = ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5)
        op_b = ComplianceOpinion.create(0.80, 0.10, 0.10, 0.5)
        g.set_erasure_opinion("A", op_a)
        g.set_erasure_opinion("B", op_b)

        adapter_result = contamination_risk("B", g)
        # B's A_j⁺ = {A, B}
        direct_result = residual_contamination(op_a, op_b)
        assert abs(adapter_result.lawfulness - direct_result.lawfulness) < TOL
        assert abs(adapter_result.violation - direct_result.violation) < TOL


# ═══════════════════════════════════════════════════════════════════
# review_due_assessment() — §8.2 composition function
# ═══════════════════════════════════════════════════════════════════


class TestReviewDueAssessment:
    """Bridge review schedule to review_due_trigger.

    Per Definition 13 (§8.2). A missed mandatory review accelerates
    decay toward vacuity (not violation). Missing evidence ≠ violation.
    """

    def test_no_review_scheduled_returns_unchanged(self):
        """No review scheduled → opinion unchanged.

        If there's no review obligation, no decay acceleration.
        """
        s = SimpleReviewSchedule()
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        result = review_due_assessment(op, "dpia-001", 500.0, s)
        assert abs(result.lawfulness - op.lawfulness) < TOL

    def test_pre_review_due_returns_unchanged(self):
        """Assessment before review-due date → unchanged.

        Review hasn't been missed yet, so no accelerated decay.
        """
        s = SimpleReviewSchedule()
        s.set_review_due("dpia-001", due_time=365.0)
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        result = review_due_assessment(op, "dpia-001", 300.0, s)
        assert abs(result.lawfulness - op.lawfulness) < TOL

    def test_post_review_due_increases_uncertainty(self):
        """Assessment after missed review → uncertainty increases.

        Definition 13: review-due moves toward vacuity (u increases),
        not violation. Missed review = missing evidence.
        """
        s = SimpleReviewSchedule(default_half_life=730.0)
        s.set_review_due("dpia-001", due_time=365.0)
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)

        pre = review_due_assessment(op, "dpia-001", 300.0, s)
        post = review_due_assessment(op, "dpia-001", 500.0, s)

        assert post.uncertainty > pre.uncertainty
        assert post.lawfulness < pre.lawfulness

    def test_longer_overdue_more_decay(self):
        """Longer overdue → more uncertainty growth."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        s.set_review_due("dpia-001", due_time=365.0)
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)

        early = review_due_assessment(op, "dpia-001", 400.0, s)
        late = review_due_assessment(op, "dpia-001", 700.0, s)

        assert late.uncertainty > early.uncertainty
        assert late.lawfulness < early.lawfulness

    def test_accelerated_half_life_is_used(self):
        """Post-trigger decay uses accelerated (shorter) half-life.

        The accelerated half-life produces faster decay than the
        normal half-life would.
        """
        s_fast = SimpleReviewSchedule(default_half_life=730.0)
        s_fast.set_review_due("dpia-001", due_time=365.0)
        s_fast.set_accelerated_half_life("dpia-001", 90.0)

        s_slow = SimpleReviewSchedule(default_half_life=730.0)
        s_slow.set_review_due("dpia-001", due_time=365.0)
        s_slow.set_accelerated_half_life("dpia-001", 365.0)

        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        fast = review_due_assessment(op, "dpia-001", 600.0, s_fast)
        slow = review_due_assessment(op, "dpia-001", 600.0, s_slow)

        # Faster decay → more uncertainty
        assert fast.uncertainty > slow.uncertainty

    def test_constraint_preservation(self):
        """l + v + u = 1 before and after review trigger."""
        s = SimpleReviewSchedule(default_half_life=365.0)
        s.set_review_due("dpia-001", due_time=100.0)
        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)

        for t in [50.0, 100.0, 200.0, 500.0]:
            result = review_due_assessment(op, "dpia-001", t, s)
            assert_valid_co(result, f"review at t={t}")

    def test_delegates_to_review_due_trigger(self):
        """Adapter must delegate to algebra's review_due_trigger()."""
        s = SimpleReviewSchedule(default_half_life=730.0)
        s.set_review_due("dpia-001", due_time=365.0)
        s.set_accelerated_half_life("dpia-001", 182.5)

        op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)

        adapter_result = review_due_assessment(op, "dpia-001", 500.0, s)
        direct_result = review_due_trigger(
            op,
            assessment_time=500.0,
            trigger_time=365.0,
            accelerated_half_life=182.5,
        )
        assert abs(adapter_result.lawfulness - direct_result.lawfulness) < TOL
        assert abs(adapter_result.violation - direct_result.violation) < TOL
        assert abs(adapter_result.uncertainty - direct_result.uncertainty) < TOL


# ═══════════════════════════════════════════════════════════════════
# END-TO-END SCENARIOS
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndScenarios:
    """Real-world compliance lifecycle scenarios using adapters."""

    def test_healthcare_erasure_request(self):
        """Patient requests erasure of clinical data.

        Scenario: Patient's data flows through 4 hospital systems:
        EHR → Research DB → Analytics → Archive.
        Archive is exempt under Art. 17(3)(d) (public health).
        """
        g = SimpleLineageGraph()
        g.add_edge("EHR", "ResearchDB")
        g.add_edge("ResearchDB", "Analytics")
        g.add_edge("ResearchDB", "Archive")
        g.add_exempt("Archive", reason="Art. 17(3)(d) public health")

        g.set_erasure_opinion(
            "EHR",
            ComplianceOpinion.create(0.95, 0.02, 0.03, 0.5),
        )
        g.set_erasure_opinion(
            "ResearchDB",
            ComplianceOpinion.create(0.85, 0.08, 0.07, 0.5),
        )
        g.set_erasure_opinion(
            "Analytics",
            ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5),
        )
        g.set_erasure_opinion(
            "Archive",
            ComplianceOpinion.create(0.10, 0.80, 0.10, 0.5),
        )

        # Scope = {EHR, ResearchDB, Analytics} (Archive exempt)
        result = erasure_scope_assessment("EHR", g)
        assert_valid_co(result, "healthcare erasure")
        # Archive excluded → composite should be reasonable
        assert result.lawfulness > 0.5

        # Contamination at Analytics
        risk = contamination_risk("Analytics", g)
        assert_valid_co(risk, "analytics contamination")

    def test_multinational_review_schedule(self):
        """DPIA reviews across multiple assessments.

        Scenario: An organization has three active DPIAs:
        - EU operations: reviewed, not yet due
        - US operations: overdue by 90 days
        - APAC operations: overdue by 365 days
        """
        s = SimpleReviewSchedule(default_half_life=365.0)
        s.set_review_due("dpia-eu", due_time=500.0)     # not yet due
        s.set_review_due("dpia-us", due_time=200.0)     # overdue
        s.set_review_due("dpia-apac", due_time=100.0)   # very overdue

        assessment_time = 465.0  # current time

        eu_op = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        us_op = ComplianceOpinion.create(0.80, 0.08, 0.12, 0.5)
        apac_op = ComplianceOpinion.create(0.75, 0.10, 0.15, 0.5)

        eu_result = review_due_assessment(eu_op, "dpia-eu", assessment_time, s)
        us_result = review_due_assessment(us_op, "dpia-us", assessment_time, s)
        apac_result = review_due_assessment(apac_op, "dpia-apac", assessment_time, s)

        # EU not yet due → unchanged
        assert abs(eu_result.lawfulness - eu_op.lawfulness) < TOL
        # US overdue → some decay
        assert us_result.uncertainty > us_op.uncertainty
        # APAC very overdue → more decay
        assert apac_result.uncertainty > us_result.uncertainty

        # Composite across all three jurisdictions
        composite = jurisdictional_meet(eu_result, us_result, apac_result)
        assert_valid_co(composite, "multinational composite")

    def test_consent_withdrawal_triggers_erasure(self):
        """§10: Consent Withdrawal → Erasure Request chain.

        Art. 17(1)(b): consent withdrawal triggers erasure when
        no alternative lawful basis exists.
        """
        # Step 1: Consent was valid
        consent_op = ComplianceOpinion.create(0.90, 0.03, 0.07, 0.5)

        # Step 2: Consent withdrawn → erasure obligation arises
        # Build lineage for erasure assessment
        g = SimpleLineageGraph()
        g.add_edge("Source", "Derived1")
        g.add_edge("Source", "Derived2")
        g.set_erasure_opinion(
            "Source",
            ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5),
        )
        g.set_erasure_opinion(
            "Derived1",
            ComplianceOpinion.create(0.70, 0.15, 0.15, 0.5),
        )
        g.set_erasure_opinion(
            "Derived2",
            ComplianceOpinion.create(0.80, 0.10, 0.10, 0.5),
        )

        erasure_result = erasure_scope_assessment("Source", g)
        assert_valid_co(erasure_result, "post-withdrawal erasure")

        # Step 3: Compliance update = meet(withdrawal, erasure)
        withdrawal_op = ComplianceOpinion.create(0.20, 0.60, 0.20, 0.5)
        final = jurisdictional_meet(withdrawal_op, erasure_result)
        assert_valid_co(final, "withdrawal+erasure composite")
        assert final.violation > final.lawfulness

    def test_full_lifecycle_with_all_adapters(self):
        """Complete lifecycle using both adapters + algebra composition.

        1. Consent assessment
        2. Multi-site meet
        3. Review-due check
        4. Erasure assessment (if triggered)
        5. Final compliance update
        """
        # 1. Two sites with consent
        site_a = ComplianceOpinion.create(0.85, 0.05, 0.10, 0.5)
        site_b = ComplianceOpinion.create(0.80, 0.08, 0.12, 0.5)

        # 2. Multi-site meet
        composite_consent = jurisdictional_meet(site_a, site_b)
        assert_valid_co(composite_consent, "step 2: meet")
        assert composite_consent.lawfulness < min(
            site_a.lawfulness, site_b.lawfulness
        )

        # 3. Review-due check on composite
        s = SimpleReviewSchedule(default_half_life=365.0)
        s.set_review_due("composite", due_time=100.0)
        reviewed = review_due_assessment(
            composite_consent, "composite", 200.0, s,
        )
        assert_valid_co(reviewed, "step 3: review")
        assert reviewed.uncertainty > composite_consent.uncertainty

        # 4. Erasure assessment
        g = SimpleLineageGraph()
        g.add_edge("raw", "processed")
        g.add_edge("processed", "model")
        for node in ["raw", "processed", "model"]:
            g.set_erasure_opinion(
                node,
                ComplianceOpinion.create(0.90, 0.05, 0.05, 0.5),
            )
        erased = erasure_scope_assessment("raw", g)
        assert_valid_co(erased, "step 4: erasure")

        # 5. Final update: meet(reviewed_consent, erasure)
        final = jurisdictional_meet(reviewed, erased)
        assert_valid_co(final, "step 5: final")
        assert final.lawfulness < reviewed.lawfulness
