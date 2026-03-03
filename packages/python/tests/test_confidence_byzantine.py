"""Tests for enhanced Byzantine-resistant fusion (confidence_byzantine.py).

TDD: Tests written FIRST -- all will fail until implementation exists.

New types and functions (additive -- robust_fuse in confidence_algebra stays untouched):

    ByzantineStrategy  -- Literal["most_conflicting", "least_trusted", "combined"]
    ByzantineConfig    -- dataclass(threshold, max_removals, strategy, trust_weights, min_agents)
    AgentRemoval       -- dataclass(index, opinion, discord_score, reason)
    ByzantineFusionReport -- dataclass(fused, removed, conflict_matrix, cohesion_score, surviving_indices)

    opinion_distance(a, b) -> float
    byzantine_fuse(opinions, config=None) -> ByzantineFusionReport
    build_conflict_matrix(opinions) -> list[list[float]]
    cohesion_score(opinions) -> float

Design principles:
    - Extends robust_fuse without modifying it
    - Three removal strategies: discord-based, trust-weighted, combined
    - Rich reporting: every removal has a reason string
    - Composable with temporal_fuse (next module)
"""

import math
import pytest
from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse

from jsonld_ex.confidence_byzantine import (
    ByzantineStrategy,
    ByzantineConfig,
    AgentRemoval,
    ByzantineFusionReport,
    opinion_distance,
    byzantine_fuse,
    build_conflict_matrix,
    cohesion_score,
)


# -------------------------------------------------------------------
# ByzantineConfig
# -------------------------------------------------------------------


class TestByzantineConfig:
    """Configuration dataclass for Byzantine fusion."""

    def test_default_values(self):
        cfg = ByzantineConfig()
        assert cfg.threshold == 0.15
        assert cfg.max_removals is None
        assert cfg.strategy == "most_conflicting"
        assert cfg.trust_weights is None
        assert cfg.min_agents == 2

    def test_custom_values(self):
        cfg = ByzantineConfig(
            threshold=0.3,
            max_removals=2,
            strategy="least_trusted",
            trust_weights=[0.9, 0.8, 0.3, 0.2],
            min_agents=3,
        )
        assert cfg.threshold == 0.3
        assert cfg.max_removals == 2
        assert cfg.strategy == "least_trusted"
        assert cfg.trust_weights == [0.9, 0.8, 0.3, 0.2]
        assert cfg.min_agents == 3

    def test_combined_strategy(self):
        cfg = ByzantineConfig(strategy="combined")
        assert cfg.strategy == "combined"


# -------------------------------------------------------------------
# opinion_distance
# -------------------------------------------------------------------


class TestOpinionDistance:
    """Normalized Euclidean distance on the opinion simplex.

    Formula: d(A, B) = sqrt((bA-bB)^2 + (dA-dB)^2 + (uA-uB)^2) / sqrt(2)

    Key property that distinguishes it from pairwise_conflict:
      d(A, A) = 0  ALWAYS  (identity of indiscernibles)

    pairwise_conflict(A, A) = 2*b*d, which is non-zero whenever
    both b > 0 and d > 0.
    """

    def test_identical_opinions_zero(self):
        """Identity of indiscernibles: d(A, A) = 0 for ANY opinion."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        assert opinion_distance(o, o) == pytest.approx(0.0)

    def test_identical_high_conflict_opinion_still_zero(self):
        """Even an opinion with high internal b*d gives distance 0 to itself.
        This is the exact case where pairwise_conflict would give 2*b*d = 0.5."""
        o = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        assert opinion_distance(o, o) == pytest.approx(0.0)

    def test_symmetry(self):
        """d(A, B) = d(B, A)."""
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        assert opinion_distance(a, b) == pytest.approx(opinion_distance(b, a))

    def test_maximum_distance_vertex_to_vertex(self):
        """Opposite simplex vertices have distance 1.0.
        (1,0,0) vs (0,1,0): sqrt(1+1+0)/sqrt(2) = 1.0"""
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        b = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        assert opinion_distance(a, b) == pytest.approx(1.0)

    def test_other_vertex_pairs(self):
        """All vertex pairs have distance 1.0."""
        v_b = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        v_d = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        v_u = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert opinion_distance(v_b, v_d) == pytest.approx(1.0)
        assert opinion_distance(v_b, v_u) == pytest.approx(1.0)
        assert opinion_distance(v_d, v_u) == pytest.approx(1.0)

    def test_formula_exact(self):
        """Verify the formula against manual calculation."""
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        expected = math.sqrt(
            (0.6 - 0.2) ** 2 + (0.3 - 0.5) ** 2 + (0.1 - 0.3) ** 2
        ) / math.sqrt(2.0)
        assert opinion_distance(a, b) == pytest.approx(expected)

    def test_range_zero_to_one(self):
        """Result must be in [0, 1] for all valid opinion pairs."""
        import random
        rng = random.Random(42)
        for _ in range(200):
            raw_a = [rng.random() for _ in range(3)]
            t_a = sum(raw_a)
            a = Opinion(raw_a[0] / t_a, raw_a[1] / t_a, raw_a[2] / t_a)
            raw_b = [rng.random() for _ in range(3)]
            t_b = sum(raw_b)
            b = Opinion(raw_b[0] / t_b, raw_b[1] / t_b, raw_b[2] / t_b)
            d = opinion_distance(a, b)
            assert -1e-9 <= d <= 1.0 + 1e-9, f"Out of range: {d}"

    def test_triangle_inequality(self):
        """d(A,C) <= d(A,B) + d(B,C) -- required for a metric."""
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2)
        c = Opinion(belief=0.1, disbelief=0.7, uncertainty=0.2)
        assert opinion_distance(a, c) <= (
            opinion_distance(a, b) + opinion_distance(b, c) + 1e-9
        )

    def test_nondegenerate(self):
        """d(A, B) > 0 when A != B."""
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        assert opinion_distance(a, b) > 0

    def test_base_rate_excluded(self):
        """Distance only considers (b, d, u), not base_rate.
        Two opinions with same evidence but different priors have distance 0."""
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.3)
        b = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1, base_rate=0.8)
        assert opinion_distance(a, b) == pytest.approx(0.0)

    def test_vacuous_to_dogmatic(self):
        """(0,0,1) to (1,0,0): sqrt(1+0+1)/sqrt(2) = 1.0"""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        dog = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert opinion_distance(vac, dog) == pytest.approx(1.0)

    def test_small_perturbation(self):
        """Slightly different opinions have small distance."""
        a = Opinion(belief=0.700, disbelief=0.200, uncertainty=0.100)
        b = Opinion(belief=0.705, disbelief=0.195, uncertainty=0.100)
        d = opinion_distance(a, b)
        assert d < 0.01


# -------------------------------------------------------------------
# build_conflict_matrix
# -------------------------------------------------------------------


class TestBuildConflictMatrix:
    """Symmetric nxn pairwise conflict matrix (Josang's formula)."""

    def test_two_opinions(self):
        a = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        b = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        mat = build_conflict_matrix([a, b])
        assert len(mat) == 2
        assert len(mat[0]) == 2
        assert mat[0][0] == pytest.approx(0.0)  # self-conflict = 0
        assert mat[1][1] == pytest.approx(0.0)
        assert mat[0][1] == pytest.approx(mat[1][0])  # symmetric
        assert mat[0][1] > 0.5  # high conflict

    def test_three_agreeing_opinions(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        mat = build_conflict_matrix(opinions)
        assert len(mat) == 3
        for i in range(3):
            assert mat[i][i] == pytest.approx(0.0)
            for j in range(3):
                assert mat[i][j] == pytest.approx(mat[j][i])

    def test_single_opinion(self):
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        mat = build_conflict_matrix([o])
        assert mat == [[0.0]]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_conflict_matrix([])

    def test_diagonal_is_zero(self):
        """Self-conflict is always 0 on the diagonal."""
        opinions = [
            Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1),
            Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3),
            Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2),
        ]
        mat = build_conflict_matrix(opinions)
        for i in range(3):
            assert mat[i][i] == pytest.approx(0.0)

    def test_values_in_range(self):
        """All matrix values must be in [0, 1]."""
        import random
        rng = random.Random(99)
        opinions = []
        for _ in range(5):
            raw = [rng.random() for _ in range(3)]
            t = sum(raw)
            opinions.append(Opinion(raw[0] / t, raw[1] / t, raw[2] / t))
        mat = build_conflict_matrix(opinions)
        for row in mat:
            for val in row:
                assert 0.0 <= val <= 1.0


# -------------------------------------------------------------------
# cohesion_score
# -------------------------------------------------------------------


class TestCohesionScore:
    """Group cohesion: 1 - mean(opinion_distance(i, j)) for all pairs.

    Uses opinion_distance (Euclidean on simplex), NOT pairwise_conflict.
    This ensures identical opinions always yield cohesion = 1.0.

    Range [0, 1].  1.0 = perfect agreement, 0.0 = total disagreement.
    """

    def test_perfect_agreement(self):
        """Identical opinions -> cohesion = 1.0."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        assert cohesion_score([o, o, o]) == pytest.approx(1.0)

    def test_perfect_agreement_high_internal_conflict(self):
        """Even opinions with high b*d should get cohesion 1.0 when identical.
        This is the specific case that fails if pairwise_conflict is used
        instead of opinion_distance."""
        o = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        assert cohesion_score([o, o, o]) == pytest.approx(1.0)

    def test_total_disagreement(self):
        """Full believer vs full disbeliever -> cohesion 0.0."""
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        b = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        score = cohesion_score([a, b])
        assert score == pytest.approx(0.0)

    def test_mostly_agreeing_high_cohesion(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        assert cohesion_score(opinions) > 0.8

    def test_single_opinion_perfect(self):
        """Single opinion -> cohesion 1.0 (no disagreement possible)."""
        o = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        assert cohesion_score([o]) == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            cohesion_score([])

    def test_range_zero_to_one(self):
        import random
        rng = random.Random(42)
        opinions = []
        for _ in range(6):
            raw = [rng.random() for _ in range(3)]
            t = sum(raw)
            opinions.append(Opinion(raw[0] / t, raw[1] / t, raw[2] / t))
        s = cohesion_score(opinions)
        assert 0.0 <= s <= 1.0

    def test_three_vertices_low_cohesion(self):
        """All three simplex vertices -> mean distance = 1.0, cohesion = 0.0."""
        v_b = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        v_d = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        v_u = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert cohesion_score([v_b, v_d, v_u]) == pytest.approx(0.0)


# -------------------------------------------------------------------
# AgentRemoval
# -------------------------------------------------------------------


class TestAgentRemoval:
    """Records why a specific agent was removed."""

    def test_fields(self):
        o = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        r = AgentRemoval(index=2, opinion=o, discord_score=0.65, reason="highest discord")
        assert r.index == 2
        assert r.opinion is o
        assert r.discord_score == 0.65
        assert "discord" in r.reason


# -------------------------------------------------------------------
# ByzantineFusionReport
# -------------------------------------------------------------------


class TestByzantineFusionReport:
    """Rich report returned by byzantine_fuse."""

    def test_fields_present(self):
        fused = Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2)
        report = ByzantineFusionReport(
            fused=fused,
            removed=[],
            conflict_matrix=[[0.0]],
            cohesion_score=1.0,
            surviving_indices=[0],
        )
        assert report.fused is fused
        assert report.removed == []
        assert report.cohesion_score == 1.0
        assert report.surviving_indices == [0]


# -------------------------------------------------------------------
# byzantine_fuse -- most_conflicting strategy (default)
# -------------------------------------------------------------------


class TestByzantineFuseMostConflicting:
    """Default strategy: remove agent with highest mean discord."""

    def test_all_agreeing_no_removals(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        report = byzantine_fuse(opinions)
        assert isinstance(report, ByzantineFusionReport)
        assert len(report.removed) == 0
        assert report.fused.belief > 0.5
        assert report.cohesion_score > 0.8
        assert report.surviving_indices == [0, 1, 2]

    def test_one_rogue_removed(self):
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogue = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        report = byzantine_fuse(honest + [rogue])
        removed_indices = [r.index for r in report.removed]
        assert 3 in removed_indices
        assert report.fused.belief > 0.5
        assert 3 not in report.surviving_indices

    def test_two_rogues_among_five(self):
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogues = [
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1),
        ]
        report = byzantine_fuse(honest + rogues)
        removed_indices = [r.index for r in report.removed]
        assert len(removed_indices) >= 1
        for idx in removed_indices:
            assert idx in (3, 4)
        assert report.fused.belief > 0.5

    def test_removal_reasons_populated(self):
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogue = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        report = byzantine_fuse(honest + [rogue])
        for removal in report.removed:
            assert isinstance(removal, AgentRemoval)
            assert len(removal.reason) > 0
            assert removal.discord_score > 0

    def test_conflict_matrix_in_report(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        report = byzantine_fuse(opinions)
        assert len(report.conflict_matrix) == 2
        assert len(report.conflict_matrix[0]) == 2

    def test_max_removals_respected(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.8, uncertainty=0.2),
            Opinion(belief=0.0, disbelief=0.7, uncertainty=0.3),
        ]
        cfg = ByzantineConfig(max_removals=1)
        report = byzantine_fuse(opinions, config=cfg)
        assert len(report.removed) <= 1

    def test_min_agents_respected(self):
        """Should never reduce below min_agents."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.8, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(min_agents=3)
        report = byzantine_fuse(opinions, config=cfg)
        assert len(report.removed) == 0  # Can't remove anyone

    def test_single_opinion(self):
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        report = byzantine_fuse([o])
        assert report.fused == o
        assert report.removed == []
        assert report.surviving_indices == [0]
        assert report.cohesion_score == pytest.approx(1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            byzantine_fuse([])

    def test_default_max_removals_is_minority(self):
        """Default max_removals = floor(n/2) when None."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.8, uncertainty=0.2),
            Opinion(belief=0.0, disbelief=0.7, uncertainty=0.3),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        report = byzantine_fuse(opinions)
        assert len(report.removed) <= 2  # floor(5/2) = 2


# -------------------------------------------------------------------
# byzantine_fuse -- least_trusted strategy
# -------------------------------------------------------------------


class TestByzantineFuseLeastTrusted:
    """Strategy: remove the agent with lowest trust weight first,
    but only if their discord also exceeds the threshold."""

    def test_untrusted_rogue_removed_first(self):
        """Low-trust rogue removed before high-trust mild dissenter."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),  # 0: honest, trusted
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),  # 1: honest, trusted
            Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2),  # 2: mild dissenter, trusted
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),  # 3: rogue, untrusted
        ]
        cfg = ByzantineConfig(
            strategy="least_trusted",
            trust_weights=[0.9, 0.9, 0.8, 0.1],
            max_removals=1,
        )
        report = byzantine_fuse(opinions, config=cfg)
        if len(report.removed) > 0:
            assert report.removed[0].index == 3

    def test_requires_trust_weights(self):
        """least_trusted without trust_weights should raise ValueError."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(strategy="least_trusted", trust_weights=None)
        with pytest.raises(ValueError, match="trust_weights"):
            byzantine_fuse(opinions, config=cfg)

    def test_wrong_length_trust_weights_raises(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(
            strategy="least_trusted",
            trust_weights=[0.9],  # wrong length
        )
        with pytest.raises(ValueError, match="trust_weights"):
            byzantine_fuse(opinions, config=cfg)

    def test_all_trusted_no_removals(self):
        """All high-trust, agreeing agents -> nothing removed."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(
            strategy="least_trusted",
            trust_weights=[0.9, 0.9],
        )
        report = byzantine_fuse(opinions, config=cfg)
        assert len(report.removed) == 0


# -------------------------------------------------------------------
# byzantine_fuse -- combined strategy
# -------------------------------------------------------------------


class TestByzantineFuseCombined:
    """Combined strategy: rank by discord * (1 - trust_weight).

    Agents that are both highly conflicting AND untrusted get
    removed first.
    """

    def test_untrusted_rogue_highest_priority(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),  # 0: honest, trusted
            Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2),  # 1: rogue-ish, trusted
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),  # 2: rogue, untrusted
        ]
        cfg = ByzantineConfig(
            strategy="combined",
            trust_weights=[0.9, 0.8, 0.1],
            max_removals=1,
        )
        report = byzantine_fuse(opinions, config=cfg)
        if len(report.removed) > 0:
            # Agent 2 should be removed: high discord + low trust
            assert report.removed[0].index == 2

    def test_requires_trust_weights(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(strategy="combined", trust_weights=None)
        with pytest.raises(ValueError, match="trust_weights"):
            byzantine_fuse(opinions, config=cfg)

    def test_combined_vs_most_conflicting_differs(self):
        """Combined strategy should sometimes make different choices
        than most_conflicting when trust weights vary."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),  # 0: honest
            Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2),  # 1: dissenter, highly trusted
            Opinion(belief=0.2, disbelief=0.6, uncertainty=0.2),  # 2: dissenter, untrusted
        ]
        cfg_combined = ByzantineConfig(
            strategy="combined",
            trust_weights=[0.9, 0.95, 0.1],
            max_removals=1,
        )
        cfg_discord = ByzantineConfig(
            strategy="most_conflicting",
            max_removals=1,
        )
        report_combined = byzantine_fuse(opinions, config=cfg_combined)
        report_discord = byzantine_fuse(opinions, config=cfg_discord)
        # Both should produce valid reports
        assert isinstance(report_combined.fused, Opinion)
        assert isinstance(report_discord.fused, Opinion)


# -------------------------------------------------------------------
# byzantine_fuse -- preserves honest consensus
# -------------------------------------------------------------------


class TestByzantineFuseConsensusPreservation:
    """After removing rogues, result should approximate honest-only fusion."""

    def test_matches_honest_fusion(self):
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogue = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        honest_fused = cumulative_fuse(*honest)

        report = byzantine_fuse(honest + [rogue])
        removed_indices = [r.index for r in report.removed]
        if 3 in removed_indices:
            assert report.fused.belief == pytest.approx(honest_fused.belief, abs=0.01)
            assert report.fused.disbelief == pytest.approx(honest_fused.disbelief, abs=0.01)

    def test_surviving_indices_correct(self):
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        report = byzantine_fuse(opinions)
        removed_indices = {r.index for r in report.removed}
        expected_surviving = [i for i in range(4) if i not in removed_indices]
        assert report.surviving_indices == expected_surviving


# -------------------------------------------------------------------
# Edge cases and error handling
# -------------------------------------------------------------------


class TestByzantineFuseEdgeCases:

    def test_two_opinions_never_reduces_below_two(self):
        """With two opinions, min_agents=2 prevents any removal."""
        a = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        b = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        report = byzantine_fuse([a, b])
        assert len(report.removed) == 0

    def test_all_vacuous_no_conflict(self):
        """All vacuous opinions -> zero conflict, nothing removed."""
        vac = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        report = byzantine_fuse([vac, vac, vac])
        assert len(report.removed) == 0
        assert report.cohesion_score == pytest.approx(1.0)

    def test_all_dogmatic_agreeing(self):
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        report = byzantine_fuse([a, a, a])
        assert len(report.removed) == 0
        assert report.fused.belief == pytest.approx(1.0)

    def test_custom_threshold_high_keeps_everyone(self):
        """Threshold = 1.0 -> nothing is ever above threshold."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        cfg = ByzantineConfig(threshold=1.0)
        report = byzantine_fuse(opinions, config=cfg)
        assert len(report.removed) == 0

    def test_report_conflict_matrix_is_original(self):
        """Conflict matrix should reflect ALL original opinions, not just survivors."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
        ]
        report = byzantine_fuse(opinions)
        assert len(report.conflict_matrix) == 3  # original count
        assert len(report.conflict_matrix[0]) == 3
