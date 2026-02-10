"""Tests for conflict detection and robust fusion operators.

TDD: Write tests FIRST, then implement.

New operators:
  - pairwise_conflict(op_a, op_b)  — Jøsang's binary conflict measure
  - conflict_metric(opinion)       — internal balance metric for fused opinions
  - robust_fuse(opinions, ...)     — Byzantine-resistant iterative fusion
"""

import math
import pytest

from jsonld_ex.confidence_algebra import Opinion, cumulative_fuse, averaging_fuse


# We'll import these once implemented:
from jsonld_ex.confidence_algebra import (
    pairwise_conflict,
    conflict_metric,
    robust_fuse,
)


# ═══════════════════════════════════════════════════════════════════
# pairwise_conflict
# ═══════════════════════════════════════════════════════════════════


class TestPairwiseConflict:
    """Jøsang's binary conflict: con(A,B) = b_A·d_B + d_A·b_B.

    Properties:
      - Range: [0, 1]
      - Symmetry: con(A,B) = con(B,A)
      - Zero when opinions agree (both believe or both disbelieve)
      - Maximum when one fully believes, other fully disbelieves
      - Zero when either opinion is vacuous (u=1)

    Reference: Jøsang (2016), §12.3.4.
    """

    def test_agreeing_opinions_zero_conflict(self):
        """Two believers should have zero conflict."""
        a = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        b = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        result = pairwise_conflict(a, b)
        assert result < 0.1  # Near-zero conflict

    def test_fully_opposing_maximum_conflict(self):
        """Full believer vs full disbeliever = maximum conflict."""
        a = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        b = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        result = pairwise_conflict(a, b)
        assert result == 1.0

    def test_symmetry(self):
        """con(A,B) = con(B,A)."""
        a = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        b = Opinion(belief=0.3, disbelief=0.5, uncertainty=0.2)
        assert pairwise_conflict(a, b) == pytest.approx(pairwise_conflict(b, a))

    def test_vacuous_opinion_zero_conflict(self):
        """A vacuous opinion (u=1) cannot conflict with anything."""
        vacuous = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        strong = Opinion(belief=0.9, disbelief=0.1, uncertainty=0.0)
        assert pairwise_conflict(vacuous, strong) == pytest.approx(0.0)

    def test_range_zero_to_one(self):
        """Result must be in [0, 1] for any valid pair."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            raw = [rng.random() for _ in range(3)]
            total = sum(raw)
            a = Opinion(raw[0]/total, raw[1]/total, raw[2]/total)

            raw2 = [rng.random() for _ in range(3)]
            total2 = sum(raw2)
            b = Opinion(raw2[0]/total2, raw2[1]/total2, raw2[2]/total2)

            c = pairwise_conflict(a, b)
            assert 0.0 <= c <= 1.0, f"Out of range: {c}"

    def test_formula_matches_josang(self):
        """Verify con(A,B) = b_A·d_B + d_A·b_B exactly."""
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        b = Opinion(belief=0.2, disbelief=0.5, uncertainty=0.3)
        expected = a.belief * b.disbelief + a.disbelief * b.belief
        assert pairwise_conflict(a, b) == pytest.approx(expected)

    def test_both_disbelievers_low_conflict(self):
        """Two opinions that both disbelieve should have low conflict."""
        a = Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1)
        b = Opinion(belief=0.0, disbelief=0.7, uncertainty=0.3)
        result = pairwise_conflict(a, b)
        assert result < 0.15  # Both on same side

    def test_type_error_non_opinion(self):
        """Must raise TypeError for non-Opinion arguments."""
        a = Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)
        with pytest.raises(TypeError):
            pairwise_conflict(a, 0.5)
        with pytest.raises(TypeError):
            pairwise_conflict(0.5, a)

    def test_identical_opinions_zero_conflict(self):
        """An opinion paired with itself should have minimal conflict."""
        a = Opinion(belief=0.6, disbelief=0.3, uncertainty=0.1)
        result = pairwise_conflict(a, a)
        # con(A,A) = b*d + d*b = 2*b*d
        expected = 2 * a.belief * a.disbelief
        assert result == pytest.approx(expected)


# ═══════════════════════════════════════════════════════════════════
# conflict_metric
# ═══════════════════════════════════════════════════════════════════


class TestConflictMetric:
    """Internal conflict/balance metric for a (typically fused) opinion.

    Formula: conflict = 1 - |b - d| - u

    Intuition:
      - When b ≈ d and u is small: high conflict (evidence cancels out)
      - When b >> d or d >> b: low conflict (clear direction)
      - When u ≈ 1: low conflict (just ignorance, not disagreement)

    Range: [0, 1]

    This measures how much the evidence *within* an opinion is
    self-contradictory (high b AND high d simultaneously), which
    typically arises when fusing opinions from agents that disagree.
    """

    def test_strong_agreement_low_conflict(self):
        """Clear belief direction → low conflict."""
        o = Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)
        assert conflict_metric(o) < 0.3

    def test_balanced_evidence_high_conflict(self):
        """Equal belief and disbelief with low uncertainty → high conflict."""
        o = Opinion(belief=0.45, disbelief=0.45, uncertainty=0.1)
        result = conflict_metric(o)
        assert result > 0.8

    def test_vacuous_zero_conflict(self):
        """Total ignorance (u=1) is NOT conflict."""
        o = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
        assert conflict_metric(o) == pytest.approx(0.0)

    def test_full_belief_zero_conflict(self):
        """Dogmatic belief has zero conflict."""
        o = Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)
        assert conflict_metric(o) == pytest.approx(0.0)

    def test_full_disbelief_zero_conflict(self):
        """Dogmatic disbelief has zero conflict."""
        o = Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)
        assert conflict_metric(o) == pytest.approx(0.0)

    def test_maximum_conflict(self):
        """b = d = 0.5, u = 0 → maximum conflict = 1.0."""
        o = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        assert conflict_metric(o) == pytest.approx(1.0)

    def test_formula_exact(self):
        """Verify formula: 1 - |b - d| - u."""
        o = Opinion(belief=0.4, disbelief=0.3, uncertainty=0.3)
        expected = 1.0 - abs(0.4 - 0.3) - 0.3
        assert conflict_metric(o) == pytest.approx(expected)

    def test_range_zero_to_one(self):
        """Result must be in [0, 1] for any valid opinion."""
        import random
        rng = random.Random(42)
        for _ in range(100):
            raw = [rng.random() for _ in range(3)]
            total = sum(raw)
            o = Opinion(raw[0]/total, raw[1]/total, raw[2]/total)
            c = conflict_metric(o)
            assert -1e-9 <= c <= 1.0 + 1e-9, f"Out of range: {c}"

    def test_type_error_non_opinion(self):
        """Must raise TypeError for non-Opinion argument."""
        with pytest.raises(TypeError):
            conflict_metric(0.5)
        with pytest.raises(TypeError):
            conflict_metric({"belief": 0.5})

    def test_distinguishes_conflict_from_ignorance(self):
        """The KEY property: conflict and ignorance produce different scores
        even when projected probability is the same (0.5)."""
        # Same P = 0.5 but radically different epistemic states
        conflict = Opinion(belief=0.5, disbelief=0.5, uncertainty=0.0)
        ignorance = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)

        assert conflict.projected_probability() == pytest.approx(
            ignorance.projected_probability(), abs=0.01
        )  # Both ≈ 0.5

        # But conflict metric separates them
        assert conflict_metric(conflict) > 0.9   # High conflict
        assert conflict_metric(ignorance) < 0.1  # No conflict


# ═══════════════════════════════════════════════════════════════════
# robust_fuse
# ═══════════════════════════════════════════════════════════════════


class TestRobustFuse:
    """Byzantine-resistant fusion via iterative conflict filtering.

    Algorithm:
      1. Compute pairwise conflict matrix
      2. Find agent with highest mean conflict (discord score)
      3. If discord exceeds threshold, remove that agent
      4. Repeat until all remaining agents are cohesive or
         max_removals reached
      5. Fuse remaining opinions via cumulative_fuse

    Returns: (fused_opinion, list_of_removed_indices)
    """

    def test_all_agreeing_no_removals(self):
        """Cohesive group → no agents removed."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        fused, removed = robust_fuse(opinions)
        assert len(removed) == 0
        assert fused.belief > 0.5

    def test_one_rogue_removed(self):
        """One dissenter among agreers → dissenter removed."""
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogue = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        all_opinions = honest + [rogue]

        fused, removed = robust_fuse(all_opinions)
        assert 3 in removed  # Rogue was at index 3
        assert fused.belief > 0.5  # Result reflects honest agents

    def test_two_rogues_among_five(self):
        """Classic 2/5 Byzantine scenario — rogues removed, honest majority wins."""
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogues = [
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1),
        ]
        all_opinions = honest + rogues

        fused, removed = robust_fuse(all_opinions)
        assert len(removed) >= 1  # At least one rogue removed
        assert fused.belief > 0.5  # Honest consensus preserved

    def test_max_removals_respected(self):
        """Should not remove more than max_removals agents."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),
            Opinion(belief=0.0, disbelief=0.8, uncertainty=0.2),
            Opinion(belief=0.0, disbelief=0.7, uncertainty=0.3),
        ]
        fused, removed = robust_fuse(opinions, max_removals=1)
        assert len(removed) <= 1

    def test_custom_threshold(self):
        """Higher threshold = less aggressive filtering."""
        mild_disagreement = [
            Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
            Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2),
            Opinion(belief=0.4, disbelief=0.4, uncertainty=0.2),
        ]
        # Tight threshold: might remove the balanced one
        _, removed_tight = robust_fuse(mild_disagreement, threshold=0.1)
        # Loose threshold: keep everyone
        _, removed_loose = robust_fuse(mild_disagreement, threshold=0.5)
        assert len(removed_loose) <= len(removed_tight)

    def test_returns_tuple(self):
        """Return type must be (Opinion, list[int])."""
        opinions = [
            Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1),
            Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2),
        ]
        result = robust_fuse(opinions)
        assert isinstance(result, tuple)
        assert len(result) == 2
        fused, removed = result
        assert isinstance(fused, Opinion)
        assert isinstance(removed, list)

    def test_single_opinion(self):
        """Single opinion → returned as-is, no removals."""
        o = Opinion(belief=0.7, disbelief=0.2, uncertainty=0.1)
        fused, removed = robust_fuse([o])
        assert fused == o
        assert removed == []

    def test_two_opinions_no_removal(self):
        """Two opinions → minimum 2 required, can't remove any."""
        a = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        b = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        fused, removed = robust_fuse([a, b])
        assert len(removed) == 0  # Can't go below 2

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            robust_fuse([])

    def test_removed_indices_are_original(self):
        """Removed indices should refer to positions in the ORIGINAL list."""
        opinions = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),  # 0: honest
            Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1),  # 1: rogue
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),  # 2: honest
            Opinion(belief=0.1, disbelief=0.8, uncertainty=0.1),  # 3: rogue
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),  # 4: honest
        ]
        fused, removed = robust_fuse(opinions)
        # All removed indices should be valid original positions
        for idx in removed:
            assert 0 <= idx < len(opinions)
        # Removed agents should be the rogues (indices 1 and/or 3)
        for idx in removed:
            assert opinions[idx].disbelief > 0.5, (
                f"Removed agent at index {idx} was honest (d={opinions[idx].disbelief})"
            )

    def test_preserves_honest_consensus(self):
        """After removing rogues, fused result should match
        fusing only the honest agents."""
        honest = [
            Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1),
            Opinion(belief=0.7, disbelief=0.1, uncertainty=0.2),
            Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1),
        ]
        rogue = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)

        # Fuse honest only (ground truth)
        honest_fused = cumulative_fuse(*honest)

        # Robust fuse with rogue included
        robust_result, removed = robust_fuse(honest + [rogue])

        # Should be close to honest fusion (exact if rogue fully removed)
        if 3 in removed:
            assert robust_result.belief == pytest.approx(honest_fused.belief, abs=0.01)

    def test_default_max_removals_is_minority(self):
        """Default max_removals = floor(n/2) — never remove majority."""
        opinions = [Opinion(belief=0.5, disbelief=0.3, uncertainty=0.2)] * 5
        # With 5 opinions, default max_removals should be 2
        fused, removed = robust_fuse(opinions)
        assert len(removed) <= 2
