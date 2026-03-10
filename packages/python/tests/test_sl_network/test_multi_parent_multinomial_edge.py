"""Tests for MultiParentMultinomialEdge dataclass.

TDD RED phase for Phase C, Step C.0.

MultiParentMultinomialEdge represents a directed conditional edge from
multiple k-ary parent nodes to a single k-ary child node.  It maps
each combination of parent states (as a tuple of state names) to a
conditional MultinomialOpinion about the child variable.

This is the multinomial generalization of MultiParentEdge, which uses
bool tuples and binomial Opinion.

References:
    Jøsang, A. (2016). Subjective Logic, Ch. 9.
    SLNetworks_plan.md §1.6.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.types import MultiParentMultinomialEdge


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _make_child_opinion(b_h: float, b_l: float) -> MultinomialOpinion:
    """Create a binary-domain MultinomialOpinion with states {H, L}."""
    u = 1.0 - b_h - b_l
    return MultinomialOpinion(
        beliefs={"H": b_h, "L": b_l},
        uncertainty=u,
        base_rates={"H": 0.5, "L": 0.5},
    )


def _make_ternary_child_opinion(
    b_a: float, b_b: float, b_c: float
) -> MultinomialOpinion:
    """Create a ternary-domain MultinomialOpinion with states {A, B, C}."""
    u = 1.0 - b_a - b_b - b_c
    return MultinomialOpinion(
        beliefs={"A": b_a, "B": b_b, "C": b_c},
        uncertainty=u,
        base_rates={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
    )


def _make_two_parent_edge() -> MultiParentMultinomialEdge:
    """Two ternary parents → binary child.

    Parent domains: X = {x1, x2, x3}, Y = {y1, y2, y3}
    Child domain: {H, L}
    Conditionals: 3×3 = 9 entries.
    """
    parent_states_x = ["x1", "x2", "x3"]
    parent_states_y = ["y1", "y2", "y3"]
    conditionals: dict[tuple[str, ...], MultinomialOpinion] = {}
    idx = 0
    for sx in parent_states_x:
        for sy in parent_states_y:
            # Vary beliefs systematically so each entry is distinct
            b_h = 0.05 * (idx + 1)
            b_l = 0.05 * (9 - idx)
            u = 1.0 - b_h - b_l
            conditionals[(sx, sy)] = _make_child_opinion(b_h, b_l)
            idx += 1

    return MultiParentMultinomialEdge(
        parent_ids=("X", "Y"),
        target_id="Z",
        conditionals=conditionals,
    )


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentMultinomialEdgeConstruction:
    """Test basic construction and field access."""

    def test_basic_construction(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.parent_ids == ("X", "Y")
        assert edge.target_id == "Z"
        assert edge.edge_type == "deduction"
        assert len(edge.conditionals) == 9

    def test_default_edge_type(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.edge_type == "deduction"

    def test_custom_edge_type(self) -> None:
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        edge = MultiParentMultinomialEdge(
            parent_ids=("A",),
            target_id="B",
            conditionals=cond,
            edge_type="deduction",
        )
        assert edge.edge_type == "deduction"

    def test_metadata_default_empty(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.metadata == {}

    def test_metadata_custom(self) -> None:
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        edge = MultiParentMultinomialEdge(
            parent_ids=("A",),
            target_id="B",
            conditionals=cond,
            metadata={"source": "test"},
        )
        assert edge.metadata["source"] == "test"

    def test_temporal_fields_default_none(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.timestamp is None
        assert edge.half_life is None
        assert edge.valid_from is None
        assert edge.valid_until is None

    def test_temporal_fields_custom(self) -> None:
        now = datetime.now(tz=timezone.utc)
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        edge = MultiParentMultinomialEdge(
            parent_ids=("A",),
            target_id="B",
            conditionals=cond,
            timestamp=now,
            half_life=3600.0,
            valid_from=now,
            valid_until=now,
        )
        assert edge.timestamp == now
        assert edge.half_life == 3600.0

    def test_frozen_immutable(self) -> None:
        edge = _make_two_parent_edge()
        with pytest.raises(AttributeError):
            edge.target_id = "other"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentMultinomialEdgeValidation:
    """Test __post_init__ validation."""

    def test_empty_parent_ids_raises(self) -> None:
        cond = {(): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="parent_ids"):
            MultiParentMultinomialEdge(
                parent_ids=(),
                target_id="Z",
                conditionals=cond,
            )

    def test_empty_target_id_raises(self) -> None:
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="target_id"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="",
                conditionals=cond,
            )

    def test_target_in_parent_ids_raises(self) -> None:
        """Target cannot also be a parent (self-loop variant)."""
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="target_id.*parent_ids"):
            MultiParentMultinomialEdge(
                parent_ids=("A", "B"),
                target_id="A",
                conditionals=cond,
            )

    def test_duplicate_parent_ids_raises(self) -> None:
        cond = {("x1", "x1"): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            MultiParentMultinomialEdge(
                parent_ids=("A", "A"),
                target_id="Z",
                conditionals=cond,
            )

    def test_empty_conditionals_raises(self) -> None:
        with pytest.raises(ValueError, match="conditionals"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals={},
            )

    def test_conditional_wrong_type_raises(self) -> None:
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals={("x1",): "not an opinion"},  # type: ignore[dict-item]
            )

    def test_inconsistent_child_domain_raises(self) -> None:
        """All conditionals must share the same child domain."""
        op_hl = _make_child_opinion(0.5, 0.3)
        op_abc = _make_ternary_child_opinion(0.3, 0.3, 0.2)
        with pytest.raises(ValueError, match="child domain"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals={("x1",): op_hl, ("x2",): op_abc},
            )

    def test_conditional_key_wrong_length_raises(self) -> None:
        """Conditional key tuples must match len(parent_ids)."""
        op = _make_child_opinion(0.5, 0.3)
        with pytest.raises(ValueError, match="[Ll]ength|parent_ids"):
            MultiParentMultinomialEdge(
                parent_ids=("A", "B"),
                target_id="Z",
                conditionals={
                    ("x1",): op,  # length 1, should be 2
                },
            )

    def test_invalid_edge_type_raises(self) -> None:
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="edge_type"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals=cond,
                edge_type="invalid",  # type: ignore[arg-type]
            )

    def test_negative_half_life_raises(self) -> None:
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="half_life"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals=cond,
                half_life=-1.0,
            )

    def test_valid_until_before_valid_from_raises(self) -> None:
        now = datetime.now(tz=timezone.utc)
        earlier = datetime(2020, 1, 1, tzinfo=timezone.utc)
        cond = {("x1",): _make_child_opinion(0.5, 0.3)}
        with pytest.raises(ValueError, match="valid_until"):
            MultiParentMultinomialEdge(
                parent_ids=("A",),
                target_id="Z",
                conditionals=cond,
                valid_from=now,
                valid_until=earlier,
            )


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentMultinomialEdgeProperties:
    """Test convenience properties."""

    def test_child_domain(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.child_domain == ("H", "L")

    def test_child_cardinality(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.child_cardinality == 2

    def test_num_parent_configs(self) -> None:
        """Two ternary parents → 9 entries."""
        edge = _make_two_parent_edge()
        assert edge.num_parent_configs == 9

    def test_parent_count(self) -> None:
        edge = _make_two_parent_edge()
        assert edge.parent_count == 2


# ═══════════════════════════════════════════════════════════════════
# HASH AND EQUALITY
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentMultinomialEdgeHash:
    """Test __hash__ based on (parent_ids, target_id)."""

    def test_hash_by_identity(self) -> None:
        edge = _make_two_parent_edge()
        assert hash(edge) == hash((("X", "Y"), "Z"))

    def test_equal_edges_same_hash(self) -> None:
        e1 = _make_two_parent_edge()
        e2 = _make_two_parent_edge()
        assert hash(e1) == hash(e2)

    def test_different_target_different_hash(self) -> None:
        e1 = _make_two_parent_edge()
        cond = {("x1", "y1"): _make_child_opinion(0.5, 0.3)}
        e2 = MultiParentMultinomialEdge(
            parent_ids=("X", "Y"),
            target_id="W",
            conditionals=cond,
        )
        assert hash(e1) != hash(e2)


# ═══════════════════════════════════════════════════════════════════
# REPR
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentMultinomialEdgeRepr:
    """Test __repr__ output."""

    def test_repr_contains_key_info(self) -> None:
        edge = _make_two_parent_edge()
        r = repr(edge)
        assert "MultiParentMultinomialEdge" in r
        assert "X" in r
        assert "Y" in r
        assert "Z" in r
        assert "9" in r  # number of conditional entries
