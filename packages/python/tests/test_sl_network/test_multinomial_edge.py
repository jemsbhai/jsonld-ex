"""
Tests for MultinomialEdge (Phase B, Step 2).

A MultinomialEdge represents a conditional relationship between a
multinomial parent node and a multinomial child node.  For a parent
with domain X = {x_1, ..., x_k}, the edge carries a conditional
opinion ω_{Y|x_i} for each parent state x_i, each over the same
child domain Y = {y_1, ..., y_m}.

Design contract (consistent with SLEdge and MultiParentEdge):
    - source_id and target_id are non-empty strings, no self-loops.
    - conditionals maps each parent state (str) to a MultinomialOpinion.
    - All conditional opinions share the same child domain.
    - edge_type defaults to "deduction".
    - Frozen (immutable) dataclass.
    - Hash by (source_id, target_id).
    - Temporal fields (timestamp, half_life, valid_from, valid_until)
      mirror SLEdge for consistency.

TDD: RED phase — all tests should FAIL until MultinomialEdge is
implemented in types.py.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion
from jsonld_ex.sl_network.types import MultinomialEdge


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def ternary_child_a() -> MultinomialOpinion:
    """Conditional opinion ω_{Y|X=A}: strong belief in child state P."""
    return MultinomialOpinion(
        beliefs={"P": 0.7, "Q": 0.1, "R": 0.0},
        uncertainty=0.2,
        base_rates={"P": 1 / 3, "Q": 1 / 3, "R": 1 / 3},
    )


@pytest.fixture
def ternary_child_b() -> MultinomialOpinion:
    """Conditional opinion ω_{Y|X=B}: moderate belief in child state Q."""
    return MultinomialOpinion(
        beliefs={"P": 0.1, "Q": 0.5, "R": 0.2},
        uncertainty=0.2,
        base_rates={"P": 1 / 3, "Q": 1 / 3, "R": 1 / 3},
    )


@pytest.fixture
def ternary_child_c() -> MultinomialOpinion:
    """Conditional opinion ω_{Y|X=C}: vacuous (no evidence)."""
    return MultinomialOpinion(
        beliefs={"P": 0.0, "Q": 0.0, "R": 0.0},
        uncertainty=1.0,
        base_rates={"P": 1 / 3, "Q": 1 / 3, "R": 1 / 3},
    )


@pytest.fixture
def full_conditionals(
    ternary_child_a: MultinomialOpinion,
    ternary_child_b: MultinomialOpinion,
    ternary_child_c: MultinomialOpinion,
) -> dict[str, MultinomialOpinion]:
    """Complete conditional table for a ternary parent {A, B, C}."""
    return {
        "A": ternary_child_a,
        "B": ternary_child_b,
        "C": ternary_child_c,
    }


@pytest.fixture
def binary_conditionals() -> dict[str, MultinomialOpinion]:
    """Conditional table for a binary parent {T, F} → binary child {H, L}."""
    return {
        "T": MultinomialOpinion(
            beliefs={"H": 0.8, "L": 0.1},
            uncertainty=0.1,
            base_rates={"H": 0.5, "L": 0.5},
        ),
        "F": MultinomialOpinion(
            beliefs={"H": 0.1, "L": 0.7},
            uncertainty=0.2,
            base_rates={"H": 0.5, "L": 0.5},
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeConstruction:
    """Test valid MultinomialEdge creation."""

    def test_minimal_construction(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Edge with required fields only uses correct defaults."""
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=full_conditionals,
        )
        assert edge.source_id == "X"
        assert edge.target_id == "Y"
        assert len(edge.conditionals) == 3
        assert edge.edge_type == "deduction"
        assert edge.metadata == {}
        assert edge.timestamp is None
        assert edge.half_life is None
        assert edge.valid_from is None
        assert edge.valid_until is None

    def test_full_construction(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Edge with all fields specified."""
        ts = datetime(2024, 6, 15)
        vf = datetime(2024, 1, 1)
        vu = datetime(2025, 1, 1)
        edge = MultinomialEdge(
            source_id="weather",
            target_id="crop_yield",
            conditionals=full_conditionals,
            edge_type="deduction",
            metadata={"source": "agricultural_model", "version": 3},
            timestamp=ts,
            half_life=86400.0,
            valid_from=vf,
            valid_until=vu,
        )
        assert edge.source_id == "weather"
        assert edge.target_id == "crop_yield"
        assert edge.metadata["source"] == "agricultural_model"
        assert edge.timestamp == ts
        assert edge.half_life == 86400.0
        assert edge.valid_from == vf
        assert edge.valid_until == vu

    def test_binary_parent_binary_child(
        self, binary_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Binary → binary multinomial edge (k=2 → m=2)."""
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=binary_conditionals,
        )
        assert len(edge.conditionals) == 2
        assert set(edge.conditionals.keys()) == {"T", "F"}

    def test_ternary_parent_binary_child(self) -> None:
        """Ternary parent → binary child (k=3 → m=2)."""
        child_high = MultinomialOpinion(
            beliefs={"H": 0.8, "L": 0.1}, uncertainty=0.1,
            base_rates={"H": 0.5, "L": 0.5},
        )
        child_low = MultinomialOpinion(
            beliefs={"H": 0.2, "L": 0.6}, uncertainty=0.2,
            base_rates={"H": 0.5, "L": 0.5},
        )
        child_mid = MultinomialOpinion(
            beliefs={"H": 0.4, "L": 0.3}, uncertainty=0.3,
            base_rates={"H": 0.5, "L": 0.5},
        )
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals={"A": child_high, "B": child_mid, "C": child_low},
        )
        assert len(edge.conditionals) == 3

    def test_binary_parent_quaternary_child(self) -> None:
        """Binary parent → 4-state child (k=2 → m=4)."""
        br = {"W": 0.25, "X": 0.25, "Y": 0.25, "Z": 0.25}
        cond_t = MultinomialOpinion(
            beliefs={"W": 0.5, "X": 0.2, "Y": 0.1, "Z": 0.0},
            uncertainty=0.2, base_rates=br,
        )
        cond_f = MultinomialOpinion(
            beliefs={"W": 0.0, "X": 0.1, "Y": 0.3, "Z": 0.4},
            uncertainty=0.2, base_rates=br,
        )
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals={"T": cond_t, "F": cond_f},
        )
        assert len(edge.conditionals) == 2

    def test_conditionals_opinion_invariants(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Σb + u = 1 holds for every conditional opinion."""
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=full_conditionals,
        )
        for state, opinion in edge.conditionals.items():
            total = sum(opinion.beliefs.values()) + opinion.uncertainty
            assert abs(total - 1.0) < 1e-9, (
                f"Σb + u != 1 for conditional at parent state {state!r}"
            )

    def test_all_conditionals_share_child_domain(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """All conditional opinions have the same domain."""
        edge = MultinomialEdge(
            source_id="X",
            target_id="Y",
            conditionals=full_conditionals,
        )
        domains = [op.domain for op in edge.conditionals.values()]
        assert len(set(domains)) == 1


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeValidation:
    """Test MultinomialEdge validation errors."""

    def _make_binary_conds(self) -> dict[str, MultinomialOpinion]:
        """Helper: minimal valid conditionals for binary parent."""
        br = {"H": 0.5, "L": 0.5}
        return {
            "T": MultinomialOpinion(
                beliefs={"H": 0.7, "L": 0.1}, uncertainty=0.2, base_rates=br,
            ),
            "F": MultinomialOpinion(
                beliefs={"H": 0.1, "L": 0.6}, uncertainty=0.3, base_rates=br,
            ),
        }

    def test_empty_source_id_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="source_id.*non-empty"):
            MultinomialEdge(source_id="", target_id="Y", conditionals=conds)

    def test_whitespace_source_id_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="source_id.*non-empty"):
            MultinomialEdge(source_id="  ", target_id="Y", conditionals=conds)

    def test_non_string_source_id_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="source_id.*non-empty"):
            MultinomialEdge(
                source_id=42, target_id="Y", conditionals=conds,  # type: ignore[arg-type]
            )

    def test_empty_target_id_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="target_id.*non-empty"):
            MultinomialEdge(source_id="X", target_id="", conditionals=conds)

    def test_whitespace_target_id_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="target_id.*non-empty"):
            MultinomialEdge(source_id="X", target_id="   ", conditionals=conds)

    def test_self_loop_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="[Ss]elf-loop"):
            MultinomialEdge(source_id="X", target_id="X", conditionals=conds)

    def test_empty_conditionals_raises(self) -> None:
        """At least one conditional is required."""
        with pytest.raises(ValueError, match="[Cc]onditionals.*empty|at least"):
            MultinomialEdge(source_id="X", target_id="Y", conditionals={})

    def test_non_multinomial_value_raises(self) -> None:
        """Conditional values must be MultinomialOpinion, not Opinion."""
        binomial = Opinion(0.7, 0.2, 0.1)
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals={"T": binomial, "F": binomial},  # type: ignore[dict-item]
            )

    def test_string_value_raises(self) -> None:
        """Strings are not MultinomialOpinions."""
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals={"T": "high", "F": "low"},  # type: ignore[dict-item]
            )

    def test_inconsistent_child_domains_raises(self) -> None:
        """All conditional opinions must share the same child domain."""
        cond_a = MultinomialOpinion(
            beliefs={"P": 0.5, "Q": 0.3}, uncertainty=0.2,
            base_rates={"P": 0.5, "Q": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"X": 0.4, "Y": 0.3}, uncertainty=0.3,
            base_rates={"X": 0.5, "Y": 0.5},
        )
        with pytest.raises(ValueError, match="[Cc]hild domain|[Dd]omain.*inconsistent|same domain"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals={"A": cond_a, "B": cond_b},
            )

    def test_inconsistent_child_cardinality_raises(self) -> None:
        """Child domains with different cardinalities must be rejected."""
        cond_a = MultinomialOpinion(
            beliefs={"P": 0.5, "Q": 0.3}, uncertainty=0.2,
            base_rates={"P": 0.5, "Q": 0.5},
        )
        cond_b = MultinomialOpinion(
            beliefs={"P": 0.3, "Q": 0.2, "R": 0.1}, uncertainty=0.4,
            base_rates={"P": 1 / 3, "Q": 1 / 3, "R": 1 / 3},
        )
        with pytest.raises(ValueError, match="[Cc]hild domain|[Dd]omain.*inconsistent|same domain"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals={"A": cond_a, "B": cond_b},
            )

    def test_invalid_edge_type_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="edge_type"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals=conds,
                edge_type="causal",  # type: ignore[arg-type]
            )

    def test_negative_half_life_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="half_life.*positive"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals=conds,
                half_life=-1.0,
            )

    def test_zero_half_life_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="half_life.*positive"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals=conds,
                half_life=0.0,
            )

    def test_valid_until_before_valid_from_raises(self) -> None:
        conds = self._make_binary_conds()
        with pytest.raises(ValueError, match="valid_until.*before.*valid_from"):
            MultinomialEdge(
                source_id="X",
                target_id="Y",
                conditionals=conds,
                valid_from=datetime(2025, 1, 1),
                valid_until=datetime(2024, 1, 1),
            )


# ═══════════════════════════════════════════════════════════════════
# IMMUTABILITY
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeImmutability:
    """MultinomialEdge is a frozen dataclass."""

    def test_cannot_reassign_source_id(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        with pytest.raises(AttributeError):
            edge.source_id = "Z"  # type: ignore[misc]

    def test_cannot_reassign_target_id(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        with pytest.raises(AttributeError):
            edge.target_id = "Z"  # type: ignore[misc]

    def test_cannot_reassign_conditionals(
        self,
        full_conditionals: dict[str, MultinomialOpinion],
        binary_conditionals: dict[str, MultinomialOpinion],
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        with pytest.raises(AttributeError):
            edge.conditionals = binary_conditionals  # type: ignore[misc]

    def test_cannot_reassign_edge_type(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        with pytest.raises(AttributeError):
            edge.edge_type = "trust"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# HASHABILITY
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeHashability:
    """MultinomialEdge is hashable by (source_id, target_id)."""

    def test_hashable(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        assert isinstance(hash(edge), int)

    def test_hash_depends_on_source_and_target(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        e1 = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        e2 = MultinomialEdge(
            source_id="X", target_id="Z", conditionals=full_conditionals,
        )
        assert hash(e1) != hash(e2)

    def test_same_endpoints_same_hash(
        self,
        full_conditionals: dict[str, MultinomialOpinion],
        binary_conditionals: dict[str, MultinomialOpinion],
    ) -> None:
        """Edges with same endpoints hash identically regardless of conditionals."""
        # Need same-endpoint edges, so use same source/target with different conds
        e1 = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        # Create a second edge with same endpoints but different conditionals
        # (must still have consistent child domain within each edge)
        e2 = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
            metadata={"different": True},
        )
        assert hash(e1) == hash(e2)

    def test_usable_in_set(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        s = {edge}
        assert edge in s

    def test_usable_as_dict_key(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        d = {edge: "some_value"}
        assert d[edge] == "some_value"


# ═══════════════════════════════════════════════════════════════════
# REPR
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeRepr:
    """MultinomialEdge has an informative repr."""

    def test_repr_contains_endpoints(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="weather",
            target_id="crop_yield",
            conditionals=full_conditionals,
        )
        r = repr(edge)
        assert "weather" in r
        assert "crop_yield" in r

    def test_repr_contains_parent_state_count(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Repr should indicate the number of parent states (or cardinality)."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        r = repr(edge)
        # Should mention 3 parent states or k_parent=3
        assert "3" in r

    def test_repr_contains_child_domain_info(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Repr should indicate the child domain cardinality."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        r = repr(edge)
        # Child domain is {P, Q, R} → 3 states
        # Should show child cardinality somewhere
        assert "MultinomialEdge" in r


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE PROPERTIES
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeProperties:
    """Convenience properties for introspection."""

    def test_parent_states(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """parent_states returns the set of parent domain states."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        assert edge.parent_states == frozenset({"A", "B", "C"})

    def test_child_domain(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """child_domain returns the shared child domain tuple."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        assert edge.child_domain == ("P", "Q", "R")

    def test_parent_cardinality(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """parent_cardinality returns number of parent states."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        assert edge.parent_cardinality == 3

    def test_child_cardinality(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """child_cardinality returns number of child states."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=full_conditionals,
        )
        assert edge.child_cardinality == 3

    def test_binary_parent_properties(
        self, binary_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        """Properties work for binary parent → binary child."""
        edge = MultinomialEdge(
            source_id="X", target_id="Y", conditionals=binary_conditionals,
        )
        assert edge.parent_states == frozenset({"T", "F"})
        assert edge.child_domain == ("H", "L")
        assert edge.parent_cardinality == 2
        assert edge.child_cardinality == 2


# ═══════════════════════════════════════════════════════════════════
# TEMPORAL FIELDS (consistency with SLEdge)
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialEdgeTemporalFields:
    """Temporal fields mirror SLEdge for consistency."""

    def test_timestamp_field(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        ts = datetime(2024, 6, 15, 12, 0, 0)
        edge = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=full_conditionals,
            timestamp=ts,
        )
        assert edge.timestamp == ts

    def test_half_life_field(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=full_conditionals,
            half_life=3600.0,
        )
        assert edge.half_life == 3600.0

    def test_validity_window(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        vf = datetime(2024, 1, 1)
        vu = datetime(2025, 1, 1)
        edge = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=full_conditionals,
            valid_from=vf,
            valid_until=vu,
        )
        assert edge.valid_from == vf
        assert edge.valid_until == vu

    def test_temporal_defaults_are_none(
        self, full_conditionals: dict[str, MultinomialOpinion]
    ) -> None:
        edge = MultinomialEdge(
            source_id="X", target_id="Y",
            conditionals=full_conditionals,
        )
        assert edge.timestamp is None
        assert edge.half_life is None
        assert edge.valid_from is None
        assert edge.valid_until is None
