"""
Tests for SLNode multinomial opinion support (Phase B, Step 1).

Extends SLNode with an optional ``multinomial_opinion`` field so that
SLNetwork can handle k-ary variables alongside existing binary nodes.

Design contract:
    - ``multinomial_opinion`` defaults to ``None`` (backward compatible).
    - When present, it must be a ``MultinomialOpinion`` instance.
    - The existing ``opinion`` field (binomial) remains required and
      unchanged — zero breaking changes.
    - Hashability and immutability are preserved.

TDD: RED phase — these tests define the contract for the extension.
All tests should FAIL until types.py is updated.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.multinomial_algebra import MultinomialOpinion, coarsen
from jsonld_ex.sl_network.types import SLNode


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def binomial_opinion() -> Opinion:
    """Standard binomial opinion: b=0.6, d=0.2, u=0.2."""
    return Opinion(belief=0.6, disbelief=0.2, uncertainty=0.2)


@pytest.fixture
def vacuous_opinion() -> Opinion:
    """Vacuous binomial opinion: b=0, d=0, u=1."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def ternary_opinion() -> MultinomialOpinion:
    """Ternary opinion over {A, B, C}: moderate belief in A."""
    return MultinomialOpinion(
        beliefs={"A": 0.5, "B": 0.2, "C": 0.1},
        uncertainty=0.2,
        base_rates={"A": 0.5, "B": 0.3, "C": 0.2},
    )


@pytest.fixture
def binary_multinomial() -> MultinomialOpinion:
    """Binary multinomial opinion over {T, F}."""
    return MultinomialOpinion(
        beliefs={"T": 0.6, "F": 0.2},
        uncertainty=0.2,
        base_rates={"T": 0.5, "F": 0.5},
    )


@pytest.fixture
def vacuous_multinomial() -> MultinomialOpinion:
    """Vacuous ternary opinion: complete ignorance."""
    return MultinomialOpinion(
        beliefs={"A": 0.0, "B": 0.0, "C": 0.0},
        uncertainty=1.0,
        base_rates={"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
    )


# ═══════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Existing SLNode usage must be completely unchanged."""

    def test_default_multinomial_is_none(self, binomial_opinion: Opinion) -> None:
        """Without multinomial_opinion arg, field defaults to None."""
        node = SLNode(node_id="X", opinion=binomial_opinion)
        assert node.multinomial_opinion is None

    def test_existing_positional_args_unchanged(self, binomial_opinion: Opinion) -> None:
        """Existing positional construction still works."""
        node = SLNode("X", binomial_opinion)
        assert node.node_id == "X"
        assert node.opinion is binomial_opinion
        assert node.node_type == "content"
        assert node.multinomial_opinion is None

    def test_existing_keyword_args_unchanged(self, binomial_opinion: Opinion) -> None:
        """Existing keyword construction still works."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            node_type="agent",
            label="test",
            metadata={"key": "val"},
        )
        assert node.node_id == "X"
        assert node.opinion is binomial_opinion
        assert node.node_type == "agent"
        assert node.label == "test"
        assert node.metadata == {"key": "val"}
        assert node.multinomial_opinion is None

    def test_temporal_fields_still_work(self, binomial_opinion: Opinion) -> None:
        """Temporal fields (timestamp, half_life) remain functional."""
        from datetime import datetime

        ts = datetime(2024, 1, 1)
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            timestamp=ts,
            half_life=3600.0,
        )
        assert node.timestamp == ts
        assert node.half_life == 3600.0
        assert node.multinomial_opinion is None

    def test_hash_unchanged_without_multinomial(self, binomial_opinion: Opinion) -> None:
        """Hash still based on node_id only."""
        node_a = SLNode(node_id="X", opinion=binomial_opinion)
        node_b = SLNode(
            node_id="X",
            opinion=Opinion(0.1, 0.1, 0.8),
        )
        assert hash(node_a) == hash(node_b)


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCTION WITH MULTINOMIAL
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialConstruction:
    """SLNode can be constructed with a MultinomialOpinion."""

    def test_explicit_multinomial_via_keyword(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """Multinomial opinion can be set via keyword argument."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        assert node.multinomial_opinion is ternary_opinion
        assert node.opinion is binomial_opinion

    def test_multinomial_with_all_fields(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """All fields can be set simultaneously."""
        from datetime import datetime

        ts = datetime(2024, 6, 15)
        node = SLNode(
            node_id="multi_node",
            opinion=binomial_opinion,
            node_type="content",
            label="A ternary proposition",
            metadata={"domain": "weather"},
            timestamp=ts,
            half_life=7200.0,
            multinomial_opinion=ternary_opinion,
        )
        assert node.node_id == "multi_node"
        assert node.opinion is binomial_opinion
        assert node.multinomial_opinion is ternary_opinion
        assert node.label == "A ternary proposition"
        assert node.timestamp == ts
        assert node.half_life == 7200.0

    def test_multinomial_none_explicit(self, binomial_opinion: Opinion) -> None:
        """Explicitly passing None is equivalent to default."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=None,
        )
        assert node.multinomial_opinion is None


# ═══════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialValidation:
    """Validation ensures type correctness of multinomial_opinion."""

    def test_rejects_non_multinomial_type(self, binomial_opinion: Opinion) -> None:
        """multinomial_opinion must be MultinomialOpinion or None."""
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            SLNode(
                node_id="X",
                opinion=binomial_opinion,
                multinomial_opinion=binomial_opinion,  # type: ignore[arg-type]
            )

    def test_rejects_string(self, binomial_opinion: Opinion) -> None:
        """Strings are not valid MultinomialOpinions."""
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            SLNode(
                node_id="X",
                opinion=binomial_opinion,
                multinomial_opinion="not an opinion",  # type: ignore[arg-type]
            )

    def test_rejects_dict(self, binomial_opinion: Opinion) -> None:
        """Dicts are not valid MultinomialOpinions."""
        with pytest.raises(TypeError, match="MultinomialOpinion"):
            SLNode(
                node_id="X",
                opinion=binomial_opinion,
                multinomial_opinion={"A": 0.5, "B": 0.5},  # type: ignore[arg-type]
            )

    def test_accepts_any_cardinality(self, binomial_opinion: Opinion) -> None:
        """MultinomialOpinion with any k >= 2 is accepted."""
        # k=4
        quad = MultinomialOpinion(
            beliefs={"W": 0.2, "X": 0.2, "Y": 0.2, "Z": 0.1},
            uncertainty=0.3,
            base_rates={"W": 0.25, "X": 0.25, "Y": 0.25, "Z": 0.25},
        )
        node = SLNode(
            node_id="quad",
            opinion=binomial_opinion,
            multinomial_opinion=quad,
        )
        assert node.multinomial_opinion is quad
        assert node.multinomial_opinion.cardinality == 4


# ═══════════════════════════════════════════════════════════════════
# IMMUTABILITY
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialImmutability:
    """The multinomial_opinion field is frozen (immutable)."""

    def test_cannot_set_multinomial_after_construction(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """Frozen dataclass prevents field mutation."""
        node = SLNode(node_id="X", opinion=binomial_opinion)
        with pytest.raises(AttributeError):
            node.multinomial_opinion = ternary_opinion  # type: ignore[misc]

    def test_cannot_replace_multinomial(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
        vacuous_multinomial: MultinomialOpinion,
    ) -> None:
        """Cannot replace an existing multinomial_opinion."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        with pytest.raises(AttributeError):
            node.multinomial_opinion = vacuous_multinomial  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════
# HASHABILITY
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialHashability:
    """Nodes with multinomial opinions remain hashable."""

    def test_hashable_with_multinomial(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """Nodes with multinomial opinions can be hashed."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        assert isinstance(hash(node), int)

    def test_hash_still_based_on_node_id(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
        vacuous_multinomial: MultinomialOpinion,
    ) -> None:
        """Hash depends on node_id, not on multinomial_opinion value."""
        node_a = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        node_b = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=vacuous_multinomial,
        )
        assert hash(node_a) == hash(node_b)

    def test_usable_in_set(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """Nodes with multinomial opinions work in sets."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        s = {node}
        assert node in s


# ═══════════════════════════════════════════════════════════════════
# REPR
# ═══════════════════════════════════════════════════════════════════


class TestMultinomialRepr:
    """Repr reflects multinomial state when present."""

    def test_repr_without_multinomial(self, binomial_opinion: Opinion) -> None:
        """Without multinomial, repr is unchanged from current behavior."""
        node = SLNode(node_id="X", opinion=binomial_opinion)
        r = repr(node)
        assert "X" in r
        assert "multinomial" not in r.lower() or "None" in r

    def test_repr_with_multinomial_shows_cardinality(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """With multinomial, repr should indicate its presence."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        r = repr(node)
        assert "X" in r
        # Should mention multinomial or k=3 somehow
        assert "k=3" in r or "multinomial" in r.lower()


# ═══════════════════════════════════════════════════════════════════
# HELPER: IS_MULTINOMIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════


class TestIsMultinomial:
    """Convenience property to check if node is multinomial."""

    def test_binary_node_is_not_multinomial(self, binomial_opinion: Opinion) -> None:
        """Node without multinomial_opinion is not multinomial."""
        node = SLNode(node_id="X", opinion=binomial_opinion)
        assert node.is_multinomial is False

    def test_multinomial_node_is_multinomial(
        self,
        binomial_opinion: Opinion,
        ternary_opinion: MultinomialOpinion,
    ) -> None:
        """Node with multinomial_opinion is multinomial."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=ternary_opinion,
        )
        assert node.is_multinomial is True

    def test_explicit_none_is_not_multinomial(self, binomial_opinion: Opinion) -> None:
        """Explicitly None multinomial_opinion is not multinomial."""
        node = SLNode(
            node_id="X",
            opinion=binomial_opinion,
            multinomial_opinion=None,
        )
        assert node.is_multinomial is False


# ═══════════════════════════════════════════════════════════════════
# COARSENING CONSISTENCY (informational, not enforced at construction)
# ═══════════════════════════════════════════════════════════════════


class TestCoarseningConsistency:
    """When both binomial and multinomial opinions are present on a
    k=2 node, verify they can be checked for consistency.

    This is NOT enforced at construction time (the user might
    intentionally set a different binomial summary). But the
    relationship should be expressible and testable.
    """

    def test_consistent_binary_multinomial_pair(self) -> None:
        """A k=2 multinomial coarsened to binomial matches the binomial opinion."""
        multi = MultinomialOpinion(
            beliefs={"T": 0.6, "F": 0.2},
            uncertainty=0.2,
            base_rates={"T": 0.5, "F": 0.5},
        )
        binomial = coarsen(multi, "T")

        node = SLNode(
            node_id="X",
            opinion=binomial,
            multinomial_opinion=multi,
        )

        # The coarsened multinomial should match the binomial
        recoarsened = coarsen(node.multinomial_opinion, "T")
        assert abs(recoarsened.belief - node.opinion.belief) < 1e-9
        assert abs(recoarsened.disbelief - node.opinion.disbelief) < 1e-9
        assert abs(recoarsened.uncertainty - node.opinion.uncertainty) < 1e-9
        assert abs(recoarsened.base_rate - node.opinion.base_rate) < 1e-9
