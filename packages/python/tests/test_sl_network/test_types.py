"""
Tests for SLNetwork data types (Tier 1, Step 1).

Covers construction, validation, immutability, hashability,
repr, and all error paths for:
    - SLNode
    - SLEdge
    - MultiParentEdge
    - InferenceStep
    - InferenceResult

TDD: These tests define the contract that types.py must satisfy.
"""

from __future__ import annotations

import pytest

from jsonld_ex.confidence_algebra import Opinion
from jsonld_ex.sl_network.types import (
    EdgeType,
    InferenceResult,
    InferenceStep,
    MultiParentEdge,
    NodeType,
    SLEdge,
    SLNode,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def high_belief() -> Opinion:
    """Strong belief opinion: b=0.8, d=0.1, u=0.1."""
    return Opinion(belief=0.8, disbelief=0.1, uncertainty=0.1)


@pytest.fixture
def moderate_belief() -> Opinion:
    """Moderate belief: b=0.5, d=0.2, u=0.3."""
    return Opinion(belief=0.5, disbelief=0.2, uncertainty=0.3)


@pytest.fixture
def vacuous() -> Opinion:
    """Complete ignorance: b=0, d=0, u=1."""
    return Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)


@pytest.fixture
def dogmatic_true() -> Opinion:
    """Dogmatic belief: b=1, d=0, u=0."""
    return Opinion(belief=1.0, disbelief=0.0, uncertainty=0.0)


@pytest.fixture
def dogmatic_false() -> Opinion:
    """Dogmatic disbelief: b=0, d=1, u=0."""
    return Opinion(belief=0.0, disbelief=1.0, uncertainty=0.0)


# ═══════════════════════════════════════════════════════════════════
# SLNode TESTS
# ═══════════════════════════════════════════════════════════════════


class TestSLNodeConstruction:
    """Test valid SLNode creation."""

    def test_minimal_construction(self, high_belief: Opinion) -> None:
        """Node with required fields only uses correct defaults."""
        node = SLNode(node_id="X", opinion=high_belief)
        assert node.node_id == "X"
        assert node.opinion is high_belief
        assert node.node_type == "content"
        assert node.label is None
        assert node.metadata == {}

    def test_full_construction(self, moderate_belief: Opinion) -> None:
        """Node with all fields specified."""
        meta = {"source": "http://example.org", "version": 2}
        node = SLNode(
            node_id="disease_present",
            opinion=moderate_belief,
            node_type="content",
            label="Disease is present",
            metadata=meta,
        )
        assert node.node_id == "disease_present"
        assert node.label == "Disease is present"
        assert node.metadata["source"] == "http://example.org"
        assert node.metadata["version"] == 2

    def test_agent_node_type(self, high_belief: Opinion) -> None:
        """Agent node type is accepted (forward compat for Tier 2)."""
        node = SLNode(node_id="agent_A", opinion=high_belief, node_type="agent")
        assert node.node_type == "agent"

    def test_opinion_invariant_preserved(self, high_belief: Opinion) -> None:
        """The b+d+u=1 invariant holds on the stored opinion."""
        node = SLNode(node_id="X", opinion=high_belief)
        op = node.opinion
        assert abs(op.belief + op.disbelief + op.uncertainty - 1.0) < 1e-9

    def test_vacuous_opinion_node(self, vacuous: Opinion) -> None:
        """Node with vacuous opinion (complete ignorance)."""
        node = SLNode(node_id="unknown", opinion=vacuous)
        assert node.opinion.uncertainty == 1.0
        assert node.opinion.belief == 0.0
        assert node.opinion.disbelief == 0.0

    def test_dogmatic_opinion_node(self, dogmatic_true: Opinion) -> None:
        """Node with dogmatic opinion (zero uncertainty)."""
        node = SLNode(node_id="certain", opinion=dogmatic_true)
        assert node.opinion.uncertainty == 0.0
        assert node.opinion.belief == 1.0


class TestSLNodeValidation:
    """Test SLNode validation errors."""

    def test_empty_node_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            SLNode(node_id="", opinion=high_belief)

    def test_whitespace_only_node_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            SLNode(node_id="   ", opinion=high_belief)

    def test_non_string_node_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            SLNode(node_id=42, opinion=high_belief)  # type: ignore[arg-type]

    def test_non_opinion_raises(self) -> None:
        with pytest.raises(TypeError, match="Opinion instance"):
            SLNode(node_id="X", opinion=0.5)  # type: ignore[arg-type]

    def test_dict_opinion_raises(self) -> None:
        """A dict is not an Opinion, even if it looks like one."""
        with pytest.raises(TypeError, match="Opinion instance"):
            SLNode(
                node_id="X",
                opinion={"belief": 0.5, "disbelief": 0.3, "uncertainty": 0.2},  # type: ignore[arg-type]
            )

    def test_invalid_node_type_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="'content' or 'agent'"):
            SLNode(node_id="X", opinion=high_belief, node_type="sensor")  # type: ignore[arg-type]


class TestSLNodeImmutability:
    """Test that SLNode is frozen."""

    def test_cannot_reassign_node_id(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        with pytest.raises(AttributeError):
            node.node_id = "Y"  # type: ignore[misc]

    def test_cannot_reassign_opinion(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        with pytest.raises(AttributeError):
            node.opinion = vacuous  # type: ignore[misc]

    def test_cannot_reassign_metadata(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        with pytest.raises(AttributeError):
            node.metadata = {"new": True}  # type: ignore[misc]

    def test_metadata_contents_are_mutable(self, high_belief: Opinion) -> None:
        """The metadata dict itself is mutable (only the field ref is frozen).

        This is expected Python behavior for frozen dataclasses with
        mutable default fields.  Documented, not a bug.
        """
        node = SLNode(node_id="X", opinion=high_belief, metadata={"a": 1})
        node.metadata["b"] = 2
        assert node.metadata == {"a": 1, "b": 2}


class TestSLNodeHashability:
    """Test SLNode hashing and equality."""

    def test_hashable(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        # Should not raise
        h = hash(node)
        assert isinstance(h, int)

    def test_hash_depends_on_node_id(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        """Two nodes with different IDs have different hashes."""
        n1 = SLNode(node_id="A", opinion=high_belief)
        n2 = SLNode(node_id="B", opinion=vacuous)
        # Not guaranteed by hash, but very likely
        assert hash(n1) != hash(n2)

    def test_same_id_same_hash(self, high_belief: Opinion, vacuous: Opinion) -> None:
        """Nodes with the same ID hash identically (identity = node_id)."""
        n1 = SLNode(node_id="X", opinion=high_belief)
        n2 = SLNode(node_id="X", opinion=vacuous)
        assert hash(n1) == hash(n2)

    def test_usable_in_set(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        s = {node}
        assert node in s

    def test_equality_checks_all_fields(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        """Equality compares all fields, not just node_id."""
        n1 = SLNode(node_id="X", opinion=high_belief)
        n2 = SLNode(node_id="X", opinion=vacuous)
        # Same node_id but different opinions → not equal
        assert n1 != n2


class TestSLNodeRepr:
    """Test SLNode string representation."""

    def test_repr_contains_id(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="fever", opinion=high_belief)
        r = repr(node)
        assert "fever" in r

    def test_repr_contains_opinion_values(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief)
        r = repr(node)
        assert "0.800" in r
        assert "0.100" in r

    def test_repr_contains_type(self, high_belief: Opinion) -> None:
        node = SLNode(node_id="X", opinion=high_belief, node_type="agent")
        assert "'agent'" in repr(node)


# ═══════════════════════════════════════════════════════════════════
# SLEdge TESTS
# ═══════════════════════════════════════════════════════════════════


class TestSLEdgeConstruction:
    """Test valid SLEdge creation."""

    def test_minimal_construction(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        assert edge.source_id == "A"
        assert edge.target_id == "B"
        assert edge.conditional is high_belief
        assert edge.counterfactual is None
        assert edge.edge_type == "deduction"
        assert edge.metadata == {}

    def test_full_construction(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        edge = SLEdge(
            source_id="symptom",
            target_id="disease",
            conditional=high_belief,
            counterfactual=vacuous,
            edge_type="deduction",
            metadata={"source": "medical_kb"},
        )
        assert edge.counterfactual is vacuous
        assert edge.metadata["source"] == "medical_kb"

    def test_opinion_invariant_on_conditional(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        c = edge.conditional
        assert abs(c.belief + c.disbelief + c.uncertainty - 1.0) < 1e-9

    def test_opinion_invariant_on_counterfactual(
        self, high_belief: Opinion, moderate_belief: Opinion
    ) -> None:
        edge = SLEdge(
            source_id="A",
            target_id="B",
            conditional=high_belief,
            counterfactual=moderate_belief,
        )
        cf = edge.counterfactual
        assert cf is not None
        assert abs(cf.belief + cf.disbelief + cf.uncertainty - 1.0) < 1e-9


class TestSLEdgeValidation:
    """Test SLEdge validation errors."""

    def test_empty_source_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="source_id.*non-empty"):
            SLEdge(source_id="", target_id="B", conditional=high_belief)

    def test_empty_target_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="target_id.*non-empty"):
            SLEdge(source_id="A", target_id="", conditional=high_belief)

    def test_whitespace_source_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="source_id.*non-empty"):
            SLEdge(source_id="  ", target_id="B", conditional=high_belief)

    def test_self_loop_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="Self-loops"):
            SLEdge(source_id="X", target_id="X", conditional=high_belief)

    def test_non_opinion_conditional_raises(self) -> None:
        with pytest.raises(TypeError, match="conditional.*Opinion"):
            SLEdge(source_id="A", target_id="B", conditional=0.9)  # type: ignore[arg-type]

    def test_non_opinion_counterfactual_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(TypeError, match="counterfactual.*Opinion"):
            SLEdge(
                source_id="A",
                target_id="B",
                conditional=high_belief,
                counterfactual="vacuous",  # type: ignore[arg-type]
            )

    def test_invalid_edge_type_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="edge_type"):
            SLEdge(
                source_id="A",
                target_id="B",
                conditional=high_belief,
                edge_type="causal",  # type: ignore[arg-type]
            )


class TestSLEdgeImmutability:
    """Test that SLEdge is frozen."""

    def test_cannot_reassign_source(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        with pytest.raises(AttributeError):
            edge.source_id = "C"  # type: ignore[misc]

    def test_cannot_reassign_conditional(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        with pytest.raises(AttributeError):
            edge.conditional = vacuous  # type: ignore[misc]


class TestSLEdgeHashability:
    """Test SLEdge hashing."""

    def test_hashable(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        h = hash(edge)
        assert isinstance(h, int)

    def test_hash_depends_on_source_target(self, high_belief: Opinion) -> None:
        e1 = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        e2 = SLEdge(source_id="A", target_id="C", conditional=high_belief)
        assert hash(e1) != hash(e2)

    def test_same_endpoints_same_hash(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        """Edges with same endpoints hash identically."""
        e1 = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        e2 = SLEdge(source_id="A", target_id="B", conditional=vacuous)
        assert hash(e1) == hash(e2)

    def test_usable_in_set(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        s = {edge}
        assert edge in s


class TestSLEdgeRepr:
    """Test SLEdge string representation."""

    def test_repr_contains_endpoints(self, high_belief: Opinion) -> None:
        edge = SLEdge(source_id="fever", target_id="infection", conditional=high_belief)
        r = repr(edge)
        assert "fever" in r
        assert "infection" in r

    def test_repr_shows_counterfactual_status(
        self, high_belief: Opinion, vacuous: Opinion
    ) -> None:
        e1 = SLEdge(source_id="A", target_id="B", conditional=high_belief)
        assert "cf=None" in repr(e1)

        e2 = SLEdge(
            source_id="A",
            target_id="B",
            conditional=high_belief,
            counterfactual=vacuous,
        )
        assert "cf=set" in repr(e2)


# ═══════════════════════════════════════════════════════════════════
# MultiParentEdge TESTS
# ═══════════════════════════════════════════════════════════════════


class TestMultiParentEdgeConstruction:
    """Test valid MultiParentEdge creation."""

    def test_two_parent_construction(self) -> None:
        """Standard 2-parent conditional table (4 entries)."""
        conditionals = {
            (True, True): Opinion(0.9, 0.05, 0.05),
            (True, False): Opinion(0.6, 0.2, 0.2),
            (False, True): Opinion(0.5, 0.3, 0.2),
            (False, False): Opinion(0.1, 0.7, 0.2),
        }
        mpe = MultiParentEdge(
            target_id="Y",
            parent_ids=("A", "B"),
            conditionals=conditionals,
        )
        assert mpe.target_id == "Y"
        assert mpe.parent_ids == ("A", "B")
        assert len(mpe.conditionals) == 4
        assert mpe.edge_type == "deduction"

    def test_three_parent_construction(self) -> None:
        """3-parent conditional table (8 entries)."""
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {}
        for a in (True, False):
            for b in (True, False):
                for c in (True, False):
                    conditionals[(a, b, c)] = op

        mpe = MultiParentEdge(
            target_id="Z",
            parent_ids=("A", "B", "C"),
            conditionals=conditionals,
        )
        assert len(mpe.conditionals) == 8

    def test_opinion_invariant_all_entries(self) -> None:
        """b+d+u=1 holds for every conditional opinion in the table."""
        conditionals = {
            (True, True): Opinion(0.9, 0.05, 0.05),
            (True, False): Opinion(0.6, 0.2, 0.2),
            (False, True): Opinion(0.5, 0.3, 0.2),
            (False, False): Opinion(0.1, 0.7, 0.2),
        }
        mpe = MultiParentEdge(
            target_id="Y",
            parent_ids=("A", "B"),
            conditionals=conditionals,
        )
        for key, op in mpe.conditionals.items():
            total = op.belief + op.disbelief + op.uncertainty
            assert abs(total - 1.0) < 1e-9, f"b+d+u != 1 for key {key}"


class TestMultiParentEdgeValidation:
    """Test MultiParentEdge validation errors."""

    def _make_conditionals_2(self) -> dict[tuple[bool, ...], Opinion]:
        """Helper: complete 2-parent conditional table."""
        op = Opinion(0.5, 0.3, 0.2)
        return {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            (False, False): op,
        }

    def test_fewer_than_two_parents_raises(self) -> None:
        """Single parent should use SLEdge, not MultiParentEdge."""
        op = Opinion(0.5, 0.3, 0.2)
        with pytest.raises(ValueError, match="at least 2 parents"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A",),
                conditionals={(True,): op, (False,): op},
            )

    def test_empty_parents_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 parents"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=(),
                conditionals={},
            )

    def test_duplicate_parent_ids_raises(self) -> None:
        conds = self._make_conditionals_2()
        with pytest.raises(ValueError, match="distinct"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "A"),
                conditionals=conds,
            )

    def test_target_in_parents_raises(self) -> None:
        """Self-loop: target cannot be its own parent."""
        conds = self._make_conditionals_2()
        with pytest.raises(ValueError, match="self-loop"):
            MultiParentEdge(
                target_id="A",
                parent_ids=("A", "B"),
                conditionals=conds,
            )

    def test_missing_conditional_entry_raises(self) -> None:
        """Incomplete table — missing one entry out of 4."""
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            # Missing: (False, False)
        }
        with pytest.raises(ValueError, match="2\\^2 = 4 entries"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "B"),
                conditionals=conditionals,
            )

    def test_extra_conditional_entry_raises(self) -> None:
        """Too many entries in the table."""
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            (False, False): op,
            (True, True, True): op,  # Wrong arity
        }
        with pytest.raises(ValueError, match="2\\^2 = 4 entries"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "B"),
                conditionals=conditionals,
            )

    def test_non_opinion_value_raises(self) -> None:
        """A float in the conditional table instead of Opinion."""
        conditionals = {
            (True, True): Opinion(0.5, 0.3, 0.2),
            (True, False): Opinion(0.5, 0.3, 0.2),
            (False, True): 0.7,  # type: ignore[dict-item]
            (False, False): Opinion(0.5, 0.3, 0.2),
        }
        with pytest.raises(TypeError, match="Opinion"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "B"),
                conditionals=conditionals,  # type: ignore[arg-type]
            )

    def test_non_bool_key_raises(self) -> None:
        """Keys must be tuples of bools, not ints."""
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            (0, 0): op,  # ints, not bools
        }
        # 0 == False in Python, so this might pass the set check
        # but should fail the isinstance(v, bool) check since
        # isinstance(0, bool) is False in Python (bool is subclass of int,
        # but 0 is an int, not a bool)
        # Actually: isinstance(0, bool) is False, isinstance(False, bool) is True
        # So (0, 0) != (False, False) in a set? Let me check:
        # Actually, 0 == False and hash(0) == hash(False), so
        # {(False, False)} == {(0, 0)} in Python. This means the key
        # (0, 0) would match (False, False) in the expected_keys set.
        # But our per-key isinstance check would catch the int.
        # However, (0, 0) hashes the same as (False, False), so
        # if both are in the dict, the dict only keeps one.
        # The test needs a key that ISN'T equivalent to a valid key.
        # Let me use a string instead.
        conditionals_bad: dict = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            ("no", "yes"): op,  # strings, not bools
        }
        with pytest.raises((ValueError, TypeError)):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "B"),
                conditionals=conditionals_bad,
            )

    def test_empty_parent_id_raises(self) -> None:
        conds = self._make_conditionals_2()
        with pytest.raises(ValueError, match="non-empty strings"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", ""),
                conditionals=conds,
            )

    def test_empty_target_id_raises(self) -> None:
        conds = self._make_conditionals_2()
        with pytest.raises(ValueError, match="target_id.*non-empty"):
            MultiParentEdge(
                target_id="",
                parent_ids=("A", "B"),
                conditionals=conds,
            )

    def test_parent_ids_must_be_tuple(self) -> None:
        """Lists are not accepted — must be a tuple for hashability."""
        conds = self._make_conditionals_2()
        with pytest.raises(TypeError, match="tuple"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=["A", "B"],  # type: ignore[arg-type]
                conditionals=conds,
            )

    def test_invalid_edge_type_raises(self) -> None:
        conds = self._make_conditionals_2()
        with pytest.raises(ValueError, match="edge_type"):
            MultiParentEdge(
                target_id="Y",
                parent_ids=("A", "B"),
                conditionals=conds,
                edge_type="causal",  # type: ignore[arg-type]
            )


class TestMultiParentEdgeHashRepr:
    """Test MultiParentEdge hashing and repr."""

    def test_hashable(self) -> None:
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            (False, False): op,
        }
        mpe = MultiParentEdge(
            target_id="Y", parent_ids=("A", "B"), conditionals=conditionals
        )
        assert isinstance(hash(mpe), int)

    def test_repr_contains_target_and_parents(self) -> None:
        op = Opinion(0.5, 0.3, 0.2)
        conditionals = {
            (True, True): op,
            (True, False): op,
            (False, True): op,
            (False, False): op,
        }
        mpe = MultiParentEdge(
            target_id="Y", parent_ids=("A", "B"), conditionals=conditionals
        )
        r = repr(mpe)
        assert "Y" in r
        assert "A" in r
        assert "B" in r
        assert "2^2=4" in r


# ═══════════════════════════════════════════════════════════════════
# InferenceStep TESTS
# ═══════════════════════════════════════════════════════════════════


class TestInferenceStepConstruction:
    """Test valid InferenceStep creation."""

    def test_minimal_step(self, high_belief: Opinion) -> None:
        step = InferenceStep(
            node_id="X",
            operation="passthrough",
            inputs={},
            result=high_belief,
        )
        assert step.node_id == "X"
        assert step.operation == "passthrough"
        assert step.inputs == {}
        assert step.result is high_belief

    def test_deduce_step(
        self, high_belief: Opinion, moderate_belief: Opinion, vacuous: Opinion
    ) -> None:
        step = InferenceStep(
            node_id="disease",
            operation="deduce",
            inputs={
                "parent": high_belief,
                "conditional": moderate_belief,
                "counterfactual": vacuous,
            },
            result=moderate_belief,
        )
        assert len(step.inputs) == 3
        assert step.inputs["parent"] is high_belief

    def test_fuse_parents_step(
        self, high_belief: Opinion, moderate_belief: Opinion
    ) -> None:
        step = InferenceStep(
            node_id="Y",
            operation="fuse_parents",
            inputs={"parent_A": high_belief, "parent_B": moderate_belief},
            result=high_belief,
        )
        assert step.operation == "fuse_parents"


class TestInferenceStepValidation:
    """Test InferenceStep validation errors."""

    def test_empty_node_id_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="node_id.*non-empty"):
            InferenceStep(
                node_id="", operation="deduce", inputs={}, result=high_belief
            )

    def test_empty_operation_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="operation.*non-empty"):
            InferenceStep(
                node_id="X", operation="", inputs={}, result=high_belief
            )

    def test_non_opinion_result_raises(self) -> None:
        with pytest.raises(TypeError, match="result.*Opinion"):
            InferenceStep(
                node_id="X", operation="deduce", inputs={}, result=0.5  # type: ignore[arg-type]
            )

    def test_non_opinion_input_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(TypeError, match="inputs.*Opinion"):
            InferenceStep(
                node_id="X",
                operation="deduce",
                inputs={"parent": 0.5},  # type: ignore[dict-item]
                result=high_belief,
            )


class TestInferenceStepHashRepr:
    """Test InferenceStep hashing and repr."""

    def test_hashable(self, high_belief: Opinion) -> None:
        step = InferenceStep(
            node_id="X", operation="deduce", inputs={}, result=high_belief
        )
        assert isinstance(hash(step), int)

    def test_repr_contains_info(self, high_belief: Opinion) -> None:
        step = InferenceStep(
            node_id="disease", operation="deduce", inputs={}, result=high_belief
        )
        r = repr(step)
        assert "disease" in r
        assert "deduce" in r


# ═══════════════════════════════════════════════════════════════════
# InferenceResult TESTS
# ═══════════════════════════════════════════════════════════════════


class TestInferenceResultConstruction:
    """Test valid InferenceResult creation."""

    def test_minimal_result(self, high_belief: Opinion) -> None:
        result = InferenceResult(
            query_node="X",
            opinion=high_belief,
            steps=[],
            intermediate_opinions={},
            topological_order=[],
        )
        assert result.query_node == "X"
        assert result.opinion is high_belief
        assert result.steps == []
        assert result.intermediate_opinions == {}
        assert result.topological_order == []

    def test_full_result(
        self, high_belief: Opinion, moderate_belief: Opinion
    ) -> None:
        step = InferenceStep(
            node_id="Y",
            operation="deduce",
            inputs={"parent": high_belief},
            result=moderate_belief,
        )
        result = InferenceResult(
            query_node="Y",
            opinion=moderate_belief,
            steps=[step],
            intermediate_opinions={"X": high_belief, "Y": moderate_belief},
            topological_order=["X", "Y"],
        )
        assert len(result.steps) == 1
        assert result.intermediate_opinions["X"] is high_belief
        assert result.topological_order == ["X", "Y"]


class TestInferenceResultValidation:
    """Test InferenceResult validation errors."""

    def test_empty_query_node_raises(self, high_belief: Opinion) -> None:
        with pytest.raises(ValueError, match="query_node.*non-empty"):
            InferenceResult(
                query_node="",
                opinion=high_belief,
                steps=[],
                intermediate_opinions={},
                topological_order=[],
            )

    def test_non_opinion_raises(self) -> None:
        with pytest.raises(TypeError, match="opinion.*Opinion"):
            InferenceResult(
                query_node="X",
                opinion={"belief": 0.5},  # type: ignore[arg-type]
                steps=[],
                intermediate_opinions={},
                topological_order=[],
            )


class TestInferenceResultHashRepr:
    """Test InferenceResult hashing and repr."""

    def test_hashable(self, high_belief: Opinion) -> None:
        result = InferenceResult(
            query_node="X",
            opinion=high_belief,
            steps=[],
            intermediate_opinions={},
            topological_order=[],
        )
        assert isinstance(hash(result), int)

    def test_repr_contains_info(self, high_belief: Opinion) -> None:
        result = InferenceResult(
            query_node="disease",
            opinion=high_belief,
            steps=[],
            intermediate_opinions={},
            topological_order=[],
        )
        r = repr(result)
        assert "disease" in r
        assert "steps=0" in r


# ═══════════════════════════════════════════════════════════════════
# CROSS-CUTTING: IMPORT AND MODULE STRUCTURE
# ═══════════════════════════════════════════════════════════════════


class TestModuleImports:
    """Verify that the sl_network package is importable and exports types."""

    def test_import_from_package(self) -> None:
        """Types are importable from the sl_network package."""
        from jsonld_ex.sl_network import (
            EdgeType,
            InferenceResult,
            InferenceStep,
            MultiParentEdge,
            SLEdge,
            SLNode,
        )
        # Verify they are the actual classes, not None
        assert SLNode is not None
        assert SLEdge is not None
        assert MultiParentEdge is not None
        assert InferenceStep is not None
        assert InferenceResult is not None

    def test_import_from_types_module(self) -> None:
        """Types are importable from the types submodule."""
        from jsonld_ex.sl_network.types import SLNode, SLEdge
        assert SLNode is not None
        assert SLEdge is not None

    def test_no_changes_to_existing_confidence_algebra(self) -> None:
        """SLNetwork types do NOT modify the existing confidence_algebra.

        This is the backward-compatibility guarantee: all existing
        operators remain unchanged.
        """
        from jsonld_ex.confidence_algebra import (
            Opinion,
            cumulative_fuse,
            deduce,
            trust_discount,
        )
        # Quick smoke test: existing operators still work
        op_a = Opinion(0.7, 0.2, 0.1)
        op_b = Opinion(0.6, 0.1, 0.3)
        fused = cumulative_fuse(op_a, op_b)
        assert abs(fused.belief + fused.disbelief + fused.uncertainty - 1.0) < 1e-9

        discounted = trust_discount(op_a, op_b)
        assert abs(
            discounted.belief + discounted.disbelief + discounted.uncertainty - 1.0
        ) < 1e-9

        deduced = deduce(op_a, op_b, op_b)
        assert abs(
            deduced.belief + deduced.disbelief + deduced.uncertainty - 1.0
        ) < 1e-9
