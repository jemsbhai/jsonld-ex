"""Tests for validation extensions."""

import pytest
from jsonld_ex.validation import validate_node, validate_document


PERSON_SHAPE = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
    "email": {"@pattern": r"^[^@]+@[^@]+$"},
    "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
}


class TestValidateNode:
    def test_valid_node(self):
        node = {"@type": "Person", "name": "John", "email": "j@x.com", "age": 30}
        result = validate_node(node, PERSON_SHAPE)
        assert result.valid

    def test_missing_required(self):
        node = {"@type": "Person", "email": "j@x.com"}
        result = validate_node(node, PERSON_SHAPE)
        assert not result.valid
        assert any(e.constraint == "required" for e in result.errors)

    def test_type_mismatch(self):
        node = {"@type": "Person", "name": 12345}
        result = validate_node(node, PERSON_SHAPE)
        assert not result.valid

    def test_below_minimum(self):
        node = {"@type": "Person", "name": "Test", "age": -5}
        result = validate_node(node, PERSON_SHAPE)
        assert any(e.constraint == "minimum" for e in result.errors)

    def test_above_maximum(self):
        node = {"@type": "Person", "name": "Test", "age": 200}
        result = validate_node(node, PERSON_SHAPE)
        assert any(e.constraint == "maximum" for e in result.errors)

    def test_pattern_mismatch(self):
        node = {"@type": "Person", "name": "Test", "email": "bad"}
        result = validate_node(node, PERSON_SHAPE)
        assert any(e.constraint == "pattern" for e in result.errors)

    def test_optional_absent(self):
        node = {"@type": "Person", "name": "Test"}
        result = validate_node(node, PERSON_SHAPE)
        assert result.valid


class TestValidateDocument:
    def test_validates_graph(self):
        doc = {
            "@graph": [
                {"@type": "Person", "name": "Alice"},
                {"@type": "Person"},  # missing name
            ]
        }
        result = validate_document(doc, [PERSON_SHAPE])
        assert not result.valid
        assert len(result.errors) == 1

    def test_empty_shapes(self):
        doc = {"@graph": [{"@type": "Person", "name": "Alice"}]}
        result = validate_document(doc, [])
        assert result.valid

    def test_nested_graph(self):
        doc = {"@graph": [{"@type": "Person", "name": "Alice"}]}
        result = validate_document(doc, [PERSON_SHAPE])
        assert result.valid


class TestValidationEdgeCases:
    def test_non_dict_node(self):
        result = validate_node("not a dict", PERSON_SHAPE)
        assert not result.valid

    def test_invalid_regex_pattern(self):
        shape = {"@type": "Thing", "code": {"@pattern": "[invalid"}}
        node = {"@type": "Thing", "code": "abc"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any("Invalid regex" in e.message for e in result.errors)

    def test_bool_not_integer(self):
        shape = {"@type": "Thing", "count": {"@type": "xsd:integer"}}
        node = {"@type": "Thing", "count": True}
        result = validate_node(node, shape)
        assert not result.valid

    def test_bool_not_double(self):
        shape = {"@type": "Thing", "score": {"@type": "xsd:double"}}
        node = {"@type": "Thing", "score": False}
        result = validate_node(node, shape)
        assert not result.valid

    def test_bool_skips_numeric_checks(self):
        """Boolean True (== 1) should not pass @minimum/@maximum."""
        shape = {"@type": "Thing", "val": {"@minimum": 0, "@maximum": 10}}
        node = {"@type": "Thing", "val": True}
        result = validate_node(node, shape)
        # Should be valid because bool is skipped by numeric checks
        # (no type constraint, and bool is not checked against min/max)
        assert result.valid

    def test_empty_dict_value_treated_as_absent(self):
        shape = {"@type": "Thing", "data": {"@required": True}}
        node = {"@type": "Thing", "data": {}}
        result = validate_node(node, shape)
        assert not result.valid

    def test_empty_list_value_treated_as_absent(self):
        shape = {"@type": "Thing", "items": {"@required": True}}
        node = {"@type": "Thing", "items": []}
        result = validate_node(node, shape)
        assert not result.valid

    def test_value_node_extraction(self):
        shape = {"@type": "Person", "name": {"@type": "xsd:string"}}
        node = {"@type": "Person", "name": {"@value": "Alice"}}
        result = validate_node(node, shape)
        assert result.valid

    def test_multiple_types(self):
        shape = {"@type": "Person", "name": {"@required": True}}
        node = {"@type": ["Person", "Agent"], "name": "Bob"}
        result = validate_node(node, shape)
        assert result.valid

    def test_wrong_type(self):
        node = {"@type": "Organization", "name": "Acme"}
        result = validate_node(node, PERSON_SHAPE)
        assert not result.valid
        assert any(e.constraint == "type" for e in result.errors)


class TestCardinalityConstraints:
    """GAP-V1: @minCount / @maxCount cardinality constraints."""

    def test_mincount_list_satisfies(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 2}}
        node = {"@type": "Thing", "tags": ["a", "b", "c"]}
        result = validate_node(node, shape)
        assert result.valid

    def test_mincount_list_too_few(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 2}}
        node = {"@type": "Thing", "tags": ["a"]}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "minCount" for e in result.errors)

    def test_mincount_single_value_counts_as_one(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 2}}
        node = {"@type": "Thing", "tags": "only-one"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "minCount" for e in result.errors)

    def test_mincount_absent_property(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 2}}
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "minCount" for e in result.errors)

    def test_mincount_one_with_single_value(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 1}}
        node = {"@type": "Thing", "tags": "present"}
        result = validate_node(node, shape)
        assert result.valid

    def test_maxcount_list_within_limit(self):
        shape = {"@type": "Thing", "tags": {"@maxCount": 3}}
        node = {"@type": "Thing", "tags": ["a", "b"]}
        result = validate_node(node, shape)
        assert result.valid

    def test_maxcount_list_exceeds(self):
        shape = {"@type": "Thing", "tags": {"@maxCount": 3}}
        node = {"@type": "Thing", "tags": ["a", "b", "c", "d", "e"]}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "maxCount" for e in result.errors)

    def test_maxcount_one_with_single_value(self):
        shape = {"@type": "Thing", "tags": {"@maxCount": 1}}
        node = {"@type": "Thing", "tags": "single"}
        result = validate_node(node, shape)
        assert result.valid

    def test_maxcount_one_with_list_of_two(self):
        shape = {"@type": "Thing", "tags": {"@maxCount": 1}}
        node = {"@type": "Thing", "tags": ["a", "b"]}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "maxCount" for e in result.errors)

    def test_combined_min_max_within_range(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 1, "@maxCount": 3}}
        node = {"@type": "Thing", "tags": ["a", "b"]}
        result = validate_node(node, shape)
        assert result.valid

    def test_exact_cardinality(self):
        shape = {"@type": "Thing", "authors": {"@minCount": 2, "@maxCount": 2}}
        node = {"@type": "Thing", "authors": ["Alice", "Bob"]}
        result = validate_node(node, shape)
        assert result.valid

    def test_mincount_zero_always_valid(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 0}}
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert result.valid

    def test_empty_list_with_mincount(self):
        shape = {"@type": "Thing", "tags": {"@minCount": 1}}
        node = {"@type": "Thing", "tags": []}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "minCount" for e in result.errors)


class TestEnumerationConstraint:
    """GAP-V2: @in enumeration constraint."""

    def test_value_in_allowed_set(self):
        shape = {"@type": "Article", "status": {"@in": ["draft", "published", "retracted"]}}
        node = {"@type": "Article", "status": "published"}
        result = validate_node(node, shape)
        assert result.valid

    def test_value_not_in_allowed_set(self):
        shape = {"@type": "Article", "status": {"@in": ["draft", "published", "retracted"]}}
        node = {"@type": "Article", "status": "archived"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "in" for e in result.errors)

    def test_in_with_integers(self):
        shape = {"@type": "Thing", "priority": {"@in": [1, 2, 3]}}
        node = {"@type": "Thing", "priority": 2}
        result = validate_node(node, shape)
        assert result.valid

    def test_in_integer_mismatch(self):
        shape = {"@type": "Thing", "priority": {"@in": [1, 2, 3]}}
        node = {"@type": "Thing", "priority": 5}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "in" for e in result.errors)

    def test_in_with_mixed_types(self):
        shape = {"@type": "Thing", "code": {"@in": ["auto", 0, 1]}}
        node = {"@type": "Thing", "code": "auto"}
        result = validate_node(node, shape)
        assert result.valid

    def test_in_with_value_node(self):
        shape = {"@type": "Article", "status": {"@in": ["draft", "published"]}}
        node = {"@type": "Article", "status": {"@value": "draft"}}
        result = validate_node(node, shape)
        assert result.valid

    def test_in_absent_optional_property(self):
        shape = {"@type": "Article", "status": {"@in": ["draft", "published"]}}
        node = {"@type": "Article"}
        result = validate_node(node, shape)
        assert result.valid  # absent optional property is fine

    def test_in_combined_with_required(self):
        shape = {"@type": "Article", "status": {"@required": True, "@in": ["draft", "published"]}}
        node = {"@type": "Article"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "required" for e in result.errors)

    def test_in_with_booleans(self):
        shape = {"@type": "Thing", "active": {"@in": [True, False]}}
        node = {"@type": "Thing", "active": True}
        result = validate_node(node, shape)
        assert result.valid

    def test_in_empty_set_always_invalid(self):
        shape = {"@type": "Thing", "val": {"@in": []}}
        node = {"@type": "Thing", "val": "anything"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "in" for e in result.errors)


# ── GAP-V3: Logical Combinators (@or, @and, @not) ─────────────────


class TestOrCombinator:
    """@or: value passes if ANY branch is satisfied."""

    def test_or_first_branch_matches(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@or": [
                    {"@type": "xsd:string", "@minLength": 1},
                    {"@type": "xsd:integer", "@minimum": 0},
                ]
            },
        }
        node = {"@type": "Thing", "val": "hello"}
        result = validate_node(node, shape)
        assert result.valid

    def test_or_second_branch_matches(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@or": [
                    {"@type": "xsd:string", "@minLength": 1},
                    {"@type": "xsd:integer", "@minimum": 0},
                ]
            },
        }
        node = {"@type": "Thing", "val": 42}
        result = validate_node(node, shape)
        assert result.valid

    def test_or_no_branch_matches(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@or": [
                    {"@type": "xsd:string", "@minLength": 5},
                    {"@type": "xsd:integer", "@minimum": 100},
                ]
            },
        }
        node = {"@type": "Thing", "val": 3}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "or" for e in result.errors)

    def test_or_with_pattern_branch(self):
        shape = {
            "@type": "Thing",
            "email": {
                "@or": [
                    {"@pattern": r"^[^@]+@[^@]+$"},
                    {"@in": ["N/A", "unknown"]},
                ]
            },
        }
        node = {"@type": "Thing", "email": "N/A"}
        result = validate_node(node, shape)
        assert result.valid

    def test_or_absent_value_no_required(self):
        """@or on absent optional property should pass."""
        shape = {
            "@type": "Thing",
            "val": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ]
            },
        }
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert result.valid

    def test_or_with_required(self):
        """@required should still be checked even with @or."""
        shape = {
            "@type": "Thing",
            "val": {
                "@required": True,
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ]
            },
        }
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "required" for e in result.errors)


class TestAndCombinator:
    """@and: value passes only if ALL branches are satisfied."""

    def test_and_all_pass(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@minLength": 3},
                    {"@pattern": r"^[A-Z]"},
                ]
            },
        }
        node = {"@type": "Thing", "val": "Hello"}
        result = validate_node(node, shape)
        assert result.valid

    def test_and_one_fails(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@minLength": 3},
                    {"@pattern": r"^[A-Z]"},
                ]
            },
        }
        node = {"@type": "Thing", "val": "hello"}  # fails pattern
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "and" for e in result.errors)

    def test_and_absent_optional(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@minLength": 1},
                ]
            },
        }
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert result.valid


class TestNotCombinator:
    """@not: value passes if inner constraints FAIL."""

    def test_not_inverts_failure_to_pass(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@not": {"@in": ["banned", "forbidden"]}
            },
        }
        node = {"@type": "Thing", "val": "allowed"}
        result = validate_node(node, shape)
        assert result.valid

    def test_not_inverts_pass_to_failure(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@not": {"@in": ["banned", "forbidden"]}
            },
        }
        node = {"@type": "Thing", "val": "banned"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "not" for e in result.errors)

    def test_not_with_pattern(self):
        """Reject values matching a pattern."""
        shape = {
            "@type": "Thing",
            "code": {
                "@not": {"@pattern": r"^DEPRECATED_"}
            },
        }
        node = {"@type": "Thing", "code": "ACTIVE_001"}
        result = validate_node(node, shape)
        assert result.valid

    def test_not_absent_optional(self):
        shape = {
            "@type": "Thing",
            "val": {
                "@not": {"@type": "xsd:integer"}
            },
        }
        node = {"@type": "Thing"}
        result = validate_node(node, shape)
        assert result.valid


class TestNestedCombinators:
    """Combinators can nest: @or inside @and, @not inside @or, etc."""

    def test_or_inside_and(self):
        """Value must be string AND (email OR 'N/A')."""
        shape = {
            "@type": "Thing",
            "contact": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@or": [
                        {"@pattern": r"^[^@]+@[^@]+$"},
                        {"@in": ["N/A"]},
                    ]},
                ]
            },
        }
        # passes: string + matches @in
        r1 = validate_node({"@type": "Thing", "contact": "N/A"}, shape)
        assert r1.valid

        # passes: string + matches pattern
        r2 = validate_node({"@type": "Thing", "contact": "a@b.c"}, shape)
        assert r2.valid

        # fails: string but neither pattern nor @in
        r3 = validate_node({"@type": "Thing", "contact": "garbage"}, shape)
        assert not r3.valid

    def test_not_inside_or(self):
        """Value is either a positive int OR not a string."""
        shape = {
            "@type": "Thing",
            "val": {
                "@or": [
                    {"@type": "xsd:integer", "@minimum": 1},
                    {"@not": {"@type": "xsd:string"}},
                ]
            },
        }
        # passes: positive int matches first branch
        r1 = validate_node({"@type": "Thing", "val": 5}, shape)
        assert r1.valid

        # fails: string fails both branches (not int, and IS string)
        r2 = validate_node({"@type": "Thing", "val": "hello"}, shape)
        assert not r2.valid


# ── GAP-V4: Cross-Property Constraints ─────────────────────────────


class TestLessThan:
    """@lessThan: this property's value must be < referenced property's value."""

    def test_less_than_passes(self):
        shape = {
            "@type": "Event",
            "startDate": {"@lessThan": "endDate"},
        }
        node = {"@type": "Event", "startDate": "2025-01-01", "endDate": "2025-12-31"}
        result = validate_node(node, shape)
        assert result.valid

    def test_less_than_fails_equal(self):
        shape = {
            "@type": "Event",
            "startDate": {"@lessThan": "endDate"},
        }
        node = {"@type": "Event", "startDate": "2025-06-15", "endDate": "2025-06-15"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "lessThan" for e in result.errors)

    def test_less_than_fails_greater(self):
        shape = {
            "@type": "Event",
            "startDate": {"@lessThan": "endDate"},
        }
        node = {"@type": "Event", "startDate": "2025-12-31", "endDate": "2025-01-01"}
        result = validate_node(node, shape)
        assert not result.valid

    def test_less_than_numeric(self):
        shape = {
            "@type": "Range",
            "low": {"@lessThan": "high"},
        }
        node = {"@type": "Range", "low": 10, "high": 20}
        result = validate_node(node, shape)
        assert result.valid

    def test_less_than_missing_other_prop_skips(self):
        """If referenced property is absent, skip the check."""
        shape = {
            "@type": "Event",
            "startDate": {"@lessThan": "endDate"},
        }
        node = {"@type": "Event", "startDate": "2025-01-01"}
        result = validate_node(node, shape)
        assert result.valid

    def test_less_than_this_absent_skips(self):
        """If this property is absent (optional), skip."""
        shape = {
            "@type": "Event",
            "startDate": {"@lessThan": "endDate"},
        }
        node = {"@type": "Event", "endDate": "2025-12-31"}
        result = validate_node(node, shape)
        assert result.valid


class TestLessThanOrEquals:
    """@lessThanOrEquals: value must be <= referenced property."""

    def test_lte_equal_passes(self):
        shape = {
            "@type": "Range",
            "low": {"@lessThanOrEquals": "high"},
        }
        node = {"@type": "Range", "low": 10, "high": 10}
        result = validate_node(node, shape)
        assert result.valid

    def test_lte_less_passes(self):
        shape = {
            "@type": "Range",
            "low": {"@lessThanOrEquals": "high"},
        }
        node = {"@type": "Range", "low": 5, "high": 10}
        result = validate_node(node, shape)
        assert result.valid

    def test_lte_greater_fails(self):
        shape = {
            "@type": "Range",
            "low": {"@lessThanOrEquals": "high"},
        }
        node = {"@type": "Range", "low": 15, "high": 10}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "lessThanOrEquals" for e in result.errors)


class TestEquals:
    """@equals: both properties must have the same value."""

    def test_equals_passes(self):
        shape = {
            "@type": "Account",
            "email": {"@equals": "confirmEmail"},
        }
        node = {"@type": "Account", "email": "a@b.c", "confirmEmail": "a@b.c"}
        result = validate_node(node, shape)
        assert result.valid

    def test_equals_fails(self):
        shape = {
            "@type": "Account",
            "email": {"@equals": "confirmEmail"},
        }
        node = {"@type": "Account", "email": "a@b.c", "confirmEmail": "x@y.z"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "equals" for e in result.errors)

    def test_equals_with_value_nodes(self):
        """@equals works with @value-wrapped nodes."""
        shape = {
            "@type": "Account",
            "email": {"@equals": "confirmEmail"},
        }
        node = {
            "@type": "Account",
            "email": {"@value": "a@b.c"},
            "confirmEmail": {"@value": "a@b.c"},
        }
        result = validate_node(node, shape)
        assert result.valid


class TestDisjoint:
    """@disjoint: property values must NOT be equal."""

    def test_disjoint_passes(self):
        shape = {
            "@type": "Transfer",
            "source": {"@disjoint": "destination"},
        }
        node = {"@type": "Transfer", "source": "acct-1", "destination": "acct-2"}
        result = validate_node(node, shape)
        assert result.valid

    def test_disjoint_fails(self):
        shape = {
            "@type": "Transfer",
            "source": {"@disjoint": "destination"},
        }
        node = {"@type": "Transfer", "source": "acct-1", "destination": "acct-1"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "disjoint" for e in result.errors)

    def test_disjoint_missing_other_skips(self):
        shape = {
            "@type": "Transfer",
            "source": {"@disjoint": "destination"},
        }
        node = {"@type": "Transfer", "source": "acct-1"}
        result = validate_node(node, shape)
        assert result.valid


class TestCrossPropertyWithOtherConstraints:
    """Cross-property constraints compose with atomic constraints."""

    def test_less_than_plus_type(self):
        shape = {
            "@type": "Range",
            "low": {
                "@type": "xsd:integer",
                "@minimum": 0,
                "@lessThan": "high",
            },
        }
        node = {"@type": "Range", "low": 5, "high": 10}
        result = validate_node(node, shape)
        assert result.valid

    def test_less_than_plus_type_type_fails(self):
        shape = {
            "@type": "Range",
            "low": {
                "@type": "xsd:integer",
                "@lessThan": "high",
            },
        }
        node = {"@type": "Range", "low": "not a number", "high": 10}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "type" for e in result.errors)
