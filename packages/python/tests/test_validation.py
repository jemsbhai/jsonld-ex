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
