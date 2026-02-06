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
