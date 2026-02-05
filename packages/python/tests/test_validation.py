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
