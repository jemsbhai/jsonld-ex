"""Tests for new validation constraints: @class, @qualifiedShape, @uniqueLang.

GAP-V8:  @class             -- instance-of check (maps to sh:class)
GAP-V9:  @qualifiedShape    -- qualified cardinality (maps to sh:qualifiedValueShape)
         @qualifiedMinCount -- minimum matching items
         @qualifiedMaxCount -- maximum matching items
GAP-V10: @uniqueLang        -- unique language tags (maps to sh:uniqueLang)

These tests are written RED-first: they MUST FAIL before implementation.
"""

import pytest
from jsonld_ex.validation import validate_node, validate_document


# ============================================================================
# GAP-V8: @class -- Instance-of Check
# ============================================================================
# @class constrains a property's value to be a node whose @type includes
# the specified class IRI. This is the jsonld-ex equivalent of sh:class.
#
# Unlike @type on the shape (which checks the *current node*'s type),
# @class checks the *property value*'s type.
# ============================================================================


class TestClassConstraintBasic:
    """@class: basic happy path and failure cases."""

    def test_class_matches_single_type(self):
        """Property value is a node with matching @type."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": "Person", "name": "Alice"},
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_class_mismatch(self):
        """Property value has wrong @type."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": "Organization", "name": "Acme"},
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_matches_in_type_list(self):
        """Property value has multiple @type values, one matches."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": ["Person", "Agent"], "name": "Alice"},
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_class_no_type_on_value(self):
        """Property value is a dict without @type -- fails."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"name": "Alice"},
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_value_is_scalar(self):
        """Property value is a string, not a node -- fails."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": "Alice",
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_value_is_value_node(self):
        """Property value is a @value node (not a typed node) -- fails."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@value": "Alice"},
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)


class TestClassConstraintOptional:
    """@class with absent/optional properties."""

    def test_class_absent_optional(self):
        """Absent optional property with @class -- valid (skip)."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {"@type": "Article"}
        result = validate_node(node, shape)
        assert result.valid

    def test_class_with_required_absent(self):
        """@class + @required, property absent -- fails on required."""
        shape = {
            "@type": "Article",
            "author": {"@required": True, "@class": "Person"},
        }
        node = {"@type": "Article"}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "required" for e in result.errors)


class TestClassConstraintComposition:
    """@class composed with other constraints."""

    def test_class_with_shape(self):
        """@class + @shape: both must pass."""
        shape = {
            "@type": "Article",
            "author": {
                "@class": "Person",
                "@shape": {
                    "@type": "Person",
                    "name": {"@required": True},
                },
            },
        }
        # Right type AND has required name
        node_ok = {
            "@type": "Article",
            "author": {"@type": "Person", "name": "Alice"},
        }
        assert validate_node(node_ok, shape).valid

        # Right type but missing name
        node_no_name = {
            "@type": "Article",
            "author": {"@type": "Person"},
        }
        result = validate_node(node_no_name, shape)
        assert not result.valid

        # Wrong type
        node_wrong_type = {
            "@type": "Article",
            "author": {"@type": "Organization", "name": "Acme"},
        }
        result2 = validate_node(node_wrong_type, shape)
        assert not result2.valid
        assert any(e.constraint == "class" for e in result2.errors)

    def test_class_with_severity_warning(self):
        """@class with @severity=warning routes to warnings, not errors."""
        shape = {
            "@type": "Article",
            "author": {
                "@class": "Person",
                "@severity": "warning",
            },
        }
        node = {
            "@type": "Article",
            "author": {"@type": "Organization", "name": "Acme"},
        }
        result = validate_node(node, shape)
        assert result.valid  # warning, not error
        assert len(result.warnings) >= 1


class TestClassConstraintEdgeCases:
    """@class edge cases and boundaries."""

    def test_class_on_list_first_item(self):
        """Property is a list -- @class checks the first item."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": [{"@type": "Person", "name": "Alice"}],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_class_on_list_first_item_wrong_type(self):
        """Property is a list, first item has wrong type."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": [{"@type": "Organization", "name": "Acme"}],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_empty_type_list(self):
        """Property value has empty @type list -- fails."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": [], "name": "Nobody"},
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_with_iri(self):
        """@class with full IRI (not just local name)."""
        shape = {
            "@type": "Article",
            "author": {"@class": "http://schema.org/Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": "http://schema.org/Person", "name": "Alice"},
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_class_null_value(self):
        """Property value is None (absent) -- skip."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {"@type": "Article", "author": None}
        # Depending on implementation: None treated as absent
        result = validate_node(node, shape)
        # None without @required should pass (absent optional)
        assert result.valid

    def test_class_empty_list(self):
        """Property value is empty list -- treated as absent."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {"@type": "Article", "author": []}
        result = validate_node(node, shape)
        assert result.valid  # empty list = absent

    def test_class_integer_value(self):
        """Property value is integer -- fails (not a node)."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {"@type": "Article", "author": 42}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_boolean_value(self):
        """Property value is boolean -- fails (not a node)."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {"@type": "Article", "author": True}
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "class" for e in result.errors)

    def test_class_error_message_includes_expected_class(self):
        """Error message should mention the expected class."""
        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        node = {
            "@type": "Article",
            "author": {"@type": "Organization"},
        }
        result = validate_node(node, shape)
        assert not result.valid
        class_errors = [e for e in result.errors if e.constraint == "class"]
        assert len(class_errors) == 1
        assert "Person" in class_errors[0].message


# ============================================================================
# GAP-V9: @qualifiedShape + @qualifiedMinCount / @qualifiedMaxCount
# ============================================================================
# @qualifiedShape defines a shape that list items are tested against.
# @qualifiedMinCount: at least N items in the list must conform.
# @qualifiedMaxCount: at most N items in the list may conform.
#
# This is the jsonld-ex equivalent of sh:qualifiedValueShape +
# sh:qualifiedMinCount / sh:qualifiedMaxCount.
#
# ML use case: "At least 2 annotations must have confidence > 0.9"
# ============================================================================


class TestQualifiedShapeBasic:
    """@qualifiedShape: basic happy path and failure cases."""

    def test_qualified_min_count_satisfied(self):
        """List has enough items matching the qualified shape."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 2,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "confidence": 0.95},
                {"@type": "Annotation", "confidence": 0.92},
                {"@type": "Annotation", "confidence": 0.5},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_min_count_not_satisfied(self):
        """List doesn't have enough items matching the qualified shape."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 2,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "confidence": 0.95},
                {"@type": "Annotation", "confidence": 0.5},
                {"@type": "Annotation", "confidence": 0.3},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "qualifiedMinCount" for e in result.errors)

    def test_qualified_max_count_satisfied(self):
        """No more than N items match the shape."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "status": {"@in": ["rejected"]},
                },
                "@qualifiedMaxCount": 1,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "status": "approved"},
                {"@type": "Annotation", "status": "rejected"},
                {"@type": "Annotation", "status": "approved"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_max_count_exceeded(self):
        """Too many items match the shape."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "status": {"@in": ["rejected"]},
                },
                "@qualifiedMaxCount": 1,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "status": "rejected"},
                {"@type": "Annotation", "status": "rejected"},
                {"@type": "Annotation", "status": "approved"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "qualifiedMaxCount" for e in result.errors)


class TestQualifiedShapeCombined:
    """@qualifiedMinCount + @qualifiedMaxCount together (range)."""

    def test_qualified_range_within(self):
        """Count of matching items is within [min, max]."""
        shape = {
            "@type": "Team",
            "members": {
                "@qualifiedShape": {
                    "@type": "Person",
                    "role": {"@in": ["lead"]},
                },
                "@qualifiedMinCount": 1,
                "@qualifiedMaxCount": 2,
            },
        }
        node = {
            "@type": "Team",
            "members": [
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "member"},
                {"@type": "Person", "role": "member"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_range_below_min(self):
        """Count of matching items is below minimum."""
        shape = {
            "@type": "Team",
            "members": {
                "@qualifiedShape": {
                    "@type": "Person",
                    "role": {"@in": ["lead"]},
                },
                "@qualifiedMinCount": 1,
                "@qualifiedMaxCount": 2,
            },
        }
        node = {
            "@type": "Team",
            "members": [
                {"@type": "Person", "role": "member"},
                {"@type": "Person", "role": "member"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "qualifiedMinCount" for e in result.errors)

    def test_qualified_range_above_max(self):
        """Count of matching items exceeds maximum."""
        shape = {
            "@type": "Team",
            "members": {
                "@qualifiedShape": {
                    "@type": "Person",
                    "role": {"@in": ["lead"]},
                },
                "@qualifiedMinCount": 1,
                "@qualifiedMaxCount": 2,
            },
        }
        node = {
            "@type": "Team",
            "members": [
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "lead"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "qualifiedMaxCount" for e in result.errors)

    def test_qualified_exact_count(self):
        """Exactly N items must match (min == max)."""
        shape = {
            "@type": "Team",
            "members": {
                "@qualifiedShape": {
                    "@type": "Person",
                    "role": {"@in": ["lead"]},
                },
                "@qualifiedMinCount": 2,
                "@qualifiedMaxCount": 2,
            },
        }
        # Exactly 2 leads
        node_ok = {
            "@type": "Team",
            "members": [
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "member"},
            ],
        }
        assert validate_node(node_ok, shape).valid

        # Only 1 lead
        node_low = {
            "@type": "Team",
            "members": [
                {"@type": "Person", "role": "lead"},
                {"@type": "Person", "role": "member"},
            ],
        }
        assert not validate_node(node_low, shape).valid


class TestQualifiedShapeEdgeCases:
    """@qualifiedShape edge cases and boundary conditions."""

    def test_qualified_absent_optional(self):
        """Absent optional property -- skip."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 1,
            },
        }
        node = {"@type": "Dataset"}
        result = validate_node(node, shape)
        # Absent property has 0 matching items -> fails qualifiedMinCount
        assert not result.valid
        assert any(e.constraint == "qualifiedMinCount" for e in result.errors)

    def test_qualified_empty_list(self):
        """Empty list -- 0 matching items."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 1,
            },
        }
        node = {"@type": "Dataset", "annotations": []}
        result = validate_node(node, shape)
        assert not result.valid

    def test_qualified_single_value_not_list(self):
        """Single (non-list) value: test it as a 1-element list."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 1,
            },
        }
        # Single matching item (not in a list)
        node_match = {
            "@type": "Dataset",
            "annotations": {"@type": "Annotation", "confidence": 0.95},
        }
        assert validate_node(node_match, shape).valid

        # Single non-matching item
        node_no_match = {
            "@type": "Dataset",
            "annotations": {"@type": "Annotation", "confidence": 0.3},
        }
        assert not validate_node(node_no_match, shape).valid

    def test_qualified_min_count_zero(self):
        """@qualifiedMinCount=0 always passes (vacuously true)."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 0,
            },
        }
        node = {"@type": "Dataset", "annotations": []}
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_max_count_zero(self):
        """@qualifiedMaxCount=0: NO items may match."""
        shape = {
            "@type": "Dataset",
            "items": {
                "@qualifiedShape": {
                    "@type": "Item",
                    "status": {"@in": ["deleted"]},
                },
                "@qualifiedMaxCount": 0,
            },
        }
        # No deleted items -- valid
        node_ok = {
            "@type": "Dataset",
            "items": [
                {"@type": "Item", "status": "active"},
                {"@type": "Item", "status": "active"},
            ],
        }
        assert validate_node(node_ok, shape).valid

        # One deleted item -- invalid
        node_bad = {
            "@type": "Dataset",
            "items": [
                {"@type": "Item", "status": "active"},
                {"@type": "Item", "status": "deleted"},
            ],
        }
        assert not validate_node(node_bad, shape).valid

    def test_qualified_all_items_match(self):
        """All items match -- count equals list length."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 3,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation"},
                {"@type": "Annotation"},
                {"@type": "Annotation"},
            ],
        }
        assert validate_node(node, shape).valid

    def test_qualified_no_items_match(self):
        """No items match -- count is 0."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 1,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "confidence": 0.1},
                {"@type": "Annotation", "confidence": 0.2},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid

    def test_qualified_shape_without_counts_is_noop(self):
        """@qualifiedShape alone (no min/max count) -- no constraint enforced."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [{"@type": "Other"}],
        }
        result = validate_node(node, shape)
        assert result.valid  # no count constraint to enforce

    def test_qualified_items_are_scalars(self):
        """List items that are scalars (not dicts) never match a shape."""
        shape = {
            "@type": "Dataset",
            "tags": {
                "@qualifiedShape": {"@type": "Tag"},
                "@qualifiedMinCount": 1,
            },
        }
        node = {
            "@type": "Dataset",
            "tags": ["foo", "bar", "baz"],
        }
        result = validate_node(node, shape)
        assert not result.valid

    def test_qualified_mixed_nodes_and_scalars(self):
        """List with mix of nodes and scalars."""
        shape = {
            "@type": "Dataset",
            "items": {
                "@qualifiedShape": {
                    "@type": "Item",
                    "score": {"@minimum": 0.5},
                },
                "@qualifiedMinCount": 2,
            },
        }
        node = {
            "@type": "Dataset",
            "items": [
                {"@type": "Item", "score": 0.9},   # matches
                "just a string",                     # doesn't match
                {"@type": "Item", "score": 0.8},   # matches
                42,                                  # doesn't match
            ],
        }
        result = validate_node(node, shape)
        assert result.valid  # 2 items match >= qualifiedMinCount 2


class TestQualifiedShapeSeverity:
    """@qualifiedShape with @severity."""

    def test_qualified_severity_warning(self):
        """Qualified constraint with severity=warning routes to warnings."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 2,
                "@severity": "warning",
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation", "confidence": 0.5},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid  # warning, not error
        assert len(result.warnings) >= 1


class TestQualifiedShapeComplex:
    """@qualifiedShape with complex inner shapes."""

    def test_qualified_with_nested_shape(self):
        """Inner qualified shape has nested constraints."""
        shape = {
            "@type": "Experiment",
            "results": {
                "@qualifiedShape": {
                    "@type": "Result",
                    "metric": {"@required": True},
                    "value": {"@minimum": 0.0, "@maximum": 1.0},
                },
                "@qualifiedMinCount": 1,
            },
        }
        # One result matches fully
        node = {
            "@type": "Experiment",
            "results": [
                {"@type": "Result", "metric": "accuracy", "value": 0.95},
                {"@type": "Result", "value": 0.5},  # missing metric
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_with_pattern_constraint(self):
        """Inner shape uses pattern matching."""
        shape = {
            "@type": "Dataset",
            "files": {
                "@qualifiedShape": {
                    "@type": "File",
                    "format": {"@pattern": r"\.csv$"},
                },
                "@qualifiedMinCount": 1,
            },
        }
        node = {
            "@type": "Dataset",
            "files": [
                {"@type": "File", "format": "data.json"},
                {"@type": "File", "format": "results.csv"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_qualified_error_message_includes_counts(self):
        """Error message should report actual vs required count."""
        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 3,
            },
        }
        node = {
            "@type": "Dataset",
            "annotations": [
                {"@type": "Annotation"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        qmc_errors = [e for e in result.errors if e.constraint == "qualifiedMinCount"]
        assert len(qmc_errors) == 1
        # Error should mention the counts
        assert "1" in qmc_errors[0].message  # actual
        assert "3" in qmc_errors[0].message  # required


# ============================================================================
# GAP-V10: @uniqueLang -- Unique Language Tags
# ============================================================================
# @uniqueLang constrains a property so that no two values share the same
# @language tag. This is the jsonld-ex equivalent of sh:uniqueLang.
#
# Per RDF semantics, language tags are case-insensitive ("en" == "EN").
# ============================================================================


class TestUniqueLangBasic:
    """@uniqueLang: basic happy path and failure cases."""

    def test_unique_lang_all_different(self):
        """All @language tags are different -- valid."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telefon", "@language": "de"},
                {"@value": "Telefono", "@language": "es"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_duplicate(self):
        """Duplicate @language tag -- invalid."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "en"},
                {"@value": "Telefon", "@language": "de"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "uniqueLang" for e in result.errors)

    def test_unique_lang_case_insensitive(self):
        """Language tags are case-insensitive per RDF: 'en' == 'EN'."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "EN"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        assert any(e.constraint == "uniqueLang" for e in result.errors)

    def test_unique_lang_subtags(self):
        """Language subtags: 'en-US' and 'en-GB' are different."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Color", "@language": "en-US"},
                {"@value": "Colour", "@language": "en-GB"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_subtags_duplicate(self):
        """Duplicate subtags: 'en-US' and 'en-us' are same."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Color", "@language": "en-US"},
                {"@value": "Colour", "@language": "en-us"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid


class TestUniqueLangEdgeCases:
    """@uniqueLang edge cases and boundaries."""

    def test_unique_lang_single_value(self):
        """Single value (not list) -- always unique."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": {"@value": "Phone", "@language": "en"},
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_single_value_no_lang(self):
        """Single value without @language -- valid (nothing to duplicate)."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": {"@value": "Phone"},
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_absent_property(self):
        """Absent property -- skip."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {"@type": "Product"}
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_empty_list(self):
        """Empty list -- valid (no duplicates possible)."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {"@type": "Product", "name": []}
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_no_language_tags(self):
        """List of values without any @language tags -- valid."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone"},
                {"@value": "Telephone"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_mixed_tagged_untagged(self):
        """Mix of tagged and untagged values -- only tagged checked."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Generic phone"},  # plain string in list
                {"@value": "Telefon", "@language": "de"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_false_skips_check(self):
        """@uniqueLang: false -- no constraint enforced."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": False},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "en"},  # duplicate
            ],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_plain_strings_in_list(self):
        """List of plain strings (not @value nodes) -- valid."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": ["Phone", "Telephone"],
        }
        result = validate_node(node, shape)
        assert result.valid

    def test_unique_lang_three_way_duplicate(self):
        """Three values with same language -- fails."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "A", "@language": "en"},
                {"@value": "B", "@language": "en"},
                {"@value": "C", "@language": "en"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid

    def test_unique_lang_scalar_string(self):
        """Property is plain scalar string, not a list -- valid."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {"@type": "Product", "name": "Phone"}
        result = validate_node(node, shape)
        assert result.valid


class TestUniqueLangSeverity:
    """@uniqueLang with severity levels."""

    def test_unique_lang_severity_warning(self):
        """@uniqueLang with severity=warning routes to warnings."""
        shape = {
            "@type": "Product",
            "name": {
                "@uniqueLang": True,
                "@severity": "warning",
            },
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "en"},
            ],
        }
        result = validate_node(node, shape)
        assert result.valid  # warning, not error
        assert len(result.warnings) >= 1


class TestUniqueLangComposition:
    """@uniqueLang combined with other constraints."""

    def test_unique_lang_with_min_count(self):
        """@uniqueLang + @minCount: both enforced."""
        shape = {
            "@type": "Product",
            "name": {
                "@uniqueLang": True,
                "@minCount": 2,
            },
        }
        # 2 items, unique languages -- valid
        node_ok = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telefon", "@language": "de"},
            ],
        }
        assert validate_node(node_ok, shape).valid

        # 2 items, same language -- fails uniqueLang
        node_dup = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "en"},
            ],
        }
        result = validate_node(node_dup, shape)
        assert not result.valid
        assert any(e.constraint == "uniqueLang" for e in result.errors)

        # 1 item -- fails minCount
        node_few = {
            "@type": "Product",
            "name": [{"@value": "Phone", "@language": "en"}],
        }
        result2 = validate_node(node_few, shape)
        assert not result2.valid
        assert any(e.constraint == "minCount" for e in result2.errors)

    def test_unique_lang_error_message_includes_language(self):
        """Error message should identify the duplicate language tag."""
        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        node = {
            "@type": "Product",
            "name": [
                {"@value": "Phone", "@language": "en"},
                {"@value": "Telephone", "@language": "en"},
            ],
        }
        result = validate_node(node, shape)
        assert not result.valid
        ul_errors = [e for e in result.errors if e.constraint == "uniqueLang"]
        assert len(ul_errors) == 1
        assert "en" in ul_errors[0].message.lower()


# ============================================================================
# SHACL Round-Trip Tests for New Constraints
# ============================================================================
# Verify that shape_to_shacl() and shacl_to_shape() handle the new
# constraint types correctly.
# ============================================================================


class TestClassShaclRoundTrip:
    """@class <-> sh:class round-trip."""

    def test_class_to_shacl(self):
        """shape_to_shacl() maps @class to sh:class."""
        from jsonld_ex.owl_interop import shape_to_shacl, SHACL

        shape = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        shacl = shape_to_shacl(shape)
        graph = shacl["@graph"]
        assert len(graph) >= 1
        shape_node = graph[0]
        props = shape_node.get(f"{SHACL}property", [])
        assert len(props) >= 1
        author_prop = props[0]
        assert f"{SHACL}class" in author_prop or "sh:class" in author_prop

    def test_class_from_shacl(self):
        """shacl_to_shape() maps sh:class back to @class."""
        from jsonld_ex.owl_interop import shape_to_shacl, shacl_to_shape

        original = {
            "@type": "Article",
            "author": {"@class": "Person"},
        }
        shacl = shape_to_shacl(original)
        recovered, warnings = shacl_to_shape(shacl)
        assert recovered.get("author", {}).get("@class") == "Person"

    def test_class_full_round_trip(self):
        """jsonld-ex -> SHACL -> jsonld-ex: @class preserved."""
        from jsonld_ex.owl_interop import shape_to_shacl, shacl_to_shape

        original = {
            "@type": "Article",
            "author": {"@class": "Person"},
            "reviewer": {"@class": "http://schema.org/Person"},
        }
        shacl = shape_to_shacl(original)
        recovered, warnings = shacl_to_shape(shacl)
        assert recovered["author"]["@class"] == "Person"
        assert recovered["reviewer"]["@class"] == "http://schema.org/Person"


class TestQualifiedShapeShaclRoundTrip:
    """@qualifiedShape <-> sh:qualifiedValueShape round-trip."""

    def test_qualified_to_shacl(self):
        """shape_to_shacl() maps @qualifiedShape to sh:qualifiedValueShape."""
        from jsonld_ex.owl_interop import shape_to_shacl, SHACL

        shape = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {"@type": "Annotation"},
                "@qualifiedMinCount": 2,
            },
        }
        shacl = shape_to_shacl(shape)
        graph = shacl["@graph"]
        shape_node = graph[0]
        props = shape_node.get(f"{SHACL}property", [])
        assert len(props) >= 1
        ann_prop = props[0]
        assert (
            f"{SHACL}qualifiedValueShape" in ann_prop
            or "sh:qualifiedValueShape" in ann_prop
        )
        assert (
            f"{SHACL}qualifiedMinCount" in ann_prop
            or "sh:qualifiedMinCount" in ann_prop
        )

    def test_qualified_from_shacl(self):
        """shacl_to_shape() maps sh:qualifiedValueShape back."""
        from jsonld_ex.owl_interop import shape_to_shacl, shacl_to_shape

        original = {
            "@type": "Dataset",
            "annotations": {
                "@qualifiedShape": {
                    "@type": "Annotation",
                    "confidence": {"@minimum": 0.9},
                },
                "@qualifiedMinCount": 2,
                "@qualifiedMaxCount": 5,
            },
        }
        shacl = shape_to_shacl(original)
        recovered, warnings = shacl_to_shape(shacl)
        ann = recovered.get("annotations", {})
        assert ann.get("@qualifiedMinCount") == 2
        assert ann.get("@qualifiedMaxCount") == 5
        assert "@qualifiedShape" in ann


class TestUniqueLangShaclRoundTrip:
    """@uniqueLang <-> sh:uniqueLang round-trip."""

    def test_unique_lang_to_shacl(self):
        """shape_to_shacl() maps @uniqueLang to sh:uniqueLang."""
        from jsonld_ex.owl_interop import shape_to_shacl, SHACL

        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        shacl = shape_to_shacl(shape)
        graph = shacl["@graph"]
        shape_node = graph[0]
        props = shape_node.get(f"{SHACL}property", [])
        assert len(props) >= 1
        name_prop = props[0]
        assert (
            name_prop.get(f"{SHACL}uniqueLang") is True
            or name_prop.get("sh:uniqueLang") is True
        )

    def test_unique_lang_from_shacl(self):
        """shacl_to_shape() maps sh:uniqueLang back to @uniqueLang."""
        from jsonld_ex.owl_interop import shape_to_shacl, shacl_to_shape

        original = {
            "@type": "Product",
            "name": {"@uniqueLang": True},
        }
        shacl = shape_to_shacl(original)
        recovered, warnings = shacl_to_shape(shacl)
        assert recovered.get("name", {}).get("@uniqueLang") is True

    def test_unique_lang_false_not_emitted(self):
        """@uniqueLang: false should not produce sh:uniqueLang in SHACL."""
        from jsonld_ex.owl_interop import shape_to_shacl, SHACL

        shape = {
            "@type": "Product",
            "name": {"@uniqueLang": False},
        }
        shacl = shape_to_shacl(shape)
        graph = shacl["@graph"]
        shape_node = graph[0]
        props = shape_node.get(f"{SHACL}property", [])
        # Either no properties or uniqueLang not set
        if props:
            name_prop = props[0]
            assert name_prop.get(f"{SHACL}uniqueLang") is not True
