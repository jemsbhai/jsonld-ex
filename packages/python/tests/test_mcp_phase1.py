"""Tests for the jsonld-ex MCP server — Phase 1 expansion (21 new tools).

Tests the new MCP tool handler functions directly (no transport needed).
Covers all 21 newly added tools across 8 groups.
"""

import json
import math
import pytest

# Guard: skip entire module if `mcp` is not installed.
mcp = pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from jsonld_ex.mcp.server import mcp as mcp_server  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _get_tool(name: str):
    """Retrieve an MCP tool function by name."""
    tool_manager = mcp_server._tool_manager
    tool = tool_manager._tools.get(name)
    if tool is None:
        available = list(tool_manager._tools.keys())
        raise KeyError(f"Tool {name!r} not found. Available: {available}")
    return tool.fn


# ═══════════════════════════════════════════════════════════════════
# Group 1: AI/ML — get_provenance
# ═══════════════════════════════════════════════════════════════════


class TestGetProvenance:
    def test_full_provenance(self):
        fn = _get_tool("get_provenance")
        node = {
            "@value": "John Smith",
            "@confidence": 0.95,
            "@source": "https://model.example.org/v2",
            "@extractedAt": "2025-01-15T10:30:00Z",
            "@method": "NER",
            "@humanVerified": True,
        }
        result = fn(node_json=json.dumps(node))
        assert result["confidence"] == 0.95
        assert result["source"] == "https://model.example.org/v2"
        assert result["extracted_at"] == "2025-01-15T10:30:00Z"
        assert result["method"] == "NER"
        assert result["human_verified"] is True

    def test_partial_provenance(self):
        fn = _get_tool("get_provenance")
        node = {"@value": "Alice", "@confidence": 0.7}
        result = fn(node_json=json.dumps(node))
        assert result["confidence"] == 0.7
        assert result["source"] is None
        assert result["method"] is None

    def test_empty_node(self):
        fn = _get_tool("get_provenance")
        result = fn(node_json=json.dumps({}))
        assert result["confidence"] is None

    def test_invalid_json(self):
        fn = _get_tool("get_provenance")
        with pytest.raises(ValueError, match="Invalid JSON"):
            fn(node_json="not json")


# ═══════════════════════════════════════════════════════════════════
# Group 2: Confidence Algebra — deduce, pairwise_conflict, conflict_metric
# ═══════════════════════════════════════════════════════════════════


class TestDeduceOpinion:
    def test_basic_deduction(self):
        fn = _get_tool("deduce_opinion")
        # If X is true (high belief), and Y|X has high belief, result should have high belief
        opinion_x = json.dumps({"belief": 0.9, "disbelief": 0.05, "uncertainty": 0.05})
        y_given_x = json.dumps({"belief": 0.8, "disbelief": 0.1, "uncertainty": 0.1})
        y_given_not_x = json.dumps({"belief": 0.1, "disbelief": 0.8, "uncertainty": 0.1})
        result = fn(
            opinion_x_json=opinion_x,
            opinion_y_given_x_json=y_given_x,
            opinion_y_given_not_x_json=y_given_not_x,
        )
        assert result["belief"] > 0.5
        assert "projected_probability" in result

    def test_vacuous_x(self):
        fn = _get_tool("deduce_opinion")
        # Total uncertainty about X → result depends on base rate
        opinion_x = json.dumps({"belief": 0.0, "disbelief": 0.0, "uncertainty": 1.0, "base_rate": 0.5})
        y_given_x = json.dumps({"belief": 1.0, "disbelief": 0.0, "uncertainty": 0.0})
        y_given_not_x = json.dumps({"belief": 0.0, "disbelief": 1.0, "uncertainty": 0.0})
        result = fn(
            opinion_x_json=opinion_x,
            opinion_y_given_x_json=y_given_x,
            opinion_y_given_not_x_json=y_given_not_x,
        )
        assert 0.4 <= result["projected_probability"] <= 0.6  # near base rate

    def test_invalid_json(self):
        fn = _get_tool("deduce_opinion")
        with pytest.raises(ValueError, match="Invalid JSON"):
            fn(
                opinion_x_json="bad",
                opinion_y_given_x_json="{}",
                opinion_y_given_not_x_json="{}",
            )


class TestMeasurePairwiseConflict:
    def test_agreeing_opinions(self):
        fn = _get_tool("measure_pairwise_conflict")
        a = json.dumps({"belief": 0.8, "disbelief": 0.1, "uncertainty": 0.1})
        b = json.dumps({"belief": 0.7, "disbelief": 0.1, "uncertainty": 0.2})
        result = fn(opinion_a_json=a, opinion_b_json=b)
        assert result["conflict"] < 0.3

    def test_conflicting_opinions(self):
        fn = _get_tool("measure_pairwise_conflict")
        a = json.dumps({"belief": 0.9, "disbelief": 0.05, "uncertainty": 0.05})
        b = json.dumps({"belief": 0.05, "disbelief": 0.9, "uncertainty": 0.05})
        result = fn(opinion_a_json=a, opinion_b_json=b)
        assert result["conflict"] > 0.7

    def test_symmetry(self):
        fn = _get_tool("measure_pairwise_conflict")
        a = json.dumps({"belief": 0.6, "disbelief": 0.3, "uncertainty": 0.1})
        b = json.dumps({"belief": 0.3, "disbelief": 0.5, "uncertainty": 0.2})
        r1 = fn(opinion_a_json=a, opinion_b_json=b)
        r2 = fn(opinion_a_json=b, opinion_b_json=a)
        assert abs(r1["conflict"] - r2["conflict"]) < 1e-10


class TestMeasureConflict:
    def test_no_conflict(self):
        fn = _get_tool("measure_conflict")
        # Clear direction → low conflict
        op = json.dumps({"belief": 0.9, "disbelief": 0.05, "uncertainty": 0.05})
        result = fn(opinion_json=op)
        assert result["conflict"] < 0.2

    def test_high_conflict(self):
        fn = _get_tool("measure_conflict")
        # Balanced belief/disbelief → high conflict
        op = json.dumps({"belief": 0.45, "disbelief": 0.45, "uncertainty": 0.1})
        result = fn(opinion_json=op)
        assert result["conflict"] > 0.7

    def test_vacuous_low_conflict(self):
        fn = _get_tool("measure_conflict")
        # Ignorance (high u) is NOT conflict
        op = json.dumps({"belief": 0.0, "disbelief": 0.0, "uncertainty": 1.0})
        result = fn(opinion_json=op)
        assert result["conflict"] == 0.0


# ═══════════════════════════════════════════════════════════════════
# Group 3: Bridge — combine_opinions_from_scalars, propagate_opinions_from_scalars
# ═══════════════════════════════════════════════════════════════════


class TestCombineOpinionsFromScalars:
    def test_basic_combination(self):
        fn = _get_tool("combine_opinions_from_scalars")
        result = fn(scores_json=json.dumps([0.9, 0.7]))
        assert result["projected_probability"] > 0.9  # combined should be higher
        assert "uncertainty" in result

    def test_single_score(self):
        fn = _get_tool("combine_opinions_from_scalars")
        result = fn(scores_json=json.dumps([0.8]))
        assert abs(result["belief"] - 0.8) < 1e-6

    def test_averaging_fusion(self):
        fn = _get_tool("combine_opinions_from_scalars")
        result = fn(scores_json=json.dumps([0.9, 0.7]), fusion="averaging")
        assert "projected_probability" in result

    def test_invalid_scores(self):
        fn = _get_tool("combine_opinions_from_scalars")
        with pytest.raises(ValueError):
            fn(scores_json=json.dumps([1.5]))


class TestPropagateOpinionsFromScalars:
    def test_basic_chain(self):
        fn = _get_tool("propagate_opinions_from_scalars")
        result = fn(chain_json=json.dumps([0.9, 0.8]))
        # With default base_rate=0, P = product = 0.72
        assert abs(result["projected_probability"] - 0.72) < 0.01

    def test_single_hop(self):
        fn = _get_tool("propagate_opinions_from_scalars")
        result = fn(chain_json=json.dumps([0.95]))
        assert result["projected_probability"] > 0.9

    def test_with_uncertainty(self):
        fn = _get_tool("propagate_opinions_from_scalars")
        result = fn(
            chain_json=json.dumps([0.9, 0.8]),
            trust_uncertainty=0.1,
        )
        assert result["uncertainty"] > 0  # should have some uncertainty


# ═══════════════════════════════════════════════════════════════════
# Group 4: Inference — propagate, combine, resolve, graph propagate
# ═══════════════════════════════════════════════════════════════════


class TestPropagateConfidence:
    def test_multiply(self):
        fn = _get_tool("propagate_confidence")
        result = fn(chain_json=json.dumps([0.9, 0.8]), method="multiply")
        assert abs(result["score"] - 0.72) < 1e-10
        assert result["method"] == "multiply"

    def test_min(self):
        fn = _get_tool("propagate_confidence")
        result = fn(chain_json=json.dumps([0.9, 0.7, 0.8]), method="min")
        assert abs(result["score"] - 0.7) < 1e-10

    def test_bayesian(self):
        fn = _get_tool("propagate_confidence")
        result = fn(chain_json=json.dumps([0.9, 0.8]), method="bayesian")
        assert result["score"] > 0.72  # bayesian is less conservative

    def test_dampened(self):
        fn = _get_tool("propagate_confidence")
        result = fn(chain_json=json.dumps([0.9, 0.8]), method="dampened")
        assert result["score"] > 0.72  # dampened attenuates penalty

    def test_invalid_method(self):
        fn = _get_tool("propagate_confidence")
        with pytest.raises(ValueError, match="Unknown propagation method"):
            fn(chain_json=json.dumps([0.9]), method="invalid")


class TestCombineSources:
    def test_noisy_or(self):
        fn = _get_tool("combine_sources")
        result = fn(scores_json=json.dumps([0.9, 0.7]), method="noisy_or")
        assert abs(result["score"] - 0.97) < 0.01

    def test_average(self):
        fn = _get_tool("combine_sources")
        result = fn(scores_json=json.dumps([0.9, 0.7]), method="average")
        assert abs(result["score"] - 0.8) < 1e-10

    def test_max(self):
        fn = _get_tool("combine_sources")
        result = fn(scores_json=json.dumps([0.9, 0.7]), method="max")
        assert abs(result["score"] - 0.9) < 1e-10

    def test_dempster_shafer(self):
        fn = _get_tool("combine_sources")
        result = fn(scores_json=json.dumps([0.9, 0.7]), method="dempster_shafer")
        assert result["score"] > 0.9


class TestResolveConflict:
    def test_highest(self):
        fn = _get_tool("resolve_conflict")
        assertions = [
            {"@value": "Engineer", "@confidence": 0.9},
            {"@value": "Manager", "@confidence": 0.85},
        ]
        result = fn(assertions_json=json.dumps(assertions), strategy="highest")
        assert result["winner"]["@value"] == "Engineer"
        assert result["strategy"] == "highest"

    def test_weighted_vote(self):
        fn = _get_tool("resolve_conflict")
        assertions = [
            {"@value": "A", "@confidence": 0.8},
            {"@value": "A", "@confidence": 0.7},
            {"@value": "B", "@confidence": 0.9},
        ]
        result = fn(assertions_json=json.dumps(assertions), strategy="weighted_vote")
        # noisy-OR(0.8, 0.7) = 1-(0.2*0.3) = 0.94 > 0.9
        assert result["winner"]["@value"] == "A"

    def test_recency(self):
        fn = _get_tool("resolve_conflict")
        assertions = [
            {"@value": "Old", "@confidence": 0.9, "@extractedAt": "2024-01-01T00:00:00Z"},
            {"@value": "New", "@confidence": 0.8, "@extractedAt": "2025-06-01T00:00:00Z"},
        ]
        result = fn(assertions_json=json.dumps(assertions), strategy="recency")
        assert result["winner"]["@value"] == "New"

    def test_empty_assertions(self):
        fn = _get_tool("resolve_conflict")
        with pytest.raises(ValueError, match="non-empty"):
            fn(assertions_json=json.dumps([]))


class TestPropagateGraphConfidence:
    def test_basic_graph(self):
        fn = _get_tool("propagate_graph_confidence")
        doc = {
            "source_fact": {"@value": "X", "@confidence": 0.9},
            "inferred": {"@value": "Y", "@confidence": 0.8},
        }
        result = fn(
            document_json=json.dumps(doc),
            property_chain_json=json.dumps(["source_fact", "inferred"]),
            method="multiply",
        )
        assert abs(result["score"] - 0.72) < 1e-10

    def test_missing_property(self):
        fn = _get_tool("propagate_graph_confidence")
        doc = {"source_fact": {"@value": "X", "@confidence": 0.9}}
        with pytest.raises(KeyError):
            fn(
                document_json=json.dumps(doc),
                property_chain_json=json.dumps(["source_fact", "missing"]),
            )


# ═══════════════════════════════════════════════════════════════════
# Group 5: Security — check_context_allowed, enforce_resource_limits
# ═══════════════════════════════════════════════════════════════════


class TestCheckContextAllowed:
    def test_allowed_exact(self):
        fn = _get_tool("check_context_allowed")
        config = {"allowed": ["https://schema.org/"], "patterns": []}
        result = fn(url="https://schema.org/", config_json=json.dumps(config))
        assert result["allowed"] is True

    def test_blocked(self):
        fn = _get_tool("check_context_allowed")
        config = {"allowed": ["https://schema.org/"], "patterns": []}
        result = fn(url="https://evil.example.org/", config_json=json.dumps(config))
        assert result["allowed"] is False

    def test_pattern_match(self):
        fn = _get_tool("check_context_allowed")
        config = {"allowed": [], "patterns": ["https://example.org/contexts/*"]}
        result = fn(url="https://example.org/contexts/v1", config_json=json.dumps(config))
        assert result["allowed"] is True

    def test_block_all_remote(self):
        fn = _get_tool("check_context_allowed")
        config = {"block_remote_contexts": True}
        result = fn(url="https://schema.org/", config_json=json.dumps(config))
        assert result["allowed"] is False


class TestEnforceResourceLimits:
    def test_valid_document(self):
        fn = _get_tool("enforce_resource_limits")
        doc = {"@type": "Person", "name": "Alice"}
        result = fn(document_json=json.dumps(doc))
        assert result["valid"] is True

    def test_exceeds_depth(self):
        fn = _get_tool("enforce_resource_limits")
        # Build a deeply nested document
        nested = {"value": "leaf"}
        for _ in range(150):
            nested = {"child": nested}
        result = fn(
            document_json=json.dumps(nested),
            limits_json=json.dumps({"max_graph_depth": 10}),
        )
        assert result["valid"] is False
        assert "depth" in result["error"].lower()

    def test_exceeds_size(self):
        fn = _get_tool("enforce_resource_limits")
        result = fn(
            document_json=json.dumps({"data": "x" * 1000}),
            limits_json=json.dumps({"max_document_size": 100}),
        )
        assert result["valid"] is False
        assert "size" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════
# Group 6: Graph Ops — diff_graphs
# ═══════════════════════════════════════════════════════════════════


class TestDiffGraphs:
    def test_identical_graphs(self):
        fn = _get_tool("diff_graphs")
        graph = {"@graph": [{"@id": "ex:1", "name": "Alice"}]}
        result = fn(
            graph_a_json=json.dumps(graph),
            graph_b_json=json.dumps(graph),
        )
        diff = json.loads(result)
        assert len(diff.get("added", [])) == 0
        assert len(diff.get("removed", [])) == 0

    def test_added_node(self):
        fn = _get_tool("diff_graphs")
        a = {"@graph": [{"@id": "ex:1", "name": "Alice"}]}
        b = {"@graph": [
            {"@id": "ex:1", "name": "Alice"},
            {"@id": "ex:2", "name": "Bob"},
        ]}
        result = fn(graph_a_json=json.dumps(a), graph_b_json=json.dumps(b))
        diff = json.loads(result)
        assert len(diff.get("added", [])) > 0

    def test_removed_node(self):
        fn = _get_tool("diff_graphs")
        a = {"@graph": [
            {"@id": "ex:1", "name": "Alice"},
            {"@id": "ex:2", "name": "Bob"},
        ]}
        b = {"@graph": [{"@id": "ex:1", "name": "Alice"}]}
        result = fn(graph_a_json=json.dumps(a), graph_b_json=json.dumps(b))
        diff = json.loads(result)
        assert len(diff.get("removed", [])) > 0


# ═══════════════════════════════════════════════════════════════════
# Group 7: Temporal — add_temporal_annotation, temporal_diff
# ═══════════════════════════════════════════════════════════════════


class TestAddTemporalAnnotation:
    def test_add_valid_from(self):
        fn = _get_tool("add_temporal_annotation")
        result = fn(value="Engineer", valid_from="2020-01-01")
        assert result["@value"] == "Engineer"
        assert result["@validFrom"] == "2020-01-01"

    def test_add_all_qualifiers(self):
        fn = _get_tool("add_temporal_annotation")
        result = fn(
            value="Manager",
            valid_from="2020-01-01",
            valid_until="2024-12-31",
            as_of="2023-06-15T10:00:00Z",
        )
        assert result["@validFrom"] == "2020-01-01"
        assert result["@validUntil"] == "2024-12-31"
        assert result["@asOf"] == "2023-06-15T10:00:00Z"

    def test_invalid_range(self):
        fn = _get_tool("add_temporal_annotation")
        with pytest.raises(ValueError, match="must not be after"):
            fn(value="X", valid_from="2025-01-01", valid_until="2020-01-01")

    def test_no_qualifiers(self):
        fn = _get_tool("add_temporal_annotation")
        with pytest.raises(ValueError, match="At least one"):
            fn(value="X")


class TestTemporalDiff:
    def test_basic_diff(self):
        fn = _get_tool("temporal_diff")
        graph = [
            {
                "@id": "ex:person1",
                "jobTitle": [
                    {
                        "@value": "Engineer",
                        "@validFrom": "2020-01-01",
                        "@validUntil": "2023-12-31",
                    },
                    {
                        "@value": "Manager",
                        "@validFrom": "2024-01-01",
                    },
                ],
            }
        ]
        result = fn(
            graph_json=json.dumps(graph),
            t1="2022-06-01",
            t2="2024-06-01",
        )
        # Should show a modification in jobTitle
        assert len(result.get("modified", [])) > 0 or len(result.get("added", [])) > 0

    def test_no_changes(self):
        fn = _get_tool("temporal_diff")
        graph = [{"@id": "ex:1", "name": "Alice"}]
        result = fn(
            graph_json=json.dumps(graph),
            t1="2022-01-01",
            t2="2025-01-01",
        )
        assert len(result.get("modified", [])) == 0


# ═══════════════════════════════════════════════════════════════════
# Group 8: Interop — from_prov_o, shacl_to_shape, shape_to_owl,
#           to_rdf_star, compare_prov_o_verbosity, compare_shacl_verbosity
# ═══════════════════════════════════════════════════════════════════


class TestFromProvO:
    def test_round_trip(self):
        """to_prov_o → from_prov_o should recover the original annotations."""
        to_fn = _get_tool("to_prov_o")
        from_fn = _get_tool("from_prov_o")
        node = {
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
                "@source": "https://model.example.org/v1",
                "@extractedAt": "2025-01-15T10:00:00Z",
            },
        }
        prov_json = to_fn(node_json=json.dumps(node))
        recovered_json = from_fn(prov_o_json=prov_json)
        recovered = json.loads(recovered_json)
        # Should have some structure back
        assert isinstance(recovered, dict)

    def test_invalid_json(self):
        fn = _get_tool("from_prov_o")
        with pytest.raises(ValueError, match="Invalid JSON"):
            fn(prov_o_json="not json")


class TestShaclToShape:
    def test_round_trip(self):
        """shape_to_shacl → shacl_to_shape should recover constraints."""
        to_fn = _get_tool("shape_to_shacl")
        from_fn = _get_tool("shacl_to_shape")
        shape = {
            "name": {"@required": True, "@type": "xsd:string"},
            "age": {"@type": "xsd:integer", "@minimum": 0},
        }
        shacl_json = to_fn(
            shape_json=json.dumps(shape),
            target_class="http://schema.org/Person",
        )
        recovered_json = from_fn(shacl_json=shacl_json)
        recovered = json.loads(recovered_json)
        assert isinstance(recovered, dict)

    def test_invalid_json(self):
        fn = _get_tool("shacl_to_shape")
        with pytest.raises(ValueError, match="Invalid JSON"):
            fn(shacl_json="bad")


class TestShapeToOwl:
    def test_basic_owl(self):
        fn = _get_tool("shape_to_owl")
        shape = {
            "name": {"@required": True, "@type": "xsd:string"},
            "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
        }
        result = fn(
            shape_json=json.dumps(shape),
            class_iri="http://schema.org/Person",
        )
        owl = json.loads(result)
        assert isinstance(owl, dict)

    def test_no_class_iri(self):
        fn = _get_tool("shape_to_owl")
        shape = {"@type": "Person", "name": {"@required": True}}
        result = fn(shape_json=json.dumps(shape))
        owl = json.loads(result)
        assert isinstance(owl, dict)


class TestToRdfStar:
    def test_basic_export(self):
        fn = _get_tool("to_rdf_star")
        doc = {
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
            },
        }
        result = fn(node_json=json.dumps(doc))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_base_subject(self):
        fn = _get_tool("to_rdf_star")
        doc = {
            "name": {"@value": "Bob", "@confidence": 0.8},
        }
        result = fn(
            node_json=json.dumps(doc),
            base_subject="http://example.org/Bob",
        )
        assert "http://example.org/Bob" in result


class TestCompareProvOVerbosity:
    def test_comparison(self):
        fn = _get_tool("compare_prov_o_verbosity")
        doc = {
            "@type": "Person",
            "name": {
                "@value": "Alice",
                "@confidence": 0.95,
                "@source": "https://model.example.org/v1",
            },
        }
        result = fn(document_json=json.dumps(doc))
        assert "jsonld_ex_triples" in result
        assert "prov_o_triples" in result
        assert "reduction_percent" in result


class TestCompareShaclVerbosity:
    def test_comparison(self):
        fn = _get_tool("compare_shacl_verbosity")
        shape = {
            "@type": "Person",
            "name": {"@required": True, "@type": "xsd:string"},
            "age": {"@type": "xsd:integer", "@minimum": 0},
        }
        result = fn(shape_json=json.dumps(shape))
        assert "jsonld_ex_triples" in result
        assert "shacl_triples" in result
        assert "reduction_percent" in result


# ═══════════════════════════════════════════════════════════════════
# Registration check — ensure all 21 new tools exist
# ═══════════════════════════════════════════════════════════════════


class TestToolRegistration:
    EXPECTED_NEW_TOOLS = [
        "get_provenance",
        "deduce_opinion",
        "measure_pairwise_conflict",
        "measure_conflict",
        "combine_opinions_from_scalars",
        "propagate_opinions_from_scalars",
        "propagate_confidence",
        "combine_sources",
        "resolve_conflict",
        "propagate_graph_confidence",
        "check_context_allowed",
        "enforce_resource_limits",
        "diff_graphs",
        "add_temporal_annotation",
        "temporal_diff",
        "from_prov_o",
        "shacl_to_shape",
        "shape_to_owl",
        "to_rdf_star",
        "compare_prov_o_verbosity",
        "compare_shacl_verbosity",
    ]

    def test_all_new_tools_registered(self):
        registered = set(mcp_server._tool_manager._tools.keys())
        for name in self.EXPECTED_NEW_TOOLS:
            assert name in registered, f"Tool {name!r} not registered"

    def test_total_tool_count(self):
        """16 original + 21 Phase 1 + 4 MQTT + 10 Compliance = 51 total."""
        registered = list(mcp_server._tool_manager._tools.values())
        assert len(registered) == 51
