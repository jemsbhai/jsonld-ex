"""Tests for owl_interop module — PROV-O, SHACL, OWL, RDF-star bidirectional mapping."""

import json
import pytest

from jsonld_ex.ai_ml import annotate, JSONLD_EX_NAMESPACE
from jsonld_ex.owl_interop import (
    PROV,
    SHACL,
    OWL,
    XSD,
    RDFS,
    ConversionReport,
    VerbosityComparison,
    to_prov_o,
    from_prov_o,
    shape_to_shacl,
    shacl_to_shape,
    shape_to_owl_restrictions,
    to_rdf_star_ntriples,
    compare_with_prov_o,
    compare_with_shacl,
)


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture
def annotated_person():
    """Simple document with a single annotated property."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/alice",
        "name": annotate(
            "Alice Smith",
            confidence=0.95,
            source="https://models.example.org/gpt4",
            extracted_at="2025-01-15T10:30:00Z",
            method="NER",
        ),
    }


@pytest.fixture
def multi_annotated():
    """Document with multiple annotated properties."""
    return {
        "@context": "http://schema.org/",
        "@type": "Person",
        "@id": "http://example.org/bob",
        "name": annotate("Bob Jones", confidence=0.92, source="https://models.example.org/gpt4"),
        "email": annotate(
            "bob@example.com",
            confidence=0.88,
            source="https://models.example.org/gpt4",
            method="regex-extraction",
            human_verified=True,
        ),
        "jobTitle": annotate(
            "Engineer",
            confidence=0.75,
            extracted_at="2025-01-10T08:00:00Z",
        ),
    }


@pytest.fixture
def person_shape():
    """Shape definition for a Person."""
    return {
        "@type": "http://schema.org/Person",
        "http://schema.org/name": {
            "@required": True,
            "@type": "xsd:string",
            "@minLength": 1,
        },
        "http://schema.org/email": {
            "@pattern": "^[^@]+@[^@]+$",
        },
        "http://schema.org/age": {
            "@type": "xsd:integer",
            "@minimum": 0,
            "@maximum": 150,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# PROV-O TESTS
# ═══════════════════════════════════════════════════════════════════


class TestToProvO:
    """Tests for jsonld-ex → PROV-O conversion."""

    def test_basic_conversion(self, annotated_person):
        prov_doc, report = to_prov_o(annotated_person)

        assert report.success is True
        assert report.nodes_converted == 1
        assert "@graph" in prov_doc

    def test_prov_context_present(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        ctx = prov_doc["@context"]

        assert ctx["prov"] == PROV
        assert ctx["xsd"] == XSD

    def test_entity_created(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        assert len(entities) == 1

        entity = entities[0]
        assert entity[f"{PROV}value"] == "Alice Smith"

    def test_confidence_preserved(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        entity = entities[0]
        assert entity[f"{JSONLD_EX_NAMESPACE}confidence"] == 0.95

    def test_software_agent_created(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        agents = [n for n in graph if n.get("@type") == f"{PROV}SoftwareAgent"]
        assert len(agents) == 1
        assert agents[0]["@id"] == "https://models.example.org/gpt4"

    def test_generated_at_time(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        entity = entities[0]

        gen_time = entity.get(f"{PROV}generatedAtTime")
        assert gen_time is not None
        assert gen_time["@value"] == "2025-01-15T10:30:00Z"
        assert gen_time["@type"] == f"{XSD}dateTime"

    def test_activity_created_for_method(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        activities = [n for n in graph if n.get("@type") == f"{PROV}Activity"]
        assert len(activities) == 1
        assert activities[0][f"{RDFS}label"] == "NER"

    def test_multiple_annotations(self, multi_annotated):
        prov_doc, report = to_prov_o(multi_annotated)

        assert report.nodes_converted == 3

        graph = prov_doc["@graph"]
        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        assert len(entities) == 3

    def test_human_verified(self, multi_annotated):
        prov_doc, _ = to_prov_o(multi_annotated)
        graph = prov_doc["@graph"]

        persons = [n for n in graph if n.get("@type") == f"{PROV}Person"]
        assert len(persons) >= 1
        assert persons[0][f"{RDFS}label"] == "Human Verifier"

    def test_triple_count_higher_than_input(self, annotated_person):
        _, report = to_prov_o(annotated_person)

        # PROV-O should produce MORE triples than the original (that's the point)
        assert report.triples_output > report.triples_input

    def test_preserves_type(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        graph = prov_doc["@graph"]

        # The main node should still have its @type
        main = [n for n in graph if n.get("@type") == "Person"]
        assert len(main) == 1

    def test_unannotated_value_passthrough(self):
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "name": "plain string",
        }
        prov_doc, report = to_prov_o(doc)
        assert report.nodes_converted == 0

        graph = prov_doc["@graph"]
        main = [n for n in graph if n.get("@type") == "Person"]
        assert main[0]["name"] == "plain string"


class TestFromProvO:
    """Tests for PROV-O → jsonld-ex round-trip."""

    def test_round_trip_preserves_value(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        assert restored["name"]["@value"] == "Alice Smith"

    def test_round_trip_preserves_confidence(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        restored, _ = from_prov_o(prov_doc)

        assert restored["name"]["@confidence"] == 0.95

    def test_round_trip_preserves_source(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        restored, _ = from_prov_o(prov_doc)

        assert restored["name"]["@source"] == "https://models.example.org/gpt4"

    def test_round_trip_preserves_method(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        restored, _ = from_prov_o(prov_doc)

        assert restored["name"]["@method"] == "NER"

    def test_round_trip_preserves_extracted_at(self, annotated_person):
        prov_doc, _ = to_prov_o(annotated_person)
        restored, _ = from_prov_o(prov_doc)

        assert restored["name"]["@extractedAt"] == "2025-01-15T10:30:00Z"

    def test_no_main_node_error(self):
        """PROV-O doc with only PROV types should error gracefully."""
        prov_doc = {
            "@context": {"prov": PROV},
            "@graph": [
                {"@id": "_:e1", "@type": f"{PROV}Entity", f"{PROV}value": "x"}
            ],
        }
        _, report = from_prov_o(prov_doc)
        assert report.success is False
        assert len(report.errors) > 0


# ═══════════════════════════════════════════════════════════════════
# SHACL TESTS
# ═══════════════════════════════════════════════════════════════════


class TestShapeToShacl:
    """Tests for jsonld-ex @shape → SHACL conversion."""

    def test_basic_conversion(self, person_shape):
        shacl = shape_to_shacl(person_shape)

        assert "@graph" in shacl
        graph = shacl["@graph"]
        assert len(graph) == 1

        node_shape = graph[0]
        assert node_shape["@type"] == f"{SHACL}NodeShape"

    def test_target_class(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        node_shape = shacl["@graph"][0]

        target = node_shape[f"{SHACL}targetClass"]
        assert target["@id"] == "http://schema.org/Person"

    def test_required_to_min_count(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        properties = shacl["@graph"][0][f"{SHACL}property"]

        name_prop = next(p for p in properties if p[f"{SHACL}path"]["@id"] == "http://schema.org/name")
        assert name_prop[f"{SHACL}minCount"] == 1

    def test_type_to_datatype(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        properties = shacl["@graph"][0][f"{SHACL}property"]

        name_prop = next(p for p in properties if p[f"{SHACL}path"]["@id"] == "http://schema.org/name")
        assert name_prop[f"{SHACL}datatype"]["@id"] == f"{XSD}string"

    def test_min_max_to_inclusive(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        properties = shacl["@graph"][0][f"{SHACL}property"]

        age_prop = next(p for p in properties if p[f"{SHACL}path"]["@id"] == "http://schema.org/age")
        assert age_prop[f"{SHACL}minInclusive"] == 0
        assert age_prop[f"{SHACL}maxInclusive"] == 150

    def test_pattern_mapping(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        properties = shacl["@graph"][0][f"{SHACL}property"]

        email_prop = next(p for p in properties if p[f"{SHACL}path"]["@id"] == "http://schema.org/email")
        assert email_prop[f"{SHACL}pattern"] == "^[^@]+@[^@]+$"

    def test_min_length_mapping(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        properties = shacl["@graph"][0][f"{SHACL}property"]

        name_prop = next(p for p in properties if p[f"{SHACL}path"]["@id"] == "http://schema.org/name")
        assert name_prop[f"{SHACL}minLength"] == 1

    def test_custom_target_class(self, person_shape):
        shacl = shape_to_shacl(person_shape, target_class="http://example.org/Employee")
        node_shape = shacl["@graph"][0]
        assert node_shape[f"{SHACL}targetClass"]["@id"] == "http://example.org/Employee"

    def test_no_type_raises(self):
        with pytest.raises(ValueError, match="@type"):
            shape_to_shacl({"name": {"@required": True}})


class TestShaclToShape:
    """Tests for SHACL → jsonld-ex @shape round-trip."""

    def test_round_trip(self, person_shape):
        shacl = shape_to_shacl(person_shape)
        restored, warnings = shacl_to_shape(shacl)

        assert restored["@type"] == "http://schema.org/Person"
        assert restored["http://schema.org/name"]["@required"] is True
        assert restored["http://schema.org/age"]["@minimum"] == 0

    def test_unsupported_shacl_feature_warning(self):
        """SHACL features without @shape equivalent should produce warnings."""
        shacl_doc = {
            "@context": {"sh": SHACL},
            "@graph": [{
                "@id": "_:s",
                "@type": f"{SHACL}NodeShape",
                f"{SHACL}targetClass": {"@id": "http://example.org/Thing"},
                f"{SHACL}property": [{
                    f"{SHACL}path": {"@id": "http://example.org/name"},
                    f"{SHACL}minCount": 1,
                    f"{SHACL}sparql": {"@value": "some SPARQL query"},
                }],
            }],
        }
        shape, warnings = shacl_to_shape(shacl_doc)

        assert any("sh:sparql" in w for w in warnings)
        assert shape["http://example.org/name"]["@required"] is True

    def test_no_node_shape_warning(self):
        shacl_doc = {"@graph": [{"@id": "_:x", "@type": "something:Else"}]}
        _, warnings = shacl_to_shape(shacl_doc)
        assert any("NodeShape" in w for w in warnings)


# ═══════════════════════════════════════════════════════════════════
# OWL TESTS
# ═══════════════════════════════════════════════════════════════════


class TestShapeToOwl:
    """Tests for jsonld-ex @shape → OWL class restrictions."""

    def test_basic_conversion(self, person_shape):
        owl = shape_to_owl_restrictions(person_shape)

        assert "@graph" in owl
        graph = owl["@graph"]
        assert len(graph) == 1

        cls = graph[0]
        assert cls["@type"] == f"{OWL}Class"
        assert cls["@id"] == "http://schema.org/Person"

    def test_required_to_min_cardinality(self, person_shape):
        owl = shape_to_owl_restrictions(person_shape)
        cls = owl["@graph"][0]

        restrictions = cls[f"{RDFS}subClassOf"]
        if not isinstance(restrictions, list):
            restrictions = [restrictions]

        cardinality_restrictions = [
            r for r in restrictions
            if f"{OWL}minCardinality" in r
        ]
        assert len(cardinality_restrictions) >= 1

    def test_type_to_all_values_from(self, person_shape):
        owl = shape_to_owl_restrictions(person_shape)
        cls = owl["@graph"][0]

        restrictions = cls[f"{RDFS}subClassOf"]
        if not isinstance(restrictions, list):
            restrictions = [restrictions]

        avf_restrictions = [
            r for r in restrictions
            if f"{OWL}allValuesFrom" in r
        ]
        # name (xsd:string) and age (xsd:integer) have @type
        assert len(avf_restrictions) >= 2

    def test_no_type_raises(self):
        with pytest.raises(ValueError, match="@type"):
            shape_to_owl_restrictions({"name": {"@required": True}})


# ═══════════════════════════════════════════════════════════════════
# RDF-STAR TESTS
# ═══════════════════════════════════════════════════════════════════


class TestToRdfStar:
    """Tests for jsonld-ex → RDF-star N-Triples."""

    def test_basic_output(self, annotated_person):
        ntriples, report = to_rdf_star_ntriples(annotated_person)

        assert report.success is True
        assert len(ntriples) > 0
        assert report.nodes_converted == 1

    def test_embedded_triple_syntax(self, annotated_person):
        ntriples, _ = to_rdf_star_ntriples(annotated_person)

        # Should contain << ... >> embedded triple syntax
        assert "<<" in ntriples
        assert ">>" in ntriples

    def test_confidence_annotation(self, annotated_person):
        ntriples, _ = to_rdf_star_ntriples(annotated_person)

        assert f"{JSONLD_EX_NAMESPACE}confidence" in ntriples
        assert "0.95" in ntriples

    def test_source_annotation(self, annotated_person):
        ntriples, _ = to_rdf_star_ntriples(annotated_person)

        assert f"{JSONLD_EX_NAMESPACE}source" in ntriples
        assert "models.example.org/gpt4" in ntriples

    def test_unannotated_plain_triple(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": "Plain",
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert report.nodes_converted == 0
        assert "Plain" in ntriples
        assert "<<" not in ntriples

    def test_all_annotation_fields(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate(
                "Full",
                confidence=0.99,
                source="https://example.org/model",
                extracted_at="2025-01-01T00:00:00Z",
                method="NER",
                human_verified=True,
            ),
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert f"{JSONLD_EX_NAMESPACE}confidence" in ntriples
        assert f"{JSONLD_EX_NAMESPACE}source" in ntriples
        assert f"{JSONLD_EX_NAMESPACE}extractedAt" in ntriples
        assert f"{JSONLD_EX_NAMESPACE}method" in ntriples
        assert f"{JSONLD_EX_NAMESPACE}humanVerified" in ntriples
        assert report.nodes_converted == 1


# ═══════════════════════════════════════════════════════════════════
# COMPARISON / VERBOSITY TESTS
# ═══════════════════════════════════════════════════════════════════


class TestComparisons:
    """Verbosity comparison utilities."""

    def test_prov_o_comparison(self, annotated_person):
        comparison = compare_with_prov_o(annotated_person)

        assert isinstance(comparison, VerbosityComparison)
        assert comparison.alternative_name == "PROV-O"
        assert comparison.alternative_triples > comparison.jsonld_ex_triples
        assert comparison.triple_reduction_pct > 0
        assert comparison.alternative_bytes > comparison.jsonld_ex_bytes
        assert comparison.byte_reduction_pct > 0

    def test_shacl_comparison(self, person_shape):
        comparison = compare_with_shacl(person_shape)

        assert isinstance(comparison, VerbosityComparison)
        assert comparison.alternative_name == "SHACL"
        assert comparison.alternative_triples > comparison.jsonld_ex_triples
        assert comparison.alternative_bytes > comparison.jsonld_ex_bytes

    def test_multi_annotation_scaling(self, multi_annotated):
        """Verbosity gap should widen with more annotations."""
        comparison = compare_with_prov_o(multi_annotated)

        # 3 annotated properties → PROV-O should create even more nodes
        assert comparison.alternative_triples > comparison.jsonld_ex_triples
        assert comparison.triple_reduction_pct > 30  # at least 30% reduction
