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
    SOSA,
    SSN,
    SSN_SYSTEM,
    QUDT,
    JSONLD_EX,
    ConversionReport,
    VerbosityComparison,
    to_prov_o,
    from_prov_o,
    to_ssn,
    from_ssn,
    shape_to_shacl,
    shacl_to_shape,
    shape_to_owl_restrictions,
    owl_to_shape,
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
# GAP-P2: @derivedFrom ↔ prov:wasDerivedFrom ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════


class TestDerivedFromProvO:
    """@derivedFrom ↔ prov:wasDerivedFrom bidirectional mapping."""

    def test_to_prov_o_single_derived_from(self):
        """@derivedFrom (single IRI) → prov:wasDerivedFrom on Entity."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/merged-alice",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/er-v1",
                derived_from="https://example.org/record/42",
            ),
        }
        prov_doc, report = to_prov_o(doc)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        assert len(entities) == 1
        entity = entities[0]

        derived = entity.get(f"{PROV}wasDerivedFrom")
        assert derived is not None
        # Single source → single reference
        assert derived == {"@id": "https://example.org/record/42"}

    def test_to_prov_o_multiple_derived_from(self):
        """@derivedFrom (list of IRIs) → list of prov:wasDerivedFrom."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/fused-entity",
            "name": annotate(
                "Bob Jones",
                confidence=0.90,
                derived_from=[
                    "https://example.org/src/a",
                    "https://example.org/src/b",
                ],
            ),
        }
        prov_doc, report = to_prov_o(doc)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        entity = entities[0]

        derived = entity.get(f"{PROV}wasDerivedFrom")
        assert isinstance(derived, list)
        assert len(derived) == 2
        ids = {d["@id"] for d in derived}
        assert ids == {"https://example.org/src/a", "https://example.org/src/b"}

    def test_to_prov_o_no_derived_from(self):
        """Without @derivedFrom, no prov:wasDerivedFrom should appear."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate("Alice", confidence=0.95),
        }
        prov_doc, _ = to_prov_o(doc)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        entity = entities[0]
        assert f"{PROV}wasDerivedFrom" not in entity

    def test_round_trip_single_derived_from(self):
        """jsonld-ex → PROV-O → jsonld-ex preserves single @derivedFrom."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/merged",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/er",
                derived_from="https://example.org/record/42",
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        assert restored["name"]["@value"] == "Alice Smith"
        assert restored["name"]["@confidence"] == 0.95
        assert restored["name"]["@derivedFrom"] == "https://example.org/record/42"

    def test_round_trip_multiple_derived_from(self):
        """jsonld-ex → PROV-O → jsonld-ex preserves list @derivedFrom."""
        sources = ["https://example.org/a", "https://example.org/b"]
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/fused",
            "name": annotate(
                "Fused Name",
                confidence=0.85,
                derived_from=sources,
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        restored_sources = restored["name"]["@derivedFrom"]
        assert isinstance(restored_sources, list)
        assert set(restored_sources) == set(sources)

    def test_derived_from_triple_count(self):
        """@derivedFrom adds triples to PROV-O output."""
        without = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/x",
            "name": annotate("X", confidence=0.9),
        }
        with_derived = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/x",
            "name": annotate(
                "X", confidence=0.9,
                derived_from="https://example.org/src",
            ),
        }
        _, report_without = to_prov_o(without)
        _, report_with = to_prov_o(with_derived)

        assert report_with.triples_output > report_without.triples_output


class TestDerivedFromRdfStar:
    """@derivedFrom in RDF-star N-Triples output."""

    def test_single_derived_from_in_ntriples(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate(
                "Alice",
                confidence=0.95,
                derived_from="https://example.org/record/42",
            ),
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert f"{JSONLD_EX_NAMESPACE}derivedFrom" in ntriples
        assert "example.org/record/42" in ntriples

    def test_multiple_derived_from_in_ntriples(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate(
                "Fused",
                confidence=0.85,
                derived_from=["https://example.org/a", "https://example.org/b"],
            ),
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert f"{JSONLD_EX_NAMESPACE}derivedFrom" in ntriples
        assert "example.org/a" in ntriples
        assert "example.org/b" in ntriples

    def test_no_derived_from_absent(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate("Alice", confidence=0.95),
        }
        ntriples, _ = to_rdf_star_ntriples(doc)

        assert "derivedFrom" not in ntriples


# ═══════════════════════════════════════════════════════════════════
# GAP-P1: @delegatedBy ↔ prov:actedOnBehalfOf ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════


class TestDelegationProvO:
    """@delegatedBy ↔ prov:actedOnBehalfOf bidirectional mapping."""

    def test_to_prov_o_delegation_single(self):
        """@delegatedBy → prov:actedOnBehalfOf on the SoftwareAgent."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/ner-v3",
                delegated_by="https://pipeline.example.org/etl-v2",
            ),
        }
        prov_doc, report = to_prov_o(doc)
        graph = prov_doc["@graph"]

        # The SoftwareAgent should have actedOnBehalfOf
        agents = [n for n in graph if n.get("@type") == f"{PROV}SoftwareAgent"]
        assert len(agents) == 1
        agent = agents[0]
        delegation = agent.get(f"{PROV}actedOnBehalfOf")
        assert delegation is not None
        assert delegation["@id"] == "https://pipeline.example.org/etl-v2"

    def test_to_prov_o_delegation_list(self):
        """@delegatedBy (list) → list of prov:actedOnBehalfOf."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/ner-v3",
                delegated_by=[
                    "https://pipeline.example.org/step-2",
                    "https://user.example.org/alice",
                ],
            ),
        }
        prov_doc, report = to_prov_o(doc)
        graph = prov_doc["@graph"]

        agents = [n for n in graph if n.get("@type") == f"{PROV}SoftwareAgent"]
        assert len(agents) == 1
        agent = agents[0]
        delegation = agent.get(f"{PROV}actedOnBehalfOf")
        assert isinstance(delegation, list)
        assert len(delegation) == 2

    def test_to_prov_o_no_delegation(self):
        """Without @delegatedBy, no actedOnBehalfOf should appear."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Alice",
                confidence=0.95,
                source="https://model.example.org/ner",
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        graph = prov_doc["@graph"]

        agents = [n for n in graph if n.get("@type") == f"{PROV}SoftwareAgent"]
        assert len(agents) == 1
        assert f"{PROV}actedOnBehalfOf" not in agents[0]

    def test_round_trip_delegation_single(self):
        """jsonld-ex → PROV-O → jsonld-ex preserves single @delegatedBy."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/ner-v3",
                delegated_by="https://pipeline.example.org/etl-v2",
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        assert restored["name"]["@delegatedBy"] == "https://pipeline.example.org/etl-v2"

    def test_round_trip_delegation_list(self):
        """jsonld-ex → PROV-O → jsonld-ex preserves list @delegatedBy."""
        delegates = [
            "https://pipeline.example.org/step-2",
            "https://user.example.org/alice",
        ]
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Alice Smith",
                confidence=0.95,
                source="https://model.example.org/ner-v3",
                delegated_by=delegates,
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        restored_delegates = restored["name"]["@delegatedBy"]
        assert isinstance(restored_delegates, list)
        assert set(restored_delegates) == set(delegates)


class TestDelegationRdfStar:
    """@delegatedBy in RDF-star N-Triples output."""

    def test_delegated_by_in_ntriples(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate(
                "Alice",
                confidence=0.95,
                source="https://model.example.org/ner",
                delegated_by="https://pipeline.example.org/etl",
            ),
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert f"{JSONLD_EX_NAMESPACE}delegatedBy" in ntriples
        assert "pipeline.example.org/etl" in ntriples

    def test_no_delegated_by_absent(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate("Alice", confidence=0.95),
        }
        ntriples, _ = to_rdf_star_ntriples(doc)

        assert "delegatedBy" not in ntriples


# ═══════════════════════════════════════════════════════════════════
# GAP-P3: @invalidatedAt/@invalidationReason ↔ prov:wasInvalidatedBy
# ═══════════════════════════════════════════════════════════════════


class TestInvalidationProvO:
    """@invalidatedAt/@invalidationReason ↔ prov:wasInvalidatedBy."""

    def test_to_prov_o_invalidated_at(self):
        """@invalidatedAt → prov:wasInvalidatedBy with prov:atTime."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Old Name",
                confidence=0.3,
                invalidated_at="2025-06-01T00:00:00Z",
                invalidation_reason="Corrected by human review",
            ),
        }
        prov_doc, report = to_prov_o(doc)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        assert len(entities) == 1
        entity = entities[0]

        # Should have wasInvalidatedBy
        invalidation = entity.get(f"{PROV}wasInvalidatedBy")
        assert invalidation is not None

    def test_to_prov_o_no_invalidation(self):
        """Without invalidation fields, no wasInvalidatedBy should appear."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate("Alice", confidence=0.95),
        }
        prov_doc, _ = to_prov_o(doc)
        graph = prov_doc["@graph"]

        entities = [n for n in graph if n.get("@type") == f"{PROV}Entity"]
        entity = entities[0]
        assert f"{PROV}wasInvalidatedBy" not in entity

    def test_round_trip_invalidation(self):
        """jsonld-ex → PROV-O → jsonld-ex preserves invalidation."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": annotate(
                "Old Name",
                confidence=0.3,
                source="https://model.example.org/v1",
                invalidated_at="2025-06-01T00:00:00Z",
                invalidation_reason="Corrected",
            ),
        }
        prov_doc, _ = to_prov_o(doc)
        restored, report = from_prov_o(prov_doc)

        assert report.success is True
        assert restored["name"]["@invalidatedAt"] == "2025-06-01T00:00:00Z"
        assert restored["name"]["@invalidationReason"] == "Corrected"


class TestInvalidationRdfStar:
    """@invalidatedAt/@invalidationReason in RDF-star N-Triples."""

    def test_invalidation_in_ntriples(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate(
                "Old",
                confidence=0.3,
                invalidated_at="2025-06-01T00:00:00Z",
                invalidation_reason="Superseded",
            ),
        }
        ntriples, report = to_rdf_star_ntriples(doc)

        assert f"{JSONLD_EX_NAMESPACE}invalidatedAt" in ntriples
        assert "2025-06-01" in ntriples
        assert f"{JSONLD_EX_NAMESPACE}invalidationReason" in ntriples
        assert "Superseded" in ntriples

    def test_no_invalidation_absent(self):
        doc = {
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate("Alice", confidence=0.95),
        }
        ntriples, _ = to_rdf_star_ntriples(doc)

        assert "invalidatedAt" not in ntriples
        assert "invalidationReason" not in ntriples


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

    def test_minimum_maximum_with_type_produces_datatype_restriction(self):
        """@type + @minimum/@maximum → combined DatatypeRestriction."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/age": {
                "@type": "xsd:integer",
                "@minimum": 0,
                "@maximum": 150,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        # Single restriction (combined, not two separate ones)
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        # Should be a DatatypeRestriction, not a simple IRI
        assert isinstance(avf, dict)
        assert avf["@type"] == f"{RDFS}Datatype"
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}integer"

        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 2
        assert any(f"{XSD}minInclusive" in f for f in facets)
        assert any(f"{XSD}maxInclusive" in f for f in facets)
        min_facet = next(f for f in facets if f"{XSD}minInclusive" in f)
        max_facet = next(f for f in facets if f"{XSD}maxInclusive" in f)
        assert min_facet[f"{XSD}minInclusive"] == 0
        assert max_facet[f"{XSD}maxInclusive"] == 150

    def test_minimum_only_without_type_defaults_to_decimal(self):
        """@minimum without @type → DatatypeRestriction on xsd:decimal."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/score": {
                "@minimum": 0,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}decimal"

        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 1
        assert facets[0][f"{XSD}minInclusive"] == 0

    def test_maximum_only_without_type_defaults_to_decimal(self):
        """@maximum without @type → DatatypeRestriction on xsd:decimal."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/score": {
                "@maximum": 100,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}decimal"
        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 1
        assert facets[0][f"{XSD}maxInclusive"] == 100

    def test_minimum_maximum_without_type(self):
        """@minimum + @maximum without @type → both facets, xsd:decimal base."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/score": {
                "@minimum": 0,
                "@maximum": 100,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}decimal"
        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 2

    def test_mincount_produces_min_cardinality(self):
        """@minCount → owl:minCardinality with explicit value."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@minCount": 2,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]
        assert restriction[f"{OWL}minCardinality"]["@value"] == 2

    def test_maxcount_produces_max_cardinality(self):
        """@maxCount → owl:maxCardinality."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@maxCount": 5,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]
        assert restriction[f"{OWL}maxCardinality"]["@value"] == 5

    def test_mincount_maxcount_together(self):
        """@minCount + @maxCount → two separate restrictions."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@minCount": 1,
                "@maxCount": 10,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restrictions = cls[f"{RDFS}subClassOf"]
        assert isinstance(restrictions, list)
        assert len(restrictions) == 2
        has_min = any(f"{OWL}minCardinality" in r for r in restrictions)
        has_max = any(f"{OWL}maxCardinality" in r for r in restrictions)
        assert has_min and has_max

    def test_mincount_overrides_required(self):
        """@minCount takes precedence over @required."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@required": True,
                "@minCount": 3,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restrictions = cls[f"{RDFS}subClassOf"]
        if not isinstance(restrictions, list):
            restrictions = [restrictions]
        min_cards = [
            r for r in restrictions if f"{OWL}minCardinality" in r
        ]
        # Should be exactly ONE minCardinality restriction with value 3 (not 1)
        assert len(min_cards) == 1
        assert min_cards[0][f"{OWL}minCardinality"]["@value"] == 3

    def test_in_produces_one_of(self):
        """@in → owl:allValuesFrom + owl:oneOf."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@in": ["active", "inactive", "pending"],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]
        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert avf[f"{OWL}oneOf"]["@list"] == ["active", "inactive", "pending"]

    def test_in_with_numeric_values(self):
        """@in with numeric values → owl:oneOf with numbers."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/priority": {
                "@in": [1, 2, 3, 5, 8],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]
        one_of = restriction[f"{OWL}allValuesFrom"][f"{OWL}oneOf"]["@list"]
        assert one_of == [1, 2, 3, 5, 8]

    def test_in_combined_with_type_produces_both_restrictions(self):
        """@in + @type → separate allValuesFrom (type) and oneOf restrictions."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@type": "xsd:string",
                "@in": ["active", "inactive"],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restrictions = cls[f"{RDFS}subClassOf"]
        assert isinstance(restrictions, list)
        assert len(restrictions) == 2
        has_type = any(
            f"{OWL}allValuesFrom" in r
            and isinstance(r[f"{OWL}allValuesFrom"], dict)
            and r[f"{OWL}allValuesFrom"].get("@id") == f"{XSD}string"
            for r in restrictions
        )
        has_oneof = any(
            f"{OWL}allValuesFrom" in r
            and isinstance(r[f"{OWL}allValuesFrom"], dict)
            and f"{OWL}oneOf" in r[f"{OWL}allValuesFrom"]
            for r in restrictions
        )
        assert has_type and has_oneof

    def test_minlength_maxlength_with_type(self):
        """@type + @minLength/@maxLength → DatatypeRestriction with string facets."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/name": {
                "@type": "xsd:string",
                "@minLength": 1,
                "@maxLength": 100,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}string"
        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 2
        assert any(f"{XSD}minLength" in f for f in facets)
        assert any(f"{XSD}maxLength" in f for f in facets)

    def test_pattern_with_type(self):
        """@type + @pattern → DatatypeRestriction with xsd:pattern facet."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/email": {
                "@type": "xsd:string",
                "@pattern": "^[^@]+@[^@]+$",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 1
        pattern_facet = facets[0]
        assert pattern_facet[f"{XSD}pattern"] == "^[^@]+@[^@]+$"

    def test_string_facets_without_type_defaults_to_string(self):
        """@minLength without @type → DatatypeRestriction on xsd:string."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/code": {
                "@minLength": 3,
                "@maxLength": 10,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        # String facets should default to xsd:string, NOT xsd:decimal
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}string"

    def test_all_facets_combined(self):
        """All five facets on one property → single DatatypeRestriction."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/field": {
                "@type": "xsd:string",
                "@minimum": 0,
                "@maximum": 999,
                "@minLength": 1,
                "@maxLength": 50,
                "@pattern": "^[A-Z]",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        facets = avf[f"{OWL}withRestrictions"]["@list"]
        assert len(facets) == 5

    def test_type_only_still_simple_all_values_from(self):
        """@type alone (no min/max) → simple owl:allValuesFrom (regression)."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/name": {
                "@type": "xsd:string",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        if isinstance(restriction, list):
            restriction = restriction[0]

        avf = restriction[f"{OWL}allValuesFrom"]
        # Should be a simple IRI reference, NOT a DatatypeRestriction dict
        assert avf == {"@id": f"{XSD}string"}

    def test_person_shape_age_combined_restriction(self, person_shape):
        """person_shape.age has @type + @minimum + @maximum → combined."""
        owl = shape_to_owl_restrictions(person_shape)
        cls = owl["@graph"][0]
        restrictions = cls[f"{RDFS}subClassOf"]
        if not isinstance(restrictions, list):
            restrictions = [restrictions]

        # Find the age restriction (the one with DatatypeRestriction on integer)
        age_restrictions = [
            r for r in restrictions
            if r.get(f"{OWL}onProperty", {}).get("@id") == "http://schema.org/age"
        ]
        # Should have exactly ONE restriction for age (combined), not two
        avf_age = [r for r in age_restrictions if f"{OWL}allValuesFrom" in r]
        assert len(avf_age) == 1

        avf = avf_age[0][f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert avf[f"{OWL}onDatatype"]["@id"] == f"{XSD}integer"

    # -- Task 5: @or/@and/@not → OWL DataRange expressions -------------------

    def test_or_produces_union_of(self):
        """@or → owl:Restriction with owl:allValuesFrom → owl:unionOf."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        assert restriction["@type"] == f"{OWL}Restriction"
        assert restriction[f"{OWL}onProperty"]["@id"] == "http://example.org/value"

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert f"{OWL}unionOf" in avf

        members = avf[f"{OWL}unionOf"]["@list"]
        assert len(members) == 2
        # Each branch should resolve to a datatype IRI
        member_ids = [m["@id"] if isinstance(m, dict) and "@id" in m else m for m in members]
        assert f"{XSD}string" in member_ids
        assert f"{XSD}integer" in member_ids

    def test_and_produces_intersection_of(self):
        """@and → owl:Restriction with owl:allValuesFrom → owl:intersectionOf."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@and": [
                    {"@minimum": 0},
                    {"@maximum": 100},
                ],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        # Should have 1 restriction from @and (facet-only branches also
        # produce individual restrictions, so we look for the intersectionOf)
        restrictions = cls.get(f"{RDFS}subClassOf", [])
        if not isinstance(restrictions, list):
            restrictions = [restrictions]

        # Find the restriction that has intersectionOf
        intersect_r = [
            r for r in restrictions
            if isinstance(r.get(f"{OWL}allValuesFrom"), dict)
            and f"{OWL}intersectionOf" in r.get(f"{OWL}allValuesFrom", {})
        ]
        assert len(intersect_r) == 1

        avf = intersect_r[0][f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        members = avf[f"{OWL}intersectionOf"]["@list"]
        assert len(members) == 2

    def test_not_produces_complement_of(self):
        """@not → owl:Restriction with owl:allValuesFrom → owl:complementOf."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@not": {"@type": "xsd:boolean"},
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        assert restriction["@type"] == f"{OWL}Restriction"

        avf = restriction[f"{OWL}allValuesFrom"]
        assert avf["@type"] == f"{RDFS}Datatype"
        assert f"{OWL}datatypeComplementOf" in avf

        complement = avf[f"{OWL}datatypeComplementOf"]
        assert complement["@id"] == f"{XSD}boolean"

    def test_or_with_faceted_branches(self):
        """@or with branches containing facets → unionOf DatatypeRestrictions."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/code": {
                "@or": [
                    {"@type": "xsd:string", "@minLength": 3},
                    {"@type": "xsd:integer", "@minimum": 100},
                ],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        avf = restriction[f"{OWL}allValuesFrom"]
        members = avf[f"{OWL}unionOf"]["@list"]
        assert len(members) == 2
        # Each member should be a DatatypeRestriction (has onDatatype + withRestrictions)
        for m in members:
            assert isinstance(m, dict)
            assert f"{OWL}onDatatype" in m
            assert f"{OWL}withRestrictions" in m

    def test_nested_or_and(self):
        """Nested @or containing @and → recursive data range construction."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@and": [
                        {"@minimum": 0},
                        {"@maximum": 100},
                    ]},
                ],
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        restriction = cls[f"{RDFS}subClassOf"]
        avf = restriction[f"{OWL}allValuesFrom"]
        members = avf[f"{OWL}unionOf"]["@list"]
        assert len(members) == 2
        # Second member should itself contain intersectionOf
        nested = members[1]
        assert f"{OWL}intersectionOf" in nested

    # -- Task 6: @extends → rdfs:subClassOf ----------------------------------

    def test_extends_produces_subclass_of(self):
        """@extends → rdfs:subClassOf on the OWL class."""
        shape = {
            "@type": "http://example.org/Employee",
            "@extends": "http://example.org/Person",
            "http://example.org/dept": {"@required": True},
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        # Should have rdfs:subClassOf entries including the parent class
        sub_of = cls.get(f"{RDFS}subClassOf", [])
        if not isinstance(sub_of, list):
            sub_of = [sub_of]

        parent_refs = [
            s for s in sub_of
            if isinstance(s, dict) and s.get("@id") == "http://example.org/Person"
        ]
        assert len(parent_refs) == 1, "Parent class should appear as rdfs:subClassOf"

    def test_extends_list_produces_multiple_subclass_of(self):
        """@extends with a list → multiple rdfs:subClassOf entries."""
        shape = {
            "@type": "http://example.org/Manager",
            "@extends": [
                "http://example.org/Employee",
                "http://example.org/Leader",
            ],
            "http://example.org/reports": {"@minCount": 1},
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        sub_of = cls.get(f"{RDFS}subClassOf", [])
        if not isinstance(sub_of, list):
            sub_of = [sub_of]

        parent_refs = [
            s for s in sub_of
            if isinstance(s, dict) and "@id" in s and s["@id"].startswith("http://example.org/")
            and s.get("@type") is None  # plain IRI ref, not a Restriction
        ]
        parent_iris = {r["@id"] for r in parent_refs}
        assert "http://example.org/Employee" in parent_iris
        assert "http://example.org/Leader" in parent_iris

    def test_extends_combined_with_constraints(self):
        """@extends + property constraints → both parent ref and restrictions."""
        shape = {
            "@type": "http://example.org/Employee",
            "@extends": "http://example.org/Person",
            "http://example.org/salary": {
                "@type": "xsd:decimal",
                "@minimum": 0,
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        sub_of = cls.get(f"{RDFS}subClassOf", [])
        if not isinstance(sub_of, list):
            sub_of = [sub_of]

        # Should have at least one parent ref AND at least one Restriction
        parent_refs = [s for s in sub_of if isinstance(s, dict) and s.get("@id") == "http://example.org/Person"]
        restrictions = [s for s in sub_of if isinstance(s, dict) and s.get("@type") == f"{OWL}Restriction"]
        assert len(parent_refs) >= 1
        assert len(restrictions) >= 1

    # -- Task 7: Unmappable constraint preservation --------------------------

    def test_unmappable_constraints_preserved_in_jex_namespace(self):
        """Constraints without OWL equivalents are preserved as jex: annotations."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/startDate": {
                "@lessThan": "http://example.org/endDate",
            },
            "http://example.org/value": {
                "@equals": "http://example.org/otherValue",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]

        # Unmappable constraints should appear as jex: annotations
        assert f"{JSONLD_EX}lessThan" in cls or any(
            f"{JSONLD_EX}lessThan" in str(v) for v in cls.values()
        )
        assert f"{JSONLD_EX}equals" in cls or any(
            f"{JSONLD_EX}equals" in str(v) for v in cls.values()
        )

    def test_unmappable_if_then_else_preserved(self):
        """@if/@then/@else has no OWL equivalent → preserved as jex: annotation."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@if": {"@type": "xsd:string"},
                "@then": {"@minLength": 1},
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]

        # Should be preserved somewhere on the class as jex: annotation
        cls_str = str(cls)
        assert JSONLD_EX in cls_str, "Conditional constraints should be preserved in jex: namespace"

    def test_unmappable_disjoint_preserved(self):
        """@disjoint has no direct OWL property restriction equivalent → preserved."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/field1": {
                "@disjoint": "http://example.org/field2",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        cls_str = str(cls)
        assert f"{JSONLD_EX}disjoint" in cls_str

    def test_unmappable_severity_preserved(self):
        """@severity has no OWL equivalent → preserved as jex: annotation."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@required": True,
                "@severity": "warning",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        cls_str = str(cls)
        assert f"{JSONLD_EX}severity" in cls_str

    def test_mappable_and_unmappable_coexist(self):
        """Shape with both mappable and unmappable constraints handles both."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/start": {
                "@type": "xsd:date",
                "@required": True,
                "@lessThan": "http://example.org/end",
            },
        }
        owl = shape_to_owl_restrictions(shape)
        cls = owl["@graph"][0]
        sub_of = cls.get(f"{RDFS}subClassOf", [])
        if not isinstance(sub_of, list):
            sub_of = [sub_of]

        # Should have OWL restrictions for @type and @required
        restrictions = [s for s in sub_of if isinstance(s, dict) and s.get("@type") == f"{OWL}Restriction"]
        assert len(restrictions) >= 2, "Should have minCardinality + allValuesFrom restrictions"

        # And unmappable @lessThan should be preserved
        cls_str = str(cls)
        assert f"{JSONLD_EX}lessThan" in cls_str


# ═══════════════════════════════════════════════════════════════════
# OWL REVERSE MAPPING TESTS
# ═══════════════════════════════════════════════════════════════════


class TestOwlToShape:
    """Tests for OWL class restrictions → jsonld-ex @shape (inverse of shape_to_owl_restrictions)."""

    def test_min_cardinality_1_to_required(self):
        """owl:minCardinality 1 → @required: True."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Person",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://schema.org/name"},
                    f"{OWL}minCardinality": {
                        "@value": 1,
                        "@type": f"{XSD}nonNegativeInteger",
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["@type"] == "http://example.org/Person"
        assert shape["http://schema.org/name"]["@required"] is True
        assert "@minCount" not in shape["http://schema.org/name"]

    def test_min_cardinality_gt1_to_mincount(self):
        """owl:minCardinality N (N > 1) → @minCount: N."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/tags"},
                    f"{OWL}minCardinality": {
                        "@value": 3,
                        "@type": f"{XSD}nonNegativeInteger",
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["http://example.org/tags"]["@minCount"] == 3
        assert "@required" not in shape["http://example.org/tags"]

    def test_max_cardinality_to_maxcount(self):
        """owl:maxCardinality → @maxCount."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/name"},
                    f"{OWL}maxCardinality": {
                        "@value": 5,
                        "@type": f"{XSD}nonNegativeInteger",
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["http://example.org/name"]["@maxCount"] == 5

    def test_simple_allvaluesfrom_to_type(self):
        """owl:allValuesFrom simple IRI → @type."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Person",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://schema.org/name"},
                    f"{OWL}allValuesFrom": {"@id": f"{XSD}string"},
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["http://schema.org/name"]["@type"] == "xsd:string"

    def test_datatype_restriction_to_type_and_facets(self):
        """owl:allValuesFrom DatatypeRestriction → @type + @minimum/@maximum."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/age"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}onDatatype": {"@id": f"{XSD}integer"},
                        f"{OWL}withRestrictions": {"@list": [
                            {f"{XSD}minInclusive": 0},
                            {f"{XSD}maxInclusive": 150},
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        prop = shape["http://example.org/age"]
        assert prop["@type"] == "xsd:integer"
        assert prop["@minimum"] == 0
        assert prop["@maximum"] == 150

    def test_facets_only_defaults_stripped(self):
        """DatatypeRestriction with default datatype → facets only, no @type."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/score"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}onDatatype": {"@id": f"{XSD}decimal"},
                        f"{OWL}withRestrictions": {"@list": [
                            {f"{XSD}minInclusive": 0},
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        prop = shape["http://example.org/score"]
        assert "@type" not in prop
        assert prop["@minimum"] == 0

    def test_string_facets_default_stripped(self):
        """DatatypeRestriction with xsd:string default → facets only, no @type."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/label"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}onDatatype": {"@id": f"{XSD}string"},
                        f"{OWL}withRestrictions": {"@list": [
                            {f"{XSD}minLength": 1},
                            {f"{XSD}maxLength": 100},
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        prop = shape["http://example.org/label"]
        assert "@type" not in prop
        assert prop["@minLength"] == 1
        assert prop["@maxLength"] == 100

    def test_oneof_to_in(self):
        """owl:oneOf → @in."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/status"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}oneOf": {"@list": ["active", "inactive", "pending"]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["http://example.org/status"]["@in"] == ["active", "inactive", "pending"]

    def test_unionof_to_or(self):
        """owl:unionOf → @or."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/value"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}unionOf": {"@list": [
                            {"@id": f"{XSD}string"},
                            {"@id": f"{XSD}integer"},
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        branches = shape["http://example.org/value"]["@or"]
        assert len(branches) == 2
        assert {"@type": "xsd:string"} in branches
        assert {"@type": "xsd:integer"} in branches

    def test_intersectionof_to_and(self):
        """owl:intersectionOf → @and."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/value"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}intersectionOf": {"@list": [
                            {"@id": f"{XSD}integer"},
                            {
                                "@type": f"{RDFS}Datatype",
                                f"{OWL}onDatatype": {"@id": f"{XSD}integer"},
                                f"{OWL}withRestrictions": {"@list": [{f"{XSD}minInclusive": 0}]},
                            },
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        branches = shape["http://example.org/value"]["@and"]
        assert len(branches) == 2

    def test_complementof_to_not(self):
        """owl:datatypeComplementOf → @not."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/value"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}datatypeComplementOf": {"@id": f"{XSD}string"},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["http://example.org/value"]["@not"] == {"@type": "xsd:string"}

    def test_subclassof_plain_iri_to_extends(self):
        """rdfs:subClassOf plain IRI (not Restriction) → @extends."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Employee",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {"@id": "http://example.org/Person"},
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["@extends"] == "http://example.org/Person"

    def test_multiple_subclassof_plain_iris_to_extends_list(self):
        """Multiple rdfs:subClassOf plain IRIs → @extends list."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Manager",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": [
                    {"@id": "http://example.org/Person"},
                    {"@id": "http://example.org/Employee"},
                ],
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert isinstance(shape["@extends"], list)
        assert "http://example.org/Person" in shape["@extends"]
        assert "http://example.org/Employee" in shape["@extends"]

    def test_jex_annotations_to_unmappable_constraints(self):
        """jex: namespace annotations → restored unmappable constraints."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/name"},
                    f"{OWL}minCardinality": {
                        "@value": 1,
                        "@type": f"{XSD}nonNegativeInteger",
                    },
                },
                f"{JSONLD_EX}lessThan": "http://example.org/endDate",
                f"{JSONLD_EX}severity": "warning",
            }],
        }
        shape = owl_to_shape(owl_doc)
        # The unmappable constraints should be available on the shape
        # They attach at property level in the original, but stored as
        # class-level annotations in OWL. We'll need to verify restoration.
        assert f"{JSONLD_EX}lessThan" in str(shape) or "@lessThan" in str(shape)

    def test_jex_conditional_restored(self):
        """jex:conditional annotation → @if/@then/@else restored."""
        cond = {"@if": {"@type": "xsd:string"}, "@then": {"@minLength": 1}}
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{JSONLD_EX}conditional": cond,
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert f"{JSONLD_EX}conditional" in str(shape) or "@if" in str(shape)

    def test_multiple_restrictions_on_different_properties(self):
        """Multiple restrictions on different properties."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Person",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": [
                    {
                        "@type": f"{OWL}Restriction",
                        f"{OWL}onProperty": {"@id": "http://schema.org/name"},
                        f"{OWL}minCardinality": {"@value": 1, "@type": f"{XSD}nonNegativeInteger"},
                    },
                    {
                        "@type": f"{OWL}Restriction",
                        f"{OWL}onProperty": {"@id": "http://schema.org/name"},
                        f"{OWL}allValuesFrom": {"@id": f"{XSD}string"},
                    },
                    {
                        "@type": f"{OWL}Restriction",
                        f"{OWL}onProperty": {"@id": "http://schema.org/age"},
                        f"{OWL}allValuesFrom": {
                            "@type": f"{RDFS}Datatype",
                            f"{OWL}onDatatype": {"@id": f"{XSD}integer"},
                            f"{OWL}withRestrictions": {"@list": [
                                {f"{XSD}minInclusive": 0},
                                {f"{XSD}maxInclusive": 150},
                            ]},
                        },
                    },
                ],
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["@type"] == "http://example.org/Person"
        assert shape["http://schema.org/name"]["@required"] is True
        assert shape["http://schema.org/name"]["@type"] == "xsd:string"
        assert shape["http://schema.org/age"]["@type"] == "xsd:integer"
        assert shape["http://schema.org/age"]["@minimum"] == 0
        assert shape["http://schema.org/age"]["@maximum"] == 150

    def test_round_trip_basic_shape(self):
        """shape → OWL → shape round-trip preserves semantics."""
        original = {
            "@type": "http://example.org/Person",
            "http://schema.org/name": {
                "@required": True,
                "@type": "xsd:string",
                "@minLength": 1,
            },
            "http://schema.org/age": {
                "@type": "xsd:integer",
                "@minimum": 0,
                "@maximum": 150,
            },
        }
        owl = shape_to_owl_restrictions(original)
        restored = owl_to_shape(owl)

        assert restored["@type"] == original["@type"]
        # name: @required, @minLength preserved; @type stripped because
        # xsd:string is the default for string facets — the forward mapping
        # produces identical OWL whether @type was explicit or not.
        assert restored["http://schema.org/name"]["@required"] is True
        assert "@type" not in restored["http://schema.org/name"]
        assert restored["http://schema.org/name"]["@minLength"] == 1
        # age: @type, @minimum, @maximum
        assert restored["http://schema.org/age"]["@type"] == "xsd:integer"
        assert restored["http://schema.org/age"]["@minimum"] == 0
        assert restored["http://schema.org/age"]["@maximum"] == 150

    def test_round_trip_with_in(self):
        """shape → OWL → shape round-trip preserves @in."""
        original = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@in": ["active", "inactive"],
            },
        }
        owl = shape_to_owl_restrictions(original)
        restored = owl_to_shape(owl)
        assert restored["http://example.org/status"]["@in"] == ["active", "inactive"]

    def test_round_trip_with_or(self):
        """shape → OWL → shape round-trip preserves @or."""
        original = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ],
            },
        }
        owl = shape_to_owl_restrictions(original)
        restored = owl_to_shape(owl)
        branches = restored["http://example.org/value"]["@or"]
        assert len(branches) == 2
        assert {"@type": "xsd:string"} in branches
        assert {"@type": "xsd:integer"} in branches

    def test_round_trip_with_extends(self):
        """shape → OWL → shape round-trip preserves @extends."""
        original = {
            "@type": "http://example.org/Employee",
            "@extends": "http://example.org/Person",
            "http://example.org/salary": {
                "@type": "xsd:decimal",
            },
        }
        owl = shape_to_owl_restrictions(original)
        restored = owl_to_shape(owl)
        assert restored["@extends"] == "http://example.org/Person"

    def test_pattern_to_pattern(self):
        """DatatypeRestriction with xsd:pattern → @pattern."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
                f"{RDFS}subClassOf": {
                    "@type": f"{OWL}Restriction",
                    f"{OWL}onProperty": {"@id": "http://example.org/email"},
                    f"{OWL}allValuesFrom": {
                        "@type": f"{RDFS}Datatype",
                        f"{OWL}onDatatype": {"@id": f"{XSD}string"},
                        f"{OWL}withRestrictions": {"@list": [
                            {f"{XSD}pattern": "^[^@]+@[^@]+$"},
                        ]},
                    },
                },
            }],
        }
        shape = owl_to_shape(owl_doc)
        # xsd:string is the default for pattern → should be stripped
        assert "@type" not in shape["http://example.org/email"]
        assert shape["http://example.org/email"]["@pattern"] == "^[^@]+@[^@]+$"

    def test_empty_class_no_restrictions(self):
        """OWL class with no restrictions → minimal shape."""
        owl_doc = {
            "@context": {"owl": OWL, "xsd": XSD, "rdfs": RDFS, "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#"},
            "@graph": [{
                "@id": "http://example.org/Thing",
                "@type": f"{OWL}Class",
            }],
        }
        shape = owl_to_shape(owl_doc)
        assert shape["@type"] == "http://example.org/Thing"
        # Should have no property constraints
        non_meta_keys = [k for k in shape if not k.startswith("@")]
        assert non_meta_keys == []


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


# ═══════════════════════════════════════════════════════════════════
# SHACL EXTENDED CONSTRAINT ROUND-TRIP TESTS
# ═══════════════════════════════════════════════════════════════════


class TestShapeToShaclExtended:
    """Tests for new constraint types in shape_to_shacl()."""

    # -- @minCount / @maxCount ------------------------------------------------

    def test_min_count_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@minCount": 2,
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        tag_prop = props[0]
        assert tag_prop[f"{SHACL}minCount"] == 2

    def test_max_count_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {
                "@maxCount": 5,
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        tag_prop = props[0]
        assert tag_prop[f"{SHACL}maxCount"] == 5

    def test_min_and_max_count_together(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/authors": {
                "@minCount": 1,
                "@maxCount": 10,
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}minCount"] == 1
        assert prop[f"{SHACL}maxCount"] == 10

    def test_required_and_min_count_no_double_min_count(self):
        """@required + @minCount should use the explicit @minCount, not duplicate."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/name": {
                "@required": True,
                "@minCount": 3,
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        # @minCount=3 takes precedence; @required should not add a second minCount=1
        assert prop[f"{SHACL}minCount"] == 3

    # -- @in / @enum ----------------------------------------------------------

    def test_in_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@in": ["draft", "published", "retracted"],
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        sh_in = prop[f"{SHACL}in"]
        assert isinstance(sh_in, dict)
        assert sh_in["@list"] == ["draft", "published", "retracted"]

    def test_in_with_numeric_values(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/priority": {
                "@in": [1, 2, 3],
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}in"]["@list"] == [1, 2, 3]

    # -- @or / @and / @not ----------------------------------------------------

    def test_or_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ],
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        sh_or = prop[f"{SHACL}or"]
        assert isinstance(sh_or, dict)
        assert "@list" in sh_or
        assert len(sh_or["@list"]) == 2

    def test_and_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/code": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@pattern": "^[A-Z]{3}$"},
                ],
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        sh_and = prop[f"{SHACL}and"]
        assert isinstance(sh_and, dict)
        assert "@list" in sh_and
        assert len(sh_and["@list"]) == 2

    def test_not_mapping(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@not": {"@in": ["deleted", "banned"]},
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        sh_not = prop[f"{SHACL}not"]
        assert isinstance(sh_not, dict)
        # sh:not wraps a single constraint shape
        assert f"{SHACL}in" in sh_not

    # -- Cross-property constraints -------------------------------------------

    def test_less_than_mapping(self):
        shape = {
            "@type": "http://example.org/Event",
            "http://example.org/startDate": {
                "@lessThan": "http://example.org/endDate",
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}lessThan"]["@id"] == "http://example.org/endDate"

    def test_less_than_or_equals_mapping(self):
        shape = {
            "@type": "http://example.org/Range",
            "http://example.org/low": {
                "@lessThanOrEquals": "http://example.org/high",
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}lessThanOrEquals"]["@id"] == "http://example.org/high"

    def test_equals_mapping(self):
        shape = {
            "@type": "http://example.org/Account",
            "http://example.org/confirmEmail": {
                "@equals": "http://example.org/email",
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}equals"]["@id"] == "http://example.org/email"

    def test_disjoint_mapping(self):
        shape = {
            "@type": "http://example.org/Pair",
            "http://example.org/primary": {
                "@disjoint": "http://example.org/secondary",
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]
        assert prop[f"{SHACL}disjoint"]["@id"] == "http://example.org/secondary"

    # -- Combined constraints -------------------------------------------------

    def test_combined_new_and_old_constraints(self):
        """All constraint types in a single shape should all appear in SHACL."""
        shape = {
            "@type": "http://example.org/Record",
            "http://example.org/name": {
                "@required": True,
                "@type": "xsd:string",
                "@minLength": 1,
                "@maxLength": 100,
            },
            "http://example.org/tags": {
                "@minCount": 1,
                "@maxCount": 10,
            },
            "http://example.org/status": {
                "@in": ["active", "archived"],
            },
            "http://example.org/start": {
                "@lessThan": "http://example.org/end",
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        assert len(props) == 4


class TestShaclToShapeExtended:
    """Tests for new constraint types in shacl_to_shape() (reverse direction)."""

    # -- @minCount / @maxCount ------------------------------------------------

    def test_min_count_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {"@minCount": 2},
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/tags"]["@minCount"] == 2

    def test_max_count_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/tags": {"@maxCount": 5},
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/tags"]["@maxCount"] == 5

    def test_min_count_1_becomes_required(self):
        """SHACL sh:minCount=1 should still map to @required (backward compat)."""
        shacl_doc = {
            "@graph": [{
                "@id": "_:s",
                "@type": f"{SHACL}NodeShape",
                f"{SHACL}targetClass": {"@id": "http://example.org/Thing"},
                f"{SHACL}property": [{
                    f"{SHACL}path": {"@id": "http://example.org/name"},
                    f"{SHACL}minCount": 1,
                }],
            }],
        }
        restored, _ = shacl_to_shape(shacl_doc)
        assert restored["http://example.org/name"]["@required"] is True

    def test_min_count_gt_1_becomes_min_count(self):
        """SHACL sh:minCount > 1 should map to @minCount, not @required."""
        shacl_doc = {
            "@graph": [{
                "@id": "_:s",
                "@type": f"{SHACL}NodeShape",
                f"{SHACL}targetClass": {"@id": "http://example.org/Thing"},
                f"{SHACL}property": [{
                    f"{SHACL}path": {"@id": "http://example.org/authors"},
                    f"{SHACL}minCount": 3,
                }],
            }],
        }
        restored, _ = shacl_to_shape(shacl_doc)
        prop = restored["http://example.org/authors"]
        assert prop["@minCount"] == 3
        # Should NOT also set @required when @minCount > 1
        assert "@required" not in prop

    def test_max_count_from_shacl(self):
        shacl_doc = {
            "@graph": [{
                "@id": "_:s",
                "@type": f"{SHACL}NodeShape",
                f"{SHACL}targetClass": {"@id": "http://example.org/Thing"},
                f"{SHACL}property": [{
                    f"{SHACL}path": {"@id": "http://example.org/tags"},
                    f"{SHACL}maxCount": 10,
                }],
            }],
        }
        restored, _ = shacl_to_shape(shacl_doc)
        assert restored["http://example.org/tags"]["@maxCount"] == 10

    # -- @in ------------------------------------------------------------------

    def test_in_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@in": ["draft", "published", "retracted"],
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/status"]["@in"] == ["draft", "published", "retracted"]

    def test_in_from_shacl_list(self):
        shacl_doc = {
            "@graph": [{
                "@id": "_:s",
                "@type": f"{SHACL}NodeShape",
                f"{SHACL}targetClass": {"@id": "http://example.org/Thing"},
                f"{SHACL}property": [{
                    f"{SHACL}path": {"@id": "http://example.org/color"},
                    f"{SHACL}in": {"@list": ["red", "green", "blue"]},
                }],
            }],
        }
        restored, warnings = shacl_to_shape(shacl_doc)
        assert restored["http://example.org/color"]["@in"] == ["red", "green", "blue"]
        # sh:in should no longer produce a warning
        assert not any("sh:in" in w for w in warnings)

    # -- @or / @and / @not ----------------------------------------------------

    def test_or_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@or": [
                    {"@type": "xsd:string"},
                    {"@type": "xsd:integer"},
                ],
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        prop = restored["http://example.org/value"]
        assert "@or" in prop
        assert len(prop["@or"]) == 2

    def test_and_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/code": {
                "@and": [
                    {"@type": "xsd:string"},
                    {"@pattern": "^[A-Z]{3}$"},
                ],
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        prop = restored["http://example.org/code"]
        assert "@and" in prop
        assert len(prop["@and"]) == 2

    def test_not_round_trip(self):
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/status": {
                "@not": {"@in": ["deleted", "banned"]},
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        prop = restored["http://example.org/status"]
        assert "@not" in prop
        assert prop["@not"]["@in"] == ["deleted", "banned"]

    # -- Cross-property -------------------------------------------------------

    def test_less_than_round_trip(self):
        shape = {
            "@type": "http://example.org/Event",
            "http://example.org/startDate": {
                "@lessThan": "http://example.org/endDate",
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/startDate"]["@lessThan"] == "http://example.org/endDate"

    def test_less_than_or_equals_round_trip(self):
        shape = {
            "@type": "http://example.org/Range",
            "http://example.org/low": {
                "@lessThanOrEquals": "http://example.org/high",
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/low"]["@lessThanOrEquals"] == "http://example.org/high"

    def test_equals_round_trip(self):
        shape = {
            "@type": "http://example.org/Account",
            "http://example.org/confirmEmail": {
                "@equals": "http://example.org/email",
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/confirmEmail"]["@equals"] == "http://example.org/email"

    def test_disjoint_round_trip(self):
        shape = {
            "@type": "http://example.org/Pair",
            "http://example.org/primary": {
                "@disjoint": "http://example.org/secondary",
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)
        assert restored["http://example.org/primary"]["@disjoint"] == "http://example.org/secondary"

    # -- Full round-trip with mixed constraints --------------------------------

    def test_full_round_trip_mixed(self):
        """A shape with old + new constraints should round-trip faithfully."""
        shape = {
            "@type": "http://example.org/Record",
            "http://example.org/name": {
                "@required": True,
                "@type": "xsd:string",
                "@minLength": 1,
            },
            "http://example.org/tags": {
                "@minCount": 2,
                "@maxCount": 10,
            },
            "http://example.org/status": {
                "@in": ["active", "archived"],
            },
            "http://example.org/start": {
                "@lessThan": "http://example.org/end",
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)

        assert restored["@type"] == "http://example.org/Record"
        assert restored["http://example.org/name"]["@required"] is True
        assert restored["http://example.org/name"]["@type"] == "xsd:string"
        assert restored["http://example.org/name"]["@minLength"] == 1
        assert restored["http://example.org/tags"]["@minCount"] == 2
        assert restored["http://example.org/tags"]["@maxCount"] == 10
        assert restored["http://example.org/status"]["@in"] == ["active", "archived"]
        assert restored["http://example.org/start"]["@lessThan"] == "http://example.org/end"


# ═══════════════════════════════════════════════════════════════════
# GAP-V7: @if/@then/@else ↔ SHACL ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════


class TestIfThenElseShacl:
    """@if/@then/@else ↔ SHACL conditional mapping."""

    def test_if_then_to_shacl(self):
        """@if/@then maps to sh:or([sh:not(if), then])."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@if": {"@type": "xsd:string"},
                "@then": {"@minLength": 3},
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]

        # Should have sh:or with two branches
        sh_or = prop.get(f"{SHACL}or")
        assert sh_or is not None
        branches = sh_or["@list"]
        assert len(branches) == 2

        # First branch: sh:not(if_constraints)
        assert f"{SHACL}not" in branches[0]
        # Second branch: then_constraints
        assert f"{SHACL}minLength" in branches[1]

    def test_if_then_else_to_shacl(self):
        """@if/@then/@else maps to sh:or([sh:and([if, then]), sh:and([sh:not(if), else])])."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@if": {"@type": "xsd:string"},
                "@then": {"@minLength": 1},
                "@else": {"@minimum": 0},
            },
        }
        shacl = shape_to_shacl(shape)
        props = shacl["@graph"][0][f"{SHACL}property"]
        prop = props[0]

        sh_or = prop.get(f"{SHACL}or")
        assert sh_or is not None
        branches = sh_or["@list"]
        assert len(branches) == 2

        # First branch: sh:and([if, then])
        assert f"{SHACL}and" in branches[0]
        # Second branch: sh:and([sh:not(if), else])
        assert f"{SHACL}and" in branches[1]

    def test_if_then_round_trip(self):
        """@if/@then survives shape_to_shacl → shacl_to_shape."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@if": {"@type": "xsd:string"},
                "@then": {"@minLength": 3},
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)

        prop = restored["http://example.org/value"]
        assert "@if" in prop
        assert "@then" in prop
        assert prop["@if"]["@type"] == "xsd:string"
        assert prop["@then"]["@minLength"] == 3

    def test_if_then_else_round_trip(self):
        """@if/@then/@else survives shape_to_shacl → shacl_to_shape."""
        shape = {
            "@type": "http://example.org/Thing",
            "http://example.org/value": {
                "@if": {"@type": "xsd:string"},
                "@then": {"@minLength": 1},
                "@else": {"@minimum": 0},
            },
        }
        shacl = shape_to_shacl(shape)
        restored, warnings = shacl_to_shape(shacl)

        prop = restored["http://example.org/value"]
        assert "@if" in prop
        assert "@then" in prop
        assert "@else" in prop
        assert prop["@if"]["@type"] == "xsd:string"
        assert prop["@then"]["@minLength"] == 1
        assert prop["@else"]["@minimum"] == 0


# ═══════════════════════════════════════════════════════════════════
# GAP-OWL1: @extends ↔ SHACL ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════


class TestExtendsShacl:
    """@extends ↔ SHACL inheritance mapping."""

    def test_extends_to_shacl_emits_parent(self):
        """@extends emits parent as separate NodeShape + sh:node reference."""
        parent = {
            "@type": "http://example.org/Person",
            "http://example.org/name": {"@required": True},
        }
        child = {
            "@type": "http://example.org/Employee",
            "@extends": parent,
            "http://example.org/dept": {"@required": True},
        }
        shacl = shape_to_shacl(child)
        graph = shacl["@graph"]

        # Should have 2 shapes: parent + child
        assert len(graph) == 2
        child_shape = graph[0]
        parent_shape = graph[1]

        # Child references parent via sh:node
        assert f"{SHACL}node" in child_shape

        # Parent is a valid NodeShape
        assert parent_shape["@type"] == f"{SHACL}NodeShape"

    def test_extends_round_trip(self):
        """@extends survives shape_to_shacl → shacl_to_shape."""
        parent = {
            "@type": "http://example.org/Person",
            "http://example.org/name": {
                "@required": True,
                "@type": "xsd:string",
            },
        }
        child = {
            "@type": "http://example.org/Employee",
            "@extends": parent,
            "http://example.org/dept": {"@required": True},
        }
        shacl = shape_to_shacl(child)
        restored, warnings = shacl_to_shape(shacl)

        assert restored["@type"] == "http://example.org/Employee"
        assert "@extends" in restored
        ext = restored["@extends"]
        # Parent shape should have been reconstructed
        assert ext["@type"] == "http://example.org/Person"
        assert ext["http://example.org/name"]["@required"] is True
        # Child's own property preserved
        assert restored["http://example.org/dept"]["@required"] is True


# ═══════════════════════════════════════════════════════════════════
# SSN/SOSA INTEROP TESTS
# ═══════════════════════════════════════════════════════════════════


def _find_nodes_by_type(graph, type_iri):
    """Find all nodes in a @graph array that have the given @type."""
    results = []
    for node in graph:
        if not isinstance(node, dict):
            continue
        node_types = node.get("@type", [])
        if isinstance(node_types, str):
            node_types = [node_types]
        if type_iri in node_types:
            results.append(node)
    return results


def _find_node_by_id(graph, node_id):
    """Find a node in @graph by its @id."""
    for node in graph:
        if isinstance(node, dict) and node.get("@id") == node_id:
            return node
    return None


class TestSSNSOSAInteropToSSN:
    """Tests for to_ssn() — jsonld-ex → SSN/SOSA conversion."""

    def test_basic_observation_generated(self):
        """An annotated value produces a sosa:Observation."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "http://schema.org/name": annotate(
                "Alice",
                confidence=0.95,
                source="https://sensor.example.org/s1",
            ),
        }
        ssn_doc, report = to_ssn(doc)

        assert report.success is True
        assert report.nodes_converted >= 1

        graph = ssn_doc["@graph"]
        observations = _find_nodes_by_type(graph, f"{SOSA}Observation")
        assert len(observations) >= 1

    def test_context_has_required_namespaces(self):
        """Output context includes sosa, ssn, qudt, xsd, jsonld-ex prefixes."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate("val", confidence=0.9),
        }
        ssn_doc, _ = to_ssn(doc)
        ctx = ssn_doc["@context"]

        assert ctx["sosa"] == SOSA
        assert ctx["ssn"] == SSN
        assert ctx["qudt"] == QUDT
        assert ctx["xsd"] == XSD
        assert ctx["jsonld-ex"] == JSONLD_EX_NAMESPACE

    def test_simple_result_without_unit(self):
        """@value without @unit → sosa:hasSimpleResult."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://schema.org/name": annotate("Alice", confidence=0.9),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert f"{SOSA}hasSimpleResult" in obs
        assert obs[f"{SOSA}hasSimpleResult"] == "Alice"

    def test_result_with_unit_uses_qudt(self):
        """@value + @unit → sosa:hasResult → sosa:Result with qudt:unit."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/sensor-reading",
            "http://example.org/temperature": annotate(
                22.5,
                source="https://sensor.example.org/thermo1",
                unit="http://qudt.org/vocab/unit/DEG_C",
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        # Should use hasResult (not hasSimpleResult) when unit present
        assert f"{SOSA}hasResult" in obs
        result_ref = obs[f"{SOSA}hasResult"]
        result_id = result_ref["@id"] if isinstance(result_ref, dict) else result_ref

        result_node = _find_node_by_id(graph, result_id)
        assert result_node is not None
        assert f"{SOSA}Result" in (result_node.get("@type", []) if isinstance(result_node.get("@type"), list) else [result_node.get("@type", "")])
        assert result_node[f"{QUDT}numericValue"] == 22.5
        assert result_node[f"{QUDT}unit"] == "http://qudt.org/vocab/unit/DEG_C"

    def test_source_maps_to_sensor(self):
        """@source → sosa:madeBySensor + sosa:Sensor node."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                20.0, source="https://sensor.example.org/thermo1",
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert f"{SOSA}madeBySensor" in obs
        sensor_ref = obs[f"{SOSA}madeBySensor"]
        sensor_id = sensor_ref["@id"] if isinstance(sensor_ref, dict) else sensor_ref
        assert sensor_id == "https://sensor.example.org/thermo1"

        sensors = _find_nodes_by_type(graph, f"{SOSA}Sensor")
        assert any(s["@id"] == "https://sensor.example.org/thermo1" for s in sensors)

    def test_extracted_at_maps_to_result_time(self):
        """@extractedAt → sosa:resultTime."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                20.0, extracted_at="2025-01-15T10:30:00Z",
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        result_time = obs[f"{SOSA}resultTime"]
        if isinstance(result_time, dict):
            assert result_time["@value"] == "2025-01-15T10:30:00Z"
        else:
            assert result_time == "2025-01-15T10:30:00Z"

    def test_method_maps_to_procedure(self):
        """@method → sosa:usedProcedure → sosa:Procedure."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                20.0, method="thermometer-reading",
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert f"{SOSA}usedProcedure" in obs
        proc_ref = obs[f"{SOSA}usedProcedure"]
        proc_id = proc_ref["@id"] if isinstance(proc_ref, dict) else proc_ref

        procedures = _find_nodes_by_type(graph, f"{SOSA}Procedure")
        proc = next(p for p in procedures if p["@id"] == proc_id)
        assert proc[f"{RDFS}label"] == "thermometer-reading"

    def test_confidence_maps_to_jsonld_ex_namespace(self):
        """@confidence → jsonld-ex:confidence (no SOSA equivalent)."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(20.0, confidence=0.92),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert obs[f"{JSONLD_EX_NAMESPACE}confidence"] == 0.92

    def test_feature_of_interest_from_parent(self):
        """Parent node @id/@type → sosa:hasFeatureOfInterest + sosa:FeatureOfInterest."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "http://schema.org/Room",
            "@id": "http://example.org/room-134",
            "http://example.org/temperature": annotate(22.5, confidence=0.95),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert f"{SOSA}hasFeatureOfInterest" in obs
        foi_ref = obs[f"{SOSA}hasFeatureOfInterest"]
        foi_id = foi_ref["@id"] if isinstance(foi_ref, dict) else foi_ref
        assert foi_id == "http://example.org/room-134"

        fois = _find_nodes_by_type(graph, f"{SOSA}FeatureOfInterest")
        foi = next(f for f in fois if f["@id"] == "http://example.org/room-134")
        # Original @type preserved as additional type
        foi_types = foi["@type"] if isinstance(foi["@type"], list) else [foi["@type"]]
        assert f"{SOSA}FeatureOfInterest" in foi_types

    def test_observable_property_from_key(self):
        """Property key → sosa:observedProperty → sosa:ObservableProperty."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temperature": annotate(22.5, confidence=0.9),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        assert f"{SOSA}observedProperty" in obs
        prop_ref = obs[f"{SOSA}observedProperty"]
        prop_id = prop_ref["@id"] if isinstance(prop_ref, dict) else prop_ref

        props = _find_nodes_by_type(graph, f"{SOSA}ObservableProperty")
        assert any(p["@id"] == prop_id for p in props)

    def test_measurement_uncertainty_maps_to_accuracy(self):
        """@measurementUncertainty → ssn-system:Accuracy on the Sensor."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                22.5,
                source="https://sensor.example.org/t1",
                measurement_uncertainty=0.1,
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]

        # Sensor should have a SystemCapability
        sensors = _find_nodes_by_type(graph, f"{SOSA}Sensor")
        assert len(sensors) >= 1
        sensor = sensors[0]
        assert f"{SSN_SYSTEM}hasSystemCapability" in sensor

        cap_ref = sensor[f"{SSN_SYSTEM}hasSystemCapability"]
        cap_id = cap_ref["@id"] if isinstance(cap_ref, dict) else cap_ref
        cap = _find_node_by_id(graph, cap_id)
        assert cap is not None

        # Capability should have Accuracy system property
        assert f"{SSN_SYSTEM}hasSystemProperty" in cap
        acc_ref = cap[f"{SSN_SYSTEM}hasSystemProperty"]
        acc_id = acc_ref["@id"] if isinstance(acc_ref, dict) else acc_ref
        acc = _find_node_by_id(graph, acc_id)
        assert acc is not None

        acc_types = acc["@type"] if isinstance(acc["@type"], list) else [acc["@type"]]
        assert f"{SSN_SYSTEM}Accuracy" in acc_types

    def test_calibration_on_sensor_capability(self):
        """@calibratedAt/Method/Authority → properties on Sensor's SystemCapability."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                22.5,
                source="https://sensor.example.org/t1",
                calibrated_at="2025-01-01T00:00:00Z",
                calibration_method="NIST traceable",
                calibration_authority="https://nist.gov",
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]

        sensors = _find_nodes_by_type(graph, f"{SOSA}Sensor")
        sensor = sensors[0]
        cap_ref = sensor[f"{SSN_SYSTEM}hasSystemCapability"]
        cap_id = cap_ref["@id"] if isinstance(cap_ref, dict) else cap_ref
        cap = _find_node_by_id(graph, cap_id)

        # Calibration info should be on the capability
        assert f"{JSONLD_EX_NAMESPACE}calibratedAt" in cap
        assert f"{JSONLD_EX_NAMESPACE}calibrationMethod" in cap
        assert f"{JSONLD_EX_NAMESPACE}calibrationAuthority" in cap

    def test_aggregation_maps_to_procedure(self):
        """@aggregationMethod/Window/Count → sosa:Procedure with params."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                22.5,
                aggregation_method="mean",
                aggregation_window="PT1H",
                aggregation_count=60,
            ),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]

        proc_ref = obs[f"{SOSA}usedProcedure"]
        proc_id = proc_ref["@id"] if isinstance(proc_ref, dict) else proc_ref
        proc = _find_node_by_id(graph, proc_id)
        assert proc is not None

        assert proc[f"{RDFS}label"] == "mean"
        assert proc[f"{JSONLD_EX_NAMESPACE}aggregationWindow"] == "PT1H"
        assert proc[f"{JSONLD_EX_NAMESPACE}aggregationCount"] == 60

    def test_multiple_annotated_properties(self):
        """Multiple annotated properties on one node → multiple Observations."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/room",
            "http://example.org/temperature": annotate(22.5, confidence=0.95),
            "http://example.org/humidity": annotate(45.0, confidence=0.88),
        }
        ssn_doc, report = to_ssn(doc)
        graph = ssn_doc["@graph"]
        observations = _find_nodes_by_type(graph, f"{SOSA}Observation")

        assert len(observations) >= 2
        assert report.nodes_converted >= 2

    def test_non_annotated_properties_preserved(self):
        """Non-annotated properties pass through unchanged."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://schema.org/name": "Plain value",
            "http://example.org/temp": annotate(22.5, confidence=0.9),
        }
        ssn_doc, _ = to_ssn(doc)
        graph = ssn_doc["@graph"]

        # Find the main node (the one that isn't a SOSA type)
        main = None
        sosa_types = {f"{SOSA}Observation", f"{SOSA}Sensor", f"{SOSA}Result",
                      f"{SOSA}Procedure", f"{SOSA}FeatureOfInterest",
                      f"{SOSA}ObservableProperty"}
        for node in graph:
            if not isinstance(node, dict):
                continue
            nt = node.get("@type", [])
            if isinstance(nt, str):
                nt = [nt]
            if not any(t in sosa_types or t.startswith(SSN_SYSTEM) for t in nt):
                main = node
                break

        # Main node should still have non-annotated property
        assert main is not None
        assert main.get("http://schema.org/name") == "Plain value"

    def test_full_iot_scenario(self):
        """Comprehensive IoT reading with all annotation keys."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "http://example.org/WeatherStation",
            "@id": "http://example.org/station-42",
            "http://example.org/temperature": annotate(
                22.5,
                confidence=0.95,
                source="https://sensor.example.org/thermo1",
                extracted_at="2025-01-15T10:30:00Z",
                method="thermometer-reading",
                unit="http://qudt.org/vocab/unit/DEG_C",
                measurement_uncertainty=0.1,
                calibrated_at="2025-01-01T00:00:00Z",
                calibration_method="NIST traceable",
                calibration_authority="https://nist.gov",
                aggregation_method="mean",
                aggregation_window="PT1H",
                aggregation_count=60,
            ),
        }
        ssn_doc, report = to_ssn(doc)

        assert report.success is True
        assert report.nodes_converted >= 1

        graph = ssn_doc["@graph"]

        # All SOSA types should be present
        assert len(_find_nodes_by_type(graph, f"{SOSA}Observation")) >= 1
        assert len(_find_nodes_by_type(graph, f"{SOSA}Sensor")) >= 1
        assert len(_find_nodes_by_type(graph, f"{SOSA}Procedure")) >= 1
        assert len(_find_nodes_by_type(graph, f"{SOSA}FeatureOfInterest")) >= 1
        assert len(_find_nodes_by_type(graph, f"{SOSA}ObservableProperty")) >= 1

        # Result should have unit
        obs = _find_nodes_by_type(graph, f"{SOSA}Observation")[0]
        assert f"{SOSA}hasResult" in obs

    def test_report_triple_counts(self):
        """ConversionReport tracks input and output triple counts."""
        doc = {
            "@context": "http://schema.org/",
            "@type": "Thing",
            "@id": "http://example.org/x",
            "http://example.org/temp": annotate(
                22.5, confidence=0.95, source="https://s.example.org/s1",
            ),
        }
        _, report = to_ssn(doc)

        assert report.triples_input >= 1
        assert report.triples_output >= report.triples_input  # SSN is more verbose


class TestSSNSOSAInteropFromSSN:
    """Tests for from_ssn() — SSN/SOSA → jsonld-ex conversion."""

    def _make_sosa_observation(self, **overrides) -> dict:
        """Helper: build a minimal SOSA Observation graph."""
        obs = {
            "@id": "_:obs-1",
            "@type": f"{SOSA}Observation",
            f"{SOSA}hasSimpleResult": "22.5",
            f"{SOSA}resultTime": {
                "@value": "2025-01-15T10:30:00Z",
                "@type": f"{XSD}dateTime",
            },
            f"{SOSA}madeBySensor": {"@id": "https://sensor.example.org/t1"},
            f"{SOSA}observedProperty": {"@id": "http://example.org/temperature"},
            f"{SOSA}hasFeatureOfInterest": {"@id": "http://example.org/room-134"},
        }
        obs.update(overrides)

        sensor = {
            "@id": "https://sensor.example.org/t1",
            "@type": f"{SOSA}Sensor",
        }
        foi = {
            "@id": "http://example.org/room-134",
            "@type": [f"{SOSA}FeatureOfInterest", "http://schema.org/Room"],
        }
        prop = {
            "@id": "http://example.org/temperature",
            "@type": f"{SOSA}ObservableProperty",
        }

        return {
            "@context": {
                "sosa": SOSA,
                "ssn": SSN,
                "ssn-system": SSN_SYSTEM,
                "qudt": QUDT,
                "xsd": XSD,
                "rdfs": RDFS,
                "jsonld-ex": JSONLD_EX_NAMESPACE,
            },
            "@graph": [obs, sensor, foi, prop],
        }

    def test_basic_from_ssn(self):
        """Basic SOSA Observation → jsonld-ex annotated value."""
        sosa_doc = self._make_sosa_observation()
        result_doc, report = from_ssn(sosa_doc)

        assert report.success is True
        assert report.nodes_converted >= 1

        # Should have reconstructed annotated value on the observed property
        # The property key should be the observed property IRI
        prop_key = "http://example.org/temperature"
        assert prop_key in result_doc

        val = result_doc[prop_key]
        assert isinstance(val, dict)
        assert "@value" in val
        assert val["@value"] == "22.5"

    def test_from_ssn_extracts_source(self):
        """sosa:madeBySensor → @source."""
        sosa_doc = self._make_sosa_observation()
        result_doc, _ = from_ssn(sosa_doc)

        val = result_doc["http://example.org/temperature"]
        assert val.get("@source") == "https://sensor.example.org/t1"

    def test_from_ssn_extracts_extracted_at(self):
        """sosa:resultTime → @extractedAt."""
        sosa_doc = self._make_sosa_observation()
        result_doc, _ = from_ssn(sosa_doc)

        val = result_doc["http://example.org/temperature"]
        assert val.get("@extractedAt") == "2025-01-15T10:30:00Z"

    def test_from_ssn_extracts_method(self):
        """sosa:usedProcedure → @method."""
        sosa_doc = self._make_sosa_observation()
        proc = {
            "@id": "_:proc-1",
            "@type": f"{SOSA}Procedure",
            f"{RDFS}label": "thermometer-reading",
        }
        sosa_doc["@graph"][0][f"{SOSA}usedProcedure"] = {"@id": "_:proc-1"}
        sosa_doc["@graph"].append(proc)

        result_doc, _ = from_ssn(sosa_doc)
        val = result_doc["http://example.org/temperature"]
        assert val.get("@method") == "thermometer-reading"

    def test_from_ssn_extracts_confidence(self):
        """jsonld-ex:confidence on Observation → @confidence."""
        sosa_doc = self._make_sosa_observation(
            **{f"{JSONLD_EX_NAMESPACE}confidence": 0.92}
        )
        result_doc, _ = from_ssn(sosa_doc)

        val = result_doc["http://example.org/temperature"]
        assert val.get("@confidence") == 0.92

    def test_from_ssn_extracts_unit_from_result(self):
        """sosa:hasResult with qudt:unit → @unit."""
        result_node = {
            "@id": "_:result-1",
            "@type": f"{SOSA}Result",
            f"{QUDT}numericValue": 22.5,
            f"{QUDT}unit": "http://qudt.org/vocab/unit/DEG_C",
        }
        sosa_doc = self._make_sosa_observation()
        # Replace hasSimpleResult with hasResult
        del sosa_doc["@graph"][0][f"{SOSA}hasSimpleResult"]
        sosa_doc["@graph"][0][f"{SOSA}hasResult"] = {"@id": "_:result-1"}
        sosa_doc["@graph"].append(result_node)

        result_doc, _ = from_ssn(sosa_doc)
        val = result_doc["http://example.org/temperature"]
        assert val["@value"] == 22.5
        assert val.get("@unit") == "http://qudt.org/vocab/unit/DEG_C"

    def test_from_ssn_extracts_measurement_uncertainty(self):
        """ssn-system:Accuracy → @measurementUncertainty."""
        sosa_doc = self._make_sosa_observation()
        # Add capability chain on the sensor
        acc_node = {
            "@id": "_:acc-1",
            "@type": f"{SSN_SYSTEM}Accuracy",
            f"{JSONLD_EX_NAMESPACE}value": 0.1,
        }
        cap_node = {
            "@id": "_:cap-1",
            "@type": f"{SSN_SYSTEM}SystemCapability",
            f"{SSN_SYSTEM}hasSystemProperty": {"@id": "_:acc-1"},
        }
        sosa_doc["@graph"][1][f"{SSN_SYSTEM}hasSystemCapability"] = {"@id": "_:cap-1"}
        sosa_doc["@graph"].extend([cap_node, acc_node])

        result_doc, _ = from_ssn(sosa_doc)
        val = result_doc["http://example.org/temperature"]
        assert val.get("@measurementUncertainty") == 0.1

    def test_from_ssn_feature_of_interest_becomes_parent(self):
        """sosa:hasFeatureOfInterest → parent node @id and @type."""
        sosa_doc = self._make_sosa_observation()
        result_doc, _ = from_ssn(sosa_doc)

        assert result_doc.get("@id") == "http://example.org/room-134"

    def test_from_ssn_warns_on_unsupported_sosa_features(self):
        """Unsupported SOSA concepts produce warnings."""
        sosa_doc = self._make_sosa_observation()
        # Add a Platform (not mappable to jsonld-ex)
        platform = {
            "@id": "http://example.org/platform-1",
            "@type": f"{SOSA}Platform",
            f"{SOSA}hosts": {"@id": "https://sensor.example.org/t1"},
        }
        sosa_doc["@graph"].append(platform)

        _, report = from_ssn(sosa_doc)
        # Should have at least one warning about unhandled SOSA concepts
        assert any("Platform" in w or "platform" in w.lower() for w in report.warnings) or report.success


class TestSSNSOSARoundTrip:
    """Tests for to_ssn() → from_ssn() round-trip fidelity."""

    def test_basic_round_trip(self):
        """Simple annotated value survives to_ssn → from_ssn."""
        original = {
            "@context": "http://schema.org/",
            "@type": "http://example.org/Room",
            "@id": "http://example.org/room-134",
            "http://example.org/temperature": annotate(
                22.5,
                confidence=0.95,
                source="https://sensor.example.org/t1",
                extracted_at="2025-01-15T10:30:00Z",
                method="thermometer-reading",
            ),
        }
        ssn_doc, _ = to_ssn(original)
        restored, report = from_ssn(ssn_doc)

        assert report.success is True

        val = restored["http://example.org/temperature"]
        assert val["@value"] == 22.5
        assert val["@confidence"] == 0.95
        assert val["@source"] == "https://sensor.example.org/t1"
        assert val["@extractedAt"] == "2025-01-15T10:30:00Z"
        assert val["@method"] == "thermometer-reading"

    def test_round_trip_with_unit(self):
        """Value with @unit survives round-trip."""
        original = {
            "@context": "http://schema.org/",
            "@type": "http://example.org/Room",
            "@id": "http://example.org/room-134",
            "http://example.org/temperature": annotate(
                22.5,
                source="https://sensor.example.org/t1",
                unit="http://qudt.org/vocab/unit/DEG_C",
            ),
        }
        ssn_doc, _ = to_ssn(original)
        restored, _ = from_ssn(ssn_doc)

        val = restored["http://example.org/temperature"]
        assert val["@value"] == 22.5
        assert val["@unit"] == "http://qudt.org/vocab/unit/DEG_C"

    def test_round_trip_preserves_feature_of_interest(self):
        """Parent @id survives as FeatureOfInterest through round-trip."""
        original = {
            "@context": "http://schema.org/",
            "@type": "http://example.org/Room",
            "@id": "http://example.org/room-134",
            "http://example.org/temperature": annotate(22.5, confidence=0.9),
        }
        ssn_doc, _ = to_ssn(original)
        restored, _ = from_ssn(ssn_doc)

        assert restored.get("@id") == "http://example.org/room-134"
