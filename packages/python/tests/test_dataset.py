"""Tests for dataset metadata extensions (GAP-D1, D2, D3)."""

import copy
import pytest
from jsonld_ex.dataset import (
    create_dataset_metadata,
    validate_dataset_metadata,
    add_distribution,
    add_file_set,
    add_record_set,
    create_field,
    to_croissant,
    from_croissant,
    DATASET_CONTEXT,
    CROISSANT_CONTEXT,
    DATASET_SHAPE,
)


# ── GAP-D1: Dataset Metadata ────────────────────────────────────────


class TestCreateDatasetMetadata:
    """Tests for create_dataset_metadata()."""

    def test_minimal_with_name_only(self):
        ds = create_dataset_metadata(name="test-dataset")
        assert ds["@type"] == "sc:Dataset"
        assert ds["name"] == "test-dataset"
        assert "@context" in ds

    def test_full_metadata(self):
        ds = create_dataset_metadata(
            name="mnist",
            description="Handwritten digit images",
            version="1.0.0",
            license="https://creativecommons.org/licenses/by/4.0/",
            url="https://yann.lecun.com/exdb/mnist/",
            date_published="2024-01-15",
            creator="Yann LeCun",
            keywords=["computer-vision", "digits", "classification"],
            citation="@article{lecun1998gradient, ...}",
        )
        assert ds["name"] == "mnist"
        assert ds["description"] == "Handwritten digit images"
        assert ds["version"] == "1.0.0"
        assert ds["license"] == "https://creativecommons.org/licenses/by/4.0/"
        assert ds["url"] == "https://yann.lecun.com/exdb/mnist/"
        assert ds["datePublished"] == "2024-01-15"
        assert ds["keywords"] == ["computer-vision", "digits", "classification"]

    def test_type_is_dataset(self):
        ds = create_dataset_metadata(name="test")
        assert ds["@type"] == "sc:Dataset"

    def test_context_present_and_valid(self):
        ds = create_dataset_metadata(name="test")
        ctx = ds["@context"]
        assert isinstance(ctx, dict)
        # Must map to schema.org
        assert ctx.get("@vocab") == "https://schema.org/" or "sc" in ctx

    def test_creator_string_wraps_to_person(self):
        ds = create_dataset_metadata(name="test", creator="Alice Smith")
        creator = ds["creator"]
        assert isinstance(creator, dict)
        assert creator["@type"] == "Person"
        assert creator["name"] == "Alice Smith"

    def test_creator_dict_passthrough(self):
        person = {"@type": "Person", "name": "Bob", "email": "bob@example.com"}
        ds = create_dataset_metadata(name="test", creator=person)
        assert ds["creator"] == person

    def test_creator_list_of_strings(self):
        ds = create_dataset_metadata(name="test", creator=["Alice", "Bob"])
        creators = ds["creator"]
        assert isinstance(creators, list)
        assert len(creators) == 2
        assert all(c["@type"] == "Person" for c in creators)
        assert creators[0]["name"] == "Alice"
        assert creators[1]["name"] == "Bob"

    def test_creator_list_mixed(self):
        ds = create_dataset_metadata(
            name="test",
            creator=[
                "Alice",
                {"@type": "Organization", "name": "Acme Corp"},
            ],
        )
        creators = ds["creator"]
        assert len(creators) == 2
        assert creators[0]["@type"] == "Person"
        assert creators[1]["@type"] == "Organization"

    def test_keywords_as_list(self):
        ds = create_dataset_metadata(name="test", keywords=["nlp", "text"])
        assert ds["keywords"] == ["nlp", "text"]

    def test_keywords_as_single_string(self):
        ds = create_dataset_metadata(name="test", keywords="nlp")
        assert ds["keywords"] == ["nlp"]

    def test_optional_fields_absent_when_none(self):
        ds = create_dataset_metadata(name="test")
        assert "description" not in ds
        assert "version" not in ds
        assert "license" not in ds
        assert "url" not in ds
        assert "datePublished" not in ds
        assert "creator" not in ds
        assert "keywords" not in ds
        assert "citation" not in ds
        assert "publisher" not in ds

    def test_publisher_string_wraps(self):
        ds = create_dataset_metadata(name="test", publisher="Acme Corp")
        pub = ds["publisher"]
        assert isinstance(pub, dict)
        assert pub["@type"] == "Organization"
        assert pub["name"] == "Acme Corp"

    def test_publisher_dict_passthrough(self):
        org = {"@type": "Organization", "name": "DeepMind", "url": "https://deepmind.com"}
        ds = create_dataset_metadata(name="test", publisher=org)
        assert ds["publisher"] == org

    def test_in_language_string(self):
        ds = create_dataset_metadata(name="test", in_language="en")
        assert ds["inLanguage"] == "en"

    def test_in_language_list(self):
        ds = create_dataset_metadata(name="test", in_language=["en", "fr"])
        assert ds["inLanguage"] == ["en", "fr"]

    def test_same_as(self):
        ds = create_dataset_metadata(
            name="test",
            same_as="https://huggingface.co/datasets/mnist",
        )
        assert ds["sameAs"] == "https://huggingface.co/datasets/mnist"

    def test_same_as_list(self):
        ds = create_dataset_metadata(
            name="test",
            same_as=["https://hf.co/datasets/mnist", "https://kaggle.com/mnist"],
        )
        assert ds["sameAs"] == ["https://hf.co/datasets/mnist", "https://kaggle.com/mnist"]

    def test_date_created_and_modified(self):
        ds = create_dataset_metadata(
            name="test",
            date_created="2023-01-01",
            date_modified="2024-06-15",
        )
        assert ds["dateCreated"] == "2023-01-01"
        assert ds["dateModified"] == "2024-06-15"

    def test_is_live_dataset(self):
        ds = create_dataset_metadata(name="test", is_live=True)
        assert ds["isLiveDataset"] is True

    def test_is_live_default_absent(self):
        ds = create_dataset_metadata(name="test")
        assert "isLiveDataset" not in ds

    def test_citation_as_bibtex(self):
        bib = "@article{doe2024, title={Test}, author={Doe}}"
        ds = create_dataset_metadata(name="test", citation=bib)
        assert ds["citeAs"] == bib

    def test_empty_name_raises(self):
        with pytest.raises(ValueError):
            create_dataset_metadata(name="")

    def test_none_name_raises(self):
        with pytest.raises((ValueError, TypeError)):
            create_dataset_metadata(name=None)

    def test_distribution_initialized_as_empty_list(self):
        ds = create_dataset_metadata(name="test")
        assert ds["distribution"] == []

    def test_record_set_initialized_as_empty_list(self):
        ds = create_dataset_metadata(name="test")
        assert ds["recordSet"] == []


class TestValidateDatasetMetadata:
    """Tests for validate_dataset_metadata()."""

    def test_valid_full_metadata(self):
        ds = create_dataset_metadata(
            name="test-dataset",
            description="A test dataset",
            version="1.0.0",
            license="https://creativecommons.org/licenses/by/4.0/",
            url="https://example.com/dataset",
            date_published="2024-01-15",
            creator="Alice",
        )
        result = validate_dataset_metadata(ds)
        assert result.valid, f"Errors: {[e.message for e in result.errors]}"

    def test_valid_minimal_metadata(self):
        ds = create_dataset_metadata(name="test-dataset")
        result = validate_dataset_metadata(ds)
        assert result.valid

    def test_missing_name_fails(self):
        ds = {"@type": "sc:Dataset", "@context": DATASET_CONTEXT}
        result = validate_dataset_metadata(ds)
        assert not result.valid
        assert any("name" in e.path or "name" in e.message for e in result.errors)

    def test_empty_name_fails(self):
        ds = create_dataset_metadata(name="placeholder")
        ds["name"] = ""
        result = validate_dataset_metadata(ds)
        assert not result.valid

    def test_non_string_name_fails(self):
        ds = create_dataset_metadata(name="placeholder")
        ds["name"] = 12345
        result = validate_dataset_metadata(ds)
        assert not result.valid

    def test_roundtrip_create_then_validate(self):
        ds = create_dataset_metadata(
            name="roundtrip-test",
            description="Testing round-trip",
            version="0.1.0",
            license="MIT",
            url="https://example.com",
            date_published="2025-01-01",
            creator="Tester",
            keywords=["test"],
        )
        result = validate_dataset_metadata(ds)
        assert result.valid, f"Errors: {[e.message for e in result.errors]}"

    def test_wrong_type_fails(self):
        ds = create_dataset_metadata(name="test")
        ds["@type"] = "Person"
        result = validate_dataset_metadata(ds)
        assert not result.valid

    def test_dataset_shape_is_exported(self):
        """DATASET_SHAPE should be a valid shape dict usable with validate_node."""
        assert isinstance(DATASET_SHAPE, dict)
        assert "@type" in DATASET_SHAPE
        assert "name" in DATASET_SHAPE


# ── GAP-D2: Distributions and Structure ─────────────────────────────


class TestAddDistribution:
    """Tests for add_distribution() — FileObject resources."""

    def test_add_single_file(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds,
            name="data.csv",
            content_url="https://example.com/data.csv",
            encoding_format="text/csv",
            sha256="abc123",
        )
        assert len(ds["distribution"]) == 1
        fo = ds["distribution"][0]
        assert fo["@type"] == "cr:FileObject"
        assert fo["@id"] == "data.csv"
        assert fo["name"] == "data.csv"
        assert fo["contentUrl"] == "https://example.com/data.csv"
        assert fo["encodingFormat"] == "text/csv"
        assert fo["sha256"] == "abc123"

    def test_add_multiple_files(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(ds, name="train.csv", content_url="https://x.com/train.csv", encoding_format="text/csv")
        ds = add_distribution(ds, name="test.csv", content_url="https://x.com/test.csv", encoding_format="text/csv")
        assert len(ds["distribution"]) == 2

    def test_optional_fields(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds,
            name="archive.tar.gz",
            content_url="https://example.com/archive.tar.gz",
            encoding_format="application/x-gzip",
            content_size="25585843 B",
            description="Main archive file",
        )
        fo = ds["distribution"][0]
        assert fo["contentSize"] == "25585843 B"
        assert fo["description"] == "Main archive file"

    def test_custom_id(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds,
            name="data.csv",
            content_url="https://example.com/data.csv",
            encoding_format="text/csv",
            file_id="my-custom-id",
        )
        assert ds["distribution"][0]["@id"] == "my-custom-id"

    def test_does_not_mutate_original(self):
        ds = create_dataset_metadata(name="test")
        original_dist_len = len(ds["distribution"])
        ds2 = add_distribution(ds, name="f.csv", content_url="u", encoding_format="text/csv")
        # Original should be unchanged if we copy internally
        assert len(ds2["distribution"]) == original_dist_len + 1

    def test_missing_name_raises(self):
        ds = create_dataset_metadata(name="test")
        with pytest.raises((ValueError, TypeError)):
            add_distribution(ds, name="", content_url="u", encoding_format="text/csv")


class TestAddFileSet:
    """Tests for add_file_set() — FileSet resources."""

    def test_add_file_set(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds, name="archive.tar", content_url="https://x.com/a.tar",
            encoding_format="application/x-tar", file_id="archive",
        )
        ds = add_file_set(
            ds,
            name="image-files",
            contained_in="archive",
            encoding_format="image/jpeg",
            includes="*.jpg",
        )
        # FileSet is also added to distribution
        file_sets = [d for d in ds["distribution"] if d["@type"] == "cr:FileSet"]
        assert len(file_sets) == 1
        fs = file_sets[0]
        assert fs["@id"] == "image-files"
        assert fs["containedIn"] == {"@id": "archive"}
        assert fs["includes"] == "*.jpg"

    def test_file_set_custom_id(self):
        ds = create_dataset_metadata(name="test")
        ds = add_file_set(
            ds, name="images", contained_in="archive",
            encoding_format="image/png", includes="*.png",
            file_set_id="my-images",
        )
        file_sets = [d for d in ds["distribution"] if d["@type"] == "cr:FileSet"]
        assert file_sets[0]["@id"] == "my-images"


class TestAddRecordSet:
    """Tests for add_record_set() and create_field()."""

    def test_create_field(self):
        f = create_field(
            name="age",
            data_type="sc:Integer",
            description="The age in years",
            source={"fileObject": {"@id": "data.csv"}, "extract": {"column": "age"}},
        )
        assert f["@type"] == "cr:Field"
        assert f["dataType"] == "sc:Integer"
        assert f["description"] == "The age in years"
        assert "source" in f

    def test_create_field_minimal(self):
        f = create_field(name="value", data_type="sc:Float")
        assert f["@type"] == "cr:Field"
        assert f["name"] == "value"
        assert f["dataType"] == "sc:Float"

    def test_add_record_set(self):
        ds = create_dataset_metadata(name="test")
        fields = [
            create_field("name", data_type="sc:Text"),
            create_field("age", data_type="sc:Integer"),
        ]
        ds = add_record_set(ds, name="examples", fields=fields, description="Example records")
        assert len(ds["recordSet"]) == 1
        rs = ds["recordSet"][0]
        assert rs["@type"] == "cr:RecordSet"
        assert rs["@id"] == "examples"
        assert rs["name"] == "examples"
        assert rs["description"] == "Example records"
        assert len(rs["field"]) == 2

    def test_field_ids_prefixed_by_record_set(self):
        """Croissant convention: field @id = recordset_id/field_name."""
        ds = create_dataset_metadata(name="test")
        fields = [create_field("col_a", data_type="sc:Text")]
        ds = add_record_set(ds, name="records", fields=fields)
        f = ds["recordSet"][0]["field"][0]
        assert f["@id"] == "records/col_a"

    def test_add_multiple_record_sets(self):
        ds = create_dataset_metadata(name="test")
        ds = add_record_set(ds, name="train", fields=[create_field("x", data_type="sc:Float")])
        ds = add_record_set(ds, name="test", fields=[create_field("x", data_type="sc:Float")])
        assert len(ds["recordSet"]) == 2


# ── GAP-D3: Croissant Interoperability ──────────────────────────────


class TestToCroissant:
    """Tests for to_croissant() conversion."""

    def test_basic_conversion(self):
        ds = create_dataset_metadata(
            name="test-dataset",
            description="A test",
            version="1.0.0",
            license="https://creativecommons.org/licenses/by/4.0/",
            url="https://example.com",
            date_published="2024-01-01",
            creator="Alice",
        )
        cr = to_croissant(ds)
        assert cr["@type"] == "sc:Dataset"
        assert cr["conformsTo"] == "http://mlcommons.org/croissant/1.0"
        assert cr["name"] == "test-dataset"

    def test_croissant_context(self):
        ds = create_dataset_metadata(name="test", description="d")
        cr = to_croissant(ds)
        ctx = cr["@context"]
        # Must have Croissant namespace
        assert ctx.get("cr") == "http://mlcommons.org/croissant/"
        # Must have schema.org
        assert ctx.get("@vocab") == "https://schema.org/" or ctx.get("sc") == "https://schema.org/"
        # Must have Dublin Core
        assert ctx.get("dct") == "http://purl.org/dc/terms/"

    def test_conforms_to_added(self):
        ds = create_dataset_metadata(name="test", description="d")
        cr = to_croissant(ds)
        assert cr["conformsTo"] == "http://mlcommons.org/croissant/1.0"

    def test_citation_mapped_to_cite_as(self):
        bib = "@article{test2024, title={T}}"
        ds = create_dataset_metadata(name="test", citation=bib)
        cr = to_croissant(ds)
        assert cr["citeAs"] == bib

    def test_distribution_preserved(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds, name="data.csv", content_url="https://x.com/data.csv",
            encoding_format="text/csv", sha256="abc",
        )
        cr = to_croissant(ds)
        assert len(cr["distribution"]) == 1
        assert cr["distribution"][0]["@type"] == "cr:FileObject"

    def test_record_set_preserved(self):
        ds = create_dataset_metadata(name="test")
        ds = add_record_set(
            ds, name="examples",
            fields=[create_field("col", data_type="sc:Text")],
        )
        cr = to_croissant(ds)
        assert len(cr["recordSet"]) == 1
        assert cr["recordSet"][0]["@type"] == "cr:RecordSet"

    def test_jsonld_ex_annotations_preserved(self):
        """jsonld-ex provenance annotations should survive conversion."""
        ds = create_dataset_metadata(name="test", description="d")
        # Simulate an annotated field
        ds["customAnnotation"] = {"@value": "test", "@confidence": 0.9}
        cr = to_croissant(ds)
        assert cr.get("customAnnotation") == {"@value": "test", "@confidence": 0.9}

    def test_does_not_mutate_input(self):
        ds = create_dataset_metadata(name="test")
        ds_copy = copy.deepcopy(ds)
        to_croissant(ds)
        assert ds == ds_copy


class TestFromCroissant:
    """Tests for from_croissant() conversion."""

    def test_basic_import(self):
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "imported-dataset",
            "description": "Imported from Croissant",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "license": "https://creativecommons.org/licenses/by/4.0/",
            "url": "https://example.com",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(croissant_doc)
        assert ds["@type"] == "sc:Dataset"
        assert ds["name"] == "imported-dataset"
        # Should have our context, not Croissant's
        assert ds["@context"] == DATASET_CONTEXT

    def test_cite_as_mapped_to_citation(self):
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "citeAs": "@article{test, title={T}}",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(croissant_doc)
        assert ds["citeAs"] == "@article{test, title={T}}"

    def test_distribution_imported(self):
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": [
                {
                    "@type": "cr:FileObject",
                    "@id": "data.csv",
                    "name": "data.csv",
                    "contentUrl": "https://example.com/data.csv",
                    "encodingFormat": "text/csv",
                    "sha256": "abc123",
                }
            ],
            "recordSet": [],
        }
        ds = from_croissant(croissant_doc)
        assert len(ds["distribution"]) == 1
        assert ds["distribution"][0]["@type"] == "cr:FileObject"

    def test_record_set_imported(self):
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": [],
            "recordSet": [
                {
                    "@type": "cr:RecordSet",
                    "@id": "default",
                    "name": "default",
                    "field": [
                        {
                            "@type": "cr:Field",
                            "@id": "default/col",
                            "name": "default/col",
                            "dataType": "sc:Text",
                        }
                    ],
                }
            ],
        }
        ds = from_croissant(croissant_doc)
        assert len(ds["recordSet"]) == 1
        assert len(ds["recordSet"][0]["field"]) == 1

    def test_conforms_to_stripped(self):
        """Our native format doesn't need conformsTo."""
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(croissant_doc)
        assert "conformsTo" not in ds

    def test_does_not_mutate_input(self):
        croissant_doc = {
            "@context": CROISSANT_CONTEXT,
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": [],
            "recordSet": [],
        }
        original = copy.deepcopy(croissant_doc)
        from_croissant(croissant_doc)
        assert croissant_doc == original


class TestCroissantRoundTrip:
    """Round-trip fidelity tests: jsonld-ex → Croissant → jsonld-ex."""

    def test_metadata_roundtrip(self):
        ds = create_dataset_metadata(
            name="roundtrip-test",
            description="Testing round-trip fidelity",
            version="2.0.0",
            license="https://creativecommons.org/licenses/by/4.0/",
            url="https://example.com/dataset",
            date_published="2024-06-15",
            creator="Alice Smith",
            keywords=["ml", "test"],
            citation="@article{test2024, title={Test}}",
        )
        cr = to_croissant(ds)
        ds2 = from_croissant(cr)

        assert ds2["name"] == ds["name"]
        assert ds2["description"] == ds["description"]
        assert ds2["version"] == ds["version"]
        assert ds2["license"] == ds["license"]
        assert ds2["url"] == ds["url"]
        assert ds2["datePublished"] == ds["datePublished"]
        assert ds2["keywords"] == ds["keywords"]

    def test_distribution_roundtrip(self):
        ds = create_dataset_metadata(name="test")
        ds = add_distribution(
            ds, name="data.csv", content_url="https://x.com/data.csv",
            encoding_format="text/csv", sha256="hash123",
        )
        cr = to_croissant(ds)
        ds2 = from_croissant(cr)

        assert len(ds2["distribution"]) == 1
        fo = ds2["distribution"][0]
        assert fo["name"] == "data.csv"
        assert fo["contentUrl"] == "https://x.com/data.csv"
        assert fo["sha256"] == "hash123"

    def test_record_set_roundtrip(self):
        ds = create_dataset_metadata(name="test")
        fields = [
            create_field("name", data_type="sc:Text", description="Name column"),
            create_field("score", data_type="sc:Float", description="Score value"),
        ]
        ds = add_record_set(ds, name="records", fields=fields, description="Main records")
        cr = to_croissant(ds)
        ds2 = from_croissant(cr)

        assert len(ds2["recordSet"]) == 1
        rs = ds2["recordSet"][0]
        assert rs["name"] == "records"
        assert len(rs["field"]) == 2

    def test_full_dataset_roundtrip(self):
        """Comprehensive round-trip with distributions, file sets, record sets."""
        ds = create_dataset_metadata(
            name="full-test",
            description="Full round-trip test",
            version="1.0.0",
            license="MIT",
            url="https://example.com",
            date_published="2025-01-01",
            creator=["Alice", {"@type": "Organization", "name": "Lab"}],
            keywords=["test", "roundtrip"],
        )
        ds = add_distribution(
            ds, name="archive.tar.gz",
            content_url="https://example.com/archive.tar.gz",
            encoding_format="application/x-gzip",
            sha256="deadbeef", file_id="archive",
        )
        ds = add_file_set(
            ds, name="csv-files", contained_in="archive",
            encoding_format="text/csv", includes="*.csv",
        )
        ds = add_record_set(
            ds, name="train",
            fields=[
                create_field("feature", data_type="sc:Float"),
                create_field("label", data_type="sc:Integer"),
            ],
            description="Training split",
        )

        cr = to_croissant(ds)
        ds2 = from_croissant(cr)

        assert ds2["name"] == "full-test"
        assert len(ds2["distribution"]) == 2  # FileObject + FileSet
        assert len(ds2["recordSet"]) == 1
        assert len(ds2["recordSet"][0]["field"]) == 2


class TestCroissantContextExported:
    """Verify the vendored CROISSANT_CONTEXT is complete."""

    def test_has_croissant_namespace(self):
        assert CROISSANT_CONTEXT["cr"] == "http://mlcommons.org/croissant/"

    def test_has_schema_org(self):
        assert CROISSANT_CONTEXT.get("@vocab") == "https://schema.org/" or \
               CROISSANT_CONTEXT.get("sc") == "https://schema.org/"

    def test_has_dublin_core(self):
        assert CROISSANT_CONTEXT["dct"] == "http://purl.org/dc/terms/"

    def test_has_core_croissant_terms(self):
        """Key Croissant terms must be present."""
        for term in ["recordSet", "field", "extract", "dataType",
                      "source", "fileObject", "fileSet", "conformsTo"]:
            assert term in CROISSANT_CONTEXT, f"Missing Croissant term: {term}"
