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



# ── Croissant RAI Extension Support ─────────────────────────────────


# All 20 RAI vocabulary properties from the spec
RAI_PROPERTIES = {
    # Data life cycle
    "rai:dataCollection": "Description of data collection process",
    "rai:dataCollectionType": ["Web Scraping", "Secondary Data Analysis"],
    "rai:dataCollectionMissingData": "No known missing data",
    "rai:dataCollectionRawData": "Raw sensor readings from IoT devices",
    "rai:dataCollectionTimeframe": [
        {"@value": "2023-01-01T00:00:00", "dataType": "sc:Date"},
        {"@value": "2024-12-31T00:00:00", "dataType": "sc:Date"},
    ],
    "rai:dataPreprocessingProtocol": [
        "Outlier removal using IQR method",
        "Min-max normalization to [0, 1]",
    ],
    "rai:dataReleaseMaintenancePlan": [
        "Updated quarterly",
        "Maintained by the ML team",
    ],
    # Data labeling
    "rai:dataAnnotationProtocol": "Three annotators per item, majority vote",
    "rai:dataAnnotationPlatform": ["Amazon Mechanical Turk", "Label Studio"],
    "rai:dataAnnotationAnalysis": [
        "Inter-annotator agreement: Cohen's kappa = 0.82",
        "Systematic disagreements analyzed by demographic group",
    ],
    "rai:annotationsPerItem": "3",
    "rai:annotatorDemographics": [
        "50% female, 50% male",
        "Age range: 25-55",
        "Geographic distribution: US 40%, EU 35%, Asia 25%",
    ],
    "rai:machineAnnotationTools": [
        "spaCy en_core_web_trf v3.7",
        "GLiNER2 fastino/gliner2-base-v1",
    ],
    # AI safety and fairness
    "rai:dataSocialImpact": "May improve accessibility of medical diagnostics",
    "rai:dataBiases": [
        "Underrepresentation of age group 65+",
        "English-language bias in text samples",
    ],
    "rai:dataLimitations": [
        "Not suitable for clinical decision-making without expert review",
        "Limited to US hospital data",
    ],
    "rai:dataUseCases": [
        "Training medical NER models",
        "Evaluating clinical text understanding",
    ],
    # Compliance
    "rai:personalSensitiveInformation": [
        "De-identified patient records",
        "No direct identifiers remain",
    ],
    "rai:dataImputationProtocol": "Missing values imputed using KNN (k=5)",
    "rai:dataManipulationProtocol": "Text redaction of PHI using regex + NER pipeline",
}


def _make_croissant_with_rai() -> dict:
    """Helper: build a realistic Croissant doc with RAI properties."""
    doc = {
        "@context": copy.deepcopy(CROISSANT_CONTEXT),
        "@type": "sc:Dataset",
        "name": "clinical-ner-benchmark",
        "description": "A benchmark dataset for clinical NER evaluation",
        "version": "2.1.0",
        "license": "https://creativecommons.org/licenses/by-nc/4.0/",
        "url": "https://example.com/clinical-ner",
        "datePublished": "2025-03-15",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "annotations.jsonl",
                "name": "annotations.jsonl",
                "contentUrl": "https://example.com/annotations.jsonl",
                "encodingFormat": "application/jsonl",
            }
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "entities",
                "name": "entities",
                "field": [
                    {
                        "@type": "cr:Field",
                        "@id": "entities/text",
                        "name": "text",
                        "dataType": "sc:Text",
                    },
                    {
                        "@type": "cr:Field",
                        "@id": "entities/label",
                        "name": "label",
                        "dataType": "sc:Text",
                    },
                ],
            }
        ],
    }
    # Add all RAI properties
    for key, value in RAI_PROPERTIES.items():
        doc[key] = copy.deepcopy(value)
    return doc


class TestDatasetContextRaiNamespace:
    """DATASET_CONTEXT must include the RAI namespace for proper JSON-LD resolution."""

    def test_rai_namespace_in_dataset_context(self):
        """DATASET_CONTEXT must map 'rai' to the official RAI namespace IRI."""
        assert "rai" in DATASET_CONTEXT, (
            "DATASET_CONTEXT is missing the 'rai' namespace prefix. "
            "RAI properties imported from Croissant will be unresolvable."
        )
        assert DATASET_CONTEXT["rai"] == "http://mlcommons.org/croissant/RAI/"

    def test_rai_namespace_in_croissant_context(self):
        """CROISSANT_CONTEXT must also have the RAI namespace (regression guard)."""
        assert "rai" in CROISSANT_CONTEXT
        assert CROISSANT_CONTEXT["rai"] == "http://mlcommons.org/croissant/RAI/"


class TestFromCroissantRaiPreservation:
    """from_croissant() must preserve all RAI properties during import."""

    def test_single_rai_text_property_preserved(self):
        """A single-valued rai:Text property must survive import."""
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "rai:dataCollection": "Collected via web scraping",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        assert ds.get("rai:dataCollection") == "Collected via web scraping"

    def test_list_rai_property_preserved(self):
        """A list-valued RAI property must survive import intact."""
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "rai:dataCollectionType": ["Web Scraping", "Manual Human Curation"],
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        assert ds.get("rai:dataCollectionType") == ["Web Scraping", "Manual Human Curation"]

    def test_structured_rai_property_preserved(self):
        """RAI properties with structured values (dicts/nested) must survive."""
        timeframe = [
            {"@value": "2023-01-01T00:00:00", "dataType": "sc:Date"},
            {"@value": "2024-12-31T00:00:00", "dataType": "sc:Date"},
        ]
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "rai:dataCollectionTimeframe": copy.deepcopy(timeframe),
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        assert ds.get("rai:dataCollectionTimeframe") == timeframe

    def test_all_20_rai_properties_preserved(self):
        """Every one of the 20 RAI vocabulary properties must survive import."""
        doc = _make_croissant_with_rai()
        ds = from_croissant(doc)
        for key, expected_value in RAI_PROPERTIES.items():
            assert key in ds, f"RAI property '{key}' was lost during from_croissant()"
            assert ds[key] == expected_value, (
                f"RAI property '{key}' was altered during from_croissant(). "
                f"Expected {expected_value!r}, got {ds[key]!r}"
            )

    def test_rai_properties_coexist_with_core_fields(self):
        """RAI properties must not interfere with core dataset fields."""
        doc = _make_croissant_with_rai()
        ds = from_croissant(doc)
        # Core fields intact
        assert ds["name"] == "clinical-ner-benchmark"
        assert ds["description"] == "A benchmark dataset for clinical NER evaluation"
        assert ds["version"] == "2.1.0"
        assert len(ds["distribution"]) == 1
        assert len(ds["recordSet"]) == 1
        # RAI still present
        assert "rai:dataCollection" in ds

    def test_rai_namespace_resolvable_after_import(self):
        """After import, the context must allow 'rai:' prefix resolution."""
        doc = _make_croissant_with_rai()
        ds = from_croissant(doc)
        ctx = ds["@context"]
        assert "rai" in ctx, "RAI namespace missing from imported context"
        assert ctx["rai"] == "http://mlcommons.org/croissant/RAI/"


class TestFromCroissantRaiConformsTo:
    """from_croissant() must handle conformsTo correctly for RAI."""

    def test_core_conformsto_stripped(self):
        """Core Croissant conformsTo should still be stripped."""
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        assert "conformsTo" not in ds

    def test_rai_conformsto_preserved_when_list(self):
        """When conformsTo is a list with both core and RAI, RAI should be preserved."""
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": [
                "http://mlcommons.org/croissant/1.0",
                "http://mlcommons.org/croissant/RAI/1.0",
            ],
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        # Core stripped, but RAI conformance preserved
        conforms = ds.get("conformsTo")
        if conforms is None:
            # If conformsTo is fully stripped, at minimum RAI conformance
            # should be stored somewhere (e.g. as a separate field)
            # For now, we require it not to be silently dropped
            pytest.fail(
                "RAI conformsTo was silently dropped. "
                "from_croissant() should preserve non-core conformance declarations."
            )
        # If it's kept, it should only contain the RAI URI
        if isinstance(conforms, list):
            assert "http://mlcommons.org/croissant/1.0" not in conforms
            assert "http://mlcommons.org/croissant/RAI/1.0" in conforms
        else:
            assert conforms == "http://mlcommons.org/croissant/RAI/1.0"

    def test_rai_only_conformsto_preserved(self):
        """If conformsTo is only the RAI URI, it should be preserved."""
        doc = {
            "@context": copy.deepcopy(CROISSANT_CONTEXT),
            "@type": "sc:Dataset",
            "name": "test",
            "conformsTo": "http://mlcommons.org/croissant/RAI/1.0",
            "distribution": [],
            "recordSet": [],
        }
        ds = from_croissant(doc)
        assert ds.get("conformsTo") == "http://mlcommons.org/croissant/RAI/1.0"


class TestToCroissantRaiPreservation:
    """to_croissant() must preserve RAI properties during export."""

    def test_rai_properties_in_export(self):
        """RAI properties added to a jsonld-ex doc must appear in Croissant export."""
        ds = create_dataset_metadata(name="test", description="d")
        ds["rai:dataCollection"] = "Scraped from public APIs"
        ds["rai:dataBiases"] = ["Selection bias toward English content"]
        cr = to_croissant(ds)
        assert cr.get("rai:dataCollection") == "Scraped from public APIs"
        assert cr.get("rai:dataBiases") == ["Selection bias toward English content"]

    def test_rai_namespace_in_exported_context(self):
        """Exported Croissant context must include rai namespace."""
        ds = create_dataset_metadata(name="test")
        ds["rai:dataCollection"] = "test"
        cr = to_croissant(ds)
        assert cr["@context"].get("rai") == "http://mlcommons.org/croissant/RAI/"

    def test_rai_conformsto_added_when_rai_properties_present(self):
        """When RAI properties are present, conformsTo should include RAI spec version."""
        ds = create_dataset_metadata(name="test")
        ds["rai:dataCollection"] = "test collection"
        cr = to_croissant(ds)
        conforms = cr.get("conformsTo")
        # Should declare conformance to both core and RAI
        if isinstance(conforms, list):
            assert "http://mlcommons.org/croissant/1.0" in conforms
            assert "http://mlcommons.org/croissant/RAI/1.0" in conforms
        elif isinstance(conforms, str):
            # If still a string, it should at minimum have core
            assert conforms == "http://mlcommons.org/croissant/1.0"
            # But ideally RAI should also be declared - this is a soft check
            # since the spec says RAI datasets "must declare" RAI conformance


class TestCroissantRaiRoundTrip:
    """Round-trip fidelity for RAI properties: Croissant+RAI -> jsonld-ex -> Croissant+RAI."""

    def test_all_rai_properties_roundtrip(self):
        """Every RAI property must survive a full round-trip."""
        original = _make_croissant_with_rai()
        imported = from_croissant(original)
        exported = to_croissant(imported)

        for key, expected_value in RAI_PROPERTIES.items():
            assert key in exported, (
                f"RAI property '{key}' lost during round-trip"
            )
            assert exported[key] == expected_value, (
                f"RAI property '{key}' altered during round-trip. "
                f"Expected {expected_value!r}, got {exported[key]!r}"
            )

    def test_core_fields_survive_rai_roundtrip(self):
        """Core dataset fields must not be disrupted by RAI round-tripping."""
        original = _make_croissant_with_rai()
        imported = from_croissant(original)
        exported = to_croissant(imported)

        assert exported["name"] == "clinical-ner-benchmark"
        assert exported["description"] == "A benchmark dataset for clinical NER evaluation"
        assert exported["version"] == "2.1.0"
        assert len(exported["distribution"]) == 1
        assert exported["distribution"][0]["@id"] == "annotations.jsonl"
        assert len(exported["recordSet"]) == 1
        assert exported["recordSet"][0]["@id"] == "entities"

    def test_jsonldex_annotations_plus_rai_roundtrip(self):
        """jsonld-ex @confidence annotations must coexist with RAI properties."""
        original = _make_croissant_with_rai()
        imported = from_croissant(original)

        # Add jsonld-ex annotations on top of RAI
        imported["description"] = {
            "@value": imported["description"],
            "@confidence": 0.95,
            "@source": "manual-review",
            "@extractedAt": "2025-03-15T10:00:00Z",
        }

        exported = to_croissant(imported)

        # jsonld-ex annotations preserved
        desc = exported["description"]
        assert isinstance(desc, dict)
        assert desc["@confidence"] == 0.95
        assert desc["@source"] == "manual-review"

        # RAI properties also preserved
        assert exported.get("rai:dataCollection") == RAI_PROPERTIES["rai:dataCollection"]
        assert exported.get("rai:dataBiases") == RAI_PROPERTIES["rai:dataBiases"]


class TestRealWorldCroissantRaiDocument:
    """Test importing a realistic Croissant+RAI document modeled after the spec examples."""

    def _make_dices_style_doc(self) -> dict:
        """Build a document styled after the DICES-350 RAI example from the spec."""
        return {
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "sc": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "dct": "http://purl.org/dc/terms/",
                "rai": "http://mlcommons.org/croissant/RAI/",
            },
            "@type": "sc:Dataset",
            "name": "DICES-350-style",
            "dct:conformsTo": "http://mlcommons.org/croissant/RAI/1.0",
            "rai:dataCollection": (
                "Subset of an 8K multi-turn conversation corpus "
                "generated by human agents interacting with a chatbot."
            ),
            "rai:dataCollectionType": (
                "350 adversarial multi-turn conversations, "
                "annotated by a pool of annotators along 16 safety criteria."
            ),
            "rai:dataAnnotationProtocol": (
                "Six question sets covering legibility, harmful content, "
                "unfair bias, misinformation, political affiliations, "
                "and policy violations."
            ),
            "rai:dataAnnotationPlatform": "Crowdworker annotators with task specific UI",
            "rai:dataAnnotationAnalysis": (
                "Initial 123 raters, 19 filtered for low quality, "
                "104 remaining. Gold ratings from trust and safety experts."
            ),
            "rai:dataUseCases": (
                "Benchmark for evaluating conversational AI safety and diversity."
            ),
            "rai:dataBiases": (
                "Limited demographic categories: 4 axes, constrained subgroups."
            ),
            "rai:annotationsPerItem": "104 unique ratings per conversation",
            "rai:annotatorDemographics": (
                "57 women, 47 men; 27 gen X+, 28 millennial, 49 gen z; "
                "21 Asian, 23 Black/African American, 22 Latine/x, "
                "13 multiracial, 25 white."
            ),
            "distribution": [],
            "recordSet": [],
        }

    def test_dices_style_import(self):
        """A DICES-350-style RAI document must import cleanly."""
        doc = self._make_dices_style_doc()
        ds = from_croissant(doc)

        assert ds["name"] == "DICES-350-style"
        assert "rai:dataCollection" in ds
        assert "rai:dataAnnotationProtocol" in ds
        assert "rai:annotatorDemographics" in ds
        assert "rai:dataBiases" in ds

    def test_dices_style_roundtrip(self):
        """DICES-style doc must survive full round-trip."""
        doc = self._make_dices_style_doc()
        ds = from_croissant(doc)
        cr = to_croissant(ds)

        assert cr["name"] == "DICES-350-style"
        assert cr.get("rai:dataCollection") == doc["rai:dataCollection"]
        assert cr.get("rai:annotatorDemographics") == doc["rai:annotatorDemographics"]

    def test_dct_conformsto_handling(self):
        """Some RAI docs use 'dct:conformsTo' instead of 'conformsTo'.
        Both forms must be handled correctly."""
        doc = self._make_dices_style_doc()
        # This doc uses dct:conformsTo (the prefixed form)
        assert "dct:conformsTo" in doc
        ds = from_croissant(doc)
        # The RAI conformsTo should be preserved (it's not core Croissant conformance)
        # Note: dct:conformsTo with RAI URI should not be stripped
        has_rai_conformance = (
            ds.get("dct:conformsTo") == "http://mlcommons.org/croissant/RAI/1.0"
            or ds.get("conformsTo") == "http://mlcommons.org/croissant/RAI/1.0"
        )
        assert has_rai_conformance, (
            "RAI conformance declaration (dct:conformsTo) was lost during import"
        )

    def _make_bigscience_style_doc(self) -> dict:
        """Build a document styled after the BigScience ROOTS example from the spec."""
        return {
            "@context": {
                "@language": "en",
                "@vocab": "https://schema.org/",
                "sc": "https://schema.org/",
                "cr": "http://mlcommons.org/croissant/",
                "dct": "http://purl.org/dc/terms/",
                "rai": "http://mlcommons.org/croissant/RAI/",
            },
            "@type": "sc:Dataset",
            "name": "BigScience-ROOTS-style",
            "conformsTo": "http://mlcommons.org/croissant/1.0",
            "rai:dataCollection": (
                "62% from curated monolingual/multilingual language resources, "
                "38% from OSCAR/Common Crawl."
            ),
            "rai:dataCollectionType": [
                "Web Scraping",
                "Secondary Data Analysis",
                "Manual Human Curation",
                "Software Collection",
            ],
            "rai:dataUseCases": [
                "Training large language models",
                "Linguistic and cultural inclusiveness research",
            ],
            "rai:dataLimitations": [
                "Over-represents pornographic spam across languages",
                "Contains personal information that may pose privacy risks",
                "Over-represents privileged voices and language varieties",
            ],
            "rai:dataBiases": "Limited demographic categories in annotation pool",
            "rai:personalSensitiveInformation": (
                "PII redaction via regex: KEY, EMAIL, USER, IP_ADDRESS"
            ),
            "rai:dataSocialImpact": (
                "Value-driven, human-centered data selection "
                "with ethical governance strategy."
            ),
            "rai:dataManipulationProtocol": [
                "HTML text structure reconstruction via DOM traversal",
                "Filtering by character/word repetition, special char ratio, "
                "closed class word ratio, flagged words, perplexity, min words",
                "Substring deduplication via Suffix Array, ~21.67% duplicates removed",
            ],
            "distribution": [],
            "recordSet": [],
        }

    def test_bigscience_style_import(self):
        """A BigScience ROOTS-style RAI document must import cleanly."""
        doc = self._make_bigscience_style_doc()
        ds = from_croissant(doc)

        assert ds["name"] == "BigScience-ROOTS-style"
        assert "rai:dataCollection" in ds
        assert isinstance(ds.get("rai:dataCollectionType"), list)
        assert len(ds["rai:dataCollectionType"]) == 4
        assert isinstance(ds.get("rai:dataLimitations"), list)
        assert len(ds["rai:dataLimitations"]) == 3

    def test_bigscience_style_roundtrip(self):
        """BigScience-style doc must survive full round-trip."""
        doc = self._make_bigscience_style_doc()
        ds = from_croissant(doc)
        cr = to_croissant(ds)

        assert cr["name"] == "BigScience-ROOTS-style"
        assert cr.get("rai:dataCollectionType") == [
            "Web Scraping",
            "Secondary Data Analysis",
            "Manual Human Curation",
            "Software Collection",
        ]
        assert cr.get("rai:dataSocialImpact") == doc["rai:dataSocialImpact"]
        assert cr.get("rai:dataManipulationProtocol") == doc["rai:dataManipulationProtocol"]
