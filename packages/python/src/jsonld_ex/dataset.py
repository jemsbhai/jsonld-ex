"""
Dataset Metadata Extensions for JSON-LD (GAP-D1, D2, D3)

Provides dataset-level metadata compatible with schema.org/Dataset
and bidirectional interoperability with Croissant (MLCommons).

Layers:
  D1 — Dataset metadata creation and validation
  D2 — Distribution (FileObject/FileSet) and RecordSet structure
  D3 — Croissant interoperability (to_croissant / from_croissant)
"""

from __future__ import annotations

import copy
from typing import Any, Optional, Sequence, Union

from jsonld_ex.validation import validate_node, ValidationResult

# ── Namespace constants ─────────────────────────────────────────────

SCHEMA_ORG = "https://schema.org/"
CROISSANT_NS = "http://mlcommons.org/croissant/"
CROISSANT_SPEC_VERSION = "http://mlcommons.org/croissant/1.0"
DCT_NS = "http://purl.org/dc/terms/"

# ── Contexts ────────────────────────────────────────────────────────

DATASET_CONTEXT: dict[str, Any] = {
    "@vocab": SCHEMA_ORG,
    "sc": SCHEMA_ORG,
    "cr": CROISSANT_NS,
    "dct": DCT_NS,
    "citeAs": "cr:citeAs",
    "conformsTo": "dct:conformsTo",
    "recordSet": "cr:recordSet",
    "field": "cr:field",
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "source": "cr:source",
    "extract": "cr:extract",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "isLiveDataset": "cr:isLiveDataset",
    "@language": "en",
}

CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": SCHEMA_ORG,
    "sc": SCHEMA_ORG,
    "cr": CROISSANT_NS,
    "dct": DCT_NS,
    "rai": "http://mlcommons.org/croissant/RAI/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

# ── Validation shape ────────────────────────────────────────────────

DATASET_SHAPE: dict[str, Any] = {
    "@type": "sc:Dataset",
    "name": {
        "@required": True,
        "@type": "xsd:string",
        "@minLength": 1,
    },
}

# ── Dataset-level metadata fields recognized by schema.org/Dataset ─
# Used to distinguish metadata fields from extension data during interop.

_SCHEMA_DATASET_FIELDS = {
    "@context", "@type", "@id",
    "name", "description", "version", "license", "url",
    "datePublished", "dateCreated", "dateModified",
    "creator", "publisher", "keywords", "inLanguage",
    "sameAs", "citeAs", "citation",
    "distribution", "recordSet",
    "conformsTo", "isLiveDataset",
}


# ── GAP-D1: Dataset Metadata ────────────────────────────────────────


def create_dataset_metadata(
    name: str,
    *,
    description: Optional[str] = None,
    version: Optional[str] = None,
    license: Optional[str] = None,
    url: Optional[str] = None,
    date_published: Optional[str] = None,
    creator: Optional[Union[str, dict, list]] = None,
    keywords: Optional[Union[str, list[str]]] = None,
    citation: Optional[str] = None,
    publisher: Optional[Union[str, dict]] = None,
    in_language: Optional[Union[str, list[str]]] = None,
    same_as: Optional[Union[str, list[str]]] = None,
    date_created: Optional[str] = None,
    date_modified: Optional[str] = None,
    is_live: Optional[bool] = None,
) -> dict[str, Any]:
    """Create a JSON-LD Dataset metadata document.

    Returns a full JSON-LD document with ``@context`` and ``@type``
    set to ``sc:Dataset``.  All optional fields are omitted when
    ``None`` to keep the output clean.

    Parameters
    ----------
    name : str
        Dataset name (required, non-empty).
    description : str, optional
        Human-readable description.
    version : str, optional
        Semantic version string (e.g. ``"1.0.0"``).
    license : str, optional
        License URL or identifier.
    url : str, optional
        Landing page URL for the dataset.
    date_published : str, optional
        ISO 8601 date string.
    creator : str | dict | list, optional
        Creator(s).  Strings are wrapped as ``Person`` nodes.
    keywords : str | list[str], optional
        Keywords / tags.
    citation : str, optional
        BibTeX citation string.
    publisher : str | dict, optional
        Publisher.  Strings are wrapped as ``Organization`` nodes.
    in_language : str | list[str], optional
        Language tag(s) (e.g. ``"en"``).
    same_as : str | list[str], optional
        Alternate URL(s) for the same dataset.
    date_created : str, optional
        ISO 8601 creation date.
    date_modified : str, optional
        ISO 8601 last-modified date.
    is_live : bool, optional
        Whether this is a live (continuously updated) dataset.

    Returns
    -------
    dict
        A JSON-LD document representing the dataset metadata.
    """
    if not name or not isinstance(name, str):
        raise ValueError("Dataset name must be a non-empty string")

    doc: dict[str, Any] = {
        "@context": copy.deepcopy(DATASET_CONTEXT),
        "@type": "sc:Dataset",
        "name": name,
    }

    if description is not None:
        doc["description"] = description
    if version is not None:
        doc["version"] = version
    if license is not None:
        doc["license"] = license
    if url is not None:
        doc["url"] = url
    if date_published is not None:
        doc["datePublished"] = date_published
    if creator is not None:
        doc["creator"] = _normalize_creator(creator)
    if keywords is not None:
        doc["keywords"] = [keywords] if isinstance(keywords, str) else keywords
    if citation is not None:
        doc["citeAs"] = citation
    if publisher is not None:
        doc["publisher"] = _normalize_publisher(publisher)
    if in_language is not None:
        doc["inLanguage"] = in_language
    if same_as is not None:
        doc["sameAs"] = same_as
    if date_created is not None:
        doc["dateCreated"] = date_created
    if date_modified is not None:
        doc["dateModified"] = date_modified
    if is_live is not None:
        doc["isLiveDataset"] = is_live

    # Initialize empty containers for resources and structure
    doc["distribution"] = []
    doc["recordSet"] = []

    return doc


def validate_dataset_metadata(doc: dict[str, Any]) -> ValidationResult:
    """Validate a dataset metadata document against :data:`DATASET_SHAPE`.

    Uses the library's own ``validate_node`` so that jsonld-ex
    validation dogfoods itself.
    """
    return validate_node(doc, DATASET_SHAPE)


# ── GAP-D2: Distributions and Structure ─────────────────────────────


def add_distribution(
    dataset: dict[str, Any],
    *,
    name: str,
    content_url: str,
    encoding_format: str,
    sha256: Optional[str] = None,
    content_size: Optional[str] = None,
    description: Optional[str] = None,
    file_id: Optional[str] = None,
) -> dict[str, Any]:
    """Add a ``cr:FileObject`` to the dataset's distribution list.

    Parameters
    ----------
    dataset : dict
        The dataset document to extend.
    name : str
        File name (also used as ``@id`` if *file_id* not given).
    content_url : str
        Download URL for the file.
    encoding_format : str
        MIME type (e.g. ``"text/csv"``).
    sha256 : str, optional
        SHA-256 checksum of the file content.
    content_size : str, optional
        Human-readable size (e.g. ``"25585843 B"``).
    description : str, optional
        Description of the file.
    file_id : str, optional
        Explicit ``@id``; defaults to *name*.

    Returns
    -------
    dict
        The dataset document with the new FileObject appended.
    """
    if not name:
        raise ValueError("Distribution name must be non-empty")

    dataset = copy.deepcopy(dataset)

    fo: dict[str, Any] = {
        "@type": "cr:FileObject",
        "@id": file_id or name,
        "name": name,
        "contentUrl": content_url,
        "encodingFormat": encoding_format,
    }
    if sha256 is not None:
        fo["sha256"] = sha256
    if content_size is not None:
        fo["contentSize"] = content_size
    if description is not None:
        fo["description"] = description

    dataset.setdefault("distribution", []).append(fo)
    return dataset


def add_file_set(
    dataset: dict[str, Any],
    *,
    name: str,
    contained_in: str,
    encoding_format: str,
    includes: str,
    description: Optional[str] = None,
    file_set_id: Optional[str] = None,
) -> dict[str, Any]:
    """Add a ``cr:FileSet`` to the dataset's distribution list.

    Parameters
    ----------
    dataset : dict
        The dataset document to extend.
    name : str
        Name of the file set (also used as ``@id`` if *file_set_id* not given).
    contained_in : str
        ``@id`` of the parent FileObject (e.g. an archive).
    encoding_format : str
        MIME type of the contained files (e.g. ``"image/jpeg"``).
    includes : str
        Glob pattern for included files (e.g. ``"*.jpg"``).
    description : str, optional
        Description of the file set.
    file_set_id : str, optional
        Explicit ``@id``; defaults to *name*.

    Returns
    -------
    dict
        The dataset document with the new FileSet appended.
    """
    dataset = copy.deepcopy(dataset)

    fs: dict[str, Any] = {
        "@type": "cr:FileSet",
        "@id": file_set_id or name,
        "name": name,
        "containedIn": {"@id": contained_in},
        "encodingFormat": encoding_format,
        "includes": includes,
    }
    if description is not None:
        fs["description"] = description

    dataset.setdefault("distribution", []).append(fs)
    return dataset


def create_field(
    name: str,
    *,
    data_type: str,
    description: Optional[str] = None,
    source: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a ``cr:Field`` definition for use in a RecordSet.

    The ``@id`` is set later by :func:`add_record_set` to follow
    the Croissant convention ``recordset_id/field_name``.

    Parameters
    ----------
    name : str
        Field name.
    data_type : str
        Data type (e.g. ``"sc:Text"``, ``"sc:Integer"``, ``"sc:Float"``).
    description : str, optional
        Human-readable description.
    source : dict, optional
        Data source definition (fileObject/fileSet reference + extract).

    Returns
    -------
    dict
        A ``cr:Field`` JSON-LD node.
    """
    f: dict[str, Any] = {
        "@type": "cr:Field",
        "name": name,
        "dataType": data_type,
    }
    if description is not None:
        f["description"] = description
    if source is not None:
        f["source"] = source
    return f


def add_record_set(
    dataset: dict[str, Any],
    *,
    name: str,
    fields: Sequence[dict[str, Any]],
    description: Optional[str] = None,
    record_set_id: Optional[str] = None,
) -> dict[str, Any]:
    """Add a ``cr:RecordSet`` to the dataset.

    Field ``@id`` values are automatically prefixed with the
    RecordSet ``@id`` following the Croissant convention.

    Parameters
    ----------
    dataset : dict
        The dataset document to extend.
    name : str
        RecordSet name (also used as ``@id`` if *record_set_id* not given).
    fields : list[dict]
        List of fields created via :func:`create_field`.
    description : str, optional
        Description of the record set.
    record_set_id : str, optional
        Explicit ``@id``; defaults to *name*.

    Returns
    -------
    dict
        The dataset document with the new RecordSet appended.
    """
    dataset = copy.deepcopy(dataset)
    rs_id = record_set_id or name

    # Prefix field @ids with record set id (Croissant convention)
    prefixed_fields = []
    for f in fields:
        f = copy.deepcopy(f)
        f["@id"] = f"{rs_id}/{f['name']}"
        prefixed_fields.append(f)

    rs: dict[str, Any] = {
        "@type": "cr:RecordSet",
        "@id": rs_id,
        "name": name,
        "field": prefixed_fields,
    }
    if description is not None:
        rs["description"] = description

    dataset.setdefault("recordSet", []).append(rs)
    return dataset


# ── GAP-D3: Croissant Interoperability ──────────────────────────────


def to_croissant(dataset: dict[str, Any]) -> dict[str, Any]:
    """Convert a jsonld-ex dataset document to Croissant-compatible JSON-LD.

    Adds the Croissant ``@context``, ``conformsTo`` declaration,
    and ensures all types use Croissant prefixes.

    Parameters
    ----------
    dataset : dict
        A jsonld-ex dataset document (as returned by
        :func:`create_dataset_metadata`).

    Returns
    -------
    dict
        A Croissant-compatible JSON-LD document.
    """
    doc = copy.deepcopy(dataset)

    # Replace context with full Croissant context
    doc["@context"] = copy.deepcopy(CROISSANT_CONTEXT)

    # Ensure conformsTo is set
    doc["conformsTo"] = CROISSANT_SPEC_VERSION

    return doc


def from_croissant(croissant_doc: dict[str, Any]) -> dict[str, Any]:
    """Import a Croissant JSON-LD document into jsonld-ex format.

    Replaces the Croissant ``@context`` with the jsonld-ex
    :data:`DATASET_CONTEXT` and removes the ``conformsTo``
    declaration (which is Croissant-specific).

    Parameters
    ----------
    croissant_doc : dict
        A Croissant JSON-LD document.

    Returns
    -------
    dict
        A jsonld-ex dataset document.
    """
    doc = copy.deepcopy(croissant_doc)

    # Replace context with our own
    doc["@context"] = copy.deepcopy(DATASET_CONTEXT)

    # Remove Croissant-specific conformsTo
    doc.pop("conformsTo", None)

    return doc


# ── Internal helpers ────────────────────────────────────────────────


def _normalize_creator(
    creator: Union[str, dict, list],
) -> Union[dict, list[dict]]:
    """Normalize creator input to Person/Organization nodes."""
    if isinstance(creator, list):
        return [_normalize_single_creator(c) for c in creator]
    return _normalize_single_creator(creator)


def _normalize_single_creator(creator: Union[str, dict]) -> dict:
    """Wrap a string creator as a Person node; pass dicts through."""
    if isinstance(creator, str):
        return {"@type": "Person", "name": creator}
    return creator


def _normalize_publisher(publisher: Union[str, dict]) -> dict:
    """Wrap a string publisher as an Organization node."""
    if isinstance(publisher, str):
        return {"@type": "Organization", "name": publisher}
    return publisher
