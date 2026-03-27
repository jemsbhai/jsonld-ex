"""Vector Embedding Extensions for JSON-LD."""

from __future__ import annotations
import math
from typing import Any, Optional


# ── Maximum allowed bit-width for quantized vectors ────────────────────
_MAX_BIT_WIDTH = 32


def quantization_descriptor(
    method: str,
    bit_width: int,
    *,
    rotation_seed: Optional[int] = None,
    has_residual_qjl: Optional[bool] = None,
    codebook_size: Optional[int] = None,
    subvector_count: Optional[int] = None,
) -> dict[str, Any]:
    """Create a quantization metadata descriptor for a ``@vector`` field.

    Describes how a vector embedding has been (or should be) quantized,
    enabling downstream consumers to reconstruct, dequantize, or reason
    about compression fidelity.

    Parameters
    ----------
    method:
        Name of the quantization algorithm (e.g. ``"turboquant"``,
        ``"polarquant"``, ``"qjl"``, ``"scalar"``,
        ``"product_quantization"``).  Any non-empty string is accepted
        so that users can describe custom quantizers.
    bit_width:
        Bits per coordinate/element.  Must be a positive integer
        in ``[1, 32]``.
    rotation_seed:
        Optional RNG seed for random rotation (used by TurboQuant and
        PolarQuant).  Non-negative integer.
    has_residual_qjl:
        Whether a 1-bit QJL residual correction stage is applied
        (TurboQuant's two-stage approach).
    codebook_size:
        Number of codewords for codebook-based methods (e.g. product
        quantization).  Positive integer.
    subvector_count:
        Number of subvector partitions for product quantization.
        Positive integer.

    Returns
    -------
    dict
        A descriptor dict suitable for use as ``@quantization`` in a
        ``@vector`` term definition.
    """
    # ── method validation ──
    if not isinstance(method, str):
        raise TypeError(f"method must be a string, got: {type(method).__name__}")
    if not method.strip():
        raise ValueError("method must be a non-empty string")

    # ── bit_width validation ──
    if isinstance(bit_width, bool) or not isinstance(bit_width, int):
        raise TypeError(f"bitWidth must be an integer, got: {type(bit_width).__name__}")
    if bit_width < 1 or bit_width > _MAX_BIT_WIDTH:
        raise ValueError(
            f"bitWidth must be in [1, {_MAX_BIT_WIDTH}], got: {bit_width}"
        )

    desc: dict[str, Any] = {
        "method": method,
        "bitWidth": bit_width,
    }

    # ── optional fields ──
    if rotation_seed is not None:
        if isinstance(rotation_seed, bool) or not isinstance(rotation_seed, int):
            raise TypeError(
                f"rotationSeed must be an integer, got: {type(rotation_seed).__name__}"
            )
        if rotation_seed < 0:
            raise ValueError(f"rotationSeed must be non-negative, got: {rotation_seed}")
        desc["rotationSeed"] = rotation_seed

    if has_residual_qjl is not None:
        desc["hasResidualQJL"] = has_residual_qjl

    if codebook_size is not None:
        if isinstance(codebook_size, bool) or not isinstance(codebook_size, int):
            raise TypeError(
                f"codebookSize must be an integer, got: {type(codebook_size).__name__}"
            )
        if codebook_size < 1:
            raise ValueError(f"codebookSize must be positive, got: {codebook_size}")
        desc["codebookSize"] = codebook_size

    if subvector_count is not None:
        if isinstance(subvector_count, bool) or not isinstance(subvector_count, int):
            raise TypeError(
                f"subvectorCount must be an integer, got: {type(subvector_count).__name__}"
            )
        if subvector_count < 1:
            raise ValueError(
                f"subvectorCount must be positive, got: {subvector_count}"
            )
        desc["subvectorCount"] = subvector_count

    return desc


def validate_quantization_descriptor(
    desc: Any,
) -> tuple[bool, list[str]]:
    """Validate a quantization descriptor dict.

    Accepts dicts produced by :func:`quantization_descriptor` **or**
    deserialized from JSON documents.  Unknown fields are allowed for
    extensibility.

    Returns
    -------
    tuple[bool, list[str]]
        ``(valid, errors)`` where *valid* is ``True`` when all checks
        pass and *errors* lists human-readable problem descriptions.
    """
    errors: list[str] = []

    if not isinstance(desc, dict):
        errors.append(
            f"Quantization descriptor must be a dict, got: {type(desc).__name__}"
        )
        return False, errors

    # ── required: method ──
    method = desc.get("method")
    if method is None:
        errors.append("Missing required field 'method'")
    elif not isinstance(method, str):
        errors.append(f"'method' must be a string, got: {type(method).__name__}")
    elif not method.strip():
        errors.append("'method' must be a non-empty string")

    # ── required: bitWidth ──
    bit_width = desc.get("bitWidth")
    if bit_width is None:
        errors.append("Missing required field 'bitWidth'")
    elif isinstance(bit_width, bool) or not isinstance(bit_width, int):
        errors.append(f"'bitWidth' must be an integer, got: {type(bit_width).__name__}")
    elif bit_width < 1 or bit_width > _MAX_BIT_WIDTH:
        errors.append(f"'bitWidth' must be in [1, {_MAX_BIT_WIDTH}], got: {bit_width}")

    # ── optional: rotationSeed ──
    if "rotationSeed" in desc:
        rs = desc["rotationSeed"]
        if isinstance(rs, bool) or not isinstance(rs, int):
            errors.append(
                f"'rotationSeed' must be an integer, got: {type(rs).__name__}"
            )
        elif rs < 0:
            errors.append(f"'rotationSeed' must be non-negative, got: {rs}")

    # ── optional: hasResidualQJL ──
    if "hasResidualQJL" in desc:
        qjl = desc["hasResidualQJL"]
        if not isinstance(qjl, bool):
            errors.append(
                f"'hasResidualQJL' must be a boolean, got: {type(qjl).__name__}"
            )

    # ── optional: codebookSize ──
    if "codebookSize" in desc:
        cs = desc["codebookSize"]
        if isinstance(cs, bool) or not isinstance(cs, int):
            errors.append(
                f"'codebookSize' must be an integer, got: {type(cs).__name__}"
            )
        elif cs < 1:
            errors.append(f"'codebookSize' must be positive, got: {cs}")

    # ── optional: subvectorCount ──
    if "subvectorCount" in desc:
        sc = desc["subvectorCount"]
        if isinstance(sc, bool) or not isinstance(sc, int):
            errors.append(
                f"'subvectorCount' must be an integer, got: {type(sc).__name__}"
            )
        elif sc < 1:
            errors.append(f"'subvectorCount' must be positive, got: {sc}")

    return len(errors) == 0, errors


def vector_term_definition(
    term_name: str,
    iri: str,
    dimensions: Optional[int] = None,
    *,
    similarity: Optional[str] = None,
    quantization: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a context term definition for a vector embedding property.

    Parameters
    ----------
    term_name:
        The JSON key that will hold the vector in documents.
    iri:
        The IRI that this term maps to.
    dimensions:
        Optional expected dimensionality (positive integer).
    similarity:
        Optional name of the recommended similarity metric for this
        vector (e.g. ``"cosine"``, ``"euclidean"``).  Stored as
        ``@similarity`` in the term definition.  This is declarative
        metadata — validation against the metric registry happens at
        use-time, not at definition-time.
    quantization:
        Optional quantization descriptor dict (produced by
        :func:`quantization_descriptor` or equivalent).  Stored as
        ``@quantization`` in the term definition.  Describes the
        compression method, bit-width, and algorithm-specific
        parameters for quantized vector storage.
    """
    defn: dict[str, Any] = {"@id": iri, "@container": "@vector"}
    if dimensions is not None:
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
            raise ValueError(f"@dimensions must be a positive integer, got: {dimensions}")
        defn["@dimensions"] = dimensions
    if similarity is not None:
        if not isinstance(similarity, str):
            raise TypeError(
                f"similarity must be a string, got: {type(similarity).__name__}"
            )
        if not similarity.strip():
            raise ValueError("similarity must be a non-empty string")
        defn["@similarity"] = similarity
    if quantization is not None:
        if not isinstance(quantization, dict):
            raise TypeError(
                f"quantization must be a dict, got: {type(quantization).__name__}"
            )
        ok, errs = validate_quantization_descriptor(quantization)
        if not ok:
            raise ValueError(
                f"Invalid quantization descriptor: {'; '.join(errs)}"
            )
        defn["@quantization"] = quantization
    return {term_name: defn}


def validate_vector(
    vector: Any, expected_dimensions: Optional[int] = None
) -> tuple[bool, list[str]]:
    """Validate a vector embedding. Returns (valid, errors)."""
    errors: list[str] = []
    if not isinstance(vector, (list, tuple)):
        errors.append(f"Vector must be a list, got: {type(vector).__name__}")
        return False, errors
    if len(vector) == 0:
        errors.append("Vector must not be empty")
        return False, errors
    for i, v in enumerate(vector):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            errors.append(f"Vector element [{i}] must be a finite number, got: {v}")
    if expected_dimensions is not None and len(vector) != expected_dimensions:
        errors.append(
            f"Vector dimension mismatch: expected {expected_dimensions}, got {len(vector)}"
        )
    return len(errors) == 0, errors


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Raises ``ValueError`` when either vector has zero magnitude,
    since cosine similarity is mathematically undefined (0/0) in that case.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be empty")
    for i, (x, y) in enumerate(zip(a, b)):
        for label, v in (("a", x), ("b", y)):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise TypeError(f"Vector {label}[{i}] must be a number, got: {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                raise ValueError(f"Vector {label}[{i}] must be finite, got: {v}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero-magnitude vector")
    return dot / (norm_a * norm_b)


def extract_vectors(
    node: dict[str, Any], vector_properties: list[str]
) -> dict[str, list[float]]:
    """Extract vector embeddings from a JSON-LD node."""
    vectors: dict[str, list[float]] = {}
    if not isinstance(node, dict):
        return vectors
    for prop in vector_properties:
        value = node.get(prop)
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
            vectors[prop] = value
    return vectors


def strip_vectors_for_rdf(doc: Any, vector_properties: list[str]) -> Any:
    """Remove vector embeddings before RDF conversion."""
    if isinstance(doc, list):
        return [strip_vectors_for_rdf(item, vector_properties) for item in doc]
    if not isinstance(doc, dict):
        return doc
    return {
        k: strip_vectors_for_rdf(v, vector_properties)
        for k, v in doc.items()
        if k not in vector_properties
    }
