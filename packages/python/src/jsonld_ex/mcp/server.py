"""
jsonld-ex MCP Server — Model Context Protocol integration.

Exposes jsonld-ex capabilities as MCP tools for LLM agents.
16 read-only tools across 6 groups: AI/ML annotation, confidence
algebra, security, vector operations, graph operations, and
interoperability.

Usage::

    python -m jsonld_ex.mcp          # stdio transport (default)
    python -m jsonld_ex.mcp --http   # streamable HTTP

Requires: pip install jsonld-ex[mcp]
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from jsonld_ex.ai_ml import (
    annotate,
    get_confidence,
    filter_by_confidence as _filter_by_confidence_raw,
)
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    robust_fuse,
)
from jsonld_ex.confidence_decay import (
    decay_opinion as _decay_opinion_raw,
    exponential_decay,
    linear_decay,
    step_decay,
)
from jsonld_ex.security import (
    compute_integrity as _compute_integrity_raw,
    verify_integrity as _verify_integrity_raw,
)
from jsonld_ex.validation import validate_node, ValidationResult
from jsonld_ex.vector import validate_vector as _validate_vector_raw
from jsonld_ex.merge import merge_graphs as _merge_graphs_raw
from jsonld_ex.temporal import query_at_time as _query_at_time_raw
from jsonld_ex.owl_interop import (
    to_prov_o as _to_prov_o_raw,
    shape_to_shacl as _shape_to_shacl_raw,
)


# ═══════════════════════════════════════════════════════════════════
# Server instance
# ═══════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "jsonld-ex",
    instructions=(
        "JSON-LD 1.2 extensions for AI/ML data exchange. "
        "Provides confidence algebra (Subjective Logic), "
        "provenance tracking, security verification, vector "
        "operations, temporal queries, and standards interop "
        "(PROV-O, SHACL). All tools are read-only and stateless."
    ),
)


# ── Helpers ────────────────────────────────────────────────────────

def _parse_json(s: str, label: str = "input") -> Any:
    """Parse a JSON string, raising ValueError with a clear message."""
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {label}: {e}") from e


def _opinion_from_dict(d: dict) -> Opinion:
    """Create an Opinion from a dict with belief/disbelief/uncertainty/base_rate."""
    return Opinion(
        belief=d["belief"],
        disbelief=d["disbelief"],
        uncertainty=d["uncertainty"],
        base_rate=d.get("base_rate", 0.5),
    )


def _opinion_to_dict(o: Opinion) -> dict:
    """Serialize an Opinion to a plain dict."""
    return {
        "belief": o.belief,
        "disbelief": o.disbelief,
        "uncertainty": o.uncertainty,
        "base_rate": o.base_rate,
        "projected_probability": o.projected_probability(),
    }


_DECAY_FUNCTIONS = {
    "exponential": exponential_decay,
    "linear": linear_decay,
    "step": step_decay,
}

_MERGE_STRATEGIES = {"confidence", "source_priority", "latest", "highest", "weighted_vote", "union", "recency"}
_FUSION_METHODS = {"cumulative", "averaging", "robust"}


# ═══════════════════════════════════════════════════════════════════
# Group 1: AI/ML Annotation Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def annotate_value(
    value: Any,
    confidence: Optional[float] = None,
    source: Optional[str] = None,
    extracted_at: Optional[str] = None,
    method: Optional[str] = None,
    human_verified: Optional[bool] = None,
) -> dict:
    """Create an annotated JSON-LD value with AI/ML provenance metadata.

    Wraps a value with optional @confidence, @source, @extractedAt,
    @method, and @humanVerified annotations following the jsonld-ex
    extension vocabulary. Use this to tag any data extracted by an
    ML model or tool with its provenance chain.

    Args:
        value: The value to annotate (string, number, bool, etc.).
        confidence: Confidence score in [0.0, 1.0].
        source: IRI of the data source or model that produced the value.
        extracted_at: ISO 8601 timestamp of extraction.
        method: Description of the extraction method (e.g. "NER", "OCR").
        human_verified: Whether a human has verified this value.

    Returns:
        Annotated JSON-LD value dict with @value and metadata keys.
    """
    return annotate(
        value=value,
        confidence=confidence,
        source=source,
        extracted_at=extracted_at,
        method=method,
        human_verified=human_verified,
    )


@mcp.tool()
def get_confidence_score(node_json: str) -> Optional[float]:
    """Extract the @confidence score from a JSON-LD annotated value node.

    Handles both compact form ({"@confidence": 0.9}) and expanded
    form (with full namespace IRIs).

    Args:
        node_json: JSON string of an annotated value node.

    Returns:
        The confidence score as a float, or null if none found.
    """
    node = _parse_json(node_json, "node_json")
    return get_confidence(node)


@mcp.tool()
def filter_by_confidence(document_json: str, min_confidence: float) -> str:
    """Filter a JSON-LD document's @graph nodes by minimum confidence.

    Walks the document's @graph array and retains only nodes where
    at least one annotated property meets or exceeds min_confidence.

    Args:
        document_json: JSON string of a JSON-LD document with @graph.
        min_confidence: Minimum confidence threshold in [0.0, 1.0].

    Returns:
        JSON string of the filtered document.
    """
    doc = _parse_json(document_json, "document_json")
    graph = doc.get("@graph", [])

    # Walk each node, keep if any annotated value meets threshold
    kept = []
    for node in graph:
        dominated = False
        for key, val in node.items():
            if key.startswith("@"):
                continue
            values = val if isinstance(val, list) else [val]
            for v in values:
                c = get_confidence(v)
                if c is not None and c >= min_confidence:
                    dominated = True
                    break
            if dominated:
                break
        if dominated:
            kept.append(node)

    result = dict(doc)
    result["@graph"] = kept
    return json.dumps(result)


# ═══════════════════════════════════════════════════════════════════
# Group 2: Confidence Algebra Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def create_opinion(
    belief: float,
    disbelief: float,
    uncertainty: float,
    base_rate: float = 0.5,
) -> dict:
    """Create a Subjective Logic opinion (Jøsang 2016).

    An opinion ω = (b, d, u, a) represents nuanced belief that
    distinguishes evidence-for, evidence-against, and absence of
    evidence — unlike a scalar confidence score.

    Constraint: belief + disbelief + uncertainty must equal 1.0.

    Args:
        belief: Evidence FOR the proposition, in [0, 1].
        disbelief: Evidence AGAINST the proposition, in [0, 1].
        uncertainty: Lack of evidence, in [0, 1].
        base_rate: Prior probability, in [0, 1]. Default 0.5.

    Returns:
        Dict with belief, disbelief, uncertainty, base_rate,
        and projected_probability.
    """
    o = Opinion(belief=belief, disbelief=disbelief,
                uncertainty=uncertainty, base_rate=base_rate)
    return _opinion_to_dict(o)


@mcp.tool()
def fuse_opinions(opinions_json: str, method: str = "cumulative") -> dict:
    """Fuse multiple Subjective Logic opinions into one.

    Combines evidence from independent or correlated sources using
    formally proven operators from Jøsang's Subjective Logic.

    Methods:
        - "cumulative": For independent sources. Reduces uncertainty
          additively. Commutative and associative.
        - "averaging": For correlated/dependent sources. Avoids
          double-counting. Commutative but NOT associative.
        - "robust": Byzantine-resistant. Detects and removes outlier
          agents before cumulative fusion.

    Args:
        opinions_json: JSON array of opinion objects, each with
            belief, disbelief, uncertainty, and optional base_rate.
        method: Fusion method — "cumulative", "averaging", or "robust".

    Returns:
        Fused opinion dict with belief, disbelief, uncertainty,
        base_rate, and projected_probability.
    """
    if method not in _FUSION_METHODS:
        raise ValueError(f"Unknown fusion method: {method!r}. Choose from: {_FUSION_METHODS}")

    data = _parse_json(opinions_json, "opinions_json")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("opinions_json must be a non-empty JSON array")

    opinions = [_opinion_from_dict(d) for d in data]

    if method == "cumulative":
        fused = cumulative_fuse(*opinions)
    elif method == "averaging":
        fused = averaging_fuse(*opinions)
    elif method == "robust":
        fused, _removed = robust_fuse(opinions)
    else:
        raise ValueError(f"Unknown method: {method}")

    return _opinion_to_dict(fused)


@mcp.tool()
def discount_opinion(opinion_json: str, trust_json: str) -> dict:
    """Discount an opinion through a trust chain (Jøsang §14.3).

    If agent A trusts agent B with trust opinion ω_AB, and B holds
    opinion ω_Bx about proposition x, then A's derived opinion is
    the trust-discounted result. Full trust preserves the opinion;
    zero trust yields total uncertainty.

    Args:
        opinion_json: JSON string of the opinion to discount.
        trust_json: JSON string of the trust opinion toward the source.

    Returns:
        Discounted opinion dict.
    """
    op_data = _parse_json(opinion_json, "opinion_json")
    trust_data = _parse_json(trust_json, "trust_json")
    opinion = _opinion_from_dict(op_data)
    trust = _opinion_from_dict(trust_data)
    result = trust_discount(trust, opinion)
    return _opinion_to_dict(result)


@mcp.tool()
def decay_opinion(
    opinion_json: str,
    elapsed_seconds: float,
    half_life_seconds: float,
    decay_function: str = "exponential",
) -> dict:
    """Apply temporal decay to a Subjective Logic opinion.

    Models evidence aging: as time passes, belief and disbelief
    migrate toward uncertainty, reflecting that old evidence is
    less reliable. The belief/disbelief ratio is preserved.

    Args:
        opinion_json: JSON string of the opinion to decay.
        elapsed_seconds: Time elapsed since opinion was formed.
        half_life_seconds: Time for belief/disbelief to halve.
        decay_function: "exponential" (default), "linear", or "step".

    Returns:
        Decayed opinion dict.
    """
    if decay_function not in _DECAY_FUNCTIONS:
        raise ValueError(
            f"Unknown decay function: {decay_function!r}. "
            f"Choose from: {set(_DECAY_FUNCTIONS.keys())}"
        )

    op_data = _parse_json(opinion_json, "opinion_json")
    opinion = _opinion_from_dict(op_data)
    fn = _DECAY_FUNCTIONS[decay_function]
    result = _decay_opinion_raw(opinion, elapsed_seconds, half_life_seconds, decay_fn=fn)
    return _opinion_to_dict(result)


# ═══════════════════════════════════════════════════════════════════
# Group 3: Security & Integrity Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def compute_integrity(context_json: str, algorithm: str = "sha256") -> str:
    """Compute a cryptographic integrity hash for a JSON-LD context.

    Produces a Subresource Integrity (SRI) style hash string that
    can be used to verify context documents haven't been tampered
    with. Supports SHA-256, SHA-384, and SHA-512.

    Args:
        context_json: JSON string of the context to hash.
        algorithm: Hash algorithm — "sha256" (default), "sha384", "sha512".

    Returns:
        Integrity string in "{algorithm}-{base64hash}" format.
    """
    return _compute_integrity_raw(context_json, algorithm)


@mcp.tool()
def verify_integrity(context_json: str, declared_hash: str) -> bool:
    """Verify a JSON-LD context against its declared integrity hash.

    Recomputes the hash and compares to the declared value.
    Returns True if the context is unmodified, False if tampered.

    Args:
        context_json: JSON string of the context to verify.
        declared_hash: Expected integrity hash string.

    Returns:
        True if hashes match, False otherwise.
    """
    return _verify_integrity_raw(context_json, declared_hash)


@mcp.tool()
def validate_document(document_json: str, shape_json: str) -> dict:
    """Validate a JSON-LD document against a shape definition.

    Checks property constraints including @required, @type, @minimum,
    @maximum, @minLength, @maxLength, and @pattern. This is the
    jsonld-ex native validation that maps bidirectionally to SHACL.

    Args:
        document_json: JSON string of the document node to validate.
        shape_json: JSON string of the shape definition.

    Returns:
        Dict with 'valid' (bool) and 'errors' (list of error dicts).
    """
    doc = _parse_json(document_json, "document_json")
    shape = _parse_json(shape_json, "shape_json")
    result: ValidationResult = validate_node(doc, shape)
    return {
        "valid": result.valid,
        "errors": [
            {"path": e.path, "constraint": e.constraint, "message": e.message}
            for e in result.errors
        ],
        "warnings": [
            {"path": w.path, "code": w.code, "message": w.message}
            for w in result.warnings
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# Group 4: Vector Operations Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def cosine_similarity(vector_a_json: str, vector_b_json: str) -> float:
    """Compute cosine similarity between two embedding vectors.

    Standard cosine similarity: dot(a,b) / (||a|| * ||b||).
    Returns a value in [-1, 1] where 1 = identical direction,
    0 = orthogonal, -1 = opposite.

    Args:
        vector_a_json: JSON array of floats for vector A.
        vector_b_json: JSON array of floats for vector B.

    Returns:
        Cosine similarity as a float.
    """
    a = _parse_json(vector_a_json, "vector_a_json")
    b = _parse_json(vector_b_json, "vector_b_json")
    if not isinstance(a, list) or not isinstance(b, list):
        raise ValueError("Both inputs must be JSON arrays of numbers")
    if len(a) != len(b):
        raise ValueError(f"Dimension mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("Vectors must not be empty")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        raise ValueError("Cannot compute cosine similarity with zero-magnitude vector")

    return dot / (norm_a * norm_b)


@mcp.tool()
def validate_vector(node_json: str) -> dict:
    """Validate a vector embedding node.

    Checks that the vector is a non-empty list of finite numbers
    and optionally validates dimension count against @dimensions.

    Args:
        node_json: JSON string of a node with vector data.
            Expected format: {"@value": [0.1, 0.2, ...], "@dimensions": 3}

    Returns:
        Dict with 'valid' (bool), 'dimensions' (int), and 'errors' (list).
    """
    node = _parse_json(node_json, "node_json")
    vector = node.get("@value", node)
    expected_dims = node.get("@dimensions")

    valid, errors = _validate_vector_raw(vector, expected_dims)
    return {
        "valid": valid,
        "dimensions": len(vector) if isinstance(vector, (list, tuple)) else 0,
        "errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════
# Group 5: Graph Operations Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def merge_graphs(
    graph_a_json: str,
    graph_b_json: str,
    strategy: str = "highest",
) -> str:
    """Merge two JSON-LD graphs with confidence-aware conflict resolution.

    Aligns nodes by @id, boosts confidence where sources agree
    (noisy-OR), and resolves conflicts using the chosen strategy.

    Strategies:
        - "highest": Pick value with highest @confidence.
        - "weighted_vote": Weighted majority by confidence.
        - "union": Keep all conflicting values.
        - "recency": Pick most recently extracted value.

    Args:
        graph_a_json: JSON string of the first graph document.
        graph_b_json: JSON string of the second graph document.
        strategy: Conflict resolution strategy.

    Returns:
        JSON string of the merged graph document.
    """
    a = _parse_json(graph_a_json, "graph_a_json")
    b = _parse_json(graph_b_json, "graph_b_json")
    merged, _report = _merge_graphs_raw([a, b], conflict_strategy=strategy)
    return json.dumps(merged)


@mcp.tool()
def query_at_time(document_json: str, query_time: str) -> str:
    """Query a JSON-LD document for the state at a specific time.

    Filters properties by their @validFrom/@validUntil temporal
    bounds. Properties without temporal bounds are treated as
    always-valid.

    Args:
        document_json: JSON string of a document with temporal annotations.
        query_time: ISO 8601 timestamp to query at.

    Returns:
        JSON string of the temporally filtered document.
    """
    doc = _parse_json(document_json, "document_json")

    # Handle both single-node and @graph documents
    if "@graph" in doc:
        nodes = doc["@graph"]
    else:
        nodes = [doc]

    result = _query_at_time_raw(nodes, query_time)

    if "@graph" in doc:
        out = dict(doc)
        out["@graph"] = result
        return json.dumps(out)
    elif result:
        return json.dumps(result[0])
    else:
        return json.dumps({})


# ═══════════════════════════════════════════════════════════════════
# Group 6: Interoperability Tools
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def to_prov_o(node_json: str) -> str:
    """Convert jsonld-ex annotations to W3C PROV-O provenance graph.

    Maps @confidence, @source, @extractedAt, @method, and
    @humanVerified to standard PROV-O entities, activities, and
    agents. Enables interop with any PROV-O consuming system.

    Args:
        node_json: JSON string of a jsonld-ex annotated document.

    Returns:
        JSON string of the equivalent PROV-O graph.
    """
    doc = _parse_json(node_json, "node_json")
    prov_doc, _report = _to_prov_o_raw(doc)
    return json.dumps(prov_doc)


@mcp.tool()
def shape_to_shacl(shape_json: str, target_class: str) -> str:
    """Convert a jsonld-ex @shape to W3C SHACL constraint graph.

    Maps @required to sh:minCount, @type to sh:datatype,
    @minimum/@maximum to sh:minInclusive/sh:maxInclusive, etc.
    Enables validation interop with any SHACL processor.

    Args:
        shape_json: JSON string of a jsonld-ex shape definition.
        target_class: IRI of the target class for the SHACL shape.

    Returns:
        JSON string of the equivalent SHACL shape graph.
    """
    shape = _parse_json(shape_json, "shape_json")
    shacl_doc = _shape_to_shacl_raw(shape, target_class=target_class)
    return json.dumps(shacl_doc)


# ═══════════════════════════════════════════════════════════════════
# Resources
# ═══════════════════════════════════════════════════════════════════


@mcp.resource("jsonld-ex://context/ai-ml")
def get_ai_ml_context() -> str:
    """JSON-LD context for AI/ML annotation extensions."""
    return json.dumps({
        "@context": {
            "jsonld-ex": "http://www.w3.org/ns/jsonld-ex/",
            "@confidence": "jsonld-ex:confidence",
            "@source": "jsonld-ex:source",
            "@extractedAt": "jsonld-ex:extractedAt",
            "@method": "jsonld-ex:method",
            "@humanVerified": "jsonld-ex:humanVerified",
        }
    })


@mcp.resource("jsonld-ex://context/security")
def get_security_context() -> str:
    """JSON-LD context for security extensions."""
    return json.dumps({
        "@context": {
            "jsonld-ex": "http://www.w3.org/ns/jsonld-ex/",
            "@integrity": "jsonld-ex:integrity",
            "@maxContextDepth": "jsonld-ex:maxContextDepth",
            "@maxGraphDepth": "jsonld-ex:maxGraphDepth",
            "@maxDocumentSize": "jsonld-ex:maxDocumentSize",
        }
    })


@mcp.resource("jsonld-ex://schema/opinion")
def get_opinion_schema() -> str:
    """JSON Schema for a Subjective Logic opinion object."""
    return json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SubjectiveLogicOpinion",
        "description": "A Subjective Logic opinion per Jøsang (2016).",
        "type": "object",
        "required": ["belief", "disbelief", "uncertainty"],
        "properties": {
            "belief": {"type": "number", "minimum": 0, "maximum": 1},
            "disbelief": {"type": "number", "minimum": 0, "maximum": 1},
            "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
            "base_rate": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        },
        "additionalProperties": False,
    })


# ═══════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════


@mcp.prompt()
def annotate_tool_results(tool_name: str, result_json: str) -> str:
    """Guide for adding provenance annotations to MCP tool outputs.

    Generates a workflow prompt for wrapping any tool's output
    with jsonld-ex provenance metadata (@source, @confidence,
    @extractedAt, @method).
    """
    return (
        f"You have a result from the '{tool_name}' tool:\n"
        f"```json\n{result_json}\n```\n\n"
        "To add provenance tracking, use the annotate_value tool:\n"
        "1. Call annotate_value with:\n"
        f"   - value: the result above (or individual fields)\n"
        f"   - source: the tool/model IRI (e.g. 'mcp://{tool_name}')\n"
        "   - confidence: your confidence in the result (0.0 to 1.0)\n"
        "   - extracted_at: current ISO 8601 timestamp\n"
        f"   - method: '{tool_name}'\n"
        "2. The annotated result can then be embedded in a JSON-LD document.\n"
        "3. Use compute_integrity to generate a hash for tamper detection.\n"
    )


@mcp.prompt()
def trust_chain_analysis(chain_description: str) -> str:
    """Guide for multi-hop trust propagation analysis.

    Generates a step-by-step workflow for evaluating trust
    through a chain of agents using trust discounting and
    fusion operators.
    """
    return (
        f"Analyze the trust chain: {chain_description}\n\n"
        "Step-by-step workflow:\n"
        "1. For each agent in the chain, create_opinion with their "
        "belief, disbelief, and uncertainty about the claim.\n"
        "2. For each trust link (A trusts B), create_opinion with "
        "A's trust assessment of B.\n"
        "3. Apply discount_opinion to propagate each opinion through "
        "its trust link (right to left through the chain).\n"
        "4. If multiple independent chains reach the same conclusion, "
        "use fuse_opinions with method='cumulative' to combine.\n"
        "5. If chains share sources (correlated), use method='averaging'.\n"
        "6. If you suspect adversarial agents, use method='robust'.\n"
        "7. Check the final projected_probability and uncertainty "
        "to assess overall confidence.\n"
    )
