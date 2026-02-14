"""
jsonld-ex MCP Server — Model Context Protocol integration.

Exposes jsonld-ex capabilities as 51 MCP tools for LLM agents,
covering 10 groups: AI/ML annotation, confidence algebra (Subjective
Logic), confidence bridge, inference/propagation, security, vector
operations, graph operations, temporal extensions, standards
interoperability (PROV-O, SHACL, OWL, RDF-Star), MQTT/IoT transport
optimization, and compliance algebra (GDPR regulatory modeling).

Usage::

    python -m jsonld_ex.mcp          # stdio transport (default)
    python -m jsonld_ex.mcp --http   # streamable HTTP

Requires: pip install jsonld-ex[mcp]
"""

from __future__ import annotations

import base64
import json
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from jsonld_ex.ai_ml import (
    annotate,
    get_confidence,
    get_provenance as _get_provenance_raw,
)
from jsonld_ex.confidence_algebra import (
    Opinion,
    cumulative_fuse,
    averaging_fuse,
    trust_discount,
    deduce as _deduce_raw,
    pairwise_conflict as _pairwise_conflict_raw,
    conflict_metric as _conflict_metric_raw,
    robust_fuse,
)
from jsonld_ex.confidence_bridge import (
    combine_opinions_from_scalars as _combine_opinions_raw,
    propagate_opinions_from_scalars as _propagate_opinions_raw,
)
from jsonld_ex.confidence_decay import (
    decay_opinion as _decay_opinion_raw,
    exponential_decay,
    linear_decay,
    step_decay,
)
from jsonld_ex.inference import (
    propagate_confidence as _propagate_confidence_raw,
    combine_sources as _combine_sources_raw,
    resolve_conflict as _resolve_conflict_raw,
    propagate_graph_confidence as _propagate_graph_confidence_raw,
)
from jsonld_ex.security import (
    compute_integrity as _compute_integrity_raw,
    verify_integrity as _verify_integrity_raw,
    is_context_allowed as _is_context_allowed_raw,
    enforce_resource_limits as _enforce_resource_limits_raw,
)
from jsonld_ex.validation import validate_node, ValidationResult
from jsonld_ex.vector import (
    validate_vector as _validate_vector_raw,
    cosine_similarity as _cosine_similarity_raw,
)
from jsonld_ex.merge import (
    merge_graphs as _merge_graphs_raw,
    diff_graphs as _diff_graphs_raw,
)
from jsonld_ex.temporal import (
    add_temporal as _add_temporal_raw,
    query_at_time as _query_at_time_raw,
    temporal_diff as _temporal_diff_raw,
)
from jsonld_ex.owl_interop import (
    to_prov_o as _to_prov_o_raw,
    from_prov_o as _from_prov_o_raw,
    shape_to_shacl as _shape_to_shacl_raw,
    shacl_to_shape as _shacl_to_shape_raw,
    shape_to_owl_restrictions as _shape_to_owl_raw,
    to_rdf_star_ntriples as _to_rdf_star_raw,
    compare_with_prov_o as _compare_prov_o_raw,
    compare_with_shacl as _compare_shacl_raw,
)
from jsonld_ex.compliance_algebra import (
    ComplianceOpinion,
    jurisdictional_meet as _jurisdictional_meet_raw,
    compliance_propagation as _compliance_propagation_raw,
    consent_validity as _consent_validity_raw,
    withdrawal_override as _withdrawal_override_raw,
    expiry_trigger as _expiry_trigger_raw,
    review_due_trigger as _review_due_trigger_raw,
    regulatory_change_trigger as _regulatory_change_trigger_raw,
    erasure_scope_opinion as _erasure_scope_opinion_raw,
    residual_contamination as _residual_contamination_raw,
)

from jsonld_ex.mqtt import (
    to_mqtt_payload as _to_mqtt_payload_raw,
    from_mqtt_payload as _from_mqtt_payload_raw,
    derive_mqtt_topic as _derive_mqtt_topic_raw,
    derive_mqtt_qos_detailed as _derive_mqtt_qos_detailed_raw,
    derive_mqtt5_properties as _derive_mqtt5_properties_raw,
)


# ═══════════════════════════════════════════════════════════════════
# Server instance
# ═══════════════════════════════════════════════════════════════════

mcp = FastMCP(
    "jsonld-ex",
    instructions=(
        "JSON-LD 1.2 extensions for AI/ML data exchange. "
        "Provides confidence algebra (Subjective Logic), "
        "compliance algebra (GDPR regulatory modeling), "
        "provenance tracking, security verification, vector "
        "operations, temporal queries, standards interop "
        "(PROV-O, SHACL), and MQTT/IoT transport optimization. "
        "All tools are read-only and stateless."
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
# Group 1b: AI/ML — get_provenance
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def get_provenance(node_json: str) -> dict:
    """Extract all provenance metadata from a JSON-LD annotated node.

    Returns a structured dict with confidence, source, extracted_at,
    method, and human_verified fields. Fields not present on the node
    are returned as null.

    Args:
        node_json: JSON string of an annotated value node.

    Returns:
        Dict with all provenance metadata fields.
    """
    node = _parse_json(node_json, "node_json")
    prov = _get_provenance_raw(node)
    return {
        "confidence": prov.confidence,
        "source": prov.source,
        "extracted_at": prov.extracted_at,
        "method": prov.method,
        "human_verified": prov.human_verified,
    }


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
# Group 2b: Confidence Algebra — deduce, pairwise conflict, conflict metric
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def deduce_opinion(
    opinion_x_json: str,
    opinion_y_given_x_json: str,
    opinion_y_given_not_x_json: str,
) -> dict:
    """Subjective Logic deduction (Jøsang Def. 12.6).

    Given an opinion about proposition X and conditional opinions
    about Y given X and Y given ¬X, derive an opinion about Y.
    This is the Subjective Logic analogue of modus ponens.

    Args:
        opinion_x_json: JSON opinion about X.
        opinion_y_given_x_json: JSON opinion about Y assuming X is true.
        opinion_y_given_not_x_json: JSON opinion about Y assuming X is false.

    Returns:
        Deduced opinion about Y.
    """
    x = _opinion_from_dict(_parse_json(opinion_x_json, "opinion_x_json"))
    ygx = _opinion_from_dict(_parse_json(opinion_y_given_x_json, "opinion_y_given_x_json"))
    ygnx = _opinion_from_dict(_parse_json(opinion_y_given_not_x_json, "opinion_y_given_not_x_json"))
    result = _deduce_raw(x, ygx, ygnx)
    return _opinion_to_dict(result)


@mcp.tool()
def measure_pairwise_conflict(opinion_a_json: str, opinion_b_json: str) -> dict:
    """Measure pairwise conflict between two opinions (Jøsang §12.3.4).

    Computes con(A, B) = b_A·d_B + d_A·b_B. High conflict means
    the opinions strongly disagree (one believes what the other
    disbelieves). Zero when opinions agree or either is vacuous.

    Args:
        opinion_a_json: JSON opinion A.
        opinion_b_json: JSON opinion B.

    Returns:
        Dict with 'conflict' score in [0, 1].
    """
    a = _opinion_from_dict(_parse_json(opinion_a_json, "opinion_a_json"))
    b = _opinion_from_dict(_parse_json(opinion_b_json, "opinion_b_json"))
    return {"conflict": _pairwise_conflict_raw(a, b)}


@mcp.tool()
def measure_conflict(opinion_json: str) -> dict:
    """Measure internal conflict within a single opinion.

    Detects when an opinion has high belief AND high disbelief
    simultaneously (evidence cancelling out). Distinguishes
    genuine conflict (b ≈ d, low u) from ignorance (high u).

    Formula: conflict = 1 − |b − d| − u

    Args:
        opinion_json: JSON opinion to evaluate.

    Returns:
        Dict with 'conflict' score in [0, 1].
    """
    op = _opinion_from_dict(_parse_json(opinion_json, "opinion_json"))
    return {"conflict": _conflict_metric_raw(op)}


# ═══════════════════════════════════════════════════════════════════
# Group 2c: Confidence Bridge — scalar-to-opinion wrappers
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def combine_opinions_from_scalars(
    scores_json: str,
    uncertainty: float = 0.0,
    fusion: str = "cumulative",
    base_rate: float = 0.5,
) -> dict:
    """Combine scalar confidence scores via the formal algebra.

    Lifts scalar scores to Subjective Logic opinions, fuses them,
    and returns the full opinion (preserving uncertainty metadata).
    This bridges legacy confidence values to the richer algebra.

    Default mapping (uncertainty=0): each score p becomes
    Opinion(b=p, d=0, u=1−p), treating "not confident" as
    "uncertain" rather than "disbelieving."

    Args:
        scores_json: JSON array of confidence scores, each in [0, 1].
        uncertainty: Uncertainty to assign to each source (default 0.0).
        fusion: "cumulative" (independent) or "averaging" (correlated).
        base_rate: Prior probability for each opinion.

    Returns:
        Fused opinion dict.
    """
    scores = _parse_json(scores_json, "scores_json")
    result = _combine_opinions_raw(scores, uncertainty=uncertainty, fusion=fusion, base_rate=base_rate)
    return _opinion_to_dict(result)


@mcp.tool()
def propagate_opinions_from_scalars(
    chain_json: str,
    trust_uncertainty: float = 0.0,
    base_rate: float = 0.0,
) -> dict:
    """Propagate confidence through a chain via trust discount.

    Lifts scalar chain scores to trust opinions and applies
    iterated trust discount. With defaults (trust_uncertainty=0,
    base_rate=0), produces the same scalar as multiply-chain
    propagation, proving the exact equivalence.

    Args:
        chain_json: JSON array of confidence scores along the path.
        trust_uncertainty: Uncertainty in each trust link.
        base_rate: Prior probability for the assertion.

    Returns:
        Propagated opinion dict.
    """
    chain = _parse_json(chain_json, "chain_json")
    result = _propagate_opinions_raw(chain, trust_uncertainty=trust_uncertainty, base_rate=base_rate)
    return _opinion_to_dict(result)


# ═══════════════════════════════════════════════════════════════════
# Group 2d: Inference — scalar propagation and conflict resolution
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def propagate_confidence(chain_json: str, method: str = "multiply") -> dict:
    """Propagate confidence through an inference chain.

    Given scores [c₁, c₂, …, cₙ] representing confidence at each
    step of a derivation, compute the combined confidence.

    Methods:
        - "multiply": Product of all scores (conservative).
        - "bayesian": Sequential Bayesian update.
        - "min": Weakest-link (returns min).
        - "dampened": product^(1/√n) to prevent collapse.

    Args:
        chain_json: JSON array of confidence scores in [0, 1].
        method: Propagation method.

    Returns:
        Dict with 'score', 'method', and 'input_scores'.
    """
    chain = _parse_json(chain_json, "chain_json")
    result = _propagate_confidence_raw(chain, method=method)
    return {
        "score": result.score,
        "method": result.method,
        "input_scores": result.input_scores,
    }


@mcp.tool()
def combine_sources(scores_json: str, method: str = "noisy_or") -> dict:
    """Combine confidence from multiple independent sources.

    When multiple sources independently assert the same fact,
    the combined confidence should generally be higher.

    Methods:
        - "noisy_or": 1 − ∏(1−pᵢ). At-least-one correct (Pearl 1988).
        - "average": Arithmetic mean.
        - "max": Optimistic (highest score).
        - "dempster_shafer": Dempster's rule of combination.

    Args:
        scores_json: JSON array of confidence scores.
        method: Combination method.

    Returns:
        Dict with 'score', 'method', and 'input_scores'.
    """
    scores = _parse_json(scores_json, "scores_json")
    result = _combine_sources_raw(scores, method=method)
    return {
        "score": result.score,
        "method": result.method,
        "input_scores": result.input_scores,
    }


@mcp.tool()
def resolve_conflict(assertions_json: str, strategy: str = "highest") -> dict:
    """Resolve conflicting assertions using confidence and metadata.

    Given multiple assertions for the same property from different
    sources, select a winner and produce an auditable report.

    Strategies:
        - "highest": Pick highest @confidence.
        - "weighted_vote": Group by @value, combine via noisy-OR.
        - "recency": Prefer most recently extracted.

    Args:
        assertions_json: JSON array of assertions, each with @value
            and @confidence (and optionally @extractedAt for recency).
        strategy: Resolution strategy.

    Returns:
        Dict with 'winner', 'strategy', 'candidates',
        'confidence_scores', and 'reason'.
    """
    assertions = _parse_json(assertions_json, "assertions_json")
    report = _resolve_conflict_raw(assertions, strategy=strategy)
    return {
        "winner": report.winner,
        "strategy": report.strategy,
        "candidates": report.candidates,
        "confidence_scores": report.confidence_scores,
        "reason": report.reason,
    }


@mcp.tool()
def propagate_graph_confidence(
    document_json: str,
    property_chain_json: str,
    method: str = "multiply",
) -> dict:
    """Propagate confidence along a property chain in a JSON-LD graph.

    Extract the confidence at each step of a derivation path and
    compute the combined confidence of the final conclusion.

    Args:
        document_json: JSON string of a JSON-LD document.
        property_chain_json: JSON array of property names forming
            the inference path.
        method: Propagation method (multiply, bayesian, min, dampened).

    Returns:
        Dict with 'score', 'method', 'input_scores', and
        'provenance_trail'.
    """
    doc = _parse_json(document_json, "document_json")
    chain = _parse_json(property_chain_json, "property_chain_json")
    result = _propagate_graph_confidence_raw(doc, chain, method=method)
    return {
        "score": result.score,
        "method": result.method,
        "input_scores": result.input_scores,
        "provenance_trail": result.provenance_trail,
    }


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
# Group 3b: Security — context allowlist & resource limits
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def check_context_allowed(url: str, config_json: str) -> dict:
    """Check if a context URL is permitted by an allowlist.

    Evaluates a context URL against a security configuration with
    allowed URLs, glob patterns, and a block-all-remote flag.

    Args:
        url: The context URL to check.
        config_json: JSON config with 'allowed' (list of exact URLs),
            'patterns' (list of glob patterns), and optionally
            'block_remote_contexts' (bool).

    Returns:
        Dict with 'allowed' (bool) and 'url'.
    """
    config = _parse_json(config_json, "config_json")
    allowed = _is_context_allowed_raw(url, config)
    return {"allowed": allowed, "url": url}


@mcp.tool()
def enforce_resource_limits(
    document_json: str,
    limits_json: Optional[str] = None,
) -> dict:
    """Validate a JSON-LD document against resource limits.

    Checks document size and nesting depth against configurable
    limits to prevent resource exhaustion attacks.

    Default limits:
        - max_context_depth: 10
        - max_graph_depth: 100
        - max_document_size: 10 MB
        - max_expansion_time: 30 seconds

    Args:
        document_json: JSON string of the document to validate.
        limits_json: Optional JSON string of limit overrides.

    Returns:
        Dict with 'valid' (bool) and 'error' (str or null).
    """
    limits = _parse_json(limits_json, "limits_json") if limits_json else None
    try:
        _enforce_resource_limits_raw(document_json, limits)
        return {"valid": True, "error": None}
    except (ValueError, TypeError) as e:
        return {"valid": False, "error": str(e)}


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
    return _cosine_similarity_raw(a, b)


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
    if strategy not in _MERGE_STRATEGIES:
        raise ValueError(f"Unknown merge strategy: {strategy!r}. Choose from: {sorted(_MERGE_STRATEGIES)}")
    a = _parse_json(graph_a_json, "graph_a_json")
    b = _parse_json(graph_b_json, "graph_b_json")
    merged, _report = _merge_graphs_raw([a, b], conflict_strategy=strategy)
    return json.dumps(merged)


@mcp.tool()
def diff_graphs(graph_a_json: str, graph_b_json: str) -> str:
    """Compute a semantic diff between two JSON-LD graphs.

    Compares nodes by @id and properties by value (ignoring
    annotation metadata like @confidence). Returns added, removed,
    modified, and unchanged elements.

    Args:
        graph_a_json: JSON string of the first (baseline) graph.
        graph_b_json: JSON string of the second (updated) graph.

    Returns:
        JSON string with 'added', 'removed', 'modified', 'unchanged'.
    """
    a = _parse_json(graph_a_json, "graph_a_json")
    b = _parse_json(graph_b_json, "graph_b_json")
    result = _diff_graphs_raw(a, b)
    return json.dumps(result)


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
# Group 5b: Temporal — add_temporal_annotation, temporal_diff
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def add_temporal_annotation(
    value: Any,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    as_of: Optional[str] = None,
) -> dict:
    """Add temporal qualifiers to a value.

    Creates or enriches an annotated-value wrapper with
    @validFrom, @validUntil, and/or @asOf timestamps.
    Composes naturally with existing provenance annotations.

    Args:
        value: The value to annotate.
        valid_from: ISO 8601 — when the assertion becomes true.
        valid_until: ISO 8601 — when the assertion expires.
        as_of: ISO 8601 — observation timestamp.

    Returns:
        Annotated value dict with temporal qualifiers.
    """
    return _add_temporal_raw(
        value, valid_from=valid_from, valid_until=valid_until, as_of=as_of
    )


@mcp.tool()
def temporal_diff(graph_json: str, t1: str, t2: str) -> dict:
    """Compute what changed between two points in time.

    Snapshots the graph at t1 and t2, then compares property
    values for each @id node.

    Args:
        graph_json: JSON array of JSON-LD nodes with temporal annotations.
        t1: Earlier ISO 8601 timestamp.
        t2: Later ISO 8601 timestamp.

    Returns:
        Dict with 'added', 'removed', 'modified', 'unchanged'.
    """
    graph = _parse_json(graph_json, "graph_json")
    result = _temporal_diff_raw(graph, t1, t2)
    return {
        "added": result.added,
        "removed": result.removed,
        "modified": result.modified,
        "unchanged": result.unchanged,
    }


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
# Group 6b: Interop — from_prov_o, shacl_to_shape, OWL, RDF-Star, verbosity
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def from_prov_o(prov_o_json: str) -> str:
    """Convert W3C PROV-O back to jsonld-ex annotations.

    Round-trip counterpart to to_prov_o. Recovers @confidence,
    @source, @extractedAt, etc. from PROV-O entities/activities.

    Args:
        prov_o_json: JSON string of a PROV-O provenance graph.

    Returns:
        JSON string of the recovered jsonld-ex annotated document.
    """
    prov_doc = _parse_json(prov_o_json, "prov_o_json")
    recovered, _report = _from_prov_o_raw(prov_doc)
    return json.dumps(recovered)


@mcp.tool()
def shacl_to_shape(shacl_json: str) -> str:
    """Convert W3C SHACL constraints back to jsonld-ex @shape.

    Round-trip counterpart to shape_to_shacl. Recovers @required,
    @type, @minimum, @maximum, etc. from SHACL property shapes.

    Args:
        shacl_json: JSON string of a SHACL shape graph.

    Returns:
        JSON string of the recovered jsonld-ex shape definition.
    """
    shacl_doc = _parse_json(shacl_json, "shacl_json")
    shape, _warnings = _shacl_to_shape_raw(shacl_doc)
    return json.dumps(shape)


@mcp.tool()
def shape_to_owl(shape_json: str, class_iri: Optional[str] = None) -> str:
    """Convert a jsonld-ex @shape to OWL class restrictions.

    Maps @required to owl:someValuesFrom, @type to owl:onDatatype,
    @minimum/@maximum to xsd:minInclusive/xsd:maxInclusive.

    Args:
        shape_json: JSON string of a jsonld-ex shape definition.
        class_iri: Optional IRI for the OWL class.

    Returns:
        JSON string of OWL restriction axioms.
    """
    shape = _parse_json(shape_json, "shape_json")
    owl_doc = _shape_to_owl_raw(shape, class_iri=class_iri)
    return json.dumps(owl_doc)


@mcp.tool()
def to_rdf_star(node_json: str, base_subject: Optional[str] = None) -> str:
    """Export jsonld-ex annotations as RDF-Star N-Triples.

    Produces RDF-Star syntax where annotation metadata (@confidence,
    @source, etc.) is expressed as annotations on the base triple
    using << >> syntax.

    Args:
        node_json: JSON string of an annotated jsonld-ex document.
        base_subject: Optional base subject IRI. Auto-generated if omitted.

    Returns:
        RDF-Star N-Triples string.
    """
    doc = _parse_json(node_json, "node_json")
    ntriples, _report = _to_rdf_star_raw(doc, base_subject=base_subject)
    return ntriples


@mcp.tool()
def compare_prov_o_verbosity(document_json: str) -> dict:
    """Compare jsonld-ex vs PROV-O verbosity for the same annotations.

    Converts a document to PROV-O and counts the resulting triples,
    then compares with the jsonld-ex triple count to measure
    payload reduction.

    Args:
        document_json: JSON string of a jsonld-ex annotated document.

    Returns:
        Dict with jsonld_ex_triples, prov_o_triples, reduction_percent.
    """
    doc = _parse_json(document_json, "document_json")
    comparison = _compare_prov_o_raw(doc)
    return {
        "jsonld_ex_triples": comparison.jsonld_ex_triples,
        "prov_o_triples": comparison.alternative_triples,
        "reduction_percent": comparison.triple_reduction_pct,
    }


@mcp.tool()
def compare_shacl_verbosity(shape_json: str) -> dict:
    """Compare jsonld-ex @shape vs SHACL verbosity.

    Converts a shape to SHACL and counts the resulting triples,
    then compares with the jsonld-ex triple count to measure
    payload reduction.

    Args:
        shape_json: JSON string of a jsonld-ex shape definition.

    Returns:
        Dict with jsonld_ex_triples, shacl_triples, reduction_percent.
    """
    shape = _parse_json(shape_json, "shape_json")
    comparison = _compare_shacl_raw(shape)
    return {
        "jsonld_ex_triples": comparison.jsonld_ex_triples,
        "shacl_triples": comparison.alternative_triples,
        "reduction_percent": comparison.triple_reduction_pct,
    }


# ═══════════════════════════════════════════════════════════════════
# Group 7: MQTT / IoT Transport
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def mqtt_encode(
    document_json: str,
    compress: bool = True,
    max_payload: int = 256_000,
) -> dict:
    """Serialize a JSON-LD document for MQTT transmission.

    Produces a base64-encoded payload (CBOR or JSON) suitable for
    MQTT PUBLISH, along with size statistics and MQTT 5.0 PUBLISH
    properties (payload_format_indicator, content_type,
    message_expiry_interval, user_properties).

    Per MQTT spec: default max_payload is 256 KB (MQTT v3.1.1).
    Set to 268_435_455 for MQTT v5.0 maximum.

    Args:
        document_json: JSON string of the document to serialize.
        compress: Use CBOR compression (True) or JSON (False).
        max_payload: Maximum payload size in bytes.

    Returns:
        Dict with payload_base64, payload_bytes, format, and
        mqtt5_properties.
    """
    doc = _parse_json(document_json, "document_json")
    payload = _to_mqtt_payload_raw(doc, compress=compress, max_payload=max_payload)
    mqtt5_props = _derive_mqtt5_properties_raw(doc, compress=compress)
    return {
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "payload_bytes": len(payload),
        "format": "application/cbor" if compress else "application/ld+json",
        "mqtt5_properties": mqtt5_props,
    }


@mcp.tool()
def mqtt_decode(
    payload_base64: str,
    compressed: bool = True,
    context: Optional[str] = None,
) -> str:
    """Deserialize an MQTT payload back to a JSON-LD document.

    Accepts base64-encoded bytes (as produced by mqtt_encode) and
    restores the original JSON-LD document.  Optionally reattaches
    an @context that was stripped during serialization.

    Args:
        payload_base64: Base64-encoded MQTT payload.
        compressed: Whether the payload is CBOR (True) or JSON (False).
        context: Optional @context IRI to reattach.

    Returns:
        JSON string of the restored document.
    """
    payload = base64.b64decode(payload_base64)
    ctx: Any = context  # Optional[str] -> Optional[Any]
    doc = _from_mqtt_payload_raw(payload, context=ctx, compressed=compressed)
    return json.dumps(doc)


@mcp.tool()
def mqtt_derive_topic(
    document_json: str,
    prefix: str = "ld",
) -> dict:
    """Derive an MQTT topic name from JSON-LD document metadata.

    Generates a hierarchical MQTT topic following the pattern
    ``{prefix}/{@type}/{@id_fragment}``.

    Per MQTT spec (v3.1.1 §4.7, v5.0 §4.7):
    - Topic names are UTF-8, max 65,535 bytes.
    - Wildcards (# +) and null are forbidden in PUBLISH topics.
    - Leading $ is reserved for broker system topics.

    Args:
        document_json: JSON string of the document.
        prefix: Topic prefix (default "ld").

    Returns:
        Dict with topic, topic_bytes, type_segment, id_segment.
    """
    doc = _parse_json(document_json, "document_json")
    topic = _derive_mqtt_topic_raw(doc, prefix=prefix)
    parts = topic.split("/", 2)
    return {
        "topic": topic,
        "topic_bytes": len(topic.encode("utf-8")),
        "prefix": parts[0] if len(parts) > 0 else prefix,
        "type_segment": parts[1] if len(parts) > 1 else "unknown",
        "id_segment": parts[2] if len(parts) > 2 else "unknown",
    }


@mcp.tool()
def mqtt_derive_qos(document_json: str) -> dict:
    """Map JSON-LD confidence metadata to MQTT QoS level.

    Heuristic mapping per MQTT spec QoS semantics:
    - QoS 0 (at most once):  @confidence < 0.5  — noisy/low-priority
    - QoS 1 (at least once): 0.5 ≤ @confidence < 0.9  — normal
    - QoS 2 (exactly once):  @confidence ≥ 0.9 or @humanVerified
      — critical/verified

    Falls back to property-level confidence if no document-level
    annotation is found.  Default: QoS 1.

    Args:
        document_json: JSON string of the document.

    Returns:
        Dict with qos (0, 1, or 2), reasoning, and confidence_used.
    """
    doc = _parse_json(document_json, "document_json")
    return _derive_mqtt_qos_detailed_raw(doc)


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


# ═════════════════════════════════════════════════════════════════
# Group 8: Compliance Algebra Tools
# ═════════════════════════════════════════════════════════════════


def _compliance_to_dict(co: ComplianceOpinion) -> dict:
    """Serialize a ComplianceOpinion to a dict with both alias sets."""
    return {
        "lawfulness": co.lawfulness,
        "violation": co.violation,
        "uncertainty": co.uncertainty,
        "base_rate": co.base_rate,
        "belief": co.belief,
        "disbelief": co.disbelief,
        "projected_probability": co.projected_probability(),
    }


@mcp.tool()
def create_compliance_opinion(
    lawfulness: float,
    violation: float,
    uncertainty: float,
    base_rate: float = 0.5,
) -> dict:
    """Create a compliance opinion ω = (l, v, u, a) per the Compliance Algebra.

    Models regulatory compliance as an uncertain epistemic state using
    Jøsang's Subjective Logic. Distinguishes between:
      - Evidence of compliance (lawfulness / belief)
      - Evidence of violation (violation / disbelief)
      - Absence of evidence (uncertainty)

    This is strictly more informative than binary COMPLIANT/NON-COMPLIANT:
    it distinguishes \"we know it's compliant\" from \"we don't know\" from
    \"we have conflicting evidence.\"

    Constraint: lawfulness + violation + uncertainty = 1.0.
    Projected probability: P(ω) = l + a·u.

    Args:
        lawfulness: Evidence of compliance, in [0, 1].
        violation: Evidence of violation, in [0, 1].
        uncertainty: Absence of evidence, in [0, 1].
        base_rate: Prior compliance probability, in [0, 1]. Default 0.5.

    Returns:
        Dict with lawfulness, violation, uncertainty, base_rate,
        belief, disbelief, and projected_probability.
    """
    co = ComplianceOpinion.create(
        lawfulness=lawfulness,
        violation=violation,
        uncertainty=uncertainty,
        base_rate=base_rate,
    )
    return _compliance_to_dict(co)


@mcp.tool()
def jurisdictional_meet(opinions_json: str) -> dict:
    """Compute the jurisdictional meet — conjunction of compliance requirements.

    Models composite compliance across multiple regulatory jurisdictions
    (e.g., GDPR + CCPA + UK-DPA). An organization must satisfy ALL
    requirements simultaneously.

    Lawfulness is conjunctive: l⊓ = l₁·l₂.
    Violation is disjunctive: v⊓ = v₁ + v₂ − v₁·v₂.

    Algebraic properties (Theorem 1):
      - Commutative and associative (order-independent)
      - Identity: (1,0,0,1) — a jurisdiction with certain compliance
      - Annihilator: (0,1,0,0) — a jurisdiction with certain violation
      - Monotonic: composite l ≤ min(lᵢ), composite v ≥ max(vᵢ)

    Args:
        opinions_json: JSON array of opinion objects, each with
            belief, disbelief, uncertainty, and optional base_rate.

    Returns:
        Compliance opinion dict for the composite assessment.
    """
    data = _parse_json(opinions_json, "opinions_json")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("opinions_json must be a non-empty JSON array")
    opinions = [_opinion_from_dict(d) for d in data]
    result = _jurisdictional_meet_raw(*opinions)
    return _compliance_to_dict(result)


@mcp.tool()
def compliance_propagation(
    source_json: str,
    derivation_trust_json: str,
    purpose_compat_json: str,
) -> dict:
    """Propagate compliance through a data derivation step.

    Models how compliance degrades when data is derived, transformed,
    or transferred. The derived dataset's compliance depends on:
      - Source compliance (ω_S): is the source data compliant?
      - Derivation trust (τ): is the derivation process lawful? (Art. 5/25)
      - Purpose compatibility (π): is the new purpose compatible? (Art. 6(4))

    Result: l_D = t·p·l_S (multiplicative degradation).

    Key insight: even if each step is 95% compliant, a 10-step pipeline
    drops to ~60% — invisible to binary pass/fail checking.

    Args:
        source_json: JSON opinion for source compliance.
        derivation_trust_json: JSON opinion for derivation process trust.
        purpose_compat_json: JSON opinion for purpose compatibility.

    Returns:
        Compliance opinion dict for the derived dataset.
    """
    source = _opinion_from_dict(_parse_json(source_json, "source_json"))
    trust = _opinion_from_dict(_parse_json(derivation_trust_json, "derivation_trust_json"))
    purpose = _opinion_from_dict(_parse_json(purpose_compat_json, "purpose_compat_json"))
    result = _compliance_propagation_raw(source, trust, purpose)
    return _compliance_to_dict(result)


@mcp.tool()
def consent_validity(conditions_json: str) -> dict:
    """Assess GDPR Art. 7 consent validity via six-condition composition.

    Valid consent requires ALL six conditions to be satisfied
    (GDPR Art. 4(11) and Art. 7):
      1. Freely given (Art. 7(4))
      2. Specific to purpose (Art. 4(11))
      3. Informed (Art. 4(11))
      4. Unambiguous indication (Art. 4(11))
      5. Demonstrable by controller (Art. 7(1))
      6. Distinguishable from other matters (Art. 7(2))

    Uses jurisdictional meet (conjunction): a single weak condition
    dominates the composite result.

    Args:
        conditions_json: JSON array of exactly 6 opinion objects,
            one per consent condition, each with belief, disbelief,
            uncertainty, and optional base_rate.

    Returns:
        Compliance opinion dict for composite consent validity.
    """
    data = _parse_json(conditions_json, "conditions_json")
    if not isinstance(data, list) or len(data) != 6:
        raise ValueError(
            f"consent_validity requires exactly 6 opinions, got {len(data) if isinstance(data, list) else 'non-list'}"
        )
    opinions = [_opinion_from_dict(d) for d in data]
    result = _consent_validity_raw(*opinions)
    return _compliance_to_dict(result)


@mcp.tool()
def withdrawal_override(
    consent_json: str,
    withdrawal_json: str,
    assessment_time: float,
    withdrawal_time: float,
) -> dict:
    """Apply consent withdrawal override (GDPR Art. 7(3)).

    Models the proposition replacement at consent withdrawal:
    before withdrawal, compliance depends on consent validity;
    after withdrawal, it depends on whether the controller has
    ceased processing.

    This is NOT fusion (combining evidence about one proposition).
    It is replacement (switching to a different proposition entirely).
    No existing SL operator models this — it is a novel contribution.

    Sharp boundary at withdrawal_time:
      - assessment_time < withdrawal_time → consent opinion
      - assessment_time ≥ withdrawal_time → withdrawal opinion

    Args:
        consent_json: JSON opinion for pre-withdrawal consent validity.
        withdrawal_json: JSON opinion for post-withdrawal processing cessation.
        assessment_time: Time of compliance assessment.
        withdrawal_time: Time of consent withdrawal.

    Returns:
        Compliance opinion dict (consent or withdrawal, depending on time).
    """
    consent = _opinion_from_dict(_parse_json(consent_json, "consent_json"))
    withdrawal = _opinion_from_dict(_parse_json(withdrawal_json, "withdrawal_json"))
    result = _withdrawal_override_raw(consent, withdrawal, assessment_time, withdrawal_time)
    return _compliance_to_dict(result)


@mcp.tool()
def expiry_trigger(
    opinion_json: str,
    assessment_time: float,
    trigger_time: float,
    residual_factor: float = 0.0,
) -> dict:
    """Apply an expiry trigger — asymmetric l→v transition.

    At expiry time, lawfulness transfers to violation (not uncertainty).
    An expired deadline is a known fact, not a source of epistemic
    uncertainty. This asymmetry is unique to the compliance algebra.

    Pre-trigger: opinion unchanged.
    Post-trigger: l' = γ·l, v' = v + (1−γ)·l, u' = u.

    The residual_factor γ controls the transition:
      - γ = 0.0: hard expiry (all lawfulness → violation)
      - γ = 0.5: partial expiry (50% retained)
      - γ = 1.0: no immediate effect

    Models: data retention deadlines (Art. 5(1)(e)), consent expiry,
    adequacy decision sunset clauses.

    Args:
        opinion_json: JSON opinion to evaluate.
        assessment_time: Time of assessment.
        trigger_time: Expiry trigger time.
        residual_factor: Fraction of lawfulness retained, in [0, 1]. Default 0.

    Returns:
        Compliance opinion dict after applying expiry.
    """
    opinion = _opinion_from_dict(_parse_json(opinion_json, "opinion_json"))
    result = _expiry_trigger_raw(opinion, assessment_time, trigger_time, residual_factor)
    return _compliance_to_dict(result)


@mcp.tool()
def review_due_trigger(
    opinion_json: str,
    assessment_time: float,
    trigger_time: float,
    accelerated_half_life: float,
) -> dict:
    """Apply a review-due trigger — accelerated decay toward vacuity.

    A missed mandatory review (e.g., DPIA review under Art. 35(11),
    adequacy review under Art. 45(3)) accelerates uncertainty growth.

    Unlike expiry, review-due moves toward vacuity (increased u),
    not violation: a missed review means we lack current evidence,
    not that we know the situation is non-compliant.

    Pre-trigger: opinion unchanged.
    Post-trigger: applies accelerated exponential decay with
    elapsed = assessment_time − trigger_time.

    Args:
        opinion_json: JSON opinion to evaluate.
        assessment_time: Time of assessment.
        trigger_time: Review-due trigger time.
        accelerated_half_life: Shorter half-life for post-trigger decay.

    Returns:
        Compliance opinion dict after applying accelerated decay.
    """
    opinion = _opinion_from_dict(_parse_json(opinion_json, "opinion_json"))
    result = _review_due_trigger_raw(opinion, assessment_time, trigger_time, accelerated_half_life)
    return _compliance_to_dict(result)


@mcp.tool()
def regulatory_change_trigger(
    opinion_json: str,
    assessment_time: float,
    trigger_time: float,
    new_opinion_json: str,
) -> dict:
    """Apply a regulatory change trigger — proposition replacement.

    At trigger time, the compliance opinion is replaced by a new
    assessment reflecting the changed legal framework. Models
    discrete legal events: adequacy decision revocation, new
    regulation taking effect, court ruling.

    Same proposition-replacement semantics as withdrawal_override.
    Trigger ordering is non-commutative (Theorem 4(e)): the order
    of regulatory events matters.

    Args:
        opinion_json: JSON opinion for current assessment.
        assessment_time: Time of assessment.
        trigger_time: Regulatory change time.
        new_opinion_json: JSON opinion under the new legal framework.

    Returns:
        Compliance opinion dict (original or new, depending on time).
    """
    opinion = _opinion_from_dict(_parse_json(opinion_json, "opinion_json"))
    new_op = _opinion_from_dict(_parse_json(new_opinion_json, "new_opinion_json"))
    result = _regulatory_change_trigger_raw(opinion, assessment_time, trigger_time, new_op)
    return _compliance_to_dict(result)


@mcp.tool()
def erasure_scope_opinion(opinions_json: str) -> dict:
    """Compute composite erasure completeness across data lineage.

    Complete erasure (GDPR Art. 17) requires ALL nodes in the data
    lineage graph to be erased — a conjunction. A single node with
    retained data compromises the entire erasure.

    Composite erasure confidence degrades exponentially: e_R = ∏ e_i.
    For 10 nodes at 99% each: e_R ≈ 90.4%. For 100 nodes: e_R ≈ 36.6%.
    This makes complete erasure assurance practically impossible for
    large data lineage graphs — a result with direct implications
    for Art. 17 compliance.

    Args:
        opinions_json: JSON array of erasure completeness opinions,
            one per node in the erasure scope.

    Returns:
        Compliance opinion dict for composite erasure completeness.
    """
    data = _parse_json(opinions_json, "opinions_json")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("opinions_json must be a non-empty JSON array")
    opinions = [_opinion_from_dict(d) for d in data]
    result = _erasure_scope_opinion_raw(*opinions)
    return _compliance_to_dict(result)


@mcp.tool()
def residual_contamination(opinions_json: str) -> dict:
    """Compute residual contamination risk at a node.

    A node is contaminated if personal data persists in the node
    OR any of its ancestors — a disjunction of per-node persistence.

    Given erasure opinions for the node and its ancestors:
      - violation (r):   contamination risk = 1 − ∏(1−ē_i)
      - lawfulness (r̄): clean probability = ∏ e_i
      - uncertainty:      u_r = ∏(1−ē_i) − ∏ e_i

    Risk is monotonically non-decreasing in ancestor count:
    more ancestors = more potential contamination sources.

    Args:
        opinions_json: JSON array of erasure opinions for the node
            and its ancestors.

    Returns:
        Compliance opinion dict where lawfulness = clean probability,
        violation = contamination risk.
    """
    data = _parse_json(opinions_json, "opinions_json")
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("opinions_json must be a non-empty JSON array")
    opinions = [_opinion_from_dict(d) for d in data]
    result = _residual_contamination_raw(*opinions)
    return _compliance_to_dict(result)


# ═════════════════════════════════════════════════════════════════
# Group 8b: Compliance Algebra — Resources
# ═════════════════════════════════════════════════════════════════


@mcp.resource("jsonld-ex://schema/compliance-opinion")
def get_compliance_opinion_schema() -> str:
    """JSON Schema for a Compliance Algebra opinion object."""
    return json.dumps({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ComplianceOpinion",
        "description": (
            "A compliance opinion \u03c9 = (l, v, u, a) modeling regulatory "
            "compliance as an uncertain epistemic state. Extends Subjective "
            "Logic opinions with compliance-domain semantics. "
            "Constraint: lawfulness + violation + uncertainty = 1."
        ),
        "type": "object",
        "required": ["lawfulness", "violation", "uncertainty"],
        "properties": {
            "lawfulness": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Evidence of compliance (alias: belief)",
            },
            "violation": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Evidence of violation (alias: disbelief)",
            },
            "uncertainty": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Absence of evidence",
            },
            "base_rate": {
                "type": "number", "minimum": 0, "maximum": 1,
                "default": 0.5,
                "description": "Prior compliance probability",
            },
        },
        "additionalProperties": False,
    })


@mcp.resource("jsonld-ex://context/compliance")
def get_compliance_context() -> str:
    """JSON-LD context for compliance algebra extensions."""
    return json.dumps({
        "@context": {
            "jsonld-ex": "http://www.w3.org/ns/jsonld-ex/",
            "@lawfulness": "jsonld-ex:lawfulness",
            "@violation": "jsonld-ex:violation",
            "@jurisdiction": "jsonld-ex:jurisdiction",
            "@legalBasis": "jsonld-ex:legalBasis",
            "@consentStatus": "jsonld-ex:consentStatus",
            "@retentionUntil": "jsonld-ex:retentionUntil",
            "@erasureStatus": "jsonld-ex:erasureStatus",
        }
    })


# ═════════════════════════════════════════════════════════════════
# Group 8c: Compliance Algebra — Prompts
# ═════════════════════════════════════════════════════════════════


@mcp.prompt()
def gdpr_compliance_assessment(jurisdiction_list: str) -> str:
    """Guide for multi-jurisdictional GDPR compliance assessment.

    Generates a step-by-step workflow for assessing compliance
    across multiple regulatory jurisdictions using the compliance
    algebra operators.
    """
    return (
        f"Assess compliance across jurisdictions: {jurisdiction_list}\n\n"
        "Step-by-step workflow:\n"
        "1. For each jurisdiction, use create_compliance_opinion to model\n"
        "   the current compliance status with lawfulness, violation, and\n"
        "   uncertainty components.\n"
        "2. Use jurisdictional_meet to compute the composite compliance\n"
        "   across all jurisdictions. The result satisfies all requirements\n"
        "   simultaneously (conjunction).\n"
        "3. Check the composite projected_probability:\n"
        "   - P > 0.8: strong composite compliance\n"
        "   - 0.5 < P < 0.8: moderate — investigate weaker jurisdictions\n"
        "   - P < 0.5: composite non-compliance likely\n"
        "4. Check uncertainty: high u means absence of evidence, not\n"
        "   evidence of compliance. Gather more evidence before deciding.\n"
        "5. If data flows between jurisdictions, use compliance_propagation\n"
        "   to model how compliance degrades through derivation chains.\n"
        "6. For temporal aspects (consent expiry, retention deadlines),\n"
        "   apply expiry_trigger or review_due_trigger to model how\n"
        "   compliance evolves over time.\n"
    )


@mcp.prompt()
def consent_lifecycle(purpose: str) -> str:
    """Guide for managing the full GDPR consent lifecycle.

    Generates a step-by-step workflow for consent assessment,
    monitoring, withdrawal handling, and erasure propagation
    for a specific processing purpose.
    """
    return (
        f"Manage consent lifecycle for purpose: {purpose}\n\n"
        "Step-by-step workflow:\n"
        "1. ASSESS CONSENT VALIDITY:\n"
        "   Use consent_validity with 6 opinions (one per Art. 7 condition):\n"
        "   freely_given, specific, informed, unambiguous, demonstrable,\n"
        "   distinguishable. Each should reflect current evidence.\n"
        "2. MONITOR CONSENT:\n"
        "   Periodically re-assess. Use review_due_trigger if mandatory\n"
        "   review deadlines apply. Use expiry_trigger if consent has\n"
        "   a fixed duration.\n"
        "3. HANDLE WITHDRAWAL:\n"
        "   When the data subject withdraws consent, use withdrawal_override\n"
        "   with the withdrawal time. Post-withdrawal, compliance depends\n"
        "   on whether processing has ceased, NOT on original consent.\n"
        "4. PROPAGATE TO DERIVED DATA:\n"
        "   Use compliance_propagation to assess how withdrawal affects\n"
        "   downstream datasets derived from the original.\n"
        "5. ERASURE OBLIGATION:\n"
        "   Use erasure_scope_opinion to assess completeness of erasure\n"
        "   across all nodes. Use residual_contamination to check for\n"
        "   lingering personal data in the lineage graph.\n"
        "6. REGULATORY CHANGES:\n"
        "   If the legal framework changes, use regulatory_change_trigger\n"
        "   to replace the compliance assessment with a new evaluation.\n"
    )
