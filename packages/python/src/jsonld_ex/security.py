"""Security Extensions for JSON-LD."""

from __future__ import annotations
import hashlib
import base64
import json
import re
from typing import Any, Optional


DEFAULT_RESOURCE_LIMITS = {
    "max_context_depth": 10,
    "max_graph_depth": 100,
    "max_document_size": 10 * 1024 * 1024,  # 10 MB
    "max_expansion_time": 30,  # seconds
}

SUPPORTED_ALGORITHMS = ("sha256", "sha384", "sha512")


def compute_integrity(
    context: str | dict | Any, algorithm: str = "sha256"
) -> str:
    """Compute an integrity hash for a context."""
    if context is None:
        raise TypeError("Context must not be None")
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    try:
        content = context if isinstance(context, str) else json.dumps(context, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Context is not JSON-serializable: {exc}") from exc
    h = hashlib.new(algorithm, content.encode("utf-8")).digest()
    b64 = base64.b64encode(h).decode("ascii")
    return f"{algorithm}-{b64}"


def verify_integrity(context: str | dict | Any, declared: str) -> bool:
    """Verify context content against its declared integrity hash."""
    parts = declared.split("-", 1)
    if len(parts) != 2 or parts[0] not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Invalid integrity string: {declared}")
    computed = compute_integrity(context, parts[0])
    return computed == declared


def integrity_context(
    url: str, content: str | dict | Any, algorithm: str = "sha256"
) -> dict[str, str]:
    """Create a context reference with integrity verification."""
    return {"@id": url, "@integrity": compute_integrity(content, algorithm)}


def is_context_allowed(url: str, config: dict[str, Any]) -> bool:
    """Check if a context URL is permitted by an allowlist configuration."""
    if config.get("block_remote_contexts", False):
        return False
    allowed = config.get("allowed", [])
    if url in allowed:
        return True
    for pattern in config.get("patterns", []):
        if isinstance(pattern, str):
            regex = "^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
            if re.match(regex, url):
                return True
    if allowed or config.get("patterns"):
        return False
    return True


_MAX_RECURSION_DEPTH = 500  # Safety cap for _measure_depth


def enforce_resource_limits(
    document: str | dict | Any,
    limits: Optional[dict[str, int]] = None,
) -> None:
    """Validate document against resource limits before processing."""
    if document is None:
        raise TypeError("Document must not be None")
    resolved = {**DEFAULT_RESOURCE_LIMITS, **(limits or {})}
    if isinstance(document, str):
        content = document
        if len(content) > resolved["max_document_size"]:
            raise ValueError(
                f"Document size {len(content)} exceeds limit {resolved['max_document_size']}"
            )
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Document is not valid JSON: {exc}") from exc
    elif isinstance(document, (dict, list)):
        try:
            content = json.dumps(document)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"Document is not JSON-serializable: {exc}") from exc
        if len(content) > resolved["max_document_size"]:
            raise ValueError(
                f"Document size {len(content)} exceeds limit {resolved['max_document_size']}"
            )
        parsed = document
    else:
        raise TypeError(f"Document must be a str, dict, or list, got: {type(document).__name__}")
    depth = _measure_depth(parsed)
    if depth > resolved["max_graph_depth"]:
        raise ValueError(
            f"Document depth {depth} exceeds limit {resolved['max_graph_depth']}"
        )


def _measure_depth(obj: Any, current: int = 0) -> int:
    if current > _MAX_RECURSION_DEPTH:
        return current  # Safety cap to prevent stack overflow
    if obj is None or not isinstance(obj, (dict, list)):
        return current
    max_depth = current
    items = obj if isinstance(obj, list) else obj.values()
    for item in items:
        max_depth = max(max_depth, _measure_depth(item, current + 1))
    return max_depth
