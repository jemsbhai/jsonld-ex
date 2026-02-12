"""Validation Extensions for JSON-LD (@shape)."""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence


@dataclass
class ValidationError:
    path: str
    constraint: str
    message: str
    value: Any = None


@dataclass
class ValidationWarning:
    path: str
    code: str
    message: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)


XSD = "http://www.w3.org/2001/XMLSchema#"


def validate_node(node: dict[str, Any], shape: dict[str, Any]) -> ValidationResult:
    """Validate a JSON-LD node against a shape definition."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    if not isinstance(node, dict):
        errors.append(ValidationError(".", "type", "Node must be a dict"))
        return ValidationResult(False, errors, warnings)

    # Type check
    if "@type" in shape:
        node_types = _get_types(node)
        if shape["@type"] not in node_types:
            errors.append(ValidationError(
                "@type", "type",
                f'Expected type "{shape["@type"]}", found: {node_types}',
                node_types,
            ))

    # Property constraints
    for prop, constraint in shape.items():
        if prop.startswith("@") or not isinstance(constraint, dict):
            continue

        value = node.get(prop)

        # ── Cardinality constraints (operate on raw value before extraction) ──
        count = _count_values(value)

        if "@minCount" in constraint:
            min_count = constraint["@minCount"]
            if count < min_count:
                errors.append(ValidationError(
                    prop, "minCount",
                    f"Expected at least {min_count} value(s), found {count}",
                    value,
                ))

        if "@maxCount" in constraint:
            max_count = constraint["@maxCount"]
            if count > max_count:
                errors.append(ValidationError(
                    prop, "maxCount",
                    f"Expected at most {max_count} value(s), found {count}",
                    value,
                ))

        # ── Extract scalar for remaining constraints ──
        raw = _extract_raw(value)

        if constraint.get("@required") and raw is None:
            errors.append(ValidationError(prop, "required", f'Property "{prop}" is required'))
            continue

        if raw is None:
            continue

        # Type check
        expected_type = constraint.get("@type")
        if expected_type:
            type_err = _validate_type(raw, expected_type)
            if type_err:
                errors.append(ValidationError(prop, "type", type_err, raw))

        # Numeric (exclude booleans — they are int subclass in Python)
        if "@minimum" in constraint and isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if raw < constraint["@minimum"]:
                errors.append(ValidationError(
                    prop, "minimum",
                    f"Value {raw} below minimum {constraint['@minimum']}", raw,
                ))

        if "@maximum" in constraint and isinstance(raw, (int, float)) and not isinstance(raw, bool):
            if raw > constraint["@maximum"]:
                errors.append(ValidationError(
                    prop, "maximum",
                    f"Value {raw} exceeds maximum {constraint['@maximum']}", raw,
                ))

        # String length
        if "@minLength" in constraint and isinstance(raw, str):
            if len(raw) < constraint["@minLength"]:
                errors.append(ValidationError(
                    prop, "minLength",
                    f"Length {len(raw)} below minimum {constraint['@minLength']}", raw,
                ))

        if "@maxLength" in constraint and isinstance(raw, str):
            if len(raw) > constraint["@maxLength"]:
                errors.append(ValidationError(
                    prop, "maxLength",
                    f"Length {len(raw)} exceeds maximum {constraint['@maxLength']}", raw,
                ))

        # Enumeration
        if "@in" in constraint:
            allowed = constraint["@in"]
            if raw not in allowed:
                errors.append(ValidationError(
                    prop, "in",
                    f"Value {raw!r} not in allowed set {allowed}", raw,
                ))

        # Pattern
        if "@pattern" in constraint and isinstance(raw, str):
            try:
                if not re.search(constraint["@pattern"], raw):
                    errors.append(ValidationError(
                        prop, "pattern",
                        f'"{raw}" does not match pattern "{constraint["@pattern"]}"', raw,
                    ))
            except re.error as exc:
                errors.append(ValidationError(
                    prop, "pattern",
                    f'Invalid regex pattern "{constraint["@pattern"]}": {exc}', raw,
                ))

    return ValidationResult(len(errors) == 0, errors, warnings)


def validate_document(
    doc: dict[str, Any], shapes: Sequence[dict[str, Any]]
) -> ValidationResult:
    """Validate all matching nodes in a document against shapes."""
    all_errors: list[ValidationError] = []
    all_warnings: list[ValidationWarning] = []

    for node in _extract_nodes(doc):
        node_types = _get_types(node)
        for shape in shapes:
            if shape.get("@type") in node_types:
                result = validate_node(node, shape)
                for e in result.errors:
                    e.path = f"{node.get('@id', 'anonymous')}/{e.path}"
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)

    return ValidationResult(len(all_errors) == 0, all_errors, all_warnings)


# ── Internal ───────────────────────────────────────────────────────

def _count_values(value: Any) -> int:
    """Count the number of values for cardinality checks.

    - ``None`` (absent property) → 0
    - A list → ``len(list)``
    - Any other single value → 1
    """
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def _get_types(node: dict) -> list[str]:
    t = node.get("@type")
    if t is None:
        return []
    return t if isinstance(t, list) else [t]


def _extract_raw(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict) and "@value" in value:
        return value["@value"]
    if isinstance(value, dict) and not any(k.startswith("@") for k in value):
        return None  # Plain dict without JSON-LD keywords — treat as absent
    if isinstance(value, list) and len(value) > 0:
        return _extract_raw(value[0])
    if isinstance(value, list) and len(value) == 0:
        return None
    return value


def _extract_nodes(doc: Any) -> list[dict]:
    if isinstance(doc, list):
        nodes = []
        for item in doc:
            nodes.extend(_extract_nodes(item))
        return nodes
    if not isinstance(doc, dict):
        return []
    nodes = []
    if "@type" in doc:
        nodes.append(doc)
    if "@graph" in doc:
        nodes.extend(_extract_nodes(doc["@graph"]))
    return nodes


def _validate_type(value: Any, expected: str) -> Optional[str]:
    xsd_type = expected.replace("xsd:", XSD) if expected.startswith("xsd:") else expected
    checks = {
        f"{XSD}string": lambda v: isinstance(v, str),
        f"{XSD}integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        f"{XSD}double": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        f"{XSD}float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        f"{XSD}decimal": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        f"{XSD}boolean": lambda v: isinstance(v, bool),
    }
    checker = checks.get(xsd_type)
    if checker and not checker(value):
        short = expected if expected.startswith("xsd:") else xsd_type
        return f"Expected {short}, got {type(value).__name__}: {value}"
    return None
