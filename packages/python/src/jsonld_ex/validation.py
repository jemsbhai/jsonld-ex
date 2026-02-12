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


def validate_node(
    node: dict[str, Any],
    shape: dict[str, Any],
    *,
    shape_registry: dict[str, dict[str, Any]] | None = None,
) -> ValidationResult:
    """Validate a JSON-LD node against a shape definition."""
    errors: list[ValidationError] = []
    warnings: list[ValidationWarning] = []

    if not isinstance(node, dict):
        errors.append(ValidationError(".", "type", "Node must be a dict"))
        return ValidationResult(False, errors, warnings)

    # -- Resolve @extends (GAP-OWL1) ------------------------------------------
    if "@extends" in shape:
        shape, extend_warnings = _resolve_extends(
            shape, shape_registry or {},
        )
        warnings.extend(extend_warnings)

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
        severity = constraint.get("@severity", "error")

        # -- Cardinality constraints (operate on raw value before extraction) --
        count = _count_values(value)

        if "@minCount" in constraint:
            min_count = constraint["@minCount"]
            if count < min_count:
                _emit(
                    errors, warnings, severity, prop, "minCount",
                    f"Expected at least {min_count} value(s), found {count}",
                    value,
                )

        if "@maxCount" in constraint:
            max_count = constraint["@maxCount"]
            if count > max_count:
                _emit(
                    errors, warnings, severity, prop, "maxCount",
                    f"Expected at most {max_count} value(s), found {count}",
                    value,
                )

        # -- Extract scalar for remaining constraints --
        raw = _extract_raw(value)

        if constraint.get("@required") and raw is None:
            _emit(
                errors, warnings, severity, prop, "required",
                f'Property "{prop}" is required',
            )
            continue

        if raw is None and value is None:
            continue

        # -- Nested shape validation (GAP-V5) --
        if "@shape" in constraint:
            inner_shape = constraint["@shape"]
            # Resolve the target node: raw dict, or original value if list
            target = value
            if isinstance(target, list) and len(target) > 0:
                target = target[0]
            if not isinstance(target, dict):
                _emit(
                    errors, warnings, severity, prop, "shape",
                    f"Expected a node (dict) for @shape validation, "
                    f"got {type(target).__name__}",
                    target,
                )
            else:
                inner_result = validate_node(target, inner_shape)
                if not inner_result.valid:
                    for e in inner_result.errors:
                        e.path = f"{prop}/{e.path}"
                    _emit(
                        errors, warnings, severity, prop, "shape",
                        f"Nested shape validation failed: "
                        f"{inner_result.errors[0].message}",
                        target,
                    )
            # @shape is the primary constraint; skip scalar checks
            continue

        if raw is None:
            continue

        # -- Evaluate constraints (including logical and cross-property) --
        prop_errors = _check_constraints(prop, raw, constraint, node)
        for e in prop_errors:
            _emit(errors, warnings, severity, e.path, e.constraint, e.message, e.value)

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


# -- Shape inheritance (@extends, GAP-OWL1) -----------------------------------


def _resolve_extends(
    shape: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    _seen: frozenset[int] | None = None,
) -> tuple[dict[str, Any], list[ValidationWarning]]:
    """Resolve ``@extends`` by deep-merging parent constraints into *shape*.

    Returns a new shape dict (original is not mutated) and any warnings
    produced during resolution (e.g. missing named references).

    Merge semantics (per-property, recursive):
      - Parent provides base constraint keys.
      - Child keys override parent keys at the same property.
      - Properties only in parent are inherited as-is.
    """
    warnings: list[ValidationWarning] = []
    _seen = _seen or frozenset()

    # Guard against circular inheritance
    shape_id = id(shape)
    if shape_id in _seen:
        return shape, warnings
    _seen = _seen | {shape_id}

    extends = shape.get("@extends")
    if extends is None:
        return shape, warnings

    # Normalise to list
    parents_raw = extends if isinstance(extends, list) else [extends]

    # Resolve each parent
    parents: list[dict[str, Any]] = []
    for ref in parents_raw:
        if isinstance(ref, dict):
            # Inline parent — recursively resolve its own @extends
            resolved, pw = _resolve_extends(ref, registry, _seen)
            warnings.extend(pw)
            parents.append(resolved)
        elif isinstance(ref, str):
            if ref in registry:
                resolved, pw = _resolve_extends(registry[ref], registry, _seen)
                warnings.extend(pw)
                parents.append(resolved)
            else:
                warnings.append(ValidationWarning(
                    "@extends", "unresolved",
                    f"Shape reference '{ref}' not found in registry",
                ))
        # Other types silently ignored

    # Build merged shape: stack parents left-to-right, then child on top
    merged: dict[str, Any] = {}
    for parent in parents:
        _merge_shape_into(merged, parent)
    _merge_shape_into(merged, shape)

    # Remove @extends from merged result (already resolved)
    merged.pop("@extends", None)
    return merged, warnings


def _merge_shape_into(
    target: dict[str, Any],
    source: dict[str, Any],
) -> None:
    """Merge *source* shape into *target* in-place.

    For property constraint dicts, keys from *source* are applied first,
    then *target*'s existing keys override (child wins).
    For non-dict values (like ``@type``), *source* overwrites *target*.
    """
    for key, val in source.items():
        if key == "@extends":
            continue  # Already resolved
        if (
            key in target
            and isinstance(target[key], dict)
            and isinstance(val, dict)
        ):
            # Deep merge: source (later writer) overrides target for same keys
            merged_prop = {**target[key], **val}
            target[key] = merged_prop
        else:
            # Source (later writer) wins for top-level keys (e.g. @type)
            target[key] = val


# -- Severity routing ---------------------------------------------------------


def _emit(
    errors: list[ValidationError],
    warnings: list[ValidationWarning],
    severity: str,
    path: str,
    constraint: str,
    message: str,
    value: Any = None,
) -> None:
    """Route a finding to errors or warnings based on severity."""
    if severity in ("warning", "info"):
        warnings.append(ValidationWarning(path, constraint, message))
    else:
        errors.append(ValidationError(path, constraint, message, value))


# -- Constraint evaluation ---------------------------------------------------


def _check_constraints(
    prop: str,
    raw: Any,
    constraint: dict[str, Any],
    node: dict[str, Any] | None = None,
) -> list[ValidationError]:
    """Evaluate atomic, logical, and cross-property constraints against *raw*.

    This function is the single point of evaluation for all value-level
    constraints.  Logical combinators (``@or``, ``@and``, ``@not``) call
    it recursively on their sub-constraint dicts.  Cross-property
    constraints (``@lessThan``, ``@lessThanOrEquals``, ``@equals``,
    ``@disjoint``) use *node* to look up sibling property values.
    """
    errors: list[ValidationError] = []

    # -- Logical combinators --------------------------------------------------
    if "@or" in constraint:
        branches = constraint["@or"]
        for branch in branches:
            if not _check_constraints(prop, raw, branch, node):
                break  # at least one branch passed (no errors)
        else:
            # None of the branches passed
            errors.append(ValidationError(
                prop, "or",
                f"Value {raw!r} did not satisfy any @or branch",
                raw,
            ))

    if "@and" in constraint:
        branches = constraint["@and"]
        for branch in branches:
            branch_errors = _check_constraints(prop, raw, branch, node)
            if branch_errors:
                errors.append(ValidationError(
                    prop, "and",
                    f"Value {raw!r} failed an @and branch: "
                    f"{branch_errors[0].message}",
                    raw,
                ))
                break  # fail fast

    if "@not" in constraint:
        inner = constraint["@not"]
        inner_errors = _check_constraints(prop, raw, inner, node)
        if not inner_errors:
            # Inner constraints passed -- @not inverts this to failure
            errors.append(ValidationError(
                prop, "not",
                f"Value {raw!r} must NOT satisfy {inner}",
                raw,
            ))

    # -- Conditional: @if / @then / @else (GAP-V7) ----------------------------
    if "@if" in constraint:
        if_constraints = constraint["@if"]
        if_errors = _check_constraints(prop, raw, if_constraints, node)
        if not if_errors:
            # Condition met → evaluate @then
            if "@then" in constraint:
                then_errors = _check_constraints(
                    prop, raw, constraint["@then"], node,
                )
                if then_errors:
                    errors.append(ValidationError(
                        prop, "conditional",
                        f"Value {raw!r} met @if condition but failed "
                        f"@then: {then_errors[0].message}",
                        raw,
                    ))
        else:
            # Condition NOT met → evaluate @else if present (vacuous truth
            # when @else absent)
            if "@else" in constraint:
                else_errors = _check_constraints(
                    prop, raw, constraint["@else"], node,
                )
                if else_errors:
                    errors.append(ValidationError(
                        prop, "conditional",
                        f"Value {raw!r} failed @else branch: "
                        f"{else_errors[0].message}",
                        raw,
                    ))

    # -- Cross-property constraints -------------------------------------------
    if node is not None:
        if "@lessThan" in constraint:
            other_prop = constraint["@lessThan"]
            other_raw = _extract_raw(node.get(other_prop))
            if other_raw is not None:
                try:
                    if not (raw < other_raw):
                        errors.append(ValidationError(
                            prop, "lessThan",
                            f"Value {raw!r} is not less than "
                            f"{other_prop}={other_raw!r}",
                            raw,
                        ))
                except TypeError:
                    errors.append(ValidationError(
                        prop, "lessThan",
                        f"Cannot compare {type(raw).__name__} with "
                        f"{type(other_raw).__name__}",
                        raw,
                    ))

        if "@lessThanOrEquals" in constraint:
            other_prop = constraint["@lessThanOrEquals"]
            other_raw = _extract_raw(node.get(other_prop))
            if other_raw is not None:
                try:
                    if not (raw <= other_raw):
                        errors.append(ValidationError(
                            prop, "lessThanOrEquals",
                            f"Value {raw!r} is not <= "
                            f"{other_prop}={other_raw!r}",
                            raw,
                        ))
                except TypeError:
                    errors.append(ValidationError(
                        prop, "lessThanOrEquals",
                        f"Cannot compare {type(raw).__name__} with "
                        f"{type(other_raw).__name__}",
                        raw,
                    ))

        if "@equals" in constraint:
            other_prop = constraint["@equals"]
            other_raw = _extract_raw(node.get(other_prop))
            if other_raw is not None:
                if raw != other_raw:
                    errors.append(ValidationError(
                        prop, "equals",
                        f"Value {raw!r} != {other_prop}={other_raw!r}",
                        raw,
                    ))

        if "@disjoint" in constraint:
            other_prop = constraint["@disjoint"]
            other_raw = _extract_raw(node.get(other_prop))
            if other_raw is not None:
                if raw == other_raw:
                    errors.append(ValidationError(
                        prop, "disjoint",
                        f"Value {raw!r} must differ from "
                        f"{other_prop}={other_raw!r}",
                        raw,
                    ))

    # -- Atomic constraints ---------------------------------------------------

    # Type check
    expected_type = constraint.get("@type")
    if expected_type:
        type_err = _validate_type(raw, expected_type)
        if type_err:
            errors.append(ValidationError(prop, "type", type_err, raw))

    # Numeric (exclude booleans -- they are int subclass in Python)
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

    return errors


# -- Internal -----------------------------------------------------------------

def _count_values(value: Any) -> int:
    """Count the number of values for cardinality checks.

    - ``None`` (absent property) -> 0
    - A list -> ``len(list)``
    - Any other single value -> 1
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
        return None  # Plain dict without JSON-LD keywords -- treat as absent
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
