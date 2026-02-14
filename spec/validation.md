# Validation Extensions

**Status:** Draft Specification v0.1.0  
**Date:** 2026-02-12  
**Part of:** [JSON-LD Extensions for AI/ML (jsonld-ex)](ai-ml-extensions.md)  
**Canonical URL:** [https://jsonld-ex.github.io/ns/validation](https://jsonld-ex.github.io/ns/validation)

---

## 1. Introduction

This document specifies the validation extensions for jsonld-ex. A shape-based constraint language — expressed entirely in JSON-LD-native syntax — enables processors to validate the structure, types, cardinalities, and relationships of nodes in a JSON-LD document without requiring external tools or languages.

The formal property definitions for all 22 validation keywords (IRIs, types, ranges) are in the [Vocabulary](vocabulary.md) specification, §16. This document defines the shape model, constraint semantics, inheritance resolution, validation algorithms, error reporting, and the relationship to existing constraint languages.

### 1.1 Motivation

JSON-LD 1.1 has no built-in validation mechanism. A document that is syntactically valid JSON-LD may be semantically broken: a `Person` node with a numeric `name`, a `Product` with a negative `price`, or a date range where the start exceeds the end. Detecting these errors currently requires one of three approaches:

1. **SHACL** (Shapes Constraint Language) — a W3C Recommendation that operates on RDF graphs. SHACL is powerful but requires RDF materialization, introduces a separate constraint language with its own syntax, and has a steep learning curve for developers who work primarily with JSON.

2. **ShEx** (Shape Expressions) — a concise constraint language for RDF. Like SHACL, ShEx requires RDF-level thinking and tooling.

3. **JSON Schema** — validates JSON syntax but has no awareness of JSON-LD semantics. A JSON Schema can enforce that a field is a string, but cannot enforce that a term maps to a specific IRI or that a value object carries a valid `@confidence` annotation.

The jsonld-ex validation extensions address this gap with a constraint language that:

- Uses JSON-LD-native syntax. Shapes are JSON objects with `@`-prefixed keywords — no new language to learn.
- Operates on JSON-LD nodes directly, without requiring RDF materialization.
- Supports the constraints most commonly needed in practice: type checking, numeric bounds, string patterns, cardinality, enumerations, logical combinators, conditional constraints, and cross-property comparisons.
- Maps cleanly to SHACL for interoperability with RDF-native tooling (every jsonld-ex constraint has a documented SHACL equivalent).
- Supports shape inheritance via `@extends`, enabling reuse and composition of constraint definitions.

### 1.2 Conformance

The key words "MUST", "MUST NOT", "SHOULD", and "MAY" in this document are to be interpreted as described in [RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

### 1.3 Terminology

- **Shape** — a JSON object that declares constraints on a JSON-LD node. A shape specifies the expected `@type`, required properties, and per-property constraints.
- **Constraint** — a single requirement on a property value, expressed as a keyword–value pair within a property constraint object (e.g., `"@minimum": 0`).
- **Property constraint object** — the JSON object associated with a property name inside a shape. It contains one or more constraints that the property's value must satisfy.
- **Node** — a JSON-LD node object (a JSON object with `@type` and/or `@id`).
- **Value object** — a JSON-LD value object (a JSON object with `@value`).
- **Raw value** — the scalar value extracted from a property for constraint evaluation. For a value object `{"@value": "Alice"}`, the raw value is `"Alice"`. For a plain scalar `42`, the raw value is `42`. For a list, the raw value is extracted from the first element.
- **Shape registry** — a named mapping from string identifiers to shape definitions, used to resolve `@extends` references.
- **Severity** — the classification of a constraint violation as an error (causing validation failure) or a warning/info (reported but not causing failure).

---

## 2. Shape Definition Model

A shape is a JSON object that constrains the structure of a JSON-LD node. Shapes are the top-level unit of validation: each shape targets a specific node type and declares constraints on its properties.

### 2.1 Shape Structure

A shape has the following structure:

```json
{
  "@shape": {
    "@type": "TypeName",
    "@extends": "ParentShapeName",
    "propertyA": { ... constraint object ... },
    "propertyB": { ... constraint object ... }
  }
}
```

The keys of a shape are:

| Key | Required | Description |
|-----|----------|-------------|
| `@type` | No | The expected `@type` of the node. If present, the node's `@type` (or list of types) MUST include this value. |
| `@extends` | No | One or more parent shapes to inherit constraints from (see §8). |
| _property names_ | No | Each non-`@` key names a property to constrain. Its value is a property constraint object. |

Keys beginning with `@` that are not recognized validation keywords are ignored. Non-dict values for property keys are ignored.

### 2.2 Type Matching

When a shape declares `@type`, the validator checks whether the node's type set includes the declared type.

**Algorithm:**

1. Extract the node's type(s): if `@type` is a string, the type set is `[string]`; if `@type` is an array, the type set is that array; if `@type` is absent, the type set is empty.
2. If the shape declares `@type` and the shape's type is NOT in the node's type set, emit a type error.

A node with multiple types (e.g., `"@type": ["Person", "Employee"]`) satisfies a shape that declares `"@type": "Person"` or a shape that declares `"@type": "Employee"`.

**Example:**

```json
// Shape
{
  "@shape": {
    "@type": "Person",
    "name": {"@required": true}
  }
}

// Valid node
{
  "@type": ["Person", "Employee"],
  "name": "Alice"
}

// Invalid node (missing type)
{
  "@type": "Organization",
  "name": "Acme Corp"
}
```

### 2.3 Property Constraint Objects

Each property in a shape maps to a constraint object — a JSON object containing one or more constraint keywords. The constraint keywords are evaluated against the property's value in the node being validated.

**Example:**

```json
{
  "@shape": {
    "@type": "Product",
    "name": {
      "@required": true,
      "@type": "xsd:string",
      "@minLength": 1,
      "@maxLength": 200
    },
    "price": {
      "@type": "xsd:double",
      "@minimum": 0
    }
  }
}
```

In this shape, the `name` property has four constraints (required, type, minimum length, maximum length) and the `price` property has two (type, minimum value).

### 2.4 Value Extraction

Before evaluating constraints, the validator extracts the **raw value** from the property:

1. If the property is absent from the node, the raw value is `null`.
2. If the property value is a value object (a dict containing `@value`), the raw value is the `@value` content.
3. If the property value is a list, the raw value is extracted from the first element (recursively applying rules 2–3).
4. If the property value is an empty list, the raw value is `null`.
5. If the property value is a plain dict with no JSON-LD keywords (no keys starting with `@`), the raw value is `null` (treated as a structured node, not a scalar).
6. Otherwise, the property value is used as the raw value directly.

This extraction ensures that constraints operate on the semantic value regardless of whether it is wrapped in a value object:

```json
// Both of these yield raw value "Alice" for constraint evaluation:
{"name": "Alice"}
{"name": {"@value": "Alice", "@confidence": 0.95}}
```

**Processing note:** Cardinality constraints (`@minCount`, `@maxCount`) operate on the original property value _before_ raw value extraction, since they need to count the number of values rather than inspect a single scalar. See §4 for details.

---

## 3. Atomic Constraints

Atomic constraints evaluate a single property's raw value against a single requirement. They are the building blocks of property constraint objects.

### 3.1 @required

When `@required` is `true`, the property MUST be present in the node and its raw value MUST NOT be `null`.

If the raw value is `null` (property absent, empty list, or a plain dict without `@value`), the validator emits a violation.

When a `@required` constraint is violated, the remaining constraints on the same property are skipped — there is no value to evaluate them against.

**SHACL equivalent:** `sh:minCount 1`

**Example:**

```json
// Shape
{"name": {"@required": true}}

// Valid
{"name": "Alice"}
{"name": {"@value": "Alice"}}

// Invalid (property absent)
{"email": "alice@example.com"}

// Invalid (null raw value from empty list)
{"name": []}
```

### 3.2 @type (Datatype Constraint)

When `@type` is present in a property constraint object, the raw value MUST conform to the specified XSD datatype.

**Supported datatypes:**

| Datatype | Compact Form | Accepted Python/JSON Types |
|----------|-------------|---------------------------|
| `http://www.w3.org/2001/XMLSchema#string` | `xsd:string` | String |
| `http://www.w3.org/2001/XMLSchema#integer` | `xsd:integer` | Integer (not boolean) |
| `http://www.w3.org/2001/XMLSchema#double` | `xsd:double` | Integer or float (not boolean) |
| `http://www.w3.org/2001/XMLSchema#float` | `xsd:float` | Integer or float (not boolean) |
| `http://www.w3.org/2001/XMLSchema#decimal` | `xsd:decimal` | Integer or float (not boolean) |
| `http://www.w3.org/2001/XMLSchema#boolean` | `xsd:boolean` | Boolean |

The compact form (e.g., `xsd:string`) is expanded by replacing the `xsd:` prefix with `http://www.w3.org/2001/XMLSchema#` before matching.

Booleans are explicitly excluded from numeric type checks. Although `bool` is a subclass of `int` in Python (and JSON booleans are sometimes coerced to numbers in other environments), a boolean value does NOT satisfy `xsd:integer`, `xsd:double`, `xsd:float`, or `xsd:decimal`.

If the raw value's type does not match, the validator emits a type violation. If the declared type is not in the supported set, no type check is performed (the constraint is silently skipped, allowing forward compatibility with future type extensions).

**SHACL equivalent:** `sh:datatype`

**Example:**

```json
// Shape
{
  "name": {"@type": "xsd:string"},
  "age": {"@type": "xsd:integer"},
  "active": {"@type": "xsd:boolean"}
}

// Valid
{"name": "Alice", "age": 30, "active": true}

// Invalid: name is numeric
{"name": 12345, "age": 30, "active": true}

// Invalid: age is boolean (boolean excluded from integer)
{"name": "Alice", "age": true, "active": true}
```

### 3.3 @minimum and @maximum

`@minimum` specifies that the numeric raw value MUST be greater than or equal to the given number. `@maximum` specifies that the numeric raw value MUST be less than or equal to the given number.

These constraints apply only when the raw value is numeric (integer or float) and not a boolean. If the raw value is not numeric, the constraints are silently skipped.

Both constraints may be present on the same property to define a valid range.

**SHACL equivalents:** `sh:minInclusive`, `sh:maxInclusive`

**Example:**

```json
// Shape
{
  "age": {
    "@type": "xsd:integer",
    "@minimum": 0,
    "@maximum": 150
  },
  "confidence": {
    "@type": "xsd:double",
    "@minimum": 0.0,
    "@maximum": 1.0
  }
}

// Valid
{"age": 30, "confidence": 0.85}

// Invalid: age below minimum
{"age": -1, "confidence": 0.85}

// Invalid: confidence above maximum
{"age": 30, "confidence": 1.5}
```

### 3.4 @minLength and @maxLength

`@minLength` specifies that the string raw value MUST have at least the given number of characters. `@maxLength` specifies that the string raw value MUST have at most the given number of characters.

These constraints apply only when the raw value is a string. If the raw value is not a string, the constraints are silently skipped.

**SHACL equivalents:** `sh:minLength`, `sh:maxLength`

**Example:**

```json
// Shape
{
  "name": {
    "@type": "xsd:string",
    "@minLength": 1,
    "@maxLength": 100
  }
}

// Valid
{"name": "Alice"}

// Invalid: empty string below minLength
{"name": ""}

// Invalid: exceeds maxLength (if length > 100)
{"name": "A very long name..."}
```

### 3.5 @pattern

`@pattern` specifies a regular expression that the string raw value MUST match. The match uses a **search** semantic (the pattern may match anywhere within the string), not a full-match semantic.

The constraint applies only when the raw value is a string. If the raw value is not a string, the constraint is silently skipped.

If the pattern is an invalid regular expression, the validator emits a pattern error indicating the regex syntax problem rather than silently passing.

**SHACL equivalent:** `sh:pattern`

**Example:**

```json
// Shape
{
  "email": {
    "@pattern": "^[^@]+@[^@]+$"
  },
  "zipCode": {
    "@pattern": "^\\d{5}(-\\d{4})?$"
  }
}

// Valid
{"email": "alice@example.com", "zipCode": "90210"}

// Invalid: email missing @
{"email": "alice.example.com"}
```

### 3.6 @in

`@in` specifies an array of allowed values. The raw value MUST be a member of the array (tested by equality).

**SHACL equivalent:** `sh:in`

**Example:**

```json
// Shape
{
  "status": {
    "@in": ["active", "inactive", "pending"]
  },
  "priority": {
    "@in": [1, 2, 3, 4, 5]
  }
}

// Valid
{"status": "active", "priority": 3}

// Invalid: status not in allowed set
{"status": "archived"}
```

---

## 4. Cardinality Constraints

Cardinality constraints operate on the property value _before_ raw value extraction. They count the number of values rather than inspecting a single scalar.

### 4.1 Value Counting

The count of values for a property is determined as follows:

| Property Value | Count |
|---------------|-------|
| Absent (`null`) | 0 |
| A list | Length of the list |
| Any other single value | 1 |

### 4.2 @minCount

`@minCount` specifies the minimum number of values the property MUST have. If the count is less than `@minCount`, the validator emits a violation.

Setting `@minCount` to 1 is semantically similar to `@required: true`, but there is a behavioral difference: `@required` causes remaining constraints to be skipped on violation, while `@minCount` does not.

**SHACL equivalent:** `sh:minCount`

### 4.3 @maxCount

`@maxCount` specifies the maximum number of values the property MUST have. If the count exceeds `@maxCount`, the validator emits a violation.

**SHACL equivalent:** `sh:maxCount`

**Example:**

```json
// Shape
{
  "email": {
    "@minCount": 1,
    "@maxCount": 3,
    "@pattern": "^[^@]+@[^@]+$"
  }
}

// Valid: single value (count = 1)
{"email": "alice@example.com"}

// Valid: two values (count = 2)
{"email": ["alice@example.com", "alice@work.com"]}

// Invalid: no values (count = 0, below minCount)
{}

// Invalid: four values (count = 4, above maxCount)
{"email": ["a@b.com", "c@d.com", "e@f.com", "g@h.com"]}
```

---

## 5. Logical Combinators

Logical combinators compose constraints using boolean logic. They enable complex validation rules that cannot be expressed with atomic constraints alone.

All three combinators — `@or`, `@and`, `@not` — evaluate their sub-constraints by recursively invoking the same constraint evaluation mechanism used for atomic constraints. This means sub-constraints may themselves contain logical combinators, enabling arbitrary nesting.

### 5.1 @or (Disjunction)

`@or` specifies an array of constraint objects. The raw value MUST satisfy **at least one** of the branches. If no branch is satisfied, the validator emits a violation.

Evaluation is short-circuit: branches are tested in order, and evaluation stops at the first branch that produces no errors.

**SHACL equivalent:** `sh:or`

**Example:**

```json
// Shape: value must be either a short code or a full name
{
  "identifier": {
    "@or": [
      {"@type": "xsd:integer", "@minimum": 1000, "@maximum": 9999},
      {"@type": "xsd:string", "@minLength": 3, "@maxLength": 50}
    ]
  }
}

// Valid: satisfies first branch
{"identifier": 1234}

// Valid: satisfies second branch
{"identifier": "ACME-WIDGET-001"}

// Invalid: satisfies neither branch
{"identifier": 99}
```

### 5.2 @and (Conjunction)

`@and` specifies an array of constraint objects. The raw value MUST satisfy **all** of the branches. If any branch fails, the validator emits a violation.

Evaluation is short-circuit: branches are tested in order, and evaluation stops at the first branch that produces errors (fail-fast).

**SHACL equivalent:** `sh:and`

**Example:**

```json
// Shape: value must be a string that is both a valid email and at least 5 characters
{
  "email": {
    "@and": [
      {"@type": "xsd:string", "@minLength": 5},
      {"@pattern": "^[^@]+@[^@]+$"}
    ]
  }
}

// Valid: satisfies both branches
{"email": "alice@example.com"}

// Invalid: fails minLength branch (too short)
{"email": "a@b"}
```

### 5.3 @not (Negation)

`@not` specifies a single constraint object. The raw value MUST NOT satisfy the inner constraint. If the inner constraint produces no errors (i.e., the value satisfies it), the `@not` constraint emits a violation.

**SHACL equivalent:** `sh:not`

**Example:**

```json
// Shape: status must not be "deleted"
{
  "status": {
    "@not": {"@in": ["deleted", "archived"]}
  }
}

// Valid: "active" is not in the forbidden set
{"status": "active"}

// Invalid: "deleted" satisfies the inner @in, so @not fails
{"status": "deleted"}
```

### 5.4 Nesting and Composition

Logical combinators may be nested arbitrarily:

```json
// Shape: value is either (an integer >= 0) or (a string that is not empty)
{
  "value": {
    "@or": [
      {"@and": [
        {"@type": "xsd:integer"},
        {"@minimum": 0}
      ]},
      {"@and": [
        {"@type": "xsd:string"},
        {"@not": {"@maxLength": 0}}
      ]}
    ]
  }
}
```

Logical combinators may also appear alongside atomic constraints in the same property constraint object. When they do, all constraints — atomic and logical — are evaluated independently, and all violations are collected:

```json
{
  "score": {
    "@type": "xsd:double",
    "@minimum": 0,
    "@or": [
      {"@maximum": 1.0},
      {"@in": [-1]}
    ]
  }
}
```

In this example, the `@type` and `@minimum` constraints are evaluated independently of the `@or` combinator. A value of `0.5` satisfies all three. A value of `2.0` satisfies `@type` and `@minimum` but fails the `@or`. A value of `"hello"` fails `@type` (the `@minimum` is silently skipped for non-numeric values).

---

## 6. Conditional Constraints

Conditional constraints enable if-then-else logic within validation. They allow different constraints to apply depending on the value itself.

### 6.1 @if / @then / @else

The `@if`, `@then`, and `@else` keywords form a conditional constraint:

1. Evaluate the `@if` constraint against the raw value.
2. If the `@if` constraint is **satisfied** (produces no errors), evaluate the `@then` constraint. If `@then` produces errors, emit a conditional violation.
3. If the `@if` constraint is **not satisfied** (produces errors), evaluate the `@else` constraint (if present). If `@else` produces errors, emit a conditional violation.
4. If `@if` is not satisfied and `@else` is absent, the conditional constraint is **vacuously true** — no violation is emitted.

The `@if` keyword is required for conditional evaluation. `@then` and `@else` are each optional, but at least one SHOULD be present for the conditional to have any effect.

**Example:**

```json
// Shape: if age >= 18, category must be "adult" or "senior";
//        otherwise, category must be "minor"
{
  "age": {
    "@if": {"@minimum": 18},
    "@then": {"@in": ["adult", "senior"]},
    "@else": {"@in": ["minor"]}
  }
}
```

Note that in this example, the `@if`, `@then`, and `@else` all evaluate against the _same_ raw value (the value of the `age` property). This is a property-level conditional, not a node-level conditional. The constraint is useful when the acceptable values of a property depend on the value's own characteristics (e.g., range-dependent enumeration).

### 6.2 Interaction with Other Constraints

Conditional constraints may appear alongside atomic constraints, logical combinators, and cross-property constraints in the same property constraint object. All constraints are evaluated independently:

```json
{
  "score": {
    "@type": "xsd:double",
    "@minimum": 0,
    "@if": {"@minimum": 0.5},
    "@then": {"@maximum": 1.0}
  }
}
```

Here, `@type` and `@minimum` are always enforced. The conditional adds: if `score >= 0.5`, then `score <= 1.0`.

---

## 7. Cross-Property Constraints

Cross-property constraints compare a property's value to the value of another property on the same node. They enable relational validation that cannot be expressed with per-property constraints alone.

All cross-property constraints reference a **sibling property** by name. The sibling's value is extracted using the same raw-value extraction rules (§2.4). If the sibling property is absent or its raw value is `null`, the cross-property constraint is silently skipped (no violation, no error).

If the comparison raises a type error (e.g., comparing a string to an integer), the validator emits a violation indicating that the types are incomparable.

### 7.1 @lessThan

The raw value of the constrained property MUST be strictly less than the raw value of the named sibling property.

**SHACL equivalent:** `sh:lessThan`

### 7.2 @lessThanOrEquals

The raw value of the constrained property MUST be less than or equal to the raw value of the named sibling property.

**SHACL equivalent:** `sh:lessThanOrEquals`

### 7.3 @equals

The raw value of the constrained property MUST be equal to the raw value of the named sibling property.

**SHACL equivalent:** `sh:equals`

### 7.4 @disjoint

The raw value of the constrained property MUST differ from (not be equal to) the raw value of the named sibling property.

**SHACL equivalent:** `sh:disjoint`

### 7.5 Cross-Property Constraint Example

```json
// Shape: startDate must be before endDate, and confirmEmail must match email
{
  "@shape": {
    "@type": "Event",
    "startDate": {
      "@required": true,
      "@lessThan": "endDate"
    },
    "endDate": {
      "@required": true
    },
    "email": {
      "@required": true,
      "@pattern": "^[^@]+@[^@]+$"
    },
    "confirmEmail": {
      "@equals": "email"
    },
    "alternateEmail": {
      "@disjoint": "email"
    }
  }
}

// Valid
{
  "@type": "Event",
  "startDate": "2026-01-01",
  "endDate": "2026-12-31",
  "email": "organizer@example.com",
  "confirmEmail": "organizer@example.com",
  "alternateEmail": "backup@example.com"
}

// Invalid: startDate not less than endDate
{
  "@type": "Event",
  "startDate": "2026-12-31",
  "endDate": "2026-01-01",
  "email": "organizer@example.com",
  "confirmEmail": "organizer@example.com",
  "alternateEmail": "organizer@example.com"
}
```

In the invalid example, two violations are emitted: `startDate` is not less than `endDate`, and `alternateEmail` is not disjoint from `email`.

---

## 8. Shape Inheritance

Shape inheritance enables constraint reuse and composition. A shape may declare one or more parent shapes via `@extends`, inheriting all of the parents' property constraints.

### 8.1 The @extends Keyword

`@extends` accepts:

- A **string** — a named reference resolved against the shape registry.
- An **inline shape object** — a shape definition provided directly.
- An **array** of strings and/or inline shape objects — multiple parents.

### 8.2 Resolution Algorithm

**Input:** A shape with `@extends`, a shape registry (mapping names to shape definitions), and a set of already-visited shapes (initially empty, for cycle detection).

**Procedure:**

1. Record the current shape's identity in the visited set. If it was already present, return the shape unchanged (cycle detected and broken).
2. Extract the `@extends` value. Normalize it to an array if it is a single value.
3. For each parent reference, in order:
   a. If the reference is a dict (inline shape), recursively resolve its own `@extends` (if any).
   b. If the reference is a string, look it up in the shape registry. If found, recursively resolve its `@extends`. If not found, emit a warning (`"unresolved"`) and skip the reference.
   c. Other types are silently ignored.
4. Build the merged shape by applying parents left-to-right, then the child on top:
   a. Start with an empty merged shape.
   b. For each resolved parent, merge it into the accumulated result.
   c. Finally, merge the child shape (the original shape minus `@extends`) on top.
5. Remove the `@extends` key from the merged result.
6. Return the merged shape and any accumulated warnings.

### 8.3 Merge Semantics

When merging a source shape into a target shape:

- For **top-level keys** (e.g., `@type`): the source value overwrites the target value. Later sources (including the child) override earlier ones.
- For **property constraint objects** (keys whose values are dicts in both source and target): the merge is shallow at the constraint level. All keys from both the target and source are included. Where both define the same constraint key, the source value wins.
- The `@extends` key in any source is ignored during merging (it has already been resolved).

### 8.4 Inheritance Order

With multiple parents `["A", "B"]`, the merge order is:

1. Start with empty shape.
2. Merge parent A.
3. Merge parent B (B's constraints override A's for the same keys).
4. Merge the child shape (child overrides everything).

This follows a left-to-right, child-wins resolution strategy.

### 8.5 Circular Inheritance Detection

If a shape's inheritance chain contains a cycle (A extends B, B extends A), the cycle is detected via the visited set and broken by returning the shape at the point where the cycle is detected. No error is raised; the shape is resolved with the constraints accumulated up to the cycle point.

### 8.6 Shape Inheritance Example

```json
// Shape registry
{
  "NamedEntity": {
    "name": {
      "@required": true,
      "@type": "xsd:string",
      "@minLength": 1
    }
  },
  "Timestamped": {
    "createdAt": {"@required": true},
    "updatedAt": {"@required": true}
  }
}

// Child shape extending both
{
  "@shape": {
    "@type": "Person",
    "@extends": ["NamedEntity", "Timestamped"],
    "name": {
      "@maxLength": 200
    },
    "email": {
      "@required": true,
      "@pattern": "^[^@]+@[^@]+$"
    }
  }
}

// Resolved shape (after inheritance):
{
  "@type": "Person",
  "name": {
    "@required": true,
    "@type": "xsd:string",
    "@minLength": 1,
    "@maxLength": 200
  },
  "createdAt": {"@required": true},
  "updatedAt": {"@required": true},
  "email": {
    "@required": true,
    "@pattern": "^[^@]+@[^@]+$"
  }
}
```

In this example, the child's `name` constraint adds `@maxLength` to the inherited `@required`, `@type`, and `@minLength` from `NamedEntity`. The `createdAt` and `updatedAt` constraints are inherited from `Timestamped` unchanged. The `email` constraint is entirely new in the child.

---

## 9. Severity Levels

The `@severity` keyword controls how constraint violations are classified. It enables shapes to distinguish between hard errors (which cause validation failure) and soft warnings (which are reported but do not prevent a document from being considered valid).

### 9.1 Severity Values

| Value | Behavior |
|-------|----------|
| `"error"` | Violation is recorded as a `ValidationError`. The node fails validation. This is the **default**. |
| `"warning"` | Violation is recorded as a `ValidationWarning`. The node does NOT fail validation. |
| `"info"` | Violation is recorded as a `ValidationWarning`. The node does NOT fail validation. |

Both `"warning"` and `"info"` route to the warnings list. The distinction is semantic: `"warning"` suggests the violation should be reviewed; `"info"` is purely informational.

### 9.2 Scope

`@severity` is set per property constraint object. All constraints within the same property constraint object share the same severity:

```json
{
  "nickname": {
    "@severity": "warning",
    "@type": "xsd:string",
    "@maxLength": 50
  }
}
```

If `nickname` is numeric (failing `@type`) or too long (failing `@maxLength`), both violations are reported as warnings rather than errors.

### 9.3 Interaction with Validation Result

A `ValidationResult` is `valid: true` if and only if the `errors` list is empty. The `warnings` list does not affect validity:

```json
// Result with warnings but no errors
{
  "valid": true,
  "errors": [],
  "warnings": [
    {"path": "nickname", "code": "type", "message": "Expected xsd:string, got int: 42"}
  ]
}
```

**SHACL equivalent:** `sh:severity` (with values `sh:Violation`, `sh:Warning`, `sh:Info`)

---

## 10. Nested Shape Validation

A property constraint object may contain a `@shape` key, indicating that the property's value is itself a node that must conform to an inner shape. This enables recursive, hierarchical validation.

### 10.1 Behavior

When `@shape` is present in a property constraint object:

1. Determine the target node:
   a. If the property value is a list, use the first element.
   b. Otherwise, use the property value directly.
2. If the target is not a dict, emit a violation (expected a node for shape validation).
3. If the target is a dict, validate it against the inner shape by recursively invoking the node validation algorithm.
4. If the inner validation fails, emit a violation with the first inner error message. All inner error paths are prefixed with the property name (e.g., `address/street`).
5. When `@shape` is the primary constraint, scalar constraints (`@type`, `@minimum`, etc.) in the same property constraint object are **not evaluated** — the `@shape` takes precedence and subsequent processing continues to the next property.

### 10.2 Example

```json
// Shape with nested shape
{
  "@shape": {
    "@type": "Person",
    "name": {"@required": true, "@type": "xsd:string"},
    "address": {
      "@shape": {
        "@type": "PostalAddress",
        "streetAddress": {"@required": true},
        "postalCode": {"@pattern": "^\\d{5}$"}
      }
    }
  }
}

// Valid
{
  "@type": "Person",
  "name": "Alice",
  "address": {
    "@type": "PostalAddress",
    "streetAddress": "123 Main St",
    "postalCode": "90210"
  }
}

// Invalid: nested node fails
{
  "@type": "Person",
  "name": "Alice",
  "address": {
    "@type": "PostalAddress",
    "postalCode": "ABCDE"
  }
}
```

In the invalid example, two errors are emitted for the `address` property: `streetAddress` is required (missing), and `postalCode` does not match the pattern. The error paths are `address/streetAddress` and `address/postalCode`.

---

## 11. Document-Level Validation

In addition to validating individual nodes, the validation extensions support document-level validation, which automatically matches shapes to nodes by type and validates all matching pairs.

### 11.1 Node Extraction

Nodes are extracted from a document recursively:

1. If the document is a list, recurse into each element and collect all nodes.
2. If the document is a dict:
   a. If it has an `@type` key, it is a node — include it.
   b. If it has an `@graph` key, recurse into the `@graph` value and collect nodes.
3. Other types are ignored.

### 11.2 Type-Based Shape Matching

For each extracted node, its type set is compared to the `@type` declared by each shape. If a shape's `@type` is in the node's type set, the node is validated against that shape.

A single node may match multiple shapes (if multiple shapes declare types that appear in the node's type set). Each matching shape is evaluated independently, and all violations are collected.

### 11.3 Error Path Prefixing

When validating a document with multiple nodes, error paths are prefixed with the node's `@id` (or `"anonymous"` if no `@id` is present) to distinguish which node produced the violation:

```
"http://example.org/alice/name" → required violation
"anonymous/email" → pattern violation
```

### 11.4 Example

```json
// Document with @graph
{
  "@context": "https://schema.org/",
  "@graph": [
    {
      "@id": "http://example.org/alice",
      "@type": "Person",
      "name": "Alice",
      "email": "alice@example.com"
    },
    {
      "@id": "http://example.org/acme",
      "@type": "Organization",
      "name": "Acme Corp"
    },
    {
      "@type": "Person",
      "email": "invalid-email"
    }
  ]
}

// Shapes
[
  {
    "@type": "Person",
    "name": {"@required": true, "@type": "xsd:string"},
    "email": {"@pattern": "^[^@]+@[^@]+$"}
  },
  {
    "@type": "Organization",
    "name": {"@required": true}
  }
]
```

Validation results:
- `http://example.org/alice` — valid (satisfies Person shape).
- `http://example.org/acme` — valid (satisfies Organization shape).
- `anonymous` — two violations: `name` is required but absent; `email` does not match the pattern.

---

## 12. Error Reporting

The validation extensions define a structured error reporting model with three data types.

### 12.1 ValidationError

A `ValidationError` represents a constraint violation at error severity. Its fields are:

| Field | Type | Description |
|-------|------|-------------|
| `path` | String | The property path where the violation occurred (e.g., `"name"`, `"address/street"`, `"http://example.org/alice/email"`). |
| `constraint` | String | The constraint keyword that was violated (e.g., `"required"`, `"minimum"`, `"pattern"`, `"type"`, `"or"`, `"and"`, `"not"`, `"conditional"`, `"lessThan"`, `"equals"`, `"disjoint"`, `"shape"`, `"minCount"`, `"maxCount"`). |
| `message` | String | A human-readable description of the violation, including the actual value and the expected constraint. |
| `value` | Any | The value that caused the violation (may be `null` for required violations). |

### 12.2 ValidationWarning

A `ValidationWarning` represents a constraint violation at warning or info severity, or an informational notice from the validation process (e.g., unresolved shape references during inheritance). Its fields are:

| Field | Type | Description |
|-------|------|-------------|
| `path` | String | The property path or context (e.g., `"nickname"`, `"@extends"`). |
| `code` | String | A short identifier for the warning type (e.g., `"type"`, `"pattern"`, `"unresolved"`). |
| `message` | String | A human-readable description. |

### 12.3 ValidationResult

A `ValidationResult` aggregates errors and warnings from a validation pass. Its fields are:

| Field | Type | Description |
|-------|------|-------------|
| `valid` | Boolean | `true` if the `errors` list is empty; `false` otherwise. |
| `errors` | List of `ValidationError` | All error-severity violations. |
| `warnings` | List of `ValidationWarning` | All warning/info-severity violations and informational notices. |

### 12.4 Error Message Guidelines

Error messages SHOULD:

- Include the actual value that caused the violation.
- Include the constraint threshold or expected value.
- Be human-readable without requiring knowledge of the shape definition.

**Examples:**

```
Property "name" is required
Value 200 exceeds maximum 150
Length 0 below minimum 1
"invalid" does not match pattern "^[^@]+@[^@]+$"
Expected xsd:string, got int: 42
Value 'X' not in allowed set ['A', 'B', 'C']
Expected at least 2 value(s), found 1
Value 42 did not satisfy any @or branch
Value "2026-12-31" is not less than endDate="2026-01-01"
```

---

## 13. Validation Algorithm Summary

This section provides a consolidated description of the complete node validation algorithm, combining all mechanisms described in §§2–12.

### 13.1 Node Validation

**Input:** A node (JSON object), a shape (JSON object), and an optional shape registry (mapping of names to shapes).

**Output:** A `ValidationResult`.

**Procedure:**

1. **Input check.** If the node is not a dict, return invalid with a type error.
2. **Resolve inheritance.** If the shape contains `@extends`, resolve it using the shape registry (§8.2). Collect any resolution warnings.
3. **Type check.** If the shape declares `@type`, extract the node's type set. If the shape's type is not in the type set, emit a type error.
4. **For each property** in the shape (keys that do not start with `@` and whose values are dicts):
   a. Retrieve the property value from the node.
   b. Determine the severity from `@severity` (default: `"error"`).
   c. **Cardinality.** Evaluate `@minCount` and `@maxCount` against the value count (§4).
   d. **Extract raw value** (§2.4).
   e. **Required check.** If `@required` is `true` and the raw value is `null`, emit a required violation and skip to the next property.
   f. If the raw value is `null` and the property is absent, skip to the next property.
   g. **Nested shape.** If `@shape` is present, validate the property value as a nested node (§10). Skip scalar constraints and continue to the next property.
   h. If the raw value is `null` (after extraction), skip to the next property.
   i. **Constraint evaluation.** Evaluate all constraints — atomic (§3), logical (§5), conditional (§6), and cross-property (§7) — against the raw value. Collect all violations.
   j. **Severity routing.** For each violation, route it to errors or warnings based on the property's severity.
5. **Return** the `ValidationResult` with `valid = (errors is empty)`, the collected errors, and the collected warnings.

### 13.2 Document Validation

**Input:** A document (JSON value) and a list of shapes.

**Output:** A `ValidationResult`.

**Procedure:**

1. Extract all nodes from the document (§11.1).
2. For each node, determine its type set.
3. For each shape, if the shape's `@type` is in the node's type set, validate the node against the shape (§13.1).
4. Prefix all error paths with the node's `@id` (or `"anonymous"`).
5. Collect all errors and warnings across all node-shape validations.
6. Return the `ValidationResult`.

---

## 14. Backward Compatibility

### 14.1 Non-Extended Processors

A standard JSON-LD 1.1 processor that does not implement the jsonld-ex validation extensions will:

- **`@shape` and all validation keywords** — These keywords are not defined in JSON-LD 1.1. A non-extended processor will treat them as regular JSON properties. A document containing `@shape` will expand normally — the shape definition will be preserved as data rather than interpreted as validation instructions.
- **No processing impact** — Validation is an opt-in processing step. Documents that include shape definitions are valid JSON-LD with or without a validation-aware processor. The shapes are simply inert metadata when no validator is present.

### 14.2 Graceful Degradation

The validation extensions degrade gracefully:

- Documents with shapes embedded alongside data are valid JSON-LD.
- Shapes can be stored in separate documents and applied selectively by validation-aware processors.
- A processor that implements only a subset of constraint keywords (e.g., atomic constraints but not logical combinators) can still provide partial validation. Unknown constraint keywords within a property constraint object are silently ignored, preserving forward compatibility.

### 14.3 Migration Path

Adopting the validation extensions requires no changes to existing JSON-LD documents. Shapes are additive — they constrain existing data without modifying it. Organizations can:

1. Start by defining shapes for their most critical document types.
2. Integrate validation into their processing pipelines as an optional step.
3. Gradually increase shape coverage as confidence in the validation framework grows.

---

## 15. Relationship to Existing Standards

### 15.1 SHACL (Shapes Constraint Language)

[SHACL](https://www.w3.org/TR/shacl/) is a W3C Recommendation for validating RDF graphs. It is the most comprehensive constraint language for linked data. The jsonld-ex validation extensions are designed to complement SHACL, not replace it.

| Aspect | SHACL | jsonld-ex Validation |
|--------|-------|---------------------|
| **Operates on** | RDF graphs (triples) | JSON-LD node objects (JSON) |
| **Requires RDF materialization** | Yes | No |
| **Syntax** | RDF (Turtle, JSON-LD, etc.) | JSON-LD native (`@`-prefixed keywords) |
| **Expressiveness** | Full (SPARQL-based constraints, property paths, etc.) | Practical subset (covers ~80% of common validation needs) |
| **Learning curve** | Steep (requires RDF knowledge) | Low (JSON-native, familiar to web developers) |
| **Constraint mapping** | N/A | Every jsonld-ex constraint has a documented SHACL equivalent |
| **Logical combinators** | `sh:or`, `sh:and`, `sh:not`, `sh:xone` | `@or`, `@and`, `@not` (no exclusive-or) |
| **Conditional constraints** | Via SPARQL-based constraints | `@if` / `@then` / `@else` |
| **Cross-property constraints** | `sh:lessThan`, `sh:equals`, `sh:disjoint`, etc. | `@lessThan`, `@lessThanOrEquals`, `@equals`, `@disjoint` |
| **Shape inheritance** | `rdfs:subClassOf` and target declarations | `@extends` with explicit merge semantics |
| **Severity** | `sh:Violation`, `sh:Warning`, `sh:Info` | `"error"`, `"warning"`, `"info"` |

**Design decision:** The jsonld-ex validation extensions intentionally do not attempt to replicate SHACL's full expressiveness. SPARQL-based constraints, complex property paths, and advanced target selection are out of scope. The goal is to cover the validation needs of the vast majority of JSON-LD consumers — type checking, required properties, numeric bounds, string patterns, cardinalities, and basic logical composition — in a syntax that requires no RDF knowledge.

For use cases that require SHACL's full power, the clean constraint mapping (documented in the vocabulary specification) enables straightforward conversion from jsonld-ex shapes to SHACL shapes.

### 15.2 ShEx (Shape Expressions)

[ShEx](https://shex.io/shex-semantics/) is a concise constraint language for RDF, focused on structural validation. Like SHACL, ShEx operates on RDF graphs and requires RDF-level thinking.

| Aspect | ShEx | jsonld-ex Validation |
|--------|------|---------------------|
| **Operates on** | RDF graphs | JSON-LD node objects |
| **Syntax** | ShExC (compact) or ShExJ (JSON) | JSON-LD native |
| **Expressiveness** | Structural validation with recursion | Similar structural coverage, plus cross-property constraints |
| **Inheritance** | `EXTENDS` keyword | `@extends` keyword (similar semantics) |

### 15.3 JSON Schema

[JSON Schema](https://json-schema.org/) validates JSON documents against structural schemas. It has wide adoption in the web development community.

| Aspect | JSON Schema | jsonld-ex Validation |
|--------|-------------|---------------------|
| **Operates on** | JSON documents | JSON-LD node objects |
| **Semantic awareness** | None (syntax-only) | JSON-LD aware (understands `@value`, `@type`, `@id`) |
| **Value extraction** | Direct property access | Extracts raw values from value objects |
| **Linked Data interoperability** | None | Maps to SHACL; operates within JSON-LD ecosystem |
| **Conditional** | `if` / `then` / `else` | `@if` / `@then` / `@else` |
| **Composition** | `anyOf`, `allOf`, `not`, `oneOf` | `@or`, `@and`, `@not` |
| **Cross-property** | Limited (via `dependencies`) | `@lessThan`, `@equals`, `@disjoint`, etc. |

**Key distinction:** JSON Schema validates the syntactic structure of a JSON document. It cannot validate that a term maps to a specific IRI, that a value object's `@value` is a valid integer, or that a `@confidence` annotation falls within [0, 1]. The jsonld-ex validation extensions understand JSON-LD's data model and extract semantic values before validation.

### 15.4 SHACL Constraint Mapping

The following table provides the complete mapping from jsonld-ex validation keywords to SHACL constraint components:

| jsonld-ex Keyword | SHACL Equivalent | Notes |
|-------------------|-----------------|-------|
| `@required: true` | `sh:minCount 1` | |
| `@type` (datatype) | `sh:datatype` | |
| `@minimum` | `sh:minInclusive` | |
| `@maximum` | `sh:maxInclusive` | |
| `@minLength` | `sh:minLength` | |
| `@maxLength` | `sh:maxLength` | |
| `@pattern` | `sh:pattern` | jsonld-ex uses search semantics; SHACL uses full match |
| `@in` | `sh:in` | |
| `@minCount` | `sh:minCount` | |
| `@maxCount` | `sh:maxCount` | |
| `@severity` | `sh:severity` | `"error"` → `sh:Violation`, `"warning"` → `sh:Warning`, `"info"` → `sh:Info` |
| `@or` | `sh:or` | |
| `@and` | `sh:and` | |
| `@not` | `sh:not` | |
| `@if`/`@then`/`@else` | SPARQL-based constraint | No direct SHACL equivalent; requires `sh:sparql` |
| `@lessThan` | `sh:lessThan` | |
| `@lessThanOrEquals` | `sh:lessThanOrEquals` | |
| `@equals` | `sh:equals` | |
| `@disjoint` | `sh:disjoint` | |
| `@extends` | Shape targeting + `rdfs:subClassOf` | Different mechanism; similar outcome |
| `@shape` (nested) | Nested `sh:node` | |

**Pattern semantics note:** The jsonld-ex `@pattern` uses a search semantic (the regex may match anywhere in the string), while SHACL's `sh:pattern` uses a full-match semantic (the regex must match the entire string). When converting jsonld-ex patterns to SHACL, patterns that do not already have `^` and `$` anchors should be wrapped in `^.*` and `.*$` to preserve the search behavior.

---

## 16. Reference Implementation

The reference implementation is in the `jsonld_ex.validation` module of the [jsonld-ex Python package](https://pypi.org/project/jsonld-ex/).

### 16.1 Public API

| Function | Spec Section | Description |
|----------|-------------|-------------|
| `validate_node(node, shape, shape_registry=None)` | §13.1 | Validate a single JSON-LD node against a shape. Returns a `ValidationResult`. |
| `validate_document(doc, shapes)` | §13.2 | Validate all matching nodes in a document against a list of shapes. Returns a `ValidationResult`. |

### 16.2 Data Types

| Type | Fields | Description |
|------|--------|-------------|
| `ValidationResult` | `valid`, `errors`, `warnings` | Aggregated validation outcome (§12.3). |
| `ValidationError` | `path`, `constraint`, `message`, `value` | An error-severity violation (§12.1). |
| `ValidationWarning` | `path`, `code`, `message` | A warning/info-severity violation or informational notice (§12.2). |

### 16.3 Internal Functions

| Function | Spec Section |
|----------|-------------|
| `_resolve_extends(shape, registry, _seen)` | §8.2 |
| `_merge_shape_into(target, source)` | §8.3 |
| `_check_constraints(prop, raw, constraint, node)` | §§3, 5, 6, 7 |
| `_emit(errors, warnings, severity, ...)` | §9 |
| `_extract_raw(value)` | §2.4 |
| `_count_values(value)` | §4.1 |
| `_get_types(node)` | §2.2 |
| `_extract_nodes(doc)` | §11.1 |
| `_validate_type(value, expected)` | §3.2 |

---

## References

- W3C. (2020). JSON-LD 1.1. W3C Recommendation. https://www.w3.org/TR/json-ld11/
- W3C. (2017). SHACL: Shapes Constraint Language. W3C Recommendation. https://www.w3.org/TR/shacl/
- Prud'hommeaux, E., Labra Gayo, J. E., & Solbrig, H. (2014). Shape Expressions: An RDF Validation and Transformation Language. In *Proceedings of the 10th International Conference on Semantic Systems (SEM '14)*, 32–40. ACM.
- Wright, A., Andrews, H., & Hutton, B. (2022). JSON Schema: A Media Type for Describing JSON Documents. Internet-Draft. https://json-schema.org/
- Bradner, S. (1997). RFC 2119: Key words for use in RFCs to Indicate Requirement Levels. IETF. https://www.rfc-editor.org/rfc/rfc2119
