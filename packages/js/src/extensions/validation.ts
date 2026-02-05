/**
 * Validation Extensions for JSON-LD (@shape)
 *
 * Provides lightweight, native JSON-LD validation without requiring
 * external SHACL or ShEx tooling. Designed to be simple enough for
 * common use cases while mapping to SHACL for complex scenarios.
 */

import {
  KEYWORD_SHAPE, KEYWORD_REQUIRED, KEYWORD_MINIMUM, KEYWORD_MAXIMUM,
  KEYWORD_MIN_LENGTH, KEYWORD_MAX_LENGTH, KEYWORD_PATTERN,
} from '../keywords';
import {
  ShapeDefinition, PropertyShape, ValidationResult,
  ValidationError, ValidationWarning,
} from '../types';

// ── Shape Validator ───────────────────────────────────────────────

/**
 * Validates a JSON-LD node against a shape definition.
 *
 * @example
 * ```ts
 * const shape = {
 *   "@type": "Person",
 *   "name": { "@required": true, "@type": "xsd:string", "@minLength": 1 },
 *   "email": { "@pattern": "^[^@]+@[^@]+$" },
 *   "age": { "@type": "xsd:integer", "@minimum": 0, "@maximum": 150 },
 * };
 *
 * const result = validateNode(personNode, shape);
 * if (!result.valid) {
 *   console.log(result.errors);
 * }
 * ```
 */
export function validateNode(
  node: any,
  shape: ShapeDefinition
): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationWarning[] = [];

  if (node == null || typeof node !== 'object') {
    errors.push({
      path: '.',
      constraint: 'type',
      message: 'Node must be an object',
    });
    return { valid: false, errors, warnings };
  }

  // Type check
  if (shape['@type']) {
    const nodeTypes = getNodeTypes(node);
    if (!nodeTypes.includes(shape['@type'])) {
      errors.push({
        path: '@type',
        constraint: 'type',
        message: `Expected type "${shape['@type']}", found: [${nodeTypes.join(', ')}]`,
        value: nodeTypes,
      });
    }
  }

  // Property constraints
  for (const [prop, constraint] of Object.entries(shape)) {
    if (prop.startsWith('@')) continue; // Skip JSON-LD keywords in shape

    if (typeof constraint !== 'object' || constraint == null) continue;
    const propShape = constraint as PropertyShape;

    const value = node[prop];
    const rawValue = extractRawValue(value);

    // Required check
    if (propShape['@required'] && (rawValue === undefined || rawValue === null)) {
      errors.push({
        path: prop,
        constraint: 'required',
        message: `Property "${prop}" is required`,
      });
      continue; // Skip further checks if missing
    }

    // Skip further validation if value is absent and not required
    if (rawValue === undefined || rawValue === null) continue;

    // Type check
    if (propShape['@type']) {
      const typeError = validateType(rawValue, propShape['@type']);
      if (typeError) {
        errors.push({
          path: prop,
          constraint: 'type',
          message: typeError,
          value: rawValue,
        });
      }
    }

    // Numeric constraints
    if (propShape['@minimum'] !== undefined && typeof rawValue === 'number') {
      if (rawValue < propShape['@minimum']) {
        errors.push({
          path: prop,
          constraint: 'minimum',
          message: `Value ${rawValue} is below minimum ${propShape['@minimum']}`,
          value: rawValue,
        });
      }
    }

    if (propShape['@maximum'] !== undefined && typeof rawValue === 'number') {
      if (rawValue > propShape['@maximum']) {
        errors.push({
          path: prop,
          constraint: 'maximum',
          message: `Value ${rawValue} exceeds maximum ${propShape['@maximum']}`,
          value: rawValue,
        });
      }
    }

    // String length constraints
    if (propShape['@minLength'] !== undefined && typeof rawValue === 'string') {
      if (rawValue.length < propShape['@minLength']) {
        errors.push({
          path: prop,
          constraint: 'minLength',
          message: `String length ${rawValue.length} is below minimum ${propShape['@minLength']}`,
          value: rawValue,
        });
      }
    }

    if (propShape['@maxLength'] !== undefined && typeof rawValue === 'string') {
      if (rawValue.length > propShape['@maxLength']) {
        errors.push({
          path: prop,
          constraint: 'maxLength',
          message: `String length ${rawValue.length} exceeds maximum ${propShape['@maxLength']}`,
          value: rawValue,
        });
      }
    }

    // Pattern constraint
    if (propShape['@pattern'] && typeof rawValue === 'string') {
      const regex = new RegExp(propShape['@pattern']);
      if (!regex.test(rawValue)) {
        errors.push({
          path: prop,
          constraint: 'pattern',
          message: `Value "${rawValue}" does not match pattern "${propShape['@pattern']}"`,
          value: rawValue,
        });
      }
    }
  }

  return { valid: errors.length === 0, errors, warnings };
}

/**
 * Validates an entire JSON-LD document against embedded @shape definitions.
 * Shapes can be defined in the document's @context or alongside nodes.
 */
export function validateDocument(
  doc: any,
  shapes: ShapeDefinition[]
): ValidationResult {
  const allErrors: ValidationError[] = [];
  const allWarnings: ValidationWarning[] = [];

  const nodes = extractNodes(doc);

  for (const node of nodes) {
    const nodeTypes = getNodeTypes(node);

    for (const shape of shapes) {
      if (nodeTypes.includes(shape['@type'])) {
        const result = validateNode(node, shape);
        allErrors.push(...result.errors.map(e => ({
          ...e,
          path: `${node['@id'] ?? 'anonymous'}/${e.path}`,
        })));
        allWarnings.push(...result.warnings);
      }
    }
  }

  return {
    valid: allErrors.length === 0,
    errors: allErrors,
    warnings: allWarnings,
  };
}

// ── Internal Helpers ──────────────────────────────────────────────

function getNodeTypes(node: any): string[] {
  const type = node['@type'];
  if (!type) return [];
  return Array.isArray(type) ? type : [type];
}

function extractRawValue(value: any): any {
  if (value == null) return undefined;
  // JSON-LD expanded value object
  if (typeof value === 'object' && '@value' in value) {
    return value['@value'];
  }
  // Array of values — take first
  if (Array.isArray(value) && value.length > 0) {
    return extractRawValue(value[0]);
  }
  return value;
}

function extractNodes(doc: any): any[] {
  if (Array.isArray(doc)) {
    return doc.flatMap(extractNodes);
  }
  if (doc == null || typeof doc !== 'object') return [];

  const nodes: any[] = [];
  if (doc['@type']) {
    nodes.push(doc);
  }
  if (doc['@graph']) {
    nodes.push(...extractNodes(doc['@graph']));
  }
  return nodes;
}

function validateType(value: any, expectedType: string): string | null {
  const xsdPrefix = 'http://www.w3.org/2001/XMLSchema#';
  const type = expectedType.startsWith('xsd:')
    ? expectedType.replace('xsd:', xsdPrefix)
    : expectedType;

  switch (type) {
    case `${xsdPrefix}string`:
      if (typeof value !== 'string') {
        return `Expected string, got ${typeof value}`;
      }
      break;
    case `${xsdPrefix}integer`:
      if (typeof value !== 'number' || !Number.isInteger(value)) {
        return `Expected integer, got ${typeof value === 'number' ? value : typeof value}`;
      }
      break;
    case `${xsdPrefix}double`:
    case `${xsdPrefix}float`:
    case `${xsdPrefix}decimal`:
      if (typeof value !== 'number') {
        return `Expected number, got ${typeof value}`;
      }
      break;
    case `${xsdPrefix}boolean`:
      if (typeof value !== 'boolean') {
        return `Expected boolean, got ${typeof value}`;
      }
      break;
    case `${xsdPrefix}dateTime`:
      if (typeof value !== 'string' || isNaN(Date.parse(value))) {
        return `Expected ISO 8601 dateTime, got "${value}"`;
      }
      break;
  }
  return null;
}
