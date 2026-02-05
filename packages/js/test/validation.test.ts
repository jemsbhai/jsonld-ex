import { validateNode, validateDocument } from '../src/extensions/validation';

describe('Validation Extensions (@shape)', () => {
  const personShape = {
    '@type': 'Person',
    'name': { '@required': true, '@type': 'xsd:string', '@minLength': 1 },
    'email': { '@pattern': '^[^@]+@[^@]+$' },
    'age': { '@type': 'xsd:integer', '@minimum': 0, '@maximum': 150 },
  };

  describe('validateNode()', () => {
    it('passes valid node', () => {
      const node = {
        '@type': 'Person',
        'name': 'John Smith',
        'email': 'john@example.com',
        'age': 30,
      };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('fails on missing required property', () => {
      const node = { '@type': 'Person', 'email': 'john@example.com' };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.constraint === 'required')).toBe(true);
    });

    it('fails on type mismatch', () => {
      const node = { '@type': 'Person', 'name': 12345 };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.constraint === 'type')).toBe(true);
    });

    it('fails on value below minimum', () => {
      const node = { '@type': 'Person', 'name': 'Test', 'age': -5 };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.constraint === 'minimum')).toBe(true);
    });

    it('fails on value above maximum', () => {
      const node = { '@type': 'Person', 'name': 'Test', 'age': 200 };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.constraint === 'maximum')).toBe(true);
    });

    it('fails on pattern mismatch', () => {
      const node = { '@type': 'Person', 'name': 'Test', 'email': 'not-an-email' };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.constraint === 'pattern')).toBe(true);
    });

    it('fails on wrong node type', () => {
      const node = { '@type': 'Organization', 'name': 'Acme' };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(false);
      expect(result.errors.some(e => e.path === '@type')).toBe(true);
    });

    it('skips validation for absent optional properties', () => {
      const node = { '@type': 'Person', 'name': 'Test' };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(true);
    });

    it('handles @value wrapped values', () => {
      const node = {
        '@type': 'Person',
        'name': { '@value': 'Test' },
        'age': { '@value': 25 },
      };
      const result = validateNode(node, personShape);
      expect(result.valid).toBe(true);
    });
  });

  describe('validateDocument()', () => {
    it('validates all matching nodes in a graph', () => {
      const doc = {
        '@graph': [
          { '@type': 'Person', 'name': 'Alice', 'age': 30 },
          { '@type': 'Person', 'age': 25 },  // missing required name
          { '@type': 'Organization', 'name': 'Acme' },  // not matched
        ],
      };
      const result = validateDocument(doc, [personShape]);
      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(1); // Only the Person missing name
    });
  });
});
