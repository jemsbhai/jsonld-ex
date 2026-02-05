import { validateVector, cosineSimilarity, vectorTermDefinition } from '../src/extensions/vector';

describe('Vector Extensions', () => {
  describe('vectorTermDefinition()', () => {
    it('creates a term definition without dimensions', () => {
      const def = vectorTermDefinition('embedding', 'http://example.org/embedding');
      expect(def.embedding['@container']).toBe('@vector');
      expect(def.embedding['@id']).toBe('http://example.org/embedding');
      expect(def.embedding['@dimensions']).toBeUndefined();
    });

    it('creates a term definition with dimensions', () => {
      const def = vectorTermDefinition('embedding', 'http://example.org/embedding', 768);
      expect(def.embedding['@dimensions']).toBe(768);
    });

    it('rejects invalid dimensions', () => {
      expect(() => vectorTermDefinition('e', 'http://x.org/e', 0)).toThrow(RangeError);
      expect(() => vectorTermDefinition('e', 'http://x.org/e', -1)).toThrow(RangeError);
      expect(() => vectorTermDefinition('e', 'http://x.org/e', 1.5)).toThrow(RangeError);
    });
  });

  describe('validateVector()', () => {
    it('accepts valid vector', () => {
      const result = validateVector([0.1, -0.2, 0.3]);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('rejects non-array', () => {
      const result = validateVector('not a vector');
      expect(result.valid).toBe(false);
    });

    it('rejects empty array', () => {
      const result = validateVector([]);
      expect(result.valid).toBe(false);
    });

    it('rejects non-finite numbers', () => {
      const result = validateVector([1.0, NaN, 3.0]);
      expect(result.valid).toBe(false);
      expect(result.errors[0]).toContain('finite number');
    });

    it('validates dimension count', () => {
      const result = validateVector([0.1, 0.2, 0.3], 5);
      expect(result.valid).toBe(false);
      expect(result.errors[0]).toContain('dimension mismatch');
    });

    it('passes correct dimension count', () => {
      const result = validateVector([0.1, 0.2, 0.3], 3);
      expect(result.valid).toBe(true);
    });
  });

  describe('cosineSimilarity()', () => {
    it('computes similarity of identical vectors', () => {
      expect(cosineSimilarity([1, 0, 0], [1, 0, 0])).toBeCloseTo(1.0);
    });

    it('computes similarity of orthogonal vectors', () => {
      expect(cosineSimilarity([1, 0, 0], [0, 1, 0])).toBeCloseTo(0.0);
    });

    it('computes similarity of opposite vectors', () => {
      expect(cosineSimilarity([1, 0], [-1, 0])).toBeCloseTo(-1.0);
    });

    it('throws on dimension mismatch', () => {
      expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow();
    });
  });
});
