import {
  annotate,
  getConfidence,
  getProvenance,
  filterByConfidence,
  aggregateConfidence,
} from '../src/extensions/ai-ml';

describe('AI/ML Extensions', () => {
  describe('annotate()', () => {
    it('creates an annotated value with confidence', () => {
      const result = annotate('John Smith', { confidence: 0.95 });
      expect(result).toEqual({
        '@value': 'John Smith',
        '@confidence': 0.95,
      });
    });

    it('creates an annotated value with full provenance', () => {
      const result = annotate('John Smith', {
        confidence: 0.95,
        source: 'https://model.example.org/ner-v2',
        extractedAt: '2026-01-15T10:30:00Z',
        method: 'NER',
        humanVerified: false,
      });

      expect(result['@value']).toBe('John Smith');
      expect(result['@confidence']).toBe(0.95);
      expect(result['@source']).toBe('https://model.example.org/ner-v2');
      expect(result['@extractedAt']).toBe('2026-01-15T10:30:00Z');
      expect(result['@method']).toBe('NER');
      expect(result['@humanVerified']).toBe(false);
    });

    it('rejects invalid confidence scores', () => {
      expect(() => annotate('x', { confidence: 1.5 })).toThrow(RangeError);
      expect(() => annotate('x', { confidence: -0.1 })).toThrow(RangeError);
    });

    it('works with numeric values', () => {
      const result = annotate(42, { confidence: 0.8 });
      expect(result['@value']).toBe(42);
    });

    it('works with boolean values', () => {
      const result = annotate(true, { confidence: 1.0 });
      expect(result['@value']).toBe(true);
    });
  });

  describe('getConfidence()', () => {
    it('extracts confidence from compact form', () => {
      const node = { '@value': 'test', '@confidence': 0.9 };
      expect(getConfidence(node)).toBe(0.9);
    });

    it('extracts confidence from expanded form', () => {
      const node = {
        'http://www.w3.org/ns/jsonld-ex/confidence': [{ '@value': 0.85 }],
      };
      expect(getConfidence(node)).toBe(0.85);
    });

    it('returns undefined for missing confidence', () => {
      expect(getConfidence({ '@value': 'test' })).toBeUndefined();
      expect(getConfidence(null)).toBeUndefined();
      expect(getConfidence(undefined)).toBeUndefined();
    });
  });

  describe('getProvenance()', () => {
    it('extracts all provenance fields', () => {
      const node = {
        '@value': 'test',
        '@confidence': 0.9,
        '@source': 'https://model.example.org/v1',
        '@method': 'NER',
      };
      const prov = getProvenance(node);
      expect(prov.confidence).toBe(0.9);
      expect(prov.source).toBe('https://model.example.org/v1');
      expect(prov.method).toBe('NER');
    });

    it('returns empty object for node without provenance', () => {
      const prov = getProvenance({ '@value': 'test' });
      expect(Object.keys(prov)).toHaveLength(0);
    });
  });

  describe('filterByConfidence()', () => {
    const graph = [
      { '@id': '#a', name: { '@value': 'Alice', '@confidence': 0.95 } },
      { '@id': '#b', name: { '@value': 'Bob', '@confidence': 0.6 } },
      { '@id': '#c', name: { '@value': 'Charlie', '@confidence': 0.3 } },
      { '@id': '#d', description: 'no confidence on name' },
    ];

    it('filters nodes above threshold', () => {
      const result = filterByConfidence(graph, 'name', 0.5);
      expect(result).toHaveLength(2);
      expect(result[0]['@id']).toBe('#a');
      expect(result[1]['@id']).toBe('#b');
    });

    it('returns empty for high threshold', () => {
      const result = filterByConfidence(graph, 'name', 0.99);
      expect(result).toHaveLength(0);
    });

    it('rejects invalid threshold', () => {
      expect(() => filterByConfidence(graph, 'name', 2.0)).toThrow(RangeError);
    });
  });

  describe('aggregateConfidence()', () => {
    it('computes mean', () => {
      expect(aggregateConfidence([0.8, 0.6, 0.4], 'mean')).toBeCloseTo(0.6);
    });

    it('computes max', () => {
      expect(aggregateConfidence([0.8, 0.6, 0.4], 'max')).toBe(0.8);
    });

    it('computes min', () => {
      expect(aggregateConfidence([0.8, 0.6, 0.4], 'min')).toBe(0.4);
    });

    it('computes weighted average', () => {
      const result = aggregateConfidence([0.9, 0.5], 'weighted', [3, 1]);
      expect(result).toBeCloseTo(0.8);
    });

    it('returns 0 for empty array', () => {
      expect(aggregateConfidence([])).toBe(0);
    });
  });
});
