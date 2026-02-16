# Interoperability Guide

**Module:** `jsonld_ex.owl_interop`, `jsonld_ex.dpv_interop`, `jsonld_ex.dataset`

## Design Philosophy

jsonld-ex does not replace existing W3C standards — it bridges them. Every interop mapping is bidirectional where possible, enabling round-trip fidelity while providing the compact inline syntax that JSON-LD developers expect.

## Standards Coverage

| Standard | Direction | Functions | Typical Reduction |
|----------|-----------|-----------|-------------------|
| **PROV-O** | ↔ Bidirectional | `to_prov_o`, `from_prov_o`, `compare_with_prov_o` | 60–75% fewer triples |
| **SHACL** | ↔ Bidirectional | `shape_to_shacl`, `shacl_to_shape`, `compare_with_shacl` | 50–70% fewer triples |
| **OWL** | ↔ Bidirectional | `shape_to_owl_restrictions`, `owl_to_shape` | — |
| **RDF-Star** | ↔ Bidirectional | `to_rdf_star_ntriples`, `from_rdf_star_ntriples`, `to_rdf_star_turtle` | — |
| **SSN/SOSA** | ↔ Bidirectional | `to_ssn`, `from_ssn` | — |
| **Croissant** | ↔ Bidirectional | `to_croissant`, `from_croissant` | — |
| **DPV v2.2** | ↔ Bidirectional | `to_dpv`, `from_dpv`, `compare_with_dpv` | — |

## PROV-O (W3C Provenance Ontology)

jsonld-ex inline annotations (`@confidence`, `@source`, `@method`, `@extractedAt`) map to PROV-O's Entity/Activity/Agent graph model. The inline syntax achieves the same semantic expressiveness with significantly fewer triples.

```python
from jsonld_ex import to_prov_o, from_prov_o, compare_with_prov_o

doc = {
    "@type": "Person",
    "name": {
        "@value": "Alice",
        "@confidence": 0.95,
        "@source": "https://model.example.org/v2",
        "@method": "NER",
        "@extractedAt": "2026-01-15T10:00:00Z",
    },
}

# Convert to full PROV-O graph
prov_doc, report = to_prov_o(doc)
# Creates prov:Entity, prov:Activity, prov:Agent nodes
# with prov:wasGeneratedBy, prov:wasAssociatedWith, etc.

# Round-trip back to inline annotations
recovered, report = from_prov_o(prov_doc)
# recovered["name"]["@confidence"] == 0.95

# Measure verbosity difference
comparison = compare_with_prov_o(doc)
# comparison.jsonld_ex_triples → e.g. 6
# comparison.alternative_triples → e.g. 18
# comparison.triple_reduction_pct → e.g. 66.7
```

### Mapping Table

| jsonld-ex | PROV-O |
|-----------|--------|
| `@source` | `prov:Agent` + `prov:wasAssociatedWith` |
| `@extractedAt` | `prov:Activity` + `prov:endedAtTime` |
| `@method` | `prov:Activity` type annotation |
| `@confidence` | Custom `prov:value` on Entity |
| `@humanVerified` | `prov:Agent` with human type |

## SHACL (Shapes Constraint Language)

jsonld-ex `@shape` definitions map bidirectionally to SHACL property shapes. The inline syntax avoids the verbose RDF graph structure that SHACL requires.

```python
from jsonld_ex import shape_to_shacl, shacl_to_shape, compare_with_shacl

shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string", "@minLength": 1},
    "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
    "email": {"@pattern": "^[^@]+@[^@]+$"},
}

# Convert to SHACL
shacl_doc = shape_to_shacl(shape, target_class="http://schema.org/Person")

# Round-trip back
recovered, warnings = shacl_to_shape(shacl_doc)

# Verbosity comparison
comparison = compare_with_shacl(shape)
```

### Constraint Mapping

| jsonld-ex | SHACL |
|-----------|-------|
| `@required: true` | `sh:minCount 1` |
| `@type` | `sh:datatype` |
| `@minimum` | `sh:minInclusive` |
| `@maximum` | `sh:maxInclusive` |
| `@minLength` | `sh:minLength` |
| `@maxLength` | `sh:maxLength` |
| `@pattern` | `sh:pattern` |
| `@minCount` | `sh:minCount` |
| `@maxCount` | `sh:maxCount` |
| `@in` | `sh:in` |
| `@and` | `sh:and` |
| `@or` | `sh:or` |
| `@not` | `sh:not` |

## OWL (Web Ontology Language)

Shapes can be exported as OWL class restrictions for integration with OWL reasoners.

```python
from jsonld_ex import shape_to_owl_restrictions, owl_to_shape

owl_doc = shape_to_owl_restrictions(shape, class_iri="http://example.org/Person")
recovered = owl_to_shape(owl_doc)
```

## RDF-Star

Annotations can be exported as RDF-Star triples, where metadata is expressed as annotations on the base triple using `<< >>` syntax.

```python
from jsonld_ex import to_rdf_star_ntriples, from_rdf_star_ntriples

ntriples, report = to_rdf_star_ntriples(doc, base_subject="http://example.org/alice")
# << <http://example.org/alice> <schema:name> "Alice" >> <jex:confidence> "0.95"^^xsd:decimal .

recovered = from_rdf_star_ntriples(ntriples)
```

## SSN/SOSA (Semantic Sensor Network)

IoT sensor observations map to the W3C SSN/SOSA ontology, bridging jsonld-ex annotations with the established sensor web vocabulary.

```python
from jsonld_ex import to_ssn, from_ssn

doc = {
    "@type": "Observation",
    "@id": "obs:temp-001",
    "value": {
        "@value": 23.5,
        "@confidence": 0.92,
        "@source": "sensor:dht22-01",
        "@extractedAt": "2026-01-15T10:00:00Z",
    },
}

ssn_doc = to_ssn(doc)
# Maps to sosa:Observation, sosa:hasResult, sosa:madeBySensor, etc.

recovered = from_ssn(ssn_doc)
```

## Croissant (ML Dataset Metadata)

Dataset metadata interoperates with the Croissant format used by HuggingFace, Google Dataset Search, and other ML platforms.

```python
from jsonld_ex import create_dataset_metadata, to_croissant, from_croissant

metadata = create_dataset_metadata(
    name="NER-2026",
    description="Named entity recognition dataset with confidence annotations",
    license="MIT",
    version="1.0.0",
)

croissant_doc = to_croissant(metadata)
recovered = from_croissant(croissant_doc)
```

## DPV v2.2 (Data Privacy Vocabulary)

Data protection annotations interoperate with the W3C Data Privacy Vocabulary, enabling compliance metadata exchange with DPV-consuming systems.

```python
from jsonld_ex import to_dpv, from_dpv, compare_with_dpv

doc = {
    "@type": "Person",
    "name": {
        "@value": "John Doe",
        "@personalDataCategory": "regular",
        "@legalBasis": "consent",
        "@jurisdiction": "EU",
    },
}

dpv_doc = to_dpv(doc)
recovered = from_dpv(dpv_doc)
comparison = compare_with_dpv(doc)
```

## Round-Trip Fidelity

All bidirectional mappings are tested for round-trip fidelity: converting to the target standard and back should preserve all semantic information. Where lossless round-trip is not possible (e.g., OWL has no direct equivalent for `@confidence`), the mapping documents what is preserved and what requires convention.
