

import {
    toProvO,
    fromProvO,
    shapeToShacl,
    shaclToShape,
    shapeToOwlRestrictions,
    owlToShape
} from '../owl.js';
import {
    PROV,
    SHACL,
    OWL,
    XSD,
    RDFS,
    JSONLD_EX
} from '../owl.js';
import { JsonLdNode } from '../types.js';

describe('OWL Interoperability (src/owl.ts)', () => {

    // ── PROV-O Tests ──────────────────────────────────────────────────────────

    describe('PROV-O Mapping', () => {
        const annotatedPerson = {
            "@context": "http://schema.org/",
            "@type": "Person",
            "@id": "http://example.org/alice",
            "name": {
                "@value": "Alice Smith",
                "@confidence": 0.95,
                "@source": "https://models.example.org/gpt4",
                "@extractedAt": "2025-01-15T10:30:00Z",
                "@method": "NER"
            }
        };

        it('should convert annotated document to PROV-O graph', () => {
            const { doc: provDoc, report } = toProvO(annotatedPerson);

            expect(report.success).toBe(true);
            expect(provDoc['@graph']).toBeDefined();
            const graph = provDoc['@graph'] as JsonLdNode[];

            // Check for Entity creation
            const entity = graph.find(n => n['@type'] === `${PROV}Entity`);
            expect(entity).toBeDefined();
            expect(entity![`${PROV}value`]).toBe("Alice Smith");
            expect(entity![`${JSONLD_EX}confidence`]).toBe(0.95);

            // Check for SoftwareAgent
            const agent = graph.find(n => n['@type'] === `${PROV}SoftwareAgent`);
            expect(agent).toBeDefined();
            expect(agent!['@id']).toBe("https://models.example.org/gpt4");

            // Check relationship
            expect(entity![`${PROV}wasAttributedTo`]).toEqual({ '@id': "https://models.example.org/gpt4" });

            // Check Activity (Method)
            const activity = graph.find(n => n['@type'] === `${PROV}Activity`);
            expect(activity).toBeDefined();
            expect(activity![`${RDFS}label`]).toBe("NER");
        });

        it('should round-trip PROV-O back to jsonld-ex', () => {
            const { doc: provDoc } = toProvO(annotatedPerson);
            const { doc: restored, report } = fromProvO(provDoc);

            expect(report.success).toBe(true);
            expect(restored['name']['@value']).toBe("Alice Smith");
            expect(restored['name']['@confidence']).toBe(0.95);
            expect(restored['name']['@source']).toBe("https://models.example.org/gpt4");
            expect(restored['name']['@extractedAt']).toBe("2025-01-15T10:30:00Z");
            expect(restored['name']['@method']).toBe("NER");
        });
    });

    // ── SHACL Tests ───────────────────────────────────────────────────────────

    describe('SHACL Mapping', () => {
        const personShape = {
            "@type": "http://schema.org/Person",
            "http://schema.org/name": {
                "@required": true,
                "@type": "xsd:string",
                "@minLength": 1
            },
            "http://schema.org/email": {
                "@pattern": "^[^@]+@[^@]+$"
            },
            "http://schema.org/age": {
                "@type": "xsd:integer",
                "@minimum": 0,
                "@maximum": 150
            }
        };

        it('should convert @shape to SHACL graph', () => {
            const shacl = shapeToShacl(personShape);
            expect(shacl['@graph']).toBeDefined();
            const graph = shacl['@graph'] as JsonLdNode[];
            expect(graph.length).toBeGreaterThan(0);

            const nodeShape = graph.find(n => n['@type'] === `${SHACL}NodeShape`);
            expect(nodeShape).toBeDefined();
            expect(nodeShape![`${SHACL}targetClass`]['@id']).toBe("http://schema.org/Person");

            const properties = nodeShape![`${SHACL}property`] as JsonLdNode[];

            // Check name property
            const nameProp = properties.find(p => p[`${SHACL}path`]['@id'] === "http://schema.org/name");
            expect(nameProp).toBeDefined();
            expect(nameProp![`${SHACL}minCount`]).toBe(1);
            expect(nameProp![`${SHACL}datatype`]['@id']).toBe(`${XSD}string`);
            expect(nameProp![`${SHACL}minLength`]).toBe(1);

            // Check age property
            const ageProp = properties.find(p => p[`${SHACL}path`]['@id'] === "http://schema.org/age");
            expect(ageProp).toBeDefined();
            expect(ageProp![`${SHACL}minInclusive`]).toBe(0);
            expect(ageProp![`${SHACL}maxInclusive`]).toBe(150);
        });

        it('should round-trip SHACL back to @shape', () => {
            const shacl = shapeToShacl(personShape);
            const { shape: restored, warnings } = shaclToShape(shacl);

            expect(warnings).toEqual([]);
            expect(restored['@type']).toBe("http://schema.org/Person");

            const name = restored["http://schema.org/name"];
            expect(name['@required']).toBe(true);
            // xsd:string might be lost if default or mapped, checking if present
            // In our implementation we kept xsd:string explicit
            expect(name['@type']).toBe("xsd:string");
            expect(name['@minLength']).toBe(1);

            const age = restored["http://schema.org/age"];


            expect(age['@minimum']).toBe(0);
            expect(age['@maximum']).toBe(150);
            expect(age['@type']).toBe("xsd:integer");
        });

        it('should handle logical operators', () => {
            const logicalShape = {
                "@type": "Thing",
                "prop": {
                    "@or": [
                        { "@type": "xsd:string" },
                        { "@type": "xsd:integer" }
                    ]
                }
            };

            const shacl = shapeToShacl(logicalShape);
            const nodeShape = (shacl['@graph'] as any[])[0];
            const prop = nodeShape[`${SHACL}property`];
            // Since it's a single property, prop is object or array. Implementation pushes to array.
            // Wait, shapeToShacl pushes to properties array. So it is array[0]
            const p = prop[0];

            expect(p[`${SHACL}or`]).toBeDefined();
            expect(p[`${SHACL}or`]['@list']).toHaveLength(2);

            const { shape: restored } = shaclToShape(shacl);
            expect(restored['prop']['@or']).toBeDefined();
            expect(restored['prop']['@or']).toHaveLength(2);
        });
    });

    // ── OWL Tests ─────────────────────────────────────────────────────────────

    describe('OWL Mapping', () => {
        const personShape = {
            "@type": "http://schema.org/Person",
            "http://schema.org/name": {
                "@required": true,
                "@type": "xsd:string",
                "@minLength": 1
            },
            "http://schema.org/age": {
                "@type": "xsd:integer",
                "@minimum": 0,
                "@maximum": 150
            }
        };

        it('should convert @shape to OWL class restrictions', () => {
            const owl = shapeToOwlRestrictions(personShape);
            const graph = owl['@graph'] as JsonLdNode[];
            const owlClass = graph[0];

            expect(owlClass['@type']).toBe(`${OWL}Class`);
            expect(owlClass['@id']).toBe("http://schema.org/Person");

            const subClassOf = owlClass[`${RDFS}subClassOf`];
            expect(subClassOf).toBeDefined();

            const restrictions = Array.isArray(subClassOf) ? subClassOf : [subClassOf];

            // Check name restriction (minCardinality)
            const nameCard = restrictions.find(r =>
                r[`${OWL}onProperty`]['@id'] === "http://schema.org/name" &&
                r[`${OWL}minCardinality`]
            );
            expect(nameCard).toBeDefined();
            expect(nameCard![`${OWL}minCardinality`]['@value']).toBe(1);

            // Check age restriction (datarange with facets)
            const ageRes = restrictions.find(r =>
                r[`${OWL}onProperty`]['@id'] === "http://schema.org/age" &&
                r[`${OWL}allValuesFrom`] &&
                r[`${OWL}allValuesFrom`][`${OWL}withRestrictions`]
            );
            expect(ageRes).toBeDefined();

            const avf = ageRes![`${OWL}allValuesFrom`];
            expect(avf[`${OWL}onDatatype`]['@id']).toBe(`${XSD}integer`.replace('xsd:', XSD));

            const facets = avf[`${OWL}withRestrictions`]['@list'];
            expect(facets).toBeDefined();
            expect(facets.some((f: any) => f[`${XSD}minInclusive`] === 0)).toBe(true);
            expect(facets.some((f: any) => f[`${XSD}maxInclusive`] === 150)).toBe(true);
        });

        it('should round-trip OWL back to @shape', () => {
            const owl = shapeToOwlRestrictions(personShape);
            const restored = owlToShape(owl);

            expect(restored['@type']).toBe("http://schema.org/Person");

            const name = restored["http://schema.org/name"];
            expect(name['@required']).toBe(true);
            // xsd:string? In implementation we default unmapped to string if text facets present?
            // "name" has minLength=1 so it should map back to xsd:string if not explicit?
            // In shapeToOwlRestrictions we put it in allValuesFrom.
            // In owlToShape we parse allValuesFrom.

            // Let's check specifics:
            // name in input had @type: xsd:string.
            // shapeToOwlRestrictions creates allValuesFrom with onDatatype xsd:string and facets.
            // owlToShape parses allValuesFrom, sees onDatatype, sets @type.
            // It sees withRestrictions (minLength), sets @minLength.
            expect(name['@type']).toBe("xsd:string");
            expect(name['@minLength']).toBe(1);

            const age = restored["http://schema.org/age"];
            expect(age['@minimum']).toBe(0);
            expect(age['@maximum']).toBe(150);
            expect(age['@type']).toBe("xsd:integer");
        });

        it('should handle unmappable annotations via jsonld-ex namespace', () => {
            const shapeWithExtra = {
                "@type": "Thing",
                "prop": {
                    "@type": "xsd:string",
                    "@equals": "otherProp"
                }
            };

            const owl = shapeToOwlRestrictions(shapeWithExtra);
            const owlClass = (owl['@graph'] as JsonLdNode[])[0];

            // Check if custom annotation is attached to restriction
            const subClassOf = owlClass[`${RDFS}subClassOf`];
            const restrictions = Array.isArray(subClassOf) ? subClassOf : [subClassOf];
            const restriction = restrictions.find((r: any) =>
                r[`${OWL}onProperty`]['@id'] === "prop"
            );
            expect(restriction).toBeDefined();
            expect(restriction![`${JSONLD_EX}equals`]).toBe("otherProp");

            const restored = owlToShape(owl);
            expect(restored['@type']).toBe("Thing");
            expect(restored['prop']['@equals']).toBe("otherProp");
        });
    });

});
