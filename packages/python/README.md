# jsonld-ex

**JSON-LD 1.2 Extensions for AI/ML Data Exchange, Security, and Validation**

Reference implementation of proposed JSON-LD 1.2 extensions. Wraps [PyLD](https://github.com/digitalbazaar/pyld) for core processing and adds extension layers.

## Install

```bash
pip install jsonld-ex
```

## Quick Start

```python
from jsonld_ex import JsonLdEx, annotate

# Annotate a value with AI/ML provenance
name = annotate(
    "John Smith",
    confidence=0.95,
    source="https://ml-model.example.org/ner-v2",
    method="NER",
)
# {'@value': 'John Smith', '@confidence': 0.95, '@source': '...', '@method': 'NER'}

# Validate against a shape
from jsonld_ex import validate_node

shape = {
    "@type": "Person",
    "name": {"@required": True, "@type": "xsd:string"},
    "age": {"@type": "xsd:integer", "@minimum": 0, "@maximum": 150},
}

result = validate_node({"@type": "Person", "name": "John", "age": 30}, shape)
assert result.valid
```

## Features

- **AI/ML Extensions**: `@confidence`, `@source`, `@extractedAt`, `@method`, `@humanVerified`
- **Vector Embeddings**: `@vector` container type with dimension validation
- **Security**: `@integrity` context verification, allowlists, resource limits
- **Validation**: `@shape` native validation framework

## Documentation

Full documentation and specifications: [github.com/jemsbhai/jsonld-ex](https://github.com/jemsbhai/jsonld-ex)

## License

MIT
