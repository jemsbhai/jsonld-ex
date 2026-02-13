
/**
 * Dataset Metadata Extensions for JSON-LD (GAP-D1, D2, D3)
 *
 * Provides dataset-level metadata compatible with schema.org/Dataset
 * and bidirectional interoperability with Croissant (MLCommons).
 */

import {
    Dataset,
    FileObject,
    FileSet,
    RecordSet,
    Field,
    JsonLdNode
} from './types.js';

// ── Namespace constants ─────────────────────────────────────────────

export const SCHEMA_ORG = "https://schema.org/";
export const CROISSANT_NS = "http://mlcommons.org/croissant/";
export const CROISSANT_SPEC_VERSION = "http://mlcommons.org/croissant/1.0";
export const DCT_NS = "http://purl.org/dc/terms/";

// ── Contexts ────────────────────────────────────────────────────────

const DATASET_CONTEXT = {
    "@vocab": SCHEMA_ORG,
    "sc": SCHEMA_ORG,
    "cr": CROISSANT_NS,
    "dct": DCT_NS,
    "citeAs": "cr:citeAs",
    "conformsTo": "dct:conformsTo",
    "recordSet": "cr:recordSet",
    "field": "cr:field",
    "dataType": { "@id": "cr:dataType", "@type": "@vocab" },
    "source": "cr:source",
    "extract": "cr:extract",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "isLiveDataset": "cr:isLiveDataset",
    "@language": "en",
};

const CROISSANT_CONTEXT = {
    "@language": "en",
    "@vocab": SCHEMA_ORG,
    "sc": SCHEMA_ORG,
    "cr": CROISSANT_NS,
    "dct": DCT_NS,
    "rai": "http://mlcommons.org/croissant/RAI/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "data": { "@id": "cr:data", "@type": "@json" },
    "dataType": { "@id": "cr:dataType", "@type": "@vocab" },
    "examples": { "@id": "cr:examples", "@type": "@json" },
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
};

// ── Validation shape ────────────────────────────────────────────────

export const DATASET_SHAPE = {
    "@type": "sc:Dataset",
    "name": {
        "@required": true,
        "@type": "xsd:string",
        "@minLength": 1,
    },
};

// ── GAP-D1: Dataset Metadata ────────────────────────────────────────

export interface DatasetOptions {
    description?: string;
    version?: string;
    license?: string;
    url?: string;
    datePublished?: string;
    creator?: string | any | any[];
    keywords?: string | string[];
    citation?: string;
    publisher?: string | any;
    inLanguage?: string | string[];
    sameAs?: string | string[];
    dateCreated?: string;
    dateModified?: string;
    isLive?: boolean;
}

/**
 * Create a JSON-LD Dataset metadata document.
 */
export function createDatasetMetadata(
    name: string,
    options: DatasetOptions = {}
): Dataset {
    if (!name || typeof name !== 'string') {
        throw new Error("Dataset name must be a non-empty string");
    }

    const doc: Dataset = {
        "@context": JSON.parse(JSON.stringify(DATASET_CONTEXT)),
        "@type": "sc:Dataset",
        "name": name,
    };

    if (options.description) doc.description = options.description;
    if (options.version) doc.version = options.version;
    if (options.license) doc.license = options.license;
    if (options.url) doc.url = options.url;
    if (options.datePublished) doc.datePublished = options.datePublished;
    if (options.creator) doc.creator = normalizeCreator(options.creator);
    if (options.keywords) doc.keywords = options.keywords;
    if (options.citation) doc.citeAs = options.citation;
    if (options.publisher) doc.publisher = normalizePublisher(options.publisher);
    if (options.inLanguage) doc.inLanguage = options.inLanguage;
    if (options.sameAs) doc.sameAs = options.sameAs;
    if (options.dateCreated) doc.dateCreated = options.dateCreated;
    if (options.dateModified) doc.dateModified = options.dateModified;
    if (options.isLive !== undefined) doc.isLiveDataset = options.isLive;

    // Initialize containers
    doc.distribution = [];
    doc.recordSet = [];

    return doc;
}

// ── GAP-D2: Distributions and Structure ─────────────────────────────

export interface DistributionOptions {
    sha256?: string;
    contentSize?: string;
    description?: string;
    fileId?: string;
}

export function addDistribution(
    dataset: Dataset,
    name: string,
    contentUrl: string,
    encodingFormat: string,
    options: DistributionOptions = {}
): Dataset {
    if (!name) throw new Error("Distribution name must be non-empty");

    const ds = JSON.parse(JSON.stringify(dataset)) as Dataset;

    const fo: FileObject = {
        "@type": "cr:FileObject",
        "@id": options.fileId || name,
        "name": name,
        "contentUrl": contentUrl,
        "encodingFormat": encodingFormat,
    };

    if (options.sha256) fo.sha256 = options.sha256;
    if (options.contentSize) fo.contentSize = options.contentSize;
    if (options.description) fo.description = options.description;

    if (!ds.distribution) ds.distribution = [];
    ds.distribution.push(fo);

    return ds;
}

export interface FileSetOptions {
    description?: string;
    fileSetId?: string;
}

export function addFileSet(
    dataset: Dataset,
    name: string,
    containedIn: string,
    encodingFormat: string,
    includes: string,
    options: FileSetOptions = {}
): Dataset {
    const ds = JSON.parse(JSON.stringify(dataset)) as Dataset;

    const fs: FileSet = {
        "@type": "cr:FileSet",
        "@id": options.fileSetId || name,
        "name": name,
        "containedIn": { "@id": containedIn },
        "encodingFormat": encodingFormat,
        "includes": includes
    };

    if (options.description) fs.description = options.description;

    if (!ds.distribution) ds.distribution = [];
    ds.distribution.push(fs);

    return ds;
}

export interface FieldOptions {
    description?: string;
    source?: any;
}

export function createField(
    name: string,
    dataType: string,
    options: FieldOptions = {}
): Field {
    const f: Field = {
        "@type": "cr:Field",
        "name": name,
        "dataType": dataType
    };
    if (options.description) f.description = options.description;
    if (options.source) f.source = options.source;
    return f;
}

export interface RecordSetOptions {
    description?: string;
    recordSetId?: string;
}

export function addRecordSet(
    dataset: Dataset,
    name: string,
    fields: Field[],
    options: RecordSetOptions = {}
): Dataset {
    const ds = JSON.parse(JSON.stringify(dataset)) as Dataset;
    const rsId = options.recordSetId || name;

    const prefixedFields = fields.map(f => {
        const fieldCopy = JSON.parse(JSON.stringify(f));
        fieldCopy['@id'] = `${rsId}/${fieldCopy.name}`;
        return fieldCopy;
    });

    const rs: RecordSet = {
        "@type": "cr:RecordSet",
        "@id": rsId,
        "name": name,
        "field": prefixedFields
    };

    if (options.description) rs.description = options.description;

    if (!ds.recordSet) ds.recordSet = [];
    ds.recordSet.push(rs);

    return ds;
}

// ── GAP-D3: Croissant Interoperability ──────────────────────────────

export function toCroissant(dataset: Dataset): Dataset {
    const doc = JSON.parse(JSON.stringify(dataset));
    doc['@context'] = JSON.parse(JSON.stringify(CROISSANT_CONTEXT));
    doc['conformsTo'] = CROISSANT_SPEC_VERSION;
    return doc;
}

export function fromCroissant(croissantDoc: Dataset): Dataset {
    const doc = JSON.parse(JSON.stringify(croissantDoc));
    doc['@context'] = JSON.parse(JSON.stringify(DATASET_CONTEXT));
    delete doc['conformsTo'];
    return doc;
}

// ── Internal Helpers ──────────────────────────────────────────────

function normalizeCreator(creator: any): any {
    if (Array.isArray(creator)) {
        return creator.map(normalizeSingleCreator);
    }
    return normalizeSingleCreator(creator);
}

function normalizeSingleCreator(creator: any): any {
    if (typeof creator === 'string') {
        return { "@type": "Person", "name": creator };
    }
    return creator;
}

function normalizePublisher(publisher: any): any {
    if (typeof publisher === 'string') {
        return { "@type": "Organization", "name": publisher };
    }
    return publisher;
}
