
/**
 * MQTT Transport Optimization for JSON-LD-Ex.
 *
 * Optimises jsonld-ex documents for IoT pub/sub via MQTT, with:
 *   - CBOR or JSON payload serialization
 *   - Automatic MQTT topic derivation from `@type` and `@id`
 *   - QoS level mapping from `@confidence`
 *   - MQTT 5.0 PUBLISH property derivation
 */

import { toCbor, fromCbor } from './cbor.js';
import { getConfidence } from './extensions/ai-ml.js';
import { JsonLdNode } from './types.js';

// MQTT spec: topic name MUST NOT exceed 65,535 bytes (UTF-8 encoded).
const MAX_TOPIC_BYTES = 65_535;

/**
 * Serialize a jsonld-ex document for MQTT transmission.
 *
 * When `compress` is true, uses CBOR encoding with context
 * compression for minimal payload size. Falls back to compact
 * JSON if false.
 *
 * @param doc - JSON-LD document with jsonld-ex extensions.
 * @param compress - Use CBOR encoding (true) or JSON (false).
 * @param maxPayload - Maximum payload size in bytes (default: 256KB).
 * @param contextRegistry - Context URL → integer mapping for CBOR compression.
 * @returns Encoded buffer ready for MQTT publish.
 */
export function toMqttPayload(
    doc: JsonLdNode,
    compress: boolean = true,
    maxPayload: number = 256_000,
    contextRegistry?: Record<string, number>
): Buffer {
    let payload: Buffer;

    if (compress) {
        payload = toCbor(doc, contextRegistry);
    } else {
        const jsonStr = JSON.stringify(doc);
        payload = Buffer.from(jsonStr, 'utf-8');
    }

    if (payload.length > maxPayload) {
        throw new Error(
            `Payload size ${payload.length} bytes exceeds max_payload (${maxPayload} bytes)`
        );
    }

    return payload;
}

/**
 * Deserialize an MQTT payload back to a jsonld-ex document.
 *
 * @param payload - Raw bytes received from MQTT.
 * @param context - Optional `@context` to reattach (if it was stripped during serialization).
 * @param compressed - Whether the payload is CBOR (true) or JSON (false).
 * @param contextRegistry - Registry used during serialization, for CBOR context decompression.
 * @returns Restored JSON-LD document.
 */
export function fromMqttPayload(
    payload: Buffer | Uint8Array,
    context?: any,
    compressed: boolean = true,
    contextRegistry?: Record<string, number>
): JsonLdNode {
    let doc: JsonLdNode;

    if (compressed) {
        doc = fromCbor(payload, contextRegistry);
    } else {
        const str = Buffer.isBuffer(payload)
            ? payload.toString('utf-8')
            : new TextDecoder().decode(payload);
        doc = JSON.parse(str);
    }

    if (context !== undefined && doc['@context'] === undefined) {
        doc['@context'] = context;
    }

    return doc;
}

// ── Topic Derivation ──────────────────────────────────────────────

/**
 * Generate an MQTT topic from JSON-LD document metadata.
 *
 * Pattern: `{prefix}/{@type}/{@id_fragment}`
 *
 * @param doc - JSON-LD document or node.
 * @param prefix - Topic prefix (default "ld").
 * @returns MQTT topic string.
 */
export function deriveMqttTopic(
    doc: JsonLdNode,
    prefix: string = 'ld'
): string {
    // Extract type
    let typeVal: any = doc['@type'] || 'unknown';
    if (Array.isArray(typeVal)) {
        typeVal = typeVal[0] || 'unknown';
    }
    const typeStr = toLocalName(String(typeVal));

    // Extract id fragment
    let idVal: any = doc['@id'] || 'unknown';
    const idStr = toLocalName(String(idVal));

    // Sanitise
    const safeType = sanitiseTopicSegment(typeStr);
    const safeId = sanitiseTopicSegment(idStr);

    const topic = `${prefix}/${safeType}/${safeId}`;

    const topicBytes = Buffer.byteLength(topic, 'utf-8');
    if (topicBytes > MAX_TOPIC_BYTES) {
        throw new Error(
            `Generated MQTT topic is ${topicBytes} bytes, exceeding limit of ${MAX_TOPIC_BYTES} bytes.`
        );
    }

    return topic;
}

function toLocalName(iri: string): string {
    if (iri.includes('#')) {
        return iri.split('#').pop()!;
    }
    if (iri.includes('/')) {
        return iri.split('/').pop()!;
    }
    if (iri.includes(':')) {
        return iri.split(':').pop()!;
    }
    return iri;
}

function sanitiseTopicSegment(segment: string): string {
    // MQTT wildcards # and + are forbidden, as is null
    let sanitised = segment.replace(/[#+\x00]/g, '_');
    // Strip leading $ (reserved for broker system topics)
    sanitised = sanitised.replace(/^\$+/, '');

    return sanitised || 'unknown';
}

// ── QoS Derivation ────────────────────────────────────────────────

export interface QosResult {
    qos: 0 | 1 | 2;
    reasoning: string;
    confidenceUsed?: number;
    humanVerified: boolean;
}

/**
 * Map document confidence to MQTT QoS level.
 */
export function deriveMqttQos(doc: JsonLdNode): 0 | 1 | 2 {
    return deriveMqttQosDetailed(doc).qos;
}

/**
 * Map document confidence to MQTT QoS level with reasoning.
 */
export function deriveMqttQosDetailed(doc: JsonLdNode): QosResult {
    let conf = getConfidence(doc);
    const humanVerified = doc['@humanVerified'] === true;
    let source = 'document-level';

    // Document-level humanVerified
    if (humanVerified) {
        return {
            qos: 2,
            reasoning: `@humanVerified=true (${source}) → QoS 2 (exactly once)`,
            confidenceUsed: conf,
            humanVerified: true
        };
    }

    // If no document-level confidence, scan first annotated property
    if (conf === undefined) {
        for (const [key, val] of Object.entries(doc)) {
            if (key.startsWith('@')) continue;

            if (val && typeof val === 'object' && !Array.isArray(val)) {
                const propConf = getConfidence(val);
                const propHv = val['@humanVerified'] === true;

                if (propHv) {
                    return {
                        qos: 2,
                        reasoning: `@humanVerified=true (property '${key}') → QoS 2 (exactly once)`,
                        confidenceUsed: propConf,
                        humanVerified: true
                    };
                }

                if (propConf !== undefined) {
                    conf = propConf;
                    source = `property '${key}'`;
                    break;
                }
            }
        }
    }

    if (conf === undefined) {
        return {
            qos: 1,
            reasoning: 'No confidence metadata found → QoS 1 (default)',
            confidenceUsed: undefined,
            humanVerified: false
        };
    }

    let qos: 0 | 1 | 2;
    let reasoning: string;

    if (conf >= 0.9) {
        qos = 2;
        reasoning = `@confidence=${conf} ≥ 0.9 (${source}) → QoS 2 (exactly once)`;
    } else if (conf >= 0.5) {
        qos = 1;
        reasoning = `0.5 ≤ @confidence=${conf} < 0.9 (${source}) → QoS 1 (at least once)`;
    } else {
        qos = 0;
        reasoning = `@confidence=${conf} < 0.5 (${source}) → QoS 0 (at most once)`;
    }

    return {
        qos,
        reasoning,
        confidenceUsed: conf,
        humanVerified: false
    };
}

// ── MQTT 5.0 Properties ───────────────────────────────────────────

/**
 * Derive MQTT 5.0 PUBLISH packet properties from a JSON-LD document.
 */
export function deriveMqtt5Properties(
    doc: JsonLdNode,
    compress: boolean = true
): Record<string, any> {
    const props: Record<string, any> = {};

    // Payload Format Indicator: 0 = unspecified (CBOR), 1 = UTF-8 (JSON)
    props.payloadFormatIndicator = compress ? 0 : 1;

    // Content Type
    props.contentType = compress ? 'application/cbor' : 'application/ld+json';

    // Message Expiry Interval
    const expiry = deriveExpirySeconds(doc);
    if (expiry !== undefined) {
        props.messageExpiryInterval = expiry;
    }

    // User Properties
    const userProps: Record<string, string> = {};

    let typeVal = doc['@type'];
    if (Array.isArray(typeVal)) typeVal = typeVal[0];
    if (typeVal) userProps['jsonld_type'] = String(typeVal);

    const conf = getConfidence(doc);
    if (conf !== undefined) userProps['jsonld_confidence'] = String(conf);

    const source = doc['@source'];
    if (source) userProps['jsonld_source'] = String(source);

    const docId = doc['@id'];
    if (docId) userProps['jsonld_id'] = String(docId);

    if (Object.keys(userProps).length > 0) {
        props.userProperties = userProps;
    }

    return props;
}

function deriveExpirySeconds(doc: JsonLdNode): number | undefined {
    let validUntil = doc['@validUntil'];

    // Fall back to property-level search
    if (!validUntil) {
        for (const [key, val] of Object.entries(doc)) {
            if (key.startsWith('@')) continue;
            if (val && typeof val === 'object' && '@validUntil' in val) {
                validUntil = val['@validUntil'];
                break;
            }
        }
    }

    if (!validUntil || typeof validUntil !== 'string') {
        return undefined;
    }

    try {
        const expiryDt = new Date(validUntil);
        if (isNaN(expiryDt.getTime())) return undefined;

        const now = new Date();
        const diffMs = expiryDt.getTime() - now.getTime();

        if (diffMs <= 0) return undefined;

        // MQTT Message Expiry Interval is uint32 (seconds)
        const seconds = Math.ceil(diffMs / 1000);
        return Math.min(seconds, 0xFFFFFFFF);
    } catch {
        return undefined;
    }
}
