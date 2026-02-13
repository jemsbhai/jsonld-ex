
import {
    toMqttPayload,
    fromMqttPayload,
    deriveMqttTopic,
    deriveMqttQos,
    deriveMqtt5Properties
} from '../mqtt.js';

describe('MQTT Parity Tests', () => {

    describe('Payload Round-Trip', () => {
        it('should round-trip compressed CBOR payload', () => {
            const doc = {
                '@context': 'http://schema.org/',
                '@type': 'SensorReading',
                '@id': 'urn:sensor:imu-001',
                'value': { '@value': 42.5, '@confidence': 0.9 }
            };

            const payload = toMqttPayload(doc, true);
            const restored = fromMqttPayload(payload, undefined, true);

            expect(restored['@type']).toBe('SensorReading');
            expect(restored['value']['@value']).toBe(42.5);
            expect(restored['value']['@confidence']).toBe(0.9);
        });

        it('should round-trip uncompressed JSON payload', () => {
            const doc = { '@type': 'Event', 'name': 'test' };
            const payload = toMqttPayload(doc, false);
            const restored = fromMqttPayload(payload, undefined, false);

            expect(restored).toEqual(doc);
        });

        it('should reattach context', () => {
            const doc = { '@type': 'Person', 'name': 'Alice' };
            const payload = toMqttPayload(doc, false);
            const restored = fromMqttPayload(payload, 'http://schema.org/', false);

            expect(restored['@context']).toBe('http://schema.org/');
        });

        it('should contain smaller payload for compressed vs json', () => {
            const doc = {
                '@context': 'http://schema.org/',
                '@type': 'SensorReading',
                'value': { '@value': 42.5, '@confidence': 0.9 },
                'timestamp': '2025-01-15T10:30:00Z'
            };
            const compressed = toMqttPayload(doc, true);
            const uncompressed = toMqttPayload(doc, false);

            expect(compressed.length).toBeLessThan(uncompressed.length);
        });
    });

    describe('Topic Derivation', () => {
        it('should derive basic topic', () => {
            const doc = { '@type': 'SensorReading', '@id': 'urn:sensor:imu-001' };
            expect(deriveMqttTopic(doc)).toBe('ld/SensorReading/imu-001');
        });

        it('should handle custom prefix', () => {
            const doc = { '@type': 'SensorReading', '@id': 'urn:sensor:imu-001' };
            expect(deriveMqttTopic(doc, 'devices')).toBe('devices/SensorReading/imu-001');
        });

        it('should handle fragments', () => {
            const doc = { '@type': 'Widget', '@id': 'http://example.org/things#widget-5' };
            expect(deriveMqttTopic(doc)).toBe('ld/Widget/widget-5');
        });

        it('should handle sanitisation', () => {
            const doc = { '@type': 'Type#Sub', '@id': 'urn:x:1' };
            const topic = deriveMqttTopic(doc);
            expect(topic).not.toContain('#');
            // ld/Sub/1
            expect(topic).toBe('ld/Sub/1');
        });

        it('should strip leading $', () => {
            const doc = { '@type': '$SYS', '@id': 'urn:x:1' };
            const topic = deriveMqttTopic(doc);
            // $SYS -> SYS
            expect(topic).toBe('ld/SYS/1');
        });
    });

    describe('QoS Derivation', () => {
        it('should map high confidence to QoS 2', () => {
            const doc = { '@confidence': 0.95, '@type': 'Alert' };
            expect(deriveMqttQos(doc)).toBe(2);
        });

        it('should map medium confidence to QoS 1', () => {
            const doc = { '@confidence': 0.7, '@type': 'Reading' };
            expect(deriveMqttQos(doc)).toBe(1);
        });

        it('should map low confidence to QoS 0', () => {
            const doc = { '@confidence': 0.3, '@type': 'Noise' };
            expect(deriveMqttQos(doc)).toBe(0);
        });

        it('should map humanVerified to QoS 2', () => {
            const doc = { '@humanVerified': true, '@type': 'Diagnosis' };
            expect(deriveMqttQos(doc)).toBe(2);
        });

        it('should default to QoS 1', () => {
            const doc = { '@type': 'Event', 'name': 'test' };
            expect(deriveMqttQos(doc)).toBe(1);
        });

        it('should use property level confidence', () => {
            const doc = {
                '@type': 'Reading',
                'value': { '@value': 42, '@confidence': 0.95 }
            };
            expect(deriveMqttQos(doc)).toBe(2);
        });
    });

    describe('MQTT 5.0 Properties', () => {
        it('should set payload format indicator', () => {
            const propsCompressed = deriveMqtt5Properties({ '@type': 'T' }, true);
            expect(propsCompressed.payloadFormatIndicator).toBe(0);

            const propsJson = deriveMqtt5Properties({ '@type': 'T' }, false);
            expect(propsJson.payloadFormatIndicator).toBe(1);
        });

        it('should set content type', () => {
            const propsCompressed = deriveMqtt5Properties({ '@type': 'T' }, true);
            expect(propsCompressed.contentType).toBe('application/cbor');

            const propsJson = deriveMqtt5Properties({ '@type': 'T' }, false);
            expect(propsJson.contentType).toBe('application/ld+json');
        });

        it('should set message expiry interval', () => {
            // 1 hour in future
            const future = new Date(Date.now() + 3600 * 1000).toISOString();
            const doc = { '@type': 'T', '@validUntil': future };

            const props = deriveMqtt5Properties(doc);
            expect(props.messageExpiryInterval).toBeGreaterThanOrEqual(3595);
            expect(props.messageExpiryInterval).toBeLessThanOrEqual(3605);
        });

        it('should set user properties', () => {
            const doc = {
                '@type': 'SensorReading',
                '@id': 'urn:sensor:imu-001',
                '@confidence': 0.92,
                '@source': 'https://model.example.org/v3'
            };
            const props = deriveMqtt5Properties(doc);
            const userProps = props.userProperties;

            expect(userProps.jsonld_type).toBe('SensorReading');
            // Note: number might be stringified
            expect(Number(userProps.jsonld_confidence)).toBe(0.92);
            expect(userProps.jsonld_source).toBe('https://model.example.org/v3');
            expect(userProps.jsonld_id).toBe('urn:sensor:imu-001');
        });
    });

});
