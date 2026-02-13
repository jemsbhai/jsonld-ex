
import {
    createDatasetMetadata,
    addDistribution,
    addFileSet,
    addRecordSet,
    createField,
    toCroissant,
    SCHEMA_ORG,
    CROISSANT_NS
} from '../dataset.js';

describe('Dataset Parity Tests', () => {

    it('should create basic dataset metadata', () => {
        const ds = createDatasetMetadata('My Dataset', {
            description: 'A test dataset',
            version: '1.0',
            creator: 'John Doe',
            isLive: true
        });

        expect(ds['@type']).toBe('sc:Dataset');
        expect(ds.name).toBe('My Dataset');
        expect(ds.description).toBe('A test dataset');
        expect(ds.version).toBe('1.0');
        expect(ds.isLiveDataset).toBe(true);
        expect(ds.creator).toEqual({ '@type': 'Person', 'name': 'John Doe' });
    });

    it('should add distributions', () => {
        let ds = createDatasetMetadata('My Dataset');
        ds = addDistribution(ds, 'data.csv', 'http://example.com/data.csv', 'text/csv');

        expect(ds.distribution).toHaveLength(1);
        expect(ds.distribution![0]['@type']).toBe('cr:FileObject');
        expect(ds.distribution![0].name).toBe('data.csv');
        expect(ds.distribution![0].encodingFormat).toBe('text/csv');
    });

    it('should add record sets and fields', () => {
        let ds = createDatasetMetadata('My Dataset');

        const field1 = createField('col1', 'sc:Integer', { description: 'Column 1' });
        const field2 = createField('col2', 'sc:Text');

        ds = addRecordSet(ds, 'default', [field1, field2]);

        expect(ds.recordSet).toHaveLength(1);
        const rs = ds.recordSet![0];
        expect(rs['@type']).toBe('cr:RecordSet');
        expect(rs.name).toBe('default');
        expect(rs.field).toHaveLength(2);

        // Check ID prefixing
        expect(rs.field![0]['@id']).toBe('default/col1');
        expect(rs.field![1]['@id']).toBe('default/col2');
    });

    it('should convert to Croissant format', () => {
        const ds = createDatasetMetadata('My Dataset');
        const croissantIds = toCroissant(ds);

        // Check context replacement
        expect(croissantIds['@context']['@vocab']).toBe(SCHEMA_ORG);
        expect(croissantIds['@context']['cr']).toBe(CROISSANT_NS);
        expect(croissantIds['conformsTo']).toBeDefined();
    });

});
