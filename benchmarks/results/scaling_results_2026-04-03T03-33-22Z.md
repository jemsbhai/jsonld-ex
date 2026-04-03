# EN4.1 / EN4.3 Scalability Benchmarks

Generated: 2026-04-03T03-33-22Z

## Scaling Results

### annotate_batch

| N | Mean (ms) | Std (ms) | 95% CI (ms) | Peak Mem (MB) | Throughput (nodes/sec) |
|---|-----------|----------|-------------|---------------|----------------------|
| 1,000 | 0.4 | 0.0 | [0.4, 0.4] | 0.2 | 2,394,330 |
| 10,000 | 4.5 | 0.3 | [4.3, 4.6] | 1.8 | 2,229,543 |
| 100,000 | 51.3 | 6.7 | [46.5, 56.0] | 18.3 | 1,950,272 |
| 1,000,000 | 579.2 | 33.8 | [537.2, 621.2] | 183.5 | 1,726,487 |

### filter_by_confidence

| N | Mean (ms) | Std (ms) | 95% CI (ms) | Peak Mem (MB) | Throughput (nodes/sec) |
|---|-----------|----------|-------------|---------------|----------------------|
| 1,000 | 0.5 | 0.0 | [0.5, 0.5] | 0.0 | 2,108,578 |
| 10,000 | 4.8 | 0.1 | [4.7, 4.8] | 0.1 | 2,098,138 |
| 100,000 | 65.4 | 13.9 | [55.4, 75.3] | 0.4 | 1,529,591 |
| 1,000,000 | 595.4 | 47.6 | [536.3, 654.5] | 4.5 | 1,679,475 |

### merge_graphs

| N | Mean (ms) | Std (ms) | 95% CI (ms) | Peak Mem (MB) | Throughput (nodes/sec) |
|---|-----------|----------|-------------|---------------|----------------------|
| 1,000 | 22.4 | 1.5 | [21.8, 23.0] | 1.4 | 44,616 |
| 10,000 | 290.4 | 24.4 | [279.0, 301.8] | 13.4 | 34,431 |
| 100,000 | 3611.7 | 246.1 | [3435.7, 3787.7] | 135.6 | 27,688 |
| 1,000,000 | 33870.5 | 4703.8 | [28030.9, 39710.2] | 1349.8 | 29,524 |

### validate_batch

| N | Mean (ms) | Std (ms) | 95% CI (ms) | Peak Mem (MB) | Throughput (nodes/sec) |
|---|-----------|----------|-------------|---------------|----------------------|
| 1,000 | 4.4 | 0.4 | [4.3, 4.6] | 0.2 | 224,816 |
| 10,000 | 52.5 | 7.4 | [49.0, 56.0] | 2.0 | 190,449 |
| 100,000 | 748.1 | 98.3 | [677.8, 818.4] | 19.8 | 133,669 |
| 1,000,000 | 6128.8 | 326.9 | [5723.0, 6534.6] | 198.8 | 163,164 |

## Linearity Check

| Operation | Smallest N | Largest N | Throughput (small) | Throughput (large) | Ratio | Linear? |
|-----------|-----------|----------|-------------------|-------------------|-------|---------|
| annotate_batch | 1,000 | 1,000,000 | 2,394,330 | 1,726,487 | 0.721 | YES |
| filter_by_confidence | 1,000 | 1,000,000 | 2,108,578 | 1,679,475 | 0.796 | YES |
| merge_graphs | 1,000 | 1,000,000 | 44,616 | 29,524 | 0.662 | YES |
| validate_batch | 1,000 | 1,000,000 | 224,816 | 163,164 | 0.726 | YES |

## EN4.4: Batch vs Per-Item API Overhead (n=10,000)

| Operation | Per-item (ms) | Batch (ms) | Speedup |
|-----------|---------------|------------|---------|
| annotate | 3.6 | 6.9 | 0.52x |
| validate | 47.5 | 44.0 | 1.08x |
| filter | 0.6 | 5.1 | 0.11x |

## Analysis

All operations demonstrate linear scaling from 1,000 to 1,000,000 nodes (throughput ratio > 0.5 between smallest and largest scale).

- **annotate_batch**: 2,394,330 -> 1,726,487 nodes/sec (ratio 0.721)
- **filter_by_confidence**: 2,108,578 -> 1,679,475 nodes/sec (ratio 0.796)
- **merge_graphs**: 44,616 -> 29,524 nodes/sec (ratio 0.662)
- **validate_batch**: 224,816 -> 163,164 nodes/sec (ratio 0.726)

Batch API provides 0.57x average speedup over per-item calls at n=10,000.
