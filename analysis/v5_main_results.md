# V5 main results — QASPER unseen-paper test

Questions/papers: 40/40. Generator: `openai:gpt-4o-mini`.
Thresholds and estimator families were frozen on dev before this test run.

| system | citation P | citation R | citation F1 | answer F1 | evidence recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_flat | 0.0250 | 0.0250 | 0.0250 | 0.0077 | 0.0250 | 1488.3 | 4238.0 |
| B_static | 0.1250 | 0.1025 | 0.1062 | 0.0368 | 0.1255 | 2889.9 | 4522.5 |
| prior | 0.0250 | 0.0250 | 0.0250 | 0.0077 | 0.0250 | 2223.8 | 4059.3 |
| learned_v5 | 0.0667 | 0.0500 | 0.0560 | 0.0162 | 0.0663 | 1922.3 | 4870.2 |

## Paired comparison against B_flat

- B_static: citation-F1 paired p=0.0350; paper-clustered 95% CI for difference [0.0167, 0.1479].
- prior: citation-F1 paired p=1.0000; paper-clustered 95% CI for difference [0.0000, 0.0000].
- learned_v5: citation-F1 paired p=0.2178; paper-clustered 95% CI for difference [0.0000, 0.0786].
