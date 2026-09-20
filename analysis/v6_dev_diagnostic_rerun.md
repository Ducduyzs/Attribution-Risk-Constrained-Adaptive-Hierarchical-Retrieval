# V5 results — qasper-v0.3 dev-diagnostic-v6

Questions/papers: 20/20. Generator: `openai:gpt-4o-mini`.
Thresholds and estimator families were frozen on dev before this test run.

| system | citation P | citation R | citation F1 | answer F1 | evidence recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_flat | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1484.0 | 4205.1 |
| learned_v5 | 0.0526 | 0.0263 | 0.0351 | 0.0091 | 0.0375 | 1872.5 | 4181.1 |

## Paired comparison against B_flat

- learned_v5: citation-F1 paired p=0.5325; paper-clustered 95% CI for difference [0.0000, 0.1053].
