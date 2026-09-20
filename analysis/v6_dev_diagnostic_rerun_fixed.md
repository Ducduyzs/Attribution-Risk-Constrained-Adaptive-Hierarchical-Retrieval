# V5 results — qasper-v0.3 dev-diagnostic-v6-fixed

Questions/papers: 20/20. Generator: `openai:gpt-4o-mini`.
Thresholds and estimator families were frozen on dev before this test run.

| system | citation P | citation R | citation F1 | answer F1 | evidence recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_flat | 0.5053 | 0.6465 | 0.4942 | 0.1676 | 0.7077 | 1484.0 | 5431.8 |
| learned_v5 | 0.4105 | 0.5982 | 0.4439 | 0.1596 | 0.6568 | 1872.5 | 6271.0 |

## Paired comparison against B_flat

- learned_v5: citation-F1 paired p=0.2707; paper-clustered 95% CI for difference [-0.1365, 0.0322].
