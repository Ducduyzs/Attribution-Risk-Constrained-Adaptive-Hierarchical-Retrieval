# V5 results — scifact dev-ood-frozen

Questions/papers: 40/40. Generator: `openai:gpt-4o-mini`.
Thresholds and estimator families were frozen on dev before this test run.

| system | citation P | citation R | citation F1 | answer F1 | evidence recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_flat | 0.4000 | 0.4000 | 0.3833 | 0.0000 | 0.4083 | 384.8 | 3918.5 |
| B_static | 0.8375 | 0.7083 | 0.7450 | 0.0000 | 0.7533 | 394.8 | 3610.4 |
| prior | 0.7500 | 0.6625 | 0.6825 | 0.0000 | 0.7075 | 389.4 | 3515.0 |
| learned_v5 | 0.8250 | 0.7042 | 0.7358 | 0.0000 | 0.7575 | 383.7 | 3226.0 |

## Paired comparison against B_flat

- B_static: citation-F1 paired p=0.0010; paper-clustered 95% CI for difference [0.2292, 0.4950].
- prior: citation-F1 paired p=0.0010; paper-clustered 95% CI for difference [0.1750, 0.4283].
- learned_v5: citation-F1 paired p=0.0010; paper-clustered 95% CI for difference [0.2250, 0.4867].
