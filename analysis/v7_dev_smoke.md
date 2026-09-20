# QASPER results — qasper-v0.3 dev-smoke

Questions/papers: 2/2. Generator: `openai:gpt-4o-mini-2024-07-18`.
Configured thresholds and estimator families are recorded in the run provenance.

| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B_flat | 0.5333 | 0.5357 | 0.2114 | 0.8750 | 1548.0 | 7245.3 |
| B_static | 0.5952 | 0.5833 | 0.1697 | 0.8750 | 2561.5 | 6260.9 |

## Paired comparison against B_flat

- B_static: official evidence-F1 paired p=0.7483; paper-clustered 95% CI for difference [-0.0667, 0.1905].
