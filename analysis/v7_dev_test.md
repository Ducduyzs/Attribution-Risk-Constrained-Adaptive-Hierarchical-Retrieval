# QASPER results — qasper-v0.3 test-unseen-paper

Questions/papers: 5/5. Generator: `openai:gpt-4o-mini`.
Configured thresholds and estimator families are recorded in the run provenance.

| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B3_flat_neural | 0.2444 | 0.2133 | 0.1359 | 0.3000 | 1745.8 | 3540.9 |
| B4_static_hierarchy | 0.2800 | 0.2800 | 0.1841 | 0.3000 | 3435.6 | 2896.5 |
| prior | 0.2571 | 0.2133 | 0.1766 | 0.3000 | 2625.4 | 2713.6 |

## Paired comparison against B3_flat_neural

- B4_static_hierarchy: official evidence-F1 paired p=0.5385; paper-clustered 95% CI for difference [0.0000, 0.1067].
- prior: official evidence-F1 paired p=0.5385; paper-clustered 95% CI for difference [0.0000, 0.0381].
