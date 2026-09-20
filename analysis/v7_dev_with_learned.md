# QASPER results — qasper-v0.3 test-unseen-paper

Questions/papers: 5/5. Generator: `openai:gpt-4o-mini`.
Configured thresholds and estimator families are recorded in the run provenance.

| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B3_flat_neural | 0.3000 | 0.2333 | 0.1958 | 0.3000 | 1745.8 | 3456.7 |
| B4_static_hierarchy | 0.2571 | 0.2800 | 0.1515 | 0.3000 | 3435.6 | 2816.9 |
| prior | 0.2571 | 0.2133 | 0.1861 | 0.3000 | 2625.4 | 2633.3 |
| learned_v7 | 0.2571 | 0.2133 | 0.1403 | 0.3000 | 2625.4 | 3031.7 |

## Paired comparison against B3_flat_neural

- B4_static_hierarchy: official evidence-F1 paired p=0.5385; paper-clustered 95% CI for difference [-0.1286, 0.0000].
- prior: official evidence-F1 paired p=0.5385; paper-clustered 95% CI for difference [-0.1286, 0.0000].
- learned_v7: official evidence-F1 paired p=0.5385; paper-clustered 95% CI for difference [-0.1286, 0.0000].
