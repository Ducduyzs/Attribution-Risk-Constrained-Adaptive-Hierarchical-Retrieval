# QASPER results — qasper-v0.3 baseline-dev-75-title

Questions/papers: 75/30. Generator: `openai:gpt-4o-mini`.
Configured thresholds and estimator families are recorded in the run provenance.

| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B3_flat_neural | 0.3596 | 0.3541 | 0.2086 | 0.4533 | 1585.2 | 3440.2 |
| B4_static_hierarchy | 0.3634 | 0.3649 | 0.2146 | 0.4799 | 2920.8 | 3732.8 |
| prior | 0.3934 | 0.3932 | 0.2381 | 0.4739 | 2223.7 | 3366.8 |
| B5_raptor_faithful | 0.3577 | 0.3587 | 0.2334 | 0.4382 | 1450.4 | 3004.3 |

## Paired comparison against B3_flat_neural

- B4_static_hierarchy: official evidence-F1 paired p=0.9021; paper-clustered 95% CI for difference [-0.0610, 0.0705].
- prior: official evidence-F1 paired p=0.1868; paper-clustered 95% CI for difference [-0.0107, 0.0815].
- B5_raptor_faithful: official evidence-F1 paired p=0.9680; paper-clustered 95% CI for difference [-0.0515, 0.0424].
