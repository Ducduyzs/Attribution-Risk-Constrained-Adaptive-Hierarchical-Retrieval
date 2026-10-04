# QASPER results — qasper-v0.3 baseline-dev-75-llmctx

Questions/papers: 75/30. Generator: `openai:gpt-4o-mini`.
Configured thresholds and estimator families are recorded in the run provenance.

| system | official evidence F1 | leaf attribution F1 | answer F1 | evidence-span recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|
| B3_flat_neural | 0.3768 | 0.3780 | 0.2018 | 0.4791 | 1591.1 | 3213.9 |
| B4_static_hierarchy | 0.3619 | 0.3621 | 0.2186 | 0.4666 | 2982.6 | 3719.3 |
| prior | 0.4048 | 0.3862 | 0.2384 | 0.5028 | 2219.8 | 3503.5 |
| B5_raptor_faithful | 0.3468 | 0.3723 | 0.2106 | 0.4533 | 1448.9 | 3145.0 |
| prior_raptor | 0.3472 | 0.3772 | 0.2220 | 0.4460 | 1839.5 | 3058.6 |

## Paired comparison against B3_flat_neural

- B4_static_hierarchy: official evidence-F1 paired p=0.6244; paper-clustered 95% CI for difference [-0.0701, 0.0459].
- prior: official evidence-F1 paired p=0.1758; paper-clustered 95% CI for difference [-0.0081, 0.0688].
- B5_raptor_faithful: official evidence-F1 paired p=0.2927; paper-clustered 95% CI for difference [-0.0772, 0.0171].
- prior_raptor: official evidence-F1 paired p=0.3237; paper-clustered 95% CI for difference [-0.0882, 0.0221].
