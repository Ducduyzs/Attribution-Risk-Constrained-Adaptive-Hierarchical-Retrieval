# Main eval (dev, 30 papers/75 questions) — faithful baselines

| system | recall@5 | citation F1 | answer F1 | official ev F1 | tokens |
|---|---:|---:|---:|---:|---:|
| B5_raptor_faithful | 0.5847 | 0.2774 | 0.1988 | 0.3424 | 1442.8 |
| B6_longrag_faithful_controlled | 0.4049 | 0.1617 | 0.1765 | 0.2279 | 1562.4 |
| B3_flat_neural | 0.5229 | 0.2431 | 0.1985 | 0.3042 | 1572.7 |
| B4_static_hierarchy | 0.5229 | 0.2433 | 0.2124 | 0.3407 | 2931.7 |
| edahr_prior | 0.5229 | 0.2137 | 0.1862 | 0.3013 | 2215.2 |

## Paired vs B3 (citation F1)

- B5_raptor_faithful: p=0.1608, clustered 95% CI [-0.0124, 0.0909]
- B6_longrag_faithful_controlled: p=0.0070, clustered 95% CI [-0.1444, -0.0283]
- B4_static_hierarchy: p=0.9940, clustered 95% CI [-0.0538, 0.0566]
- edahr_prior: p=0.4006, clustered 95% CI [-0.0940, 0.0406]
