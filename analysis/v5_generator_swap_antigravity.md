# V5 exploratory generator swap — QASPER unseen-paper test

Questions/papers: 10/10. Generator: `antigravity:gemini-3.7-flash`.
Thresholds and estimator families were frozen on dev before this test run.
This small matched subset is a generator-sensitivity check, not the main QASPER table.

| system | citation P | citation R | citation F1 | answer F1 | evidence recall | context tokens | latency ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| B_flat | 0.8000 | 0.3750 | 0.4852 | 0.2092 | 0.6700 | 1524.8 | 11635.1 |
| B_static | 0.6333 | 0.3750 | 0.4200 | 0.1952 | 0.5300 | 3284.6 | 10271.2 |
| prior | 0.8000 | 0.4083 | 0.4919 | 0.2360 | 0.7100 | 2322.4 | 9518.7 |
| learned_v5 | 0.8667 | 0.4417 | 0.5519 | 0.2258 | 0.7100 | 1814.2 | 11582.9 |

## Paired comparison against B_flat

- B_static: citation-F1 paired p=0.6054; paper-clustered 95% CI for difference [-0.2738, 0.1733].
- prior: citation-F1 paired p=0.8272; paper-clustered 95% CI for difference [-0.1000, 0.1200].
- learned_v5: citation-F1 paired p=0.5375; paper-clustered 95% CI for difference [0.0000, 0.2000].
