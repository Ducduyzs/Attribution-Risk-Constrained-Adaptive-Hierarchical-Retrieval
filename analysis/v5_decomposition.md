# V5 Attribution Decomposition

> Diagnostic decomposition of existing artifacts. It is not the final v5 result table
> unless every evaluated row has a stable question ID and was generated after the
> source-scoped retrieval, selector guard, and v5 reward fixes.

Total rows: 344
Citation-evaluable rows: 292 (84.9%)
Rows with stable question IDs: 0 (0.0%)
Drift/rescue cases: 1

## Input provenance

- `data\artifacts\artifacts_B_flat.jsonl` — SHA-256 `d04b06269296e96c2b335158c757f5e3aca268ea0f4988e74c50877b9d9fe4d2`
- `data\artifacts\artifacts_B_static.jsonl` — SHA-256 `b204ef943c23bea99729f24b71985e468bd340da5ed7d5026bf9fb26897f0fdb`
- `data\artifacts\artifacts_prior.jsonl` — SHA-256 `ce2c354ef51eceb00679e2f4febc1ca54810faaa03fe71705aa1879b142bd252`
- `data\artifacts\artifacts_learned.jsonl` — SHA-256 `b3c705458ef5326bc41c8f4db40ce467e371569d7e4e9c76b33a7c73210f5375`

## Aggregate decomposition

| system | evaluable | mean rescue | mean harmful drift |
|---|---:|---:|---:|
| B_flat | 73 | 0.0000 | 0.0000 |
| B_static | 73 | 0.0068 | 0.0000 |
| learned | 73 | 0.0000 | 0.0000 |
| prior | 73 | 0.0000 | 0.0000 |
