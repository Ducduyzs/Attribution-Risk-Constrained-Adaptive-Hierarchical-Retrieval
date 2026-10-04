# V9 ranking diagnostics (DEV, no new generation)

| System | Recall@5 | Recall@10 | Pool recall | MRR | Rerank candidates |
|---|---:|---:|---:|---:|---:|
| flat24 | 0.5229 | 0.7529 | 0.9422 | 0.5326 | 21.9467 |
| all_leaf | 0.5251 | 0.7518 | 0.9867 | 0.5322 | 29.0933 |
| contextual24 | 0.5469 | 0.7198 | 0.9422 | 0.6041 | 21.9467 |
| contextual_all | 0.5491 | 0.7220 | 0.9867 | 0.6040 | 29.0933 |
| raptor | 0.6024 | 0.7356 | 0.7911 | 0.5838 | 12.7867 |

Full paired paper-clustered results and Holm adjustments are in v9_ranking_diagnostics.json.
Gold is used only for scoring. Pool recall is not evidence visible in the packed context.
