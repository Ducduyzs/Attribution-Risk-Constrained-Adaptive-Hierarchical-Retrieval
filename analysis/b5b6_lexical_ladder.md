# Offline full-ladder benchmark (B0-B6 + edahr) — QASPER dev subset

Papers/questions: 10/30. Backend: BM25 retrieval, lexical rerank, extractive generation, token-overlap verification (no neural weights, no LLM key). All systems share ingestion/hierarchy/generation/verification; only retrieval granularity and hierarchy control differ.

| system | recall@5 | citation F1 | answer F1 | ctx tokens |
|---|---:|---:|---:|---:|
| B0_bm25 | 0.2061 | 0.0919 | 0.0822 | 1363.1 |
| B1_dense | 0.2061 | 0.0919 | 0.0822 | 1363.1 |
| B2_hybrid_rrf | 0.2061 | 0.0640 | 0.0732 | 3387.0 |
| B3_flat_neural | 0.2246 | 0.0924 | 0.0873 | 1511.3 |
| B4_static_hierarchy | 0.2246 | 0.0663 | 0.0844 | 2982.4 |
| B5_raptor | 0.3080 | 0.1217 | 0.0931 | 1528.0 |
| B6_longrag | 0.2913 | 0.0730 | 0.0894 | 2852.6 |
| edahr | 0.2246 | 0.0663 | 0.0844 | 2982.4 |

## Paired comparison vs B3_flat_neural (citation F1)

- B0_bm25: paired p=0.9361; paper-clustered 95% CI [-0.0156, 0.0208]
- B1_dense: paired p=0.9361; paper-clustered 95% CI [-0.0156, 0.0208]
- B2_hybrid_rrf: paired p=0.5475; paper-clustered 95% CI [-0.1086, 0.0494]
- B4_static_hierarchy: paired p=0.4735; paper-clustered 95% CI [-0.0966, 0.0335]
- B5_raptor: paired p=0.3786; paper-clustered 95% CI [-0.0303, 0.0871]
- B6_longrag: paired p=0.6244; paper-clustered 95% CI [-0.1038, 0.0569]
- edahr: paired p=0.4735; paper-clustered 95% CI [-0.0966, 0.0335]
