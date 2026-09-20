# Fair-evaluation protocol cho RAPTOR/LongRAG faithful baselines

Ngày: 2026-09-19. Áp dụng cho mọi run từ Task 8 trở đi.

## 1. Nguyên tắc

- Chuẩn hóa **input / output contract / metrics / manifest / logging /
  provenance**. Không ép RAPTOR và LongRAG dùng chung retrieval algorithm:
  RAPTOR giữ semantic tree retrieval, LongRAG giữ semantic long-unit
  retrieval — đó là đặc trưng của phương pháp gốc.
- Controlled run được phép dùng chung generator/reader để isolate retrieval;
  primary run giữ cấu hình faithful (reader gốc của từng phương pháp).

## 2. Adapter contract (Task 5/6 phải implement)

Mỗi faithful baseline là một `Retriever` theo `src/edahr/interfaces.py`
(`search(query, k, source) -> list[Hit]` với `node_id` là **child leaf**),
cộng thêm:

- `index_metadata() -> dict`: model IDs, hyperparams, corpus/manifest hashes,
  index build time, cache paths.
- `retrieval_trace(query) -> dict`: tree nodes / long units đã xét, scores,
  mapping unit/node → leaf ids, latency.
- Leaf mapping bắt buộc: mọi retrieved node/unit phải expose
  `evidence_child_ids` về child leaves (RAPTOR) hoặc paragraph IDs QASPER
  (LongRAG) để evaluator chấm ở cả hai mức.

## 3. Run types

| Run | Mục đích | Cấu hình |
|---|---|---|
| Faithful primary | So sánh đúng phương pháp gốc | RAPTOR: SBERT/OpenAI embeddings + abstractive summaries + collapsed-tree; LongRAG: bge-large-en-v1.5 units + long-context LLM reader |
| Controlled | Isolate retrieval granularity | Cả hai + B3/B4/edahr dùng chung generator, reranker, verifier, token budget |
| Smoke | Kiểm tra plumbing | 2–5 papers, Task 8 |
| Development | Chọn ngưỡng/cấu hình adapter | Chỉ dev manifest, cấm test |
| Frozen main | Số liệu báo cáo | Manifest đã hash, config khóa (Task 9) |

## 4. Seeds, top-k, budget

- Seeds: `42` chính; lặp `1337` khi tài nguyên cho phép. Mọi RNG
  (clustering, sampling, bootstrap) nhận seed tường minh qua config.
- Retrieval: RAPTOR collapsed `top_k=10`, `max_tokens=3500` (official defaults);
  LongRAG document-units `k=2` primary (paper Table 5: Qasper F1 25.9%),
  sweep `{1,2,5}` ghi đủ. Flat baselines: `candidate_k=80`, `rerank_k=24`.
- Context budget: `context_token_budget=7000`, `final_context_k=8`
  (Settings mặc định); LongRAG reader log tokens thật đã nạp (~30K regime
  của paper được ghi nhận nhưng budget   của repo giữ 7000 trừ khi protocol sửa đổi có ghi log rõ ràng — mọi sai lệch phải log).

## 5. Metrics (định nghĩa + nguồn dữ liệu)

| Metric | Định nghĩa | Nguồn |
|---|---|---|
| Recall@k / MRR / nDCG | Chuẩn IR trên ranked child ids vs gold children | `src/edahr/evaluation.py`, `run_benchmark` rows |
| Official QASPER paragraph Evidence F1 | Max-over-annotations paragraph F1 | `qasper_evidence_f1`, predicted từ `paragraph_texts` của evidence leaves |
| Leaf attribution P/R/F1 | Citation precision/recall/F1 trên child ids | `citation_precision/recall/f1` |
| Rescue / harmful drift / kept-correct / kept-wrong | Phân rã evidence theo retrieved set | `run_benchmark` rows |
| Answer EM / F1 | QASPER official normalization, max-over-annotations | `qasper_answer_exact_match/token_f1` |
| Context tokens | Tổng token blocks đã dùng | `result.metrics["context_tokens"]` |
| Latency | Retrieval/rerank/generation/verification/total ms | `result.metrics` + `latency_stats` |
| Peak memory / GPU memory | `tracemalloc` hoặc `nvidia-smi` sampling khi có GPU | run metadata |
| Index construction time | Wall-clock build + cache size MB | index metadata |
| Inference cost | API tokens × đơn giá / GPU-giờ | run metadata |
| Paired p-value | `paired_bootstrap_test`, 1000 iters | `significance_vs_baseline` |
| Paper-clustered 95% CI | `paired_cluster_bootstrap` theo source | `clustered_ci_vs_baseline` |

Không tuyên bố superiority khi CI chứa 0 hoặc mẫu thiếu lực thống kê.

## 6. Manifest và leakage

- Manifest tạo, lưu và **hash SHA-256 trước inference** (Task 4).
- Cấm `eligible[:N]` làm manifest chính.
- Cấm dùng test split để chọn hyperparameters, thresholds, prompts, models.
- Test manifest stratified hiện có (`data/manifests_test/`, 180 papers,
  seed 42, `scripts/create_test_manifest.py`) dùng cho frozen main;
  baseline manifests Task 4 dùng cho dev/smoke.
