# Bàn giao v8 — Các hướng đã thử và vì sao chưa thắng

Cập nhật 2026-10-04. Mục đích: cung cấp đủ bối cảnh để một AI/người khác đề xuất
**hướng mới**, không lặp lại các hướng đã thất bại. Thay thế `CONTEXT_2026-09-30.md`.

## 1. Bài toán

QA có trích dẫn trên bài báo khoa học (QASPER v0.3, tiếng Anh). Pipeline: chia bài
thành document → section → parent (4 child, chồng 1) → child/leaf (~220 token) →
BGE-M3 (dense + sparse + ColBERT) → reranker `bge-reranker-v2-m3` → **mở rộng thích ứng**
child→parent→section (đóng góp đề xuất) → ghép context theo ngân sách token →
gpt-4o-mini sinh claim kèm context ID (strict schema) → verifier NLI
(`DeBERTa-v3-base-mnli-fever-anli`) chọn leaf được trích dẫn cho từng claim.

Mục tiêu bài báo: phương pháp đề xuất (`prior` = mở rộng thích ứng bằng utility tay;
`learned_v7` = gate học được) phải **thắng baseline mạnh nhất (RAPTOR)** về attribution,
ưu tiên ít token (người dùng quan tâm chi phí).

## 2. Bảng chuẩn hiện tại (dev 75 câu / 30 bài, gpt-4o-mini, verifier đã sửa)

| Hệ thống | Citation F1 | Evidence F1 (official) | Answer F1 | Token context |
|---|---:|---:|---:|---:|
| oracle_evidence (context = gold) | 0,621 | 0,602 | 0,270 | 517 |
| **B5 RAPTOR faithful** | **0,436** | **0,404** | **0,244** | 1452 |
| prior_raptor | 0,397 | 0,389 | 0,226 | 1871 |
| **prior (đề xuất)** | 0,387 | 0,396 | 0,231 | 2215 |
| B4 static hierarchy | 0,365 | 0,373 | 0,229 | 2932 |
| B3 flat neural | 0,370 | 0,369 | 0,201 | 1572 |
| full_document | 0,343 | 0,337 | 0,205 | 4355 |
| B6 LongRAG faithful | 0,313 | 0,287 | 0,195 | 1562 |

Thống kê (paper-clustered, Holm trên 15 so sánh): **không so sánh nào có ý nghĩa sau
Holm**. prior − RAPTOR citation F1 = −0,049 [−0,109; +0,010]. Nguồn:
`analysis/v8_unified_gpt4omini.md`, `analysis/stats_v8_dev.md`.

## 3. Sự thật cấu trúc quan trọng (ràng buộc mọi ý tưởng)

| Sự thật | Số liệu | Hệ quả |
|---|---|---|
| Bài QASPER ngắn | 11–56 leaf/bài (trung vị 26); 1,6k–8,4k token/bài | Truy hồi trong *một* bài |
| `candidate_k = 80` > số leaf mỗi bài | Retrieval tầng đầu luôn trả cả bài | Embedding gần như không quyết định thứ hạng |
| `rerank_k = 24` | 43% câu hỏi đưa cả bài vào reranker | **Reranker (trên văn bản thô) quyết định** |
| Retrieval nền | flat recall@5 0,52; recall@10 0,75; MRR 0,53 | Nửa evidence không vào top-5 |
| Mở rộng parent thêm gold chưa retrieve | ~1% nhóm ứng viên (section ~1,3%) | Mở rộng phân cấp gần như không có dư địa |
| Harmful drift do mở rộng | 0–1 leaf / 75 câu | Không có rủi ro để kiểm soát |
| Claim mỗi câu trả lời | ~2 | Ngay cả oracle chỉ đạt citation recall 0,54–0,57 |
| `max_evidence_per_claim = 1` | nới lên 2–3: +0,01–0,03 citation F1 | Không phải nút thắt |

## 4. Các hướng đã thử

| # | Hướng | Kết quả | Vì sao |
|---|---|---|---|
| 1 | Gate học được v7 (RF/GB/HGB trên nhãn KEEP/EXPAND counterfactual) | dev AUC 0,54 / 0,51; train 0,995 | Nhãn = chênh reward sau 1 lần sinh LLM → nhiễu; dư địa ~1% |
| 2 | H1: nhãn từ gold (mở rộng có thêm gold?) | AUC 0,9 nhưng là rò rỉ cấu trúc; có điều kiện ≈ 0,5 | Chỉ 8–12 mẫu dương dev |
| 3 | H2: học xếp hạng theo cặp | AUC 0,585 (CI loại 0,5) | Dưới ngưỡng 0,60 |
| 4 | H3: feature tương đối theo câu hỏi | AUC 0,565–0,586 | Dưới ngưỡng |
| 5 | H4: leaf mở rộng chỉ để đọc, không được trích | Không tác dụng | Drift ≈ 0 |
| 6 | H5: conformal risk control cho ngưỡng trích leaf mở rộng | Không tác dụng (đã cài, đúng lý thuyết) | Drift ≈ 0 |
| 7 | full_document (không giới hạn ngân sách) | Thua flat nhẹ, tốn 2,8× token | Context dài không giúp |
| 8 | Nhiều trích dẫn mỗi claim (k=2,3,all) | +0,01–0,03 citation F1, evidence F1 giảm | Nút thắt là số claim |
| 9 | Contextual chunk `title` (tên bài + mục vào embedding) | Không đổi; hại RAPTOR (−0,077 cit F1) | Tên bài giống nhau trong cùng bài; embedding không quyết định |
| 10 | Contextual chunk `llm` (Anthropic, 870 chunk) | Không đổi; hại RAPTOR (−0,063) | Reranker chấm văn bản thô quyết định thứ hạng |
| 11 | prior_raptor (mở rộng của prior trên retrieval RAPTOR) | −0,038 cit F1 so với RAPTOR, +29% token | Mở rộng thêm nhiễu, không thêm gold |

## 5. Phát hiện phụ có giá trị (dùng được cho bài báo)

- **Bug verifier đã sửa**: guard phủ định/số liệu phủ quyết claim NLI-entailed trên cả leaf
  220 token → ~25% claim bị loại oan (~60% là gold). Sửa xong +0,07–0,11 citation F1 mọi hệ thống.
- **Evaluator official AllenAI** vendor + audit: khớp tuyệt đối 30.690 dự đoán (sửa 3 sai lệch).
- **Lỗi provenance**: bảng cũ (`artifacts/baselines/main`) do Qwen2.5-7B sinh, metadata ghi gpt-4o-mini.
- Trên Qwen-7B prior thấp nhất; trên gpt-4o-mini prior vượt flat/static → kết quả phụ thuộc generator.

## 6. Chưa thử (gợi ý điểm xuất phát, không phải khuyến nghị)

- Đưa ngữ cảnh chunk vào **đầu vào reranker** (ngữ cảnh `llm` đã có sẵn, ~0,2 USD).
- Reranker mạnh hơn hoặc fine-tune reranker trên QASPER train; LLM làm reranker.
- Viết lại/mở rộng câu hỏi (HyDE, multi-query) — nhưng tầng embedding không quyết định.
- Tăng độ phủ evidence của generator (nhiều claim hơn, claim theo từng evidence, trích mức câu).
- Xử lý riêng theo loại câu (yes/no, unanswerable, extractive, abstractive).
- Truy hồi lặp / đa bước; abstain dựa trên "sufficient context".
- Benchmark có bài dài hơn (PeerQA ~12k token/bài) nơi retrieval và mở rộng có thể quan trọng hơn.
- Hướng B: bài phân tích/kết quả âm dựa trên mục 3–5.

## 7. Ràng buộc thực tế

- **Chi phí**: người dùng quan tâm chi phí. gpt-4o-mini; giới hạn tài khoản 200k token/phút.
  Mỗi lượt dev 75 câu × 1 hệ thống ≈ 0,03–0,07 USD API.
- **GPU**: thí nghiệm nặng chạy trên **RTX 3090 thuê (Vast.ai)**, không chạy trên laptop
  local (RTX 4050 6GB). Phân tích offline/CPU chạy local được.
- **Đánh giá**: evaluator official (`scripts/audit_evaluator.py`), thống kê theo cụm bài báo
  (`src/edahr/statistics.py`, `scripts/stats_report.py`). Test 180 câu
  (`data/manifests_test/`) **chưa mở** — chỉ chạy một lần sau khi khóa protocol.
- **Offline replay**: `src/edahr/replay.py` chấm lại verification từ trace đã lưu, không tốn API.

## 8. File chính

| Nội dung | File |
|---|---|
| Bảng chuẩn v8 | `analysis/v8_unified_gpt4omini.md` |
| H1–H5, bug verifier | `analysis/h1_h5_results.md` |
| full_document / oracle | `analysis/v8_fulldoc_dev.md` |
| Ngữ cảnh title / llm, prior_raptor | `analysis/v8_title_context.md`, `analysis/v8_llm_context_and_prior_raptor.md` |
| Thống kê | `analysis/stats_v8_dev.md` |
| Tổng quan 40 bài báo + 8 hướng | `analysis/literature_review_improvement_directions.md` |
| Audit cũ (P0) | `analysis/v7_publication_readiness_audit.md` |
| Benchmark runner | `scripts/run_qasper_benchmark.py` |

Git: các commit v8 (`fe15a0a` … `a8b3d2e`) chỉ ở local, **chưa push**.
