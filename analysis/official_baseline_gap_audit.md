# Audit khoảng cách: baseline RAPTOR/LongRAG hiện tại vs paper gốc

Ngày: 2026-09-19. Phạm vi: `src/edahr/baselines.py` (1076 dòng),
`scripts/run_lexical_ladder_b5b6.py`, `tests/test_baselines.py`,
`analysis/b5b6_lexical_ladder.{md,json}`, `tools/create_ieee_paper.py`.

Kết luận trước: implementation hiện tại là **RAPTOR-style lexical rút gọn**
và **LongRAG-style long-unit rút gọn**. Tuyệt đối không gọi chúng là
"official RAPTOR" / "official LongRAG" / "faithful" trong bất kỳ bảng,
label hay claim nào cho đến khi Task 5/6 hoàn thành.

## 1. RAPTOR — bảng so sánh với paper gốc (Sarthi et al., ICLR 2024)

| Thành phần paper gốc | Hiện có trong repo | Khoảng cách | Ảnh hưởng công bằng | Hướng triển khai (Task 5) |
|---|---|---|---|---|
| Chunk ~100 tokens, embed bằng SBERT (`all-mpnet-base-v2` trong official code) | Chunk 220-token của pipeline EDAHR; embed bằng **TF-IDF** (`baselines.py:217-223,245-246`) | Không có semantic embedding; tương đồng từ vựng thay vì tương đồng ngữ nghĩa | Cao — cụm và summary kém ngữ nghĩa, bất lợi cho RAPTOR trên câu hỏi diễn đạt lại | Dùng SBERT/sentence-transformers thật, cache embedding, ghi model ID |
| Clustering: GMM trên UMAP-reduced embeddings, số cụm theo BIC | **KMeans trên TF-IDF**, `n_clusters=ceil(n/4)` (`baselines.py:234,247`) | Thuật toán, số cụm và không gian vector đều khác | Trung bình–cao — cấu trúc cây khác, khó quy kết chênh lệch về phương pháp | Reimplement GMM+UMAP+BIC hoặc adapter official repo; khóa seed |
| Summary: **abstractive** bằng LLM (GPT-3.5/4, ~120 từ) | **Extractive centroid sentences**, tối đa 5 câu/220 từ (`baselines.py:266-294`) | Mất khả năng trừu tượng hóa, hợp nhất và diễn đạt lại đa nguồn | Cao — đúng điểm mạnh RAPTOR bị loại bỏ | Summarizer LLM có cấu hình rõ, cache theo hash, provenance model version |
| Retrieval: tree traversal HOẶC **collapsed tree** trên embeddings của mọi tầng | Collapsed-tree nhưng summary **chấm BM25 lexical** (`_LexicalScorer`, `baselines.py:312-357,391`), fuse với leaf ranking bằng **RRF** (`baselines.py:420-424`) | RRF không có trong paper; internal nodes không được chấm bằng cùng embedding space | Cao — cơ chế retrieval khác gốc, RRF có thể bơm/pha loãng tín hiệu | Score mọi node bằng cùng semantic retriever; chọn traversal/collapsed theo paper |
| Reader: GPT-3.5/4, UnifiedQA, etc. | Extractive generator: 3 câu đầu của top blocks (`run_lexical_ladder_b5b6.py: LexicalReranker/ExtractiveGenerator/OverlapVerifier`) | Không đo khả năng đọc đa tầng của RAPTOR | Cao — answer F1 ~0.09 phản ánh generator đồ chơi, không phản ánh RAPTOR | Controlled run: cùng LLM reader cho mọi baseline (Task 3) |
| Verifier: không có (paper đo QA accuracy) | Token-overlap verifier | Metric citation F1 đo overlap từ, không phải support ngữ nghĩa | Trung bình — làm méo so sánh grounding | Dùng NLI verifier của repo cho mọi baseline khi chạy controlled |

## 2. LongRAG — bảng so sánh với paper gốc (Jiang et al., arXiv:2406.15319)

| Thành phần paper gốc | Hiện có trong repo | Khoảng cách | Ảnh hưởng công bằng | Hướng triển khai (Task 6) |
|---|---|---|---|---|
| Long units 4K-token gom theo hyperlink (Wikipedia); với QASPER/MultiFieldQA: **mỗi document là 1 unit** | **Mỗi section là 1 unit** (`LongRagRetriever._build_units`, `baselines.py:454-470`); toy single-section còn bị chẻ đôi `#a/#b` (`baselines.py:462-467`) | Granularity sai: section ≪ long unit gốc; chẻ `#a/#b` là artifact của test toy, không tồn tại trong paper | Cao — không còn là "long retriever" theo nghĩa paper; kết quả không ngoại suy được | Units theo paper: document-level cho QASPER (ghi rõ adaptation), hoặc gom sections liên quan có tài liệu hóa |
| Long retriever: **bge-large-en-v1.5 embeddings**, corpus 22M→600K units | **Không có retriever riêng**: `unit_score = mean(top-2 BM25 leaf scores)` (`baselines.py:488-492`) | Thứ tự unit hoàn toàn suy từ leaf BM25 — đây là flat retrieval khoác áo long-unit | Rất cao — baseline mất đặc trưng cốt lõi, mọi chênh lệch chỉ là hiệu ứng sắp xếp lại | Semantic retriever thật trên unit texts (bge-large-en-v1.5 hoặc tương đương có lý giải), gọi thật trong primary run |
| Long reader: LLM long-context (GPT-4o/Gemini), ~30K tokens, zero-shot | `LongRagPipeline` đọc section blocks + extractive generator (`baselines.py:864-971`) | Không có long-context reader thật; budget/token order không theo paper | Cao — không kiểm chứng được claim "long reader khai thác long units" | Long-context LLM reader, context construction đúng thứ tự + budget, log tokens thật |
| Đánh giá: NQ/HotpotQA EM, Qasper F1 25.9% | 30 câu dev, answer F1 ~0.089 bằng token-F1 trên extractive claims | Quy mô và metric không so được với paper | Trung bình — chỉ dùng nội bộ, cấm trích dẫn cạnh số paper gốc | Frozen manifest, official QASPER paragraph F1 + leaf F1 (Task 3/4/9) |

## 3. Shortcut chung của benchmark hiện tại (xác nhận đủ 8 điểm bắt buộc)

1. **RAPTOR dùng TF-IDF thay semantic embeddings** — `baselines.py:217-223,245-246`.
2. **RAPTOR dùng extractive centroid thay abstractive summaries** — `baselines.py:266-294`.
3. **LongRAG dùng section hiện có làm long unit** — `baselines.py:454-470`.
4. **LongRAG tổng hợp BM25 leaf scores để rank unit** — `baselines.py:488-492`.
5. **Benchmark dùng extractive generator + token-overlap verifier** — `scripts/run_lexical_ladder_b5b6.py` (classes `ExtractiveGenerator`, `OverlapVerifier`).
6. **B1_dense là BM25 stand-in** — `make_baseline_pipeline` B1 dùng `index_factory` (`baselines.py:1005-1011`); trong run offline `index_factory` trả `Bm25ChildRetriever`, nên B1 ≡ B0 (xác nhận trong `b5b6_lexical_ladder.md`: hai dòng đồng nhất 0.2061/0.0919/0.0822).
7. **Benchmark chỉ ~30 questions/10 papers** — `run_lexical_ladder_b5b6.py:90-99` (`eligible[:max_questions]`, defaults 10/40, thực chạy 10 papers/30 questions).
8. **Artifact thiếu provenance/per-query traces publication-grade** — `b5b6_lexical_ladder.json` chỉ có summaries + paired stats, không có per-query predictions, retrieval traces, config/model/seed/manifest hashes, latency/memory/indexing-time/cost.

## 4. Rủi ro nhãn sai trong bài báo hiện tại

- `tools/create_ieee_paper.py` Table V ghi "B5 RAPTOR" / "B6 LongRAG" — phải đổi thành
  "B5 RAPTOR-style lexical" / "B6 LongRAG-style long-unit" (Task 12), kèm chú thích
  backend lexical và sample size.
- Docstring `baselines.py:988` còn ghi "Wire one of B0..B4" trong khi đã có B5/B6 —
  sửa khi refactor Task 5/6.

## 5. Checklist chuyển sang Task 2

- [x] Mọi shortcut liệt kê kèm file:dòng.
- [x] Không gọi implementation hiện tại là official/faithful.
- [ ] Task 2: xác minh official repo, model IDs, hyperparameters, license.
