# Smoke review: faithful baselines trên Vast RTX 3090 (Task 8)

Ngày chạy: 2026-09-19. GPU: RTX 3090 24GB (Vast.ai).
Manifest: `manifests/qasper_baseline_dev_*` (hash trong run_metadata).
Smoke: 3 papers đầu manifest (`1603.09631`, `1909.13104`, `1910.01863`),
6 questions, 5/6 multi-evidence, 1 far-evidence — đạt tiêu chí Task 8.
Artifacts: `artifacts/baselines/{raptor,longrag}/smoke/` (+ `index/`).

## 1. RAPTOR faithful (B5_raptor_faithful) — PASS controlled

- Index: SBERT `multi-qa-mpnet-base-cos-v1` thật, UMAP+GMM/BIC, summaries
  **abstractive BART** (đã đọc mẫu L1: diễn đạt mới, không copy câu).
  3 cây/paper, 56–82 nodes, **6 layers** — đúng recursive multi-level.
- Retrieval traces (đọc 3/6):
  - Q1 "How was this data collected?": 58 candidates (26 leaves + 32 internal),
    top-10 trộn L0/L2/L3/L4/L5 — collapsed-tree thật.
  - Q2 "average length of dialog": top-3 toàn L3/L4/L5 summaries (28–34 leaves
    mỗi node) — abstraction được dùng khi phù hợp.
  - Q3 "datasets used": top-3 toàn leaves — specificity được giữ.
  Ba query, ba chế độ granularity khác nhau: đúng hành vi RAPTOR.
- Controlled (shared local Qwen-7B generator + BGE rerank + NLI):
  recall@5 **0.55**, MRR 0.67, citation F1 **0.219** (CI [0.067, 0.333]),
  answer F1 0.214, official evidence F1 **0.40**. verified claims 6/15.
- Leaf mapping đúng: mọi Hit là child id; source filter đúng (test + trace
  per-paper).

## 2. LongRAG faithful (B6_longrag_faithful) — PASS controlled, reader PARTIAL

- Index: units **document-level** (1 unit/paper), retriever
  `bge-large-en-v1.5` + max-over-512tok subchunk, không re-rank — đúng paper.
- Retrieval traces (đọc 3/6): source-scoped (1 unit scored), semantic scores
  0.61–0.67, chọn đúng paper cho cả 3 query.
- Controlled: hit_rate@5 **0.83**, recall@5 **0.575**, citation F1 **0.20**,
  answer F1 **0.27**, official evidence F1 **0.39**.
- Per-query variance khỏe (rec5 0–1, citF1 0–0.4): số liệu thật, không đều giả.
- Reader primary (local Qwen-7B): **2/6 rows** rồi CUDA OOM.
  Nguyên nhân: prompt = cả paper unit (dài) + KV cache 7B fp16 trong khi mọi
  model khác còn resident (~21–22GB/24GB, fragmentation). 2 rows có answer
  đọc được (ansF1 0.26/0.30, evF1 0.075/0.038 — thấp vì predict cả unit).
  Lỗi đã ghi `errors.jsonl`. Task 9 phải có memory hygiene
  (del BART sau indexing, `empty_cache`, `expandable_segments`).

## 3. Blockers & adaptations đã ghi nhận (không giấu)

- OpenAI hết credits; Gemini free-tier 20 req/ngày cạn trên mọi flash model
  đã thử → summarizer = local BART abstractive, generator/reader = local
  Qwen-7B. Mọi thay thế ghi trong index/run metadata.
- Qwen-3B compliance 0% (citations rỗng) → nâng 7B: cit-precision 0.5–0.58.
- Không tổng hợp số liệu reader (2/6, thiếu lực) — chỉ dùng controlled runs.

## 4. Kết luận smoke

RAPTOR + LongRAG faithful **chạy thật end-to-end**, retrieval đúng phương pháp
gốc, artifacts đầy đủ provenance. Đủ điều kiện sang Task 9 với memory hygiene
và manifest dev đầy đủ (30 papers/75 questions).
