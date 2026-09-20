# Nguồn chính thức cho baseline RAPTOR và LongRAG

Ngày truy cập: 2026-09-19. Quy tắc: chỉ dùng official repo + paper/proceedings;
không dùng blog/tổng hợp.

## 1. RAPTOR

- Paper: P. Sarthi et al., "RAPTOR: Recursive Abstractive Processing for
  Tree-Organized Retrieval", Proc. ICLR 2024. arXiv:2401.18059.
  URL: https://arxiv.org/abs/2401.18059
  URL proceedings: https://proceedings.iclr.cc/paper_files/paper/2024/file/8a2acd174940dbca361a6398a4f9df91-Paper-Conference.pdf
- Official repository: https://github.com/parthsarthi03/raptor
  (mô tả repo: "The official implementation of RAPTOR", tác giả Parth Sarthi).
  Branch mặc định: `master`. Tạo: 2024-02-27. License: **MIT**
  (`LICENSE.txt`: "Copyright (c) Parth Sarthi").
  Commit/tag: không pin trong tài liệu này — khi vendor/clone phải ghi commit
  SHA vào run metadata (xem protocol Task 3). Truy cập 2026-09-19: cây file gồm
  `raptor/{EmbeddingModels,FaissRetriever,QAModels,RetrievalAugmentation,
  Retrievers,SummarizationModels,cluster_tree_builder,cluster_utils,
  tree_builder,tree_retriever,tree_structures,utils}.py`.
- requirements (`requirements.txt` tại master):
  `faiss-cpu, numpy==1.26.3, openai==1.3.3, scikit-learn,
  sentence-transformers==2.2.2, tenacity==8.2.3, tiktoken==0.5.1, torch,
  transformers==4.38.1, umap-learn==0.5.5, urllib3==1.26.6`.
- Embedding: mặc định OpenAI `text-embedding-ada-002`
  (`raptor/EmbeddingModels.py: OpenAIEmbeddingModel`); SBERT thay thế
  `sentence-transformers/multi-qa-mpnet-base-cos-v1` (`SBertEmbeddingModel`).
- Chunk: `tb_max_tokens=100` tokens (tokenizer `cl100k_base`)
  (`raptor/RetrievalAugmentation.py: RetrievalAugmentationConfig`).
- Tree: `tb_num_layers=5`; summary length 100 (`tb_summarization_length=100`).
- Summarizer: `gpt-3.5-turbo` (hoặc `text-davinci-003`), prompt:
  "Write a summary of the following, including as many key details as
  possible: {context}:" (`raptor/SummarizationModels.py`).
- Clustering (`raptor/cluster_utils.py`):
  UMAP global (`n_neighbors=int(sqrt(N-1))`, cosine, `dim=10`) → GMM chọn số
  cụm bằng BIC (`max_clusters=50`, `RANDOM_SEED=224`) → UMAP local
  (`num_neighbors=10`) → GMM local; soft assignment `threshold=0.1`;
  cụm vượt `max_length_in_cluster=3500` tokens bị recluster đệ quy.
- Retrieval (`raptor/tree_retriever.py` + `RetrievalAugmentation.retrieve`):
  mặc định **collapsed tree** (`collapse_tree=True`), cosine distance trên mọi
  node, `top_k=10`, `max_tokens=3500`; chế độ traversal: `top_k=5` /
  `threshold=0.5` mỗi tầng. QA mặc định `GPT3TurboQAModel`.
- Model IDs cần cho faithful run:
  `text-embedding-ada-002` (hoặc `sentence-transformers/multi-qa-mpnet-base-cos-v1`
  cho offline), summarizer `gpt-3.5-turbo`, QA `gpt-3.5-turbo`.
- Dataset requirements: bản gốc dùng QuALITY/HotpotQA/MultiHop; adapt sang
  QASPER cần cây **per-paper** (official code build một cây trên toàn corpus
  đưa vào — khác biệt bắt buộc số 1) + map node→QASPER paragraph IDs
  (khác biệt bắt buộc số 2).
- Hardware ước tính: không cần GPU nếu dùng OpenAI API + SBERT CPU; RAM
  ~8–16 GB cho cây 10–200 papers; FAISS CPU. Dùng local LLM thay GPT-3.5 thì
  cần GPU ≥24 GB VRAM.
- Khác biệt bắt buộc khi adapter sang QASPER: (1) một cây RAPTOR mỗi paper
  thay vì một cây toàn corpus; (2) giữ `paragraph_id` QASPER trên leaf để chấm
  official Evidence F1; (3) controlled run dùng chung generator của repo thay
  vì GPT-3.5 mặc định của official code (ghi rõ trong metadata).
- Blocker ghi nhận 2026-09-19: OpenAI API key của môi trường hết credits
  (`credit_balance_exhausted`); Gemini free-tier chỉ 20 req/ngày/model
  (đã cạn cho `gemini-3.6-flash`, `gemini-3.8-flash`/`3.7-flash` chập chờn
  503). Summarizer chạy bằng **local abstractive `facebook/bart-large-cnn`**
  trên GPU (`summarizer_provider="local"`), đúng official prompt ở tầng gọi;
  thay thế này được ghi trong `summarizer_provider/model` của mọi index/run
  metadata. Khi có quota API, chạy lại với `gpt-3.5-turbo` đúng official và
  đối chiếu.

## 2. LongRAG (Jiang–Ma–Chen, technical report)

- Paper: Z. Jiang, X. Ma, W. Chen, "LongRAG: Enhancing Retrieval-Augmented
  Generation with Long-context LLMs", arXiv:2406.15319 (v3, 2024-09-01).
  URL: https://arxiv.org/abs/2406.15319
  Website: https://tiger-ai-lab.github.io/LongRAG/
- Official repository: https://github.com/TIGER-AI-Lab/LongRAG
  (mô tả: 'Official repo for "LongRAG..."'). Branch `main`. Tạo: 2024-06-18.
  License: **MIT** (theo metadata repo GitHub). Commit: ghi SHA khi clone
  (xem protocol Task 3).
- requirements (`requirements.txt` tại main):
  `anthropic==0.28.1, datasets==2.19.1, google-generativeai==0.5.4,
  huggingface-hub==0.23.0, openai==1.14.2, tiktoken==0.6.0,
  transformers==4.38.1`.
- Long units: toàn bộ document hoặc gom documents liên quan, ≥4K tokens
  (trung bình 6K); **với Qasper: mỗi document là đúng 1 unit** (paper §3.1,
  Table 5: `Document` granularity, `k ∈ {1,2,5,10}`, Qasper F1 tốt nhất
  25.9% tại `k=2`).
- Retriever: toolkit **Tevatron**, base embedding **`bge-large-en-v1.5`**
  (BAAI, `BAAI/bge-large-en-v1.5`); điểm unit:
  `sim(q,g) ≈ max_{g'⊆g, |g'|=512 tok}(E_Q(q)^T E_C(g'))`; FAISS inner-product;
  **không re-rank**; top-k: 4–8 (grouped), ~10 (document), ~100+ (passage).
- Reader: **Gemini-1.5-Pro** và **GPT-4o**, input = concat top-k units
  (~30K tokens), zero-shot answer generation.
- Model IDs cần cho faithful run: `BAAI/bge-large-en-v1.5` (+ Tevatron/FAISS),
  reader `gemini-1.5-pro` hoặc `gpt-4o` (ghi snapshot ID thực tế vào metadata).
- Dataset requirements: NQ/HotpotQA (Wikipedia dumps + group_documents.sh);
  Qasper/MultiFieldQA-en: document-level units — trùng khớp đánh giá của repo.
- Hardware ước tính: embedding bge-large chạy CPU được (GPU nhanh hơn);
  reader là API nên không cần GPU local; index FAISS toàn bộ dev QASPER
  (~10–200 papers) vừa RAM laptop (vài GB).
- Khác biệt bắt buộc khi adapter sang QASPER: (1) corpus = papers trong frozen
  manifest, một unit = một paper (đúng paper, khác với section-units hiện tại);
  (2) map retrieved units → QASPER paragraph IDs qua char offsets để chấm
  official Evidence F1; (3) controlled run có thể thay reader bằng generator
  chung của repo (ghi rõ), primary run giữ long-context LLM reader.

## 3. Ghi chú license/model access

- Cả hai official repos đều MIT — được phép vendor/adapt với điều kiện giữ
  copyright notice.
- Model/API cần quyền truy cập: OpenAI API (RAPTOR default; SBERT offline là
  đường thay thế), Gemini/GPT-4o (LongRAG reader). Không hard-code key;
  dùng `config.local.json` (đã gitignore) hoặc biến môi trường.
- Tên định danh dùng trong repo sau Task 5/6: `B5_raptor_faithful`,
  `B6_longrag_faithful`; tên cũ đổi thành `B5_raptor_lightweight`,
  `B6_longrag_lightweight`.
