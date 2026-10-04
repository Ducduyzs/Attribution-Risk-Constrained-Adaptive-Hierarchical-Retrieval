# Tổng quan tài liệu và hướng cải tiến thuật toán

Ngày: 2026-10-03. Mục đích: đối chiếu công nghệ đang dùng với 40 bài báo liên quan,
rồi đề xuất hướng cải tiến cho **hướng A** (làm cho phương pháp thắng baseline).

Ký hiệu: ✔ = đã kiểm chứng bằng web search trong phiên này (có link);
○ = trích từ trí nhớ, **phải kiểm tra lại arXiv ID/venue trước khi đưa vào bài báo**.

## 0. Vấn đề cần giải (số liệu thật của dự án)

| Hiện tượng | Số liệu | Nguồn |
|---|---|---|
| Gate học được không có tín hiệu | dev AUC 0,537 / 0,511; CV theo bài báo ≈ 0,50 với RF/GB/HGB | `checkpoints/*_v7_final.metadata.json` |
| Overfit | train AUC 0,995 / 0,997 | như trên |
| Đề xuất thua baseline | citation F1: prior 0,214 < flat 0,243 < RAPTOR 0,277 | `artifacts/baselines/main` |
| Mở rộng tốn token mà không lợi | prior 2215 token so với flat 1573 | như trên |
| Retrieval là nút cổ chai | flat: hit@10 0,95 nhưng recall@5 0,52, recall@10 0,75 | B3 summary |

Kết luận chẩn đoán: (1) **nhãn** của gate quá nhiễu (phụ thuộc một lần sinh của LLM),
(2) **feature** không mang tín hiệu tổng quát, (3) **cơ chế** mở rộng làm generator trích
dẫn sai leaf (drift), (4) retrieval nền chỉ lấy được ~½ evidence ở top-5.

## 1. Danh mục tài liệu theo thành phần

### 1.1 Nền tảng và dữ liệu (stack đang dùng)

| # | Bài báo | Venue | Liên quan |
|---|---|---|---|
| 1 | ○ Docling Technical Report (Auer et al.) | arXiv 2408.09869, 2024 | Parser PDF đang dùng |
| 2 | ○ BGE M3-Embedding: multi-lingual, multi-functionality, multi-granularity (Chen et al.) | Findings ACL 2024 | Dense + sparse + multi-vector đang dùng |
| 3 | ○ ColBERTv2 (Santhanam et al.) | NAACL 2022 | Late interaction trong BGE-M3 |
| 4 | ○ QASPER (Dasigi et al.) | NAACL 2021 | Benchmark chính |
| 5 | ○ SciFact (Wadden et al.) | EMNLP 2020 | OOD rationale attribution |
| 6 | ✔ [PeerQA](https://aclanthology.org/2025.naacl-long.22) (Baumgärtner et al.) | NAACL 2025 | OOD QA khoa học; **decontextualization luôn cải thiện retrieval** |
| 7 | ✔ [SciDQA](https://arxiv.org/abs/2411.05338) (Singh, Sarkar, Cohan) | EMNLP 2024 | QA khoa học sâu, câu hỏi từ peer review |

### 1.2 Granularity và cấu trúc phân cấp (đối thủ trực tiếp)

| # | Bài báo | Venue | Ý chính / liên quan |
|---|---|---|---|
| 8 | ○ RAPTOR (Sarthi et al.) | ICLR 2024 | Cây tóm tắt đệ quy; **baseline mạnh nhất hiện tại (0,277)** |
| 9 | ○ LongRAG (Jiang et al.) | arXiv 2406.15319, 2024 | Đơn vị retrieval dài + reader long-context |
| 10 | ✔ [Dense X Retrieval](https://arxiv.org/abs/2312.06648) (Chen et al.) | EMNLP 2024 | Đơn vị *proposition* tốt hơn passage; lợi nhất ở ngân sách 100–200 từ |
| 11 | ✔ [Mix-of-Granularity](https://arxiv.org/abs/2406.00456) (Zhong et al.) | COLING 2025 | **Router chọn granularity theo query, huấn luyện bằng soft label** — gần nhất với gate của ta |
| 12 | ✔ [HiChunk](https://arxiv.org/abs/2509.11552) (Lu et al.) | ACL 2026 | Chunking phân cấp + **Auto-Merge thích ứng theo query**; benchmark evidence-dense. **Bắt buộc so sánh/trích dẫn** |
| 13 | ✔ [Late Chunking](https://arxiv.org/abs/2409.04701) (Günther et al.) | arXiv 2024 | Embed cả tài liệu rồi mới pool theo chunk; +2,7–3,6% nDCG, không cần train |
| 14 | ✔ [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (Anthropic) | Blog kỹ thuật 2024 (không peer-review) | Thêm 50–100 token ngữ cảnh vào chunk trước khi embed/BM25; giảm 49% lỗi top-20 |

### 1.3 Retrieval thích ứng và quyết định khi nào dùng context

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 15 | ○ Self-RAG (Asai et al.) | ICLR 2024 | Token phản tư quyết định retrieve/critique |
| 16 | ✔ [Adaptive-RAG](https://aclanthology.org/2024.naacl-long.389) (Jeong et al.) | NAACL 2024 | Classifier nhỏ chọn chiến lược theo độ phức tạp, **nhãn tự thu thập** |
| 17 | ○ Corrective RAG (Yan et al.) | arXiv 2401.15884, 2024 | Evaluator đánh giá chất lượng retrieval rồi sửa |
| 18 | ○ FLARE (Jiang et al.) | EMNLP 2023 | Retrieve chủ động khi độ tin cậy thấp |
| 19 | ✔ [Sufficient Context](https://arxiv.org/abs/2411.06037) (Joren et al.) | ICLR 2025 | Phân loại context *đủ/không đủ*; dùng để abstain có hướng dẫn, +2–10% độ chính xác khi trả lời |

### 1.4 Chọn lọc/nén context

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 20 | ✔ [FILCO](https://arxiv.org/abs/2311.08377) (Wang et al.) | arXiv 2023 | Lọc context mức câu; **nhãn oracle rẻ: string inclusion, lexical overlap, CXMI** |
| 21 | ○ RECOMP (Xu et al.) | ICLR 2024 | Nén context trích xuất/tóm tắt |
| 22 | ○ Lost in the Middle (Liu et al.) | TACL 2024 | Vị trí thông tin trong context ảnh hưởng mạnh |
| 23 | ○ The Power of Noise (Cuconasu et al.) | SIGIR 2024 | Tài liệu nhiễu có thể ảnh hưởng ngoài dự đoán |

### 1.5 Attribution / citation

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 24 | ○ ALCE (Gao et al.) | EMNLP 2023 | Benchmark sinh văn bản kèm citation |
| 25 | ✔ [Attribute or Abstain / LAB](https://arxiv.org/abs/2407.07799) (Buchmann et al.) | EMNLP 2024 | Tài liệu dài: **citation 1 bước tốt nhất cho model lớn; retrieval giúp model nhỏ**; evidence khó cho claim phức tạp |
| 26 | ✔ [LongCite](https://arxiv.org/abs/2409.02897) (Zhang et al.) | Findings ACL 2025 | Citation **mức câu**; pipeline coarse-to-fine |
| 27 | ✔ [Fine-grained rewards for citations](https://arxiv.org/abs/2402.04315) (Huang et al.) | ACL 2024 | Reward cục bộ mức câu cho citation quality |
| 28 | ✔ [AttributionBench](https://arxiv.org/abs/2402.15089) (Li et al.) | Findings ACL 2024 | Đánh giá attribution tự động rất khó (~80% macro-F1) |
| 29 | ○ RARR (Gao et al.) | ACL 2023 | Attribution hậu kiểm (post-hoc) và sửa |

### 1.6 Kiểm chứng (verifier)

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 30 | ✔ [MiniCheck](https://arxiv.org/abs/2404.10774) (Tang et al.) | EMNLP 2024 | Model 770M đạt mức GPT-4 khi kiểm grounding, rẻ hơn 400× |
| 31 | ○ AlignScore (Zha et al.) | ACL 2023 | Hàm alignment thống nhất cho factual consistency |
| 32 | ○ FActScore (Min et al.) | EMNLP 2023 | Chấm theo atomic fact |
| 33 | ○ TRUE (Honovich et al.) | NAACL 2022 | Đánh giá lại các metric factual consistency, NLI mạnh |

### 1.7 Kiểm soát rủi ro có bảo đảm

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 34 | ✔ [Conformal Factuality](https://proceedings.mlr.press/v235/mohri24a.html) (Mohri, Hashimoto) | ICML 2024 | Bảo đảm xác suất đúng 80–90% bằng back-off |
| 35 | ✔ [C-RAG](https://proceedings.mlr.press/v235/kang24a.html) (Kang et al.) | ICML 2024 | **Chứng nhận cận trên rủi ro sinh của RAG**, kể cả khi phân phối dịch chuyển |
| 36 | ○ Conformal Risk Control (Angelopoulos et al.) | ICLR 2024 | Kiểm soát kỳ vọng của loss bị chặn bất kỳ ≤ α |

### 1.8 Học policy từ phản hồi counterfactual / nhãn nhiễu

| # | Bài báo | Venue | Ý chính |
|---|---|---|---|
| 37 | ○ Counterfactual Risk Minimization (Swaminathan, Joachims) | ICML 2015 | Học policy từ logged bandit feedback |
| 38 | ○ Doubly Robust Policy Evaluation and Learning (Dudík, Langford, Li) | ICML 2011 | Ước lượng giá trị policy ít phương sai |
| 39 | ✔ [ROPO: Robust Preference Optimization](https://arxiv.org/abs/2404.04102) | arXiv 2024 | Học preference chịu nhiễu: giảm trọng số cặp có nhãn không chắc |
| 40 | ✔ [MCite-RL](https://arxiv.org/abs/2608.21808) | arXiv 2026 | Reward kết hợp retrieval, độ đúng và citation ở mức kết quả và quá trình |

## 2. Hướng cải tiến (xếp theo chi phí và khả năng thành công)

### H1 — Đổi mục tiêu của gate sang nhãn không phụ thuộc generator  *(rẻ, làm ngay)*
- **Vấn đề nó giải:** nhãn hiện tại = chênh lệch reward sau *một* lần LLM sinh → phương sai
  lớn; CV ≈ 0,50 cho thấy nhãn gần như ngẫu nhiên đối với feature.
- **Ý tưởng (từ FILCO #20, MoG #11, Sufficient Context #19):** nhãn tính trực tiếp từ gold
  evidence trên train, không gọi LLM: mở rộng parent/section có *thêm gold paragraph mới*
  (rescue) hay chỉ thêm leaf không phải gold (drift)? Dùng **soft label** = tỷ lệ
  rescue/(rescue+drift), giống MoG.
- **Đo:** CV theo bài báo và dev AUC; tiêu chí > 0,60.
- **Chi phí:** 0 API, chạy lại trên rollouts/hierarchy đã có.
- **Rủi ro:** nhãn "có thêm gold" khác với "trả lời tốt hơn"; cần báo cáo tương quan với
  end-to-end.

### H2 — Học xếp hạng theo cặp thay vì phân loại từng điểm  *(rẻ)*
- **Ý tưởng (#11, #37–#39):** với mỗi nhóm ứng viên, học *KEEP vs EXPAND vs SECTION cái nào
  tốt hơn* (pairwise/listwise) thay vì xác suất tuyệt đối; loss chịu nhiễu kiểu ROPO; nếu
  dùng nhãn end-to-end thì lấy trung bình k lần sinh (A1) hoặc ước lượng doubly robust.
- **Lý do:** so sánh tương đối trong cùng câu hỏi tự triệt tiêu thang đo riêng của bài báo
  — đúng chỗ gây overfit hiện tại.

### H3 — Feature tương đối theo query, bỏ feature mang thang đo của bài báo  *(rẻ)*
- **Ý tưởng:** thay feature tuyệt đối (độ dài, số token, mật độ thô) bằng feature *tương
  đối trong query*: chênh điểm cross-encoder parent so với child tốt nhất, hạng trong
  danh sách rerank, entropy điểm thành viên, tỷ lệ token mới so với leaf đã có. Đây là A3.
- **Bước xa hơn (MiniCheck #30):** fine-tune một cross-encoder nhỏ làm gate trực tiếp
  "parent này có thêm evidence cho query không" — thường mạnh hơn RF trên feature tay.

### H4 — Tách "được đọc" và "được trích dẫn" khi mở rộng  *(rẻ, đánh trúng drift)*
- **Vấn đề nó giải:** drift sinh ra vì generator trích dẫn leaf mới do mở rộng đưa vào.
- **Ý tưởng (Dense X #10, LongCite #26, RARR #29):** mở rộng chỉ để *cung cấp ngữ cảnh*;
  chỉ các leaf được retrieve ban đầu (hoặc qua ngưỡng verifier cao hơn) mới được phép
  trích dẫn. Hiện thực bằng enum citation hợp lệ theo từng request (đã có hạ tầng strict
  schema) hoặc re-attribution hậu kiểm về leaf.
- **Kỳ vọng:** giữ lợi ích "rescue" cho câu trả lời mà chặn phần "harmful drift" trong
  citation. Đây là đóng góp thuật toán mới, dễ kiểm chứng bằng decomposition sẵn có.

### H5 — Biến "risk-constrained" thành bảo đảm thống kê bằng conformal  *(trung bình, giá trị cao)*
- **Ý tưởng (C-RAG #35, Conformal Factuality #34, Conformal Risk Control #36):** không cần
  gate dự đoán hoàn hảo. Hiệu chỉnh ngưỡng mở rộng trên tập calibration sao cho
  **tỷ lệ harmful drift ≤ α với bảo đảm hữu hạn mẫu**; ngoài ngưỡng thì back-off về flat.
- **Lý do mạnh:** khớp đúng tên đề tài ("Attribution-Risk-Constrained"), cho claim có định
  lý thay vì phụ thuộc vào AUC; ngay cả gate yếu vẫn cho bảo đảm hợp lệ (chỉ là mở rộng
  ít hơn). Báo cáo đường cong rủi ro–độ bao phủ theo α.

### H6 — Nâng retrieval nền  *(rẻ đến trung bình, lợi chắc chắn)*
- **Vấn đề:** recall@5 = 0,52 nghĩa là một nửa evidence không vào top-5; gate không thể bù.
- **Ý tưởng:** decontextualization/contextual chunk (PeerQA #6, Contextual Retrieval #14),
  late chunking (#13), đơn vị proposition hoặc câu (#10).
- **Lưu ý thiết kế:** cải tiến này phải áp dụng **cho mọi baseline** để so sánh công bằng;
  nó nâng nền chứ không tự tạo khoảng cách với RAPTOR.

### H7 — Verifier mạnh hơn và abstain theo "sufficient context"  *(trung bình)*
- Thay NLI bằng MiniCheck/AlignScore (#30, #31) — rẻ, mạnh hơn, giảm sai số khi lọc claim.
- Dùng phân loại "context đủ/không đủ" (#19) để abstain; cải thiện độ chính xác chọn lọc
  (AURC) — metric đã có trong code.

### H8 — Huấn luyện generator theo reward citation  *(đắt, để sau)*
- Fine-grained reward / RL (#27, #40) hoặc SFT kiểu LongCite (#26). Hiệu quả nhưng cần
  model mở và GPU; không phù hợp giai đoạn hiện tại.

## 3. Đề xuất thứ tự thực hiện

| Bước | Nội dung | Chi phí API | Thời gian ước | Tiêu chí dừng/tiếp |
|---|---|---|---|---|
| 1 | H1 + H3 trên rollouts sẵn có (A2 + A3) | 0 | 2–3 ngày | CV theo bài báo > 0,58, dev AUC > 0,60 |
| 2 | H4 tách đọc/trích dẫn, chạy decomposition trên 75 câu dev | thấp | 2–3 ngày | harmful drift giảm, citation F1 ≥ flat |
| 3 | H5 conformal threshold trên dev (chia calibration/eval theo bài báo) | 0 | 2 ngày | drift ≤ α đạt thực nghiệm |
| 4 | H6 cho mọi hệ thống, chạy lại bảng 75 câu | trung bình | 3–5 ngày | khoảng cách với RAPTOR |
| 5 | Nếu bước 1 thất bại: giữ H4 + H5 (không cần gate mạnh) hoặc chuyển hướng B | — | — | — |

**Nhận định:** H4 và H5 là hai hướng hứa hẹn nhất vì **không phụ thuộc vào việc gate học
được tín hiệu** — điểm yếu đã được chứng minh của hệ thống hiện tại. H4 tấn công trực tiếp
cơ chế gây drift; H5 biến tiêu đề "risk-constrained" thành bảo đảm có thể chứng minh. Hai
hướng này cùng tạo đóng góp khác biệt rõ với HiChunk (#12) và MoG (#11), vốn chỉ tối ưu
retrieval/granularity mà không kiểm soát rủi ro attribution.
