# Contextual chunk — chế độ `title` (2026-10-04)

75 câu dev / 30 bài, generator gpt-4o-mini, verifier đã sửa. So sánh từng hệ thống
có (`chunk_context="title"`, `artifacts/baselines/v8_dev_title/`) và không có ngữ
cảnh (bảng v8). Chênh lệch = title − none, CI 95% bootstrap theo bài báo, p sign-flip.
Chi tiết: `analysis/v8_title_vs_none.json`.

| Hệ thống | recall@5 | MRR | Citation F1 | Evidence F1 | Answer F1 |
|---|---|---|---|---|---|
| flat / static / prior (cùng index) | +0,002 [+0,000; +0,007] | −0,007 [−0,021; +0,000] | flat −0,016; static −0,000; prior +0,006 (đều n.s.) | −0,010 / −0,009 / −0,003 (n.s.) | +0,008 / **−0,014** / +0,008 |
| RAPTOR | **−0,064 [−0,115; −0,016]** | −0,012 (n.s.) | **−0,077 [−0,132; −0,027]** | **−0,046 [−0,091; −0,005]** | −0,010 (n.s.) |

## Kết luận

- **Không có tác dụng với BGE-M3** (flat/static/prior): retrieval QASPER được giới hạn
  trong *một bài báo*, nên tên bài báo giống hệt nhau ở mọi chunk ứng viên — không
  mang thông tin phân biệt. Tên mục vốn đã có trong `embedding_text` mặc định. Kết quả
  này đoán trước được từ thiết kế; lẽ ra nên loại chế độ `title` trước khi chạy.
- **Làm hại RAPTOR**: leaf được thêm tiền tố giống nhau trong khi các node tóm tắt vẫn
  embed văn bản thô, làm lệch cân bằng leaf/summary trong collapsed-tree retrieval.
- Chỉ ngữ cảnh **riêng cho từng chunk** (chế độ `llm`) mới có thể giúp phân biệt trong
  cùng một bài báo. Chưa chạy; ước tính 0,39–0,72 USD sinh ngữ cảnh cho dev.

Chi phí lượt này: API ~0,2 USD; ~1 giờ RTX 3090.
