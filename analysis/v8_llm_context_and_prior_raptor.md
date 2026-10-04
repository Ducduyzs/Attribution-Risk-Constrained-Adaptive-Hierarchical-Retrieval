# Ngữ cảnh `llm` và `prior_raptor` (2026-10-04)

75 câu dev / 30 bài, generator gpt-4o-mini. Chênh lệch so với bản không ngữ cảnh,
CI 95% bootstrap theo bài báo. Chi tiết: `analysis/v8_llm_vs_none.json`.

## 1. `prior_raptor` (prior mở rộng + verify trên retrieval RAPTOR, không ngữ cảnh)

| Hệ thống | Citation F1 | Evidence F1 | Answer F1 | Token |
|---|---:|---:|---:|---:|
| B5 RAPTOR | **0,436** | **0,404** | **0,244** | 1452 |
| prior_raptor | 0,397 | 0,389 | 0,226 | 1871 |
| prior | 0,387 | 0,396 | 0,231 | 2215 |

prior_raptor − RAPTOR: citation −0,038 [−0,095; +0,013]. Cùng retrieval (recall@5
0,606), nên phần kém đi đến từ bước mở rộng — khớp với phát hiện dư địa mở rộng ~1%.

## 2. Ngữ cảnh `llm` (Anthropic Contextual Retrieval, 870 chunk, 0,43 USD)

| Hệ thống | recall@5 | Citation F1 | Evidence F1 | Answer F1 |
|---|---|---|---|---|
| flat / static / prior | +0,010 [+0,000; +0,026] | +0,008 / −0,003 / −0,001 (n.s.) | +0,007 / −0,011 / +0,009 (n.s.) | +0,001 / −0,010 / +0,008 (n.s.) |
| RAPTOR | **−0,049** [−0,102; −0,005] | −0,063 [−0,134; −0,001] | −0,057 (n.s.) | **−0,033** [−0,067; −0,007] |
| prior_raptor | −0,049 | −0,020 (n.s.) | −0,042 (n.s.) | −0,004 (n.s.) |

## 3. Vì sao ngữ cảnh ở tầng embedding không có tác dụng

- Mỗi bài có 11–56 chunk (trung vị 26), còn `candidate_k = 80`: retrieval tầng đầu luôn
  trả **toàn bộ bài báo**. 24 chunk đầu (`rerank_k`) vào reranker, và 43% câu hỏi đưa
  cả bài vào reranker.
- Reranker (`bge-reranker-v2-m3`) chấm trên **văn bản thô** → thứ hạng cuối gần như do
  reranker quyết định; embedding (dù có ngữ cảnh) chỉ chọn vài chunk bị loại khỏi pool.
- RAPTOR bị hại vì chỉ leaf có ngữ cảnh, còn node tóm tắt thì không, làm lệch cân bằng
  leaf/summary trong collapsed-tree retrieval (giống chế độ `title`).

Hệ quả: trên QASPER (truy hồi trong một bài, bài ngắn), muốn cải thiện retrieval phải
tác động vào **reranker** (đưa ngữ cảnh vào đầu vào reranker hoặc reranker mạnh hơn),
không phải embedding.

## Chi phí

Ngữ cảnh 0,43 USD (89% prompt cache); hai lượt chạy ~0,3 USD API; ~1,5 giờ RTX 3090.
