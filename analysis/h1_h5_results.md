# Kết quả H1–H5 (2026-10-04)

Toàn bộ chạy offline, không gọi API. Tái lập:
`python scripts/replay_policies.py` (→ `analysis/replay_policies.json`) và
`python scripts/gate_experiments.py` (→ `analysis/gate_experiments.json`).

## 0. Bug verifier tìm được trong quá trình làm (đã sửa)

Guard phủ định/số liệu (`_lexical_conflict`) được áp lên **toàn bộ leaf 220 token** và
**phủ quyết cả claim có NLI entailment cao**. Theo đặc tả v7 §8 nó chỉ được chặn bước
lexical fallback. Ảnh hưởng: ~25% claim bị loại oan ở mọi hệ thống, ~60% trong số đó
bị loại trên chính leaf gold. Đã sửa trong `src/edahr/verification.py`.

> **Lưu ý generator:** các row này do **Qwen2.5-7B local** sinh (hard-code trong
> `scripts/run_faithful_main.py`), dù `run_metadata`/`config.json` ghi `gpt-4o-mini`.

Replay offline (`src/edahr/replay.py`) tái tạo đúng 375/375 row đã lưu với luật cũ,
sau đó chấm luật mới — 75 câu dev / 30 bài, Answer F1 dùng tham chiếu official:

| Hệ thống | Citation F1 cũ → mới | Evidence F1 cũ → mới | Answer F1 cũ → mới |
|---|---|---|---|
| B5 RAPTOR faithful | 0,277 → **0,386** | 0,335 → **0,403** | 0,199 → 0,244 |
| B3 flat neural | 0,243 → 0,341 | 0,297 → 0,386 | 0,199 → **0,274** |
| B4 static hierarchy | 0,243 → 0,323 | 0,333 → 0,395 | 0,212 → 0,266 |
| edahr_prior | 0,214 → 0,286 | 0,293 → 0,351 | 0,186 → 0,256 |
| B6 LongRAG | 0,162 → 0,254 | 0,221 → 0,306 | 0,177 → 0,238 |

Mức tăng có CI 95% theo bài báo không chứa 0 ở 13/15 cặp (citation F1 cả 5 hệ thống).
Thứ hạng không đổi: phương pháp đề xuất vẫn thấp nhất trong nhóm phân cấp.
**Mọi artifact và nhãn rollout cũ đều được tạo với bug này.**

## 1. H4 (tách đọc/trích dẫn) và H5 (conformal risk control)

Đã cài: `expanded_citation_threshold` (config + verifier), `src/edahr/risk_control.py`
(CRC theo đơn vị bài báo, loss drift bị chặn và đơn điệu — đã kiểm), cross-fit 5 fold.

Kết quả: **gần như không có tác dụng**, vì harmful drift trên 75 câu dev chỉ 0–1 leaf
mỗi hệ thống (rescue = 0). Ở cấu hình hiện tại (rerank_k = 24, parent 4 child) tập
retrieve đã phủ gần hết leaf của parent được mở rộng → không có drift để kiểm soát.

## 2. H1–H3 (gate có học được không?)

Rollouts v7: train 2881 nhóm / 534 bài, dev 924 nhóm / 164 bài.

| Nhãn | Feature | Học | CV AUC train | Dev AUC [CI 95% theo bài] |
|---|---|---|---|---|
| T0 cũ (parent) | F0 | điểm (hiện tại) | 0,510 | 0,543 [0,490; 0,595] |
| T0 cũ (parent) | F0 | **cặp (H2)** | 0,533 | 0,585 [0,533; 0,637] |
| T0 cũ (parent) | **F3 (H3)** | điểm | 0,576 | 0,565 [0,516; 0,612] |
| T0 cũ (section) | F3 | cặp | 0,549 | 0,586 [0,532; 0,639] |
| T1 (H1, parent) | F0/F3 | — | 0,87–0,93 | 0,83–0,94 *(rò rỉ cấu trúc)* |
| T1 (H1), **có điều kiện** | F0/F3 | — | — | 0,49–0,63, CI chứa 0,5 |

- H2/H3 nâng AUC trên nhãn cũ từ ~0,54 lên ~0,58 (CI không chứa 0,5) — **chưa đạt
  ngưỡng 0,60**.
- H1 đạt 0,9 chỉ vì luật tầm thường "node còn leaf chưa retrieve" đã cho AUC 0,93–0,94.
  Chỉ xét nhóm mà mở rộng thật sự thêm leaf mới, AUC ≈ ngẫu nhiên (8–12 mẫu dương dev).

## 3. Phát hiện gốc rễ

Mở rộng **parent chỉ thêm được gold leaf chưa retrieve ở ~1% nhóm** (train 28/2881,
dev 8/924); section ~1,2–1,3%. Nhãn v5 lại gán ~14–15% là "nên mở rộng" → phần lớn là
nhiễu từ một lần sinh LLM (và bug guard). Evidence mà retrieval bỏ sót hiếm khi nằm cạnh
evidence đã retrieve, nên mở rộng theo cấu trúc phân cấp gần như không có dư địa cứu
evidence trên QASPER ở cấu hình này. Điều này giải thích đồng thời: gate ≈ ngẫu nhiên,
drift ≈ 0, và static/prior ≈ flat.

## 4. Hệ quả cho hướng A/B

Hướng A theo khung hiện tại (gate mở rộng thắng baseline) không có dư địa trên QASPER.
Các lựa chọn còn lại:
1. Đổi bài toán sang chế độ **ngân sách chặt** (top-k nhỏ, ví dụ 3–5 leaf) — nơi mở rộng
   mới có thể cứu evidence; cần đo lại tỷ lệ "mở rộng thêm gold" trước khi đầu tư.
2. Hướng B: báo cáo kết quả âm có định lượng (headroom ~1%, nhãn end-to-end nhiễu,
   drift ≈ 0) + benchmark có kiểm soát + bug verifier như bài học phương pháp luận.
