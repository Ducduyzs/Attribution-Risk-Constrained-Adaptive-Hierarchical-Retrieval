# V9 — Kết quả ma trận chọn evidence theo ngân sách (DEV, thăm dò)

2026-10-04. 12 nhánh × 3 ngân sách (512 / 1024 / 2048 token cl100k, tính cả header)
× 75 câu = 2700 lượt; gpt-4o-mini, verifier chỉ đọc đoạn thực sự hiển thị. Thiết kế:
`analysis/v9_directions_protocol.md`. Audit thực thi: `analysis/v9_execution_audit.json`
(đủ ma trận, không vượt ngân sách, khớp evaluator official, test chưa mở). Chi phí API
~0,6 USD (2,93M token vào, 0,28M ra). **Dev 75 câu đã dùng nhiều lần → chỉ thăm dò.**
Số v9 không so trực tiếp với v8 (tokenizer, giới hạn 120 từ, verify phần hiển thị).

## 1. Kết quả chính

| Nhánh | Leaf F1 512 / 1024 / 2048 | Evidence F1 512 / 1024 / 2048 | Answer F1 512 / 1024 / 2048 |
|---|---|---|---|
| **raptor** | **0,335 / 0,366 / 0,407** | **0,386** / 0,367 / 0,419 | **0,236** / 0,222 / **0,237** |
| flat24 | 0,271 / 0,297 / 0,377 | 0,306 / 0,334 / 0,374 | 0,212 / 0,217 / 0,191 |
| all_leaf | 0,271 / 0,300 / 0,379 | 0,310 / 0,339 / 0,365 | 0,212 / 0,216 / 0,178 |
| contextual24 | 0,281 / 0,331 / 0,332 | 0,319 / 0,331 / 0,331 | 0,216 / 0,204 / 0,204 |
| contextual_all | 0,286 / 0,328 / 0,318 | 0,326 / 0,325 / 0,323 | 0,218 / 0,203 / 0,203 |
| mmr | 0,301 / 0,346 / 0,388 | 0,366 / 0,352 / 0,372 | 0,229 / 0,208 / 0,204 |
| coverage | 0,266 / 0,330 / 0,371 | 0,325 / 0,358 / 0,383 | 0,213 / 0,220 / 0,234 |
| sentence | 0,311 / 0,321 / 0,309 | **0,386 / 0,381** / 0,340 | 0,213 / 0,212 / 0,203 |
| window | 0,283 / 0,295 / 0,351 | 0,363 / 0,355 / **0,429** | 0,232 / **0,229** / **0,237** |
| dependency | 0,250 / 0,272 / 0,331 | 0,282 / 0,309 / 0,338 | 0,189 / 0,208 / 0,186 |
| evidence_first | 0,232 / 0,330 / 0,374 | 0,249 / 0,360 / 0,378 | 0,213 / 0,218 / 0,230 |
| dependency_first | 0,219 / 0,265 / 0,264 | 0,259 / 0,289 / 0,237 | 0,189 / 0,186 / 0,192 |

Holm trên 99 so sánh với RAPTOR: chỉ **2** có ý nghĩa, đều **kém hơn** RAPTOR
(evidence_first@512 evidence F1 −0,138; dependency_first@2048 −0,181). Không nhánh nào
vượt RAPTOR có ý nghĩa. Lưu ý: ngân sách là trần — RAPTOR dùng ít token nhất ở 2048
(1545 so với 1810–1953 của các nhánh câu/MMR).

## 2. Chẩn đoán thất thoát evidence (ngân sách 1024)

| Nhánh | Gold trong pool | Leaf gold được đóng gói | Phủ ký tự paragraph gold | Generator trích gold | Còn sau verify |
|---|---:|---:|---:|---:|---:|
| raptor | 0,791 | 0,496 | 0,512 | 0,390 | 0,336 |
| flat24 | 0,942 | 0,446 | 0,453 | 0,343 | 0,276 |
| all_leaf | 0,987 | 0,448 | 0,458 | 0,345 | 0,279 |
| contextual_all | 0,987 | 0,519 | 0,519 | 0,397 | 0,305 |
| mmr | 0,987 | 0,442 | 0,454 | 0,354 | 0,325 |
| sentence | 0,987 | 0,755 | 0,410 | 0,515 | 0,320 |
| dependency | 0,987 | 0,652 | 0,322 | 0,383 | 0,263 |

**Nút thất thoát lớn nhất là bước đóng gói (pool → context):** gold có trong pool
94–99% nhưng chỉ ~45% leaf gold vào được context 1024 token. RAPTOR có pool *kém hơn*
(0,79) nhưng đóng gói được nhiều gold hơn (0,50) — lợi thế của RAPTOR nằm ở việc chọn
cái gì đưa vào context, không phải ở recall của pool. Rerank toàn bài (all_leaf) gần
như không đổi gì so với top-24.

## 3. Đối chứng thành phần

- **MMR − all_leaf**: Evidence F1 +0,056 @512 (p=0,04), các mức khác dương nhưng n.s.
  → giảm trùng lặp giúp ở ngân sách chặt.
- **Coverage − all_leaf**: Answer F1 +0,056 @2048 (p=0,01), còn lại n.s.
- **Contextual reranker − all_leaf**: +0,015–0,028 leaf F1 @512/1024 (n.s.), **−0,060 @2048**
  (p=0,03) → không ổn định.
- **Evidence-first − all_leaf**: Evidence F1 −0,061 @512 (p<0,01); Answer F1 +0,052 @2048
  (p=0,01) → hại ở ngân sách chặt, có thể giúp câu trả lời khi context rộng.
- **Window − sentence**: @2048 Evidence F1 +0,089 (p=0,09); @512/1024 âm nhẹ, n.s.
- **Dependency − window**: âm ở mọi ngân sách (Evidence F1 −0,05 đến −0,09); bộ nhóm
  heuristic làm *giảm* phủ ký tự paragraph gold (0,32 so với 0,43). **Giả thuyết "bảo
  toàn phụ thuộc" ở dạng heuristic hiện tại không được ủng hộ.**

## 4. Kết luận

1. Không nhánh nào vượt RAPTOR; dependency (ứng viên chính của hướng mới) kém nhất.
2. Thất thoát chính nằm ở **chọn/đóng gói context dưới ngân sách**, không ở retrieval
   pool, không ở reranker. Đây là chẩn đoán mới có giá trị.
3. Các tín hiệu nhỏ nhưng nhất quán: MMR ở ngân sách chặt; sentence-level cho Evidence
   F1 ở 512/1024; window ở 2048. Đều n.s. so với RAPTOR.
4. Cần đánh giá người (file `artifacts/v9_results/human_audit_unlabelled.json` chưa gán
   nhãn) trước mọi kết luận về chất lượng attribution thực.
