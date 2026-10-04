# V10 — Bộ chọn context theo đồng thuận hai tín hiệu (thiết kế + đăng ký trước)

2026-10-04. Phân tích offline (CPU, không API) trên bảng xếp hạng v9 đã cache.
Script: `scripts/v10_packing_analysis.py`, `scripts/v10_selector_search.py`.
Kết quả: `analysis/v10_packing_analysis.json`, `analysis/v10_selector_search.json`.

## 1. Vì sao RAPTOR đóng gói nhiều gold hơn (DEV 74 câu có gold)

Các nhánh flat/all_leaf/raptor của v9 dùng **cùng bộ đóng gói**; chỉ tập ứng viên khác.

| | all_leaf | RAPTOR pool |
|---|---:|---:|
| Số ứng viên / bài | 29,3 | 12,8 |
| Hạng của gold tốt nhất (theo cross-encoder) | 3,73 | 2,72 |
| Gold trong top-3 | 66% | 72% |
| Leaf không-gold xếp trên gold | 2,73 | RAPTOR loại ~1,0 |

RAPTOR loại bớt leaf mà cross-encoder chấm cao nhưng tín hiệu ngữ nghĩa khác không ủng hộ.

## 2. Kiểm định giả thuyết về "tín hiệu thứ hai" (độ phủ ký tự paragraph gold)

| Bộ chọn | 512 | 1024 | 2048 | Kết luận |
|---|---:|---:|---:|---|
| RAPTOR (v9) | 0,411 | 0,519 | 0,690 | tham chiếu |
| chỉ cross-encoder | 0,325 | 0,464 | 0,678 | |
| RRF + BGE-M3 / sentence / contextual reranker | 0,318–0,371 | 0,475–0,527 | 0,628–0,691 | thu hẹp một phần |
| Lọc cứng 44% theo BGE / ctx / sentence | 0,323–0,329 | 0,460–0,471 | 0,629–0,663 | không giúp |
| Hỗ trợ lân cận (support) | 0,320–0,327 | 0,423–0,429 | 0,611–0,658 | **bác bỏ** |
| Phân cụm không tóm tắt (cluster) | 0,344 | 0,454 | 0,637 | **bác bỏ** |
| **RRF(cross-encoder, SBERT query–leaf)** | **0,451** | **0,537** | 0,689 | **ứng viên** |

`rerank+sbert` so với RAPTOR: +0,041 (p=0,25) @512, +0,018 (p=0,61) @1024, −0,001 @2048
— chưa có ý nghĩa thống kê; đã thử ~20 biến thể trên cùng dev → **có thiên lệch chọn lọc**.

**Giả thuyết cơ chế:** lợi thế của RAPTOR trên QASPER đến từ tín hiệu tương đồng
câu hỏi–leaf của SBERT `multi-qa-mpnet-base-cos-v1` (bổ sung cho cross-encoder), không
phải từ cây, tóm tắt hay phân cụm.

## 3. Bộ chọn đề xuất: Agreement Ranking (AR)

1. Cross-encoder chấm mọi leaf của bài (như all_leaf).
2. SBERT tính cosine câu hỏi–leaf (cùng model RAPTOR dùng; 110M tham số).
3. Thứ tự = RRF (k=60) của hai hạng; điểm cho bộ đóng gói = phân bố điểm của
   cross-encoder gán theo thứ tự mới (bộ đóng gói giữ nguyên).
4. Không tham số học, không cây, không tóm tắt BART, không gọi LLM khi index.

Chi phí so với RAPTOR: bỏ hoàn toàn bước dựng cây (~1,5 phút GPU / bài với BART).

## 4. Protocol xác nhận (đăng ký trước khi xem dữ liệu mới)

- **Tập**: QASPER dev, chỉ các bài **chưa dùng** (không thuộc 30 bài baseline-dev, không
  thuộc 164 bài frozen-dev của gate v7): 104 bài / 267 câu có gold. Chọn 1 câu mỗi bài,
  seed 20261004, tối đa 100 bài. Ghi manifest + SHA-256 trước khi chạy. Test 180 câu
  vẫn không mở.
- **Nhánh** (đóng băng): `raptor`, `all_leaf`, `agreement` (RRF k=60, SBERT như trên).
  Không thêm/sửa nhánh sau khi xem kết quả.
- **Ngân sách**: 512 / 1024 / 2048 (cl100k, gồm header), generator gpt-4o-mini, verifier
  chỉ đọc phần hiển thị — giống hệt v9.
- **Chỉ số chính**: official Evidence F1, `agreement` − `raptor`, gộp 3 ngân sách theo
  câu hỏi (trung bình 3 ngân sách mỗi câu), kiểm định sign-flip theo bài báo.
- **Giả thuyết**: (H-a) agreement không kém RAPTOR quá 0,02 (non-inferiority);
  (H-b) agreement > all_leaf. Phụ: độ phủ ký tự gold, Leaf F1, Answer F1, token.
- **Họ so sánh** cho Holm: 2 giả thuyết chính. Các chỉ số phụ chỉ báo cáo mô tả.
- **Dừng**: nếu H-a thất bại, không tinh chỉnh tiếp trên tập này; báo cáo kết quả âm.

Ước tính: GPU ~2,5 giờ RTX 3090 (dựng ~100 cây RAPTOR cho nhánh tham chiếu, rerank,
NLI); API ~0,3 USD (3 nhánh × 3 ngân sách × ~100 câu, có cache).
