# V10 — Kết quả xác nhận Agreement Ranking (đăng ký trước)

2026-10-04. Protocol: `analysis/v10_selector_design.md` (commit `44a6014`); manifest đóng
băng trước khi chạy (commit `5efec97`): 100 bài QASPER dev chưa từng dùng, 1 câu/bài.
3 nhánh × 3 ngân sách × 100 câu = 900 lượt; gpt-4o-mini; verifier chỉ đọc phần hiển thị;
pipeline v9 dùng nguyên. Test 180 câu không mở. Dữ liệu: `artifacts/v10_confirm/`.

## 1. Giả thuyết chính (Holm trên 2 giả thuyết)

Chỉ số: official Evidence F1, trung bình 3 ngân sách mỗi câu; kiểm định sign-flip theo bài.

| Giả thuyết | Chênh lệch | CI 95% | p (một phía) | p (Holm) | Kết luận |
|---|---:|---|---:|---:|---|
| H-a: agreement không kém RAPTOR quá 0,02 | +0,032 | [−0,006; +0,072] | 0,005 | **0,010** | **Được ủng hộ** |
| H-b: agreement tốt hơn all_leaf | +0,023 | [−0,018; +0,065] | 0,149 | 0,149 | **Không được ủng hộ** |

## 2. Bảng theo ngân sách

| Nhánh | Ngân sách | Evidence F1 | Phủ ký tự gold | Leaf F1 | Answer F1 | Token |
|---|---:|---:|---:|---:|---:|---:|
| raptor | 512 | 0,285 | 0,307 | 0,240 | 0,224 | 420 |
| raptor | 1024 | 0,300 | 0,408 | 0,290 | 0,218 | 894 |
| raptor | 2048 | 0,327 | 0,480 | 0,328 | 0,221 | 1518 |
| all_leaf | 512 | 0,270 | 0,297 | 0,231 | 0,207 | 412 |
| all_leaf | 1024 | 0,335 | 0,450 | 0,302 | 0,217 | 881 |
| all_leaf | 2048 | 0,334 | 0,557 | 0,312 | 0,225 | 1554 |
| agreement | 512 | **0,329** | 0,284 | 0,245 | 0,224 | 417 |
| agreement | 1024 | **0,350** | 0,424 | 0,301 | 0,227 | 883 |
| agreement | 2048 | 0,328 | 0,553 | 0,340 | 0,204 | 1528 |

(Chỉ số phụ — chỉ mô tả, theo protocol.)

## 3. Chi phí

| | RAPTOR | Agreement |
|---|---:|---:|
| Dựng index (100 bài, RTX 3090) | **7044 s** (cây + tóm tắt BART) | **17 s** (SBERT) |
| Truy vấn | 3,7 s tổng | gộp trong 17 s |

Agreement rẻ hơn khoảng **400 lần** ở bước index, không cần gọi model tóm tắt.

## 4. Diễn giải trung thực

- **Xác nhận được**: Agreement Ranking không kém RAPTOR (biên 0,02) trên bài báo mới,
  với chi phí index nhỏ hơn ~400 lần. Ước lượng điểm còn cao hơn RAPTOR (+0,032) nhưng
  CI chứa 0 → **không** được tuyên bố vượt RAPTOR.
- **Không xác nhận được**: Agreement tốt hơn reranking toàn bài (H-b). Phần đóng góp riêng
  của tín hiệu SBERT chưa được chứng minh.
- **Phát hiện trên dev không tái lập**: trên 100 bài mới, RAPTOR **không** đóng gói gold tốt
  hơn all_leaf (phủ ký tự 0,408 so với 0,450 @1024; 0,480 so với 0,557 @2048). Lợi thế
  "RAPTOR đóng gói tốt hơn" quan sát trên 30 bài dev nhiều khả năng do mẫu nhỏ/dev đã
  dùng nhiều lần. Do đó giả thuyết cơ chế ("lợi thế RAPTOR đến từ SBERT") yếu đi: trên dữ
  liệu mới, cây RAPTOR không đem lại lợi ích so với rerank toàn bài.
- Hệ quả thực tiễn: với bài báo khoa học ngắn (QASPER), **cây phân cấp kiểu RAPTOR không
  đáng chi phí** — rerank toàn bài hoặc Agreement Ranking đạt ngang hoặc hơn với chi phí
  index thấp hơn hàng trăm lần.
