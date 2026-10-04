# V10 — Kết quả test một lần (đăng ký trước)

2026-10-04. Protocol đóng băng: `analysis/v10_test_protocol.md` (commit `c469658`); code
chạy có SHA-256 khớp commit. Tập test QASPER 180 câu / 180 bài, mở đúng một lần.
3 nhánh × 3 ngân sách × 180 câu = 1620 lượt; gpt-4o-mini; verifier chỉ đọc phần hiển thị.
Dữ liệu: `artifacts/v10_test/`. Kịch bản chạy không lỗi, không phải chạy lại phần nào.

## 1. Giả thuyết chính (Holm trên 3)

Chỉ số: official Evidence F1, trung bình 3 ngân sách mỗi câu; sign-flip theo bài; biên 0,02.

| Mã | Giả thuyết | Chênh lệch | CI 95% | p (một phía) | p (Holm) | Kết luận |
|---|---|---:|---|---:|---:|---|
| H-a | agreement không kém RAPTOR | +0,020 | [−0,007; +0,048] | 0,002 | **0,006** | **Được ủng hộ** |
| T2 | all_leaf không kém RAPTOR | −0,008 | [−0,037; +0,023] | 0,201 | 0,201 | **Không được ủng hộ** |
| H-b | agreement tốt hơn all_leaf | +0,028 | [+0,001; +0,056] | 0,026 | 0,052 | **Không được ủng hộ** (sát ngưỡng sau Holm) |

## 2. Theo ngân sách (chỉ mô tả)

| Nhánh | Ngân sách | Evidence F1 | Phủ ký tự gold | Leaf F1 | Answer F1 | Token context |
|---|---:|---:|---:|---:|---:|---:|
| raptor | 512 | 0,346 | 0,398 | 0,327 | 0,213 | 410 |
| raptor | 1024 | 0,353 | 0,558 | 0,366 | 0,230 | 894 |
| raptor | 2048 | 0,372 | 0,613 | 0,406 | 0,223 | 1495 |
| all_leaf | 512 | 0,310 | 0,378 | 0,302 | 0,207 | 413 |
| all_leaf | 1024 | 0,361 | 0,560 | 0,364 | 0,216 | 889 |
| all_leaf | 2048 | 0,378 | 0,683 | 0,413 | 0,224 | 1544 |
| agreement | 512 | **0,357** | **0,443** | **0,347** | 0,214 | 412 |
| agreement | 1024 | **0,384** | **0,601** | **0,407** | 0,229 | 884 |
| agreement | 2048 | **0,390** | **0,707** | 0,412 | **0,238** | 1522 |

Agreement có ước lượng điểm cao nhất về Evidence F1 và phủ gold ở cả ba ngân sách;
chênh lệch từng ngân sách với RAPTOR (+0,011 / +0,030 / +0,019) đều có CI chứa 0.

## 3. Chi phí index (180 bài, RTX 3090)

| RAPTOR (cây + tóm tắt BART) | Agreement (SBERT) |
|---:|---:|
| 12 151 s (3,4 giờ) | 29 s |

Chi phí API sinh đáp án tương đương giữa các nhánh (~770–1950 token/lượt theo ngân sách).

## 4. Kết luận được phép viết

1. **Agreement Ranking không kém RAPTOR** về Evidence F1 trên tập test chưa từng xem,
   lặp lại kết quả xác nhận trên 100 bài dev mới, với chi phí index nhỏ hơn ~400 lần.
   Không được viết rằng nó *vượt* RAPTOR (CI chứa 0).
2. **Không chứng minh được** rằng rerank toàn bài đơn thuần không kém RAPTOR (T2), và
   **không chứng minh được** Agreement tốt hơn rerank toàn bài sau hiệu chỉnh Holm (H-b,
   p = 0,052). Tín hiệu SBERT có xu hướng giúp (+0,028, CI chưa phân tích đa giả thuyết
   loại 0) nhưng chưa đạt mức xác nhận.
3. Kết luận thực tiễn: với bài báo khoa học ngắn, có thể thay cây RAPTOR bằng một tín hiệu
   tương đồng câu hỏi–leaf rẻ mà không mất chất lượng evidence đo được.

Giới hạn: một bộ dữ liệu (QASPER), một generator (gpt-4o-mini), chỉ số evidence tự động;
mẫu kiểm tra thủ công attribution (`human_audit_unlabelled.json` của v9) chưa gán nhãn.
