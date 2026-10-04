# V10 — Protocol chạy test một lần (đăng ký trước)

2026-10-04. Đóng băng trước khi mở tập test. Sau lần chạy này, không thay đổi phương
pháp, nhánh, ngân sách, chỉ số hay phân tích dựa trên kết quả test.

## Tập test

- `data/manifests_test/qasper_test_stratified_{papers,questions}.jsonl`: 180 câu / 180 bài
  QASPER v0.3 test, phân tầng, seed 42 (tạo trước v8, chưa mở).
- SHA-256: questions `de0241e2…c2c7b` (= `selection_sha256` trong
  `test_manifest_metadata.json`), papers `a8096fda…dfa1ad`. Runner từ chối nếu lệch.

## Hệ thống (đóng băng, giống hệt lượt xác nhận v10)

`raptor`, `all_leaf`, `agreement` — `scripts/run_v10_confirm.py --split test --open-test-once`.
Ngân sách 512 / 1024 / 2048 token cl100k (gồm header); gpt-4o-mini, temperature 0; giới
hạn 120 từ; verifier NLI chỉ đọc phần hiển thị; pipeline v9 dùng nguyên; cấu hình
`artifacts/baselines/main/config.json`.

## Giả thuyết (Holm trên 3)

Chỉ số chính: official Evidence F1, trung bình 3 ngân sách mỗi câu; đơn vị lấy mẫu là
bài báo; kiểm định sign-flip theo bài; biên non-inferiority 0,02.

| Mã | Giả thuyết | Kiểm định |
|---|---|---|
| H-a | agreement không kém RAPTOR quá 0,02 (lặp lại kết quả xác nhận) | non-inferiority |
| T2 | all_leaf (rerank toàn bài) không kém RAPTOR quá 0,02 ("cây không đáng chi phí") | non-inferiority |
| H-b | agreement tốt hơn all_leaf | superiority một phía |

Chỉ số phụ (chỉ mô tả, không kiểm định): kết quả theo từng ngân sách, phủ ký tự paragraph
gold, Leaf F1, Answer F1, token context, token API, thời gian index RAPTOR và SBERT.

## Quy tắc

- Chạy đúng một lần; nếu lỗi kỹ thuật (crash, mạng), chỉ được chạy tiếp phần dở dang với
  cùng code/protocol (runner có cache và kiểm tra hash).
- Báo cáo mọi giả thuyết bất kể kết quả; không bỏ giả thuyết thất bại.
- Nhánh ngoài danh sách (flat, prior, full_document, oracle) không được thêm vào họ kiểm
  định sau khi xem kết quả.

## Ước tính chi phí

API ~0,45 USD (1620 lượt sinh). GPU ~3,5–4 giờ RTX 3090 (dựng 180 cây RAPTOR ~3,5 giờ;
rerank, SBERT, NLI < 30 phút).
