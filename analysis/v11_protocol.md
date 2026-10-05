# V11 — Protocol kiểm tra độ bền (đăng ký trước)

2026-10-05. Mục tiêu: kiểm tra kết luận v10 ("Agreement Ranking không kém RAPTOR") có
đứng vững khi đổi **generator** và đổi **bộ dữ liệu** (bài dài hơn). Không thay đổi
phương pháp: 3 nhánh `raptor`, `all_leaf`, `agreement`; ngân sách 512/1024/2048; bộ đóng
gói, verifier, chấm điểm và họ giả thuyết đúng như v10 (`scripts/run_v10_confirm.py`
dùng nguyên, gọi qua `scripts/run_v11.py`).

## A. Generator thứ hai: Qwen2.5-7B-Instruct (local, fp16, greedy)

- Cùng prompt grounded + cùng chỉ dẫn 120 từ như gpt-4o-mini; JSON được kiểm theo cùng
  hợp đồng citation (lỗi định dạng được ghi nhận, không bị che).
- Chạy trên **QASPER xác nhận** (100 câu) và **QASPER test** (180 câu), tái dùng nguyên
  bảng xếp hạng v10 (retrieval không chạy lại).
- Tập test đã mở ở v10; lượt này chỉ đổi generator cho các nhánh đã đóng băng, không dùng
  để chỉnh phương pháp → báo cáo là **kiểm tra độ bền**, không phải xác nhận mới.

## B. Bộ dữ liệu thứ hai: PeerQA (Baumgärtner et al., NAACL 2025)

- Câu hỏi: bản phát hành chính thức `peerqa-data-v1.0` (CC BY-NC-SA 4.0). Bài báo: tải PDF
  bằng script chính thức của PeerQA (OpenReview/EGU), trích văn bản bằng GROBID 0.8 theo
  pipeline chính thức. Chuyển đổi: `src/edahr/peerqa.py`.
- Đưa vào: mọi câu hỏi `answerable_mapped` có ít nhất một đoạn evidence ánh xạ được, thuộc
  các bài tải và trích xuất thành công. Loại trừ (ghi số lượng): bài không tải/trích được,
  câu không ánh xạ được evidence. Manifest + SHA-256 được ghi **trước** mọi lượt chạy.
- Chưa từng dùng PeerQA cho thiết kế → đây là đánh giá xác nhận độc lập.
- Generator chính: gpt-4o-mini. Qwen2.5-7B: phụ.

## Giả thuyết (mỗi bộ dữ liệu × generator báo cáo riêng)

Chỉ số chính: official-style paragraph Evidence F1 (cùng công thức QASPER), trung bình 3
ngân sách mỗi câu; đơn vị lấy mẫu là bài báo; sign-flip theo bài; biên 0,02; Holm trên 3:

| Mã | Giả thuyết |
|---|---|
| H-a | agreement không kém RAPTOR quá 0,02 |
| T2 | all_leaf không kém RAPTOR quá 0,02 |
| H-b | agreement tốt hơn all_leaf |

(QASPER xác nhận giữ họ 2 giả thuyết H-a, H-b như v10.) Kết quả **chính** của v11 là
PeerQA × gpt-4o-mini. Các ô khác là kiểm tra độ bền; báo cáo đủ, không chọn lọc.

## Ghi chú

- Answer F1 trên PeerQA là token-F1 với câu trả lời tự do của tác giả (khác chỉ số chính
  thức của PeerQA: ROUGE-L/AlignScore/Prometheus) → chỉ mô tả.
- Bài PeerQA dài (~12k token) → nhiều leaf hơn; RAPTOR được dựng đầy đủ, không cắt.

## Sửa đổi 1 (2026-10-05, trước mọi lượt chạy PeerQA)

Bản phát hành chính thức `peerqa-data-v1.0` không tải tự động được (trang tudatalib chặn
bằng kiểm tra chống bot; không vượt qua cơ chế này). Thay bằng **`mteb/PeerQA`** trên
Hugging Face (CC BY-NC-SA 4.0): bản đóng gói MTEB của PeerQA, gồm toàn văn các bài thuộc
nhóm NLPeer (được phép phát hành lại) và evidence do tác giả gán (qrels).

- Quy mô: **70 bài, 136 câu hỏi** (tất cả có evidence; tất cả ánh xạ được tới leaf gold).
  Trung vị ~5.200 từ/bài (dài hơn QASPER).
- Đơn vị evidence là đơn vị corpus của MTEB (gần với câu) → Evidence F1 tính ở mức đơn vị
  này (chặt hơn mức đoạn của QASPER); so sánh giữa các nhánh vẫn công bằng.
- Không có câu trả lời tự do → **Answer F1 không tính** trên PeerQA.
- Heading: một đơn vị là heading chỉ khi có đơn vị sau ghi nó làm heading; các đơn vị khác
  (kể cả chú thích hình/bảng không có heading) là văn bản thường. Quy tắc chốt trước khi chạy.
- Manifest: `scripts/v11_make_peerqa_manifest.py`; hash nguồn và manifest trong
  `manifests/peerqa_metadata.json`. Manifest dẫn xuất không commit (giấy phép NC-SA).
- Giả thuyết, nhánh, ngân sách, phân tích: **không đổi**.

## Sửa đổi 2 (2026-10-05, kỹ thuật, trước mọi kết quả PeerQA)

Prepare dừng ở bước dựng cây RAPTOR: UMAP không dựng được đồ thị láng giềng cho một tập
đơn vị gần như trùng nhau (code RAPTOR gốc cũng dừng ở trường hợp này). Thêm adaptation
**A6**: tập điểm bị ảnh hưởng được giữ làm một cụm — đúng cách code gốc xử lý tập quá nhỏ
cho UMAP; số lần kích hoạt ghi vào metadata cây. *Đính chính:* bản sửa đầu tiên (commit
`480f03e`) chỉ bọc bước UMAP cục bộ do chẩn đoán sai; lỗi thật ở bước **toàn cục**, lần
chạy lại thứ nhất dừng ở cùng chỗ; bản sửa hoàn chỉnh bọc cả hai bước. Chỉ tác động nhánh
lỗi; cây QASPER đã dựng không đổi. Chưa có bảng xếp hạng, đáp án hay điểm PeerQA nào.
