# V11-B — PeerQA (bộ dữ liệu thứ hai, bài dài hơn; đăng ký trước)

2026-10-05. Protocol: `analysis/v11_protocol.md` (+ sửa đổi 1–3, đều ghi trước mọi đáp án
hay điểm PeerQA). Dữ liệu: `mteb/PeerQA` (NLPeer subset), **70 bài, 136 câu**, evidence do
tác giả gán; trung vị ~5.200 từ/bài. 3 nhánh × 3 ngân sách × 136 câu = 1224 lượt cho mỗi
generator. Kết quả chính: gpt-4o-mini. Dữ liệu: `artifacts/v11_peerqa_{gpt-4o-mini,qwen25-7b}/`.

## 1. Giả thuyết (Evidence F1 mức đơn vị câu, trung bình 3 ngân sách; Holm trên 3)

| Giả thuyết | gpt-4o-mini (chính) | Qwen2.5-7B |
|---|---|---|
| H-a agreement không kém RAPTOR (biên 0,02) | +0,011 [−0,003; +0,025], Holm p<0,001 **✔** | +0,005 [−0,016; +0,024], Holm p=0,016 **✔** |
| T2 all_leaf không kém RAPTOR (biên 0,02) | +0,005 [−0,006; +0,016], Holm p<0,001 **✔** | +0,002 [−0,014; +0,017], Holm p=0,015 **✔** |
| H-b agreement tốt hơn all_leaf | +0,006, p=0,22 ✘ | +0,003, p=0,41 ✘ |

**Lưu ý quan trọng về biên.** Biên non-inferiority được chốt là 0,02 *tuyệt đối* (từ thang
QASPER, Evidence F1 ~0,3–0,4). Trên PeerQA, Evidence F1 tính ở mức câu nên thấp hơn nhiều
(~0,13–0,18); biên 0,02 tương đương ~13% giá trị của RAPTOR → **lỏng hơn tương đối** so với
QASPER. Tuy vậy, ước lượng điểm của cả hai bộ chọn rẻ đều ≥ RAPTOR và CI chỉ chạm tới
−0,016 (Qwen) / −0,006 (gpt-4o-mini).

## 2. Theo ngân sách (gpt-4o-mini; độ phủ gold không phụ thuộc generator)

| Nhánh | Evidence F1 512 / 1024 / 2048 | **Phủ ký tự gold** 512 / 1024 / 2048 | Leaf F1 512 / 1024 / 2048 |
|---|---|---|---|
| raptor | 0,129 / 0,143 / 0,136 | 0,373 / 0,515 / 0,638 | 0,271 / 0,296 / 0,302 |
| all_leaf | 0,132 / 0,142 / 0,150 | **0,412 / 0,573 / 0,703** | 0,304 / 0,317 / 0,338 |
| agreement | **0,135 / 0,146 / 0,161** | 0,380 / 0,549 / 0,684 | 0,275 / 0,314 / 0,338 |

Trên bài dài, **RAPTOR đóng gói ít gold nhất ở mọi ngân sách**; rerank toàn bài đóng gói
nhiều nhất. Giả thuyết "RAPTOR mạnh hơn trên tài liệu dài" không được ủng hộ.

## 3. Chi phí

| | RAPTOR | Agreement |
|---|---:|---:|
| Dựng index 70 bài (RTX 3090) | 3.097 s | 32 s |
| API gpt-4o-mini (1224 lượt) | — | 0,29 USD (dùng chung cho cả 3 nhánh) |

Qwen JSON hợp lệ 99,2%; gpt-4o-mini 100%.

## 4. Kết luận

1. Trên bộ dữ liệu thứ hai với bài dài hơn, **cả Agreement Ranking lẫn rerank toàn bài đều
   không kém RAPTOR**, với cả hai generator, trong khi rẻ hơn ~100 lần ở bước index.
2. Không chứng minh được Agreement tốt hơn rerank toàn bài (ở mọi tập, mọi generator).
3. Hạn chế: biên 0,02 tương đối lỏng trên thang Evidence F1 mức câu; chỉ một phần PeerQA
   (NLPeer, 70 bài); không có câu trả lời tự do nên không đánh giá chất lượng trả lời.
