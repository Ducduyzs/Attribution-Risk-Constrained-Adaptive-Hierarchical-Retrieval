# V11-A — Độ bền theo generator: Qwen2.5-7B-Instruct (đăng ký trước)

2026-10-05. Protocol: `analysis/v11_protocol.md` (commit `78440e9`). Cùng 3 nhánh, cùng
ngân sách, cùng **context** (bảng xếp hạng v10 tái dùng nguyên), cùng verifier; chỉ đổi
generator từ gpt-4o-mini sang Qwen2.5-7B-Instruct (fp16, greedy, chạy local trên RTX 3090,
không dùng API). Dữ liệu: `artifacts/v11_qasper-{confirm,test}_qwen25-7b/`.

## 1. Giả thuyết (Evidence F1, trung bình 3 ngân sách)

| Tập | Giả thuyết | gpt-4o-mini (v10) | **Qwen2.5-7B (v11)** |
|---|---|---|---|
| Xác nhận (100) | H-a agreement ≥ RAPTOR − 0,02 | +0,032, Holm p=0,010 ✔ | −0,033 [−0,074; +0,008], p=0,73 ✘ |
| Xác nhận (100) | H-b agreement > all_leaf | +0,023, p=0,15 ✘ | −0,043 [−0,090; +0,003] ✘ |
| Test (180) | H-a agreement ≥ RAPTOR − 0,02 | +0,020, Holm p=0,006 ✔ | +0,007 [−0,027; +0,043], Holm p=0,19 ✘ |
| Test (180) | T2 all_leaf ≥ RAPTOR − 0,02 | −0,008, p=0,20 ✘ | −0,007 [−0,043; +0,030], Holm p=0,37 ✘ |
| Test (180) | H-b agreement > all_leaf | +0,028, Holm p=0,052 ✘ | +0,014 [−0,017; +0,046], Holm p=0,37 ✘ |

## 2. Evidence F1 theo ngân sách (Qwen, test 180)

| Nhánh | 512 | 1024 | 2048 |
|---|---:|---:|---:|
| raptor | 0,355 | 0,384 | 0,388 |
| all_leaf | 0,331 | 0,385 | 0,390 |
| agreement | **0,362** | **0,391** | **0,397** |

## 3. Diễn giải

- **Context giống hệt** giữa hai generator (độ phủ gold được đóng gói trùng từng số), nên
  mọi khác biệt đến từ cách generator *trích dẫn* trong cùng context.
- Với Qwen, kết luận "agreement không kém RAPTOR" **không được xác nhận** ở mức thống kê:
  trên tập test, ước lượng điểm vẫn cao nhất ở cả 3 ngân sách (+0,007 so với RAPTOR) nhưng
  CI rộng; trên tập xác nhận 100 câu, ước lượng điểm âm (−0,033).
- Kết luận phải viết lại: Agreement Ranking không kém RAPTOR **với gpt-4o-mini**; với một
  generator mở 7B, chênh lệch giữa ba bộ chọn nằm trong dao động do generator, không có bộ
  chọn nào khác biệt có ý nghĩa. Lợi thế chi phí index (~400×) không phụ thuộc generator.
- Qwen tạo JSON hợp lệ ở > 99% lượt (lỗi định dạng ghi nhận, không bị che); trung bình
  ~1,3 claim / câu trả lời, ít hơn gpt-4o-mini.
