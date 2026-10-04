# Bảng thống nhất v8 — 75 câu dev / 30 bài, generator gpt-4o-mini (2026-10-04)

Tất cả hệ thống dùng cùng generator `gpt-4o-mini`, cùng verifier đã sửa guard, cùng
tham chiếu official. Hai lượt chạy: `artifacts/baselines/v8_fulldoc_dev/` (oracle,
flat, full_document — RTX 4050 local) và `artifacts/baselines/v8_dev_gpt4omini/`
(static, prior, RAPTOR, LongRAG — RTX 3090 Vast.ai). Thay thế bảng
`artifacts/baselines/main` (do Qwen2.5-7B sinh, metadata ghi sai).

| Hệ thống | Cit P | Cit R | **Cit F1** | **Evidence F1** | Answer F1 | Token context | Độ trễ (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle_evidence (cận trên) | 0,851 | 0,537 | 0,621 | 0,602 | 0,270 | 517 | 2,3 |
| B5 RAPTOR faithful | **0,496** | **0,455** | **0,436** | **0,404** | **0,244** | 1452 | 3,1 |
| prior (đề xuất) | 0,480 | 0,410 | 0,387 | 0,396 | 0,231 | 2215 | 3,3 |
| B4 static hierarchy | 0,453 | 0,365 | 0,365 | 0,373 | 0,229 | 2932 | 3,8 |
| B3 flat neural | 0,452 | 0,374 | 0,370 | 0,369 | 0,201 | 1572 | 5,8 |
| full_document | 0,405 | 0,347 | 0,343 | 0,337 | 0,205 | 4355 | 4,2 |
| B6 LongRAG faithful | 0,391 | 0,304 | 0,313 | 0,287 | 0,195 | 1562 | 3,1 |

So sánh cặp (chênh lệch trung bình, CI 95% bootstrap theo bài báo):

| Cặp | Citation F1 | Evidence F1 | Answer F1 |
|---|---|---|---|
| prior − flat | +0,017 [−0,023; +0,057] | +0,027 [−0,028; +0,078] | **+0,030 [+0,003; +0,062]** |
| prior − static | +0,022 [−0,029; +0,080] | +0,024 [−0,031; +0,080] | +0,002 [−0,028; +0,040] |
| prior − RAPTOR | −0,049 [−0,110; +0,009] | −0,008 [−0,059; +0,043] | −0,013 [−0,041; +0,015] |
| RAPTOR − flat | **+0,065 [+0,003; +0,137]** | +0,035 [−0,018; +0,088] | **+0,043 [+0,009; +0,077]** |
| static − flat | −0,005 [−0,078; +0,061] | +0,003 [−0,075; +0,083] | +0,028 [−0,014; +0,069] |
| full_document − flat | −0,027 [−0,085; +0,040] | −0,032 [−0,088; +0,024] | +0,004 [−0,039; +0,046] |

## Nhận xét

- Với gpt-4o-mini, **prior vượt flat và static** trên mọi metric (chỉ Answer F1 có ý
  nghĩa thống kê) và dùng ít token hơn static 24%. Khác với bảng Qwen-7B cũ, nơi prior
  thấp nhất.
- **RAPTOR vẫn tốt nhất** và rẻ nhất trong nhóm phân cấp (1452 token); prior kém RAPTOR
  0,049 citation F1 (CI chạm 0) và dùng nhiều hơn 53% token.
- Đưa cả bài báo không giúp; LongRAG kém nhất. Oracle cách xa mọi hệ thống → nút cổ
  chai là tìm đúng evidence.
- 75 câu / 30 bài vẫn nhỏ: phần lớn CI rộng ±0,05–0,08.

## Chi phí lượt chạy

API gpt-4o-mini cho 7 hệ thống × 75 câu: ước tính ~0,35 USD (theo token context).
GPU thuê: ~1 giờ RTX 3090 (dựng 30 cây RAPTOR ~45 phút). Cache 30 cây RAPTOR đã lưu ở
`artifacts/baselines/raptor/index/` (dùng lại được khi cùng cấu hình).
