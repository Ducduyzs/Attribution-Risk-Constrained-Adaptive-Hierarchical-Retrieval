# Bàn giao cho AI tiếp theo — Attribution-Risk-Constrained AHR

Cập nhật: 2026-08-27, múi giờ Asia/Saigon.

## 1. Mục tiêu nghiên cứu đã chốt

Tên hướng hiện tại:

**Attribution-Risk-Constrained Adaptive Hierarchical Retrieval for Verifiable
Scientific Document Question Answering**.

Luận điểm cần kiểm tra: mở rộng phân cấp có thể cứu evidence bị bỏ sót, nhưng
cũng gây attribution drift khi generator/verifier gán claim sang sibling leaf
không đúng. Learned gate phải kiểm soát rủi ro này, không chỉ tăng recall.

Benchmark chính hiện làm bằng **tiếng Anh** vì QASPER và SciFact là tiếng Anh.
Tiếng Việt là hướng cross-lingual riêng sau khi pipeline tiếng Anh được đóng
băng; không dịch QASPER rồi gọi đó là benchmark gốc.

## 2. Repository và mốc Git an toàn

Repository local:

`D:\AI PROJECT\Evidence-Density-Aware Adaptive Hierarchical Retrieval for Scientific Document Question Answering`

Remote:

`https://github.com/Ducduyzs/Attribution-Risk-Constrained-Adaptive-Hierarchical-Retrieval.git`

Branch: `main`.

Mốc frozen-v5 đã commit và push:

`52cf54c Freeze reproducible v5 evaluation artifacts`

Commit này có mã nguồn QASPER/SciFact, rollout fresh, checkpoint joblib,
artifact per-query, bảng kết quả, provenance và 66 tests. Có thể quay lại mốc
này nếu phần v6 chưa commit gặp vấn đề.

Không được commit `config.local.json`. File này chứa OpenAI/Gemini key và đã
được `.gitignore`. Không in hoặc sao chép giá trị key vào log/tài liệu.

## 3. Kết quả frozen-v5

### QASPER unseen-paper test — OpenAI, 40 câu/40 paper

| system | citation precision | citation recall | citation F1 |
|---|---:|---:|---:|
| B_flat | 0.0250 | 0.0250 | 0.0250 |
| B_static | 0.1250 | 0.1025 | **0.1062** |
| prior | 0.0250 | 0.0250 | 0.0250 |
| learned_v5 | 0.0667 | 0.0500 | 0.0560 |

Learned-v5 tốt hơn flat nhưng chưa có ý nghĩa thống kê:
`p=0.2178`, clustered 95% CI `[0.0000, 0.0786]`. Static là hệ thống tốt nhất
và có cải thiện có ý nghĩa so với flat (`p=0.0350`). Không được tuyên bố
learned-v5 vượt static.

### Antigravity generator swap — 10 câu/10 paper

Citation F1: flat `0.4852`, static `0.4200`, prior `0.4919`, learned-v5
`0.5519`. Mẫu quá nhỏ, chỉ dùng làm generator-sensitivity check.

### SciFact OOD rationale attribution — 40 paper

Citation F1: flat `0.3833`, static `0.7450`, learned-v5 `0.7358`.
Learned-v5 so với flat: `p=0.0010`, clustered 95% CI `[0.2250, 0.4867]`.
SciFact không phải QA endpoint tương đương QASPER; không diễn giải answer F1
như stance accuracy.

Tài liệu chính:

- `analysis/v5_main_results.md`
- `analysis/v5_generator_swap_antigravity.md`
- `analysis/v5_scifact_ood.md`
- `analysis/v5_failure_analysis_fresh.md`
- `analysis/v5_training_report.md`
- `analysis/v6_next_experiment_protocol.md`

## 4. Dữ liệu và checkpoint frozen

QASPER local đã đầy đủ; người dùng chưa cần cung cấp thêm dataset:

- train: 888 papers, 2,593 questions;
- dev: 281 papers, 1,005 questions;
- test: 416 papers, 1,451 questions;
- tổng cộng 1,585 papers, 5,049 stable question IDs.

Fresh v5 dùng một câu mỗi paper:

- train: 120 rows, 110 citation-evaluable;
- dev: 40 rows, 36 citation-evaluable.

Checkpoint cuối:

- `checkpoints/policy_parent_v5_final.joblib`: random forest, threshold 0.50,
  SHA-256 `ab0b22478bbb3295f709e051cf0f449267b6de0727acc5587224cc571823498c`;
- `checkpoints/policy_section_v5_final.joblib`: gradient boosting, threshold
  0.32, SHA-256
  `1d7d7aeae407cf2f12622df2e366e2e38907865b17291c1a8d61de9f03fd4dbb`.

Hai checkpoint đã được tái huấn luyện vào file tạm và hash trùng từng byte.

## 5. Trạng thái v6 chưa commit

Sau commit `52cf54c`, có một vòng instrumentation đang nằm trong working tree.
Toàn bộ tests hiện đạt: **67 passed**.

Các file v6 đã sửa:

- `src/edahr/schemas.py`: `Result` thêm `raw_generation` và
  `verification_trace`;
- `src/edahr/verification.py`: trace từng claim và từng candidate leaf; tách
  nguyên nhân invalid citation, low confidence, base-support rejection và
  sibling-guard rejection;
- `src/edahr/verification.py`: sửa bug attribution support — sau lexical
  fallback phải dùng `effective_support`, không giữ NLI thô;
- `src/edahr/pipeline.py`: lưu raw generation và verification trace;
- `src/edahr/baselines.py`: sửa `generated_claim_count`/`verified_claim_count`
  lấy từ metrics đúng nghĩa và ghi trace vào artifact;
- `scripts/run_qasper_benchmark.py`: thêm `--systems` để chạy subset systems;
- `tests/test_verification.py`: test hồi quy lexical fallback + trace.

Không discard các thay đổi này trước khi đọc `git diff`. Chúng chưa được
commit vì người dùng yêu cầu tạm dừng.

## 6. Phát hiện dev ngay trước khi dừng

Từ 40 rollout QASPER dev frozen cũ:

| branch | generation empty | verifier rejected all | verified some |
|---|---:|---:|---:|
| keep/flat | 7 | 22 | 11 |
| parent | 6 | 2 | 32 |
| section | 4 | 1 | 35 |

Nút thắt của flat chủ yếu là verifier loại toàn bộ claim, không phải retrieval.
Trong QASPER frozen test, retrieval `hit@10 = 0.95` nhưng citation F1 rất thấp.

Một benchmark dev mới đã được khởi động bằng OpenAI rồi **dừng bằng Ctrl+C theo
yêu cầu người dùng**. Lệnh dùng 20 dev papers và hai systems. Quá trình dừng
trước khi hoàn thành B_flat; script chỉ ghi artifact sau khi hoàn thành cả
system nên `data/artifacts/v6_dev_diagnostic` hiện không có kết quả hợp lệ.
Không diễn giải số liệu từ lần chạy dở.

## 7. File của người dùng — tuyệt đối không đụng vào

Các file sau không thuộc commit nghiên cứu hiện tại:

- `opencode.jsonc` — tracked nhưng đang có thay đổi của người dùng;
- `filelist.txt`;
- `kien thuc.docx`;
- `needtodo.txt`;
- các checkpoint `policy_*_v5_final.ts` — MLP chẩn đoán đã bị joblib thay thế;
- các rollout `*_smoke*` và Antigravity smoke.

Không stage, sửa, xóa hoặc hoàn nguyên các file trên nếu người dùng chưa yêu
cầu rõ ràng.

## 8. Việc AI tiếp theo nên làm

1. Đọc tài liệu này, `analysis/v6_next_experiment_protocol.md` và `git diff`.
2. Chạy `D:\edahr_env\Scripts\python.exe -m pytest -q`; kỳ vọng 67 passed.
3. Review trace schema, đặc biệt xác nhận raw/effective NLI support và sibling
   guard. Sau đó commit riêng instrumentation v6, không stage file người dùng.
4. Chạy lại dev diagnostic vào thư mục mới, không dùng thư mục lần chạy dở:

```powershell
D:\edahr_env\Scripts\python.exe scripts\run_qasper_benchmark.py `
  --paper-manifest qasper_dev_papers.jsonl `
  --question-manifest qasper_dev_questions.jsonl `
  --dataset-name qasper-v0.3 --split-name dev-diagnostic-v6 `
  --questions 20 --systems B_flat learned_v5 `
  --config config.local.json --provider openai --model gpt-4o-mini `
  --artifact-dir data\artifacts\v6_dev_diagnostic_rerun `
  --report analysis\v6_dev_diagnostic_rerun.md
```

5. Từ `verification_trace`, lập bảng đếm status và phân phối:
   `nli_support`, `lexical_coverage`, `effective_support`, wrong-leaf rate và
   sibling-guard rejection. Chỉ dùng QASPER dev để hiệu chỉnh threshold/prompt.
6. Nếu lỗi chủ yếu là `below_support_threshold`, chạy threshold sweep offline
   từ trace, không gọi lại generator. Nếu lỗi chủ yếu là invalid citation hoặc
   generation empty, sửa generation contract trên dev.
7. Tăng rollout lên khoảng 400–800 train và 150–250 dev, tối đa 3 câu/paper;
   giữ paper-disjoint và inverse-paper weighting.
8. Trước evaluation mới, bổ sung `--exclude-selection` hoặc manifest frozen để
   loại toàn bộ 40 QASPER test question IDs đã xem. Không dùng test/SciFact
   frozen để chọn model, threshold hoặc prompt.
9. Chỉ sau khi khóa v6 trên dev mới chạy một partition QASPER test mới tối
   thiểu 150–200 papers, rồi generator swap 50–100 matched papers.

## 9. Điều kiện khoa học bắt buộc

- B_static phải được báo cáo vì hiện là comparator mạnh nhất.
- Không gọi random held-out QASPER papers là true OOD; gọi là unseen-paper test.
- SciFact hiện chỉ hỗ trợ claim rationale-attribution OOD.
- Citation precision và recall phải non-inferior so với flat theo margin đã
  khóa trên dev; mọi cải thiện cần paper-clustered confidence interval.
- Lưu stable question ID, paper ID, manifest hash, checkpoint hash, config
  hash, package versions, seed và per-query artifact.
- Nếu learned gate vẫn không thắng static, thu hẹp contribution thành hiện
  tượng attribution drift + diagnostic/risk-constrained framework; không viết
  claim vượt quá dữ liệu.
