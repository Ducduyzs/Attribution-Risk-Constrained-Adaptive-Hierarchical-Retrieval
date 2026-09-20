# Bàn giao cho AI tiếp theo — Hoàn thiện pipeline để đăng báo

Cập nhật: 2026-08-27 — Asia/Saigon.

## 1. Nhiệm vụ tổng quát

Tiếp tục hoàn thiện hệ thống:

**Attribution-Risk-Constrained Adaptive Hierarchical Retrieval for Verifiable
Scientific Document Question Answering**.

Mục tiêu không phải chỉ làm pipeline chạy được. Mục tiêu là tạo một thực nghiệm
có thể bảo vệ trước reviewer: đúng evaluator chính thức, không rò rỉ test,
baseline đủ mạnh, artifact tái lập được và claim không vượt quá bằng chứng.

Benchmark chính hiện là **tiếng Anh**. QASPER và SciFact đều là tiếng Anh.
Tiếng Việt chỉ được làm như thí nghiệm cross-lingual riêng sau khi pipeline
tiếng Anh đã đóng băng.

## 2. Repository và trạng thái Git

Repository local:

`D:\AI PROJECT\Evidence-Density-Aware Adaptive Hierarchical Retrieval for Scientific Document Question Answering`

Remote:

`https://github.com/Ducduyzs/Attribution-Risk-Constrained-Adaptive-Hierarchical-Retrieval.git`

Branch: `main`.

Mốc frozen-v5 đã push:

`52cf54c Freeze reproducible v5 evaluation artifacts`

Hai commit v6 hiện có ở local nhưng chưa push:

- `f79b3e0` — verification trace, raw generation, claim counters;
- `5369f06` — generation prompt yêu cầu context ID rõ ràng.

Trạng thái kiểm thử gần nhất: **67 passed**.

Trước mọi thay đổi, chạy:

```powershell
git status --short --branch
git diff
D:\edahr_env\Scripts\python.exe -m pytest -q
```

Không được commit `config.local.json` hoặc in API key. Hai key OpenAI/Gemini đã
được cấu hình local.

## 3. Kết quả v6 hiện tại

### Trước generation-contract fix — 20 QASPER dev papers

- B_flat citation F1: `0.000`;
- learned-v5 citation F1: `0.035`;
- gần như mọi generated claim dùng citation sai định dạng.

Nguyên nhân: generator trả numeric index như `1`, `2`, `3` thay vì context ID.

### Sau generation-contract fix — cùng 20 QASPER dev papers

- B_flat citation precision/recall/F1: `0.5053 / 0.6465 / 0.4942`;
- learned-v5: `0.4105 / 0.5982 / 0.4439`;
- learned-v5 so với flat: `p=0.2707`;
- paper-clustered 95% CI: `[-0.1365, 0.0322]`.

Kết luận trung thực: prompt fix thành công, nhưng learned-v5 hiện chưa vượt
flat. Không được dùng kết quả này để claim superiority.

Lần `v6_dev_diagnostic_rerun_fixed2` mới chỉ hoàn thành B_flat. Chưa có
learned-v5 tương ứng, nên chưa được xem là benchmark hoàn chỉnh.

## 4. Quy tắc quan trọng nhất

**Không chạy rollout train/dev lớn trước khi hoàn thành Mục 5–8 dưới đây.**

Nếu evaluator hoặc generation contract tiếp tục thay đổi sau khi rollout lớn,
toàn bộ label, checkpoint và bảng kết quả sẽ phải tạo lại.

Không dùng 40 QASPER test papers frozen-v5 hoặc 40 SciFact dev papers đã xem để
chọn prompt, model, feature, threshold hay hyperparameter.

## 5. Blocker P0 — strict citation contract

File chính: `src/edahr/models.py`.

Generation hiện vẫn dùng JSON mode và prompt tự do. Bộ lọc citation rỗng chỉ bỏ
chuỗi rỗng; nó không bảo đảm claim có context ID hợp lệ.

Phải thực hiện:

1. Với OpenAI, thay `response_format={"type": "json_object"}` bằng strict
   `json_schema`.
2. Tạo enum động từ context IDs hợp lệ của mỗi request.
3. JSON citation value phải là `"C1"`, không phải `"[C1]"`. Dấu ngoặc vuông
   chỉ dùng làm marker trong phần evidence của prompt.
4. Claim có `answerable=true` phải có ít nhất một citation thuộc enum.
5. Không âm thầm xóa claim/citation sai rồi coi như generator đúng. Phải lưu
   counter hoặc trace cho schema rejection/invalid citation.
6. Thêm unit tests cho: numeric ID, `[C1]`, `C1`, blank citation, unknown ID,
   claims rỗng và refusal.
7. Tạo validation tương đương cho Gemini và Antigravity.

OpenAI GPT-4o mini hỗ trợ Structured Outputs:

`https://developers.openai.com/api/docs/models/gpt-4o-mini`

API reference:

`https://developers.openai.com/api/reference/resources/chat`

Acceptance criteria:

- invalid citation rate bằng 0 trên smoke/dev run mới;
- không còn answerable claim có citation tuple rỗng;
- raw generator failures vẫn xuất hiện trong telemetry, không bị che giấu.

## 6. Blocker P0 — official QASPER evaluator

Files chính:

- `src/edahr/qasper.py`;
- `src/edahr/baselines.py`;
- `src/edahr/evaluation.py`;
- `src/edahr/rollouts.py`.

Hiện converter hợp nhất evidence của nhiều annotator thành union. Đây không
phải cách evaluator chính thức QASPER chấm điểm.

Phải thực hiện:

1. Lưu `reference_answers` riêng cho từng annotation.
2. Lưu `reference_evidence_sets` riêng cho từng annotation.
3. Answer F1/EM: chấm từng reference rồi lấy max.
4. Evidence F1: chấm từng evidence set rồi lấy max.
5. Dùng normalization chính thức: lowercase, bỏ punctuation, articles và
   khoảng trắng thừa.
6. Không dùng `record["answer"]` đơn lẻ trong benchmark.
7. Thêm regression fixture và đối chiếu kết quả với evaluator QASPER chính
   thức từng giá trị.

Evaluator chính thức:

`https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py`

Acceptance criteria:

- fixture nội bộ cho đúng Answer F1 và Evidence F1 như official script;
- multi-annotator evidence không còn bị union trong metric chính;
- rollout reward/label dùng cùng định nghĩa evidence với benchmark.

## 7. Blocker P0 — paragraph evidence và child attribution

QASPER gán gold evidence ở paragraph level, còn hệ thống tạo overlapping child
chunks. Một gold paragraph có thể map vào nhiều leaves và làm sai denominator.

Phải giữ stable `paragraph_id` từ QASPER ingestion và ánh xạ mỗi child chunk về
paragraph nguồn bằng char offsets.

Bài báo phải báo cáo hai metric độc lập:

1. **Official QASPER Evidence F1** ở paragraph level.
2. **Leaf Attribution F1 / harmful drift / rescue rate** ở child level.

Không được gọi leaf citation F1 là official QASPER Evidence F1.

Acceptance criteria:

- mỗi predicted child truy ngược được về paragraph ID;
- overlapping child chunks không nhân đôi một gold paragraph;
- artifact chứa cả paragraph-level và child-level IDs.

## 8. Blocker P0 — verifier safety

File chính: `src/edahr/models.py`, class `NliVerifier`.

Nếu code không tìm thấy label chứa `entail`, nó hiện trả xác suất lớn nhất của
mọi label. Điều này có thể coi contradiction là support.

Phải thực hiện:

1. Xác định entailment index từ `model.config.id2label`.
2. Nếu không xác định được entailment label, fail fast với lỗi rõ ràng.
3. Không dùng `max(all labels)` làm fallback.
4. Test nhiều label ordering khác nhau.
5. Lexical fallback chỉ được cứu claim khi không có contradiction mạnh.
6. Thêm guard cho negation và số liệu trái ngược.

Acceptance criteria:

- contradiction không bao giờ được tính thành entailment;
- test có cặp “improves” / “does not improve” và số liệu khác nhau;
- trace lưu raw NLI, lexical score, effective score và quyết định cuối.

## 9. Sửa training protocol trước khi retrain

File chính: `scripts/train_tree_policy.py`.

Code hiện ưu tiên accuracy trước balanced accuracy, trái với mô tả giao thức.

Phải thực hiện:

- thêm `--selection-metric`;
- mặc định paper-weighted balanced accuracy;
- chọn estimator/hyperparameter bằng group CV theo paper trên train;
- dùng dev để chọn threshold và risk constraints;
- lưu toàn bộ candidate model/threshold table;
- nếu gọi output là risk probability, thêm calibration và báo Brier/ECE;
- không chọn bất kỳ tham số nào từ frozen test.

Hai checkpoint v5 hiện tại được huấn luyện từ rollout trước generation-contract
fix. Chúng chỉ là historical baselines, không phải checkpoint v6 cuối.

## 10. Tạo lại fresh v6 train/dev

Chỉ thực hiện sau khi Mục 5–9 đạt tests.

Khuyến nghị:

- train: 400–800 questions;
- dev: 150–250 questions;
- tối đa 3 questions/paper;
- official train/dev paper-disjoint;
- inverse-paper weighting;
- cùng model snapshot, prompt hash, verifier config và evaluator version;
- crash-safe resume;
- lưu stable question ID, paper ID, config hash và selection manifest hash.

Dùng model snapshot cố định như `gpt-4o-mini-2024-07-18`, không dùng alias thay
đổi theo thời gian cho bảng cuối.

Sau đó train độc lập:

- parent gate;
- section gate.

## 11. Dev benchmark và ablation bắt buộc

Chạy cùng question manifest và generator settings:

- BM25;
- dense-only;
- hybrid/RRF;
- flat neural;
- B_static;
- prior heuristic;
- learned-v6;
- parent-only;
- section-only;
- không drift penalty;
- không sibling guard;
- không verifier;
- oracle evidence/context upper bound;
- full-document long-context baseline;
- citation-in-generation hoặc post-hoc attribution baseline.

B_static phải luôn xuất hiện vì hiện là comparator mạnh nhất.

Related-work comparators cần xem xét: Attribute or Abstain, RAPTOR và LongRAG.

Dev acceptance trước khi mở test mới:

- citation precision và recall không kém flat quá margin đã khóa;
- harmful drift dưới ceiling đã khóa;
- learned-v6 phải có lợi ích rõ so với prior;
- báo rescue/harmful-drift/kept-correct/kept-wrong decomposition;
- nếu static vẫn tốt hơn, không được che kết quả.

## 12. Tạo test partition mới chưa từng xem

Phải bổ sung benchmark selection có:

- seed cố định;
- selection manifest được ghi trước inference;
- SHA-256 của manifest;
- `--exclude-selection` hoặc danh sách question IDs đã sử dụng;
- loại 40 QASPER test IDs frozen-v5 và mọi test smoke/diagnostic IDs;
- stratification theo answer type, query type, paper length và evidence count.

Main test khuyến nghị tối thiểu 150–200 papers. Không thay prompt, feature,
model family hoặc threshold sau khi xem kết quả.

Thống kê phải gồm:

- macro metrics;
- paper-clustered 95% CI;
- cluster-aware paired significance;
- effect size;
- correction cho multiple comparisons;
- latency, context tokens và API cost.

## 13. Generator swap và OOD

Sau main test:

- chạy OpenAI và Antigravity/Gemini trên cùng matched subset 50–100 questions;
- không trộn hai generator vào cùng bảng chính;
- báo độ nhạy generator riêng.

SciFact chỉ được gọi là rationale-attribution OOD. Không dùng answer F1 hiện tại
như stance accuracy. Muốn claim OOD scientific QA phải bổ sung PeerQA hoặc một
scientific-document QA dataset tương đương.

## 14. Git, provenance và file không được chạm vào

Cần review rồi push hai commit v6. Các report và script trace hiện còn untracked
phải được lưu có chọn lọc cùng manifest/provenance.

Không stage hoặc sửa nếu người dùng chưa yêu cầu:

- `opencode.jsonc`;
- `filelist.txt`;
- `kien thuc.docx`;
- `needtodo.txt`;
- `checkpoints/policy_*_v5_final.ts`;
- rollout smoke và Antigravity smoke.

Mọi result table phải lưu:

- git commit và dirty-worktree flag;
- prompt/config/model snapshot;
- package versions và hardware;
- dataset/manifest/checkpoint SHA-256;
- seed;
- per-query artifact;
- generator usage/cost nếu có.

## 15. Claim được phép viết

Claim an toàn hiện tại:

> Hierarchical expansion can rescue missing evidence but can also introduce
> attribution drift. A risk-constrained adaptive gate provides a measurable
> framework for controlling this trade-off under distribution shift.

Chưa được viết rằng learned-v6 vượt mọi hierarchical retrieval baseline.

Nếu learned-v6 không vượt static ở test mới, contribution phải thu hẹp thành:

- phát hiện attribution drift;
- decomposition và diagnostic framework;
- risk-constrained gating;
- negative result trung thực về giới hạn generalization.

## 16. Definition of Done để bắt đầu viết paper

Chỉ đánh dấu hoàn thành khi tất cả điều kiện sau đạt:

- strict citation schema hoạt động cho mọi provider;
- evaluator khớp official QASPER;
- paragraph và child metrics được tách rõ;
- NLI/lexical verifier qua safety tests;
- fresh v6 train/dev được tạo sau mọi fix;
- parent/section v6 checkpoints có provenance;
- dev baselines và ablations hoàn chỉnh;
- test manifest mới được khóa trước inference;
- main test đạt cỡ mẫu và thống kê yêu cầu;
- generator swap hoàn thành;
- OOD claim được đặt tên chính xác;
- artifact, code, reports đã commit và push;
- không có API key hoặc file cá nhân trong Git;
- paper claim khớp đúng với kết quả.

Nếu một P0 blocker chưa đạt, không chạy experiment quy mô lớn và không tuyên
bố pipeline đã sẵn sàng đăng báo.
