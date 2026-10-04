# V9: exploratory development of all proposed directions

2026-10-04. All runs use the frozen 75-question / 30-paper DEV manifest only.
The unseen 180-question test is not opened. No confirmatory or novelty claim is
made from this exploratory matrix.

## Matrix

Budgets: 512, 1024, 2048 cl100k_base tokens, INCLUDING rendered evidence headers.
Shared GPT-4o-mini, temperature 0, answer-length instruction 120 words; NLI uses
only visible, cited fragments. Actual API prompt/output usage is recorded.

| Arm | Intervention |
|---|---|
| flat24 | Original top-24 rerank pool and top-8/knapsack packer |
| all_leaf | Rerank all leaves in the scoped paper; original packer |
| contextual24 | Cached LLM context prepended at reranker input only |
| contextual_all | Same contextual reranker over all leaves |
| mmr | All-leaf scores, relevance/redundancy selection |
| coverage | All-leaf scores, marginal query-token coverage per sqrt(cost) |
| sentence | Rerank deduplicated extractive sentences; relevance selection |
| window | Sentence seeds with +/-1 sentence, same section; coverage selector |
| dependency | Bounded heuristic anaphora/qualifier groups; coverage selector |
| evidence_first | Exact all_leaf context; evidence plan in structured response |
| dependency_first | Dependency selection + evidence-first generation |
| raptor | Cached faithful RAPTOR retrieval, shared reranker and packer |

The dependency module is a HEURISTIC prototype, not a trained dependency model.
Coverage is lexical, not measured semantic sufficiency. These methods must be
compared with their component controls before being proposed as a contribution.

## Fairness and provenance

- All arms share exact context budgets and generator/verifier configuration.
- All-leaf/sentence arms explicitly spend more reranking compute than flat24;
  cached prepare runtime is recorded separately. This is not a free improvement.
- RAPTOR retains its existing cached tree settings, including the recorded BART
  summarizer substitution. It is not represented as the authors' original setup.
- Visible-only verification and exact token accounting differ from v8. Compare
  methods WITHIN v9; do not interpret v8-to-v9 changes as a method gain.
- Gold labels do not enter candidate scoring, grouping, selection or generation.
- Files store code/manifest hashes and sanitized configuration. Resume rejects
  a changed protocol. Generation and NLI caches are content-addressed.
- Generation has an identical answer-length instruction; compliance is logged,
  not enforced by clipping a potentially supported answer.
- Results use official max-over-reference evidence F1 and answer F1, plus the
  existing leaf overlap F1. Leaf overlap is not claim-level entailment quality.
- Stage diagnostics distinguish leaf touch, full-leaf text retention, exact
  paragraph character coverage, generated citations and verified citations.
- Paragraph attribution uses overlap with the quoted fragment. Character
  coverage is conservative; it cannot establish semantic sufficiency.
- NLI acceptance is only a proxy. A blinded unlabelled human-audit sample is
  exported; the runner cannot manufacture human labels.
- All completed method x budget x metric contrasts against RAPTOR form one
  Holm family. Paper-clustered inference remains exploratory on this reused dev.

## Execution

GPU work runs only on the user-provided RTX 3090 Vast instance. Local work is
source editing and CPU unit tests. The SSH relay is optional; direct SSH works.
GPU verification runs as a supervisor one-shot job with autorestart disabled.
Generation runs on the local CPU/API client: API credentials never leave the
local machine. Only public QASPER prepared contexts and generated answers are
transferred over SSH. Per-question atomic outputs support resumable execution.

Commands (on remote project root):

```bash
/venv/main/bin/python -m pytest tests/test_experimental_v9.py -q
/venv/main/bin/python scripts/run_v9_directions.py --phase prepare
/venv/main/bin/python scripts/run_v9_directions.py --phase verify --output artifacts/v9_results
/venv/main/bin/python scripts/run_v9_directions.py --phase report --output artifacts/v9_results
```

Local generation after syncing prepared rankings:

```powershell
python -X utf8 -u scripts/run_v9_directions.py --phase generate --api-config config.local.json --prepared-from artifacts/v9_directions --output artifacts/v9_results --budgets 512 1024 2048
```

Expected matrix: 12 methods x 3 budgets x 75 questions = 2700 evaluated rows.
Exact duplicate prompts may share generator cache entries. Original v8
artifacts and user changes are preserved. Results are synchronized to the
local workspace before the task is considered complete.
