# V7 publication-readiness audit

Audit date: 2026-09-01. Verdict: **do not run the one-shot main test yet**.

## Evidence completed

- Fresh QASPER train rollouts: 800 questions / 534 papers, 3,091 rows,
  93% citation-evaluable.
- Fresh QASPER dev rollouts: 250 questions / 164 papers, 962 rows,
  96% citation-evaluable.
- Final parent and section checkpoint artifacts were trained from those frozen
  rollouts.
- The test manifest bug was corrected. The replacement contains exactly 180
  questions from 180 unique papers, uses empirical document word-count
  tertiles, and is deterministic (selection SHA-256
  6f29aa8f159a78d5ae0a704971e167dfd7b16980892f9a6626579ae0a13e04cd).
- Regression suite: 77 tests and 4 subtests pass.

## Publication blockers

### P0 — learned gates do not generalize

The parent gate has dev AUC 0.5366 and balanced accuracy 0.5432; the section
gate has dev AUC 0.5110 and balanced accuracy 0.5190. Their train AUC values
are 0.9954 and 0.9973 respectively. Group-disjoint train CV is also close to
chance. This is strong overfitting and weak learnable signal, not a shortage
of rollout rows.

Required before test:

1. Stabilize the counterfactual target (for example repeated deterministic
   generations and aggregation, or a retrieval/verification-only target).
2. Audit feature/label shift by split and remove features that encode
   paper-specific scale.
3. Report calibration (Brier score and reliability/ECE) because the gate is
   described as a probability/risk estimator.
4. Predeclare a minimum dev acceptance criterion. A reasonable default is
   paper-grouped AUC above 0.60 with a clustered interval excluding 0.50,
   followed by end-to-end superiority or non-inferiority checks.

If this remains unmet, the scientific claim must be narrowed to a diagnostic
or negative-result study; the learned policy cannot be presented as an
effective generalizing component.

### P0 — three benchmark labels are semantically false

The current oracle_evidence is a flat-neural retrieval run, not gold-evidence
context. oracle_context is static hierarchical expansion with a large token
budget, not oracle context. full_document is another flat-neural retrieval
run, not a full-document long-context baseline.

Implement these systems literally or remove them from all tables and scripts.
Never publish results under the current names. Also build/run systems lazily:
the current script constructs several neural indexes simultaneously, creating
an avoidable RAM/VRAM crash risk.

### P0 — evaluator audit is not independent

scripts/audit_evaluator.py calls the local metric implementation and compares
only with hand-written expectations. It does not execute AllenAI's official
QASPER evaluator despite its docstring.

Vendor or pin the official evaluator, generate predictions in its native
format, and require exact agreement on a multi-annotator fixture and edge
cases. Record the upstream URL, commit, license, and file hash.

Official reference:
<https://github.com/allenai/qasper-led-baseline/blob/main/scripts/evaluator.py>

### P0 — dev acceptance benchmark is missing

The existing v7 reports contain only smoke subsets. Run the corrected full dev
benchmark before test. At minimum compare BM25, dense, hybrid, flat neural,
static hierarchy, prior adaptive, and learned adaptive. Ablations must match
their names; disabling rollback is not the same experiment as removing the
drift term from training.

Predeclare all of the following:

- primary metric and comparison;
- citation precision/recall non-inferiority margins;
- harmful-drift ceiling;
- invalid-citation rate requirement (zero);
- answer F1 and official evidence F1;
- context tokens, latency, and API cost;
- stopping rule that keeps the test set closed when dev fails.

### P0 — statistical report is incomplete

Add paired effect sizes, paper-clustered confidence intervals and hypothesis
tests, correction for multiple comparisons (for example Holm), and explicit
non-inferiority tests. Current code mixes an unclustered paired p-value with a
clustered interval. Report the number of papers as the independent sampling
unit.

## P1 requirements

- Generator swap on one frozen matched subset, with exact dated model IDs,
  prompt hash, decoding parameters, and paired analysis.
- SciFact OOD runner/report whose primary metric is rationale attribution.
  The QASPER-specific report currently cannot substantiate an OOD claim.
- Manual blinded audit of a predeclared random sample for citation correctness,
  unsupported claims, abstention errors, and an error taxonomy.
- Complete provenance: code commit and dirty state, manifest/checkpoint/config
  hashes, model revisions, package versions, hardware, seeds, prompts, token
  usage, latency, and monetary cost.
- CI on a clean environment. A lock file alone is insufficient; no CI workflow
  is currently present.
- Data/model/API licenses, privacy statement, limitations, compute disclosure,
  and an artifact README with commands that reproduce every table.

## One-shot test gate

The test command may run exactly once only after:

1. all P0 implementation defects above are fixed and regression-tested;
2. the method, thresholds, prompts, baselines, metrics, and analyses are frozen;
3. the full dev result meets the predeclared acceptance criteria;
4. the repository/provenance snapshot is archived before execution.

After the test run, no method or threshold changes may use test labels. Any
follow-up becomes a new protocol/version and requires a new untouched split.
