# Next Experiment Protocol

## Frozen evidence

- QASPER test (40 unseen papers) and SciFact dev OOD (40 evidence papers) are
  frozen evaluation sets. Do not tune thresholds, features, prompts, or model
  families from their per-query outcomes.
- The QASPER failure analysis is diagnostic only. Its legacy artifacts lack
  generated/verified claim counters, so 33--37 post-context failures per system
  cannot yet be split into empty generation and verifier rejection.
- SciFact is a rationale-attribution endpoint. Citation metrics are valid;
  answer F1 is not a stance metric because the generator emits free-form claims
  rather than a normalized `supported`/`contradicted` label.

## QASPER development cycle

1. Generate a larger train-only rollout with 3 questions per paper and a fresh
   dev-only rollout with the same cap. Keep official paper-disjoint splits.
2. Train parent and section gates with `train_tree_policy.py`. The learner now
   assigns inverse row-count weights per `source`, so each paper has equal total
   influence even when it contributes several questions.
3. Select estimator family and threshold on QASPER dev only. Primary selection
   metric is paper-weighted balanced accuracy; citation precision non-inferiority
   and harmful-drift constraints remain mandatory.
4. Run a dev-only verifier diagnostic with the new claim counters. If failures
   are mostly `generation_empty`, change the generation contract. If they are
   mostly `verifier_rejected_all`, calibrate NLI/lexical verification on dev.
5. Create a new untouched QASPER evaluation partition before claiming an
   improvement. The existing 40-paper test remains a historical frozen result.

## Acceptance criteria

- Stable question IDs and at least 80% citation-evaluable rows.
- No paper overlap among train, dev, and new evaluation partition.
- Learned policy is non-inferior to flat in citation precision and recall under
  the dev-frozen margins, and improves over flat with a paper-clustered interval
  consistent with the stated claim.
- Report static as the strongest comparator whenever it remains better.
- Preserve per-query artifacts, checkpoint/data hashes, config hash, package
  versions, seed, and a dirty-worktree indicator for every result table.

## OOD policy

SciFact dev is evaluated once with QASPER-frozen checkpoints. Future changes
must use SciFact train or cross-validation folds for development and a different
held-out fold for evaluation; do not tune against the completed 40-paper OOD run.