# V5 execution log and evidence status

## Scope frozen for the next paper-quality run

Method: **Attribution-Risk-Constrained Adaptive Hierarchical Retrieval**.
The central test is whether a learned expansion gate preserves or improves
citation precision/recall under paper-level distribution shift while retaining
legitimate evidence rescue. Claims about “OOD” are not made from a random
QASPER test split alone; the paper must call it an unseen-paper split unless a
second corpus or explicit domain/time shift is used.

## Completed engineering gates

- Retrieval is scoped to a paper before top-k for dense, sparse, BM25, RRF and
  legacy retrievers.
- Pipeline and rollout verification use the same retrieved-leaf guard.
- A claim is rejected when sibling filtering leaves no candidate leaf.
- Benchmark artifacts expose stable identity fields and the R/V/G leaf sets,
  with rescued, harmful-drift, kept-correct and kept-wrong decomposition.
- Questions without mappable gold leaves are marked non-evaluable instead of
  contributing artificial zeros to citation macro metrics.
- V5 reward includes answer F1, citation precision/recall, rescue, harmful
  drift, ambiguity, token/latency cost and an explicit empty-evidence penalty.
- V5 labels reject precision loss, recall loss, excessive harmful drift and
  expansion-to-empty degeneracy.
- Parent and section policies are independent runtime checkpoints and support
  level-specific ablations.
- Training emits a sidecar with checkpoint/data SHA-256 hashes, feature size,
  split report, seed and all v5 constraint parameters.
- Full offline test suite: 66 passing tests after these changes.

## Existing artifacts: diagnostic only

The four historical artifact files contain 344 rows; 292 (84.9%) have mappable
gold leaves, but none has a stable question ID. Their decomposition found only
one rescue/drift case. This is useful as a migration check, not as the final
paper table: these rows predate the corrected source scoping, verifier guard,
v5 reward and identity schema. `analysis/v5_decomposition.md` records their
hashes and diagnostic summary.

The existing `checkpoints/policy_parent_v5.ts` is therefore provisional. It
must not be described as the final v5 model because its source rollout does not
meet the new artifact contract and it lacks the new sidecar provenance.

## Required fresh runs

1. Create train/dev/test manifests with stable question IDs and paper-disjoint
   splits. Report how many questions have mappable gold leaves; target at least
   80%, and never silently score the remainder as citation failures.
2. Generate fresh train and dev counterfactual rollouts with the corrected
   pipeline. Train separate parent and section v5 checkpoints.
3. Select thresholds and constraint parameters on dev only. Freeze them before
   touching test.
4. Evaluate B_flat, B_static, the prior gate and the learned v5 gate on the same
   paper-level test split, with identical generator/verifier settings.
5. Report per-query artifacts, macro metrics, paper-clustered confidence
   intervals, seeds, package/model versions, checkpoint hashes and data hashes.
6. Run at least two generator conditions. The Gemini swap remains
   blocked-by-quota until the provider quota resets; do not relabel it as a code
   failure or silently substitute it in the main table.
7. Use a second corpus or explicit domain/time split for a true OOD claim.
   Otherwise label QASPER held-out papers precisely as “unseen-paper test”.

## Acceptance rule

The learned v5 system is acceptable for the main paper claim only if its
citation precision and recall are non-inferior to B_flat within the predeclared
margin, its harmful-drift rate satisfies the dev-frozen ceiling, and any claimed
recall gain has a paper-clustered confidence interval that does not contradict
the claim. If it fails, report the negative result and narrow the contribution
to the measured attribution-drift phenomenon and its diagnostic framework.

## Frozen test outcomes

The 40-question/40-paper QASPER unseen-paper test used `openai:gpt-4o-mini`.
Learned v5 improved citation F1 from 0.0250 (B_flat) to 0.0560, but the paired
comparison was not significant (p=0.2178; paper-clustered 95% CI for the
difference [0.0000, 0.0786]). B_static reached 0.1062 and was the only system
with a significant improvement over B_flat (p=0.0350). These results do not
support a superiority claim for learned v5 on the main test.

An exploratory generator swap used `antigravity:gemini-3.7-flash` on the first
10 matched unseen test papers. Learned v5 reached citation F1 0.5519 versus
0.4852 for B_flat, with p=0.5375 and a clustered 95% CI [0.0000, 0.2000]. The
sample is too small for a comparative claim, but it confirms strong generator
sensitivity and satisfies the requirement to evaluate a second generator
condition without mixing it into the main table.

## True out-of-domain result

The QASPER-frozen checkpoints were evaluated once on 40 labeled SciFact dev
evidence papers. Learned v5 reached citation F1 0.7358 versus 0.3833 for B_flat
(paired p=0.0010; paper-clustered 95% CI for the difference [0.2250, 0.4867]).
B_static remained slightly higher at 0.7450. Failure counters show 6/40 empty
generations for learned/static versus 22/40 for flat, with no retrieval miss in
the selected subset. SciFact answer F1 is not interpreted because the current
generator is not a normalized stance classifier; this run supports only the
rationale-attribution OOD claim.
