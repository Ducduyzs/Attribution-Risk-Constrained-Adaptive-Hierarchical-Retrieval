# Frozen V5 training report

## Data contract

The final gates were trained from fresh, paper-scoped QASPER v0.3 rollouts.
Every row has a stable question ID. The official train and dev papers are
disjoint, and the held-out test papers were not used for fitting, estimator
selection or threshold selection.

| split | rows | papers | citation-evaluable | evaluable rate |
|---|---:|---:|---:|---:|
| train | 120 | 120 | 110 | 91.67% |
| dev | 40 | 40 | 36 | 90.00% |

Train rollout SHA-256:
`167f49d7d07a807976695d6d46649fc9b03ca68c36273d26843127dba33ca981`.

Dev rollout SHA-256:
`ac65d9097e0af60be7a7ff3422c1259abe0d9321f5ed01c0cde37d04e857b400`.

## Frozen checkpoints

| gate | estimator | threshold | train balanced accuracy | dev balanced accuracy | dev AUC |
|---|---|---:|---:|---:|---:|
| parent | random forest | 0.50 | 0.8833 | 0.7207 | 0.5953 |
| section | gradient boosting | 0.32 | 0.8818 | 0.6823 | 0.6806 |

Parent checkpoint SHA-256:
`ab0b22478bbb3295f709e051cf0f449267b6de0727acc5587224cc571823498c`.

Section checkpoint SHA-256:
`1d7d7aeae407cf2f12622df2e366e2e38907865b17291c1a8d61de9f03fd4dbb`.

The current training script reproduced both checkpoint hashes exactly in an
independent rerun. Paper weighting is inverse rows per source, normalized to
mean one. Because this frozen run used one question per paper, every training
and development row received weight 1.

## Interpretation

The final tree gates replace the weaker diagnostic TorchScript MLPs. Model
family and threshold were selected on dev before the frozen test was executed.
The development set is small, so the checkpoints are a reproducible v5
baseline rather than the final evidence for a superiority claim. The next
cycle must increase train/dev coverage and preserve a new untouched evaluation
partition.
