# Shared GPT-4o controlled reader

All systems use the same prompt and GPT-4o reader over leaf-normalized candidate contexts.

| System | N | Answer F1 | Evidence F1 | Mean input tokens | API cost (USD) |
|---|---:|---:|---:|---:|---:|
| B3_flat_neural | 75 | 0.3208 | 0.1897 | 1848.6 | 0.3744 |
| B4_static_hierarchy | 75 | 0.2796 | 0.1429 | 3371.3 | 0.6627 |
| edahr_prior | 75 | 0.3139 | 0.1673 | 2575.2 | 0.5120 |
| B5_raptor_faithful | 75 | 0.3126 | 0.1990 | 1711.8 | 0.3490 |

## Paired answer-F1 comparisons versus B3

- B4_static_hierarchy: delta=-0.0413, p=0.0210, paper-clustered 95% CI [-0.0741, -0.0108].
- edahr_prior: delta=-0.0069, p=0.5395, paper-clustered 95% CI [-0.0285, 0.0144].
- B5_raptor_faithful: delta=-0.0082, p=0.4875, paper-clustered 95% CI [-0.0320, 0.0137].

Estimated total API token charge: $1.8981.
No superiority claim is warranted when the paired confidence interval contains zero.
