# prior vs baselines — v8 dev, 75 questions / 30 papers (exploratory)

Family of 15 comparisons, Holm-adjusted. Paper = sampling unit. Non-inferiority margin 0.02 (absolute).

| Baseline | Metric | Diff | 95% CI (paper bootstrap) | p | p (Holm) | d_z | Non-inferior @0.02 |
|---|---|---:|---|---:|---:|---:|---|
| B5_raptor_faithful | Citation F1 | -0.049 | [-0.109, +0.010] | 0.123 | 1.000 | -0.35 | no (p=0.816) |
| B5_raptor_faithful | Evidence F1 | -0.008 | [-0.056, +0.043] | 0.773 | 1.000 | -0.08 | no (p=0.325) |
| B5_raptor_faithful | Answer F1 | -0.013 | [-0.042, +0.015] | 0.388 | 1.000 | -0.04 | no (p=0.325) |
| B4_static_hierarchy | Citation F1 | +0.022 | [-0.030, +0.077] | 0.441 | 1.000 | +0.11 | no (p=0.068) |
| B4_static_hierarchy | Evidence F1 | +0.024 | [-0.031, +0.080] | 0.431 | 1.000 | +0.15 | no (p=0.069) |
| B4_static_hierarchy | Answer F1 | +0.002 | [-0.028, +0.039] | 0.927 | 1.000 | +0.04 | no (p=0.115) |
| B3_flat_neural | Citation F1 | +0.017 | [-0.023, +0.058] | 0.417 | 1.000 | +0.11 | yes (p=0.044) |
| B3_flat_neural | Evidence F1 | +0.027 | [-0.028, +0.080] | 0.355 | 1.000 | +0.19 | no (p=0.053) |
| B3_flat_neural | Answer F1 | +0.030 | [+0.003, +0.064] | 0.054 | 0.697 | +0.37 | yes (p=0.000) |
| full_document | Citation F1 | +0.044 | [-0.005, +0.095] | 0.093 | 0.928 | +0.29 | yes (p=0.009) |
| full_document | Evidence F1 | +0.059 | [+0.018, +0.101] | 0.010 | 0.134 | +0.42 | yes (p=0.000) |
| full_document | Answer F1 | +0.026 | [-0.009, +0.068] | 0.204 | 1.000 | +0.29 | yes (p=0.008) |
| B6_longrag_faithful | Citation F1 | +0.074 | [+0.001, +0.150] | 0.064 | 0.745 | +0.31 | yes (p=0.010) |
| B6_longrag_faithful | Evidence F1 | +0.109 | [+0.038, +0.186] | 0.006 | 0.083 | +0.55 | yes (p=0.000) |
| B6_longrag_faithful | Answer F1 | +0.036 | [+0.002, +0.074] | 0.062 | 0.745 | +0.37 | yes (p=0.001) |
