# V9 directions — exploratory DEV only

Exact cl100k_base context+headers budgets; shared GPT-4o-mini and visible-fragment verifier.
Scores are not directly interchangeable with v8 (tokenizer, length instruction, visible verification).
All methods are heuristic prototypes, not validated novelty claims. Test split untouched.

| System | Budget | N | Leaf F1 | Official evidence F1 | Answer F1 | Raw answer F1 | Context tokens | API tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mmr | 1024 | 75 | 0.3459 | 0.3524 | 0.2080 | 0.2065 | 958.2 | 1345.0 |
| flat24 | 2048 | 75 | 0.3775 | 0.3741 | 0.1915 | 0.2080 | 1601.4 | 2005.4 |
| all_leaf | 1024 | 75 | 0.3004 | 0.3387 | 0.2164 | 0.2164 | 892.7 | 1271.4 |
| window | 512 | 75 | 0.2834 | 0.3632 | 0.2319 | 0.2216 | 433.2 | 828.5 |
| window | 1024 | 75 | 0.2949 | 0.3554 | 0.2287 | 0.2282 | 891.3 | 1355.1 |
| mmr | 512 | 75 | 0.3011 | 0.3657 | 0.2290 | 0.2264 | 464.0 | 824.5 |
| contextual24 | 1024 | 75 | 0.3313 | 0.3308 | 0.2036 | 0.2015 | 895.7 | 1283.1 |
| coverage | 512 | 75 | 0.2660 | 0.3249 | 0.2129 | 0.2172 | 466.8 | 823.1 |
| raptor | 2048 | 75 | 0.4068 | 0.4186 | 0.2368 | 0.2259 | 1545.3 | 1944.7 |
| sentence | 2048 | 75 | 0.3092 | 0.3403 | 0.2026 | 0.1904 | 1822.1 | 2399.7 |
| all_leaf | 512 | 75 | 0.2705 | 0.3101 | 0.2119 | 0.2083 | 407.6 | 761.5 |
| contextual_all | 512 | 75 | 0.2856 | 0.3262 | 0.2179 | 0.2178 | 407.2 | 761.6 |
| dependency | 512 | 75 | 0.2500 | 0.2824 | 0.1889 | 0.2037 | 442.3 | 841.2 |
| flat24 | 512 | 75 | 0.2705 | 0.3057 | 0.2123 | 0.2085 | 407.9 | 761.4 |
| flat24 | 1024 | 75 | 0.2975 | 0.3340 | 0.2165 | 0.2160 | 889.5 | 1267.4 |
| contextual_all | 2048 | 75 | 0.3183 | 0.3230 | 0.2033 | 0.2151 | 1595.5 | 2003.0 |
| dependency | 2048 | 75 | 0.3307 | 0.3378 | 0.1861 | 0.1961 | 1821.5 | 2388.4 |
| dependency_first | 1024 | 75 | 0.2649 | 0.2885 | 0.1864 | 0.2211 | 899.6 | 1532.6 |
| dependency | 1024 | 75 | 0.2721 | 0.3091 | 0.2077 | 0.2051 | 899.6 | 1363.3 |
| evidence_first | 2048 | 75 | 0.3744 | 0.3777 | 0.2303 | 0.2378 | 1600.2 | 2140.9 |
| coverage | 1024 | 75 | 0.3298 | 0.3580 | 0.2205 | 0.2216 | 962.8 | 1346.2 |
| dependency_first | 512 | 75 | 0.2192 | 0.2594 | 0.1890 | 0.2165 | 442.3 | 989.9 |
| evidence_first | 1024 | 75 | 0.3301 | 0.3601 | 0.2179 | 0.2221 | 892.7 | 1399.3 |
| coverage | 2048 | 75 | 0.3714 | 0.3831 | 0.2341 | 0.2165 | 1952.9 | 2373.9 |
| contextual24 | 512 | 75 | 0.2805 | 0.3192 | 0.2157 | 0.2134 | 408.4 | 762.1 |
| raptor | 512 | 75 | 0.3354 | 0.3864 | 0.2359 | 0.2338 | 402.8 | 751.7 |
| sentence | 1024 | 75 | 0.3207 | 0.3814 | 0.2121 | 0.1888 | 903.9 | 1376.6 |
| window | 2048 | 75 | 0.3513 | 0.4292 | 0.2366 | 0.2213 | 1809.7 | 2385.0 |
| evidence_first | 512 | 75 | 0.2320 | 0.2489 | 0.2127 | 0.2267 | 407.6 | 880.0 |
| mmr | 2048 | 75 | 0.3878 | 0.3717 | 0.2041 | 0.1964 | 1936.9 | 2356.7 |
| dependency_first | 2048 | 75 | 0.2641 | 0.2372 | 0.1921 | 0.2249 | 1821.5 | 2604.7 |
| sentence | 512 | 75 | 0.3110 | 0.3859 | 0.2134 | 0.2104 | 446.6 | 851.4 |
| raptor | 1024 | 75 | 0.3659 | 0.3670 | 0.2217 | 0.2140 | 898.1 | 1276.8 |
| contextual24 | 2048 | 75 | 0.3318 | 0.3313 | 0.2042 | 0.2130 | 1595.1 | 2004.7 |
| contextual_all | 1024 | 75 | 0.3280 | 0.3246 | 0.2027 | 0.2013 | 897.7 | 1284.5 |
| all_leaf | 2048 | 75 | 0.3786 | 0.3655 | 0.1781 | 0.1941 | 1600.2 | 2006.4 |

## Stage diagnostics

| System@budget | Pool gold recall | Packed leaf touch | Gold paragraph character coverage | Raw citation recall | Verified citation recall |
|---|---:|---:|---:|---:|---:|
| mmr@1024 | 0.987 | 0.442 | 0.454 | 0.354 | 0.325 |
| flat24@2048 | 0.942 | 0.701 | 0.665 | 0.517 | 0.374 |
| all_leaf@1024 | 0.987 | 0.448 | 0.458 | 0.345 | 0.279 |
| window@512 | 0.987 | 0.412 | 0.305 | 0.305 | 0.248 |
| window@1024 | 0.987 | 0.550 | 0.430 | 0.363 | 0.269 |
| mmr@512 | 0.987 | 0.318 | 0.370 | 0.281 | 0.261 |
| contextual24@1024 | 0.942 | 0.517 | 0.514 | 0.399 | 0.307 |
| coverage@512 | 0.987 | 0.306 | 0.375 | 0.269 | 0.240 |
| raptor@2048 | 0.791 | 0.702 | 0.681 | 0.527 | 0.399 |
| sentence@2048 | 0.987 | 0.886 | 0.576 | 0.560 | 0.302 |
| all_leaf@512 | 0.987 | 0.267 | 0.320 | 0.235 | 0.231 |
| contextual_all@512 | 0.987 | 0.340 | 0.410 | 0.285 | 0.255 |
| dependency@512 | 0.987 | 0.545 | 0.257 | 0.319 | 0.245 |
| flat24@512 | 0.942 | 0.267 | 0.316 | 0.235 | 0.231 |
| flat24@1024 | 0.942 | 0.446 | 0.453 | 0.343 | 0.276 |
| contextual_all@2048 | 0.987 | 0.688 | 0.673 | 0.496 | 0.319 |
| dependency@2048 | 0.987 | 0.779 | 0.403 | 0.432 | 0.332 |
| dependency_first@1024 | 0.987 | 0.652 | 0.322 | 0.371 | 0.241 |
| dependency@1024 | 0.987 | 0.652 | 0.322 | 0.383 | 0.263 |
| evidence_first@2048 | 0.987 | 0.700 | 0.669 | 0.525 | 0.368 |
| coverage@1024 | 0.987 | 0.461 | 0.461 | 0.367 | 0.309 |
| dependency_first@512 | 0.987 | 0.545 | 0.257 | 0.304 | 0.210 |
| evidence_first@1024 | 0.987 | 0.448 | 0.458 | 0.360 | 0.306 |
| coverage@2048 | 0.987 | 0.688 | 0.645 | 0.481 | 0.374 |
| contextual24@512 | 0.942 | 0.334 | 0.396 | 0.278 | 0.250 |
| raptor@512 | 0.791 | 0.341 | 0.405 | 0.304 | 0.297 |
| sentence@1024 | 0.987 | 0.755 | 0.410 | 0.515 | 0.320 |
| window@2048 | 0.987 | 0.717 | 0.532 | 0.447 | 0.326 |
| evidence_first@512 | 0.987 | 0.267 | 0.320 | 0.248 | 0.216 |
| mmr@2048 | 0.987 | 0.626 | 0.620 | 0.442 | 0.361 |
| dependency_first@2048 | 0.987 | 0.779 | 0.403 | 0.453 | 0.248 |
| sentence@512 | 0.987 | 0.584 | 0.302 | 0.412 | 0.312 |
| raptor@1024 | 0.791 | 0.496 | 0.512 | 0.390 | 0.336 |
| contextual24@2048 | 0.942 | 0.686 | 0.669 | 0.500 | 0.339 |
| contextual_all@1024 | 0.987 | 0.519 | 0.519 | 0.397 | 0.305 |
| all_leaf@2048 | 0.987 | 0.700 | 0.669 | 0.520 | 0.376 |

Leaf touch is NOT full text retention. Character coverage is conservative exact-span coverage, not semantic sufficiency.
Claim acceptance uses the same NLI verifier and is NOT an independent attribution quality measurement.
Compute/setup and contextual/RAPTOR precomputation costs are separate from online API usage.
Paired cluster tests are exploratory; Holm family includes every completed method/budget/metric vs RAPTOR.
