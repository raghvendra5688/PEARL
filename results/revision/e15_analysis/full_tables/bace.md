**BACE**: mean ± SD test MCC over five seeds (42 to 46). Bold: best of the three stages within the row and tree.

| Embedding, loss | XGBoost: FT Embed | XGBoost: FT Embed+PC | XGBoost: RAFE | LightGBM: FT Embed | LightGBM: FT Embed+PC | LightGBM: RAFE |
|---|---|---|---|---|---|---|
| ChemBERTa-MLM FL | 0.401 ± 0.014 | 0.422 ± 0.024 | **0.433 ± 0.011** | 0.415 ± 0.021 | 0.418 ± 0.007 | **0.423 ± 0.006** |
| ChemBERTa-MLM WL | 0.471 ± 0.012 | **0.480 ± 0.021** | 0.461 ± 0.007 | 0.494 ± 0.005 | **0.498 ± 0.004** | 0.494 ± 0.009 |
| ChemBERTa-MTR FL | 0.458 ± 0.003 | **0.463 ± 0.010** | 0.460 ± 0.018 | 0.466 ± 0.021 | 0.475 ± 0.014 | **0.476 ± 0.015** |
| ChemBERTa-MTR WL | 0.433 ± 0.027 | **0.440 ± 0.023** | 0.431 ± 0.005 | **0.448 ± 0.018** | 0.428 ± 0.005 | 0.431 ± 0.018 |
| MolFormer FL | 0.512 ± 0.021 | **0.522 ± 0.010** | 0.508 ± 0.018 | 0.521 ± 0.008 | **0.541 ± 0.018** | 0.520 ± 0.018 |
| MolFormer WL | **0.542 ± 0.005** | 0.538 ± 0.016 | 0.527 ± 0.011 | 0.539 ± 0.010 | 0.549 ± 0.020 | **0.553 ± 0.017** |
| Uni-Mol FL | 0.444 ± 0.017 | 0.479 ± 0.025 | **0.496 ± 0.041** | **0.478 ± 0.027** | 0.468 ± 0.016 | 0.471 ± 0.023 |
| Uni-Mol WL | 0.429 ± 0.018 | **0.440 ± 0.017** | 0.419 ± 0.036 | 0.421 ± 0.027 | **0.439 ± 0.016** | 0.424 ± 0.015 |
| *PC-only (473 descriptors)* | 0.477 ± 0.036 |  |  | 0.485 ± 0.021 |  |  |
| *Mean difference over rows* | RAFE−FT: +0.005 | RAFE−FT+PC: -0.006 | 8 rows | RAFE−FT: +0.001 | RAFE−FT+PC: -0.003 | 8 rows |

Pending trees: CatBoost.
