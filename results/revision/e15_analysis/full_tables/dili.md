**DILI**: mean ± SD test MCC over five seeds (42 to 46). Bold: best of the three stages within the row and tree.

| Embedding, loss | XGBoost: FT Embed | XGBoost: FT Embed+PC | XGBoost: RAFE | LightGBM: FT Embed | LightGBM: FT Embed+PC | LightGBM: RAFE |
|---|---|---|---|---|---|---|
| ChemBERTa-MLM FL | 0.536 ± 0.025 | 0.522 ± 0.020 | **0.637 ± 0.024** | 0.613 ± 0.047 | **0.645 ± 0.039** | 0.617 ± 0.060 |
| ChemBERTa-MLM WL | 0.563 ± 0.059 | 0.542 ± 0.057 | **0.565 ± 0.024** | **0.579 ± 0.021** | 0.574 ± 0.030 | 0.564 ± 0.021 |
| ChemBERTa-MTR FL | 0.412 ± 0.046 | 0.467 ± 0.030 | **0.543 ± 0.040** | 0.416 ± 0.021 | 0.521 ± 0.025 | **0.535 ± 0.052** |
| ChemBERTa-MTR WL | 0.546 ± 0.026 | 0.542 ± 0.028 | **0.562 ± 0.025** | **0.558 ± 0.017** | 0.534 ± 0.025 | 0.538 ± 0.031 |
| MolFormer FL | 0.507 ± 0.038 | 0.537 ± 0.018 | **0.541 ± 0.021** | 0.545 ± 0.018 | 0.553 ± 0.031 | **0.608 ± 0.023** |
| MolFormer WL | 0.446 ± 0.032 | **0.472 ± 0.019** | 0.461 ± 0.027 | 0.491 ± 0.024 | **0.517 ± 0.018** | 0.461 ± 0.037 |
| Uni-Mol FL | 0.520 ± 0.075 | 0.693 ± 0.019 | **0.697 ± 0.008** | 0.574 ± 0.061 | **0.690 ± 0.032** | 0.658 ± 0.039 |
| Uni-Mol WL | 0.576 ± 0.012 | 0.653 ± 0.027 | **0.677 ± 0.013** | 0.554 ± 0.035 | **0.675 ± 0.008** | 0.658 ± 0.024 |
| *PC-only (473 descriptors)* | 0.672 ± 0.034 |  |  | 0.654 ± 0.015 |  |  |
| *Mean difference over rows* | RAFE−FT: +0.072 | RAFE−FT+PC: +0.032 | 8 rows | RAFE−FT: +0.039 | RAFE−FT+PC: -0.009 | 8 rows |

Pending trees: CatBoost.
