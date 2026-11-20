**Half-Life**: mean ± SD test Spearman over five seeds (42 to 46). Bold: best of the three stages within the row and tree.

| Embedding, loss | XGBoost: FT Embed | XGBoost: FT Embed+PC | XGBoost: RAFE | LightGBM: FT Embed | LightGBM: FT Embed+PC | LightGBM: RAFE |
|---|---|---|---|---|---|---|
| ChemBERTa-MLM Huber | 0.350 ± 0.019 | **0.370 ± 0.025** | 0.351 ± 0.034 | 0.325 ± 0.007 | **0.349 ± 0.010** | 0.332 ± 0.027 |
| ChemBERTa-MTR Huber | 0.472 ± 0.011 | 0.483 ± 0.017 | **0.496 ± 0.007** | **0.508 ± 0.023** | 0.506 ± 0.014 | 0.504 ± 0.016 |
| MolFormer Huber | 0.592 ± 0.021 | **0.595 ± 0.017** | 0.559 ± 0.024 | 0.550 ± 0.010 | **0.558 ± 0.036** | 0.512 ± 0.020 |
| Uni-Mol Huber | 0.385 ± 0.036 | 0.379 ± 0.041 | **0.427 ± 0.028** | 0.412 ± 0.027 | **0.428 ± 0.024** | 0.402 ± 0.015 |
| *PC-only (473 descriptors)* | 0.499 ± 0.021 |  |  | 0.477 ± 0.016 |  |  |
| *RAFE minus FT Embed, pooled over rows and seeds [95% CI]* | +0.009 [-0.023, +0.039] |  |  | -0.011 [-0.040, +0.017] |  |  |
| *RAFE minus FT Embed+PC, pooled [95% CI]* | +0.002 [-0.024, +0.026] |  |  | -0.023 [-0.044, -0.002] |  |  |

Pending trees: CatBoost.
