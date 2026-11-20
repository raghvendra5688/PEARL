**BBBP**: mean ± SD test MCC over five seeds (42 to 46). Bold: best of the three stages within the row and tree.

| Embedding, loss | XGBoost: FT Embed | XGBoost: FT Embed+PC | XGBoost: RAFE | LightGBM: FT Embed | LightGBM: FT Embed+PC | LightGBM: RAFE |
|---|---|---|---|---|---|---|
| ChemBERTa-MLM FL | 0.732 ± 0.009 | **0.746 ± 0.013** | 0.733 ± 0.006 | 0.741 ± 0.012 | **0.751 ± 0.009** | 0.737 ± 0.007 |
| ChemBERTa-MLM WL | **0.711 ± 0.006** | 0.707 ± 0.013 | 0.705 ± 0.001 | 0.700 ± 0.031 | 0.698 ± 0.017 | **0.703 ± 0.011** |
| ChemBERTa-MTR FL | 0.683 ± 0.013 | **0.690 ± 0.005** | 0.659 ± 0.005 | **0.675 ± 0.006** | 0.671 ± 0.010 | 0.673 ± 0.011 |
| ChemBERTa-MTR WL | 0.701 ± 0.005 | 0.703 ± 0.012 | **0.708 ± 0.010** | **0.696 ± 0.006** | 0.676 ± 0.013 | 0.687 ± 0.007 |
| MolFormer FL | 0.783 ± 0.010 | **0.785 ± 0.016** | 0.779 ± 0.010 | **0.769 ± 0.026** | 0.755 ± 0.015 | 0.767 ± 0.015 |
| MolFormer WL | 0.730 ± 0.014 | **0.737 ± 0.008** | 0.732 ± 0.006 | 0.752 ± 0.014 | **0.761 ± 0.012** | 0.746 ± 0.013 |
| Uni-Mol FL | 0.721 ± 0.017 | 0.734 ± 0.021 | **0.735 ± 0.031** | 0.731 ± 0.017 | 0.742 ± 0.024 | **0.745 ± 0.011** |
| Uni-Mol WL | 0.701 ± 0.016 | 0.729 ± 0.022 | **0.735 ± 0.012** | 0.713 ± 0.023 | **0.743 ± 0.012** | 0.728 ± 0.029 |
| *PC-only (473 descriptors)* | 0.731 ± 0.033 |  |  | 0.733 ± 0.030 |  |  |
| *Mean difference over rows* | RAFE−FT: +0.003 | RAFE−FT+PC: -0.006 | 8 rows | RAFE−FT: +0.001 | RAFE−FT+PC: -0.002 | 8 rows |

Pending trees: CatBoost.
